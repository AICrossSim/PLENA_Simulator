"""Single modulation-generation kernel end-to-end through the manager.

adaLN-Zero modulation: from one vec (selected by pipe_idx out of a
(1, pipelined, 1, HD) input) produce shift/scale/gate, each broadcast across
SEQ rows:

    m = silu(vec[pipe_idx])
    shift = m @ Wshift^T   (broadcast over SEQ)
    scale = m @ Wscale^T
    gate  = m @ Wgate^T

Geometry from plena_settings.toml. Uses the manager graph driver + HBM-bin
relay (MX-E4M3 round-trip per hop) — same proven path as _validate_linear /
_validate_block, so HBM packing / scale / DMA are handled by the manager.

Run: tools/manager/run.sh _validate_modulation
"""

import os
import sys
from pathlib import Path

# Use the stable register allocator for this validation (set before any
# compiler import / subprocess so the env propagates to the compile step).
os.environ["PLENA_ALLOC_MODE"] = "stable"

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.tensor import Role
from manager.pipeline import Manager
from _tb_cache import mx_roundtrip as mxr


def main() -> int:
    s = load_behavior_settings()
    MLEN, HLEN = s.mlen, s.hlen
    HEAD = s.hardware_lane_count * 2
    # HD in units of MLEN blocks. Default 2 so the multi-block N/K path
    # (sum-over-kb accumulation + W[kb, nb] tile indexing) is the default
    # coverage; override with MOD_HD_BLOCKS=1 for the single-block path.
    HD = 2* MLEN
    PIPELINED = 4
    PIPE_IDX = 2
    NSB = 2                       # single output tile (broadcast)
    SEQ = NSB * MLEN
    print(f"geometry: mlen={MLEN} hlen={HLEN} HEAD={HEAD} HD={HD}  "
          f"pipelined={PIPELINED} pipe_idx={PIPE_IDX} SEQ={SEQ}")

    torch.manual_seed(0)
    vec2d = torch.randn(PIPELINED, HD) * 0.5
    Wsh = torch.randn(HD, HD) * 0.1
    Wsc = torch.randn(HD, HD) * 0.1
    Wgt = torch.randn(HD, HD) * 0.1

    # VEC_hbm is (1, MLEN, 1, HD): the pipelined vecs live in the first
    # PIPELINED rows, the rest is zero padding (DMA stages a full MLEN-row
    # tile, like linear's A).
    vec_padded = torch.zeros(MLEN, HD)
    vec_padded[:PIPELINED] = vec2d
    vec = vec_padded.view(1, MLEN, 1, HD).contiguous()
    # weights are (N, K) row-major (nn.Linear convention); kernel issues
    # transpose_B=True, same as linear_min — host does NOT transpose.
    wsh = Wsh.view(1, HD, 1, HD).contiguous()
    wsc = Wsc.view(1, HD, 1, HD).contiguous()
    wgt = Wgt.view(1, HD, 1, HD).contiguous()

    # ---- golden (MX round-trip each hop) ----
    # SiLU(vec) = vec * sigmoid(vec), then m @ W. The kernel computes silu in
    # f16 per K-slice via 1/(1+exp(-x)); mirror that in f16 here (cast through
    # half so exp/reciprocal rounding matches the hardware path).
    vec_eff = mxr(vec2d)
    x = vec_eff[PIPE_IDX].half().float()                     # (HD,)
    sig = 1.0 / (1.0 + torch.exp(-x))
    m = (x * sig).half().float()                             # silu(vec)

    # mv computes vec @ W (NO transpose — mv path ignores transpose_B).
    def lin(w2d):
        return mxr(m @ mxr(w2d))                              # (HD,) vec@W
    shift_row = lin(Wsh); scale_row = lin(Wsc); gate_row = lin(Wgt)

    def bcast(row):
        return mxr(row.view(1, HD).expand(SEQ, HD).contiguous())  # (SEQ, HD)

    SHIFT_g = bcast(shift_row); SCALE_g = bcast(scale_row); GATE_g = bcast(gate_row)

    mgr = Manager(settings=s)
    mgr.place("VEC_hbm",   [1, MLEN, 1, HD],      Role.IO,     data=vec)
    mgr.place("W_SHIFT",   [1, HD, 1, HD],        Role.WEIGHT, data=wsh)
    mgr.place("W_SCALE",   [1, HD, 1, HD],        Role.WEIGHT, data=wsc)
    mgr.place("W_GATE",    [1, HD, 1, HD],        Role.WEIGHT, data=wgt)
    mgr.place("SHIFT_hbm", [1, SEQ, 1, HD],       Role.IO,     data=torch.zeros(1, SEQ, 1, HD))
    mgr.place("SCALE_hbm", [1, SEQ, 1, HD],       Role.IO,     data=torch.zeros(1, SEQ, 1, HD))
    mgr.place("GATE_hbm",  [1, SEQ, 1, HD],       Role.IO,     data=torch.zeros(1, SEQ, 1, HD))

    out = mgr.run_kernel(
        "tilelang_tvm_compiler.kernels.modulation_gen_min:make_modulation_gen_min",
        asm_name="modulation_solo",
        kernel_kwargs={"hd": HD, "pipelined": PIPELINED,
                       "pipe_idx": PIPE_IDX, "seq": SEQ},
        compare={
            "SHIFT_hbm": SHIFT_g.reshape(-1).numpy(),  # SHIFT-only isolation
        },
    )

    print()
    fails = 0
    for cmp in out["compares"]:
        ok = cmp.ok(cos_thresh=0.8)
        fails += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'} {cmp.name}: "
              f"cosine={cmp.cosine:.6f} nrmse={cmp.nrmse:.6f}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'} (cosine >= 0.8)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
