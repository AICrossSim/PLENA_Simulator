"""Step-5a validation: single gelu kernel end-to-end through the Manager.

Mirrors tvm_gelu_min_test.py's golden (MX-roundtrip both sides) but drives the
whole flow through the Manager: plan addresses -> seek-write bin -> fp_sram ->
compile with overrides -> assemble -> run emulator -> read back -> compare.

HBM compare needs NO stride reorder: data lands in HBM as plain BSHD (that's
the whole reason we compare on HBM, not VRAM — VRAM holds the chunk-group-major
intermediate). So read_tensor reads BSHD order directly and lines up with the
BSHD golden, even at head_count=8 (chunks_per_batch=2).

Run:
  cd PLENA_Simulator
  LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib \
    PYTHONPATH=tools:transactional_emulator/testbench \
    ./.venv/bin/python3 tools/manager/_validate_step5a.py
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.tensor import Role
from manager.pipeline import Manager
from _tb_cache import mx_roundtrip   # same MX-E4M3 round-trip the testbench uses


def main() -> int:
    s = load_behavior_settings()
    HEAD_COUNT = 8                      # matches the standalone gelu testbench
    NUM_S_BLOCKS = 2
    ROWS = s.mlen
    HLEN = s.hlen
    SEQ_LEN = NUM_S_BLOCKS * ROWS
    shape = (1, SEQ_LEN, HEAD_COUNT, HLEN)
    print(f"BEHAVIOR mlen={s.mlen} hlen={s.hlen}; gelu shape={shape} "
          f"(epb={HEAD_COUNT*HLEN}, chunks={(HEAD_COUNT*HLEN + s.mlen-1)//s.mlen})")

    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float32) * 0.5

    # golden: gelu on the MX-roundtripped input, then MX-roundtrip the output
    # (the kernel reads X back as MX and writes Y as MX).
    x_eff = mx_roundtrip(x)
    y_golden = mx_roundtrip(torch.nn.functional.gelu(x_eff, approximate="tanh"))

    mgr = Manager(settings=s)
    mgr.place("X_hbm", shape, Role.IO, data=x)
    mgr.place("Y_hbm", shape, Role.IO, data=torch.zeros_like(x))

    out = mgr.run_kernel(
        "tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min",
        asm_name="gelu_e2e",
        kernel_kwargs={"rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
                       "num_s_blocks": NUM_S_BLOCKS, "batch": 1},
        compare={"Y_hbm": y_golden.reshape(-1).numpy()},
    )

    cmp = out["compares"][0]
    print(f"\nY_hbm: cosine={cmp.cosine:.6f} nrmse={cmp.nrmse:.6f}")
    ok = cmp.ok(cos_thresh=0.85)
    print(f"\n{'PASS' if ok else 'FAIL'} (threshold cosine >= 0.85)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
