"""Step-5b validation: 2-kernel chain via HBM-bin RELAY (no asm concat).

gelu_a reads X, writes MID; gelu_b reads MID, writes OUT. gelu_a's Y_hbm and
gelu_b's X_hbm map to the SAME managed tensor (MID) -> same HBM address. The
two kernels run as independent emulator invocations; gelu_b's --hbm is gelu_a's
hbm_dump, so MID flows through the shared bin with no instruction-stream
concatenation.

Golden OUT = MX(gelu(MX(gelu(MX(x))))) — MX round-trip at every HBM hop.

Run:
  cd PLENA_Simulator
  LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib \
    PYTHONPATH=tools:compiler:transactional_emulator/testbench \
    ./.venv/bin/python3 tools/manager/_validate_step5b.py
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.tensor import Role
from manager.pipeline import Manager, KernelStep
from _tb_cache import mx_roundtrip


def main() -> int:
    s = load_behavior_settings()
    HEAD_COUNT = 8
    NUM_S_BLOCKS = 2
    ROWS = s.mlen
    HLEN = s.hlen
    SEQ_LEN = NUM_S_BLOCKS * ROWS
    shape = (1, SEQ_LEN, HEAD_COUNT, HLEN)
    GELU = "tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min"
    kw = {"rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
          "num_s_blocks": NUM_S_BLOCKS, "batch": 1}

    def gelu(t):
        return mx_roundtrip(torch.nn.functional.gelu(t, approximate="tanh"))

    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float32) * 0.5
    x_eff = mx_roundtrip(x)
    mid_golden = gelu(x_eff)               # after gelu_a
    out_golden = gelu(mid_golden)          # after gelu_b

    mgr = Manager(settings=s)
    # three managed tensors; MID is shared (gelu_a out == gelu_b in)
    mgr.place("X", shape, Role.IO, data=x)
    mgr.place("MID", shape, Role.ACTIVATION, data=None)   # produced by gelu_a
    mgr.place("OUT", shape, Role.IO, data=None)
    print(f"addrs: X={mgr.layout.tensors['X'].hbm_addr} "
          f"MID={mgr.layout.tensors['MID'].hbm_addr} "
          f"OUT={mgr.layout.tensors['OUT'].hbm_addr}")

    steps = [
        KernelStep(GELU, "gelu_a", kw,
                   tensor_map={"X_hbm": "X", "Y_hbm": "MID"},
                   compare={"MID": mid_golden.reshape(-1).numpy()}),
        KernelStep(GELU, "gelu_b", kw,
                   tensor_map={"X_hbm": "MID", "Y_hbm": "OUT"},
                   compare={"OUT": out_golden.reshape(-1).numpy()}),
    ]

    out = mgr.run_pipeline(steps)

    print()
    fails = 0
    for cmp in out["compares"]:
        ok = cmp.ok(cos_thresh=0.85)
        if not ok:
            fails += 1
        print(f"  {'OK ' if ok else 'FAIL'} {cmp.name}: cosine={cmp.cosine:.6f} "
              f"nrmse={cmp.nrmse:.6f}")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
