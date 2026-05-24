"""Step-3 validation: manager-planned addresses reach the compiler.

Place gelu's HBM tensors via HbmLayout, compile through the manager runner
with those addresses as overrides, and assert every planned address lands
exactly in the compiled buffer_addrs.json.

Run (torch venv + nix libstdc++):
  cd PLENA_Simulator
  LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib \
    PYTHONPATH=tools ./.venv/bin/python3 tools/manager/_validate_step3.py
"""

import sys

import torch

from manager.geometry import load_behavior_settings
from manager.tensor import HbmLayout, Role
from manager.runner import compile_kernel


def main() -> int:
    s = load_behavior_settings()
    print(f"BEHAVIOR: mlen={s.mlen} hlen={s.hlen} blen={s.blen} "
          f"lane_count={s.hardware_lane_count}")

    # gelu geometry (mirrors tvm_gelu_min_test.py)
    HEAD_COUNT = 8
    NUM_S_BLOCKS = 2
    ROWS = s.mlen
    HLEN = s.hlen
    SEQ_LEN = NUM_S_BLOCKS * ROWS
    shape = (1, SEQ_LEN, HEAD_COUNT, HLEN)

    # Plan addresses: manager owns the HBM space, bump-allocates X then Y.
    layout = HbmLayout(s, base=0)
    torch.manual_seed(0)
    X = layout.place("X_hbm", shape, Role.IO,
                    data=torch.randn(*shape) * 0.5)
    Y = layout.place("Y_hbm", shape, Role.IO,
                    data=torch.zeros(*shape))
    planned = layout.overrides()
    print(f"planned: X_hbm@{X.hbm_addr} (packed {X.packed_bytes(s)})  "
          f"Y_hbm@{Y.hbm_addr}")

    ck = compile_kernel(
        "tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min",
        asm_name="gelu_mgr",
        settings=s,
        kernel_kwargs={
            "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
            "num_s_blocks": NUM_S_BLOCKS, "batch": 1,
        },
        hbm_overrides=planned,
    )
    print(f"compiled: {ck.isa_text.count(chr(10))} ISA lines, "
          f"ir_dir={ck.ir_dir}")

    fails = 0
    for name, want in planned.items():
        got = ck.address_of(name)
        ok = got == want
        if not ok:
            fails += 1
        print(f"  {'OK ' if ok else 'FAIL'} {name}: planned={want} compiled={got}")

    # also report hoisted constants discovered (for step 4 / const pool)
    consts = ck.hoisted_constants()
    print(f"  hoisted FP constants: {len(consts)} -> "
          f"{sorted(round(v,5) for v in consts.values())}")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
