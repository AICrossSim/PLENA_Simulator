"""Step-4 validation: global FPRAM constant pool across chained kernels.

Compile gelu + silu, collect both kernels' hoisted constants into one
ConstPool, and verify:
  A. dedup — values shared between kernels (1.0, -1.0) get ONE slot.
  B. recompiling each kernel with the pool's fpram overrides lands every
     constant at its assigned slot.
  C. fp_sram.bin holds each allocated slot's value at the right index.
  D. scratch_base sits above the const ceiling.

Run (torch venv + nix libstdc++):
  cd PLENA_Simulator
  LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib \
    PYTHONPATH=tools ./.venv/bin/python3 tools/manager/_validate_step4.py
"""

import sys
import tempfile

import numpy as np

from manager.geometry import load_behavior_settings
from manager.runner import compile_kernel
from manager.const_pool import ConstPool, FPRAM_USER_BASE


def _gelu_kwargs(s):
    return {"rows": s.mlen, "hlen": s.hlen, "head_count": 8,
            "num_s_blocks": 2, "batch": 1}


def _ln_kwargs(s):
    return {"rows": s.mlen, "hidden_size": 128, "num_s_blocks": 2, "batch": 1}


# (asm_name, kernel_spec, kwargs_fn). gelu twice (2nd run must add 0 new slots —
# pure dedup); layernorm's constants don't overlap gelu's (adds new slots).
_KERNELS = [
    ("gelu_a", "tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min", _gelu_kwargs),
    ("ln_b", "tilelang_tvm_compiler.kernels.layernorm_min:make_layernorm_min", _ln_kwargs),
    ("gelu_c", "tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min", _gelu_kwargs),
]


def main() -> int:
    s = load_behavior_settings()
    fails = 0

    # --- compile each kernel once (no overrides) to discover constants ---
    compiled = {}
    for name, spec, kwfn in _KERNELS:
        compiled[name] = compile_kernel(spec, asm_name=name, settings=s,
                                        kernel_kwargs=kwfn(s))
        print(f"{name} consts: "
              f"{sorted(round(v,5) for v in compiled[name].hoisted_constants().values())}")

    # --- collect into one global pool, tracking slot growth per kernel ---
    pool = ConstPool()
    growth = {}
    for name, _, _ in _KERNELS:
        before = pool.const_ceiling
        pool.collect(name, compiled[name].hoisted_constants())
        growth[name] = pool.const_ceiling - before

    all_values = [v for c in compiled.values() for v in c.hoisted_constants().values()]
    distinct_values = len({np.float16(v).view(np.uint16) for v in all_values})
    slots_used = pool.const_ceiling - FPRAM_USER_BASE
    print(f"\n[A] dedup: distinct={distinct_values}, slots used={slots_used}, "
          f"per-kernel growth={growth}, scratch_base={pool.scratch_base}")
    ok_a = slots_used == distinct_values
    fails += 0 if ok_a else 1
    print(f"  {'OK ' if ok_a else 'FAIL'} slots == distinct values")
    # 2nd gelu run must add zero new slots (every value already present)
    ok_dedup = growth["gelu_c"] == 0
    fails += 0 if ok_dedup else 1
    print(f"  {'OK ' if ok_dedup else 'FAIL'} gelu_c added 0 slots (pure dedup)")
    # the two gelu runs must map identical const names to identical slots
    ok_same = pool.overrides_for("gelu_a") == pool.overrides_for("gelu_c")
    fails += 0 if ok_same else 1
    print(f"  {'OK ' if ok_same else 'FAIL'} gelu_a / gelu_c slot maps identical")

    # --- B: recompile each with pool overrides; assert slots land ---
    print("\n[B] recompile with fpram overrides; constants land at assigned slots")
    for name, spec, kwfn in _KERNELS:
        ov = pool.overrides_for(name)
        ck = compile_kernel(spec, asm_name=name, settings=s,
                            kernel_kwargs=kwfn(s), fpram_overrides=ov)
        for cname, want in ov.items():
            got = ck.address_of(cname)
            ok = got == want
            fails += 0 if ok else 1
            print(f"  {'OK ' if ok else 'FAIL'} {name}/{cname}: planned={want} got={got}")

    # --- C: fp_sram.bin content ---
    print("\n[C] fp_sram.bin slot values")
    binp = tempfile.mktemp(suffix=".bin")
    n = pool.write_fp_sram(binp)
    arr = np.frombuffer(open(binp, "rb").read(), dtype=np.float16)
    ok_c = True
    for slot, value in pool._value_of_slot.items():
        if not np.isclose(float(arr[slot]), float(np.float16(value)), atol=1e-3):
            ok_c = False
            print(f"  FAIL slot {slot}: bin={arr[slot]} expected={value}")
    fails += 0 if ok_c else 1
    print(f"  {'OK ' if ok_c else 'FAIL'} {n} slots written, all values match")

    # --- D: scratch above const ceiling ---
    ok_d = pool.scratch_base >= pool.const_ceiling and pool.const_ceiling > FPRAM_USER_BASE
    fails += 0 if ok_d else 1
    print(f"\n[D] {'OK ' if ok_d else 'FAIL'} scratch_base({pool.scratch_base}) "
          f">= const_ceiling({pool.const_ceiling}) > base({FPRAM_USER_BASE})")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
