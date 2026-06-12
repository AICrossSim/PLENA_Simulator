"""COMPILE-ONLY flash-attention BACKWARD: lower the bwd kernel through the
real manager pipeline so its .isa lands in <build_root>/ir/flash_attention_bwd/.

Same machinery as _compile_block.py (Manager(compile_only=True): per-kernel
compile + assemble, no golden / no HBM bin / no emulator / no compare), but
the graph holds a single node: the flash-attention backward kernel. The ISA
is meant to be fed straight to the analytic model
(tools/power/plena_isa_energy.py), not the simulator — correctness of the
kernel is intentionally not checked for the softmax statistics (L/D zeroed).
dQ now takes a real transpose-A GEMM (``transpose_A=True`` -> M_TMM_A), so the
dQ = dS @ K step is numerically exact and no longer a pre-transposed proxy;
see flash_attention_bwd_min.py.

Run:
  PLENA_ALLOC_MODE=stable tools/manager/run.sh _compile_bwd managerbuild_bwd
  PLENA_ALLOC_MODE=stable tools/manager/run.sh _compile_bwd managerbuild_bwd plena_settings.SSB_build_1.toml

Note: use PLENA_ALLOC_MODE=stable — the deep q/kv loop nest exhausts the GP
pool under gp_only_spill. stable gives every value a fixed IntRAM home.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.pipeline import Manager


# Real Open-Sora v2 single-stream attention dims (padded to MLEN multiples).
NSB = 9                 # seq blocks: S = 9*1024 = 9216 (real 8828 padded)
HD_REAL = 3072          # hidden


def build_graph(s):
    MLEN, HLEN = s.mlen, s.hlen
    S = NSB * MLEN          # 9216
    HD = HD_REAL           # 3072
    HEAD = HD // HLEN      # 24

    BWD = "tilelang_tvm_compiler.kernels.flash_attention_bwd_min:make_flash_attention_bwd_min"

    bshd = [1, S, HEAD, HLEN]

    bwd_kw = {
        "rows": MLEN,
        "hlen": HLEN,
        "head_count": HEAD,
        "num_kv_blocks": NSB,
        "num_q_blocks": NSB,
    }

    def t(shape, role):
        return {"shape": shape, "role": role}

    tensors = {
        "Q":  t(bshd, "io"),
        "K":  t(bshd, "io"),
        "V":  t(bshd, "io"),
        "dO": t(bshd, "io"),
        "dQ": t(bshd, "io"),
        "dK": t(bshd, "io"),
        "dV": t(bshd, "io"),
    }

    nodes = [
        {
            "name": "flash_attention_bwd",
            "kernel": BWD,
            "kwargs": bwd_kw,
            "in":  {"Q_hbm": "Q", "K_hbm": "K", "V_hbm": "V", "dO_hbm": "dO"},
            "out": {"dQ_hbm": "dQ", "dK_hbm": "dK", "dV_hbm": "dV"},
        },
    ]
    return {"tensors": tensors, "nodes": nodes}


def main() -> int:
    import time as _time
    build_root = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    toml_path = sys.argv[2] if len(sys.argv) > 2 else None
    s = load_behavior_settings(toml_path)
    print(f"[compile-only bwd] settings toml: {toml_path or 'plena_settings.toml (default)'} "
          f"| MLEN={s.mlen} HLEN={s.hlen} BLEN={s.blen} VLEN={s.vlen}")
    graph = build_graph(s)

    mgr = Manager(settings=s, build_root=build_root, compile_only=True)
    print(f"[compile-only bwd] flash_attention_bwd -> {mgr.ir_dir}")
    _t0 = _time.time()
    out = mgr.run_graph(graph, data=None, compare=None)
    _wall = _time.time() - _t0

    print(f"\n[compile-only bwd] done: {len(out['results'])} kernel(s) compiled in {_wall:.1f}s")
    print(f"  ISA at: {mgr.ir_dir}/<kernel>/<kernel>.isa")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
