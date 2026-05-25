"""Graph-driver validation: express step5b's gelu->gelu chain as a declarative
compute graph and run it via Manager.run_graph. Must reproduce step5b's cosines
(graph driver == hand-written KernelStep).

Run:
  cd PLENA_Simulator
  LD_LIBRARY_PATH=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib \
    PYTHONPATH=tools:compiler:transactional_emulator/testbench \
    ./.venv/bin/python3 tools/manager/_validate_graph.py
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.pipeline import Manager
from _tb_cache import mx_roundtrip


def main() -> int:
    s = load_behavior_settings()
    HEAD_COUNT, NUM_S_BLOCKS = 8, 2
    ROWS, HLEN = s.mlen, s.hlen
    SEQ_LEN = NUM_S_BLOCKS * ROWS
    shape = [1, SEQ_LEN, HEAD_COUNT, HLEN]
    GELU = "tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min"
    kw = {"rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
          "num_s_blocks": NUM_S_BLOCKS, "batch": 1}

    def gelu(t):
        return mx_roundtrip(torch.nn.functional.gelu(t, approximate="tanh"))

    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float32) * 0.5
    mid_golden = gelu(mx_roundtrip(x))
    out_golden = gelu(mid_golden)

    # The whole chain as ONE declarative graph. Note MID is named identically
    # in gelu_a.out and gelu_b.in -> shared address -> bin relay, automatically.
    graph = {
        "tensors": {
            "X":   {"shape": shape, "role": "io"},
            "MID": {"shape": shape, "role": "activation"},
            "OUT": {"shape": shape, "role": "io"},
        },
        "nodes": [
            {"name": "gelu_a", "kernel": GELU, "kwargs": kw,
             "in": {"X_hbm": "X"},   "out": {"Y_hbm": "MID"}},
            {"name": "gelu_b", "kernel": GELU, "kwargs": kw,
             "in": {"X_hbm": "MID"}, "out": {"Y_hbm": "OUT"}},
        ],
    }

    mgr = Manager(settings=s)
    out = mgr.run_graph(
        graph,
        data={"X": x},
        compare={"MID": mid_golden.reshape(-1).numpy(),
                 "OUT": out_golden.reshape(-1).numpy()},
    )

    print()
    fails = 0
    for cmp in out["compares"]:
        ok = cmp.ok(cos_thresh=0.85)
        fails += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'} {cmp.name}: cosine={cmp.cosine:.6f} "
              f"nrmse={cmp.nrmse:.6f}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
