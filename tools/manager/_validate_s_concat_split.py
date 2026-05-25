"""Smoke test for s_concat_min + s_split_min (double-stream S-axis fuse/split).

Chain: two head-packed segments A=[1,S_a,1,HD], B=[1,S_b,1,HD]
  s_concat(A, B) -> CAT = [1, S_a+S_b, 1, HD]   (A's rows then B's rows)
  s_split(CAT)   -> A2 = CAT[0:S_a], B2 = CAT[S_a:]   (must recover A, B)

Verifies both directions: CAT matches torch.cat(dim=seq), and the round-trip
A2/B2 recover the originals. Each HBM hop MX-E4M3 round-trips, so golden uses
mxr on both sides; residual is pure MX quant noise (threshold 0.8).

Run: tools/manager/run.sh _validate_s_concat_split
"""

import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "transactional_emulator" / "testbench"))

from manager.geometry import load_behavior_settings
from manager.pipeline import Manager
from _tb_cache import mx_roundtrip as mxr


def main() -> int:
    s = load_behavior_settings()
    MLEN, HLEN = s.mlen, s.hlen
    HEAD = s.hardware_lane_count * 2
    HD = HEAD * HLEN
    NSB_A, NSB_B = 2, 1          # txt = 2 seq blocks, img = 1 (asymmetric on purpose)
    S_A, S_B = NSB_A * MLEN, NSB_B * MLEN
    print(f"geometry: mlen={MLEN} hlen={HLEN} HEAD={HEAD} HD={HD}  "
          f"S_a={S_A} (nsb={NSB_A}) S_b={S_B} (nsb={NSB_B})")

    hdv_a = [1, S_A, 1, HD]
    hdv_b = [1, S_B, 1, HD]
    hdv_cat = [1, S_A + S_B, 1, HD]

    torch.manual_seed(0)
    a = torch.randn(1, S_A, 1, HD) * 0.5
    b = torch.randn(1, S_B, 1, HD) * 0.5

    # golden (MX roundtrip per hop)
    ae, be = mxr(a), mxr(b)
    CAT_g = mxr(torch.cat([ae, be], dim=1))          # stack along seq axis
    A2_g = mxr(mxr(CAT_g)[:, 0:S_A])
    B2_g = mxr(mxr(CAT_g)[:, S_A:S_A + S_B])

    SC = "tilelang_tvm_compiler.kernels.s_concat_min:make_s_concat_min"
    SS = "tilelang_tvm_compiler.kernels.s_split_min:make_s_split_min"
    cat_kw = {"hd": HD, "num_s_blocks_a": NSB_A, "num_s_blocks_b": NSB_B, "batch": 1}

    def t(shape, role): return {"shape": shape, "role": role}
    tensors = {
        "A": t(hdv_a, "io"), "B": t(hdv_b, "io"),
        "CAT": t(hdv_cat, "activation"),
        "A2": t(hdv_a, "io"), "B2": t(hdv_b, "io"),
    }
    nodes = [
        {"name": "s_concat", "kernel": SC, "kwargs": cat_kw,
         "in": {"A_hbm": "A", "B_hbm": "B"}, "out": {"Y_hbm": "CAT"}},
        {"name": "s_split", "kernel": SS, "kwargs": cat_kw,
         "in": {"X_hbm": "CAT"}, "out": {"A_hbm": "A2", "B_hbm": "B2"}},
    ]
    graph = {"tensors": tensors, "nodes": nodes}
    data = {"A": a, "B": b}
    compare = {
        "CAT": CAT_g.reshape(-1).numpy(),
        "A2": A2_g.reshape(-1).numpy(),
        "B2": B2_g.reshape(-1).numpy(),
    }

    mgr = Manager(settings=s)
    out = mgr.run_graph(graph, data=data, compare=compare)

    print()
    fails = 0
    for cmp in out["compares"]:
        ok = cmp.ok(cos_thresh=0.8)
        fails += 0 if ok else 1
        print(f"  {'OK ' if ok else 'FAIL'} {cmp.name}: cosine={cmp.cosine:.6f} nrmse={cmp.nrmse:.6f}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
