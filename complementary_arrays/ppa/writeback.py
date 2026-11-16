"""Write-back ratios of re-delivered expert weights, from the evaluation snapshots of one model and format.

Only experts read in more than one pass have their BF16 block written back to the main SRAM, once; the re-delivered
bytes are (passes - 1) blocks. Per batch and array height m:
    r_m = (routed experts with c > m) / sum over those experts of (ceil(c / m) - 1),
with c the tokens of an expert in a window, so that write-back bytes = r_m x re-delivered bytes.

usage: python -m complementary_arrays.ppa.writeback OUT.json SNAPSHOT_DIR --experts E [--pattern s1_eval_b{batch}.npz]
  A snapshot file holds routes [windows, batch, top-k] and live [windows] (live requests first in every row).
"""

import argparse
import json
from pathlib import Path

import numpy as np

HEIGHTS = (2, 4, 8, 16)


def ratios(routes, live, experts):
    counts = np.concatenate(
        [np.bincount(routes[i, : live[i]].ravel().astype(np.int64), minlength=experts) for i in range(len(live))]
    )
    counts = counts[counts > 0]
    out = {}
    for m in HEIGHTS:
        hot = counts[counts > m]
        extra = (np.ceil(hot / m) - 1).sum()
        out[str(m)] = dict(
            ratio=float(len(hot) / extra) if extra else None,
            reread_experts_per_window=len(hot) / len(live),
            extra_passes_per_window=float(extra) / len(live),
        )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("snapshots", type=Path)
    p.add_argument("--experts", type=int, required=True, help="routed experts per layer (256 Qwen3.5, 64 DeepSeek)")
    p.add_argument("--pattern", default="s1_eval_b{batch}.npz")
    p.add_argument("--batches", default="2,8,32,64,128")
    a = p.parse_args()
    res = {}
    for b in (int(x) for x in a.batches.split(",")):
        with np.load(a.snapshots / a.pattern.format(batch=b)) as z:
            res[f"B{b}"] = ratios(z["routes"], z["live"], a.experts)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
