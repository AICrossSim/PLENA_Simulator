"""Summary of a capture directory: per capture (<model>_<weights>_<mode>) and source, requests, mean prompt and
generated length, share stopped by the token limit, stored rows and bytes. Prints a Markdown table and writes
summary.json into the directory.

usage: python -m complementary_arrays.gpu.capture.summary CAPTURE_DIR
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    d = Path(sys.argv[1])
    tags = defaultdict(list)
    for p in sorted(d.glob("*_chunk*.npz")):
        tags[p.name.rsplit("_chunk", 1)[0]].append(p)
    out = {}
    md = [
        "| capture | source | requests | mean prompt tokens | mean generated tokens | stopped at the limit | rows |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, files in tags.items():
        acc = defaultdict(lambda: dict(n=0, p=0, g=0, cut=0))
        size = rows = 0
        for f in files:
            size += f.stat().st_size
            with np.load(f) as z:
                rows += int(z["routes"].shape[0])
                for s, p, g, e in zip(z["source"], z["prompt_tokens"], z["gen_tokens"], z["finish_eos"], strict=True):
                    a = acc[str(s)]
                    a["n"] += 1
                    a["p"] += int(p)
                    a["g"] += int(g)
                    a["cut"] += int(not e)
        out[tag] = dict(files=len(files), bytes=size, rows=rows, per_source={})
        for s in sorted(acc):
            a = acc[s]
            r = dict(
                requests=a["n"],
                mean_prompt=a["p"] / a["n"],
                mean_gen=a["g"] / a["n"],
                truncated_share=a["cut"] / a["n"],
            )
            out[tag]["per_source"][s] = r
            md.append(
                f"| {tag} | {s} | {a['n']} | {r['mean_prompt']:.0f} | {r['mean_gen']:.0f} | {100 * r['truncated_share']:.1f}% | |"
            )
        n = max(1, sum(a["n"] for a in acc.values()))
        md.append(
            f"| {tag} | all | {n} | | {sum(a['g'] for a in acc.values()) / n:.0f} | "
            f"{100 * sum(a['cut'] for a in acc.values()) / n:.1f}% | {rows:,} ({size / 2**30:.2f} GiB) |"
        )
    (d / "summary.json").write_text(json.dumps(out, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
