"""Final chips: per cell and stream count, the het chip is the faster on the dev select windows (select_main) of the
top10 and hetopt searches (a tie keeps top10); hom and single chips are the top10 results."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from . import results as rs
from .organisations import FAMILIES


def select(out: Path) -> list[tuple]:
    out, chosen = Path(out), []
    top10 = out / "search" / "top10"
    for cell_dir in sorted(top10.iterdir()) if top10.exists() else []:
        cell = cell_dir.name
        for fam in FAMILIES.values():
            if not all(rs.search_path(out, "top10", cell, o).exists() for o in fam):
                continue
            for o in fam:
                stages = ["top10"]
                hetopt = rs.search_path(out, "hetopt", cell, o)
                if o.startswith("het") and hetopt.exists():
                    stages.append("hetopt")
                best = {s: rs.read_json(rs.search_path(out, s, cell, o))["best_main"] for s in stages}
                stage = min(stages, key=lambda s: best[s]["select_main"])
                rs.write_json(
                    rs.final_path(out, cell, o),
                    dict(
                        best_main=best[stage],
                        selected_from=stage,
                        source=str(rs.search_path(out, stage, cell, o).relative_to(out)),
                        candidates_select_main={s: b["select_main"] for s, b in best.items()},
                        provisional=o.startswith("het") and not hetopt.exists(),
                    ),
                )
                chosen.append((cell, o, stage))
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    a = ap.parse_args(argv)
    if a.out is None:
        ap.error("--out (or COMPLEMENTARY_ARRAYS_OUT) is required")
    chosen = select(a.out)
    print(len(chosen), "final chips")
    for row in chosen:
        if row[1].startswith("het"):
            print(*row)


if __name__ == "__main__":
    main()
