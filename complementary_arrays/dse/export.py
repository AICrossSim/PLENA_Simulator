"""Result table of the final chips and the preregistered verdict.

Per cell and stream count: chip shapes, policies and multipliers, heldout cycles per batch and the het speed-ups
over hom, single and the faster of the two (B32..B128 geometric means with the two-way intervals of bootstrap.json),
the sealed-group values, the top10-only speed-ups and the fullhom / hetcap controls. Verdict: in the decision cells
(Qwen3.5 and DeepSeek, BF16 and INT8) het passes when vs_best >= 1.05 and the lower two-way bound exceeds 1; the
hypothesis needs 4 of the 8 cells per stream count.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

from ..sim import workload as wl
from . import results as rs
from .organisations import FAMILIES

FORMAT_NAMES = {"bf16": "BF16", "w8g128": "INT8 (group 128)", "w4g32": "INT4 (group 32)"}


def _gm(xs) -> float:
    return math.exp(wl.sequential_sum(math.log(x) for x in xs) / len(xs))


def _shape(base) -> str:
    t = "({}, {}, {})".format
    if len(base) == 1:
        return t(*base[0])
    return t(*base[0]) + " x2" if list(base[0]) == list(base[1]) else t(*base[0]) + " + " + t(*base[1])


def _policy(p) -> str:
    return f"p{p[0]}" + (f" s{p[1]}" if p[1] is not None else "")


def export(out: Path) -> dict:
    out = Path(out)
    boot = {(r["model"], r["fmt"], r["P"], r["streams"], r["version"]): r for r in rs.read_json(out / "bootstrap.json")}

    def interval(key, part, ratio, field="ci_two_way"):
        r = boot.get(key)
        return r[part]["B32-128"][ratio][field] if r and r.get(part) and "B32-128" in r[part] else None

    def point(key, part, ratio):
        r = boot.get(key)
        return r[part]["B32-128"][ratio]["point"] if r and r.get(part) and "B32-128" in r[part] else None

    rows, verdict = [], []
    for S, fam in FAMILIES.items():
        for fmt in rs.FORMAT_ORDER:
            for model in rs.MODEL_ORDER:
                for P in rs.BUDGETS:
                    cell = rs.cell_name(model, fmt, P)
                    if not all(rs.final_path(out, cell, o).exists() for o in fam):
                        continue
                    X = {o: rs.read_json(rs.final_path(out, cell, o)) for o in fam}
                    single, hom, het = fam
                    b = {o: X[o]["best_main"] for o in fam}
                    cyc = {o: [b[o]["report"][f"B{B}"]["cycles"] for B in wl.BATCHES] for o in fam}
                    key, top10 = (model, fmt, P, S, "final"), (model, fmt, P, S, "top10")
                    row = dict(
                        key=f"{model}-{fmt}-{P}-s{S}",
                        model=wl.MODEL_NAMES[model],
                        quant=FORMAT_NAMES[fmt],
                        mac=f"{P // 1024}K",
                        streams=S,
                        het=_shape(b[het]["base"]),
                        hom=_shape(b[hom]["base"]),
                        single=_shape(b[single]["base"]),
                        het_pol=_policy(b[het]["policy"]),
                        hom_pol=_policy(b[hom]["policy"]),
                        single_pol=_policy(b[single]["policy"]),
                        mac_used={
                            o: sum(n * k for _, n, k in b[o]["base"])
                            * (2 if o.startswith("hom") and len(b[o]["base"]) == 1 else 1)
                            for o in fam
                        },
                        rescued=[o for o in fam if b[o].get("rescued")],
                        valid=[o for o in fam if b[o]["valid"]],
                        selected_from={o: X[o]["selected_from"] for o in fam},
                        provisional=[o for o in fam if X[o]["provisional"]],
                        shapes_mnk={o: [list(map(int, s)) for s in b[o]["base"]] for o in fam},
                        top10_vs_hom=point(top10, "heldout", "vs_hom"),
                        top10_vs_single=point(top10, "heldout", "vs_single"),
                        cycles=cyc,
                        vs_hom=[cyc[hom][i] / cyc[het][i] for i in range(8)],
                        vs_single=[cyc[single][i] / cyc[het][i] for i in range(8)],
                        vs_hom_main=_gm(cyc[hom][4:]) / _gm(cyc[het][4:]),
                        vs_single_main=_gm(cyc[single][4:]) / _gm(cyc[het][4:]),
                        vs_best_main=min(_gm(cyc[hom][4:]), _gm(cyc[single][4:])) / _gm(cyc[het][4:]),
                        vs_hom_ci=interval(key, "heldout", "vs_hom"),
                        vs_single_ci=interval(key, "heldout", "vs_single"),
                        vs_best_ci=interval(key, "heldout", "vs_best"),
                        sealed_vs_hom=point(key, "sealed", "vs_hom"),
                        sealed_vs_hom_ci=interval(key, "sealed", "vs_hom"),
                        sealed_vs_single=point(key, "sealed", "vs_single"),
                    )
                    if S == 2:
                        full, cap = rs.search_path(out, "fullhom", cell, hom), rs.search_path(out, "hetcap", cell, het)
                        if full.exists():
                            F = rs.read_json(full)["best_main"]
                            row.update(
                                fullhom=_shape(F["base"]), vs_fullhom_main=F["report_main"] / b[het]["report_main"]
                            )
                        if cap.exists():
                            C = rs.read_json(cap)
                            row.update(
                                hetcap=_shape(C["best_main"]["base"]),
                                hetcap_mac=C.get("mac_cap"),
                                hetcap_vs_hom_main=b[hom]["report_main"] / C["best_main"]["report_main"],
                            )
                    rows.append(row)
                    if model in ("qwen", "deepseek") and fmt in ("bf16", "w8g128") and row["vs_best_ci"]:
                        layer = interval(key, "heldout", "vs_best", "ci_layer")
                        verdict.append(
                            dict(
                                cell=row["key"],
                                streams=S,
                                vs_best=row["vs_best_main"],
                                ci_two_way=row["vs_best_ci"],
                                ci_layer=layer,
                                pass_two_way=row["vs_best_main"] >= 1.05 and row["vs_best_ci"][0] > 1,
                                pass_layer=row["vs_best_main"] >= 1.05 and layer[0] > 1,
                                rescued=bool(row["rescued"]),
                            )
                        )
    summary = {}
    for S in FAMILIES:
        v = [x for x in verdict if x["streams"] == S]
        summary[S] = dict(
            n=len(v),
            pass_two_way=sum(x["pass_two_way"] for x in v),
            pass_layer=sum(x["pass_layer"] for x in v),
            pass_two_way_without_rescue=sum(x["pass_two_way"] and not x["rescued"] for x in v),
        )
    return dict(rows=rows, verdict=verdict, verdict_summary=summary)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    a = ap.parse_args(argv)
    if a.out is None:
        ap.error("--out (or COMPLEMENTARY_ARRAYS_OUT) is required")
    res = export(a.out)
    rs.write_json(Path(a.out) / "export.json", res, indent=None)
    print(len(res["rows"]), "rows; verdict", res["verdict_summary"])


if __name__ == "__main__":
    main()
