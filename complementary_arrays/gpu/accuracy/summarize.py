"""Accuracy tables of the paper from the stored results: BF16 rows are scores, other rows changes in points against BF16
(paired over items; '*' marks a paired bootstrap 95% interval, 2,000 draws, seed 37, that excludes zero).

  standard    emulated formats (evaluate.py): WikiText-2 perplexity, MMLU, GSM8K (1,024-token cap), ARC-c, ARC-e,
              HellaSwag (normalized), PIQA, WinoGrande, LAMBADA, GPQA diagnostic
  reasoning   GPU kernels (serve.py + generate.py / bfcl.py / lcb_score.sh / swebench.py): GPQA Diamond x8, MATH-500,
              AIME x8, LiveCodeBench x4 (Qwen3.5 with the 81,920-token runs), BFCL non-live / live / multi-turn (BFCL's
              class scores; changes paired over items), SWE-bench Verified resolved of 100; KV-cache rows add the
              change of an emulated KV-cache format against the BF16 cache to the NVFP4 row
  main        the body's table: perplexity and MMLU (emulated), GSM8K (4,096-token runs where present, else the
              1,024-token cap, marked with a dagger), MATH-500, LiveCodeBench and BFCL live (GPU kernels)

GPQA answers of the generation runs: the simple-evals 'Answer: X' line, else the last 'answer is X', '**X)', line-initial
'X)' or 'option X' (DeepSeek-V2-Lite often omits the answer line).

usage: python -m complementary_arrays.gpu.accuracy.summarize OUT_STEM --emulated DIR [--gsm4096 DIR] --runs DIR
       --tasks DIR [--kv mxint8=DIR --kv mxint4=DIR] [--swebench DIR]
  DIR layouts: emulated and gsm4096 <kind>/<format>.json; runs and every kv DIR <kind>_<mode>/ (generate.py,
  lcb_score.sh and bfcl.py outputs); swebench runs/<format>/<instance>/done.json.
"""

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np

KINDS = {"qwen35": "Qwen3.5", "deepseek": "DeepSeek-V2-Lite"}
EMULATED = (("bf16", "BF16"), ("w8g128", "INT8"), ("fp8b128", "FP8"), ("w4g32", "INT4"), ("nvfp4", "NVFP4"))
STANDARD = (
    ("mmlu", "acc", "correct", "MMLU"),
    ("gsm8k", "acc", "correct", "GSM8K"),
    ("arc_c", "acc", "correct", "ARC-c"),
    ("arc_e", "acc", "correct", "ARC-e"),
    ("hellaswag", "acc_norm", "correct_norm", "HellaSwag"),
    ("piqa", "acc", "correct", "PIQA"),
    ("winogrande", "acc", "correct", "WinoGrande"),
    ("lambada", "acc", "correct", "LAMBADA"),
    ("gpqa", "acc", "correct", "GPQA"),
)
GPU = (("bf16", "BF16"), ("fp8_w8a16", "FP8"), ("nvfp4_w4a16", "NVFP4"))
EMULATED_OF = {"bf16": "bf16", "fp8_w8a16": "fp8b128", "nvfp4_w4a16": "nvfp4"}
TASKS = {"qwen35": ("gpqa", "math500", "aime_long", "lcb_long"), "deepseek": ("gpqa", "math500", "aime", "lcb")}
BFCL = (
    ("non_live", "Non-Live Overall Acc", "data_non_live.csv"),
    ("live", "Live Overall Acc", "data_live.csv"),
    ("multi_turn", "Multi Turn Overall Acc", "data_multi_turn.csv"),
)
STRICT = re.compile(r"(?i)Answer[ \t]*:[ \t]*\$?\(?([A-D])\)?\$?")
LOOSE = [
    re.compile(p)
    for p in (
        r"(?i)answer is[:\s]*\**\(?([A-D])\b",
        r"\*\*\(?([A-D])\)",
        r"(?m)^\(?([A-D])\)",
        r"(?i)\boption\s+\(?([A-D])\b",
    )
]


def paired(x, y, draws=2000, seed=37):
    """Mean of x - y in points and whether the bootstrap 95% interval excludes zero."""
    d = np.asarray(x, float) - np.asarray(y, float)
    means = d[np.random.default_rng(seed).integers(0, len(d), (draws, len(d)))].mean(1)
    lo, hi = np.percentile(means, 2.5), np.percentile(means, 97.5)
    return 100 * d.mean(), bool(lo > 0 or hi < 0)


def emulated(root, kind):
    out = {}
    for name in (kind, "qwen" if kind == "qwen35" else kind):
        for f in (Path(root) / name).glob("*.json") if root else ():
            out.setdefault(f.stem, json.loads(f.read_text()).get("evals"))
    return out


def gpqa_answer(text):
    m = STRICT.findall(text or "")
    if m:
        return m[-1].upper()
    best = None
    for p in LOOSE:  # the last position of any pattern
        for x in p.finditer(text or ""):
            if best is None or x.start() > best[0]:
                best = (x.start(), x.group(1).upper())
    return best[1] if best else None


def task_scores(d, task, gold):
    """{item: mean score over its samples} of one generation run, or None."""
    if task.startswith("lcb"):
        p = d / f"{task}_scored.jsonl"
        return {r["id"]: r["score"] for r in map(json.loads, open(p))} if p.exists() else None
    p = d / f"{task}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(line) for line in open(p)]
    if task == "gpqa":
        for r in rows:
            for x in r["samples"]:
                x["score"] = float(gpqa_answer(x["final"]) == gold[r["id"]])
    return {r["id"]: float(np.mean([s["score"] for s in r["samples"]])) for r in rows}


def bfcl_scores(d, kind):
    """({class: BFCL's class score}, {class: {item: 0/1}}) of one run, relevance categories excluded as in BFCL's
    non-live and live summaries."""
    sd = d / "bfcl" / "score"
    if not (sd / "data_overall.csv").exists():
        return None, None
    official, items = {}, {}
    for cls, col, fn in BFCL:
        with open(sd / fn) as f:
            v = next(csv.DictReader(f)).get(col, "N/A")
        official[cls] = float(v.rstrip("%")) if v not in ("N/A", "") else None
        it = {}
        for p in glob.glob(str(sd / f"plena-{kind}" / cls / "*_score.json")):
            lines = [json.loads(line) for line in open(p)]
            fails = {r["id"] for r in lines[1:] if "id" in r}
            cat = Path(p).name.replace("BFCL_v4_", "").replace("_score.json", "")
            if cls != "multi_turn" and "relevance" in cat:
                continue
            res = d / "bfcl" / "result" / f"plena-{kind}" / cls / f"BFCL_v4_{cat}_result.json"
            for i in (json.loads(line)["id"] for line in open(res)) if res.exists() else ():
                it[i] = 0.0 if i in fails else 1.0
        items[cls] = it
    return official, items


def delta(x, ref):
    ids = sorted(set(x) & set(ref))
    return paired([x[i] for i in ids], [ref[i] for i in ids]) if ids else None


def fmt_delta(d, star=True):
    if d is None:
        return "-"
    return ("0.0" if d[0] == 0 else f"{d[0]:+.1f}") + ("*" if star and d[1] else "")


def standard_table(a):
    out, res = (
        [
            "| Model | Weights | WikiText-2 ppl. | " + " | ".join(s[3] for s in STANDARD) + " |",
            "|---|---|---|" + "---|" * len(STANDARD),
        ],
        {},
    )
    for kind, name in KINDS.items():
        e = emulated(a.emulated, kind)
        if "bf16" not in e:
            continue
        for fmt, label in EMULATED:
            if fmt not in e:
                continue
            row = [f"{e[fmt]['wikitext2']['perplexity']:.3f}"]
            for task, metric, items, _ in STANDARD:
                r, ref = e[fmt].get(task), e["bf16"].get(task)
                if r is None:
                    row.append("-")
                elif fmt == "bf16":
                    row.append(f"{100 * r[metric]:.1f}")
                else:
                    d = paired(r[items], ref[items])
                    row.append(fmt_delta(d, star=False))
                    res[f"{kind}|{fmt}|{task}"] = d[0]
            out.append(f"| {name if fmt == 'bf16' else ''} | {label} | " + " | ".join(row) + " |")
    return out, res


def reasoning_table(a):
    gold = {r["id"]: r["gold"] for r in map(json.loads, open(Path(a.tasks) / "gpqa.jsonl"))}
    heads = ["GPQA", "MATH-500", "AIME", "LCB", "BFCL non-live", "BFCL live", "BFCL multi-turn", "SWE-bench"]
    out = ["| Model | Weights | " + " | ".join(heads) + " |", "|---|---|" + "---|" * len(heads)]
    res = {}
    kv = dict(k.split("=", 1) for k in a.kv or [])
    for kind, name in KINDS.items():
        runs = {m: Path(a.runs) / f"{kind}_{m}" for m, _ in GPU if (Path(a.runs) / f"{kind}_{m}").exists()}
        if "bf16" not in runs:
            continue
        scores = {m: {t: task_scores(d, t, gold) for t in TASKS[kind]} for m, d in runs.items()}
        bfcl = {m: bfcl_scores(d, kind) for m, d in runs.items()}
        swe = {}
        for m, _ in GPU:
            done = list(Path(a.swebench).glob(f"runs/{m}/*/done.json")) if a.swebench and kind == "qwen35" else []
            if done:
                swe[m] = sum(json.loads(p.read_text())["resolved"] for p in done)
        for m, label in GPU:
            if m not in runs:
                continue
            row = []
            for t in TASKS[kind]:
                sc, ref = scores[m][t], scores["bf16"][t]
                if sc is None:
                    row.append("-")
                    continue
                v = 100 * np.mean(list(sc.values())) if m == "bf16" else delta(sc, ref)
                row.append(f"{v:.1f}" if m == "bf16" else fmt_delta(v))
                res[f"{kind}|{m}|{t}"] = v if m == "bf16" else v[0]
            official, items = bfcl[m]
            for cls, _, _ in BFCL:
                if not official or official.get(cls) is None:
                    row.append("-")
                    continue
                v = official[cls] if m == "bf16" else delta(items[cls], bfcl["bf16"][1][cls])
                row.append(f"{v:.1f}" if m == "bf16" else fmt_delta(v))
                res[f"{kind}|{m}|bfcl_{cls}"] = v if m == "bf16" else v[0]
            row.append(str(swe[m]) if m in swe else "-")
            out.append(f"| {name if m == 'bf16' else ''} | {label} | " + " | ".join(row) + " |")
        for kvf, root in kv.items():  # NVFP4 + change of the KV-cache format against the BF16 cache
            d = Path(root) / f"{kind}_nvfp4_w4a16"
            if not d.exists() or "nvfp4_w4a16" not in runs:
                continue
            row = []
            for t in TASKS[kind]:
                sc, ref = task_scores(d, t, gold), scores["nvfp4_w4a16"][t]
                if sc is None or ref is None:
                    row.append("-")
                    continue
                v = res[f"{kind}|nvfp4_w4a16|{t}"] + delta(sc, ref)[0]
                row.append(fmt_delta((v, False)))
                res[f"{kind}|nvfp4_w4a16+{kvf}|{t}"] = v
            label = f"NVFP4 + {kvf[2:].replace('int', '')}-bit KV cache"
            out.append(f"| | {label} | " + " | ".join(row) + " | - | - | - | - |")
    return out, res


def main_table(a, std, rsn):
    heads = ["PPL", "MMLU", "GSM8K", "MATH", "LCB", "BFCL"]
    out = ["| Model | Weights | " + " | ".join(heads) + " |", "|---|---|" + "---|" * len(heads)]
    for kind, name in KINDS.items():
        e, g = emulated(a.emulated, kind), emulated(a.gsm4096, kind)
        if "bf16" not in e:
            continue
        for m, label in GPU:
            f = EMULATED_OF[m]
            if f not in e:
                continue
            row = [f"{e[f]['wikitext2']['perplexity']:.2f}"]
            row.append(
                f"{100 * e[f]['mmlu']['acc']:.1f}" if m == "bf16" else fmt_delta((std[f"{kind}|{f}|mmlu"], False))
            )
            long = f in g and "bf16" in g and g[f].get("gsm8k")
            src, ref = (g, g["bf16"]) if long else (e, e["bf16"])
            if m == "bf16":
                row.append(f"{100 * src[f]['gsm8k']['acc']:.1f}")
            else:
                d = paired(src[f]["gsm8k"]["correct"], ref["gsm8k"]["correct"])
                row.append(f"{d[0]:+.1f}" + ("" if long else "†"))
            for key in (TASKS[kind][1], TASKS[kind][3], "bfcl_live"):
                v = rsn.get(f"{kind}|{m}|{key}")
                row.append("-" if v is None else f"{v:.1f}" if m == "bf16" else f"{v:+.1f}")
            out.append(f"| {name if m == 'bf16' else ''} | {label} | " + " | ".join(row) + " |")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path, help="output stem: OUT.md and OUT.json")
    p.add_argument("--emulated", type=Path, required=True)
    p.add_argument("--gsm4096", type=Path)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--kv", action="append", help="KV-cache format=directory of its runs")
    p.add_argument("--swebench", type=Path)
    a = p.parse_args()
    std, std_res = standard_table(a)
    rsn, rsn_res = reasoning_table(a)
    md = [
        "## Standard tasks, emulated formats",
        "",
        *std,
        "",
        "## Reasoning and tool use, GPU kernels",
        "",
        *rsn,
        "",
        "## Main table",
        "",
        *main_table(a, std_res, rsn_res),
        "",
    ]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".md").write_text("\n".join(md))
    a.out.with_suffix(".json").write_text(json.dumps(dict(standard=std_res, reasoning=rsn_res), indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
