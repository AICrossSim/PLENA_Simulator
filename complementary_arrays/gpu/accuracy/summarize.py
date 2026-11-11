"""Accuracy tables from evaluate.py results, each format against its model's reference format.

Differences carry paired bootstrap 95% intervals: the same items are resampled with replacement.
"""

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = {"qwen35": "Qwen3.5-35B-A3B", "deepseek": "DeepSeek-V2-Lite-Chat", "qwen3next": "Qwen3-Next-80B-A3B-Instruct"}
FORMATS = {"bf16": "BF16", "fp8": "FP8 (checkpoint)", "w8g128": "INT8, group 128", "w4g32": "INT4, group 32"}
BITS_PER_WEIGHT = {"bf16": 16, "fp8": 8 + 32 / (128 * 128), "w8g128": 8 + 16 / 128, "w4g32": 4 + 16 / 32}
METRICS = (
    ("mmlu", "acc", "correct"),
    ("gsm8k", "acc", "correct"),
    ("arc_c", "acc_norm", "correct_norm"),
    ("arc_e", "acc_norm", "correct_norm"),
    ("hellaswag", "acc_norm", "correct_norm"),
    ("piqa", "acc_norm", "correct_norm"),
    ("winogrande", "acc", "correct"),
    ("lambada", "acc", "correct"),
)


def paired_bootstrap(x, y, draws, seed):
    """Mean of x - y and the 2.5 / 97.5 percentiles of the mean over `draws` item resamples."""
    d = np.asarray(x, float) - np.asarray(y, float)
    means = d[np.random.default_rng(seed).integers(0, len(d), (draws, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def model_table(results, draws, seed):
    reference = "bf16" if "bf16" in results else "fp8"
    ref = results[reference]["tasks"]
    head = ["Format", "Bits/weight", "Weight rel. L2", "WikiText-2 ppl"] + [f"{t} {m}" for t, m, _ in METRICS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for fmt in (f for f in FORMATS if f in results):
        tasks, quant = results[fmt]["tasks"], results[fmt].get("quantization")
        ppl = f"{tasks['wikitext2']['perplexity']:.3f}" if "wikitext2" in tasks else "-"
        row = [FORMATS[fmt], f"{BITS_PER_WEIGHT[fmt]:.2f}", f"{quant['rel_l2']:.4f}" if quant else "0", ppl]
        for task, metric, items in METRICS:
            if task not in tasks:
                row.append("-")
                continue
            cell = f"{100 * tasks[task][metric]:.1f}"
            if fmt != reference and task in ref:
                d, lo, hi = paired_bootstrap(tasks[task][items], ref[task][items], draws, seed)
                cell += f" ({100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}])"
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    gsm = [
        f"{FORMATS[f]}: {r['tasks']['gsm8k']['mean_output_tokens']:.0f} tokens, {r['tasks']['gsm8k']['truncated']} truncated"
        for f, r in results.items()
        if "gsm8k" in r["tasks"]
    ]
    if gsm:
        lines += ["", "GSM8K mean output length and truncated answers: " + "; ".join(gsm) + "."]
    return reference, lines


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results", type=Path, help="evaluate.py output directory (<model-kind>/<format>.json)")
    p.add_argument("--out", type=Path, help="markdown file, default <results>/accuracy.md")
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=37)
    a = p.parse_args()
    lines = [
        "# Accuracy under expert-weight formats",
        "",
        "Cells: accuracy in % (difference to the reference in points [paired bootstrap 95% interval, "
        f"{a.draws} draws]). WikiText-2 is perplexity.",
        "",
    ]
    for kind, name in MODELS.items():
        files = {f.stem: f for f in (a.results / kind).glob("*.json")}
        results = {fmt: json.loads(files[fmt].read_text()) for fmt in FORMATS if fmt in files}
        if "bf16" not in results and "fp8" not in results:
            continue
        reference, table = model_table(results, a.draws, a.seed)
        lines += [f"## {name} (reference: {FORMATS[reference]})", "", *table, ""]
    text = "\n".join(lines) + "\n"
    (a.out or a.results / "accuracy.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
