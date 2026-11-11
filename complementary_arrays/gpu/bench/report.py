"""Summarize the GPU measurements: kernel time against the weight-read floor, and the MoE share of the decode step."""

import argparse
import json
from pathlib import Path

MODELS = {"qwen": "Qwen3.5-35B-A3B", "deepseek": "DeepSeek-V2-Lite"}
FORMATS = {"bf16": "BF16", "w8g128": "INT8, group 128", "w4g32": "INT4, group 32"}
KERNELS = {"triton": "vLLM Triton", "flashinfer": "FlashInfer CUTLASS", "trtllm": "TensorRT-LLM", "marlin": "Marlin"}
QUANTIZATION = {"none": "BF16", "fp8": "FP8"}


def read_jsonl(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def table(head, rows):
    return (
        ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
        + [""]
    )


def kernel_section(summary, hbm_tbps):
    """Per model, format and batch: the fastest kernel's layer time and its fraction of the read floor."""
    best = {}
    for r in summary:  # a later measurement of the same kernel replaces an earlier one
        best.setdefault((r["model"], r["format"], r["batch"]), {})[r["kernel"]] = r
    batches = sorted({r["batch"] for r in summary})
    lines, fractions = [], []
    for model, name in MODELS.items():
        rows = []
        for fmt, label in FORMATS.items():
            cells, used = [], set()
            for b in batches:
                if (model, fmt, b) not in best:
                    cells.append("-")
                    continue
                r = min(best[(model, fmt, b)].values(), key=lambda r: r["layer_us"])
                fraction = r["bytes"] / (hbm_tbps * 1e6) / r["layer_us"]
                fractions.append(fraction)
                used.add(KERNELS[r["kernel"]])
                cells.append(f"{r['layer_us']:.1f} ({100 * fraction:.0f}%)")
            if used:
                rows.append([label, *cells, ", ".join(sorted(used))])
        if rows:
            lines += [f"**{name}**", "", *table(["Format", *[f"B{b}" for b in batches], "Fastest kernels"], rows)]
    if fractions:
        lines.insert(
            0,
            f"The fastest kernel per format reaches {100 * min(fractions):.0f}-{100 * max(fractions):.0f}% of the read floor.\n",
        )
    return lines


def e2e_section(records):
    """Per quantization and batch: decode step with and without MoE layers and the MoE share."""
    steps = {(r["quantization"], r["batch"], r["variant"]): r["decode_step_ms"] for r in records}
    rows, shares = [], []
    for quant, b in sorted({(q, b) for q, b, _ in steps}, key=lambda k: (list(QUANTIZATION).index(k[0]), k[1])):
        if (quant, b, "full") in steps and (quant, b, "no_moe") in steps:
            full, rest = steps[(quant, b, "full")], steps[(quant, b, "no_moe")]
            shares.append((full - rest) / full)
            rows.append(
                [QUANTIZATION[quant], b, f"{full:.2f}", f"{rest:.2f}", f"{full - rest:.2f}", f"{100 * shares[-1]:.0f}%"]
            )
    head = ["Weights", "Batch", "Step (ms)", "Step without MoE (ms)", "MoE (ms)", "MoE share"]
    lines = table(head, rows)
    if shares:
        lines.insert(0, f"MoE layers take {100 * min(shares):.0f}-{100 * max(shares):.0f}% of a decode step.\n")
    return lines


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kernels", type=Path, required=True, help="moe_layer.py output directory")
    p.add_argument("--e2e", type=Path, help="e2e_moe_share.py output directory")
    p.add_argument("--hbm-tbps", type=float, default=8.0, help="HBM bandwidth of the read floor")
    p.add_argument("--out", type=Path, help="markdown file")
    a = p.parse_args()
    lines = [
        "# GPU baseline",
        "",
        f"## One MoE decode layer: time in us (fraction of the weight-read floor at {a.hbm_tbps:g} TB/s)",
        "",
        *kernel_section(read_jsonl(a.kernels / "summary.jsonl"), a.hbm_tbps),
    ]
    if a.e2e:
        lines += ["## DeepSeek-V2-Lite decode step in vLLM", "", *e2e_section(read_jsonl(a.e2e / "e2e_ablation.jsonl"))]
    text = "\n".join(lines) + "\n"
    if a.out:
        a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
