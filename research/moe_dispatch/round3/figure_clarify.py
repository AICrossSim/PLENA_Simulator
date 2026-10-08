"""Clarify figure labels using saved observations only; no simulation imports.

The original five PDFs are preserved. Numerical sources and their generators
are read-only. Figure payloads and source SHA receipts make the plotted values
auditable independently of their presentation labels.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
BATCHES = (2, 4, 8, 16, 64, 96, 128)
METHODS = ("nominal", "random", "static", "btb", "ema", "ours")
FIGURES = ("fig_headroom_vs_bw", "fig_ablation", "fig_gpqa_inflight", "fig_main_by_batch", "fig_predictor")
SOURCES = ("E2/bounds_summary.csv", "E3/ablation.csv", "E3/gpqa_full_results.json.gz",
           "E4/heldout_main_table.csv", "E5/predictor/predictor_table.csv")
PROTECTED = ("model.py", "runtime.py", "optimizer.py", "config.py", "search.py", "sensitivity.py",
             "ablation.py", "evaluations.py", "bounds.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def curve_points(segments):
    """Convert existing arrays to display units; verify the stated proxy."""
    for segment in segments:
        rates = segment["hbm_rate_Bpc_core"]
        if not math.isclose(sum(rates), segment["hbm_rate_Bpc"], rel_tol=1e-12, abs_tol=1e-8):
            raise AssertionError("Saved global/core HBM rates differ")
        for rate, proxy in zip(rates, segment["inflight_bytes"]):
            if not math.isclose(proxy, rate * 65, rel_tol=1e-12, abs_tol=1e-8):
                raise AssertionError("Saved curve does not satisfy the 65ns service-equivalent proxy")
    return {"time_ms": [s["start"] / 1e6 for s in segments],
            "core0_proxy_KiB": [s["inflight_bytes"][0] / 1024 for s in segments],
            "core1_proxy_KiB": [s["inflight_bytes"][1] / 1024 for s in segments],
            "global_HBM_GBps": [s["hbm_rate_Bpc"] for s in segments]}


def payloads(root=ROOT):
    bounds = read_csv(root / SOURCES[0])
    ablation = read_csv(root / SOURCES[1])
    with gzip.open(root / SOURCES[2], "rt") as f:
        gpqa = json.load(f)
    main = read_csv(root / SOURCES[3])
    predictor = read_csv(root / SOURCES[4])
    headroom = []
    for batch in (*BATCHES, "all"):
        rows = sorted((r for r in bounds if r["set"] == "heldout" and r["batch"] == str(batch)),
                      key=lambda r: float(r["bw_GBps"]))
        headroom.append(dict(batch=batch, service_cap_GBps=[float(r["bw_GBps"]) for r in rows],
                             max_gain_pct=[float(r["max_gain_vs_b1_pct"]) for r in rows]))
    ab = [dict(config=r["config"], latency_GM_ms=float(r["geomean_ms"]), iso=r["iso"] == "True")
          for r in ablation if r["batch"] == "all"]
    curves = {label: curve_points(gpqa[label]["segments"]) for label in ("H0", "H3", "H2") if label in gpqa}
    names = ("B1", "B2", "H51", "H42", "H33")
    main_rows = {r["design"]: r for r in main if r["constraint_group"] == "C0" and
                 r["onchip_mode"] == "pipelined" and r["dispatch"] == "fixed"}
    main_values = {name: [float(main_rows[name][f"B{batch}"]) for batch in BATCHES] for name in names}
    pred = {}
    for mode in ("pipelined", "port_tight"):
        rows = [r for r in predictor if float(r["bw_GBps"]) == 256 and r["onchip_mode"] == mode and r["method"] != "oracle"]
        pred[mode] = {name: [float(next(r for r in rows if r["design"] == name and r["method"] == method)["ratio_vs_no_pred"])
                            for method in METHODS] for name in sorted({r["design"] for r in rows})}
    return dict(headroom=headroom, ablation=ab, gpqa=curves, main=main_values, predictor=pred)


def plt_module():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def gpqa_figure(curves):
    plt = plt_module()
    labels = list(curves)
    fig, axes = plt.subplots(len(labels), 1, figsize=(12, 3.3 * len(labels)))
    if len(labels) == 1:
        axes = [axes]
    for label, ax in zip(labels, axes):
        points = curves[label]
        for core in (0, 1):
            ax.step(points["time_ms"], points[f"core{core}_proxy_KiB"], where="post", label=f"core{core} service-equivalent bytes")
        other = ax.twinx()
        other.step(points["time_ms"], points["global_HBM_GBps"], color="#d95f02", alpha=.55,
                   where="post", label="global HBM supply rate")
        ax.set(title=f"{label}: GPQA B128 layer 13; BF16 phase-fluid analytical estimate",
               xlabel="Time (ms; 1 model cycle = 1 ns)", ylabel="Service-equivalent bytes\n(KiB; proxy)")
        other.set_ylabel("Global HBM supply rate\n(GB/s; analytical)")
        lines, names = ax.get_legend_handles_labels()
        right_lines, right_names = other.get_legend_handles_labels()
        ax.legend(lines + right_lines, names + right_names, loc="upper left", fontsize=8)
    fig.text(.5, .01,
             "Proxy = current per-core HBM service rate × 65 ns; not outstanding-request occupancy.\n"
             "single_fetcher means exactly one positive supply stream, not one core with in-flight requests.\n"
             "GB/s uses decimal bytes; KiB uses 1024 bytes. 256 GB/s is the shared analytical service cap.",
             ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    return fig


def draw(data, root=ROOT):
    plt = plt_module()
    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    for series in data["headroom"]:
        batch = series["batch"]
        ax.plot(series["service_cap_GBps"], series["max_gain_pct"], marker="o",
                linewidth=2.5 if batch == "all" else 1, label="All (paired GM)" if batch == "all" else f"B{batch}")
    ax.axhline(5, color="black", linestyle="--", label="5% calibration gate")
    ax.set(title="BF16 pipelined: mandatory-floor headroom vs historical 126-selected B1",
           xlabel="Shared HBM service cap (GB/s; analytical)",
           ylabel="Maximum possible latency reduction\nvs historical frozen B1 (%)")
    ax.grid(alpha=.25); ax.legend(ncol=3, fontsize=8)
    fig.text(.5, .01, "135 held-out windows; 100 × [1 − GM(per-window lower bound / B1 latency)].", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1)); fig.savefig(figures / "fig_headroom_vs_bw.pdf"); plt.close(fig)

    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(10, 4.8))
    rows = data["ablation"]
    bars = ax.bar([r["config"] for r in rows], [r["latency_GM_ms"] for r in rows],
                  color=["#377eb8" if r["iso"] else "#aaaaaa" for r in rows])
    for bar, row in zip(bars, rows):
        if not row["iso"]:
            bar.set_hatch("//")
    ax.set(title="BF16 post-router FFN; 256 GB/s service cap; phase-fluid analytical estimate",
           ylabel="135 held-out windows: latency GM (ms)", xlabel="126-selected frozen hardware and buffer ablations")
    ax.legend(handles=[Patch(facecolor="#377eb8", label="Iso-resource"),
                       Patch(facecolor="#aaaaaa", hatch="//", label="Non-iso infinite-W diagnostic")], fontsize=9)
    fig.tight_layout(); fig.savefig(figures / "fig_ablation.pdf"); plt.close(fig)

    fig = gpqa_figure(data["gpqa"])
    fig.savefig(figures / "fig_gpqa_inflight.pdf"); plt.close(fig)

    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 4.6)); xs = np.arange(len(BATCHES)); width = .16
    for i, (name, values) in enumerate(data["main"].items()):
        ax.bar(xs + (i - 2) * width, values, width, label=name)
    ax.set_xticks(xs, [f"B{batch}" for batch in BATCHES])
    ax.set(ylabel="MoE layer analytical latency (ms, GM)", xlabel="Held-out batch",
           title="BF16 pipelined; 256 GB/s service cap; 256-selected C0; fixed dispatch")
    ax.legend(ncol=5); ax.grid(axis="y", alpha=.2); fig.tight_layout()
    fig.savefig(figures / "fig_main_by_batch.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (mode, rows) in zip(axes, data["predictor"].items()):
        for name, values in rows.items():
            ax.plot(METHODS, values, marker="o", label=name)
        ax.axhline(1, color="grey", linestyle="--")
        ax.set(title=mode, ylabel="Paired-GM latency / nominal (×)", xlabel="Service-time estimator")
        ax.legend()
    fig.suptitle("BF16 phase-fluid analytical estimate; 256 GB/s service cap; 135 held-out windows")
    fig.text(.5, .01, "Dimensionless paired per-window latency ratios. Oracle is an accuracy reference and is omitted here.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .93)); fig.savefig(figures / "fig_predictor.pdf"); plt.close(fig)


def main():
    before = {name: sha(ROOT / name) for name in SOURCES + PROTECTED}
    archive = ROOT / "archive/figures_before_clarification"
    archive.mkdir(parents=True, exist_ok=True)
    for name in FIGURES:
        target = archive / (name + ".pdf")
        if not target.exists():
            shutil.copy2(ROOT / "figures" / target.name, target)
    data = payloads()
    draw(data)
    after = {name: sha(ROOT / name) for name in SOURCES + PROTECTED}
    if before != after:
        raise AssertionError("Numerical inputs or frozen numerical source changed during figure generation")
    output = ROOT / "figures/CLARIFIED_PLOT_VALUES.json"
    output.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    receipt = dict(scope="presentation-only redraw from unchanged saved values; no numerical simulation/search",
                   source_sha256={name: before[name] for name in SOURCES},
                   protected_numerical_source_sha256={name: before[name] for name in PROTECTED},
                   numerical_inputs_unchanged=True, script_sha256=sha(Path(__file__)),
                   plotted_values_sha256=sha(output),
                   prior_PDF_sha256={name: sha(archive / (name + ".pdf")) for name in FIGURES},
                   clarified_PDF_sha256={name: sha(ROOT / "figures" / (name + ".pdf")) for name in FIGURES})
    (ROOT / "figures/CLARIFICATION_RECEIPT.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"figures": len(FIGURES), "numerical_inputs_unchanged": True,
                      "receipt": "figures/CLARIFICATION_RECEIPT.json"}), flush=True)


if __name__ == "__main__":
    main()
