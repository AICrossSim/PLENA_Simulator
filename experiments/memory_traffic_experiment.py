#!/usr/bin/env python3
"""
Memory Traffic Experiment: Flash Attention vs Standard Attention

This script compares memory traffic with and without flash attention
across different models and workloads, and generates visualization plots.

Usage:
    python experiments/memory_traffic_experiment.py
    python experiments/memory_traffic_experiment.py --output-dir ./results
    python experiments/memory_traffic_experiment.py --no-plot  # Skip plotting
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Try to import plotting libraries (optional)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/numpy not available, plotting disabled")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "analytic_models" / "memory"))

from llm_memory_model import LLMMemoryModel
from memory_model import load_memory_config_from_toml


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    model_name: str
    batch_size: int
    input_seq_len: int
    output_seq_len: int
    use_flash_attention: bool


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config: ExperimentConfig
    prefill_read_bytes: int
    prefill_write_bytes: int
    prefill_total_bytes: int
    decode_read_bytes: int
    decode_write_bytes: int
    decode_total_bytes: int


def run_experiment(
    model_path: str,
    memory_config_path: str,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run a single experiment and return results."""
    memory_config = load_memory_config_from_toml(memory_config_path)

    model = LLMMemoryModel(
        model_config_path=model_path,
        memory_config=memory_config,
        batch_size=config.batch_size,
        input_seq_len=config.input_seq_len,
        output_seq_len=config.output_seq_len,
        use_flash_attention=config.use_flash_attention,
    )

    # Compute traffic
    prefill_traffic = model.compute_prefill_traffic()
    decode_traffic = model.compute_decode_traffic(num_output_tokens=0)  # First decode token

    return ExperimentResult(
        config=config,
        prefill_read_bytes=prefill_traffic.total_traffic.read_bytes,
        prefill_write_bytes=prefill_traffic.total_traffic.write_bytes,
        prefill_total_bytes=prefill_traffic.total_traffic.total_bytes,
        decode_read_bytes=decode_traffic.total_traffic.read_bytes,
        decode_write_bytes=decode_traffic.total_traffic.write_bytes,
        decode_total_bytes=decode_traffic.total_traffic.total_bytes,
    )


def run_model_comparison(
    model_lib_path: Path,
    memory_config_path: str,
    models: list[str],
    batch_size: int = 1,
    input_seq_len: int = 2048,
    output_seq_len: int = 128,
) -> dict[str, dict[str, ExperimentResult]]:
    """Compare flash vs standard attention across multiple models."""
    results = {}

    for model_name in models:
        model_path = model_lib_path / f"{model_name}.json"
        if not model_path.exists():
            print(f"Warning: Model {model_name} not found, skipping...")
            continue

        results[model_name] = {}

        for use_flash in [True, False]:
            config = ExperimentConfig(
                model_name=model_name,
                batch_size=batch_size,
                input_seq_len=input_seq_len,
                output_seq_len=output_seq_len,
                use_flash_attention=use_flash,
            )

            result = run_experiment(str(model_path), memory_config_path, config)
            key = "flash" if use_flash else "standard"
            results[model_name][key] = result
            print(f"  {model_name} ({key}): prefill={result.prefill_total_bytes / 1e9:.2f} GB, "
                  f"decode={result.decode_total_bytes / 1e6:.2f} MB")

    return results


def run_workload_comparison(
    model_lib_path: Path,
    memory_config_path: str,
    model_name: str,
    workloads: list[dict],
) -> list[dict[str, ExperimentResult]]:
    """Compare flash vs standard attention across different workloads."""
    model_path = model_lib_path / f"{model_name}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found")

    results = []

    for workload in workloads:
        workload_results = {"workload": workload}

        for use_flash in [True, False]:
            config = ExperimentConfig(
                model_name=model_name,
                batch_size=workload.get("batch_size", 1),
                input_seq_len=workload["input_seq_len"],
                output_seq_len=workload.get("output_seq_len", 128),
                use_flash_attention=use_flash,
            )

            result = run_experiment(str(model_path), memory_config_path, config)
            key = "flash" if use_flash else "standard"
            workload_results[key] = result

        results.append(workload_results)
        print(f"  input_seq={workload['input_seq_len']}: "
              f"flash={workload_results['flash'].prefill_total_bytes / 1e9:.2f} GB, "
              f"standard={workload_results['standard'].prefill_total_bytes / 1e9:.2f} GB")

    return results


def plot_model_comparison(
    results: dict[str, dict[str, ExperimentResult]],
    output_path: Path,
    title: str = "Memory Traffic: Flash vs Standard Attention",
):
    """Plot memory traffic comparison across models."""
    models = list(results.keys())
    n_models = len(models)

    # Extract data
    flash_prefill = [results[m]["flash"].prefill_total_bytes / 1e9 for m in models]
    std_prefill = [results[m]["standard"].prefill_total_bytes / 1e9 for m in models]
    flash_decode = [results[m]["flash"].decode_total_bytes / 1e6 for m in models]
    std_decode = [results[m]["standard"].decode_total_bytes / 1e6 for m in models]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Prefill comparison
    x = np.arange(n_models)
    width = 0.35

    ax1 = axes[0]
    bars1 = ax1.bar(x - width / 2, flash_prefill, width, label="Flash Attention", color="#2ecc71")
    bars2 = ax1.bar(x + width / 2, std_prefill, width, label="Standard Attention", color="#e74c3c")

    ax1.set_xlabel("Model")
    ax1.set_ylabel("Memory Traffic (GB)")
    ax1.set_title("Prefill Phase")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f"{height:.1f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

    # Decode comparison
    ax2 = axes[1]
    bars3 = ax2.bar(x - width / 2, flash_decode, width, label="Flash Attention", color="#2ecc71")
    bars4 = ax2.bar(x + width / 2, std_decode, width, label="Standard Attention", color="#e74c3c")

    ax2.set_xlabel("Model")
    ax2.set_ylabel("Memory Traffic (MB)")
    ax2.set_title("Decode Phase (First Token)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f"{height:.0f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f"{height:.0f}",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "model_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/model_comparison.png")
    plt.close()


def plot_workload_comparison(
    results: list[dict],
    output_path: Path,
    model_name: str,
):
    """Plot memory traffic comparison across workloads."""
    seq_lens = [r["workload"]["input_seq_len"] for r in results]

    flash_prefill = [r["flash"].prefill_total_bytes / 1e9 for r in results]
    std_prefill = [r["standard"].prefill_total_bytes / 1e9 for r in results]

    # Calculate savings
    savings = [(s - f) / s * 100 for f, s in zip(flash_prefill, std_prefill)]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Traffic comparison
    ax1 = axes[0]
    ax1.plot(seq_lens, flash_prefill, "o-", label="Flash Attention", color="#2ecc71", linewidth=2, markersize=8)
    ax1.plot(seq_lens, std_prefill, "s-", label="Standard Attention", color="#e74c3c", linewidth=2, markersize=8)
    ax1.set_xlabel("Input Sequence Length")
    ax1.set_ylabel("Prefill Memory Traffic (GB)")
    ax1.set_title(f"Memory Traffic vs Sequence Length ({model_name})")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log", base=2)

    # Savings percentage
    ax2 = axes[1]
    bars = ax2.bar(range(len(seq_lens)), savings, color="#3498db")
    ax2.set_xlabel("Input Sequence Length")
    ax2.set_ylabel("Memory Savings (%)")
    ax2.set_title("Flash Attention Memory Savings")
    ax2.set_xticks(range(len(seq_lens)))
    ax2.set_xticklabels([str(s) for s in seq_lens])
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, saving in zip(bars, savings):
        height = bar.get_height()
        ax2.annotate(f"{saving:.1f}%",
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path / "workload_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "workload_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/workload_comparison.png")
    plt.close()


def plot_read_write_breakdown(
    results: dict[str, dict[str, ExperimentResult]],
    output_path: Path,
):
    """Plot read/write breakdown for each model."""
    models = list(results.keys())
    n_models = len(models)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Flash Attention - Prefill
    ax = axes[0, 0]
    flash_read = [results[m]["flash"].prefill_read_bytes / 1e9 for m in models]
    flash_write = [results[m]["flash"].prefill_write_bytes / 1e9 for m in models]
    x = np.arange(n_models)
    ax.bar(x, flash_read, label="Read", color="#3498db")
    ax.bar(x, flash_write, bottom=flash_read, label="Write", color="#e67e22")
    ax.set_title("Flash Attention - Prefill")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Traffic (GB)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Standard Attention - Prefill
    ax = axes[0, 1]
    std_read = [results[m]["standard"].prefill_read_bytes / 1e9 for m in models]
    std_write = [results[m]["standard"].prefill_write_bytes / 1e9 for m in models]
    ax.bar(x, std_read, label="Read", color="#3498db")
    ax.bar(x, std_write, bottom=std_read, label="Write", color="#e67e22")
    ax.set_title("Standard Attention - Prefill")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Traffic (GB)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Flash Attention - Decode
    ax = axes[1, 0]
    flash_read = [results[m]["flash"].decode_read_bytes / 1e6 for m in models]
    flash_write = [results[m]["flash"].decode_write_bytes / 1e6 for m in models]
    ax.bar(x, flash_read, label="Read", color="#3498db")
    ax.bar(x, flash_write, bottom=flash_read, label="Write", color="#e67e22")
    ax.set_title("Flash Attention - Decode")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Traffic (MB)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Standard Attention - Decode
    ax = axes[1, 1]
    std_read = [results[m]["standard"].decode_read_bytes / 1e6 for m in models]
    std_write = [results[m]["standard"].decode_write_bytes / 1e6 for m in models]
    ax.bar(x, std_read, label="Read", color="#3498db")
    ax.bar(x, std_write, bottom=std_read, label="Write", color="#e67e22")
    ax.set_title("Standard Attention - Decode")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Traffic (MB)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Read/Write Traffic Breakdown", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "read_write_breakdown.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "read_write_breakdown.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/read_write_breakdown.png")
    plt.close()


def save_results_json(
    model_results: dict,
    workload_results: list,
    output_path: Path,
):
    """Save results to JSON file."""
    data = {
        "model_comparison": {},
        "workload_comparison": [],
    }

    # Model comparison
    for model_name, results in model_results.items():
        data["model_comparison"][model_name] = {
            "flash": {
                "prefill_total_gb": results["flash"].prefill_total_bytes / 1e9,
                "decode_total_mb": results["flash"].decode_total_bytes / 1e6,
            },
            "standard": {
                "prefill_total_gb": results["standard"].prefill_total_bytes / 1e9,
                "decode_total_mb": results["standard"].decode_total_bytes / 1e6,
            },
        }

    # Workload comparison
    for result in workload_results:
        data["workload_comparison"].append({
            "input_seq_len": result["workload"]["input_seq_len"],
            "flash_prefill_gb": result["flash"].prefill_total_bytes / 1e9,
            "standard_prefill_gb": result["standard"].prefill_total_bytes / 1e9,
        })

    with open(output_path / "results.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}/results.json")


def main():
    parser = argparse.ArgumentParser(description="Memory Traffic Experiment")
    parser.add_argument(
        "--model-lib",
        default="compiler/doc/Model_Lib",
        help="Path to Model_Lib directory",
    )
    parser.add_argument(
        "--config",
        default="plena_settings.toml",
        help="Path to hardware config TOML",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/memory_traffic",
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["llama-3-8b", "llama-3.1-8b", "llama-3.1-70b", "qwen2_5_7b"],
        help="Models to compare",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation (useful if matplotlib is not available)",
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    model_lib_path = project_root / args.model_lib
    memory_config_path = str(project_root / args.config)
    output_path = project_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MEMORY TRAFFIC EXPERIMENT")
    print("=" * 70)

    # 1. Model comparison
    print("\n[1] Model Comparison (Flash vs Standard Attention)")
    print("-" * 70)
    model_results = run_model_comparison(
        model_lib_path=model_lib_path,
        memory_config_path=memory_config_path,
        models=args.models,
        batch_size=1,
        input_seq_len=2048,
        output_seq_len=128,
    )

    # 2. Workload comparison (varying sequence length)
    print("\n[2] Workload Comparison (Varying Sequence Length)")
    print("-" * 70)
    workloads = [
        {"input_seq_len": 512},
        {"input_seq_len": 1024},
        {"input_seq_len": 2048},
        {"input_seq_len": 4096},
        {"input_seq_len": 8192},
    ]
    workload_results = run_workload_comparison(
        model_lib_path=model_lib_path,
        memory_config_path=memory_config_path,
        model_name="llama-3-8b",
        workloads=workloads,
    )

    # 3. Generate plots
    print("\n[3] Generating Plots")
    print("-" * 70)
    if args.no_plot:
        print("  Skipping plots (--no-plot flag)")
    elif not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not available)")
    else:
        plot_model_comparison(model_results, output_path)
        plot_workload_comparison(workload_results, output_path, "llama-3-8b")
        plot_read_write_breakdown(model_results, output_path)

    # 4. Save results
    print("\n[4] Saving Results")
    print("-" * 70)
    save_results_json(model_results, workload_results, output_path)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
