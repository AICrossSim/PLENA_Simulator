#!/usr/bin/env python3
"""
Memory IO Experiment: HBM Bandwidth Utilization Analysis

This script analyzes memory IO bandwidth by combining:
1. Memory traffic model (bytes transferred to/from HBM)
2. Performance model (execution time)

Calculates achieved HBM bandwidth = traffic / execution_time for both:
- Prefill phase (process entire input sequence)
- Decode phase (generate single output token)

Usage:
    python experiments/memory_io_experiment.py
    python experiments/memory_io_experiment.py --models llama-3-8b llama-3.1-8b
    python experiments/memory_io_experiment.py --output-dir ./results --no-plot
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
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
sys.path.insert(0, str(Path(__file__).parent.parent / "analytic_models" / "performance"))

from llm_memory_model import LLMMemoryModel
from memory_model import load_memory_config_from_toml
from llama_model import LLaMAModel
from perf_model import load_hardware_config_from_toml


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    model_name: str
    batch_size: int
    input_seq_len: int
    output_seq_len: int
    use_flash_attention: bool = True
    frequency_hz: float = 1e9


@dataclass
class BandwidthResult:
    """Bandwidth analysis result for a single phase."""

    traffic_bytes: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    execution_time_seconds: float = 0.0  # compute-bound time
    achieved_bandwidth_gbps: float = 0.0
    peak_bandwidth_gbps: float = 0.0
    utilization: float = 0.0  # ratio (0-1)

    @property
    def memory_bound_time_seconds(self) -> float:
        """Time if limited by memory bandwidth (traffic / peak_bw)."""
        if self.peak_bandwidth_gbps > 0:
            return self.traffic_bytes / (self.peak_bandwidth_gbps * 1e9)
        return 0.0

    @property
    def is_memory_bound(self) -> bool:
        """True if memory-bound (utilization > 100%)."""
        return self.utilization > 1.0

    @property
    def effective_time_seconds(self) -> float:
        """Actual execution time (max of compute-bound and memory-bound)."""
        return max(self.execution_time_seconds, self.memory_bound_time_seconds)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config: ExperimentConfig
    prefill: BandwidthResult = field(default_factory=BandwidthResult)
    decode: BandwidthResult = field(default_factory=BandwidthResult)


def run_experiment(
    model_path: str,
    memory_config_path: str,
    hardware_config_path: str,
    isa_lib_path: str,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Run a single experiment and return bandwidth results."""
    # Load configurations
    memory_config = load_memory_config_from_toml(memory_config_path)
    hardware_config = load_hardware_config_from_toml(hardware_config_path)

    # Initialize memory model
    mem_model = LLMMemoryModel(
        model_config_path=model_path,
        memory_config=memory_config,
        batch_size=config.batch_size,
        input_seq_len=config.input_seq_len,
        output_seq_len=config.output_seq_len,
        use_flash_attention=config.use_flash_attention,
    )

    # Initialize performance model
    perf_model = LLaMAModel(
        model_config_path=model_path,
        hardware_config=hardware_config,
        custom_isa_path=isa_lib_path,
        batch_size=config.batch_size,
        input_seq_len=config.input_seq_len,
        output_seq_len=config.output_seq_len,
        frequency_hz=config.frequency_hz,
    )

    # Compute peak bandwidth (same for both phases)
    peak_bw = mem_model.mem.compute_peak_bandwidth(config.frequency_hz)

    result = ExperimentResult(config=config)

    # =========================================================================
    # Prefill Phase Analysis
    # =========================================================================
    prefill_traffic = mem_model.compute_prefill_traffic()
    prefill_time = perf_model.compute_prefill_time(verbose=False)

    result.prefill.traffic_bytes = prefill_traffic.total_traffic.total_bytes
    result.prefill.read_bytes = prefill_traffic.total_traffic.read_bytes
    result.prefill.write_bytes = prefill_traffic.total_traffic.write_bytes
    result.prefill.execution_time_seconds = prefill_time
    result.prefill.peak_bandwidth_gbps = peak_bw

    if prefill_time > 0:
        result.prefill.achieved_bandwidth_gbps = (
            prefill_traffic.total_traffic.total_bytes / prefill_time / 1e9
        )
        result.prefill.utilization = result.prefill.achieved_bandwidth_gbps / peak_bw

    # =========================================================================
    # Decode Phase Analysis (Single Token)
    # For decode, we analyze the traffic and time for generating a SINGLE token
    # - Memory traffic: num_output_tokens=0 means first decode token (KV cache = input_seq_len)
    # - Performance: compute_decode_time(1) gives time for 1 output token
    # =========================================================================
    decode_traffic = mem_model.compute_decode_traffic(num_output_tokens=0)
    decode_time = perf_model.compute_decode_time(output_token_size=1, verbose=False)

    result.decode.traffic_bytes = decode_traffic.total_traffic.total_bytes
    result.decode.read_bytes = decode_traffic.total_traffic.read_bytes
    result.decode.write_bytes = decode_traffic.total_traffic.write_bytes
    result.decode.execution_time_seconds = decode_time
    result.decode.peak_bandwidth_gbps = peak_bw

    if decode_time > 0:
        result.decode.achieved_bandwidth_gbps = (
            decode_traffic.total_traffic.total_bytes / decode_time / 1e9
        )
        result.decode.utilization = result.decode.achieved_bandwidth_gbps / peak_bw

    return result


def run_model_comparison(
    model_lib_path: Path,
    memory_config_path: str,
    hardware_config_path: str,
    isa_lib_path: str,
    models: list[str],
    batch_size: int = 1,
    input_seq_len: int = 2048,
    output_seq_len: int = 128,
) -> dict[str, ExperimentResult]:
    """Run bandwidth analysis across multiple models."""
    results = {}

    for model_name in models:
        model_path = model_lib_path / f"{model_name}.json"
        if not model_path.exists():
            print(f"Warning: Model {model_name} not found, skipping...")
            continue

        config = ExperimentConfig(
            model_name=model_name,
            batch_size=batch_size,
            input_seq_len=input_seq_len,
            output_seq_len=output_seq_len,
        )

        result = run_experiment(
            str(model_path),
            memory_config_path,
            hardware_config_path,
            isa_lib_path,
            config,
        )
        results[model_name] = result

        print(f"  {model_name}:")
        print(f"    Prefill: {result.prefill.traffic_bytes / 1e9:.2f} GB")
        if result.prefill.is_memory_bound:
            print(f"             Compute time: {result.prefill.execution_time_seconds * 1e3:.2f} ms")
            print(f"             Memory time:  {result.prefill.memory_bound_time_seconds * 1e3:.2f} ms (MEMORY-BOUND)")
        else:
            print(f"             Time: {result.prefill.execution_time_seconds * 1e3:.2f} ms")
        print(f"             BW: {result.prefill.achieved_bandwidth_gbps:.1f} / {result.prefill.peak_bandwidth_gbps:.0f} GB/s ({result.prefill.utilization * 100:.1f}%)")

        print(f"    Decode:  {result.decode.traffic_bytes / 1e6:.2f} MB (single token)")
        if result.decode.is_memory_bound:
            print(f"             Compute time: {result.decode.execution_time_seconds * 1e6:.2f} us")
            print(f"             Memory time:  {result.decode.memory_bound_time_seconds * 1e6:.2f} us (MEMORY-BOUND)")
        else:
            print(f"             Time: {result.decode.execution_time_seconds * 1e6:.2f} us")
        print(f"             BW: {result.decode.achieved_bandwidth_gbps:.1f} / {result.decode.peak_bandwidth_gbps:.0f} GB/s ({result.decode.utilization * 100:.1f}%)")

    return results


def run_sequence_length_comparison(
    model_lib_path: Path,
    memory_config_path: str,
    hardware_config_path: str,
    isa_lib_path: str,
    model_name: str,
    sequence_lengths: list[int],
    batch_size: int = 1,
) -> list[ExperimentResult]:
    """Analyze bandwidth utilization across different sequence lengths."""
    model_path = model_lib_path / f"{model_name}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found")

    results = []

    for seq_len in sequence_lengths:
        config = ExperimentConfig(
            model_name=model_name,
            batch_size=batch_size,
            input_seq_len=seq_len,
            output_seq_len=128,
        )

        result = run_experiment(
            str(model_path),
            memory_config_path,
            hardware_config_path,
            isa_lib_path,
            config,
        )
        results.append(result)

        print(f"  seq_len={seq_len}: "
              f"Prefill BW={result.prefill.achieved_bandwidth_gbps:.1f} GB/s ({result.prefill.utilization * 100:.1f}%), "
              f"Decode BW={result.decode.achieved_bandwidth_gbps:.1f} GB/s ({result.decode.utilization * 100:.1f}%)")

    return results


def run_batch_size_comparison(
    model_lib_path: Path,
    memory_config_path: str,
    hardware_config_path: str,
    isa_lib_path: str,
    model_name: str,
    batch_sizes: list[int],
    input_seq_len: int = 2048,
) -> list[ExperimentResult]:
    """Analyze bandwidth utilization across different batch sizes."""
    model_path = model_lib_path / f"{model_name}.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model {model_name} not found")

    results = []

    for batch_size in batch_sizes:
        config = ExperimentConfig(
            model_name=model_name,
            batch_size=batch_size,
            input_seq_len=input_seq_len,
            output_seq_len=128,
        )

        result = run_experiment(
            str(model_path),
            memory_config_path,
            hardware_config_path,
            isa_lib_path,
            config,
        )
        results.append(result)

        print(f"  batch_size={batch_size}: "
              f"Prefill BW={result.prefill.achieved_bandwidth_gbps:.1f} GB/s ({result.prefill.utilization * 100:.1f}%), "
              f"Decode BW={result.decode.achieved_bandwidth_gbps:.1f} GB/s ({result.decode.utilization * 100:.1f}%)")

    return results


def plot_model_comparison(
    results: dict[str, ExperimentResult],
    output_path: Path,
):
    """Plot bandwidth utilization comparison across models."""
    models = list(results.keys())
    n_models = len(models)

    # Extract data
    prefill_bw = [results[m].prefill.achieved_bandwidth_gbps for m in models]
    decode_bw = [results[m].decode.achieved_bandwidth_gbps for m in models]
    prefill_util = [results[m].prefill.utilization * 100 for m in models]
    decode_util = [results[m].decode.utilization * 100 for m in models]
    peak_bw = results[models[0]].prefill.peak_bandwidth_gbps

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x = np.arange(n_models)
    width = 0.35

    # Plot 1: Achieved Bandwidth (Prefill vs Decode)
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width / 2, prefill_bw, width, label="Prefill", color="#3498db")
    bars2 = ax1.bar(x + width / 2, decode_bw, width, label="Decode", color="#e74c3c")
    ax1.axhline(y=peak_bw, color="green", linestyle="--", linewidth=2, label=f"Peak ({peak_bw:.0f} GB/s)")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Achieved Bandwidth (GB/s)")
    ax1.set_title("Achieved HBM Bandwidth")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f"{height:.0f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f"{height:.0f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Plot 2: Bandwidth Utilization
    ax2 = axes[0, 1]
    bars3 = ax2.bar(x - width / 2, prefill_util, width, label="Prefill", color="#3498db")
    bars4 = ax2.bar(x + width / 2, decode_util, width, label="Decode", color="#e74c3c")
    ax2.axhline(y=100, color="green", linestyle="--", linewidth=2, label="100% Utilization")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Bandwidth Utilization (%)")
    ax2.set_title("HBM Bandwidth Utilization")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax2.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for bar in bars4:
        height = bar.get_height()
        ax2.annotate(f"{height:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Plot 3: Traffic Amount (Prefill)
    ax3 = axes[1, 0]
    prefill_read = [results[m].prefill.read_bytes / 1e9 for m in models]
    prefill_write = [results[m].prefill.write_bytes / 1e9 for m in models]
    ax3.bar(x, prefill_read, label="Read", color="#3498db")
    ax3.bar(x, prefill_write, bottom=prefill_read, label="Write", color="#e67e22")
    ax3.set_xlabel("Model")
    ax3.set_ylabel("Traffic (GB)")
    ax3.set_title("Prefill Memory Traffic (Read/Write)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha="right")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.3)

    # Plot 4: Traffic Amount (Decode - single token)
    ax4 = axes[1, 1]
    decode_read = [results[m].decode.read_bytes / 1e6 for m in models]
    decode_write = [results[m].decode.write_bytes / 1e6 for m in models]
    ax4.bar(x, decode_read, label="Read", color="#3498db")
    ax4.bar(x, decode_write, bottom=decode_read, label="Write", color="#e67e22")
    ax4.set_xlabel("Model")
    ax4.set_ylabel("Traffic (MB)")
    ax4.set_title("Decode Memory Traffic (Single Token, Read/Write)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha="right")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)

    plt.suptitle("Memory IO Bandwidth Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "model_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/model_comparison.png")
    plt.close()


def plot_sequence_length_comparison(
    results: list[ExperimentResult],
    output_path: Path,
    model_name: str,
):
    """Plot bandwidth utilization vs sequence length."""
    seq_lens = [r.config.input_seq_len for r in results]
    prefill_bw = [r.prefill.achieved_bandwidth_gbps for r in results]
    decode_bw = [r.decode.achieved_bandwidth_gbps for r in results]
    prefill_util = [r.prefill.utilization * 100 for r in results]
    decode_util = [r.decode.utilization * 100 for r in results]
    peak_bw = results[0].prefill.peak_bandwidth_gbps

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Achieved Bandwidth vs Sequence Length
    ax1 = axes[0]
    ax1.plot(seq_lens, prefill_bw, "o-", label="Prefill", color="#3498db", linewidth=2, markersize=8)
    ax1.plot(seq_lens, decode_bw, "s-", label="Decode", color="#e74c3c", linewidth=2, markersize=8)
    ax1.axhline(y=peak_bw, color="green", linestyle="--", linewidth=2, label=f"Peak ({peak_bw:.0f} GB/s)")
    ax1.set_xlabel("Input Sequence Length")
    ax1.set_ylabel("Achieved Bandwidth (GB/s)")
    ax1.set_title(f"Achieved Bandwidth vs Sequence Length ({model_name})")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xscale("log", base=2)

    # Plot 2: Utilization vs Sequence Length
    ax2 = axes[1]
    ax2.plot(seq_lens, prefill_util, "o-", label="Prefill", color="#3498db", linewidth=2, markersize=8)
    ax2.plot(seq_lens, decode_util, "s-", label="Decode", color="#e74c3c", linewidth=2, markersize=8)
    ax2.axhline(y=100, color="green", linestyle="--", linewidth=2, label="100% Utilization")
    ax2.set_xlabel("Input Sequence Length")
    ax2.set_ylabel("Bandwidth Utilization (%)")
    ax2.set_title(f"Bandwidth Utilization vs Sequence Length ({model_name})")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_xscale("log", base=2)

    plt.tight_layout()
    plt.savefig(output_path / "sequence_length_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "sequence_length_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/sequence_length_comparison.png")
    plt.close()


def plot_batch_size_comparison(
    results: list[ExperimentResult],
    output_path: Path,
    model_name: str,
):
    """Plot bandwidth utilization vs batch size."""
    batch_sizes = [r.config.batch_size for r in results]
    prefill_bw = [r.prefill.achieved_bandwidth_gbps for r in results]
    decode_bw = [r.decode.achieved_bandwidth_gbps for r in results]
    prefill_util = [r.prefill.utilization * 100 for r in results]
    decode_util = [r.decode.utilization * 100 for r in results]
    peak_bw = results[0].prefill.peak_bandwidth_gbps

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Achieved Bandwidth vs Batch Size
    ax1 = axes[0]
    ax1.plot(batch_sizes, prefill_bw, "o-", label="Prefill", color="#3498db", linewidth=2, markersize=8)
    ax1.plot(batch_sizes, decode_bw, "s-", label="Decode", color="#e74c3c", linewidth=2, markersize=8)
    ax1.axhline(y=peak_bw, color="green", linestyle="--", linewidth=2, label=f"Peak ({peak_bw:.0f} GB/s)")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Achieved Bandwidth (GB/s)")
    ax1.set_title(f"Achieved Bandwidth vs Batch Size ({model_name})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Utilization vs Batch Size
    ax2 = axes[1]
    ax2.plot(batch_sizes, prefill_util, "o-", label="Prefill", color="#3498db", linewidth=2, markersize=8)
    ax2.plot(batch_sizes, decode_util, "s-", label="Decode", color="#e74c3c", linewidth=2, markersize=8)
    ax2.axhline(y=100, color="green", linestyle="--", linewidth=2, label="100% Utilization")
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Bandwidth Utilization (%)")
    ax2.set_title(f"Bandwidth Utilization vs Batch Size ({model_name})")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "batch_size_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "batch_size_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/batch_size_comparison.png")
    plt.close()


def save_results_json(
    model_results: dict[str, ExperimentResult],
    seq_len_results: list[ExperimentResult],
    batch_size_results: list[ExperimentResult],
    output_path: Path,
):
    """Save all results to JSON file."""
    data = {
        "model_comparison": {},
        "sequence_length_comparison": [],
        "batch_size_comparison": [],
    }

    # Model comparison
    for model_name, result in model_results.items():
        data["model_comparison"][model_name] = {
            "prefill": {
                "traffic_gb": result.prefill.traffic_bytes / 1e9,
                "read_gb": result.prefill.read_bytes / 1e9,
                "write_gb": result.prefill.write_bytes / 1e9,
                "execution_time_ms": result.prefill.execution_time_seconds * 1e3,
                "achieved_bandwidth_gbps": result.prefill.achieved_bandwidth_gbps,
                "peak_bandwidth_gbps": result.prefill.peak_bandwidth_gbps,
                "utilization_percent": result.prefill.utilization * 100,
            },
            "decode": {
                "traffic_mb": result.decode.traffic_bytes / 1e6,
                "read_mb": result.decode.read_bytes / 1e6,
                "write_mb": result.decode.write_bytes / 1e6,
                "execution_time_us": result.decode.execution_time_seconds * 1e6,
                "achieved_bandwidth_gbps": result.decode.achieved_bandwidth_gbps,
                "peak_bandwidth_gbps": result.decode.peak_bandwidth_gbps,
                "utilization_percent": result.decode.utilization * 100,
            },
        }

    # Sequence length comparison
    for result in seq_len_results:
        data["sequence_length_comparison"].append({
            "input_seq_len": result.config.input_seq_len,
            "prefill_bandwidth_gbps": result.prefill.achieved_bandwidth_gbps,
            "prefill_utilization_percent": result.prefill.utilization * 100,
            "decode_bandwidth_gbps": result.decode.achieved_bandwidth_gbps,
            "decode_utilization_percent": result.decode.utilization * 100,
        })

    # Batch size comparison
    for result in batch_size_results:
        data["batch_size_comparison"].append({
            "batch_size": result.config.batch_size,
            "prefill_bandwidth_gbps": result.prefill.achieved_bandwidth_gbps,
            "prefill_utilization_percent": result.prefill.utilization * 100,
            "decode_bandwidth_gbps": result.decode.achieved_bandwidth_gbps,
            "decode_utilization_percent": result.decode.utilization * 100,
        })

    with open(output_path / "results.json", "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {output_path}/results.json")


def main():
    parser = argparse.ArgumentParser(description="Memory IO Bandwidth Experiment")
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
        "--isa-lib",
        default="analytic_models/performance/customISA_lib.json",
        help="Path to customISA_lib.json",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/memory_io",
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
    hardware_config_path = str(project_root / args.config)
    isa_lib_path = str(project_root / args.isa_lib)
    output_path = project_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("MEMORY IO BANDWIDTH EXPERIMENT")
    print("=" * 70)
    print("Analyzing: HBM bandwidth = memory_traffic / execution_time")
    print("  - Prefill: Process entire input sequence")
    print("  - Decode:  Generate single output token (KV cache = input_seq_len)")
    print()
    print("NOTE: Utilization >100% indicates MEMORY-BOUND operation:")
    print("  The compute model finishes faster than memory can supply data.")
    print("  Real execution would be limited by memory bandwidth.")
    print("=" * 70)

    # 1. Model comparison
    print("\n[1] Model Comparison")
    print("-" * 70)
    model_results = run_model_comparison(
        model_lib_path=model_lib_path,
        memory_config_path=memory_config_path,
        hardware_config_path=hardware_config_path,
        isa_lib_path=isa_lib_path,
        models=args.models,
        batch_size=1,
        input_seq_len=2048,
        output_seq_len=128,
    )

    # 2. Sequence length comparison
    print("\n[2] Sequence Length Comparison (llama-3-8b)")
    print("-" * 70)
    sequence_lengths = [512, 1024, 2048, 4096, 8192]
    seq_len_results = run_sequence_length_comparison(
        model_lib_path=model_lib_path,
        memory_config_path=memory_config_path,
        hardware_config_path=hardware_config_path,
        isa_lib_path=isa_lib_path,
        model_name="llama-3-8b",
        sequence_lengths=sequence_lengths,
    )

    # 3. Batch size comparison
    print("\n[3] Batch Size Comparison (llama-3-8b)")
    print("-" * 70)
    batch_sizes = [1, 2, 4, 8, 16]
    batch_size_results = run_batch_size_comparison(
        model_lib_path=model_lib_path,
        memory_config_path=memory_config_path,
        hardware_config_path=hardware_config_path,
        isa_lib_path=isa_lib_path,
        model_name="llama-3-8b",
        batch_sizes=batch_sizes,
    )

    # 4. Generate plots
    print("\n[4] Generating Plots")
    print("-" * 70)
    if args.no_plot:
        print("  Skipping plots (--no-plot flag)")
    elif not HAS_MATPLOTLIB:
        print("  Skipping plots (matplotlib not available)")
    else:
        plot_model_comparison(model_results, output_path)
        plot_sequence_length_comparison(seq_len_results, output_path, "llama-3-8b")
        plot_batch_size_comparison(batch_size_results, output_path, "llama-3-8b")

    # 5. Save results
    print("\n[5] Saving Results")
    print("-" * 70)
    save_results_json(model_results, seq_len_results, batch_size_results, output_path)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
