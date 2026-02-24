#!/usr/bin/env python3
"""
Compute vs Memory Time Experiment: Transformer Block Analysis

This script analyzes the balance between computation time and memory access time
for each component of a transformer block:
- Computation time: cycles from performance model / frequency
- Memory access time: memory traffic / HBM bandwidth

Analyzes both prefill and decode phases separately.

Usage:
    python experiments/compute_vs_memory_experiment.py
    python experiments/compute_vs_memory_experiment.py --model llama-3.1-70b
    python experiments/compute_vs_memory_experiment.py --batch-size 8 --input-seq 4096
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "analytic_models" / "memory"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analytic_models" / "performance"))

from llm_memory_model import LLMMemoryModel
from memory_model import MemoryTraffic, load_memory_config_from_toml
from llama_model import LLaMAModel
from perf_model import PerfModel, load_hardware_config_from_toml


@dataclass
class LayerProfile:
    """Profile for a single transformer layer component."""

    name: str
    compute_cycles: int = 0
    compute_time_us: float = 0.0  # microseconds
    memory_traffic_bytes: int = 0
    memory_time_us: float = 0.0  # microseconds (traffic / bandwidth)

    @property
    def total_time_us(self) -> float:
        """Effective time (max of compute and memory)."""
        return max(self.compute_time_us, self.memory_time_us)

    @property
    def is_memory_bound(self) -> bool:
        """True if memory access takes longer than compute."""
        return self.memory_time_us > self.compute_time_us

    @property
    def compute_ratio(self) -> float:
        """Ratio of compute time to total time."""
        total = self.compute_time_us + self.memory_time_us
        return self.compute_time_us / total if total > 0 else 0.0

    @property
    def memory_ratio(self) -> float:
        """Ratio of memory time to total time."""
        total = self.compute_time_us + self.memory_time_us
        return self.memory_time_us / total if total > 0 else 0.0


@dataclass
class TransformerBlockProfile:
    """Profile for an entire transformer block."""

    layers: list[LayerProfile] = field(default_factory=list)

    @property
    def total_compute_time_us(self) -> float:
        return sum(layer.compute_time_us for layer in self.layers)

    @property
    def total_memory_time_us(self) -> float:
        return sum(layer.memory_time_us for layer in self.layers)

    @property
    def total_memory_traffic_bytes(self) -> int:
        return sum(layer.memory_traffic_bytes for layer in self.layers)

    def to_dict(self) -> dict:
        return {
            "layers": [
                {
                    "name": layer.name,
                    "compute_cycles": layer.compute_cycles,
                    "compute_time_us": layer.compute_time_us,
                    "memory_traffic_bytes": layer.memory_traffic_bytes,
                    "memory_time_us": layer.memory_time_us,
                    "is_memory_bound": layer.is_memory_bound,
                    "compute_ratio": layer.compute_ratio,
                    "memory_ratio": layer.memory_ratio,
                }
                for layer in self.layers
            ],
            "total_compute_time_us": self.total_compute_time_us,
            "total_memory_time_us": self.total_memory_time_us,
            "total_memory_traffic_mb": self.total_memory_traffic_bytes / 1e6,
        }


@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""

    model_name: str
    batch_size: int
    input_seq_len: int
    output_seq_len: int
    frequency_hz: float = 1e9


@dataclass
class ExperimentResult:
    """Results from compute vs memory analysis."""

    config: ExperimentConfig
    prefill_profile: TransformerBlockProfile = field(default_factory=TransformerBlockProfile)
    decode_profile: TransformerBlockProfile = field(default_factory=TransformerBlockProfile)
    peak_bandwidth_gbps: float = 0.0

    def to_dict(self) -> dict:
        return {
            "config": {
                "model_name": self.config.model_name,
                "batch_size": self.config.batch_size,
                "input_seq_len": self.config.input_seq_len,
                "output_seq_len": self.config.output_seq_len,
                "frequency_hz": self.config.frequency_hz,
            },
            "peak_bandwidth_gbps": self.peak_bandwidth_gbps,
            "prefill": self.prefill_profile.to_dict(),
            "decode": self.decode_profile.to_dict(),
        }


def analyze_transformer_block(
    model_path: str,
    memory_config_path: str,
    hardware_config_path: str,
    isa_lib_path: str,
    config: ExperimentConfig,
) -> ExperimentResult:
    """Analyze compute vs memory time for each transformer block component."""

    # Load configurations
    memory_config = load_memory_config_from_toml(memory_config_path)
    hardware_config = load_hardware_config_from_toml(hardware_config_path)

    # Load model config
    with open(model_path) as f:
        model_param = json.load(f)

    hidden_size = model_param["hidden_size"]
    num_attention_heads = model_param["num_attention_heads"]
    num_hidden_layers = model_param["num_hidden_layers"]
    intermediate_size = model_param["intermediate_size"]
    num_kv_heads = model_param["num_key_value_heads"]
    head_dim = model_param.get("head_dim", hidden_size // num_attention_heads)

    # Initialize performance model
    perf = PerfModel(hardware_config, isa_lib_path)

    # Initialize memory model
    mem_model = LLMMemoryModel(
        model_config_path=model_path,
        memory_config=memory_config,
        batch_size=config.batch_size,
        input_seq_len=config.input_seq_len,
        output_seq_len=config.output_seq_len,
        frequency_hz=config.frequency_hz,
    )

    # Compute peak bandwidth
    peak_bw_gbps = mem_model.mem.compute_peak_bandwidth(config.frequency_hz)
    peak_bw_bytes_per_us = peak_bw_gbps * 1e9 / 1e6  # bytes per microsecond

    result = ExperimentResult(config=config, peak_bandwidth_gbps=peak_bw_gbps)

    # =========================================================================
    # PREFILL PHASE ANALYSIS
    # =========================================================================
    mode = "prefill"
    seq_len = config.input_seq_len
    kv_size = config.input_seq_len
    batch_size = config.batch_size

    prefill_layers = []

    # 1. RMS Norm (pre-attention)
    rms_cycles = perf.rms_layer(hidden_size, seq_len, batch_size, mode)
    rms_weights_bytes = mem_model.mem.layer_norm_weights(hidden_size)
    prefill_layers.append(
        LayerProfile(
            name="RMS Norm",
            compute_cycles=rms_cycles,
            compute_time_us=rms_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=rms_weights_bytes,
            memory_time_us=rms_weights_bytes / peak_bw_bytes_per_us,
        )
    )

    # 2. QKV Projection
    proj_cycles = perf.projection(
        hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
    )
    proj_traffic = mem_model.mem.projection_traffic(
        hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
    )
    prefill_layers.append(
        LayerProfile(
            name="QKV Projection",
            compute_cycles=proj_cycles,
            compute_time_us=proj_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=proj_traffic.total_bytes,
            memory_time_us=proj_traffic.total_bytes / peak_bw_bytes_per_us,
        )
    )

    # 3. Flash Attention
    attn_cycles = perf.flash_attention(
        num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
    )
    attn_traffic = mem_model.mem.flash_attention_traffic(
        num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
    )
    prefill_layers.append(
        LayerProfile(
            name="Flash Attention",
            compute_cycles=attn_cycles,
            compute_time_us=attn_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=attn_traffic.total_bytes,
            memory_time_us=attn_traffic.total_bytes / peak_bw_bytes_per_us,
        )
    )

    # 4. Output Projection
    out_proj_cycles = perf.projection(
        hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
    ) // 3  # Rough estimate: output proj is ~1/3 of QKV projection
    out_proj_traffic = mem_model.mem.output_projection_traffic(
        hidden_size, num_attention_heads, head_dim, seq_len, batch_size, mode
    )
    prefill_layers.append(
        LayerProfile(
            name="Output Projection",
            compute_cycles=out_proj_cycles,
            compute_time_us=out_proj_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=out_proj_traffic.total_bytes,
            memory_time_us=out_proj_traffic.total_bytes / peak_bw_bytes_per_us,
        )
    )

    # 5. Residual Connection
    res_cycles = perf.residual(hidden_size, seq_len, batch_size, mode)
    # Residual is on-chip operation, minimal HBM traffic
    prefill_layers.append(
        LayerProfile(
            name="Residual",
            compute_cycles=res_cycles,
            compute_time_us=res_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=0,
            memory_time_us=0.0,
        )
    )

    # 6. RMS Norm (pre-FFN)
    prefill_layers.append(
        LayerProfile(
            name="RMS Norm (FFN)",
            compute_cycles=rms_cycles,
            compute_time_us=rms_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=rms_weights_bytes,
            memory_time_us=rms_weights_bytes / peak_bw_bytes_per_us,
        )
    )

    # 7. Feed Forward Network
    ffn_cycles = perf.feed_forward(hidden_size, intermediate_size, seq_len, batch_size, mode)
    ffn_traffic = mem_model.mem.ffn_traffic(hidden_size, intermediate_size, seq_len, batch_size, mode)
    prefill_layers.append(
        LayerProfile(
            name="FFN",
            compute_cycles=ffn_cycles,
            compute_time_us=ffn_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=ffn_traffic.total_bytes,
            memory_time_us=ffn_traffic.total_bytes / peak_bw_bytes_per_us,
        )
    )

    result.prefill_profile = TransformerBlockProfile(layers=prefill_layers)

    # =========================================================================
    # DECODE PHASE ANALYSIS (single token generation)
    # =========================================================================
    mode = "decode"
    seq_len = 1  # Generate 1 token
    kv_size = config.input_seq_len + config.output_seq_len  # Full KV cache

    decode_layers = []

    # 1. RMS Norm (pre-attention)
    rms_cycles = perf.rms_layer(hidden_size, seq_len, batch_size, mode)
    decode_layers.append(
        LayerProfile(
            name="RMS Norm",
            compute_cycles=rms_cycles,
            compute_time_us=rms_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=rms_weights_bytes,
            memory_time_us=rms_weights_bytes / peak_bw_bytes_per_us,
        )
    )

    # 2. QKV Projection
    proj_cycles = perf.projection(
        hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
    )
    proj_traffic = mem_model.mem.projection_traffic(
        hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
    )
    decode_layers.append(
        LayerProfile(
            name="QKV Projection",
            compute_cycles=proj_cycles,
            compute_time_us=proj_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=proj_traffic.total_bytes,
            memory_time_us=proj_traffic.total_bytes / peak_bw_bytes_per_us,
        )
    )

    # 3. Flash Attention
    attn_cycles = perf.flash_attention(
        num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
    )
    attn_traffic = mem_model.mem.flash_attention_traffic(
        num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
    )
    decode_layers.append(
        LayerProfile(
            name="Flash Attention",
            compute_cycles=attn_cycles,
            compute_time_us=attn_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=attn_traffic.total_bytes,
            memory_time_us=attn_traffic.total_bytes / peak_bw_bytes_per_us,
        )
    )

    # 4. Output Projection
    out_proj_cycles = perf.projection(
        hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
    ) // 3
    out_proj_traffic = mem_model.mem.output_projection_traffic(
        hidden_size, num_attention_heads, head_dim, seq_len, batch_size, mode
    )
    decode_layers.append(
        LayerProfile(
            name="Output Projection",
            compute_cycles=out_proj_cycles,
            compute_time_us=out_proj_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=out_proj_traffic.total_bytes,
            memory_time_us=out_proj_traffic.total_bytes / peak_bw_bytes_per_us,
        )
    )

    # 5. Residual Connection
    res_cycles = perf.residual(hidden_size, seq_len, batch_size, mode)
    decode_layers.append(
        LayerProfile(
            name="Residual",
            compute_cycles=res_cycles,
            compute_time_us=res_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=0,
            memory_time_us=0.0,
        )
    )

    # 6. RMS Norm (pre-FFN)
    decode_layers.append(
        LayerProfile(
            name="RMS Norm (FFN)",
            compute_cycles=rms_cycles,
            compute_time_us=rms_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=rms_weights_bytes,
            memory_time_us=rms_weights_bytes / peak_bw_bytes_per_us,
        )
    )

    # 7. Feed Forward Network
    ffn_cycles = perf.feed_forward(hidden_size, intermediate_size, seq_len, batch_size, mode)
    ffn_traffic = mem_model.mem.ffn_traffic(hidden_size, intermediate_size, seq_len, batch_size, mode)
    decode_layers.append(
        LayerProfile(
            name="FFN",
            compute_cycles=ffn_cycles,
            compute_time_us=ffn_cycles / config.frequency_hz * 1e6,
            memory_traffic_bytes=ffn_traffic.total_bytes,
            memory_time_us=ffn_traffic.total_bytes / peak_bw_bytes_per_us,
        )
    )

    result.decode_profile = TransformerBlockProfile(layers=decode_layers)

    return result


def print_profile(name: str, profile: TransformerBlockProfile, peak_bw_gbps: float):
    """Print a formatted profile."""
    print(f"\n{'=' * 80}")
    print(f"{name} - Transformer Block Analysis (One Layer)")
    print(f"{'=' * 80}")
    print(f"Peak HBM Bandwidth: {peak_bw_gbps:.1f} GB/s")
    print()
    print(f"{'Layer':<20} {'Compute (us)':<15} {'Memory (us)':<15} {'Traffic (MB)':<15} {'Bound':<10}")
    print("-" * 80)

    for layer in profile.layers:
        bound = "MEMORY" if layer.is_memory_bound else "COMPUTE"
        print(
            f"{layer.name:<20} "
            f"{layer.compute_time_us:>12.2f}   "
            f"{layer.memory_time_us:>12.2f}   "
            f"{layer.memory_traffic_bytes / 1e6:>12.3f}   "
            f"{bound:<10}"
        )

    print("-" * 80)
    print(
        f"{'TOTAL':<20} "
        f"{profile.total_compute_time_us:>12.2f}   "
        f"{profile.total_memory_time_us:>12.2f}   "
        f"{profile.total_memory_traffic_bytes / 1e6:>12.3f}"
    )


def plot_compute_vs_memory(
    result: ExperimentResult,
    output_path: Path,
):
    """Plot compute vs memory time comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    phases = [
        ("Prefill", result.prefill_profile),
        ("Decode", result.decode_profile),
    ]

    for idx, (phase_name, profile) in enumerate(phases):
        layer_names = [layer.name for layer in profile.layers]
        compute_times = [layer.compute_time_us for layer in profile.layers]
        memory_times = [layer.memory_time_us for layer in profile.layers]

        # Plot 1: Stacked bar chart
        ax1 = axes[idx, 0]
        x = np.arange(len(layer_names))
        width = 0.6

        bars1 = ax1.bar(x, compute_times, width, label="Compute Time", color="#3498db")
        bars2 = ax1.bar(x, memory_times, width, bottom=compute_times, label="Memory Time", color="#e74c3c")

        ax1.set_xlabel("Layer Component")
        ax1.set_ylabel("Time (μs)")
        ax1.set_title(f"{phase_name}: Compute vs Memory Time per Layer")
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_names, rotation=45, ha="right")
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Plot 2: Side-by-side comparison
        ax2 = axes[idx, 1]
        width = 0.35
        bars3 = ax2.bar(x - width / 2, compute_times, width, label="Compute Time", color="#3498db")
        bars4 = ax2.bar(x + width / 2, memory_times, width, label="Memory Time", color="#e74c3c")

        ax2.set_xlabel("Layer Component")
        ax2.set_ylabel("Time (μs)")
        ax2.set_title(f"{phase_name}: Compute vs Memory Time Comparison")
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_names, rotation=45, ha="right")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        # Add bound indicators
        for i, layer in enumerate(profile.layers):
            if layer.is_memory_bound:
                ax2.annotate(
                    "MEM",
                    xy=(i + width / 2, layer.memory_time_us),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color="red",
                )

    plt.suptitle(
        f"Compute vs Memory Time Analysis - {result.config.model_name}\n"
        f"(batch={result.config.batch_size}, input_seq={result.config.input_seq_len}, "
        f"output_seq={result.config.output_seq_len})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path / "compute_vs_memory.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "compute_vs_memory.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/compute_vs_memory.png")
    plt.close()


def plot_breakdown_pie(
    result: ExperimentResult,
    output_path: Path,
):
    """Plot pie charts showing compute/memory breakdown per layer."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    phases = [
        ("Prefill", result.prefill_profile),
        ("Decode", result.decode_profile),
    ]

    colors = plt.cm.Set3(np.linspace(0, 1, 7))

    for idx, (phase_name, profile) in enumerate(phases):
        layer_names = [layer.name for layer in profile.layers]

        # Compute time breakdown
        compute_times = [layer.compute_time_us for layer in profile.layers]
        ax1 = axes[idx, 0]
        wedges, texts, autotexts = ax1.pie(
            compute_times,
            labels=layer_names,
            autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
            colors=colors,
            startangle=90,
        )
        ax1.set_title(f"{phase_name}: Compute Time Breakdown")

        # Memory time breakdown
        memory_times = [layer.memory_time_us for layer in profile.layers]
        ax2 = axes[idx, 1]
        # Filter out zero values for cleaner pie chart
        non_zero_indices = [i for i, t in enumerate(memory_times) if t > 0]
        if non_zero_indices:
            filtered_names = [layer_names[i] for i in non_zero_indices]
            filtered_times = [memory_times[i] for i in non_zero_indices]
            filtered_colors = [colors[i] for i in non_zero_indices]
            ax2.pie(
                filtered_times,
                labels=filtered_names,
                autopct=lambda pct: f"{pct:.1f}%" if pct > 3 else "",
                colors=filtered_colors,
                startangle=90,
            )
        ax2.set_title(f"{phase_name}: Memory Time Breakdown")

    plt.suptitle(
        f"Time Breakdown Analysis - {result.config.model_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path / "time_breakdown.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "time_breakdown.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/time_breakdown.png")
    plt.close()


def plot_summary_comparison(
    result: ExperimentResult,
    output_path: Path,
):
    """Plot summary comparison between prefill and decode."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    phases = ["Prefill", "Decode"]
    profiles = [result.prefill_profile, result.decode_profile]

    # Plot 1: Total compute vs memory time
    ax1 = axes[0]
    compute_totals = [p.total_compute_time_us for p in profiles]
    memory_totals = [p.total_memory_time_us for p in profiles]
    x = np.arange(len(phases))
    width = 0.35

    ax1.bar(x - width / 2, compute_totals, width, label="Compute", color="#3498db")
    ax1.bar(x + width / 2, memory_totals, width, label="Memory", color="#e74c3c")
    ax1.set_xlabel("Phase")
    ax1.set_ylabel("Time (μs)")
    ax1.set_title("Total Compute vs Memory Time")
    ax1.set_xticks(x)
    ax1.set_xticklabels(phases)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Add value labels
    for i, (c, m) in enumerate(zip(compute_totals, memory_totals)):
        ax1.text(i - width / 2, c + 0.01 * max(compute_totals + memory_totals), f"{c:.1f}", ha="center", fontsize=9)
        ax1.text(i + width / 2, m + 0.01 * max(compute_totals + memory_totals), f"{m:.1f}", ha="center", fontsize=9)

    # Plot 2: Memory traffic comparison
    ax2 = axes[1]
    traffic_totals = [p.total_memory_traffic_bytes / 1e6 for p in profiles]
    bars = ax2.bar(phases, traffic_totals, color=["#3498db", "#e74c3c"])
    ax2.set_xlabel("Phase")
    ax2.set_ylabel("Memory Traffic (MB)")
    ax2.set_title("Total HBM Traffic")
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, traffic_totals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.2f}", ha="center", va="bottom", fontsize=10)

    # Plot 3: Compute/Memory ratio per layer
    ax3 = axes[2]
    layer_names = [layer.name for layer in result.prefill_profile.layers]
    prefill_ratios = [layer.compute_ratio for layer in result.prefill_profile.layers]
    decode_ratios = [layer.compute_ratio for layer in result.decode_profile.layers]

    x = np.arange(len(layer_names))
    width = 0.35

    ax3.bar(x - width / 2, prefill_ratios, width, label="Prefill", color="#3498db")
    ax3.bar(x + width / 2, decode_ratios, width, label="Decode", color="#e74c3c")
    ax3.axhline(y=0.5, color="green", linestyle="--", linewidth=1, label="50% (balanced)")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Compute Ratio")
    ax3.set_title("Compute Ratio per Layer\n(>0.5 = compute-bound, <0.5 = memory-bound)")
    ax3.set_xticks(x)
    ax3.set_xticklabels(layer_names, rotation=45, ha="right")
    ax3.legend(loc="upper right")
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_ylim(0, 1)

    plt.suptitle(
        f"Summary Analysis - {result.config.model_name}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path / "summary_comparison.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "summary_comparison.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/summary_comparison.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compute vs Memory Time Experiment")
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
        default="experiments/results/compute_vs_memory",
        help="Output directory for plots and results",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="llama-3-8b",
        help="Model to analyze",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size",
    )
    parser.add_argument(
        "--input-seq",
        "-i",
        type=int,
        default=2048,
        help="Input sequence length",
    )
    parser.add_argument(
        "--output-seq",
        "-o",
        type=int,
        default=128,
        help="Output sequence length",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plot generation",
    )
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    model_lib_path = project_root / args.model_lib
    model_path = model_lib_path / f"{args.model}.json"
    memory_config_path = str(project_root / args.config)
    hardware_config_path = str(project_root / args.config)
    isa_lib_path = str(project_root / args.isa_lib)
    output_path = project_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"Error: Model {args.model} not found at {model_path}")
        return

    print("=" * 80)
    print("COMPUTE VS MEMORY TIME EXPERIMENT")
    print("=" * 80)
    print(f"Model:           {args.model}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Input Seq Len:   {args.input_seq}")
    print(f"Output Seq Len:  {args.output_seq}")
    print("=" * 80)

    config = ExperimentConfig(
        model_name=args.model,
        batch_size=args.batch_size,
        input_seq_len=args.input_seq,
        output_seq_len=args.output_seq,
    )

    # Run analysis
    result = analyze_transformer_block(
        str(model_path),
        memory_config_path,
        hardware_config_path,
        isa_lib_path,
        config,
    )

    # Print results
    print_profile("PREFILL PHASE", result.prefill_profile, result.peak_bandwidth_gbps)
    print_profile("DECODE PHASE", result.decode_profile, result.peak_bandwidth_gbps)

    # Generate plots
    if not args.no_plot:
        print("\nGenerating plots...")
        plot_compute_vs_memory(result, output_path)
        plot_breakdown_pie(result, output_path)
        plot_summary_comparison(result, output_path)

    # Save results as JSON
    with open(output_path / "results.json", "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Saved: {output_path}/results.json")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
