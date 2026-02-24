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
    decode_short_profile: TransformerBlockProfile = field(default_factory=TransformerBlockProfile)
    decode_long_profile: TransformerBlockProfile = field(default_factory=TransformerBlockProfile)
    decode_short_ctx: int = 0  # KV cache context length for short decode
    decode_long_ctx: int = 0   # KV cache context length for long decode
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
            "decode_short": {"context_length": self.decode_short_ctx, **self.decode_short_profile.to_dict()},
            "decode_long": {"context_length": self.decode_long_ctx, **self.decode_long_profile.to_dict()},
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
    # DECODE PHASE ANALYSIS - Helper function
    # =========================================================================
    def compute_decode_profile(kv_size: int) -> TransformerBlockProfile:
        """Compute decode profile for a given KV cache context length."""
        mode = "decode"
        seq_len = 1  # Generate 1 token

        decode_layers = []

        # 1. RMS Norm (pre-attention)
        rms_cycles_d = perf.rms_layer(hidden_size, seq_len, batch_size, mode)
        decode_layers.append(
            LayerProfile(
                name="RMS Norm",
                compute_cycles=rms_cycles_d,
                compute_time_us=rms_cycles_d / config.frequency_hz * 1e6,
                memory_traffic_bytes=rms_weights_bytes,
                memory_time_us=rms_weights_bytes / peak_bw_bytes_per_us,
            )
        )

        # 2. QKV Projection
        proj_cycles_d = perf.projection(
            hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
        )
        proj_traffic_d = mem_model.mem.projection_traffic(
            hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
        )
        decode_layers.append(
            LayerProfile(
                name="QKV Projection",
                compute_cycles=proj_cycles_d,
                compute_time_us=proj_cycles_d / config.frequency_hz * 1e6,
                memory_traffic_bytes=proj_traffic_d.total_bytes,
                memory_time_us=proj_traffic_d.total_bytes / peak_bw_bytes_per_us,
            )
        )

        # 3. Flash Attention (depends on kv_size)
        attn_cycles_d = perf.flash_attention(
            num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
        )
        attn_traffic_d = mem_model.mem.flash_attention_traffic(
            num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
        )
        decode_layers.append(
            LayerProfile(
                name="Flash Attention",
                compute_cycles=attn_cycles_d,
                compute_time_us=attn_cycles_d / config.frequency_hz * 1e6,
                memory_traffic_bytes=attn_traffic_d.total_bytes,
                memory_time_us=attn_traffic_d.total_bytes / peak_bw_bytes_per_us,
            )
        )

        # 4. Output Projection
        out_proj_cycles_d = perf.projection(
            hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
        ) // 3
        out_proj_traffic_d = mem_model.mem.output_projection_traffic(
            hidden_size, num_attention_heads, head_dim, seq_len, batch_size, mode
        )
        decode_layers.append(
            LayerProfile(
                name="Output Projection",
                compute_cycles=out_proj_cycles_d,
                compute_time_us=out_proj_cycles_d / config.frequency_hz * 1e6,
                memory_traffic_bytes=out_proj_traffic_d.total_bytes,
                memory_time_us=out_proj_traffic_d.total_bytes / peak_bw_bytes_per_us,
            )
        )

        # 5. Residual Connection
        res_cycles_d = perf.residual(hidden_size, seq_len, batch_size, mode)
        decode_layers.append(
            LayerProfile(
                name="Residual",
                compute_cycles=res_cycles_d,
                compute_time_us=res_cycles_d / config.frequency_hz * 1e6,
                memory_traffic_bytes=0,
                memory_time_us=0.0,
            )
        )

        # 6. RMS Norm (pre-FFN)
        decode_layers.append(
            LayerProfile(
                name="RMS Norm (FFN)",
                compute_cycles=rms_cycles_d,
                compute_time_us=rms_cycles_d / config.frequency_hz * 1e6,
                memory_traffic_bytes=rms_weights_bytes,
                memory_time_us=rms_weights_bytes / peak_bw_bytes_per_us,
            )
        )

        # 7. Feed Forward Network
        ffn_cycles_d = perf.feed_forward(hidden_size, intermediate_size, seq_len, batch_size, mode)
        ffn_traffic_d = mem_model.mem.ffn_traffic(hidden_size, intermediate_size, seq_len, batch_size, mode)
        decode_layers.append(
            LayerProfile(
                name="FFN",
                compute_cycles=ffn_cycles_d,
                compute_time_us=ffn_cycles_d / config.frequency_hz * 1e6,
                memory_traffic_bytes=ffn_traffic_d.total_bytes,
                memory_time_us=ffn_traffic_d.total_bytes / peak_bw_bytes_per_us,
            )
        )

        return TransformerBlockProfile(layers=decode_layers)

    # Decode with short context (first token after prefill)
    result.decode_short_ctx = config.input_seq_len
    result.decode_short_profile = compute_decode_profile(result.decode_short_ctx)

    # Decode with long context (last token)
    result.decode_long_ctx = config.input_seq_len + config.output_seq_len
    result.decode_long_profile = compute_decode_profile(result.decode_long_ctx)

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


def plot_execution_timeline(
    result: ExperimentResult,
    output_path: Path,
):
    """
    Plot horizontal execution timeline showing compute vs memory bottlenecks.

    Two plots side by side: Prefill and Decode.
    Compute uses lighter colors, Memory uses darker colors.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 2.5))

    phases = [
        (f"Prefill (seq={result.config.input_seq_len})", result.prefill_profile),
        (f"Decode (ctx={result.decode_long_ctx})", result.decode_long_profile),
    ]

    # Key layers with light (compute) and dark (memory) colors
    key_layers = {
        "QKV Projection": {"compute": "#e7d4e8", "memory": "#762a83"},
        "Flash Attention": {"compute": "#d9f0d3", "memory": "#1b7837"},
        "FFN": {"compute": "#af8dc3", "memory": "#7fbf7b"},
    }
    others_color = "#a0a0a0"  # Single grey for others
    short_names = {"QKV Projection": "QKV", "Flash Attention": "Attn", "FFN": "FFN"}
    bar_height = 0.5  # Narrower bars

    for idx, (ax, (phase_name, profile)) in enumerate(zip(axes, phases)):
        current_time = 0.0

        for layer in profile.layers:
            effective_time = layer.total_time_us

            if layer.name in key_layers:
                compute_color = key_layers[layer.name]["compute"]
                memory_color = key_layers[layer.name]["memory"]
            else:
                compute_color = others_color
                memory_color = others_color

            if layer.is_memory_bound:
                # Memory bottleneck - full bar on Memory row (dark)
                ax.barh(y=0, width=effective_time, left=current_time,
                        height=bar_height, color=memory_color, edgecolor="black", linewidth=0.5)
                if layer.compute_time_us > 0:
                    ax.barh(y=1, width=layer.compute_time_us, left=current_time,
                            height=bar_height, color=compute_color, edgecolor="black", linewidth=0.5)
            else:
                # Compute bottleneck - full bar on Compute row (light)
                ax.barh(y=1, width=effective_time, left=current_time,
                        height=bar_height, color=compute_color, edgecolor="black", linewidth=0.5)
                if layer.memory_time_us > 0:
                    ax.barh(y=0, width=layer.memory_time_us, left=current_time,
                            height=bar_height, color=memory_color, edgecolor="black", linewidth=0.5)

            # Label key layers on BOTH compute and memory bars
            if layer.name in key_layers:
                short_name = short_names[layer.name]
                # Label on compute bar
                if layer.compute_time_us > 0:
                    ax.text(current_time + layer.compute_time_us / 2, 1,
                            short_name, ha="center", va="center",
                            fontsize=6, fontweight="bold", color="black")
                # Label on memory bar
                if layer.memory_time_us > 0:
                    ax.text(current_time + layer.memory_time_us / 2, 0,
                            short_name, ha="center", va="center",
                            fontsize=6, fontweight="bold", color="white")

            current_time += effective_time

        ax.set_yticks([0, 1])
        if idx == 0:
            ax.set_yticklabels(["Memory", "Compute"])
        else:
            ax.set_yticklabels(["", ""])
        ax.set_xlabel("Time (μs)")
        ax.set_title(f"{phase_name} - Total: {current_time:.2f} μs", fontsize=9)
        ax.set_xlim(0, current_time * 1.05)
        ax.set_ylim(-0.5, 1.5)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"Transformer Block Execution - {result.config.model_name} (batch={result.config.batch_size})",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path / "execution_timeline.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "execution_timeline.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/execution_timeline.png")
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
    print_profile(f"DECODE PHASE (ctx={result.decode_long_ctx})", result.decode_long_profile, result.peak_bandwidth_gbps)

    # Generate plot
    if not args.no_plot:
        print("\nGenerating plot...")
        plot_execution_timeline(result, output_path)

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
