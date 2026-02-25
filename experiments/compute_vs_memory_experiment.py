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

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "analytic_models" / "memory"))
sys.path.insert(0, str(Path(__file__).parent.parent / "analytic_models" / "performance"))

from llm_memory_model import LLMMemoryModel
from memory_model import load_memory_config_from_toml
from perf_model import PerfModel, load_hardware_config_from_toml


@dataclass
class ExecutionSegment:
    """A segment of execution with aligned compute and memory info."""

    name: str
    cycles: int
    systolic_utilization: float  # 0-100%
    memory_read_bytes: int = 0
    memory_write_bytes: int = 0
    # Time fields (set after creation based on frequency/bandwidth)
    compute_time_us: float = 0.0
    memory_time_us: float = 0.0

    @property
    def memory_bytes(self) -> int:
        return self.memory_read_bytes + self.memory_write_bytes

    @property
    def effective_time_us(self) -> float:
        """Effective time is max of compute and memory (they overlap)."""
        return max(self.compute_time_us, self.memory_time_us)

    @property
    def is_memory_bound(self) -> bool:
        return self.memory_time_us > self.compute_time_us


@dataclass
class LayerProfile:
    """Profile for a single transformer layer component."""

    name: str
    group: str = ""  # Group name for plotting (e.g., "Attention", "FFN")
    compute_cycles: int = 0
    compute_time_us: float = 0.0  # microseconds
    memory_traffic_bytes: int = 0
    memory_time_us: float = 0.0  # microseconds (traffic / bandwidth)
    # Execution segments with aligned compute and memory
    segments: list = field(default_factory=list)

    @property
    def total_time_us(self) -> float:
        """Effective time (max of compute and memory)."""
        return max(self.compute_time_us, self.memory_time_us)

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

    @property
    def is_memory_bound(self) -> bool:
        """True if memory time exceeds compute time."""
        return self.memory_time_us > self.compute_time_us


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
                    "segments": [
                        {
                            "name": seg.name,
                            "cycles": seg.cycles,
                            "systolic_utilization": seg.systolic_utilization,
                            "compute_time_us": seg.compute_time_us,
                            "memory_time_us": seg.memory_time_us,
                            "effective_time_us": seg.effective_time_us,
                        }
                        for seg in layer.segments
                    ],
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
class HardwareOverride:
    """Hardware parameter overrides for experiments."""

    mlen: int = None
    blen: int = None
    vlen: int = None
    hbm_bandwidth_gbps: float = None  # HBM bandwidth in GB/s
    use_flash_attention: bool = True  # Whether to use flash attention


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
    hardware_override: HardwareOverride = None
    # Per-tile flash attention data
    prefill_flash_attn_tiles: dict = None
    decode_short_flash_attn_tiles: dict = None
    decode_long_flash_attn_tiles: dict = None

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
    hardware_override: HardwareOverride = None,
) -> ExperimentResult:
    """Analyze compute vs memory time for each transformer block component."""

    # Load configurations
    memory_config = load_memory_config_from_toml(memory_config_path)
    hardware_config = load_hardware_config_from_toml(hardware_config_path)

    # Apply hardware overrides if specified
    if hardware_override:
        if hardware_override.mlen is not None:
            memory_config.MLEN = hardware_override.mlen
            hardware_config.MLEN = hardware_override.mlen
        if hardware_override.blen is not None:
            memory_config.BLEN = hardware_override.blen
            hardware_config.BLEN = hardware_override.blen
        if hardware_override.vlen is not None:
            memory_config.VLEN = hardware_override.vlen
            hardware_config.VLEN = hardware_override.vlen

    # Load model config
    with open(model_path) as f:
        model_param = json.load(f)

    hidden_size = model_param["hidden_size"]
    num_attention_heads = model_param["num_attention_heads"]
    num_hidden_layers = model_param["num_hidden_layers"]
    intermediate_size = model_param["intermediate_size"]
    num_kv_heads = model_param["num_key_value_heads"]
    head_dim = model_param.get("head_dim", hidden_size // num_attention_heads)

    # Initialize memory model
    mem_model = LLMMemoryModel(
        model_config_path=model_path,
        memory_config=memory_config,
        batch_size=config.batch_size,
        input_seq_len=config.input_seq_len,
        output_seq_len=config.output_seq_len,
        frequency_hz=config.frequency_hz,
    )

    # Initialize performance model with memory model for combined segments
    perf = PerfModel(hardware_config, isa_lib_path, memory_model=mem_model.mem)

    # Compute peak bandwidth (override if specified)
    if hardware_override and hardware_override.hbm_bandwidth_gbps is not None:
        peak_bw_gbps = hardware_override.hbm_bandwidth_gbps
    else:
        peak_bw_gbps = mem_model.mem.compute_peak_bandwidth(config.frequency_hz)
    peak_bw_bytes_per_us = peak_bw_gbps * 1e9 / 1e6  # bytes per microsecond

    result = ExperimentResult(config=config, peak_bandwidth_gbps=peak_bw_gbps, hardware_override=hardware_override)

    # Determine attention type
    use_flash_attention = hardware_override.use_flash_attention if hardware_override else True

    # Helper to convert perf segments to ExecutionSegment objects
    def convert_segments(perf_segs: list) -> list:
        """Convert perf segments (with integrated memory info) to ExecutionSegment objects."""
        converted = []
        for ps in perf_segs:
            read_bytes = ps.get("memory_read_bytes", 0)
            write_bytes = ps.get("memory_write_bytes", 0)
            compute_time = ps["cycles"] / config.frequency_hz * 1e6
            memory_time = (read_bytes + write_bytes) / peak_bw_bytes_per_us if peak_bw_bytes_per_us > 0 else 0
            converted.append(ExecutionSegment(
                name=ps["name"],
                cycles=ps["cycles"],
                systolic_utilization=ps["systolic_utilization"],
                memory_read_bytes=read_bytes,
                memory_write_bytes=write_bytes,
                compute_time_us=compute_time,
                memory_time_us=memory_time,
            ))
        return converted

    # Helper to create LayerProfile from perf result
    def make_layer_profile(name: str, group: str, perf_result: dict) -> LayerProfile:
        """Create LayerProfile from perf result with integrated memory info."""
        segments = convert_segments(perf_result["segments"])
        total_mem_bytes = sum(s.memory_read_bytes + s.memory_write_bytes for s in segments)
        return LayerProfile(
            name=name,
            group=group,
            compute_cycles=perf_result["total_cycles"],
            compute_time_us=perf_result["total_cycles"] / config.frequency_hz * 1e6,
            memory_traffic_bytes=total_mem_bytes,
            memory_time_us=total_mem_bytes / peak_bw_bytes_per_us if peak_bw_bytes_per_us > 0 else 0,
            segments=segments,
        )

    # =========================================================================
    # PREFILL PHASE ANALYSIS
    # =========================================================================
    mode = "prefill"
    seq_len = config.input_seq_len
    kv_size = config.input_seq_len
    batch_size = config.batch_size

    prefill_layers = []

    # 1. RMS Norm (pre-attention)
    rms_result = perf.rms_layer_with_util(hidden_size, seq_len, batch_size, mode)
    prefill_layers.append(make_layer_profile("RMS Norm", "Others", rms_result))

    # 2. QKV Projection
    proj_result = perf.projection_with_util(
        hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
    )
    prefill_layers.append(make_layer_profile("QKV Projection", "QKV", proj_result))

    # # 3. Attention (Flash or Self)
    # if use_flash_attention:
    #     attn_result = perf.flash_attention_with_util(
    #         num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
    #     )
    #     attn_name = "Flash Attention"
    #     # Store tile-level data for prefill
    #     result.prefill_flash_attn_tiles = attn_result
    # else:
    #     attn_result = perf.self_attention_with_util(
    #         num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
    #     )
    #     attn_name = "Self Attention"
    # prefill_layers.append(make_layer_profile(attn_name, "Attention", attn_result))

    # # 4. Residual Connection
    # res_result = perf.residual_with_util(hidden_size, seq_len, batch_size, mode)
    # prefill_layers.append(make_layer_profile("Residual", "Others", res_result))

    # # 5. RMS Norm (pre-FFN)
    # rms_ffn_result = perf.rms_layer_with_util(hidden_size, seq_len, batch_size, mode)
    # prefill_layers.append(make_layer_profile("RMS Norm (FFN)", "Others", rms_ffn_result))

    # # 6. FFN
    # ffn_result = perf.feed_forward_with_util(hidden_size, intermediate_size, seq_len, batch_size, mode)
    # prefill_layers.append(make_layer_profile("FFN", "FFN", ffn_result))

    result.prefill_profile = TransformerBlockProfile(layers=prefill_layers)

    # =========================================================================
    # DECODE PHASE ANALYSIS - Helper function
    # =========================================================================
    def compute_decode_profile(kv_size: int) -> tuple:
        """Compute decode profile for a given KV cache context length.

        Returns:
            tuple: (TransformerBlockProfile, flash_attn_tiles or None)
        """
        mode = "decode"
        seq_len = 1  # Generate 1 token

        decode_layers = []
        flash_attn_tiles = None

        # 1. RMS Norm (pre-attention)
        rms_result_d = perf.rms_layer_with_util(hidden_size, seq_len, batch_size, mode)
        decode_layers.append(make_layer_profile("RMS Norm", "Others", rms_result_d))

        # 2. QKV Projection
        proj_result_d = perf.projection_with_util(
            hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
        )
        decode_layers.append(make_layer_profile("QKV Projection", "QKV", proj_result_d))

        # # 3. Attention (Flash or Self, depends on kv_size)
        # if use_flash_attention:
        #     attn_result_d = perf.flash_attention_with_util(
        #         num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
        #     )
        #     attn_name_d = "Flash Attention"
        #     flash_attn_tiles = attn_result_d  # Store tile-level data
        # else:
        #     attn_result_d = perf.self_attention_with_util(
        #         num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
        #     )
        #     attn_name_d = "Self Attention"
        # decode_layers.append(make_layer_profile(attn_name_d, "Attention", attn_result_d))

        # # 4. Residual Connection
        # res_result_d = perf.residual_with_util(hidden_size, seq_len, batch_size, mode)
        # decode_layers.append(make_layer_profile("Residual", "Others", res_result_d))

        # # 5. RMS Norm (pre-FFN)
        # rms_ffn_result_d = perf.rms_layer_with_util(hidden_size, seq_len, batch_size, mode)
        # decode_layers.append(make_layer_profile("RMS Norm (FFN)", "Others", rms_ffn_result_d))

        # # 6. Feed Forward Network
        # ffn_result_d = perf.feed_forward_with_util(hidden_size, intermediate_size, seq_len, batch_size, mode)
        # decode_layers.append(make_layer_profile("FFN", "FFN", ffn_result_d))

        return TransformerBlockProfile(layers=decode_layers), flash_attn_tiles

    # Decode with short context (first token after prefill)
    result.decode_short_ctx = config.input_seq_len
    result.decode_short_profile, result.decode_short_flash_attn_tiles = compute_decode_profile(result.decode_short_ctx)

    # Decode with long context (last token)
    result.decode_long_ctx = config.input_seq_len + config.output_seq_len
    result.decode_long_profile, result.decode_long_flash_attn_tiles = compute_decode_profile(result.decode_long_ctx)

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


def plot_combined_timeline(
    result: ExperimentResult,
    output_path: Path,
):
    """
    Plot combined execution timeline: Compute, Memory, and Systolic Utilization.

    Single seamless plot per phase with 3 tracks sharing the same x-axis.
    Uses segment-level effective time (max of compute and memory) for timeline.
    Groups: Attention, FFN, Others with distinct colors.
    """
    FONTSIZE = 8

    # Create figure with 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 2.2))

    phases = [
        (f"Prefill (seq={result.config.input_seq_len})", result.prefill_profile),
        (f"Decode (ctx={result.decode_long_ctx})", result.decode_long_profile),
    ]

    # Group-based colors: light for compute, dark for memory
    group_colors = {
        "QKV": {"compute": "#e7d4e8", "memory": "#762a83"},
        "Attention": {"compute": "#d9f0d3", "memory": "#1b7837"},
        "FFN": {"compute": "#af8dc3", "memory": "#7fbf7b"},
        "Others": {"compute": "#d0d0d0", "memory": "#808080"},
    }

    # Groups to annotate with bidirectional arrows
    annotate_groups = ["QKV", "Attention", "FFN"]

    # Y positions: Compute=top, Memory=middle, Util=bottom
    y_compute = 0.75
    y_memory = 0.45
    bar_height = 0.18

    for col_idx, (ax, (phase_name, profile)) in enumerate(zip(axes, phases)):
        current_time = 0.0

        # Compute total time using segment-level effective time
        total_time = 0.0
        for layer in profile.layers:
            for seg in layer.segments:
                total_time += seg.effective_time_us

        # Track group spans for labeling
        group_spans = {}  # group -> (start, end)

        # =================================================================
        # Draw Compute and Memory bars at segment level
        # =================================================================
        for layer in profile.layers:
            group = layer.group if layer.group else "Others"
            colors = group_colors.get(group, group_colors["Others"])

            for seg in layer.segments:
                seg_effective_time = seg.effective_time_us

                # Track group spans
                if group not in group_spans:
                    group_spans[group] = [current_time, current_time + seg_effective_time]
                else:
                    group_spans[group][1] = current_time + seg_effective_time

                # Compute bar (top track)
                if seg.compute_time_us > 0:
                    ax.barh(y=y_compute, width=seg.compute_time_us, left=current_time,
                            height=bar_height, color=colors["compute"], edgecolor="black", linewidth=0.3)

                # Memory bar (middle track)
                if seg.memory_time_us > 0:
                    ax.barh(y=y_memory, width=seg.memory_time_us, left=current_time,
                            height=bar_height, color=colors["memory"], edgecolor="black", linewidth=0.3)

                current_time += seg_effective_time

        # Add group labels with dotted lines and vertical end markers (QKV, Attention, FFN)
        marker_y = 0.90  # Y position for markers (above compute bars)
        tick_height = 0.04  # Height of vertical end markers
        for group, (start, end) in group_spans.items():
            if group in annotate_groups:
                mid = (start + end) / 2
                width = end - start
                if width > total_time * 0.03:  # Only annotate if wide enough
                    # Draw dotted line spanning the group
                    ax.plot([start, end], [marker_y, marker_y],
                            color="black", linestyle=":", linewidth=1.0)
                    # Draw vertical line segments at both ends
                    ax.plot([start, start], [marker_y - tick_height/2, marker_y + tick_height/2],
                            color="black", linewidth=1.0)
                    ax.plot([end, end], [marker_y - tick_height/2, marker_y + tick_height/2],
                            color="black", linewidth=1.0)
                    # Add label in the center
                    ax.text(mid, marker_y + 0.03, group, ha="center", va="bottom",
                            fontsize=FONTSIZE - 1, fontweight="bold", color="black")

        # =================================================================
        # Draw Systolic Utilization (bottom track, scaled to 0-0.25 range)
        # =================================================================
        util_y_max = 0.25  # 100% utilization maps to this y value
        times = [0.0]
        utils = [0.0]
        seg_time = 0.0

        for layer in profile.layers:
            for seg in layer.segments:
                seg_eff_time = seg.effective_time_us
                util = seg.systolic_utilization / 100.0 * util_y_max  # Scale to 0-0.25
                times.append(seg_time)
                utils.append(util)
                seg_time += seg_eff_time
                times.append(seg_time)
                utils.append(util)

        times.append(seg_time)
        utils.append(0.0)

        ax.fill_between(times, utils, alpha=0.4, color="red", step="post")
        ax.plot(times, utils, color="red", linewidth=1.0, drawstyle="steps-post")

        # Add 100% reference line for utilization
        ax.axhline(y=util_y_max, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        if col_idx == 0:
            ax.text(-current_time * 0.02, util_y_max, "100%", ha="right", va="center",
                    fontsize=FONTSIZE - 2, color="red")

        # =================================================================
        # Group boundary markers
        # =================================================================
        for group, (start, end) in group_spans.items():
            ax.axvline(x=end, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

        # =================================================================
        # Formatting - seamless, no box
        # =================================================================
        ax.set_xlim(0, current_time * 1.02)
        ax.set_ylim(0, 1.0)

        # Custom y-axis labels
        ax.set_yticks([0.125, y_memory, y_compute])
        if col_idx == 0:
            ax.set_yticklabels(["Util.", "Mem.", "Comp."], fontsize=FONTSIZE)
        else:
            ax.set_yticklabels(["", "", ""])

        ax.tick_params(axis="x", labelsize=FONTSIZE)
        ax.set_xlabel("Time (μs)", fontsize=FONTSIZE)
        ax.set_title(f"{phase_name} ({current_time:.1f} μs)", fontsize=FONTSIZE)

        # Remove top and right spines for cleaner look
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"Transformer Block - {result.config.model_name} (batch={result.config.batch_size})",
        fontsize=FONTSIZE + 2, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path / "execution_timeline.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / "execution_timeline.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/execution_timeline.png")
    plt.close()


def plot_three_panel_timeline(
    results: list[ExperimentResult],
    titles: list[str],
    output_path: Path,
    filename: str = "three_panel_timeline",
    main_title: str = None,
):
    """
    Plot three vertically stacked execution timelines.

    Args:
        results: List of 3 ExperimentResult objects
        titles: List of 3 titles for each subplot
        output_path: Output directory
        filename: Output filename (without extension)
        main_title: Main title for the entire figure
    """
    FONTSIZE = 9

    # Create figure with 3 rows, 2 columns - wider and thinner
    fig, axes = plt.subplots(3, 2, figsize=(18, 6.5))

    # Group-based colors: light for compute, dark for memory
    group_colors = {
        "QKV": {"compute": "#e7d4e8", "memory": "#762a83"},
        "Attention": {"compute": "#d9f0d3", "memory": "#1b7837"},
        "FFN": {"compute": "#af8dc3", "memory": "#7fbf7b"},
        "Others": {"compute": "#d0d0d0", "memory": "#808080"},
    }

    # Groups to annotate with bidirectional arrows
    annotate_groups = ["QKV", "Attention", "FFN"]

    # Y positions: Compute=top, Memory=middle, Util=bottom
    y_compute = 0.75
    y_memory = 0.45
    bar_height = 0.18

    # First pass: compute max time for each column (prefill and decode) across all results
    max_time_prefill = 0.0
    max_time_decode = 0.0
    for result in results:
        # Compute prefill total time
        prefill_time = sum(seg.effective_time_us for layer in result.prefill_profile.layers for seg in layer.segments)
        max_time_prefill = max(max_time_prefill, prefill_time)
        # Compute decode total time
        decode_time = sum(seg.effective_time_us for layer in result.decode_long_profile.layers for seg in layer.segments)
        max_time_decode = max(max_time_decode, decode_time)

    # Store the shared x-axis limits
    shared_xlim = [max_time_prefill * 1.02, max_time_decode * 1.02]

    for row_idx, (result, row_title) in enumerate(zip(results, titles)):
        phases = [
            (f"Prefill (seq={result.config.input_seq_len})", result.prefill_profile),
            (f"Decode (ctx={result.decode_long_ctx})", result.decode_long_profile),
        ]

        for col_idx, (ax, (phase_name, profile)) in enumerate(zip(axes[row_idx], phases)):
            current_time = 0.0

            # Compute total time using segment-level effective time
            total_time = 0.0
            for layer in profile.layers:
                for seg in layer.segments:
                    total_time += seg.effective_time_us

            # Track group spans for labeling
            group_spans = {}  # group -> (start, end)

            # =================================================================
            # Draw Compute and Memory bars at segment level
            # =================================================================
            for layer in profile.layers:
                group = layer.group if layer.group else "Others"
                colors = group_colors.get(group, group_colors["Others"])

                for seg in layer.segments:
                    seg_effective_time = seg.effective_time_us

                    # Track group spans
                    if group not in group_spans:
                        group_spans[group] = [current_time, current_time + seg_effective_time]
                    else:
                        group_spans[group][1] = current_time + seg_effective_time

                    # Compute bar (top track)
                    if seg.compute_time_us > 0:
                        ax.barh(y=y_compute, width=seg.compute_time_us, left=current_time,
                                height=bar_height, color=colors["compute"], edgecolor="black", linewidth=0.3)

                    # Memory bar (middle track)
                    if seg.memory_time_us > 0:
                        ax.barh(y=y_memory, width=seg.memory_time_us, left=current_time,
                                height=bar_height, color=colors["memory"], edgecolor="black", linewidth=0.3)

                    current_time += seg_effective_time

            # Add group labels with dotted lines and vertical end markers (QKV, Attention, FFN)
            marker_y = 0.90  # Y position for markers (above compute bars)
            tick_height = 0.04  # Height of vertical end markers
            for group, (start, end) in group_spans.items():
                if group in annotate_groups:
                    mid = (start + end) / 2
                    width = end - start
                    if width > total_time * 0.03:  # Only annotate if wide enough
                        # Draw dotted line spanning the group
                        ax.plot([start, end], [marker_y, marker_y],
                                color="black", linestyle=":", linewidth=1.0)
                        # Draw vertical line segments at both ends
                        ax.plot([start, start], [marker_y - tick_height/2, marker_y + tick_height/2],
                                color="black", linewidth=1.0)
                        ax.plot([end, end], [marker_y - tick_height/2, marker_y + tick_height/2],
                                color="black", linewidth=1.0)
                        # Add label in the center
                        ax.text(mid, marker_y + 0.03, group, ha="center", va="bottom",
                                fontsize=FONTSIZE - 1, fontweight="bold", color="black")

            # =================================================================
            # Draw Systolic Utilization (bottom track, scaled to 0-0.25 range)
            # =================================================================
            util_y_max = 0.25  # 100% utilization maps to this y value
            times = [0.0]
            utils = [0.0]
            seg_time = 0.0

            for layer in profile.layers:
                for seg in layer.segments:
                    seg_eff_time = seg.effective_time_us
                    util = seg.systolic_utilization / 100.0 * util_y_max  # Scale to 0-0.25
                    times.append(seg_time)
                    utils.append(util)
                    seg_time += seg_eff_time
                    times.append(seg_time)
                    utils.append(util)

            times.append(seg_time)
            utils.append(0.0)

            ax.fill_between(times, utils, alpha=0.4, color="red", step="post")
            ax.plot(times, utils, color="red", linewidth=1.0, drawstyle="steps-post")

            # Add 100% reference line for utilization
            ax.axhline(y=util_y_max, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            if col_idx == 0:
                ax.text(-current_time * 0.02, util_y_max, "100%", ha="right", va="center",
                        fontsize=FONTSIZE - 2, color="red")

            # =================================================================
            # Group boundary markers
            # =================================================================
            for group, (start, end) in group_spans.items():
                ax.axvline(x=end, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

            # =================================================================
            # Formatting - seamless, no box
            # =================================================================
            # Use shared x-axis limits for all panels in the same column
            ax.set_xlim(0, shared_xlim[col_idx])
            ax.set_ylim(0, 1.0)

            # Custom y-axis labels
            ax.set_yticks([0.125, y_memory, y_compute])
            if col_idx == 0:
                ax.set_yticklabels(["Util.", "Mem.", "Comp."], fontsize=FONTSIZE)
            else:
                ax.set_yticklabels(["", "", ""])

            ax.tick_params(axis="x", labelsize=FONTSIZE)

            # Only show x-label on bottom row
            if row_idx == 2:
                ax.set_xlabel("Time (μs)", fontsize=FONTSIZE)

            # Only show Prefill/Decode title on first row, no time shown
            if row_idx == 0:
                # Extract just "Prefill" or "Decode" from phase_name
                phase_label = "Prefill" if "Prefill" in phase_name else "Decode"
                ax.set_title(phase_label, fontsize=FONTSIZE + 1, fontweight='bold')

            # Remove top and right spines for cleaner look
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", alpha=0.3)

        # Add row title above the row (centered between the two columns)
        # Position it above the left subplot
        axes[row_idx, 0].annotate(
            row_title, xy=(1.0, 1.15), xycoords='axes fraction',
            fontsize=FONTSIZE, fontweight='bold',
            ha='center', va='bottom'
        )

    # Add main title at center top
    if main_title:
        fig.suptitle(main_title, fontsize=FONTSIZE + 3, fontweight='bold', y=0.98)

    plt.tight_layout()
    # wspace: horizontal space between left/right (smaller = closer)
    # hspace: vertical space between rows (larger = more space)
    plt.subplots_adjust(top=0.88, hspace=0.75, wspace=0.08)
    plt.savefig(output_path / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / f"{filename}.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/{filename}.png")
    plt.close()


def plot_tile_level_attention(
    tile_data: dict,
    frequency_hz: float,
    peak_bw_gbps: float,
    output_path: Path,
    title: str = "Flash Attention Per-Tile Analysis",
    filename: str = "tile_level_attention",
):
    """
    Plot per-tile level breakdown of flash attention.

    Args:
        tile_data: dict from flash_attention_with_util containing tile-level info
        frequency_hz: clock frequency in Hz
        peak_bw_gbps: peak HBM bandwidth in GB/s
        output_path: output directory
        title: plot title
        filename: output filename (without extension)
    """
    import numpy as np

    FONTSIZE = 9
    tr = tile_data["tr"]
    tc = tile_data["tc"]
    tiles = tile_data["tiles"]
    mlen = tile_data["mlen"]
    seq_len = tile_data["seq_len"]
    kv_size = tile_data["kv_size"]

    # Set peak_bw_bytes_per_us to the minimum of (HBM BW in bytes/us) and (MLEN per cycle at frequency)
    peak_bw_bytes_per_us = min(
        peak_bw_gbps * 1e9 / 1e6,
        mlen * frequency_hz / 1e6
    )

    # Create figure with 2 rows: compute time heatmap, memory time heatmap
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Prepare data matrices
    compute_time_matrix = np.zeros((tr, tc))
    memory_time_matrix = np.zeros((tr, tc))
    total_cycles_matrix = np.zeros((tr, tc))
    total_mem_bytes_matrix = np.zeros((tr, tc))

    for r in range(tr):
        for c in range(tc):
            tile = tiles[r][c]
            total_cycles = tile["total_cycles"]
            compute_time_us = total_cycles / frequency_hz * 1e6

            total_mem_bytes = sum(
                s["memory_read_bytes"] + s["memory_write_bytes"]
                for s in tile["segments"]
            )
            memory_time_us = total_mem_bytes / peak_bw_bytes_per_us if peak_bw_bytes_per_us > 0 else 0

            compute_time_matrix[r, c] = compute_time_us
            memory_time_matrix[r, c] = memory_time_us
            total_cycles_matrix[r, c] = total_cycles
            total_mem_bytes_matrix[r, c] = total_mem_bytes

    # Plot 1: Compute time heatmap
    ax1 = axes[0, 0]
    im1 = ax1.imshow(compute_time_matrix, cmap="Blues", aspect="auto")
    ax1.set_xlabel("tc (KV tiles)", fontsize=FONTSIZE)
    ax1.set_ylabel("tr (Q tiles)", fontsize=FONTSIZE)
    ax1.set_title("Compute Time per Tile (μs)", fontsize=FONTSIZE + 1)
    plt.colorbar(im1, ax=ax1, label="μs")

    # Plot 2: Memory time heatmap
    ax2 = axes[0, 1]
    im2 = ax2.imshow(memory_time_matrix, cmap="Oranges", aspect="auto")
    ax2.set_xlabel("tc (KV tiles)", fontsize=FONTSIZE)
    ax2.set_ylabel("tr (Q tiles)", fontsize=FONTSIZE)
    ax2.set_title("Memory Time per Tile (μs)", fontsize=FONTSIZE + 1)
    plt.colorbar(im2, ax=ax2, label="μs")

    # Plot 3: Effective time (max of compute and memory)
    ax3 = axes[1, 0]
    effective_time_matrix = np.maximum(compute_time_matrix, memory_time_matrix)
    im3 = ax3.imshow(effective_time_matrix, cmap="Greens", aspect="auto")
    ax3.set_xlabel("tc (KV tiles)", fontsize=FONTSIZE)
    ax3.set_ylabel("tr (Q tiles)", fontsize=FONTSIZE)
    ax3.set_title("Effective Time per Tile (μs)", fontsize=FONTSIZE + 1)
    plt.colorbar(im3, ax=ax3, label="μs")

    # Plot 4: Memory bound indicator (memory_time > compute_time)
    ax4 = axes[1, 1]
    bound_matrix = (memory_time_matrix > compute_time_matrix).astype(float)
    ax4.imshow(bound_matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax4.set_xlabel("tc (KV tiles)", fontsize=FONTSIZE)
    ax4.set_ylabel("tr (Q tiles)", fontsize=FONTSIZE)
    ax4.set_title("Memory Bound (red=yes, green=no)", fontsize=FONTSIZE + 1)

    # Add summary text
    total_compute_time = np.sum(compute_time_matrix)
    total_memory_time = np.sum(memory_time_matrix)
    total_effective_time = np.sum(effective_time_matrix)
    pct_memory_bound = np.sum(bound_matrix) / (tr * tc) * 100

    fig.suptitle(
        f"{title}\n"
        f"seq_len={seq_len}, kv_size={kv_size}, mlen={mlen}, tiles={tr}x{tc}\n"
        f"Total: compute={total_compute_time:.1f}μs, memory={total_memory_time:.1f}μs, "
        f"effective={total_effective_time:.1f}μs, {pct_memory_bound:.0f}% tiles memory-bound",
        fontsize=FONTSIZE + 1,
    )

    plt.tight_layout()
    plt.savefig(output_path / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / f"{filename}.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/{filename}.png")
    plt.close()


def plot_tile_timeline(
    tile_data: dict,
    frequency_hz: float,
    peak_bw_gbps: float,
    output_path: Path,
    title: str = "Flash Attention Tile Timeline",
    filename: str = "tile_timeline",
):
    """
    Plot timeline showing per-tile execution with compute and memory bars.

    Args:
        tile_data: dict from flash_attention_with_util containing tile-level info
        frequency_hz: clock frequency in Hz
        peak_bw_gbps: peak HBM bandwidth in GB/s
        output_path: output directory
        title: plot title
        filename: output filename (without extension)
    """
    FONTSIZE = 9
    tr = tile_data["tr"]
    tc = tile_data["tc"]
    tiles = tile_data["tiles"]

    peak_bw_bytes_per_us = peak_bw_gbps * 1e9 / 1e6

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 6))

    # Colors for segments
    segment_colors = {
        "QKT": {"compute": "#a6cee3", "memory": "#1f78b4"},
        "softmax": {"compute": "#b2df8a", "memory": "#33a02c"},
        "PV": {"compute": "#fb9a99", "memory": "#e31a1c"},
        "writeback": {"compute": "#fdbf6f", "memory": "#ff7f00"},
    }

    y_compute = 0.7
    y_memory = 0.3
    bar_height = 0.25

    current_time = 0.0
    tile_boundaries = [0.0]

    # Draw tiles sequentially
    for r in range(tr):
        for c in range(tc):
            tile = tiles[r][c]
            tile_start = current_time

            for seg in tile["segments"]:
                compute_time = seg["cycles"] / frequency_hz * 1e6
                mem_bytes = seg["memory_read_bytes"] + seg["memory_write_bytes"]
                memory_time = mem_bytes / peak_bw_bytes_per_us if peak_bw_bytes_per_us > 0 else 0
                effective_time = max(compute_time, memory_time)

                colors = segment_colors.get(seg["name"], {"compute": "#cccccc", "memory": "#888888"})

                # Compute bar
                if compute_time > 0:
                    ax.barh(y=y_compute, width=compute_time, left=current_time,
                            height=bar_height, color=colors["compute"], edgecolor="black", linewidth=0.3)

                # Memory bar
                if memory_time > 0:
                    ax.barh(y=y_memory, width=memory_time, left=current_time,
                            height=bar_height, color=colors["memory"], edgecolor="black", linewidth=0.3)

                current_time += effective_time

            # Mark tile boundary
            tile_boundaries.append(current_time)
            ax.axvline(x=current_time, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

            # Add tile label at top
            tile_mid = (tile_start + current_time) / 2
            if tc <= 10 or c % max(1, tc // 10) == 0:
                ax.text(tile_mid, 0.95, f"({r},{c})", ha="center", va="bottom",
                        fontsize=FONTSIZE - 2, rotation=45)

    # Formatting
    ax.set_xlim(0, current_time * 1.02)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([y_memory, y_compute])
    ax.set_yticklabels(["Memory", "Compute"], fontsize=FONTSIZE)
    ax.set_xlabel("Time (μs)", fontsize=FONTSIZE)
    ax.set_title(f"{title} (tr={tr}, tc={tc}, total={current_time:.1f}μs)", fontsize=FONTSIZE + 1)

    # Legend
    legend_elements = []
    for seg_name, colors in segment_colors.items():
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=colors["compute"], label=f"{seg_name} (compute)"))
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, fc=colors["memory"], label=f"{seg_name} (memory)"))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=FONTSIZE - 1, ncol=4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / f"{filename}.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/{filename}.png")
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
    parser.add_argument(
        "--three-panel",
        action="store_true",
        help="Generate three-panel comparison plot",
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

    if args.three_panel:
        # Generate three-panel comparison with specific settings
        # Override model and sequence lengths for three-panel mode
        three_panel_model = "llama-3.1-70b"
        three_panel_input_seq = 80000
        three_panel_output_seq = 5600

        model_path_3p = model_lib_path / f"{three_panel_model}.json"
        if not model_path_3p.exists():
            print(f"Error: Model {three_panel_model} not found at {model_path_3p}")
            return

        three_panel_batch_size = 16

        config_3p = ExperimentConfig(
            model_name=three_panel_model,
            batch_size=three_panel_batch_size,
            input_seq_len=three_panel_input_seq,
            output_seq_len=three_panel_output_seq,
        )

        print("\nGenerating three-panel comparison...")
        print(f"Model: {three_panel_model}")
        print(f"Input tokens: {three_panel_input_seq}, Output tokens: {three_panel_output_seq}")

        # Configuration 1: MLEN=128, BLEN=128, HBM=1024GB/s, VLEN=1024, Self-Attention
        hw1 = HardwareOverride(
            mlen=128, blen=128, vlen=1024,
            hbm_bandwidth_gbps=1024.0,
            use_flash_attention=False
        )
        result1 = analyze_transformer_block(
            str(model_path_3p), memory_config_path, hardware_config_path,
            isa_lib_path, config_3p, hardware_override=hw1
        )
        print_profile("Config 1 (Self-Attn) PREFILL", result1.prefill_profile, result1.peak_bandwidth_gbps)

        # Configuration 2: MLEN=128, BLEN=128, HBM=1024GB/s, VLEN=1024, Flash-Attention
        hw2 = HardwareOverride(
            mlen=128, blen=128, vlen=1024,
            hbm_bandwidth_gbps=1024.0,
            use_flash_attention=True
        )
        result2 = analyze_transformer_block(
            str(model_path_3p), memory_config_path, hardware_config_path,
            isa_lib_path, config_3p, hardware_override=hw2
        )
        print_profile("Config 2 (Flash-Attn) PREFILL", result2.prefill_profile, result2.peak_bandwidth_gbps)

        # Configuration 3: MLEN=1024, BLEN=16, VLEN=1024, Flash-Attention
        hw3 = HardwareOverride(
            mlen=1024, blen=16, vlen=1024,
            hbm_bandwidth_gbps=1024.0,
            use_flash_attention=True
        )
        result3 = analyze_transformer_block(
            str(model_path_3p), memory_config_path, hardware_config_path,
            isa_lib_path, config_3p, hardware_override=hw3
        )
        print_profile("Config 3 (MLEN=1024) PREFILL", result3.prefill_profile, result3.peak_bandwidth_gbps)

        # Plot
        if not args.no_plot:
            titles = [
                "MLEN=128, BLEN=128, VLEN=1024, HBM=1024GB/s, Self-Attention",
                "MLEN=128, BLEN=128, VLEN=1024, HBM=1024GB/s, Flash-Attention",
                "MLEN=1024, BLEN=16, VLEN=1024, HBM=1024GB/s, Flash-Attention",
            ]
            main_title = f"Transformer Block Analysis - {three_panel_model} (input={three_panel_input_seq}, output={three_panel_output_seq}, batch={three_panel_batch_size})"
            plot_three_panel_timeline([result1, result2, result3], titles, output_path, main_title=main_title)
    else:
        # Run standard analysis
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

        # Generate combined plot
        if not args.no_plot:
            print("\nGenerating plot...")
            plot_combined_timeline(result, output_path)

            # Generate tile-level plots for flash attention
            if result.prefill_flash_attn_tiles is not None:
                print("\nGenerating tile-level flash attention plots...")
                # Prefill tile analysis
                plot_tile_level_attention(
                    result.prefill_flash_attn_tiles,
                    config.frequency_hz,
                    result.peak_bandwidth_gbps,
                    output_path,
                    title=f"Flash Attention Prefill (seq={config.input_seq_len})",
                    filename="tile_level_prefill",
                )
                plot_tile_timeline(
                    result.prefill_flash_attn_tiles,
                    config.frequency_hz,
                    result.peak_bandwidth_gbps,
                    output_path,
                    title=f"Flash Attention Prefill Timeline",
                    filename="tile_timeline_prefill",
                )

            if result.decode_long_flash_attn_tiles is not None:
                # Decode tile analysis
                plot_tile_level_attention(
                    result.decode_long_flash_attn_tiles,
                    config.frequency_hz,
                    result.peak_bandwidth_gbps,
                    output_path,
                    title=f"Flash Attention Decode (ctx={result.decode_long_ctx})",
                    filename="tile_level_decode",
                )
                plot_tile_timeline(
                    result.decode_long_flash_attn_tiles,
                    config.frequency_hz,
                    result.peak_bandwidth_gbps,
                    output_path,
                    title=f"Flash Attention Decode Timeline",
                    filename="tile_timeline_decode",
                )

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
