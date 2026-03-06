#!/usr/bin/env python3
"""
System Model Experiment: Transformer Block Analysis with Tile-Level Modeling.

Uses the fused SystemModel for combined compute and memory analysis.
Focuses on three-panel timeline visualization.

Usage:
    python experiments/sys_model_experiment.py
    python experiments/sys_model_experiment.py --model llama-3.1-70b
    python experiments/sys_model_experiment.py --batch-size 8 --input-seq 4096
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "analytic_models" / "performance"))

from sys_model import SystemModel, load_system_config_from_toml


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Section:
    """A single section with offset, duration, and optional value (for utilization)."""
    offset_us: float = 0.0
    duration_us: float = 0.0
    value: float = 0.0  # For utilization sections (percentage)

    @property
    def end_us(self) -> float:
        return self.offset_us + self.duration_us


@dataclass
class ExecutionSegment:
    """A segment of execution with multiple tunable sections for compute, memory, and utilization.

    Each of compute, memory, systolic, and bandwidth can have MULTIPLE sections,
    allowing fine-grained control over timing and values within a single segment.

    Section format (list of dicts or Section objects):
        {"offset_us": float, "duration_us": float, "value": float (for util)}

    Example - QKT with prefetch, compute, and writeback phases:
        compute_sections = [
            {"offset_us": 5.0, "duration_us": 10.0},  # Main compute after prefetch
        ]
        memory_sections = [
            {"offset_us": 0.0, "duration_us": 5.0},   # Prefetch read
            {"offset_us": 15.0, "duration_us": 3.0},  # Writeback
        ]
        systolic_sections = [
            {"offset_us": 5.0, "duration_us": 10.0, "value": 95.0},  # High util during compute
        ]
        bandwidth_sections = [
            {"offset_us": 0.0, "duration_us": 5.0, "value": 80.0},   # Prefetch BW
            {"offset_us": 15.0, "duration_us": 3.0, "value": 60.0},  # Writeback BW
        ]

    Backward compatible: If sections are not provided, falls back to single values.
    """

    name: str
    cycles: int = 0
    memory_read_bytes: int = 0
    memory_write_bytes: int = 0

    # Multiple sections for each type (list of Section or dict)
    compute_sections: list = field(default_factory=list)
    memory_sections: list = field(default_factory=list)
    systolic_sections: list = field(default_factory=list)
    bandwidth_sections: list = field(default_factory=list)

    # Legacy single-value fields (used if sections are empty)
    compute_time_us: float = 0.0
    compute_offset_us: float = 0.0
    memory_time_us: float = 0.0
    memory_offset_us: float = 0.0
    systolic_utilization: float = 0.0
    systolic_offset_us: float = None
    systolic_duration_us: float = None
    bandwidth_utilization: float = 0.0
    bandwidth_offset_us: float = None
    bandwidth_duration_us: float = None

    def __post_init__(self):
        """Convert dict sections to Section objects and handle legacy fields."""
        self.compute_sections = [self._to_section(s) for s in self.compute_sections]
        self.memory_sections = [self._to_section(s) for s in self.memory_sections]
        self.systolic_sections = [self._to_section(s) for s in self.systolic_sections]
        self.bandwidth_sections = [self._to_section(s) for s in self.bandwidth_sections]

        # If no sections provided, create from legacy single values
        if not self.compute_sections and self.compute_time_us > 0:
            self.compute_sections = [Section(self.compute_offset_us, self.compute_time_us)]
        if not self.memory_sections and self.memory_time_us > 0:
            self.memory_sections = [Section(self.memory_offset_us, self.memory_time_us)]
        if not self.systolic_sections and self.systolic_utilization > 0:
            offset = self.systolic_offset_us if self.systolic_offset_us is not None else self.compute_offset_us
            duration = self.systolic_duration_us if self.systolic_duration_us is not None else self.compute_time_us
            self.systolic_sections = [Section(offset, duration, self.systolic_utilization)]
        if not self.bandwidth_sections and self.bandwidth_utilization > 0:
            offset = self.bandwidth_offset_us if self.bandwidth_offset_us is not None else self.memory_offset_us
            duration = self.bandwidth_duration_us if self.bandwidth_duration_us is not None else self.memory_time_us
            self.bandwidth_sections = [Section(offset, duration, self.bandwidth_utilization)]

    @staticmethod
    def _to_section(s) -> Section:
        if isinstance(s, Section):
            return s
        return Section(
            offset_us=s.get("offset_us", 0.0),
            duration_us=s.get("duration_us", 0.0),
            value=s.get("value", 0.0),
        )

    @property
    def effective_time_us(self) -> float:
        """Total segment duration based on all sections."""
        max_end = 0.0
        for sec in self.compute_sections + self.memory_sections + self.systolic_sections + self.bandwidth_sections:
            max_end = max(max_end, sec.end_us)
        return max_end

    @property
    def is_memory_bound(self) -> bool:
        compute_total = sum(s.duration_us for s in self.compute_sections)
        memory_total = sum(s.duration_us for s in self.memory_sections)
        return memory_total > compute_total


@dataclass
class LayerProfile:
    """Profile for a single transformer layer component."""

    name: str
    group: str = ""
    segments: list = field(default_factory=list)

    @property
    def compute_time_us(self) -> float:
        return sum(s.compute_time_us for s in self.segments)

    @property
    def memory_time_us(self) -> float:
        return sum(s.memory_time_us for s in self.segments)

    @property
    def total_time_us(self) -> float:
        return max(self.compute_time_us, self.memory_time_us)


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
    use_flash_attention: bool = True
    use_prefetch: bool = True  # Whether to use prefetching/preloading for attention


@dataclass
class ExperimentResult:
    """Results from compute vs memory analysis."""

    config: ExperimentConfig
    prefill_profile: TransformerBlockProfile = field(default_factory=TransformerBlockProfile)
    decode_profile: TransformerBlockProfile = field(default_factory=TransformerBlockProfile)
    decode_ctx: int = 0
    peak_bandwidth_gbps: float = 0.0
    hardware_override: HardwareOverride = None


# =============================================================================
# Analysis Functions
# =============================================================================


def analyze_transformer_block(
    model_path: str,
    config_path: str,
    isa_lib_path: str,
    config: ExperimentConfig,
    hardware_override: HardwareOverride = None,
) -> ExperimentResult:
    """Analyze compute vs memory time for transformer block using SystemModel."""

    # Load system config
    sys_config = load_system_config_from_toml(config_path)

    # Apply hardware overrides
    if hardware_override:
        if hardware_override.mlen is not None:
            sys_config.MLEN = hardware_override.mlen
        if hardware_override.blen is not None:
            sys_config.BLEN = hardware_override.blen
        if hardware_override.vlen is not None:
            sys_config.VLEN = hardware_override.vlen

    # Load model config
    with open(model_path) as f:
        model_param = json.load(f)

    hidden_size = model_param["hidden_size"]
    num_attention_heads = model_param["num_attention_heads"]
    num_kv_heads = model_param["num_key_value_heads"]
    head_dim = model_param.get("head_dim", hidden_size // num_attention_heads)
    intermediate_size = model_param["intermediate_size"]

    # Initialize system model
    sys_model = SystemModel(sys_config, isa_lib_path, frequency_hz=config.frequency_hz)

    # Compute peak bandwidth
    peak_bw_gbps = sys_config.HBM_WIDTH // 8 * config.frequency_hz / 1e9

    result = ExperimentResult(
        config=config,
        peak_bandwidth_gbps=peak_bw_gbps,
        hardware_override=hardware_override,
    )

    # Helper to convert sys_model result to LayerProfile
    def make_layer_profile(name: str, group: str, sys_result: dict) -> LayerProfile:
        segments = []
        for seg in sys_result["segments"]:
            segments.append(ExecutionSegment(
                name=seg["name"],
                cycles=seg.get("cycles", 0),
                memory_read_bytes=seg.get("memory_read_bytes", 0),
                memory_write_bytes=seg.get("memory_write_bytes", 0),
                # Multiple sections (new flexible format)
                compute_sections=seg.get("compute_sections", []),
                memory_sections=seg.get("memory_sections", []),
                systolic_sections=seg.get("systolic_sections", []),
                bandwidth_sections=seg.get("bandwidth_sections", []),
                # Legacy single-value fields (backward compatible)
                compute_time_us=seg.get("compute_time_us", 0.0),
                compute_offset_us=seg.get("compute_offset_us", 0.0),
                memory_time_us=seg.get("memory_time_us", 0.0),
                memory_offset_us=seg.get("memory_offset_us", 0.0),
                systolic_utilization=seg.get("systolic_utilization", 0.0),
                systolic_offset_us=seg.get("systolic_offset_us"),
                systolic_duration_us=seg.get("systolic_duration_us"),
                bandwidth_utilization=seg.get("bandwidth_utilization", 0.0),
                bandwidth_offset_us=seg.get("bandwidth_offset_us"),
                bandwidth_duration_us=seg.get("bandwidth_duration_us"),
            ))
        return LayerProfile(name=name, group=group, segments=segments)

    # =========================================================================
    # PREFILL PHASE
    # =========================================================================
    mode = "prefill"
    seq_len = config.input_seq_len
    kv_size = config.input_seq_len
    batch_size = config.batch_size

    prefill_layers = []

    # RMS Norm
    rms_result = sys_model.rms_norm(hidden_size, seq_len, batch_size, mode)
    prefill_layers.append(make_layer_profile("RMS Norm", "Others", rms_result))

    # QKV Projection
    proj_result = sys_model.projection(
        hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
    )
    prefill_layers.append(make_layer_profile("QKV Projection", "QKV", proj_result))

    # Self-Attention or Flash-Attention based on config
    use_flash = hardware_override.use_flash_attention if hardware_override else True
    use_prefetch = hardware_override.use_prefetch if hardware_override else True

    if use_flash:
        if use_prefetch:
            attn_result = sys_model.flash_attention(
                num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
            )
        else:
            attn_result = sys_model.flash_attention_no_prefetch(
                num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
            )
        prefill_layers.append(make_layer_profile("Flash Attention", "Attention", attn_result))
    else:
        if use_prefetch:
            attn_result = sys_model.self_attention(
                num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
            )
        else:
            attn_result = sys_model.self_attention_no_prefetch(
                num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
            )
        prefill_layers.append(make_layer_profile("Self Attention", "Attention", attn_result))

    # Feed Forward - use no_prefetch version when prefetch is disabled
    if use_prefetch:
        ffn_result = sys_model.feed_forward(hidden_size, intermediate_size, seq_len, batch_size, mode)
    else:
        ffn_result = sys_model.feed_forward_no_prefetch(hidden_size, intermediate_size, seq_len, batch_size, mode)
    prefill_layers.append(make_layer_profile("Feed Forward", "FFN", ffn_result))

    result.prefill_profile = TransformerBlockProfile(layers=prefill_layers)

    # =========================================================================
    # DECODE PHASE
    # =========================================================================
    mode = "decode"
    seq_len = 1
    kv_size = config.input_seq_len + config.output_seq_len
    result.decode_ctx = kv_size

    decode_layers = []

    # RMS Norm
    rms_result = sys_model.rms_norm(hidden_size, seq_len, batch_size, mode)
    decode_layers.append(make_layer_profile("RMS Norm", "Others", rms_result))

    # QKV Projection
    proj_result = sys_model.projection(
        hidden_size, num_attention_heads, num_kv_heads, head_dim, seq_len, batch_size, mode
    )
    decode_layers.append(make_layer_profile("QKV Projection", "QKV", proj_result))

    # Self-Attention or Flash-Attention based on config
    if use_flash:
        if use_prefetch:
            attn_result = sys_model.flash_attention(
                num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
            )
        else:
            attn_result = sys_model.flash_attention_no_prefetch(
                num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
            )
        decode_layers.append(make_layer_profile("Flash Attention", "Attention", attn_result))
    else:
        if use_prefetch:
            attn_result = sys_model.self_attention(
                num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
            )
        else:
            attn_result = sys_model.self_attention_no_prefetch(
                num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
            )
        decode_layers.append(make_layer_profile("Self Attention", "Attention", attn_result))

    # Feed Forward - use no_prefetch version when prefetch is disabled
    if use_prefetch:
        ffn_result = sys_model.feed_forward(hidden_size, intermediate_size, seq_len, batch_size, mode)
    else:
        ffn_result = sys_model.feed_forward_no_prefetch(hidden_size, intermediate_size, seq_len, batch_size, mode)
    decode_layers.append(make_layer_profile("Feed Forward", "FFN", ffn_result))

    result.decode_profile = TransformerBlockProfile(layers=decode_layers)

    return result


# =============================================================================
# Plotting Functions
# =============================================================================


def plot_three_panel_timeline(
    results: list[ExperimentResult],
    titles: list[str],
    output_path: Path,
    filename: str = "three_panel_timeline",
    results_small: list[ExperimentResult] = None,  # Optional smaller config results
):
    """
    Plot three vertically stacked execution timelines.

    Each row has 4 columns (small prefill + small decode + large prefill + large decode) with:
    - Combined CA + Mem subplot:
      - CA bars (top) from compute_sections, with CA util (SU) % above
      - Mem bars (bottom) from memory_sections, with Mem util (BWU) % above
      - Group labels (Attention, FFN) in the middle between CA and Mem
    """
    FONTSIZE = 13

    num_cols = 4 if results_small else 2
    # 2 rows per result: combined CA+Mem + spacer
    # height_ratios for 3 results = 6 rows total
    fig_width = 6 * num_cols if results_small else 14
    fig = plt.figure(figsize=(fig_width, 9))  # Flattened: wider and shorter
    gs = GridSpec(6, num_cols, figure=fig,
                  height_ratios=[1, 0.35, 1, 0.35, 1, 0.35],
                  hspace=0.20, wspace=0.15)

    # Group colors - using the specified palette
    # #7b3294 (dark purple), #c2a5cf (light purple), #a6dba0 (light green), #008837 (bright green)
    group_colors = {
        "QKV": {"compute": "#d0d0d0", "memory": "#d0d0d0"},  # Grey (same as Others)
        "Attention": {"compute": "#a6dba0", "memory": "#a6dba0"},  # Light green
        "FFN": {"compute": "#c2a5cf", "memory": "#c2a5cf"},  # Light purple
        "Others": {"compute": "#d0d0d0", "memory": "#d0d0d0"},  # Grey
    }

    annotate_groups = ["Attention", "FFN"]

    # First pass: compute max time for each column across all rows
    if results_small:
        # 4 columns: small_prefill, small_decode, large_prefill, large_decode
        max_small_prefill = max(
            sum(seg.effective_time_us for layer in r.prefill_profile.layers for seg in layer.segments)
            for r in results_small
        )
        max_small_decode = max(
            sum(seg.effective_time_us for layer in r.decode_profile.layers for seg in layer.segments)
            for r in results_small
        )
        max_large_prefill = max(
            sum(seg.effective_time_us for layer in r.prefill_profile.layers for seg in layer.segments)
            for r in results
        )
        max_large_decode = max(
            sum(seg.effective_time_us for layer in r.decode_profile.layers for seg in layer.segments)
            for r in results
        )
        shared_xlim = [
            max(max_small_prefill * 1.02, 1.0),
            max(max_small_decode * 1.02, 1.0),
            max(max_large_prefill * 1.02, 1.0),
            max(max_large_decode * 1.02, 1.0),
        ]
    else:
        # 2 columns: prefill, decode
        max_time_prefill = max(
            sum(seg.effective_time_us for layer in r.prefill_profile.layers for seg in layer.segments)
            for r in results
        )
        max_time_decode = max(
            sum(seg.effective_time_us for layer in r.decode_profile.layers for seg in layer.segments)
            for r in results
        )
        shared_xlim = [max(max_time_prefill * 1.02, 1.0), max(max_time_decode * 1.02, 1.0)]

    for row_idx, (result, row_title) in enumerate(zip(results, titles)):
        # Build phases list based on whether we have small results
        if results_small:
            result_small = results_small[row_idx]
            phases = [
                (f"Prefill ({result_small.config.input_seq_len/1000:.1f}k)", result_small.prefill_profile),
                (f"Decode ({result_small.decode_ctx/1000:.1f}k)", result_small.decode_profile),
                (f"Prefill ({result.config.input_seq_len/1000:.0f}k)", result.prefill_profile),
                (f"Decode ({result.decode_ctx/1000:.0f}k)", result.decode_profile),
            ]
        else:
            phases = [
                (f"Prefill (seq={result.config.input_seq_len})", result.prefill_profile),
                (f"Decode (ctx={result.decode_ctx})", result.decode_profile),
            ]

        ax_timing_left = None
        # 2 rows per result: combined CA+Mem + spacer
        grid_row = row_idx * 2

        for col_idx, (phase_name, profile) in enumerate(phases):
            ax = fig.add_subplot(gs[grid_row, col_idx])

            if col_idx == 0:
                ax_timing_left = ax

            current_time = 0.0
            total_time = sum(seg.effective_time_us for layer in profile.layers for seg in layer.segments)

            group_spans = {}
            # Collect group-level utilization data for averaging
            group_util_data = {}  # {group: {"su_sum": float, "bw_sum": float, "duration": float}}

            # =================================================================
            # First pass: collect group spans and utilization data
            # =================================================================
            for layer in profile.layers:
                group = layer.group if layer.group else "Others"
                for seg in layer.segments:
                    seg_effective_time = seg.effective_time_us

                    if group not in group_spans:
                        group_spans[group] = [current_time, current_time + seg_effective_time]
                    else:
                        group_spans[group][1] = current_time + seg_effective_time

                    # Collect utilization data per group
                    if group not in group_util_data:
                        group_util_data[group] = {"su_weighted": 0.0, "bw_weighted": 0.0, "duration": 0.0}

                    # Weighted average by duration
                    for sec in seg.systolic_sections:
                        group_util_data[group]["su_weighted"] += sec.value * sec.duration_us
                    for sec in seg.bandwidth_sections:
                        group_util_data[group]["bw_weighted"] += sec.value * sec.duration_us
                    group_util_data[group]["duration"] += seg_effective_time

                    current_time += seg_effective_time

            # Calculate average utilizations per group
            group_avg_util = {}
            for group, data in group_util_data.items():
                if data["duration"] > 0:
                    avg_su = data["su_weighted"] / data["duration"]
                    avg_bw = data["bw_weighted"] / data["duration"]
                    group_avg_util[group] = {"su": avg_su, "bw": avg_bw}
                else:
                    group_avg_util[group] = {"su": 0, "bw": 0}

            # =================================================================
            # COMBINED CA + MEM SUBPLOT
            # =================================================================
            y_ca = 0.62       # CA bars at top (closer together)
            y_mem = 0.38      # Mem bars at bottom (closer together)
            bar_height = 0.18 # Reduced bar height (half of original)
            y_mid = 0.50      # Middle between CA and Mem for group labels

            # Box background colors for groups
            # #7b3294 - dark purple, #c2a5cf - light purple
            # #a6dba0 - light green, #008837 - bright green
            box_colors = {
                "Attention": "#a6dba0",  # Light green for Attention
                "FFN": "#c2a5cf",        # Light purple for FFN
            }

            current_time = 0.0

            # Draw CA bars (compute_sections)
            for layer in profile.layers:
                group = layer.group if layer.group else "Others"
                colors = group_colors.get(group, group_colors["Others"])

                for seg in layer.segments:
                    seg_effective_time = seg.effective_time_us

                    for sec in seg.compute_sections:
                        if sec.duration_us > 0:
                            ax.barh(
                                y=y_ca, width=sec.duration_us,
                                left=current_time + sec.offset_us,
                                height=bar_height, color=colors["compute"], edgecolor="black", linewidth=0.3, zorder=2
                            )

                    current_time += seg_effective_time

            # Draw Mem bars (memory_sections) with dotted/hatch pattern
            current_time = 0.0
            for layer in profile.layers:
                group = layer.group if layer.group else "Others"
                colors = group_colors.get(group, group_colors["Others"])

                for seg in layer.segments:
                    seg_effective_time = seg.effective_time_us

                    for sec in seg.memory_sections:
                        if sec.duration_us > 0:
                            ax.barh(
                                y=y_mem, width=sec.duration_us,
                                left=current_time + sec.offset_us,
                                height=bar_height, color=colors["memory"], edgecolor="black", linewidth=0.3,
                                hatch='//', zorder=2  # Slash pattern for Mem bars
                            )

                    current_time += seg_effective_time

            # Group boundaries and boxes
            from matplotlib.patches import FancyBboxPatch

            for group, (start, end) in group_spans.items():
                ax.axvline(x=end, color="gray", linestyle="--", alpha=0.3, linewidth=0.5, zorder=1)

            # Draw separate boxes for Attention and FFN with colored backgrounds
            # Colors: #7b3294 (dark purple), #c2a5cf (light purple), #a6dba0 (light green), #008837 (bright green)
            box_bottom = y_mem - bar_height/2 - 0.02
            box_top = y_ca + bar_height/2 + 0.02
            box_height = box_top - box_bottom

            skip_attention_positions = [(0, 0), (2, 0)]
            for group in ["Attention", "FFN"]:
                if group in group_spans:
                    start, end = group_spans[group]
                    block_width = end - start
                    block_center = (start + end) / 2

                    # Get background color for this group
                    bg_color = box_colors.get(group, "#d0d0d0")
                    # Darker edge colors: green for Attention, purple for FFN
                    edge_color = "#008837" if group == "Attention" else "#7b3294"

                    # Draw box with colored background
                    rect = FancyBboxPatch(
                        (start, box_bottom), block_width, box_height,
                        boxstyle="round,pad=0.01,rounding_size=0.02",
                        facecolor=bg_color, edgecolor=edge_color,
                        linewidth=1.5, linestyle='-', zorder=0, alpha=0.3
                    )
                    ax.add_patch(rect)

                    # Add group label below the box
                    if group == "Attention" and (row_idx, col_idx) in skip_attention_positions:
                        continue

                    if block_width / shared_xlim[col_idx] > 0.05:
                        label_color = "#008837" if group == "Attention" else "#7b3294"
                        ax.text(
                            block_center, box_bottom - 0.03, group,
                            ha='center', va='top', fontsize=FONTSIZE - 1,
                            fontweight='normal', color=label_color, zorder=3
                        )

            # Helper to format with slash instead of dot
            def fmt_util(val):
                return f"{val:.1f}".replace('.', '/')

            # Calculate total average for this phase
            phase_comp_total = 0.0
            phase_mem_total = 0.0
            phase_duration = 0.0
            for layer in profile.layers:
                for seg in layer.segments:
                    for sec in seg.systolic_sections:
                        phase_comp_total += sec.value * sec.duration_us
                    for sec in seg.bandwidth_sections:
                        phase_mem_total += sec.value * sec.duration_us
                    phase_duration += seg.effective_time_us
            if phase_duration > 0:
                total_comp = phase_comp_total / phase_duration
                total_mem = phase_mem_total / phase_duration
            else:
                total_comp, total_mem = 0.0, 0.0

            # Get utilization values
            attn_comp = group_avg_util.get("Attention", {}).get("su", 0)
            attn_mem = group_avg_util.get("Attention", {}).get("bw", 0)
            ffn_comp = group_avg_util.get("FFN", {}).get("su", 0)
            ffn_mem = group_avg_util.get("FFN", {}).get("bw", 0)

            # Calculate overall averages across Attention and FFN
            avg_comp = (attn_comp + ffn_comp) / 2 if (attn_comp > 0 or ffn_comp > 0) else 0
            avg_mem = (attn_mem + ffn_mem) / 2 if (attn_mem > 0 or ffn_mem > 0) else 0

            # Right side: Attn and FFN on same lines (FFN rightmost, Attn to its left)
            right_x = shared_xlim[col_idx]  # Right edge
            util_y_top = 0.93
            util_y_line2 = 0.86

            # Determine Attn position: middle for (0,0), (0,1), (1,1), otherwise left of FFN
            attn_mid_positions = [(0, 0), (0, 1), (1, 1)]
            if (row_idx, col_idx) in attn_mid_positions:
                attn_x = shared_xlim[col_idx] / 2  # Center
                attn_ha = 'center'
            else:
                attn_x = right_x - shared_xlim[col_idx] * 0.28  # Left of FFN
                attn_ha = 'right'

            # Line 1: Attn comp (green) | FFN comp (purple)
            ax.text(
                right_x, util_y_top,
                f"FFN comp:{fmt_util(ffn_comp)}%",
                ha='right', va='top', fontsize=FONTSIZE - 2,
                fontweight='normal', color='#7b3294', zorder=10
            )
            ax.text(
                attn_x, util_y_top,
                f"Attn comp:{fmt_util(attn_comp)}%",
                ha=attn_ha, va='top', fontsize=FONTSIZE - 2,
                fontweight='normal', color='#008837', zorder=10
            )

            # Line 2: Attn mem (green) | FFN mem (purple)
            ax.text(
                right_x, util_y_line2,
                f"FFN mem:{fmt_util(ffn_mem)}%",
                ha='right', va='top', fontsize=FONTSIZE - 2,
                fontweight='normal', color='#7b3294', zorder=10
            )
            ax.text(
                attn_x, util_y_line2,
                f"Attn mem:{fmt_util(attn_mem)}%",
                ha=attn_ha, va='top', fontsize=FONTSIZE - 2,
                fontweight='normal', color='#008837', zorder=10
            )

            # Left side: Avg utils (starting from 0 on x-axis)
            left_x = 0  # Left edge
            ax.text(
                left_x, util_y_top,
                f"Avg comp:{fmt_util(avg_comp)}%",
                ha='left', va='top', fontsize=FONTSIZE - 2,
                fontweight='normal', color='black', zorder=10
            )
            ax.text(
                left_x, util_y_line2,
                f"Avg mem:{fmt_util(avg_mem)}%",
                ha='left', va='top', fontsize=FONTSIZE - 2,
                fontweight='normal', color='black', zorder=10
            )

            # For first row, third column: mark stalling gaps with dark red (Attention only)
            if row_idx == 0 and col_idx == 2:
                gap_time = 0.0
                stall_boxes = []
                for layer in profile.layers:
                    layer_group = layer.group if layer.group else "Others"
                    for seg in layer.segments:
                        seg_eff_time = seg.effective_time_us
                        if layer_group == "Attention":
                            sorted_compute = sorted(seg.compute_sections, key=lambda s: s.offset_us)
                            cursor = 0.0
                            for sec in sorted_compute:
                                if sec.offset_us > cursor:
                                    stall_boxes.append((gap_time + cursor, sec.offset_us - cursor))
                                cursor = sec.offset_us + sec.duration_us
                            if cursor < seg_eff_time:
                                stall_boxes.append((gap_time + cursor, seg_eff_time - cursor))
                        gap_time += seg_eff_time

                from matplotlib.patches import Rectangle
                significant_stalls = []
                for (stall_start, stall_width) in stall_boxes:
                    if stall_width > 0:
                        rect = Rectangle(
                            (stall_start, y_ca - bar_height/2), stall_width, bar_height,
                            facecolor='none', edgecolor='darkred',
                            linewidth=1.5, linestyle='--', zorder=5
                        )
                        ax.add_patch(rect)
                        if stall_width / shared_xlim[col_idx] > 0.01:
                            significant_stalls.append((stall_start, stall_width))

                if significant_stalls:
                    attn_start, attn_end = group_spans.get("Attention", (0, 0))
                    text_x = (attn_start + attn_end) / 2
                    text_y = y_mid - 0.08

                    ax.text(
                        text_x, text_y, 'Stalling for\nmem traffic',
                        fontsize=FONTSIZE - 1, ha='center', va='top',
                        color='darkred', zorder=10
                    )

                    for (stall_start, stall_width) in significant_stalls:
                        stall_center = stall_start + stall_width / 2
                        ax.annotate(
                            '',
                            xy=(stall_center, y_ca - bar_height/2),
                            xytext=(text_x, text_y + 0.02),
                            arrowprops=dict(arrowstyle='->', color='darkred', lw=1.2),
                            zorder=10
                        )

            ax.set_xlim(0, shared_xlim[col_idx])
            ax.set_ylim(0.10, 0.95)
            ax.set_yticks([y_mem, y_ca])
            if col_idx == 0:
                ax.set_yticklabels(["Mem", "CA"], fontsize=FONTSIZE + 2)
            else:
                ax.set_yticklabels(["", ""])

            ax.set_title(phase_name, fontsize=FONTSIZE + 1, fontweight='normal')
            ax.tick_params(axis="x", labelsize=FONTSIZE + 2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="x", alpha=0.3)
            ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

            if row_idx == 2:
                ax.set_xlabel("Time (μs)", fontsize=FONTSIZE + 2)

        # Row title - centered in the middle of all four figures
        if ax_timing_left is not None:
            # Position at the center of the row (middle of 4 columns = 2.0)
            # Third row (row_idx == 2) gets dark red color and "(Ours)" suffix
            title_text = f"{row_title} (Ours)" if row_idx == 2 else row_title
            title_color = 'darkred' if row_idx == 2 else 'black'
            ax_timing_left.annotate(
                title_text, xy=(num_cols / 2, 1.12), xycoords='axes fraction',
                fontsize=FONTSIZE + 3, ha='center', va='bottom', fontweight='normal',
                color=title_color
            )

        # Add grey horizontal line separator between hardware configs (except after last row)
        if row_idx < len(results) - 1:
            # Draw line across the figure below this row's CA subplot
            line_y = 1.0 - (row_idx + 1) * (1.0 / len(results)) + 0.02
            fig.add_artist(plt.Line2D([0.05, 0.95], [line_y, line_y],
                                       transform=fig.transFigure, color='grey',
                                       linewidth=1.5, linestyle='-', alpha=0.6))

    # Calculate overall average utilization across all results (use last result = "Ours")
    final_result = results[-1]
    overall_comp_util = 0.0
    overall_mem_util = 0.0
    total_duration = 0.0

    for phase_profile in [final_result.prefill_profile, final_result.decode_profile]:
        for layer in phase_profile.layers:
            for seg in layer.segments:
                seg_duration = seg.effective_time_us
                for sec in seg.systolic_sections:
                    overall_comp_util += sec.value * sec.duration_us
                for sec in seg.bandwidth_sections:
                    overall_mem_util += sec.value * sec.duration_us
                total_duration += seg_duration

    if total_duration > 0:
        avg_comp_util = overall_comp_util / total_duration
        avg_mem_util = overall_mem_util / total_duration
    else:
        avg_comp_util = 0.0
        avg_mem_util = 0.0

    # Add legend at bottom center
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#a6dba0', edgecolor='#008837', linewidth=1.0, label='Attention'),
        Patch(facecolor='#c2a5cf', edgecolor='#7b3294', linewidth=1.0, label='FFN'),
        Patch(facecolor='#d0d0d0', edgecolor='black', linewidth=0.5, label='Others'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=FONTSIZE + 1,
               frameon=True, fancybox=True, shadow=False, bbox_to_anchor=(0.5, -0.02))

    # Add overall average utilization text at the bottom (use slash instead of dot)
    def fmt_util_bottom(val):
        return f"{val:.1f}".replace('.', '/')

    fig.text(
        0.5, -0.06,
        f"Overall Avg (Ours): c:{fmt_util_bottom(avg_comp_util)}%  |  m:{fmt_util_bottom(avg_mem_util)}%",
        ha='center', va='top', fontsize=FONTSIZE,
        fontweight='normal', color='darkred'
    )

    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.06, hspace=0.10, wspace=0.12)
    plt.savefig(output_path / f"{filename}.png", dpi=150, bbox_inches="tight")
    plt.savefig(output_path / f"{filename}.pdf", bbox_inches="tight")
    print(f"Saved: {output_path}/{filename}.png")
    plt.close()


def print_profile(name: str, profile: TransformerBlockProfile, peak_bw_gbps: float):
    """Print a formatted profile."""
    print(f"\n{'=' * 80}")
    print(f"{name} - Transformer Block Analysis (One Layer)")
    print(f"{'=' * 80}")
    print(f"Peak HBM Bandwidth: {peak_bw_gbps:.1f} GB/s")
    print()
    print(f"{'Layer':<20} {'Compute (us)':<15} {'Memory (us)':<15} {'Bound':<10}")
    print("-" * 60)

    for layer in profile.layers:
        bound = "MEMORY" if layer.memory_time_us > layer.compute_time_us else "COMPUTE"
        print(f"{layer.name:<20} {layer.compute_time_us:>12.2f}   {layer.memory_time_us:>12.2f}   {bound:<10}")

    print("-" * 60)
    print(f"{'TOTAL':<20} {profile.total_compute_time_us:>12.2f}   {profile.total_memory_time_us:>12.2f}")


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="System Model Experiment")
    parser.add_argument("--model-lib", default="compiler/doc/Model_Lib", help="Path to Model_Lib directory")
    parser.add_argument("--config", default="plena_settings.toml", help="Path to hardware config TOML")
    parser.add_argument("--isa-lib", default="analytic_models/performance/customISA_lib.json", help="Path to customISA_lib.json")
    parser.add_argument("--output-dir", default="experiments/results/sys_model", help="Output directory")
    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent
    model_lib_path = project_root / args.model_lib
    config_path = str(project_root / args.config)
    isa_lib_path = str(project_root / args.isa_lib)
    output_path = project_root / args.output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    # Three-panel configuration
    three_panel_model = "llama-3.1-70b"
    three_panel_batch_size = 16

    # Large config: 90k input, 98k decode context
    large_input_seq = 90000
    large_output_seq = 8000

    model_path = model_lib_path / f"{three_panel_model}.json"
    if not model_path.exists():
        print(f"Error: Model {three_panel_model} not found at {model_path}")
        return

    print("=" * 80)
    print("SYSTEM MODEL EXPERIMENT - TWO COLUMN PANEL")
    print("=" * 80)
    print(f"Model:           {three_panel_model}")
    print(f"Batch Size:      {three_panel_batch_size}")
    print(f"Config:          Prefill {large_input_seq}, Decode ctx {large_input_seq + large_output_seq}")
    print("=" * 80)

    # Large config
    config_large = ExperimentConfig(
        model_name=three_panel_model,
        batch_size=three_panel_batch_size,
        input_seq_len=large_input_seq,
        output_seq_len=large_output_seq,
    )

    # Configuration 1: Self-Attention WITHOUT memory overlap (no prefetch)
    hw1 = HardwareOverride(mlen=128, blen=128, vlen=1024, use_flash_attention=False, use_prefetch=False)
    result1_large = analyze_transformer_block(str(model_path), config_path, isa_lib_path, config_large, hardware_override=hw1)
    print_profile("Config 1: Self-Attention (No Memory Overlap) PREFILL", result1_large.prefill_profile, result1_large.peak_bandwidth_gbps)

    # Configuration 2: Flash-Attention WITHOUT memory overlap (no prefetch)
    hw2 = HardwareOverride(mlen=128, blen=128, vlen=1024, use_flash_attention=True, use_prefetch=False)
    result2_large = analyze_transformer_block(str(model_path), config_path, isa_lib_path, config_large, hardware_override=hw2)
    print_profile("Config 2: Flash-Attention (No Memory Overlap) PREFILL", result2_large.prefill_profile, result2_large.peak_bandwidth_gbps)

    # Configuration 3: Flash-Attention WITH memory overlap (with prefetch)
    hw3 = HardwareOverride(mlen=1024, blen=16, vlen=1024, use_flash_attention=True, use_prefetch=True)
    result3_large = analyze_transformer_block(str(model_path), config_path, isa_lib_path, config_large, hardware_override=hw3)
    print_profile("Config 3: Flash-Attention (With Memory Overlap) PREFILL", result3_large.prefill_profile, result3_large.peak_bandwidth_gbps)

    # Plot with 2 columns (Prefill 90k, Decode 98k)
    titles = [
        "Self-Attention (No Memory Overlap)",
        "Flash-Attention (No Memory Overlap)",
        "Flash-Attention (With Memory Overlap)",
    ]
    plot_three_panel_timeline(
        [result1_large, result2_large, result3_large],
        titles,
        output_path,
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
