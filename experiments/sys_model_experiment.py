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

    if use_flash:
        attn_result = sys_model.flash_attention(
            num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
        )
        prefill_layers.append(make_layer_profile("Flash Attention", "Attention", attn_result))
    else:
        attn_result = sys_model.self_attention(
            num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
        )
        prefill_layers.append(make_layer_profile("Self Attention", "Attention", attn_result))

    # Feed Forward
    ffn_result = sys_model.feed_forward(hidden_size, intermediate_size, seq_len, batch_size, mode)
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
        attn_result = sys_model.flash_attention(
            num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
        )
        decode_layers.append(make_layer_profile("Flash Attention", "Attention", attn_result))
    else:
        attn_result = sys_model.self_attention(
            num_attention_heads, num_kv_heads, head_dim, seq_len, kv_size, batch_size, mode
        )
        decode_layers.append(make_layer_profile("Self Attention", "Attention", attn_result))

    # Feed Forward
    ffn_result = sys_model.feed_forward(hidden_size, intermediate_size, seq_len, batch_size, mode)
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
    - Top: Compute + Memory timing bars
    - Bottom: Systolic and Bandwidth utilization
    """
    FONTSIZE = 13

    num_cols = 4 if results_small else 2
    fig = plt.figure(figsize=(6 * num_cols, 14))
    gs = GridSpec(8, num_cols, figure=fig, height_ratios=[1.2, 1, 0.35, 1.2, 1, 0.35, 1.2, 1], hspace=0.18, wspace=0.12)

    # Group colors - same color for compute and memory within each group
    group_colors = {
        "QKV": {"compute": "#c2a5cf", "memory": "#c2a5cf"},  # Light purple
        "Attention": {"compute": "#a6dba0", "memory": "#a6dba0"},  # Light green
        "FFN": {"compute": "#7b3294", "memory": "#7b3294"},  # Dark purple
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
                (f"Prefill ({result_small.config.input_seq_len//1000}k)", result_small.prefill_profile),
                (f"Decode ({result_small.decode_ctx//1000}k)", result_small.decode_profile),
                (f"Prefill ({result.config.input_seq_len//1000}k)", result.prefill_profile),
                (f"Decode ({result.decode_ctx//1000}k)", result.decode_profile),
            ]
        else:
            phases = [
                (f"Prefill (seq={result.config.input_seq_len})", result.prefill_profile),
                (f"Decode (ctx={result.decode_ctx})", result.decode_profile),
            ]

        ax_timing_left = None
        grid_row_timing = row_idx * 3
        grid_row_util = grid_row_timing + 1

        for col_idx, (phase_name, profile) in enumerate(phases):
            ax_timing = fig.add_subplot(gs[grid_row_timing, col_idx])
            ax_util = fig.add_subplot(gs[grid_row_util, col_idx], sharex=ax_timing)

            if col_idx == 0:
                ax_timing_left = ax_timing

            current_time = 0.0
            total_time = sum(seg.effective_time_us for layer in profile.layers for seg in layer.segments)

            group_spans = {}

            # =================================================================
            # TIMING SUBPLOT
            # =================================================================
            y_compute = 0.62
            y_memory = 0.38
            bar_height = 0.12

            for layer in profile.layers:
                group = layer.group if layer.group else "Others"
                colors = group_colors.get(group, group_colors["Others"])

                for seg in layer.segments:
                    seg_effective_time = seg.effective_time_us

                    if group not in group_spans:
                        group_spans[group] = [current_time, current_time + seg_effective_time]
                    else:
                        group_spans[group][1] = current_time + seg_effective_time

                    # Draw multiple compute sections
                    for sec in seg.compute_sections:
                        if sec.duration_us > 0:
                            ax_timing.barh(
                                y=y_compute, width=sec.duration_us,
                                left=current_time + sec.offset_us,
                                height=bar_height, color=colors["compute"], edgecolor="black", linewidth=0.3, zorder=2
                            )

                    # Draw multiple memory sections
                    for sec in seg.memory_sections:
                        if sec.duration_us > 0:
                            ax_timing.barh(
                                y=y_memory, width=sec.duration_us,
                                left=current_time + sec.offset_us,
                                height=bar_height, color=colors["memory"], edgecolor="black", linewidth=0.3, zorder=2
                            )

                    current_time += seg_effective_time


            # Group boundaries
            for group, (start, end) in group_spans.items():
                ax_timing.axvline(x=end, color="gray", linestyle="--", alpha=0.3, linewidth=0.5, zorder=1)

            # Box around timing bars
            box_margin = 0.03
            box_y_bottom = y_memory - bar_height/2 - box_margin
            box_y_top = y_compute + bar_height/2 + box_margin
            box = FancyBboxPatch(
                (0, box_y_bottom), shared_xlim[col_idx], box_y_top - box_y_bottom,
                boxstyle="round,pad=0.005,rounding_size=0.015",
                facecolor="#f5f5f5", edgecolor="#888888", linewidth=0.8, linestyle="-", zorder=1,
            )
            ax_timing.add_patch(box)

            # Timing formatting
            ax_timing.set_xlim(0, shared_xlim[col_idx])
            ax_timing.set_ylim(0.25, 0.75)
            ax_timing.set_yticks([y_memory, y_compute])
            if col_idx == 0:
                ax_timing.set_yticklabels(["Mem", "Comp"], fontsize=FONTSIZE + 2)
            else:
                ax_timing.set_yticklabels(["", ""])
            ax_timing.tick_params(axis="x", labelbottom=False, labelsize=FONTSIZE)
            ax_timing.spines["top"].set_visible(False)
            ax_timing.spines["right"].set_visible(False)
            ax_timing.spines["bottom"].set_visible(False)
            ax_timing.grid(axis="x", alpha=0.3)

            ax_timing.set_title(phase_name, fontsize=FONTSIZE + 1, fontweight='normal')

            # =================================================================
            # UTILIZATION SUBPLOT
            # =================================================================
            y_systolic = 0.55
            y_bw = 0.15
            util_height = 0.30

            systolic_times = [0.0]
            systolic_utils = [y_systolic]
            bw_times = [0.0]
            bw_utils = [y_bw]
            seg_time = 0.0

            for layer in profile.layers:
                for seg in layer.segments:
                    seg_eff_time = seg.effective_time_us

                    # Build systolic utilization from multiple sections
                    sorted_sys = sorted(seg.systolic_sections, key=lambda s: s.offset_us)
                    sys_cursor = 0.0
                    for sec in sorted_sys:
                        sec_start = sec.offset_us
                        sec_end = sec.offset_us + sec.duration_us
                        s_util_y = y_systolic + (sec.value / 100.0) * util_height

                        # Gap before this section
                        if sec_start > sys_cursor:
                            systolic_times.extend([seg_time + sys_cursor, seg_time + sec_start])
                            systolic_utils.extend([y_systolic, y_systolic])
                        # This section's utilization
                        systolic_times.extend([seg_time + sec_start, seg_time + sec_end])
                        systolic_utils.extend([s_util_y, s_util_y])
                        sys_cursor = sec_end

                    # Fill remaining gap to segment end
                    if sys_cursor < seg_eff_time:
                        systolic_times.extend([seg_time + sys_cursor, seg_time + seg_eff_time])
                        systolic_utils.extend([y_systolic, y_systolic])

                    # Build bandwidth utilization from multiple sections
                    sorted_bw = sorted(seg.bandwidth_sections, key=lambda s: s.offset_us)
                    bw_cursor = 0.0
                    for sec in sorted_bw:
                        sec_start = sec.offset_us
                        sec_end = sec.offset_us + sec.duration_us
                        b_util_y = y_bw + (sec.value / 100.0) * util_height

                        # Gap before this section
                        if sec_start > bw_cursor:
                            bw_times.extend([seg_time + bw_cursor, seg_time + sec_start])
                            bw_utils.extend([y_bw, y_bw])
                        # This section's utilization
                        bw_times.extend([seg_time + sec_start, seg_time + sec_end])
                        bw_utils.extend([b_util_y, b_util_y])
                        bw_cursor = sec_end

                    # Fill remaining gap to segment end
                    if bw_cursor < seg_eff_time:
                        bw_times.extend([seg_time + bw_cursor, seg_time + seg_eff_time])
                        bw_utils.extend([y_bw, y_bw])

                    seg_time += seg_eff_time

            systolic_times.append(seg_time)
            systolic_utils.append(y_systolic)
            bw_times.append(seg_time)
            bw_utils.append(y_bw)

            ax_util.fill_between(systolic_times, y_systolic, systolic_utils, step="post", color="#018571", alpha=0.2)
            ax_util.plot(systolic_times, systolic_utils, color="#018571", linewidth=1.2, drawstyle="steps-post")
            ax_util.fill_between(bw_times, y_bw, bw_utils, step="post", color="#2171b5", alpha=0.2)
            ax_util.plot(bw_times, bw_utils, color="#2171b5", linewidth=1.2, drawstyle="steps-post")

            ax_util.axhline(y=y_systolic + util_height, color="#018571", linestyle="--", alpha=0.4, linewidth=0.6)
            ax_util.axhline(y=y_bw + util_height, color="#2171b5", linestyle="--", alpha=0.4, linewidth=0.6)

            for group, (start, end) in group_spans.items():
                ax_util.axvline(x=end, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

            ax_util.set_xlim(0, shared_xlim[col_idx])
            ax_util.set_ylim(0.0, 0.95)
            ax_util.set_yticks([y_bw + util_height/2, y_systolic + util_height/2])

            if col_idx == 0:
                ax_util.set_yticklabels(["BW", "Sys"], fontsize=FONTSIZE + 2)
            else:
                ax_util.set_yticklabels(["", ""])

            ax_util.tick_params(axis="x", labelsize=FONTSIZE + 2)
            ax_util.spines["top"].set_visible(False)
            ax_util.spines["right"].set_visible(False)
            ax_util.grid(axis="x", alpha=0.3)

            # Use scientific notation for x-axis
            ax_util.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))

            if row_idx == 2:
                ax_util.set_xlabel("Time (μs)", fontsize=FONTSIZE + 2)

        # Row title - centered in the middle of all four figures
        if ax_timing_left is not None:
            # Position at the center of the row (middle of 4 columns = 2.0)
            # Third row (row_idx == 2) gets red color and "(Ours)" suffix
            title_text = f"{row_title} (Ours)" if row_idx == 2 else row_title
            title_color = 'red' if row_idx == 2 else 'black'
            ax_timing_left.annotate(
                title_text, xy=(num_cols / 2, 1.15), xycoords='axes fraction',
                fontsize=FONTSIZE + 3, ha='center', va='bottom', fontweight='normal',
                color=title_color
            )

        # Add grey horizontal line separator between hardware configs (except after last row)
        if row_idx < len(results) - 1:
            # Draw line across the figure below this row's utilization subplot
            # Adjust offset: lower line (row_idx=1) needs to be higher
            offset = -0.005 if row_idx == 0 else 0
            line_y = 1.0 - (row_idx + 1) * (1.0 / len(results)) + offset
            fig.add_artist(plt.Line2D([0.05, 0.95], [line_y, line_y],
                                       transform=fig.transFigure, color='grey',
                                       linewidth=1.5, linestyle='-', alpha=0.6))

    # Add legend at bottom center
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#a6dba0', edgecolor='black', linewidth=0.5, label='Attention'),
        Patch(facecolor='#7b3294', edgecolor='black', linewidth=0.5, label='FFN'),
        Patch(facecolor='#d0d0d0', edgecolor='black', linewidth=0.5, label='Others'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=FONTSIZE + 1,
               frameon=True, fancybox=True, shadow=False, bbox_to_anchor=(0.5, -0.025))

    plt.subplots_adjust(left=0.04, right=0.98, top=0.95, bottom=0.04, hspace=0.12, wspace=0.12)
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

    # Large config: 80k input, 85.6k decode context
    large_input_seq = 80000
    large_output_seq = 5000

    # Small config: 8.5k input, 20k decode context
    small_input_seq = 2000
    small_output_seq = 500  # 8500 + 11500 = 20000

    model_path = model_lib_path / f"{three_panel_model}.json"
    if not model_path.exists():
        print(f"Error: Model {three_panel_model} not found at {model_path}")
        return

    print("=" * 80)
    print("SYSTEM MODEL EXPERIMENT - FOUR COLUMN PANEL")
    print("=" * 80)
    print(f"Model:           {three_panel_model}")
    print(f"Batch Size:      {three_panel_batch_size}")
    print(f"Small Config:    Prefill {small_input_seq}, Decode ctx {small_input_seq + small_output_seq}")
    print(f"Large Config:    Prefill {large_input_seq}, Decode ctx {large_input_seq + large_output_seq}")
    print("=" * 80)

    # Large config
    config_large = ExperimentConfig(
        model_name=three_panel_model,
        batch_size=three_panel_batch_size,
        input_seq_len=large_input_seq,
        output_seq_len=large_output_seq,
    )

    # Small config
    config_small = ExperimentConfig(
        model_name=three_panel_model,
        batch_size=three_panel_batch_size,
        input_seq_len=small_input_seq,
        output_seq_len=small_output_seq,
    )

    # Configuration 1: MLEN=128, BLEN=128, VLEN=1024, Self-Attention
    hw1 = HardwareOverride(mlen=128, blen=128, vlen=1024, use_flash_attention=False)
    result1_large = analyze_transformer_block(str(model_path), config_path, isa_lib_path, config_large, hardware_override=hw1)
    result1_small = analyze_transformer_block(str(model_path), config_path, isa_lib_path, config_small, hardware_override=hw1)
    print_profile("Config 1 (Self-Attn) PREFILL", result1_large.prefill_profile, result1_large.peak_bandwidth_gbps)

    # Configuration 2: MLEN=128, BLEN=128, VLEN=1024, Flash-Attention
    hw2 = HardwareOverride(mlen=128, blen=128, vlen=1024, use_flash_attention=True)
    result2_large = analyze_transformer_block(str(model_path), config_path, isa_lib_path, config_large, hardware_override=hw2)
    result2_small = analyze_transformer_block(str(model_path), config_path, isa_lib_path, config_small, hardware_override=hw2)
    print_profile("Config 2 (Flash-Attn) PREFILL", result2_large.prefill_profile, result2_large.peak_bandwidth_gbps)

    # Configuration 3: MLEN=1024, BLEN=16, VLEN=1024, Flash-Attention
    hw3 = HardwareOverride(mlen=1024, blen=16, vlen=1024, use_flash_attention=True)
    result3_large = analyze_transformer_block(str(model_path), config_path, isa_lib_path, config_large, hardware_override=hw3)
    result3_small = analyze_transformer_block(str(model_path), config_path, isa_lib_path, config_small, hardware_override=hw3)
    print_profile("Config 3 (MLEN=1024) PREFILL", result3_large.prefill_profile, result3_large.peak_bandwidth_gbps)

    # Plot with 4 columns
    titles = [
        "MLEN=128, BLEN=128, VLEN=1024, Self-Attention",
        "MLEN=128, BLEN=128, VLEN=1024, Flash-Attention",
        "MLEN=1024, BLEN=16, VLEN=1024, Flash-Attention",
    ]
    plot_three_panel_timeline(
        [result1_large, result2_large, result3_large],
        titles,
        output_path,
        results_small=[result1_small, result2_small, result3_small],
    )

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
