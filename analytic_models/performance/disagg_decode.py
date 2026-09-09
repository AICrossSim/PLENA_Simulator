"""
Decode-chip analytic model for disaggregated serving on PLENA.

Models the decode chip only. The prefill chip transfers its BF16 KV cache
unchanged; the decode chip quantizes that cache once during admission.
Compiler-trace timing composes each sequential stage as
``sum(max(stage_compute, stage_request_memory))``.  The explicitly named
legacy aggregate-bandwidth mode retains ``max(compute, HBM bytes/bandwidth)``
for compatibility and sensitivity analysis. One code path serves every model;
only the model JSON differs (``--model``).

Precision has two widths:
  N — the HBM stream widths (attnW / ffnW / KV). These set memory_time and the
      MLEN bandwidth cap.
  M — the MAC compute width (`--m-bits`, default = widest MAC operand
      max(attnW, ffnW, KV), plus the activation compute width when a CSV point
      carries it). M sets the iso-area compute density: at fixed area an M-bit
      array fits (4/M)^k times the reference multipliers (`--density-exp`).
      Default k = 0 means compute is precision-neutral (the memory-bound
      "upcast back to original compute" assumption); k = 2 turns on the density
      layer once a Synopsys DC sweep calibrates it.
HBM is fixed technology, not a free knob: `--hbm-gen`/`--hbm-channels` set
bandwidth AND capacity together from a real generation x channel count.

Modes:
  default     report one (precision, hardware, batch) point
  --search    right-size the hardware for a fixed precision
  --sweep     accuracy-vs-cost front on the decode chip (from the software DSE)
  --codesign  joint precision x hardware: right-size the array per precision
  --compare   theoretical comparison from published device peak specifications

Algorithmic memory pressure and realized matrix-issue serialization are
classified independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ANALYTIC = _HERE.parent
_SIMULATOR_ROOT = _ANALYTIC.parent
_COMPILER_ROOT = _SIMULATOR_ROOT / "compiler"
_RTL_ROOT = _SIMULATOR_ROOT.parent / "PLENA_RTL"
for _sub in ("performance", "memory", "utilisation", "roofline", "disagg_serve"):
    p = str(_ANALYTIC / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from perf_model import PerfModel, load_hardware_config_from_toml           # noqa: E402
from decode_timing import (                                                # noqa: E402
    DRAIN_OVERLAPPED,
    EMULATOR_EVIDENCE_TIER,
    IDEAL_MATRIX_PIPELINE,
    RTL_SERIALIZED,
    STEP_COMPOSITION,
    TIMING_MODES,
    TimingEvidence,
    cycles_to_seconds,
    validate_timing_evidence,
)
from decode_crossover import DecodeCrossoverPoint                          # noqa: E402
from packed_q1_timing import (                                             # noqa: E402
    PackedQ1TimingContract,
    validate_packed_q1_timing_contract,
)
from memory_model import (                                                  # noqa: E402
    MemoryConfig,
    MemoryModel,
    MemoryTraffic,
    conservative_unique_experts,
    expected_unique_experts,
    load_memory_config_from_toml,
)
from llm_memory_model import LLMMemoryModel                                # noqa: E402
from packed_kv import (                                                     # noqa: E402
    DENSE_SELECTOR,
    DRAIN_ACCUMULATOR_BYTES_PER_CHIP,
    PACKED_KV_MODES,
    architecture_option_area_mm2,
    kv_head_reuse_status,
    traffic_from_precision,
)
from physical_ledger import (                                               # noqa: E402
    DecodeStepTrafficLedger,
    KVLedger,
    PhysicalDecodeLedger,
    PlaneBytes,
    SRAMLedger,
    WeightLedger,
    build_physical_decode_ledger,
    decode_step_traffic_ledger,
    kv_ledger,
    weight_ledger,
)
from body_weight_layout import (                                            # noqa: E402
    EXPERT_ID_PARALLEL,
    EXPERT_PARALLEL_MODES,
    EXPERT_TENSOR_PARALLEL,
    BodyWeightPhysicalLayout,
    build_body_weight_physical_layout,
)
from utilisation_model import PLENAUtilization                             # noqa: E402
from hbm_technology import HBM_TECHNOLOGIES, hbm_technology                # noqa: E402
from decode_power import decode_power                                      # noqa: E402
from handoff import LINK_GENS                                               # noqa: E402
from emulator_calibration import EmulatorCalibration                       # noqa: E402
from router_trace_summary import validate_model_overlay                    # noqa: E402
try:                                                                        # noqa: E402
    from .compiler_trace_timing import (
        COMPILER_TRACE,
        FULL_MODEL_DECODE_SCOPE,
        LEGACY_AGGREGATE_BANDWIDTH,
        TRACE_STEP_COMPOSITION,
        canonical_sha256,
        resolve_decode_step_timing,
    )
except ImportError:                                                         # noqa: E402
    from compiler_trace_timing import (
        COMPILER_TRACE,
        FULL_MODEL_DECODE_SCOPE,
        LEGACY_AGGREGATE_BANDWIDTH,
        TRACE_STEP_COMPOSITION,
        canonical_sha256,
        resolve_decode_step_timing,
    )

FREQ_HZ = 1.0e9                      # 1 GHz clock
SCALE_BITS = 8                       # one 8-bit shared scale per MX block (E8M0)
ACT_BITS = 16                        # activations stored bf16 on-chip (never HBM); computed low-precision
LM_HEAD_BITS = 16                    # vocab projection left unquantised by the software DSE
EMBED_BITS = 16                      # embedding table stored at bf16
DECODE_BF16_HEAD = "decode_bf16_unmodeled"
DECODE_MX_HEAD = "decode_local_mx_head"
EXTERNAL_BF16_HEAD = "external_bf16_service"
OUTPUT_HEAD_LOCATIONS = frozenset(
    {DECODE_BF16_HEAD, DECODE_MX_HEAD, EXTERNAL_BF16_HEAD}
)
DEFAULT_LINK_GENERATION = "nvlink4"
COMPILER_TRACE_TIMING_SET_SCHEMA = "plena-compiler-trace-timing-set-v1"
SRAM_POLICIES = frozenset(
    {
        "streaming",
        "projection_resident",
        "kv_resident_25",
        "kv_resident_50",
        "kv_resident_75",
        "kv_resident_100",
    }
)

BODY_PARALLEL_TIMING_SCHEMA = "plena-body-rank-local-timing/v1"
LAYER_EXACT_MOE_ROUTE_PROJECTION_SCHEMA = (
    "plena-layer-exact-moe-route-projection/v1"
)


def decoder_owns_output_head(location: str) -> bool:
    """Return whether LM-head work and storage belong to the decode ledger."""

    if location not in OUTPUT_HEAD_LOCATIONS:
        raise ValueError(f"unsupported output-head location {location!r}")
    return location in {DECODE_BF16_HEAD, DECODE_MX_HEAD}


def _validate_layer_exact_moe_route_projection(
    projection: Mapping[str, object],
    dims: Mapping[str, object],
    *,
    batch: int,
    tp: int,
    kvp: int,
    expert_parallel_mode: str,
) -> tuple[tuple[float, ...], tuple[PlaneBytes, ...], tuple[PlaneBytes, ...]]:
    """Validate a content-addressed, layer-specific routed-expert override.

    The normal decode model owns every non-expert operation.  This hook may
    replace only the routed-expert cycles and its streamed element/scale
    planes.  Keeping one record per decoder layer prevents a global maximum
    from inventing a route pattern that never occurred.
    """

    if not isinstance(projection, Mapping):
        raise ValueError("layer-exact MoE route projection must be an object")
    body = dict(projection)
    observed_hash = body.pop("content_hash", None)
    if observed_hash != canonical_sha256(body):
        raise ValueError("layer-exact MoE route projection content hash mismatch")
    if projection.get("schema") != LAYER_EXACT_MOE_ROUTE_PROJECTION_SCHEMA:
        raise ValueError("unsupported layer-exact MoE route projection schema")
    if (
        int(projection.get("batch_size", -1)) != batch
        or int(projection.get("tensor_parallel_degree", -1)) != tp
        or int(projection.get("kv_parallel_degree", -1)) != kvp
        or projection.get("expert_parallel_mode") != expert_parallel_mode
    ):
        raise ValueError("layer-exact MoE route projection topology differs")
    if (
        expert_parallel_mode == EXPERT_ID_PARALLEL
        and projection.get("mapping")
        != "replicated_hidden_local_route_filter_then_output_allreduce"
    ):
        raise ValueError("layer-exact expert-ID mapping token differs")
    classification = projection.get("classification")
    if not isinstance(classification, Mapping) or any(
        classification.get(field) is not False
        for field in (
            "publication_rankable",
            "hardware_rankable",
            "selection_eligible",
            "timing_selection_allowed",
        )
    ):
        raise ValueError("layer-exact MoE route projection must remain fail-closed")
    if projection.get("global_layer_collapse_allowed") is not False:
        raise ValueError("layer-exact MoE route projection permits layer collapse")
    layers = projection.get("layers")
    layer_count = int(dims["layers"])
    if (
        not isinstance(layers, list)
        or len(layers) != layer_count
        or [int(row.get("layer", -1)) for row in layers]
        != list(range(layer_count))
    ):
        raise ValueError("layer-exact MoE route projection lacks layer coverage")

    expected_assignments = batch * int(dims["experts_per_token"])
    expected_collectives = int(tp > 1)
    cycles: list[float] = []
    rank_planes: list[PlaneBytes] = []
    system_planes: list[PlaneBytes] = []
    for row in layers:
        if (
            int(row.get("logical_route_assignments", -1))
            != expected_assignments
            or int(row.get("physical_whole_expert_assignments_across_kvp", -1))
            != expected_assignments * kvp
            or int(row.get("source_hidden_dispatch_bytes", -1)) != 0
            or int(row.get("expert_output_collective_count", -1))
            != expected_collectives
        ):
            raise ValueError("layer-exact MoE route projection does not conserve")
        value = float(row.get("slowest_rank_expert_stage_cycles", -1))
        if not math.isfinite(value) or value <= 0:
            raise ValueError("layer-exact expert cycles must be finite and positive")

        def plane(field: str) -> PlaneBytes:
            raw = row.get(field)
            if not isinstance(raw, Mapping):
                raise ValueError("layer-exact expert plane is missing")
            element = int(raw.get("element_aligned", -1))
            scale = int(raw.get("scale_aligned", -1))
            total = int(raw.get("total_aligned", -1))
            if min(element, scale) < 0 or total != element + scale or total <= 0:
                raise ValueError("layer-exact expert plane does not conserve")
            return PlaneBytes(
                element_raw=element,
                element_aligned=element,
                scale_raw=scale,
                scale_aligned=scale,
            )

        rank_plane = plane("slowest_rank_expert_streamed")
        system_plane = plane("system_expert_streamed")
        if system_plane.total_aligned < rank_plane.total_aligned:
            raise ValueError("system expert traffic is below one rank")
        cycles.append(value)
        rank_planes.append(rank_plane)
        system_planes.append(system_plane)
    return tuple(cycles), tuple(rank_planes), tuple(system_planes)
# Mirrors the MXFP formats the software quantiser supports: 4-bit E1M2/E2M1,
# 6-bit E2M3/E3M2, 8-bit E3M4/E4M3/E5M2.
MXFP_FORMATS = {"E1M2": (1, 2), "E2M1": (2, 1), "E2M3": (2, 3), "E3M2": (3, 2),
                "E3M4": (3, 4), "E4M3": (4, 3), "E5M2": (5, 2)}

# Reference array: 4096 multipliers = 0.237 mm^2 (7nm), multiplying 4-bit MX
# operands. MLEN*BLEN counts positions in this 4-bit-MAC unit (the area budget).
REF_MULTIPLIERS = 4096
REF_MM2 = 0.237
MM2_PER_MULTIPLIER = REF_MM2 / REF_MULTIPLIERS

# Iso-area compute density: at fixed area an M-bit array fits (REF_MAC_BITS/M)^k
# times the reference multiplier count (multiplier area grows ~quadratically with
# width, so k ~= 2 once calibrated). Default k = 0 makes compute precision-neutral
# (the memory-bound "upcast back to original compute" assumption). Set
# --density-exp 2.0 to turn on the density layer.
REF_MAC_BITS = 4
DENSITY_EXP = 0.0

RESULTS_PROVENANCE_SCHEMA = "decode-sweep-results-provenance"
WORKSPACE_PROVENANCE_SCHEMA = "decode-sweep-provenance"

HBM_GENS = {
    name: {
        "ch_bits": 64,
        "gbps": technology.pin_rate_gbps,
        "stack_ch": technology.interface_units_per_stack,
        "ch_gb": technology.capacity_gb_per_interface_unit,
    }
    for name, technology in HBM_TECHNOLOGIES.items()
}


def hbm_overrides(gen: str, channels: int) -> dict:
    """Return coupled bandwidth and capacity for 64-bit interface units."""

    return hbm_technology(gen).overrides(channels, clock_hz=FREQ_HZ)


# Theoretical roofline references for --compare. GPUs/TPU use published BF16
# peak and HBM specifications; these rows are not measured baselines.
# The PLENA entry is filled in from the live config at runtime (plena_device)
# so the compare table and the main report describe the same chip.
# `count` sizes each system (16 small accelerators vs 4 big GPUs, roughly equal
# multiplier budgets). `sq_dim` is the square tile the decode batch must fill
# (PLENA's tile is its small BLEN).
DEVICES = {
    "plena": dict(label="PLENA-sys", kind="plena", count=16),
    "a100":  dict(label="A100",    kind="gpu", count=4,  peak_tflops=312.0,  hbm_gb=80, hbm_tbs=2.039, sq_dim=128),
    "h100":  dict(label="H100",    kind="gpu", count=4,  peak_tflops=989.5,  hbm_gb=80, hbm_tbs=3.35, sq_dim=128),
    "tpu":   dict(label="TPU-v6e", kind="tpu", count=16, peak_tflops=918.0,  hbm_gb=32, hbm_tbs=1.56, sq_dim=256),
}


def plena_device(hw_cfg, base_mem) -> dict:
    """PLENA compare entry built from the live config, so geometry, bandwidth and
    capacity have one source of truth and the compare table matches the model."""
    return {**DEVICES["plena"], "mlen": hw_cfg.MLEN, "blen": hw_cfg.BLEN,
            "hbm_gb": base_mem.HBM_SIZE / 1e9, "hbm_tbs": peak_hbm_bw_bytes(hw_cfg) / 1e12}


def device_peaks(dev: dict, prec: dict | None = None) -> tuple[float, float, float]:
    """System totals (peak compute FLOP/s, peak bandwidth B/s, capacity bytes).
    PLENA peak = 2*MLEN*BLEN*clock x density(M); if `prec` is given, MLEN is also
    capped by the bandwidth bound MLEN*max(W,KV) <= HBM_WIDTH."""
    n = dev["count"]
    if dev["kind"] == "plena":
        mlen, density = dev["mlen"], 1.0
        if prec is not None:
            mlen = min(mlen, int(dev["hbm_tbs"] * 1e12 * 8 / FREQ_HZ // stream_bits(prec)))
            density = compute_density(prec)
        per_compute = 2 * mlen * dev["blen"] * FREQ_HZ * density
    else:
        per_compute = dev["peak_tflops"] * 1e12
    return per_compute * n, dev["hbm_tbs"] * 1e12 * n, dev["hbm_gb"] * 1e9 * n


def fill_util(batch: int, m_tile: int) -> float:
    """Fraction of the array's M-tile filled by the decode batch: B / (ceil(B/m_tile)*m_tile).

    A large square tile (GPU 128, TPU 256) sits mostly idle at small batch, while
    PLENA's small BLEN fills at low batch — the flattened-array advantage.
    """
    return batch / (math.ceil(batch / m_tile) * m_tile)


def effective_bits(fmt: str, width, block: int) -> float:
    """Average stored bits per element, including the per-block shared scale (SCALE_BITS/block)."""
    elem = int(width) if fmt == "mxint" else 1 + int(width[0]) + int(width[1])   # MXFP: sign+exp+frac
    return elem + SCALE_BITS / block


def element_bits(fmt: str, width) -> int:
    """Element width without the block scale -- the bits HBM streams per operand."""
    return int(width) if fmt == "mxint" else 1 + int(width[0]) + int(width[1])


def width_label(fmt: str, width) -> str:
    return f"MXINT{int(width)}" if fmt == "mxint" else f"MXFP_E{width[0]}M{width[1]}"


def precision_from_components(attn_bits, ffn_bits, kv_bits,
                             attn_label="attnW", ffn_label="ffnW", kv_label="KV",
                             attn_elem=None, ffn_elem=None, kv_elem=None,
                             m_bits=0, density_exp=DENSITY_EXP,
                             block_size=8, *, key_bits=None, value_bits=None,
                             key_label=None, value_label=None,
                             key_elem=None, value_elem=None) -> dict:
    """One decode precision point (activations are fixed bf16 in storage).

    `*_elem` are the element widths N that HBM streams (they set the bandwidth cap
    and memory time); they default to the rounded effective bits. `m_bits` is the
    MAC compute width M: HBM streams N-bit operands, the array multiplies at M,
    writeback requantises to N. It drives the iso-area density; 0 means the widest
    streamed operand max(attnW, ffnW, KV). Callers that know the activation compute
    width should fold it in and pass m_bits explicitly (the DSE bridge does). With
    the default density_exp = 0 the density is 1 regardless of M."""
    attn_e = int(round(float(attn_bits))) if attn_elem is None else int(attn_elem)
    ffn_e = int(round(float(ffn_bits))) if ffn_elem is None else int(ffn_elem)
    key_b = float(kv_bits) if key_bits is None else float(key_bits)
    value_b = float(kv_bits) if value_bits is None else float(value_bits)
    key_e = (
        int(round(key_b))
        if key_elem is None and kv_elem is None
        else int(kv_elem if key_elem is None else key_elem)
    )
    value_e = (
        int(round(value_b))
        if value_elem is None and kv_elem is None
        else int(kv_elem if value_elem is None else value_elem)
    )
    kv_e = max(key_e, value_e)
    aggregate_kv_bits = (key_b + value_b) / 2.0
    return {
        "attn_bits": float(attn_bits), "ffn_bits": float(ffn_bits),
        "kv_bits": aggregate_kv_bits, "key_bits": key_b, "value_bits": value_b,
        "a_bits": float(ACT_BITS),
        "attn_label": attn_label, "ffn_label": ffn_label, "kv_label": kv_label,
        "key_label": key_label or kv_label, "value_label": value_label or kv_label,
        "attn_elem": attn_e, "ffn_elem": ffn_e, "kv_elem": kv_e,
        "key_elem": key_e, "value_elem": value_e,
        "m_bits": int(m_bits) if m_bits else max(attn_e, ffn_e, key_e, value_e),
        # P = accumulator width (INT accumulate in the MX array). P never reaches
        # HBM: every writeback requantises P -> N first (new KV written at kv_bits,
        # activations stay on-chip), so all HBM byte counts here depend only on N.
        "p_bits": 32,
        "density_exp": float(density_exp),
        "block_size": int(block_size),
    }


def compute_density(prec) -> float:
    """Iso-area multiplier gain of M-bit MACs over the 4-bit reference: (4/M)^k.
    Scales peak compute and the cycle model; the N widths scale memory time."""
    return (REF_MAC_BITS / prec["m_bits"]) ** prec["density_exp"]


def _parse_width(fmt: str, tok):
    """Parse one width token: an int for MXINT, an (exp, frac) tuple for MXFP."""
    return int(tok) if fmt == "mxint" else MXFP_FORMATS[tok]


def build_precision(args) -> dict:
    """Build a precision spec from CLI args. attn and ffn share the weight format
    (--w-fmt) but keep independent widths; KV has its own format (--kv-fmt), so
    MXINT weights with MXFP KV is expressible."""
    wf, kf = args.w_fmt, args.kv_fmt
    aw, fw, kv = _parse_width(wf, args.attn_w), _parse_width(wf, args.ffn_w), _parse_width(kf, args.kv)
    return precision_from_components(
        effective_bits(wf, aw, args.block),
        effective_bits(wf, fw, args.block),
        effective_bits(kf, kv, args.block),
        width_label(wf, aw), width_label(wf, fw), width_label(kf, kv),
        attn_elem=element_bits(wf, aw), ffn_elem=element_bits(wf, fw),
        kv_elem=element_bits(kf, kv),
        m_bits=args.m_bits, density_exp=args.density_exp,
        block_size=args.block)


def stream_bits(prec) -> int:
    """Widest operand HBM streams in decode: max(attnW, ffnW, KV) bits."""
    return max(
        prec["attn_elem"],
        prec["ffn_elem"],
        prec.get("key_elem", prec["kv_elem"]),
        prec.get("value_elem", prec["kv_elem"]),
    )


def mlen_bandwidth_cap(hw_cfg, prec) -> int:
    """Largest MLEN HBM can feed per cycle: HBM_WIDTH / stream_bits. Wider operands => smaller MLEN."""
    return hw_cfg.HBM_WIDTH // stream_bits(prec)


def load_model_dims(path: str) -> dict:
    """Architecture sizes from the model JSON. `num_experts`/`experts_per_token`
    default to (1, 1) for a dense model (>1 experts uses the MoE FFN path).
    `sliding_window` + `layer_types` give the attention split; no sliding_window
    means all full attention (n_sliding = 0)."""
    with open(path) as f:
        p = json.load(f)
    ah = p["num_attention_heads"]
    layers = p["num_hidden_layers"]
    window = p.get("sliding_window") or 0                      # null / 0 -> no windowing
    if p.get("use_sliding_window") is False:                   # window size set but disabled (some Qwen)
        window = 0
    n_sliding = (sum(1 for lt in p.get("layer_types", []) if lt == "sliding_attention")
                 if window > 0 else 0)
    model_type = str(p.get("model_type", ""))
    num_experts = int(p.get("num_experts", p.get("num_local_experts", 1)))
    experts_per_token = int(
        p.get("num_experts_per_tok", p.get("experts_per_token", 1))
    )
    if num_experts <= 0 or not 1 <= experts_per_token <= num_experts:
        raise ValueError("invalid MoE expert count or top-k routing width")
    for field in ("num_shared_experts", "shared_expert_intermediate_size"):
        value = p.get(field)
        if value not in (None, 0):
            raise ValueError(
                f"{field} enables shared experts, which the decode model does not price"
            )
    dense_inter = int(p["intermediate_size"])
    inter = int(p.get("moe_intermediate_size", dense_inter)) if num_experts > 1 else dense_inter
    routing_imbalance = float(p.get("moe_routing_imbalance_factor", 1.0))
    if not math.isfinite(routing_imbalance) or routing_imbalance < 1.0:
        raise ValueError("moe_routing_imbalance_factor must be finite and >= 1")
    unique_experts = p.get("moe_unique_experts_per_step")
    if unique_experts is not None:
        unique_experts = int(unique_experts)
        if not experts_per_token <= unique_experts <= num_experts:
            raise ValueError(
                "moe_unique_experts_per_step must be between top-k and num_experts"
            )
    route_repricing = validate_model_overlay(p)
    return {"hidden": p["hidden_size"], "heads": ah, "kv_heads": p["num_key_value_heads"],
            "head_dim": p.get("head_dim", p["hidden_size"] // ah), "layers": layers,
            "inter": inter, "dense_inter": dense_inter, "vocab": p["vocab_size"],
            "tie_embeddings": p.get("tie_word_embeddings", False),
            "model_type": model_type,
            "qk_norm": bool(
                p.get(
                    "qk_norm",
                    p.get("use_qk_norm", model_type.startswith("qwen3")),
                )
            ),
            "num_experts": num_experts,
            "experts_per_token": experts_per_token,
            "router_weight_bits": int(p.get("router_weight_bits", 16)),
            "norm_topk_prob": bool(p.get("norm_topk_prob", False)),
            "moe_routing_imbalance_factor": routing_imbalance,
            "moe_unique_experts_per_step": unique_experts,
            "moe_route_repricing": (
                dict(route_repricing) if route_repricing is not None else None
            ),
            "sliding_window": window, "n_sliding": n_sliding, "n_full": layers - n_sliding}


def is_moe(d: dict) -> bool:
    return d.get("num_experts", 1) > 1


def _moe_route_accounting(dims: Mapping[str, object], batch: int) -> dict:
    """Return conserved route-count and scheduling-penalty provenance."""

    assignments = int(batch) * int(dims["experts_per_token"])
    return {
        "routes_per_step": assignments,
        "physical_route_assignments_per_step": assignments,
        "route_assignment_accounting": "conserved_batch_times_topk",
        "routing_imbalance_factor": float(
            dims.get("moe_routing_imbalance_factor", 1.0)
        ),
        "routing_imbalance_application": (
            "expert_stage_ragged_schedule_cycle_penalty_only"
        ),
    }


def _moe_workload_accounting(
    perf: PerfModel,
    dims: Mapping[str, object],
    batch: int,
) -> dict:
    """Build one-source routed-expert timing and provenance for report rows."""

    route_unique_override = dims.get("moe_unique_experts_per_step")
    route_repricing = dims.get("moe_route_repricing")
    active_expert_source = (
        (
            "verified_trace_summary_conservative_observed_max"
            if route_repricing is not None
            else "explicit_unique_expert_override"
        )
        if route_unique_override is not None
        else None
    )
    expert_timing = perf.moe_decode_expert_timing(
        int(dims["hidden"]),
        int(batch),
        int(dims["num_experts"]),
        int(dims["experts_per_token"]),
        int(dims["inter"]),
        active_experts=(
            int(route_unique_override)
            if route_unique_override is not None
            else None
        ),
        active_expert_source=active_expert_source,
        routing_imbalance_factor=float(
            dims.get("moe_routing_imbalance_factor", 1.0)
        ),
    )
    route_accounting = _moe_route_accounting(dims, batch)
    if int(expert_timing["route_assignments"]) != int(
        route_accounting["physical_route_assignments_per_step"]
    ):
        raise AssertionError("MoE route and timing ledgers disagree")

    expected_unique = expected_unique_experts(
        int(dims["num_experts"]),
        int(dims["experts_per_token"]),
        int(batch),
    )
    return {
        "schema": "plena-routed-moe-decode-workload/v1",
        "num_experts": int(dims["num_experts"]),
        "experts_per_token": int(dims["experts_per_token"]),
        "expert_intermediate_size": int(dims["inter"]),
        "tokens_per_step": int(batch),
        **route_accounting,
        "expected_unique_experts_uniform": expected_unique,
        "charged_unique_experts_per_layer": int(
            expert_timing["active_experts"]
        ),
        "unique_expert_source": str(
            expert_timing["active_expert_source"]
        ),
        "expert_row_tiles_per_layer": int(
            expert_timing["expert_row_tiles"]
        ),
        "expert_padded_rows_per_layer": int(
            expert_timing["expert_padded_rows"]
        ),
        "expert_padding_rows_per_layer": int(
            expert_timing["expert_padding_rows"]
        ),
        "expert_batch_ledger_source": str(
            expert_timing["assignment_distribution"]
        ),
        "expert_batch_ledger": expert_timing,
        "router_weight_precision_bits": int(
            dims.get("router_weight_bits", 16)
        ),
        "norm_topk_prob": bool(dims.get("norm_topk_prob", False)),
        "route_repricing": (
            dict(route_repricing) if isinstance(route_repricing, Mapping) else None
        ),
        "timing_events": [
            "router_gemm",
            "topk_selection",
            "route_probability_softmax",
            "expert_gate_up",
            "expert_silu_multiply",
            "expert_down",
            "weighted_expert_combine",
        ],
        "provenance": {
            "expert_matrix_cycles_included": True,
            "expert_matrix_drain_cycles_included": True,
            "expert_auxiliary_issue_cycles_included": True,
            "expert_blen_padding_included": True,
            "router_and_vector_cycles_included": True,
            "expert_hbm_traffic_included": True,
            "router_bf16_hbm_traffic_included": True,
            "moe_specific_area_calibrated": False,
            "moe_specific_dynamic_power_calibrated": False,
            "publication_rankable": False,
            "route_repricing_publication_rankable": False,
            "unrankable_reason": (
                "routed-MoE control area and dynamic power lack matched "
                "emulator/RTL calibration"
            ),
        },
    }


def _attn_split(d: dict) -> tuple[int, int, int]:
    """Attention layers by KV span: (full count, sliding-window count, window).
    An all-full model is (layers, 0, 0), collapsing every windowed branch below."""
    return d["n_full"], d["n_sliding"], d["sliding_window"]


def peak_hbm_bw_bytes(hw_cfg) -> float:
    """Peak HBM bandwidth = (HBM_WIDTH / 8) * clock  [bytes/s]."""
    return (hw_cfg.HBM_WIDTH / 8.0) * FREQ_HZ


def matrix_overfetch_factor(hw_cfg) -> float:
    """Wasted-read factor on matrix loads. A prefetch reads M_LOAD elements rounded
    up to a multiple of MLEN, so M_LOAD > MLEN reads M_LOAD/MLEN x the needed
    bytes. Best case M_LOAD = MLEN gives factor 1."""
    mlen = hw_cfg.MLEN
    m_load = getattr(hw_cfg, "HBM_M_Prefetch_Amount", mlen)
    return math.ceil(max(m_load, mlen) / mlen)


# Area model: "proxy" (mm^2 per multiplier) or "calibrated" (precision-aware
# MatrixMachine census, DC-fitted). Set once from the CLI so area_mm2() serves
# every caller without passing a flag through each call site.
_AREA_MODEL = "proxy"
_AREA_PREC: dict | None = None


def set_area_model(model: str, prec: dict | None = None) -> None:
    global _AREA_MODEL, _AREA_PREC
    _AREA_MODEL, _AREA_PREC = model, prec


def area_multipliers(hw_cfg) -> int:
    """Matrix-array multiplier count = MLEN * BLEN (sets silicon area)."""
    return hw_cfg.MLEN * hw_cfg.BLEN


def area_mm2(hw_cfg) -> float:
    if _AREA_MODEL == "calibrated" and _AREA_PREC is not None:
        sys.path.insert(0, str(_HERE.parent.parent))
        from analytic_models.disagg_serve.area import area_mm2 as _calibrated
        return _calibrated("calibrated", hw_cfg, _AREA_PREC)
    return area_multipliers(hw_cfg) * MM2_PER_MULTIPLIER


def system_area(
    hw_cfg,
    prec: Mapping,
    *,
    chip_count: int,
    link_ports: int,
    link_generation: str = DEFAULT_LINK_GENERATION,
    kv_head_reuse: bool = False,
    drain_overlapped: bool = False,
    kv_heads: int = 1,
    hbm_interface_units: int = 0,
) -> dict:
    """Return fail-closed full-chip plus HBM-PHY and C2C-PHY aggregate area."""

    if link_generation not in LINK_GENS:
        raise ValueError("unsupported link generation")
    sys.path.insert(0, str(_HERE.parent.parent))
    from analytic_models.disagg_serve.area import system_area_mm2

    result = system_area_mm2(
        hw_cfg,
        prec,
        chip_count=chip_count,
        ports_per_chip=link_ports,
        link_bandwidth_gbps=LINK_GENS[link_generation] * 2.0 * 8.0 / 1e9,
        hbm_interface_units_per_chip=hbm_interface_units,
    )
    options = architecture_option_area_mm2(
        mlen=int(hw_cfg.MLEN),
        hlen=int(hw_cfg.HLEN),
        kv_heads=kv_heads,
        kv_head_reuse=kv_head_reuse,
        drain_overlapped=drain_overlapped,
    )
    per_chip_mm2 = float(options["area_mm2_per_chip"])
    if per_chip_mm2:
        system_addition_mm2 = per_chip_mm2 * chip_count
        result = dict(result)
        result["base_area_mm2"] = float(result["area_mm2"])
        result["area_mm2"] = float(result["area_mm2"]) + system_addition_mm2
        result["chip_area_mm2"] = (
            float(result["chip_area_mm2"]) + per_chip_mm2
        )
        for name in ("area", "system_area"):
            if name in result:
                result[name] = float(result[name]) + system_addition_mm2 * 1e6
        if "chip_area" in result:
            result["chip_area"] = float(result["chip_area"]) + per_chip_mm2 * 1e6
        raw_breakdown = dict(result.get("breakdown", {}))
        mm_breakdown = dict(result.get("breakdown_mm2", {}))
        for name, area in dict(options["breakdown_mm2_per_chip"]).items():
            raw_breakdown[name] = float(area) * chip_count * 1e6
            mm_breakdown[name] = float(area) * chip_count
        result["breakdown"] = raw_breakdown
        result["breakdown_mm2"] = mm_breakdown

        chip = dict(result.get("chip", {}))
        if chip:
            for name in ("area", "chip_area"):
                if name in chip:
                    chip[name] = float(chip[name]) + per_chip_mm2 * 1e6
            chip_breakdown = dict(chip.get("breakdown", {}))
            chip_evidence = dict(chip.get("block_evidence", {}))
            for name, area in dict(options["breakdown_mm2_per_chip"]).items():
                chip_breakdown[name] = float(area) * 1e6
                chip_evidence[name] = dict(options["evidence"])[name]
            chip["breakdown"] = chip_breakdown
            chip["block_evidence"] = chip_evidence
            reuse_area = float(
                dict(options["breakdown_mm2_per_chip"]).get(
                    "KVHeadReuseControl",
                    0.0,
                )
            )
            bank_area = float(
                dict(options["breakdown_mm2_per_chip"]).get(
                    "DrainOverlapAccumulatorBank",
                    0.0,
                )
            )
            if "logic_area" in chip:
                chip["logic_area"] = float(chip["logic_area"]) + reuse_area * 1e6
            if "sram_macro_area" in chip:
                chip["sram_macro_area"] = (
                    float(chip["sram_macro_area"]) + bank_area * 1e6
                )
            result["chip"] = chip
        # Reuse control is an explicit structural proxy and therefore lowers
        # the combined evidence tier even when the base chip is inside its fit.
        if kv_head_reuse:
            result["evidence_tier"] = "declared_structural_estimate"
    result["architecture_options"] = options
    return result


def onchip_activation_bytes(d: dict, batch: int) -> int:
    """Decode activation working set held on-chip in FP16 (Vector SRAM); never written to HBM."""
    return math.ceil(batch * (d["hidden"] + d["inter"]) * ACT_BITS / 8)


# Per-component HBM footprint + decode traffic.
# `MemoryModel` carries a single `weight_bits`, but we need three (attn / FFN / vocab),
# so we reset it per component and sum. This is the only place per-component
# precision enters the byte model. The FFN term picks dense vs MoE by expert count.
def weight_footprint_bytes(mem: MemoryModel, d: dict, prec: dict) -> dict:
    """HBM weight storage: attention @ attn_bits, FFN/experts @ ffn_bits,
    embedding/norms/router @ BF16 and the canonical local head at profile W.
    An optional external head is excluded. MoE keeps every expert resident,
    which sets the capacity wall for these models.
    """
    h, ah, kvh, hd = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"]
    inter, vocab, layers = d["inter"], d["vocab"], d["layers"]

    mem.weight_bits = EMBED_BITS
    embedding = mem.embedding_weights(vocab, h)
    norms = mem.layer_norm_weights(h) * 2 * layers
    if bool(d.get("qk_norm", False)):
        norms += mem.layer_norm_weights(hd) * 2 * layers
    mem.weight_bits = (
        prec["head_bits"]
        if bool(prec.get("lm_head_quantized", False))
        else LM_HEAD_BITS
    )
    # An MX serving head is a separately packed matrix even when a checkpoint
    # ties its BF16 embedding table.
    lm_head = mem.lm_head_weights(
        h,
        vocab,
        d.get("tie_embeddings", False)
        and not bool(prec.get("lm_head_quantized", False)),
    )

    mem.weight_bits = prec["attn_bits"]
    attention = mem.attention_weights(h, ah, kvh, hd) * layers
    mem.weight_bits = prec["ffn_bits"]
    if is_moe(d):
        router = mem._bits_to_bytes(
            h * d["num_experts"],
            d.get("router_weight_bits", 16),
        )
        experts = mem.moe_expert_weights(h, inter, d["num_experts"])
        ffn = (router + experts) * layers
    else:
        ffn = mem.ffn_weights(h, inter) * layers

    total = embedding + norms + lm_head + attention + ffn
    return {"embedding": embedding, "norms": norms, "lm_head": lm_head,
            "attention": attention, "ffn": ffn, "total": total}


def _packed_kv_traffic(d: dict, prec: dict, mlen: int):
    return traffic_from_precision(
        kv_heads=d["kv_heads"],
        head_dim=d["head_dim"],
        mlen=mlen,
        element_bits=prec["kv_elem"],
        effective_bits=prec["kv_bits"],
    )


def kv_footprint_bytes(
    mem: MemoryModel,
    d: dict,
    prec: dict,
    ctx: int,
    batch: int,
    *,
    mlen: int | None = None,
    kv_layout: str = DENSE_SELECTOR,
    tp: int = 1,
) -> int:
    """HBM KV-cache footprint at kv_bits. Full layers store the whole `ctx`;
    sliding-window layers store only the last `window` tokens -- windowing's
    capacity win."""
    mem.kv_cache_bits = prec["kv_bits"]
    kvh, hd = d["kv_heads"], d["head_dim"]
    n_full, n_slide, window = _attn_split(d)
    total = mem.kv_cache_footprint(kvh, hd, n_full, ctx, batch).total_bytes
    if n_slide:
        total += mem.kv_cache_footprint(kvh, hd, n_slide, min(ctx, window), batch).total_bytes
    if mlen is None:
        return total
    return kv_ledger(
        d,
        prec,
        context=ctx,
        batch=batch,
        mlen=mlen,
        kv_layout=kv_layout,
        tp=tp,
    ).total_bytes


def decode_traffic(
    mem: MemoryModel,
    d: dict,
    kv_size: int,
    batch: int,
    prec: dict,
    *,
    mlen: int | None = None,
    kv_layout: str = DENSE_SELECTOR,
    physical_weights=None,
    physical_step=None,
    include_lm_head: bool = True,
    tp: int = 1,
) -> MemoryTraffic:
    """Per-token HBM traffic: read every weight once (attn @ attn_bits,
    FFN/experts @ ffn_bits, local head @ profile W), read the whole KV cache @
    kv_bits, and write the new token's KV. Embedding/norm/router remain BF16;
    MoE reads the router plus active experts rather than every expert."""
    h, ah, kvh, hd = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"]
    inter, vocab, layers = d["inter"], d["vocab"], d["layers"]

    mem.weight_bits = prec["attn_bits"]
    mem.kv_cache_bits = prec["kv_bits"]
    n_full, n_slide, window = _attn_split(d)
    proj = mem.projection_traffic(h, ah, kvh, hd, 1, batch, "decode")     # QKV weights + one-token KV write
    out_proj = mem.output_projection_traffic(h, ah, hd, 1, batch, "decode")
    # Full layers read the whole KV; windowed layers read only the last `window` keys.
    attn = (proj + out_proj) * layers + mem.flash_attention_traffic(ah, kvh, hd, 1, kv_size, batch, "decode") * n_full
    if n_slide:
        attn += mem.flash_attention_traffic(ah, kvh, hd, 1, min(kv_size, window), batch, "decode") * n_slide

    mem.weight_bits = prec["ffn_bits"]
    if is_moe(d):
        ffn = mem.moe_traffic(
            h,
            inter,
            d["num_experts"],
            d["experts_per_token"],
            1,
            batch,
            "decode",
            unique_experts=d.get("moe_unique_experts_per_step"),
            router_weight_bits=d.get("router_weight_bits", 16),
        ) * layers
    else:
        ffn = mem.ffn_traffic(h, inter, 1, batch, "decode") * layers

    mem.weight_bits = (
        prec["head_bits"]
        if bool(prec.get("lm_head_quantized", False))
        else LM_HEAD_BITS
    )
    head = mem.lm_head_traffic(h, vocab) if include_lm_head else MemoryTraffic()
    traffic = attn + ffn + head
    if mlen is None:
        return traffic

    read_per_full_layer = 2 * mem._bits_to_bytes(
        kv_size * batch * kvh * hd,
        prec["kv_bits"],
    )
    # Subtracted below to remove the estimate the physical ledger replaces, so
    # this stays at one sweep: the ledger is what charges for the rest.
    logical_kv_read = read_per_full_layer * n_full
    if n_slide:
        logical_kv_read += (
            2
            * mem._bits_to_bytes(
                min(kv_size, window) * batch * kvh * hd,
                prec["kv_bits"],
            )
            * n_slide
        )
    logical_kv_write = (
        2
        * mem._bits_to_bytes(batch * kvh * hd, prec["kv_bits"])
        * layers
    )
    weights = physical_weights or weight_ledger(
        d,
        prec,
        include_lm_head=include_lm_head,
        batch=batch,
        unique_experts=d.get("moe_unique_experts_per_step"),
        mlen=mlen,
    )
    step_traffic = physical_step or decode_step_traffic_ledger(
        d,
        prec,
        context=kv_size,
        batch=batch,
        mlen=mlen,
        kv_layout=kv_layout,
        weights=weights,
        include_lm_head=include_lm_head,
        tp=tp,
    )
    mem.weight_bits = prec["attn_bits"]
    logical_weight_read = (
        mem.attention_weights(h, ah, kvh, hd) * layers
    )
    mem.weight_bits = prec["ffn_bits"]
    if is_moe(d):
        router = mem._bits_to_bytes(
            h * d["num_experts"],
            d.get("router_weight_bits", 16),
        )
        unique_experts = d.get("moe_unique_experts_per_step")
        if unique_experts is None:
            unique_experts = conservative_unique_experts(
                d["num_experts"],
                d["experts_per_token"],
                batch,
            )
        active = (
            mem._bits_to_bytes(3 * h * inter, prec["ffn_bits"])
            * unique_experts
        )
        logical_weight_read += (router + active) * layers
    else:
        logical_weight_read += mem.ffn_weights(h, inter) * layers
    if include_lm_head:
        mem.weight_bits = (
            prec["head_bits"]
            if bool(prec.get("lm_head_quantized", False))
            else LM_HEAD_BITS
        )
        logical_weight_read += mem.lm_head_weights(
            h,
            vocab,
            tie_embeddings=False,
        )
    return MemoryTraffic(
        read_bytes=traffic.read_bytes
        - logical_weight_read
        - logical_kv_read
        + step_traffic.read_bytes,
        write_bytes=traffic.write_bytes
        - logical_kv_write
        + step_traffic.write_bytes,
    )


# Compute cycles + FLOPs per decode token
def _ffn_label_cycles(perf: PerfModel, d: dict, batch: int) -> tuple[str, int]:
    """FFN cycles including routed-MoE router, top-k, experts, and combine."""
    if is_moe(d):
        cycles = perf.mlp_moe(
            d["hidden"],
            1,
            batch,
            d["num_experts"],
            d["experts_per_token"],
            d["inter"],
            "decode",
            include_input_norm=False,
            active_experts=d.get("moe_unique_experts_per_step"),
            active_expert_source=(
                (
                    "verified_trace_summary_conservative_observed_max"
                    if d.get("moe_route_repricing") is not None
                    else "explicit_unique_expert_override"
                )
                if d.get("moe_unique_experts_per_step") is not None
                else None
            ),
            routing_imbalance_factor=d.get(
                "moe_routing_imbalance_factor",
                1.0,
            ),
        )
        return (
            f"MoE router+top-k+{d['experts_per_token']}/{d['num_experts']} experts+combine",
            cycles,
        )
    return (
        "FFN (gate/up/down)",
        perf.feed_forward(d["hidden"], d["inter"], 1, batch, "decode"),
    )


def _flash_cycles(
    perf: PerfModel,
    d: dict,
    kv: int,
    batch: int,
    *,
    kv_layout: str,
    packed_q1_timing_contract: PackedQ1TimingContract | None,
    batch_packed: bool = False,
    kv_head_reuse: bool | None = None,
) -> int:
    """Per-token flash-attention cycles over the stack: full layers attend to the
    whole KV, sliding-window layers to only the last `window` keys."""
    ah, kvh, hd = d["heads"], d["kv_heads"], d["head_dim"]
    n_full, n_slide, window = _attn_split(d)
    packed_q1 = (
        kv_layout == DENSE_SELECTOR and kv_head_reuse is not False
    )
    total = perf.flash_attention(
        ah,
        kvh,
        hd,
        1,
        kv,
        batch,
        "decode",
        packed_q1=packed_q1,
        packed_q1_contract=packed_q1_timing_contract,
        batch_packed=batch_packed,
    ) * n_full
    if n_slide:
        total += perf.flash_attention(
            ah,
            kvh,
            hd,
            1,
            min(kv, window),
            batch,
            "decode",
            packed_q1=packed_q1,
            packed_q1_contract=packed_q1_timing_contract,
            batch_packed=batch_packed,
        ) * n_slide
    return total


def decode_token_components(
    perf: PerfModel,
    d: dict,
    kv: int,
    batch: int,
    *,
    include_lm_head: bool = True,
    kv_layout: str = DENSE_SELECTOR,
    packed_q1_timing_contract: PackedQ1TimingContract | None = None,
    batch_packed_attention: bool = False,
    kv_head_reuse: bool | None = None,
) -> dict:
    """Cycles to generate one token: per-layer ops x layers + once-per-token head
    ops. Windowed layers are charged only for their `window` keys."""
    h, ah, kvh, hd, layers = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"], d["layers"]
    ffn_label, ffn_cyc = _ffn_label_cycles(perf, d, batch)
    comp = {
        f"RMSNorm (x2) x{layers} layers":            perf.rms_layer(h, 1, batch, "decode") * 2 * layers,
        f"Q/K/V proj + RoPE x{layers} layers":       perf.projection(
            h, ah, kvh, hd, 1, batch, "decode",
            kv_projection_width=d.get("kv_projection_width"),
        ) * layers,
        f"Flash attention x{layers} layers":         _flash_cycles(
            perf,
            d,
            kv,
            batch,
            kv_layout=kv_layout,
            packed_q1_timing_contract=packed_q1_timing_contract,
            batch_packed=batch_packed_attention,
            kv_head_reuse=kv_head_reuse,
        ),
        f"Output projection (W_O) x{layers} layers": perf.output_projection(h, ah, hd, 1, batch, "decode") * layers,
        f"Residual adds (x2) x{layers} layers":      perf.residual(h, 1, batch, "decode") * 2 * layers,
        f"{ffn_label} x{layers} layers":             ffn_cyc * layers,
    }
    if bool(d.get("qk_norm", False)):
        comp[f"Q/K RMSNorm x{layers} layers"] = (
            perf.rms_layer(hd, 1, batch * ah, "decode")
            + perf.rms_layer(hd, 1, batch * kvh, "decode")
        ) * layers
    comp["Embedding lookup"] = perf.embeddings(h, 1, batch, "decode")
    comp["Final RMSNorm"] = perf.rms_layer(h, 1, batch, "decode")
    if include_lm_head:
        comp["LM head"] = perf.lm_head(h, d["vocab"], batch)
        head_padding = perf.lm_head_padding_preparation(
            h,
            d["vocab"],
            batch,
        )
        # The activation tail and BLEN-inactive rows are initialized locally on
        # every rank.  Padded vocabulary rows are masked across the sharded
        # output space before the bounded selector consumes them.
        comp["LM head activation padding zero-fill"] = int(
            head_padding["zero_fill_cycles_per_rank"]
        )
        comp["LM head padded-vocab mask"] = int(
            head_padding["padded_vocab_mask_cycles_aggregate"]
        )
        comp["Vocab streamed top-k/top-p selection"] = (
            perf.lm_head_streaming_selection(
            d["vocab"],
            batch,
            top_k=20,
            )
        )
    return comp


def decode_token_cycles(
    perf: PerfModel,
    d: dict,
    kv: int,
    batch: int,
    *,
    include_lm_head: bool = True,
    kv_layout: str = DENSE_SELECTOR,
    packed_q1_timing_contract: PackedQ1TimingContract | None = None,
    batch_packed_attention: bool = False,
    kv_head_reuse: bool | None = None,
) -> int:
    return sum(
        decode_token_components(
            perf,
            d,
            kv,
            batch,
            include_lm_head=include_lm_head,
            kv_layout=kv_layout,
            packed_q1_timing_contract=packed_q1_timing_contract,
            batch_packed_attention=batch_packed_attention,
            kv_head_reuse=kv_head_reuse,
        ).values()
    )


def decode_step_flops(
    d: dict,
    kv: int,
    batch: int,
    *,
    include_lm_head: bool = True,
) -> int:
    """FLOPs for one decode step over the batch (2 per MAC), for arithmetic
    intensity. MoE counts only top-k experts; sliding layers only `window` keys."""
    h, ah, kvh, hd = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"]
    qkvo = h * ah * hd + 2 * (h * kvh * hd) + (ah * hd) * h    # Q, K, V, O projections
    ffn = 2 * h * d["inter"] + d["inter"] * h                  # gate, up, down (one expert)
    if is_moe(d):
        ffn *= d["experts_per_token"]
    n_full, n_slide, window = _attn_split(d)
    attn = 2 * ah * hd * kv * n_full                           # QK^T + attention @ V (full layers)
    if n_slide:
        attn += 2 * ah * hd * min(kv, window) * n_slide        # windowed layers
    body = (qkvo + ffn) * d["layers"] + attn
    head = h * d["vocab"] if include_lm_head else 0
    return 2 * batch * (body + head)


def _parallel_topology(hw_over, n_chips: int) -> dict[str, int | str]:
    """Resolve an explicit TP x KVP topology from hardware overrides."""

    values = dict(hw_over or {})
    has_explicit_topology = "TP" in values or "KVP" in values
    if has_explicit_topology:
        tp = int(values.get("TP", 1))
        kvp = int(values.get("KVP", 1))
        chips = tp * kvp
        if n_chips not in (0, chips):
            raise ValueError("n_chips differs from TP * KVP")
    else:
        chips = int(n_chips)
        tp = chips if chips > 0 else 0
        kvp = 1
    ports = int(
        values.get(
            "LINK_PORTS",
            0 if chips in (0, 1) else int(tp > 1) + int(kvp > 1),
        )
    )
    policy = str(values.get("SRAM_POLICY", "streaming"))
    link_generation = str(
        values.get("LINK_GENERATION", DEFAULT_LINK_GENERATION)
    )
    expert_parallel_mode = str(
        values.get("EXPERT_PARALLEL_MODE", EXPERT_TENSOR_PARALLEL)
    )
    e2_explicit = (
        "KV_HEAD_REUSE" in values or "DRAIN_OVERLAPPED" in values
    )
    kv_head_reuse = values.get("KV_HEAD_REUSE", False)
    drain_overlapped = values.get("DRAIN_OVERLAPPED", False)
    if not isinstance(kv_head_reuse, bool) or not isinstance(
        drain_overlapped,
        bool,
    ):
        raise TypeError("architecture options must be boolean")
    if tp < 0 or kvp <= 0 or chips < 0:
        raise ValueError("parallel dimensions must be non-negative")
    if chips > 0 and tp * kvp != chips:
        raise ValueError("chip count must equal TP * KVP")
    if policy not in SRAM_POLICIES:
        raise ValueError(f"unsupported SRAM policy {policy!r}")
    if link_generation not in LINK_GENS:
        raise ValueError(
            f"unsupported link generation {link_generation!r}"
        )
    if expert_parallel_mode not in EXPERT_PARALLEL_MODES:
        raise ValueError(
            f"unsupported expert parallel mode {expert_parallel_mode!r}"
        )
    required_ports = int(tp > 1) + int(kvp > 1)
    if chips == 1 and ports != 0:
        raise ValueError("single-chip decode cannot use link ports")
    if chips > 1 and ports < required_ports:
        raise ValueError(
            "link ports cannot serve every active parallel dimension"
        )
    return {
        "tp": tp,
        "kvp": kvp,
        "chip_count": chips,
        "explicit_topology": has_explicit_topology,
        "legacy_ideal_parallelism": not has_explicit_topology,
        "link_ports": ports,
        "sram_policy": policy,
        "link_generation": link_generation,
        "expert_parallel_mode": expert_parallel_mode,
        "architecture_knobs_explicit": e2_explicit,
        "kv_head_reuse": kv_head_reuse,
        "drain_overlapped": drain_overlapped,
    }


def _validate_parallel_model(d: dict, topology: Mapping[str, int | str]) -> None:
    tp = int(topology["tp"])
    if tp == 0:
        return
    if int(d["hidden"]) % tp:
        raise ValueError("TP must divide hidden size")
    if int(d["heads"]) % tp:
        raise ValueError("TP must divide query-head count")
    if int(d["kv_heads"]) % tp:
        raise ValueError("TP must divide KV-head count")


def _kv_resident_fraction(policy: str) -> float:
    if not policy.startswith("kv_resident_"):
        return 0.0
    return int(policy.rsplit("_", 1)[1]) / 100.0


def _balanced_partition_widths(value: int, ranks: int) -> tuple[int, ...]:
    """Return deterministic rank widths without hiding an uneven tail."""

    if value <= 0 or ranks <= 0 or ranks > value:
        raise ValueError("partition dimensions must be positive")
    quotient, remainder = divmod(value, ranks)
    return tuple(quotient + int(rank < remainder) for rank in range(ranks))


def _partitioned_component_cycles(
    perf: PerfModel,
    d: dict,
    kv: int,
    batch: int,
    *,
    tp: int,
    kvp: int,
    include_lm_head: bool,
    kv_layout: str,
    packed_q1_timing_contract,
    batch_packed_attention: bool,
    kv_head_reuse: bool | None,
    body_layout: BodyWeightPhysicalLayout | None = None,
    expert_parallel_mode: str = EXPERT_TENSOR_PARALLEL,
) -> dict[str, float]:
    """Operation-shaped slowest-rank component cycles for explicit TP x KVP.

    Replicated vector/control operations are not divided by TP. Q/K/V, O, the
    expert intermediate dimension, the local vocabulary shard, and the
    KVP-local cache span are re-evaluated at their actual rank-local shapes.
    """

    if tp <= 0 or kvp <= 0:
        raise ValueError("parallel degrees must be positive")
    if expert_parallel_mode not in EXPERT_PARALLEL_MODES:
        raise ValueError("unsupported expert parallel mode")
    if int(d["heads"]) % tp or int(d["kv_heads"]) % tp:
        raise ValueError("TP must own complete query and KV heads")

    hidden = int(d["hidden"])
    layers = int(d["layers"])
    local_heads = int(d["heads"]) // tp
    local_kv_heads = int(d["kv_heads"]) // tp
    local_inter_widths = _balanced_partition_widths(int(d["inter"]), tp)
    local_vocab_widths = _balanced_partition_widths(int(d["vocab"]), tp)
    if body_layout is not None:
        provenance = body_layout.provenance
        if (
            int(provenance["tp"]) != tp
            or int(provenance["kvp"]) != kvp
            or int(provenance["mlen"]) != int(perf.mlen)
            or str(provenance["expert_parallel_mode"])
            != expert_parallel_mode
        ):
            raise ValueError("body layout differs from the timing topology")
        rank_shapes = provenance.get("rank_shapes")
        if not isinstance(rank_shapes, Mapping):
            raise ValueError("body layout omitted rank-local timing shapes")
        expert_shapes = rank_shapes.get("experts")
        if not isinstance(expert_shapes, Sequence) or len(expert_shapes) != tp:
            raise ValueError("body layout omitted expert rank shapes")
        local_inter_widths = tuple(
            int(shape["gate_up"]["logical_rows"])
            for shape in expert_shapes
        )
        if include_lm_head:
            head_shapes = rank_shapes.get("local_head")
            if not isinstance(head_shapes, Sequence) or len(head_shapes) != tp:
                raise ValueError("body layout omitted local-head rank shapes")
            local_vocab_widths = tuple(
                int(shape["logical_rows"])
                for shape in head_shapes
            )
    components: dict[str, float] = {
        "replicated_rmsnorm": float(
            perf.rms_layer(hidden, 1, batch, "decode") * (2 * layers + 1)
        ),
        "rank_local_qkv_projection": float(
            perf.projection(
                hidden,
                local_heads,
                local_kv_heads,
                int(d["head_dim"]),
                1,
                batch,
                "decode",
            )
            * layers
        ),
        "rank_local_output_projection": float(
            perf.output_projection(
                hidden,
                local_heads,
                int(d["head_dim"]),
                1,
                batch,
                "decode",
            )
            * layers
        ),
        "replicated_residual": float(
            perf.residual(hidden, 1, batch, "decode") * 2 * layers
        ),
        "replicated_embedding_lookup": float(
            perf.embeddings(hidden, 1, batch, "decode")
        ),
    }
    if bool(d.get("qk_norm", False)):
        components["rank_local_qk_rmsnorm"] = float(
            (
                perf.rms_layer(
                    int(d["head_dim"]),
                    1,
                    batch * local_heads,
                    "decode",
                )
                + perf.rms_layer(
                    int(d["head_dim"]),
                    1,
                    batch * local_kv_heads,
                    "decode",
                )
            )
            * layers
        )

    full_layers, sliding_layers, window = _attn_split(d)
    full_local_context = math.ceil(kv / kvp)
    sliding_local_context = (
        math.ceil(min(kv, window) / kvp) if sliding_layers else 0
    )
    local_trace_contract = (
        packed_q1_timing_contract if tp == kvp == 1 else None
    )

    def flash(context: int, count: int) -> int:
        if count == 0:
            return 0
        return perf.flash_attention(
            local_heads,
            local_kv_heads,
            int(d["head_dim"]),
            1,
            context,
            batch,
            "decode",
            packed_q1=(kv_layout == DENSE_SELECTOR and kv_head_reuse is not False),
            packed_q1_contract=local_trace_contract,
            batch_packed=batch_packed_attention,
        ) * count

    components["kvp_local_flash_attention"] = float(
        flash(full_local_context, full_layers)
        + flash(sliding_local_context, sliding_layers)
    )

    if is_moe(d):
        active_experts = d.get("moe_unique_experts_per_step")
        active_source = (
            (
                "verified_trace_summary_conservative_observed_max"
                if d.get("moe_route_repricing") is not None
                else "explicit_unique_expert_override"
            )
            if active_experts is not None
            else None
        )
        common = {
            "active_experts": active_experts,
            "active_expert_source": active_source,
            "routing_imbalance_factor": float(
                d.get("moe_routing_imbalance_factor", 1.0)
            ),
        }
        full_expert = perf.moe_decode_expert_timing(
            hidden,
            batch,
            int(d["num_experts"]),
            int(d["experts_per_token"]),
            int(d["inter"]),
            **common,
        )
        full_moe = perf.mlp_moe(
            hidden,
            1,
            batch,
            int(d["num_experts"]),
            int(d["experts_per_token"]),
            int(d["inter"]),
            "decode",
            include_input_norm=False,
            **common,
        )
        replicated_control = full_moe - int(full_expert["expert_stage_cycles"])
        components["replicated_router_topk_softmax_combine"] = float(
            replicated_control * layers
        )
        if expert_parallel_mode == EXPERT_TENSOR_PARALLEL:
            local_expert_cycles = max(
                int(
                    perf.moe_decode_expert_timing(
                        hidden,
                        batch,
                        int(d["num_experts"]),
                        int(d["experts_per_token"]),
                        width,
                        **common,
                    )["expert_stage_cycles"]
                )
                for width in local_inter_widths
            )
        else:
            # Safe projected bound: no route balance is assumed without a
            # measured per-rank route trace.
            local_expert_cycles = int(full_expert["expert_stage_cycles"])
        components["rank_local_routed_experts"] = float(
            local_expert_cycles * layers
        )
    else:
        components["rank_local_dense_ffn"] = float(
            max(
                perf.feed_forward(hidden, width, 1, batch, "decode")
                for width in local_inter_widths
            )
            * layers
        )

    if include_lm_head:
        physical_hidden = math.ceil(hidden / perf.mlen) * perf.mlen
        physical_batch = math.ceil(batch / perf.blen) * perf.blen
        activation_tail = (
            batch * (physical_hidden - hidden)
            + (physical_batch - batch) * physical_hidden
        )
        activation_padding_cycles = (
            math.ceil(activation_tail / perf.vlen) * perf.instr["V_BASIC"]
        )
        rank_head_schedules = []
        for width in local_vocab_widths:
            matrix_cycles = perf.lm_head(physical_hidden, width, batch)
            selection_cycles = perf.lm_head_streaming_selection(
                width,
                batch,
                top_k=20,
            )
            mask_elements = batch * (
                math.ceil(width / perf.mlen) * perf.mlen - width
            )
            mask_cycles = (
                math.ceil(mask_elements / perf.vlen)
                * perf.instr["V_BASIC"]
            )
            rank_head_schedules.append(
                (
                    matrix_cycles
                    + selection_cycles
                    + activation_padding_cycles
                    + mask_cycles,
                    matrix_cycles,
                    selection_cycles,
                    mask_cycles,
                )
            )
        _, matrix_cycles, selection_cycles, mask_cycles = max(
            rank_head_schedules,
            key=lambda value: value[0],
        )
        components["rank_local_lm_head"] = float(matrix_cycles)
        components["rank_local_vocab_selection"] = float(selection_cycles)
        components["rank_local_head_padding"] = float(
            activation_padding_cycles + mask_cycles
        )

    return components


def _partitioned_components(
    perf: PerfModel,
    d: dict,
    kv: int,
    batch: int,
    *,
    tp: int,
    kvp: int,
    include_lm_head: bool,
    kv_layout: str,
    packed_q1_timing_contract,
    batch_packed_attention: bool,
    kv_head_reuse: bool | None,
    body_layout: BodyWeightPhysicalLayout | None = None,
    expert_parallel_mode: str = EXPERT_TENSOR_PARALLEL,
) -> float:
    """Sum the explicit slowest-rank operation schedule without global division."""

    components = _partitioned_component_cycles(
        perf,
        d,
        kv,
        batch,
        tp=tp,
        kvp=kvp,
        include_lm_head=include_lm_head,
        kv_layout=kv_layout,
        packed_q1_timing_contract=packed_q1_timing_contract,
        batch_packed_attention=batch_packed_attention,
        kv_head_reuse=kv_head_reuse,
        body_layout=body_layout,
        expert_parallel_mode=expert_parallel_mode,
    )
    return math.fsum(components.values())


def _partitioned_peak_flops(
    d: dict,
    kv: int,
    batch: int,
    *,
    tp: int,
    kvp: int,
    include_lm_head: bool,
    expert_parallel_mode: str = EXPERT_TENSOR_PARALLEL,
) -> float:
    """Useful FLOPs executed by the slowest explicit-topology rank."""

    if int(d["heads"]) % tp or int(d["kv_heads"]) % tp:
        raise ValueError("TP must own complete attention heads")
    hidden = int(d["hidden"])
    heads = int(d["heads"]) // tp
    kv_heads = int(d["kv_heads"]) // tp
    head_dim = int(d["head_dim"])
    query = heads * head_dim
    kv_width = kv_heads * head_dim
    layers = int(d["layers"])
    qkvo_macs = (
        hidden * query
        + 2 * hidden * kv_width
        + query * hidden
    ) * layers
    if is_moe(d):
        local_inter = (
            max(_balanced_partition_widths(int(d["inter"]), tp))
            if expert_parallel_mode == EXPERT_TENSOR_PARALLEL
            else int(d["inter"])
        )
        expert_macs = (
            3
            * hidden
            * local_inter
            * int(d["experts_per_token"])
            * layers
        )
        # Router is replicated and therefore receives no TP speedup.
        router_macs = hidden * int(d["num_experts"]) * layers
    else:
        local_inter = max(_balanced_partition_widths(int(d["inter"]), tp))
        expert_macs = 3 * hidden * local_inter * layers
        router_macs = 0
    n_full, n_slide, window = _attn_split(d)
    attention_macs = (
        2 * heads * head_dim * math.ceil(kv / kvp) * n_full
    )
    if n_slide:
        attention_macs += (
            2
            * heads
            * head_dim
            * math.ceil(min(kv, window) / kvp)
            * n_slide
        )
    head_macs = (
        hidden * max(_balanced_partition_widths(int(d["vocab"]), tp))
        if include_lm_head
        else 0
    )
    return float(
        2
        * batch
        * (qkvo_macs + expert_macs + router_macs + attention_macs + head_macs)
    )


def collective_cost_per_step(
    d: dict,
    *,
    batch: int,
    tp: int,
    kvp: int,
    link_ports: int,
    link_generation: str = DEFAULT_LINK_GENERATION,
    activation_bytes: int = ACT_BITS // 8,
    include_local_output_selection: bool = False,
    local_output_top_k: int = 20,
    expert_parallel_mode: str = EXPERT_TENSOR_PARALLEL,
) -> dict[str, float]:
    """Dependency-bound collective time and physical bytes for one step.

    TP performs two ring all-reduces per decoder layer.  KVP performs a stable
    distributed-softmax reduction: max, then sum plus the partial value vector.
    """

    if (
        batch <= 0
        or tp <= 0
        or kvp <= 0
        or activation_bytes <= 0
        or local_output_top_k <= 0
    ):
        raise ValueError("collective dimensions must be positive")
    if link_generation not in LINK_GENS:
        raise ValueError("unsupported link generation")
    if expert_parallel_mode not in EXPERT_PARALLEL_MODES:
        raise ValueError("unsupported expert parallel mode")
    active_dimensions = int(tp > 1) + int(kvp > 1)
    if active_dimensions == 0:
        return {
            "tp_bytes": 0.0,
            "kvp_bytes": 0.0,
            "local_output_selection_bytes": 0.0,
            "local_output_selection_time_s": 0.0,
            "expert_routing_bytes": 0.0,
            "expert_routing_time_s": 0.0,
            "expert_output_collective_slowest_rank_bytes": 0.0,
            "expert_output_collective_system_bytes": 0.0,
            "expert_output_collective_time_s": 0.0,
            "total_bytes": 0.0,
            "time_s": 0.0,
        }
    if link_ports < active_dimensions:
        raise ValueError("insufficient link ports for the topology")

    def ring_bytes(payload: float, ranks: int) -> float:
        return 0.0 if ranks == 1 else 2.0 * (ranks - 1) / ranks * payload

    layers = int(d["layers"])
    tp_payload = batch * int(d["hidden"]) * activation_bytes
    tp_bytes = 2.0 * layers * ring_bytes(tp_payload, tp)
    local_heads = int(d["heads"]) // tp
    kvp_payload = (
        batch
        * local_heads
        * (int(d["head_dim"]) + 2)
        * activation_bytes
    )
    kvp_bytes = layers * ring_bytes(kvp_payload, kvp)
    # Each active dimension receives at least one dedicated port.  Extra ports
    # are shared deterministically, preserving a conservative slowest dimension.
    base_ports = link_ports // active_dimensions
    spare = link_ports % active_dimensions
    dimension_ports = []
    if tp > 1:
        dimension_ports.append(base_ports + int(spare > 0))
        spare = max(0, spare - 1)
    if kvp > 1:
        dimension_ports.append(base_ports + int(spare > 0))
    index = 0
    tp_time = 0.0
    kvp_time = 0.0
    local_output_selection_bytes = 0.0
    local_output_selection_time_s = 0.0
    expert_routing_bytes = 0.0
    expert_routing_time_s = 0.0
    expert_output_collective_slowest_rank_bytes = 0.0
    expert_output_collective_system_bytes = 0.0
    expert_output_collective_time_s = 0.0
    if tp > 1:
        tp_ports = dimension_ports[index]
        tp_time = tp_bytes / (LINK_GENS[link_generation] * tp_ports)
        if include_local_output_selection:
            # Each TP rank emits top-k (FP32 score, uint32 token-id) pairs.
            # The designated sampling owner receives every remote rank's
            # bounded candidate list, selects from at most TP*k candidates,
            # and broadcasts one uint32 token id.  KVP replicas execute the
            # same TP-local protocol.  This charges the slowest owner path and
            # every physical byte without ever materializing full logits.
            candidate_bytes = batch * local_output_top_k * (4 + 4)
            token_bytes = batch * 4
            slowest_bytes = (tp - 1) * (candidate_bytes + token_bytes)
            local_output_selection_bytes = slowest_bytes * kvp
            local_output_selection_time_s = slowest_bytes / (
                LINK_GENS[link_generation] * tp_ports
            )
        if expert_parallel_mode == EXPERT_ID_PARALLEL:
            # The first decoder all-reduce leaves the normalized hidden row on
            # every TP rank. Replicated router/top-k work therefore permits a
            # zero-byte local expert-ID filter; dispatching the hidden row again
            # would double-charge traffic. The second already-charged per-layer
            # all-reduce combines the whole-expert partial output. Expose that
            # subset without adding it to the total a second time.
            expert_output_collective_slowest_rank_bytes = tp_bytes / 2.0
            expert_output_collective_system_bytes = (
                expert_output_collective_slowest_rank_bytes * tp * kvp
            )
            expert_output_collective_time_s = tp_time / 2.0
        index += 1
    if kvp > 1:
        kvp_time = kvp_bytes / (LINK_GENS[link_generation] * dimension_ports[index])
    chip_count = tp * kvp
    return {
        "tp_bytes": tp_bytes,
        "kvp_bytes": kvp_bytes,
        # These are slowest-rank byte counts.  Every rank participates in one
        # collective along each active mesh dimension, so aggregate traffic is
        # their sum across the complete TP x KVP mesh.
        "local_output_selection_bytes": local_output_selection_bytes,
        "local_output_selection_time_s": local_output_selection_time_s,
        "expert_routing_bytes": expert_routing_bytes,
        "expert_routing_time_s": expert_routing_time_s,
        "expert_output_collective_slowest_rank_bytes": (
            expert_output_collective_slowest_rank_bytes
        ),
        "expert_output_collective_system_bytes": (
            expert_output_collective_system_bytes
        ),
        "expert_output_collective_time_s": expert_output_collective_time_s,
        "total_bytes": (
            (tp_bytes + kvp_bytes) * chip_count
            + local_output_selection_bytes
        ),
        "time_s": (
            tp_time
            + kvp_time
            + local_output_selection_time_s
        ),
    }


def _traffic_for_policy(
    traffic: DecodeStepTrafficLedger,
    weights: WeightLedger,
    policy: str,
) -> DecodeStepTrafficLedger:
    if policy == "streaming":
        return traffic
    updates: dict[str, int] = {}
    if policy == "projection_resident":
        updates["weight_element_read_bytes"] = max(
            0,
            traffic.weight_element_read_bytes
            - weights.attention.element_aligned,
        )
        updates["weight_scale_read_bytes"] = max(
            0,
            traffic.weight_scale_read_bytes
            - weights.attention.scale_aligned,
        )
    fraction = _kv_resident_fraction(policy)
    if fraction:
        for name in (
            "kv_element_read_bytes",
            "kv_scale_read_bytes",
            "kv_element_write_bytes",
            "kv_scale_write_bytes",
        ):
            updates[name] = int(
                round(getattr(traffic, name) * (1.0 - fraction))
            )
    return replace(traffic, **updates)


def _traffic_for_kv_head_reuse(
    traffic: DecodeStepTrafficLedger,
    *,
    kv_heads: int,
    kv_head_reuse: bool | None,
) -> DecodeStepTrafficLedger:
    """Apply the explicit compiler schedule without changing packed storage."""

    if kv_head_reuse is None or kv_head_reuse:
        return traffic
    if kv_heads <= 0:
        raise ValueError("kv_heads must be positive")
    return replace(
        traffic,
        kv_element_read_bytes=traffic.kv_element_read_bytes * kv_heads,
        kv_scale_read_bytes=traffic.kv_scale_read_bytes * kv_heads,
    )


def _partitioned_kv_ledgers(
    dims: Mapping[str, object],
    precision: Mapping[str, object],
    *,
    context: int,
    batch: int,
    mlen: int,
    kv_layout: str,
    tp: int,
    kvp: int,
) -> tuple[KVLedger, KVLedger, dict[str, object]]:
    """Return exact slowest-rank and aggregate-system PackedKV storage.

    Head partitioning happens before the PackedKV row shape is formed, and the
    full/sliding context spans are partitioned independently across KVP.  The
    slowest rank therefore uses the largest integer context shard rather than
    an average byte count.
    """

    if context <= 0 or batch <= 0 or tp <= 0 or kvp <= 0:
        raise ValueError("partitioned KV dimensions must be positive")
    global_kv_heads = int(dims["kv_heads"])
    if global_kv_heads % tp:
        raise ValueError("TP must own complete KV heads")
    local_kv_heads = global_kv_heads // tp
    block = int(precision.get("block_size", 8))
    key = traffic_from_precision(
        kv_heads=local_kv_heads,
        head_dim=int(dims["head_dim"]),
        mlen=mlen,
        element_bits=int(precision.get("key_elem", precision["kv_elem"])),
        effective_bits=float(precision.get("key_bits", precision["kv_bits"])),
        block_size=block,
    )
    value = traffic_from_precision(
        kv_heads=local_kv_heads,
        head_dim=int(dims["head_dim"]),
        mlen=mlen,
        element_bits=int(precision.get("value_elem", precision["kv_elem"])),
        effective_bits=float(precision.get("value_bits", precision["kv_bits"])),
        block_size=block,
    )
    full_layers, sliding_layers, window = _attn_split(dict(dims))
    sliding_tokens = min(context, window) if sliding_layers else 0
    global_token_layers = full_layers * context + sliding_layers * sliding_tokens
    slowest_token_layers = (
        full_layers * math.ceil(context / kvp)
        + sliding_layers * math.ceil(sliding_tokens / kvp)
    )
    element_per_token = (
        key.storage_element_bytes(kv_layout)
        + value.storage_element_bytes(kv_layout)
    )
    scale_per_token = (
        key.storage_scale_bytes(kv_layout)
        + value.storage_scale_bytes(kv_layout)
    )

    def ledger(token_layers: int, rank_copies: int) -> KVLedger:
        per_batch_element = token_layers * element_per_token
        per_batch_scale = token_layers * scale_per_token
        return KVLedger(
            element_bytes=per_batch_element * batch * rank_copies,
            scale_bytes=per_batch_scale * batch * rank_copies,
            per_batch_element_bytes=per_batch_element * rank_copies,
            per_batch_scale_bytes=per_batch_scale * rank_copies,
            layout_id=(
                f"key={key.layout_id};value={value.layout_id};"
                f"tp={tp};kvp={kvp};partition=rank_local_heads_context"
            ),
        )

    slowest = ledger(slowest_token_layers, 1)
    system = ledger(global_token_layers, tp)
    provenance = {
        "local_kv_heads": local_kv_heads,
        "full_context_tokens_global": context,
        "full_context_tokens_slowest_kvp_rank": math.ceil(context / kvp),
        "sliding_context_tokens_global": sliding_tokens,
        "sliding_context_tokens_slowest_kvp_rank": (
            math.ceil(sliding_tokens / kvp) if sliding_layers else 0
        ),
        "partition_order": "tp_complete_heads_then_kvp_integer_context_shards",
        "system_storage_is_sum_of_rank_local_packed_rows": True,
        "slowest_rank_capacity_checked": True,
    }
    return slowest, system, provenance


def _partitioned_step_traffic_pair(
    dims: Mapping[str, object],
    precision: Mapping[str, object],
    *,
    context: int,
    batch: int,
    mlen: int,
    kv_layout: str,
    tp: int,
    kvp: int,
    weights: BodyWeightPhysicalLayout,
    kv_head_reuse: bool | None,
) -> tuple[DecodeStepTrafficLedger, DecodeStepTrafficLedger]:
    """Return slowest-rank and aggregate-system physical step traffic."""

    local_kv_heads = int(dims["kv_heads"]) // tp
    block = int(precision.get("block_size", 8))
    key = traffic_from_precision(
        kv_heads=local_kv_heads,
        head_dim=int(dims["head_dim"]),
        mlen=mlen,
        element_bits=int(precision.get("key_elem", precision["kv_elem"])),
        effective_bits=float(precision.get("key_bits", precision["kv_bits"])),
        block_size=block,
    )
    value = traffic_from_precision(
        kv_heads=local_kv_heads,
        head_dim=int(dims["head_dim"]),
        mlen=mlen,
        element_bits=int(precision.get("value_elem", precision["kv_elem"])),
        effective_bits=float(precision.get("value_bits", precision["kv_bits"])),
        block_size=block,
    )
    full_layers, sliding_layers, window = _attn_split(dict(dims))
    sliding_tokens = min(context, window) if sliding_layers else 0
    slowest_read_token_layers = (
        full_layers * math.ceil(context / kvp)
        + sliding_layers * math.ceil(sliding_tokens / kvp)
    )
    system_read_token_layers = (
        full_layers * context + sliding_layers * sliding_tokens
    ) * tp
    read_factor = (
        local_kv_heads if kv_head_reuse is False else 1
    )
    key_read_element = key.read_element_bytes(kv_layout)
    value_read_element = value.read_element_bytes(kv_layout)
    key_read_scale = key.read_scale_bytes(kv_layout)
    value_read_scale = value.read_scale_bytes(kv_layout)
    write_element = (
        key.storage_element_bytes(kv_layout)
        + value.storage_element_bytes(kv_layout)
    )
    write_scale = (
        key.storage_scale_bytes(kv_layout)
        + value.storage_scale_bytes(kv_layout)
    )
    embedding_row_bytes = math.ceil(int(dims["hidden"]) * 2 / 64) * 64

    def make(
        ledger: WeightLedger,
        *,
        read_token_layers: int,
        write_rank_copies: int,
        embedding_copies: int,
    ) -> DecodeStepTrafficLedger:
        quantized = (
            ledger.attention + ledger.ffn_streamed + ledger.lm_head_streamed
        )
        return DecodeStepTrafficLedger(
            weight_element_read_bytes=quantized.element_aligned,
            weight_scale_read_bytes=quantized.scale_aligned,
            bf16_weight_read_bytes=(
                ledger.bf16_streamed.total_aligned
                + batch * embedding_row_bytes * embedding_copies
            ),
            activation_read_bytes=0,
            activation_write_bytes=0,
            kv_element_read_bytes=(
                batch
                * read_token_layers
                * (key_read_element + value_read_element)
                * read_factor
            ),
            kv_scale_read_bytes=(
                batch
                * read_token_layers
                * (key_read_scale + value_read_scale)
                * read_factor
            ),
            # One KVP owner writes the new context position. Across the system
            # there is one owner per TP head shard, not one write per KVP rank.
            kv_element_write_bytes=(
                batch * int(dims["layers"]) * write_element * write_rank_copies
            ),
            kv_scale_write_bytes=(
                batch * int(dims["layers"]) * write_scale * write_rank_copies
            ),
        )

    return (
        make(
            weights.slowest_rank,
            read_token_layers=slowest_read_token_layers,
            write_rank_copies=1,
            embedding_copies=1,
        ),
        make(
            weights.system,
            read_token_layers=system_read_token_layers,
            write_rank_copies=tp,
            embedding_copies=tp * kvp,
        ),
    )


def _partition_step_traffic(
    traffic: DecodeStepTrafficLedger,
    *,
    tp: int,
    kvp: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return slowest-rank and aggregate-system physical traffic."""

    rank: dict[str, float] = {}
    system: dict[str, float] = {}
    chips = tp * kvp
    for name in traffic.__dataclass_fields__:
        value = float(getattr(traffic, name))
        divisor = chips if name.startswith("kv_") else tp
        rank[name] = value / divisor
        system[name] = rank[name] * chips
    return rank, system


def _partition_plane(
    plane: PlaneBytes,
    *,
    shards: int,
    replicas: int,
) -> PlaneBytes:
    """Aggregate aligned bytes after equal sharding and replication."""

    copies = shards * replicas

    def aggregate(value: int) -> int:
        return math.ceil(value / shards) * copies if value else 0

    return PlaneBytes(
        element_raw=aggregate(plane.element_raw),
        element_aligned=aggregate(plane.element_aligned),
        scale_raw=aggregate(plane.scale_raw),
        scale_aligned=aggregate(plane.scale_aligned),
    )


def _partition_weight_ledger(
    weights: WeightLedger,
    *,
    tp: int,
    kvp: int,
    sram_policy: str,
) -> WeightLedger:
    def part(value: PlaneBytes) -> PlaneBytes:
        return _partition_plane(value, shards=tp, replicas=kvp)

    attention = PlaneBytes() if sram_policy == "projection_resident" else part(
        weights.attention
    )
    return WeightLedger(
        attention=attention,
        ffn_resident=part(weights.ffn_resident),
        ffn_streamed=part(weights.ffn_streamed),
        lm_head_resident=part(weights.lm_head_resident),
        lm_head_streamed=part(weights.lm_head_streamed),
        bf16_embedding=part(weights.bf16_embedding),
        bf16_norms=part(weights.bf16_norms),
        bf16_lm_head_resident=part(weights.bf16_lm_head_resident),
        bf16_lm_head_streamed=part(weights.bf16_lm_head_streamed),
        bf16_router_resident=part(weights.bf16_router_resident),
        bf16_router_streamed=part(weights.bf16_router_streamed),
    )


def _partition_physical_ledger(
    ledger: PhysicalDecodeLedger,
    *,
    tp: int,
    kvp: int,
    hbm_per_chip: int,
    sram_policy: str,
    batch: int,
    body_weight_layout: BodyWeightPhysicalLayout | None = None,
    slowest_rank_kv: KVLedger | None = None,
    system_kv: KVLedger | None = None,
) -> PhysicalDecodeLedger:
    """Build aggregate HBM accounting while retaining per-chip SRAM limits."""

    if (
        body_weight_layout is None
        and tp == kvp == 1
        and sram_policy == "streaming"
    ):
        return ledger
    chips = tp * kvp
    if body_weight_layout is not None:
        if slowest_rank_kv is None or system_kv is None:
            raise ValueError("rank-local body layout requires rank-local KV ledgers")
        if int(body_weight_layout.provenance["tp"]) != tp or int(
            body_weight_layout.provenance["kvp"]
        ) != kvp:
            raise ValueError("body layout topology differs from capacity topology")
        rank_weights = body_weight_layout.slowest_rank
        weights = body_weight_layout.system
        if sram_policy == "projection_resident":
            rank_attention = rank_weights.attention
            rank_weights = replace(rank_weights, attention=PlaneBytes())
            weights = replace(weights, attention=PlaneBytes())
        else:
            rank_attention = PlaneBytes()
        kv_fraction = _kv_resident_fraction(sram_policy)

        def hbm_kv_exact(value: KVLedger) -> KVLedger:
            return KVLedger(
                element_bytes=int(round(value.element_bytes * (1.0 - kv_fraction))),
                scale_bytes=int(round(value.scale_bytes * (1.0 - kv_fraction))),
                per_batch_element_bytes=int(
                    round(value.per_batch_element_bytes * (1.0 - kv_fraction))
                ),
                per_batch_scale_bytes=int(
                    round(value.per_batch_scale_bytes * (1.0 - kv_fraction))
                ),
                layout_id=(
                    value.layout_id
                    if kv_fraction == 0.0
                    else f"{value.layout_id}:{sram_policy}"
                ),
            )

        rank_hbm_kv = hbm_kv_exact(slowest_rank_kv)
        kv = hbm_kv_exact(system_kv)
        sram = ledger.sram
        vector_per_sequence = sram.vector_bytes_per_sequence + math.ceil(
            slowest_rank_kv.per_batch_bytes * kv_fraction
        )
        vector_required = max(
            vector_per_sequence * batch,
            sram.output_head_workspace_bytes,
        )
        matrix_required = (
            sram.matrix_required_bytes + rank_attention.total_aligned
        )
        body_max_vector_batch = (
            sram.vector_capacity_bytes // max(vector_per_sequence, 1)
        )
        if sram.output_head_logit_tile_bytes:
            per_sequence_selection_state = (
                sram.output_head_selection_state_bytes // max(batch, 1)
            )
            head_max_vector_batch = max(
                0,
                (
                    sram.vector_capacity_bytes
                    - sram.output_head_logit_tile_bytes
                )
                // max(per_sequence_selection_state, 1),
            )
            max_vector_batch = min(body_max_vector_batch, head_max_vector_batch)
        else:
            max_vector_batch = body_max_vector_batch
        max_synchronous = (
            min(max_vector_batch, sram.max_synchronous_batch)
            if matrix_required <= sram.matrix_capacity_bytes
            else 0
        )
        sram = SRAMLedger(
            vector_capacity_bytes=sram.vector_capacity_bytes,
            vector_bytes_per_sequence=vector_per_sequence,
            vector_required_bytes=vector_required,
            matrix_capacity_bytes=sram.matrix_capacity_bytes,
            matrix_required_bytes=matrix_required,
            matrix_tile_capacity=sram.matrix_tile_capacity,
            matrix_required_tiles=sram.matrix_required_tiles,
            max_vector_batch=max_vector_batch,
            max_synchronous_batch=max_synchronous,
            output_head_logit_tile_bytes=sram.output_head_logit_tile_bytes,
            output_head_selection_state_bytes=(
                sram.output_head_selection_state_bytes
            ),
            output_head_workspace_bytes=sram.output_head_workspace_bytes,
        )
        hbm_capacity = hbm_per_chip * chips
        system_reserve = ledger.runtime_hbm_reserve_bytes * chips
        system_fixed = weights.resident.total_aligned + system_reserve
        rank_fixed = (
            rank_weights.resident.total_aligned
            + ledger.runtime_hbm_reserve_bytes
        )
        system_max = max(0, hbm_capacity - system_fixed) // max(
            kv.per_batch_bytes, 1
        )
        rank_max = max(0, hbm_per_chip - rank_fixed) // max(
            rank_hbm_kv.per_batch_bytes, 1
        )
        max_resident = min(system_max, rank_max)
        max_runtime = (
            min(max_resident, sram.max_vector_batch)
            if sram.matrix_required_bytes <= sram.matrix_capacity_bytes
            else 0
        )
        return PhysicalDecodeLedger(
            weights=weights,
            kv=kv,
            sram=sram,
            hbm_capacity_bytes=hbm_capacity,
            runtime_hbm_reserve_bytes=system_reserve,
            hbm_required_bytes=system_fixed + kv.total_bytes,
            max_resident_batch=max_resident,
            max_runtime_batch=max_runtime,
            kv_layout=ledger.kv_layout,
            slowest_rank_hbm_required_bytes=(
                rank_fixed + rank_hbm_kv.total_bytes
            ),
            per_chip_hbm_capacity_bytes=hbm_per_chip,
        )

    weights = _partition_weight_ledger(
        ledger.weights,
        tp=tp,
        kvp=kvp,
        sram_policy=sram_policy,
    )
    kv_fraction = _kv_resident_fraction(sram_policy)

    def hbm_kv(value: int) -> int:
        retained = int(round(value * (1.0 - kv_fraction)))
        return math.ceil(retained / chips) * chips if retained else 0

    kv = KVLedger(
        element_bytes=hbm_kv(ledger.kv.element_bytes),
        scale_bytes=hbm_kv(ledger.kv.scale_bytes),
        per_batch_element_bytes=hbm_kv(
            ledger.kv.per_batch_element_bytes
        ),
        per_batch_scale_bytes=hbm_kv(ledger.kv.per_batch_scale_bytes),
        layout_id=(
            ledger.kv.layout_id
            if sram_policy == "streaming"
            else f"{ledger.kv.layout_id}:{sram_policy}"
        ),
    )
    sram = ledger.sram
    vector_per_sequence = sram.vector_bytes_per_sequence + math.ceil(
        ledger.kv.per_batch_bytes * kv_fraction / chips
    )
    vector_required = max(
        vector_per_sequence * batch,
        sram.output_head_workspace_bytes,
    )
    projection_bytes = (
        math.ceil(ledger.weights.attention.total_aligned / tp)
        if sram_policy == "projection_resident"
        else 0
    )
    matrix_required = sram.matrix_required_bytes + projection_bytes
    body_max_vector_batch = (
        sram.vector_capacity_bytes // max(vector_per_sequence, 1)
    )
    if sram.output_head_logit_tile_bytes:
        per_sequence_selection_state = (
            sram.output_head_selection_state_bytes // max(batch, 1)
        )
        head_max_vector_batch = max(
            0,
            (
                sram.vector_capacity_bytes
                - sram.output_head_logit_tile_bytes
            )
            // max(per_sequence_selection_state, 1),
        )
        max_vector_batch = min(
            body_max_vector_batch,
            head_max_vector_batch,
        )
    else:
        max_vector_batch = body_max_vector_batch
    max_synchronous = (
        min(max_vector_batch, sram.max_synchronous_batch)
        if matrix_required <= sram.matrix_capacity_bytes
        else 0
    )
    sram = SRAMLedger(
        vector_capacity_bytes=sram.vector_capacity_bytes,
        vector_bytes_per_sequence=vector_per_sequence,
        vector_required_bytes=vector_required,
        matrix_capacity_bytes=sram.matrix_capacity_bytes,
        matrix_required_bytes=matrix_required,
        matrix_tile_capacity=sram.matrix_tile_capacity,
        matrix_required_tiles=sram.matrix_required_tiles,
        max_vector_batch=max_vector_batch,
        max_synchronous_batch=max_synchronous,
        output_head_logit_tile_bytes=(
            sram.output_head_logit_tile_bytes
        ),
        output_head_selection_state_bytes=(
            sram.output_head_selection_state_bytes
        ),
        output_head_workspace_bytes=sram.output_head_workspace_bytes,
    )
    hbm_capacity = hbm_per_chip * chips
    runtime_reserve = ledger.runtime_hbm_reserve_bytes * chips
    resident_fixed = weights.resident.total_aligned + runtime_reserve
    available_kv = max(0, hbm_capacity - resident_fixed)
    max_resident = available_kv // max(kv.per_batch_bytes, 1)
    max_runtime = (
        min(max_resident, sram.max_vector_batch)
        if sram.matrix_required_bytes <= sram.matrix_capacity_bytes
        else 0
    )
    return PhysicalDecodeLedger(
        weights=weights,
        kv=kv,
        sram=sram,
        hbm_capacity_bytes=hbm_capacity,
        runtime_hbm_reserve_bytes=runtime_reserve,
        hbm_required_bytes=resident_fixed + kv.total_bytes,
        max_resident_batch=max_resident,
        max_runtime_batch=max_runtime,
        kv_layout=ledger.kv_layout,
    )


def run_decode_loop(perf, mem, d, prec, input_seq, output_seq, batch, peak_bw, stride, overfetch,
                    batch_packed_attention=False, n_chips=1, bw_model=None, hbm_gen="HBM2", hbm_channels=8,
                    hbm_pin_rate_gbps=None,
                    kv_layout=DENSE_SELECTOR, ideal_perf=None,
                    physical_weights=None, include_lm_head=True,
                    packed_q1_timing_contract=None,
                    tp=1, kvp=1, link_ports=0,
                    link_generation=DEFAULT_LINK_GENERATION,
                    sram_policy="streaming",
                    legacy_ideal_parallelism=False,
                    kv_head_reuse=None,
                    body_weight_layout: BodyWeightPhysicalLayout | None = None,
                    expert_parallel_mode=EXPERT_TENSOR_PARALLEL,
                    layer_exact_moe_route_projection: Mapping[str, object] | None = None,
                    execution_mode=LEGACY_AGGREGATE_BANDWIDTH,
                    trace_timing_provider=None,
                    trace_request_factory: Callable[[int], object] | None = None):
    """Walk the growing-context decode under one explicit timing contract.

    Compiler-trace timing is accepted only from a provider whose builder
    declares a full-model, independent-request decode step.  The physical
    traffic ledger remains authoritative for capacity, energy, and reporting;
    aggregate-bandwidth time is used only by the named compatibility mode.

    `stride` subsamples the loop for speed.  Multi-chip points explicitly shard
    attention heads/FFN columns over TP and cache sequence ranges over KVP;
    weights are sharded by TP and replicated by KVP.

    memory_time: with a `bw_model` (disagg_serve.memory.CalibratedBandwidth) bytes
    are priced at the measured effective bandwidth per class for (hbm_gen,
    hbm_channels); otherwise at aggregate peak. The returned decomposition keeps
    classical peak, ideal matrix issue, and realized timing as separate views."""
    # M-bit density scales the compute side: the cycle model is calibrated on the
    # 4-bit array, and density x more (or fewer) MACs finish density x faster.
    density = compute_density(prec)
    if tp <= 0 or kvp <= 0 or tp * kvp != n_chips:
        raise ValueError("run_decode_loop requires n_chips == TP * KVP")
    single_chip_compatibility = (
        tp == 1 and kvp == 1 and sram_policy == "streaming"
        and kv_head_reuse is None
    )
    compatibility_path = (
        single_chip_compatibility or legacy_ideal_parallelism
    )
    if ideal_perf is None:
        ideal_perf = perf
    if execution_mode == COMPILER_TRACE:
        if trace_timing_provider is None or trace_request_factory is None:
            raise RuntimeError(
                "compiler_trace mode requires a provider and request factory"
            )
        if getattr(trace_timing_provider, "artifact_scope", None) != (
            FULL_MODEL_DECODE_SCOPE
        ):
            raise ValueError(
                "compiler_trace mode requires full-model independent-request "
                "decode artifacts"
            )
    elif execution_mode == LEGACY_AGGREGATE_BANDWIDTH:
        if trace_timing_provider is not None or trace_request_factory is not None:
            raise ValueError(
                "legacy_aggregate_bandwidth rejects compiler-trace inputs"
            )
    else:
        raise ValueError(f"unsupported decode execution mode {execution_mode!r}")
    exact_route_cycles: tuple[float, ...] | None = None
    exact_route_rank_planes: tuple[PlaneBytes, ...] | None = None
    exact_route_system_planes: tuple[PlaneBytes, ...] | None = None
    exact_route_rank_total = PlaneBytes()
    exact_route_system_total = PlaneBytes()
    if layer_exact_moe_route_projection is not None:
        if (
            execution_mode != LEGACY_AGGREGATE_BANDWIDTH
            or compatibility_path
            or body_weight_layout is None
            or not is_moe(d)
        ):
            raise ValueError(
                "layer-exact MoE route projection requires the explicit-topology "
                "legacy analytic body path"
            )
        (
            exact_route_cycles,
            exact_route_rank_planes,
            exact_route_system_planes,
        ) = _validate_layer_exact_moe_route_projection(
            layer_exact_moe_route_projection,
            d,
            batch=batch,
            tp=tp,
            kvp=kvp,
            expert_parallel_mode=expert_parallel_mode,
        )
        for value in exact_route_rank_planes:
            exact_route_rank_total += value
        for value in exact_route_system_planes:
            exact_route_system_total += value
    total_time, total_bytes, first_step = 0.0, 0, None
    compute_time_total = 0.0
    peak_compute_time_total = 0.0
    ideal_compute_time_total = 0.0
    memory_time_total = 0.0
    collective_time_total = 0.0
    collective_bytes_total = 0.0
    collective_breakdown_per_batch_step: dict[str, float] | None = None
    traffic_totals = {
        "weight_element_read_bytes": 0.0,
        "weight_scale_read_bytes": 0.0,
        "bf16_weight_read_bytes": 0.0,
        "activation_read_bytes": 0.0,
        "activation_write_bytes": 0.0,
        "kv_element_read_bytes": 0.0,
        "kv_scale_read_bytes": 0.0,
        "kv_element_write_bytes": 0.0,
        "kv_scale_write_bytes": 0.0,
    }
    mem_bound = 0
    classical_mem_bound = 0
    architecture_issue_mem_bound = 0
    serialization_bound = 0
    peak_compute_per_chip_second = (
        2 * perf.mlen * perf.blen * FREQ_HZ * density
    )
    peak_compute_per_second = (
        peak_compute_per_chip_second * n_chips
    )
    def step_seconds(model: PerfModel, kv: int) -> float:
        """Wall-clock of one decode step under `model`'s timing contract."""
        if not compatibility_path:
            components = _partitioned_component_cycles(
                model,
                d,
                kv,
                batch,
                tp=tp,
                kvp=kvp,
                include_lm_head=include_lm_head,
                kv_layout=kv_layout,
                packed_q1_timing_contract=packed_q1_timing_contract,
                batch_packed_attention=batch_packed_attention,
                kv_head_reuse=kv_head_reuse,
                body_layout=body_weight_layout,
                expert_parallel_mode=expert_parallel_mode,
            )
            if exact_route_cycles is not None:
                if "rank_local_routed_experts" not in components:
                    raise AssertionError("exact MoE route hook found no expert stage")
                components["rank_local_routed_experts"] = math.fsum(
                    exact_route_cycles
                )
            cycles = math.fsum(components.values())
            return cycles_to_seconds(
                cycles,
                frequency_hz=FREQ_HZ,
                compute_density=density,
                chip_count=1,
            )
        cycles = decode_token_cycles(
            model,
            d,
            kv,
            batch,
            include_lm_head=include_lm_head,
            kv_layout=kv_layout,
            packed_q1_timing_contract=packed_q1_timing_contract,
            batch_packed_attention=batch_packed_attention,
            kv_head_reuse=kv_head_reuse,
        )
        if include_lm_head and n_chips > 1:
            # Legacy ideal parallelism divides aggregate work by chip count,
            # but each rank must still initialize its own padded activation
            # rows/columns.  Add the copies removed by the final division.
            head_padding = model.lm_head_padding_preparation(
                int(d["hidden"]),
                int(d["vocab"]),
                batch,
            )
            cycles += int(head_padding["zero_fill_cycles_per_rank"]) * (
                n_chips - 1
            )
        return cycles_to_seconds(
            cycles,
            frequency_hz=FREQ_HZ,
            compute_density=density,
            chip_count=n_chips,
        )

    trace_requests: dict[int, object] = {}
    trace_timing_evidence: dict[str, object] | None = None
    if execution_mode == COMPILER_TRACE:
        sampled_contexts = tuple(
            input_seq + offset for offset in range(0, output_seq, stride)
        )
        trace_requests = {
            context: trace_request_factory(context)
            for context in sampled_contexts
        }
        prepared = trace_timing_provider.prepare(trace_requests.values())
        if len(prepared) != len(sampled_contexts):
            raise RuntimeError("compiler trace provider changed the request count")
        for context, result in zip(sampled_contexts, prepared):
            if result.context_tokens != context or result.batch != batch:
                raise ValueError(
                    "compiler trace request differs from the serving context or batch"
                )
            if result.artifact_scope != FULL_MODEL_DECODE_SCOPE:
                raise ValueError(
                    "compiler trace result lacks full-model decode scope"
                )
        descriptor_ids = {
            result.compiler_inputs_sha256 for result in prepared
        }
        if len(descriptor_ids) != 1:
            raise ValueError(
                "compiler trace requests do not share one decode-point descriptor"
            )
        lowering_ids = {
            result.compiler_lowering_sha256 for result in prepared
        }
        if None in lowering_ids or len(lowering_ids) != 1:
            raise ValueError(
                "compiler trace requests do not share one lowering key"
            )
        artifact_record_ids = {
            result.artifact_record_sha256 for result in prepared
        }
        if None in artifact_record_ids:
            raise ValueError("compiler trace timing lacks artifact-record identity")
        trace_timing_evidence = {
            "schema_version": COMPILER_TRACE_TIMING_SET_SCHEMA,
            "execution_mode": COMPILER_TRACE,
            "artifact_scope": FULL_MODEL_DECODE_SCOPE,
            "request_count": len(prepared),
            "compiler_input_descriptor_sha256": next(iter(descriptor_ids)),
            "compiler_lowering_key_sha256": next(iter(lowering_ids)),
            "compiler_artifact_set_sha256": canonical_sha256(
                sorted(artifact_record_ids)
            ),
            "request_set_sha256": canonical_sha256(
                [result.provenance for result in prepared]
            ),
            "compiler_source_sha256": canonical_sha256(
                sorted({result.compiler_source_sha256 for result in prepared})
            ),
            "latency_library_sha256": canonical_sha256(
                sorted({result.latency_library_sha256 for result in prepared})
            ),
            "request_memory_sidecar_set_sha256": canonical_sha256(
                sorted(
                    {
                        result.request_memory_sidecar_sha256
                        for result in prepared
                    }
                )
            ),
            "request_memory_calibration_ids": sorted(
                {result.memory_calibration_id for result in prepared}
            ),
            "step_composition": TRACE_STEP_COMPOSITION,
        }

    t = 0
    while t < output_seq:
        kv = input_seq + t                                    # KV cache grows by one each token
        if execution_mode == LEGACY_AGGREGATE_BANDWIDTH:
            legacy_compute_time = step_seconds(perf, kv)
            ideal_compute_time = step_seconds(ideal_perf, kv)
        else:
            legacy_compute_time = None
            ideal_compute_time = None
        if compatibility_path:
            peak_compute_time = decode_step_flops(
                d,
                kv,
                batch,
                include_lm_head=include_lm_head,
            ) / peak_compute_per_second
        else:
            peak_compute_time = _partitioned_peak_flops(
                d,
                kv,
                batch,
                tp=tp,
                kvp=kvp,
                include_lm_head=include_lm_head,
                expert_parallel_mode=expert_parallel_mode,
            ) / peak_compute_per_chip_second
        step_traffic = decode_step_traffic_ledger(
            d,
            prec,
            context=kv,
            batch=batch,
            mlen=perf.mlen,
            kv_layout=kv_layout,
            weights=physical_weights,
            include_lm_head=include_lm_head,
            tp=tp,
        )
        step_traffic = _traffic_for_kv_head_reuse(
            step_traffic,
            # Rank-local head count: the ledger already sums the tp ranks.
            kv_heads=int(d["kv_heads"]) // tp,
            kv_head_reuse=kv_head_reuse,
        )
        if compatibility_path:
            tr = decode_traffic(
                mem,
                d,
                kv,
                batch,
                prec,
                mlen=perf.mlen,
                kv_layout=kv_layout,
                physical_weights=physical_weights,
                physical_step=step_traffic,
                include_lm_head=include_lm_head,
            )
            rank_read_bytes = tr.read_bytes
            rank_write_bytes = tr.write_bytes
            system_traffic = {
                name: float(getattr(step_traffic, name))
                for name in traffic_totals
            }
        else:
            if body_weight_layout is not None:
                rank_step, system_step = _partitioned_step_traffic_pair(
                    d,
                    prec,
                    context=kv,
                    batch=batch,
                    mlen=perf.mlen,
                    kv_layout=kv_layout,
                    tp=tp,
                    kvp=kvp,
                    weights=body_weight_layout,
                    kv_head_reuse=kv_head_reuse,
                )
                rank_policy = _traffic_for_policy(
                    rank_step,
                    body_weight_layout.slowest_rank,
                    sram_policy,
                )
                system_policy = _traffic_for_policy(
                    system_step,
                    body_weight_layout.system,
                    sram_policy,
                )
                rank_traffic = {
                    name: float(getattr(rank_policy, name))
                    for name in traffic_totals
                }
                system_traffic = {
                    name: float(getattr(system_policy, name))
                    for name in traffic_totals
                }
                if exact_route_rank_planes is not None:
                    replacements = (
                        (
                            rank_traffic,
                            body_weight_layout.slowest_rank.ffn_streamed,
                            exact_route_rank_total,
                        ),
                        (
                            system_traffic,
                            body_weight_layout.system.ffn_streamed,
                            exact_route_system_total,
                        ),
                    )
                    for target, original, replacement in replacements:
                        target["weight_element_read_bytes"] += float(
                            replacement.element_aligned
                            - original.element_aligned
                        )
                        target["weight_scale_read_bytes"] += float(
                            replacement.scale_aligned
                            - original.scale_aligned
                        )
                        if (
                            target["weight_element_read_bytes"] < 0
                            or target["weight_scale_read_bytes"] < 0
                        ):
                            raise AssertionError(
                                "layer-exact expert replacement made traffic negative"
                            )
            else:
                policy_traffic = _traffic_for_policy(
                    step_traffic,
                    physical_weights,
                    sram_policy,
                )
                rank_traffic, system_traffic = _partition_step_traffic(
                    policy_traffic,
                    tp=tp,
                    kvp=kvp,
                )
            rank_read_bytes = sum(
                value
                for name, value in rank_traffic.items()
                if name.endswith("_read_bytes")
            )
            rank_write_bytes = sum(
                value
                for name, value in rank_traffic.items()
                if name.endswith("_write_bytes")
            )
        rank_bytes = rank_read_bytes * overfetch + rank_write_bytes
        system_bytes = (
            rank_bytes
            if compatibility_path
            else sum(
                value
                * (overfetch if name.endswith("_read_bytes") else 1.0)
                for name, value in system_traffic.items()
            )
        )
        legacy_memory_time = None
        if execution_mode == LEGACY_AGGREGATE_BANDWIDTH:
            if bw_model is not None:
                # One H_PREFETCH_M moves an MLEN x MLEN weight tile at the
                # widest streamed width. That per-DMA size keys the calibrated
                # aggregate-bandwidth curve used only by compatibility mode.
                wt_transfer = (
                    perf.mlen
                    * perf.mlen
                    * max(prec["attn_elem"], prec["ffn_elem"])
                    / 8
                )
                legacy_memory_time = bw_model.memory_time(
                    {
                        "weights_kv": rank_read_bytes * overfetch,
                        "writeback": rank_write_bytes,
                    },
                    hbm_gen,
                    hbm_channels,
                    transfer_bytes=wt_transfer,
                    pin_rate_gbps=hbm_pin_rate_gbps,
                ) / (n_chips if legacy_ideal_parallelism else 1)
            else:
                legacy_memory_time = rank_bytes / (
                    peak_bw * (n_chips if legacy_ideal_parallelism else 1)
                )

        collective = (
            {
                "tp_bytes": 0.0,
                "kvp_bytes": 0.0,
                "local_output_selection_bytes": 0.0,
                "local_output_selection_time_s": 0.0,
                "expert_routing_bytes": 0.0,
                "expert_routing_time_s": 0.0,
                "expert_output_collective_slowest_rank_bytes": 0.0,
                "expert_output_collective_system_bytes": 0.0,
                "expert_output_collective_time_s": 0.0,
                "total_bytes": 0.0,
                "time_s": 0.0,
            }
            if legacy_ideal_parallelism
            else collective_cost_per_step(
                d,
                batch=batch,
                tp=tp,
                kvp=kvp,
                link_ports=link_ports,
                link_generation=link_generation,
                include_local_output_selection=bool(
                    prec.get("lm_head_quantized", False)
                ),
                local_output_top_k=int(prec.get("lm_head_top_k", 20)),
                expert_parallel_mode=expert_parallel_mode,
            )
        )
        collective_time = collective["time_s"]
        if collective_breakdown_per_batch_step is None:
            collective_breakdown_per_batch_step = dict(collective)
        if execution_mode == COMPILER_TRACE:
            resolved_step = resolve_decode_step_timing(
                COMPILER_TRACE,
                trace_timing_provider=trace_timing_provider,
                trace_request=trace_requests[kv],
            )
        else:
            resolved_step = resolve_decode_step_timing(
                LEGACY_AGGREGATE_BANDWIDTH,
                legacy_compute_seconds=legacy_compute_time,
                legacy_memory_seconds=legacy_memory_time,
            )
        compute_time = resolved_step.compute_seconds
        memory_time = resolved_step.memory_seconds
        if execution_mode == COMPILER_TRACE:
            # The trace's instruction cycles are the issue-time diagnostic in
            # compiler mode. No legacy PackedQ1 or aggregate-BW calculation is
            # allowed to gate the trace timing path.
            ideal_compute_time = compute_time
        step_time = resolved_step.total_seconds + collective_time
        span = min(stride, output_seq - t)                    # tokens this sample stands for
        total_time += step_time * span
        total_bytes += system_bytes * span
        compute_time_total += compute_time * span
        peak_compute_time_total += peak_compute_time * span
        ideal_compute_time_total += ideal_compute_time * span
        memory_time_total += memory_time * span
        collective_time_total += collective_time * span
        collective_bytes_total += collective["total_bytes"] * span
        for name in traffic_totals:
            value = system_traffic[name]
            if name.endswith("_read_bytes"):
                value *= overfetch
            traffic_totals[name] += value * span
        if memory_time >= compute_time:                       # memory paced this token
            mem_bound += span
        if memory_time >= peak_compute_time:
            classical_mem_bound += span
        if memory_time >= ideal_compute_time:
            architecture_issue_mem_bound += span
            if memory_time < compute_time:
                serialization_bound += span
        if first_step is None:
            first_step = step_time
        t += stride
    per_step = {
        name: value / output_seq
        for name, value in traffic_totals.items()
    }
    per_generated = {
        name: value / batch
        for name, value in per_step.items()
    }
    read_bytes_total = sum(
        value for name, value in traffic_totals.items() if name.endswith("_read_bytes")
    )
    write_bytes_total = sum(
        value for name, value in traffic_totals.items() if name.endswith("_write_bytes")
    )
    route_projection_receipt = None
    if layer_exact_moe_route_projection is not None:
        route_projection_receipt = {
            "schema": LAYER_EXACT_MOE_ROUTE_PROJECTION_SCHEMA,
            "content_hash": layer_exact_moe_route_projection["content_hash"],
            "mapping": layer_exact_moe_route_projection["mapping"],
            "layer_count": len(exact_route_cycles or ()),
            "sum_of_per_layer_slowest_rank_expert_stage_cycles": math.fsum(
                exact_route_cycles or ()
            ),
            "sum_of_per_layer_slowest_rank_expert_streamed_element_bytes": (
                exact_route_rank_total.element_aligned
            ),
            "sum_of_per_layer_slowest_rank_expert_streamed_scale_bytes": (
                exact_route_rank_total.scale_aligned
            ),
            "sum_of_per_layer_slowest_rank_expert_streamed_bytes": (
                exact_route_rank_total.total_aligned
            ),
            "system_expert_streamed_bytes": (
                exact_route_system_total.total_aligned
            ),
            "source_hidden_dispatch_bytes": 0,
            "global_layer_collapse_allowed": False,
            "other_decode_components_changed": False,
        }
    return {"total_time": total_time, "tpot": total_time / output_seq,
            "tps": (batch * output_seq) / total_time, "first_step": first_step,
            "step_composition": STEP_COMPOSITION,
            "execution_mode": execution_mode,
            "compiler_trace_timing": trace_timing_evidence,
            "batch_packed_attention": batch_packed_attention,
            "read_bytes_per_second": read_bytes_total / total_time,
            "write_bytes_per_second": write_bytes_total / total_time,
            "array_active_fraction": min(1.0, compute_time_total / total_time),
            "avg_bytes_per_batch_step": total_bytes / output_seq,
            "avg_bytes_per_generated_token": total_bytes / (output_seq * batch),
            "avg_bytes_per_token": total_bytes / output_seq,
            "avg_peak_compute_seconds": peak_compute_time_total / output_seq,
            "avg_realized_compute_seconds": compute_time_total / output_seq,
            "avg_ideal_compute_seconds": ideal_compute_time_total / output_seq,
            "avg_memory_seconds": memory_time_total / output_seq,
            "avg_collective_seconds": collective_time_total / output_seq,
            "collective_bytes_per_batch_step": (
                collective_bytes_total / output_seq
            ),
            "collective_bytes_per_generated_token": (
                collective_bytes_total / (output_seq * batch)
            ),
            "collective_breakdown_per_batch_step": (
                collective_breakdown_per_batch_step or {}
            ),
            "link_bytes_per_second": (
                collective_bytes_total / total_time
            ),
            "parallelism": {
                "tp": tp,
                "kvp": kvp,
                "chip_count": n_chips,
                "link_ports": link_ports,
                "link_generation": link_generation,
                "sram_policy": sram_policy,
                "collective_composition": "dependency_bound_additive",
                "kv_head_reuse": kv_head_reuse,
            },
            "layer_exact_moe_route_projection": route_projection_receipt,
            "traffic_breakdown_per_batch_step": per_step,
            "traffic_breakdown_per_generated_token": per_generated,
            "frac_mem_bound": mem_bound / output_seq,
            "frac_classical_mem_bound": classical_mem_bound / output_seq,
            "frac_architecture_issue_mem_bound": (
                architecture_issue_mem_bound / output_seq
            ),
            "frac_algorithmic_mem_bound": (
                architecture_issue_mem_bound / output_seq
            ),
            "frac_communication_bound": (
                collective_time_total / total_time
            ),
            "frac_serialization_bound": serialization_bound / output_seq}


def _timing_cache_schedule(
    d: dict,
    input_seq: int,
    output_seq: int,
    stride: int,
) -> tuple[int, ...]:
    """Return every cache length consumed by the sampled attention path."""

    contexts = tuple(
        input_seq + offset
        for offset in range(0, output_seq, stride)
    )
    _, n_sliding, window = _attn_split(d)
    values = set(contexts)
    values.add(input_seq + output_seq // 2)
    if n_sliding:
        values.update(min(context, window) for context in contexts)
    return tuple(sorted(values))


def evaluate(model_path, dims, hw_cfg, isa_path, base_mem, prec, batch,
             input_seq, output_seq, hw_over=None, stride=1, n_chips=0,
             batch_packed_attention=False,
             bw_model=None, hbm_gen="HBM2", hbm_channels=8,
             kv_layout=DENSE_SELECTOR, timing_mode=RTL_SERIALIZED,
             timing_evidence=None, runtime_hbm_reserve_bytes=0,
             output_head_location=DECODE_BF16_HEAD,
             packed_q1_timing_contract=None,
             execution_mode=LEGACY_AGGREGATE_BANDWIDTH,
             trace_timing_provider=None,
             trace_request_factory: Callable[[int], object] | None = None):
    """Metrics for one (hardware, precision, batch) point. `n_chips`: 0 = auto
    (fewest HBM stacks that hold the model), else a fixed count; a model that fits
    one stack resolves to 1 chip. `runtime_hbm_reserve_bytes` is a per-chip
    reserve on every topology path (legacy aggregate and explicit TP x KVP)."""
    prec = dict(prec)
    local_mx_head = output_head_location == DECODE_MX_HEAD
    prec["lm_head_quantized"] = local_mx_head
    prec["lm_head_top_k"] = 20
    if local_mx_head:
        required_head_precision = {
            "profile_id",
            "head_bits",
            "head_elem",
            "head_label",
            "head_activation_bits",
            "head_activation_elem",
            "head_activation_label",
            "head_vector_format",
            "head_matrix_storage_format",
            "head_logit_container_format",
            "head_bf16_container_precision_recovery",
            "head_operand_family_supported",
            "head_operand_family_binding",
            "head_numerical_oracle_rule",
            "head_partial_conversion_rule",
            "head_hardware_bit_parity_verified",
            "head_accumulation_chain",
            "head_numerical_matrix_mlen",
        }
        missing_head_precision = sorted(required_head_precision - set(prec))
        if missing_head_precision:
            raise ValueError(
                "decode-local MX head lacks profile W/A bindings: "
                + ",".join(missing_head_precision)
            )
        if (
            prec["head_bits"] != prec["attn_bits"]
            or prec["head_elem"] != prec["attn_elem"]
            or prec["head_label"] != prec["attn_label"]
            or int(prec.get("block_size", 0)) != 8
        ):
            raise ValueError(
                "decode-local MX head must follow profile W with E8M0 block8"
            )
        if (
            prec["head_matrix_storage_format"]
            != prec["head_vector_format"]
            or prec["head_logit_container_format"] != "BF16"
            or prec["head_bf16_container_precision_recovery"] is not False
        ):
            raise ValueError(
                "local-head matrix format and BF16 container binding differ"
            )
    route_repricing = dims.get("moe_route_repricing")
    if route_repricing is not None:
        if int(route_repricing["batch_size"]) != int(batch):
            raise ValueError(
                "MoE route overlay batch does not match the evaluated batch"
            )
        if execution_mode != LEGACY_AGGREGATE_BANDWIDTH:
            raise ValueError(
                "MoE route overlays are supported only by the legacy analytic "
                "sensitivity path; compiler-trace timing is not route-bound"
            )
    include_lm_head = decoder_owns_output_head(output_head_location)
    topology = _parallel_topology(hw_over, n_chips)
    architecture_knobs_explicit = bool(
        topology["architecture_knobs_explicit"]
    )
    effective_kv_head_reuse = (
        bool(topology["kv_head_reuse"])
        if architecture_knobs_explicit
        else None
    )
    resolved_timing_mode = (
        (
            DRAIN_OVERLAPPED
            if bool(topology["drain_overlapped"])
            else RTL_SERIALIZED
        )
        if architecture_knobs_explicit
        else timing_mode
    )
    perf = PerfModel(hw_cfg, isa_path, timing_mode=resolved_timing_mode)
    ideal_perf = (
        perf
        if resolved_timing_mode == IDEAL_MATRIX_PIPELINE
        else PerfModel(hw_cfg, isa_path, timing_mode=IDEAL_MATRIX_PIPELINE)
    )
    timing_cache_tokens = _timing_cache_schedule(
        dims,
        input_seq,
        output_seq,
        stride,
    )
    packed_q1_timing_required = (
        kv_layout == DENSE_SELECTOR and effective_kv_head_reuse is not False
    )
    if execution_mode == COMPILER_TRACE:
        base_timing_calibrated = True
        base_timing_reason = "compiler_trace_timing_validated"
        packed_q1_timing_validated = True
        packed_q1_timing_reason = "covered_by_full_model_compiler_trace"
    elif execution_mode == LEGACY_AGGREGATE_BANDWIDTH:
        base_timing_calibrated, base_timing_reason = validate_timing_evidence(
            resolved_timing_mode,
            timing_evidence,
        )
        if packed_q1_timing_required:
            packed_q1_timing_validated, packed_q1_timing_reason = (
                validate_packed_q1_timing_contract(
                    packed_q1_timing_contract,
                    timing_mode=resolved_timing_mode,
                    mlen=hw_cfg.MLEN,
                    blen=hw_cfg.BLEN,
                    hlen=hw_cfg.HLEN,
                    query_heads=dims["heads"],
                    kv_heads=dims["kv_heads"],
                    head_dim=dims["head_dim"],
                    batch=batch,
                    cache_tokens=timing_cache_tokens,
                    compiler_root=_COMPILER_ROOT,
                    rtl_root=_RTL_ROOT,
                    latency_library_path=Path(isa_path),
                )
            )
        elif kv_layout == DENSE_SELECTOR:
            packed_q1_timing_validated = True
            packed_q1_timing_reason = "not_required_for_per_head_schedule"
        else:
            packed_q1_timing_validated = False
            packed_q1_timing_reason = "packed_q1_timing_requires_dense_selector"
    else:
        raise ValueError(f"unsupported decode execution mode {execution_mode!r}")
    timing_calibrated = (
        base_timing_calibrated and packed_q1_timing_validated
    )
    timing_reason = (
        base_timing_reason
        if timing_calibrated or packed_q1_timing_validated
        else packed_q1_timing_reason
    )
    timing_evidence_id = (
        timing_evidence.evidence_id
        if timing_calibrated and execution_mode != COMPILER_TRACE
        else None
    )
    active_packed_q1_contract = (
        packed_q1_timing_contract
        if (
            execution_mode != COMPILER_TRACE
            and packed_q1_timing_required
            and packed_q1_timing_validated
        )
        else None
    )
    packed_q1_timing_contract_id = (
        packed_q1_timing_contract.contract_id
        if (
            execution_mode != COMPILER_TRACE
            and packed_q1_timing_required
            and packed_q1_timing_validated
        )
        else None
    )
    if (
        execution_mode == LEGACY_AGGREGATE_BANDWIDTH
        and bool(topology["explicit_topology"])
        and int(topology["tp"]) * int(topology["kvp"]) > 1
    ):
        # The available PackedQ1 artifact is bound to the global head/context
        # geometry. Rank-local timing below is source-derived and exact to the
        # analytic schedule, but it is not a compiler/RTL timing receipt.
        timing_calibrated = False
        timing_reason = "rank_local_body_timing_lacks_matched_compiler_trace"
        timing_evidence_id = None
        packed_q1_timing_validated = False
        packed_q1_timing_reason = (
            "global_packed_q1_trace_cannot_price_rank_local_geometry"
        )
        packed_q1_timing_contract_id = None
        active_packed_q1_contract = None
    # Activations bf16; KV at kv_bits. weight_bits is set per-component in decode_traffic.
    mem_cfg = base_mem.model_copy(update={"weight_bits": prec["ffn_bits"], "activation_bits": ACT_BITS,
                                          "kv_cache_bits": prec["kv_bits"], **(hw_over or {})})
    mem_model = LLMMemoryModel(model_path, mem_cfg, batch_size=batch,
                               input_seq_len=input_seq, output_seq_len=output_seq)
    mem = mem_model.mem
    peak_bw = peak_hbm_bw_bytes(hw_cfg)
    technology = hbm_technology(hbm_gen)
    hbm_pin_rate_gbps = technology.pin_rate_gbps
    active_bw_model = bw_model
    bandwidth_calibration_id = None
    bandwidth_reason = "peak_bandwidth_sensitivity"
    if execution_mode == COMPILER_TRACE:
        active_bw_model = None
        bandwidth_reason = "not_applicable_compiler_trace"
    elif bw_model is not None:
        bandwidth_calibration_id = (
            bw_model.operating_point_calibration_id(
                hbm_gen,
                hbm_pin_rate_gbps,
            )
        )
        if bandwidth_calibration_id is None:
            active_bw_model = None
            bandwidth_reason = "generation_rate_not_calibrated"
        else:
            bandwidth_reason = "emulator_dma_calibrated"

    # HBM holds only weights + KV cache; activations stay on-chip. The footprint
    # is chip-count independent, so resolve the chip count from it.
    ctx = input_seq + output_seq
    physical_weights = weight_ledger(
        dims,
        prec,
        include_lm_head=include_lm_head,
        batch=batch,
        unique_experts=dims.get("moe_unique_experts_per_step"),
        mlen=int(hw_cfg.MLEN),
    )
    # Per-chip capacity honours hw_over: a searched HBM_SIZE (channel count) must
    # gate the fit check, not the TOML default.
    hbm_per_chip = mem_cfg.HBM_SIZE
    probe = build_physical_decode_ledger(
        dims,
        prec,
        hw_cfg,
        context=ctx,
        batch=batch,
        hbm_capacity_bytes=hbm_per_chip,
        runtime_hbm_reserve_bytes=runtime_hbm_reserve_bytes,
        kv_layout=kv_layout,
        include_lm_head=include_lm_head,
    )
    chips = int(topology["chip_count"]) or max(
        1,
        math.ceil(probe.hbm_required_bytes / hbm_per_chip),
    )
    if not bool(topology["explicit_topology"]):
        topology = {
            **topology,
            "tp": chips,
            "kvp": 1,
            "chip_count": chips,
            "link_ports": 0 if chips == 1 else 1,
        }
    else:
        _validate_parallel_model(dims, topology)
    body_weight_layout = None
    slowest_rank_kv = None
    system_kv = None
    kv_partition_provenance = None
    exact_body_layout_supported = (
        is_moe(dims)
        and bool(topology["explicit_topology"])
        and (local_mx_head or not include_lm_head)
    )
    if exact_body_layout_supported:
        unique_experts = dims.get("moe_unique_experts_per_step")
        if unique_experts is None:
            unique_experts = conservative_unique_experts(
                int(dims["num_experts"]),
                int(dims["experts_per_token"]),
                int(batch),
            )
        layout_precision = {
            **prec,
            "head_elem": int(prec.get("head_elem", prec["attn_elem"])),
            "head_bits": float(prec.get("head_bits", prec["attn_bits"])),
        }
        body_weight_layout = build_body_weight_physical_layout(
            dims,
            layout_precision,
            mlen=int(hw_cfg.MLEN),
            tp=int(topology["tp"]),
            kvp=int(topology["kvp"]),
            batch=batch,
            unique_experts=int(unique_experts),
            expert_parallel_mode=str(topology["expert_parallel_mode"]),
            # The current route-summary overlay deliberately excludes per-rank
            # identities/counts, so expert-ID mode remains a conservative
            # projected bound rather than inventing balanced placement.
            active_experts_per_rank=None,
            include_lm_head=local_mx_head,
        )
        slowest_rank_kv, system_kv, kv_partition_provenance = (
            _partitioned_kv_ledgers(
                dims,
                prec,
                context=ctx,
                batch=batch,
                mlen=int(hw_cfg.MLEN),
                kv_layout=kv_layout,
                tp=int(topology["tp"]),
                kvp=int(topology["kvp"]),
            )
        )
    fp_sram_depth = int(getattr(hw_cfg, "FP_SRAM_DEPTH", 512))
    reuse_kv_heads = int(dims["kv_heads"])
    if architecture_knobs_explicit:
        tp = int(topology["tp"])
        if reuse_kv_heads % tp:
            raise ValueError("KV heads must divide evenly across TP ranks")
        reuse_kv_heads //= tp
    reuse_status = kv_head_reuse_status(
        enabled=(
            bool(topology["kv_head_reuse"])
            if architecture_knobs_explicit
            else True
        ),
        mlen=int(hw_cfg.MLEN),
        hlen=int(hw_cfg.HLEN),
        blen=int(hw_cfg.BLEN),
        # The head-broadcast state is private to a TP rank.  Pricing the
        # global model head count here incorrectly rejects legal GQA reuse
        # schedules after the attention tensors have already been sharded.
        kv_heads=reuse_kv_heads,
        fp_sram_depth=fp_sram_depth,
    )
    if architecture_knobs_explicit and not bool(reuse_status["supported"]):
        raise ValueError(
            "KV_HEAD_REUSE rejected: " + str(reuse_status["legality_reason"])
        )
    base_ledger = build_physical_decode_ledger(
        dims,
        prec,
        hw_cfg,
        context=ctx,
        batch=batch,
        hbm_capacity_bytes=hbm_per_chip,
        runtime_hbm_reserve_bytes=runtime_hbm_reserve_bytes,
        kv_layout=kv_layout,
        include_lm_head=include_lm_head,
        tp=int(topology["tp"]),
    )
    if bool(topology["legacy_ideal_parallelism"]):
        ledger = build_physical_decode_ledger(
            dims,
            prec,
            hw_cfg,
            context=ctx,
            batch=batch,
            hbm_capacity_bytes=hbm_per_chip * chips,
            # `runtime_hbm_reserve_bytes` is per chip on every path: the
            # explicit-topology ledger scales it by the chip count inside
            # `_partition_physical_ledger`, so the aggregate legacy ledger
            # must reserve the same total or the two topologies disagree
            # about feasibility for the identical physical system.
            runtime_hbm_reserve_bytes=runtime_hbm_reserve_bytes * chips,
            kv_layout=kv_layout,
            include_lm_head=include_lm_head,
            tp=int(topology["tp"]),
        )
    else:
        ledger = _partition_physical_ledger(
            base_ledger,
            tp=int(topology["tp"]),
            kvp=int(topology["kvp"]),
            hbm_per_chip=hbm_per_chip,
            sram_policy=str(topology["sram_policy"]),
            batch=batch,
            body_weight_layout=body_weight_layout,
            slowest_rank_kv=slowest_rank_kv,
            system_kv=system_kv,
        )
    hbm_capacity = ledger.hbm_capacity_bytes
    resident = ledger.weights.resident
    quantized_resident = (
        ledger.weights.attention
        + ledger.weights.ffn_resident
        + ledger.weights.lm_head_resident
    )
    lm_head_resident = (
        ledger.weights.lm_head_resident
        + ledger.weights.bf16_lm_head_resident
    )
    wf = {
        "embedding": ledger.weights.bf16_embedding.total_aligned,
        "norms": ledger.weights.bf16_norms.total_aligned,
        "lm_head": lm_head_resident.total_aligned,
        "lm_head_element_plane": (
            ledger.weights.lm_head_resident.element_aligned
        ),
        "lm_head_scale_plane": (
            ledger.weights.lm_head_resident.scale_aligned
        ),
        "lm_head_bf16": (
            ledger.weights.bf16_lm_head_resident.total_aligned
        ),
        "attention": ledger.weights.attention.total_aligned,
        "ffn": ledger.weights.ffn_resident.total_aligned,
        "element_plane": quantized_resident.element_aligned,
        "scale_plane": quantized_resident.scale_aligned,
        "bf16": ledger.weights.bf16_resident.total_aligned,
        "total": resident.total_aligned,
    }
    kv_bytes = ledger.kv.total_bytes
    hbm_required = ledger.hbm_required_bytes

    loop = run_decode_loop(perf, mem, dims, prec, input_seq, output_seq, batch, peak_bw,
                           stride, matrix_overfetch_factor(hw_cfg), n_chips=chips,
                           bw_model=active_bw_model, hbm_gen=hbm_gen,
                           hbm_channels=hbm_channels,
                           hbm_pin_rate_gbps=hbm_pin_rate_gbps,
                           kv_layout=kv_layout, ideal_perf=ideal_perf,
                           physical_weights=physical_weights,
                           include_lm_head=include_lm_head,
                           batch_packed_attention=batch_packed_attention,
                           packed_q1_timing_contract=active_packed_q1_contract,
                           tp=int(topology["tp"]),
                           kvp=int(topology["kvp"]),
                           link_ports=int(topology["link_ports"]),
                           link_generation=str(topology["link_generation"]),
                           sram_policy=str(topology["sram_policy"]),
                           legacy_ideal_parallelism=bool(
                               topology["legacy_ideal_parallelism"]
                           ),
                           kv_head_reuse=effective_kv_head_reuse,
                           body_weight_layout=body_weight_layout,
                           expert_parallel_mode=str(
                               topology["expert_parallel_mode"]
                           ),
                           execution_mode=execution_mode,
                           trace_timing_provider=trace_timing_provider,
                           trace_request_factory=trace_request_factory)
    local_output_head = None
    if local_mx_head:
        head_rank_shapes = (
            list(body_weight_layout.provenance["rank_shapes"]["local_head"])
            if body_weight_layout is not None
            else []
        )
        physical_head_batch = (
            math.ceil(int(batch) / int(hw_cfg.BLEN)) * int(hw_cfg.BLEN)
        )
        default_physical_hidden = (
            math.ceil(int(dims["hidden"]) / int(hw_cfg.MLEN))
            * int(hw_cfg.MLEN)
        )
        activation_zero_fill_elements = (
            batch * (default_physical_hidden - int(dims["hidden"]))
            + (physical_head_batch - batch) * default_physical_hidden
        )
        activation_zero_fill_events = math.ceil(
            activation_zero_fill_elements / int(hw_cfg.VLEN)
        )
        activation_zero_fill_cycles = (
            activation_zero_fill_events * perf.instr["V_BASIC"]
        )
        rank_head_schedules: list[dict[str, int]] = []
        source_shapes = head_rank_shapes or [
            {
                "logical_rows": int(dims["vocab"]),
                "physical_rows": (
                    math.ceil(int(dims["vocab"]) / int(hw_cfg.MLEN))
                    * int(hw_cfg.MLEN)
                ),
                "physical_columns": default_physical_hidden,
            }
        ]
        for rank_index, shape in enumerate(source_shapes):
            rank_vocab = int(shape["logical_rows"])
            rank_physical_vocab = int(shape["physical_rows"])
            rank_physical_hidden = int(shape["physical_columns"])
            mask_elements = batch * (rank_physical_vocab - rank_vocab)
            mask_events = math.ceil(mask_elements / int(hw_cfg.VLEN))
            matrix_cycles = perf.lm_head(
                rank_physical_hidden,
                rank_vocab,
                batch,
            )
            selection_cycles = perf.lm_head_streaming_selection(
                rank_vocab,
                batch,
                top_k=20,
            )
            mask_cycles = mask_events * perf.instr["V_BASIC"]
            rank_head_schedules.append(
                {
                    "rank": rank_index,
                    "logical_vocab": rank_vocab,
                    "physical_vocab": rank_physical_vocab,
                    "physical_hidden": rank_physical_hidden,
                    "matrix_cycles": matrix_cycles,
                    "selection_cycles": selection_cycles,
                    "argmax_cycles": perf.lm_head_argmax_diagnostic(
                        rank_vocab,
                        batch,
                    ),
                    "mask_elements": mask_elements,
                    "mask_events": mask_events,
                    "mask_cycles": mask_cycles,
                    "serving_cycles": (
                        matrix_cycles
                        + selection_cycles
                        + activation_zero_fill_cycles
                        + mask_cycles
                    ),
                }
            )
        slowest_head_schedule = max(
            rank_head_schedules,
            key=lambda value: value["serving_cycles"],
        )
        physical_head_vocab = slowest_head_schedule["physical_vocab"]
        physical_head_hidden = slowest_head_schedule["physical_hidden"]
        head_matrix_cycles = slowest_head_schedule["matrix_cycles"]
        head_selection_cycles = slowest_head_schedule["selection_cycles"]
        head_argmax_cycles = slowest_head_schedule["argmax_cycles"]
        padded_vocab_mask_elements_slowest_rank = slowest_head_schedule[
            "mask_elements"
        ]
        padded_vocab_mask_events_slowest_rank = slowest_head_schedule[
            "mask_events"
        ]
        padded_vocab_mask_elements_system = (
            sum(value["mask_elements"] for value in rank_head_schedules)
            * (int(topology["kvp"]) if head_rank_shapes else 1)
        )
        # Vector events cannot be coalesced across independent TP ranks or KVP
        # replicas: each rank rounds its own tail to a complete VLEN issue.
        padded_vocab_mask_events_system = (
            sum(value["mask_events"] for value in rank_head_schedules)
            * (int(topology["kvp"]) if head_rank_shapes else 1)
        )
        head_padding = {
            "zero_fill_elements_per_rank": activation_zero_fill_elements,
            "zero_fill_vector_events_per_rank": activation_zero_fill_events,
            "zero_fill_cycles_per_rank": activation_zero_fill_cycles,
            "padded_vocab_mask_elements_slowest_rank": (
                padded_vocab_mask_elements_slowest_rank
            ),
            "padded_vocab_mask_vector_events_slowest_rank": (
                padded_vocab_mask_events_slowest_rank
            ),
            "padded_vocab_mask_cycles_slowest_rank": (
                padded_vocab_mask_events_slowest_rank
                * perf.instr["V_BASIC"]
            ),
            "padded_vocab_mask_elements_system": (
                padded_vocab_mask_elements_system
            ),
            "padded_vocab_mask_vector_events_system": (
                padded_vocab_mask_events_system
            ),
            "padded_vocab_mask_cycles_system": (
                padded_vocab_mask_events_system * perf.instr["V_BASIC"]
            ),
        }
        compute_divisor = 1 if body_weight_layout is not None else (
            chips
            if bool(topology["legacy_ideal_parallelism"])
            else int(topology["tp"])
        )
        slowest_rank_cycles = (
            (head_matrix_cycles + head_selection_cycles) / compute_divisor
            + int(head_padding["zero_fill_cycles_per_rank"])
            + int(head_padding["padded_vocab_mask_cycles_slowest_rank"])
            / compute_divisor
        )
        head_compute_time_s = cycles_to_seconds(
            slowest_rank_cycles,
            frequency_hz=FREQ_HZ,
            compute_density=compute_density(prec),
            chip_count=1,
        )
        base_head_streamed = (
            body_weight_layout.slowest_rank.lm_head_streamed
            + body_weight_layout.slowest_rank.bf16_lm_head_streamed
            if body_weight_layout is not None
            else physical_weights.lm_head_streamed
            + physical_weights.bf16_lm_head_streamed
        )
        rank_head_read_bytes = (
            float(base_head_streamed.total_aligned)
            / (1 if body_weight_layout is not None else (
                chips
                if bool(topology["legacy_ideal_parallelism"])
                else int(topology["tp"])
            ))
        )
        overfetch = matrix_overfetch_factor(hw_cfg)
        rank_head_hbm_read_bytes = rank_head_read_bytes * overfetch
        if active_bw_model is not None:
            head_transfer_bytes = (
                perf.mlen * perf.mlen * int(prec["head_elem"]) / 8
            )
            head_memory_time_s = active_bw_model.memory_time(
                {"weights_kv": rank_head_hbm_read_bytes},
                hbm_gen,
                hbm_channels,
                transfer_bytes=head_transfer_bytes,
                pin_rate_gbps=hbm_pin_rate_gbps,
            ) / (
                chips
                if bool(topology["legacy_ideal_parallelism"])
                else 1
            )
        else:
            head_memory_time_s = rank_head_hbm_read_bytes / (
                peak_bw
                * (
                    chips
                    if bool(topology["legacy_ideal_parallelism"])
                    else 1
                )
            )
        resident_head = (
            ledger.weights.lm_head_resident
            + ledger.weights.bf16_lm_head_resident
        )
        streamed_head = (
            ledger.weights.lm_head_streamed
            + ledger.weights.bf16_lm_head_streamed
        )
        aggregate_head_hbm_read_bytes = (
            float(streamed_head.total_aligned) * overfetch
        )
        selection_collective = (
            {
                "local_output_selection_bytes": 0.0,
                "local_output_selection_time_s": 0.0,
            }
            if bool(topology["legacy_ideal_parallelism"])
            else collective_cost_per_step(
                dims,
                batch=batch,
                tp=int(topology["tp"]),
                kvp=int(topology["kvp"]),
                link_ports=int(topology["link_ports"]),
                link_generation=str(topology["link_generation"]),
                include_local_output_selection=True,
                local_output_top_k=20,
                expert_parallel_mode=str(topology["expert_parallel_mode"]),
            )
        )
        selection_collective_time_s = float(
            selection_collective["local_output_selection_time_s"]
        )
        selection_collective_bytes = float(
            selection_collective["local_output_selection_bytes"]
        )
        isolated_head_roofline_s = max(
            head_compute_time_s,
            head_memory_time_s,
        )
        isolated_head_time_s = (
            isolated_head_roofline_s + selection_collective_time_s
        )
        local_head_failures = []
        if body_weight_layout is None:
            local_head_failures.append(
                "body_weight_physical_padding_unmodelled"
            )
            if int(topology["tp"]) > 1:
                local_head_failures.append(
                    "local_head_tp_sharded_physical_shape_unmodelled"
                )
        if execution_mode != LEGACY_AGGREGATE_BANDWIDTH:
            local_head_failures.append(
                "compiler_trace_head_stage_breakdown_unavailable"
            )
        if not bool(prec["head_operand_family_supported"]):
            local_head_failures.append(
                "mixed_matrix_family_unsupported_without_trace_evidence"
            )
        numerical_matrix_mlen = int(prec["head_numerical_matrix_mlen"])
        candidate_matrix_mlen = int(hw_cfg.MLEN)
        numerical_matrix_mlen_exact_match = (
            numerical_matrix_mlen == candidate_matrix_mlen
        )
        if not numerical_matrix_mlen_exact_match:
            local_head_failures.append("numerical_matrix_mlen_mismatch")
        if (
            bool(topology["legacy_ideal_parallelism"])
            and int(topology["tp"]) > 1
        ):
            local_head_failures.append(
                "legacy_ideal_parallelism_omits_global_topk_merge"
            )
        if head_rank_shapes:
            physical_head_compute_vocab_system = sum(
                math.ceil(int(shape["logical_rows"]) / int(hw_cfg.BLEN))
                * int(hw_cfg.BLEN)
                for shape in head_rank_shapes
            ) * int(topology["kvp"])
            algorithmic_head_flops = (
                2
                * batch
                * int(dims["hidden"])
                * int(dims["vocab"])
                * int(topology["kvp"])
            )
        else:
            physical_head_compute_vocab_system = physical_head_vocab
            algorithmic_head_flops = (
                2 * batch * int(dims["hidden"]) * int(dims["vocab"])
            )
        physical_head_flops = (
            2
            * physical_head_batch
            * physical_head_hidden
            * physical_head_compute_vocab_system
        )
        local_output_head = {
            "schema_version": "decode-local-mx-head-breakdown/v3",
            "passed": not local_head_failures,
            "failures": local_head_failures,
            "operator": "decode_lm_head",
            "profile_id": str(prec["profile_id"]),
            "numerical_matrix_mlen": numerical_matrix_mlen,
            "candidate_matrix_mlen": candidate_matrix_mlen,
            "numerical_matrix_mlen_exact_match": (
                numerical_matrix_mlen_exact_match
            ),
            "batch_geometry": {
                "active_rows": int(batch),
                "physical_rows": physical_head_batch,
                "zero_padded_rows": physical_head_batch - int(batch),
            },
            "padding_preparation": {
                "schedule": "v_basic_full_width_chunks_before_matrix_and_selection",
                "weight_padding": (
                    "offline_zero_fill_included_in_head_hbm_planes"
                ),
                "activation_zero_fill_elements_per_rank": int(
                    head_padding["zero_fill_elements_per_rank"]
                ),
                "activation_zero_fill_vector_events_per_rank": int(
                    head_padding["zero_fill_vector_events_per_rank"]
                ),
                "activation_zero_fill_cycles_per_rank": int(
                    head_padding["zero_fill_cycles_per_rank"]
                ),
                "padded_vocab_mask": "negative_infinity",
                "slowest_serving_rank": int(
                    slowest_head_schedule["rank"]
                ),
                "padded_vocab_mask_by_tp_rank": [
                    {
                        "rank": int(value["rank"]),
                        "logical_vocab": int(value["logical_vocab"]),
                        "physical_vocab": int(value["physical_vocab"]),
                        "elements": int(value["mask_elements"]),
                        "vector_events": int(value["mask_events"]),
                        "cycles": int(value["mask_cycles"]),
                    }
                    for value in rank_head_schedules
                ],
                "padded_vocab_mask_elements_slowest_rank": int(
                    head_padding["padded_vocab_mask_elements_slowest_rank"]
                ),
                "padded_vocab_mask_vector_events_slowest_rank": int(
                    head_padding[
                        "padded_vocab_mask_vector_events_slowest_rank"
                    ]
                ),
                "padded_vocab_mask_cycles_slowest_rank": int(
                    head_padding["padded_vocab_mask_cycles_slowest_rank"]
                ),
                "padded_vocab_mask_elements_system": int(
                    head_padding["padded_vocab_mask_elements_system"]
                ),
                "padded_vocab_mask_vector_events_system": int(
                    head_padding["padded_vocab_mask_vector_events_system"]
                ),
                "padded_vocab_mask_cycles_system": int(
                    head_padding["padded_vocab_mask_cycles_system"]
                ),
                "analytic_cycles_charged": True,
                "compiler_lowered": False,
            },
            "precision_policy": "profile_w_a_bf16_logits",
            "weight_format": str(prec["head_label"]),
            "activation_format": str(prec["head_activation_label"]),
            "weight_element_bits": int(prec["head_elem"]),
            "weight_effective_bits": float(prec["head_bits"]),
            "activation_element_bits": int(prec["head_activation_elem"]),
            "activation_effective_bits": float(
                prec["head_activation_bits"]
            ),
            "block_size": int(prec["block_size"]),
            "scale_format": "E8M0",
            "accumulator_dtype": "signed_fixed16.16",
            "operand_family_binding": str(
                prec["head_operand_family_binding"]
            ),
            "numerical_oracle_rule": str(
                prec["head_numerical_oracle_rule"]
            ),
            "partial_conversion_rule": str(
                prec["head_partial_conversion_rule"]
            ),
            "operand_family_deployment_supported": bool(
                prec["head_operand_family_supported"]
            ),
            "hardware_bit_parity_verified": bool(
                prec["head_hardware_bit_parity_verified"]
            ),
            "accumulation_chain": list(prec["head_accumulation_chain"]),
            "matrix_numeric_format": str(prec["head_vector_format"]),
            "matrix_storage_format": str(
                prec["head_matrix_storage_format"]
            ),
            "logit_container_format": str(
                prec["head_logit_container_format"]
            ),
            "bf16_container_precision_recovery": bool(
                prec["head_bf16_container_precision_recovery"]
            ),
            "selection": {
                "serving_policy": "streaming_topk20_topp0.95_minp0",
                "diagnostic_policy": "argmax_lowest_token_id_on_tie",
                "full_vocab_logits_materialized": False,
                "top_k": 20,
                "top_p": 0.95,
                "min_p": 0.0,
                "logit_tile_bytes_per_chip": (
                    ledger.sram.output_head_logit_tile_bytes
                ),
                "state_bytes_per_chip": (
                    ledger.sram.output_head_selection_state_bytes
                ),
                "workspace_bytes_per_chip": (
                    ledger.sram.output_head_workspace_bytes
                ),
                "distributed_merge": {
                    "mode": "tp_topk20_gather_owner_then_u32_token_broadcast",
                    "candidate_pair_bytes": 8,
                    "candidate_count_per_rank_per_sequence": 20,
                    "aggregate_link_bytes_per_batch_step": (
                        selection_collective_bytes
                    ),
                    "slowest_path_serialization_time_s": (
                        selection_collective_time_s
                    ),
                    "charged_to_system_collective_energy": True,
                },
            },
            "cycles_per_batch_step": {
                "matrix_slowest_rank": float(
                    head_matrix_cycles / compute_divisor
                ),
                "selection_slowest_rank": float(
                    head_selection_cycles / compute_divisor
                ),
                "argmax_diagnostic_slowest_rank": float(
                    head_argmax_cycles / compute_divisor
                ),
                "activation_zero_fill_per_rank": int(
                    head_padding["zero_fill_cycles_per_rank"]
                ),
                "padded_vocab_mask_slowest_rank": int(
                    head_padding["padded_vocab_mask_cycles_slowest_rank"]
                ),
                "padded_vocab_mask_system": int(
                    head_padding["padded_vocab_mask_cycles_system"]
                ),
                "serving_slowest_rank": float(slowest_rank_cycles),
            },
            "time_s_per_batch_step": {
                "compute_slowest_rank": head_compute_time_s,
                "hbm_slowest_rank": head_memory_time_s,
                "local_roofline": isolated_head_roofline_s,
                "distributed_selection_collective": (
                    selection_collective_time_s
                ),
                "isolated_head_with_collective": isolated_head_time_s,
                "scope": (
                    "local_roofline_plus_dependency_bound_tp_selection_merge"
                ),
            },
            "hbm_read_bytes_per_batch_step": {
                "aggregate_system_before_overfetch": float(
                    streamed_head.total_aligned
                ),
                "aggregate_system_after_overfetch": (
                    aggregate_head_hbm_read_bytes
                ),
                "slowest_rank_after_overfetch": rank_head_hbm_read_bytes,
            },
            "resident_bytes": {
                "aggregate_system": int(resident_head.total_aligned),
                "element_plane": int(
                    ledger.weights.lm_head_resident.element_aligned
                ),
                "scale_plane": int(
                    ledger.weights.lm_head_resident.scale_aligned
                ),
                "bf16_plane": int(
                    ledger.weights.bf16_lm_head_resident.total_aligned
                ),
            },
            "algorithmic_flops_per_batch_step": (
                algorithmic_head_flops
            ),
            "flops_per_batch_step": (
                physical_head_flops
            ),
            "padding_flops_per_batch_step": (
                physical_head_flops - algorithmic_head_flops
            ),
            "fractions": {
                "isolated_roofline_time_over_decoder_tpot": (
                    isolated_head_time_s / float(loop["tpot"])
                ),
                "hbm_read_over_decoder_traffic": (
                    aggregate_head_hbm_read_bytes
                    / float(loop["avg_bytes_per_batch_step"])
                ),
                "resident_over_decoder_capacity": (
                    resident_head.total_aligned
                    / int(ledger.hbm_required_bytes)
                ),
            },
            "topology": {
                "tp": int(topology["tp"]),
                "kvp": int(topology["kvp"]),
                "chip_count": chips,
            },
        }
    loop["local_output_head"] = local_output_head
    if execution_mode == COMPILER_TRACE:
        trace_evidence = loop.get("compiler_trace_timing")
        if not isinstance(trace_evidence, Mapping):
            raise RuntimeError("compiler trace timing omitted its provenance")
        timing_reason = "compiler_trace_timing_validated"
        timing_evidence_id = (
            "compiler-trace-timing-" + canonical_sha256(trace_evidence)
        )
    option_area = architecture_option_area_mm2(
        mlen=int(hw_cfg.MLEN),
        hlen=int(hw_cfg.HLEN),
        # Reuse control is private to the rank-local GQA head group, just like
        # its legality and KV traffic schedule above.
        kv_heads=reuse_kv_heads,
        kv_head_reuse=(
            bool(topology["kv_head_reuse"])
            if architecture_knobs_explicit
            else False
        ),
        drain_overlapped=(
            bool(topology["drain_overlapped"])
            if architecture_knobs_explicit
            else False
        ),
    )
    reuse_status = {
        **reuse_status,
        "requested": (
            bool(topology["kv_head_reuse"])
            if architecture_knobs_explicit
            else None
        ),
        "legacy_implicit_default": not architecture_knobs_explicit,
        "legality_enforced": architecture_knobs_explicit,
    }
    drain_status = {
        "requested": (
            bool(topology["drain_overlapped"])
            if architecture_knobs_explicit
            else None
        ),
        "enabled": (
            bool(topology["drain_overlapped"])
            if architecture_knobs_explicit
            else resolved_timing_mode == DRAIN_OVERLAPPED
        ),
        "timing_mode": resolved_timing_mode,
        "timing_calibrated": timing_calibrated,
        "timing_evidence_id": timing_evidence_id,
        "timing_reason": timing_reason,
        "second_accumulator_bank_bytes_per_chip": (
            DRAIN_ACCUMULATOR_BYTES_PER_CHIP
            if (
                architecture_knobs_explicit
                and bool(topology["drain_overlapped"])
            )
            else 0
        ),
        "evidence_tier": (
            "not_applicable"
            if not (
                bool(topology["drain_overlapped"])
                if architecture_knobs_explicit
                else resolved_timing_mode == DRAIN_OVERLAPPED
            )
            else (
                (
                    "compiler_trace_request_calibrated"
                    if execution_mode == COMPILER_TRACE
                    else (
                        "matched_analytic_emulator_timing"
                        if timing_evidence is not None
                        and timing_evidence.evidence_tier == EMULATOR_EVIDENCE_TIER
                        else "matched_emulator_rtl_timing"
                    )
                )
                if timing_calibrated
                else "analytic_codesign_unrankable"
            )
        ),
    }
    architecture_options = {
        "schema": "plena-decode-architecture-options",
        "explicit": architecture_knobs_explicit,
        "expert_parallel_mode": str(topology["expert_parallel_mode"]),
        "kv_head_reuse": reuse_status,
        "drain_overlapped": drain_status,
        "area": option_area,
    }
    moe_workload = (
        _moe_workload_accounting(perf, dims, batch)
        if is_moe(dims)
        else None
    )
    if moe_workload is not None and body_weight_layout is not None:
        moe_provenance = dict(moe_workload["provenance"])
        expert_mode = str(topology["expert_parallel_mode"])
        route_exact = bool(
            body_weight_layout.provenance["expert_route_assignment_exact"]
        )
        moe_provenance.update(
            {
                "expert_parallel_mode": expert_mode,
                "per_rank_route_assignment_exact": route_exact,
                "expert_id_timing_semantics": (
                    "not_applicable"
                    if expert_mode == EXPERT_TENSOR_PARALLEL
                    else "conservative_slowest_rank_all_routes_bound"
                ),
                "expert_id_route_filter_transport": (
                    "not_applicable"
                    if expert_mode == EXPERT_TENSOR_PARALLEL
                    else (
                        "replicated_hidden_local_route_filter_then_"
                        "output_allreduce"
                    )
                ),
                "expert_id_route_filter_collective_bytes": 0,
                "expert_id_output_collective_charged": (
                    expert_mode == EXPERT_ID_PARALLEL
                    and int(topology["tp"]) > 1
                ),
            }
        )
        moe_workload = {
            **moe_workload,
            "provenance": moe_provenance,
        }
    body_physical_layout = None
    if body_weight_layout is not None:
        body_physical_layout = body_weight_layout.to_dict()
        body_provenance = dict(body_physical_layout["provenance"])
        body_blockers = list(body_provenance.get("blockers", []))
        body_provenance.update(
            {
                "analytic_layout_valid": True,
                "analytic_timing_valid": True,
                "timing_schema": BODY_PARALLEL_TIMING_SCHEMA,
                "timing_partition": (
                    "replicated_control_vector;rank_local_qkv_o_expert_"
                    "intermediate_head;kvp_integer_context_slowest_rank"
                ),
                "global_cycle_division_used": False,
                "capacity_system_aggregate_checked": True,
                "capacity_slowest_rank_checked": True,
                "expert_collective_accounting": {
                    "route_filter_mapping": body_provenance[
                        "expert_routing_mapping"
                    ],
                    "route_filter_bytes": float(
                        loop["collective_breakdown_per_batch_step"].get(
                            "expert_routing_bytes", 0.0
                        )
                    ),
                    "route_filter_time_s": float(
                        loop["collective_breakdown_per_batch_step"].get(
                            "expert_routing_time_s", 0.0
                        )
                    ),
                    "output_allreduce_slowest_rank_bytes": float(
                        loop["collective_breakdown_per_batch_step"].get(
                            "expert_output_collective_slowest_rank_bytes",
                            0.0,
                        )
                    ),
                    "output_allreduce_system_bytes": float(
                        loop["collective_breakdown_per_batch_step"].get(
                            "expert_output_collective_system_bytes", 0.0
                        )
                    ),
                    "output_allreduce_time_s": float(
                        loop["collective_breakdown_per_batch_step"].get(
                            "expert_output_collective_time_s", 0.0
                        )
                    ),
                    "output_allreduce_already_in_system_collective_total": True,
                },
                "compiler_timing_parity_verified": False,
                "rtl_timing_parity_verified": False,
                "publication_rankable": False,
                "selection_eligible": False,
                "blockers": body_blockers,
            }
        )
        body_physical_layout["provenance"] = body_provenance
        body_physical_layout["kv_partition"] = dict(
            kv_partition_provenance or {}
        )
        body_physical_layout["capacity"] = {
            "system_required_bytes": int(ledger.hbm_required_bytes),
            "system_capacity_bytes": int(ledger.hbm_capacity_bytes),
            "slowest_rank_required_bytes": int(
                ledger.slowest_rank_hbm_required_bytes or 0
            ),
            "per_chip_capacity_bytes": int(
                ledger.per_chip_hbm_capacity_bytes or hbm_per_chip
            ),
            "system_feasible": (
                ledger.hbm_required_bytes <= ledger.hbm_capacity_bytes
            ),
            "slowest_rank_feasible": (
                int(ledger.slowest_rank_hbm_required_bytes or 0)
                <= int(ledger.per_chip_hbm_capacity_bytes or hbm_per_chip)
            ),
            "overall_feasible": bool(ledger.fits_hbm),
        }
    traffic_per_token = loop["traffic_breakdown_per_generated_token"]
    kv_read_bytes_per_token = float(
        traffic_per_token["kv_element_read_bytes"]
        + traffic_per_token["kv_scale_read_bytes"]
    )
    kv_write_bytes_per_token = float(
        traffic_per_token["kv_element_write_bytes"]
        + traffic_per_token["kv_scale_write_bytes"]
    )
    capacity_throughput_chain = {
        "schema": "plena-kv-capacity-throughput-chain",
        "kv_storage_bytes_per_active_sequence": int(
            ledger.kv.per_batch_bytes
        ),
        "kv_storage_bytes_per_sequence_context_token": (
            float(ledger.kv.per_batch_bytes) / ctx
        ),
        "kv_read_bytes_per_generated_token": kv_read_bytes_per_token,
        "kv_write_bytes_per_generated_token": kv_write_bytes_per_token,
        "max_feasible_batch": int(ledger.max_runtime_batch),
        "evaluated_batch": int(batch),
        "capacity_binding": bool(batch == ledger.max_runtime_batch),
        "runtime_feasible": bool(ledger.fits_runtime),
        "evaluated_throughput_tokens_per_second": (
            float(loop["tps"]) if ledger.fits_runtime else None
        ),
        "throughput_semantics": (
            "measured_at_evaluated_batch_no_capacity_extrapolation"
        ),
        "byte_unit": "physical_bytes",
        "batch_unit": "active_sequences",
    }
    loop.update(hbm_required=hbm_required, fits_in_hbm=ledger.fits_hbm,
                hbm_capacity=hbm_capacity, hbm_per_chip=hbm_per_chip, n_chips=chips,
                weight_footprint=wf, kv_footprint=kv_bytes,
                physical_ledger=ledger,
                runtime_hbm_reserve_bytes=ledger.runtime_hbm_reserve_bytes,
                fits_onchip_sram=ledger.sram.fits,
                fits_runtime=ledger.fits_runtime,
                max_resident_batch=ledger.max_resident_batch,
                max_synchronous_batch=ledger.sram.max_synchronous_batch,
                max_runtime_batch=ledger.max_runtime_batch,
                mem=mem, perf=perf, ideal_perf=ideal_perf, peak_bw=peak_bw, dims=dims,
                bw_model=active_bw_model, hbm_gen=hbm_gen,
                hbm_channels=hbm_channels,
                hbm_pin_rate_gbps=hbm_pin_rate_gbps,
                kv_layout=kv_layout, kv_layout_id=ledger.kv.layout_id,
                timing_mode=resolved_timing_mode,
                timing_calibrated=timing_calibrated,
                timing_reason=timing_reason,
                timing_evidence_id=timing_evidence_id,
                timing_evidence_tier=(
                    timing_evidence.evidence_tier
                    if timing_evidence is not None
                    and execution_mode != COMPILER_TRACE
                    else None
                ),
                packed_q1_timing_validated=packed_q1_timing_validated,
                packed_q1_timing_reason=packed_q1_timing_reason,
                packed_q1_timing_contract_id=packed_q1_timing_contract_id,
                packed_q1_timing_cache_tokens=timing_cache_tokens,
                packed_q1_timing_contract=active_packed_q1_contract,
                bandwidth_calibration_id=bandwidth_calibration_id,
                bandwidth_reason=bandwidth_reason,
                output_head_location=output_head_location,
                parallelism=dict(topology),
                architecture_options=architecture_options,
                moe_workload=moe_workload,
                body_physical_layout=body_physical_layout,
                capacity_throughput_chain=capacity_throughput_chain)
    option_logic_area = float(
        dict(option_area["breakdown_mm2_per_chip"]).get(
            "KVHeadReuseControl",
            0.0,
        )
    )
    loop["power"] = decode_power(
        technology,
        # decode_power takes one chip's capacity and traffic, then applies the
        # chip count. The loop rates and hbm_capacity above are system totals.
        capacity_bytes=hbm_per_chip,
        read_bytes_per_second=loop["read_bytes_per_second"] / chips,
        write_bytes_per_second=loop["write_bytes_per_second"] / chips,
        multipliers=hw_cfg.MLEN * hw_cfg.BLEN,
        clock_hz=FREQ_HZ,
        mac_bits=prec["m_bits"],
        array_active_fraction=loop["array_active_fraction"],
        tokens_per_second=loop["tps"],
        chip_count=chips,
        sram_read_bytes_per_second=(
            loop["read_bytes_per_second"] / chips
            if bool(topology["explicit_topology"])
            else 0.0
        ),
        sram_write_bytes_per_second=(
            loop["write_bytes_per_second"] / chips
            if bool(topology["explicit_topology"])
            else 0.0
        ),
        logic_area_mm2=(
            area_mm2(hw_cfg) + option_logic_area
            if bool(topology["explicit_topology"])
            else 0.0
        ),
        link_bytes_per_second=loop["link_bytes_per_second"],
        link_generation=str(topology["link_generation"]),
        token_latency_s=loop["tpot"],
    )
    return loop


def max_batch_capacity(result, batch: int) -> int:
    """Largest runtime batch after HBM reserve and SRAM feasibility."""
    if "max_runtime_batch" in result:
        return int(result["max_runtime_batch"])
    kv_per_batch = result["kv_footprint"] / max(batch, 1)
    return int((result["hbm_capacity"] - result["weight_footprint"]["total"]) // max(kv_per_batch, 1))


def decode_bound_label(result, *, short: bool = False) -> str:
    """Classify memory, serialization, and arithmetic limits separately."""

    if (
        not result.get("timing_calibrated", False)
        or (
            result.get("execution_mode") != COMPILER_TRACE
            and result.get("bandwidth_calibration_id") is None
        )
    ):
        return "sens" if short else "unavailable"
    if result["frac_mem_bound"] >= 0.5:
        return "mem" if short else "memory"
    if result.get("frac_serialization_bound", 0.0) >= 0.5:
        return "ser" if short else "serialization"
    return "cmp" if short else "compute"


def classical_roofline_bound_label(
    result,
    *,
    short: bool = False,
) -> str:
    """Classify physical memory time against the theoretical compute ceiling.

    Both sides of this comparison are analytic: bytes divided by the memory
    bandwidth in force, against the array's peak arithmetic rate. It therefore
    needs no emulator calibration, and the bandwidth basis is reported alongside
    it so a peak-bandwidth reading is not mistaken for a measured one.
    """

    if result["frac_classical_mem_bound"] >= 0.5:
        return "mem" if short else "memory"
    return "cmp" if short else "compute"


def architecture_issue_bound_label(
    result,
    *,
    short: bool = False,
) -> str:
    """Classify physical memory time against ideal matrix issue timing."""

    if (
        not result.get("timing_calibrated", False)
        or (
            result.get("execution_mode") != COMPILER_TRACE
            and result.get("bandwidth_calibration_id") is None
        )
    ):
        return "sens" if short else "unavailable"
    fraction = result.get(
        "frac_architecture_issue_mem_bound",
        result["frac_algorithmic_mem_bound"],
    )
    if fraction >= 0.5:
        return "mem" if short else "memory"
    return "cmp" if short else "compute"


def decode_crossover_point(
    result: dict,
    *,
    context: int,
    batch: int,
) -> DecodeCrossoverPoint:
    """Convert one fixed-context evaluation into a crossover record."""

    return DecodeCrossoverPoint(
        context=context,
        batch=batch,
        peak_compute_seconds=float(result["avg_peak_compute_seconds"]),
        ideal_compute_seconds=float(result["avg_ideal_compute_seconds"]),
        realized_compute_seconds=float(
            result["avg_realized_compute_seconds"]
        ),
        memory_seconds=float(result["avg_memory_seconds"]),
        physical_bytes_per_batch_step=float(
            result["avg_bytes_per_batch_step"]
        ),
        capacity_required_bytes=int(result["hbm_required"]),
        capacity_available_bytes=int(result["hbm_capacity"]),
        timing_mode=str(result["timing_mode"]),
        timing_calibrated=bool(result["timing_calibrated"]),
        timing_evidence_id=result.get("timing_evidence_id"),
        packed_q1_timing_contract_id=result.get(
            "packed_q1_timing_contract_id"
        ),
        bandwidth_calibration_id=result.get("bandwidth_calibration_id"),
        step_composition=str(result["step_composition"]),
    )


def _fmt_bytes(n):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.3f} {unit}"
    return f"{n:.0f} B"


def _decode_utilization(hw_cfg, d: dict, kv: int, batch: int) -> tuple[float, float]:
    """(attention, FFN) matrix-array utilisation for the decode step, a display
    metric (attainable / theoretical ops). Attention blends full and
    sliding-window layers (windowed layers share the projection but shrink the
    flash part). FFN is the per-GEMM fill, same for a dense FFN or one MoE expert."""
    plena = PLENAUtilization({"MLEN": hw_cfg.MLEN, "BLEN": hw_cfg.BLEN, "VLEN": hw_cfg.VLEN})
    h, ah, kvh, hd = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"]
    n_full, n_slide, window = _attn_split(d)
    proj_a, proj_t = plena.projection_utilization(h, ah, kvh, hd, 1, batch, "decode")   # kv-independent
    fa_a, fa_t = plena.flash_attention_utilization(ah, kvh, hd, 1, kv, batch, "decode")
    att_a, att_t = (proj_a + fa_a) * n_full, (proj_t + fa_t) * n_full
    if n_slide:
        s_a, s_t = plena.flash_attention_utilization(ah, kvh, hd, 1, min(kv, window), batch, "decode")
        att_a += (proj_a + s_a) * n_slide
        att_t += (proj_t + s_t) * n_slide
    ffn_a, ffn_t = plena.ffn_utilization(h, d["inter"], 1, batch, "decode")
    return (att_a / att_t if att_t else 0.0), (ffn_a / ffn_t if ffn_t else 0.0)


def print_report(args, dims, hw_cfg, prec, result):
    bar = "=" * 78
    n_chips = result["n_chips"]
    density = compute_density(prec)
    include_lm_head = decoder_owns_output_head(result["output_head_location"])
    metric_scope = (
        "decode plus local BF16 head sensitivity"
        if include_lm_head
        else "decode body only; remote BF16 service excluded"
    )
    peak_compute = 2 * hw_cfg.MLEN * hw_cfg.BLEN * FREQ_HZ * density   # per chip, at M-bit MACs
    peak_bw = result["peak_bw"]                                  # per chip
    peak_compute_sys, peak_bw_sys = peak_compute * n_chips, peak_bw * n_chips   # aggregate over the group
    moe_str = f", MoE {dims['experts_per_token']}/{dims['num_experts']} experts" if is_moe(dims) else ""
    sw_str = (f", sliding {dims['n_sliding']}/{dims['layers']} layers @ w{dims['sliding_window']}"
              if dims["n_sliding"] else "")
    print(bar)
    print("  DECODE-CHIP REPORT — Disaggregated Serving on PLENA")
    print(bar)
    print(f"  Model:     {args.model}  (hidden={dims['hidden']}, layers={dims['layers']}, "
          f"heads={dims['heads']}/{dims['kv_heads']}KV, head_dim={dims['head_dim']}, inter={dims['inter']}{moe_str}{sw_str})")
    print(f"  Workload:  batch={args.batch}  input_seq={args.input_seq} (handed-off KV)  "
          f"output_seq={args.output_seq}")
    print(f"  Precision: attnW:{prec['attn_label']} ffnW:{prec['ffn_label']} KV:{prec['kv_label']} "
          f"(block {args.block})  ->  {prec['attn_bits']:.3f}/{prec['ffn_bits']:.3f}/{prec['kv_bits']:.3f} eff bits")
    print(f"             activations: bf16 in on-chip SRAM, never in HBM (computed low-precision)   [prefill: separate BF16]")
    print(f"             KV layout: {result['kv_layout']}")
    print(f"             timing: {result['timing_mode']} "
          f"({'calibrated' if result['timing_calibrated'] else result['timing_reason']})")
    print(f"             output head: {result['output_head_location']}")
    print(f"             metric scope: {metric_scope}")
    print(f"             HBM timing: {result['bandwidth_reason']}")
    if result["timing_evidence_id"] is not None:
        print(f"             timing evidence: {result['timing_evidence_id']}")
    if result["packed_q1_timing_contract_id"] is not None:
        print(
            "             PackedKV q1 timing: "
            f"{result['packed_q1_timing_contract_id']}"
        )
    else:
        print(
            "             PackedKV q1 timing: "
            f"{result['packed_q1_timing_reason']}"
        )
    # Two-level scheme: HBM streams the N-bit operands above; the array multiplies at M-bit.
    print(f"             compute MACs: M={prec['m_bits']}-bit -> iso-area density x{density:.2f} "
          f"vs the {REF_MAC_BITS}-bit reference array (k={prec['density_exp']:.1f})")
    hbm_src = (f"  [{args.hbm_gen} x {args.hbm_channels} ch]" if args.hbm_gen else "")
    print(f"  Hardware:  MLEN={hw_cfg.MLEN} BLEN={hw_cfg.BLEN} VLEN={hw_cfg.VLEN} HLEN={hw_cfg.HLEN}  "
          f"clock={FREQ_HZ/1e9:.0f} GHz")
    print(f"             peak compute = 2*MLEN*BLEN*clock x density = {peak_compute/1e12:.2f} TFLOP/s   "
          f"peak HBM BW = {peak_bw/1e9:.0f} GB/s{hbm_src}")
    mac_note = (f" -> {int(area_multipliers(hw_cfg) * density):,} M-bit MACs at iso-area"
                if density != 1.0 else "")
    print(f"             matrix array = {area_multipliers(hw_cfg):,} multipliers (~{area_mm2(hw_cfg):.3f} mm^2){mac_note}")
    if n_chips > 1:   # model exceeds one HBM stack -> tensor-parallel over UALink
        print(f"             system: {n_chips} chips (ideal tensor-parallel, UALink) -> aggregate "
              f"{peak_compute_sys/1e12:.2f} TFLOP/s, {peak_bw_sys/1e9:.0f} GB/s, {result['hbm_capacity']/1e9:.0f} GB")
    # HBM feeds MLEN operands/cycle, so the widest of (attnW, ffnW, KV) caps how wide MLEN can be.
    print(f"             bandwidth bound: MLEN <= HBM_WIDTH / max(attnW,ffnW,KV) = "
          f"{hw_cfg.HBM_WIDTH}/{stream_bits(prec)} = {mlen_bandwidth_cap(hw_cfg, prec)}")

    # TTFT is the prefill chip's job; the decode chip's first step makes token #2.
    print("\n[1] LATENCY  (TTFT from prefill)")
    print(f"      First decode step (kv={args.input_seq}):  {result['first_step']*1e3:.3f} ms")
    print(f"      TPOT ({metric_scope}):  {result['tpot']*1e3:.3f} ms")
    print(f"      Total generation ({args.output_seq} tok):    {result['total_time']*1e3:.2f} ms")

    avg_kv = args.input_seq + args.output_seq // 2
    print("\n[2] PERFORMANCE")
    print(f"      TPS (batch*output / total):      {result['tps']:.1f} tokens/s")
    print(f"      Per-stream rate (1 / TPOT):      {1.0/result['tpot']:.1f} tokens/s")
    print(f"      Achieved compute:                "
          f"{decode_step_flops(dims, avg_kv, args.batch, include_lm_head=include_lm_head) / result['tpot'] / 1e12:.2f} TFLOP/s")

    wf, kv_bytes = result["weight_footprint"], result["kv_footprint"]
    max_batch = max_batch_capacity(result, args.batch)
    ffn_name = "experts" if is_moe(dims) else "ffn"
    bf16_roles = "emb/norms/lm_head" if include_lm_head else "emb/norms"
    print("\n[3] MEMORY  (HBM = weights + KV; activations stay on-chip)")
    print(f"      Weights (HBM):        {_fmt_bytes(wf['total'])}  "
          f"(attn {_fmt_bytes(wf['attention'])} @ {prec['attn_bits']:.2f}b, "
          f"{ffn_name} {_fmt_bytes(wf['ffn'])} @ {prec['ffn_bits']:.2f}b, {bf16_roles} @ bf16)")
    kv_note = (f", {dims['n_sliding']} sliding layers capped @ {dims['sliding_window']}"
               if dims["n_sliding"] else "")
    print(f"      KV cache (HBM):       {_fmt_bytes(kv_bytes)}  "
          f"(context={args.input_seq+args.output_seq}, batch={args.batch}{kv_note})")
    cap_note = f" ({n_chips} chips)" if n_chips > 1 else ""
    print(f"      HBM used / capacity:  {_fmt_bytes(result['hbm_required'])} / "
          f"{_fmt_bytes(result['hbm_capacity'])}{cap_note}  ->  {'FITS' if result['fits_in_hbm'] else 'EXCEEDS'} "
          f"({result['hbm_required']/result['hbm_capacity']*100:.1f}%)")
    print(f"      Activations on-chip:  {_fmt_bytes(onchip_activation_bytes(dims, args.batch))}")
    print(f"      Max batch (Capacity bound): {max_batch}  (KV grows with batch x context)")
    print(f"      HBM bytes / decode step: {_fmt_bytes(result['avg_bytes_per_batch_step'])}  "
          f"({_fmt_bytes(result['avg_bytes_per_generated_token'])}/generated token x "
          f"{args.batch} batch)")

    attn_util, ffn_util = _decode_utilization(hw_cfg, dims, avg_kv, args.batch)
    perf = result["perf"]
    compute_time = cycles_to_seconds(
        decode_token_cycles(
            perf,
            dims,
            avg_kv,
            args.batch,
            include_lm_head=include_lm_head,
            kv_layout=result["kv_layout"],
            packed_q1_timing_contract=result[
                "packed_q1_timing_contract"
            ],
            batch_packed_attention=result["batch_packed_attention"],
        ),
        frequency_hz=FREQ_HZ,
        compute_density=density,
        chip_count=n_chips,
    )
    tr = decode_traffic(
        result["mem"],
        dims,
        avg_kv,
        args.batch,
        prec,
        mlen=hw_cfg.MLEN,
        kv_layout=result["kv_layout"],
        include_lm_head=include_lm_head,
    )
    bytes_tok = tr.read_bytes * matrix_overfetch_factor(hw_cfg) + tr.write_bytes
    achieved_bw = bytes_tok / max(compute_time, bytes_tok / peak_bw_sys)
    print("\n[4] UTILISATION  (@ avg context)")
    print(f"      Matrix array, attention:  {attn_util*100:.1f}% of peak")
    print(f"      Matrix array, FFN:        {ffn_util*100:.1f}% of peak")
    print(f"      HBM bandwidth:            {achieved_bw/1e9:.1f} / {peak_bw_sys/1e9:.0f} GB/s "
          f"({achieved_bw/peak_bw_sys*100:.1f}% of peak)")

    ridge = peak_compute_sys / peak_bw_sys                    # == per-chip ridge (n cancels)
    flops = decode_step_flops(
        dims,
        avg_kv,
        args.batch,
        include_lm_head=include_lm_head,
    )
    ai = flops / bytes_tok
    peak_compute_time = flops / peak_compute_sys             # roofline compute ceiling (aggregate)
    ideal_issue_time = cycles_to_seconds(
        decode_token_cycles(
            result["ideal_perf"],
            dims,
            avg_kv,
            args.batch,
            include_lm_head=include_lm_head,
            kv_layout=result["kv_layout"],
            packed_q1_timing_contract=result[
                "packed_q1_timing_contract"
            ],
            batch_packed_attention=result["batch_packed_attention"],
        ),
        frequency_hz=FREQ_HZ,
        compute_density=density,
        chip_count=n_chips,
    )
    # Price memory the same way the decode loop did: calibrated effective
    # bandwidth when --bw-model calibrated, else aggregate peak.
    if result.get("bw_model") is not None:
        wt_transfer = hw_cfg.MLEN * hw_cfg.MLEN * max(prec["attn_elem"], prec["ffn_elem"]) / 8
        memory_time = result["bw_model"].memory_time(
            {"weights_kv": tr.read_bytes * matrix_overfetch_factor(hw_cfg),
             "writeback": tr.write_bytes},
            result["hbm_gen"], result["hbm_channels"],
            transfer_bytes=wt_transfer,
            pin_rate_gbps=result["hbm_pin_rate_gbps"]) / result["n_chips"]
        mem_label = "bytes / calibrated effective BW (size-aware)"
    else:
        memory_time = bytes_tok / peak_bw_sys
        mem_label = "bytes / peak HBM BW"
    classical_mem_bound = memory_time >= peak_compute_time
    architecture_issue_mem_bound = memory_time >= ideal_issue_time
    mem_bound = memory_time >= compute_time
    serialization_bound = (
        not mem_bound
        and architecture_issue_mem_bound
        and result["timing_mode"] == RTL_SERIALIZED
    )
    compute_util_pct = peak_compute_time / compute_time * 100
    print(f"\n[5] ROOFLINE  (decode step @ avg context kv={avg_kv})")
    print(f"      Arithmetic intensity:  {ai:.2f} FLOP/byte   (ridge {ridge:.1f} FLOP/byte)")
    print(f"      memory  time / batch step:  {memory_time*1e3:.3f} ms   ({mem_label})")
    print(f"      compute time / batch step:  {peak_compute_time*1e3:.3f} ms theoretical peak")
    print(f"                             {ideal_issue_time*1e3:.3f} ms ideal matrix issue")
    timing_state = "calibrated" if result["timing_calibrated"] else result["timing_reason"]
    print(f"                             {compute_time*1e3:.3f} ms achieved "
          f"({result['timing_mode']}, {timing_state}; {compute_util_pct:.0f}% of the compute ceiling realised)")
    # The classical roofline is analytic on both sides, so it is always
    # reported; the suffix records whether the bandwidth was measured or peak.
    bandwidth_calibrated = result.get("bandwidth_calibration_id") is not None
    bandwidth_basis = "measured BW" if bandwidth_calibrated else "peak BW"
    classical_label = (
        f"{'MEMORY' if classical_mem_bound else 'COMPUTE'} ({bandwidth_basis})"
    )
    # The issue-level views depend on trace-calibrated instruction timing, so
    # they stay labelled as modelled until that evidence exists.
    timing_basis = "trace-calibrated" if result["timing_calibrated"] else "modelled"
    architecture_label = (
        f"{'MEMORY' if architecture_issue_mem_bound else 'COMPUTE'} "
        f"({timing_basis}, {bandwidth_basis})"
    )
    realized_label = (
        f"{'MEMORY' if mem_bound else 'MATRIX-SERIALIZATION' if serialization_bound else 'COMPUTE'} "
        f"({timing_basis}, {bandwidth_basis})"
    )
    print(f"      -> Classical roofline: {classical_label}")
    print(f"      -> Architecture issue: {architecture_label}")
    print(f"      -> Realized RTL:       {realized_label}")
    comps = decode_token_components(
        perf,
        dims,
        avg_kv,
        args.batch,
        include_lm_head=include_lm_head,
        kv_layout=result["kv_layout"],
        packed_q1_timing_contract=result["packed_q1_timing_contract"],
        batch_packed_attention=result["batch_packed_attention"],
    )
    tot = sum(comps.values())
    print("      Decode-step cycles per operation:")
    for name, c in comps.items():
        print(f"        {name:<34} {c:>14,d} cyc  ({c/tot*100:5.1f}%)")

    # Prefill transfers BF16 KV; decode-cache quantization occurs during admission.
    sys.path.insert(0, str(_HERE.parent.parent))
    from analytic_models.disagg_serve import handoff as _handoff
    power = result["power"]
    print(f"\n[6] POWER AND ENERGY EFFICIENCY  ({n_chips} decode chip"
          f"{'s' if n_chips != 1 else ''})")
    print(f"      Memory power:      {power.memory_watts:8.1f} W "
          f"({power.memory_fraction*100:.0f}% of total)")
    print(f"      Compute power:     {power.compute_watts:8.1f} W "
          f"(@ {result['array_active_fraction']*100:.0f}% array activity, "
          f"{prec['m_bits']}-bit MAC)")
    print(f"      Total:             {power.total_watts:8.1f} W")
    print(f"      Throughput:        {power.tokens_per_second:8.1f} tok/s")
    print(f"      Energy efficiency: {power.tokens_per_joule:8.2f} tok/J")

    print(f"\n[7] KV HAND-OFF  (BF16 transfer + decode admission)")
    print(_handoff.report(dims, prec, args.input_seq, args.batch,
                          link_gen=args.link_gen,
                          link_bw=args.link_bw * 1e9 if args.link_bw else None))
    print(bar)


# Hardware + batch search (precision fixed). VLEN is not independent: the compiler
# requires VLEN == MLEN, so the vector unit follows the matrix unit everywhere.
SEARCH_SPACE = {
    "MLEN": [64, 128, 256, 512, 1024, 2048],   # matrix reduction tile (capped by bandwidth); VLEN follows
    "BLEN": [4, 8, 16, 32, 64, 128, 256],      # sub-array width; area = MLEN*BLEN; BLEN=batch fills the FFN tile
    "HLEN": [16, 32, 64, 128],       # head lane; MLEN//HLEN heads run in parallel
    "BATCH": [1, 4, 8, 16, 64, 256],    # serving knob (throughput vs latency); KV capacity caps it
}
RIGHTSIZE_TPS_TOL = 0.01             # within 1% of the best TPS still counts as "peak"


def _valid(mlen, blen, vlen, hlen, hidden=0) -> bool:
    """Legal array geometry: MLEN divisible by BLEN and HLEN, MLEN >= HLEN >= BLEN,
    plus the compiler constraints VLEN == MLEN and hidden % VLEN == 0 (hidden=0
    skips the divisibility check when model dims aren't available)."""
    return (mlen % blen == 0 and mlen % hlen == 0 and blen <= hlen <= mlen
            and vlen == mlen and (hidden == 0 or hidden % mlen == 0))


def _bandwidth_ok(mlen, hw_cfg, prec) -> bool:
    """Bandwidth check: the array needs MLEN operands per cycle, so HBM must supply
    MLEN * stream_bits <= HBM_WIDTH, else the array starves."""
    return mlen <= mlen_bandwidth_cap(hw_cfg, prec)


def _candidate(hw_cfg, dim, value):
    """One swept value with the other axes at baseline. VLEN follows MLEN (compiler
    constraint) and M_LOAD follows MLEN (no wasted reads)."""
    mlen, blen, hlen = hw_cfg.MLEN, hw_cfg.BLEN, hw_cfg.HLEN
    batch = None
    if dim == "BATCH":   batch = value
    elif dim == "MLEN":  mlen = value
    elif dim == "BLEN":  blen = value
    elif dim == "HLEN":  hlen = value
    vlen = mlen
    hw2 = hw_cfg.model_copy(update={"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen,
                                    "HBM_M_Prefetch_Amount": mlen})
    return hw2, {"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen}, batch, (mlen, blen, vlen, hlen)


def _timing_kwargs(args) -> dict:
    return {
        "timing_mode": args.timing_mode,
        "timing_evidence": getattr(args, "_timing_evidence", None),
        "packed_q1_timing_contract": getattr(
            args,
            "_packed_q1_timing_contract",
            None,
        ),
        "batch_packed_attention": args.batch_packed_attention,
        "bw_model": getattr(args, "_bw_model", None),
        "hbm_gen": args.hbm_gen or "HBM2",
        "hbm_channels": args.hbm_channels or 8,
        "output_head_location": args.output_head_location,
    }


def run_search(args, model_path, dims, base_hw, isa, base_mem, prec):
    """Two phases: (1) right-size each hardware axis at the user's batch -- the
    smallest array that still reaches ~peak TPS; (2) sweep batch on the right-sized
    chip to show its throughput/latency trade-off."""
    stride = max(1, args.output_seq // 24)
    # Clamp the starting MLEN to this precision's bandwidth cap so every sweep is
    # feasible, and tie VLEN to MLEN (compiler constraint).
    cap = mlen_bandwidth_cap(base_hw, prec)
    start_mlen = min(base_hw.MLEN, cap)
    base_hw = base_hw.model_copy(update={"MLEN": start_mlen, "VLEN": start_mlen,
                                         "HBM_M_Prefetch_Amount": start_mlen})

    def sweep_axis(dim, values, hw, batch_fixed):
        """Sweep one axis with the others fixed; print the table, return (rows, eligible)."""
        print(f"\n  -- {dim} sweep --")
        print(f"     {'value':>6} | {'TPOT(ms)':>9} | {'TPS':>9} | {'area(mm^2)':>10} | {'bound':>7} | fits")
        rows, eligible = [], []
        for v in values:
            hw2, mem_over, batch_v, geo = _candidate(hw, dim, v)
            batch = batch_v if batch_v is not None else batch_fixed
            if not (_valid(*geo, hidden=dims["hidden"]) and _bandwidth_ok(geo[0], hw2, prec)):
                continue
            try:
                r = evaluate(model_path, dims, hw2, isa, base_mem, prec, batch,
                             args.input_seq, args.output_seq, mem_over, stride=stride,
                             n_chips=args.chips, kv_layout=args.kv_layout,
                             **_timing_kwargs(args))
            except Exception as e:                            # keep the sweep alive
                print(f"     {v:>6} |  (skipped: {type(e).__name__})")
                continue
            area = area_multipliers(hw2)
            bound = decode_bound_label(r, short=True)
            print(f"     {v:>6} | {r['tpot']*1e3:>9.3f} | {r['tps']:>9.1f} | "
                  f"{area*MM2_PER_MULTIPLIER:>10.3f} | {bound:>7} | {'yes' if r['fits_in_hbm'] else 'NO'}")
            rows.append({"value": v, "tps": r["tps"], "tpot": r["tpot"], "area": area, "fits": r["fits_in_hbm"]})
            if r["fits_in_hbm"] and r["timing_calibrated"]:
                eligible.append({"value": v, "tps": r["tps"], "area": area, "tpot": r["tpot"]})
        return rows, eligible

    print("\n" + "#" * 78)
    print("[6] HARDWARE + BATCH SEARCH  (precision fixed; one axis at a time)")
    print(f"    Baseline: MLEN={base_hw.MLEN} BLEN={base_hw.BLEN} VLEN={base_hw.VLEN} "
          f"HLEN={base_hw.HLEN}  ({area_multipliers(base_hw):,} mult, {area_mm2(base_hw):.3f} mm^2)  "
          f"batch={args.batch}")
    print("    Right-size each axis = the smallest array still reaching ~peak TPS at your batch.")
    print("#" * 78)

    # Phase 1: right-size each hardware axis at the user's batch (VLEN follows MLEN).
    sweeps, best = {}, {}
    for dim in ("MLEN", "BLEN", "HLEN"):
        rows, eligible = sweep_axis(dim, SEARCH_SPACE[dim], base_hw, args.batch)
        sweeps[dim] = rows
        if not eligible:
            continue
        peak = max(e["tps"] for e in eligible)
        knee = min((e for e in eligible if e["tps"] >= (1 - RIGHTSIZE_TPS_TOL) * peak), key=lambda e: e["area"])
        best[dim] = knee["value"]
        print(f"     -> right-size {dim}={knee['value']}  (TPS={knee['tps']:.1f}, "
              f"{knee['area']*MM2_PER_MULTIPLIER:.3f} mm^2)")

    # Assemble the right-sized chip (VLEN == MLEN by the compiler constraint).
    mlen, blen = best.get("MLEN", base_hw.MLEN), best.get("BLEN", base_hw.BLEN)
    vlen, hlen = mlen, best.get("HLEN", base_hw.HLEN)
    print(f"\n  -- Right-sized decode chip @ batch={args.batch} --")
    best_hw, best_result = None, None
    if _valid(mlen, blen, vlen, hlen, hidden=dims["hidden"]) and _bandwidth_ok(mlen, base_hw, prec):
        best_hw = base_hw.model_copy(update={"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen,
                                             "HBM_M_Prefetch_Amount": mlen})
        best_result = evaluate(model_path, dims, best_hw, isa, base_mem, prec, args.batch, args.input_seq,
                               args.output_seq, {"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen},
                               stride=stride, n_chips=args.chips,
                               kv_layout=args.kv_layout, **_timing_kwargs(args))
        area = area_multipliers(best_hw)
        print(f"     MLEN={mlen} BLEN={blen} VLEN={vlen} HLEN={hlen} batch={args.batch}")
        print(f"     -> TPOT={best_result['tpot']*1e3:.3f} ms   TPS={best_result['tps']:.1f}   "
              f"{decode_bound_label(best_result)}-bound")
        print(f"     -> {area:,} mult (~{area_mm2(best_hw):.3f} mm^2 = {area/REF_MULTIPLIERS:.2f}x the "
              f"{REF_MM2:.3f} mm^2 baseline)   fits={'yes' if best_result['fits_in_hbm'] else 'NO'}")
    else:
        print("     (best-per-axis combination breaks the geometry/bandwidth rules)")

    # Phase 2: batch trade-off on the right-sized chip. BLEN=batch fills only the
    # FFN GEMM tile, and HBM capacity (KV cache) caps the batch regardless.
    rows, eligible = sweep_axis("BATCH", SEARCH_SPACE["BATCH"], best_hw or base_hw, args.batch)
    sweeps["BATCH"] = rows
    if eligible:
        peak = max(e["tps"] for e in eligible)
        eff = min((e for e in eligible if e["tps"] >= (1 - RIGHTSIZE_TPS_TOL) * peak), key=lambda e: e["value"])
        cap = f"; KV capacity caps batch at {max_batch_capacity(best_result, args.batch):,}" if best_result else ""
        print(f"     -> efficient batch={eff['value']}  (TPS={eff['tps']:.1f}, TPOT={eff['tpot']*1e3:.1f} ms; "
              f"fills the FFN BLEN tile -- attention gains nothing past it{cap})")
    print("#" * 78)
    return sweeps, best_hw, best_result


# Software DSE bridge: accuracy (continuation PPL) <-> decode cost
def _canonical_hash(value: Mapping) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_hash(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"results provenance has an invalid {field}")
    return value


def _validate_model_descriptor(value: object) -> Mapping:
    if not isinstance(value, Mapping):
        raise ValueError("results provenance is missing the model descriptor")
    for field in ("name", "revision", "tokenizer_revision", "dtype"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            raise ValueError(f"results provenance model has an invalid {field}")
    return value


def _validate_dataset_descriptors(value: object) -> Mapping:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("results provenance is missing dataset descriptors")
    for role, descriptor in value.items():
        if not isinstance(role, str) or not isinstance(descriptor, Mapping):
            raise ValueError("results provenance has an invalid dataset descriptor")
        for field in ("name", "revision", "split"):
            if not isinstance(descriptor.get(field), str) or not descriptor[field].strip():
                raise ValueError(
                    f"results provenance dataset {role!r} has an invalid {field}"
                )
        if "config" not in descriptor or not isinstance(
            descriptor["config"], (str, type(None))
        ):
            raise ValueError(
                f"results provenance dataset {role!r} has an invalid config"
            )
    return value


def _load_hashed_json(path: Path, label: str) -> tuple[dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    recorded_hash = _require_hash(value.get("content_hash"), "content_hash")
    body = dict(value)
    body.pop("content_hash")
    if _canonical_hash(body) != recorded_hash:
        raise ValueError(f"{label} content hash mismatch")
    return body, recorded_hash


def _normalise_model_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _validate_precision_source(path: Path, expected_model: str | None) -> bytes:
    sidecar_path = path.parent / "sweep_results_provenance.json"
    provenance, _ = _load_hashed_json(sidecar_path, "results provenance")
    if provenance.get("schema_version") != RESULTS_PROVENANCE_SCHEMA:
        raise ValueError("results provenance schema is unsupported")
    if not isinstance(provenance.get("created_at_utc"), str) or not provenance[
        "created_at_utc"
    ].strip():
        raise ValueError("results provenance has an invalid creation timestamp")

    model = _validate_model_descriptor(provenance.get("model"))
    datasets = _validate_dataset_descriptors(provenance.get("datasets"))
    manifest_hash = _require_hash(provenance.get("manifest_hash"), "manifest_hash")
    run_plan_hash = _require_hash(provenance.get("run_plan_hash"), "run_plan_hash")
    quantizer_hash = _require_hash(
        provenance.get("quantizer_provenance_hash"),
        "quantizer_provenance_hash",
    )
    if expected_model is not None:
        expected_name = Path(expected_model).name.removesuffix(".json")
        recorded_name = str(model["name"]).rsplit("/", 1)[-1]
        expected = _normalise_model_name(expected_name)
        recorded = _normalise_model_name(recorded_name)
        if not expected or recorded != expected:
            raise ValueError(
                f"precision source model {model['name']!r} does not match {expected_model!r}"
            )

    tables = provenance.get("tables")
    if not isinstance(tables, list):
        raise ValueError("results provenance is missing its table inventory")
    matches = [
        table
        for table in tables
        if isinstance(table, Mapping) and table.get("filename") == path.name
    ]
    if len(matches) != 1:
        raise ValueError(f"results provenance does not bind {path.name}")
    table = matches[0]
    expected_size = table.get("size_bytes")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError("results provenance has an invalid table size")
    source_bytes = path.read_bytes()
    if len(source_bytes) != expected_size or hashlib.sha256(source_bytes).hexdigest() != _require_hash(
        table.get("sha256"), "table sha256"
    ):
        raise ValueError(f"precision source checksum mismatch: {path}")

    workspace_binding = provenance.get("workspace_provenance")
    if not isinstance(workspace_binding, Mapping):
        raise ValueError("results provenance is missing workspace provenance")
    workspace_name = workspace_binding.get("path")
    if not isinstance(workspace_name, str) or not workspace_name:
        raise ValueError("results provenance has an invalid workspace path")
    workspace_path = Path(workspace_name)
    if not workspace_path.is_absolute():
        workspace_path = sidecar_path.parent / workspace_path
    workspace, workspace_hash = _load_hashed_json(
        workspace_path, "workspace provenance"
    )
    if workspace_hash != _require_hash(
        workspace_binding.get("content_hash"), "workspace content_hash"
    ):
        raise ValueError("workspace provenance binding mismatch")
    if workspace.get("schema_version") != WORKSPACE_PROVENANCE_SCHEMA:
        raise ValueError("workspace provenance schema is unsupported")
    if (
        workspace.get("manifest_hash") != manifest_hash
        or workspace.get("run_plan_hash") != run_plan_hash
        or workspace.get("quantizer_provenance_hash") != quantizer_hash
        or workspace.get("model") != model
        or workspace.get("datasets") != datasets
    ):
        raise ValueError("workspace provenance does not bind the precision source")
    return source_bytes


def _format_element_bits(token: str) -> int:
    canonical = token.upper()
    if canonical == "BF16":
        return 16
    if canonical.startswith("MXINT") and canonical.removeprefix("MXINT").isdigit():
        return int(canonical.removeprefix("MXINT"))
    if canonical.startswith("E") and "M" in canonical:
        exponent, mantissa = canonical.removeprefix("E").split("M", 1)
        if exponent.isdigit() and mantissa.isdigit():
            return 1 + int(exponent) + int(mantissa)
    raise ValueError(f"unsupported precision format {token!r} in software CSV")


def load_precision_points(path: str | Path, *, expected_model: str | None = None) -> list[dict]:
    """Read only checksum-bound software-sweep accuracy and precision points.

    The CSV must be listed in its sibling ``sweep_results_provenance.json``;
    that receipt must in turn bind the originating workspace provenance. Only
    the canonical numerical-results schema is accepted; retired accuracy CSVs
    cannot be interpreted as current results merely by adding a new sidecar.
    """
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_bytes = _validate_precision_source(source_path, expected_model)
    pts = []
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("precision CSV is not valid UTF-8") from error
    with io.StringIO(source_text, newline="") as f:
        reader = csv.DictReader(f)
        required_fields = {
            "profile_id",
            "kind",
            "weight_format",
            "activation_format",
            "key_format",
            "value_format",
            "vector_format",
            "state",
            "mean_nll",
        }
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            raise ValueError(
                "precision CSV does not use the canonical numerical-results schema"
            )
        for row_number, r in enumerate(reader, start=2):
            if r["state"] != "succeeded":
                continue
            if r["kind"] != "quantized":
                continue
            try:
                mean_nll = float(r["mean_nll"])
                if not math.isfinite(mean_nll):
                    raise ValueError("mean_nll is not finite")
                if r["key_format"] != r["value_format"]:
                    raise ValueError("key and value formats differ")
                weight_element = _format_element_bits(r["weight_format"])
                activation_element = _format_element_bits(r["activation_format"])
                kv_element = _format_element_bits(r["key_format"])
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"invalid successful precision row {row_number}: {error}"
                ) from error
            scale_share = 0.0 if r["kind"] == "bf16_reference" else SCALE_BITS / 8
            weight_bits = weight_element + scale_share
            activation_bits = activation_element + scale_share
            kv_bits = kv_element + scale_share
            pts.append(
                {
                    "tag": r["profile_id"],
                    "ppl": math.exp(mean_nll),
                    "attn_bits": weight_bits,
                    "ffn_bits": weight_bits,
                    "kv_bits": kv_bits,
                    "sw_mb": 2 * weight_bits + kv_bits,
                    "gptq": False,
                    "act_bits": activation_bits,
                    "attn_elem": weight_element,
                    "ffn_elem": weight_element,
                    "kv_elem": kv_element,
                    "act_elem": activation_element,
                    "block": 8,
                    "vector_format": r["vector_format"],
                }
            )
    if not pts:
        raise ValueError("precision CSV contains no successful numerical points")
    return pts


def pareto_front(points: list[dict]) -> list[dict]:
    """Keep only the best precisions: as memory cost rises, keep each point that lowers perplexity."""
    best_at_cost: dict[float, dict] = {}
    for point in points:
        incumbent = best_at_cost.get(point["sw_mb"])
        if incumbent is None or point["ppl"] < incumbent["ppl"]:
            best_at_cost[point["sw_mb"]] = point
    front, best_ppl = [], float("inf")
    for p in (best_at_cost[cost] for cost in sorted(best_at_cost)):
        if p["ppl"] < best_ppl - 1e-9:
            front.append(p)
            best_ppl = p["ppl"]
    return front


def _prec_from_point(p: dict, args) -> dict:
    """CSV point -> precision spec. Element widths subtract the per-block scale
    share before rounding (eff = elem + SCALE_BITS/block). The activation
    compute width joins the M-bit default when the CSV carries it."""
    share = SCALE_BITS / p.get("block", 8)

    def _elem(bits):
        return max(1, round(float(bits) - share))

    elems = [
        int(p.get("attn_elem", _elem(p["attn_bits"]))),
        int(p.get("ffn_elem", _elem(p["ffn_bits"]))),
        int(p.get("kv_elem", _elem(p["kv_bits"]))),
    ]
    if p.get("act_bits") is not None:
        elems.append(int(p.get("act_elem", _elem(p["act_bits"]))))
    return precision_from_components(p["attn_bits"], p["ffn_bits"], p["kv_bits"],
                                     attn_elem=elems[0], ffn_elem=elems[1], kv_elem=elems[2],
                                     m_bits=args.m_bits or max(elems),
                                     density_exp=args.density_exp,
                                     block_size=int(p.get("block", 8)))


def run_precision_sweep(args, model_path, dims, hw_cfg, isa, base_mem):
    """Run each point on the software PPL-vs-memory front on the decode chip ->
    MB/token, TPS, HBM-fit. When compute-bound, TPS barely moves with precision,
    so the trade-off is PPL vs MB."""
    points = load_precision_points(args.sweep, expected_model=args.model)
    front = pareto_front(points)
    stride = max(1, args.output_seq // 24)
    print("=" * 96)
    print(f"  PRECISION SWEEP (accuracy vs decode cost) — {args.model}  batch={args.batch}  "
          f"in={args.input_seq} out={args.output_seq}")
    print(f"  {len(points)} precisions in {Path(args.sweep).name}; "
          f"{len(front)} best ones (perplexity-vs-memory front)")
    print("=" * 96)
    print(f"  {'precision':<40} {'perplexity':>10} {'attn/ffn/kv':>13} {'MB/tok':>8} {'TPS':>8} fits")
    rows = []
    for p in front:
        prec = _prec_from_point(p, args)
        # Clamp MLEN to this precision's bandwidth cap so the TPS is feasible;
        # VLEN follows MLEN. Memory cost (MB/token) does not depend on MLEN.
        mlen = min(hw_cfg.MLEN, mlen_bandwidth_cap(hw_cfg, prec))
        hw = hw_cfg.model_copy(update={"MLEN": mlen, "VLEN": mlen, "HBM_M_Prefetch_Amount": mlen})
        r = evaluate(model_path, dims, hw, isa, base_mem, prec, args.batch,
                     args.input_seq, args.output_seq, {"MLEN": mlen, "VLEN": mlen},
                     stride=stride, n_chips=args.chips,
                     kv_layout=args.kv_layout, **_timing_kwargs(args))
        if not r["timing_calibrated"]:
            raise ValueError(
                "precision sweep point lacks matching PackedKV timing evidence"
            )
        mb = r["avg_bytes_per_generated_token"] / 1e6
        label = f"{prec['attn_elem']}/{prec['ffn_elem']}/{prec['kv_elem']}"
        print(f"  {p['tag'][:40]:<40} {p['ppl']:>10.3f} {label:>13} {mb:>8.1f} {r['tps']:>8.1f}  "
              f"{'yes' if r['fits_in_hbm'] else 'NO'}")
        rows.append({"tag": p["tag"], "ppl": p["ppl"], "mb": mb, "fits": r["fits_in_hbm"],
                     "gptq": p["gptq"], "label": label})
    print("=" * 96)
    return rows


def right_size(args, model_path, dims, base_hw, isa, base_mem, prec, stride):
    """Smallest area at peak throughput (TPS). VLEN follows MLEN (compiler constraint)."""
    cap = mlen_bandwidth_cap(base_hw, prec)
    start_mlen = min(base_hw.MLEN, cap)
    start = base_hw.model_copy(update={"MLEN": start_mlen, "VLEN": start_mlen,
                                       "HBM_M_Prefetch_Amount": start_mlen})
    best = {}
    for dim in ("MLEN", "BLEN", "HLEN"):
        eligible = []
        for v in SEARCH_SPACE[dim]:
            hw2, mem_over, _, geo = _candidate(start, dim, v)
            if not (_valid(*geo, hidden=dims["hidden"]) and _bandwidth_ok(geo[0], hw2, prec)):
                continue
            try:
                r = evaluate(model_path, dims, hw2, isa, base_mem, prec, args.batch,
                             args.input_seq, args.output_seq, mem_over, stride=stride,
                             n_chips=args.chips, kv_layout=args.kv_layout,
                             **_timing_kwargs(args))
            except Exception:
                continue
            if r["fits_in_hbm"] and r["timing_calibrated"]:
                eligible.append({"value": v, "tps": r["tps"], "area": area_multipliers(hw2)})
        if eligible:
            peak = max(e["tps"] for e in eligible)
            best[dim] = min((e for e in eligible if e["tps"] >= (1 - RIGHTSIZE_TPS_TOL) * peak),
                            key=lambda e: e["area"])["value"]
    mlen, blen = best.get("MLEN", start.MLEN), best.get("BLEN", start.BLEN)
    vlen, hlen = mlen, best.get("HLEN", start.HLEN)
    if _valid(mlen, blen, vlen, hlen, hidden=dims["hidden"]) and _bandwidth_ok(mlen, start, prec):
        return start.model_copy(update={"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen,
                                        "HBM_M_Prefetch_Amount": mlen})
    return start


def run_codesign(args, model_path, dims, base_hw, isa, base_mem):
    points = pareto_front(
        load_precision_points(args.codesign, expected_model=args.model)
    )
    stride = max(1, args.output_seq // 24)
    print("\n" + "#" * 104)
    print(f"[7] PRECISION x HARDWARE CO-DESIGN — {args.model}  batch={args.batch}  "
          f"in={args.input_seq} out={args.output_seq}")
    print("    Each precision: cap MLEN by bandwidth, then right-size the whole array")
    print("    ('bound' shows where each precision lands, no assumption made).")
    print("#" * 104)
    print(f"  {'precision':<34} {'PPL':>9} {'maxW|KV':>7} {'MLEN':>5} {'BLEN':>5} {'area(mm^2)':>10} "
          f"{'TPS':>8} {'max-batch':>10} {'chips':>5} {'bound':>5} fits")
    rows = []
    for p in points:
        prec = _prec_from_point(p, args)
        hw = right_size(args, model_path, dims, base_hw, isa, base_mem, prec, stride)
        r = evaluate(model_path, dims, hw, isa, base_mem, prec, args.batch, args.input_seq, args.output_seq,
                     {"MLEN": hw.MLEN, "BLEN": hw.BLEN, "VLEN": hw.VLEN, "HLEN": hw.HLEN},
                     stride=stride, n_chips=args.chips,
                     kv_layout=args.kv_layout, **_timing_kwargs(args))
        if not r["timing_calibrated"]:
            continue
        max_batch = max_batch_capacity(r, args.batch)
        bound = decode_bound_label(r, short=True)
        label = f"{prec['attn_elem']}/{prec['ffn_elem']}/{prec['kv_elem']}"
        print(f"  {p['tag'][:34]:<34} {p['ppl']:>9.3f} {stream_bits(prec):>7} {hw.MLEN:>5} {hw.BLEN:>5} "
              f"{area_mm2(hw):>10.3f} {r['tps']:>8.1f} {max_batch:>10,} {r['n_chips']:>5} {bound:>5} {'yes' if r['fits_in_hbm'] else 'NO'}")
        rows.append({"ppl": p["ppl"], "tps": r["tps"], "max_batch": max_batch, "fits": r["fits_in_hbm"],
                     "area": area_mm2(hw), "bound": bound, "label": label, "n_chips": r["n_chips"]})
    print("#" * 104)
    return rows


# =============================================================================
# Tier-2: theoretical system comparison from published peak specifications
# =============================================================================
def run_compare(args, model_path, dims, hw_cfg, base_mem, prec, device_names):
    """Theoretical throughput at each device's max-fitting aggregate-HBM batch.

    Workload bytes/FLOPs are recomputed per device because they scale with its
    batch; util = how well the decode batch fills the M-tile (PLENA's small BLEN
    fills earlier than a GPU)."""
    kv, ctx = args.input_seq + args.output_seq // 2, args.input_seq + args.output_seq
    mem = LLMMemoryModel(model_path, base_mem.model_copy(update={
        "weight_bits": prec["ffn_bits"], "activation_bits": ACT_BITS, "kv_cache_bits": prec["kv_bits"]}),
        batch_size=1, input_seq_len=args.input_seq, output_seq_len=args.output_seq).mem
    wf = weight_footprint_bytes(mem, dims, prec)["total"]       # sharded once across the system
    kv_per_batch = kv_footprint_bytes(
        mem,
        dims,
        prec,
        ctx,
        1,
        mlen=hw_cfg.MLEN,
        kv_layout=args.kv_layout,
    )

    print("\n" + "=" * 112)
    print(f"[8] THEORETICAL ROOFLINE COMPARISON — {args.model} decode  in={args.input_seq} out={args.output_seq}")
    print("    Published peak specifications only; these rows are not measured baselines.")
    print(f"    precision attnW/ffnW/KV = {prec['attn_bits']:.2f}/{prec['ffn_bits']:.2f}/{prec['kv_bits']:.2f}"
          f"   BS = largest fitting each system's aggregate HBM (full-capacity utilisation)")
    print("=" * 112)
    print(f"  {'device':<11} {'peakTF':>8} {'BW(TB/s)':>9} {'cap(GB)':>8} {'BS':>8} {'util%':>6} "
          f"{'bound':>6} {'TPS':>11} {'xA100':>7}")
    rows, a100 = [], None
    for name in device_names:
        dev = plena_device(hw_cfg, base_mem) if name == "plena" else DEVICES[name]
        peak_c, peak_bw, cap = device_peaks(dev, prec)         # PLENA peak is bandwidth-capped per precision
        bs = max(1, int((cap - wf) // max(kv_per_batch, 1)))   # largest batch whose KV still fits
        tr = decode_traffic(
            mem,
            dims,
            kv,
            bs,
            prec,
            mlen=hw_cfg.MLEN if name == "plena" else None,
            kv_layout=args.kv_layout,
        )
        bytes_step = tr.read_bytes + tr.write_bytes
        flops_step = decode_step_flops(dims, kv, bs)
        m_tile = dev["blen"] if dev["kind"] == "plena" else dev["sq_dim"]
        util = fill_util(bs, m_tile)
        comp_t, mem_t = flops_step / (peak_c * util), bytes_step / peak_bw
        bound = "mem" if mem_t >= comp_t else "cmp"            # wall this device hits
        tps = bs / max(comp_t, mem_t)                          # tokens/s at the max-fitting batch
        r = dict(
            label=dev["label"],
            peak_c=peak_c,
            peak_bw=peak_bw,
            cap=cap,
            bs=bs,
            util=util,
            bound=bound,
            tps=tps,
            evidence_scope="theoretical_roofline_reference",
            measured=False,
        )
        rows.append(r)
        if name == "a100":
            a100 = r
    for r in rows:
        xt = r["tps"] / a100["tps"] if a100 else float("nan")
        print(f"  {r['label']:<11} {r['peak_c']/1e12:>8.0f} {r['peak_bw']/1e12:>9.2f} {r['cap']/1e9:>8.0f} "
              f"{r['bs']:>8,} {r['util']*100:>5.0f}% {r['bound']:>6} {r['tps']:>11.1f} {xt:>6.2f}x")
    print("=" * 112)
    print("    BS = largest batch fitting each system's aggregate HBM; util = decode-batch fill of the M-tile.")
    print("    PLENA row = the configured chip (geometry/HBM from the live config, compute at M-bit density).")
    print("    Publication claims require separately sealed device measurements.")
    return rows


def configure(args):
    """Resolve the model, hardware and precision one point is evaluated at.

    The HBM generation replaces the configured bandwidth and capacity before the
    array-shape overrides apply, so a caller that sets `--hbm-gen` gets that
    technology's per-chip capacity in both the hardware config and the memory
    config. Evaluating a point without this leaves the TOML's capacity in place
    and silently resolves a different chip count.
    """
    model_path = resolve_model_path(args.model, args.model_lib)
    dims = load_model_dims(model_path)
    hw_cfg = load_hardware_config_from_toml(args.config)
    base_mem = load_memory_config_from_toml(args.config)
    if args.hbm_gen:
        over = hbm_overrides(args.hbm_gen, args.hbm_channels)
        args.hbm_channels = over.pop("channels")
        hw_cfg, base_mem = hw_cfg.model_copy(update=over), base_mem.model_copy(update=over)
    tile_over = {
        key: value
        for key, value in (
            ("MLEN", args.mlen),
            ("BLEN", args.blen),
            ("HLEN", getattr(args, "hlen", 0)),
        )
        if value
    }
    if tile_over:
        if "MLEN" in tile_over:
            tile_over["VLEN"] = tile_over["MLEN"]
            tile_over["HBM_M_Prefetch_Amount"] = tile_over["MLEN"]
        hw_cfg = hw_cfg.model_copy(update=tile_over)
        hw_cfg = hw_cfg.model_copy(
            update={"BROADCAST_AMOUNT": hw_cfg.MLEN // hw_cfg.HLEN}
        )
    return model_path, dims, hw_cfg, base_mem


def resolve_model_path(model_name, model_lib):
    p = Path(model_lib) / f"{model_name}.json"
    if not p.exists():
        avail = ", ".join(sorted(f.stem for f in Path(model_lib).glob("*.json")))
        raise FileNotFoundError(f"Model '{model_name}' not found in {model_lib}. Available: {avail}")
    return str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--model-lib", default=str(_ANALYTIC.parent / "compiler" / "doc" / "Model_Lib"))
    ap.add_argument("--config", default=str(_ANALYTIC.parent / "plena_settings.toml"))
    ap.add_argument("--isa-lib", default=str(_HERE / "customISA_lib.json"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--input-seq", type=int, default=256)
    ap.add_argument("--output-seq", type=int, default=16384)
    ap.add_argument("--w-fmt", choices=["mxint", "mxfp"], default="mxint",
                    help="weight format, shared by attention + FFN weights")
    ap.add_argument("--kv-fmt", choices=["mxint", "mxfp"], default="mxint",
                    help="KV-cache format, independent of the weights (mixed precision)")
    ap.add_argument("--attn-w", default="4", help="attention-projection weight width (q/k/v/o)")
    ap.add_argument("--ffn-w", default="4", help="FFN/expert-projection weight width (gate/up/down)")
    ap.add_argument("--kv", default="4", help="KV-cache width")
    ap.add_argument("--block", type=int, default=8,
                    help="MX block size for element and E8M0 scale accounting")
    ap.add_argument(
        "--kv-layout",
        choices=PACKED_KV_MODES,
        default=DENSE_SELECTOR,
        help="physical KV-cache layout used for traffic and capacity",
    )
    ap.add_argument(
        "--timing-mode",
        choices=TIMING_MODES,
        default=RTL_SERIALIZED,
        help="matrix timing contract; the pipeline oracle cannot rank without matching evidence",
    )
    ap.add_argument(
        "--timing-evidence",
        default=None,
        help="trace-calibration JSON for the selected timing mode",
    )
    ap.add_argument(
        "--packed-q1-timing-contract",
        default=None,
        help="content-addressed compiler opcode-count contract for cached PackedKV timing",
    )
    ap.add_argument(
        "--output-head-location",
        choices=sorted(OUTPUT_HEAD_LOCATIONS),
        default=DECODE_BF16_HEAD,
        help="decode boundary; the external service is composed by the exact system evaluator",
    )
    ap.add_argument("--m-bits", type=int, default=0,
                    help="MAC compute width M: HBM streams N-bit operands, the array multiplies at M, "
                         "writeback requantises to N. 0 = widest operand max(attnW, ffnW, KV) "
                         "(+ the activation compute width for CSV points)")
    ap.add_argument("--density-exp", type=float, default=DENSITY_EXP,
                    help="iso-area density exponent k: an M-bit array fits (4/M)^k x the reference "
                         "multipliers. Default 0 = precision-neutral compute; set 2.0 to enable the "
                         "density layer (until a Synopsys DC sweep calibrates it)")
    ap.add_argument("--hbm-gen", choices=sorted(HBM_GENS), default=None,
                    help="set HBM bandwidth and capacity from this generation x --hbm-channels "
                         "instead of the TOML HBM_WIDTH/HBM_SIZE")
    ap.add_argument("--hbm-channels", type=int, default=0,
                    help="64-bit HBM interface units; 0 = one full stack of --hbm-gen")
    ap.add_argument("--chips", type=int, default=0,
                    help="tensor-parallel decode chips over UALink (0 = auto: fewest HBM stacks that hold "
                         "the model). Large models (405B / 235B) need several stacks.")
    ap.add_argument("--area-model", choices=("proxy", "calibrated"), default="proxy",
                    help="chip-area model: 'proxy' = mm^2/multiplier; 'calibrated' = precision-aware "
                         "MatrixMachine structural census (DC-fitted, validated at large MLEN; "
                         "matches the known 0.237 mm^2 at 4x1024 MXINT4)")
    ap.add_argument("--link-gen", choices=("nvlink3", "nvlink4", "ualink", "pcie5"), default="nvlink4",
                    help="prefill->decode interconnect for the KV hand-off timing")
    ap.add_argument("--link-bw", type=float, default=0,
                    help="override the interconnect bandwidth, GB/s per direction (0 = --link-gen preset)")
    ap.add_argument("--mlen", type=int, default=0,
                    help="override the TOML MLEN (matrix tile length), e.g. 2048 for the baseline array")
    ap.add_argument("--blen", type=int, default=0,
                    help="override the TOML BLEN (block/batch tile), e.g. 32 for the baseline array")
    ap.add_argument("--bw-model", choices=("peak", "calibrated"), default="peak",
                    help="memory-time pricing: 'peak' = bytes / aggregate peak bandwidth; 'calibrated' = "
                         "per-class effective bandwidth measured on the emulator "
                         "(disagg_serve/calibration_bw.csv)")
    ap.add_argument("--batch-packed-attention", action="store_true",
                    help="pack the decode batch into the matrix unit's query dimension instead of "
                         "issuing one attention program per batch element; the lowering currently "
                         "emits per-element programs, so this reports the available headroom")
    ap.add_argument(
        "--emulator-calibration",
        type=Path,
        default=None,
        help="analytic-vs-emulator agreement artifact from "
             "decode_stage_validation.py --emit-calibration",
    )
    ap.add_argument("--search", action="store_true", help="right-size the decode hardware for this precision")
    ap.add_argument(
        "--sweep",
        metavar="CSV",
        help="precision sweep over an explicitly selected, provenance-bound software CSV",
    )
    ap.add_argument(
        "--codesign",
        metavar="CSV",
        help="joint precision x hardware co-design over a provenance-bound software CSV",
    )
    ap.add_argument("--compare", action="store_true",
                    help="theoretical comparison from published A100/H100/TPU peak specifications")
    ap.add_argument("--compare-devices", default="plena,a100,h100",
                    help="comma list from {plena,a100,h100,tpu}")
    args = ap.parse_args()
    args._timing_evidence = (
        TimingEvidence.load(args.timing_evidence)
        if args.timing_evidence
        else None
    )
    args._packed_q1_timing_contract = (
        PackedQ1TimingContract.load(args.packed_q1_timing_contract)
        if args.packed_q1_timing_contract
        else None
    )

    model_path, dims, hw_cfg, base_mem = configure(args)

    # Software-DSE CSVs come from the separate software search; skip until they exist.
    for flag in ("sweep", "codesign"):
        path = getattr(args, flag)
        if path and not Path(path).exists():
            print(f"[--{flag} skipped: {path} not found -- run the software DSE to generate it]")
            setattr(args, flag, None)

    args._emulator_calibration = (
        EmulatorCalibration.load(args.emulator_calibration)
        if args.emulator_calibration
        else None
    )

    prec = build_precision(args)
    if args.area_model == "calibrated":
        set_area_model("calibrated", prec)
    bw_model = None
    if args.bw_model == "calibrated":
        if args.hbm_gen is None:
            raise ValueError(
                "--bw-model calibrated requires an explicit --hbm-gen "
                "and matching technology operating point"
            )
        # Repo root on sys.path so the disagg_serve package resolves when this
        # file is run as a script from anywhere.
        sys.path.insert(0, str(_HERE.parent.parent))
        from analytic_models.disagg_serve.memory import CalibratedBandwidth
        bw_model = CalibratedBandwidth.load()
    args._bw_model = bw_model
    stride = max(1, args.output_seq // 256)            # subsample the context loop for speed
    result = evaluate(model_path, dims, hw_cfg, args.isa_lib, base_mem, prec,
                      args.batch, args.input_seq, args.output_seq, stride=stride, n_chips=args.chips,
                      kv_layout=args.kv_layout, **_timing_kwargs(args))
    print_report(args, dims, hw_cfg, prec, result)

    # A search may still enumerate candidates without absolute trace calibration,
    # but its ordering remains a model-only sensitivity until every required
    # compiler, emulator, and RTL anchor is present.
    ranking_requested = args.search or args.sweep or args.codesign
    if ranking_requested and not result["timing_calibrated"]:
        emulator = args._emulator_calibration
        print(
            f"\n[!] Rankings below are not trace-calibrated: "
            f"{result['timing_reason']} for {args.timing_mode}."
        )
        if emulator is not None and emulator.passed:
            print(
                f"    Per-stage terms are {emulator.label} against "
                f"{emulator.configuration}: worst stage "
                f"{emulator.worst_stage_error:.1%}; "
                f"{emulator.uncovered_fraction:.1%} of the measured layer carries "
                f"no analytic term.\n"
                f"    Absolute cycle counts are validated against the "
                f"transactional emulator but not against RTL."
            )
        else:
            print(
                f"    Candidate ordering is a model-only sensitivity, not a "
                f"publication or deployment ranking.\n"
                f"    Supply --emulator-calibration for emulator agreement."
            )
        print(
            f"    Supply --timing-evidence to promote these to "
            f"trace-calibrated results."
        )
    if args.search:
        run_search(args, model_path, dims, hw_cfg, args.isa_lib, base_mem, prec)
    if args.sweep:
        run_precision_sweep(args, model_path, dims, hw_cfg, args.isa_lib, base_mem)
    if args.codesign:
        run_codesign(args, model_path, dims, hw_cfg, args.isa_lib, base_mem)
    if args.compare:
        names = [d.strip() for d in args.compare_devices.split(",") if d.strip() in DEVICES]
        if not names:
            raise SystemExit(f"--compare-devices: no valid devices in '{args.compare_devices}' "
                             f"(choose from {', '.join(DEVICES)})")
        run_compare(args, model_path, dims, hw_cfg, base_mem, prec, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
