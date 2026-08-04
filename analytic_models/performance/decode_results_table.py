"""Decode performance sensitivities with explicit validation tiers.

The instruction-level PLENA rows come from the analytic decode model in
``disagg_decode.py``. A content-addressed calibration is loaded and reported,
but it covers the transactional-emulator test geometry rather than the headline
array and does not promote these rows to RTL evidence. Device-reference rows use
peak specifications through one traffic model and are theoretical roofline
bounds, not measured baselines.

No accuracy file is ingested here. The available Llama measurements predate the
current quantiser arithmetic, and no Qwen3-32B accuracy point exists, so every
accuracy cell remains blank until a provenance-checked sweep produces one.

Prefill runs unquantised BF16 on a separate chip; only the decode precision
varies. The BF16 KV cache crosses the boundary unchanged and decode quantises it
once on admission.

The first-decode metric is the first batch-step at the initial cache length. It
excludes prefill, handoff, admission, and queueing, so it is not TTFT.

Comparability rules, because a decode comparison is easy to get wrong:

- GPUs retain the reference system's aggregate HBM. PLENA uses the smallest
  whole-chip count that meets that capacity, which can provide a small excess;
  every row reports the resulting maximum-fitting batch.
- The GPU rows are peak rooflines, so the PLENA peak roofline is reported on the
  same footing and is the like-for-like number. The instruction-level rows are
  reported separately, with `%peak` showing how much of PLENA's own roofline each
  configuration realises; charging PLENA for issue and drain overheads that the
  GPU rows do not model at all would not be a comparison.
- The head lane width follows each model's head dimension, because HLEN *is* the
  head lane: one broadcast matmul covers `MLEN / HLEN` query heads of one KV
  group.

The configuration axis reports three independent choices. "Packed" is avoided
because it is overloaded elsewhere: the KV *cache layout* is packed (all KV heads
of a token share one MLEN-wide row) in every row below, and that is not what
varies here.

- `single-q` vs `batched-q`: whether a batch of `q_len=1` tokens is lowered one
  token at a time or packed into query rows. The current decoder has the batched
  matrix arms, while the single-token form uses `M_BTMV` and `M_BMV_WO`, which
  carry ISA encodings but have no decoder arm. The single-q sensitivity includes
  the source-emitted `M_BMV_WO` and `M_MV_WO` drains at their serialized
  MLEN-wide cost. It remains an unsupported sensitivity, not the comparison
  baseline.

  The reference decode lowering that produces the compiler-trace artifacts does
  emit the opcode-compatible batched form: it packs the batch into query rows
  (`s_q == batch`) and its emitted decode layer contains only `M_MM`, `M_TMM`,
  `M_BTMM`, `M_MM_WO` and `M_BMM_WO`, each of which the decoder implements. At
  the headline geometry it resolves the packed group layout with a GQA ratio of
  8 against 8 broadcast lanes, so the batched path is selected there too. What
  is still missing for a deployable Qwen claim is a numerical run at that
  geometry: the address-resolved request-memory sidecar exceeds its exact
  dynamic-instruction limit when the testbench forces the query-row count to
  MLEN, so the whole-layer evidence stops at the emulator test geometry.
- `KV read per-head` vs `selector timing + KV read 1x`: the emitted lowering
  prefetches the packed KV row once for each KV head, which reads it `kv_heads`
  times. The selector row combines one read per token with the selector compute
  timing; it is not a traffic-only attribution.

  The opt-in emitter contains that reordered loop and `M_BTMM` carries a
  head-selector field that picks the head window out of a resident tile. Full
  decoder runs at MLEN=64, HLEN=16, BLEN=4, kv=128 and a fixed four-row tile
  are emulator-verified for hkv=2 and hkv=4: reuse is output-bit-identical to
  the per-head schedule, while issue-origin physical K+V reads fall from
  1,048,576 B to 524,288 B and from 2,097,152 B to 524,288 B respectively.
  The run receipts recompute those tensor totals from hash-bound assembly and
  emulator op-stats and reconcile all issue-origin reads to the emulator's
  global physical counter. They are synthetic compiler/emulator measurements,
  not headline-geometry or RTL results. The active MXFP profile marks the
  selector unsupported; selector capability remains profile-dependent and is
  present only in the MXINT profile. Keeping `g` groups live also keeps their
  softmax state live: this fixed four-row geometry uses
  `6 + 3 * BLEN * (MLEN / HLEN) * g` scalar slots, 102 at `g = 2` and 198 at
  `g = 4`. Those synthetic depths must not be extrapolated to Qwen geometry.

  Only a full M_BTMM query block has to fit: packed query rows are separate
  sequences holding their own caches, so tiling the batch adds no KV-cache
  reread traffic. Execution effects remain a separate measurement question.

Usage:
    decode_results_table.py --models qwen3-32b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

from disagg_decode import (  # noqa: E402
    DENSITY_EXP,
    DEVICES,
    LLMMemoryModel,
    area_mm2,
    build_precision,
    configure,
    decode_step_flops,
    decode_traffic,
    device_peaks,
    evaluate,
    fill_util,
    kv_footprint_bytes,
    load_model_dims,
    max_batch_capacity,
    plena_device,
    resolve_model_path,
    weight_footprint_bytes,
)
from packed_kv import DENSE_COMPILER, DENSE_SELECTOR  # noqa: E402
from decode_timing import DRAIN_OVERLAPPED, RTL_SERIALIZED  # noqa: E402
from emulator_calibration import (  # noqa: E402
    EmulatorCalibration,
    sha256_file,
)
from fp_sram_sweep import (  # noqa: E402
    FP_SLOT_BITS,
    RTL_FP_SRAM_DEPTH,
    depth_for_reuse,
    evaluate as evaluate_fp_sram,
)
from decode_power import REFERENCE_CONFIGURATION  # noqa: E402
from hbm_technology import hbm_technology  # noqa: E402

#: Activations stay BF16 in on-chip SRAM and never reach HBM.
ACT_BITS = 16

#: Published TDP point used for one accelerator; form-factor details follow.
DEVICE_TDP_WATTS = {"a100": 400.0, "h100": 700.0}

# Primary vendor specifications for the exact GPU form factors used below.
DEVICE_REFERENCE_SOURCES = {
    "a100": {
        "sku": "NVIDIA A100 80GB SXM",
        "source_title": "NVIDIA A100 Tensor Core GPU Data Sheet",
        "source_url": (
            "https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/"
            "a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf"
        ),
        "compute_basis": "dense BF16 Tensor Core peak (without sparsity)",
        "tdp_basis": "400 W standard SXM configuration",
    },
    "h100": {
        "sku": "NVIDIA H100 SXM 80GB",
        "source_title": "NVIDIA H100 Tensor Core GPU product specifications",
        "source_url": "https://www.nvidia.com/en-us/data-center/h100/",
        "compute_basis": (
            "dense BF16 Tensor Core peak, one half of the vendor's "
            "sparsity-marked BF16 value"
        ),
        "tdp_basis": "700 W maximum configured SXM TDP",
    },
}

_REPOSITORY = _HERE.parents[1]
_PATH_ARGUMENTS = frozenset(
    {"model_lib", "config", "isa_lib", "emulator_calibration"}
)
_RESULTS_SOURCE_PATHS = {
    "decode_results_table": _HERE / "decode_results_table.py",
    "disagg_decode": _HERE / "disagg_decode.py",
    "perf_model": _HERE / "perf_model.py",
    "packed_q1_timing": _HERE / "packed_q1_timing.py",
    "decode_timing": _HERE / "decode_timing.py",
    "fp_sram_sweep": _HERE / "fp_sram_sweep.py",
    "emulator_calibration": _HERE / "emulator_calibration.py",
    "packed_kv": _HERE.parent / "disagg_serve" / "packed_kv.py",
    "physical_ledger": _HERE.parent / "disagg_serve" / "physical_ledger.py",
    "decode_power": _HERE.parent / "disagg_serve" / "decode_power.py",
    "hbm_technology": _HERE.parent / "disagg_serve" / "hbm_technology.py",
    "memory_model": _HERE.parent / "memory" / "memory_model.py",
    "llm_memory_model": _HERE.parent / "memory" / "llm_memory_model.py",
    "area_calibration": _HERE.parent / "area" / "calibration_provenance.py",
    "area_full_chip_anchors": (
        _HERE.parent / "area" / "calibration" / "full_chip_anchors.csv"
    ),
    "area_structural_coefficients": (
        _HERE.parent
        / "area"
        / "calibration"
        / "matrix_structural_coefficients.json"
    ),
}

PEAK_ROOFLINE = "peak roofline"


def _repository_path(value: str | Path) -> str:
    """Return a stable repository-relative path when the input is in-tree."""
    path = Path(value).resolve()
    try:
        return str(path.relative_to(_REPOSITORY))
    except ValueError:
        return str(path)


def _json_safe(value):
    """Return an RFC-8259-compatible representation of nested result data."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _canonical_hash(value: dict[str, object]) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def results_source_provenance(
    paths: dict[str, Path] | None = None,
) -> dict[str, dict[str, str]]:
    """Hash every implementation and calibration source used by the table."""
    provenance: dict[str, dict[str, str]] = {}
    for name, source in (paths or _RESULTS_SOURCE_PATHS).items():
        path = Path(source).resolve()
        try:
            repository_path = str(path.relative_to(_REPOSITORY))
        except ValueError:
            repository_path = str(path)
        provenance[name] = {
            "repository_path": repository_path,
            "sha256": sha256_file(path),
        }
    return provenance


def validate_calibration_inputs(calibration: EmulatorCalibration, args) -> None:
    """Require the calibration's measured settings and ISA to match this run."""
    observed = dict(calibration.provenance_hashes)
    expected = {
        "settings": sha256_file(args.config),
        "isa_lib": sha256_file(args.isa_lib),
    }
    for role, digest in expected.items():
        if observed.get(role) != digest:
            raise ValueError(
                f"calibration {role} hash does not match the active results input"
            )


def _device_reference_provenance() -> dict[str, dict[str, object]]:
    references: dict[str, dict[str, object]] = {}
    for name in ("a100", "h100"):
        references[name] = {
            **DEVICE_REFERENCE_SOURCES[name],
            "peak_tflops": DEVICES[name]["peak_tflops"],
            "hbm_gb": DEVICES[name]["hbm_gb"],
            "hbm_tbs": DEVICES[name]["hbm_tbs"],
            "tdp_w": DEVICE_TDP_WATTS[name],
            "device_count": DEVICES[name]["count"],
        }
    return references


def precision_label(precision: dict[str, object]) -> str:
    """Display element formats without conflating them with scale overhead."""
    return "/".join(
        (
            str(precision["attn_label"]),
            str(precision["ffn_label"]),
            "BF16",
            str(precision["kv_label"]),
        )
    )


@dataclass
class Row:
    """One line of the results table."""

    model: str
    device: str
    configuration: str
    precision: str
    batch: int
    first_decode_ms: float
    tps: float
    tpot_ms: float
    power_w: float
    tokens_per_joule: float
    area_mm2: float
    evidence_tier: str
    accuracy: dict[str, float] = field(default_factory=dict)


def head_lane_width(args, model: str) -> int:
    """The head lane the array is shaped to for this model."""
    if args.hlen:
        return args.hlen
    return load_model_dims(resolve_model_path(model, args.model_lib))["head_dim"]


def build_point(args, model: str):
    """Resolve the model, hardware, precision and memory model for one point."""
    args.model = model
    requested_hlen = args.hlen
    args.hlen = head_lane_width(args, model)
    model_path, dims, hardware, base_mem = configure(args)
    args.hlen = requested_hlen
    return model_path, dims, hardware, base_mem, build_precision(args)


def capacity_match(args, model: str, target_capacity: float) -> tuple[int, int]:
    """Chips holding `target_capacity`, and the largest batch that fits them."""
    args.chips, args.batch = 0, 1
    model_path, dims, hardware, base_mem, precision = build_point(args, model)
    chips = max(1, math.ceil(target_capacity / base_mem.HBM_SIZE))
    probe = evaluate(
        model_path,
        dims,
        hardware,
        args.isa_lib,
        base_mem,
        precision,
        1,
        args.input_seq,
        args.output_seq,
        stride=max(1, args.output_seq // 32),
        n_chips=chips,
        kv_layout=DENSE_COMPILER,
        batch_packed_attention=True,
        hbm_gen=args.hbm_gen,
        hbm_channels=args.hbm_channels,
    )
    return chips, max(1, max_batch_capacity(probe, 1))


def modelled_row(
    args, model: str, chips: int, batch: int, batch_packed: bool,
    kv_layout: str, timing_mode: str = RTL_SERIALIZED,
) -> Row:
    """One instruction-level operating point of the decode chip."""
    args.chips, args.batch = chips, batch
    model_path, dims, hardware, base_mem, precision = build_point(args, model)
    result = evaluate(
        model_path,
        dims,
        hardware,
        args.isa_lib,
        base_mem,
        precision,
        batch,
        args.input_seq,
        args.output_seq,
        stride=max(1, args.output_seq // 256),
        n_chips=chips,
        kv_layout=kv_layout,
        batch_packed_attention=batch_packed,
        hbm_gen=args.hbm_gen,
        hbm_channels=args.hbm_channels,
        timing_mode=timing_mode,
    )
    power = result["power"]
    co_design = kv_layout == DENSE_SELECTOR or timing_mode != RTL_SERIALIZED
    if not batch_packed:
        evidence_tier = "analytic unsupported"
    elif co_design:
        evidence_tier = "analytic co-design"
    else:
        evidence_tier = "analytic baseline"
    kv_schedule = (
        "selector timing + KV read 1x"
        if kv_layout == DENSE_SELECTOR
        else "KV read per-head"
    )
    return Row(
        model=model,
        device=f"PLENA x{chips}",
        # Three independent choices, named so they cannot be confused: how the
        # batch is lowered, how often the KV row is read, and whether the
        # writeout overlaps the next accumulate.
        configuration=(
            f"{'batched-q' if batch_packed else 'single-q'} + {kv_schedule}"
            + (" + drain ovl" if timing_mode == DRAIN_OVERLAPPED else "")
            + (" (unsupported)" if not batch_packed else "")
        ),
        precision=precision_label(precision),
        batch=batch,
        first_decode_ms=result["first_step"] * 1e3,
        tps=result["tps"],
        tpot_ms=result["tpot"] * 1e3,
        # The co-design throughput sensitivities are useful before physical
        # implementation, but their added SRAM/bank area and power are unknown.
        # Leaving those cells blank prevents the baseline proxy from being
        # mistaken for an implementation-inclusive cost.
        power_w=float("nan") if co_design else power.total_watts,
        tokens_per_joule=float("nan") if co_design else power.tokens_per_joule,
        area_mm2=float("nan") if co_design else area_mm2(hardware),
        evidence_tier=evidence_tier,
    )


def roofline_rows(args, model: str, chips: int) -> list[Row]:
    """Peak-specification rooflines at each system's max-fitting batch.

    Every row uses one formula — `batch / max(compute_time, memory_time)` from
    peak compute and peak bandwidth — so the devices are comparable to each
    other, PLENA included.
    """
    model_path, dims, hardware, base_mem, precision = build_point(args, model)
    kv = args.input_seq + args.output_seq // 2
    context = args.input_seq + args.output_seq
    memory = LLMMemoryModel(
        model_path,
        base_mem.model_copy(
            update={
                "weight_bits": precision["ffn_bits"],
                "activation_bits": ACT_BITS,
                "kv_cache_bits": precision["kv_bits"],
            }
        ),
        batch_size=1,
        input_seq_len=args.input_seq,
        output_seq_len=args.output_seq,
    ).mem
    weights = weight_footprint_bytes(memory, dims, precision)["total"]

    rows: list[Row] = []
    for name in ("plena", "a100", "h100"):
        plena = name == "plena"
        device = (
            {**plena_device(hardware, base_mem), "count": chips} if plena else DEVICES[name]
        )
        peak_compute, peak_bw, capacity = device_peaks(device, precision)
        tile_mlen = hardware.MLEN if plena else None
        kv_per_batch = kv_footprint_bytes(
            memory, dims, precision, context, 1,
            mlen=tile_mlen, kv_layout=args.roofline_kv_layout,
        )
        batch = max(1, int((capacity - weights) // max(kv_per_batch, 1)))
        traffic = decode_traffic(
            memory, dims, kv, batch, precision,
            mlen=tile_mlen, kv_layout=args.roofline_kv_layout,
        )
        step_bytes = traffic.read_bytes + traffic.write_bytes
        step_flops = decode_step_flops(dims, kv, batch)
        # The decode batch must fill the device's matrix tile: PLENA's is BLEN,
        # a GPU's is its square tensor-core tile.
        tile = device["blen"] if plena else device["sq_dim"]
        utilisation = fill_util(batch, tile)
        step_time = max(step_flops / (peak_compute * utilisation), step_bytes / peak_bw)
        watts = DEVICE_TDP_WATTS.get(name, float("nan"))
        system_watts = watts * device["count"] if watts == watts else float("nan")
        tps = batch / step_time
        rows.append(
            Row(
                model=model,
                device=("PLENA" if plena else device["label"]) + f" x{device['count']}",
                configuration=PEAK_ROOFLINE,
                precision=precision_label(precision),
                batch=batch,
                first_decode_ms=float("nan"),
                tps=tps,
                tpot_ms=step_time * 1e3,
                power_w=system_watts,
                tokens_per_joule=(
                    tps / system_watts if system_watts == system_watts else float("nan")
                ),
                area_mm2=area_mm2(hardware) if plena else float("nan"),
                evidence_tier="peak-roofline bound",
            )
        )
    return rows


def render(
    rows: list[Row],
    *,
    args,
    calibration: EmulatorCalibration,
    calibration_hash: str,
    geometries: dict[str, dict[str, int]],
) -> str:
    table_precision = build_precision(args)
    header = (
        f"{'model':<14}{'device':<11}{'configuration':<48}{'tier':<22}"
        f"{'attnW/ffnW/A/KV':<32}"
        f"{'batch':>7}{'1st dec ms':>11}{'TPS':>10}{'xA100':>7}{'%peak':>7}"
        f"{'TPOT ms':>9}{'power W':>9}{'tok/J':>8}{'array/chip mm2':>15}"
        f"{'ppl':>8}{'d ppl':>7}{'GSM8K':>7}{'IFEval':>7}{'MXblk':>7}"
    )
    a100_tps = {
        row.model: row.tps for row in rows if row.device.startswith("A100")
    }
    peak_tps = {
        row.model: row.tps
        for row in rows
        if row.device.startswith("PLENA") and row.configuration == PEAK_ROOFLINE
    }

    def cell(value: float, spec: str) -> str:
        width = int(spec.lstrip(">").split(".")[0])
        if value != value:
            return format("-", f">{width}")
        return format(value, spec)

    lines = ["=" * len(header), header, "-" * len(header)]
    for row in rows:
        reference = a100_tps.get(row.model, float("nan"))
        peak = peak_tps.get(row.model, float("nan"))
        realised = (
            row.tps / peak
            if peak == peak and row.configuration != PEAK_ROOFLINE
            else float("nan")
        )
        lines.append(
            f"{row.model:<14}{row.device:<11}{row.configuration:<48}"
            f"{row.evidence_tier:<22}{row.precision:<32}{row.batch:>7,}"
            f"{cell(row.first_decode_ms, '>11.1f')}"
            f"{cell(row.tps, '>10.1f')}"
            f"{cell(row.tps / reference, '>7.2f')}"
            f"{cell(realised * 100, '>7.1f')}"
            f"{cell(row.tpot_ms, '>9.1f')}"
            f"{cell(row.power_w, '>9.1f')}"
            f"{cell(row.tokens_per_joule, '>8.2f')}"
            f"{cell(row.area_mm2, '>15.3f')}"
            f"{cell(row.accuracy.get('ppl', float('nan')), '>8.2f')}"
            f"{cell(row.accuracy.get('ppl', float('nan')) - row.accuracy.get('ppl_bf16', float('nan')), '>7.2f')}"
            f"{cell(row.accuracy.get('gsm8k', float('nan')), '>7.3f')}"
            f"{cell(row.accuracy.get('ifeval', float('nan')), '>7.3f')}"
            f"{cell(row.accuracy.get('block', float('nan')), '>7.0f')}"
        )
    lines += [
        "=" * len(header),
        "  PLENA instruction rows are unsealed analytic sensitivities. They have",
        "  neither headline-geometry RTL evidence nor the complete timing anchors",
        "  required for a publication or deployment ranking.",
        f"  Emulator calibration: {calibration.calibration_id} ({calibration.label}),",
        f"  SHA-256 {calibration_hash}; {calibration.configuration}; worst stage",
        f"  {calibration.worst_stage_error:.1%}, coverage {1.0 - calibration.uncovered_fraction:.1%}.",
        "  This calibration is transactional-emulator evidence at its recorded",
        "  geometry, not RTL evidence and not a headline-geometry measurement.",
        "  Peak-roofline rows: batch / max(compute, memory) from peak compute and",
        "  peak bandwidth. They assume free unpack/requantisation of the listed",
        "  low-precision storage formats and are not measured baselines.",
        "  First-decode is the first decode batch-step at the initial cache length;",
        "  it excludes prefill, handoff, admission, and queueing and is not TTFT.",
        "  TPOT is total decode-loop time / output positions; TPS is batch / TPOT.",
        "  xA100 is throughput against the A100 peak roofline. Compare peak with",
        "  peak: the PLENA peak-roofline row is the like-for-like number, and",
        "  %peak is how much of it each modelled configuration realises.",
        "  Power: PLENA is an analytic sensitivity anchored to the MemExplorer",
        "  literature model output, not measured or trace-calibrated power.",
        "  GPUs use cited TDP x device count (H100 at its 700 W configured",
        "  maximum), so GPU tok/J is a peak/TDP proxy rather than an",
        "  achieved efficiency or a formal energy-efficiency bound.",
        "  PLENA system power assumes traffic and array activity are balanced",
        "  evenly across the tensor-parallel chips.",
        "  PLENA uses the smallest whole-chip count meeting the GPU reference HBM",
        "  capacity; the reported batch reflects any resulting excess capacity.",
        "  The unsupported single-q row prices its source-emitted M_BMV_WO and",
        "  M_MV_WO drains at the serialized MLEN-wide cost. This resolves the",
        "  former inversion, but missing decoder arms still make it invalid.",
        "  Co-design rows compare with the batched-q + KV read per-head",
        "  analytic baseline only.",
        "  Batched-q is opcode-compatible and the reference decode lowering emits",
        "  it, but no whole-layer run exists at the headline geometry, so no row",
        "  is labelled deployable.",
        "  Drain overlap is an analytic co-design bound. Its raw register cost is",
        "  derived below from the live 32-bit accumulators. No synthesis result",
        "  exists and the default is off.",
        "  Selector timing + KV read 1x is a combined analytic co-design bound:",
        "  both selector compute timing and KV traffic differ from the baseline.",
        "  The selector, hkv>1 loop hoist, and complete decoder are",
        "  emulator-verified at MLEN=64, HLEN=16 for hkv=2/4 with exact HBM",
        "  reconciliation, but not at the headline Qwen",
        "  geometry. Its headline compute model also differs from the per-head row,",
        "  and the active MXFP RTL profile does not implement the selector.",
        "  Co-design area, power, and tok/J are blank until the added storage and",
        "  control are synthesised and power-characterised. Other PLENA area cells",
        "  are a per-chip analytic array proxy, not full-chip, system aggregate,",
        "  or synthesised area.",
        "  Precision lists element formats as attention-weight / FFN-weight /",
        "  activation / KV. Effective HBM bits include the shared scale overhead:",
        f"  block {table_precision['block_size']} accounts "
        f"{table_precision['attn_bits']:.2f}/{table_precision['ffn_bits']:.2f}/"
        f"{table_precision['kv_bits']:.2f} bits for attention-weight/FFN-weight/KV.",
        "  Accuracy is intentionally blank: no provenance-valid Qwen3-32B point exists.",
        "  Prefill remains unquantised BF16 on a separate chip.",
    ]
    for model, dims in geometries.items():
        system_chips = next(
            int(row.device.rsplit("x", 1)[1])
            for row in rows
            if row.model == model and row.device.startswith("PLENA x")
        )
        hlen = args.hlen or dims["head_dim"]
        if args.mlen % hlen:
            lines.append(f"  {model}: HLEN={hlen} does not divide MLEN={args.mlen}.")
            continue
        broadcast_heads = args.mlen // hlen
        point = evaluate_fp_sram(
            RTL_FP_SRAM_DEPTH,
            broadcast_heads=broadcast_heads,
            kv_heads=dims["kv_heads"],
            blen=args.blen,
        )
        one_read = depth_for_reuse(broadcast_heads, args.blen, dims["kv_heads"])
        extra_bytes = max(0, one_read - RTL_FP_SRAM_DEPTH) * FP_SLOT_BITS / 8
        if point is None:
            lines.append(
                f"  {model}: MLEN={args.mlen}, HLEN={hlen}, BLEN={args.blen}; "
                f"the RTL's {RTL_FP_SRAM_DEPTH}-slot scalar SRAM cannot hold one query block."
            )
            continue
        lines.append(
            f"  {model}: MLEN={args.mlen}, HLEN={hlen}, BLEN={args.blen}, "
            f"FP depth={RTL_FP_SRAM_DEPTH}; {point.groups_live} groups fit "
            f"({point.live_slots} slots, KV read {point.kv_read_factor}x)."
        )
        lines.append(
            f"  One-read traffic requires {one_read} slots "
            f"({extra_bytes:,.0f} B/chip, {extra_bytes * system_chips:,.0f} B "
            f"across the {system_chips}-chip system of raw scalar storage)."
        )
        accumulator_bytes = (
            (broadcast_heads + 1) * args.blen * args.blen * 32 // 8
        )
        lines.append(
            f"  Duplicating packed plus plain accumulator state is "
            f"{accumulator_bytes:,} B/chip "
            f"({accumulator_bytes * system_chips:,} B across the "
            f"{system_chips}-chip system) of raw registers before implementation overhead."
        )
    return "\n".join(lines)


def build_results_document(
    *,
    args,
    requested_arguments: dict[str, object],
    rows: list[Row],
    models: list[str],
    geometries: dict[str, dict[str, int]],
    calibration,
    calibration_hash: str,
) -> dict[str, object]:
    """Build a strict, content-addressed publication artifact."""
    semantic_arguments = {
        name: _repository_path(value) if name in _PATH_ARGUMENTS else value
        for name, value in requested_arguments.items()
        if name != "json"
    }
    resolved_geometry = {
        model: {
            **geometries[model],
            "mlen": args.mlen,
            "blen": args.blen,
            "hlen": args.hlen or geometries[model]["head_dim"],
            "mx_block": args.block,
        }
        for model in models
    }
    resolved_system = {}
    for model in models:
        instruction_row = next(
            row
            for row in rows
            if row.model == model and row.device.startswith("PLENA x")
        )
        resolved_system[model] = {
            "plena_chips": int(instruction_row.device.rsplit("x", 1)[1]),
            "instruction_batch": instruction_row.batch,
            "sizing_policy": (
                "smallest whole PLENA chip count meeting aggregate A100 x4 HBM; "
                "largest batch fitting that capacity"
            ),
            "reference_capacity_bytes": (
                DEVICES["a100"]["hbm_gb"] * DEVICES["a100"]["count"] * 1e9
            ),
        }
    instruction_points = [
        {
            "batch_packed_attention": False,
            "kv_layout": DENSE_COMPILER,
            "timing_mode": RTL_SERIALIZED,
            "evidence_tier": "analytic unsupported",
        },
        {
            "batch_packed_attention": True,
            "kv_layout": DENSE_COMPILER,
            "timing_mode": RTL_SERIALIZED,
            "evidence_tier": "analytic baseline",
        },
        {
            "batch_packed_attention": True,
            "kv_layout": DENSE_COMPILER,
            "timing_mode": DRAIN_OVERLAPPED,
            "evidence_tier": "analytic co-design",
        },
        {
            "batch_packed_attention": True,
            "kv_layout": DENSE_SELECTOR,
            "timing_mode": RTL_SERIALIZED,
            "evidence_tier": "analytic co-design",
        },
    ]
    model_inputs = {
        model: {
            "path": _repository_path(resolve_model_path(model, args.model_lib)),
            "sha256": sha256_file(Path(resolve_model_path(model, args.model_lib))),
        }
        for model in models
    }
    body = {
        "schema": "plena-decode-results",
        "schema_version": 1,
        "metric_definitions": {
            "first_decode_ms": (
                "first decode batch-step at the initial cache length; excludes "
                "prefill, handoff, admission, and queueing"
            ),
            "tpot_ms": "total decode-loop time / output positions",
            "tps": "batch * output positions / total decode-loop time",
        },
        "evaluation": {
            "arguments": _json_safe(semantic_arguments),
            "resolved_geometry": resolved_geometry,
            "resolved_system": resolved_system,
            "instruction_points": instruction_points,
            "sequence_lengths": {
                "input_tokens": args.input_seq,
                "output_tokens": args.output_seq,
            },
            "kv_cache_layout": {
                "instruction_baseline": DENSE_COMPILER,
                "selector_co_design": DENSE_SELECTOR,
                "roofline": args.roofline_kv_layout,
            },
        },
        "device_references": _device_reference_provenance(),
        "model_assumptions": {
            "plena_power": {
                "evidence_tier": (
                    "analytic sensitivity anchored to literature-reported model output"
                ),
                "reference_configuration": REFERENCE_CONFIGURATION,
                "hbm_energy": asdict(hbm_technology(args.hbm_gen)),
                "system_scaling": (
                    "per-chip capacity and evenly divided system traffic/activity "
                    "are multiplied once by the resolved chip count"
                ),
            }
        },
        "provenance": {
            "emulator_calibration": {
                "calibration_id": calibration.calibration_id,
                "label": calibration.label,
                "path": _repository_path(args.emulator_calibration),
                "sha256": calibration_hash,
                "execution_contract": calibration.execution_contract.to_dict(),
                "measured_input_hashes": {
                    name: digest
                    for name, digest in calibration.provenance_hashes
                    if not name.startswith("analytic_source:")
                },
                "analytic_source_hashes": {
                    name: digest
                    for name, digest in calibration.provenance_hashes
                    if name.startswith("analytic_source:")
                },
            },
            "inputs": {
                "config": {
                    "path": _repository_path(args.config),
                    "sha256": sha256_file(args.config),
                },
                "isa_library": {
                    "path": _repository_path(args.isa_lib),
                    "sha256": sha256_file(args.isa_lib),
                },
                "models": model_inputs,
            },
            "sources": results_source_provenance(),
        },
        "rows": [asdict(row) for row in rows],
    }
    safe_body = _json_safe(body)
    safe_body["content_hash"] = _canonical_hash(safe_body)
    return safe_body


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="qwen3-32b")
    parser.add_argument(
        "--model-lib", default=str(_HERE.parent.parent / "compiler" / "doc" / "Model_Lib")
    )
    parser.add_argument("--config", default=str(_HERE.parent.parent / "plena_settings.toml"))
    parser.add_argument("--isa-lib", default=str(_HERE / "customISA_lib.json"))
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--input-seq", type=int, default=256)
    parser.add_argument("--output-seq", type=int, default=16384)
    parser.add_argument("--mlen", type=int, default=1024)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument(
        "--hlen", type=int, default=0,
        help="head lane width; 0 follows each model's head dimension",
    )
    parser.add_argument("--chips", type=int, default=0)
    parser.add_argument("--hbm-gen", default="HBM3E")
    parser.add_argument("--hbm-channels", type=int, default=32)
    parser.add_argument("--w-fmt", default="mxint")
    parser.add_argument("--kv-fmt", default="mxint")
    parser.add_argument("--attn-w", default="4")
    parser.add_argument("--ffn-w", default="4")
    parser.add_argument("--kv", default="4")
    parser.add_argument("--block", type=int, default=8)
    parser.add_argument("--m-bits", type=int, default=0)
    parser.add_argument("--density-exp", type=float, default=DENSITY_EXP)
    parser.add_argument(
        "--roofline-kv-layout", default=DENSE_SELECTOR,
        help="KV traffic mode for the peak-roofline rows",
    )
    parser.add_argument(
        "--emulator-calibration",
        type=Path,
        default=_HERE / "calibration" / "decode_kv1024.json",
        help="content-addressed analytic-versus-emulator evidence to report",
    )
    parser.add_argument("--json", type=Path, help="also write the rows here")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    requested_arguments = dict(vars(args))

    if args.block % args.blen:
        raise SystemExit(
            f"MX block size {args.block} must be divisible by BLEN {args.blen} "
            "for the current MXINT RTL datapath"
        )
    if not args.emulator_calibration.is_file():
        raise SystemExit(
            f"emulator calibration is missing: {args.emulator_calibration}; "
            "regenerate the decode dump and calibration before reporting results"
        )
    try:
        calibration = EmulatorCalibration.load(args.emulator_calibration)
        validate_calibration_inputs(calibration, args)
    except ValueError as error:
        raise SystemExit(
            f"emulator calibration is stale or invalid: {error}; regenerate it "
            "from a fresh emulator run"
        ) from error
    if not calibration.passed:
        raise SystemExit(
            f"emulator calibration {calibration.calibration_id} does not pass its "
            "fixed stage, total, and coverage gates"
        )
    calibration_hash = sha256_file(args.emulator_calibration)

    reference_capacity = DEVICES["a100"]["hbm_gb"] * DEVICES["a100"]["count"] * 1e9
    rows: list[Row] = []
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    geometries = {
        model: load_model_dims(resolve_model_path(model, args.model_lib))
        for model in models
    }
    for model in models:
        chips, batch = capacity_match(args, model, reference_capacity)
        for batch_packed, kv_layout, timing_mode in (
            (False, DENSE_COMPILER, RTL_SERIALIZED),
            (True, DENSE_COMPILER, RTL_SERIALIZED),
            (True, DENSE_COMPILER, DRAIN_OVERLAPPED),
            (True, DENSE_SELECTOR, RTL_SERIALIZED),
        ):
            rows.append(
                modelled_row(
                    args, model, chips, batch, batch_packed, kv_layout, timing_mode
                )
            )
        rows.extend(roofline_rows(args, model, chips))

    print(
        render(
            rows,
            args=args,
            calibration=calibration,
            calibration_hash=calibration_hash,
            geometries=geometries,
        )
    )
    if args.json:
        document = build_results_document(
            args=args,
            requested_arguments=requested_arguments,
            rows=rows,
            models=models,
            geometries=geometries,
            calibration=calibration,
            calibration_hash=calibration_hash,
        )
        args.json.write_text(
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        print(f"\nrows -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
