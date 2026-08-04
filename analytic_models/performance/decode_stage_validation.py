"""Analytical vs emulator cycles per decode stage.

Attributes the transactional emulator's per-instruction samples (`--op-stats`)
to the pipeline stages of one decode layer and compares each stage against the
analytic model evaluated at the same array geometry and the same operand widths
the program actually computes.

The emulator cycles are split by the opcode class executing when they are
observed. Those values include dependency stalls charged to that opcode; in
particular, the HBM column is time observed at HBM opcodes rather than all
dynamic memory service. The stage totals remain the complete timing evidence.

The emulator program is the decoder_decode testbench: one layer, one new token
per sequence, checked against a PyTorch golden.

Usage:
    decode_stage_validation.py --asm ASM --op-stats JSONL --settings TOML
                               --isa-lib JSON [--kv-size N]
"""

from __future__ import annotations

import argparse
import json
import math
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from analytic_models.disagg_serve.memory import (
    CALIBRATED_PIN_RATES_GBPS,
    RequestLatencyModel,
    load_request_observations,
)
from compiler.aten.execution_trace import (
    NO_DMA,
    CompilationArtifact,
    SECTION_TO_STAGE,
    build_execution_trace,
)

try:
    from .decode_cost_model import DecodeCostModel
    from .decode_timing import DRAIN_OVERLAPPED, RTL_SERIALIZED, TIMING_MODES
    from .disagg_decode import decode_token_components
    from .perf_model import PerfModel, load_hardware_config_from_toml
    from .compiler_trace_timing import (
        HBMOperatingPoint,
        MATRIX_ENGINE,
        VECTOR_ENGINE,
        RequestModelStageMemoryPricer,
        request_memory_sidecar_from_compiler,
    )
except ImportError:
    from decode_cost_model import DecodeCostModel
    from decode_timing import DRAIN_OVERLAPPED, RTL_SERIALIZED, TIMING_MODES
    from disagg_decode import decode_token_components
    from perf_model import PerfModel, load_hardware_config_from_toml
    from compiler_trace_timing import (
        HBMOperatingPoint,
        MATRIX_ENGINE,
        VECTOR_ENGINE,
        RequestModelStageMemoryPricer,
        request_memory_sidecar_from_compiler,
    )
from packed_kv import DENSE_COMPILER
from transactional_emulator.testbench.emulator_runner import (
    acquire_build_directory,
    acquire_emulator_execution,
    validate_emulator_run_receipt,
)

try:
    from .emulator_calibration import (
        EMULATOR_CALIBRATION_STAGES,
        EMULATOR_PRECISION_ROLES,
        EMULATOR_STAGE_ERROR_LIMIT,
        EMULATOR_TOTAL_ERROR_LIMIT,
        EMULATOR_UNCOVERED_FRACTION_LIMIT,
        EmulatorCalibration,
        EmulatorExecutionContract,
        StageCalibration,
        calibration_source_hashes,
        sha256_file,
    )
except ImportError:
    from emulator_calibration import (
        EMULATOR_CALIBRATION_STAGES,
        EMULATOR_PRECISION_ROLES,
        EMULATOR_STAGE_ERROR_LIMIT,
        EMULATOR_TOTAL_ERROR_LIMIT,
        EMULATOR_UNCOVERED_FRACTION_LIMIT,
        EmulatorCalibration,
        EmulatorExecutionContract,
        StageCalibration,
        calibration_source_hashes,
        sha256_file,
    )

PICOSECONDS_PER_CYCLE = 1000
COMPILATION_ARTIFACT_NAME = "compilation_artifact.json"
DEFAULT_REQUEST_MEMORY_CALIBRATION = (
    Path(__file__).resolve().parents[1]
    / "disagg_serve"
    / "calibration_dma_requests.csv"
)
PUBLICATION_STAGE_ERROR_TARGET = EMULATOR_STAGE_ERROR_LIMIT
PUBLICATION_COVERAGE_TARGET = 1.0 - EMULATOR_UNCOVERED_FRACTION_LIMIT

# The compiler identifies the physical operations below as separate sequential
# stages. The validation table groups only adjacent operations that the emulator
# reports as one canonical stage, and sums their individual costs. This preserves
# the stage composition contract instead of creating a larger, more optimistic
# overlap window.
TRACE_STAGE_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Activation load + RMSNorm", ("Activation load", "RMSNorm")),
    (
        "Q/K/V + W_O projection + RoPE",
        ("Q/K/V + W_O projection", "RoPE"),
    ),
    ("KV store", ("KV store",)),
    ("Flash attention", ("Flash attention",)),
    ("Residual add", ("Residual add",)),
    ("FFN (gate/up/down)", ("FFN (gate/up/down)",)),
    ("LM head", ("LM head",)),
)

# Closed-form component groupings are retained only as a diagnostic. Unlike the
# compiler artifact, that model has no separate KV-store term and therefore
# cannot emit a complete calibration artifact.
ANALYTIC_STAGE_KEYS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Activation load + RMSNorm",
        ("Embedding lookup", "RMSNorm (x2)", "Final RMSNorm"),
    ),
    (
        "Q/K/V + W_O projection + RoPE",
        ("Q/K/V proj + RoPE", "Output projection (W_O)"),
    ),
    ("KV store", ()),
    ("Flash attention", ("Flash attention",)),
    ("Residual add", ("Residual adds (x2)",)),
    ("FFN (gate/up/down)", ("FFN (gate/up/down)",)),
    ("LM head", ("LM head",)),
)
if tuple(stage for stage, _keys in ANALYTIC_STAGE_KEYS) != EMULATOR_CALIBRATION_STAGES:
    raise RuntimeError("analytic and calibration stage contracts disagree")
if tuple(stage for stage, _keys in TRACE_STAGE_KEYS) != EMULATOR_CALIBRATION_STAGES:
    raise RuntimeError("compiler trace and calibration stage contracts disagree")

_TRACE_STAGE_TO_CANONICAL = {
    raw_stage: canonical_stage
    for canonical_stage, raw_stages in TRACE_STAGE_KEYS
    for raw_stage in raw_stages
}


@lru_cache(maxsize=4)
def _fit_request_latency_model(
    calibration_path: str,
    calibration_sha256: str,
) -> RequestLatencyModel:
    """Reuse an immutable request fit without accepting changed evidence."""

    path = Path(calibration_path)
    if sha256_file(path) != calibration_sha256:
        raise ValueError("request-memory calibration changed while it was being priced")
    return RequestLatencyModel.fit(load_request_observations(path))


@dataclass(frozen=True)
class StageValidationSummary:
    """Worst-stage and coverage result for a decode-stage comparison."""

    stage_errors: tuple[tuple[str, float], ...]
    worst_stage: str | None
    worst_stage_error: float
    coverage: float
    modelled_cycles: int
    measured_cycles: int
    measured_layer_cycles: int

    def meets_target(
        self,
        *,
        max_worst_stage_error: float = 0.05,
        min_coverage: float = 0.99,
    ) -> bool:
        return (
            self.worst_stage is not None
            and abs(self.worst_stage_error) <= max_worst_stage_error
            and self.coverage >= min_coverage
        )


def summarize_stage_validation(
    modelled: dict[str, int],
    measured: dict[str, dict[str, int]],
) -> StageValidationSummary:
    """Summarize canonical stages without hiding the worst component error."""

    stage_errors: list[tuple[str, float]] = []
    modelled_cycles = 0
    measured_cycles = 0
    for stage, _keys in ANALYTIC_STAGE_KEYS:
        if stage not in modelled or stage not in measured:
            continue
        observed = sum(measured[stage].values())
        if observed <= 0:
            continue
        predicted = int(modelled[stage])
        stage_errors.append((stage, (predicted - observed) / observed))
        modelled_cycles += predicted
        measured_cycles += observed
    measured_layer_cycles = sum(
        sum(bucket.values()) for bucket in measured.values()
    )
    coverage = (
        measured_cycles / measured_layer_cycles
        if measured_layer_cycles
        else float("nan")
    )
    if stage_errors:
        worst_stage, worst_error = max(
            stage_errors,
            key=lambda item: abs(item[1]),
        )
    else:
        worst_stage, worst_error = None, float("nan")
    return StageValidationSummary(
        stage_errors=tuple(stage_errors),
        worst_stage=worst_stage,
        worst_stage_error=worst_error,
        coverage=coverage,
        modelled_cycles=modelled_cycles,
        measured_cycles=measured_cycles,
        measured_layer_cycles=measured_layer_cycles,
    )


def require_complete_calibration_stages(
    modelled: dict[str, int], measured: dict[str, dict[str, int]]
) -> None:
    """Refuse to emit calibration from a partial analytic/measured stage set."""
    expected = set(EMULATOR_CALIBRATION_STAGES)
    missing_modelled = sorted(expected - set(modelled))
    missing_measured = sorted(expected - set(measured))
    unexpected_modelled = sorted(set(modelled) - expected)
    if missing_modelled or missing_measured or unexpected_modelled:
        details = []
        if missing_modelled:
            details.append("modelled missing " + ", ".join(missing_modelled))
        if missing_measured:
            details.append("measured missing " + ", ".join(missing_measured))
        if unexpected_modelled:
            details.append("modelled unexpected " + ", ".join(unexpected_modelled))
        raise ValueError(
            "calibration requires exactly all canonical decode stages: "
            + "; ".join(details)
        )

MATRIX_WRITEOUT_SUFFIX = "_WO"


def instruction_class(op: str) -> str:
    """Instruction class used to split the measured cycles."""
    if op.startswith("M_"):
        return "writeout" if op.endswith(MATRIX_WRITEOUT_SUFFIX) else "matrix"
    if op.startswith("V_"):
        return "vector"
    if op.startswith("H_"):
        return "hbm"
    return "scalar"


def _classify(section: str) -> str | None:
    for prefix, stage in SECTION_TO_STAGE:
        if section.startswith(prefix):
            if stage is None:
                return None
            return _TRACE_STAGE_TO_CANONICAL.get(stage, stage)
    return None


def instruction_stages(asm_path: Path) -> list[str]:
    """Stage name per program-counter index.

    The assembler drops comments and blank lines, so the program counter is the
    index among the remaining lines.
    """
    stages: list[str] = []
    current = "Setup"
    for line in asm_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(";"):
            stage = _classify(stripped)
            if stage is not None:
                current = stage
            continue
        stages.append(current)
    return stages


def emulator_stage_cycles(
    op_stats_path: Path, stages: list[str]
) -> dict[str, dict[str, int]]:
    """Emulator cycles per stage, split by instruction class."""
    totals: dict[str, dict[str, int]] = {}
    with op_stats_path.open() as source:
        for line in source:
            record = json.loads(line)
            if record.get("aggregate"):
                continue
            pc = record["pc"]
            stage = stages[pc] if pc < len(stages) else "Unattributed"
            bucket = totals.setdefault(stage, {})
            key = instruction_class(record["op"])
            bucket[key] = bucket.get(key, 0) + record["dt_ps"]
    return {
        stage: {
            key: value // PICOSECONDS_PER_CYCLE for key, value in bucket.items()
        }
        for stage, bucket in totals.items()
    }


def analytic_stage_cycles(
    settings_path: Path,
    isa_path: Path,
    *,
    hidden: int,
    heads: int,
    kv_heads: int,
    kv_projection_width: int,
    head_dim: int,
    inter: int,
    vocab: int,
    kv_size: int,
    batch: int,
    mlen: int,
    blen: int,
    hlen: int,
    timing_mode: str,
) -> dict[str, int]:
    """Analytic cycles per stage for one decode layer at the measured geometry."""
    hardware = load_hardware_config_from_toml(str(settings_path))
    # The emulator run fixes the array geometry; the settings file carries the
    # analytic default, so override it to the geometry actually measured.
    hardware = hardware.model_copy(
        update={
            "MLEN": mlen,
            "VLEN": mlen,
            "BLEN": blen,
            "HLEN": hlen,
            "BROADCAST_AMOUNT": mlen // hlen,
        }
    )
    perf = PerfModel(hardware, str(isa_path), timing_mode=timing_mode)
    dims = {
        "hidden": hidden,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "layers": 1,
        "inter": inter,
        "vocab": vocab,
        # The testbench layer attends to the whole cache.
        "n_full": 1,
        "n_sliding": 0,
        "sliding_window": 0,
        # The lowering zero-pads K/V out_features to a tile boundary.
        "kv_projection_width": kv_projection_width,
    }
    # The testbench calls its `s_q` shared-cache query rows "batch", but they are
    # consecutive positions in one cache rather than independent cached-q1
    # requests. It uses the dense KV path and the matrix-shaped lowering, so the
    # attention term must retain that shape without generalising it to serving.
    components = decode_token_components(
        perf, dims, kv_size, batch,
        include_lm_head=True,
        kv_layout=DENSE_COMPILER,
        batch_packed_attention=True,
    )
    # Component keys carry a " x1 layers" suffix; index them by their prefix.
    by_prefix = {key.split(" x1 layers")[0]: value for key, value in components.items()}
    return {
        stage: sum(by_prefix[key] for key in keys)
        for stage, keys in ANALYTIC_STAGE_KEYS
        if keys and all(key in by_prefix for key in keys)
    }


def compiler_trace_stage_cycles(
    asm_path: Path,
    settings_path: Path,
    isa_path: Path,
    *,
    compiler_artifact_path: Path | None = None,
    request_memory_calibration_path: Path = DEFAULT_REQUEST_MEMORY_CALIBRATION,
    hbm_generation: str = "HBM2",
    hbm_channels: int = 8,
    mlen: int,
    blen: int,
    hlen: int,
    timing_mode: str,
) -> dict[str, int]:
    """Price the exact compiler artifact with calibrated physical requests.

    Compute and request-memory time overlap only inside each compiler stage.
    Canonical validation stages then sum their constituent sequential compiler
    stages. A real DMA trace without its address-resolved compiler sidecar is
    rejected instead of reverting to aggregate HBM bandwidth.
    """

    hardware = load_hardware_config_from_toml(str(settings_path)).model_copy(
        update={
            "MLEN": mlen,
            "VLEN": mlen,
            "BLEN": blen,
            "HLEN": hlen,
            "BROADCAST_AMOUNT": mlen // hlen,
        }
    )
    perf = PerfModel(hardware, str(isa_path), timing_mode=timing_mode)
    vector_store_amount = int(
        getattr(
            hardware,
            "HBM_V_Writeback_Amount",
            hardware.HBM_V_Prefetch_Amount,
        )
    )
    assembly = asm_path.read_text(encoding="utf-8")
    artifact: CompilationArtifact | None = None
    if compiler_artifact_path is not None:
        artifact_path = Path(compiler_artifact_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"compiler compilation artifact is missing: {artifact_path}"
            )
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("compiler compilation artifact must be a JSON object")
        artifact = CompilationArtifact.from_dict(payload)
        if artifact.assembly != assembly:
            raise ValueError(
                "compiler compilation artifact assembly differs from the emulator assembly"
            )
        trace = artifact.execution_trace
    else:
        trace = build_execution_trace(
            assembly,
            mlen=mlen,
            blen=blen,
            vlen=mlen,
            hlen=hlen,
            vector_prefetch_amount=hardware.HBM_V_Prefetch_Amount,
            vector_store_amount=vector_store_amount,
        )

    expected_geometry = (mlen, blen, mlen, hlen)
    observed_geometry = (trace.mlen, trace.blen, trace.vlen, trace.hlen)
    if observed_geometry != expected_geometry:
        raise ValueError(
            "compiler compilation artifact geometry differs from the emulator run: "
            f"artifact={observed_geometry}, emulator={expected_geometry}"
        )

    has_dma = any(entry.dma_direction != NO_DMA for entry in trace.entries)
    if has_dma and (artifact is None or artifact.request_memory is None):
        raise RuntimeError(
            "compiler DMA timing requires the persisted address-resolved "
            "compilation_artifact.json"
        )

    compute_cost = DecodeCostModel.from_perf_model(perf).evaluate(trace)
    compute_by_stage = {
        stage.stage: stage.compute_cycles for stage in compute_cost.stages
    }
    memory_by_stage: dict[str, int] = {}
    matrix_by_stage: dict[str, int] = {}
    vector_by_stage: dict[str, int] = {}
    if has_dma:
        calibration_path = Path(request_memory_calibration_path)
        if not calibration_path.is_file():
            raise FileNotFoundError(
                f"request-memory calibration is missing: {calibration_path}"
            )
        generation = hbm_generation.upper()
        try:
            pin_rate_gbps = CALIBRATED_PIN_RATES_GBPS[generation]
        except KeyError as error:
            raise ValueError(
                f"request-memory calibration has no {generation} operating point"
            ) from error
        hbm = HBMOperatingPoint(
            generation=generation,
            channels=hbm_channels,
            pin_rate_gbps=pin_rate_gbps,
        )
        assert artifact is not None and artifact.request_memory is not None
        request_sidecar = request_memory_sidecar_from_compiler(
            artifact.request_memory,
            hbm,
        )
        request_sidecar.validate(trace, hbm)
        pricer = RequestModelStageMemoryPricer(
            _fit_request_latency_model(
                str(calibration_path.resolve()),
                sha256_file(calibration_path),
            )
        )
        frequency_hz = 1e12 / PICOSECONDS_PER_CYCLE
        for stage, engines in pricer.price_trace_by_engine(
            trace,
            request_sidecar,
            hbm,
        ).items():
            matrix_by_stage[stage] = math.ceil(
                engines.get(MATRIX_ENGINE, 0.0) * frequency_hz
            )
            vector_by_stage[stage] = math.ceil(
                engines.get(VECTOR_ENGINE, 0.0) * frequency_hz
            )
            memory_by_stage[stage] = matrix_by_stage[stage] + vector_by_stage[stage]

    raw_cycles = {
        stage: max(
            compute_by_stage.get(stage, 0),
            matrix_by_stage.get(stage, 0),
        )
        + vector_by_stage.get(stage, 0)
        for stage in trace.stage_order
    }
    return {
        canonical_stage: sum(raw_cycles.get(stage, 0) for stage in raw_stages)
        for canonical_stage, raw_stages in TRACE_STAGE_KEYS
        if any(stage in raw_cycles for stage in raw_stages)
    }


MANIFEST_NAME = "decode_run_manifest.json"
RUN_STATS_NAME = "rust_emulator_run_stats.json"

# Repeated source-default kv=1024 runs are deterministic. The retained floor is
# a conservative HBM-attribution/configuration-sensitivity bound observed when
# changing equivalent settings presentation or row-tile geometry; it is not a
# claim of run-to-run nondeterminism.
EMULATOR_NOISE_FLOOR = 0.0001

#: Shape the run manifest fixes, and the flag that would contradict each. Only
#: attention scales with the cache length, so a mismatched cache length shows up
#: as one broken stage and five clean ones, which reads as a modelling error in
#: that stage rather than as a mismatched comparison.
MANIFEST_DIMENSIONS = (
    ("kv_size", "kv_size", "kv-size"),
    ("inter", "inter", "inter"),
    ("vocab", "vocab", "vocab"),
    ("mlen", "geometry.mlen", "mlen"),
    ("blen", "geometry.blen", "blen"),
    ("hlen", "geometry.hlen", "hlen"),
    ("batch", "geometry.batch", "batch"),
    ("hidden", "geometry.hidden", "hidden"),
    ("head_dim", "geometry.head_dim", "head-dim"),
    ("heads", "geometry.query_heads", "heads"),
    ("kv_heads", "geometry.kv_heads", "kv-heads"),
)


def _lookup(manifest: dict, path: str):
    value = manifest
    for key in path.split("."):
        value = value[key]
    return value


def validate_compiler_artifact_binding(
    artifact_path: Path,
    asm_path: Path,
    manifest: dict,
    receipt: dict,
) -> CompilationArtifact:
    """Bind the persisted compiler artifact to the measured emulator run."""

    build_artifact = asm_path.resolve().parent / COMPILATION_ARTIFACT_NAME
    if artifact_path.resolve() != build_artifact:
        raise ValueError(
            "the compiler artifact must be the one persisted beside the emulator assembly"
        )
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("compiler compilation artifact must be a JSON object")
    artifact = CompilationArtifact.from_dict(payload)
    if artifact.assembly != asm_path.read_text(encoding="utf-8"):
        raise ValueError(
            "compiler compilation artifact assembly differs from the emulator assembly"
        )
    artifact_digest = sha256_file(artifact_path)
    if receipt.get("artifacts", {}).get("compiler_artifact_sha256") != artifact_digest:
        raise ValueError(
            "emulator receipt does not bind the requested compiler compilation artifact"
        )

    compiler_trace = manifest.get("compiler_trace")
    if not isinstance(compiler_trace, dict):
        raise ValueError("decode run manifest lacks compiler-trace provenance")
    if compiler_trace.get("schema_version") != payload.get("schema_version"):
        raise ValueError("decode run manifest names a different compiler artifact schema")
    if (
        compiler_trace.get("assembly_sha256")
        != artifact.execution_trace.assembly_sha256
    ):
        raise ValueError("decode run manifest names a different compiler assembly")
    if artifact.request_memory is None:
        raise ValueError(
            "decode compiler artifact lacks its address-resolved request-memory sidecar"
        )
    if (
        compiler_trace.get("request_memory_sidecar_sha256")
        != artifact.request_memory.sidecar_sha256
    ):
        raise ValueError(
            "decode run manifest names a different compiler request-memory sidecar"
        )
    return artifact


def run_shape(op_stats: Path, args) -> dict[str, int]:
    """The shape the dump was generated for, from the manifest beside it.

    Taking any dimension from a default rather than from the dump lets the two
    sides of the comparison describe different programs. The manifest is written
    next to the op-stats file by the same run, so it is the only thing that can
    say what was measured.
    """
    path = op_stats.resolve().parent / MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing: run "
            f"transactional_emulator/testbench/misc/decoder_decode_test.py "
            f"--kv-size N to generate a dump before validating against it."
        )
    manifest = json.loads(path.read_text())
    if manifest.get("kv_head_reuse") is not False:
        raise SystemExit(
            "the calibration model describes the default KV schedule; "
            "regenerate the dump without --kv-head-reuse"
        )
    shape = {}
    for name, location, flag in MANIFEST_DIMENSIONS:
        recorded = int(_lookup(manifest, location))
        requested = getattr(args, name)
        if requested and requested != recorded:
            raise SystemExit(
                f"the dump was generated with {name}={recorded} but "
                f"--{flag}={requested} was requested. Regenerate the dump for "
                f"that shape first; comparing an analytic model at one shape "
                f"against a dump at another is not a validation."
            )
        shape[name] = recorded
    return shape


def calibration_execution_contract(
    receipt: dict,
    manifest: dict,
    settings_path: Path,
    timing_mode: str,
) -> EmulatorExecutionContract:
    """Bind the comparison to the emulator behavior and precision it measured."""
    behavior = receipt.get("behavior_config")
    if not isinstance(behavior, dict):
        raise ValueError("emulator receipt lacks a behavior configuration")
    command = receipt.get("command")
    if not isinstance(command, list) or "--blocking-prefetch" in command:
        raise ValueError(
            "calibration timing requires the normal asynchronous-prefetch emulator run"
        )
    if timing_mode not in {RTL_SERIALIZED, DRAIN_OVERLAPPED}:
        raise ValueError(
            f"timing mode {timing_mode!r} has no matching emulator execution mode"
        )
    expected_overlap = timing_mode == DRAIN_OVERLAPPED
    if behavior.get("DRAIN_OVERLAPPED") is not expected_overlap:
        raise ValueError(
            "requested timing mode disagrees with the emulator drain behavior"
        )

    try:
        manifest_depth = int(manifest["geometry"]["fp_sram_depth"])
        receipt_depth = int(behavior["FP_SRAM_DEPTH"])
        hbm_gen = str(behavior["HBM_GEN"]).upper()
        hbm_channels = int(behavior["HBM_CHANNELS"])
        receipt_precision = behavior["PRECISION"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("emulator receipt lacks the calibration execution fields") from error
    if not isinstance(receipt_precision, dict):
        raise ValueError("emulator receipt lacks its precision contract")

    with settings_path.open("rb") as source:
        settings = tomllib.load(source)
    try:
        analytic_depth = int(settings["ANALYTIC"]["CONFIG"]["FP_SRAM_DEPTH"]["value"])
        analytic_precision = settings["ANALYTIC"]["PRECISION"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("analytic settings lack the calibration execution fields") from error
    if manifest_depth != receipt_depth or analytic_depth != receipt_depth:
        raise ValueError(
            "manifest, analytic, and emulator FP SRAM depths do not match"
        )

    missing_precision = [
        name
        for name in EMULATOR_PRECISION_ROLES
        if name not in receipt_precision or name not in analytic_precision
    ]
    if missing_precision:
        raise ValueError(
            "calibration precision contract is incomplete: "
            + ", ".join(missing_precision)
        )
    mismatched_precision = [
        name
        for name in EMULATOR_PRECISION_ROLES
        if receipt_precision[name] != analytic_precision[name]
    ]
    if mismatched_precision:
        raise ValueError(
            "analytic and emulator precision contracts disagree: "
            + ", ".join(mismatched_precision)
        )

    return EmulatorExecutionContract(
        timing_mode=timing_mode,
        drain_overlapped=expected_overlap,
        fp_sram_depth=receipt_depth,
        hbm_gen=hbm_gen,
        hbm_channels=hbm_channels,
        precision=tuple(
            (
                name,
                json.dumps(
                    receipt_precision[name],
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            )
            for name in EMULATOR_PRECISION_ROLES
        ),
    )


def assert_op_stats_current(
    op_stats: Path,
    asm_path: Path,
    settings_path: Path,
) -> dict:
    """Refuse an op-stats file left behind by an earlier run.

    The successful-run receipt binds the assembly, settings, run manifest,
    op-stats, inputs, and copied VRAM dump by content hash. The instruction
    records must also sum to both their aggregate and the independently logged
    emulator latency.
    """
    build_dir = op_stats.resolve().parent
    try:
        receipt = validate_emulator_run_receipt(
            build_dir,
            settings_file=settings_path,
        )
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    expected_asm = build_dir / "generated_asm_code.asm"
    if (
        asm_path.resolve() != expected_asm
        or sha256_file(asm_path)
        != receipt["artifacts"].get("asm_source_sha256")
    ):
        raise SystemExit("the requested assembly is not the emulator-run assembly")

    individual_total_ps = 0
    aggregate_total_ps = None
    aggregate_seen = False
    with op_stats.open() as source:
        for line_number, line in enumerate(source, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"op-stats line {line_number} is not valid JSON"
                ) from error
            if record.get("aggregate"):
                if aggregate_seen:
                    raise SystemExit("op-stats contains multiple aggregate records")
                aggregate_seen = True
                aggregate_total_ps = int(record["total_dt_ps"])
                continue
            if aggregate_seen:
                raise SystemExit("op-stats contains instructions after its aggregate")
            individual_total_ps += int(record["dt_ps"])
    if aggregate_total_ps is None:
        raise SystemExit("op-stats is missing its aggregate record")
    if aggregate_total_ps != individual_total_ps:
        raise SystemExit(
            "op-stats instruction times do not sum to its aggregate record"
        )
    measured = aggregate_total_ps // PICOSECONDS_PER_CYCLE
    recorded = receipt["sim_latency_ns"]
    if measured != int(recorded):
        raise SystemExit(
            f"{op_stats} totals {measured:,} cycles but the emulator run beside "
            f"it took {int(recorded):,}. The op-stats file is from a different "
            f"run; regenerate the dump with the emulator asked for --op-stats "
            f"before validating against it."
        )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    """The command line. Every shape dimension defaults to the run manifest.

    A non-zero default here would let the comparison silently describe a
    different program from the one the dump was generated for.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asm", type=Path, required=True)
    parser.add_argument("--op-stats", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--isa-lib", type=Path, required=True)
    parser.add_argument(
        "--compiler-artifact",
        type=Path,
        help="persisted compilation_artifact.json; defaults to the artifact "
             "beside --asm",
    )
    parser.add_argument(
        "--request-memory-calibration",
        type=Path,
        default=DEFAULT_REQUEST_MEMORY_CALIBRATION,
        help="request-level Ramulator calibration used to price the compiler "
             "DMA sidecar",
    )
    # Every dimension below defaults to the run manifest. A value is accepted
    # only when it agrees with what the dump was generated for.
    parser.add_argument("--kv-size", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--heads", type=int, default=0)
    parser.add_argument("--kv-heads", type=int, default=0)
    parser.add_argument("--head-dim", type=int, default=0)
    parser.add_argument("--inter", type=int, default=0)
    parser.add_argument("--vocab", type=int, default=0)
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--mlen", type=int, default=0)
    parser.add_argument("--blen", type=int, default=0)
    parser.add_argument("--hlen", type=int, default=0)
    parser.add_argument(
        "--timing-mode",
        choices=TIMING_MODES,
        default=RTL_SERIALIZED,
        help="matrix timing contract; must match the contract the dump was "
             "generated under",
    )
    parser.add_argument(
        "--cost-source",
        choices=("compiler_trace", "closed_form"),
        default="compiler_trace",
        help="use dynamic counts from emitted assembly (default), or retain "
             "the tensor-shape formula as a diagnostic comparison",
    )
    parser.add_argument(
        "--emit-calibration",
        type=Path,
        help="write the analytic-vs-emulator calibration artifact here",
    )
    parser.add_argument(
        "--logical-kv-width",
        action="store_true",
        help="cost K/V projection at kv_heads*head_dim. The emitted program "
             "pads K/V out_features to MLEN, so the default costs the padded "
             "width the program actually computes.",
    )
    return parser


def _run_validation(args: argparse.Namespace) -> int:
    receipt = assert_op_stats_current(args.op_stats, args.asm, args.settings)
    shape = run_shape(args.op_stats, args)
    for name, value in shape.items():
        setattr(args, name, value)
    manifest = json.loads(
        (args.op_stats.resolve().parent / MANIFEST_NAME).read_text()
    )
    try:
        execution_contract = calibration_execution_contract(
            receipt, manifest, args.settings, args.timing_mode
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    persisted_artifact = args.asm.resolve().parent / COMPILATION_ARTIFACT_NAME
    compiler_artifact_path = (
        args.compiler_artifact.resolve()
        if args.compiler_artifact is not None
        else (persisted_artifact if persisted_artifact.is_file() else None)
    )
    if compiler_artifact_path is not None:
        try:
            validate_compiler_artifact_binding(
                compiler_artifact_path,
                args.asm,
                manifest,
                receipt,
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise SystemExit(str(error)) from error

    # The emitted program pads the K/V projection's out_features to MLEN, but
    # attention still runs over the model's real KV heads. Widening kv_heads
    # would change the GQA structure of the attention term as well, so the
    # padding is applied to the projection width alone.
    kv_projection_width = (
        args.kv_heads * args.head_dim if args.logical_kv_width else args.mlen
    )

    stages = instruction_stages(args.asm)
    measured = emulator_stage_cycles(args.op_stats, stages)
    if args.cost_source == "compiler_trace":
        if args.logical_kv_width:
            raise SystemExit(
                "--logical-kv-width is a closed-form sensitivity and cannot "
                "replace the physical work in emitted compiler assembly"
            )
        try:
            modelled = compiler_trace_stage_cycles(
                args.asm,
                args.settings,
                args.isa_lib,
                compiler_artifact_path=compiler_artifact_path,
                request_memory_calibration_path=args.request_memory_calibration,
                hbm_generation=execution_contract.hbm_gen,
                hbm_channels=execution_contract.hbm_channels,
                mlen=args.mlen,
                blen=args.blen,
                hlen=args.hlen,
                timing_mode=args.timing_mode,
            )
        except (
            FileNotFoundError,
            KeyError,
            RuntimeError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise SystemExit(str(error)) from error
    else:
        modelled = analytic_stage_cycles(
            args.settings,
            args.isa_lib,
            hidden=args.hidden,
            heads=args.heads,
            kv_heads=args.kv_heads,
            kv_projection_width=kv_projection_width,
            head_dim=args.head_dim,
            inter=args.inter,
            vocab=args.vocab,
            kv_size=args.kv_size,
            batch=args.batch,
            mlen=args.mlen,
            blen=args.blen,
            hlen=args.hlen,
            timing_mode=args.timing_mode,
        )

    width = "logical" if args.logical_kv_width else "padded to MLEN"
    print(
        f"Decode layer: kv={args.kv_size} batch={args.batch} hidden={args.hidden} "
        f"inter={args.inter} MLEN={args.mlen} BLEN={args.blen} HLEN={args.hlen}"
    )
    print(
        f"K/V projection width: {width} ({kv_projection_width}); "
        f"attention kv_heads={args.kv_heads}\n"
    )
    print(
        "Cost source: "
        + (
            "compiler structured execution trace"
            if args.cost_source == "compiler_trace"
            else "closed-form tensor-shape diagnostic"
        )
        + "\n"
    )

    columns = (
        f"{'stage':<38}{'analytic':>10}{'measured':>10}{'error':>8}"
        f"{'matrix':>10}{'writeout':>10}{'vector':>9}{'scalar':>9}{'HBM':>7}"
    )
    print(columns)
    print("-" * len(columns))

    measured_total = 0
    for stage, _ in ANALYTIC_STAGE_KEYS:
        if stage not in modelled or stage not in measured:
            continue
        bucket = measured[stage]
        analytic = modelled[stage]
        total = sum(bucket.values())
        error = (analytic - total) / total if total else float("nan")
        measured_total += total
        print(
            f"{stage:<38}{analytic:>10,}{total:>10,}{error:>7.1%}"
            f"{bucket.get('matrix', 0):>10,}{bucket.get('writeout', 0):>10,}"
            f"{bucket.get('vector', 0):>9,}{bucket.get('scalar', 0):>9,}"
            f"{bucket.get('hbm', 0):>7,}"
        )
    print("-" * len(columns))

    summary = summarize_stage_validation(modelled, measured)
    layer_total = summary.measured_layer_cycles
    covered = summary.coverage
    if summary.worst_stage is not None:
        worst_stage = summary.worst_stage
        worst_error = summary.worst_stage_error
        print(
            f"\nWorst stage: {worst_stage} at {worst_error:+.1%}; this is the "
            f"error bound to report independently of the stage mix."
        )
        if worst_stage == "FFN (gate/up/down)" and worst_error < 0:
            print(
                "The FFN loop bookkeeping and prefetch issue counts are "
                "source-derived; class columns include dependency stalls at "
                "the executing opcode, while the stage total preserves all "
                "dynamic DMA completion and consumer-wait time."
            )
    print(f"Those stages are {covered:.1%} of the measured decode layer.")
    print(
        f"The retained HBM-attribution/configuration-sensitivity floor is "
        f"{EMULATOR_NOISE_FLOOR:.2%} of a stage, so a stage error is only "
        f"distinguishable from zero above that."
    )

    other = {
        stage: sum(bucket.values())
        for stage, bucket in sorted(measured.items())
        if stage not in dict(ANALYTIC_STAGE_KEYS)
    }
    if other:
        print("\nStages with no analytic decode-layer term:")
        for name, value in other.items():
            print(f"  {name:<22}{value:>10,}")
    print(
        f"\nEmulator total: "
        f"{sum(sum(bucket.values()) for bucket in measured.values()):,} cycles"
    )

    total_error = (
        (summary.modelled_cycles - summary.measured_cycles)
        / summary.measured_cycles
        if summary.measured_cycles
        else float("nan")
    )
    accepted = summary.meets_target(
        max_worst_stage_error=PUBLICATION_STAGE_ERROR_TARGET,
        min_coverage=PUBLICATION_COVERAGE_TARGET,
    ) and abs(total_error) <= EMULATOR_TOTAL_ERROR_LIMIT
    print(
        "Publication acceptance: "
        f"{'PASS' if accepted else 'FAIL'} "
        f"(worst-stage absolute error <= {PUBLICATION_STAGE_ERROR_TARGET:.0%}; "
        f"coverage >= {PUBLICATION_COVERAGE_TARGET:.0%}; "
        f"total absolute error <= {EMULATOR_TOTAL_ERROR_LIMIT:.0%})"
    )

    if args.emit_calibration:
        if compiler_artifact_path is None:
            raise SystemExit(
                "calibration emission requires the persisted compiler compilation artifact"
            )
        request_calibration_path = args.request_memory_calibration.resolve()
        if not request_calibration_path.is_file():
            raise SystemExit(
                f"request-memory calibration is missing: {request_calibration_path}"
            )
        calibration_modelled = {
            stage: modelled[stage]
            for stage, _keys in ANALYTIC_STAGE_KEYS
            if stage in modelled
        }
        require_complete_calibration_stages(calibration_modelled, measured)
        calibration = EmulatorCalibration(
            configuration=(
                f"decoder_decode kv={args.kv_size} batch={args.batch} "
                f"MLEN={args.mlen} BLEN={args.blen} HLEN={args.hlen} "
                f"timing={args.timing_mode} "
                f"kv_projection={'logical' if args.logical_kv_width else 'padded'}"
            ),
            stages=tuple(
                StageCalibration(
                    stage=stage,
                    analytical_cycles=calibration_modelled[stage],
                    emulator_cycles=sum(measured[stage].values()),
                )
                for stage, _ in ANALYTIC_STAGE_KEYS
            ),
            uncovered_cycles=layer_total - measured_total,
            execution_contract=execution_contract,
            provenance_hashes=tuple(
                sorted(
                    {
                        "op_stats": sha256_file(args.op_stats),
                        "assembly": sha256_file(args.asm),
                        "isa_lib": sha256_file(args.isa_lib),
                        "settings": sha256_file(args.settings),
                        "run_manifest": sha256_file(
                            args.op_stats.resolve().parent / MANIFEST_NAME
                        ),
                        "run_receipt": sha256_file(
                            args.op_stats.resolve().parent / RUN_STATS_NAME
                        ),
                        "emulator_binary": receipt["emulator_binary_sha256"],
                        "compiler_artifact": sha256_file(compiler_artifact_path),
                        "request_memory_calibration": sha256_file(
                            request_calibration_path
                        ),
                        **calibration_source_hashes(),
                    }.items()
                )
            ),
        )
        if calibration.passed != accepted:
            raise RuntimeError(
                "calibration artifact and command acceptance contracts disagree"
            )
        args.emit_calibration.parent.mkdir(parents=True, exist_ok=True)
        args.emit_calibration.write_text(
            json.dumps(calibration.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        print(
            f"\ncalibration: {calibration.label} "
            f"(worst stage {calibration.worst_stage_error:.1%}, "
            f"uncovered {calibration.uncovered_fraction:.1%}) "
            f"-> {args.emit_calibration}"
        )
    return 0 if accepted else 2


def main() -> int:
    args = build_parser().parse_args()
    build_lease = acquire_build_directory(args.op_stats.resolve().parent)
    try:
        execution_lease = acquire_emulator_execution()
        try:
            return _run_validation(args)
        finally:
            execution_lease.release()
    finally:
        build_lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
