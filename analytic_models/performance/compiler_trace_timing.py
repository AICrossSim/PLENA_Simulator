"""Fail-closed production timing from compiler execution traces.

The compiler trace remains the instruction-cost contract.  Physical HBM
requests live in a separate sidecar bound to the trace assembly digest and to
each DMA entry fingerprint.  This keeps request addresses and memory topology
out of :class:`ExecutionTraceEntry` while still making production timing
dependent on exact compiler, latency-library, and memory-calibration evidence.

Compiler mode composes sequential stages as
``sum(max(compute, matrix DMA) + vector DMA)``.  The matrix tile store is
double buffered and its prefetch is issued ahead of the compute that reads
it, while the vector load and writeback engine holds a single request that
the consuming instruction waits on.
Request pricing carries the last open row in every physical channel-bank pair
through the complete trace and charges fitted row conflicts at descriptor
boundaries.  Isolated request-fit identity remains separate from that stream
composition identity.
The historical closed-form path is available only through the explicitly
named ``legacy_aggregate_bandwidth`` mode; compiler mode never falls back to
it when an artifact, descriptor, opcode price, or calibration is missing.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import threading
import tempfile
from contextlib import redirect_stdout
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

from analytic_models.disagg_serve.memory import (
    HBM_BANK_GROUPS,
    HBM_BANKS_PER_GROUP,
    HBM_CHANNEL_WIDTH_BITS,
    HBM_PSEUDOCHANNELS,
    HBM_TRANSACTION_BYTES,
    REQUEST_STREAM_COMPOSITION_SCHEMA,
    DMARequestDescriptor,
)
from compiler.aten.execution_trace import (
    HBM_READ,
    HBM_WRITE,
    NO_DMA,
    CompilationArtifact,
    CompilerDMARequest,
    CompilerDMARequestRun,
    CompilerRequestMemoryTrace,
    CompilerTraceRequestBinding,
    ExecutionTrace,
    ExecutionTraceEntry,
    TensorTraceMetadata,
    build_execution_trace,
    execution_trace_entry_sha256,
)

try:
    from .decode_timing import RTL_SERIALIZED
except ImportError:
    from decode_timing import RTL_SERIALIZED


COMPILER_TRACE = "compiler_trace"
LEGACY_AGGREGATE_BANDWIDTH = "legacy_aggregate_bandwidth"
# The full-model scope covers the critical rank's compiler and HBM execution
# for an independent-request batch.  System-level collectives are composed by
# the serving model after this timing result.
FULL_MODEL_DECODE_SCOPE = "full_model_decode_step_independent_request_batch"
REFERENCE_DECODE_SCOPE = (
    "one_decoder_layer_plus_final_norm_and_lm_head_shared_cache_rows"
)
DECODE_EXECUTION_MODES = (
    COMPILER_TRACE,
    LEGACY_AGGREGATE_BANDWIDTH,
)
REQUEST_MEMORY_SIDECAR_SCHEMA = "plena-request-memory-sidecar-v2"
TRACE_STEP_COMPOSITION = "max_compute_matrix_dma_plus_vector_dma"
COMPILER_TRACE_TIMING_SCHEMA = "plena-compiler-trace-timing-v1"
FULL_MODEL_ARTIFACT_SET_SCHEMA = "plena-full-model-decode-artifacts-v1"
FULL_MODEL_ARTIFACT_RECORD_SCHEMA = "plena-full-model-decode-artifact-v1"
FULL_MODEL_ARTIFACT_FAMILY_SCHEMA = "plena-full-model-decode-artifact-family-v1"
FULL_MODEL_LOWERING_KEY_SCHEMA = "plena-full-model-decode-lowering-key-v1"
FULL_MODEL_BUILD_PLAN_SCHEMA = "plena-full-model-decode-build-plan-v1"
FULL_MODEL_FAMILY_KEY_SCHEMA = "plena-full-model-decode-family-key-v1"
FULL_MODEL_LAZY_INSTANTIATION_SCHEMA = (
    "plena-full-model-decode-lazy-instantiation-v1"
)
FULL_MODEL_CONTEXT_RESOLUTION_SCHEMA = (
    "plena-full-model-decode-context-resolution-v1"
)
FULL_MODEL_BATCH_RESOLUTION_SCHEMA = (
    "plena-full-model-decode-batch-resolution-v1"
)
FULL_MODEL_NATIVE_TEMPLATE_KEY_SCHEMA = (
    "plena-full-model-decode-native-template-key-v1"
)
FULL_MODEL_CONTEXT_RESOLUTION_MODE = "exact_full_block_loop_and_masked_tail"
FULL_MODEL_BATCH_RESOLUTION_MODE = "exact_independent_batch_slab_replication"
FULL_MODEL_STORAGE_RESOLUTION_MODE = "exact_physical_width_and_format_binding"
FULL_MODEL_CACHE_SEMANTICS = (
    "external_per_layer_packed_kv_independent_batch_slabs"
)
FULL_MODEL_ARTIFACT_ID_PREFIX = "compiler-trace-artifacts-"
DEFAULT_MAX_TRACE_GENERATION_CALLS = 25_000
DEFAULT_MAX_PROJECTED_TRACE_BYTES = 64 * 1024**3
_REQUEST_CALIBRATION_ID = re.compile(r"^request-latency-[0-9a-f]{64}$")
_SIMULATOR_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = _SIMULATOR_ROOT / "plena_settings.toml"
DEFAULT_LATENCY_LIBRARY = Path(__file__).resolve().parent / "customISA_lib.json"
DEFAULT_REQUEST_CALIBRATION = (
    _SIMULATOR_ROOT
    / "analytic_models"
    / "disagg_serve"
    / "calibration_dma_requests.csv"
)
_REFERENCE_COMPILER_SOURCE_PATHS = (
    Path("compiler/aten"),
    Path("compiler/asm_templates"),
    Path("compiler/doc/operation.svh"),
    Path("compiler/doc/configuration.svh"),
    Path("transactional_emulator/testbench/misc/decoder_decode_asm_gen.py"),
)
_NATIVE_COMPILER_SOURCE_PATHS = (
    Path("compiler/aten"),
    Path("compiler/asm_templates"),
    Path("compiler/doc/operation.svh"),
    Path("compiler/doc/configuration.svh"),
)


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")


def _require_request_calibration_id(value: str) -> None:
    if _REQUEST_CALIBRATION_ID.fullmatch(value) is None:
        raise ValueError(
            "request-memory calibration identity must be "
            "request-latency-<lowercase SHA-256>"
        )


def canonical_sha256(value: object) -> str:
    """Return a deterministic digest for compiler inputs or sidecar evidence."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_decode_compiler_source_sha256(
    simulator_root: str | Path = _SIMULATOR_ROOT,
) -> str:
    """Hash every executable source used by the decoder reference lowering."""

    root = Path(simulator_root).resolve()
    files: list[Path] = []
    for relative in _REFERENCE_COMPILER_SOURCE_PATHS:
        path = root / relative
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*.py")
                if candidate.is_file()
                and "tests" not in candidate.relative_to(path).parts
                and not candidate.name.startswith("test_")
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"compiler timing source is missing: {path}")
    digest = hashlib.sha256()
    digest.update(b"plena-reference-decode-compiler-source-v1\0")
    for path in sorted(set(files), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def native_decode_compiler_source_sha256(
    simulator_root: str | Path = _SIMULATOR_ROOT,
) -> str:
    """Hash every executable source used by native decoder trace lowering."""

    root = Path(simulator_root).resolve()
    files: list[Path] = []
    for relative in _NATIVE_COMPILER_SOURCE_PATHS:
        path = root / relative
        if path.is_dir():
            files.extend(
                candidate
                for candidate in path.rglob("*.py")
                if candidate.is_file()
                and "tests" not in candidate.relative_to(path).parts
                and not candidate.name.startswith("test_")
            )
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"compiler timing source is missing: {path}")
    digest = hashlib.sha256()
    digest.update(b"plena-native-decode-compiler-source-v1\0")
    for path in sorted(
        set(files),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


@dataclass(frozen=True)
class ArrayGeometry:
    """Exact array geometry used to compile and price a decode program."""

    mlen: int
    blen: int
    vlen: int
    hlen: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.as_tuple()
        ):
            raise ValueError("array geometry values must be positive integers")

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.mlen, self.blen, self.vlen, self.hlen

    def to_dict(self) -> dict[str, int]:
        return {
            "mlen": self.mlen,
            "blen": self.blen,
            "vlen": self.vlen,
            "hlen": self.hlen,
        }

    @classmethod
    def from_trace(cls, trace: ExecutionTrace) -> "ArrayGeometry":
        return cls(trace.mlen, trace.blen, trace.vlen, trace.hlen)


@dataclass(frozen=True)
class HBMOperatingPoint:
    """Physical HBM organization attached to one timing request."""

    generation: str
    channels: int
    pin_rate_gbps: float
    channel_width_bits: int = HBM_CHANNEL_WIDTH_BITS
    pseudochannels: int = HBM_PSEUDOCHANNELS
    bank_groups: int = HBM_BANK_GROUPS
    banks_per_group: int = HBM_BANKS_PER_GROUP
    transaction_bytes: int = HBM_TRANSACTION_BYTES

    def __post_init__(self) -> None:
        if not self.generation:
            raise ValueError("HBM generation must be explicit")
        integer_values = (
            self.channels,
            self.channel_width_bits,
            self.pseudochannels,
            self.bank_groups,
            self.banks_per_group,
            self.transaction_bytes,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_values
        ):
            raise ValueError("HBM geometry values must be positive integers")
        if self.channels & (self.channels - 1):
            raise ValueError("HBM channel count must be a power of two")
        if not math.isfinite(self.pin_rate_gbps) or self.pin_rate_gbps <= 0:
            raise ValueError("HBM pin rate must be finite and positive")

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "generation": self.generation,
            "channels": self.channels,
            "pin_rate_gbps": self.pin_rate_gbps,
            "channel_width_bits": self.channel_width_bits,
            "pseudochannels": self.pseudochannels,
            "bank_groups": self.bank_groups,
            "banks_per_group": self.banks_per_group,
            "transaction_bytes": self.transaction_bytes,
        }


@dataclass(frozen=True)
class ReferenceDecodeLowering:
    """Inputs for the emulator-validated one-layer decoder reference program.

    This lowering contains one decoder layer followed by final RMSNorm and the
    LM head.  Its query rows intentionally share one cache, matching the
    transactional validation harness; it is not an independent-request serving
    batch and is therefore kept separate from full-model serving composition.
    """

    intermediate_size: int
    vocabulary_size: int
    kv_heads: int = 1
    kv_head_reuse: bool = False
    row_tile: int | None = None
    activation_element_bits: int = 8
    weight_element_bits: int = 8
    kv_element_bits: int = 8
    block_size: int = 8
    scale_bits: int = 8

    def __post_init__(self) -> None:
        integer_values = {
            "intermediate size": self.intermediate_size,
            "vocabulary size": self.vocabulary_size,
            "KV heads": self.kv_heads,
            "block size": self.block_size,
            "scale bits": self.scale_bits,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in integer_values.values()
        ):
            raise ValueError("reference decode dimensions must be positive integers")
        for label, width in (
            ("activation", self.activation_element_bits),
            ("weight", self.weight_element_bits),
            ("KV", self.kv_element_bits),
        ):
            if width not in {2, 4, 8}:
                raise ValueError(f"{label} element width must be 2, 4, or 8 bits")
        if self.row_tile is not None and (
            isinstance(self.row_tile, bool)
            or not isinstance(self.row_tile, int)
            or self.row_tile <= 0
        ):
            raise ValueError("reference decode row tile must be a positive integer")

    def to_dict(self) -> dict[str, int | bool | None]:
        return {
            "intermediate_size": self.intermediate_size,
            "vocabulary_size": self.vocabulary_size,
            "kv_heads": self.kv_heads,
            "kv_head_reuse": self.kv_head_reuse,
            "row_tile": self.row_tile,
            "activation_element_bits": self.activation_element_bits,
            "weight_element_bits": self.weight_element_bits,
            "kv_element_bits": self.kv_element_bits,
            "block_size": self.block_size,
            "scale_bits": self.scale_bits,
        }


@dataclass(frozen=True)
class CompilerTraceTimingRequest:
    """Complete cache key for one compiled decode step.

    ``compiler_inputs_sha256`` identifies the exact full DSE point.  Full-model
    requests additionally carry its canonical descriptor and a separately
    hashed minimal lowering key.  This preserves exact provenance while
    allowing traces to be reused across fields proven not to alter assembly or
    request addresses.  Context and batch remain explicit.
    """

    compiler_inputs_sha256: str
    compiler_source_sha256: str
    context_tokens: int
    batch: int
    geometry: ArrayGeometry
    hbm: HBMOperatingPoint
    frequency_hz: float
    compiler_lowering_sha256: str | None = None
    compiler_point_descriptor_json: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(self.compiler_inputs_sha256, "compiler-input identity")
        _require_sha256(self.compiler_source_sha256, "compiler-source identity")
        if (self.compiler_lowering_sha256 is None) != (
            self.compiler_point_descriptor_json is None
        ):
            raise ValueError(
                "full-model timing requests require both point and lowering identities"
            )
        if self.compiler_lowering_sha256 is not None:
            _require_sha256(
                self.compiler_lowering_sha256,
                "compiler-lowering identity",
            )
            descriptor = json.loads(self.compiler_point_descriptor_json)
            if not isinstance(descriptor, Mapping):
                raise TypeError("compiler point descriptor must be an object")
            canonical, point_identity, _ = _point_descriptor(descriptor)
            if canonical != self.compiler_point_descriptor_json:
                raise ValueError("compiler point descriptor is not canonical")
            if point_identity != self.compiler_inputs_sha256:
                raise ValueError("compiler point descriptor identity differs")
            _, lowering_identity, _ = full_model_decode_lowering_key(descriptor)
            if lowering_identity != self.compiler_lowering_sha256:
                raise ValueError("compiler lowering identity differs from its point")
        for label, value in (
            ("context length", self.context_tokens),
            ("batch", self.batch),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        if not math.isfinite(self.frequency_hz) or self.frequency_hz <= 0:
            raise ValueError("timing frequency must be finite and positive")

    @property
    def request_id(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        value = {
            "compiler_inputs_sha256": self.compiler_inputs_sha256,
            "compiler_source_sha256": self.compiler_source_sha256,
            "context_tokens": self.context_tokens,
            "batch": self.batch,
            "geometry": self.geometry.to_dict(),
            "hbm": self.hbm.to_dict(),
            "frequency_hz": self.frequency_hz,
        }
        if self.compiler_lowering_sha256 is not None:
            value["compiler_lowering_sha256"] = self.compiler_lowering_sha256
        return value


def trace_entry_fingerprint(entry: ExecutionTraceEntry) -> str:
    """Digest the settled nine-field trace entry without changing its key."""

    return canonical_sha256(entry.to_dict())


MATRIX_ENGINE = "matrix"
VECTOR_ENGINE = "vector"
#: The DMA engine each opcode issues on. H_PREFETCH_M fills the double-buffered
#: matrix tile store; H_PREFETCH_V and H_STORE_V share the single-slot vector
#: load/writeback engine.
DMA_ENGINE_BY_OPCODE = {
    "H_PREFETCH_M": MATRIX_ENGINE,
    "H_PREFETCH_V": VECTOR_ENGINE,
    "H_STORE_V": VECTOR_ENGINE,
}


def dma_engine(opcode: str) -> str:
    """Return the DMA engine an opcode issues on."""

    try:
        return DMA_ENGINE_BY_OPCODE[opcode]
    except KeyError as error:
        raise ValueError(f"opcode {opcode!r} does not issue DMA") from error


@dataclass(frozen=True)
class RequestDescriptorRun:
    """One exact or affine sequence of physical DMA descriptors.

    The first descriptor and byte steps preserve every address while keeping a
    hardware cache-block loop independent of its trip count in serialized
    evidence.
    """

    descriptor: DMARequestDescriptor
    repetitions: int = 1
    address_step_bytes: int = 0
    scale_address_step_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.repetitions, bool)
            or not isinstance(self.repetitions, int)
            or self.repetitions <= 0
        ):
            raise ValueError("descriptor repetitions must be a positive integer")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                self.address_step_bytes,
                self.scale_address_step_bytes,
            )
        ):
            raise ValueError("descriptor affine steps must be non-negative integers")
        if self.repetitions == 1 and (
            self.address_step_bytes or self.scale_address_step_bytes
        ):
            raise ValueError("a single descriptor cannot carry an affine step")
        if self.repetitions > 1:
            ratio_numerator = (
                self.descriptor.element_bits * self.descriptor.block_size
            )
            if ratio_numerator % self.descriptor.scale_bits:
                raise ValueError("descriptor affine run has a fractional scale ratio")
            ratio = ratio_numerator // self.descriptor.scale_bits
            if self.address_step_bytes != self.scale_address_step_bytes * ratio:
                raise ValueError("descriptor affine element and scale steps disagree")

    @property
    def logical_bytes_per_request(self) -> int:
        return self.descriptor.rows * (
            self.descriptor.element_bytes_per_row
            + self.descriptor.scale_bytes_per_row
        )

    @property
    def address_varying(self) -> bool:
        return bool(self.address_step_bytes or self.scale_address_step_bytes)

    def descriptor_at(self, index: int) -> DMARequestDescriptor:
        """Resolve one descriptor from the affine sequence."""

        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("descriptor run index must be an integer")
        if not 0 <= index < self.repetitions:
            raise IndexError("descriptor run index is outside its repetitions")
        if index == 0:
            return self.descriptor
        return replace(
            self.descriptor,
            address=(
                self.descriptor.address + index * self.address_step_bytes
            ),
            scale_address=(
                self.descriptor.resolved_scale_address
                + index * self.scale_address_step_bytes
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "descriptor": self.descriptor.to_dict(),
            "repetitions": self.repetitions,
            "address_step_bytes": self.address_step_bytes,
            "scale_address_step_bytes": self.scale_address_step_bytes,
        }


@dataclass(frozen=True)
class TraceRequestBinding:
    """Physical requests for one DMA entry in the immutable compiler trace."""

    trace_entry_index: int
    trace_entry_sha256: str
    runs: tuple[RequestDescriptorRun, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.trace_entry_index, bool)
            or not isinstance(self.trace_entry_index, int)
            or self.trace_entry_index < 0
        ):
            raise ValueError("trace entry index must be a non-negative integer")
        _require_sha256(self.trace_entry_sha256, "trace-entry identity")
        if not self.runs:
            raise ValueError("a DMA trace binding requires descriptor runs")

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_entry_index": self.trace_entry_index,
            "trace_entry_sha256": self.trace_entry_sha256,
            "runs": [run.to_dict() for run in self.runs],
        }


@dataclass(frozen=True)
class RequestMemorySidecar:
    """Compiler-derived physical-memory evidence bound beside a trace."""

    trace_assembly_sha256: str
    geometry: ArrayGeometry
    bindings: tuple[TraceRequestBinding, ...]
    schema_version: str = REQUEST_MEMORY_SIDECAR_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_MEMORY_SIDECAR_SCHEMA:
            raise ValueError("unsupported request-memory sidecar schema")
        _require_sha256(self.trace_assembly_sha256, "sidecar assembly identity")
        indexes = [binding.trace_entry_index for binding in self.bindings]
        if len(indexes) != len(set(indexes)):
            raise ValueError("request-memory sidecar contains duplicate trace entries")

    @property
    def sidecar_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trace_assembly_sha256": self.trace_assembly_sha256,
            "geometry": self.geometry.to_dict(),
            "bindings": [binding.to_dict() for binding in self.bindings],
        }

    def validate(
        self,
        trace: ExecutionTrace,
        hbm: HBMOperatingPoint,
    ) -> None:
        """Require exact trace coverage, transfer sizes, and HBM provenance."""

        if self.trace_assembly_sha256 != trace.assembly_sha256:
            raise ValueError("request-memory sidecar assembly differs from the trace")
        if self.geometry != ArrayGeometry.from_trace(trace):
            raise ValueError("request-memory sidecar geometry differs from the trace")
        expected_indexes = {
            index
            for index, entry in enumerate(trace.entries)
            if entry.dma_direction != NO_DMA
        }
        observed_indexes = {binding.trace_entry_index for binding in self.bindings}
        if observed_indexes != expected_indexes:
            missing = sorted(expected_indexes - observed_indexes)
            unexpected = sorted(observed_indexes - expected_indexes)
            raise ValueError(
                "request-memory sidecar DMA coverage differs from the trace: "
                f"missing={missing}, unexpected={unexpected}"
            )

        expected_direction = {HBM_READ: "read", HBM_WRITE: "write"}
        for binding in self.bindings:
            entry = trace.entries[binding.trace_entry_index]
            if binding.trace_entry_sha256 != trace_entry_fingerprint(entry):
                raise ValueError("request-memory binding entry identity is stale")
            if sum(run.repetitions for run in binding.runs) != entry.dynamic_count:
                raise ValueError(
                    "request-memory descriptor multiplicity differs from the trace"
                )
            for run in binding.runs:
                descriptor = run.descriptor
                if descriptor.opcode != entry.opcode:
                    raise ValueError("request descriptor opcode differs from the trace")
                if descriptor.tensor != entry.tensor:
                    raise ValueError("request descriptor tensor differs from the trace")
                if descriptor.direction != expected_direction[entry.dma_direction]:
                    raise ValueError(
                        "request descriptor direction differs from the trace"
                    )
                if run.logical_bytes_per_request != entry.dma_bytes:
                    raise ValueError(
                        "request descriptor byte count differs from the trace"
                    )
                if (
                    descriptor.hbm_generation != hbm.generation
                    or descriptor.channels != hbm.channels
                    or not math.isclose(
                        descriptor.pin_rate_gbps,
                        hbm.pin_rate_gbps,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        "request descriptor HBM operating point differs from timing provenance"
                    )

    def runs_by_stage(
        self,
        trace: ExecutionTrace,
    ) -> dict[str, tuple[RequestDescriptorRun, ...]]:
        grouped: dict[str, list[RequestDescriptorRun]] = {}
        for binding in self.bindings:
            stage = trace.entries[binding.trace_entry_index].stage
            grouped.setdefault(stage, []).extend(binding.runs)
        return {stage: tuple(runs) for stage, runs in grouped.items()}


def request_memory_sidecar_from_compiler(
    source: CompilerRequestMemoryTrace,
    hbm: HBMOperatingPoint,
) -> RequestMemorySidecar:
    """Bind a compiler-owned address trace to one physical HBM operating point."""

    if not isinstance(source, CompilerRequestMemoryTrace):
        raise TypeError("compiler request-memory sidecar has the wrong type")
    geometry = ArrayGeometry(
        source.mlen,
        source.blen,
        source.vlen,
        source.hlen,
    )
    bindings = []
    for binding in source.bindings:
        runs = []
        for run in binding.runs:
            request = run.request
            runs.append(
                RequestDescriptorRun(
                    DMARequestDescriptor(
                        opcode=request.opcode,
                        hbm_generation=hbm.generation,
                        channels=hbm.channels,
                        address=request.address,
                        rows=request.rows,
                        elements_per_row=request.elements_per_row,
                        stride_bytes=request.stride_bytes,
                        element_bits=request.element_bits,
                        direction=request.direction,
                        pin_rate_gbps=hbm.pin_rate_gbps,
                        tensor=request.tensor,
                        scale_bits=request.scale_bits,
                        block_size=request.block_size,
                        scale_address=request.scale_address,
                        scale_stride_bytes=request.scale_stride_bytes,
                        partial_write_rmw=request.partial_write_rmw,
                    ),
                    repetitions=run.repetitions,
                    address_step_bytes=run.address_step_bytes,
                    scale_address_step_bytes=run.scale_address_step_bytes,
                )
            )
        bindings.append(
            TraceRequestBinding(
                trace_entry_index=binding.trace_entry_index,
                trace_entry_sha256=binding.trace_entry_sha256,
                runs=tuple(runs),
            )
        )
    return RequestMemorySidecar(
        trace_assembly_sha256=source.trace_assembly_sha256,
        geometry=geometry,
        bindings=tuple(bindings),
    )


@dataclass(frozen=True)
class BoundCompilerTrace:
    """Trace artifact and its separate request-memory sidecar."""

    execution_trace: ExecutionTrace
    compiler_source_sha256: str
    request_memory: RequestMemorySidecar | None
    compiler_lowering_sha256: str | None = None
    artifact_record_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_sha256(
            self.compiler_source_sha256,
            "artifact compiler-source identity",
        )
        if (self.compiler_lowering_sha256 is None) != (
            self.artifact_record_sha256 is None
        ):
            raise ValueError(
                "full-model artifacts require both lowering and record identities"
            )
        if self.compiler_lowering_sha256 is not None:
            _require_sha256(
                self.compiler_lowering_sha256,
                "artifact compiler-lowering identity",
            )
            _require_sha256(
                self.artifact_record_sha256,
                "compiler artifact-record identity",
            )

    @classmethod
    def from_compilation_artifact(
        cls,
        artifact: CompilationArtifact,
        *,
        compiler_source_sha256: str,
        hbm: HBMOperatingPoint,
        compiler_lowering_sha256: str | None = None,
        artifact_record_sha256: str | None = None,
    ) -> "BoundCompilerTrace":
        if artifact.request_memory is None:
            raise RuntimeError(
                "compiler artifact lacks an address-resolved request-memory sidecar"
            )
        return cls(
            execution_trace=artifact.execution_trace,
            compiler_source_sha256=compiler_source_sha256,
            request_memory=request_memory_sidecar_from_compiler(
                artifact.request_memory,
                hbm,
            ),
            compiler_lowering_sha256=compiler_lowering_sha256,
            artifact_record_sha256=artifact_record_sha256,
        )


def _canonical_json_text(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _point_descriptor(value: object) -> tuple[str, str, dict[str, object]]:
    if callable(getattr(value, "to_dict", None)):
        mapping = value.to_dict()
    elif isinstance(value, Mapping):
        mapping = dict(value)
    else:
        raise TypeError("compiler point descriptor must be an object")
    if not isinstance(mapping, dict):
        raise TypeError("compiler point descriptor must serialize to an object")
    canonical = _canonical_json_text(mapping)
    identity = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    declared = getattr(value, "descriptor_sha256", identity)
    if declared != identity:
        raise ValueError("compiler point descriptor identity is inconsistent")
    return canonical, identity, mapping


def _resolved_storage_formats(
    precision: Mapping[str, object],
) -> tuple[str, str, str, str]:
    """Resolve the legacy shared KV format and the exact K/V formats.

    A refined precision point can use ``kv_format='split'`` while carrying the
    physical formats in ``key_format`` and ``value_format``.  The frontend's
    shared KV argument remains a backwards-compatible default, so for a split
    point it is bound to the key format and the two explicit arguments remain
    authoritative.
    """

    weight = str(precision.get("weight_format", "")).lower()
    declared_kv = str(precision.get("kv_format", "")).lower()
    key = str(precision.get("key_format", declared_kv)).lower()
    value = str(precision.get("value_format", declared_kv)).lower()
    shared_kv = declared_kv if declared_kv in {"mxint", "mxfp"} else key
    return weight, shared_kv, key, value


@dataclass(frozen=True)
class FullModelDecodeContextResolution:
    """Exact non-materialized global and critical-rank context parameters."""

    context_tokens: int
    mlen: int
    batch: int
    kv_parallel_degree: int
    kv_parallel_rank: int
    local_context_tokens: int
    global_full_block_count: int
    global_tail_columns: int
    global_cache_block_count: int
    global_cache_rows_per_batch: int
    global_append_token_index: int
    local_full_block_count: int
    local_tail_columns: int
    local_cache_block_count: int
    local_cache_rows_per_batch: int
    local_append_token_index: int
    schema_version: str = FULL_MODEL_CONTEXT_RESOLUTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FULL_MODEL_CONTEXT_RESOLUTION_SCHEMA:
            raise ValueError("unsupported full-model context-resolution schema")
        if min(
            self.context_tokens,
            self.mlen,
            self.batch,
            self.kv_parallel_degree,
            self.local_context_tokens,
        ) <= 0:
            raise ValueError("context-resolution dimensions must be positive")
        owner = (self.context_tokens - 1) % self.kv_parallel_degree
        if self.kv_parallel_rank != owner:
            raise ValueError("context resolution must bind the current-token KVP rank")
        local_context = 1 + (
            self.context_tokens - 1 - self.kv_parallel_rank
        ) // self.kv_parallel_degree
        global_full, global_tail = divmod(self.context_tokens, self.mlen)
        global_blocks = global_full + int(global_tail > 0)
        local_full, local_tail = divmod(local_context, self.mlen)
        local_blocks = local_full + int(local_tail > 0)
        expected = (
            local_context,
            global_full,
            global_tail,
            global_blocks,
            global_blocks * self.mlen,
            self.context_tokens - 1,
            local_full,
            local_tail,
            local_blocks,
            local_blocks * self.mlen,
            local_context - 1,
        )
        observed = (
            self.local_context_tokens,
            self.global_full_block_count,
            self.global_tail_columns,
            self.global_cache_block_count,
            self.global_cache_rows_per_batch,
            self.global_append_token_index,
            self.local_full_block_count,
            self.local_tail_columns,
            self.local_cache_block_count,
            self.local_cache_rows_per_batch,
            self.local_append_token_index,
        )
        if observed != expected:
            raise ValueError("full-model context-resolution parameters are inconsistent")

    @classmethod
    def resolve(
        cls,
        *,
        context_tokens: int,
        mlen: int,
        batch: int,
        kv_parallel_degree: int = 1,
        kv_parallel_rank: int | None = None,
    ) -> "FullModelDecodeContextResolution":
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (context_tokens, mlen, batch, kv_parallel_degree)
        ):
            raise TypeError("context-resolution inputs must be integers")
        if min(context_tokens, mlen, batch, kv_parallel_degree) <= 0:
            raise ValueError("context-resolution inputs must be positive")
        owner = (context_tokens - 1) % kv_parallel_degree
        rank = owner if kv_parallel_rank is None else kv_parallel_rank
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise TypeError("context-resolution KVP rank must be an integer")
        if not 0 <= rank < kv_parallel_degree:
            raise ValueError("context-resolution KVP rank is outside its degree")
        local_context = 1 + (context_tokens - 1 - rank) // kv_parallel_degree
        global_full, global_tail = divmod(context_tokens, mlen)
        global_blocks = global_full + int(global_tail > 0)
        local_full, local_tail = divmod(local_context, mlen)
        local_blocks = local_full + int(local_tail > 0)
        return cls(
            context_tokens=context_tokens,
            mlen=mlen,
            batch=batch,
            kv_parallel_degree=kv_parallel_degree,
            kv_parallel_rank=rank,
            local_context_tokens=local_context,
            global_full_block_count=global_full,
            global_tail_columns=global_tail,
            global_cache_block_count=global_blocks,
            global_cache_rows_per_batch=global_blocks * mlen,
            global_append_token_index=context_tokens - 1,
            local_full_block_count=local_full,
            local_tail_columns=local_tail,
            local_cache_block_count=local_blocks,
            local_cache_rows_per_batch=local_blocks * mlen,
            local_append_token_index=local_context - 1,
        )

    @property
    def has_masked_tail(self) -> bool:
        return self.local_tail_columns > 0

    @property
    def attention_block_count(self) -> int:
        return self.local_cache_block_count

    @property
    def resolution_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "resolution_mode": FULL_MODEL_CONTEXT_RESOLUTION_MODE,
            "batch_resolution_mode": FULL_MODEL_BATCH_RESOLUTION_MODE,
            "batch_resolution": FullModelDecodeBatchResolution.resolve(
                self.batch
            ).to_dict(),
            "context_tokens": self.context_tokens,
            "mlen": self.mlen,
            "batch": self.batch,
            "kv_parallel_degree": self.kv_parallel_degree,
            "kv_parallel_rank": self.kv_parallel_rank,
            "local_context_tokens": self.local_context_tokens,
            "global_full_block_count": self.global_full_block_count,
            "global_tail_columns": self.global_tail_columns,
            "global_cache_block_count": self.global_cache_block_count,
            "global_cache_rows_per_batch": self.global_cache_rows_per_batch,
            "global_append_token_index": self.global_append_token_index,
            "local_full_block_count": self.local_full_block_count,
            "local_tail_columns": self.local_tail_columns,
            "has_masked_tail": self.has_masked_tail,
            "attention_block_count": self.attention_block_count,
            "local_cache_block_count": self.local_cache_block_count,
            "local_cache_rows_per_batch": self.local_cache_rows_per_batch,
            "local_append_token_index": self.local_append_token_index,
            "materialized_context_rows": 0,
        }


@dataclass(frozen=True)
class FullModelDecodeBatchResolution:
    """Exact binding of an independent-request batch to a native recipe.

    The recipe identity is batch-free, but the compiled record is not: query
    tiles, activation/cache addresses, weight reuse, and tails are emitted for
    the requested batch.  A record from another batch can never be rebound.
    """

    batch: int
    native_template_batch: int
    slab_ordinal_start: int
    slab_ordinal_stop: int
    slab_ordinal_step: int
    independent_slab_count: int
    resolved_active_rows: int
    schema_version: str = FULL_MODEL_BATCH_RESOLUTION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FULL_MODEL_BATCH_RESOLUTION_SCHEMA:
            raise ValueError("unsupported full-model batch-resolution schema")
        if isinstance(self.batch, bool) or not isinstance(self.batch, int):
            raise TypeError("batch-resolution batch must be an integer")
        expected = (1, 0, self.batch, 1, self.batch, self.batch)
        observed = (
            self.native_template_batch,
            self.slab_ordinal_start,
            self.slab_ordinal_stop,
            self.slab_ordinal_step,
            self.independent_slab_count,
            self.resolved_active_rows,
        )
        if self.batch <= 0 or observed != expected:
            raise ValueError("full-model batch-resolution parameters are inconsistent")

    @classmethod
    def resolve(cls, batch: int) -> "FullModelDecodeBatchResolution":
        if isinstance(batch, bool) or not isinstance(batch, int):
            raise TypeError("batch-resolution batch must be an integer")
        return cls(
            batch=batch,
            native_template_batch=1,
            slab_ordinal_start=0,
            slab_ordinal_stop=batch,
            slab_ordinal_step=1,
            independent_slab_count=batch,
            resolved_active_rows=batch,
        )

    @property
    def is_identity(self) -> bool:
        return self.batch == self.native_template_batch

    @property
    def resolution_sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "resolution_mode": FULL_MODEL_BATCH_RESOLUTION_MODE,
            "composition_operator": "exact_batch_record_instantiation",
            "batch": self.batch,
            "native_template_batch": self.native_template_batch,
            "slab_ordinal_range": {
                "start": self.slab_ordinal_start,
                "stop": self.slab_ordinal_stop,
                "step": self.slab_ordinal_step,
            },
            "independent_slab_count": self.independent_slab_count,
            "resolved_active_rows": self.resolved_active_rows,
            "identity": self.is_identity,
            "requires_exact_batch_record": True,
            "artifact_alias_permitted": False,
            "materialized_batch_alias_records": 0,
        }


def full_model_decode_batch_resolution(
    point_descriptor: object,
) -> FullModelDecodeBatchResolution:
    """Resolve the sealed independent-request batch composition contract."""

    _, _, descriptor = _point_descriptor(point_descriptor)
    serving = descriptor.get("serving")
    if not isinstance(serving, Mapping):
        raise TypeError("compiler point serving batch is missing")
    return FullModelDecodeBatchResolution.resolve(int(serving["batch"]))


def full_model_decode_context_resolution(
    point_descriptor: object,
    *,
    context_tokens: int,
) -> FullModelDecodeContextResolution:
    """Resolve exact loop, tail, append, and independent-batch parameters."""

    _, _, descriptor = _point_descriptor(point_descriptor)
    hardware = descriptor.get("hardware")
    serving = descriptor.get("serving")
    if not isinstance(hardware, Mapping) or not isinstance(serving, Mapping):
        raise TypeError("compiler point timing geometry is missing")
    geometry = hardware.get("array_geometry")
    topology = hardware.get("topology")
    if not isinstance(geometry, Mapping) or not isinstance(topology, Mapping):
        raise TypeError("compiler point array geometry is missing")
    kv_parallel_degree = int(topology["kvp"])
    kv_parallel_rank = (int(context_tokens) - 1) % kv_parallel_degree
    return FullModelDecodeContextResolution.resolve(
        context_tokens=int(context_tokens),
        mlen=int(geometry["mlen"]),
        batch=int(serving["batch"]),
        kv_parallel_degree=kv_parallel_degree,
        kv_parallel_rank=kv_parallel_rank,
    )


def full_model_decode_lowering_key(
    point_descriptor: object,
) -> tuple[str, str, dict[str, object]]:
    """Derive the minimal trace/address key from one exact DSE point.

    HBM technology, channel count, full timing configuration, link transport,
    output-head service placement, workload length/stride, runtime capacity
    reserve, and drain timing do not alter the decoder assembly or its logical
    request addresses.  They remain sealed in the full point descriptor and
    are applied while pricing or composing the system.  The fields retained
    here shape tensor geometry, quantized storage, rank-local work, cache
    layout, or compiler scheduling.
    """

    _, _, descriptor = _point_descriptor(point_descriptor)
    required_sections = {
        "schema_version",
        "artifact_scope",
        "model",
        "precision",
        "hardware",
        "serving",
        "compiler",
    }
    if set(descriptor) != required_sections:
        raise ValueError("compiler point fields differ from the exact schema")
    if descriptor["schema_version"] != "plena-compiler-trace-point-v1":
        raise ValueError("unsupported compiler point descriptor schema")
    if descriptor["artifact_scope"] != FULL_MODEL_DECODE_SCOPE:
        raise ValueError("compiler point lacks full-model decode scope")
    model = descriptor["model"]
    precision = descriptor["precision"]
    hardware = descriptor["hardware"]
    serving = descriptor["serving"]
    compiler = descriptor["compiler"]
    if not all(
        isinstance(item, Mapping)
        for item in (model, precision, hardware, serving, compiler)
    ):
        raise TypeError("compiler point sections must be objects")
    dimensions = model.get("dimensions")
    geometry = hardware.get("array_geometry")
    topology = hardware.get("topology")
    if not all(
        isinstance(item, Mapping)
        for item in (dimensions, geometry, topology)
    ):
        raise TypeError("compiler point lowering geometry is missing")
    if model.get("layer_scope") != "all_decoder_layers":
        raise ValueError("compiler point must cover all decoder layers")
    model_identity = str(model.get("model_json_sha256", ""))
    settings_identity = str(compiler.get("settings_sha256", ""))
    _require_sha256(model_identity, "compiler point model identity")
    _require_sha256(settings_identity, "compiler point settings identity")
    geometry_value = {
        name: int(geometry[name])
        for name in ("mlen", "blen", "vlen", "hlen")
    }
    if any(value <= 0 for value in geometry_value.values()):
        raise ValueError("compiler point array geometry must be positive")
    if geometry_value["mlen"] % geometry_value["hlen"]:
        raise ValueError("compiler point MLEN must be divisible by HLEN")
    for name in ("architecture_knobs_explicit", "kv_head_reuse"):
        if not isinstance(topology[name], bool):
            raise TypeError(f"compiler point topology {name} must be boolean")
    architecture_knobs_explicit = topology["architecture_knobs_explicit"]
    effective_kv_head_reuse = (
        topology["kv_head_reuse"]
        if architecture_knobs_explicit
        else True
    )
    rank_partition = {
        "tp": int(topology["tp"]),
        "kvp": int(topology["kvp"]),
        "kv_head_reuse": effective_kv_head_reuse,
    }
    if (
        rank_partition["tp"] <= 0
        or rank_partition["kvp"] <= 0
        or int(topology["chip_count"])
        != rank_partition["tp"] * rank_partition["kvp"]
    ):
        raise ValueError("compiler point rank partition is inconsistent")
    batch = int(serving["batch"])
    if batch <= 0:
        raise ValueError("compiler point batch must be positive")
    specification = precision.get("specification")
    if not isinstance(specification, Mapping):
        raise TypeError("compiler point precision specification is missing")
    precision_lowering = {
        "attention_weight_element_bits": int(specification["attn_elem"]),
        "ffn_weight_element_bits": int(specification["ffn_elem"]),
        "key_element_bits": int(
            specification.get("key_elem", specification["kv_elem"])
        ),
        "value_element_bits": int(
            specification.get("value_elem", specification["kv_elem"])
        ),
        # Native decoder activations are BF16 storage.  Narrow activation/MAC
        # widths alter the RTL price, not the compiler trace or HBM addresses.
        "activation_storage_bits": 16,
        "weight_scale_bits": 8,
        "block_size": int(precision["block_size"]),
    }
    if any(value <= 0 for value in precision_lowering.values()):
        raise ValueError("compiler point storage precision must be positive")
    key = {
        "schema_version": FULL_MODEL_LOWERING_KEY_SCHEMA,
        "artifact_scope": FULL_MODEL_DECODE_SCOPE,
        "model": {
            "model_json_sha256": model_identity,
            "dimensions": dict(dimensions),
            "layer_scope": "all_decoder_layers",
        },
        "precision": precision_lowering,
        "array_geometry": geometry_value,
        "rank_partition": rank_partition,
        "serving": {
            "kv_layout": str(serving["kv_layout"]),
            # Batch changes query-tile counts and address-resolved activation
            # and cache slabs, so exact records remain batch-specific even
            # though they share one batch-free native template recipe.
            "batch": batch,
        },
        "compiler_lowering": {
            "settings_sha256": settings_identity,
            "query_tokens": 1,
            "cache_semantics": FULL_MODEL_CACHE_SEMANTICS,
            "mram_tile_capacity": 4,
            "attention_broadcast_amount": (
                geometry_value["mlen"] // geometry_value["hlen"]
            ),
            "output_head_included": False,
            "batch_resolution": FULL_MODEL_BATCH_RESOLUTION_MODE,
            "context_resolution": FULL_MODEL_CONTEXT_RESOLUTION_MODE,
            "storage_resolution": FULL_MODEL_STORAGE_RESOLUTION_MODE,
        },
    }
    canonical = _canonical_json_text(key)
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest(), key


def full_model_decode_family_key(
    point_descriptor: object,
) -> tuple[str, str, dict[str, object]]:
    """Derive the compact native-template family independent of multiplicities."""

    _, _, descriptor = _point_descriptor(point_descriptor)
    # Reuse the strict descriptor validation performed by the lowering-key
    # derivation before selecting only opcode/template-structure determinants.
    full_model_decode_lowering_key(descriptor)
    model = descriptor["model"]
    precision = descriptor["precision"]
    serving = descriptor["serving"]
    compiler = descriptor["compiler"]
    dimensions = model["dimensions"]
    if not all(
        isinstance(item, Mapping)
        for item in (model, precision, serving, compiler, dimensions)
    ):
        raise TypeError("compiler family descriptor is malformed")
    family = {
        "schema_version": FULL_MODEL_FAMILY_KEY_SCHEMA,
        "artifact_scope": FULL_MODEL_DECODE_SCOPE,
        "model_template": {
            "model_type": str(dimensions.get("model_type", "llama")),
            "qk_norm": bool(dimensions.get("qk_norm", False)),
            "mixture_of_experts": int(dimensions.get("num_experts", 1)) > 1,
            "sliding_attention": int(dimensions.get("n_sliding", 0)) > 0,
        },
        "storage_templates": {
            "kv_layout": str(serving["kv_layout"]),
            "block_scaled": True,
            "storage_formats": "bound_in_exact_point_receipt",
        },
        "compiler_template": {
            "settings_sha256": str(compiler["settings_sha256"]),
            "cache_semantics": FULL_MODEL_CACHE_SEMANTICS,
            "rank_partition": "tensor_parallel_x_round_robin_kv_parallel",
            "residency_pricing": "algebraic_outside_native_lowering",
            "output_head_included": False,
        },
    }
    canonical = _canonical_json_text(family)
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest(), family


def full_model_decode_native_template_key(
    point_descriptor: object,
) -> tuple[str, str, dict[str, object]]:
    """Derive the native compiler template before storage/batch/context binding."""

    _, _, descriptor = _point_descriptor(point_descriptor)
    _, _, lowering = full_model_decode_lowering_key(descriptor)
    template = {
        "schema_version": FULL_MODEL_NATIVE_TEMPLATE_KEY_SCHEMA,
        "artifact_scope": FULL_MODEL_DECODE_SCOPE,
        "model": lowering["model"],
        "array_geometry": lowering["array_geometry"],
        "rank_partition": lowering["rank_partition"],
        "serving": {
            "kv_layout": lowering["serving"]["kv_layout"],
        },
        "compiler_template": {
            "settings_sha256": lowering["compiler_lowering"]["settings_sha256"],
            "query_tokens": 1,
            "cache_semantics": FULL_MODEL_CACHE_SEMANTICS,
            "mram_tile_capacity": lowering["compiler_lowering"][
                "mram_tile_capacity"
            ],
            "attention_broadcast_amount": lowering["compiler_lowering"][
                "attention_broadcast_amount"
            ],
            "output_head_included": False,
            "storage_resolution": FULL_MODEL_STORAGE_RESOLUTION_MODE,
            "batch_resolution": FULL_MODEL_BATCH_RESOLUTION_MODE,
            "context_resolution": FULL_MODEL_CONTEXT_RESOLUTION_MODE,
        },
    }
    canonical = _canonical_json_text(template)
    return canonical, hashlib.sha256(canonical.encode("utf-8")).hexdigest(), template


def full_model_decode_generator_blockers(
    point_descriptor: object,
) -> tuple[str, ...]:
    """Return every native-generator capability mismatch for an exact point.

    Native generation is admitted only for descriptors satisfying the exact
    model, rank-partition, and dense-selector PackedKV contracts. Unsupported
    points remain visible to dry-run planning and fail before trace generation.
    """

    _, _, descriptor = _point_descriptor(point_descriptor)
    _, _, lowering = full_model_decode_lowering_key(descriptor)
    dimensions = lowering["model"]["dimensions"]
    precision = descriptor["precision"]
    specification = precision["specification"]
    geometry = lowering["array_geometry"]
    rank_partition = lowering["rank_partition"]
    serving = lowering["serving"]
    if not all(
        isinstance(item, Mapping)
        for item in (
            dimensions,
            precision,
            specification,
            geometry,
            rank_partition,
            serving,
        )
    ):
        raise TypeError("compiler generator point sections are malformed")

    blockers: list[str] = []
    if int(dimensions.get("num_experts", 1)) > 1:
        blockers.append("mixture_of_experts_not_lowered")
    if int(dimensions.get("n_sliding", 0)) > 0:
        blockers.append("sliding_attention_not_lowered")
    weight_format, _, key_format, value_format = _resolved_storage_formats(
        precision
    )
    if weight_format not in {"mxint", "mxfp"}:
        blockers.append("unsupported_weight_storage_format")
    if key_format not in {"mxint", "mxfp"}:
        blockers.append("unsupported_key_storage_format")
    if value_format not in {"mxint", "mxfp"}:
        blockers.append("unsupported_value_storage_format")
    if int(specification["attn_elem"]) != int(specification["ffn_elem"]):
        blockers.append("mixed_attention_ffn_weight_width_not_lowered")
    if str(serving["kv_layout"]) != "dense_selector":
        blockers.append("packed_kv_dense_selector_required")

    hidden = int(dimensions["hidden"])
    heads = int(dimensions["heads"])
    kv_heads = int(dimensions["kv_heads"])
    head_dim = int(dimensions["head_dim"])
    layers = int(dimensions["layers"])
    inter = int(dimensions["inter"])
    mlen = int(geometry["mlen"])
    hlen = int(geometry["hlen"])
    tp = int(rank_partition["tp"])
    kvp = int(rank_partition["kvp"])
    if min(hidden, heads, kv_heads, head_dim, layers) <= 0:
        blockers.append("nonpositive_model_dimension")
    if (
        kv_heads <= 0
        or heads % kv_heads
        or heads % tp
        or kv_heads % tp
        or inter % tp
        or tp <= 0
        or kvp <= 0
    ):
        blockers.append("invalid_grouped_query_attention_ratio")
        local_heads = heads
        local_kv_heads = kv_heads
    else:
        local_heads = heads // tp
        local_kv_heads = kv_heads // tp
    if int(geometry["vlen"]) != mlen:
        blockers.append("native_frontend_requires_vlen_equal_mlen")
    if head_dim != hlen or mlen % hlen:
        blockers.append("packed_attention_head_geometry_unsupported")
    elif local_heads // local_kv_heads > mlen // hlen:
        blockers.append("grouped_query_ratio_exceeds_head_broadcast")
    block_size = int(precision["block_size"])
    if (
        local_kv_heads * head_dim > mlen
        or local_kv_heads > 16
        or mlen % head_dim
        or head_dim % block_size
        or mlen % block_size
    ):
        blockers.append("packed_kv_layout_geometry_unsupported")
    return tuple(sorted(set(blockers)))


@dataclass(frozen=True)
class FullModelDecodeArtifactFamily:
    """Sealed native frontend recipe shared by exact lazy instantiations."""

    family_key_json: str
    compiler_source_sha256: str
    generator_contract_json: str
    schema_version: str = FULL_MODEL_ARTIFACT_FAMILY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FULL_MODEL_ARTIFACT_FAMILY_SCHEMA:
            raise ValueError("unsupported full-model artifact family schema")
        _require_sha256(
            self.compiler_source_sha256,
            "full-model artifact family compiler-source identity",
        )
        family_key = json.loads(self.family_key_json)
        contract = json.loads(self.generator_contract_json)
        if not isinstance(family_key, dict) or not isinstance(contract, dict):
            raise TypeError("artifact family key and generator contract must be objects")
        if self.family_key_json != _canonical_json_text(family_key):
            raise ValueError("full-model artifact family key is not canonical")
        if self.generator_contract_json != _canonical_json_text(contract):
            raise ValueError("full-model generator contract is not canonical")
        if family_key.get("schema_version") != FULL_MODEL_FAMILY_KEY_SCHEMA:
            raise ValueError("unsupported full-model artifact family key")
        required_contract = {
            "schema_version",
            "artifact_scope",
            "frontend",
            "lowering_mode",
            "query_tokens",
            "cache_semantics",
            "output_head_included",
            "base_sram_policy",
            "lazy_instantiation_signature",
            "unsupported_features",
        }
        if set(contract) != required_contract:
            raise ValueError("full-model generator contract fields differ from the schema")
        if (
            contract["schema_version"] != FULL_MODEL_LAZY_INSTANTIATION_SCHEMA
            or contract["artifact_scope"] != FULL_MODEL_DECODE_SCOPE
            or contract["frontend"]
            != "compiler.aten.plena_frontend.compile_native_hf_decoder"
            or contract["lowering_mode"] != "trace_only_shape_descriptor"
            or contract["query_tokens"] != 1
            or contract["cache_semantics"] != FULL_MODEL_CACHE_SEMANTICS
            or contract["output_head_included"] is not False
            or contract["base_sram_policy"] != "streaming"
            or contract["lazy_instantiation_signature"]
            != (
                "sha256(family_key,native_template,lowering_key,point,"
                "context_resolution,compiler_source)"
            )
            or contract["unsupported_features"] != "fail_closed_preflight"
        ):
            raise ValueError("unsupported full-model native generator contract")

    @classmethod
    def from_point_descriptor(
        cls,
        point_descriptor: object,
        *,
        compiler_source_sha256: str,
    ) -> "FullModelDecodeArtifactFamily":
        family_json, _, _ = full_model_decode_family_key(point_descriptor)
        contract = {
            "schema_version": FULL_MODEL_LAZY_INSTANTIATION_SCHEMA,
            "artifact_scope": FULL_MODEL_DECODE_SCOPE,
            "frontend": "compiler.aten.plena_frontend.compile_native_hf_decoder",
            "lowering_mode": "trace_only_shape_descriptor",
            "query_tokens": 1,
            "cache_semantics": FULL_MODEL_CACHE_SEMANTICS,
            "output_head_included": False,
            "base_sram_policy": "streaming",
            "lazy_instantiation_signature": (
                "sha256(family_key,native_template,lowering_key,point,"
                "context_resolution,compiler_source)"
            ),
            "unsupported_features": "fail_closed_preflight",
        }
        return cls(
            family_key_json=family_json,
            compiler_source_sha256=compiler_source_sha256,
            generator_contract_json=_canonical_json_text(contract),
        )

    @property
    def family_sha256(self) -> str:
        return canonical_sha256(self._payload())

    @property
    def family_key(self) -> dict[str, object]:
        return json.loads(self.family_key_json)

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "family_key": self.family_key,
            "compiler_source_sha256": self.compiler_source_sha256,
            "generator_contract": json.loads(self.generator_contract_json),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "family_sha256": self.family_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FullModelDecodeArtifactFamily":
        required = {
            "schema_version",
            "family_key",
            "compiler_source_sha256",
            "generator_contract",
            "family_sha256",
        }
        if set(value) != required:
            raise ValueError("full-model artifact family fields differ from the schema")
        family_key = value["family_key"]
        contract = value["generator_contract"]
        if not isinstance(family_key, Mapping) or not isinstance(contract, Mapping):
            raise TypeError("full-model artifact family objects are malformed")
        family = cls(
            family_key_json=_canonical_json_text(family_key),
            compiler_source_sha256=str(value["compiler_source_sha256"]),
            generator_contract_json=_canonical_json_text(contract),
            schema_version=str(value["schema_version"]),
        )
        if value["family_sha256"] != family.family_sha256:
            raise ValueError("full-model artifact family identity is stale")
        return family


@dataclass(frozen=True)
class FullModelDecodeArtifactRecord:
    """One exact point/context compiler artifact with rank ownership."""

    descriptor_json: str
    lowering_key_json: str
    context_tokens: int
    compiler_source_sha256: str
    lowering_contract_json: str
    compiler_receipt_json: str
    compilation_artifact: CompilationArtifact
    schema_version: str = FULL_MODEL_ARTIFACT_RECORD_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != FULL_MODEL_ARTIFACT_RECORD_SCHEMA:
            raise ValueError("unsupported full-model artifact record schema")
        _require_sha256(
            self.compiler_source_sha256,
            "full-model artifact compiler-source identity",
        )
        if (
            isinstance(self.context_tokens, bool)
            or not isinstance(self.context_tokens, int)
            or self.context_tokens <= 1
        ):
            raise ValueError("full-model decode context must exceed one token")
        if not isinstance(self.compilation_artifact, CompilationArtifact):
            raise TypeError("full-model record requires a compilation artifact")
        if self.compilation_artifact.request_memory is None:
            raise ValueError("full-model artifact lacks request-memory evidence")

        descriptor = json.loads(self.descriptor_json)
        lowering_key = json.loads(self.lowering_key_json)
        contract = json.loads(self.lowering_contract_json)
        receipt = json.loads(self.compiler_receipt_json)
        if not all(
            isinstance(item, dict)
            for item in (descriptor, lowering_key, contract, receipt)
        ):
            raise TypeError(
                "full-model descriptor, lowering key, contract, and receipt must be objects"
            )
        if self.descriptor_json != _canonical_json_text(descriptor):
            raise ValueError("full-model point descriptor is not canonical")
        if self.lowering_key_json != _canonical_json_text(lowering_key):
            raise ValueError("full-model lowering key is not canonical")
        derived_key_json, _, _ = full_model_decode_lowering_key(descriptor)
        if self.lowering_key_json != derived_key_json:
            raise ValueError("full-model lowering key differs from its point")
        if self.lowering_contract_json != _canonical_json_text(contract):
            raise ValueError("full-model lowering contract is not canonical")
        if self.compiler_receipt_json != _canonical_json_text(receipt):
            raise ValueError("full-model compiler receipt is not canonical")
        self._validate_contract(descriptor, contract, receipt)

    @property
    def representative_point_sha256(self) -> str:
        return hashlib.sha256(self.descriptor_json.encode("utf-8")).hexdigest()

    @property
    def compiler_lowering_sha256(self) -> str:
        return hashlib.sha256(self.lowering_key_json.encode("utf-8")).hexdigest()

    @property
    def compiler_family_sha256(self) -> str:
        family_json, _, _ = full_model_decode_family_key(self.descriptor)
        return hashlib.sha256(family_json.encode("utf-8")).hexdigest()

    @property
    def native_template_sha256(self) -> str:
        return full_model_decode_native_template_key(self.descriptor)[1]

    @property
    def context_resolution(self) -> FullModelDecodeContextResolution:
        return full_model_decode_context_resolution(
            self.descriptor,
            context_tokens=self.context_tokens,
        )

    @property
    def lazy_instantiation_sha256(self) -> str:
        """Identity of the smallest exact native-generation signature."""

        return canonical_sha256(
            {
                "schema_version": FULL_MODEL_LAZY_INSTANTIATION_SCHEMA,
                "compiler_family_sha256": self.compiler_family_sha256,
                "native_template_sha256": self.native_template_sha256,
                "compiler_lowering_sha256": self.compiler_lowering_sha256,
                "representative_point_sha256": self.representative_point_sha256,
                "context_resolution_sha256": (
                    self.context_resolution.resolution_sha256
                ),
                "compiler_source_sha256": self.compiler_source_sha256,
            }
        )

    @property
    def descriptor(self) -> dict[str, object]:
        return json.loads(self.descriptor_json)

    @property
    def lowering_key(self) -> dict[str, object]:
        return json.loads(self.lowering_key_json)

    @property
    def lowering_contract(self) -> dict[str, object]:
        return json.loads(self.lowering_contract_json)

    @property
    def compiler_receipt(self) -> dict[str, object]:
        return json.loads(self.compiler_receipt_json)

    def _validate_contract(
        self,
        descriptor: dict[str, object],
        contract: dict[str, object],
        receipt: dict[str, object],
    ) -> None:
        required_contract = {
            "schema_version",
            "artifact_scope",
            "compiler_lowering_sha256",
            "query_tokens",
            "context_tokens",
            "context_resolution",
            "batch",
            "cache_semantics",
            "layer_scope",
            "decoder_layers",
            "output_head_included",
            "output_head_ownership",
            "geometry",
            "rank_partition",
            "critical_rank",
        }
        if set(contract) != required_contract:
            raise ValueError("full-model lowering contract fields differ from the schema")
        if contract["schema_version"] != "plena-full-model-decode-lowering-v1":
            raise ValueError("unsupported full-model lowering contract schema")
        if descriptor.get("artifact_scope") != FULL_MODEL_DECODE_SCOPE:
            raise ValueError("artifact descriptor lacks full-model decode scope")
        if descriptor.get("schema_version") != "plena-compiler-trace-point-v1":
            raise ValueError("unsupported compiler point descriptor schema")
        if contract["artifact_scope"] != FULL_MODEL_DECODE_SCOPE:
            raise ValueError("lowering contract lacks full-model decode scope")
        if contract["compiler_lowering_sha256"] != self.compiler_lowering_sha256:
            raise ValueError("lowering contract differs from its lowering key")
        if contract["query_tokens"] != 1:
            raise ValueError("full-model compiler artifact must use q_len=1")
        if contract["context_tokens"] != self.context_tokens:
            raise ValueError("full-model artifact context is inconsistent")
        if contract["context_resolution"] != self.context_resolution.to_dict():
            raise ValueError("full-model artifact context algebra is inconsistent")
        if contract["cache_semantics"] != FULL_MODEL_CACHE_SEMANTICS:
            raise ValueError("full-model artifact cache semantics are unsupported")
        if contract["layer_scope"] != "all_decoder_layers":
            raise ValueError("full-model artifact omits decoder layers")
        if contract["output_head_included"] is not False:
            raise ValueError("native decoder artifact must exclude the output head")

        model = descriptor.get("model")
        descriptor_serving = descriptor.get("serving")
        lowering_key = self.lowering_key
        lowering_model = lowering_key.get("model")
        serving = lowering_key.get("serving")
        rank_partition = lowering_key.get("rank_partition")
        geometry = lowering_key.get("array_geometry")
        if not all(
            isinstance(item, Mapping)
            for item in (
                model,
                descriptor_serving,
                lowering_model,
                serving,
                rank_partition,
                geometry,
            )
        ):
            raise TypeError("full-model descriptor dimensions are missing")
        dimensions = lowering_model["dimensions"]
        if not isinstance(dimensions, Mapping):
            raise TypeError("full-model lowering dimensions are missing")
        expected_layers = int(dimensions["layers"])
        expected_batch = int(descriptor_serving["batch"])
        critical_rank = contract["critical_rank"]
        if not isinstance(critical_rank, Mapping) or set(critical_rank) != {
            "tensor_parallel_rank",
            "kv_parallel_rank",
            "kv_token_sharding",
            "owns_current_token",
        }:
            raise ValueError("full-model artifact critical-rank binding is incomplete")
        tp = int(rank_partition["tp"])
        kvp = int(rank_partition["kvp"])
        tp_rank = int(critical_rank["tensor_parallel_rank"])
        kvp_rank = int(critical_rank["kv_parallel_rank"])
        if not (0 <= tp_rank < tp and 0 <= kvp_rank < kvp):
            raise ValueError("full-model artifact critical rank is outside the topology")
        if critical_rank["kv_token_sharding"] != "round_robin":
            raise ValueError("full-model artifact requires explicit round-robin KV sharding")
        if critical_rank["owns_current_token"] is not True:
            raise ValueError("full-model critical rank must own the current-token append")
        if kvp_rank != (self.context_tokens - 1) % kvp:
            raise ValueError("full-model critical rank does not own the current token")
        if contract["decoder_layers"] != expected_layers:
            raise ValueError("full-model artifact layer count is inconsistent")
        if contract["batch"] != expected_batch:
            raise ValueError("full-model artifact batch is inconsistent")
        if contract["geometry"] != geometry:
            raise ValueError("full-model artifact geometry is inconsistent")
        if contract["rank_partition"] != rank_partition:
            raise ValueError("full-model artifact rank partition is inconsistent")
        if contract["output_head_ownership"] != "external_to_decoder_artifact":
            raise ValueError("full-model artifact output-head ownership is inconsistent")

        required_receipt = {
            "artifact_scope",
            "trace_only",
            "decoder_input_source",
            "seq_len",
            "decode_context_tokens",
            "local_decode_context_tokens",
            "local_cache_position",
            "owns_current_kv_token",
            "kv_append_enabled",
            "batch_size",
            "active_rows",
            "padded_seq_len",
            "rows_per_batch",
            "compile_seq_rows",
            "cache_rows_per_batch",
            "external_packed_kv_cache",
            "packed_kv_layout_id",
            "packed_kv_selector_enabled",
            "packed_kv_element_bits",
            "packed_key_layout_id",
            "packed_value_layout_id",
            "packed_key_row_bytes",
            "packed_value_row_bytes",
            "packed_key_element_bits",
            "packed_value_element_bits",
            "packed_key_block_size",
            "packed_value_block_size",
            "packed_key_scale_bits",
            "packed_value_scale_bits",
            "compiled_qk_norm",
            "qk_norm_segment_width",
            "qk_norm_reciprocal_fp_offset",
            "qk_norm_affine_storage_shape",
            "qk_norm_affine_pattern",
            "output_head_included",
            "output_head_location",
            "compiled_sram_policy",
            "compiled_kv_head_reuse",
            "attention_head_packing",
            "attention_head_slot_dim",
            "attention_broadcast_amount",
            "tensor_parallel_degree",
            "kv_parallel_degree",
            "tensor_parallel_rank",
            "kv_parallel_rank",
            "kv_token_sharding",
            "local_num_heads",
            "local_num_kv_heads",
            "local_inter_dim",
            "local_padded_inter_dim",
            "tensor_parallel_query_head_range",
            "tensor_parallel_kv_head_range",
            "local_padded_query_width",
            "local_packed_kv_active_elements",
            "local_kv_head_selector_count",
            "communication_lowering",
            "external_collectives",
            "hidden_size",
            "inter_dim",
            "padded_hidden_size",
            "padded_inter_dim",
            "num_layers",
            "num_heads",
            "num_kv_heads",
            "head_dim",
            "weight_element_bits",
            "weight_block_size",
            "weight_scale_bits",
            "weight_storage_format",
            "kv_storage_format",
            "key_storage_format",
            "value_storage_format",
            "mlen",
            "blen",
            "hlen",
        }
        missing_receipt = sorted(required_receipt - set(receipt))
        if missing_receipt:
            raise ValueError(
                f"full-model compiler receipt is incomplete: {missing_receipt}"
            )
        if (
            receipt["artifact_scope"] != FULL_MODEL_DECODE_SCOPE
            or receipt["trace_only"] is not True
            or receipt["decoder_input_source"] != "trace_shape_only"
            or receipt["seq_len"] != 1
            or receipt["decode_context_tokens"] != self.context_tokens
            or receipt["batch_size"] != expected_batch
            or receipt["active_rows"] != expected_batch
            or receipt["external_packed_kv_cache"] is not True
            or receipt["packed_kv_selector_enabled"] is not True
            or receipt["output_head_included"] is not False
            or receipt["output_head_location"]
            != model.get("output_head_location")
            # Residency is priced algebraically.  The reusable native trace is
            # intentionally always the streaming base program.
            or receipt["compiled_sram_policy"] != "streaming"
            or receipt["compiled_kv_head_reuse"]
            is not rank_partition["kv_head_reuse"]
            or int(receipt["tensor_parallel_degree"])
            != int(rank_partition["tp"])
            or int(receipt["kv_parallel_degree"])
            != int(rank_partition["kvp"])
        ):
            raise ValueError("full-model compiler receipt differs from its contract")
        if serving.get("kv_layout") != "dense_selector":
            raise ValueError("full-model compiler artifacts require dense-selector PackedKV")
        expected_cache_rows = self.context_resolution.local_cache_rows_per_batch
        if receipt["cache_rows_per_batch"] != expected_cache_rows:
            raise ValueError("full-model compiler cache capacity is inconsistent")
        dimension_receipt = {
            "hidden_size": "hidden",
            "inter_dim": "inter",
            "num_layers": "layers",
            "num_heads": "heads",
            "num_kv_heads": "kv_heads",
            "head_dim": "head_dim",
        }
        if any(
            int(receipt[receipt_name]) != int(dimensions[dimension_name])
            for receipt_name, dimension_name in dimension_receipt.items()
        ):
            raise ValueError("full-model compiler model dimensions are inconsistent")
        if (
            int(receipt["mlen"]) != int(geometry["mlen"])
            or int(receipt["blen"]) != int(geometry["blen"])
            or int(receipt["hlen"]) != int(geometry["hlen"])
        ):
            raise ValueError("full-model compiler geometry receipt is inconsistent")
        mlen = int(geometry["mlen"])
        hlen = int(geometry["hlen"])
        hidden = int(dimensions["hidden"])
        inter = int(dimensions["inter"])
        heads = int(dimensions["heads"])
        kv_heads = int(dimensions["kv_heads"])
        local_heads = heads // tp
        local_kv_heads = kv_heads // tp
        local_inter = inter // tp
        expected_collectives = []
        if tp > 1:
            expected_collectives.extend(
                ("attention_output_all_reduce", "ffn_down_output_all_reduce")
            )
        if kvp > 1:
            expected_collectives.append("attention_logsumexp_reduce")
        expected_rank_receipt = {
            "local_decode_context_tokens": (
                self.context_resolution.local_context_tokens
            ),
            "local_cache_position": (
                self.context_resolution.local_append_token_index
            ),
            "owns_current_kv_token": True,
            "kv_append_enabled": True,
            "local_num_heads": local_heads,
            "local_num_kv_heads": local_kv_heads,
            "local_inter_dim": local_inter,
            "local_padded_inter_dim": ((local_inter + mlen - 1) // mlen) * mlen,
            "tensor_parallel_query_head_range": [
                tp_rank * local_heads,
                (tp_rank + 1) * local_heads,
            ],
            "tensor_parallel_kv_head_range": [
                tp_rank * local_kv_heads,
                (tp_rank + 1) * local_kv_heads,
            ],
            "local_padded_query_width": local_kv_heads * mlen,
            "local_packed_kv_active_elements": (
                local_kv_heads * int(dimensions["head_dim"])
            ),
            "local_kv_head_selector_count": local_kv_heads,
            "communication_lowering": "external_collectives",
            "external_collectives": expected_collectives,
        }
        if any(
            receipt[name] != expected
            for name, expected in expected_rank_receipt.items()
        ):
            raise ValueError("full-model compiler rank-local receipt is inconsistent")
        expected_batch_tiling = {
            "padded_seq_len": mlen,
            "rows_per_batch": mlen,
            "compile_seq_rows": expected_batch * mlen,
            "padded_hidden_size": ((hidden + mlen - 1) // mlen) * mlen,
            "padded_inter_dim": ((inter + mlen - 1) // mlen) * mlen,
            "attention_head_packing": True,
            "attention_head_slot_dim": int(dimensions["head_dim"]),
            "attention_broadcast_amount": mlen // hlen,
        }
        if any(
            receipt[name] != expected
            for name, expected in expected_batch_tiling.items()
        ):
            raise ValueError("full-model compiler exact-batch tiling is inconsistent")
        precision = descriptor.get("precision")
        if not isinstance(precision, Mapping):
            raise TypeError("full-model precision descriptor is missing")
        specification = precision.get("specification")
        if not isinstance(specification, Mapping):
            raise TypeError("full-model precision specification is missing")
        from compiler.aten.plena.packed_kv import PackedKVLayout

        weight_bits = int(receipt["weight_element_bits"])
        block_size = int(precision["block_size"])
        scale_bits = 8
        key_bits = int(specification.get("key_elem", specification["kv_elem"]))
        value_bits = int(
            specification.get("value_elem", specification["kv_elem"])
        )
        key_layout = PackedKVLayout(
            kv_heads=local_kv_heads,
            head_dim=int(dimensions["head_dim"]),
            mlen=int(geometry["mlen"]),
            block_size=block_size,
            element_bits=key_bits,
            scale_bits=scale_bits,
        )
        value_layout = PackedKVLayout(
            kv_heads=local_kv_heads,
            head_dim=int(dimensions["head_dim"]),
            mlen=int(geometry["mlen"]),
            block_size=block_size,
            element_bits=value_bits,
            scale_bits=scale_bits,
        )
        weight_format, kv_format, key_format, value_format = (
            _resolved_storage_formats(precision)
        )
        if (
            int(specification["attn_elem"]) != weight_bits
            or int(specification["ffn_elem"]) != weight_bits
            or key_bits != int(receipt["packed_kv_element_bits"])
            or key_bits != int(receipt["packed_key_element_bits"])
            or value_bits != int(receipt["packed_value_element_bits"])
            or block_size != int(receipt["weight_block_size"])
            or block_size != int(receipt["packed_key_block_size"])
            or block_size != int(receipt["packed_value_block_size"])
            or scale_bits != int(receipt["weight_scale_bits"])
            or scale_bits != int(receipt["packed_key_scale_bits"])
            or scale_bits != int(receipt["packed_value_scale_bits"])
            or receipt["packed_kv_layout_id"] != key_layout.layout_id
            or receipt["packed_key_layout_id"] != key_layout.layout_id
            or receipt["packed_value_layout_id"] != value_layout.layout_id
            or int(receipt["packed_key_row_bytes"])
            != key_layout.packed_row_bytes
            or int(receipt["packed_value_row_bytes"])
            != value_layout.packed_row_bytes
            or str(receipt["weight_storage_format"]).lower() != weight_format
            or str(receipt["kv_storage_format"]).lower() != kv_format
            or str(receipt["key_storage_format"]).lower() != key_format
            or str(receipt["value_storage_format"]).lower() != value_format
        ):
            raise ValueError("full-model compiler precision receipt is inconsistent")

        qk_norm = bool(dimensions.get("qk_norm", False))
        expected_qk_metadata = (
            (
                int(dimensions["head_dim"]),
                6,
                [4, int(geometry["mlen"])],
                "shared_head_weight_repeated_per_mlen_vector",
            )
            if qk_norm
            else (None, None, None, None)
        )
        observed_qk_metadata = (
            receipt["qk_norm_segment_width"],
            receipt["qk_norm_reciprocal_fp_offset"],
            receipt["qk_norm_affine_storage_shape"],
            receipt["qk_norm_affine_pattern"],
        )
        if (
            receipt["compiled_qk_norm"] is not qk_norm
            or observed_qk_metadata != expected_qk_metadata
        ):
            raise ValueError("full-model compiler Q/K norm receipt is inconsistent")

        if (
            int(receipt["tensor_parallel_rank"]) != tp_rank
            or int(receipt["kv_parallel_rank"]) != kvp_rank
            or receipt["kv_token_sharding"]
            != critical_rank["kv_token_sharding"]
        ):
            raise ValueError("full-model compiler receipt uses a different rank")

        trace = self.compilation_artifact.execution_trace
        if ArrayGeometry.from_trace(trace).to_dict() != geometry:
            raise ValueError("full-model artifact trace geometry is inconsistent")
        if any(entry.stage == "LM head" for entry in trace.entries):
            raise ValueError("native decoder artifact unexpectedly includes an LM head")
        expected_caches = {
            f"{role}_cache_{layer}"
            for layer in range(expected_layers)
            for role in ("K", "V")
        }
        dma_tensors = {
            entry.tensor
            for entry in trace.entries
            if entry.dma_direction != NO_DMA
        }
        if not expected_caches <= dma_tensors:
            missing = sorted(expected_caches - dma_tensors)
            raise ValueError(f"full-model artifact cache coverage is incomplete: {missing}")
        for cache in expected_caches:
            stores = [
                entry
                for entry in trace.entries
                if entry.opcode == "H_STORE_V" and entry.tensor == cache
            ]
            if sum(entry.dynamic_count for entry in stores) != expected_batch:
                raise ValueError("full-model artifact cache append coverage is incomplete")
            self._validate_cache_append_context(
                cache,
                expected_batch=expected_batch,
                kv_parallel_degree=kvp,
            )

    def _validate_cache_append_context(
        self,
        cache: str,
        *,
        expected_batch: int,
        kv_parallel_degree: int,
    ) -> None:
        trace = self.compilation_artifact.execution_trace
        request_memory = self.compilation_artifact.request_memory
        if request_memory is None:
            raise ValueError("full-model artifact lacks request-memory evidence")
        bindings = {
            binding.trace_entry_index: binding
            for binding in request_memory.bindings
        }
        read_runs = []
        append_runs = []
        for index, entry in enumerate(trace.entries):
            if entry.tensor != cache:
                continue
            binding = bindings.get(index)
            if binding is None:
                continue
            if entry.dma_direction == HBM_READ:
                read_runs.extend(binding.runs)
            elif entry.opcode == "H_STORE_V" and entry.dma_direction == HBM_WRITE:
                append_runs.extend(binding.runs)
        read_count = sum(run.repetitions for run in read_runs)
        append_count = sum(run.repetitions for run in append_runs)
        if (
            append_count != expected_batch
            or not read_runs
            or read_count % expected_batch
        ):
            raise ValueError("full-model cache request ordering is incomplete")
        reads_per_batch = read_count // expected_batch

        def request_at_ordinal(
            runs: list[object],
            ordinal: int,
        ) -> object:
            for run in runs:
                repetitions = int(getattr(run, "repetitions"))
                if ordinal < repetitions:
                    return run.request_at(ordinal)
                ordinal -= repetitions
            raise IndexError("compiler DMA request ordinal is outside its runs")

        local_token_index = (self.context_tokens - 1) // kv_parallel_degree
        for batch_index in range(expected_batch):
            base_request = request_at_ordinal(
                read_runs,
                batch_index * reads_per_batch,
            )
            append_request = request_at_ordinal(append_runs, batch_index)
            packed_bits = (
                append_request.elements_per_row * append_request.element_bits
            )
            if packed_bits % 8:
                raise ValueError("full-model cache token rows must be byte addressable")
            token_bytes = packed_bits // 8
            expected_address = (
                base_request.address + local_token_index * token_bytes
            )
            if append_request.address != expected_address:
                raise ValueError(
                    "full-model cache append address differs from its context"
                )

    @property
    def record_sha256(self) -> str:
        return canonical_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "descriptor": json.loads(self.descriptor_json),
            "lowering_key": json.loads(self.lowering_key_json),
            "context_tokens": self.context_tokens,
            "compiler_source_sha256": self.compiler_source_sha256,
            "lowering_contract": json.loads(self.lowering_contract_json),
            "compiler_receipt": json.loads(self.compiler_receipt_json),
            "compilation_artifact": self.compilation_artifact.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "record_sha256": self.record_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "FullModelDecodeArtifactRecord":
        required = {
            "schema_version",
            "descriptor",
            "lowering_key",
            "context_tokens",
            "compiler_source_sha256",
            "lowering_contract",
            "compiler_receipt",
            "compilation_artifact",
            "record_sha256",
        }
        if set(value) != required:
            raise ValueError("full-model artifact record fields differ from the schema")
        descriptor = value["descriptor"]
        lowering_key = value["lowering_key"]
        contract = value["lowering_contract"]
        receipt = value["compiler_receipt"]
        artifact = value["compilation_artifact"]
        if not all(
            isinstance(item, Mapping)
            for item in (descriptor, lowering_key, contract, receipt, artifact)
        ):
            raise TypeError("full-model artifact record objects are malformed")
        record = cls(
            descriptor_json=_canonical_json_text(descriptor),
            lowering_key_json=_canonical_json_text(lowering_key),
            context_tokens=int(value["context_tokens"]),
            compiler_source_sha256=str(value["compiler_source_sha256"]),
            lowering_contract_json=_canonical_json_text(contract),
            compiler_receipt_json=_canonical_json_text(receipt),
            compilation_artifact=CompilationArtifact.from_dict(artifact),
            schema_version=str(value["schema_version"]),
        )
        if value["record_sha256"] != record.record_sha256:
            raise ValueError("full-model artifact record identity is stale")
        return record


class FullModelDecodeArtifactSet:
    """Content-addressed families plus optional pre-generated exact traces.

    Lazy records are cached in memory by their structure-changing lowering key
    and context.  They are deliberately excluded from the serialized family
    artifact, so no per-point aliases or mutable on-disk cache are created.
    """

    scope = FULL_MODEL_DECODE_SCOPE

    def __init__(
        self,
        records: Iterable[FullModelDecodeArtifactRecord] = (),
        *,
        families: Iterable[FullModelDecodeArtifactFamily] = (),
    ) -> None:
        ordered = tuple(
            sorted(
                records,
                key=lambda item: (
                    item.compiler_lowering_sha256,
                    item.context_tokens,
                ),
            )
        )
        keys = [
            (record.compiler_lowering_sha256, record.context_tokens)
            for record in ordered
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("full-model artifact set contains duplicate lowering records")
        lowering_keys: dict[str, str] = {}
        sources: set[str] = set()
        for record in ordered:
            previous = lowering_keys.setdefault(
                record.compiler_lowering_sha256,
                record.lowering_key_json,
            )
            if previous != record.lowering_key_json:
                raise ValueError("full-model lowering-key identity collision")
            sources.add(record.compiler_source_sha256)

        family_by_key: dict[str, FullModelDecodeArtifactFamily] = {}
        supplied_families = list(families)
        for record in ordered:
            supplied_families.append(
                FullModelDecodeArtifactFamily.from_point_descriptor(
                    record.descriptor,
                    compiler_source_sha256=record.compiler_source_sha256,
                )
            )
        for family in supplied_families:
            if not isinstance(family, FullModelDecodeArtifactFamily):
                raise TypeError("full-model artifact set families are malformed")
            key_sha256 = hashlib.sha256(
                family.family_key_json.encode("utf-8")
            ).hexdigest()
            previous = family_by_key.setdefault(key_sha256, family)
            if previous != family:
                raise ValueError("full-model artifact family-key identity collision")
            sources.add(family.compiler_source_sha256)
        if not ordered and not family_by_key:
            raise ValueError("full-model artifact set must contain records or families")
        if len(sources) != 1:
            raise ValueError("full-model artifact set mixes compiler revisions")
        self._records = ordered
        self._index = dict(zip(keys, ordered, strict=True))
        self._generated_index: dict[
            tuple[str, int], FullModelDecodeArtifactRecord
        ] = {}
        self._resolved_index: dict[
            tuple[str, int], FullModelDecodeArtifactRecord
        ] = {}
        self._lowering_keys = lowering_keys
        self._families = tuple(
            family_by_key[key] for key in sorted(family_by_key)
        )
        self._family_by_key = family_by_key
        self._lock = threading.RLock()
        self.compiler_source_sha256 = next(iter(sources))

    @property
    def records(self) -> tuple[FullModelDecodeArtifactRecord, ...]:
        return self._records

    @property
    def families(self) -> tuple[FullModelDecodeArtifactFamily, ...]:
        return self._families

    @property
    def artifact_set_id(self) -> str:
        return FULL_MODEL_ARTIFACT_ID_PREFIX + canonical_sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": FULL_MODEL_ARTIFACT_SET_SCHEMA,
            "artifact_scope": FULL_MODEL_DECODE_SCOPE,
            "compiler_source_sha256": self.compiler_source_sha256,
            "families": [family.to_dict() for family in self.families],
            "records": [record.to_dict() for record in self.records],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "artifact_set_id": self.artifact_set_id}

    def write(self, path: str | Path) -> Path:
        output = Path(path).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "FullModelDecodeArtifactSet":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("full-model artifact set must be an object")
        required = {
            "schema_version",
            "artifact_scope",
            "compiler_source_sha256",
            "records",
            "artifact_set_id",
        }
        supported_fields = (required, required | {"families"})
        if set(value) not in supported_fields:
            raise ValueError("full-model artifact set fields differ from the schema")
        if value["schema_version"] != FULL_MODEL_ARTIFACT_SET_SCHEMA:
            raise ValueError("unsupported full-model artifact set schema")
        if value["artifact_scope"] != FULL_MODEL_DECODE_SCOPE:
            raise ValueError("artifact set lacks full-model decode scope")
        raw_records = value["records"]
        if not isinstance(raw_records, list) or not all(
            isinstance(item, Mapping) for item in raw_records
        ):
            raise TypeError("full-model artifact records must be objects")
        raw_families = value.get("families", [])
        if not isinstance(raw_families, list) or not all(
            isinstance(item, Mapping) for item in raw_families
        ):
            raise TypeError("full-model artifact families must be objects")
        artifact_set = cls(
            (
                FullModelDecodeArtifactRecord.from_dict(item)
                for item in raw_records
            ),
            families=(
                FullModelDecodeArtifactFamily.from_dict(item)
                for item in raw_families
            ),
        )
        if value["compiler_source_sha256"] != artifact_set.compiler_source_sha256:
            raise ValueError("artifact-set compiler identity is inconsistent")
        if value["artifact_set_id"] != artifact_set.artifact_set_id:
            raise ValueError("full-model artifact-set identity is stale")
        return artifact_set

    def lowering_key(self, compiler_lowering_sha256: str) -> dict[str, object]:
        try:
            encoded = self._lowering_keys[compiler_lowering_sha256]
        except KeyError as error:
            raise KeyError("compiler lowering key is absent from the artifact set") from error
        return json.loads(encoded)

    def has_lowering_key(self, compiler_lowering_sha256: str) -> bool:
        with self._lock:
            return (
                compiler_lowering_sha256 in self._lowering_keys
            )

    def family(
        self,
        point_descriptor: object,
    ) -> FullModelDecodeArtifactFamily:
        family_json, family_key_sha256, _ = full_model_decode_family_key(
            point_descriptor
        )
        try:
            family = self._family_by_key[family_key_sha256]
        except KeyError as error:
            raise KeyError("compiler family is absent from the artifact set") from error
        if family.family_key_json != family_json:
            raise ValueError("compiler family-key identity collision")
        return family

    def contexts(self, compiler_lowering_sha256: str) -> tuple[int, ...]:
        with self._lock:
            contexts = {
                record.context_tokens
                for record in self.records
                if record.compiler_lowering_sha256 == compiler_lowering_sha256
            }
            contexts.update(
                context
                for lowering, context in self._generated_index
                if lowering == compiler_lowering_sha256
            )
        return tuple(sorted(contexts))

    def record(
        self,
        compiler_lowering_sha256: str,
        context_tokens: int,
    ) -> FullModelDecodeArtifactRecord:
        key = (compiler_lowering_sha256, context_tokens)
        with self._lock:
            record = self._index.get(key) or self._generated_index.get(key)
        if record is None:
            raise KeyError(
                "exact compiler lowering/context artifact is missing"
            )
        return record

    def resolve_record(
        self,
        point_descriptor: object,
        *,
        context_tokens: int,
    ) -> FullModelDecodeArtifactRecord:
        """Return or lazily instantiate the exact lowering/context record."""

        descriptor_json, point_sha256, descriptor = _point_descriptor(
            point_descriptor
        )
        _, lowering_sha256, _ = full_model_decode_lowering_key(descriptor)
        cache_key = (lowering_sha256, int(context_tokens))
        resolution_key = (point_sha256, int(context_tokens))
        with self._lock:
            resolved = self._resolved_index.get(resolution_key)
            record = self._index.get(cache_key) or self._generated_index.get(cache_key)
        if resolved is not None:
            return resolved
        if record is not None and record.descriptor_json == descriptor_json:
            with self._lock:
                self._resolved_index[resolution_key] = record
            return record
        if record is not None:
            record_batch = int(record.descriptor["serving"]["batch"])
            requested_batch = int(descriptor["serving"]["batch"])
            if record_batch == requested_batch:
                receipt = record.compiler_receipt
                precision = descriptor["precision"]
                model = descriptor["model"]
                if not isinstance(precision, Mapping) or not isinstance(
                    model,
                    Mapping,
                ):
                    raise TypeError("compiler point alias binding is malformed")
                weight_format, kv_format, key_format, value_format = (
                    _resolved_storage_formats(precision)
                )
                rebound_receipt = {
                    **receipt,
                    "weight_storage_format": weight_format,
                    "kv_storage_format": kv_format,
                    "key_storage_format": key_format,
                    "value_storage_format": value_format,
                    "output_head_location": model["output_head_location"],
                }
                rebound = FullModelDecodeArtifactRecord(
                    descriptor_json=descriptor_json,
                    lowering_key_json=record.lowering_key_json,
                    context_tokens=int(context_tokens),
                    compiler_source_sha256=record.compiler_source_sha256,
                    lowering_contract_json=record.lowering_contract_json,
                    compiler_receipt_json=_canonical_json_text(rebound_receipt),
                    compilation_artifact=record.compilation_artifact,
                )
                with self._lock:
                    self._resolved_index[resolution_key] = rebound
                return rebound
        family = self.family(descriptor)
        generated = FullModelDecodeLazyArtifactGenerator(family).instantiate(
            descriptor,
            context_tokens=int(context_tokens),
        )
        with self._lock:
            self._generated_index.setdefault(cache_key, generated)
            self._resolved_index[resolution_key] = generated
            self._lowering_keys.setdefault(
                lowering_sha256,
                generated.lowering_key_json,
            )
            return generated


class FullModelDecodeLazyArtifactGenerator:
    """Instantiate exact native records from one sealed compiler family."""

    def __init__(self, family: FullModelDecodeArtifactFamily) -> None:
        if not isinstance(family, FullModelDecodeArtifactFamily):
            raise TypeError("full-model artifact family is required")
        self.family = family

    def instantiate(
        self,
        point_descriptor: object,
        *,
        context_tokens: int,
    ) -> FullModelDecodeArtifactRecord:
        descriptor_json, _, descriptor = _point_descriptor(point_descriptor)
        family_json, _, _ = full_model_decode_family_key(descriptor)
        if family_json != self.family.family_key_json:
            raise ValueError("compiler point differs from the sealed artifact family")
        if (
            isinstance(context_tokens, bool)
            or not isinstance(context_tokens, int)
            or context_tokens <= 1
        ):
            raise ValueError("full-model decode context must exceed one token")
        blockers = full_model_decode_generator_blockers(descriptor)
        if blockers:
            raise ValueError(
                "native full-model generator cannot lower this point: "
                + ", ".join(blockers)
            )
        context_resolution = full_model_decode_context_resolution(
            descriptor,
            context_tokens=context_tokens,
        )

        lowering_json, _, lowering = full_model_decode_lowering_key(descriptor)
        dimensions = lowering["model"]["dimensions"]
        geometry = lowering["array_geometry"]
        rank_partition = lowering["rank_partition"]
        serving = lowering["serving"]
        descriptor_serving = descriptor["serving"]
        precision = lowering["precision"]
        model_section = descriptor["model"]
        if not all(
            isinstance(item, Mapping)
            for item in (
                dimensions,
                geometry,
                rank_partition,
                serving,
                precision,
                model_section,
                descriptor_serving,
            )
        ):
            raise TypeError("compiler lazy-instantiation descriptor is malformed")

        hidden = int(dimensions["hidden"])
        heads = int(dimensions["heads"])
        kv_heads = int(dimensions["kv_heads"])
        head_dim = int(dimensions["head_dim"])
        layers = int(dimensions["layers"])
        mlen = int(geometry["mlen"])
        hlen = int(geometry["hlen"])
        tensor_parallel_degree = int(rank_partition["tp"])
        kv_parallel_degree = int(rank_partition["kvp"])
        tensor_parallel_rank = 0
        kv_parallel_rank = (context_tokens - 1) % kv_parallel_degree
        local_kv_heads = kv_heads // tensor_parallel_degree
        config = SimpleNamespace(
            hidden_size=hidden,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            head_dim=head_dim,
            intermediate_size=int(dimensions["inter"]),
            rms_norm_eps=1.0e-5,
            rope_theta=10000.0,
            vocab_size=int(dimensions["vocab"]),
            model_type=str(dimensions.get("model_type", "llama")),
            qk_norm=bool(dimensions.get("qk_norm", False)),
            use_qk_norm=bool(dimensions.get("qk_norm", False)),
        )
        model = SimpleNamespace(
            config=config,
            layers=[None] * layers,
        )
        from compiler.aten.plena.packed_kv import PackedKVLayout
        from compiler.aten.plena_frontend import compile_native_hf_decoder

        descriptor_precision = descriptor["precision"]
        if not isinstance(descriptor_precision, Mapping):
            raise TypeError("compiler point precision descriptor is malformed")
        weight_format, kv_format, key_format, value_format = (
            _resolved_storage_formats(descriptor_precision)
        )
        key_layout = PackedKVLayout(
            kv_heads=local_kv_heads,
            head_dim=head_dim,
            mlen=mlen,
            block_size=int(precision["block_size"]),
            element_bits=int(precision["key_element_bits"]),
            scale_bits=int(precision["weight_scale_bits"]),
        )
        value_layout = PackedKVLayout(
            kv_heads=local_kv_heads,
            head_dim=head_dim,
            mlen=mlen,
            block_size=int(precision["block_size"]),
            element_bits=int(precision["value_element_bits"]),
            scale_bits=int(precision["weight_scale_bits"]),
        )

        # The frontend prints a human-oriented compilation report; generation
        # is a library operation here, so keep production launch output clean.
        with redirect_stdout(io.StringIO()):
            compilation_result = compile_native_hf_decoder(
                model,
                seq_len=1,
                batch_size=int(descriptor_serving["batch"]),
                num_layers=layers,
                mlen=mlen,
                blen=int(geometry["blen"]),
                hlen=hlen,
                broadcast_amount=mlen // hlen,
                attention_head_packing=True,
                packed_kv_layout=key_layout,
                packed_value_layout=value_layout,
                decode_context_tokens=context_tokens,
                external_packed_kv_cache=True,
                trace_only=True,
                output_head_location=str(
                    model_section["output_head_location"]
                ),
                weight_element_bits=int(
                    precision["attention_weight_element_bits"]
                ),
                weight_block_size=int(precision["block_size"]),
                weight_scale_bits=int(precision["weight_scale_bits"]),
                weight_storage_format=weight_format,
                kv_storage_format=kv_format,
                key_storage_format=key_format,
                value_storage_format=value_format,
                tensor_parallel_degree=tensor_parallel_degree,
                kv_parallel_degree=kv_parallel_degree,
                tensor_parallel_rank=tensor_parallel_rank,
                kv_parallel_rank=kv_parallel_rank,
                kv_token_sharding="round_robin",
                kv_head_reuse=bool(rank_partition["kv_head_reuse"]),
            )
        builder = FullModelDecodeArtifactSetBuilder(
            self.family.compiler_source_sha256
        )
        record = builder.add(
            json.loads(descriptor_json),
            context_tokens=context_tokens,
            compilation_result=compilation_result,
            critical_rank={
                "tensor_parallel_rank": tensor_parallel_rank,
                "kv_parallel_rank": kv_parallel_rank,
                "kv_token_sharding": "round_robin",
                "owns_current_token": True,
            },
        )
        if record.lowering_key_json != lowering_json:
            raise RuntimeError("lazy native record differs from its lowering signature")
        receipt = record.compiler_receipt
        if (
            int(receipt["batch_size"]) != context_resolution.batch
            or int(receipt["cache_rows_per_batch"])
            != context_resolution.local_cache_rows_per_batch
        ):
            raise RuntimeError("native record differs from its algebraic resolution")
        return record


@dataclass(frozen=True)
class FullModelDecodeArtifactBuildPlan:
    """Compact dry-run cardinality for algebraic full-model trace templates."""

    full_point_count: int
    unique_family_count: int
    unique_lowering_key_count: int
    unique_native_template_count: int
    unique_batch_record_count: int
    exact_point_context_resolution_count: int
    unique_lowering_context_resolution_count: int
    projected_trace_bytes: int
    lowering_key_context_counts: tuple[tuple[str, int], ...]
    capability_blocker_counts: tuple[tuple[str, int], ...] = ()
    max_trace_generation_calls: int = DEFAULT_MAX_TRACE_GENERATION_CALLS
    max_projected_trace_bytes: int = DEFAULT_MAX_PROJECTED_TRACE_BYTES
    schema_version: str = FULL_MODEL_BUILD_PLAN_SCHEMA

    @classmethod
    def from_point_contexts(
        cls,
        point_contexts: Iterable[tuple[object, Iterable[int]]],
        *,
        max_trace_generation_calls: int = DEFAULT_MAX_TRACE_GENERATION_CALLS,
        max_projected_trace_bytes: int = DEFAULT_MAX_PROJECTED_TRACE_BYTES,
    ) -> "FullModelDecodeArtifactBuildPlan":
        contexts_by_key: dict[str, set[int]] = {}
        lowerings_by_template: dict[str, dict[str, object]] = {}
        lowerings_by_batch_record: dict[str, dict[str, object]] = {}
        family_keys: set[str] = set()
        blocker_keys: dict[str, set[str]] = {}
        full_point_count = 0
        exact_point_context_resolution_count = 0
        for point_descriptor, context_tokens in point_contexts:
            _, lowering_sha256, lowering = full_model_decode_lowering_key(
                point_descriptor
            )
            _, native_template_sha256, _ = full_model_decode_native_template_key(
                point_descriptor
            )
            lowering_serving = lowering.get("serving")
            if not isinstance(lowering_serving, Mapping):
                raise TypeError("compiler point serving batch is missing")
            batch_record_sha256 = canonical_sha256(
                {
                    "native_template_sha256": native_template_sha256,
                    "batch": int(lowering_serving["batch"]),
                }
            )
            _, family_sha256, _ = full_model_decode_family_key(
                point_descriptor
            )
            family_keys.add(family_sha256)
            lowerings_by_template.setdefault(native_template_sha256, lowering)
            lowerings_by_batch_record.setdefault(batch_record_sha256, lowering)
            for blocker in full_model_decode_generator_blockers(point_descriptor):
                blocker_keys.setdefault(blocker, set()).add(lowering_sha256)
            contexts = contexts_by_key.setdefault(lowering_sha256, set())
            point_contexts_seen: set[int] = set()
            for raw_context in context_tokens:
                context = int(raw_context)
                if context <= 1:
                    raise ValueError("full-model decode context must exceed one token")
                contexts.add(context)
                point_contexts_seen.add(context)
            if not point_contexts_seen:
                raise ValueError("every compiler point requires sampled contexts")
            exact_point_context_resolution_count += len(point_contexts_seen)
            full_point_count += 1
        if not full_point_count:
            raise ValueError("artifact dry-run requires compiler points")
        counts = tuple(
            sorted(
                (lowering_sha256, len(contexts))
                for lowering_sha256, contexts in contexts_by_key.items()
            )
        )
        projected_trace_bytes = sum(
            _projected_full_model_trace_bytes(lowering)
            for lowering in lowerings_by_batch_record.values()
        )
        return cls(
            full_point_count=full_point_count,
            unique_family_count=len(family_keys),
            unique_lowering_key_count=len(counts),
            unique_native_template_count=len(lowerings_by_template),
            unique_batch_record_count=len(lowerings_by_batch_record),
            exact_point_context_resolution_count=(
                exact_point_context_resolution_count
            ),
            unique_lowering_context_resolution_count=sum(
                count for _, count in counts
            ),
            projected_trace_bytes=projected_trace_bytes,
            lowering_key_context_counts=counts,
            capability_blocker_counts=tuple(
                sorted((name, len(keys)) for name, keys in blocker_keys.items())
            ),
            max_trace_generation_calls=int(max_trace_generation_calls),
            max_projected_trace_bytes=int(max_projected_trace_bytes),
        )

    @property
    def preflight_blockers(self) -> tuple[str, ...]:
        blockers = [
            f"unsupported_native_lowering:{name}:{count}"
            for name, count in self.capability_blocker_counts
        ]
        if self.unique_batch_record_count > self.max_trace_generation_calls:
            blockers.append(
                "projected_trace_generation_calls_exceed_limit:"
                f"{self.unique_batch_record_count}>"
                f"{self.max_trace_generation_calls}"
            )
        if self.projected_trace_bytes > self.max_projected_trace_bytes:
            blockers.append(
                "projected_trace_bytes_exceed_limit:"
                f"{self.projected_trace_bytes}>{self.max_projected_trace_bytes}"
            )
        return tuple(blockers)

    @property
    def compiler_trace_preflight_feasible(self) -> bool:
        return not self.preflight_blockers

    @property
    def context_artifact_count(self) -> int:
        """No context-specific artifact aliases are materialized."""

        return 0

    def to_dict(self, *, include_lowering_keys: bool = False) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "full_point_count": self.full_point_count,
            "unique_compiler_family_artifacts": self.unique_family_count,
            "unique_lowering_key_count": self.unique_lowering_key_count,
            "context_artifact_count": self.context_artifact_count,
            "exact_point_context_resolutions": (
                self.exact_point_context_resolution_count
            ),
            "unique_lowering_context_resolutions": (
                self.unique_lowering_context_resolution_count
            ),
            "unique_native_trace_templates": self.unique_native_template_count,
            "unique_exact_batch_records": self.unique_batch_record_count,
            "unique_lazy_trace_instantiations": self.unique_batch_record_count,
            "projected_trace_generation_calls": self.unique_batch_record_count,
            "projected_trace_bytes": self.projected_trace_bytes,
            "compiler_trace_preflight_feasible": (
                self.compiler_trace_preflight_feasible
            ),
            "compiler_trace_preflight_blockers": list(self.preflight_blockers),
            "max_trace_generation_calls": self.max_trace_generation_calls,
            "max_projected_trace_bytes": self.max_projected_trace_bytes,
            "compile_count_formula": "|unique (native template, batch) keys|",
            "lazy_instantiation_formula": (
                "|unique (native template, batch) keys|; context and storage "
                "multiplicities resolve algebraically"
            ),
            "trace_byte_projection": (
                "sum(64KiB + layers*96KiB + "
                "layers*heads*2*512B) over native templates"
            ),
            "materialized_alias_records": 0,
            "materialized_context_rows": 0,
            "materialized_batch_alias_records": 0,
            "avoided_full_point_alias_records": self.full_point_count,
        }
        if include_lowering_keys:
            value["lowering_key_context_counts"] = [
                {
                    "compiler_lowering_sha256": identity,
                    "context_count": count,
                }
                for identity, count in self.lowering_key_context_counts
            ]
        return value


def _projected_full_model_trace_bytes(
    lowering_key: Mapping[str, object],
) -> int:
    """Serialized size of one body, full-block, and masked-tail template."""

    model = lowering_key.get("model")
    if not isinstance(model, Mapping):
        raise TypeError("trace-byte projection requires a complete lowering key")
    dimensions = model.get("dimensions")
    if not isinstance(dimensions, Mapping):
        raise TypeError("trace-byte projection requires model dimensions")
    layers = int(dimensions["layers"])
    heads = int(dimensions["heads"])
    return (
        64 * 1024
        + layers * 96 * 1024
        + layers * heads * 2 * 512
    )


class FullModelDecodeArtifactSetBuilder:
    """Build an artifact set only from complete, scope-bound records."""

    def __init__(self, compiler_source_sha256: str) -> None:
        _require_sha256(
            compiler_source_sha256,
            "full-model artifact compiler-source identity",
        )
        self.compiler_source_sha256 = compiler_source_sha256
        self._records: list[FullModelDecodeArtifactRecord] = []

    @staticmethod
    def dry_run(
        point_contexts: Iterable[tuple[object, Iterable[int]]],
    ) -> FullModelDecodeArtifactBuildPlan:
        return FullModelDecodeArtifactBuildPlan.from_point_contexts(
            point_contexts
        )

    def add(
        self,
        point_descriptor: object,
        *,
        context_tokens: int,
        compilation_result: Mapping[str, object],
        critical_rank: Mapping[str, object],
    ) -> FullModelDecodeArtifactRecord:
        if not isinstance(compilation_result, Mapping):
            raise TypeError("full-model compilation result must be an object")
        compilation_artifact = compilation_result.get("compilation_artifact")
        compiler_receipt = compilation_result.get("info")
        if not isinstance(compilation_artifact, CompilationArtifact):
            raise TypeError("full-model compilation result lacks its artifact")
        if not isinstance(compiler_receipt, Mapping):
            raise TypeError("full-model compilation result lacks its compiler receipt")
        descriptor_json, _, descriptor = _point_descriptor(
            point_descriptor
        )
        lowering_key_json, lowering_sha256, lowering_key = (
            full_model_decode_lowering_key(descriptor)
        )
        model = descriptor["model"]
        serving = descriptor["serving"]
        hardware = descriptor["hardware"]
        if not all(isinstance(item, Mapping) for item in (model, serving, hardware)):
            raise TypeError("compiler point descriptor sections are missing")
        dimensions = model["dimensions"]
        if not isinstance(dimensions, Mapping):
            raise TypeError("compiler point model dimensions are missing")
        contract = {
            "schema_version": "plena-full-model-decode-lowering-v1",
            "artifact_scope": FULL_MODEL_DECODE_SCOPE,
            "compiler_lowering_sha256": lowering_sha256,
            "query_tokens": 1,
            "context_tokens": context_tokens,
            "context_resolution": full_model_decode_context_resolution(
                descriptor,
                context_tokens=context_tokens,
            ).to_dict(),
            "batch": int(serving["batch"]),
            "cache_semantics": FULL_MODEL_CACHE_SEMANTICS,
            "layer_scope": "all_decoder_layers",
            "decoder_layers": int(dimensions["layers"]),
            "output_head_included": False,
            "output_head_ownership": "external_to_decoder_artifact",
            "geometry": lowering_key["array_geometry"],
            "rank_partition": lowering_key["rank_partition"],
            "critical_rank": dict(critical_rank),
        }
        record = FullModelDecodeArtifactRecord(
            descriptor_json=descriptor_json,
            lowering_key_json=lowering_key_json,
            context_tokens=context_tokens,
            compiler_source_sha256=self.compiler_source_sha256,
            lowering_contract_json=_canonical_json_text(contract),
            compiler_receipt_json=_canonical_json_text(compiler_receipt),
            compilation_artifact=compilation_artifact,
        )
        self._records.append(record)
        return record

    def build(self) -> FullModelDecodeArtifactSet:
        return FullModelDecodeArtifactSet(self._records)


class FullModelDecodeArtifactBuilder:
    """Resolve immutable compilation artifacts for one exact point."""

    scope = FULL_MODEL_DECODE_SCOPE

    def __init__(
        self,
        artifact_set: FullModelDecodeArtifactSet,
        compiler_lowering_sha256: str,
        point_descriptor: object | None = None,
    ) -> None:
        self.artifact_set = artifact_set
        self.compiler_lowering_sha256 = compiler_lowering_sha256
        self.compiler_source_sha256 = artifact_set.compiler_source_sha256
        self.point_descriptor = point_descriptor
        if point_descriptor is None:
            artifact_set.lowering_key(compiler_lowering_sha256)
        else:
            _, derived, _ = full_model_decode_lowering_key(point_descriptor)
            if derived != compiler_lowering_sha256:
                raise ValueError("artifact builder point has a different lowering key")
            artifact_set.family(point_descriptor)

    def __call__(
        self,
        request: CompilerTraceTimingRequest,
    ) -> BoundCompilerTrace:
        if request.compiler_lowering_sha256 != self.compiler_lowering_sha256:
            raise ValueError("compiler request differs from the artifact lowering key")
        if request.compiler_source_sha256 != self.compiler_source_sha256:
            raise ValueError("compiler request uses a stale compiler identity")
        if self.point_descriptor is None:
            record = self.artifact_set.record(
                self.compiler_lowering_sha256,
                request.context_tokens,
            )
        else:
            if request.compiler_point_descriptor_json != _point_descriptor(
                self.point_descriptor
            )[0]:
                raise ValueError("compiler request differs from the builder point")
            record = self.artifact_set.resolve_record(
                self.point_descriptor,
                context_tokens=request.context_tokens,
            )
        return BoundCompilerTrace.from_compilation_artifact(
            record.compilation_artifact,
            compiler_source_sha256=self.compiler_source_sha256,
            hbm=request.hbm,
            compiler_lowering_sha256=self.compiler_lowering_sha256,
            artifact_record_sha256=record.record_sha256,
        )


class ReferenceDecodeArtifactBuilder:
    """Compile the exact one-layer decoder program used for stage validation."""

    scope = REFERENCE_DECODE_SCOPE

    def __init__(
        self,
        lowering: ReferenceDecodeLowering,
        *,
        settings_path: str | Path = DEFAULT_SETTINGS,
    ) -> None:
        if not isinstance(lowering, ReferenceDecodeLowering):
            raise TypeError("reference decode lowering inputs are required")
        self.lowering = lowering
        self.settings_path = Path(settings_path).resolve()
        if not self.settings_path.is_file():
            raise FileNotFoundError(self.settings_path)
        self.configuration_sha256 = _sha256_file(self.settings_path)
        self.compiler_source_sha256 = reference_decode_compiler_source_sha256()

        from transactional_emulator.testbench.misc.decoder_decode_asm_gen import (
            decode_geometry,
        )

        geometry = decode_geometry(
            str(self.settings_path),
            kv_heads=lowering.kv_heads,
            kv_head_reuse=lowering.kv_head_reuse,
            row_tile=lowering.row_tile,
        )
        self.geometry = ArrayGeometry(
            mlen=int(geometry["mlen"]),
            blen=int(geometry["blen"]),
            vlen=int(geometry["mlen"]),
            hlen=int(geometry["hlen"]),
        )
        self.query_rows = int(geometry["batch"])

    def compiler_inputs_sha256(
        self,
        *,
        context_tokens: int,
        batch: int,
    ) -> str:
        """Seal all program-shaping inputs for one reference trace point."""

        return canonical_sha256(
            {
                "schema_version": "plena-reference-decode-lowering-inputs-v1",
                "scope": self.scope,
                "lowering": self.lowering.to_dict(),
                "configuration_sha256": self.configuration_sha256,
                "context_tokens": context_tokens,
                "batch": batch,
                "geometry": self.geometry.to_dict(),
            }
        )

    def request(
        self,
        *,
        context_tokens: int,
        hbm: HBMOperatingPoint,
        frequency_hz: float,
    ) -> CompilerTraceTimingRequest:
        """Construct the only request shape this reference lowering represents."""

        self._validate_shape(context_tokens, self.query_rows)
        return CompilerTraceTimingRequest(
            compiler_inputs_sha256=self.compiler_inputs_sha256(
                context_tokens=context_tokens,
                batch=self.query_rows,
            ),
            compiler_source_sha256=self.compiler_source_sha256,
            context_tokens=context_tokens,
            batch=self.query_rows,
            geometry=self.geometry,
            hbm=hbm,
            frequency_hz=frequency_hz,
        )

    def _validate_shape(self, context_tokens: int, batch: int) -> None:
        if batch != self.query_rows:
            raise ValueError(
                "reference decoder trace uses shared-cache query rows and requires "
                f"batch={self.query_rows}, not {batch}; it cannot price an "
                "independent-request serving batch"
            )
        if context_tokens <= batch:
            raise ValueError(
                "reference decoder context must exceed its shared-cache query rows"
            )
        if context_tokens % self.geometry.mlen:
            raise ValueError(
                "reference decoder context must be a whole number of MLEN tiles"
            )

    def __call__(
        self,
        request: CompilerTraceTimingRequest,
    ) -> BoundCompilerTrace:
        if not isinstance(request, CompilerTraceTimingRequest):
            raise TypeError("reference decoder compiler requires a timing request")
        if request.geometry != self.geometry:
            raise ValueError("reference decoder geometry differs from the request")
        self._validate_shape(request.context_tokens, request.batch)
        if request.compiler_source_sha256 != self.compiler_source_sha256:
            raise ValueError("reference decoder request has a stale compiler identity")
        expected_inputs = self.compiler_inputs_sha256(
            context_tokens=request.context_tokens,
            batch=request.batch,
        )
        if request.compiler_inputs_sha256 != expected_inputs:
            raise ValueError("reference decoder request has stale lowering inputs")
        if _sha256_file(self.settings_path) != self.configuration_sha256:
            raise RuntimeError(
                "reference decoder settings changed after request creation"
            )
        if (
            reference_decode_compiler_source_sha256()
            != self.compiler_source_sha256
        ):
            raise RuntimeError(
                "reference decoder compiler sources changed after request creation"
            )

        from transactional_emulator.testbench.misc.decoder_decode_asm_gen import (
            generate_decode_asm,
        )

        with tempfile.TemporaryDirectory(prefix="plena-decode-trace-") as build_dir:
            # The assembly templates print allocator diagnostics.  The builder
            # captures them so library use remains quiet while compiler errors
            # still propagate normally.
            with redirect_stdout(io.StringIO()):
                generated = generate_decode_asm(
                    kv_size=request.context_tokens,
                    hidden=self.geometry.mlen,
                    inter=self.lowering.intermediate_size,
                    head_dim=self.geometry.hlen,
                    build_dir=build_dir,
                    vocab=self.lowering.vocabulary_size,
                    kv_head_reuse=self.lowering.kv_head_reuse,
                    kv_heads=self.lowering.kv_heads,
                    row_tile=self.lowering.row_tile,
                    settings_toml=str(self.settings_path),
                    activation_element_bits=(
                        self.lowering.activation_element_bits
                    ),
                    weight_element_bits=self.lowering.weight_element_bits,
                    kv_element_bits=self.lowering.kv_element_bits,
                    block_size=self.lowering.block_size,
                    scale_bits=self.lowering.scale_bits,
                    verbose=False,
                )
        artifact = generated.get("compilation_artifact")
        if not isinstance(artifact, CompilationArtifact):
            raise TypeError(
                "reference decoder lowering did not return a compilation artifact"
            )
        if int(generated.get("s_q", -1)) != request.batch:
            raise ValueError(
                "reference decoder lowering changed its shared-cache row count"
            )
        return BoundCompilerTrace.from_compilation_artifact(
            artifact,
            compiler_source_sha256=self.compiler_source_sha256,
            hbm=request.hbm,
        )


class StageMemoryPricer(Protocol):
    """Calibrated callback that prices physical descriptor runs in seconds."""

    calibration_id: str
    base_calibration_id: str

    def price_trace(
        self,
        trace: ExecutionTrace,
        request_memory: RequestMemorySidecar,
        hbm: HBMOperatingPoint,
    ) -> dict[str, float]:
        ...


class RequestModelStageMemoryPricer:
    """Algebraic adapter for isolated and ordered request-model pricing."""

    def __init__(self, request_model: object) -> None:
        calibration_id = getattr(request_model, "calibration_id", None)
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise ValueError("request-memory calibration identity is required")
        _require_request_calibration_id(calibration_id)
        predictor = getattr(request_model, "predict", None)
        if not callable(predictor):
            raise TypeError("request-memory model must provide predict(descriptor)")
        stream_predictor = getattr(request_model, "predict_stream", None)
        if not callable(stream_predictor):
            raise TypeError(
                "request-memory model must provide predict_stream(runs)"
            )
        stream_composition = getattr(
            request_model,
            "stream_composition_schema",
            None,
        )
        if stream_composition != REQUEST_STREAM_COMPOSITION_SCHEMA:
            raise ValueError(
                "request-memory model has an unsupported stream composition"
            )
        self._request_model = request_model
        self.base_calibration_id = calibration_id
        self.calibration_id = "request-latency-" + canonical_sha256(
            {
                "base_request_calibration_id": calibration_id,
                "request_stream_composition": stream_composition,
            }
        )

    @staticmethod
    def _validate_hbm_geometry(hbm: HBMOperatingPoint) -> None:
        expected_geometry = (
            HBM_CHANNEL_WIDTH_BITS,
            HBM_PSEUDOCHANNELS,
            HBM_BANK_GROUPS,
            HBM_BANKS_PER_GROUP,
            HBM_TRANSACTION_BYTES,
        )
        observed_geometry = (
            hbm.channel_width_bits,
            hbm.pseudochannels,
            hbm.bank_groups,
            hbm.banks_per_group,
            hbm.transaction_bytes,
        )
        if observed_geometry != expected_geometry:
            raise ValueError(
                "HBM geometry differs from the calibrated request-memory mapper"
            )

    def price_trace(
        self,
        trace: ExecutionTrace,
        request_memory: RequestMemorySidecar,
        hbm: HBMOperatingPoint,
    ) -> dict[str, float]:
        """Price a complete trace with open-row state carried across stages."""

        return self.price_trace_with_entry_weights(
            trace,
            request_memory,
            hbm,
            entry_weight=lambda _entry: 1.0,
        )

    def price_trace_by_engine(
        self,
        trace: ExecutionTrace,
        request_memory: RequestMemorySidecar,
        hbm: HBMOperatingPoint,
    ) -> dict[str, dict[str, float]]:
        """Split each stage's priced time across the matrix and vector engines.

        The two engines expose their service time differently: the matrix tile
        prefetch is double buffered and issued ahead of the compute that reads
        it, while the vector load and writeback engine holds a single request
        that the consuming instruction waits on.  Open rows are carried through
        the one undivided stream, so the split never changes the total.
        """

        return self.price_trace_with_entry_weights(
            trace,
            request_memory,
            hbm,
            entry_weight=lambda _entry: 1.0,
            by_engine=True,
        )

    def price_trace_with_entry_weights(
        self,
        trace: ExecutionTrace,
        request_memory: RequestMemorySidecar,
        hbm: HBMOperatingPoint,
        *,
        entry_weight: Callable[[ExecutionTraceEntry], float],
        by_engine: bool = False,
    ) -> dict[str, float] | dict[str, dict[str, float]]:
        """Price an ordered stream after a declared algebraic residency map.

        Zero-weight entries are removed before stream prediction, so a fully
        resident transfer cannot perturb later open-row state.  Fractional
        residency retains the request ordering and scales its calibrated
        contribution after prediction.
        """

        self._validate_hbm_geometry(hbm)
        request_memory.validate(trace, hbm)
        ordered: list[tuple[str, RequestDescriptorRun, float]] = []
        for binding in sorted(
            request_memory.bindings,
            key=lambda item: item.trace_entry_index,
        ):
            entry = trace.entries[binding.trace_entry_index]
            weight = float(entry_weight(entry))
            if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                raise ValueError("request residency weight must be within [0, 1]")
            if weight:
                ordered.extend((entry.stage, run, weight) for run in binding.runs)
        expanded: list[tuple[str, DMARequestDescriptor, int, float]] = []
        for stage, run, weight in ordered:
            if run.address_varying:
                expanded.extend(
                    (stage, run.descriptor_at(index), 1, weight)
                    for index in range(run.repetitions)
                )
            else:
                expanded.append(
                    (stage, run.descriptor, run.repetitions, weight)
                )
        predictions = self._request_model.predict_stream(
            tuple(
                (descriptor, repetitions)
                for _stage, descriptor, repetitions, _weight in expanded
            )
        )
        if len(predictions) != len(expanded):
            raise ValueError(
                "request-memory stream model returned the wrong number of runs"
            )
        totals: dict[str, float] = {}
        by_engine_totals: dict[str, dict[str, float]] = {}
        for (stage, descriptor, _repetitions, weight), prediction in zip(
            expanded,
            predictions,
            strict=True,
        ):
            seconds = float(getattr(prediction, "seconds", float("nan")))
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(
                    "request-memory stream model returned an invalid latency"
                )
            totals[stage] = totals.get(stage, 0.0) + seconds * weight
            if by_engine:
                engine = dma_engine(descriptor.opcode)
                bucket = by_engine_totals.setdefault(stage, {})
                bucket[engine] = bucket.get(engine, 0.0) + seconds * weight
        return by_engine_totals if by_engine else totals


class ResidencyAdjustedStageMemoryPricer:
    """Apply an SRAM-residency policy to one reusable streaming trace."""

    _KV_FRACTIONS = {
        "kv_resident_25": 0.25,
        "kv_resident_50": 0.50,
        "kv_resident_75": 0.75,
        "kv_resident_100": 1.00,
    }
    _POLICIES = {"streaming", "projection_resident", *_KV_FRACTIONS}

    def __init__(
        self,
        base_pricer: RequestModelStageMemoryPricer,
        policy: str,
    ) -> None:
        if not isinstance(base_pricer, RequestModelStageMemoryPricer):
            raise TypeError("residency pricing requires the request-model pricer")
        if policy not in self._POLICIES:
            raise ValueError(f"unsupported SRAM residency policy {policy!r}")
        self._base_pricer = base_pricer
        self.policy = policy
        self.base_calibration_id = base_pricer.base_calibration_id
        self.calibration_id = "request-latency-" + canonical_sha256(
            {
                "base_stream_calibration_id": base_pricer.calibration_id,
                "residency_pricing_schema": (
                    "plena-request-residency-algebra-v1"
                ),
                "sram_policy": policy,
            }
        )

    def price_trace(
        self,
        trace: ExecutionTrace,
        request_memory: RequestMemorySidecar,
        hbm: HBMOperatingPoint,
    ) -> dict[str, float]:
        kv_fraction = self._KV_FRACTIONS.get(self.policy, 0.0)

        def entry_weight(entry: ExecutionTraceEntry) -> float:
            if (
                self.policy == "projection_resident"
                and entry.stage == "Q/K/V + W_O projection"
                and entry.tensor.startswith("W_")
            ):
                return 0.0
            if (
                kv_fraction
                and entry.tensor.startswith(("K_cache_", "V_cache_"))
            ):
                return 1.0 - kv_fraction
            return 1.0

        return self._base_pricer.price_trace_with_entry_weights(
            trace,
            request_memory,
            hbm,
            entry_weight=entry_weight,
        )


@dataclass(frozen=True)
class TraceStageTiming:
    """Compute, memory, and overlapped timing for one sequential stage."""

    stage: str
    compute_cycles: int
    matrix_memory_cycles: int
    vector_memory_cycles: int
    cycles: int
    dynamic_instructions: int
    descriptor_requests: int

    def __post_init__(self) -> None:
        if not self.stage:
            raise ValueError("trace stage timing requires a stage name")
        values = (
            self.compute_cycles,
            self.matrix_memory_cycles,
            self.vector_memory_cycles,
            self.cycles,
            self.dynamic_instructions,
            self.descriptor_requests,
        )
        if any(value < 0 for value in values):
            raise ValueError("trace stage timing values must be non-negative")
        if self.cycles != max(
            self.compute_cycles,
            self.matrix_memory_cycles,
        ) + self.vector_memory_cycles:
            raise ValueError(
                "trace stage timing must overlap compute with matrix DMA and "
                "expose vector DMA"
            )

    @property
    def memory_cycles(self) -> int:
        return self.matrix_memory_cycles + self.vector_memory_cycles

    def to_dict(self) -> dict[str, int | str]:
        return {
            "stage": self.stage,
            "compute_cycles": self.compute_cycles,
            "matrix_memory_cycles": self.matrix_memory_cycles,
            "vector_memory_cycles": self.vector_memory_cycles,
            "memory_cycles": self.memory_cycles,
            "cycles": self.cycles,
            "dynamic_instructions": self.dynamic_instructions,
            "descriptor_requests": self.descriptor_requests,
        }


@dataclass(frozen=True)
class CompilerTraceTimingResult:
    """Production timing result with exact trace and calibration provenance."""

    request_id: str
    context_tokens: int
    batch: int
    geometry: ArrayGeometry
    hbm: HBMOperatingPoint
    frequency_hz: float
    stages: tuple[TraceStageTiming, ...]
    trace_assembly_sha256: str
    compiler_inputs_sha256: str
    compiler_source_sha256: str
    latency_library_sha256: str
    request_memory_sidecar_sha256: str
    memory_calibration_id: str
    base_memory_calibration_id: str
    artifact_scope: str | None
    compiler_lowering_sha256: str | None = None
    artifact_record_sha256: str | None = None
    reason: str = "compiler_trace_timing_validated"
    execution_mode: str = COMPILER_TRACE
    step_composition: str = TRACE_STEP_COMPOSITION
    schema_version: str = COMPILER_TRACE_TIMING_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != COMPILER_TRACE_TIMING_SCHEMA:
            raise ValueError("unsupported compiler-trace timing schema")
        if self.execution_mode != COMPILER_TRACE:
            raise ValueError("compiler-trace result has the wrong execution mode")
        if self.step_composition != TRACE_STEP_COMPOSITION:
            raise ValueError("compiler-trace timing has the wrong composition")
        for value, label in (
            (self.request_id, "timing-request identity"),
            (self.trace_assembly_sha256, "trace assembly identity"),
            (self.compiler_inputs_sha256, "compiler-input identity"),
            (self.compiler_source_sha256, "compiler-source identity"),
            (self.latency_library_sha256, "latency-library identity"),
            (self.request_memory_sidecar_sha256, "request-memory sidecar identity"),
        ):
            _require_sha256(value, label)
        if not self.memory_calibration_id:
            raise ValueError("request-memory calibration identity is required")
        _require_request_calibration_id(self.memory_calibration_id)
        _require_request_calibration_id(self.base_memory_calibration_id)
        if (self.compiler_lowering_sha256 is None) != (
            self.artifact_record_sha256 is None
        ):
            raise ValueError(
                "compiler timing requires both lowering and artifact identities"
            )
        if self.compiler_lowering_sha256 is not None:
            _require_sha256(
                self.compiler_lowering_sha256,
                "compiler-lowering identity",
            )
            _require_sha256(
                self.artifact_record_sha256,
                "compiler artifact-record identity",
            )
        if self.artifact_scope == FULL_MODEL_DECODE_SCOPE and (
            self.compiler_lowering_sha256 is None
        ):
            raise ValueError(
                "full-model timing lacks lowering and artifact identities"
            )
        if self.artifact_scope is not None and (
            not isinstance(self.artifact_scope, str)
            or not self.artifact_scope.strip()
        ):
            raise ValueError("compiler trace artifact scope must be a non-empty string")

    @property
    def total_cycles(self) -> int:
        return sum(stage.cycles for stage in self.stages)

    @property
    def compute_cycles(self) -> int:
        return sum(stage.compute_cycles for stage in self.stages)

    @property
    def memory_cycles(self) -> int:
        return sum(stage.memory_cycles for stage in self.stages)

    @property
    def total_seconds(self) -> float:
        return self.total_cycles / self.frequency_hz

    @property
    def provenance(self) -> dict[str, object]:
        value = {
            "schema_version": self.schema_version,
            "execution_mode": self.execution_mode,
            "reason": self.reason,
            "request_id": self.request_id,
            "trace_assembly_sha256": self.trace_assembly_sha256,
            "compiler_inputs_sha256": self.compiler_inputs_sha256,
            "compiler_source_sha256": self.compiler_source_sha256,
            "latency_library_sha256": self.latency_library_sha256,
            "request_memory_sidecar_sha256": self.request_memory_sidecar_sha256,
            "memory_calibration_id": self.memory_calibration_id,
            "base_memory_calibration_id": self.base_memory_calibration_id,
            "request_stream_composition": REQUEST_STREAM_COMPOSITION_SCHEMA,
            "artifact_scope": self.artifact_scope,
            "geometry": self.geometry.to_dict(),
            "hbm": self.hbm.to_dict(),
            "frequency_hz": self.frequency_hz,
            "step_composition": self.step_composition,
        }
        if self.compiler_lowering_sha256 is not None:
            value["compiler_lowering_sha256"] = self.compiler_lowering_sha256
            value["artifact_record_sha256"] = self.artifact_record_sha256
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            **self.provenance,
            "context_tokens": self.context_tokens,
            "batch": self.batch,
            "total_cycles": self.total_cycles,
            "compute_cycles": self.compute_cycles,
            "memory_cycles": self.memory_cycles,
            "total_seconds": self.total_seconds,
            "stages": [stage.to_dict() for stage in self.stages],
        }


class CompilerTraceTimingProvider:
    """Compile and price exact context points with immutable in-memory reuse."""

    def __init__(
        self,
        artifact_builder: Callable[[CompilerTraceTimingRequest], BoundCompilerTrace],
        instruction_latencies: Mapping[str, int],
        *,
        latency_library_sha256: str,
        stage_memory_pricer: StageMemoryPricer | None,
    ) -> None:
        if not callable(artifact_builder):
            raise TypeError("compiler trace artifact builder must be callable")
        latencies = {
            str(opcode): int(cycles)
            for opcode, cycles in instruction_latencies.items()
        }
        if not latencies or any(cycles <= 0 for cycles in latencies.values()):
            raise ValueError("instruction latencies must be a non-empty positive map")
        _require_sha256(latency_library_sha256, "latency-library identity")
        if stage_memory_pricer is not None:
            calibration_id = getattr(stage_memory_pricer, "calibration_id", None)
            if not isinstance(calibration_id, str) or not calibration_id.strip():
                raise ValueError("stage memory pricer requires a calibration identity")
            _require_request_calibration_id(calibration_id)
            base_calibration_id = getattr(
                stage_memory_pricer,
                "base_calibration_id",
                None,
            )
            if not isinstance(base_calibration_id, str):
                raise ValueError(
                    "stage memory pricer requires its base calibration identity"
                )
            _require_request_calibration_id(base_calibration_id)
            if not callable(getattr(stage_memory_pricer, "price_trace", None)):
                raise TypeError("stage memory pricer must provide price_trace")
        self._artifact_builder = artifact_builder
        artifact_scope = getattr(artifact_builder, "scope", None)
        if artifact_scope is not None and (
            not isinstance(artifact_scope, str) or not artifact_scope.strip()
        ):
            raise ValueError("compiler trace artifact scope must be a non-empty string")
        self._artifact_scope = artifact_scope
        self._instruction_latencies = latencies
        self._latency_library_sha256 = latency_library_sha256
        self._stage_memory_pricer = stage_memory_pricer
        self._cache: dict[CompilerTraceTimingRequest, CompilerTraceTimingResult] = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    @property
    def artifact_scope(self) -> str | None:
        """Semantic scope guaranteed by the artifact builder, if declared."""

        return self._artifact_scope

    def evaluate(
        self,
        request: CompilerTraceTimingRequest,
    ) -> CompilerTraceTimingResult:
        """Resolve one exact point; any evidence mismatch raises immediately."""

        if not isinstance(request, CompilerTraceTimingRequest):
            raise TypeError("compiler trace timing requires a structured request")
        with self._lock:
            cached = self._cache.get(request)
            if cached is not None:
                self._hits += 1
                return cached

            artifact = self._artifact_builder(request)
            if not isinstance(artifact, BoundCompilerTrace):
                raise TypeError("artifact builder must return BoundCompilerTrace")
            if artifact.compiler_source_sha256 != request.compiler_source_sha256:
                raise ValueError("compiled artifact uses a different compiler source")
            if artifact.compiler_lowering_sha256 != request.compiler_lowering_sha256:
                raise ValueError("compiled artifact uses a different lowering key")
            trace = artifact.execution_trace
            if ArrayGeometry.from_trace(trace) != request.geometry:
                raise ValueError("compiler trace geometry differs from the timing request")
            if artifact.request_memory is None:
                raise RuntimeError(
                    "compiler_trace mode requires a compiler-derived request-memory sidecar"
                )
            if self._stage_memory_pricer is None:
                raise RuntimeError(
                    "compiler_trace mode requires calibrated structured request-memory pricing"
                )
            artifact.request_memory.validate(trace, request.hbm)

            missing_opcodes = sorted(
                {
                    entry.opcode
                    for entry in trace.entries
                    if entry.opcode not in self._instruction_latencies
                }
            )
            if missing_opcodes:
                raise ValueError(
                    f"compiler trace contains unpriced opcodes {missing_opcodes}"
                )

            stage_runs = artifact.request_memory.runs_by_stage(trace)
            stage_memory_seconds = self._stage_memory_pricer.price_trace_by_engine(
                trace,
                artifact.request_memory,
                request.hbm,
            )
            stages = []
            for stage in trace.stage_order:
                entries = trace.entries_for_stage(stage)
                runs = stage_runs.get(stage, ())
                engines = stage_memory_seconds.get(stage, {})
                matrix_seconds = float(engines.get(MATRIX_ENGINE, 0.0))
                vector_seconds = float(engines.get(VECTOR_ENGINE, 0.0))
                if any(
                    not math.isfinite(value) or value < 0
                    for value in (matrix_seconds, vector_seconds)
                ):
                    raise ValueError("stage memory pricer returned an invalid latency")
                compute_cycles = sum(
                    entry.dynamic_count * self._instruction_latencies[entry.opcode]
                    for entry in entries
                )
                matrix_cycles = math.ceil(matrix_seconds * request.frequency_hz)
                vector_cycles = math.ceil(vector_seconds * request.frequency_hz)
                stages.append(
                    TraceStageTiming(
                        stage=stage,
                        compute_cycles=compute_cycles,
                        matrix_memory_cycles=matrix_cycles,
                        vector_memory_cycles=vector_cycles,
                        cycles=max(compute_cycles, matrix_cycles) + vector_cycles,
                        dynamic_instructions=sum(
                            entry.dynamic_count for entry in entries
                        ),
                        descriptor_requests=sum(
                            run.repetitions for run in runs
                        ),
                    )
                )

            result = CompilerTraceTimingResult(
                request_id=request.request_id,
                context_tokens=request.context_tokens,
                batch=request.batch,
                geometry=request.geometry,
                hbm=request.hbm,
                frequency_hz=request.frequency_hz,
                stages=tuple(stages),
                trace_assembly_sha256=trace.assembly_sha256,
                compiler_inputs_sha256=request.compiler_inputs_sha256,
                compiler_source_sha256=artifact.compiler_source_sha256,
                latency_library_sha256=self._latency_library_sha256,
                request_memory_sidecar_sha256=(
                    artifact.request_memory.sidecar_sha256
                ),
                memory_calibration_id=self._stage_memory_pricer.calibration_id,
                base_memory_calibration_id=(
                    self._stage_memory_pricer.base_calibration_id
                ),
                artifact_scope=self._artifact_scope,
                compiler_lowering_sha256=artifact.compiler_lowering_sha256,
                artifact_record_sha256=artifact.artifact_record_sha256,
            )
            self._cache[request] = result
            self._misses += 1
            return result

    def prepare(
        self,
        requests: Iterable[CompilerTraceTimingRequest],
    ) -> tuple[CompilerTraceTimingResult, ...]:
        """Resolve a sampled workload once; duplicates become constant-time hits."""

        return tuple(self.evaluate(request) for request in requests)

    def cache_info(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
            }


class FullModelDecodeArtifactBinder:
    """Bind DSE point descriptors to exact artifact-set timing requests."""

    def __init__(self, artifact_set: FullModelDecodeArtifactSet) -> None:
        if not isinstance(artifact_set, FullModelDecodeArtifactSet):
            raise TypeError("full-model artifact set is required")
        self.artifact_set = artifact_set

    def bind(self, point_descriptor: object):
        canonical, identity, descriptor = _point_descriptor(point_descriptor)
        lowering_json, lowering_identity, _ = full_model_decode_lowering_key(
            descriptor
        )
        if self.artifact_set.has_lowering_key(lowering_identity):
            stored = self.artifact_set.lowering_key(lowering_identity)
            if lowering_json != _canonical_json_text(stored):
                raise ValueError("compiler point differs from the stored lowering key")
        else:
            self.artifact_set.family(descriptor)
        hardware = descriptor["hardware"]
        serving = descriptor["serving"]
        compiler = descriptor["compiler"]
        if not all(
            isinstance(item, Mapping)
            for item in (hardware, serving, compiler)
        ):
            raise TypeError("compiler point descriptor sections are missing")
        geometry_value = hardware["array_geometry"]
        hbm_value = hardware["hbm_timing_geometry"]
        if not isinstance(geometry_value, Mapping) or not isinstance(
            hbm_value,
            Mapping,
        ):
            raise TypeError("compiler point timing geometry is missing")
        geometry = ArrayGeometry(
            mlen=int(geometry_value["mlen"]),
            blen=int(geometry_value["blen"]),
            vlen=int(geometry_value["vlen"]),
            hlen=int(geometry_value["hlen"]),
        )
        hbm = HBMOperatingPoint(
            generation=str(hbm_value["generation"]),
            channels=int(hbm_value["channels"]),
            pin_rate_gbps=float(hbm_value["pin_rate_gbps"]),
            channel_width_bits=int(hbm_value["channel_width_bits"]),
            pseudochannels=int(hbm_value["pseudochannels"]),
            bank_groups=int(hbm_value["bank_groups"]),
            banks_per_group=int(hbm_value["banks_per_group"]),
            transaction_bytes=int(hbm_value["transaction_bytes"]),
        )
        batch = int(serving["batch"])
        frequency_hz = float(compiler["frequency_hz"])

        def request_for_context(context_tokens: int) -> CompilerTraceTimingRequest:
            if self.artifact_set.has_lowering_key(lowering_identity):
                known_contexts = self.artifact_set.contexts(lowering_identity)
                if known_contexts and context_tokens not in known_contexts:
                    # A family can fill a missing exact context.  Record-only
                    # legacy sets preserve their fail-closed exact behavior.
                    try:
                        self.artifact_set.family(descriptor)
                    except KeyError:
                        self.artifact_set.record(lowering_identity, context_tokens)
            elif context_tokens <= 1:
                raise ValueError("full-model decode context must exceed one token")
            return CompilerTraceTimingRequest(
                compiler_inputs_sha256=identity,
                compiler_source_sha256=(
                    self.artifact_set.compiler_source_sha256
                ),
                context_tokens=context_tokens,
                batch=batch,
                geometry=geometry,
                hbm=hbm,
                frequency_hz=frequency_hz,
                compiler_lowering_sha256=lowering_identity,
                compiler_point_descriptor_json=canonical,
            )

        return request_for_context


class FullModelDecodeArtifactTimingProvider:
    """Price content-addressed artifact records with point-specific RTL costs."""

    artifact_scope = FULL_MODEL_DECODE_SCOPE

    def __init__(
        self,
        artifact_set: FullModelDecodeArtifactSet,
        *,
        latency_library_path: str | Path,
        request_calibration_path: str | Path,
    ) -> None:
        if not isinstance(artifact_set, FullModelDecodeArtifactSet):
            raise TypeError("full-model artifact set is required")
        latency_path = Path(latency_library_path).resolve()
        calibration_path = Path(request_calibration_path).resolve()
        if not latency_path.is_file():
            raise FileNotFoundError(latency_path)
        if not calibration_path.is_file():
            raise FileNotFoundError(calibration_path)
        from analytic_models.disagg_serve.memory import (
            RequestLatencyModel,
            load_request_observations,
        )

        self.artifact_set = artifact_set
        self._latency_library_path = latency_path
        self._latency_library_sha256 = _sha256_file(latency_path)
        self._memory_pricer = RequestModelStageMemoryPricer(
            RequestLatencyModel.fit(
                load_request_observations(calibration_path)
            )
        )
        self._providers: dict[str, CompilerTraceTimingProvider] = {}
        self._lock = threading.RLock()

    def _provider(
        self,
        request: CompilerTraceTimingRequest,
    ) -> CompilerTraceTimingProvider:
        if (
            request.compiler_lowering_sha256 is None
            or request.compiler_point_descriptor_json is None
        ):
            raise ValueError("full-model timing request lacks lowering provenance")
        descriptor = json.loads(request.compiler_point_descriptor_json)
        _, lowering_identity, _ = full_model_decode_lowering_key(descriptor)
        if lowering_identity != request.compiler_lowering_sha256:
            raise ValueError("timing request lowering key is inconsistent")
        if self.artifact_set.has_lowering_key(lowering_identity):
            stored = self.artifact_set.lowering_key(lowering_identity)
            if _canonical_json_text(stored) != full_model_decode_lowering_key(
                descriptor
            )[0]:
                raise ValueError("timing request lowering key is absent")
        else:
            self.artifact_set.family(descriptor)
        compiler = descriptor["compiler"]
        hardware = descriptor["hardware"]
        if not isinstance(compiler, Mapping) or not isinstance(
            hardware,
            Mapping,
        ):
            raise TypeError("compiler point descriptor is malformed")
        configuration = hardware["configuration"]
        topology = hardware.get("topology")
        if not isinstance(configuration, Mapping) or not isinstance(
            topology,
            Mapping,
        ):
            raise TypeError("compiler hardware configuration is missing")
        sram_policy = str(topology["sram_policy"])
        timing_mode = str(compiler["timing_mode"])
        provider_key = canonical_sha256(
            {
                "schema_version": "plena-full-model-pricing-provider-v1",
                "compiler_lowering_sha256": lowering_identity,
                "latency_library_sha256": self._latency_library_sha256,
                "configuration": dict(configuration),
                "settings_sha256": compiler["settings_sha256"],
                "timing_mode": timing_mode,
                "sram_policy": sram_policy,
            }
        )
        with self._lock:
            cached = self._providers.get(provider_key)
            if cached is not None:
                return cached
            if (
                compiler["latency_library_sha256"]
                != self._latency_library_sha256
            ):
                raise ValueError(
                    "compiler artifact uses a different latency library"
                )
            try:
                from .perf_model import HardwareConfig, build_pipelined_latency
            except ImportError:
                from perf_model import HardwareConfig, build_pipelined_latency

            instruction_latencies = build_pipelined_latency(
                HardwareConfig(**dict(configuration)),
                str(self._latency_library_path),
                timing_mode=timing_mode,
            ).latencies
            latency_identity = canonical_sha256(
                {
                    "schema_version": (
                        "plena-evaluated-instruction-latencies-v1"
                    ),
                    "source_sha256": self._latency_library_sha256,
                    "configuration": dict(configuration),
                    "settings_sha256": compiler["settings_sha256"],
                    "timing_mode": timing_mode,
                    "instruction_latencies": instruction_latencies,
                }
            )
            provider = CompilerTraceTimingProvider(
                FullModelDecodeArtifactBuilder(
                    self.artifact_set,
                    lowering_identity,
                    descriptor,
                ),
                instruction_latencies,
                latency_library_sha256=latency_identity,
                stage_memory_pricer=(
                    self._memory_pricer
                    if sram_policy == "streaming"
                    else ResidencyAdjustedStageMemoryPricer(
                        self._memory_pricer,
                        sram_policy,
                    )
                ),
            )
            self._providers[provider_key] = provider
            return provider

    def evaluate(
        self,
        request: CompilerTraceTimingRequest,
    ) -> CompilerTraceTimingResult:
        if request.compiler_source_sha256 != (
            self.artifact_set.compiler_source_sha256
        ):
            raise ValueError("compiler timing request uses a stale source identity")
        return self._provider(request).evaluate(request)

    def prepare(
        self,
        requests: Iterable[CompilerTraceTimingRequest],
    ) -> tuple[CompilerTraceTimingResult, ...]:
        return tuple(self.evaluate(request) for request in requests)

    def cache_info(self) -> dict[str, int]:
        with self._lock:
            values = [provider.cache_info() for provider in self._providers.values()]
        return {
            "hits": sum(value["hits"] for value in values),
            "misses": sum(value["misses"] for value in values),
            "size": sum(value["size"] for value in values),
            "points": len(values),
        }


@dataclass(frozen=True)
class FullModelDecodeArtifactRuntime:
    artifact_set: FullModelDecodeArtifactSet
    provider: FullModelDecodeArtifactTimingProvider
    binder: FullModelDecodeArtifactBinder


def create_full_model_decode_artifact_runtime(
    artifact_set_path: str | Path,
    *,
    latency_library_path: str | Path = DEFAULT_LATENCY_LIBRARY,
    request_calibration_path: str | Path = DEFAULT_REQUEST_CALIBRATION,
) -> FullModelDecodeArtifactRuntime:
    """Load and price a complete content-addressed compiler artifact set."""

    artifact_set = FullModelDecodeArtifactSet.load(artifact_set_path)
    installed_compiler_sha256 = native_decode_compiler_source_sha256()
    if artifact_set.compiler_source_sha256 != installed_compiler_sha256:
        raise ValueError(
            "full-model artifact set was built by a different compiler source"
        )
    return FullModelDecodeArtifactRuntime(
        artifact_set=artifact_set,
        provider=FullModelDecodeArtifactTimingProvider(
            artifact_set,
            latency_library_path=latency_library_path,
            request_calibration_path=request_calibration_path,
        ),
        binder=FullModelDecodeArtifactBinder(artifact_set),
    )


@dataclass(frozen=True)
class ReferenceDecodeTimingRuntime:
    """Ready-to-use provider for the exact decoder stage-validation lowering."""

    artifact_builder: ReferenceDecodeArtifactBuilder
    provider: CompilerTraceTimingProvider
    latency_library_sha256: str
    request_calibration_id: str
    base_request_calibration_id: str

    def request(
        self,
        *,
        context_tokens: int,
        hbm: HBMOperatingPoint,
        frequency_hz: float,
    ) -> CompilerTraceTimingRequest:
        return self.artifact_builder.request(
            context_tokens=context_tokens,
            hbm=hbm,
            frequency_hz=frequency_hz,
        )

    def prewarm(
        self,
        context_tokens: Iterable[int],
        *,
        hbm: HBMOperatingPoint,
        frequency_hz: float,
    ) -> tuple[CompilerTraceTimingResult, ...]:
        requests = (
            self.request(
                context_tokens=int(tokens),
                hbm=hbm,
                frequency_hz=frequency_hz,
            )
            for tokens in context_tokens
        )
        return self.provider.prepare(requests)


def create_reference_decode_timing_runtime(
    lowering: ReferenceDecodeLowering,
    *,
    settings_path: str | Path = DEFAULT_SETTINGS,
    latency_library_path: str | Path = DEFAULT_LATENCY_LIBRARY,
    request_calibration_path: str | Path = DEFAULT_REQUEST_CALIBRATION,
    timing_mode: str = RTL_SERIALIZED,
) -> ReferenceDecodeTimingRuntime:
    """Construct a compiler, opcode, and request-calibration bound runtime."""

    from analytic_models.disagg_serve.memory import (
        RequestLatencyModel,
        load_request_observations,
    )

    try:
        from .perf_model import (
            build_pipelined_latency,
            load_hardware_config_from_toml,
        )
    except ImportError:
        from perf_model import (
            build_pipelined_latency,
            load_hardware_config_from_toml,
        )

    builder = ReferenceDecodeArtifactBuilder(
        lowering,
        settings_path=settings_path,
    )
    latency_path = Path(latency_library_path).resolve()
    calibration_path = Path(request_calibration_path).resolve()
    if not latency_path.is_file():
        raise FileNotFoundError(latency_path)
    if not calibration_path.is_file():
        raise FileNotFoundError(calibration_path)

    hardware = load_hardware_config_from_toml(str(builder.settings_path))
    hardware = hardware.model_copy(
        update={
            "MLEN": builder.geometry.mlen,
            "BLEN": builder.geometry.blen,
            "VLEN": builder.geometry.vlen,
            "HLEN": builder.geometry.hlen,
            "BROADCAST_AMOUNT": (
                builder.geometry.mlen // builder.geometry.hlen
            ),
        }
    )
    instruction_latencies = build_pipelined_latency(
        hardware,
        str(latency_path),
        timing_mode=timing_mode,
    ).latencies
    latency_identity = canonical_sha256(
        {
            "schema_version": "plena-evaluated-instruction-latencies-v1",
            "source_sha256": _sha256_file(latency_path),
            "configuration_sha256": builder.configuration_sha256,
            "timing_mode": timing_mode,
            "instruction_latencies": instruction_latencies,
        }
    )
    request_model = RequestLatencyModel.fit(
        load_request_observations(calibration_path)
    )
    pricer = RequestModelStageMemoryPricer(request_model)
    provider = CompilerTraceTimingProvider(
        builder,
        instruction_latencies,
        latency_library_sha256=latency_identity,
        stage_memory_pricer=pricer,
    )
    return ReferenceDecodeTimingRuntime(
        artifact_builder=builder,
        provider=provider,
        latency_library_sha256=latency_identity,
        request_calibration_id=pricer.calibration_id,
        base_request_calibration_id=pricer.base_calibration_id,
    )


@dataclass(frozen=True)
class ResolvedDecodeStepTiming:
    """Common production result for trace and explicit compatibility modes."""

    execution_mode: str
    reason: str
    total_seconds: float
    compute_seconds: float
    memory_seconds: float
    compiler_trace: CompilerTraceTimingResult | None = None

    def __post_init__(self) -> None:
        if self.execution_mode not in DECODE_EXECUTION_MODES:
            raise ValueError("unsupported decode execution mode")
        values = (
            self.total_seconds,
            self.compute_seconds,
            self.memory_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("resolved decode timing must be finite and non-negative")
        if self.execution_mode == COMPILER_TRACE:
            if self.compiler_trace is None:
                raise ValueError("compiler trace mode requires trace timing evidence")
            if not math.isclose(
                self.total_seconds,
                self.compiler_trace.total_seconds,
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise ValueError("resolved timing differs from compiler trace timing")
        elif self.compiler_trace is not None:
            raise ValueError("legacy timing cannot carry compiler trace evidence")


def resolve_decode_step_timing(
    execution_mode: str,
    *,
    trace_timing_provider: CompilerTraceTimingProvider | None = None,
    trace_request: CompilerTraceTimingRequest | None = None,
    legacy_compute_seconds: float | None = None,
    legacy_memory_seconds: float | None = None,
) -> ResolvedDecodeStepTiming:
    """Resolve one step without any implicit fallback between timing modes."""

    if execution_mode == COMPILER_TRACE:
        if legacy_compute_seconds is not None or legacy_memory_seconds is not None:
            raise ValueError("compiler_trace mode rejects legacy timing inputs")
        if trace_timing_provider is None or trace_request is None:
            raise RuntimeError(
                "compiler_trace mode requires a timing provider and exact request"
            )
        trace_result = trace_timing_provider.evaluate(trace_request)
        return ResolvedDecodeStepTiming(
            execution_mode=COMPILER_TRACE,
            reason=trace_result.reason,
            total_seconds=trace_result.total_seconds,
            compute_seconds=trace_result.compute_cycles / trace_result.frequency_hz,
            memory_seconds=trace_result.memory_cycles / trace_result.frequency_hz,
            compiler_trace=trace_result,
        )

    if execution_mode == LEGACY_AGGREGATE_BANDWIDTH:
        if trace_timing_provider is not None or trace_request is not None:
            raise ValueError(
                "legacy_aggregate_bandwidth mode rejects compiler trace inputs"
            )
        if legacy_compute_seconds is None or legacy_memory_seconds is None:
            raise RuntimeError(
                "legacy_aggregate_bandwidth mode requires compute and memory timing"
            )
        compute_seconds = float(legacy_compute_seconds)
        memory_seconds = float(legacy_memory_seconds)
        return ResolvedDecodeStepTiming(
            execution_mode=LEGACY_AGGREGATE_BANDWIDTH,
            reason="legacy_aggregate_bandwidth_compatibility",
            total_seconds=max(compute_seconds, memory_seconds),
            compute_seconds=compute_seconds,
            memory_seconds=memory_seconds,
        )

    raise ValueError(
        f"unsupported decode execution mode {execution_mode!r}; "
        f"expected one of {DECODE_EXECUTION_MODES}"
    )


__all__ = [
    "ArrayGeometry",
    "BoundCompilerTrace",
    "COMPILER_TRACE",
    "COMPILER_TRACE_TIMING_SCHEMA",
    "CompilerTraceTimingProvider",
    "CompilerTraceTimingRequest",
    "CompilerTraceTimingResult",
    "DEFAULT_LATENCY_LIBRARY",
    "DEFAULT_MAX_PROJECTED_TRACE_BYTES",
    "DEFAULT_MAX_TRACE_GENERATION_CALLS",
    "DEFAULT_REQUEST_CALIBRATION",
    "DEFAULT_SETTINGS",
    "DECODE_EXECUTION_MODES",
    "FULL_MODEL_DECODE_SCOPE",
    "FULL_MODEL_ARTIFACT_ID_PREFIX",
    "FULL_MODEL_ARTIFACT_FAMILY_SCHEMA",
    "FULL_MODEL_ARTIFACT_RECORD_SCHEMA",
    "FULL_MODEL_ARTIFACT_SET_SCHEMA",
    "FULL_MODEL_BUILD_PLAN_SCHEMA",
    "FULL_MODEL_BATCH_RESOLUTION_MODE",
    "FULL_MODEL_BATCH_RESOLUTION_SCHEMA",
    "FULL_MODEL_CACHE_SEMANTICS",
    "FULL_MODEL_CONTEXT_RESOLUTION_MODE",
    "FULL_MODEL_CONTEXT_RESOLUTION_SCHEMA",
    "FULL_MODEL_FAMILY_KEY_SCHEMA",
    "FULL_MODEL_LAZY_INSTANTIATION_SCHEMA",
    "FULL_MODEL_LOWERING_KEY_SCHEMA",
    "FULL_MODEL_NATIVE_TEMPLATE_KEY_SCHEMA",
    "FULL_MODEL_STORAGE_RESOLUTION_MODE",
    "FullModelDecodeArtifactBinder",
    "FullModelDecodeArtifactBuildPlan",
    "FullModelDecodeArtifactBuilder",
    "FullModelDecodeArtifactFamily",
    "FullModelDecodeLazyArtifactGenerator",
    "FullModelDecodeArtifactRecord",
    "FullModelDecodeArtifactRuntime",
    "FullModelDecodeArtifactSet",
    "FullModelDecodeArtifactSetBuilder",
    "FullModelDecodeArtifactTimingProvider",
    "FullModelDecodeBatchResolution",
    "FullModelDecodeContextResolution",
    "HBMOperatingPoint",
    "LEGACY_AGGREGATE_BANDWIDTH",
    "REQUEST_MEMORY_SIDECAR_SCHEMA",
    "REFERENCE_DECODE_SCOPE",
    "ReferenceDecodeArtifactBuilder",
    "ReferenceDecodeLowering",
    "ReferenceDecodeTimingRuntime",
    "RequestDescriptorRun",
    "RequestMemorySidecar",
    "RequestModelStageMemoryPricer",
    "ResidencyAdjustedStageMemoryPricer",
    "ResolvedDecodeStepTiming",
    "StageMemoryPricer",
    "TraceRequestBinding",
    "TRACE_STEP_COMPOSITION",
    "TraceStageTiming",
    "canonical_sha256",
    "create_reference_decode_timing_runtime",
    "create_full_model_decode_artifact_runtime",
    "full_model_decode_batch_resolution",
    "full_model_decode_context_resolution",
    "full_model_decode_lowering_key",
    "full_model_decode_native_template_key",
    "full_model_decode_family_key",
    "full_model_decode_generator_blockers",
    "native_decode_compiler_source_sha256",
    "reference_decode_compiler_source_sha256",
    "request_memory_sidecar_from_compiler",
    "resolve_decode_step_timing",
    "trace_entry_fingerprint",
]
