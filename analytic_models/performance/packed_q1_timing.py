"""Content-addressed matrix timing evidence for PackedKV cached decode."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from .decode_timing import (
        IDEAL_MATRIX_PIPELINE,
        RTL_SERIALIZED,
        TIMING_MODES,
        matrix_issue_cycles,
    )
except ImportError:
    from decode_timing import (
        IDEAL_MATRIX_PIPELINE,
        RTL_SERIALIZED,
        TIMING_MODES,
        matrix_issue_cycles,
    )

PACKED_Q1_TIMING_SCHEMA = "plena-packed-q1-timing-contract-v2"
PACKED_Q1_COUNT_RULE = "compiler-packedkv-q1-dynamic-execution-trace-counts"
PACKED_Q1_MATRIX_OPCODES = (
    "M_BTMM",
    "M_BMM_WO",
    "M_MM",
    "M_MM_WO",
)
PACKED_Q1_REDUCTION_CONTRACT = {
    "qk": "drain_each_blen_score_tile",
    "pv": "drain_each_mlen_context_partial",
    "cross_context_accumulation": "online_softmax_scale_then_v_add_vv",
    "matrix_writeout_serialized_cycles": {
        "M_MM_WO": "BLEN + 6",
        "M_BMM_WO": "(MLEN / HLEN) * BLEN + 6",
    },
    "matrix_writeout_ideal_issue_cycles": 1,
}
COMPILER_TIMING_SOURCE_PATHS = (
    "aten/execution_trace.py",
    "aten/plena/compiler.py",
    "aten/plena/program_attention.py",
    "aten/plena/isa_attention.py",
    "aten/plena/program_tensors.py",
    "assembler/assembly_to_binary.py",
    "assembler/parser.py",
    "doc/operation.svh",
)
RTL_TIMING_SOURCE_PATHS = (
    "src/basic_components/systolic_gemm_mxint/rtl/mxint_systolic_mcu.sv",
    "src/definitions/operation.svh",
    "src/frontend/rtl/decoder.sv",
    "src/control/rtl/pipeline_control.sv",
    "src/control/rtl/data_flow_control.sv",
    "src/matrix_machine/rtl/matrix_machine.sv",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _require_digest(value: str, label: str) -> str:
    value = str(value)
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _source_hashes(
    root: Path,
    relative_paths: Sequence[str],
) -> tuple[tuple[str, str], ...]:
    missing = [relative for relative in relative_paths if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"timing-contract sources are missing: {missing}")
    return tuple(
        (
            relative,
            hashlib.sha256((root / relative).read_bytes()).hexdigest(),
        )
        for relative in sorted(relative_paths)
    )


def packed_q1_matrix_histogram(
    *,
    cache_tokens: int,
    batch: int,
    mlen: int,
    blen: int,
    hlen: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    batch_packed: bool = False,
) -> tuple[tuple[str, int], ...]:
    """Return matrix-op counts for the dense-selector q1 attention lowering.

    The lowering emits one attention program per batch element, so a decode step
    fills a single query row of each BLEN-row matrix tile. ``batch_packed``
    instead gathers the batch into the query dimension, which is what the tile
    geometry allows whenever ``batch <= BLEN``; it reduces the op count by
    ``batch / ceil(batch / BLEN)`` and is the headroom that packing would
    recover.
    """

    dimensions = {
        "cache_tokens": cache_tokens,
        "batch": batch,
        "mlen": mlen,
        "blen": blen,
        "hlen": hlen,
        "query_heads": query_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in dimensions.values()
    ):
        raise ValueError("PackedKV q1 timing dimensions must be positive integers")
    if mlen % blen or mlen % hlen:
        raise ValueError("MLEN must be divisible by BLEN and HLEN")
    if head_dim != hlen:
        raise ValueError("PackedKV q1 timing requires head_dim == HLEN")
    if query_heads % kv_heads:
        raise ValueError(
            "query_heads must be divisible by kv_heads"
        )
    # Logical grouping fixes correctness; physical chunking fixes how it runs.
    # The array broadcasts one K/V head across at most MLEN/HLEN query heads, so
    # a GQA ratio wider than that is issued as several chunks over the same K/V
    # rather than being rejected. Chunking keeps K/V, mask and RoPE state
    # resident, so only the query-side work repeats.
    logical_broadcast = query_heads // kv_heads
    physical_broadcast = min(logical_broadcast, mlen // hlen)
    chunks_per_kv_head = math.ceil(logical_broadcast / physical_broadcast)

    query_tiles = math.ceil(batch / blen) if batch_packed else batch
    qk_ops = (
        query_tiles
        * kv_heads
        * chunks_per_kv_head
        * math.ceil(cache_tokens / blen)
    )
    pv_ops = (
        query_tiles
        * query_heads
        * math.ceil(cache_tokens / mlen)
        * math.ceil(head_dim / blen)
    )
    return (
        ("M_BTMM", qk_ops),
        ("M_BMM_WO", qk_ops),
        ("M_MM", pv_ops),
        ("M_MM_WO", pv_ops),
    )


def packed_q1_v_add_count(
    *,
    cache_tokens: int,
    batch: int,
    mlen: int,
    query_heads: int,
) -> int:
    """Return the exact q1 V_ADD_VV count for partials, packing, and tail masks."""

    dimensions = (cache_tokens, batch, mlen, query_heads)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in dimensions
    ):
        raise ValueError("PackedKV q1 vector-count dimensions must be positive")
    context_tiles = math.ceil(cache_tokens / mlen)
    tail_mask = int(cache_tokens % mlen != 0)
    return batch * query_heads * (context_tiles + 1 + tail_mask)


def _matrix_opcode_histogram(
    opcode_histogram: Mapping[str, int],
) -> tuple[tuple[str, int], ...]:
    unsupported = sorted(
        opcode
        for opcode, count in opcode_histogram.items()
        if opcode.startswith("M_")
        and int(count) > 0
        and opcode not in PACKED_Q1_MATRIX_OPCODES
    )
    if unsupported:
        raise ValueError(
            f"PackedKV q1 trace contains non-deployable matrix opcodes {unsupported}"
        )
    return tuple(
        (opcode, int(opcode_histogram.get(opcode, 0)))
        for opcode in PACKED_Q1_MATRIX_OPCODES
    )


@dataclass(frozen=True)
class PackedQ1TracePoint:
    """One compiler trace bound to a cache length and matrix histogram."""

    cache_tokens: int
    opcode_histogram: tuple[tuple[str, int], ...]
    assembly_sha256: str
    machine_code_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.cache_tokens, bool)
            or not isinstance(self.cache_tokens, int)
            or self.cache_tokens <= 0
        ):
            raise ValueError("cache_tokens must be a positive integer")
        normalized = tuple(
            sorted((str(opcode), int(count)) for opcode, count in self.opcode_histogram)
        )
        if len({opcode for opcode, _ in normalized}) != len(normalized):
            raise ValueError("opcode histogram contains duplicate opcodes")
        if any(count < 0 for _, count in normalized):
            raise ValueError("opcode counts must be non-negative")
        _matrix_opcode_histogram(dict(normalized))
        object.__setattr__(self, "opcode_histogram", normalized)
        _require_digest(self.assembly_sha256, "assembly hash")
        _require_digest(self.machine_code_sha256, "machine-code hash")

    @property
    def matrix_histogram(self) -> tuple[tuple[str, int], ...]:
        return _matrix_opcode_histogram(dict(self.opcode_histogram))

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PackedQ1TracePoint":
        if set(value) != {
            "cache_tokens",
            "opcode_histogram",
            "matrix_opcode_histogram",
            "assembly_sha256",
            "machine_code_sha256",
        }:
            raise ValueError("PackedKV trace-point fields differ from the schema")
        histogram = value.get("opcode_histogram")
        if not isinstance(histogram, Mapping):
            raise ValueError("opcode_histogram must be an object")
        point = cls(
            cache_tokens=int(value["cache_tokens"]),
            opcode_histogram=tuple(
                (str(opcode), int(count))
                for opcode, count in histogram.items()
            ),
            assembly_sha256=str(value["assembly_sha256"]),
            machine_code_sha256=str(value["machine_code_sha256"]),
        )
        if value["matrix_opcode_histogram"] != dict(point.matrix_histogram):
            raise ValueError("PackedKV trace-point matrix histogram is inconsistent")
        return point

    def to_dict(self) -> dict[str, object]:
        return {
            "cache_tokens": self.cache_tokens,
            "opcode_histogram": dict(self.opcode_histogram),
            "matrix_opcode_histogram": dict(self.matrix_histogram),
            "assembly_sha256": self.assembly_sha256,
            "machine_code_sha256": self.machine_code_sha256,
        }


@dataclass(frozen=True)
class PackedQ1TimingContract:
    """Exact compiler-count evidence for one geometry, batch, and cache schedule."""

    timing_mode: str
    mlen: int
    blen: int
    hlen: int
    query_heads: int
    kv_heads: int
    head_dim: int
    batch: int
    points: tuple[PackedQ1TracePoint, ...]
    compiler_source_hashes: tuple[tuple[str, str], ...]
    rtl_source_hashes: tuple[tuple[str, str], ...]
    latency_library_sha256: str
    q_len: int = 1
    count_rule: str = PACKED_Q1_COUNT_RULE

    def __post_init__(self) -> None:
        if self.timing_mode not in TIMING_MODES:
            raise ValueError(f"unknown timing mode {self.timing_mode!r}")
        if self.q_len != 1:
            raise ValueError("PackedKV timing evidence is restricted to q_len=1")
        if self.count_rule != PACKED_Q1_COUNT_RULE:
            raise ValueError("unsupported PackedKV matrix-count rule")
        if not self.points:
            raise ValueError("PackedKV timing evidence requires trace points")
        ordered = tuple(sorted(self.points, key=lambda point: point.cache_tokens))
        if len({point.cache_tokens for point in ordered}) != len(ordered):
            raise ValueError("PackedKV timing evidence repeats a cache length")
        object.__setattr__(self, "points", ordered)
        for point in ordered:
            expected = packed_q1_matrix_histogram(
                cache_tokens=point.cache_tokens,
                batch=self.batch,
                mlen=self.mlen,
                blen=self.blen,
                hlen=self.hlen,
                query_heads=self.query_heads,
                kv_heads=self.kv_heads,
                head_dim=self.head_dim,
            )
            if point.matrix_histogram != expected:
                raise ValueError(
                    f"compiler matrix histogram differs at cache={point.cache_tokens}"
                )
            observed_v_add = dict(point.opcode_histogram).get("V_ADD_VV", 0)
            expected_v_add = packed_q1_v_add_count(
                cache_tokens=point.cache_tokens,
                batch=self.batch,
                mlen=self.mlen,
                query_heads=self.query_heads,
            )
            if observed_v_add != expected_v_add:
                raise ValueError(
                    "compiler vector accumulation/packing histogram differs at "
                    f"cache={point.cache_tokens}"
                )
        self._validate_source_hashes(
            self.compiler_source_hashes,
            COMPILER_TIMING_SOURCE_PATHS,
            "compiler",
        )
        self._validate_source_hashes(
            self.rtl_source_hashes,
            RTL_TIMING_SOURCE_PATHS,
            "RTL",
        )
        _require_digest(self.latency_library_sha256, "latency-library hash")

    @staticmethod
    def _validate_source_hashes(
        values: tuple[tuple[str, str], ...],
        required: Sequence[str],
        label: str,
    ) -> None:
        normalized = tuple(sorted((str(name), str(digest)) for name, digest in values))
        if tuple(name for name, _ in normalized) != tuple(sorted(required)):
            raise ValueError(f"{label} source hash set differs from the timing contract")
        for name, digest in normalized:
            _require_digest(digest, f"{label} source {name}")

    @property
    def cache_tokens(self) -> tuple[int, ...]:
        return tuple(point.cache_tokens for point in self.points)

    def point(self, cache_tokens: int) -> PackedQ1TracePoint:
        for point in self.points:
            if point.cache_tokens == cache_tokens:
                return point
        raise KeyError(f"cache length {cache_tokens} is absent from the timing contract")

    def matrix_cycles(self, cache_tokens: int, mode: str) -> int:
        if mode not in (self.timing_mode, IDEAL_MATRIX_PIPELINE):
            raise ValueError("matrix-cycle request differs from the timing contract")
        return sum(
            count
            * matrix_issue_cycles(
                opcode,
                self.blen,
                mode,
                mlen=self.mlen,
                hlen=self.hlen,
            )
            for opcode, count in self.point(cache_tokens).matrix_histogram
        )

    def _content_dict(self) -> dict[str, object]:
        return {
            "schema_version": PACKED_Q1_TIMING_SCHEMA,
            "count_rule": self.count_rule,
            "reduction_contract": PACKED_Q1_REDUCTION_CONTRACT,
            "timing_mode": self.timing_mode,
            "q_len": self.q_len,
            "geometry": {
                "mlen": self.mlen,
                "blen": self.blen,
                "hlen": self.hlen,
                "query_heads": self.query_heads,
                "kv_heads": self.kv_heads,
                "head_dim": self.head_dim,
            },
            "batch": self.batch,
            "cache_tokens": list(self.cache_tokens),
            "compiler_source_hashes": dict(self.compiler_source_hashes),
            "rtl_source_hashes": dict(self.rtl_source_hashes),
            "latency_library_sha256": self.latency_library_sha256,
            "points": [point.to_dict() for point in self.points],
        }

    @property
    def contract_id(self) -> str:
        return "packed-q1-timing-" + hashlib.sha256(
            _canonical_bytes(self._content_dict())
        ).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return self._content_dict() | {"contract_id": self.contract_id}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "PackedQ1TimingContract":
        if set(value) != {
            "schema_version",
            "count_rule",
            "reduction_contract",
            "timing_mode",
            "q_len",
            "geometry",
            "batch",
            "cache_tokens",
            "compiler_source_hashes",
            "rtl_source_hashes",
            "latency_library_sha256",
            "points",
            "contract_id",
        }:
            raise ValueError("PackedKV timing-contract fields differ from the schema")
        if value.get("schema_version") != PACKED_Q1_TIMING_SCHEMA:
            raise ValueError("unsupported PackedKV timing-contract schema")
        if value.get("reduction_contract") != PACKED_Q1_REDUCTION_CONTRACT:
            raise ValueError("PackedKV reduction contract differs from the schema")
        geometry = value.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("PackedKV timing geometry must be an object")
        if set(geometry) != {
            "mlen",
            "blen",
            "hlen",
            "query_heads",
            "kv_heads",
            "head_dim",
        }:
            raise ValueError("PackedKV timing geometry fields differ from the schema")
        compiler_hashes = value.get("compiler_source_hashes")
        rtl_hashes = value.get("rtl_source_hashes")
        if not isinstance(compiler_hashes, Mapping) or not isinstance(
            rtl_hashes, Mapping
        ):
            raise ValueError("PackedKV source hashes must be objects")
        raw_points = value.get("points")
        if not isinstance(raw_points, list):
            raise ValueError("PackedKV timing points must be a list")
        contract = cls(
            timing_mode=str(value["timing_mode"]),
            mlen=int(geometry["mlen"]),
            blen=int(geometry["blen"]),
            hlen=int(geometry["hlen"]),
            query_heads=int(geometry["query_heads"]),
            kv_heads=int(geometry["kv_heads"]),
            head_dim=int(geometry["head_dim"]),
            batch=int(value["batch"]),
            points=tuple(PackedQ1TracePoint.from_dict(point) for point in raw_points),
            compiler_source_hashes=tuple(
                sorted((str(name), str(digest)) for name, digest in compiler_hashes.items())
            ),
            rtl_source_hashes=tuple(
                sorted((str(name), str(digest)) for name, digest in rtl_hashes.items())
            ),
            latency_library_sha256=str(value["latency_library_sha256"]),
            q_len=int(value["q_len"]),
            count_rule=str(value["count_rule"]),
        )
        if list(contract.cache_tokens) != list(value.get("cache_tokens", ())):
            raise ValueError("PackedKV cache schedule differs from its trace points")
        if value.get("contract_id") != contract.contract_id:
            raise ValueError("PackedKV timing-contract identity mismatch")
        return contract

    @classmethod
    def load(cls, path: str | Path) -> "PackedQ1TimingContract":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def validate_packed_q1_timing_contract(
    contract: PackedQ1TimingContract | None,
    *,
    timing_mode: str,
    mlen: int,
    blen: int,
    hlen: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    batch: int,
    cache_tokens: Sequence[int],
    compiler_root: Path,
    rtl_root: Path,
    latency_library_path: Path,
) -> tuple[bool, str]:
    """Validate schedule, geometry, mode, and active source hashes."""

    if contract is None:
        return False, "missing_packed_q1_timing_contract"
    expected = (
        timing_mode,
        mlen,
        blen,
        hlen,
        query_heads,
        kv_heads,
        head_dim,
        batch,
    )
    observed = (
        contract.timing_mode,
        contract.mlen,
        contract.blen,
        contract.hlen,
        contract.query_heads,
        contract.kv_heads,
        contract.head_dim,
        contract.batch,
    )
    if observed != expected:
        return False, "packed_q1_geometry_or_mode_mismatch"
    if tuple(sorted(set(int(value) for value in cache_tokens))) != contract.cache_tokens:
        return False, "packed_q1_cache_schedule_mismatch"
    if contract.compiler_source_hashes != _source_hashes(
        compiler_root,
        COMPILER_TIMING_SOURCE_PATHS,
    ):
        return False, "packed_q1_compiler_source_mismatch"
    if contract.rtl_source_hashes != _source_hashes(
        rtl_root,
        RTL_TIMING_SOURCE_PATHS,
    ):
        return False, "packed_q1_rtl_source_mismatch"
    if (
        contract.latency_library_sha256
        != hashlib.sha256(latency_library_path.read_bytes()).hexdigest()
    ):
        return False, "packed_q1_latency_library_mismatch"
    return True, "packed_q1_timing_validated"


__all__ = [
    "COMPILER_TIMING_SOURCE_PATHS",
    "PACKED_Q1_COUNT_RULE",
    "PACKED_Q1_MATRIX_OPCODES",
    "PACKED_Q1_REDUCTION_CONTRACT",
    "PACKED_Q1_TIMING_SCHEMA",
    "RTL_TIMING_SOURCE_PATHS",
    "PackedQ1TimingContract",
    "PackedQ1TracePoint",
    "packed_q1_matrix_histogram",
    "packed_q1_v_add_count",
    "validate_packed_q1_timing_contract",
]
