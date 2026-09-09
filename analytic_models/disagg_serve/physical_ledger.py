"""Aligned HBM and SRAM accounting for one decode-chip configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from .packed_kv import traffic_from_precision
    from ..memory.memory_model import conservative_unique_experts
except ImportError:
    from packed_kv import traffic_from_precision
    from memory_model import conservative_unique_experts

MIN_MATRIX_TILE_CAPACITY = 4
LOCAL_HEAD_TOP_K = 20
LOCAL_HEAD_CANDIDATE_BYTES = 8
LOCAL_HEAD_SAMPLER_STATE_BYTES = 16
LOCAL_HEAD_ARGMAX_STATE_BYTES = 8
LOCAL_HEAD_LOGIT_BYTES = 2


def _align(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0:
        raise ValueError("byte count must be non-negative and alignment positive")
    return math.ceil(value / alignment) * alignment if value else 0


def _scale_bits(effective_bits: float, element_bits: int, block_size: int) -> int:
    value = (float(effective_bits) - int(element_bits)) * int(block_size)
    rounded = round(value)
    if value < -1e-9 or not math.isclose(value, rounded, abs_tol=1e-6):
        raise ValueError("effective precision must encode an integral shared scale")
    return int(rounded)


@dataclass(frozen=True)
class PlaneBytes:
    """Raw and independently aligned element and scale planes."""

    element_raw: int = 0
    element_aligned: int = 0
    scale_raw: int = 0
    scale_aligned: int = 0

    @property
    def total_aligned(self) -> int:
        return self.element_aligned + self.scale_aligned

    def __add__(self, other: "PlaneBytes") -> "PlaneBytes":
        return PlaneBytes(
            self.element_raw + other.element_raw,
            self.element_aligned + other.element_aligned,
            self.scale_raw + other.scale_raw,
            self.scale_aligned + other.scale_aligned,
        )

def matrix_planes(
    rows: int,
    columns: int,
    instances: int,
    *,
    element_bits: int,
    effective_bits: float,
    block_size: int,
    alignment_bytes: int = 64,
) -> PlaneBytes:
    """Storage for row-wise 1xB blocked matrices, aligned per tensor plane."""

    for name, value in (
        ("rows", rows),
        ("columns", columns),
        ("instances", instances),
        ("element_bits", element_bits),
        ("block_size", block_size),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive")
    scale_bits = _scale_bits(effective_bits, element_bits, block_size)
    element_raw = math.ceil(rows * columns * element_bits / 8)
    blocks = rows * math.ceil(columns / block_size)
    scale_raw = math.ceil(blocks * scale_bits / 8)
    return PlaneBytes(
        element_raw=element_raw * instances,
        element_aligned=_align(element_raw, alignment_bytes) * instances,
        scale_raw=scale_raw * instances,
        scale_aligned=_align(scale_raw, alignment_bytes) * instances,
    )


def bf16_matrix_planes(
    rows: int,
    columns: int,
    instances: int = 1,
    *,
    alignment_bytes: int = 64,
) -> PlaneBytes:
    return matrix_planes(
        rows,
        columns,
        instances,
        element_bits=16,
        effective_bits=16,
        block_size=1,
        alignment_bytes=alignment_bytes,
    )


@dataclass(frozen=True)
class WeightLedger:
    """Resident and per-batch-step streamed weight planes."""

    attention: PlaneBytes
    ffn_resident: PlaneBytes
    ffn_streamed: PlaneBytes
    lm_head_resident: PlaneBytes
    lm_head_streamed: PlaneBytes
    bf16_embedding: PlaneBytes
    bf16_norms: PlaneBytes
    bf16_lm_head_resident: PlaneBytes
    bf16_lm_head_streamed: PlaneBytes
    bf16_router_resident: PlaneBytes = PlaneBytes()
    bf16_router_streamed: PlaneBytes = PlaneBytes()

    @property
    def bf16_resident(self) -> PlaneBytes:
        return (
            self.bf16_embedding
            + self.bf16_norms
            + self.bf16_lm_head_resident
            + self.bf16_router_resident
        )

    @property
    def bf16_streamed(self) -> PlaneBytes:
        return (
            self.bf16_norms
            + self.bf16_lm_head_streamed
            + self.bf16_router_streamed
        )

    @property
    def resident(self) -> PlaneBytes:
        return (
            self.attention
            + self.ffn_resident
            + self.lm_head_resident
            + self.bf16_resident
        )

    @property
    def streamed_per_batch_step(self) -> PlaneBytes:
        return (
            self.attention
            + self.ffn_streamed
            + self.lm_head_streamed
            + self.bf16_streamed
        )

def weight_ledger(
    dims: dict,
    precision: dict,
    *,
    alignment_bytes: int = 64,
    include_lm_head: bool = True,
    batch: int = 1,
    decode_tokens_per_sequence: int = 1,
    unique_experts: int | None = None,
    mlen: int | None = None,
) -> WeightLedger:
    """Build the immutable model-weight ledger for dense or MoE decoders."""

    hidden = int(dims["hidden"])
    query = int(dims["heads"]) * int(dims["head_dim"])
    kv = int(dims["kv_heads"]) * int(dims["head_dim"])
    inter = int(dims["inter"])
    vocab = int(dims["vocab"])
    layers = int(dims["layers"])
    block = int(precision.get("block_size", 8))

    def quant(rows: int, cols: int, count: int, role: str) -> PlaneBytes:
        return matrix_planes(
            rows,
            cols,
            count,
            element_bits=int(precision[f"{role}_elem"]),
            effective_bits=float(precision[f"{role}_bits"]),
            block_size=block,
            alignment_bytes=alignment_bytes,
        )

    attention = (
        quant(query, hidden, layers, "attn")
        + quant(kv, hidden, 2 * layers, "attn")
        + quant(hidden, query, layers, "attn")
    )

    experts = int(dims.get("num_experts", 1))
    active_experts = int(dims.get("experts_per_token", 1))
    if experts > 1:
        if unique_experts is None:
            unique_experts = conservative_unique_experts(
                experts,
                active_experts,
                int(batch) * int(decode_tokens_per_sequence),
            )
        unique_experts = int(unique_experts)
        if not active_experts <= unique_experts <= experts:
            raise ValueError(
                "unique_experts must be between experts_per_token and num_experts"
            )
        router = bf16_matrix_planes(
            experts,
            hidden,
            layers,
            alignment_bytes=alignment_bytes,
        )
        resident_experts = (
            quant(inter, hidden, 2 * layers * experts, "ffn")
            + quant(hidden, inter, layers * experts, "ffn")
        )
        streamed_experts = (
            quant(inter, hidden, 2 * layers * unique_experts, "ffn")
            + quant(hidden, inter, layers * unique_experts, "ffn")
        )
        ffn_resident = resident_experts
        ffn_streamed = streamed_experts
    else:
        router = PlaneBytes()
        ffn_resident = (
            quant(inter, hidden, 2 * layers, "ffn")
            + quant(hidden, inter, layers, "ffn")
        )
        ffn_streamed = ffn_resident

    embedding = bf16_matrix_planes(
        vocab,
        hidden,
        alignment_bytes=alignment_bytes,
    )
    norms = bf16_matrix_planes(
        1,
        hidden,
        2 * layers + 1,
        alignment_bytes=alignment_bytes,
    )
    if bool(dims.get("qk_norm", False)):
        norms += bf16_matrix_planes(
            1,
            int(dims["head_dim"]),
            2 * layers,
            alignment_bytes=alignment_bytes,
        )
    tied = bool(dims.get("tie_embeddings", False))
    if include_lm_head:
        if bool(precision.get("lm_head_quantized", False)):
            # The serving head follows the profile's W/A precision.  A tied
            # BF16 embedding cannot be reused as an MX matrix, so this remains
            # a separately packed physical plane even for tied checkpoints.
            if mlen is None or int(mlen) <= 0:
                raise ValueError(
                    "quantized local LM-head storage requires candidate MLEN"
                )
            head_mlen = int(mlen)
            physical_vocab = math.ceil(vocab / head_mlen) * head_mlen
            physical_hidden = math.ceil(hidden / head_mlen) * head_mlen
            lm_head_resident = quant(
                physical_vocab,
                physical_hidden,
                1,
                "head",
            )
            lm_head_streamed = lm_head_resident
            bf16_lm_head_resident = PlaneBytes()
            bf16_lm_head_streamed = PlaneBytes()
        else:
            lm_head_resident = PlaneBytes()
            lm_head_streamed = PlaneBytes()
            bf16_lm_head_resident = (
                PlaneBytes()
                if tied
                else bf16_matrix_planes(
                    vocab,
                    hidden,
                    alignment_bytes=alignment_bytes,
                )
            )
            bf16_lm_head_streamed = bf16_matrix_planes(
                vocab,
                hidden,
                alignment_bytes=alignment_bytes,
            )
    else:
        lm_head_resident = PlaneBytes()
        lm_head_streamed = PlaneBytes()
        bf16_lm_head_resident = PlaneBytes()
        bf16_lm_head_streamed = PlaneBytes()
    return WeightLedger(
        attention=attention,
        ffn_resident=ffn_resident,
        ffn_streamed=ffn_streamed,
        lm_head_resident=lm_head_resident,
        lm_head_streamed=lm_head_streamed,
        bf16_embedding=embedding,
        bf16_norms=norms,
        bf16_lm_head_resident=bf16_lm_head_resident,
        bf16_lm_head_streamed=bf16_lm_head_streamed,
        bf16_router_resident=router,
        bf16_router_streamed=router,
    )


@dataclass(frozen=True)
class KVLedger:
    """Physical KV element and scale planes for the full cache."""

    element_bytes: int
    scale_bytes: int
    per_batch_element_bytes: int
    per_batch_scale_bytes: int
    layout_id: str

    @property
    def total_bytes(self) -> int:
        return self.element_bytes + self.scale_bytes

    @property
    def per_batch_bytes(self) -> int:
        return self.per_batch_element_bytes + self.per_batch_scale_bytes


@dataclass(frozen=True)
class DecodeStepTrafficLedger:
    """HBM bytes for one batch decode step before matrix-load overfetch."""

    weight_element_read_bytes: int
    weight_scale_read_bytes: int
    bf16_weight_read_bytes: int
    activation_read_bytes: int
    activation_write_bytes: int
    kv_element_read_bytes: int
    kv_scale_read_bytes: int
    kv_element_write_bytes: int
    kv_scale_write_bytes: int

    @property
    def read_bytes(self) -> int:
        return (
            self.weight_element_read_bytes
            + self.weight_scale_read_bytes
            + self.bf16_weight_read_bytes
            + self.activation_read_bytes
            + self.kv_element_read_bytes
            + self.kv_scale_read_bytes
        )

    @property
    def write_bytes(self) -> int:
        return (
            self.activation_write_bytes
            + self.kv_element_write_bytes
            + self.kv_scale_write_bytes
        )

    @property
    def total_bytes(self) -> int:
        return self.read_bytes + self.write_bytes


def _rank_local_kv_heads(dims: dict, tp: int) -> int:
    """KV heads owned by one tensor-parallel rank.

    A packed KV row is formed per rank from the heads that rank owns, so the
    row count of the whole system is the sum over ``tp`` ranks. Pricing the
    packed row with the model's global head count under-counts a tensor-
    parallel cache by ``tp`` (2026-09-08 fix).
    """
    heads = int(dims["kv_heads"])
    if tp <= 0:
        raise ValueError("tp must be positive")
    if heads % tp:
        raise ValueError("TP must own complete KV heads")
    return heads // tp

def decode_step_traffic_ledger(
    dims: dict,
    precision: dict,
    *,
    context: int,
    batch: int,
    mlen: int,
    kv_layout: str,
    weights: WeightLedger | None = None,
    include_lm_head: bool = True,
    tp: int = 1,
) -> DecodeStepTrafficLedger:
    """Physical weight and KV planes moved by one cached q_len=1 step.

    Each packed query row is a different sequence holding its own cache, so the
    query-row tiling the scalar FP SRAM forces splits the batch across passes
    rather than re-reading any cache: the KV read plane stays one pass.
    """
    if context <= 0 or batch <= 0:
        raise ValueError("context and batch must be positive")
    weights = weights or weight_ledger(
        dims,
        precision,
        include_lm_head=include_lm_head,
        batch=batch,
        mlen=mlen,
    )
    quantized = (
        weights.attention
        + weights.ffn_streamed
        + weights.lm_head_streamed
    )
    embedding_row_bytes = _align(
        math.ceil(int(dims["hidden"]) * 16 / 8),
        64,
    )
    block = int(precision.get("block_size", 8))
    key_layout = traffic_from_precision(
        kv_heads=_rank_local_kv_heads(dims, tp),
        head_dim=int(dims["head_dim"]),
        mlen=mlen,
        element_bits=int(precision.get("key_elem", precision["kv_elem"])),
        effective_bits=float(precision.get("key_bits", precision["kv_bits"])),
        block_size=block,
    )
    value_layout = traffic_from_precision(
        kv_heads=_rank_local_kv_heads(dims, tp),
        head_dim=int(dims["head_dim"]),
        mlen=mlen,
        element_bits=int(precision.get("value_elem", precision["kv_elem"])),
        effective_bits=float(precision.get("value_bits", precision["kv_bits"])),
        block_size=block,
    )
    full = int(dims.get("n_full", dims["layers"]))
    sliding = int(dims.get("n_sliding", 0))
    window = int(dims.get("sliding_window", 0))
    attended_token_layers = full * context
    if sliding:
        attended_token_layers += sliding * min(context, window)
    # Per-rank rows summed over the tp ranks that each own local_kv_heads.
    read_tensors = batch * attended_token_layers * tp
    write_tensors = batch * int(dims["layers"]) * tp
    return DecodeStepTrafficLedger(
        weight_element_read_bytes=quantized.element_aligned,
        weight_scale_read_bytes=quantized.scale_aligned,
        bf16_weight_read_bytes=(
            weights.bf16_streamed.total_aligned
            + batch * embedding_row_bytes
        ),
        activation_read_bytes=0,
        activation_write_bytes=0,
        kv_element_read_bytes=(
            read_tensors
            * (
                key_layout.read_element_bytes(kv_layout)
                + value_layout.read_element_bytes(kv_layout)
            )
        ),
        kv_scale_read_bytes=(
            read_tensors
            * (
                key_layout.read_scale_bytes(kv_layout)
                + value_layout.read_scale_bytes(kv_layout)
            )
        ),
        kv_element_write_bytes=(
            write_tensors
            * (
                key_layout.storage_element_bytes(kv_layout)
                + value_layout.storage_element_bytes(kv_layout)
            )
        ),
        kv_scale_write_bytes=(
            write_tensors
            * (
                key_layout.storage_scale_bytes(kv_layout)
                + value_layout.storage_scale_bytes(kv_layout)
            )
        ),
    )


def kv_ledger(
    dims: dict,
    precision: dict,
    *,
    context: int,
    batch: int,
    mlen: int,
    kv_layout: str,
    tp: int = 1,
) -> KVLedger:
    """Account K and V planes at the full and sliding-window layer spans."""

    if context <= 0 or batch <= 0:
        raise ValueError("context and batch must be positive")
    block = int(precision.get("block_size", 8))
    key_layout = traffic_from_precision(
        kv_heads=_rank_local_kv_heads(dims, tp),
        head_dim=int(dims["head_dim"]),
        mlen=mlen,
        element_bits=int(precision.get("key_elem", precision["kv_elem"])),
        effective_bits=float(precision.get("key_bits", precision["kv_bits"])),
        block_size=block,
    )
    value_layout = traffic_from_precision(
        kv_heads=_rank_local_kv_heads(dims, tp),
        head_dim=int(dims["head_dim"]),
        mlen=mlen,
        element_bits=int(precision.get("value_elem", precision["kv_elem"])),
        effective_bits=float(precision.get("value_bits", precision["kv_bits"])),
        block_size=block,
    )
    full = int(dims.get("n_full", dims["layers"]))
    sliding = int(dims.get("n_sliding", 0))
    window = int(dims.get("sliding_window", 0))
    token_layers = full * context
    if sliding:
        token_layers += sliding * min(context, window)
    per_batch_element = token_layers * tp * (
        key_layout.storage_element_bytes(kv_layout)
        + value_layout.storage_element_bytes(kv_layout)
    )
    per_batch_scale = token_layers * tp * (
        key_layout.storage_scale_bytes(kv_layout)
        + value_layout.storage_scale_bytes(kv_layout)
    )
    layout_id = (
        key_layout.layout_id
        if key_layout.layout_id == value_layout.layout_id
        else f"key={key_layout.layout_id};value={value_layout.layout_id}"
    )
    return KVLedger(
        element_bytes=per_batch_element * batch,
        scale_bytes=per_batch_scale * batch,
        per_batch_element_bytes=per_batch_element,
        per_batch_scale_bytes=per_batch_scale,
        layout_id=layout_id,
    )


@dataclass(frozen=True)
class SRAMLedger:
    """One-layer scratch requirements and capacities."""

    vector_capacity_bytes: int
    vector_bytes_per_sequence: int
    vector_required_bytes: int
    matrix_capacity_bytes: int
    matrix_required_bytes: int
    matrix_tile_capacity: int
    matrix_required_tiles: int
    max_vector_batch: int
    max_synchronous_batch: int
    output_head_logit_tile_bytes: int
    output_head_selection_state_bytes: int
    output_head_workspace_bytes: int

    @property
    def fits(self) -> bool:
        return (
            self.vector_required_bytes <= self.vector_capacity_bytes
            and self.matrix_required_bytes <= self.matrix_capacity_bytes
        )


def sram_ledger(
    dims: dict,
    precision: dict,
    hardware,
    *,
    batch: int,
) -> SRAMLedger:
    """Conservative one-layer workspace under the fused decode schedule."""

    if batch <= 0:
        raise ValueError("batch must be positive")
    hidden = int(dims["hidden"])
    query = int(dims["heads"]) * int(dims["head_dim"])
    kv = int(dims["kv_heads"]) * int(dims["head_dim"])
    inter = int(dims["inter"])
    activation_bytes = 2
    attention_elements = 2 * hidden + 2 * query + 2 * kv
    ffn_elements = 2 * hidden + 2 * inter
    vector_per_sequence = max(attention_elements, ffn_elements) * activation_bytes
    output_head_logit_tile_bytes = 0
    output_head_selection_state_bytes = 0
    output_head_workspace_bytes = 0
    if bool(precision.get("lm_head_quantized", False)):
        top_k = int(precision.get("lm_head_top_k", LOCAL_HEAD_TOP_K))
        if top_k != LOCAL_HEAD_TOP_K:
            raise ValueError("local output head requires top_k=20")
        per_sequence_state = (
            top_k * LOCAL_HEAD_CANDIDATE_BYTES
            + LOCAL_HEAD_SAMPLER_STATE_BYTES
            + LOCAL_HEAD_ARGMAX_STATE_BYTES
        )
        output_head_logit_tile_bytes = (
            int(hardware.BLEN)
            * int(hardware.MLEN)
            * LOCAL_HEAD_LOGIT_BYTES
        )
        output_head_selection_state_bytes = per_sequence_state * batch
        output_head_workspace_bytes = (
            output_head_selection_state_bytes
            + output_head_logit_tile_bytes
        )
    vector_capacity = int(hardware.VECTOR_SRAM_SIZE) * int(hardware.VLEN) * activation_bytes
    body_max_batch = vector_capacity // max(vector_per_sequence, 1)
    if bool(precision.get("lm_head_quantized", False)):
        head_max_batch = max(
            0,
            (vector_capacity - output_head_logit_tile_bytes)
            // max(per_sequence_state, 1),
        )
        max_vector_batch = min(body_max_batch, head_max_batch)
    else:
        max_vector_batch = body_max_batch

    # HBM operands dequantize into compiler-managed BF16 Matrix SRAM tiles.
    row_bytes = int(hardware.MLEN) * 2
    matrix_depth = int(hardware.MATRIX_SRAM_SIZE)
    matrix_capacity = matrix_depth * row_bytes
    matrix_tile_capacity = matrix_depth // int(hardware.MLEN)
    prefetch_rows = int(
        getattr(hardware, "HBM_M_Prefetch_Amount", hardware.MLEN)
    )
    prefetch_tiles = math.ceil(prefetch_rows / int(hardware.MLEN))
    matrix_required_tiles = max(MIN_MATRIX_TILE_CAPACITY, prefetch_tiles)
    matrix_required = (
        matrix_required_tiles * int(hardware.MLEN) * row_bytes
    )
    max_synchronous = (
        min(max_vector_batch, int(hardware.BLEN))
        if matrix_required <= matrix_capacity
        else 0
    )
    return SRAMLedger(
        vector_capacity_bytes=vector_capacity,
        vector_bytes_per_sequence=vector_per_sequence,
        vector_required_bytes=max(
            vector_per_sequence * batch,
            output_head_workspace_bytes,
        ),
        matrix_capacity_bytes=matrix_capacity,
        matrix_required_bytes=matrix_required,
        matrix_tile_capacity=matrix_tile_capacity,
        matrix_required_tiles=matrix_required_tiles,
        max_vector_batch=max_vector_batch,
        max_synchronous_batch=max_synchronous,
        output_head_logit_tile_bytes=output_head_logit_tile_bytes,
        output_head_selection_state_bytes=(
            output_head_selection_state_bytes
        ),
        output_head_workspace_bytes=output_head_workspace_bytes,
    )


@dataclass(frozen=True)
class PhysicalDecodeLedger:
    """Capacity ledger for the evaluated context and hardware."""

    weights: WeightLedger
    kv: KVLedger
    sram: SRAMLedger
    hbm_capacity_bytes: int
    runtime_hbm_reserve_bytes: int
    hbm_required_bytes: int
    max_resident_batch: int
    max_runtime_batch: int
    kv_layout: str
    slowest_rank_hbm_required_bytes: int | None = None
    per_chip_hbm_capacity_bytes: int | None = None

    @property
    def fits_hbm(self) -> bool:
        aggregate_fits = self.hbm_required_bytes <= self.hbm_capacity_bytes
        if self.slowest_rank_hbm_required_bytes is None:
            return aggregate_fits
        if self.per_chip_hbm_capacity_bytes is None:
            raise ValueError("slowest-rank HBM use requires per-chip capacity")
        return (
            aggregate_fits
            and self.slowest_rank_hbm_required_bytes
            <= self.per_chip_hbm_capacity_bytes
        )

    @property
    def fits_runtime(self) -> bool:
        return self.fits_hbm and self.sram.fits


def build_physical_decode_ledger(
    dims: dict,
    precision: dict,
    hardware,
    *,
    context: int,
    batch: int,
    hbm_capacity_bytes: int,
    runtime_hbm_reserve_bytes: int,
    kv_layout: str,
    include_lm_head: bool = True,
    tp: int = 1,
) -> PhysicalDecodeLedger:
    """Build the aligned physical ledger used by capacity and traffic checks."""

    if hbm_capacity_bytes <= 0 or runtime_hbm_reserve_bytes < 0:
        raise ValueError("HBM capacity must be positive and reserve non-negative")
    weights = weight_ledger(
        dims,
        precision,
        include_lm_head=include_lm_head,
        batch=batch,
        mlen=int(hardware.MLEN),
    )
    kv = kv_ledger(
        dims,
        precision,
        context=context,
        batch=batch,
        mlen=int(hardware.MLEN),
        kv_layout=kv_layout,
        tp=tp,
    )
    sram = sram_ledger(dims, precision, hardware, batch=batch)
    resident_fixed = weights.resident.total_aligned + runtime_hbm_reserve_bytes
    available_kv = max(0, hbm_capacity_bytes - resident_fixed)
    max_resident = available_kv // max(kv.per_batch_bytes, 1)
    max_runtime = (
        min(max_resident, sram.max_vector_batch)
        if sram.matrix_required_bytes <= sram.matrix_capacity_bytes
        else 0
    )
    required = resident_fixed + kv.total_bytes
    return PhysicalDecodeLedger(
        weights=weights,
        kv=kv,
        sram=sram,
        hbm_capacity_bytes=hbm_capacity_bytes,
        runtime_hbm_reserve_bytes=runtime_hbm_reserve_bytes,
        hbm_required_bytes=required,
        max_resident_batch=max_resident,
        max_runtime_batch=max_runtime,
        kv_layout=kv_layout,
    )


__all__ = [
    "DecodeStepTrafficLedger",
    "KVLedger",
    "MIN_MATRIX_TILE_CAPACITY",
    "LOCAL_HEAD_CANDIDATE_BYTES",
    "LOCAL_HEAD_ARGMAX_STATE_BYTES",
    "LOCAL_HEAD_LOGIT_BYTES",
    "LOCAL_HEAD_SAMPLER_STATE_BYTES",
    "LOCAL_HEAD_TOP_K",
    "PhysicalDecodeLedger",
    "PlaneBytes",
    "SRAMLedger",
    "WeightLedger",
    "bf16_matrix_planes",
    "build_physical_decode_ledger",
    "decode_step_traffic_ledger",
    "kv_ledger",
    "matrix_planes",
    "sram_ledger",
    "weight_ledger",
]
