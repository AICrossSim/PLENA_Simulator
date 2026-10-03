"""Memory-aware latency of a dense decoder on one PLENA chip.

``PerfModel`` (``analytic_models/performance/perf_model.py``) counts pipelined
instruction cycles and charges the HBM prefetch/store instructions a single
cycle, so the TTFT and TPS of ``llama_model.py`` assume the memory system always
keeps up. This module adds the missing memory term. For each stage (the
embedding lookup and every transformer block) it estimates the DRAM bytes the
stage moves, prices them with a ``MemorySystem`` (``FixedBandwidthMemory`` for a
baseline, ``StackedDramModel`` for a 3D stack) and combines compute and memory
time with one of two overlap policies, named as in the compiler-trace latency
model:

* ``"stage-roofline"``: a stage takes the longer of the two because prefetch
  overlaps compute. DeepStack prices every operator this way
  (arXiv:2604.04750; tile-ai/DeepStack@8509061,
  ``src/tilesight/tilesight/fusion_support/hete_post_process_single_op.py``,
  ``hete_post_process_tensor_core_op``).
* ``"serial"``: compute and memory time add up.

The compute term calls ``PerfModel`` exactly as ``llama_model.py`` does,
including its decode issue factor of two, so a memory with unbounded bandwidth
reproduces llama_model's TTFT and TPS. One difference: ``head_dim`` comes from
the model config when present, whereas llama_model always uses
``hidden_size // num_attention_heads``; the two agree for Llama 3.x.

Traffic is a first-order estimate:

* a block's weights are read once per forward pass (ideal on-chip reuse);
* decode reads the layer's whole KV cache once per generated token and writes
  the new K and V rows;
* prefill attention re-reads K and V once per MLEN-row query tile, the
  H_PREFETCH_M loop of ``PerfModel.flash_attention``;
* activations reach DRAM only where ``PerfModel`` charges H_PREFETCH_V for an
  activation larger than Vector SRAM: the two RMSNorms, the projection and the
  residual of a prefill block, each read once and written once;
* storage precisions come from ``[ANALYTIC.PRECISION]`` in ``plena_settings.toml``:
  ``HBM_M_WEIGHT_TYPE`` for weights, ``HBM_M_KV_TYPE`` for KV reads,
  ``HBM_V_KV_TYPE`` for KV writes and ``HBM_V_ACT_TYPE`` for activations and
  embedding rows.

Mixture-of-experts decoders are rejected for now.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import toml

from analytic_models.performance.perf_model import PerfModel

from ._validate import finite_number, positive_int
from .memory import MemorySystem

OVERLAP_POLICIES = ("stage-roofline", "serial")

# llama_model.py multiplies decode instruction cycles by two "for instruction
# issue + memory access pipeline stages"; kept so the compute term matches it.
LLAMA_DECODE_ISSUE_FACTOR = 2

# Prefill block stages where PerfModel spills an over-sized activation through
# H_PREFETCH_V: rms_layer (twice), projection and residual.
PREFILL_ACTIVATION_SPILL_SITES = 4

_EXPERT_COUNT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts")


def _scalar_bits(spec: Mapping[str, Any], where: str) -> int:
    kind = spec.get("type")
    if kind == "Fp":
        return int(bool(spec["sign"])) + int(spec["exponent"]) + int(spec["mantissa"])
    if kind == "Int":
        return int(spec["width"])
    raise ValueError(f"{where}: unsupported data type {kind!r}")


def element_bits(spec: Mapping[str, Any], where: str = "precision") -> float:
    """Stored bits per element for one ``[*.PRECISION.*]`` table of ``plena_settings.toml``.

    An ``Mx`` block format shares one ``SCALE`` between ``block`` elements.
    """

    fmt = spec.get("format")
    if fmt == "Mx":
        block = positive_int(spec["block"], f"{where}.block")
        return _scalar_bits(spec["ELEM"], f"{where}.ELEM") + _scalar_bits(spec["SCALE"], f"{where}.SCALE") / block
    if fmt == "Plain":
        return _scalar_bits(spec["DATA_TYPE"], f"{where}.DATA_TYPE")
    if fmt is None and "type" in spec:
        return _scalar_bits(spec, where)
    raise ValueError(f"{where}: unsupported precision format {fmt!r}")


@dataclass(frozen=True)
class HbmStoragePrecision:
    """Bytes per element of the tensors kept in off-chip memory."""

    weight_bytes: float
    kv_read_bytes: float
    kv_write_bytes: float
    activation_bytes: float

    def __post_init__(self) -> None:
        for name in ("weight_bytes", "kv_read_bytes", "kv_write_bytes", "activation_bytes"):
            finite_number(getattr(self, name), name)

    @classmethod
    def from_settings(cls, settings_path: str | Path, *, section: str = "ANALYTIC") -> HbmStoragePrecision:
        data = toml.load(settings_path)
        precision = data.get(section, {}).get("PRECISION")
        if not isinstance(precision, Mapping):
            raise ValueError(f"{settings_path} has no [{section}.PRECISION] tables")

        def nbytes(key: str) -> float:
            if key not in precision:
                raise ValueError(f"{settings_path} has no [{section}.PRECISION.{key}] table")
            return element_bits(precision[key], f"{section}.PRECISION.{key}") / 8.0

        return cls(
            weight_bytes=nbytes("HBM_M_WEIGHT_TYPE"),
            kv_read_bytes=nbytes("HBM_M_KV_TYPE"),
            kv_write_bytes=nbytes("HBM_V_KV_TYPE"),
            activation_bytes=nbytes("HBM_V_ACT_TYPE"),
        )


@dataclass(frozen=True)
class DecoderShape:
    """The dimensions of a dense decoder-only transformer."""

    name: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_hidden_layers: int
    intermediate_size: int
    vocab_size: int
    head_dim: int
    tie_word_embeddings: bool

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "num_hidden_layers",
            "intermediate_size",
            "vocab_size",
            "head_dim",
        ):
            positive_int(getattr(self, name), name)
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")

    @classmethod
    def from_hf_config(cls, config: Mapping[str, Any], *, name: str) -> DecoderShape:
        for key in _EXPERT_COUNT_KEYS:
            if (config.get(key) or 1) > 1:
                raise ValueError(f"{name}: mixture-of-experts decoders ({key}={config[key]}) are not supported yet")
        hidden = config["hidden_size"]
        heads = config["num_attention_heads"]
        return cls(
            name=name,
            hidden_size=hidden,
            num_attention_heads=heads,
            num_key_value_heads=config.get("num_key_value_heads", heads),
            num_hidden_layers=config["num_hidden_layers"],
            intermediate_size=config["intermediate_size"],
            vocab_size=config["vocab_size"],
            head_dim=config.get("head_dim") or hidden // heads,
            tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> DecoderShape:
        config_path = Path(path)
        with config_path.open() as handle:
            return cls.from_hf_config(json.load(handle), name=config_path.stem)

    @property
    def q_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def block_weight_elements(self) -> int:
        """Q, K, V and O projections plus the gate, up and down projections."""

        attention = 2 * self.hidden_size * self.q_width + 2 * self.hidden_size * self.kv_width
        return attention + 3 * self.hidden_size * self.intermediate_size

    @property
    def total_weight_elements(self) -> int:
        tables = 1 if self.tie_word_embeddings else 2  # input embedding and LM head
        return self.num_hidden_layers * self.block_weight_elements + tables * self.vocab_size * self.hidden_size


Traffic = tuple[dict[str, float], dict[str, float]]


def prefill_block_traffic(
    shape: DecoderShape,
    precision: HbmStoragePrecision,
    *,
    batch_size: int,
    seq_len: int,
    mlen: int,
    vector_sram_elements: int,
) -> Traffic:
    """DRAM (reads, writes) in bytes of one prefill transformer block."""

    tokens = batch_size * seq_len
    kv_elements = 2 * tokens * shape.kv_width
    spill = max(0, shape.hidden_size * tokens - vector_sram_elements)
    activation = PREFILL_ACTIVATION_SPILL_SITES * spill * precision.activation_bytes
    reads = {
        "weights": shape.block_weight_elements * precision.weight_bytes,
        "kv_cache": math.ceil(seq_len / mlen) * kv_elements * precision.kv_read_bytes,
        "activation_spill": activation,
    }
    writes = {
        "kv_cache": kv_elements * precision.kv_write_bytes,
        "activation_spill": activation,
    }
    return reads, writes


def decode_block_traffic(
    shape: DecoderShape,
    precision: HbmStoragePrecision,
    *,
    batch_size: int,
    kv_len: int,
) -> Traffic:
    """DRAM (reads, writes) in bytes of one transformer block for one generated token."""

    reads = {
        "weights": shape.block_weight_elements * precision.weight_bytes,
        "kv_cache": 2 * batch_size * kv_len * shape.kv_width * precision.kv_read_bytes,
    }
    writes = {"kv_cache": 2 * batch_size * shape.kv_width * precision.kv_write_bytes}
    return reads, writes


@dataclass(frozen=True)
class PhaseEstimate:
    """Totals over the stages of one inference phase.

    ``memory_bound_seconds`` is the time spent in stages whose memory time
    exceeds their compute time. Byte counts include transfer quantisation.
    """

    seconds: float
    compute_seconds: float
    memory_seconds: float
    memory_bound_seconds: float
    read_bytes: float
    write_bytes: float
    dram_energy_pj: float | None


class _Phase:
    def __init__(self, track_energy: bool) -> None:
        self.seconds = 0.0
        self.compute_seconds = 0.0
        self.memory_seconds = 0.0
        self.memory_bound_seconds = 0.0
        self.read_bytes = 0.0
        self.write_bytes = 0.0
        self.energy_pj: float | None = 0.0 if track_energy else None

    def add(self, stage: PhaseEstimate, repeat: int = 1) -> None:
        self.seconds += repeat * stage.seconds
        self.compute_seconds += repeat * stage.compute_seconds
        self.memory_seconds += repeat * stage.memory_seconds
        self.memory_bound_seconds += repeat * stage.memory_bound_seconds
        self.read_bytes += repeat * stage.read_bytes
        self.write_bytes += repeat * stage.write_bytes
        if self.energy_pj is not None and stage.dram_energy_pj is not None:
            self.energy_pj += repeat * stage.dram_energy_pj

    def freeze(self) -> PhaseEstimate:
        return PhaseEstimate(
            seconds=self.seconds,
            compute_seconds=self.compute_seconds,
            memory_seconds=self.memory_seconds,
            memory_bound_seconds=self.memory_bound_seconds,
            read_bytes=self.read_bytes,
            write_bytes=self.write_bytes,
            dram_energy_pj=self.energy_pj,
        )


@dataclass(frozen=True)
class DecoderLatencyEstimate:
    model: str
    batch_size: int
    input_seq_len: int
    output_seq_len: int
    overlap_policy: str
    include_lm_head: bool
    compute_frequency_hz: float
    ttft_seconds: float
    tps: float
    prefill: PhaseEstimate
    first_token_decode: PhaseEstimate
    decode: PhaseEstimate
    weight_footprint_bytes: float
    kv_cache_footprint_bytes: float
    memory_capacity_bytes: int | None
    fits_in_memory: bool | None
    memory: dict[str, Any]
    precision: dict[str, float]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_decoder_latency(
    shape: DecoderShape,
    perf: PerfModel,
    precision: HbmStoragePrecision,
    memory: MemorySystem,
    *,
    batch_size: int,
    input_seq_len: int,
    output_seq_len: int,
    frequency_hz: float = 1e9,
    overlap_policy: str = "stage-roofline",
    include_lm_head: bool = False,
) -> DecoderLatencyEstimate:
    """TTFT and TPS of ``shape`` on one PLENA chip backed by ``memory``.

    ``frequency_hz`` is the nominal compute clock (llama_model.py's default is
    1 GHz); a ``memory`` with a thermal policy scales it. ``include_lm_head``
    adds the LM head projection, which llama_model.py leaves out, once in
    prefill and once per generated token.
    """

    batch = positive_int(batch_size, "batch_size")
    seq = positive_int(input_seq_len, "input_seq_len")
    out = positive_int(output_seq_len, "output_seq_len")
    if overlap_policy not in OVERLAP_POLICIES:
        raise ValueError(
            f"unsupported overlap policy {overlap_policy!r}; expected one of {', '.join(OVERLAP_POLICIES)}"
        )
    if not isinstance(include_lm_head, bool):
        raise TypeError("include_lm_head must be a bool")
    nominal_hz = finite_number(frequency_hz, "frequency_hz")
    frequency_scale = memory.compute_frequency_scale
    compute_hz = nominal_hz * frequency_scale
    bandwidth = memory.usable_bandwidth_bytes_per_s
    track_energy = memory.energy_pj(0.0, 0.0) is not None
    roofline = overlap_policy == "stage-roofline"

    def price(cycles: float, traffic: Traffic) -> PhaseEstimate:
        reads, writes = traffic
        read_bytes = sum(memory.quantize_bytes(value) for value in reads.values() if value > 0)
        write_bytes = sum(memory.quantize_bytes(value) for value in writes.values() if value > 0)
        compute_s = cycles / compute_hz
        memory_s = (read_bytes + write_bytes) / bandwidth
        seconds = max(compute_s, memory_s) if roofline else compute_s + memory_s
        return PhaseEstimate(
            seconds=seconds,
            compute_seconds=compute_s,
            memory_seconds=memory_s,
            memory_bound_seconds=seconds if memory_s > compute_s else 0.0,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            dram_energy_pj=memory.energy_pj(read_bytes, write_bytes),
        )

    hidden = shape.hidden_size
    heads = shape.num_attention_heads
    kv_heads = shape.num_key_value_heads
    head_dim = shape.head_dim
    layers = shape.num_hidden_layers
    lm_head = None
    if include_lm_head:
        lm_head_reads = {"lm_head_weights": shape.vocab_size * hidden * precision.weight_bytes}
        lm_head = price(perf.lm_head(hidden, shape.vocab_size, batch), (lm_head_reads, {}))

    # Prefill: embedding lookup, then the blocks, as in LLaMAModel.compute_prefill_time.
    prefill = _Phase(track_energy)
    embedding_rows = {"embedding_rows": batch * seq * hidden * precision.activation_bytes}
    prefill.add(price(perf.embeddings(hidden, seq, batch, "prefill"), (embedding_rows, {})))
    rms = perf.rms_layer(hidden, seq, batch, "prefill")
    block_cycles = (
        rms
        + perf.projection(hidden, heads, kv_heads, head_dim, seq, batch, "prefill")
        + perf.flash_attention(heads, kv_heads, head_dim, seq, seq, batch, "prefill")
        + perf.residual(hidden, seq, batch, "prefill")
        + rms
        + perf.feed_forward(hidden, shape.intermediate_size, seq, batch, "prefill")
    )
    block_traffic = prefill_block_traffic(
        shape,
        precision,
        batch_size=batch,
        seq_len=seq,
        mlen=perf.mlen,
        vector_sram_elements=perf.vector_sram_size,
    )
    prefill.add(price(block_cycles, block_traffic), layers)
    if lm_head is not None:
        prefill.add(lm_head)

    # Decode: one token at a time with a growing KV cache, as in LLaMAModel.compute_decode_time.
    kv_independent_cycles = (
        perf.rms_layer(hidden, 1, batch, "decode") * 2
        + perf.projection(hidden, heads, kv_heads, head_dim, 1, batch, "decode")
        + perf.residual(hidden, 1, batch, "decode")
        + perf.feed_forward(hidden, shape.intermediate_size, 1, batch, "decode")
    )

    def decode_token(kv_len: int) -> _Phase:
        cycles = kv_independent_cycles + perf.flash_attention(heads, kv_heads, head_dim, 1, kv_len, batch, "decode")
        traffic = decode_block_traffic(shape, precision, batch_size=batch, kv_len=kv_len)
        token = _Phase(track_energy)
        token.add(price(cycles * LLAMA_DECODE_ISSUE_FACTOR, traffic), layers)
        if lm_head is not None:
            token.add(lm_head)
        return token

    decode = _Phase(track_energy)
    first_token = decode_token(seq).freeze()
    decode.add(first_token)
    for step in range(1, out):
        decode.add(decode_token(seq + step).freeze())

    weight_footprint = shape.total_weight_elements * precision.weight_bytes
    kv_footprint = 2 * layers * (seq + out) * batch * shape.kv_width * precision.kv_write_bytes
    capacity = memory.capacity_bytes
    fits = None if capacity is None else weight_footprint + kv_footprint <= capacity

    warnings: list[str] = []
    if fits is False:
        warnings.append(
            f"weights ({weight_footprint / 2**30:.2f} GiB) and KV cache ({kv_footprint / 2**30:.2f} GiB) "
            f"exceed the memory capacity ({capacity / 2**30:.2f} GiB)"
        )
    littles_law = getattr(memory, "littles_law", None)
    bound = littles_law() if callable(littles_law) else None
    if bound is not None and bound.limited:
        warnings.append(
            f"buffering limits usable bandwidth to {bound.bandwidth_bytes_per_s / 1e9:.3f} GB/s "
            f"(needs {bound.required_buffer_bytes_per_requester:.0f} B per requester, has {bound.buffer_bytes_per_requester})"
        )
    if frequency_scale != 1.0:
        warnings.append(f"thermal policy scales the compute clock by {frequency_scale:.4f}")
    if head_dim * heads != hidden:
        warnings.append(
            f"head_dim * num_attention_heads ({head_dim * heads}) differs from hidden_size ({hidden}); "
            "weight traffic uses the true projection shapes, PerfModel's projection cycles assume hidden-wide Q/O"
        )

    return DecoderLatencyEstimate(
        model=shape.name,
        batch_size=batch,
        input_seq_len=seq,
        output_seq_len=out,
        overlap_policy=overlap_policy,
        include_lm_head=include_lm_head,
        compute_frequency_hz=compute_hz,
        ttft_seconds=prefill.seconds + first_token.seconds,
        tps=batch * out / decode.seconds,
        prefill=prefill.freeze(),
        first_token_decode=first_token,
        decode=decode.freeze(),
        weight_footprint_bytes=weight_footprint,
        kv_cache_footprint_bytes=kv_footprint,
        memory_capacity_bytes=capacity,
        fits_in_memory=fits,
        memory=memory.describe(),
        precision=asdict(precision),
        warnings=tuple(warnings),
    )
