"""Compute cycles and DRAM traffic of one transformer layer on one PLENA chip of a plan.

The compute terms are the ``PerfModel`` calls of PLENA's single-chip scripts
(``llama_model.py`` and ``gpt_oss_model.py``) with per-device shapes:

* attention heads and KV heads are split over TP (KV heads are replicated
  when TP exceeds their count) and the KV sequence over CP;
* the QKV projection kernel, whose ``PerfModel`` cost does not take a sharded
  width, is split evenly over the TP ranks;
* the dense FFN is column/row parallel over TP (``intermediate_size / tp``);
* routed experts follow ``moe.moe_cycles``.

The traffic terms are the per-device versions of ``stacked_dram.estimate``'s
block traffic: the device's weight shards, its KV cache slice and the
activation spills ``PerfModel`` charges in prefill.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..performance.perf_model import PerfModel
from ..stacked_dram.estimate import HbmStoragePrecision, Traffic
from .moe import ExpertLoad, moe_cycles
from .plan import ParallelPlan
from .workload import ModelSpec


@dataclass(frozen=True)
class DeviceShard:
    """Per-device dimensions of one layer."""

    batch: int
    tp: int
    cp: int
    heads: int
    kv_heads: int
    intermediate: int
    vocab: int

    @classmethod
    def build(cls, model: ModelSpec, plan: ParallelPlan, micro_batch: int) -> DeviceShard:
        tp, kv = plan.tp, model.num_key_value_heads
        return cls(
            batch=math.ceil(micro_batch / (plan.dp * plan.ep)),
            tp=tp,
            cp=plan.cp,
            heads=model.num_attention_heads // tp,
            kv_heads=kv // tp if tp <= kv else 1,
            intermediate=math.ceil(model.intermediate_size / tp),
            vocab=math.ceil(model.vocab_size / tp),
        )


def _spill_bytes(perf: PerfModel, precision: HbmStoragePrecision, hidden: int, tokens: int) -> float:
    """Bytes ``PerfModel`` moves through H_PREFETCH_V for one over-sized prefill activation, per direction."""

    return max(0, hidden * tokens - perf.vector_sram_size) * precision.activation_bytes


def attention_half(
    model: ModelSpec,
    shard: DeviceShard,
    perf: PerfModel,
    precision: HbmStoragePrecision,
    *,
    phase: str,
    layer_type: str,
    q_len: int,
    kv_len: int,
) -> tuple[float, Traffic]:
    """RMSNorm, QKV projection, attention and the residual after it."""

    hidden, head_dim, batch = model.hidden_size, model.head_dim, shard.batch
    cycles = perf.rms_layer(hidden, q_len, batch, phase)
    cycles += (
        perf.projection(hidden, model.num_attention_heads, model.num_key_value_heads, head_dim, q_len, batch, phase)
        / shard.tp
    )
    if layer_type == "sliding_attention":
        cycles += perf.sliding_window_attention(
            num_attention_heads=shard.heads,
            num_kv_heads=shard.kv_heads,
            head_dim=head_dim,
            seq_len=q_len,
            kv_size=kv_len,
            batch_size=batch,
            sliding_window_size=model.sliding_window,
            num_sink_tokens=1,
            mode=phase,
        )
        attended = min(kv_len, model.sliding_window)
    else:
        cycles += perf.flash_attention(shard.heads, shard.kv_heads, head_dim, q_len, kv_len, batch, phase)
        attended = kv_len
    cycles += perf.residual(hidden, q_len, batch, phase)

    kv_row = 2 * batch * shard.kv_heads * head_dim
    rereads = math.ceil(q_len / perf.mlen) if phase == "prefill" else 1
    weights = 2 * hidden * shard.heads * head_dim + 2 * hidden * shard.kv_heads * head_dim
    # Decode appends one KV row per sequence; CP ranks take turns holding it.
    writes_per_rank = q_len if phase == "prefill" else 1 / shard.cp
    reads = {
        "weights": weights * precision.weight_bytes,
        "kv_cache": rereads * kv_row * attended * precision.kv_read_bytes,
    }
    writes = {"kv_cache": kv_row * writes_per_rank * precision.kv_write_bytes}
    if phase == "prefill":
        spill = 3 * _spill_bytes(perf, precision, hidden, batch * q_len)  # rms_layer, projection, residual
        reads["activation_spill"] = spill
        writes["activation_spill"] = spill
    return cycles, (reads, writes)


def mlp_half(
    model: ModelSpec,
    shard: DeviceShard,
    perf: PerfModel,
    precision: HbmStoragePrecision,
    *,
    phase: str,
    mlp_type: str,
    q_len: int,
    expert_load: ExpertLoad | None = None,
    expert_intermediate: int | None = None,
) -> tuple[float, Traffic]:
    """RMSNorm and the dense FFN or routed experts (plus the residual of the MoE composition)."""

    hidden, batch = model.hidden_size, shard.batch
    cycles = perf.rms_layer(hidden, q_len, batch, phase)
    spill_sites = 1  # rms_layer
    if mlp_type == "moe":
        if expert_load is None or expert_intermediate is None:
            raise ValueError("MoE layers need an expert load and the experts' per-rank width")
        cycles += moe_cycles(
            perf,
            hidden_size=hidden,
            intermediate_size=expert_intermediate,
            num_experts=model.num_experts,
            top_k=model.experts_per_token,
            load=expert_load,
            mode=phase,
        )
        weights = (
            hidden * model.num_experts + expert_load.active_experts * 3 * hidden * expert_intermediate
        )  # router plus the experts this rank serves
    else:
        cycles += perf.feed_forward(hidden, shard.intermediate, q_len, batch, phase)
        weights = 3 * hidden * shard.intermediate
    if model.family == "moe":
        cycles += perf.residual(hidden, q_len, batch, phase)
        spill_sites += 1

    reads = {"weights": weights * precision.weight_bytes}
    writes: dict[str, float] = {}
    if phase == "prefill":
        spill = spill_sites * _spill_bytes(perf, precision, hidden, batch * q_len)
        reads["activation_spill"] = spill
        writes["activation_spill"] = spill
    return cycles, (reads, writes)


def embedding_lookup(
    model: ModelSpec, shard: DeviceShard, perf: PerfModel, precision: HbmStoragePrecision, *, q_len: int
) -> tuple[float, Traffic]:
    """Prefill embedding lookup on the first pipeline stage."""

    rows = {"embedding_rows": shard.batch * q_len * model.hidden_size * precision.activation_bytes}
    return perf.embeddings(model.hidden_size, q_len, shard.batch, "prefill"), (rows, {})


def lm_head(
    model: ModelSpec, shard: DeviceShard, perf: PerfModel, precision: HbmStoragePrecision
) -> tuple[float, Traffic]:
    """Vocabulary-parallel LM head on the last pipeline stage."""

    weights = {"lm_head_weights": shard.vocab * model.hidden_size * precision.weight_bytes}
    return perf.lm_head(model.hidden_size, shard.vocab, shard.batch), (weights, {})
