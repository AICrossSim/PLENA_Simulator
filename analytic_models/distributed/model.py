"""Latency and throughput of a multi-chip PLENA deployment.

Every device is a PLENA chip: ``PerfModel`` compute and a ``MemorySystem``
from ``analytic_models.stacked_dram`` (a fixed bandwidth such as an HBM
baseline, or a 3D DRAM stack). The chips are connected by a DeepStack NoC
hierarchy and run a ``ParallelPlan``. The network side (topology, routing,
collective algorithms, pipeline transfers, MoE routing statistics) is
DeepStack's (arXiv:2604.04750, tile-ai/DeepStack, the ``DeepStack``
submodule).

A transformer layer on one device is one roofline stage (its compute against
its DRAM traffic, ``stacked_dram.price_stage``) plus its collectives:

* a TP all-reduce of the attention output, and one of the dense FFN output;
* with CP, an all-reduce of the partial attention outputs in decode and an
  all-gather of K and V in prefill;
* in MoE layers, the token dispatch and combine across each EP group
  (DeepStack's ``ep_all_to_all_wrapper``) and, when the experts stay tensor
  parallel, a TP all-reduce of the expert outputs. Where TP ranks become EP
  ranks (``moe_tp_mode="replace"``), the TP all-reduce after attention stands
  in for the reduce-scatter and all-gather around the experts, as in
  DeepStack's decode driver;
* RMSNorm is replicated across TP ranks and needs no collective, and sampling
  reduces the vocabulary-parallel logits locally.

Collectives follow the layer's compute (``comm_overlap="none"``) or hide
behind it (``"full"``). The layers are cut into ``pp`` contiguous stages of
``ceil(layers / pp)``. The first stage also runs the prefill embedding lookup
and the last stage the LM head where the single-chip composition has one.

Pipelining follows DeepStack's drivers: the batch is split into ``pp``
micro-batches of ``ceil(batch / pp)`` sequences; the system emits one
micro-batch of tokens per pipeline period (the slowest stage plus one
stage-to-stage transfer), so a sequence advances by one token every ``pp``
periods. TTFT is one micro-batch's prefill through every stage plus its first
decode step. With one device the estimate reduces to the single-chip one
(``stacked_dram.estimate_decoder_latency`` for the llama composition).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from ..performance.perf_model import PerfModel
from ..stacked_dram._validate import finite_number, positive_int
from ..stacked_dram.estimate import (
    LLAMA_DECODE_ISSUE_FACTOR,
    OVERLAP_POLICIES,
    HbmStoragePrecision,
    PhaseEstimate,
    PhaseTotals,
    Traffic,
    price_stage,
)
from ..stacked_dram.memory import MemorySystem
from .comm import CommCost, Network
from .device import DeviceShard, attention_half, embedding_lookup, lm_head, mlp_half
from .moe import TRACE_COMM_TOKEN_LIMIT, ExpertLoad, check_routing, expert_load, routing_rows
from .noc import NocProfile
from .plan import ParallelPlan
from .workload import ModelSpec

COMM_OVERLAP_POLICIES = ("none", "full")
COLLECTIVES = (
    "tp_all_reduce",
    "cp_attention_combine",
    "cp_kv_all_gather",
    "ep_dispatch",
    "moe_tp_all_reduce",
    "ep_combine",
)


@dataclass(frozen=True)
class StageReport:
    """One pipeline stage processing one micro-batch, on one of its devices."""

    stage: int
    layers: int
    seconds: float
    compute_seconds: float
    memory_seconds: float
    memory_bound_seconds: float
    comm_seconds: dict[str, float]
    read_bytes: float
    write_bytes: float
    dram_energy_pj: float | None


@dataclass(frozen=True)
class PipelinePass:
    """One micro-batch through every pipeline stage."""

    stages: tuple[StageReport, ...]
    transfer_seconds: float
    period_seconds: float
    latency_seconds: float
    bottleneck_stage: int


@dataclass(frozen=True)
class DistributedEstimate:
    model: dict[str, Any]
    plan: dict[str, Any]
    noc: dict[str, Any]
    memory: dict[str, Any]
    precision: dict[str, float]
    batch_size: int
    micro_batch: int
    sequences_per_device: int
    input_seq_len: int
    output_seq_len: int
    overlap_policy: str
    comm_overlap: str
    routing: str | None
    include_lm_head: bool
    compute_frequency_hz: float
    ttft_seconds: float
    tps: float
    tps_per_sequence: float
    prefill: PipelinePass
    first_token_decode: PipelinePass
    decode_seconds: float
    weight_bytes_per_device: float
    kv_cache_bytes_per_device: float
    memory_capacity_bytes: int | None
    fits_in_memory: bool | None
    moe: dict[str, Any] | None
    noc_energy_pj_per_decode_token: float | None
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _PhaseContext:
    q_len: int
    attention_calls: tuple[tuple[str, CommCost], ...]
    ffn_calls: tuple[tuple[str, CommCost], ...]
    moe_calls: tuple[tuple[str, CommCost], ...]
    p2p: CommCost
    load: ExpertLoad | None

    def calls(self, mlp_type: str) -> tuple[tuple[str, CommCost], ...]:
        return self.attention_calls + (self.moe_calls if mlp_type == "moe" else self.ffn_calls)


def _merge(*traffics: Traffic) -> Traffic:
    reads: dict[str, float] = {}
    writes: dict[str, float] = {}
    for part_reads, part_writes in traffics:
        for name, value in part_reads.items():
            reads[name] = reads.get(name, 0.0) + value
        for name, value in part_writes.items():
            writes[name] = writes.get(name, 0.0) + value
    return reads, writes


def _validate(model: ModelSpec, plan: ParallelPlan, noc: NocProfile, routing: str, input_seq_len: int) -> None:
    if plan.world_size != noc.num_devices:
        raise ValueError(f"the plan uses {plan.world_size} devices but NoC profile {noc.name!r} has {noc.num_devices}")
    if model.num_attention_heads % plan.tp:
        raise ValueError(f"tp={plan.tp} does not divide the {model.num_attention_heads} attention heads")
    kv = model.num_key_value_heads
    if (plan.tp <= kv and kv % plan.tp) or (plan.tp > kv and plan.tp % kv):
        raise ValueError(f"tp={plan.tp} and the {kv} KV heads must divide one another")
    if plan.pp > model.num_hidden_layers:
        raise ValueError(f"pp={plan.pp} exceeds the {model.num_hidden_layers} layers")
    if plan.cp > 1:
        if plan.cp > input_seq_len:
            raise ValueError(f"cp={plan.cp} exceeds the {input_seq_len}-token prompt")
        if model.has_moe or "sliding_attention" in model.layer_types:
            raise ValueError("context parallelism is modelled for dense full-attention models only")
    if model.has_moe:
        moe_ep = plan.moe_scheme(model.num_experts).ep
        if model.num_experts % moe_ep:
            raise ValueError(f"{model.num_experts} experts do not split evenly over {moe_ep} expert-parallel ranks")
        check_routing(routing, num_experts=model.num_experts, top_k=model.experts_per_token)
    elif plan.ep > 1:
        raise ValueError("ep > 1 needs a model with MoE layers; use dp for a dense model")


def estimate_distributed(
    model: ModelSpec,
    plan: ParallelPlan,
    noc: NocProfile,
    perf: PerfModel,
    precision: HbmStoragePrecision,
    memory: MemorySystem,
    *,
    batch_size: int,
    input_seq_len: int,
    output_seq_len: int,
    frequency_hz: float = 1e9,
    overlap_policy: str = "stage-roofline",
    comm_overlap: str = "none",
    routing: str = "balanced",
    include_lm_head: bool = False,
) -> DistributedEstimate:
    """TTFT and decode throughput of ``model`` on the ``noc.num_devices`` PLENA chips of ``plan``.

    ``perf``, ``precision`` and ``memory`` describe one chip; every chip is
    identical. ``routing`` picks the MoE routing (``moe.ROUTING_MODES``).
    ``include_lm_head`` adds the LM head to every decode step (and to prefill
    for the llama composition, which leaves it out), as in
    ``stacked_dram.estimate_decoder_latency``.
    """

    batch = positive_int(batch_size, "batch_size")
    seq = positive_int(input_seq_len, "input_seq_len")
    out = positive_int(output_seq_len, "output_seq_len")
    if overlap_policy not in OVERLAP_POLICIES:
        raise ValueError(f"overlap_policy must be one of {', '.join(OVERLAP_POLICIES)}")
    if comm_overlap not in COMM_OVERLAP_POLICIES:
        raise ValueError(f"comm_overlap must be one of {', '.join(COMM_OVERLAP_POLICIES)}")
    if not isinstance(include_lm_head, bool):
        raise TypeError("include_lm_head must be a bool")
    _validate(model, plan, noc, routing, seq)

    network = Network(noc)
    dense = plan.dense_scheme()
    moe_scheme = plan.moe_scheme(model.num_experts) if model.has_moe else None
    micro = math.ceil(batch / plan.pp)
    shard = DeviceShard.build(model, plan, micro)
    frequency_scale = memory.compute_frequency_scale
    compute_hz = finite_number(frequency_hz, "frequency_hz") * frequency_scale
    track_energy = memory.energy_pj(0.0, 0.0) is not None
    hidden, head_dim, act = model.hidden_size, model.head_dim, precision.activation_bytes
    expert_width = math.ceil(model.moe_intermediate_size / moe_scheme.tp) if moe_scheme else None
    lm_head_phases = {"decode"} if include_lm_head else set()
    if include_lm_head or model.family == "moe":
        lm_head_phases.add("prefill")

    def price(cycles: float, traffic: Traffic) -> PhaseEstimate:
        return price_stage(cycles, traffic, memory=memory, compute_hz=compute_hz, overlap_policy=overlap_policy)

    def phase_context(phase: str) -> _PhaseContext:
        q_len = math.ceil(seq / plan.cp) if phase == "prefill" else 1
        activation = shard.batch * q_len * hidden * act
        tp_all_reduce = ("tp_all_reduce", network.all_reduce(dense, "tp", activation))
        if phase == "decode":
            cp = ("cp_attention_combine", network.all_reduce(dense, "cp", shard.batch * shard.heads * head_dim * act))
        else:
            kv_bytes = 2 * shard.batch * seq * shard.kv_heads * head_dim * precision.kv_write_bytes
            cp = ("cp_kv_all_gather", network.all_gather(dense, "cp", kv_bytes))
        moe_calls: tuple[tuple[str, CommCost], ...] = ()
        load = None
        if moe_scheme is not None:
            tokens = micro * q_len
            load = expert_load(
                routing,
                phase=phase,
                scheme=moe_scheme,
                micro_batch_tokens=tokens,
                group_tokens=math.ceil(micro / moe_scheme.dp) * q_len,
                num_experts=model.num_experts,
                top_k=model.experts_per_token,
            )
            rows = None
            if moe_scheme.ep > 1 and tokens < TRACE_COMM_TOKEN_LIMIT:
                rows = routing_rows(
                    routing, phase=phase, tokens=tokens, num_experts=model.num_experts, top_k=model.experts_per_token
                )
            all_to_all = network.ep_all_to_all(
                moe_scheme,
                bytes_each_token=hidden * act,
                routing=rows,
                micro_batch=micro,
                seq=q_len,
                num_experts=model.num_experts,
                top_k=model.experts_per_token,
                imbalance=load.imbalance,
                tag=f"{phase}:{routing}",
            )
            moe_calls = (
                ("ep_dispatch", all_to_all),
                ("moe_tp_all_reduce", network.all_reduce(moe_scheme, "tp", load.pairs * hidden * act)),
                ("ep_combine", all_to_all),
            )
        return _PhaseContext(
            q_len=q_len,
            attention_calls=(tp_all_reduce, cp),
            ffn_calls=(tp_all_reduce,),
            moe_calls=moe_calls,
            p2p=network.pipeline_p2p(dense, activation),
            load=load,
        )

    contexts = {phase: phase_context(phase) for phase in ("prefill", "decode")}
    attention_cache: dict[tuple[str, str, int], tuple[float, Traffic]] = {}
    mlp_cache: dict[tuple[str, str], tuple[float, Traffic]] = {}

    def block(phase: str, layer_type: str, mlp_type: str, kv_len: int) -> PhaseEstimate:
        context = contexts[phase]
        attention_key = (phase, layer_type, kv_len)
        if attention_key not in attention_cache:
            attention_cache[attention_key] = attention_half(
                model, shard, perf, precision, phase=phase, layer_type=layer_type, q_len=context.q_len, kv_len=kv_len
            )
        mlp_key = (phase, mlp_type)
        if mlp_key not in mlp_cache:
            mlp_cache[mlp_key] = mlp_half(
                model,
                shard,
                perf,
                precision,
                phase=phase,
                mlp_type=mlp_type,
                q_len=context.q_len,
                expert_load=context.load,
                expert_intermediate=expert_width,
            )
        attention_cycles, attention_traffic = attention_cache[attention_key]
        mlp_cycles, mlp_traffic = mlp_cache[mlp_key]
        factor = LLAMA_DECODE_ISSUE_FACTOR if phase == "decode" else 1
        return price((attention_cycles + mlp_cycles) * factor, _merge(attention_traffic, mlp_traffic))

    layers = model.num_hidden_layers
    per_stage = math.ceil(layers / plan.pp)
    stage_layers = [range(s * per_stage, min(layers, (s + 1) * per_stage)) for s in range(plan.pp)]
    stage_kinds = [Counter((model.layer_types[i], model.mlp_types[i]) for i in span) for span in stage_layers]

    def run(phase: str, kv_len: int) -> PipelinePass:
        context = contexts[phase]
        reports = []
        for index, kinds in enumerate(stage_kinds):
            totals = PhaseTotals(track_energy)
            seconds = 0.0
            comm = dict.fromkeys(COLLECTIVES, 0.0)
            extras = []
            if phase == "prefill" and index == 0:
                extras.append(embedding_lookup(model, shard, perf, precision, q_len=context.q_len))
            if phase in lm_head_phases and index == plan.pp - 1:
                extras.append(lm_head(model, shard, perf, precision))
            for cycles, traffic in extras:
                part = price(cycles, traffic)
                totals.add(part)
                seconds += part.seconds
            for (layer_type, mlp_type), count in kinds.items():
                part = block(phase, layer_type, mlp_type, kv_len)
                calls = context.calls(mlp_type)
                comm_seconds = sum(cost.seconds for _, cost in calls)
                totals.add(part, count)
                if comm_overlap == "none":
                    seconds += count * (part.seconds + comm_seconds)
                else:
                    seconds += count * max(part.seconds, comm_seconds)
                for name, cost in calls:
                    comm[name] += count * cost.seconds
            stage = totals.freeze()
            reports.append(
                StageReport(
                    stage=index,
                    layers=len(stage_layers[index]),
                    seconds=seconds,
                    compute_seconds=stage.compute_seconds,
                    memory_seconds=stage.memory_seconds,
                    memory_bound_seconds=stage.memory_bound_seconds,
                    comm_seconds={name: value for name, value in comm.items() if value},
                    read_bytes=stage.read_bytes,
                    write_bytes=stage.write_bytes,
                    dram_energy_pj=stage.dram_energy_pj,
                )
            )
        transfer = context.p2p.seconds
        bottleneck = max(range(plan.pp), key=lambda i: reports[i].seconds)
        return PipelinePass(
            stages=tuple(reports),
            transfer_seconds=transfer,
            period_seconds=reports[bottleneck].seconds + transfer,
            latency_seconds=sum(report.seconds for report in reports) + (plan.pp - 1) * transfer,
            bottleneck_stage=bottleneck,
        )

    prefill = run("prefill", seq)
    first_token = run("decode", math.ceil(seq / plan.cp))
    decode_seconds = first_token.period_seconds
    for step in range(1, out):
        decode_seconds += run("decode", math.ceil((seq + step) / plan.cp)).period_seconds

    # Footprint of the fullest stage's devices.
    def stage_weight_bytes(index: int) -> float:
        total = 0.0
        for i in stage_layers[index]:
            total += 2 * hidden * shard.heads * head_dim + 2 * hidden * shard.kv_heads * head_dim
            if model.mlp_types[i] == "moe":
                total += hidden * model.num_experts + model.num_experts // moe_scheme.ep * 3 * hidden * expert_width
            else:
                total += 3 * hidden * shard.intermediate
        tables = shard.vocab * hidden  # vocabulary-parallel embedding and LM head
        if index == 0:
            total += tables
        if index == plan.pp - 1 and not (plan.pp == 1 and model.tie_word_embeddings):
            total += tables
        return total * precision.weight_bytes

    def stage_kv_bytes(index: int) -> float:
        tokens = math.ceil((seq + out) / plan.cp)
        total = 0.0
        for i in stage_layers[index]:
            held = min(tokens, model.sliding_window) if model.layer_types[i] == "sliding_attention" else tokens
            total += 2 * shard.batch * held * shard.kv_heads * head_dim
        return total * precision.kv_write_bytes

    footprints = [(stage_weight_bytes(i), stage_kv_bytes(i)) for i in range(plan.pp)]
    weight_bytes, kv_bytes = max(footprints, key=sum)
    capacity = memory.capacity_bytes
    fits = None if capacity is None else weight_bytes + kv_bytes <= capacity

    # NoC energy of one decode pipeline period, per emitted token. A collective's
    # traffic spans every pipeline stage, and each stage runs its own layers once
    # per period; every stage boundary transfers once.
    noc_energy = None
    if noc.energy is not None:
        decode = contexts["decode"]
        collectives_pj = sum(
            network.energy_pj(cost) for i in range(layers) for _, cost in decode.calls(model.mlp_types[i])
        )
        noc_energy = (collectives_pj / plan.pp + network.energy_pj(decode.p2p)) / micro

    warnings: list[str] = []
    replicas = plan.dp * plan.ep
    if micro < replicas:
        warnings.append(f"the {micro}-sequence micro-batch leaves some of the {replicas} data-parallel ranks idle")
    if batch % plan.pp:
        warnings.append(f"batch {batch} is not a multiple of pp={plan.pp}; micro-batches are rounded up to {micro}")
    if any(len(span) == 0 for span in stage_layers):
        warnings.append(f"{layers} layers in stages of {per_stage} leave pipeline stages empty")
    if plan.tp > model.num_key_value_heads:
        warnings.append(f"tp={plan.tp} exceeds the {model.num_key_value_heads} KV heads; KV heads are replicated")
    if fits is False:
        warnings.append(
            f"a device of the fullest stage holds weights ({weight_bytes / 2**30:.2f} GiB) and KV cache "
            f"({kv_bytes / 2**30:.2f} GiB) beyond its memory capacity ({capacity / 2**30:.2f} GiB)"
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
    if head_dim * model.num_attention_heads != hidden:
        warnings.append(
            f"head_dim * num_attention_heads ({head_dim * model.num_attention_heads}) differs from hidden_size "
            f"({hidden}); weight traffic uses the true projection shapes, PerfModel's projection cycles assume "
            "hidden-wide Q/O"
        )

    moe_report = None
    if moe_scheme is not None:
        moe_report = {
            "tp": moe_scheme.tp,
            "ep": moe_scheme.ep,
            "dp": moe_scheme.dp,
            "experts_per_rank": model.num_experts // moe_scheme.ep,
            "prefill": contexts["prefill"].load.describe(),
            "decode": contexts["decode"].load.describe(),
        }

    return DistributedEstimate(
        model=model.describe(),
        plan=plan.describe(),
        noc=noc.describe(),
        memory=memory.describe(),
        precision=asdict(precision),
        batch_size=batch,
        micro_batch=micro,
        sequences_per_device=shard.batch,
        input_seq_len=seq,
        output_seq_len=out,
        overlap_policy=overlap_policy,
        comm_overlap=comm_overlap,
        routing=routing if model.has_moe else None,
        include_lm_head=include_lm_head,
        compute_frequency_hz=compute_hz,
        ttft_seconds=prefill.latency_seconds + first_token.latency_seconds,
        tps=micro * out / decode_seconds,
        tps_per_sequence=out / (plan.pp * decode_seconds),
        prefill=prefill,
        first_token_decode=first_token,
        decode_seconds=decode_seconds,
        weight_bytes_per_device=weight_bytes,
        kv_cache_bytes_per_device=kv_bytes,
        memory_capacity_bytes=capacity,
        fits_in_memory=fits,
        moe=moe_report,
        noc_energy_pj_per_decode_token=noc_energy,
        warnings=tuple(warnings),
    )
