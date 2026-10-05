"""Routed-expert load and compute of one MoE layer on the busiest expert-parallel rank.

Routing modes:

* ``"balanced"``: PLENA's assumption in ``PerfModel.mlp_moe``: every expert
  serves ``ceil(tokens * top_k / num_experts)`` tokens and every rank gets the
  same share. With one rank this reproduces ``mlp_moe`` exactly. For the token
  dispatch, the ``j``-th choice of token ``t`` is expert
  ``(t + j * (num_experts // top_k)) % num_experts``, so a token's experts sit
  on different EP ranks whenever ``ep >= top_k``.
* ``"random"``: experts drawn uniformly at random per token (fixed seed).
* ``"qwen3_235b"``, ``"deepseek_v3"``: routing traces packaged with DeepStack
  (``mosaic.data.routing``); the decode trace prices decode and the prefill
  trace prices prefill.

For ``"random"`` and the traces, the busiest rank's per-expert token counts
come from DeepStack's ``count_expert_frequency_flatten_wrapper`` while the
micro-batch has fewer than 4096 tokens, and from DeepStack's closed-form
imbalance estimate (``estimate_moe_routing_imbalance_overhead`` with
``estimate_experts_activated``) above it, the same switch DeepStack makes in
``mosaic/llm/moe/moe_coarse.py`` (arXiv:2604.04750, tile-ai/DeepStack).
Like DeepStack, every MoE layer uses the same routed tokens (the first rows of
the flattened trace).

The compute cost reuses ``PerfModel.mlp_moe``'s instruction counts for the
tokens and experts of one rank.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..performance.perf_model import PerfModel
from ._deepstack import (
    ROUTING_TRACES,
    ParallelScheme,
    count_expert_frequency_flatten_wrapper,
    estimate_experts_activated,
    estimate_moe_routing_imbalance_overhead,
    load_routing_trace,
)

ROUTING_MODES = ("balanced", "random", *ROUTING_TRACES)
# DeepStack switches from routing statistics to its closed-form estimate at
# these micro-batch token counts (moe_coarse.py and ep_all_to_all_wrapper.py).
TRACE_COMPUTE_TOKEN_LIMIT = 4096
TRACE_COMM_TOKEN_LIMIT = 8192
_RANDOM_SEED = 0


@dataclass(frozen=True)
class ExpertLoad:
    """Routed work of one MoE layer on the busiest expert-parallel rank."""

    tokens_local: int
    pairs: int
    expert_counts: tuple[int, ...]
    imbalance: float
    source: str

    @property
    def active_experts(self) -> int:
        return sum(1 for count in self.expert_counts if count > 0)

    def describe(self) -> dict[str, Any]:
        return {
            "tokens_per_rank": self.tokens_local,
            "token_expert_pairs_on_busiest_rank": self.pairs,
            "active_experts_on_busiest_rank": self.active_experts,
            "imbalance": self.imbalance,
            "source": self.source,
        }


@functools.cache
def _load_trace(mode: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(array) for array in load_routing_trace(mode))
    for array in arrays:
        array.setflags(write=False)
    return arrays


def _trace_rows(mode: str, phase: str, num_experts: int, top_k: int) -> np.ndarray:
    prefill, decode = _load_trace(mode)
    trace = decode if phase == "decode" else prefill
    trace_experts, trace_top_k = int(trace.max()) + 1, trace.shape[-1]
    if trace_experts != num_experts or trace_top_k < top_k:
        raise ValueError(
            f"routing trace {mode!r} has {trace_experts} experts and top-{trace_top_k}; "
            f"the model has {num_experts} experts and top-{top_k}"
        )
    return trace.reshape(-1, trace_top_k)


def check_routing(mode: str, *, num_experts: int, top_k: int) -> None:
    """Reject an unknown mode or a routing trace recorded for another expert layout."""

    if mode not in ROUTING_MODES:
        raise ValueError(f"unknown routing mode {mode!r}; expected one of {', '.join(ROUTING_MODES)}")
    if mode in ROUTING_TRACES:
        for phase in ("prefill", "decode"):
            _trace_rows(mode, phase, num_experts, top_k)


def routing_rows(mode: str, *, phase: str, tokens: int, num_experts: int, top_k: int) -> np.ndarray:
    """Expert ids, shape ``(tokens, top_k)``, for DeepStack's per-token statistics."""

    if mode == "balanced":
        stride = num_experts // top_k
        choices = np.arange(tokens, dtype=np.int64)[:, None] + stride * np.arange(top_k, dtype=np.int64)[None, :]
        return choices % num_experts
    if mode == "random":
        scores = np.random.default_rng(_RANDOM_SEED).random((tokens, num_experts))
        return np.argsort(scores, axis=1)[:, :top_k].astype(np.int64)
    check_routing(mode, num_experts=num_experts, top_k=top_k)
    rows = _trace_rows(mode, phase, num_experts, top_k)
    if rows.shape[0] < tokens:
        raise ValueError(f"routing trace {mode!r} has {rows.shape[0]} {phase} rows, {tokens} needed")
    return rows[:tokens, :top_k]


def expert_load(
    mode: str,
    *,
    phase: str,
    scheme: ParallelScheme,
    micro_batch_tokens: int,
    group_tokens: int,
    num_experts: int,
    top_k: int,
) -> ExpertLoad:
    """Busiest-rank load for ``micro_batch_tokens`` tokens split into EP groups of ``group_tokens``."""

    ep = scheme.ep
    local_experts = num_experts // ep
    tokens_local = math.ceil(group_tokens / ep)
    balanced_pairs = math.ceil(group_tokens * top_k / ep)
    if mode == "balanced":
        per_expert = math.ceil(group_tokens * top_k / num_experts)
        return ExpertLoad(tokens_local, balanced_pairs, (per_expert,) * local_experts, 1.0, "balanced")
    if micro_batch_tokens >= TRACE_COMPUTE_TOKEN_LIMIT:
        imbalance = float(
            estimate_moe_routing_imbalance_overhead(
                num_experts, top_k, ep, group_tokens, R=scheme.tp * scheme.sp * scheme.dp
            )
        )
        pairs = round(tokens_local * top_k * imbalance)
        few_experts, few_tokens, more_experts, more_tokens = estimate_experts_activated(pairs, local_experts)
        counts = (few_tokens,) * few_experts + (more_tokens,) * more_experts
        return ExpertLoad(tokens_local, pairs, counts, pairs / balanced_pairs, f"{mode} (DeepStack estimate)")
    rows = routing_rows(mode, phase=phase, tokens=micro_batch_tokens, num_experts=num_experts, top_k=top_k)
    counts = tuple(
        int(count)
        for count in count_expert_frequency_flatten_wrapper(
            flatten_numpy=rows, group_tokens=group_tokens, total_experts=num_experts, EP=ep
        )
    )
    pairs = sum(counts)
    return ExpertLoad(tokens_local, pairs, counts, pairs / (group_tokens * top_k / ep), mode)


def moe_cycles(
    perf: PerfModel,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    load: ExpertLoad,
    mode: str,
) -> float:
    """``PerfModel.mlp_moe`` for the tokens and experts of one rank.

    Normalisation, router, top-k, softmax and the weighted sum run for the
    rank's own tokens; the expert MLPs run for the token-expert pairs routed to
    it (decode: one matrix-vector pass per pair; prefill: batched per expert).
    """

    instr = perf.instr
    mlen, blen, vlen = perf.mlen, perf.blen, perf.vlen
    tokens = load.tokens_local
    cycles = math.ceil(hidden_size / vlen) * instr["V_BASIC"] * 4 * tokens
    if mode == "prefill":
        cycles += (
            (4 + math.ceil(hidden_size / mlen) * instr["M_MM"] + instr["H_PREFETCH_M"])
            * math.ceil(tokens / blen)
            * math.ceil(num_experts / blen)
        )
    else:
        cycles += (
            tokens
            * (4 + math.ceil(hidden_size / mlen) * instr["M_MV"] + instr["H_PREFETCH_M"])
            * math.ceil(num_experts / blen)
        )
    cycles += (4 + math.ceil(num_experts / vlen) * instr["V_TOPK"]) * tokens
    cycles += tokens * math.ceil(top_k / vlen) * (instr["V_EXP_V"] + instr["V_RED_MAX"] + instr["V_BASIC"])

    if mode == "prefill":
        up = 4 + math.ceil(hidden_size / mlen) * instr["M_MM"] + instr["H_PREFETCH_M"]
        down = 4 + math.ceil(intermediate_size / mlen) * instr["M_MM"] + instr["H_PREFETCH_M"]
        for count in load.expert_counts:
            if count:
                cycles += up * math.ceil(count / blen) * math.ceil(2 * intermediate_size / blen)
                cycles += down * math.ceil(count / blen) * math.ceil(hidden_size / blen)
    else:
        cycles += (
            load.pairs
            * (4 + math.ceil(hidden_size / mlen) * instr["M_MV"] + instr["H_PREFETCH_M"])
            * math.ceil(2 * intermediate_size / blen)
        )
        cycles += (
            load.pairs
            * (4 + math.ceil(intermediate_size / mlen) * instr["M_MV"] + instr["H_PREFETCH_M"])
            * math.ceil(hidden_size / blen)
        )
    cycles += load.pairs * math.ceil(intermediate_size / vlen) * 6 * instr["V_BASIC"]
    cycles += tokens * top_k * math.ceil(hidden_size / vlen) * (instr["V_MUL_VV"] + instr["V_ADD_VV"])
    return cycles
