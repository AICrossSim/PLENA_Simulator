"""Rebuild router logits from a route trace's recorded top-k.

``generate_true_routing_with_weights`` runs the real router, takes ``topk`` and
``softmax`` of the result, and writes only those two. The logits themselves are
never persisted -- and regenerating them means the 30B checkpoint and a forward
pass per sample, which is exactly what replaying a trace exists to avoid.

Replaying the router on device needs a logit row anyway, so this rebuilds one
that is *equivalent for the router's purpose*: the largest ``top_k`` entries are
the trace's experts, in the trace's order, and the softmax over them is the
trace's weights. What it does not rebuild is the tail -- the 120 unselected
experts get a floor rather than their true values, so the tail's spread is not
exercised. Selection and weighting are, which is what V_TOPK decides.

The row must survive BF16, because that is the width V_TOPK reads from VRAM. A
reconstruction that only holds in f32 would put the narrowing in the caller,
past the point that can refuse it, and a near-tie flattened there would surface
as the device picking the wrong expert.
"""

from __future__ import annotations

import math

import torch

__all__ = ["RouterLogitReconstructionError", "reconstruct_router_logits"]


class RouterLogitReconstructionError(ValueError):
    """A trace's recorded top-k cannot be expressed as a BF16 logit row."""


# Softmax is shift-invariant, so the absolute placement of the row is free. It is
# spent on two things.
#
# Positivity: VRAM is preloaded with zeros and raw log-weights are all <= 0, so a
# row laid out one column short would have a zero pad outrank every real expert.
# Silently -- the numbers stay plausible. Keeping the whole row above zero makes
# that mistake unselectable instead of invisible.
#
# Smallness: BF16 stores 7 mantissa bits (8 bits of significand counting the
# implicit one), so its resolution is relative -- one ulp in the binade
# [2^k, 2^(k+1)) is 2^-7 * 2^k. Placing the row just above 1.0 keeps the selected
# logits in [2, 4), where that is 2^-6, a couple of percent of a weight after
# exp(). A larger offset would buy nothing and cost precision proportionally.
_FLOOR_VALUE = 1.0
_FLOOR_MARGIN = 1.0

#: Recorded weights are a softmax over the selected experts, so they sum to 1.
#:
#: The reconstruction depends on it: softmax over the rebuilt logits renormalises,
#: so weights summing to anything else come back scaled by 1/sum and no logit row
#: can reproduce them. Checked explicitly, because the symptom otherwise surfaces
#: in `_verify` as "the recorded order and the recorded weights disagree" -- which
#: is a true statement about the arithmetic and a wrong diagnosis of the trace.
_WEIGHT_SUM_ATOL = 1e-3

# How far a reconstructed weight may sit from the recorded one. This is BF16's
# resolution, not a fudge factor: one ulp at the top of the selected range is
# ~2^-6, and exp(2^-6) - 1 is ~1.6%, doubled to leave room for the ladder below.
_WEIGHT_RTOL = 0.05
_WEIGHT_ATOL = 2e-3


def _bf16_step_down(values: torch.Tensor) -> torch.Tensor:
    """The largest BF16 strictly below each (positive) input.

    Done on the bit pattern rather than by subtracting a constant: the gap
    between neighbouring BF16 values is relative, so any fixed epsilon is either
    too small to register at the top of the row or far larger than needed at the
    bottom.
    """
    bits = values.to(torch.bfloat16).view(torch.int16)
    return (bits - 1).view(torch.bfloat16)


def _validate(
    topk_indices: list[list[int]],
    topk_weights: list[list[float]],
    num_experts: int,
) -> int:
    if len(topk_indices) != len(topk_weights):
        raise ValueError(f"topk_indices has {len(topk_indices)} rows but topk_weights has {len(topk_weights)}")
    if not topk_indices:
        raise ValueError("no routing rows to reconstruct")
    top_k = len(topk_indices[0])
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if top_k > num_experts:
        raise ValueError(f"top_k={top_k} exceeds num_experts={num_experts}")
    for token, (indices, weights) in enumerate(zip(topk_indices, topk_weights, strict=True)):
        if len(indices) != top_k or len(weights) != top_k:
            raise ValueError(
                f"token {token}: ragged routing row -- top_k is {top_k} but this row "
                f"has {len(indices)} indices and {len(weights)} weights"
            )
        if len(set(indices)) != top_k:
            raise ValueError(f"token {token}: repeated expert id in {indices}")
        for expert_id in indices:
            if not 0 <= expert_id < num_experts:
                raise ValueError(f"token {token}: expert id {expert_id} outside [0, {num_experts})")
        for weight in weights:
            if not weight > 0.0:
                raise ValueError(f"token {token}: route weights must be positive, got {weight!r}")
        total = math.fsum(weights)
        if abs(total - 1.0) > _WEIGHT_SUM_ATOL:
            raise RouterLogitReconstructionError(
                f"token {token}: route weights sum to {total!r}, not 1. This replay needs "
                "softmax-normalised top-k weights; weights taken from a softmax over all "
                "experts without renormalising (norm_topk_prob=false) cannot be expressed "
                "as a logit row, because softmax over the selected experts renormalises "
                "them back to 1"
            )
    return top_k


def reconstruct_router_logits(
    topk_indices: list[list[int]],
    topk_weights: list[list[float]],
    num_experts: int,
) -> torch.Tensor:
    """Build a ``(tokens, num_experts)`` BF16 row set reproducing the trace's top-k.

    Raises :class:`RouterLogitReconstructionError` when the recorded order and
    the recorded weights cannot both hold in BF16 -- rather than emitting a row
    that would make the device's correct answer look like a fault.
    """
    top_k = _validate(topk_indices, topk_weights, num_experts)
    tokens = len(topk_indices)

    logits = torch.empty(tokens, num_experts, dtype=torch.bfloat16)
    for token, (indices, weights) in enumerate(zip(topk_indices, topk_weights, strict=True)):
        # log() inverts softmax up to an additive constant, which the shift below
        # supplies. Weights that already differ enough land here in descending
        # order and the ladder is a no-op.
        raw = [math.log(weight) for weight in weights]
        shift = _FLOOR_VALUE + _FLOOR_MARGIN - min(raw)
        selected = torch.tensor([value + shift for value in raw], dtype=torch.bfloat16)

        # A strict descending ladder in BF16, not in f32: V_TOPK breaks ties by
        # low index, so two experts that round to the same BF16 value come back
        # in ascending id order regardless of what the trace recorded. Uniform
        # weights -- which `build_route_traces` substitutes for timing-only runs
        # -- are entirely ties, and every trace has near-ties in the tail of its
        # top-k. Each step is one ulp, the smallest correction that survives.
        for position in range(1, top_k):
            if selected[position] >= selected[position - 1]:
                selected[position] = _bf16_step_down(selected[position - 1])

        if float(selected.float().min()) <= _FLOOR_VALUE:
            # Only reachable if the ladder walked a selected logit down onto the
            # floor, which means the row needed more BF16 range than it has.
            raise RouterLogitReconstructionError(
                f"token {token}: the selected logits reached the unselected floor; "
                f"{top_k} experts cannot be separated in BF16 for weights {weights}"
            )

        row = torch.full((num_experts,), _FLOOR_VALUE, dtype=torch.bfloat16)
        row[torch.tensor(indices)] = selected
        logits[token] = row

    _verify(logits, topk_indices, topk_weights, top_k)
    return logits


def _verify(
    logits: torch.Tensor,
    topk_indices: list[list[int]],
    topk_weights: list[list[float]],
    top_k: int,
) -> None:
    """Confirm the finished BF16 rows really do reproduce the trace.

    Checked here rather than trusted from the construction above, because every
    step of it -- the log, the shift, the ladder, the narrowing to BF16 -- can
    individually be correct while the composition is not. The downstream gate
    reads *device* output against the trace, so if this were wrong the two would
    disagree and the device would take the blame.
    """
    values, indices = torch.topk(logits.float(), k=top_k, dim=-1)
    got_weights = torch.softmax(values, dim=-1)
    want_indices = torch.tensor(topk_indices, dtype=indices.dtype)
    want_weights = torch.tensor(topk_weights, dtype=torch.float32)

    # Compared for the whole block, then narrowed to the first offending token for
    # the message. A per-token loop launched one comparison per row, which on a
    # four-thousand-token trace is four thousand kernels to answer a question one
    # of them can answer.
    bad_order = (indices != want_indices).any(dim=-1)
    bad_weight = ~torch.isclose(got_weights, want_weights, rtol=_WEIGHT_RTOL, atol=_WEIGHT_ATOL).all(dim=-1)

    if bool(bad_order.any()):
        token = int(bad_order.nonzero()[0])
        raise RouterLogitReconstructionError(
            f"token {token}: reconstructed logits select {indices[token].tolist()}, "
            f"but the trace recorded {topk_indices[token]}"
        )
    if bool(bad_weight.any()):
        token = int(bad_weight.nonzero()[0])
        raise RouterLogitReconstructionError(
            f"token {token}: reconstructed weights {got_weights[token].tolist()} "
            f"differ from the trace's {topk_weights[token]} by more than BF16 "
            f"resolution (rtol={_WEIGHT_RTOL}); the recorded order and the "
            f"recorded weights disagree"
        )
