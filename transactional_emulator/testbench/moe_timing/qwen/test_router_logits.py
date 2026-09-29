"""Guards for the router-logit reconstruction that puts V_TOPK in the timing path.

The captured traces record ``topk_indices`` and ``topk_weights`` but not the
router logits they came from -- ``generate_true_routing_with_weights`` takes the
top-k and throws the logits away. Replaying the router on device therefore needs
logits rebuilt from what survived, and the rebuild is only worth anything if the
device's selection provably lands back on the trace's experts.

These tests are that proof, on the reconstruction alone. The device half is the
functional gate in ``qwen3_trace_replay``.
"""

from __future__ import annotations

import math

import pytest
import torch

from transactional_emulator.testbench.moe_timing.qwen.router_logits import (
    RouterLogitReconstructionError,
    reconstruct_router_logits,
)


def _topk(logits: torch.Tensor, top_k: int) -> tuple[list[list[int]], torch.Tensor]:
    values, indices = torch.topk(logits.float(), k=top_k, dim=-1)
    return indices.tolist(), torch.softmax(values, dim=-1)


def test_reconstruction_reproduces_the_traces_expert_ids() -> None:
    """The whole point: torch.topk on the rebuilt row returns the trace's experts.

    Fails if the reconstruction orders the selected logits by anything other than
    descending weight, or lets an unselected expert reach the top-k.
    """
    indices = [[5, 12, 3], [7, 5, 1]]
    weights = [[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]]

    logits = reconstruct_router_logits(indices, weights, num_experts=16)

    assert _topk(logits, top_k=3)[0] == indices


def test_reconstruction_reproduces_the_traces_weights() -> None:
    """Softmax over the selected logits returns the recorded weights.

    The device computes softmax itself, so a reconstruction that got the ordering
    right but the spacing wrong would still report the wrong route weights.
    """
    indices = [[5, 12, 3], [7, 5, 1]]
    weights = [[0.6, 0.3, 0.1], [0.5, 0.4, 0.1]]

    logits = reconstruct_router_logits(indices, weights, num_experts=16)

    got = _topk(logits, top_k=3)[1]
    assert torch.allclose(got, torch.tensor(weights), rtol=0.02, atol=0.002)


def test_every_unselected_expert_stays_below_every_selected_one() -> None:
    """Not just below the top-k -- below by a margin bf16 cannot erase.

    Fails if the floor is computed per-token from a rounded value, which can put
    an unselected expert exactly level with the weakest selected one.
    """
    indices = [[0, 1]]
    weights = [[0.5001, 0.4999]]

    logits = reconstruct_router_logits(indices, weights, num_experts=64)

    row = logits[0].float()
    selected = row[torch.tensor(indices[0])]
    unselected = row[torch.tensor([e for e in range(64) if e not in indices[0]])]
    assert unselected.max().item() < selected.min().item()


def test_output_is_positive_so_zero_padding_can_never_be_selected() -> None:
    """VRAM is preloaded with zeros, and softmax is shift-invariant.

    Raw log-weights are all <= 0, so a row that got laid out one column short
    would have a zero pad outrank every real expert -- silently, with plausible
    numbers. Shifting the whole row positive makes that mistake unselectable.
    Guards the shift, which is otherwise invisible: nothing else observes it.
    """
    indices = [[3, 9]]
    weights = [[0.9, 0.1]]

    logits = reconstruct_router_logits(indices, weights, num_experts=32)

    assert logits.float().min().item() > 0.0


def test_uniform_weights_still_yield_a_strict_order() -> None:
    """`build_route_traces` can substitute uniform weights for timing-only runs.

    log(w) is then identical across the selected experts. V_TOPK breaks ties by
    low index, so without a strict ladder the device would return the experts in
    ascending id order and disagree with the trace wherever it is not already
    sorted -- here, [9, 2, 5] would come back [2, 5, 9].
    """
    indices = [[9, 2, 5]]
    weights = [[1 / 3, 1 / 3, 1 / 3]]

    logits = reconstruct_router_logits(indices, weights, num_experts=16)

    assert _topk(logits, top_k=3)[0] == indices


def test_a_trace_whose_order_contradicts_its_weights_is_refused() -> None:
    """Recorded order and recorded weights have to agree, or one of them is wrong.

    Expert 4 is listed first but carries the smallest weight. Reconstructing in
    the recorded order needs logits whose softmax is nowhere near the recorded
    weights; reconstructing by weight returns experts in a different order than
    the trace. Either way the functional gate downstream would blame the device
    for a malformed trace, so this refuses instead -- naming the token, because a
    trace holds thousands of rows and the operator has to find the one that
    failed.
    """
    indices = [[0, 1, 2], [4, 5, 6]]
    weights = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]

    with pytest.raises(RouterLogitReconstructionError, match="token 1"):
        reconstruct_router_logits(indices, weights, num_experts=16)


def test_the_row_is_bf16_because_that_is_what_v_topk_reads() -> None:
    """V_TOPK reads BF16 rows from VRAM. Verifying in f32 would prove nothing.

    Fails if the reconstruction returns f32 and leaves the narrowing to the
    caller -- which is where the near-tie this module rejects would reappear,
    past the point that can refuse it.
    """
    logits = reconstruct_router_logits([[1, 0]], [[0.7, 0.3]], num_experts=8)

    assert logits.dtype == torch.bfloat16


def test_rejects_rows_whose_weights_do_not_match_the_index_count() -> None:
    """A malformed trace should not become a silently shorter top-k."""
    with pytest.raises(ValueError, match="top_k"):
        reconstruct_router_logits([[1, 2, 3]], [[0.5, 0.5]], num_experts=8)


def test_rejects_expert_ids_outside_the_router_width() -> None:
    """An id >= num_experts would index past the row V_TOPK scans."""
    with pytest.raises(ValueError, match="expert id"):
        reconstruct_router_logits([[0, 8]], [[0.5, 0.5]], num_experts=8)


def test_rejects_non_positive_weights() -> None:
    """log(0) is -inf and log(-w) is NaN; both would poison the whole row."""
    with pytest.raises(ValueError, match="positive"):
        reconstruct_router_logits([[0, 1]], [[1.0, 0.0]], num_experts=8)


def test_a_realistic_qwen3_row_survives_the_round_trip() -> None:
    """128 experts, top-8, softmax weights with the spread a real router gives.

    The unit cases above are small enough to pass on a reconstruction that only
    works when the weights are far apart. This is the production shape.
    """
    raw = [4.1, 3.6, 3.2, 2.9, 2.5, 2.2, 1.8, 1.5]
    denom = sum(math.exp(v) for v in raw)
    weights = [[math.exp(v) / denom for v in raw]]
    indices = [[117, 3, 64, 12, 99, 41, 8, 126]]

    logits = reconstruct_router_logits(indices, weights, num_experts=128)

    got_indices, got_weights = _topk(logits, top_k=8)
    assert got_indices == indices
    assert torch.allclose(got_weights, torch.tensor(weights), rtol=0.05, atol=0.002)
