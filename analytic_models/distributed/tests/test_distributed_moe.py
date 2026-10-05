"""Expert parallelism: plan mapping, routing and the busiest rank's work."""

from __future__ import annotations

import numpy as np
import pytest
from distributed_fixtures import FICTIONAL_MOE_128, UNBOUNDED, gpt_oss, perf_model, precision, switch_noc

from analytic_models.distributed import ModelSpec, ParallelPlan, estimate_distributed, expert_load, routing_rows
from analytic_models.distributed.moe import TRACE_COMPUTE_TOKEN_LIMIT, moe_cycles


@pytest.fixture(scope="module")
def perf():
    return perf_model()


@pytest.fixture(scope="module")
def prec():
    return precision()


def test_replace_mode_folds_tp_into_ep() -> None:
    scheme = ParallelPlan(tp=2, ep=4, dp=2).moe_scheme(32)
    assert (scheme.tp, scheme.ep, scheme.dp) == (1, 8, 2)
    # More ranks than experts: the remainder stays tensor parallel.
    scheme = ParallelPlan(tp=8, ep=2).moe_scheme(4)
    assert (scheme.tp, scheme.ep) == (4, 4)
    with pytest.raises(ValueError, match="multiple"):
        ParallelPlan(tp=2, ep=3).moe_scheme(4)
    kept = ParallelPlan(tp=2, ep=4, moe_tp_mode="keep").moe_scheme(32)
    assert (kept.tp, kept.ep) == (2, 4)
    dense = ParallelPlan(tp=2, ep=4, dp=2).dense_scheme()
    assert (dense.tp, dense.ep, dense.dp) == (2, 1, 8)


def test_balanced_rows_spread_each_token_over_ranks() -> None:
    rows = routing_rows("balanced", phase="decode", tokens=64, num_experts=32, top_k=4)
    assert rows.shape == (64, 4)
    assert np.bincount(rows.ravel(), minlength=32).tolist() == [8] * 32
    ranks = rows * 8 // 32  # DeepStack's expert-to-rank map with ep=8
    assert all(len(set(token)) == 4 for token in ranks.tolist())


def test_random_rows_are_seeded_and_distinct() -> None:
    first = routing_rows("random", phase="prefill", tokens=100, num_experts=32, top_k=4)
    again = routing_rows("random", phase="prefill", tokens=100, num_experts=32, top_k=4)
    assert np.array_equal(first, again)
    assert all(len(set(token)) == 4 for token in first.tolist())


def test_balanced_load_is_the_even_share() -> None:
    scheme = ParallelPlan(ep=8).moe_scheme(32)
    load = expert_load(
        "balanced", phase="decode", scheme=scheme, micro_batch_tokens=64, group_tokens=64, num_experts=32, top_k=4
    )
    assert (load.tokens_local, load.pairs, load.imbalance) == (8, 32, 1.0)
    assert load.expert_counts == (8, 8, 8, 8)


@pytest.mark.parametrize("routing", ["random", "qwen3_235b"])
def test_routed_load_counts_the_busiest_rank(routing) -> None:
    scheme = ParallelPlan(ep=8).moe_scheme(128)
    kwargs = {"phase": "decode", "scheme": scheme, "num_experts": 128, "top_k": 8}
    small = expert_load(routing, micro_batch_tokens=64, group_tokens=64, **kwargs)
    assert small.source == routing
    assert small.pairs == sum(small.expert_counts)
    assert small.imbalance >= 1.0
    assert len(small.expert_counts) == 16

    tokens = 2 * TRACE_COMPUTE_TOKEN_LIMIT
    large = expert_load(routing, micro_batch_tokens=tokens, group_tokens=tokens, **kwargs)
    assert large.source == f"{routing} (DeepStack estimate)"
    assert large.pairs == sum(large.expert_counts)
    assert large.imbalance >= 1.0


def test_routing_traces_must_match_the_expert_layout(perf, prec) -> None:
    with pytest.raises(ValueError, match="routing trace 'qwen3_235b' has 128 experts"):
        estimate_distributed(
            gpt_oss(), ParallelPlan(ep=8), switch_noc(8), perf, prec, UNBOUNDED,
            batch_size=8, input_seq_len=64, output_seq_len=1, routing="qwen3_235b",
        )  # fmt: skip
    model = ModelSpec.from_hf_config(FICTIONAL_MOE_128, name="fictional-moe-128")
    result = estimate_distributed(
        model, ParallelPlan(ep=8), switch_noc(8), perf, prec, UNBOUNDED,
        batch_size=8, input_seq_len=64, output_seq_len=1, routing="qwen3_235b",
    )  # fmt: skip
    assert result.moe["decode"]["source"] == "qwen3_235b"
    assert result.first_token_decode.stages[0].comm_seconds["ep_dispatch"] > 0


@pytest.mark.parametrize("mode", ["prefill", "decode"])
def test_one_rank_balanced_moe_is_perf_models_mlp_moe(perf, mode) -> None:
    tokens = 300 if mode == "prefill" else 7
    scheme = ParallelPlan().moe_scheme(32)
    load = expert_load(
        "balanced", phase=mode, scheme=scheme, micro_batch_tokens=tokens, group_tokens=tokens, num_experts=32, top_k=4
    )
    cycles = moe_cycles(perf, hidden_size=2880, intermediate_size=2880, num_experts=32, top_k=4, load=load, mode=mode)
    seq, batch = (tokens, 1) if mode == "prefill" else (1, tokens)
    assert cycles == perf.mlp_moe(2880, seq, batch, 32, 4, 2880, mode)


def test_expert_parallelism_splits_the_experts(perf, prec) -> None:
    model = gpt_oss()
    kwargs = {"batch_size": 16, "input_seq_len": 256, "output_seq_len": 2}
    single = estimate_distributed(model, ParallelPlan(), switch_noc(1), perf, prec, UNBOUNDED, **kwargs)
    split = estimate_distributed(model, ParallelPlan(ep=8), switch_noc(8), perf, prec, UNBOUNDED, **kwargs)
    assert split.moe["experts_per_rank"] == 4
    assert split.sequences_per_device == 2
    assert split.weight_bytes_per_device < single.weight_bytes_per_device / 2
    assert split.prefill.stages[0].compute_seconds < single.prefill.stages[0].compute_seconds / 4
    assert split.tps > single.tps


def test_keep_mode_all_reduces_tensor_parallel_experts(perf, prec) -> None:
    model = gpt_oss()
    kwargs = {"batch_size": 16, "input_seq_len": 256, "output_seq_len": 2}
    kept = estimate_distributed(
        model, ParallelPlan(tp=2, ep=4, moe_tp_mode="keep"), switch_noc(8), perf, prec, UNBOUNDED, **kwargs
    )
    replaced = estimate_distributed(model, ParallelPlan(tp=2, ep=4), switch_noc(8), perf, prec, UNBOUNDED, **kwargs)
    assert (kept.moe["tp"], kept.moe["ep"]) == (2, 4)
    assert (replaced.moe["tp"], replaced.moe["ep"]) == (1, 8)
    assert kept.first_token_decode.stages[0].comm_seconds["moe_tp_all_reduce"] > 0
    assert "moe_tp_all_reduce" not in replaced.first_token_decode.stages[0].comm_seconds
