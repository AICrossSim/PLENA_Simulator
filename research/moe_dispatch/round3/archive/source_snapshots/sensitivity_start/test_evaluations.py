"""消融账本、首块及时率和核空转交集的必要回归。"""
from dataclasses import replace

from research.moe_dispatch.round3.evaluations import interval_union, intersect_length, prediction_samples, old_designs, configured_params, flow_metrics


def test_union_does_not_duplicate_overlapping_waits():
    assert interval_union([(4, 8), (1, 5), (10, 12), (12, 15), (3, 3)]) == [(1, 8), (10, 15)]
    assert intersect_length((1, 8), (4, 10)) == 4


def test_wait_is_counted_only_when_core_has_no_run_actor():
    d = old_designs("pipelined")["B1"]
    p = configured_params("pipelined")
    w = {"experts": [{"Me": 2, "H": 2048, "F": 1408}]}
    r = {"tasks": [{"expert_index": 0, "core": 0, "actual_cycles": 50.0,
                     "predicted_cycles": 45.0, "first_weight_ready": 90.0, "current_end": 20.0}],
         "segments": [{"start": 0, "end": 30, "active_run": [True]},
                      {"start": 30, "end": 70, "active_run": [False]},
                      {"start": 70, "end": 100, "active_run": [True]}]}
    samples, stall = prediction_samples(r, w, d, p)
    assert stall == 40.0  # 70-cy late interval includes 30 cycles of real work.
    assert samples[0]["late_gt_64"] is True
    assert samples[0]["late_gt_256"] is False
    assert samples[0]["error_fraction"] == .1
    assert not samples[0]["success_at_2"]


def test_conditional_oracle_is_a_real_physical_second_pass():
    from research.moe_dispatch.round3.runtime import simulate
    from research.moe_dispatch.round3.oracle_replay import same_schedule_oracle
    from research.moe_dispatch.round2.common import inputs, canonical
    d = old_designs("pipelined")["B1"]
    p = configured_params("pipelined")
    plan = simulate(inputs()["heldout"][0], d, p, dispatch="fixed")
    before = canonical(plan)
    oracle = same_schedule_oracle(plan, d, p)
    assert oracle["oracle_replay"]["physically_replayed"]
    assert abs(oracle["cycles"] - plan["cycles"]) < 1e-5
    assert abs(oracle["oracle_replay"]["hbm_replay_bytes"] - plan["hbm_bytes"]) < 1
    assert oracle["oracle_replay"]["mae_pct"] < 1e-8
    assert canonical(plan) == before


def test_h2_preserves_costs_and_total_storage():
    from research.moe_dispatch.round3.ablation import configurations, all_costs_compatible
    specs, meta = configurations()
    assert meta["H2_feasible"]
    h0, h2 = specs["H0"][1], specs["H2"][1]
    assert h0.ledger()["installed_storage_B"] == h2.ledger()["installed_storage_B"]
    assert sum(h2.w_bytes) == sum(h0.w_bytes) + 16 * 1024
    assert all_costs_compatible(h0, h2, configured_params("pipelined"))


def test_global_storage_chunks_use_local_token_count_in_diagnostics():
    from research.moe_dispatch.round3.runtime import simulate
    from research.moe_dispatch.round3.evaluations import summarize_result
    d = old_designs("pipelined")["B1"]
    p = configured_params("pipelined")
    w = {"id": "test_chunked_shared", "batch": 256, "hidden": 2048, "top_k": 1,
         "experts": [{"id": -1, "Me": 256, "H": 2048, "F": 1408,
                      "is_shared": True, "token_indices": list(range(256))}]}
    result = simulate(w, d, p, dispatch="fixed")
    summary = summarize_result(w, d, p, result)
    assert result["storage_chunks"] == 2
    # Each global chunk has 128 rows and fits Z. The original 256-row
    # expert would wrongly report a 2x refetch for both local tasks.
    assert summary["refetch_tasks"] == 0
    assert summary["refetch_details"] == []
    samples, _ = prediction_samples(result, w, d, p)
    assert [(t["chunk_index"], t["original_expert_index"]) for t in samples] == [(0, 0), (1, 0)]


def test_shared_rate_uses_expert_id_and_only_big_core_interval():
    from research.moe_dispatch.round3.shared_rate_diagnostic import shared_service
    d = old_designs("pipelined")["best_hetero"]
    w = {"experts": [{"id": -1, "is_shared": True}]}
    r = {"cycles": 100, "core_finish_cycles": [80, 100], "segments": [
        {"start": 0, "end": 40, "hbm_rate_Bpc": 100, "hbm_rate_Bpc_core": [100, 0], "inflight_bytes": [10, 0],
         "active_run": [True, False], "run_expert": [-1, None]},
        {"start": 40, "end": 100, "hbm_rate_Bpc": 200, "hbm_rate_Bpc_core": [150, 50], "inflight_bytes": [10, 20],
         "active_run": [True, True], "run_expert": [3, -1]}]}
    m = flow_metrics(r, w, d)
    assert m["single_fetcher_time_pct"] == 40
    assert m["single_fetcher_GBps"] == 100
    assert m["shared_cycles"] == 60
    assert m["shared_fetch_GBps"] == 200
    own = shared_service(r, w, d)
    assert own["global_hbm_GBps"] == 200
    assert own["big_core_hbm_GBps"] == 50
    assert own["shared_interval_cycles"] == 60
