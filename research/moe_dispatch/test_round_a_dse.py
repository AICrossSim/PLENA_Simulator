from dataclasses import replace
import heapq

import round_a_dse as dse


def explicit_projection(m, n, k, core, model):
    """Independent issue-event implementation of the closed-form bound."""
    time = 0
    for first in range(0, dse.ceil(n, core.pn), model.n_group_tiles):
        count = min(model.n_group_tiles, dse.ceil(n, core.pn) - first)
        q = count * dse.ceil(m, core.pm)
        previous = [0] * q
        for _ in range(dse.ceil(k, core.pk)):
            for record in range(q):
                time = max(time, previous[record])
                previous[record] = time + model.completion_latency
                time += model.initiation_interval
        time = max(time, max(previous))
    return time


def test_exact_budget_and_symmetric_mirror_deduplication():
    geometries = dse.enumerate_geometries()
    assert len(geometries) == 96
    assert len(set(map(dse.geometry_id, geometries))) == 96
    assert all(sum(c.macs for c in g) == 12288 for g in geometries)
    assert sum(dse.family(g) == "single" for g in geometries) == 6
    assert sum(dse.family(g) == "homogeneous" for g in geometries) == 5


def test_dependency_closed_form_against_independent_event_simulator():
    for core in [dse.Core(1, 24), dse.Core(3, 4), dse.Core(4, 4)]:
        for m in [1, 2, 4, 7, 32, 96]:
            for n in [1, 13, 128]:
                for k in [1, 512, 513, 2048]:
                    for model in [dse.Model(), dse.Model(n_group_tiles=4),
                                  dse.Model(dot_latency=1)]:
                        actual = dse.projection(m, n, k, core, model)
                        assert actual["cycles"] == explicit_projection(m, n, k, core, model)
                        assert actual["useful_macs"] + actual["padding_macs"] == actual["issued_macs"]


def test_teacher_hand_counts_and_result_latency_not_issue_interval():
    rows = dse.toy_table(dse.Model())
    assert [r["useful_macs"] for r in rows] == [393216] * 3
    assert [r["padding_macs"] for r in rows] == [393216, 196608, 0]
    assert [r["issued_tiles"] for r in rows] == [64, 96, 64]
    assert [r["cycles"] for r in rows] == [224, 144, 112]
    assert [r["spatial_utilization"] for r in rows] == [.5, 2 / 3, 1]


def test_same_output_k_dependency_limits_small_context_group():
    p = dse.projection(1, 4, 2048, dse.Core(4, 4), dse.Model())
    assert p["issues"] == 4 and p["cycles"] == 84
    assert p["cycles"] != p["issues"]


def test_threshold_is_runtime_rule_and_shared_not_privileged():
    cores = (dse.Core(2, 4), dse.Core(4, 4))
    assert dse.eligible_cores(2, cores, "threshold_2") == [0]
    assert dse.eligible_cores(3, cores, "threshold_2") == [1]
    assert dse.eligible_cores(99, cores, "eft") == [0, 1]
    assert dse.eligible_cores(1, (dse.Core(3, 4), dse.Core(3, 4)), "threshold_2") == [0, 1]


def test_equal_m_but_unequal_n_is_still_asymmetric_for_threshold():
    cores = (dse.Core(1, 2), dse.Core(1, 22))
    assert dse.eligible_cores(1, cores, "threshold_2") == [0]
    assert dse.eligible_cores(2, cores, "threshold_2") == [0]
    assert dse.eligible_cores(3, cores, "threshold_2") == [1]
    assert dse.eligible_cores(1, cores, "eft") == [0, 1]


def test_vector_queue_is_shared_and_wall_utilization_differs_from_padding():
    w = {"id": "two-identical", "batch": 2, "experts": [
        {"id": i, "is_shared": False, "Me": 2, "H": 512, "F": 128}
        for i in range(2)]}
    cores = (dse.Core(3, 4), dse.Core(3, 4))
    r = dse.simulate_layer(w, cores, detail=True)
    assert r["useful_macs"] == 2 * 3 * 2 * 512 * 128
    assert r["spatial_utilization"] > r["wall_mac_utilization"]
    assert len(r["completion_order"]) == 2
    assert r["core_finish_cycles"][0] != r["core_finish_cycles"][1]


def test_population_coherence_rejects_wrong_tokens():
    w = {"id": "one", "batch": 1, "hidden": 512, "top_k": 1, "provenance": "captured_decode",
         "tokens": [{"token_index": 0, "sample_id": "capture", "routes": [{"expert_id": 3}]}],
         "experts": [{"id": 3, "Me": 1, "H": 512, "F": 128,
                      "is_shared": False, "token_indices": [0]}]}
    assert dse.verify_trace([w])["token_id_and_expert_population_coherent"]
    w["experts"][0]["token_indices"] = [1]
    import pytest
    with pytest.raises(AssertionError):
        dse.verify_trace([w])


def test_coherent_constructed_routes_are_not_certified_as_captured():
    import pytest
    w = {"id": "one", "batch": 1, "hidden": 512, "top_k": 1, "provenance": "constructed",
         "tokens": [{"token_index": 0, "sample_id": "fake", "routes": [{"expert_id": 3}]}],
         "experts": [{"id": 3, "Me": 1, "H": 512, "F": 128,
                      "is_shared": False, "token_indices": [0]}]}
    with pytest.raises(AssertionError, match="real captured"):
        dse.verify_trace([w])
