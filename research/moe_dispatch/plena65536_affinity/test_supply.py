"""Independent numeric conservation, timeline and interface-boundary tests."""
from copy import deepcopy

import pytest

from research.moe_dispatch.plena65536_dedicated.geometry import Core
from research.moe_dispatch.plena65536_shared.timing import PROFILES, phase
from research.moe_dispatch.plena65536_affinity.supply import array_feed_peak, supply_ledger


def example_window():
    return {"id": "tiny", "batch": 3, "hidden": 8,
            "experts": [{"id": -1, "is_shared": True, "Me": 3, "H": 8, "F": 12}]}


def plan_for(cores, window, owners, starts=None, profile=PROFILES[0]):
    loads = [0] * len(cores)
    jobs = []
    for expert, owner in zip(window["experts"], owners):
        if not expert["Me"]:
            continue
        m, h, f = expert["Me"], expert["H"], expert["F"]
        core = cores[owner]
        start = starts[len(jobs)] if starts is not None else loads[owner]
        end = start + phase(core, m, f, h, 2, profile) + phase(core, m, h, f, 1, profile)
        jobs.append({"expert_id": expert["id"], "core_id": owner, "start": start, "end": end})
        loads[owner] = end
    return {"cycles": max(loads), "jobs": jobs}


def test_hand_counted_os_masked_padded_output_and_postprocess_bytes():
    # Physical 2x4x4; temporal segment width R=8, hence two feed cycles/segment.
    core = Core(2, 8, 1, 2)
    window = example_window()
    plan = plan_for((core,), window, (0,))
    assert plan["cycles"] == 70  # GU:24+15, Down:16+15.
    got = supply_ledger((core,), window, plan)
    assert got["unique_weight_bytes"] == 576
    assert got["pool_weight_bytes"] == 1152
    assert got["pool_x_bytes"] == 432
    assert got["pool_context_bytes"] == 0
    assert got["pool_output_bytes"] == 384
    assert got["pool_postprocess_bytes"] == 552
    assert got["pool_operand_bytes"] == 1584
    assert got["pool_operand_full_bytes"] == 1920
    assert got["array_full_weight_bytes"] == 1280
    assert got["array_full_x_bytes"] == 640
    assert got["per_core_feed_cycles"] == [40]
    assert got["array_full_scheduled_peak_bytes_per_cycle"] == 48
    assert got["array_full_scheduled_peak_TB_s"] == 0.048
    assert got["pool_all_average_TB_s"] == 2520 / 70 / 1000
    assert got["local_double_buffer_KiB"] == 192 / 1024
    assert got["static_local_storage_KiB"] == 544 / 1024
    assert got["full_layer_weight_MiB"] is None
    assert got["pool_minimum_required_peak_TB_s"] is None
    assert not got["end_to_end_validated"]


def test_scalar_extension_and_non_scalar_charge_distinct_temporal_bandwidth():
    window = example_window()
    scalar = Core(1, 4, 2, 4)  # Identical physical PM,PN,G, but R=4 rather than 8.
    native = Core(2, 8, 1, 2)
    assert scalar.shape == native.shape
    rows = [supply_ledger((core,), window, plan_for((core,), window, (0,)))
            for core in (scalar, native)]
    assert rows[0]["array_full_scheduled_peak_TB_s"] == rows[1]["array_full_scheduled_peak_TB_s"]
    assert rows[0]["local_double_buffer_KiB"] * 2 == rows[1]["local_double_buffer_KiB"]
    assert rows[0]["pool_operand_full_bytes"] < rows[1]["pool_operand_full_bytes"]


def test_exact_peak_counts_actual_overlap_not_unconditional_sum_of_ports():
    cores = (Core(2, 8, 1, 2),) * 2
    window = example_window()
    window["experts"].append({"id": 4, "is_shared": False, "Me": 3, "H": 8, "F": 12})
    overlap = supply_ledger(cores, window, plan_for(cores, window, (0, 1)))
    serial = supply_ledger(cores, window, plan_for(cores, window, (0, 1), starts=(0, 70)))
    assert overlap["array_full_scheduled_peak_TB_s"] == 0.096
    assert serial["array_full_scheduled_peak_TB_s"] == 0.048
    assert serial["array_full_provision_peak_TB_s"] == 0.096
    assert overlap["pool_operand_bytes"] == serial["pool_operand_bytes"]
    assert overlap["pool_operand_average_TB_s"] == 2 * serial["pool_operand_average_TB_s"]


def test_same_timestamp_end_start_are_half_open_and_do_not_double_count():
    cores = (Core(2, 8, 1, 2),) * 2
    window = example_window()
    window["experts"].append({"id": 4, "is_shared": False, "Me": 3, "H": 8, "F": 12})
    # First GU feed finishes at 24; second GU starts at 24. Later overlap exists,
    # so check the single-core timelines by placing complete jobs back-to-back.
    result = array_feed_peak(cores, window, plan_for(cores, window, (0, 1), starts=(0, 70)))
    assert result["array_full_scheduled_peak_bytes_per_cycle"] == 48


@pytest.mark.parametrize("profile", PROFILES)
def test_fused_gu_and_serialized_record_timing_conserve_padded_bytes(profile):
    core = Core(2, 8, 1, 2)
    window = example_window()
    got = supply_ledger((core,), window, plan_for((core,), window, (0,), profile=profile), profile)
    assert got["array_full_operand_bytes"] == 1920
    assert got["per_core_feed_cycles"] == [40]
    if profile.serialized:
        # 12 GU output records each (2+15), 4 Down records each (4+15).
        assert got["cycles"] == 280
        assert got["feed_interval_event_count"] == 32
    elif profile.tree_stage == 2:
        assert got["cycles"] == 74
    else:
        assert got["cycles"] == 70
    assert got["pool_context_bytes"] == 0


def test_deepseek_full_layer_capacity_is_not_average_active_weight_capacity():
    window = {"id": "deepseek", "batch": 2, "hidden": 2048,
              "experts": [{"id": -1, "is_shared": True, "Me": 2, "H": 2048, "F": 2816},
                          {"id": 9, "is_shared": False, "Me": 1, "H": 2048, "F": 1408}]}
    core = Core(1, 512, 1, 128)
    got = supply_ledger((core,), window, plan_for((core,), window, (0, 0)))
    assert got["physical_multipliers"] == 65536
    assert got["full_layer_weight_MiB"] == 1089
    assert got["unique_weight_MiB"] == 49.5
    assert got["resident_payload_upper_MiB"] > got["unique_weight_MiB"]
    assert got["unique_weight_fetch_ms_at256"] == 49.5 * 1024**2 / 256e6


def test_missing_duplicate_overlap_and_inconsistent_timing_are_rejected():
    core = Core(2, 8, 1, 2)
    window = example_window()
    plan = plan_for((core,), window, (0,))
    bad = deepcopy(plan)
    bad["jobs"][0]["end"] += 1
    with pytest.raises(ValueError, match="duration disagrees"):
        supply_ledger((core,), window, bad)
    bad = deepcopy(plan)
    bad["jobs"].append(bad["jobs"][0])
    with pytest.raises(ValueError, match="every active expert"):
        supply_ledger((core,), window, bad)
    window["experts"].append({"id": 2, "Me": 3, "H": 8, "F": 12})
    bad = plan_for((core,), window, (0, 0), starts=(0, 0))
    with pytest.raises(ValueError, match="overlap"):
        supply_ledger((core,), window, bad)


def test_array_ingress_does_not_charge_internal_broadcast_as_shared_sram_bytes():
    core = Core(2, 8, 2, 4)  # PM4 PN8 G4:128 physical multipliers.
    window = example_window()
    got = supply_ledger((core,), window, plan_for((core,), window, (0,)))
    assert got["array_full_scheduled_peak_bytes_per_cycle"] == 2 * 4 * (4 + 8)
    assert got["array_full_scheduled_peak_bytes_per_cycle"] < 4 * core.macs
    assert not got["PE_fanout_traffic_included_in_ingress"]
