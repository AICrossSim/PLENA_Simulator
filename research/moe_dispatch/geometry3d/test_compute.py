"""Independent event checks for prospective 3D geometry and output records."""
from dataclasses import replace
import heapq

import pytest

from .compute import (Core, ContextLimits, TimingProfile, TIMING_PROFILES,
                                ceil, enumerate_geometries, enumeration_counts,
                                geometry_id, projection, projection_groups, paired_gate_up,
                                resource_requirements)


def event_projection(m, n, k, core, timing, limits):
    """Event/scoreboard oracle; never uses the production group formula."""
    nm, nn, nk = ceil(m, core.pm), ceil(n, core.pn), ceil(k, core.pk)
    bound = min(limits.max_records, limits.accumulator_bytes // (4 * core.pm * core.pn))
    next_issue = 0
    completion = 0
    timeline = []
    for first in range(0, nm * nn, bound):
        resident = list(range(first, min(first + bound, nm * nn)))
        previous_commit = {r: 0 for r in resident}
        pending = []
        for segment in range(nk):
            for record in resident:
                now = max(next_issue, completion, previous_commit[record])
                while pending and pending[0][0] <= now:
                    heapq.heappop(pending)
                commit = now + dict(timing.latency_by_pk)[core.pk] + timing.commit_cycles
                heapq.heappush(pending, (commit, record))
                assert len(pending) <= bound
                previous_commit[record] = commit
                timeline.append((record, segment, now, commit))
                next_issue = now + timing.initiation_interval
        completion = max(t for t, _ in pending)
    return completion, timeline


def event_paired_gate_up(m, f, h, core, timing, limits):
    """Two explicit phase scoreboards; Gate storage survives the Up phase."""
    records = ceil(m, core.pm) * ceil(f, core.pn)
    bound = min(limits.max_records, limits.accumulator_bytes // (8 * core.pm * core.pn))
    next_issue = 0
    previous_phase_done = 0
    phase_commits = []
    for first in range(0, records, bound):
        resident = list(range(first, min(first + bound, records)))
        assert 2 * len(resident) * core.record_bytes <= limits.accumulator_bytes
        gate_record_values = set()
        for phase in ("Gate", "Up"):
            previous_commit = {r: 0 for r in resident}
            for segment in range(ceil(h, core.pk)):
                for record in resident:
                    now = max(next_issue, previous_phase_done, previous_commit[record])
                    previous_commit[record] = now + dict(timing.latency_by_pk)[core.pk] + timing.commit_cycles
                    next_issue = now + timing.initiation_interval
            previous_phase_done = max(previous_commit.values())
            phase_commits.append(previous_phase_done)
            if phase == "Gate":
                gate_record_values.update(resident)
            else:
                assert gate_record_values == set(resident)
                gate_record_values.clear()  # A paired consumer may now retire.
    return previous_phase_done, phase_commits


def test_complete_declared_enumeration_and_unequal_pk_mirrors():
    designs = enumerate_geometries()
    assert enumeration_counts() == {
        "total": 16763, "single": 44, "dual": 16719, "homogeneous": 41,
        "heterogeneous": 16678, "heterogeneous_same_pk": 4662,
        "heterogeneous_unequal_pk": 12016}
    assert len(set(designs)) == len(designs)
    assert len({geometry_id(g) for g in designs}) == len(designs)
    assert all(sum(c.macs for c in g) == 12288 for g in designs)
    assert all(len(g) == 1 or g[0] <= g[1] for g in designs)
    assert (Core(4, 4, 512), Core(1, 32, 128)) not in designs
    assert (Core(1, 32, 128), Core(4, 4, 512)) in designs
    assert (Core(3, 16, 256),) in designs
    assert (Core(1, 96, 128),) in designs
    assert (Core(1, 192, 64),) in designs


def test_timing_profiles_are_explicit_anchor_and_stage_sensitivities():
    assert [dict(TIMING_PROFILES[n].latency_by_pk)[512] for n in TIMING_PROFILES] == [20] * 4
    assert dict(TIMING_PROFILES["log2_stage1"].latency_by_pk) == {32: 16, 64: 17, 128: 18, 256: 19, 512: 20, 1024: 21}
    assert dict(TIMING_PROFILES["log2_stage2"].latency_by_pk) == {32: 12, 64: 14, 128: 16, 256: 18, 512: 20, 1024: 22}
    assert dict(TIMING_PROFILES["log2_stage4"].latency_by_pk) == {32: 4, 64: 8, 128: 12, 256: 16, 512: 20, 1024: 24}
    assert all("unvalidated" in p.hypothesis or "no physical validation" in p.hypothesis
               for p in TIMING_PROFILES.values())


def test_closed_form_matches_independent_events_for_context_bytes_ii_and_tails():
    timings = list(TIMING_PROFILES.values()) + [
        TimingProfile("test_II_longer_than_latency", tuple((k, 1) for k in (32, 64, 128, 256, 512, 1024)),
                      initiation_interval=5, commit_cycles=1)]
    for core in (Core(1, 7, 64), Core(3, 5, 128), Core(4, 3, 256), Core(6, 4, 512)):
        for limits in (ContextLimits(1, 65536), ContextLimits(8, 65536),
                       ContextLimits(9, 2 * core.record_bytes)):
            for m in (1, 4, 17):
                for n in (1, 13, 65):
                    for k in (1, 63, 129, 513, 2049):
                        for timing in timings:
                            p = projection(m, n, k, core, timing, limits)
                            actual, events = event_projection(m, n, k, core, timing, limits)
                            assert p.cycles == actual
                            assert p.issues == len(events)
                            assert p.issued_macs == p.issues * core.macs
                            assert p.useful_macs + p.padding_macs == p.issued_macs
                            assert p.peak_resident_records <= limits.max_records
                            assert p.peak_accumulator_bytes <= limits.accumulator_bytes
                            previous = {}
                            for r, s, issue, commit in events:
                                if s:
                                    assert issue >= previous[r]
                                previous[r] = commit


def test_short_k_serial_dependency_has_actual_pipeline_delay():
    core = Core(1, 4, 64)
    p = projection(1, 4, 129, core, TIMING_PROFILES["log2_stage1"], ContextLimits(8, 65536))
    assert (p.k_segments, p.issues, p.cycles) == (3, 3, 54)
    assert p.last_k_elements == 1
    assert p.cycles > p.issues
    assert p.accumulator_read_bytes == 2 * core.record_bytes
    assert p.accumulator_write_bytes == 3 * core.record_bytes


def test_more_independent_records_overlap_without_free_capacity():
    core = Core(1, 1, 64)
    one = projection(1, 8, 128, core, limits=ContextLimits(1, 65536))
    eight = projection(1, 8, 128, core, limits=ContextLimits(8, 65536))
    byte_limited = projection(1, 8, 128, core, limits=ContextLimits(8, 4))
    assert one.cycles == byte_limited.cycles == 8 * 2 * 15
    assert eight.cycles == 37
    assert eight.peak_accumulator_bytes == 32
    assert one.issued_macs == eight.issued_macs


def test_projection_padding_and_group_retirement():
    core = Core(3, 7, 128)
    p = projection(5, 2 * 17, 257, core, limits=ContextLimits(3, 252))
    assert (p.m_waves, p.n_tiles, p.k_segments) == (2, 5, 3)
    assert (p.last_m_rows, p.last_n_columns, p.last_k_elements) == (2, 6, 1)
    assert p.useful_macs == 5 * 34 * 257
    assert p.issued_macs == 2 * 5 * 3 * core.macs
    assert p.group_count == 4
    groups = projection_groups(5, 34, 257, core, limits=ContextLimits(3, 252))
    assert [g.first_record for g in groups] == [0, 3, 6, 9]
    assert [g.records for g in groups] == [3, 3, 3, 1]
    assert p.cycles == sum(g.replacement_interval for g in groups[:-1]) + groups[-1].cycles


def test_paired_gate_up_retains_both_outputs_and_independent_f_tails():
    core = Core(3, 7, 128)
    limits = ContextLimits(3, 504)
    gu = paired_gate_up(5, 17, 257, core, limits=limits)
    single = projection(5, 17, 257, core, limits=ContextLimits(3, 252))
    concat = projection(5, 34, 257, core, limits=limits)
    assert (gu.m_waves, gu.n_tiles, gu.k_segments, gu.group_count) == (2, 3, 3, 2)
    assert (gu.last_m_rows, gu.last_n_columns, gu.last_k_elements) == (2, 3, 1)
    assert gu.record_bytes == 2 * core.record_bytes
    assert gu.peak_accumulator_bytes == 504
    assert gu.useful_macs == 5 * 34 * 257
    assert gu.issues == 2 * single.issues
    assert gu.cycles == 2 * single.cycles
    assert gu.issued_macs > concat.issued_macs


def test_paired_gate_up_matches_independent_phase_events():
    timings = list(TIMING_PROFILES.values()) + [
        TimingProfile("test_pair_II_longer_than_latency",
                      tuple((k, 1) for k in (32, 64, 128, 256, 512, 1024)),
                      initiation_interval=5)]
    for core in (Core(1, 7, 32), Core(3, 5, 128), Core(2, 9, 1024)):
        for bound in (1, 3, 8):
            limits = ContextLimits(bound, 2 * core.record_bytes * bound)
            for m, f, h in ((1, 1, 1), (7, 17, 513), (4, 65, 2049)):
                for timing in timings:
                    cost = paired_gate_up(m, f, h, core, timing, limits)
                    actual, phases = event_paired_gate_up(m, f, h, core, timing, limits)
                    assert cost.cycles == actual
                    assert len(phases) == 2 * cost.group_count
                    assert cost.peak_accumulator_bytes <= limits.accumulator_bytes


def test_resource_requirements_charge_full_slices_and_commit_port():
    core = Core(2, 4, 128)
    r = resource_requirements(core, limits=ContextLimits(8, 128))
    assert r["x_double_buffer_bytes"] == 1024
    assert r["weight_double_buffer_bytes"] == 2048
    assert r["resident_records"] == 4
    assert r["resident_partial_sum_bytes"] == 128
    assert r["accumulator_required_bytes_per_cycle"] == 64


def test_invalid_shapes_timing_or_storage_fail_before_timing():
    with pytest.raises(ValueError):
        Core(1, 1, 16)
    with pytest.raises(ValueError):
        Core(17, 1, 64)
    with pytest.raises(ValueError):
        projection(0, 1, 1, Core(1, 1, 64))
    with pytest.raises(ValueError):
        projection(1, 1, 1, Core(2, 4, 64), limits=ContextLimits(8, 31))
    with pytest.raises(ValueError):
        replace(TIMING_PROFILES["log2_stage1"], initiation_interval=0)
    with pytest.raises(ValueError):
        TimingProfile("missingPK", ((64, 20), (128, 20), (256, 20)))
