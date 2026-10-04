"""Independent finite-group, byte-conservation and fluid-capacity checks."""
from dataclasses import replace
import math
import unittest

from .compute import Core, TIMING_PROFILES
from .memory import memory_budget
from .model import (Phase, Settings, expert_phases, fluid_rates, private_estimate,
                    projection_phase, simulate_layer)


SINGLE_GEOMETRIES = (Core(2, 192, 32), Core(2, 96, 64), Core(3, 32, 128),
                     Core(3, 16, 256), Core(6, 4, 512), Core(3, 4, 1024))


def ceil_reference(n, d):
    return math.ceil(n / d)


def transaction_oracle_for_model(n, k, k0, kv):
    stride = ((2 * k + 31) // 32) * 32
    return {((col * stride + 2 * red) // 32) * 32
            for col in range(n) for red in range(k0, k0 + kv)}


def group_commit_oracle(records, segments, completion, initiation_interval):
    """Issue every spatial record; remember its actual previous K commit."""
    committed = [0] * records
    next_issue = 0
    for _ in range(segments):
        for record in range(records):
            issue = max(next_issue, committed[record])
            committed[record] = issue + completion
            next_issue = issue + initiation_interval
    return max(committed)


def phase_oracle(m, n, k, core, mem, settings, paired):
    """Enumerate the actual bounded N-group/M-chunk/GU-plane loop nest."""
    nm = ceil_reference(m, core.pm)
    nn = ceil_reference(n, core.pn)
    nk = ceil_reference(k, core.pk)
    planes = 2 if paired else 1
    resident_limit = min(settings.records, mem.accumulator_bytes // (planes * core.pm * core.pn * 4))
    rows_at_once = (1 if settings.flow == "bounded_os" or (paired and mem.w_slots == 1)
                    else min(nm, resident_limit))
    columns_at_once = min(nn, max(1, (mem.w_slots - 1) // planes), resident_limit // rows_at_once)
    cycles = issues = useful = issued = native_wire = xfetch = peak_acc = 0
    for n_first in range(0, nn, columns_at_once):
        n_tiles = min(columns_at_once, nn - n_first)
        for m_first in range(0, nm, rows_at_once):
            m_tiles = min(rows_at_once, nm - m_first)
            q = planes * m_tiles * n_tiles
            peak_acc = max(peak_acc, q * core.pm * core.pn * 4)
            cycles += group_commit_oracle(q, nk, settings.timing.completion_latency(core),
                                          settings.timing.initiation_interval)
            for k_first in range(0, k, core.pk):
                valid_k = min(core.pk, k - k_first)
                for plane in range(planes):
                    # W is fetched once for this N group and reused over its
                    # admitted M blocks. The two GU matrices have distinct bytes.
                    for n_tile in range(n_first, n_first + n_tiles):
                        valid_n = min(core.pn, n - n_tile * core.pn)
                        native_wire += valid_n * ceil_reference(valid_k * 2, 32) * 32
                    for m_tile in range(m_first, m_first + m_tiles):
                        valid_m = min(core.pm, m - m_tile * core.pm)
                        for n_tile in range(n_first, n_first + n_tiles):
                            valid_n = min(core.pn, n - n_tile * core.pn)
                            issues += 1
                            useful += valid_m * valid_n * valid_k
                            issued += core.pm * core.pn * core.pk
                # Same X serves Gate and Up before this slice is replaced.
                valid_rows = sum(min(core.pm, m - mt * core.pm)
                                 for mt in range(m_first, m_first + m_tiles))
                xfetch += valid_rows * valid_k * 2
    consumer_global = (2 if paired else 8) * m * n
    return dict(compute=cycles, issues=issues, useful_macs=useful, issued_macs=issued,
                hbm_bytes=native_wire, activation_bytes=xfetch + consumer_global,
                peak_accumulator_bytes=peak_acc)


def synthetic_phase(base, hbm_bytes, activation_bytes=0, control_cycles=0,
                    window=1_000_000, vector_elements=0):
    return Phase(name="synthetic", compute=int(base), issues=1, useful_macs=1,
                 issued_macs=1, hbm_bytes=hbm_bytes, weight_unique=hbm_bytes,
                 w_port_bytes=0, x_port_bytes=0, acc_port_bytes=0,
                 activation_bytes=activation_bytes, vector_elements=vector_elements,
                 control_cycles=control_cycles, base_cycles=float(base),
                 base_reason="compute_or_K_dependency", prefetch_bandwidth=window,
                 peak_accumulator_bytes=0, peak_x_bytes=0, peak_w_bytes=0)


class FinitePhaseTests(unittest.TestCase):
    def test_gu_and_down_against_explicit_commit_and_issue_oracle(self):
        for core in SINGLE_GEOMETRIES:
            for slot_limit in (1, 2, 6, 32):
                for flow in ("bounded_ws", "bounded_os"):
                    settings = Settings(prefetch_slots=slot_limit, flow=flow)
                    mem = memory_budget((core,), 16, 2048, 2816, buffer_limit=slot_limit).cores[0]
                    for paired in (False, True):
                        m, n, k = 7, 197, 1041
                        got = projection_phase(m, n, k, core, mem, settings, paired)
                        expected = phase_oracle(m, n, k, core, mem, settings, paired)
                        for key, value in expected.items():
                            self.assertEqual(getattr(got, key), value,
                                             (core, slot_limit, flow, paired, key))
                        self.assertLessEqual(got.peak_accumulator_bytes, mem.accumulator_bytes)
                        self.assertLessEqual(got.peak_x_bytes, mem.x_register_bytes)
                        self.assertLessEqual(got.peak_w_bytes, mem.w_capacity_bytes)

    def test_gu_retains_two_fp32_planes_and_reuses_x(self):
        core = Core(6, 4, 512)
        settings = Settings(prefetch_slots=6, records=3)
        mem = memory_budget((core,), 16, 2048, 2816, buffer_limit=6).cores[0]
        p = projection_phase(13, 17, 515, core, mem, settings, paired=True)
        self.assertEqual(p.issues, 2 * 3 * 5 * 2)
        self.assertEqual(p.useful_macs, 2 * 13 * 17 * 515)
        self.assertEqual(p.peak_accumulator_bytes, 2 * 3 * core.record_bytes)
        # Native byte count follows two separate F tensors; no concatenation
        # can pool their tail padding or discard Gate before matching Up.
        self.assertEqual(p.weight_unique, 2 * 17 * 1056)
        self.assertEqual(p.activation_bytes, 5 * 13 * 515 * 2 + 2 * 13 * 17)
        self.assertEqual(p.vector_elements, 3 * 13 * 17)
        self.assertEqual(p.acc_port_bytes, 2 * 3 * 5 * (2 * 2 - 1) * core.record_bytes + 8 * 13 * 17)

    def test_single_buffer_has_no_lookahead_and_more_row_reloads(self):
        core = Core(6, 4, 512)
        settings = Settings(prefetch_slots=1)
        one = memory_budget((core,), 32, 2048, 2816, buffer_limit=1).cores[0]
        two = memory_budget((core,), 32, 2048, 2816, buffer_limit=2).cores[0]
        a = projection_phase(31, 32, 1025, core, one, settings, True)
        b = projection_phase(31, 32, 1025, core, two, replace(settings, prefetch_slots=2), True)
        self.assertEqual(a.hbm_bytes, a.weight_unique * 6)
        self.assertGreater(a.hbm_bytes, b.hbm_bytes)
        self.assertGreater(a.hbm_bytes / a.prefetch_bandwidth, b.hbm_bytes / b.prefetch_bandwidth)
        self.assertTrue(a.serialized_supply)
        self.assertTrue(b.serialized_supply)
        self.assertLessEqual(a.peak_w_bytes, one.w_capacity_bytes)

    def test_no_spare_tail_fill_read_calendar_and_single_final_drain(self):
        core = Core(6, 4, 512)
        # Logical slices are much smaller than their padded physical W slots.
        # This oracle enumerates native sectors and serial stage boundaries;
        # it does not derive the model's average-native-slice formula.
        for slots in (1, 2):
            settings = Settings(prefetch_slots=slots, control=False)
            mem = memory_budget((core,), 2, 2048, 2816, buffer_limit=slots).cores[0]
            for n, k in ((1, 1), (4, 3), (1, 515), (3, 1041)):
                phase = projection_phase(1, n, k, core, mem, settings, True)
                native_payloads = []
                for k0 in range(0, k, core.pk):
                    kv = min(core.pk, k - k0)
                    # Every requested BF16 element addresses a row32 sector.
                    sectors = transaction_oracle_for_model(n, k, k0, kv)
                    native_payloads.extend([len(sectors) * 32] * 2)
                time = 0.0
                for first in range(0, len(native_payloads), slots):
                    payload_group = native_payloads[first:first + slots]
                    requested = time
                    responded = requested + 64
                    landed = responded + 1
                    # Independent continuous-credit payload service followed
                    # by padded physical operand reads, held through last use.
                    payload_done = landed + sum(payload_group) / (8192 / 65)
                    last_read_done = payload_done + len(payload_group) * (4096 / 1024)
                    time = last_read_done
                self.assertTrue(phase.serialized_supply)
                self.assertEqual(phase.hbm_bytes, sum(native_payloads))
                self.assertAlmostEqual(phase.hbm_bytes / phase.prefetch_bandwidth, time)
                self.assertEqual(phase.final_drain_cycles, 21)
                # Supply dominates this tiny case; the private estimate must
                # include every serial startup and just one final dot drain.
                self.assertAlmostEqual(private_estimate((phase,), 1, k, n, settings), time + 21)

    def test_physical_tails_require_full_operand_capacity(self):
        core = Core(6, 4, 512)
        settings = Settings()
        mem = memory_budget((core,), 2, 2048, 2816).cores[0]
        p = projection_phase(1, 1, 1, core, mem, settings, True)
        self.assertEqual(p.useful_macs, 2)
        self.assertEqual(p.issued_macs, 2 * core.macs)
        self.assertEqual(p.peak_accumulator_bytes, 2 * core.record_bytes)
        with self.assertRaises(ValueError):
            projection_phase(1, 1, 1, core, replace(mem, accumulator_bytes=2 * core.record_bytes - 1),
                             settings, True)
        with self.assertRaises(ValueError):
            projection_phase(1, 1, 1, core, replace(mem, eligible=False), settings, True)

    def test_z_chunks_reload_weights_with_exact_useful_work(self):
        core = Core(6, 4, 512)
        settings = Settings(prefetch_slots=6)
        mem = memory_budget((core,), 32, 2048, 2816, buffer_limit=6).cores[0]
        m, h, f = 17, 65, 33
        # Six BF16 Z rows fit and round to one full spatial M block.
        narrow = replace(mem, z_bytes=6 * f * 2)
        whole = replace(mem, z_bytes=ceil_reference(m, core.pm) * core.pm * f * 2)
        small = expert_phases(m, h, f, core, narrow, settings)
        full = expert_phases(m, h, f, core, whole, settings)
        self.assertEqual([p.name for p in small], ["gate_up", "down"] * 3)
        self.assertEqual(sum(p.useful_macs for p in small), 3 * m * h * f)
        self.assertEqual(sum(p.useful_macs for p in full), 3 * m * h * f)
        self.assertGreater(sum(p.hbm_bytes for p in small), sum(p.hbm_bytes for p in full))
        self.assertEqual(sum(p.weight_unique for p in small), 3 * sum(p.weight_unique for p in full))
        with self.assertRaisesRegex(ValueError, "one Z row"):
            expert_phases(m, h, f, core, replace(mem, z_bytes=f * 2 - 1), settings)


class FluidAllocationTests(unittest.TestCase):
    def assert_capacity_and_work_conservation(self, phases, settings):
        rates = fluid_rates(phases, settings)
        capacities = {"vector_elements": 64}
        if settings.hbm:
            capacities["hbm_bytes"] = settings.fabric.landing_credit_bandwidth_upper_bound
        if settings.ports:
            capacities["activation_bytes"] = 384
        if settings.control:
            capacities["control_cycles"] = 1
        usage = {resource: sum(getattr(phases[c], resource) * rates[c][1] for c in phases)
                 for resource in capacities}
        for resource, capacity in capacities.items():
            self.assertLessEqual(usage[resource], capacity * (1 + 1e-10))
        for c, (_, rate) in rates.items():
            p = phases[c]
            limit = 1 / max(1, p.base_cycles)
            if settings.hbm:
                limit = min(limit, p.prefetch_bandwidth / p.hbm_bytes)
            self.assertGreater(rate, 0)
            self.assertLessEqual(rate, limit * (1 + 1e-10))
            # A core below its local limit must demand a shared saturated
            # resource; otherwise its rate could be increased without impact.
            if rate < limit * (1 - 1e-9):
                self.assertTrue(any(getattr(p, resource) > 0 and
                                    math.isclose(usage[resource], capacity, rel_tol=1e-8)
                                    for resource, capacity in capacities.items()))
        return rates

    def test_aggregate_resources_and_unused_share_reclamation(self):
        cases = (
            ({0: synthetic_phase(1, 8192), 1: synthetic_phase(1, 8192)}, Settings(ports=False, control=False)),
            ({0: synthetic_phase(1000, 8192), 1: synthetic_phase(1, 8192)}, Settings(ports=False, control=False)),
            ({0: synthetic_phase(1, 32, 100000), 1: synthetic_phase(1, 32, 100000)}, Settings(hbm=False, control=False)),
            ({0: synthetic_phase(1, 32, 0, 100), 1: synthetic_phase(1, 32, 0, 100)}, Settings(hbm=False, ports=False)),
            ({0: synthetic_phase(100, 10000, 100000, 100), 1: synthetic_phase(10, 4000, 100, 200)}, Settings()),
            ({0: synthetic_phase(1, 32, vector_elements=1000),
              1: synthetic_phase(1, 32, vector_elements=1000)},
             Settings(hbm=False, ports=False, control=False)),
        )
        for phases, settings in cases:
            self.assert_capacity_and_work_conservation(phases, settings)
        phases, settings = cases[1]
        rates = fluid_rates(phases, settings)
        self.assertAlmostEqual(rates[0][1], 0.001)
        total = sum(rates[c][1] * phases[c].hbm_bytes for c in phases)
        self.assertAlmostEqual(total, settings.fabric.landing_credit_bandwidth_upper_bound)
        self.assertGreater(rates[1][1], settings.fabric.landing_credit_bandwidth_upper_bound / 2 / 8192)

    def test_private_port_partitions_do_not_double_aggregate_capacity(self):
        cores = (Core(4, 4, 512), Core(1, 32, 128))
        settings = Settings(hbm=False, control=False)
        budget = memory_budget(cores, 16, 2048, 2816)
        phases = {c: projection_phase(13, 47, 515, core, budget.cores[c], settings, True)
                  for c, core in enumerate(cores)}
        rates = self.assert_capacity_and_work_conservation(phases, settings)
        for c, (_, rate) in rates.items():
            p, mem = phases[c], budget.cores[c]
            self.assertLessEqual(rate * p.w_port_bytes, mem.w_bandwidth * (1 + 1e-10))
            self.assertLessEqual(rate * p.x_port_bytes, mem.x_bandwidth * (1 + 1e-10))
            self.assertLessEqual(rate * p.acc_port_bytes, mem.accumulator_bandwidth * (1 + 1e-10))
        self.assertLessEqual(sum(phases[c].x_port_bytes * rates[c][1] for c in phases), 384 * (1 + 1e-10))


class LayerTests(unittest.TestCase):
    @staticmethod
    def workload():
        # Shared expert covers every token; routed counts sum to top_k * batch.
        return {"id": "independent-post-router", "batch": 8,
                "experts": [{"Me": 8, "H": 2048, "F": 2816},
                            {"Me": 2, "H": 2048, "F": 1408},
                            {"Me": 5, "H": 2048, "F": 1408},
                            {"Me": 1, "H": 2048, "F": 1408}]}

    def test_deterministic_layer_and_required_post_router_macs(self):
        workload = self.workload()
        families = ((Core(6, 4, 512),), (Core(3, 4, 512), Core(3, 4, 512)),
                    (Core(4, 4, 512), Core(1, 32, 128)))
        required = sum(3 * e["Me"] * e["H"] * e["F"] for e in workload["experts"])
        for cores in families:
            a = simulate_layer(workload, cores, detail=True)
            b = simulate_layer(workload, cores, detail=True)
            self.assertEqual(a, b)
            self.assertEqual(a["useful_macs"], required)
            self.assertGreaterEqual(a["issued_macs"], required)
            self.assertEqual(a["cycles"], max(a["core_finish_cycles"]))
            self.assertEqual(len(a["bindings"]), len(workload["experts"]))
            self.assertEqual(a["budget"]["total_bytes"], 2_158_592)
            for p in a["phases"]:
                self.assertLessEqual(p["started"], p["stream_completed"])
                self.assertLessEqual(p["stream_completed"], p["completed"])

    def test_resource_disable_oracles_under_fixed_single_owner(self):
        workload = self.workload()
        for core in SINGLE_GEOMETRIES:
            settings = Settings()
            base = simulate_layer(workload, (core,), settings)["cycles"]
            for disabled in ({"hbm": False}, {"ports": False}, {"control": False},
                             {"hbm": False, "ports": False, "control": False}):
                oracle = simulate_layer(workload, (core,), replace(settings, **disabled))
                self.assertLessEqual(oracle["cycles"], base * (1 + 1e-10), (core, disabled))
                self.assertEqual(oracle["useful_macs"],
                                 sum(3 * e["Me"] * e["H"] * e["F"] for e in workload["experts"]))

    def test_solo_expert_private_estimate_equals_simulation_after_common_costs(self):
        workload = {"id": "solo-private-estimate", "batch": 5, "hidden": 33,
                    "experts": [{"Me": 5, "H": 33, "F": 17}]}
        for core in SINGLE_GEOMETRIES:
            for slots in (1, 2, 6, 32):
                for switches in ({}, {"hbm": False}, {"ports": False},
                                 {"hbm": False, "ports": False, "control": False}):
                    settings = Settings(prefetch_slots=slots, **switches)
                    result = simulate_layer(workload, (core,), settings, detail=True)
                    mem = memory_budget((core,), 5, 33, 17, buffer_limit=slots).cores[0]
                    phases = expert_phases(5, 33, 17, core, mem, settings)
                    estimated = private_estimate(phases, 5, 33, 17, settings)
                    clear = 5 * 33 * 4 / 384 if settings.ports else 0
                    binding = 4 if settings.control else 0
                    self.assertAlmostEqual(result["cycles"] - clear - binding, estimated, places=8,
                                           msg=(core, slots, switches))
                    self.assertAlmostEqual(result["bindings"][0]["predicted_private_cycles"], estimated)
                    for actual, phase in zip(result["phases"], phases):
                        self.assertAlmostEqual(actual["completed"] - actual["stream_completed"],
                                               phase.final_drain_cycles)
                    expected_startups = sum(64 for p in phases if settings.hbm and not p.serialized_supply)
                    self.assertEqual(result["exclusive_stream_counters"][0]["hbm_startup"], expected_startups)

    def test_declared_timing_sensitivity_keeps_same_required_work(self):
        workload = {"id": "small-timing", "batch": 2,
                    "experts": [{"Me": 2, "H": 2048, "F": 1408}]}
        required = 3 * 2 * 2048 * 1408
        # Timing profiles can change geometry rankings. They must preserve
        # the required work; no universal ranking order is asserted here.
        for name in ("conservative_flat20", "log2_stage1", "log2_stage2", "log2_stage4"):
            for core in SINGLE_GEOMETRIES:
                result = simulate_layer(workload, (core,), Settings(timing=TIMING_PROFILES[name]))
                self.assertEqual(result["useful_macs"], required)
                self.assertTrue(math.isfinite(result["cycles"]))
                self.assertGreater(result["cycles"], 0)

    def test_native_unique_report_includes_row32_logical_tails(self):
        workload = {"id": "row32-tail", "batch": 4,
                    "experts": [{"Me": 4, "H": 33, "F": 17},
                                {"Me": 2, "H": 33, "F": 17}]}
        result = simulate_layer(workload, (Core(6, 4, 512),), detail=True)
        expected = sum(2 * e["F"] * ceil_reference(2 * e["H"], 32) * 32
                       + e["H"] * ceil_reference(2 * e["F"], 32) * 32
                       for e in workload["experts"])
        self.assertEqual(result["native_unique_bytes"], expected)

    def test_api_guards_reject_invalid_descriptors_routes_and_owners(self):
        for bad in ({"records": 0}, {"records": 33}, {"flow": "is"}):
            with self.assertRaises(ValueError):
                Settings(**bad)
        workload = self.workload()
        dual = (Core(3, 4, 512), Core(3, 4, 512))
        # Q=32 paired descriptors per core, slot descriptors and controller
        # state exceed the fixed 16-KiB reserve when there are two cores.
        with self.assertRaisesRegex(ValueError, "fixed control budget"):
            simulate_layer(workload, dual, Settings(records=32))
        with self.assertRaisesRegex(ValueError, "unknown dispatch"):
            simulate_layer(workload, dual, policy="invented_policy")
        with self.assertRaisesRegex(ValueError, "one fixed owner"):
            simulate_layer(workload, dual, fixed_owners=(0,))
        with self.assertRaisesRegex(ValueError, "fixed owner is illegal"):
            simulate_layer(workload, dual, fixed_owners=(2, 0, 0, 0))
        excessive = {**workload, "experts": [{"Me": 9, "H": 2048, "F": 1408}]}
        with self.assertRaisesRegex(ValueError, "inside the declared batch"):
            simulate_layer(excessive, dual)
        mixed_hidden = {**workload, "hidden": 2048,
                        "experts": [{"Me": 1, "H": 1024, "F": 1408}]}
        with self.assertRaisesRegex(ValueError, "declared layer hidden"):
            simulate_layer(mixed_hidden, dual)


if __name__ == "__main__":
    unittest.main()
