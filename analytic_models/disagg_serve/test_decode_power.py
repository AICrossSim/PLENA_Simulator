"""Checks for the decode power and energy-efficiency model."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import decode_power as decode_power_module  # noqa: E402
from decode_power import (  # noqa: E402
    ANALYTIC_ENERGY_TIER,
    COMPUTE_ENERGY_CROSS_CHECK,
    LEAKAGE_SOURCE,
    LINK_ENERGY_PJ_PER_BIT,
    LOGIC_LEAKAGE_W_PER_MM2,
    MATRIX_MACHINE_LEAKAGE_MEASUREMENT,
    REFERENCE_CONFIGURATION,
    REFERENCE_MAC_ENERGY_PJ,
    SRAM_ACCESS_ENERGY_PJ_PER_BIT,
    SRAM_ENERGY_SOURCE,
    analytic_energy_identity,
    calibrate_reference_mac_energy,
    compute_power_watts,
    decode_power,
    logic_leakage_power_watts,
    memory_power_watts,
)
from handoff import LINK_ENERGY_PJ_PER_BIT as HANDOFF_LINK_ENERGY  # noqa: E402
from hbm_technology import HBM_TECHNOLOGIES, hbm_technology  # noqa: E402


class MemoryPowerTest(unittest.TestCase):
    def test_idle_memory_draws_only_background_power(self) -> None:
        technology = hbm_technology("HBM3E")
        watts = memory_power_watts(
            technology,
            capacity_bytes=96e9,
            read_bytes_per_second=0.0,
            write_bytes_per_second=0.0,
        )
        self.assertAlmostEqual(watts, 96 * technology.background_power_mw_per_gb / 1e3)

    def test_traffic_adds_the_literature_reference_per_bit_energy(self) -> None:
        technology = hbm_technology("HBM3E")
        base = memory_power_watts(
            technology, capacity_bytes=0.0, read_bytes_per_second=0.0, write_bytes_per_second=0.0
        )
        one_tb_read = memory_power_watts(
            technology, capacity_bytes=0.0, read_bytes_per_second=1e12, write_bytes_per_second=0.0
        )
        expected = technology.read_energy_pj_per_bit * 1e12 * 8 * 1e-12
        self.assertAlmostEqual(one_tb_read - base, expected)

    def test_newer_generations_move_fewer_joules_per_bit(self) -> None:
        rates = [
            (name, technology.read_energy_pj_per_bit)
            for name, technology in HBM_TECHNOLOGIES.items()
        ]
        ordered = ["HBM2", "HBM2E", "HBM3", "HBM3E", "HBM4"]
        values = [dict(rates)[name] for name in ordered]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_energy_coefficients_link_their_distinct_source_and_assumptions(self) -> None:
        for technology in HBM_TECHNOLOGIES.values():
            self.assertEqual(
                technology.energy_source_url,
                "https://arxiv.org/pdf/2604.16007",
            )
        hbm3e = HBM_TECHNOLOGIES["HBM3E"]
        self.assertEqual(hbm3e.background_power_mw_per_gb, 75.0)
        self.assertIn("midpoint", hbm3e.energy_source_label)

    def test_negative_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            memory_power_watts(
                hbm_technology("HBM3"),
                capacity_bytes=-1.0,
                read_bytes_per_second=0.0,
                write_bytes_per_second=0.0,
            )


class ComputePowerTest(unittest.TestCase):
    def test_multiplier_energy_grows_quadratically_with_operand_width(self) -> None:
        shared = dict(multipliers=65536, clock_hz=1e9, array_active_fraction=1.0)
        four = compute_power_watts(mac_bits=4, **shared)
        eight = compute_power_watts(mac_bits=8, **shared)
        self.assertAlmostEqual(eight / four, 4.0)

    def test_power_scales_with_array_activity(self) -> None:
        shared = dict(multipliers=65536, clock_hz=1e9, mac_bits=4)
        self.assertAlmostEqual(
            compute_power_watts(array_active_fraction=0.5, **shared) * 2,
            compute_power_watts(array_active_fraction=1.0, **shared),
        )

    def test_activity_outside_the_unit_interval_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            compute_power_watts(
                multipliers=1024, clock_hz=1e9, mac_bits=4, array_active_fraction=1.5
            )


class CalibrationTest(unittest.TestCase):
    def test_reference_is_an_explicit_literature_model_output(self) -> None:
        self.assertEqual(
            REFERENCE_CONFIGURATION["source_url"],
            "https://arxiv.org/pdf/2604.16007",
        )
        self.assertEqual(
            REFERENCE_CONFIGURATION["source_scope"],
            "literature-reported analytic/synthesis model output; not measured silicon",
        )
        self.assertEqual(REFERENCE_CONFIGURATION["reference_clock_hz"], 1.0e9)
        self.assertEqual(REFERENCE_CONFIGURATION["array_active_fraction"], 1.0)
        self.assertEqual(REFERENCE_CONFIGURATION["read_fraction"], 0.95)
        self.assertEqual(REFERENCE_CONFIGURATION["write_fraction"], 0.05)
        self.assertIn("quadratic", REFERENCE_CONFIGURATION["compute_scaling_rule"])

    def test_stored_coefficient_matches_its_literature_reference(self) -> None:
        self.assertAlmostEqual(
            calibrate_reference_mac_energy(), REFERENCE_MAC_ENERGY_PJ, places=3
        )

    def test_reference_configuration_reproduces_its_target_power(self) -> None:
        reference = REFERENCE_CONFIGURATION
        technology = hbm_technology(str(reference["hbm_generation"]))
        stacks = int(reference["hbm_stacks"])
        peak = stacks * technology.peak_bandwidth_bytes_per_s_per_stack
        utilisation = float(reference["bandwidth_utilisation"])
        rows, columns = reference["array"]
        estimate = decode_power(
            technology,
            capacity_bytes=stacks * technology.capacity_gb_per_stack * 1e9,
            read_bytes_per_second=peak * utilisation * 0.95,
            write_bytes_per_second=peak * utilisation * 0.05,
            multipliers=rows * columns,
            clock_hz=1e9,
            mac_bits=int(reference["operand_bits"]),
            array_active_fraction=1.0,
            tokens_per_second=1.0,
        )
        # The stored coefficient is rounded to three significant figures, so
        # check the anchor is reproduced to within 0.1% rather than exactly.
        target = float(reference["reference_total_watts"])
        self.assertAlmostEqual(estimate.memory_watts, 87.1260672)
        self.assertLess(abs(estimate.total_watts - target) / target, 1e-3)


class EnergyEfficiencyTest(unittest.TestCase):
    def _estimate(self, tokens_per_second: float, chip_count: int = 1):
        return decode_power(
            hbm_technology("HBM3E"),
            capacity_bytes=48e9,
            read_bytes_per_second=1e12,
            write_bytes_per_second=1e10,
            multipliers=65536,
            clock_hz=1e9,
            mac_bits=4,
            array_active_fraction=1.0,
            tokens_per_second=tokens_per_second,
            chip_count=chip_count,
        )

    def test_tokens_per_joule_is_throughput_over_power(self) -> None:
        estimate = self._estimate(500.0)
        self.assertAlmostEqual(
            estimate.tokens_per_joule, 500.0 / estimate.total_watts
        )

    def test_replicating_chips_scales_power_but_not_the_token_rate(self) -> None:
        one = self._estimate(500.0, chip_count=1)
        four = self._estimate(500.0, chip_count=4)
        self.assertAlmostEqual(four.total_watts, one.total_watts * 4)
        self.assertAlmostEqual(four.tokens_per_joule, one.tokens_per_joule / 4)

    def test_memory_dominates_decode_power_at_this_operating_point(self) -> None:
        self.assertGreater(self._estimate(500.0).memory_fraction, 0.5)

    def test_structural_terms_are_separate_and_rankable(self) -> None:
        estimate = decode_power(
            hbm_technology("HBM3E"),
            capacity_bytes=48e9,
            read_bytes_per_second=1e12,
            write_bytes_per_second=1e10,
            multipliers=65536,
            clock_hz=1e9,
            mac_bits=4,
            array_active_fraction=0.5,
            tokens_per_second=500.0,
            chip_count=2,
            sram_read_bytes_per_second=2e11,
            sram_write_bytes_per_second=1e11,
            logic_area_mm2=20.0,
            link_bytes_per_second=1e10,
            token_latency_s=0.01,
        )
        self.assertGreater(estimate.sram_watts, 0.0)
        self.assertGreater(estimate.leakage_watts, 0.0)
        self.assertGreater(estimate.link_watts, 0.0)
        self.assertEqual(estimate.energy_tier, ANALYTIC_ENERGY_TIER)
        self.assertEqual(estimate.energy_id, analytic_energy_identity())
        self.assertAlmostEqual(
            estimate.edp_j_s,
            estimate.energy_per_token_j * 0.01,
        )

    def test_link_coefficients_have_one_canonical_owner(self) -> None:
        self.assertIs(LINK_ENERGY_PJ_PER_BIT, HANDOFF_LINK_ENERGY)


class SramEnergyCoefficientTest(unittest.TestCase):
    """Pin the vendored SRAM extraction the analytic energy identity hashes.

    ``SRAM_ACCESS_ENERGY_PJ_PER_BIT`` is derived at import from
    ``sram_energy_asap7_v1.json``.  Nothing else pins the macro census or the
    resulting median, so replacing that artifact would move the coefficient -
    and therefore the ``energy_id`` stamped on every priced row - without any
    test noticing.
    """

    def _table(self) -> dict:
        return json.loads(
            decode_power_module._SRAM_ENERGY_TABLE_PATH.read_text(encoding="utf-8")
        )

    def test_macro_census_is_pinned(self) -> None:
        table = self._table()
        self.assertEqual(table["macro_count"], 36)
        self.assertEqual(len(table["macros"]), 36)
        self.assertEqual(SRAM_ENERGY_SOURCE["statistic"], "median read pJ/bit over 36 macros")

    def test_median_read_energy_per_bit_is_pinned(self) -> None:
        self.assertAlmostEqual(
            SRAM_ACCESS_ENERGY_PJ_PER_BIT,
            0.047918,
            places=6,
        )
        self.assertEqual(
            SRAM_ACCESS_ENERGY_PJ_PER_BIT,
            decode_power_module._sram_read_pj_per_bit_median(),
        )

    def test_energy_identity_is_deterministic(self) -> None:
        self.assertEqual(analytic_energy_identity(), analytic_energy_identity())
        self.assertTrue(
            analytic_energy_identity().startswith("analytic-decode-energy-")
        )

    def test_energy_identity_moves_with_the_sram_table(self) -> None:
        # The committed artifact is never mutated: a perturbed copy is written
        # to a temporary directory and the derived coefficient is substituted
        # for the duration of the check.
        table = self._table()
        baseline = decode_power_module._sram_read_pj_per_bit_median()
        for macro in table["macros"]:
            entry = macro["extraction"]["read"]["entries"][0]
            entry["rise_power_mw"] = float(entry["rise_power_mw"]) * 2.0
        with tempfile.TemporaryDirectory() as directory:
            perturbed_path = Path(directory) / "sram_energy_perturbed.json"
            perturbed_path.write_text(json.dumps(table), encoding="utf-8")
            perturbed = decode_power_module._sram_read_pj_per_bit_median(perturbed_path)
        self.assertNotAlmostEqual(perturbed, baseline)
        identity = analytic_energy_identity()
        with mock.patch.object(
            decode_power_module,
            "SRAM_ACCESS_ENERGY_PJ_PER_BIT",
            perturbed,
        ):
            self.assertNotEqual(analytic_energy_identity(), identity)
        # The committed artifact and the identity it produces are unchanged.
        self.assertEqual(
            decode_power_module._sram_read_pj_per_bit_median(),
            baseline,
        )
        self.assertEqual(analytic_energy_identity(), identity)

    def test_energy_identity_moves_with_its_declared_source_scope(self) -> None:
        identity = analytic_energy_identity()
        replaced = dict(SRAM_ENERGY_SOURCE, source_artifact="sram_energy_v2.json")
        with mock.patch.object(decode_power_module, "SRAM_ENERGY_SOURCE", replaced):
            self.assertNotEqual(analytic_energy_identity(), identity)
        self.assertEqual(analytic_energy_identity(), identity)


def _gate_level_record() -> dict:
    """Read the area package's gate-level validation artifact."""

    artifact = (
        Path(__file__).resolve().parents[1]
        / "area"
        / "calibration"
        / "matrix_gate_level_validation.json"
    )
    return json.loads(artifact.read_text(encoding="utf-8"))


class GateLevelLeakageEvidenceTest(unittest.TestCase):
    """Pin the leakage decision, which is deliberately not the measured value.

    A gate-level campaign measured leakage roughly 54x below the declared
    coefficient, but at 25 C and on the compute array alone. The resolution is
    to keep the conservative declared value and carry the measurement as a
    scoped lower bound. That is a judgement, so it is pinned in both
    directions: the default must not silently drift down onto the optimistic
    25 C figure, and the measurement must not silently disappear.
    """

    def test_the_declared_coefficient_is_unchanged(self) -> None:
        self.assertEqual(LOGIC_LEAKAGE_W_PER_MM2, 0.05)
        self.assertEqual(
            logic_leakage_power_watts(logic_area_mm2=2.0),
            2.0 * 0.05,
        )

    def test_the_measurement_is_recorded_with_corner_and_scope(self) -> None:
        record = MATRIX_MACHINE_LEAKAGE_MEASUREMENT
        self.assertAlmostEqual(record["w_per_mm2"], 9.209669e-04, places=10)
        self.assertEqual(record["temperature_c"], 25)
        self.assertEqual(record["n_points"], 8)
        self.assertIn("PVT_0P7V_25C", record["corner"])
        self.assertIn("matrix_machine only", record["block_scope"])
        self.assertIn("not full-chip logic", record["block_scope"])
        self.assertEqual(record["evidence_tier"], "gate_level_dc_measured")

    def test_the_default_stays_above_the_measured_lower_bound(self) -> None:
        record = MATRIX_MACHINE_LEAKAGE_MEASUREMENT
        self.assertEqual(record["relation_to_default"], "lower_bound")
        self.assertGreater(LOGIC_LEAKAGE_W_PER_MM2, record["w_per_mm2"])
        self.assertAlmostEqual(
            record["default_over_measured_ratio"],
            LOGIC_LEAKAGE_W_PER_MM2 / record["w_per_mm2"],
            places=12,
        )
        self.assertGreater(record["default_over_measured_ratio"], 50.0)

    def test_the_declared_scope_names_both_gaps(self) -> None:
        reason = MATRIX_MACHINE_LEAKAGE_MEASUREMENT["not_adopted_because"]
        self.assertIn("temperature", reason)
        self.assertIn("whole-chip logic", reason)
        self.assertIs(LEAKAGE_SOURCE["measured_lower_bound"], MATRIX_MACHINE_LEAKAGE_MEASUREMENT)
        self.assertIn("not DC calibrated", LEAKAGE_SOURCE["evidence_scope"])

    def test_the_recorded_density_matches_the_measurement_artifact(self) -> None:
        stored = _gate_level_record()["leakage_density"]
        self.assertAlmostEqual(
            MATRIX_MACHINE_LEAKAGE_MEASUREMENT["w_per_mm2"],
            stored["w_per_mm2"],
            places=10,
        )
        self.assertEqual(
            MATRIX_MACHINE_LEAKAGE_MEASUREMENT["n_points"],
            stored["n_points"],
        )


class GateLevelComputeEnergyCrossCheckTest(unittest.TestCase):
    """Pin the independent corroboration of the compute anchor.

    The check is real evidence and is recorded as such, but it is a declared
    activity estimate. Both halves of that statement are load bearing: dropping
    the record loses the corroboration, and dropping the caveat would promote a
    consistency check into a calibration it is not.
    """

    def test_the_anchor_is_bracketed_at_a_consistent_toggle_rate(self) -> None:
        check = COMPUTE_ENERGY_CROSS_CHECK
        self.assertEqual(check["anchor_pj_per_mac"], REFERENCE_MAC_ENERGY_PJ)
        low, high = check["envelope_pj_per_mac"]
        self.assertLess(low, REFERENCE_MAC_ENERGY_PJ)
        self.assertGreater(high, REFERENCE_MAC_ENERGY_PJ)
        implied = check["implied_toggle_rate"]
        self.assertAlmostEqual(implied["MXFP_E1M2_32x4"], 0.0797, places=3)
        self.assertAlmostEqual(implied["MXFP_E1M2_16x4"], 0.0835, places=3)
        # Two independent geometries agreeing is what makes this a check rather
        # than a coincidence at one shape.
        self.assertLess(
            abs(implied["MXFP_E1M2_32x4"] - implied["MXFP_E1M2_16x4"]),
            0.01,
        )

    def test_the_cross_check_declares_its_activity_caveat_and_scope(self) -> None:
        check = COMPUTE_ENERGY_CROSS_CHECK
        self.assertIs(check["coefficient_changed"], False)
        self.assertEqual(
            check["evidence_tier"],
            "gate_level_declared_activity_estimate",
        )
        self.assertIn("declared-activity", check["caveat"])
        self.assertIn("matrix_machine only", check["block_scope"])
        self.assertIn("PVT_0P7V_25C", check["corner"])

    def test_it_did_not_move_the_compute_coefficient(self) -> None:
        self.assertEqual(REFERENCE_MAC_ENERGY_PJ, 0.203)
        self.assertAlmostEqual(
            calibrate_reference_mac_energy(),
            REFERENCE_MAC_ENERGY_PJ,
            places=3,
        )

    def test_it_agrees_with_the_derived_measurement_artifact(self) -> None:
        stored = _gate_level_record()["compute_energy_envelope"]
        self.assertEqual(
            stored["anchor_pj_per_mac"],
            COMPUTE_ENERGY_CROSS_CHECK["anchor_pj_per_mac"],
        )
        implied = stored["implied_toggle_rate_by_geometry"]
        for geometry, value in COMPUTE_ENERGY_CROSS_CHECK[
            "implied_toggle_rate"
        ].items():
            self.assertAlmostEqual(implied[geometry], value, places=3)

    def test_both_records_are_hashed_into_the_energy_identity(self) -> None:
        identity = analytic_energy_identity()
        replaced = dict(COMPUTE_ENERGY_CROSS_CHECK, coefficient_changed=True)
        with mock.patch.object(
            decode_power_module,
            "COMPUTE_ENERGY_CROSS_CHECK",
            replaced,
        ):
            self.assertNotEqual(analytic_energy_identity(), identity)
        weaker = dict(LEAKAGE_SOURCE, measured_lower_bound=None)
        with mock.patch.object(decode_power_module, "LEAKAGE_SOURCE", weaker):
            self.assertNotEqual(analytic_energy_identity(), identity)
        self.assertEqual(analytic_energy_identity(), identity)


class LeakageChoiceSensitivityTest(unittest.TestCase):
    """Show that the 54x coefficient gap is not a 54x power gap.

    The leakage decision is defensible partly because it barely moves any
    reported figure. If the leakage term ever becomes a large share of total
    decode power, the declared coefficient stops being a safe default and this
    test is where that shows up.
    """

    def _estimate(self, leakage_w_per_mm2: float) -> float:
        technology = hbm_technology("HBM3E")
        peak = technology.peak_bandwidth_bytes_per_s_per_stack
        logic_mm2 = 0.93
        chips = 8
        estimate = decode_power(
            technology,
            capacity_bytes=technology.capacity_gb_per_stack * 1e9,
            read_bytes_per_second=peak * 0.70 * 0.95,
            write_bytes_per_second=peak * 0.70 * 0.05,
            multipliers=1024 * 8,
            clock_hz=1.0e9,
            mac_bits=4,
            array_active_fraction=0.30,
            tokens_per_second=4000.0,
            chip_count=chips,
            logic_area_mm2=logic_mm2,
        )
        substituted = logic_leakage_power_watts(
            logic_area_mm2=logic_mm2,
            leakage_w_per_mm2=leakage_w_per_mm2,
        ) * chips
        return estimate.total_watts - estimate.leakage_watts + substituted

    def test_the_leakage_term_is_a_small_share_at_a_representative_point(
        self,
    ) -> None:
        declared = self._estimate(LOGIC_LEAKAGE_W_PER_MM2)
        measured = self._estimate(
            MATRIX_MACHINE_LEAKAGE_MEASUREMENT["w_per_mm2"]
        )
        # Adopting the optimistic 25 C density would lower total power, and by
        # well under one percent: the conservative default costs almost nothing.
        self.assertLess(measured, declared)
        self.assertLess((declared - measured) / declared, 0.01)


if __name__ == "__main__":
    unittest.main()
