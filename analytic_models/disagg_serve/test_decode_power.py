"""Checks for the decode power and energy-efficiency model."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from decode_power import (  # noqa: E402
    ANALYTIC_ENERGY_TIER,
    LINK_ENERGY_PJ_PER_BIT,
    REFERENCE_CONFIGURATION,
    REFERENCE_MAC_ENERGY_PJ,
    analytic_energy_identity,
    calibrate_reference_mac_energy,
    compute_power_watts,
    decode_power,
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


if __name__ == "__main__":
    unittest.main()
