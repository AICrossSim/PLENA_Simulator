"""Tests for the analytic-vs-emulator calibration artifact."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

from emulator_calibration import (  # noqa: E402
    EMULATOR_CALIBRATION_LABEL,
    EMULATOR_CALIBRATION_STAGES,
    EMULATOR_PRECISION_ROLES,
    EMULATOR_REQUIRED_PROVENANCE_ROLES,
    EMULATOR_STAGE_ERROR_LIMIT,
    EMULATOR_TOTAL_ERROR_LIMIT,
    EMULATOR_UNCOVERED_FRACTION_LIMIT,
    EmulatorCalibration,
    EmulatorExecutionContract,
    StageCalibration,
    calibration_source_hashes,
    describe_emulator_calibration,
    sha256_file,
    validate_calibration_sources,
)

DIGEST = "a" * 64


def execution_contract(**overrides):
    values = {
        "timing_mode": "rtl_serialized",
        "drain_overlapped": False,
        "fp_sram_depth": 512,
        "hbm_gen": "HBM2",
        "hbm_channels": 8,
        "precision": tuple(
            (name, '{"format":"test"}') for name in EMULATOR_PRECISION_ROLES
        ),
    }
    values.update(overrides)
    return EmulatorExecutionContract(**values)


def provenance(**overrides):
    digests = {role: DIGEST for role in EMULATOR_REQUIRED_PROVENANCE_ROLES}
    digests.update(overrides)
    return tuple(sorted(digests.items()))


def build(stages, uncovered, **overrides):
    cycle_counts = {stage: (100, 100) for stage in EMULATOR_CALIBRATION_STAGES}
    for stage, analytical, emulator in stages:
        cycle_counts[stage] = (analytical, emulator)
    return EmulatorCalibration(
        configuration="decoder_decode kv=128",
        stages=tuple(
            StageCalibration(stage, *cycle_counts[stage])
            for stage in EMULATOR_CALIBRATION_STAGES
        ),
        uncovered_cycles=uncovered,
        provenance_hashes=provenance(),
        execution_contract=execution_contract(),
        **overrides,
    )


class TestEmulatorCalibration(unittest.TestCase):
    def test_label_is_never_trace_calibrated(self):
        calibration = build([], 0)
        self.assertEqual(calibration.label, EMULATOR_CALIBRATION_LABEL)
        self.assertNotIn("trace", calibration.label)

    def test_uncovered_fraction_is_reported(self):
        calibration = build([], 25)
        self.assertAlmostEqual(calibration.uncovered_fraction, 25 / 725)
        self.assertEqual(calibration.measured_layer_cycles, 725)

    def test_excess_uncovered_cycles_fail_closed(self):
        calibration = build([], 150)
        self.assertFalse(calibration.passed)
        self.assertEqual(calibration.label, "uncalibrated")
        ok, reason = describe_emulator_calibration(calibration)
        self.assertFalse(ok)
        self.assertEqual(reason, "emulator_calibration_failed")

    def test_stage_error_beyond_limit_fails(self):
        calibration = build([("FFN (gate/up/down)", 200, 100)], 0)
        self.assertGreater(
            calibration.worst_stage_error,
            EMULATOR_STAGE_ERROR_LIMIT,
        )
        self.assertFalse(calibration.passed)

    def test_total_error_beyond_limit_fails(self):
        # The aggregate remains an independent secondary diagnostic even
        # though the tighter worst-stage gate will also reject this payload.
        calibration = build(
            [(stage, 120, 100) for stage in EMULATOR_CALIBRATION_STAGES], 0
        )
        self.assertGreater(abs(calibration.total_error), 0.15)
        self.assertFalse(calibration.passed)

    def test_missing_calibration_is_not_calibrated(self):
        ok, reason = describe_emulator_calibration(None)
        self.assertFalse(ok)
        self.assertEqual(reason, "missing_emulator_calibration")

    def test_round_trip_preserves_identity(self):
        calibration = build([("FFN (gate/up/down)", 105, 100)], 5)
        restored = EmulatorCalibration.from_dict(calibration.to_dict())
        self.assertEqual(restored.calibration_id, calibration.calibration_id)
        self.assertEqual(restored.uncovered_fraction, calibration.uncovered_fraction)

    def test_tampered_payload_is_rejected(self):
        payload = build([("FFN (gate/up/down)", 105, 100)], 5).to_dict()
        payload["stages"][0]["emulator_cycles"] = 999
        with self.assertRaises(ValueError):
            EmulatorCalibration.from_dict(payload)

    def test_provenance_digest_covers_an_external_file(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "op_stats.jsonl"
            path.write_text('{"aggregate":true}\n')
            digest = sha256_file(path)
            self.assertEqual(len(digest), 64)
            calibration = EmulatorCalibration(
                configuration="decoder_decode kv=128",
                stages=tuple(
                    StageCalibration(stage, 100, 100)
                    for stage in EMULATOR_CALIBRATION_STAGES
                ),
                uncovered_cycles=0,
                provenance_hashes=provenance(op_stats=digest),
                execution_contract=execution_contract(),
            )
            self.assertTrue(calibration.passed)

    def test_analytic_source_provenance_fails_closed(self):
        source_hashes = calibration_source_hashes()
        calibration = EmulatorCalibration(
            configuration="decoder_decode kv=128",
            stages=tuple(
                StageCalibration(stage, 100, 100)
                for stage in EMULATOR_CALIBRATION_STAGES
            ),
            uncovered_cycles=0,
            provenance_hashes=provenance(**source_hashes),
            execution_contract=execution_contract(),
        )
        validate_calibration_sources(calibration)

        name = next(iter(source_hashes))
        stale_hashes = dict(source_hashes)
        stale_hashes[name] = "0" * 64
        stale = EmulatorCalibration(
            configuration=calibration.configuration,
            stages=calibration.stages,
            uncovered_cycles=calibration.uncovered_cycles,
            provenance_hashes=provenance(**stale_hashes),
            execution_contract=calibration.execution_contract,
        )
        with self.assertRaisesRegex(ValueError, "stale analytic source"):
            validate_calibration_sources(stale)

        incomplete = tuple(
            item for item in provenance() if item[0] != next(iter(source_hashes))
        )
        with self.assertRaisesRegex(ValueError, "missing required provenance"):
            EmulatorCalibration(
                configuration=calibration.configuration,
                stages=calibration.stages,
                uncovered_cycles=0,
                provenance_hashes=incomplete,
                execution_contract=execution_contract(),
            )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "stale.json"
            path.write_text(json.dumps(stale.to_dict()))
            with self.assertRaisesRegex(ValueError, "stale analytic source"):
                EmulatorCalibration.load(path)

    def test_missing_stage_mutation_is_rejected(self):
        payload = build([], 0).to_dict()
        payload["stages"].pop()
        with self.assertRaisesRegex(ValueError, "canonical decode stages"):
            EmulatorCalibration.from_dict(payload)

    def test_validation_limits_are_non_negotiable(self):
        self.assertEqual(EMULATOR_STAGE_ERROR_LIMIT, 0.05)
        self.assertEqual(EMULATOR_TOTAL_ERROR_LIMIT, 0.15)
        self.assertEqual(EMULATOR_UNCOVERED_FRACTION_LIMIT, 0.01)
        mutations = (
            ("stage_error_limit", EMULATOR_STAGE_ERROR_LIMIT + 0.01),
            ("total_error_limit", EMULATOR_TOTAL_ERROR_LIMIT + 0.01),
            (
                "uncovered_fraction_limit",
                EMULATOR_UNCOVERED_FRACTION_LIMIT + 0.01,
            ),
        )
        for name, value in mutations:
            with self.subTest(name=name, path="constructor"):
                with self.assertRaisesRegex(ValueError, "canonical validation limits"):
                    build([], 0, **{name: value})
            with self.subTest(name=name, path="artifact"):
                payload = build([], 0).to_dict()
                payload[name] = value
                with self.assertRaisesRegex(ValueError, "canonical validation limits"):
                    EmulatorCalibration.from_dict(payload)

        payload = build([], 0).to_dict()
        payload.pop("stage_error_limit")
        with self.assertRaisesRegex(ValueError, "canonical validation limits"):
            EmulatorCalibration.from_dict(payload)

    def test_execution_contract_mutations_are_rejected(self):
        mutations = (
            ("drain_overlapped", True, "drain behavior"),
            ("fp_sram_depth", 0, "depth"),
            ("hbm_channels", 0, "HBM geometry"),
        )
        for name, value, message in mutations:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, message):
                    execution_contract(**{name: value})

        payload = build([], 0).to_dict()
        payload["execution_contract"]["precision"].pop(EMULATOR_PRECISION_ROLES[-1])
        with self.assertRaisesRegex(ValueError, "precision contract is incomplete"):
            EmulatorCalibration.from_dict(payload)

    def test_required_provenance_roles_are_unique_and_complete(self):
        calibration = build([], 0)
        self.assertIn("emulator_binary", EMULATOR_REQUIRED_PROVENANCE_ROLES)
        with self.assertRaisesRegex(ValueError, "must be unique"):
            EmulatorCalibration(
                configuration=calibration.configuration,
                stages=calibration.stages,
                uncovered_cycles=0,
                provenance_hashes=provenance() + (("op_stats", DIGEST),),
                execution_contract=execution_contract(),
            )

        payload = calibration.to_dict()
        payload["provenance_hashes"].pop("run_receipt")
        with self.assertRaisesRegex(ValueError, "missing required provenance"):
            EmulatorCalibration.from_dict(payload)

        payload = calibration.to_dict()
        payload["provenance_hashes"].pop("emulator_binary")
        with self.assertRaisesRegex(ValueError, "missing required provenance"):
            EmulatorCalibration.from_dict(payload)

    def test_duplicate_json_provenance_key_is_rejected_on_load(self):
        payload = json.dumps(build([], 0).to_dict())
        payload = payload.replace(
            '"op_stats": "' + DIGEST + '"',
            '"op_stats": "' + DIGEST + '", "op_stats": "' + DIGEST + '"',
            1,
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text(payload)
            with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
                EmulatorCalibration.load(path)

    def test_serialised_form_is_json(self):
        calibration = build([("FFN (gate/up/down)", 105, 100)], 5)
        json.dumps(calibration.to_dict(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
