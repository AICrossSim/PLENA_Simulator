from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from compiler.aten.packedkv_profile_hook import (
    PROFILE_SCHEMA,
    TARGET,
    _MATRIX_SEMANTICS,
    _canonical_bytes,
    _content_hash,
)
from transactional_emulator.tools.packedkv_profile_hook import (
    HookError,
    MX_PHYSICAL_SEMANTICS_SCHEMA,
    REQUEST_SCHEMA,
    _worker_compile,
    _write_trace_files,
    run_hook,
)


def _profile(*, weight: str = "MXINT4") -> dict:
    return {
        "schema_version": PROFILE_SCHEMA,
        "kind": "quantized",
        "weight_format": weight,
        "activation_format": "MXINT2",
        "key_format": "MXINT4",
        "value_format": "MXINT4",
        "vector_format": "FP_E3M2",
        "block_size": 8,
        "scale_format": "E8M0",
        "scale_bits": 8,
        "accumulator_rule": "plena_fixed16_16_accumulate_truncate",
        "output_rule": "truncate_to_vector_format",
        "matrix_semantics": dict(_MATRIX_SEMANTICS),
        "method": "rtn",
        "operator_coverage": {
            "weight": ["attention_linear", "ffn_linear"],
            "activation": ["attention_linear", "ffn_linear", "qk_matmul", "pv_matmul"],
            "kv": ["kv_cache", "qk_matmul", "pv_matmul"],
            "vector": [
                "input_rmsnorm",
                "post_attention_rmsnorm",
                "q_norm",
                "k_norm",
                "rope",
                "softmax",
                "silu_gate",
                "residual",
                "final_rmsnorm",
            ],
            "bf16": ["embedding", "lm_head"],
        },
    }


def _request(profile: dict) -> dict:
    value = {
        "schema_version": REQUEST_SCHEMA,
        "stage": "emulator",
        "manifest_hash": "1" * 64,
        "profile_id": "dqp-" + hashlib.sha256(_canonical_bytes(profile)).hexdigest(),
        "profile": profile,
        "target": dict(TARGET),
        "source_tree_sha256": "2" * 64,
        "hook_template_hash": "3" * 64,
        "environment_sha256": "4" * 64,
    }
    value["content_hash"] = _content_hash(value)
    return value


class PackedKVEmulatorHookTests(unittest.TestCase):
    def _write_request(self, root: Path, request: dict) -> Path:
        path = root / "request.json"
        path.write_bytes(_canonical_bytes(request) + b"\n")
        return path

    def test_compiler_worker_materializes_real_traces(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = self._write_request(root, _request(_profile()))
            bundle_path = root / "bundle.json"
            artifact_dir = root / "artifacts"
            _worker_compile(request_path, artifact_dir, bundle_path)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
            self.assertEqual(bundle["content_hash"], _content_hash(bundle))
            evidence_target = bundle["binding"]["evidence_target"]
            self.assertEqual(
                evidence_target["target_mode"],
                "simulator_compiler_emulator",
            )
            self.assertEqual(
                evidence_target["mxint2_activation_scope"],
                "emulator_only",
            )
            self.assertFalse(evidence_target["common_deployment_valid"])
            physical = bundle["binding"]["runtime_precision_contract"][
                "physical_semantics"
            ]
            self.assertEqual(
                physical["schema_version"],
                MX_PHYSICAL_SEMANTICS_SCHEMA,
            )
            for trace in [
                bundle["traces"]["linear"],
                bundle["traces"]["roundtrip"],
                *bundle["traces"]["attention"],
            ]:
                self.assertTrue(
                    trace["assembler_metrics"]["execution_opcode_coverage_valid"]
                )
                self.assertTrue(Path(trace["machine_path"]).is_file())
                self.assertTrue(Path(trace["hbm_path"]).is_file())
                self.assertTrue(Path(trace["vram_path"]).is_file())
            for trace in bundle["traces"]["attention"]:
                metrics = trace["compiler_metrics"]
                self.assertEqual(metrics["q_len"], 1)
                self.assertEqual(
                    metrics["cache_position"],
                    metrics["cache_tokens"] - 1,
                )
            settings = Path(bundle["settings_path"]).read_text(encoding="utf-8")
            self.assertIn('schema_version = "plena-matrix-semantics/v3"', settings)
            self.assertIn("physical_k_width = 1024", settings)
            self.assertIn('format = "FP_E3M2"', settings)
            self.assertIn('source_profile_schema = "decode-precision-profile/v4"', settings)
            self.assertIn(
                'schema_version = "plena-mx-physical-semantics/v2"',
                settings,
            )
            self.assertIn(
                'plane_order = ["element", "scale"]',
                settings,
            )

    def test_missing_binary_and_toolchain_write_no_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = self._write_request(root, _request(_profile()))
            result_path = root / "result.json"
            with mock.patch(
                "transactional_emulator.tools.packedkv_profile_hook.shutil.which",
                return_value=None,
            ):
                with self.assertRaisesRegex(HookError, "cargo is not installed"):
                    run_hook(
                        request_path,
                        result_path,
                        root / "artifacts",
                        emulator_binary=root / "missing-emulator",
                    )
            self.assertFalse(result_path.exists())

    def test_unsupported_profile_is_explicitly_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = self._write_request(
                root,
                _request(_profile(weight="MXINT2")),
            )
            result = run_hook(
                request_path,
                root / "result.json",
                root / "artifacts",
            )
            self.assertFalse(result["tests"][0]["passed"])
            self.assertEqual(
                result["tests"][0]["metrics"]["reason_code"],
                "unsupported_mxint_weight",
            )

    def test_existing_artifact_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request_path = self._write_request(
                root,
                _request(_profile(weight="MXINT2")),
            )
            result = run_hook(
                request_path,
                root / "result.json",
                root / "artifacts",
            )
            artifact_path = Path(result["artifacts"][0]["path"])
            artifact_path.write_bytes(b"tampered\n")
            with self.assertRaisesRegex(HookError, "artifact hash"):
                run_hook(
                    request_path,
                    root / "result.json",
                    root / "artifacts",
                )

    def test_execution_opcode_gate_fails_closed(self) -> None:
        cases = {
            "emulator_gap": "V_PS_V gp1, gp2, gp3\n",
            "rtl_decode_gap": "M_BMV gp1, gp2, gp3\n",
            "rtl_execute_gap": "M_TMV gp1, gp2, gp3\n",
            "writeout_imm_gap": "M_MM_WO gp1, gp2, 1\n",
            "dma_mode_gap": "H_PREFETCH_M gp1, gp2, a0, 2, 2\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, assembly in cases.items():
                with self.subTest(name=name):
                    with self.assertRaisesRegex(
                        HookError,
                        "unsupported execution contract",
                    ):
                        _write_trace_files(root, name, assembly)


if __name__ == "__main__":
    unittest.main()
