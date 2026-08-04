from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from transactional_emulator.testbench.sim_env_utils import (
    _mx_quant_config,
    _mx_tensor_generator,
    _precision_for_tensor,
    map_mx_data_to_hbm_for_behave_sim,
)
from transactional_emulator.tools.packedkv_numeric import encode_mx, matrix_format


def _scale() -> dict:
    return {
        "type": "Fp",
        "sign": False,
        "exponent": 8,
        "mantissa": 0,
    }


def _mxfp(exponent: int, mantissa: int) -> dict:
    return {
        "format": "Mx",
        "block": 8,
        "ELEM": {
            "type": "Fp",
            "sign": True,
            "exponent": exponent,
            "mantissa": mantissa,
        },
        "SCALE": _scale(),
    }


def _mxint(width: int) -> dict:
    return {
        "format": "Mx",
        "block": 8,
        "ELEM": {"type": "Int", "width": width},
        "SCALE": _scale(),
    }


def _precision() -> dict:
    return {
        "HBM_M_WEIGHT_TYPE": _mxfp(4, 3),
        "HBM_M_KV_TYPE": _mxint(4),
        "HBM_V_ACT_TYPE": _mxfp(2, 1),
        "HBM_V_KV_TYPE": _mxint(8),
        "HBM_V_INT_TYPE": {
            "format": "Plain",
            "DATA_TYPE": {"type": "Int", "width": 32},
        },
    }


class SimulatorPrecisionImageTests(unittest.TestCase):
    def test_tensor_roles_select_distinct_precision_nodes(self) -> None:
        precision = _precision()
        self.assertEqual(
            _precision_for_tensor("W_Q", precision),
            precision["HBM_M_WEIGHT_TYPE"],
        )
        for name in ("K", "K_cache", "V", "V_cache"):
            self.assertEqual(
                _precision_for_tensor(name, precision),
                precision["HBM_M_KV_TYPE"],
            )
        self.assertEqual(
            _precision_for_tensor(
                "cache",
                precision,
                {"precision_role": "vector_kv"},
            ),
            precision["HBM_V_KV_TYPE"],
        )
        self.assertEqual(
            _precision_for_tensor("Q", precision),
            precision["HBM_V_ACT_TYPE"],
        )

    def test_quant_configs_preserve_family_and_element_width(self) -> None:
        precision = _precision()
        mxint = _mx_quant_config(_mxint(2), precision)
        self.assertEqual(mxint["format"], "mxint")
        self.assertEqual(mxint["element_width"], 2)
        self.assertEqual(mxint["block_size"], [1, 8])

        mxfp = _mx_quant_config(_mxfp(1, 2), precision)
        self.assertEqual(mxfp["format"], "mxfp")
        self.assertEqual(mxfp["element_width"], 4)
        self.assertEqual(mxfp["exp_width"], 1)
        self.assertEqual(mxfp["man_width"], 2)
        with self.assertRaisesRegex(ValueError, "MXINT element width 3"):
            _mx_quant_config(_mxint(3), precision)

    def test_mxint2_image_has_packed_elements_and_e8m0_scale(self) -> None:
        precision = _precision()
        config = _mx_quant_config(_mxint(2), precision)
        tensor = torch.tensor(
            [[-0.40, -0.30, -0.20, -0.10, 0.10, 0.20, 0.30, 0.40]],
            dtype=torch.float32,
        )
        generator = _mx_tensor_generator(
            shape=tuple(tensor.shape),
            quant_config=config,
            config_settings={},
            directory=None,
            filename=None,
        )
        blocks, scales = generator.quantize_tensor(tensor)
        with tempfile.TemporaryDirectory() as temporary:
            map_mx_data_to_hbm_for_behave_sim(
                blocks=blocks,
                element_width=config["element_width"],
                block_width=8,
                bias=scales,
                bias_width=8,
                directory=temporary,
                append=False,
                logical_row_elements=8,
                source_row_elements=8,
                logical_rows=1,
                source_rows=1,
            )
            payload = (
                Path(temporary) / "hbm_for_behave_sim.bin"
            ).read_bytes()
        self.assertEqual(payload[:3], bytes((0x0F, 0x50, 0x7F)))
        self.assertEqual(len(payload), 64)

    def test_mxfp_sweep_images_match_the_numeric_oracle(self) -> None:
        values = (3.0, -2.5, 1.0, -0.0, 0.0, 0.125, -0.125, 0.03125)
        tensor = torch.tensor([values], dtype=torch.float32)
        formats = {
            "E1M2": (1, 2),
            "E2M1": (2, 1),
            "E3M4": (3, 4),
            "E4M3": (4, 3),
            "E5M2": (5, 2),
        }

        for token, (exponent, mantissa) in formats.items():
            with self.subTest(token=token):
                precision = _precision()
                config = _mx_quant_config(_mxfp(exponent, mantissa), precision)
                generator = _mx_tensor_generator(
                    shape=tuple(tensor.shape),
                    quant_config=config,
                    config_settings={},
                    directory=None,
                    filename=None,
                )
                blocks, scales = generator.quantize_tensor(tensor)
                with tempfile.TemporaryDirectory() as temporary:
                    map_mx_data_to_hbm_for_behave_sim(
                        blocks=blocks,
                        element_width=config["element_width"],
                        block_width=8,
                        bias=scales,
                        bias_width=8,
                        directory=temporary,
                        append=False,
                        logical_row_elements=8,
                        source_row_elements=8,
                        logical_rows=1,
                        source_rows=1,
                    )
                    payload = (
                        Path(temporary) / "hbm_for_behave_sim.bin"
                    ).read_bytes()

                oracle = encode_mx(values, matrix_format(token))
                logical_element_bytes = (
                    8 * config["element_width"] + 7
                ) // 8
                self.assertEqual(
                    payload[:logical_element_bytes],
                    oracle.element_plane[:logical_element_bytes],
                )
                self.assertEqual(
                    payload[logical_element_bytes],
                    oracle.scale_plane[0],
                )


if __name__ == "__main__":
    unittest.main()
