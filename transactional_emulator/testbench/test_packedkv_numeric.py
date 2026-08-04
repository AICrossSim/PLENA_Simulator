from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for _source_root in (
    _WORKSPACE_ROOT / "PLENA_Simulator" / "PLENA_Tools",
    _WORKSPACE_ROOT / "mase" / "src",
):
    if not _source_root.is_dir():
        raise RuntimeError(f"required quantizer source tree is missing: {_source_root}")
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from chop.nn.quantizers import mxfp_quantizer, mxint_quantizer
from chop.nn.quantizers.mxfp.fake import extract_mxfp_components
from chop.nn.quantizers.mxfp.meta import MXFPMeta
from chop.nn.quantizers.mxint_hardware import mxint_hardware
from chop.passes.module.transforms.gptq.quantize_dispatch import _build_mxfp_meta
from plena_quant.mxfp import _mx_fp_quantize_hardware
from plena_quant.mxint import _mx_int_quantize_hardware
from plena_quant.quantizer.mxfp import mxfp_quantizer as legacy_mxfp_quantizer

from transactional_emulator.tools.packedkv_numeric import (
    FloatFormat,
    MxImage,
    canonical_mxint_vectors,
    decode_mx,
    encode_float,
    encode_mx,
    matrix_accumulate_partials,
    matrix_format,
    truncate_float,
)


MXINT_SWEEP_WIDTHS = (2, 4, 8)
MXFP_SWEEP_FORMATS = {
    "E1M2": (1, 2),
    "E2M1": (2, 1),
    "E3M4": (3, 4),
    "E4M3": (4, 3),
    "E5M2": (5, 2),
}


def _raw_int32(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.flatten().view(torch.int32))


def _element_max_exponent(exponent_bits: int) -> int:
    bias = 1 if exponent_bits == 1 else 2 ** (exponent_bits - 1) - 1
    max_code = (
        2**exponent_bits - 1
        if exponent_bits == 1
        else 2**exponent_bits - 2
    )
    return max_code - bias


class PackedKVNumericOracleTests(unittest.TestCase):
    def test_mxfp_ocp_scale_is_shared_across_quantizers(self) -> None:
        maxima = (0.0, 3.0, math.ldexp(1.0, -120), math.ldexp(1.0, 200))
        blocks = torch.tensor(
            [[maximum] + [0.0] * 7 for maximum in maxima],
            dtype=torch.float64,
        )

        for token, (exponent_bits, mantissa_bits) in MXFP_SWEEP_FORMATS.items():
            with self.subTest(token=token):
                element_max_exponent = _element_max_exponent(exponent_bits)
                expected = tuple(
                    0
                    if maximum == 0.0
                    else min(
                        128,
                        max(
                            -127,
                            math.floor(math.log2(maximum))
                            - element_max_exponent,
                        ),
                    )
                    for maximum in maxima
                )

                meta = MXFPMeta(
                    block_size=8,
                    scale_exp_bits=8,
                    element_exp_bits=exponent_bits,
                    element_frac_bits=mantissa_bits,
                    element_is_finite=(exponent_bits == 1),
                    round_mode="rn",
                )
                mase_scales, _ = extract_mxfp_components(blocks, meta)
                _, _, _, tools_scale_codes = _mx_fp_quantize_hardware(
                    blocks,
                    width=1 + exponent_bits + mantissa_bits,
                    exponent_width=exponent_bits,
                    exponent_bias_width=8,
                    block_size=[8],
                )
                oracle = encode_mx(
                    tuple(float(value) for value in blocks.flatten()),
                    matrix_format(token),
                )

                self.assertEqual(
                    tuple(int(value) for value in mase_scales.flatten()),
                    expected,
                )
                self.assertEqual(
                    tuple(int(value) - 127 for value in tools_scale_codes.flatten()),
                    expected,
                )
                self.assertEqual(
                    tuple(int(value) - 127 for value in oracle.scale_plane[:4]),
                    expected,
                )

                gptq_meta = _build_mxfp_meta(
                    {
                        "weight_block_size": 8,
                        "weight_exponent_width": exponent_bits,
                        "weight_frac_width": mantissa_bits,
                    }
                )
                self.assertEqual(gptq_meta.element_is_finite, exponent_bits == 1)
                self.assertEqual(
                    gptq_meta.element_max_exponent,
                    element_max_exponent,
                )

    def test_all_sweep_formats_are_idempotent(self) -> None:
        values = (3.0, -2.5, 1.0, -0.0, 0.0, 0.125, -0.125, 0.03125)
        source = torch.tensor([values], dtype=torch.float32)

        for width in MXINT_SWEEP_WIDTHS:
            with self.subTest(token=f"MXINT{width}"):
                fmt = matrix_format(f"MXINT{width}")
                first_image = encode_mx(values, fmt)
                second_image = encode_mx(decode_mx(first_image, fmt), fmt)
                self.assertEqual(first_image.payload, second_image.payload)

                mase_first = mxint_quantizer(
                    source, block_size=8, element_bits=width, block_dim=-1
                )
                mase_second = mxint_quantizer(
                    mase_first, block_size=8, element_bits=width, block_dim=-1
                )
                self.assertEqual(_raw_int32(mase_first), _raw_int32(mase_second))
                compatibility = mxint_hardware(
                    source,
                    {"width": width, "exponent_width": 8},
                    [1, 8],
                )
                self.assertEqual(_raw_int32(compatibility), _raw_int32(mase_first))

                tools_first, _, _ = _mx_int_quantize_hardware(
                    source,
                    width=width,
                    exponent_width=8,
                    block_size=[8],
                    skip_first_dim=False,
                )
                tools_second, _, _ = _mx_int_quantize_hardware(
                    tools_first,
                    width=width,
                    exponent_width=8,
                    block_size=[8],
                    skip_first_dim=False,
                )
                self.assertEqual(_raw_int32(tools_first), _raw_int32(tools_second))

        for token, (exponent_bits, mantissa_bits) in MXFP_SWEEP_FORMATS.items():
            with self.subTest(token=token):
                fmt = matrix_format(token)
                first_image = encode_mx(values, fmt)
                second_image = encode_mx(decode_mx(first_image, fmt), fmt)
                self.assertEqual(first_image.payload, second_image.payload)

                mase_first = mxfp_quantizer(
                    source,
                    block_size=8,
                    element_exp_bits=exponent_bits,
                    element_frac_bits=mantissa_bits,
                    block_dim=-1,
                )
                mase_second = mxfp_quantizer(
                    mase_first,
                    block_size=8,
                    element_exp_bits=exponent_bits,
                    element_frac_bits=mantissa_bits,
                    block_dim=-1,
                )
                self.assertEqual(_raw_int32(mase_first), _raw_int32(mase_second))

                tools_first, _, _, _ = _mx_fp_quantize_hardware(
                    source,
                    width=1 + exponent_bits + mantissa_bits,
                    exponent_width=exponent_bits,
                    exponent_bias_width=8,
                    block_size=[8],
                )
                tools_second, _, _, _ = _mx_fp_quantize_hardware(
                    tools_first,
                    width=1 + exponent_bits + mantissa_bits,
                    exponent_width=exponent_bits,
                    exponent_bias_width=8,
                    block_size=[8],
                )
                self.assertEqual(_raw_int32(tools_first), _raw_int32(tools_second))
                compatibility = legacy_mxfp_quantizer(
                    source,
                    width=1 + exponent_bits + mantissa_bits,
                    exponent_width=exponent_bits,
                    exponent_bias_width=8,
                    block_size=8,
                )
                self.assertEqual(_raw_int32(compatibility), _raw_int32(tools_first))

                differentiable = source.clone().requires_grad_()
                legacy_mxfp_quantizer(
                    differentiable,
                    width=1 + exponent_bits + mantissa_bits,
                    exponent_width=exponent_bits,
                    exponent_bias_width=8,
                    block_size=8,
                ).sum().backward()
                self.assertTrue(torch.equal(differentiable.grad, torch.ones_like(source)))

    def test_mxint_raw_bits_agree_with_explicit_signed_zero_exception(self) -> None:
        positive_zero_bits = 0
        negative_zero_bits = -(1 << 31)

        for width in MXINT_SWEEP_WIDTHS:
            with self.subTest(width=width):
                magnitude_bits = width - 1
                unit = 1.0 / (1 << magnitude_bits)
                qmax = ((1 << magnitude_bits) - 1) * unit
                values = (unit, -unit, qmax, -qmax, 0.0, -0.0, 0.0, 0.0)
                source = torch.tensor([values], dtype=torch.float32)
                fmt = matrix_format(f"MXINT{width}")

                oracle = torch.tensor(
                    decode_mx(encode_mx(values, fmt), fmt),
                    dtype=torch.float32,
                )
                tools, _, _ = _mx_int_quantize_hardware(
                    source,
                    width=width,
                    exponent_width=8,
                    block_size=[8],
                    skip_first_dim=False,
                )
                mase = mxint_quantizer(
                    source, block_size=8, element_bits=width, block_dim=-1
                )
                compatibility = mxint_hardware(
                    source,
                    {"width": width, "exponent_width": 8},
                    [1, 8],
                )

                oracle_bits = _raw_int32(oracle)
                tools_bits = _raw_int32(tools)
                mase_bits = _raw_int32(mase)
                self.assertEqual(oracle_bits, tools_bits)
                self.assertEqual(oracle_bits[:4], mase_bits[:4])
                self.assertEqual(oracle_bits[4:], (positive_zero_bits,) * 4)
                self.assertEqual(tools_bits[4:], (positive_zero_bits,) * 4)
                self.assertEqual(
                    mase_bits[4:],
                    (
                        positive_zero_bits,
                        negative_zero_bits,
                        positive_zero_bits,
                        positive_zero_bits,
                    ),
                )
                self.assertEqual(_raw_int32(compatibility), mase_bits)

    def test_mxint_physical_vectors_are_canonical(self) -> None:
        vectors = canonical_mxint_vectors()
        expected = {
            "2": "dd00",
            "4": "91f70000",
            "8": "01817fff00000000",
        }
        self.assertEqual(vectors["physical_semantics_id"], "plena-mx-physical-semantics/v2")
        self.assertEqual(
            vectors["mxint_scale_rule"],
            "ceil_log2_max_abs_over_qmax_fraction",
        )
        for width, element_hex in expected.items():
            self.assertEqual(vectors["widths"][width]["element_hex"], element_hex)
            self.assertEqual(vectors["widths"][width]["scale_code"], 127)
            self.assertTrue(vectors["widths"][width]["canonical_zero"])
        self.assertTrue(vectors["maximum_e8m0_finite"])
        self.assertTrue(vectors["zero_times_maximum_scale_is_zero"])

    def test_range_safe_mxint_is_byte_idempotent(self) -> None:
        values = (1.0, -1.0, 0.25, -0.25, 0.0, -0.0, 0.0, 0.0)
        for width in (2, 4, 8):
            with self.subTest(width=width):
                fmt = matrix_format(f"MXINT{width}")
                first = encode_mx(values, fmt)
                decoded = decode_mx(first, fmt)
                second = encode_mx(decoded, fmt)
                self.assertEqual(first.payload, second.payload)
                self.assertEqual(decoded[0], 1.0)
                self.assertEqual(decoded[1], -1.0)

    def test_planes_are_independently_aligned(self) -> None:
        values = tuple(float(index % 3 - 1) for index in range(48))
        image = encode_mx(
            values,
            matrix_format("MXINT4"),
            hbm_row_bytes=32,
        )
        self.assertEqual(len(image.element_plane), 32)
        self.assertEqual(len(image.scale_plane), 32)
        self.assertEqual(len(image.payload), 64)

    def test_maximum_e8m0_code_does_not_create_nan(self) -> None:
        image = MxImage(
            element_plane=bytes([0x01]).ljust(32, b"\0"),
            scale_plane=bytes([255]).ljust(32, b"\0"),
            element_count=8,
            block_size=8,
            row_bytes=32,
        )
        values = decode_mx(image, matrix_format("MXINT2"))
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertTrue(all(value == 0.0 for value in values[1:]))

    def test_minifloat_halfway_uses_even_code(self) -> None:
        fmt = FloatFormat(3, 2)
        lower = 1.0
        upper = 1.25
        self.assertEqual(encode_float((lower + upper) / 2.0, fmt) & 0b11, 0)

    def test_matrix_partials_round_before_fixed_accumulation(self) -> None:
        fmt = FloatFormat(3, 2)
        observed = matrix_accumulate_partials((0.5, -0.54), fmt)
        collapsed = matrix_accumulate_partials((0.5 - 0.54,), fmt)
        self.assertEqual(observed, 0.0)
        self.assertNotEqual(observed, collapsed)

    def test_matrix_writeout_truncates_magnitude(self) -> None:
        fmt = FloatFormat(3, 2)
        self.assertEqual(truncate_float(1.219, fmt), 1.0)
        self.assertEqual(truncate_float(-1.219, fmt), -1.0)

    def test_fixed_bank_wraparound_matches_signed_32_bit(self) -> None:
        fmt = FloatFormat(8, 7, saturating=False)
        observed = matrix_accumulate_partials(
            (32_704.0, 128.0),
            fmt,
        )
        self.assertEqual(observed, -32_640.0)


if __name__ == "__main__":
    unittest.main()
