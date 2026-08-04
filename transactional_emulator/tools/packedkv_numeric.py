"""Dependency-light numerical and physical-layout oracles for PLENA evidence."""

from __future__ import annotations

import bisect
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

MX_PHYSICAL_SEMANTICS_ID = "plena-mx-physical-semantics/v2"
MXINT_SCALE_RULE = "ceil_log2_max_abs_over_qmax_fraction"


@dataclass(frozen=True)
class FloatFormat:
    exponent: int
    mantissa: int
    signed: bool = True
    saturating: bool = True

    @property
    def bits(self) -> int:
        return int(self.signed) + self.exponent + self.mantissa

    @property
    def bias(self) -> int:
        return 1 if self.exponent == 1 else (1 << (self.exponent - 1)) - 1

    @property
    def max_finite_exponent_code(self) -> int:
        mask = (1 << self.exponent) - 1
        if self.saturating and self.exponent == 1:
            return mask
        return mask - 1


@dataclass(frozen=True)
class MatrixFormat:
    token: str
    family: str
    element_bits: int
    exponent: int | None = None
    mantissa: int | None = None


@dataclass(frozen=True)
class MxImage:
    element_plane: bytes
    scale_plane: bytes
    element_count: int
    block_size: int
    row_bytes: int

    @property
    def payload(self) -> bytes:
        return self.element_plane + self.scale_plane


def matrix_format(token: str) -> MatrixFormat:
    if token.startswith("MXINT"):
        width = int(token[5:])
        if width not in (2, 4, 8):
            raise ValueError("PLENA MXINT supports 2-, 4-, and 8-bit elements")
        return MatrixFormat(token, "mxint", width)
    if not token.startswith("E") or "M" not in token:
        raise ValueError(f"invalid matrix format {token!r}")
    exponent_text, mantissa_text = token[1:].split("M", 1)
    exponent = int(exponent_text)
    mantissa = int(mantissa_text)
    return MatrixFormat(
        token,
        "mxfp",
        1 + exponent + mantissa,
        exponent,
        mantissa,
    )


def vector_format(token: str) -> FloatFormat:
    if token == "BF16":
        return FloatFormat(8, 7, saturating=False)
    if not token.startswith("FP_E") or "M" not in token:
        raise ValueError(f"invalid vector format {token!r}")
    exponent_text, mantissa_text = token[4:].split("M", 1)
    return FloatFormat(int(exponent_text), int(mantissa_text))


def _pack_codes(codes: Iterable[int], width: int) -> bytes:
    values = tuple(int(value) for value in codes)
    if width <= 0:
        raise ValueError("code width must be positive")
    result = bytearray((len(values) * width + 7) // 8)
    mask = (1 << width) - 1
    bit_offset = 0
    for value in values:
        value &= mask
        for bit in range(width):
            if value & (1 << bit):
                position = bit_offset + bit
                result[position // 8] |= 1 << (position % 8)
        bit_offset += width
    return bytes(result)


def _unpack_codes(payload: bytes, count: int, width: int) -> tuple[int, ...]:
    if len(payload) * 8 < count * width:
        raise ValueError("packed payload is shorter than its logical code count")
    values: list[int] = []
    bit_offset = 0
    for _ in range(count):
        value = 0
        for bit in range(width):
            position = bit_offset + bit
            if payload[position // 8] & (1 << (position % 8)):
                value |= 1 << bit
        values.append(value)
        bit_offset += width
    return tuple(values)


def decode_float(code: int, fmt: FloatFormat) -> float:
    mantissa_mask = (1 << fmt.mantissa) - 1
    exponent_mask = (1 << fmt.exponent) - 1
    mantissa = code & mantissa_mask
    exponent = (code >> fmt.mantissa) & exponent_mask
    sign = -1.0 if fmt.signed and (code >> (fmt.exponent + fmt.mantissa)) & 1 else 1.0
    if exponent == 0:
        if mantissa == 0:
            return math.copysign(0.0, sign)
        magnitude = (
            mantissa
            / (1 << fmt.mantissa)
            * math.ldexp(1.0, 1 - fmt.bias)
        )
        return sign * magnitude
    if (
        not fmt.saturating
        and fmt.exponent != 1
        and exponent == exponent_mask
    ):
        return math.copysign(
            math.inf if mantissa == 0 else math.nan,
            sign,
        )
    magnitude = (
        1.0 + mantissa / (1 << fmt.mantissa)
    ) * math.ldexp(1.0, exponent - fmt.bias)
    return sign * magnitude


@lru_cache(maxsize=None)
def _positive_float_lattice(fmt: FloatFormat) -> tuple[tuple[float, ...], tuple[int, ...]]:
    max_code = (
        fmt.max_finite_exponent_code << fmt.mantissa
    ) | ((1 << fmt.mantissa) - 1)
    pairs = tuple(
        (decode_float(code, fmt), code)
        for code in range(max_code + 1)
        if math.isfinite(decode_float(code, fmt))
    )
    pairs = tuple(sorted(pairs))
    return tuple(item[0] for item in pairs), tuple(item[1] for item in pairs)


def encode_float(value: float, fmt: FloatFormat) -> int:
    if not math.isfinite(value):
        raise ValueError("PLENA evidence inputs must be finite")
    sign_bit = int(math.copysign(1.0, value) < 0.0)
    magnitude = abs(float(value))
    values, codes = _positive_float_lattice(fmt)
    index = bisect.bisect_left(values, magnitude)
    if index <= 0:
        code = codes[0]
    elif index >= len(values):
        code = codes[-1]
    else:
        lower_value, upper_value = values[index - 1], values[index]
        lower_code, upper_code = codes[index - 1], codes[index]
        lower_error = magnitude - lower_value
        upper_error = upper_value - magnitude
        if lower_error < upper_error:
            code = lower_code
        elif upper_error < lower_error:
            code = upper_code
        else:
            code = lower_code if lower_code & 1 == 0 else upper_code
    if sign_bit:
        code |= 1 << (fmt.exponent + fmt.mantissa)
    return code


def round_float(value: float, fmt: FloatFormat) -> float:
    return decode_float(encode_float(value, fmt), fmt)


def truncate_float(value: float, fmt: FloatFormat) -> float:
    """Truncate one finite value to the RTL matrix writeout format."""

    if not math.isfinite(value):
        raise ValueError("PLENA matrix writeout must be finite")
    if value == 0.0:
        return value
    magnitude = abs(float(value))
    minimum_exponent = 1 - fmt.bias
    maximum_exponent = fmt.max_finite_exponent_code - fmt.bias
    normal_floor = math.ldexp(1.0, minimum_exponent)
    exponent = min(
        maximum_exponent,
        max(minimum_exponent, math.floor(math.log2(magnitude))),
    )
    step = math.ldexp(
        1.0,
        (
            minimum_exponent
            if magnitude < normal_floor
            else exponent
        )
        - fmt.mantissa,
    )
    maximum = (
        2.0 - math.ldexp(1.0, -fmt.mantissa)
    ) * math.ldexp(1.0, maximum_exponent)
    truncated = min(maximum, math.floor(magnitude / step) * step)
    return math.copysign(truncated, value)


def _fixed16_16(value: float) -> int:
    if not math.isfinite(value):
        raise ValueError("PLENA fixed-bank input must be finite")
    bits = math.trunc(value * 65_536.0) % (1 << 32)
    return bits - (1 << 32) if bits >= (1 << 31) else bits


def matrix_accumulate_partials(
    partials: Sequence[float],
    fmt: FloatFormat,
) -> float:
    """Apply per-MM_IC storage rounding and signed 16.16 accumulation."""

    if not partials:
        raise ValueError("matrix accumulation requires at least one partial")
    accumulator = 0
    for partial in partials:
        incoming = _fixed16_16(round_float(float(partial), fmt))
        bits = (accumulator + incoming) % (1 << 32)
        accumulator = bits - (1 << 32) if bits >= (1 << 31) else bits
    return truncate_float(accumulator / 65_536.0, fmt)


def encode_plain(values: Sequence[float], fmt: FloatFormat) -> bytes:
    return _pack_codes((encode_float(value, fmt) for value in values), fmt.bits)


def decode_plain(payload: bytes, count: int, fmt: FloatFormat) -> tuple[float, ...]:
    return tuple(
        decode_float(code, fmt)
        for code in _unpack_codes(payload, count, fmt.bits)
    )


def encode_vector_rows(
    rows: Sequence[Sequence[float]],
    fmt: FloatFormat,
    row_elements: int,
) -> bytes:
    if row_elements * fmt.bits % 8:
        raise ValueError("vector rows must end on byte boundaries")
    result = bytearray()
    for row in rows:
        if len(row) != row_elements:
            raise ValueError("vector row has an unexpected logical width")
        result.extend(encode_plain(row, fmt))
    return bytes(result)


def decode_vector_row(
    payload: bytes,
    row_index: int,
    fmt: FloatFormat,
    row_elements: int,
) -> tuple[float, ...]:
    row_bytes = row_elements * fmt.bits // 8
    start = row_index * row_bytes
    end = start + row_bytes
    if end > len(payload):
        raise ValueError("vector dump does not contain the requested row")
    return decode_plain(payload[start:end], row_elements, fmt)


def _round_ties_even_nonnegative(value: float) -> int:
    floor = math.floor(value)
    fraction = value - floor
    if fraction > 0.5 or (fraction == 0.5 and floor & 1):
        return floor + 1
    return floor


def _mxint_code(value: float, width: int) -> int:
    magnitude_bits = width - 1
    magnitude_max = (1 << magnitude_bits) - 1
    magnitude = min(
        magnitude_max,
        _round_ties_even_nonnegative(
            abs(value) * (1 << magnitude_bits)
        ),
    )
    sign = int(value < 0.0 and magnitude != 0)
    return (sign << magnitude_bits) | magnitude


def _mxint_value(code: int, width: int) -> float:
    magnitude_bits = width - 1
    magnitude = code & ((1 << magnitude_bits) - 1)
    if magnitude == 0:
        return 0.0
    sign = -1.0 if code >> magnitude_bits else 1.0
    return sign * magnitude / (1 << magnitude_bits)


def _align(byte_count: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError("alignment must be positive")
    return (byte_count + alignment - 1) // alignment * alignment


def encode_mx(
    values: Sequence[float],
    fmt: MatrixFormat,
    *,
    block_size: int = 8,
    hbm_row_bytes: int = 32,
) -> MxImage:
    if block_size != 8:
        raise ValueError("PLENA native MX block size is 8")
    if len(values) % block_size:
        raise ValueError("MX payloads must contain complete native blocks")
    element_format = (
        FloatFormat(int(fmt.exponent), int(fmt.mantissa))
        if fmt.family == "mxfp"
        else None
    )
    element_codes: list[int] = []
    scale_codes: list[int] = []
    for start in range(0, len(values), block_size):
        block = tuple(float(value) for value in values[start:start + block_size])
        if not all(math.isfinite(value) for value in block):
            raise ValueError("MX inputs must be finite")
        maximum = max(abs(value) for value in block)
        if maximum == 0.0:
            exponent = 0
            scale_codes.append(127)
            element_codes.extend([0] * block_size)
            continue
        if fmt.family == "mxint":
            qmax = (
                ((1 << (fmt.element_bits - 1)) - 1)
                / (1 << (fmt.element_bits - 1))
            )
            raw_exponent = math.ceil(math.log2(maximum / qmax))
        else:
            # OCP MX scale placement uses the complete finite exponent range
            # of the element format. The encoded E8M0 scale is consumed as
            # data, so no datapath-specific scale convention is implied.
            element_max_exponent = (
                element_format.max_finite_exponent_code - element_format.bias
            )
            raw_exponent = (
                math.floor(math.log2(maximum)) - element_max_exponent
            )
        exponent = min(128, max(-127, int(raw_exponent)))
        scale_codes.append(exponent + 127)
        scaled = tuple(math.ldexp(value, -exponent) for value in block)
        if fmt.family == "mxint":
            element_codes.extend(
                _mxint_code(value, fmt.element_bits)
                for value in scaled
            )
        else:
            element_codes.extend(
                encode_float(value, element_format)
                for value in scaled
            )
    logical_element_bytes = _pack_codes(element_codes, fmt.element_bits)
    logical_scale_bytes = bytes(scale_codes)
    element_plane = logical_element_bytes.ljust(
        _align(len(logical_element_bytes), hbm_row_bytes),
        b"\0",
    )
    scale_plane = logical_scale_bytes.ljust(
        _align(len(logical_scale_bytes), hbm_row_bytes),
        b"\0",
    )
    return MxImage(
        element_plane=element_plane,
        scale_plane=scale_plane,
        element_count=len(values),
        block_size=block_size,
        row_bytes=hbm_row_bytes,
    )


def decode_mx(image: MxImage, fmt: MatrixFormat) -> tuple[float, ...]:
    logical_element_bytes = (
        image.element_count * fmt.element_bits + 7
    ) // 8
    element_codes = _unpack_codes(
        image.element_plane[:logical_element_bytes],
        image.element_count,
        fmt.element_bits,
    )
    scale_count = image.element_count // image.block_size
    scale_codes = image.scale_plane[:scale_count]
    values: list[float] = []
    element_format = (
        FloatFormat(int(fmt.exponent), int(fmt.mantissa))
        if fmt.family == "mxfp"
        else None
    )
    for block_index, scale_code in enumerate(scale_codes):
        multiplier = math.ldexp(1.0, int(scale_code) - 127)
        start = block_index * image.block_size
        for code in element_codes[start:start + image.block_size]:
            scaled = (
                _mxint_value(code, fmt.element_bits)
                if fmt.family == "mxint"
                else decode_float(code, element_format)
            )
            value = scaled * multiplier
            if fmt.family == "mxint" and scaled == 0.0:
                value = 0.0
            values.append(value)
    return tuple(values)


def mx_round(values: Sequence[float], fmt: MatrixFormat) -> tuple[float, ...]:
    return decode_mx(encode_mx(values, fmt), fmt)


def place_image(hbm: bytearray, base: int, image: MxImage) -> None:
    payload = image.payload
    end = base + len(payload)
    if base < 0 or end > len(hbm):
        raise ValueError("MX image does not fit its HBM reservation")
    hbm[base:end] = payload


def canonical_mxint_vectors() -> dict[str, object]:
    widths: dict[str, object] = {}
    for width in (2, 4, 8):
        magnitude_bits = width - 1
        unit = 1.0 / (1 << magnitude_bits)
        qmax = ((1 << magnitude_bits) - 1) * unit
        values = (
            unit,
            -unit,
            qmax,
            -qmax,
            0.0,
            -0.0,
            0.0,
            0.0,
        )
        image = encode_mx(values, matrix_format(f"MXINT{width}"))
        decoded = decode_mx(image, matrix_format(f"MXINT{width}"))
        widths[str(width)] = {
            "element_hex": image.element_plane[:width].hex(),
            "scale_code": image.scale_plane[0],
            "canonical_zero": all(
                struct.pack(">d", value) == struct.pack(">d", 0.0)
                for value in decoded[4:]
            ),
            "decoded": list(decoded),
        }
    maximum_scale = MxImage(
        element_plane=bytes([0x01, 0x00]).ljust(32, b"\0"),
        scale_plane=bytes([255]).ljust(32, b"\0"),
        element_count=8,
        block_size=8,
        row_bytes=32,
    )
    maximum_decoded = decode_mx(maximum_scale, matrix_format("MXINT2"))
    return {
        "physical_semantics_id": MX_PHYSICAL_SEMANTICS_ID,
        "mxint_scale_rule": MXINT_SCALE_RULE,
        "widths": widths,
        "maximum_e8m0_code": 255,
        "maximum_e8m0_value": maximum_decoded[0],
        "maximum_e8m0_finite": all(math.isfinite(value) for value in maximum_decoded),
        "zero_times_maximum_scale_is_zero": all(
            value == 0.0 for value in maximum_decoded[1:]
        ),
    }
