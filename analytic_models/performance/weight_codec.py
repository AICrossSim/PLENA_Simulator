"""NVFP4 block16 storage and finite decode service; no free dequantization.

Packets contain 32 output rows by K (K padded to 32). The decode candidate
transposes into the existing Matrix-view packet using fixed addresses. Its
throughput is a budget to sweep, not a synthesized hardware measurement.
"""

from dataclasses import dataclass
from math import ceil
import numpy as np

from .ltile_platform import CodecResources


def decode_e4m3_scale(bits):
    """E4M3FN (bias 7): exponent 15 is finite except mantissa 7."""
    bits = np.asarray(bits, dtype=np.uint8)
    exponent = (bits >> 3) & 15
    mantissa = bits & 7
    values = np.where(
        exponent == 0,
        mantissa.astype(np.float32) * 2**-9,
        (1 + mantissa.astype(np.float32) / 8) * np.exp2(exponent.astype(np.float32) - 7),
    )
    values = np.where((exponent == 15) & (mantissa == 7), np.nan, values)
    return np.copysign(values, np.where(bits & 128, -1.0, 1.0)).astype(np.float32)


def decode_nvfp4(packed, scales, global_scale):
    packed = np.asarray(packed)
    scales = np.asarray(scales, dtype=np.float32)
    if packed.dtype != np.uint8 or packed.ndim != 2 or packed.shape[1] % 8:
        raise ValueError("complete NVFP4 block16 rows required")
    if scales.shape != (packed.shape[0], packed.shape[1] // 8):
        raise ValueError("block16 scale shape mismatch")
    if not np.isfinite(scales).all() or not np.isfinite(global_scale):
        raise ValueError("nonfinite scales")
    table = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    out = np.empty((packed.shape[0], packed.shape[1] * 2), np.float32)
    out[:, ::2], out[:, 1::2] = table[packed & 15], table[packed >> 4]
    out *= np.repeat(scales * np.float32(global_scale), 16, axis=1)
    return out.T.copy()


@dataclass(frozen=True)
class WeightPacket:
    elements: int
    block: int = 16

    def __post_init__(self):
        if self.block != 16 or self.elements < 1 or self.elements % 32:
            raise ValueError("NVFP4 packets require complete padded 32-element words")

    @property
    def payload_bytes(self):
        return self.elements // 2 + self.elements // self.block

    @property
    def transfer_bytes(self):
        return ceil(self.payload_bytes / 64) * 64

    def service(self, resource=CodecResources()):
        if self.transfer_bytes > resource.input_bytes or 2 * self.elements > resource.output_bytes:
            raise ValueError("decode packet exceeds finite input/output storage")
        scale = ceil((self.elements // 16) / resource.scale_lanes) - 1 + resource.scale_latency
        values = (ceil(self.elements / resource.lanes) - 1) * resource.ii + resource.latency
        return dict(
            issue=1,
            arithmetic=scale + values,
            # Shared Vector-port staging, read packed data, write decoded
            # output. Matrix-view placement is charged by its DMA op.
            sram=ceil(self.transfer_bytes / 4096) * 2 + ceil(2 * self.elements / 4096),
            input_buffer_bytes=self.transfer_bytes,
            output_buffer_bytes=2 * self.elements,
        )


def storage_bytes(rows, columns):
    """Actual tensor extent, block padding, scale bytes, global scale once."""
    if min(rows, columns) < 1:
        raise ValueError("positive tensor dimensions required")
    padded = ceil(columns / 16) * 16
    payload = rows * padded // 2
    scales = rows * padded // 16
    return dict(
        values=payload,
        block_scales=scales,
        tensor_scale=4,
        padding_elements=rows * (padded - columns),
        total=ceil(payload / 64) * 64 + ceil(scales / 64) * 64 + 64,
    )


def packed_matrix_bytes(rows, columns):
    """Compiler BLEN32/K32 packet padding, interleaved block16 scale bytes."""
    if min(rows, columns) < 1:
        raise ValueError("positive matrix dimensions required")
    padded_rows = ceil(rows / 32) * 32
    padded_columns = ceil(columns / 32) * 32
    elements = padded_rows * padded_columns
    return dict(
        values=elements // 2,
        block_scales=elements // 16,
        tensor_scale=4,
        padding_elements=elements - rows * columns,
        total=elements * 9 // 16 + 64,
    )
