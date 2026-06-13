#!/usr/bin/env python3
"""Verify MXINT encoding by comparing hbm.mem with original tensor.

Usage:
    python -m verification.verify_mxint_encoding <build_dir>

Example:
    python -m verification.verify_mxint_encoding build/test/prefetch/20260512_012321
"""

import sys
from pathlib import Path

import torch
import numpy as np


def parse_hbm_mem_line(line: str) -> int:
    """Parse a hex line from hbm.mem file."""
    line = line.strip()
    if line.startswith("0x") or line.startswith("0X"):
        return int(line, 16)
    return 0


def mxint_to_float(element: int, scale: int, int_width: int = 8, scale_width: int = 8) -> float:
    """Convert single MXINT element to float.

    MXINT format: [sign(1)][magnitude(int_width-1)]
    Value = (-1)^sign * (magnitude / 2^(int_width-1)) * 2^(scale - bias)
    """
    magnitude_bits = int_width - 1
    magnitude_mask = (1 << magnitude_bits) - 1
    scale_bias = (1 << (scale_width - 1)) - 1  # 127 for 8-bit

    sign = (element >> magnitude_bits) & 1
    magnitude = element & magnitude_mask

    # Normalized mantissa in [0, 1)
    normalized = magnitude / (1 << magnitude_bits)

    # Scale factor
    actual_exp = int(scale) - scale_bias
    scale_factor = 2.0**actual_exp

    value = normalized * scale_factor
    if sign:
        value = -value

    return value


def read_hbm_mxint(
    hbm_path: Path,
    num_elements: int,
    block_size: int = 8,
    element_width: int = 8,
    scale_width: int = 8,
    row_width: int = 256,
    total_element_rows: int | None = None,
    scale_start_row: int | None = None,
):
    """Read MXINT data from hbm.mem and convert to float.

    Args:
        hbm_path: Path to hbm.mem file
        num_elements: Number of elements to read (for this tensor)
        block_size: Elements per block
        element_width: Bits per element
        scale_width: Bits per scale
        row_width: HBM row width in bits
        total_element_rows: Total element rows in HBM (for all tensors). If None, calculated from num_elements.
        scale_start_row: Row where scales start. If None, calculated from total_element_rows.
    """

    block_width = element_width * block_size  # 64 bits per block
    blocks_per_row = row_width // block_width  # 4 blocks per 256-bit row
    scales_per_row = row_width // scale_width  # 32 scales per row

    num_blocks = (num_elements + block_size - 1) // block_size
    element_rows = (num_blocks + blocks_per_row - 1) // blocks_per_row
    scale_rows = (num_blocks + scales_per_row - 1) // scales_per_row

    with open(hbm_path) as f:
        lines = [line.strip() for line in f if line.strip() and line.strip().startswith("0x")]

    # Auto-detect scale start row by finding where scale-like data begins
    # Scale data has characteristic pattern: mostly 0x7F/0x80 (biased exp around 0-1)
    if scale_start_row is None:
        if total_element_rows is not None:
            scale_start_row = total_element_rows
        else:
            # Try to auto-detect: find first row that looks like scale data
            scale_start_row = element_rows  # Default fallback
            for i in range(len(lines)):
                row_hex = lines[i][2:] if lines[i].startswith("0x") else lines[i]
                # Scale rows have mostly 0x7E, 0x7F, 0x80, 0x81 bytes (biased exp near 0)
                byte_vals = [int(row_hex[j : j + 2], 16) for j in range(0, min(len(row_hex), 16), 2)]
                if len(byte_vals) >= 4:
                    near_center = sum(1 for b in byte_vals if 0x7D <= b <= 0x82)
                    if near_center >= len(byte_vals) * 0.7:  # 70% of bytes are scale-like
                        scale_start_row = i
                        break

    print(f"[DEBUG] Scale start row: {scale_start_row} (total lines: {len(lines)})")

    # Extract raw elements
    elements = []
    blocks_extracted = 0

    for row_idx in range(element_rows):
        if row_idx >= len(lines):
            break
        row_value = parse_hbm_mem_line(lines[row_idx])

        blocks_in_row = min(blocks_per_row, num_blocks - blocks_extracted)

        for block_idx in range(blocks_in_row):
            block_start_bit = block_idx * block_width
            block_value = (row_value >> block_start_bit) & ((1 << block_width) - 1)

            for elem_idx in range(block_size):
                elem = (block_value >> (elem_idx * element_width)) & ((1 << element_width) - 1)
                elements.append(elem)

            blocks_extracted += 1

    elements = elements[:num_elements]

    # Extract scales from the correct location
    scales = []
    scales_extracted = 0

    for row_idx in range(scale_rows):
        actual_row_idx = scale_start_row + row_idx
        if actual_row_idx >= len(lines):
            break
        row_value = parse_hbm_mem_line(lines[actual_row_idx])
        scales_in_row = min(scales_per_row, num_blocks - scales_extracted)

        for scale_idx in range(scales_in_row):
            scale = (row_value >> (scale_idx * scale_width)) & ((1 << scale_width) - 1)
            scales.append(scale)
            scales_extracted += 1

    # Convert to float
    float_values = []
    for i, elem in enumerate(elements):
        block_idx = i // block_size
        scale = scales[block_idx] if block_idx < len(scales) else 127
        float_values.append(mxint_to_float(elem, scale))

    return np.array(elements), np.array(scales), np.array(float_values)


def verify_mxint_encoding(build_dir: Path, tensor_name: str = "act_tensor.pt"):
    """Verify MXINT encoding for a tensor.

    Args:
        build_dir: Path to build directory containing hbm.mem and tensor files
        tensor_name: Name of the tensor file to verify

    Returns:
        Dict with verification results
    """
    hbm_path = build_dir / "hbm.mem"
    tensor_path = build_dir / tensor_name

    if not hbm_path.exists():
        raise FileNotFoundError(f"{hbm_path} not found")

    if not tensor_path.exists():
        raise FileNotFoundError(f"{tensor_path} not found")

    # Load original tensor
    original = torch.load(tensor_path)
    if original.dtype == torch.bfloat16:
        original = original.float()
    original_flat = original.flatten().numpy()
    num_elements = len(original_flat)

    # Read and decode HBM
    raw_elements, scales, decoded = read_hbm_mxint(hbm_path, num_elements)

    # Compute error metrics
    abs_error = np.abs(decoded - original_flat)
    rel_error = abs_error / (np.abs(original_flat) + 1e-10)

    return {
        "original": original_flat,
        "decoded": decoded,
        "raw_elements": raw_elements,
        "scales": scales,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "max_abs_error": abs_error.max(),
        "mean_abs_error": abs_error.mean(),
        "max_rel_error": rel_error.max(),
        "mean_rel_error": rel_error.mean(),
        "num_elements": num_elements,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m verification.verify_mxint_encoding <build_dir>")
        print("Example: python -m verification.verify_mxint_encoding build/test/prefetch/20260512_012321")
        sys.exit(1)

    build_dir = Path(sys.argv[1])

    print("=" * 70)
    print("MXINT Encoding Verification")
    print("=" * 70)
    print(f"Build dir: {build_dir}")

    try:
        results = verify_mxint_encoding(build_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    original_flat = results["original"]
    decoded = results["decoded"]
    raw_elements = results["raw_elements"]
    scales = results["scales"]
    abs_error = results["abs_error"]
    rel_error = results["rel_error"]
    num_elements = results["num_elements"]

    print(f"Num elements: {num_elements}")
    print()

    # Show first few blocks
    print("First 2 blocks (raw):")
    for block_idx in range(min(2, len(scales))):
        start = block_idx * 8
        end = start + 8
        block_elems = raw_elements[start:end]
        scale = scales[block_idx]
        scale_bias = 127
        actual_exp = int(scale) - scale_bias

        print(f"  Block {block_idx}: scale={scale} (0x{scale:02X}) => 2^{actual_exp}")
        print(f"    Raw:     [{', '.join(f'0x{e:02X}' for e in block_elems)}]")
        print(f"    Decoded: [{', '.join(f'{decoded[start + i]:8.4f}' for i in range(8))}]")
        print(f"    Original:[{', '.join(f'{original_flat[start + i]:8.4f}' for i in range(8))}]")
        print()

    # Compute error metrics
    print("Error Metrics:")
    print(f"  Max absolute error:  {results['max_abs_error']:.6f}")
    print(f"  Mean absolute error: {results['mean_abs_error']:.6f}")
    print(f"  Max relative error:  {results['max_rel_error']:.6f}")
    print(f"  Mean relative error: {results['mean_rel_error']:.6f}")
    print()

    # Check if quantization error is reasonable (MXINT8 has ~7 bits of precision)
    # Expected relative error for 7-bit mantissa: ~1/128 = 0.0078
    expected_rel_error = 1.0 / 128

    if rel_error.mean() < expected_rel_error * 2:
        print(f"✓ PASS: Mean relative error ({rel_error.mean():.4f}) is within expected range for MXINT8")
    else:
        print(f"✗ FAIL: Mean relative error ({rel_error.mean():.4f}) exceeds expected range")
        print(f"  Expected: < {expected_rel_error * 2:.4f}")

    # Show worst cases
    worst_indices = np.argsort(abs_error)[-5:][::-1]
    print("\nWorst 5 errors:")
    print(f"  {'Index':<8} {'Original':<12} {'Decoded':<12} {'Abs Error':<12} {'Rel Error':<12}")
    for idx in worst_indices:
        print(
            f"  {idx:<8} {original_flat[idx]:<12.6f} {decoded[idx]:<12.6f} "
            f"{abs_error[idx]:<12.6f} {rel_error[idx]:<12.6f}"
        )


if __name__ == "__main__":
    main()
