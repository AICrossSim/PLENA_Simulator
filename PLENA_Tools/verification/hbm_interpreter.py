"""HBM Memory Interpreter for PLENA.

This module provides tools to decode HBM memory files (.mem hex format) back to
floating point values for verification purposes. Supports both MXFP and MXINT formats.

Usage:
    python -m verification.hbm_interpreter /path/to/hbm.mem --format mxint --original /path/to/tensor.pt

    # Dump raw data for debugging
    python -m verification.hbm_interpreter /path/to/hbm.mem --dump-raw --num-blocks 8
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .verify_rtl_sim import mx_to_float, mxint_to_float


def parse_hbm_mem_line(line: str) -> int:
    """Parse a hex line from hbm.mem file.

    Args:
        line: Line from .mem file (format: 0xHEX_VALUE)

    Returns:
        Integer value of the hex data
    """
    line = line.strip()
    if line.startswith("0x") or line.startswith("0X"):
        return int(line, 16)
    return 0


def read_hbm_mem_file(
    hbm_path: str | Path,
    num_elements: int,
    num_scales: int,
    element_width: int,
    scale_width: int,
    block_size: int = 8,
    row_width: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Read elements and scales from HBM .mem file.

    The .mem file format is:
        - Element section: blocks of elements packed into 256-bit rows
        - Scale section: scale values packed into 256-bit rows
        - Instruction section: (ignored)

    Args:
        hbm_path: Path to hbm.mem file
        num_elements: Total number of elements to read
        num_scales: Number of scale values to read
        element_width: Bit width of each element (8 for both MXFP E4M3 and MXINT8)
        scale_width: Bit width of each scale (typically 8)
        block_size: Number of elements per block/scale
        row_width: HBM row width in bits (256)

    Returns:
        Tuple of (elements array, scales array) as uint8 numpy arrays
    """
    hbm_path = Path(hbm_path)

    # Calculate packing parameters
    block_width = element_width * block_size
    blocks_per_row = row_width // block_width
    scales_per_row = row_width // scale_width

    # Read all lines
    with open(hbm_path) as f:
        lines = [line.strip() for line in f if line.strip() and line.strip().startswith("0x")]

    # Calculate row counts
    num_blocks = (num_elements + block_size - 1) // block_size
    element_rows = (num_blocks + blocks_per_row - 1) // blocks_per_row
    scale_rows = (num_scales + scales_per_row - 1) // scales_per_row

    # Extract elements
    elements = []
    blocks_extracted = 0

    for row_idx in range(element_rows):
        if row_idx >= len(lines):
            break
        row_value = parse_hbm_mem_line(lines[row_idx])

        blocks_in_this_row = min(blocks_per_row, num_blocks - blocks_extracted)

        for block_idx in range(blocks_in_this_row):
            # Extract block worth of elements
            block_start_bit = block_idx * block_width
            block_value = (row_value >> block_start_bit) & ((1 << block_width) - 1)

            # Extract individual elements from block
            for elem_idx in range(block_size):
                elem = (block_value >> (elem_idx * element_width)) & ((1 << element_width) - 1)
                elements.append(elem)

            blocks_extracted += 1

    # Truncate to exact number of elements
    elements = elements[:num_elements]

    # Extract scales
    scales = []
    scales_extracted = 0

    for row_idx in range(scale_rows):
        actual_row_idx = element_rows + row_idx
        if actual_row_idx >= len(lines):
            break
        row_value = parse_hbm_mem_line(lines[actual_row_idx])

        scales_in_this_row = min(scales_per_row, num_scales - scales_extracted)

        for scale_idx in range(scales_in_this_row):
            scale = (row_value >> (scale_idx * scale_width)) & ((1 << scale_width) - 1)
            scales.append(scale)
            scales_extracted += 1

    return np.array(elements, dtype=np.uint8), np.array(scales, dtype=np.uint8)


def interpret_hbm_to_float(
    hbm_path: str | Path,
    num_elements: int,
    mx_format: str = "mxint",
    element_width: int = 8,
    scale_width: int = 8,
    block_size: int = 8,
    exp_width: int = 4,
    man_width: int = 3,
    row_width: int = 256,
    tensor_shape: list[int] | None = None,
) -> Tensor:
    """Interpret HBM memory file and convert to floating point tensor.

    Args:
        hbm_path: Path to hbm.mem file
        num_elements: Total number of elements
        mx_format: "mxint" or "mxfp"
        element_width: Bit width of each element
        scale_width: Bit width of scale values
        block_size: Elements per block
        exp_width: MXFP exponent width (only for mxfp format)
        man_width: MXFP mantissa width (only for mxfp format)
        row_width: HBM row width in bits
        tensor_shape: Optional shape to reshape result

    Returns:
        Floating point tensor with decoded values
    """
    num_scales = (num_elements + block_size - 1) // block_size

    # Read raw data
    elements, scales = read_hbm_mem_file(
        hbm_path, num_elements, num_scales, element_width, scale_width, block_size, row_width
    )

    # Convert to float
    if mx_format.lower() == "mxint":
        float_values = mxint_to_float(
            elements,
            scales,
            int_width=element_width,
            scale_width=scale_width,
            block_size=block_size,
        )
    else:  # mxfp
        float_values = mx_to_float(
            elements,
            scales,
            exp_width=exp_width,
            man_width=man_width,
            scale_width=scale_width,
            block_size=block_size,
        )

    # Create tensor
    result = torch.from_numpy(float_values)

    if tensor_shape:
        result = result.reshape(tensor_shape)

    return result


def compare_with_original(
    hbm_path: str | Path,
    original_tensor_path: str | Path,
    mx_format: str = "mxint",
    element_width: int = 8,
    scale_width: int = 8,
    block_size: int = 8,
    exp_width: int = 4,
    man_width: int = 3,
    verbose: bool = True,
) -> dict:
    """Compare interpreted HBM data with original tensor.

    Args:
        hbm_path: Path to hbm.mem file
        original_tensor_path: Path to original .pt tensor file
        mx_format: "mxint" or "mxfp"
        element_width: Bit width of each element
        scale_width: Bit width of scale values
        block_size: Elements per block
        exp_width: MXFP exponent width
        man_width: MXFP mantissa width
        verbose: Print detailed comparison info

    Returns:
        Dictionary with comparison metrics
    """
    # Load original tensor
    original = torch.load(original_tensor_path)
    if original.dtype == torch.bfloat16:
        original = original.float()
    original_flat = original.flatten()

    num_elements = original_flat.numel()

    # Interpret HBM
    interpreted = interpret_hbm_to_float(
        hbm_path, num_elements, mx_format, element_width, scale_width, block_size, exp_width, man_width
    )

    # Compute metrics
    abs_diff = torch.abs(interpreted - original_flat)
    rel_diff = abs_diff / (torch.abs(original_flat) + 1e-10)

    metrics = {
        "num_elements": num_elements,
        "max_abs_error": abs_diff.max().item(),
        "mean_abs_error": abs_diff.mean().item(),
        "max_rel_error": rel_diff.max().item(),
        "mean_rel_error": rel_diff.mean().item(),
        "original_range": (original_flat.min().item(), original_flat.max().item()),
        "interpreted_range": (interpreted.min().item(), interpreted.max().item()),
    }

    if verbose:
        print(f"HBM Interpretation Comparison ({mx_format.upper()} format)")
        print("=" * 60)
        print(f"Number of elements: {num_elements}")
        print("Format parameters:")
        print(f"  element_width: {element_width}")
        print(f"  scale_width: {scale_width}")
        print(f"  block_size: {block_size}")
        if mx_format.lower() == "mxfp":
            print(f"  exp_width: {exp_width}")
            print(f"  man_width: {man_width}")
        print(f"\nOriginal tensor range: [{metrics['original_range'][0]:.6f}, {metrics['original_range'][1]:.6f}]")
        print(f"Interpreted range:     [{metrics['interpreted_range'][0]:.6f}, {metrics['interpreted_range'][1]:.6f}]")
        print("\nError metrics:")
        print(f"  Max absolute error: {metrics['max_abs_error']:.6f}")
        print(f"  Mean absolute error: {metrics['mean_abs_error']:.6f}")
        print(f"  Max relative error: {metrics['max_rel_error']:.6f}")
        print(f"  Mean relative error: {metrics['mean_rel_error']:.6f}")

        # Show sample values
        print("\nSample comparisons (first 10 elements):")
        print(f"{'Index':<8} {'Original':<15} {'Interpreted':<15} {'Abs Error':<15}")
        print("-" * 60)
        for i in range(min(10, num_elements)):
            print(f"{i:<8} {original_flat[i].item():<15.6f} {interpreted[i].item():<15.6f} {abs_diff[i].item():<15.6f}")

    return metrics


def dump_hbm_raw(
    hbm_path: str | Path,
    num_elements: int,
    element_width: int = 8,
    scale_width: int = 8,
    block_size: int = 8,
    row_width: int = 256,
    num_blocks_to_show: int = 4,
    mx_format: str = "mxint",
) -> None:
    """Dump raw HBM data for debugging.

    Args:
        hbm_path: Path to hbm.mem file
        num_elements: Total number of elements
        element_width: Bit width of each element
        scale_width: Bit width of scale values
        block_size: Elements per block
        row_width: HBM row width in bits
        num_blocks_to_show: Number of blocks to display
        mx_format: "mxint" or "mxfp" (affects how binary is displayed)
    """
    num_scales = (num_elements + block_size - 1) // block_size

    elements, scales = read_hbm_mem_file(
        hbm_path, num_elements, num_scales, element_width, scale_width, block_size, row_width
    )

    print(f"HBM Raw Data Dump ({mx_format.upper()} format)")
    print("=" * 80)
    print(f"Total elements: {num_elements}")
    print(f"Total blocks: {num_scales}")
    print(f"Total scales: {len(scales)}")
    print(f"Element width: {element_width} bits")
    print(f"Scale width: {scale_width} bits")
    print(f"Block size: {block_size}")
    print()

    scale_bias = (1 << (scale_width - 1)) - 1

    for block_idx in range(min(num_blocks_to_show, num_scales)):
        start_elem = block_idx * block_size
        end_elem = min(start_elem + block_size, len(elements))
        block_elements = elements[start_elem:end_elem]
        scale = scales[block_idx] if block_idx < len(scales) else 0
        actual_scale_exp = int(scale) - scale_bias

        print(f"Block {block_idx}: scale=0x{scale:02X} ({scale}) => 2^{actual_scale_exp}")
        elements_hex = " ".join(f"0x{e:02X}" for e in block_elements)
        print(f"  Raw elements: [{elements_hex}]")

        # Show binary breakdown and converted values
        print(f"  Binary breakdown (sign|{'magnitude' if mx_format == 'mxint' else 'exp|man'}) and FP values:")
        for j, elem in enumerate(block_elements):
            if mx_format.lower() == "mxint":
                sign = (elem >> (element_width - 1)) & 1
                magnitude = elem & ((1 << (element_width - 1)) - 1)
                normalized = magnitude / (1 << (element_width - 1))
                fp_val = normalized * (2.0**actual_scale_exp)
                if sign:
                    fp_val = -fp_val
                print(f"    [{j}] 0x{elem:02X} = {sign}|{magnitude:07b} => {fp_val:12.6f}")
            else:
                # MXFP: sign|exp|man - would need exp_width and man_width
                print(f"    [{j}] 0x{elem:02X} = {elem:08b}")
        print()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="HBM Memory Interpreter - decode hbm.mem to floating point")
    parser.add_argument("hbm_path", help="Path to hbm.mem file")
    parser.add_argument("--original", "-o", help="Path to original .pt tensor for comparison")
    parser.add_argument(
        "--format", "-f", choices=["mxint", "mxfp"], default="mxint", help="Data format (default: mxint)"
    )
    parser.add_argument("--num-elements", "-n", type=int, default=1024, help="Number of elements to read")
    parser.add_argument("--element-width", type=int, default=8, help="Element bit width (default: 8)")
    parser.add_argument("--scale-width", type=int, default=8, help="Scale bit width (default: 8)")
    parser.add_argument("--block-size", type=int, default=8, help="Block size (default: 8)")
    parser.add_argument("--exp-width", type=int, default=4, help="MXFP exponent width (default: 4)")
    parser.add_argument("--man-width", type=int, default=3, help="MXFP mantissa width (default: 3)")
    parser.add_argument("--dump-raw", "-d", action="store_true", help="Dump raw HBM data")
    parser.add_argument("--num-blocks", type=int, default=4, help="Number of blocks to show in raw dump")

    args = parser.parse_args()

    if args.dump_raw:
        dump_hbm_raw(
            args.hbm_path,
            args.num_elements,
            args.element_width,
            args.scale_width,
            args.block_size,
            num_blocks_to_show=args.num_blocks,
            mx_format=args.format,
        )

    if args.original:
        compare_with_original(
            args.hbm_path,
            args.original,
            mx_format=args.format,
            element_width=args.element_width,
            scale_width=args.scale_width,
            block_size=args.block_size,
            exp_width=args.exp_width,
            man_width=args.man_width,
        )
    elif not args.dump_raw:
        # Just interpret and show summary
        tensor = interpret_hbm_to_float(
            args.hbm_path,
            args.num_elements,
            mx_format=args.format,
            element_width=args.element_width,
            scale_width=args.scale_width,
            block_size=args.block_size,
            exp_width=args.exp_width,
            man_width=args.man_width,
        )
        print(f"Interpreted {args.num_elements} elements from HBM ({args.format.upper()} format)")
        print(f"Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
        print(f"Mean: {tensor.mean().item():.6f}")
        print(f"First 10 values: {tensor[:10].tolist()}")


if __name__ == "__main__":
    main()
