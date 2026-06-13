"""Vector result memory file viewer and FP converter.

Parses vector_result.mem files from RTL simulation and converts
packed hex data to floating point values with configurable precision.

Usage:
    python -m verification.view_vector_result --file /path/to/vector_result.mem --vlen 16

    Or programmatically:
    from verification.view_vector_result import view_vector_result_as_fp
    fp_values = view_vector_result_as_fp("vector_result.mem", vlen=16, exp_width=6, man_width=5)
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def fp_to_float(
    element: int,
    exp_width: int = 6,
    man_width: int = 5,
) -> float:
    """Convert FP format element to float.

    Args:
        element: Raw element data (uint, with sign+exp+mant)
        exp_width: Exponent width (default: 6 for V_FP_EXP_WIDTH)
        man_width: Mantissa width (default: 5 for V_FP_MANT_WIDTH)

    Returns:
        Float value
    """
    total_width = 1 + exp_width + man_width
    mask = (1 << total_width) - 1
    element = element & mask

    # Extract fields
    sign = (element >> (exp_width + man_width)) & 1
    exp = (element >> man_width) & ((1 << exp_width) - 1)
    man = element & ((1 << man_width) - 1)

    bias = (1 << (exp_width - 1)) - 1  # bias = 31 for exp_width=6

    if exp == 0:
        # Subnormal or zero
        if man == 0:
            return 0.0
        # Subnormal: implicit leading 0
        val = (man / (2**man_width)) * (2 ** (1 - bias))
    elif exp == (1 << exp_width) - 1:
        # Inf/NaN
        if man == 0:
            return float("-inf") if sign else float("inf")
        return float("nan")
    else:
        # Normal: implicit leading 1
        val = (1 + man / (2**man_width)) * (2 ** (exp - bias))

    return -val if sign else val


def extract_fp_elements_from_row(
    row_data: int,
    vlen: int = 16,
    exp_width: int = 6,
    man_width: int = 5,
) -> list[float]:
    """Extract FP elements from a row and convert to floats.

    Args:
        row_data: Row data as integer
        vlen: Number of elements per row (VLEN)
        exp_width: FP exponent width
        man_width: FP mantissa width

    Returns:
        List of float values
    """
    element_width = 1 + exp_width + man_width
    element_mask = (1 << element_width) - 1

    values = []
    for i in range(vlen):
        elem = (row_data >> (i * element_width)) & element_mask
        values.append(fp_to_float(elem, exp_width, man_width))

    return values


def parse_vector_result_file(filepath: str | Path) -> list[int]:
    """Parse vector SRAM result file.

    File format (hex values, one per line):
        DATA_ROW_0
        DATA_ROW_1
        ...

    Args:
        filepath: Path to vector_result.mem file

    Returns:
        List of row data values (as integers)
    """
    data = []
    filepath = Path(filepath)

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue

            # Parse hex data (with or without 0x prefix)
            try:
                if line.startswith("0x") or line.startswith("0X"):
                    data.append(int(line, 16))
                else:
                    data.append(int(line, 16))
            except ValueError:
                continue

    return data


def view_vector_result_as_fp(
    filepath: str | Path,
    vlen: int = 16,
    exp_width: int = 6,
    man_width: int = 5,
    start_row: int = 0,
    num_rows: int | None = None,
    output_file: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Parse vector_result.mem and convert all data to floating point.

    Args:
        filepath: Path to vector_result.mem file
        vlen: Number of elements per row (VLEN, default: 16)
        exp_width: FP exponent width (V_FP_EXP_WIDTH, default: 6)
        man_width: FP mantissa width (V_FP_MANT_WIDTH, default: 5)
        start_row: Starting row index (default: 0)
        num_rows: Number of rows to process (None = all remaining)
        output_file: Optional path to write FP values
        verbose: Print output to console (default: True)

    Returns:
        Dictionary containing:
            - 'fp_values': 2D list of float values [rows][elements]
            - 'flat_values': 1D numpy array of all values
            - 'num_rows': Number of rows processed
            - 'vlen': VLEN used
            - 'fp_format': String describing FP format
            - 'row_width_bits': Width of each row in bits
    """
    filepath = Path(filepath)

    # Calculate row width
    element_width = 1 + exp_width + man_width
    row_width_bits = vlen * element_width

    # Parse the file
    raw_data = parse_vector_result_file(filepath)

    if not raw_data:
        raise ValueError(f"No data found in {filepath}")

    # Determine row range
    total_rows = len(raw_data)
    end_row = total_rows if num_rows is None else min(start_row + num_rows, total_rows)

    fp_format = f"FP{element_width} (1s + {exp_width}e + {man_width}m)"
    bias = (1 << (exp_width - 1)) - 1

    if verbose:
        print("=" * 70)
        print("Vector Result FP Viewer")
        print("=" * 70)
        print(f"  File: {filepath}")
        print(f"  VLEN: {vlen}")
        print(f"  FP Format: {fp_format}")
        print(f"  Bias: {bias}")
        print(f"  Row width: {row_width_bits} bits ({row_width_bits // 4} hex chars)")
        print(f"  Total rows in file: {total_rows}")
        print(f"  Processing rows: {start_row} to {end_row - 1}")
        print("=" * 70)

    # Extract FP values
    fp_values = []
    all_values = []

    for row_idx in range(start_row, end_row):
        row_floats = extract_fp_elements_from_row(
            raw_data[row_idx],
            vlen=vlen,
            exp_width=exp_width,
            man_width=man_width,
        )
        fp_values.append(row_floats)
        all_values.extend(row_floats)

        if verbose:
            # Format row output
            print(f"Row {row_idx:4d}: ", end="")
            for i, v in enumerate(row_floats):
                if np.isnan(v):
                    print(f"{'nan':>9s}", end=" ")
                elif np.isinf(v):
                    print(f"{'inf' if v > 0 else '-inf':>9s}", end=" ")
                elif abs(v) < 0.001 and v != 0:
                    print(f"{v:9.2e}", end=" ")
                else:
                    print(f"{v:9.4f}", end=" ")
            print()

    flat_values = np.array(all_values, dtype=np.float32)

    # Write to output file if specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            f.write("# Vector Result FP Values\n")
            f.write(f"# File: {filepath}\n")
            f.write(f"# VLEN: {vlen}, FP Format: {fp_format}\n")
            f.write(f"# Rows: {start_row} to {end_row - 1}\n")
            f.write(f"# Values per row: {vlen}\n")
            f.write("#\n")
            for row_idx, row_floats in enumerate(fp_values, start=start_row):
                f.write(f"Row {row_idx:4d}: ")
                f.write(" ".join(f"{v:12.6f}" for v in row_floats))
                f.write("\n")
        if verbose:
            print(f"\nOutput written to: {output_path}")

    # Summary statistics for non-zero values
    non_zero = flat_values[flat_values != 0]
    if verbose and len(non_zero) > 0:
        print("\n" + "-" * 70)
        print("Statistics (non-zero values only):")
        print("-" * 70)
        print(f"  Total elements: {len(flat_values)}")
        print(f"  Non-zero elements: {len(non_zero)}")
        print(f"  Min: {np.min(non_zero):.6f}")
        print(f"  Max: {np.max(non_zero):.6f}")
        print(f"  Mean: {np.mean(non_zero):.6f}")
        print(f"  Std: {np.std(non_zero):.6f}")
        print("-" * 70)

    return {
        "fp_values": fp_values,
        "flat_values": flat_values,
        "num_rows": end_row - start_row,
        "vlen": vlen,
        "exp_width": exp_width,
        "man_width": man_width,
        "fp_format": fp_format,
        "row_width_bits": row_width_bits,
        "start_row": start_row,
        "end_row": end_row,
    }


def view_vector_result_as_hex(
    filepath: str | Path,
    vlen: int = 16,
    exp_width: int = 6,
    man_width: int = 5,
    start_row: int = 0,
    num_rows: int | None = None,
) -> None:
    """Display vector_result.mem with raw hex element values.

    Args:
        filepath: Path to vector_result.mem file
        vlen: Number of elements per row (VLEN)
        exp_width: FP exponent width
        man_width: FP mantissa width
        start_row: Starting row index
        num_rows: Number of rows to display
    """
    filepath = Path(filepath)
    element_width = 1 + exp_width + man_width
    element_mask = (1 << element_width) - 1
    hex_chars = (element_width + 3) // 4  # Round up to hex chars

    raw_data = parse_vector_result_file(filepath)
    total_rows = len(raw_data)
    end_row = total_rows if num_rows is None else min(start_row + num_rows, total_rows)

    print("=" * 70)
    print("Vector Result Hex Viewer")
    print("=" * 70)
    print(f"  File: {filepath}")
    print(f"  VLEN: {vlen}, Element width: {element_width} bits")
    print(f"  Rows: {start_row} to {end_row - 1}")
    print("=" * 70)

    for row_idx in range(start_row, end_row):
        row_data = raw_data[row_idx]
        print(f"Row {row_idx:4d}: ", end="")
        for i in range(vlen):
            elem = (row_data >> (i * element_width)) & element_mask
            print(f"0x{elem:0{hex_chars}x}", end=" ")
        print()


def view_vector_result_as_binary(
    filepath: str | Path,
    vlen: int = 16,
    exp_width: int = 6,
    man_width: int = 5,
    start_row: int = 0,
    num_rows: int | None = None,
) -> None:
    """Display vector_result.mem with binary breakdown of FP fields.

    Shows sign, exponent, and mantissa fields separately.

    Args:
        filepath: Path to vector_result.mem file
        vlen: Number of elements per row (VLEN)
        exp_width: FP exponent width
        man_width: FP mantissa width
        start_row: Starting row index
        num_rows: Number of rows to display
    """
    filepath = Path(filepath)
    element_width = 1 + exp_width + man_width
    element_mask = (1 << element_width) - 1

    raw_data = parse_vector_result_file(filepath)
    total_rows = len(raw_data)
    end_row = total_rows if num_rows is None else min(start_row + num_rows, total_rows)

    bias = (1 << (exp_width - 1)) - 1

    print("=" * 90)
    print("Vector Result Binary Viewer")
    print("=" * 90)
    print(f"  File: {filepath}")
    print(f"  VLEN: {vlen}, FP{element_width} (1s + {exp_width}e + {man_width}m), bias={bias}")
    print(f"  Rows: {start_row} to {end_row - 1}")
    print("=" * 90)
    print(f"{'Row':>6} {'Elem':>4} {'Sign':>4} {'Exp':>{exp_width + 2}} {'Mant':>{man_width + 2}} {'Value':>12}")
    print("-" * 90)

    for row_idx in range(start_row, end_row):
        row_data = raw_data[row_idx]
        for i in range(vlen):
            elem = (row_data >> (i * element_width)) & element_mask
            sign = (elem >> (exp_width + man_width)) & 1
            exp = (elem >> man_width) & ((1 << exp_width) - 1)
            man = elem & ((1 << man_width) - 1)
            fp_val = fp_to_float(elem, exp_width, man_width)

            sign_str = f"{sign}"
            exp_str = f"{exp:0{exp_width}b}"
            man_str = f"{man:0{man_width}b}"

            if np.isnan(fp_val):
                val_str = "nan"
            elif np.isinf(fp_val):
                val_str = "inf" if fp_val > 0 else "-inf"
            else:
                val_str = f"{fp_val:.6f}"

            print(
                f"{row_idx:>6} {i:>4} {sign_str:>4} {exp_str:>{exp_width + 2}} {man_str:>{man_width + 2}} {val_str:>12}"
            )
        print("-" * 90)


def main():
    """Main entry point for command line usage."""
    parser = argparse.ArgumentParser(description="View vector_result.mem files with FP conversion")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to vector_result.mem file")
    parser.add_argument("--vlen", type=int, default=16, help="Number of elements per row (VLEN, default: 16)")
    parser.add_argument("--exp-width", type=int, default=6, help="Exponent width (V_FP_EXP_WIDTH, default: 6)")
    parser.add_argument("--man-width", type=int, default=5, help="Mantissa width (V_FP_MANT_WIDTH, default: 5)")
    parser.add_argument("--start-row", type=int, default=0, help="Starting row index (default: 0)")
    parser.add_argument("--num-rows", type=int, default=None, help="Number of rows to display (default: all)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path for FP values")
    parser.add_argument(
        "--mode",
        choices=["fp", "hex", "binary"],
        default="fp",
        help="Display mode: fp (floating point), hex, or binary (default: fp)",
    )

    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    if args.mode == "fp":
        view_vector_result_as_fp(
            filepath,
            vlen=args.vlen,
            exp_width=args.exp_width,
            man_width=args.man_width,
            start_row=args.start_row,
            num_rows=args.num_rows,
            output_file=args.output,
        )
    elif args.mode == "hex":
        view_vector_result_as_hex(
            filepath,
            vlen=args.vlen,
            exp_width=args.exp_width,
            man_width=args.man_width,
            start_row=args.start_row,
            num_rows=args.num_rows,
        )
    elif args.mode == "binary":
        view_vector_result_as_binary(
            filepath,
            vlen=args.vlen,
            exp_width=args.exp_width,
            man_width=args.man_width,
            start_row=args.start_row,
            num_rows=args.num_rows,
        )


if __name__ == "__main__":
    main()
