"""FP scalar register-file dump viewer and converter.

Parses fp_reg_result.mem files produced by the RTL simulation (one raw hex
entry per register, written via $writememh on the scalar `fp_reg_file`) and
converts each entry to its real floating-point value using the same minifloat
format as the rest of the toolchain (1 sign + S_FP_EXP_WIDTH exp +
S_FP_MANT_WIDTH mant). Defaults match precision.svh: exp=6, mant=5 (FP12).

Usage:
    python -m verification.view_fp_reg --file /path/to/fp_reg_result.mem

    Or programmatically:
    from verification.view_fp_reg import view_fp_reg_as_fp
    regs = view_fp_reg_as_fp("fp_reg_result.mem", exp_width=6, man_width=5)
"""

import argparse
import sys
from pathlib import Path

# Reuse the canonical FP-bit -> float converter so both viewers stay in sync.
from verification.view_vector_result import fp_to_float


def parse_fp_reg_file(filepath: str | Path) -> list[int]:
    """Parse an fp_reg_result.mem dump into a list of raw register words.

    File format ($writememh output): one hex value per line, one per register,
    in ascending register-index order. Blank lines, ``//`` comments and
    ``@address`` markers are ignored.
    """
    filepath = Path(filepath)
    regs: list[int] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("@"):
                continue
            token = line.split()[0]
            try:
                regs.append(int(token, 16))
            except ValueError:
                continue
    return regs


def view_fp_reg_as_fp(
    filepath: str | Path,
    exp_width: int = 6,
    man_width: int = 5,
    output_file: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Parse fp_reg_result.mem and convert every register to a float.

    Args:
        filepath: Path to the fp_reg_result.mem dump.
        exp_width: FP exponent width (S_FP_EXP_WIDTH, default 6).
        man_width: FP mantissa width (S_FP_MANT_WIDTH, default 5).
        output_file: Optional path to also write the decoded table.
        verbose: Print the decoded table to stdout.

    Returns:
        Dict with 'raw' (list[int]), 'fp_values' (list[float]),
        'fp_format' (str), 'exp_width', 'man_width'.
    """
    filepath = Path(filepath)
    raw = parse_fp_reg_file(filepath)

    element_width = 1 + exp_width + man_width
    mask = (1 << element_width) - 1
    hex_chars = (element_width + 3) // 4
    bias = (1 << (exp_width - 1)) - 1
    fp_format = f"FP{element_width} (1s + {exp_width}e + {man_width}m)"

    raw = [r & mask for r in raw]
    fp_values = [fp_to_float(r, exp_width, man_width) for r in raw]

    lines = []
    header = (
        "=" * 60 + "\n"
        "FP Scalar Register File Dump\n"
        + "=" * 60 + "\n"
        f"  File: {filepath}\n"
        f"  FP Format: {fp_format}  bias={bias}\n"
        f"  Registers: {len(raw)}\n"
        + "=" * 60 + "\n"
        f"{'Reg':>5} {'Hex':>{hex_chars + 2}} {'Value':>16}"
    )
    lines.append(header)
    for idx, (r, v) in enumerate(zip(raw, fp_values)):
        lines.append(f"f{idx:<4d} 0x{r:0{hex_chars}x} {v:16.8g}")

    text = "\n".join(lines)
    if verbose:
        print(text)
    if output_file:
        Path(output_file).write_text(text + "\n")
        if verbose:
            print(f"\nOutput written to: {output_file}")

    return {
        "raw": raw,
        "fp_values": fp_values,
        "fp_format": fp_format,
        "exp_width": exp_width,
        "man_width": man_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="View fp_reg_result.mem with FP conversion")
    parser.add_argument("--file", "-f", type=str, required=True, help="Path to fp_reg_result.mem")
    parser.add_argument("--exp-width", type=int, default=6, help="Exponent width (S_FP_EXP_WIDTH, default 6)")
    parser.add_argument("--man-width", type=int, default=5, help="Mantissa width (S_FP_MANT_WIDTH, default 5)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file path for decoded values")
    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    view_fp_reg_as_fp(
        filepath,
        exp_width=args.exp_width,
        man_width=args.man_width,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
