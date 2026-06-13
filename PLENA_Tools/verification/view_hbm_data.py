#!/usr/bin/env python3
"""View HBM data in floating point format.

Prints initial HBM data (activations, weights) and result HBM data
converted from MX format to floating point for comparison.

Usage:
    python -m verification.view_hbm_data --workload-dir build/test/prefetch
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def parse_hbm_mem_file(filepath: Path) -> list[int]:
    """Parse HBM .mem file to list of 256-bit row values."""
    rows = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("0x") or line.startswith("0X"):
                rows.append(int(line, 16))
    return rows


def extract_mx_data(
    rows: list[int],
    start_row: int,
    num_elements: int,
    element_width: int = 8,
    scale_width: int = 8,
    block_size: int = 8,
    row_width: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract MX elements and scales from HBM rows.

    Layout: elements packed in rows, then scales packed in following rows.
    """
    elements_per_row = row_width // element_width
    scales_per_row = row_width // scale_width

    num_blocks = (num_elements + block_size - 1) // block_size
    element_rows_needed = (num_elements + elements_per_row - 1) // elements_per_row
    scale_rows_needed = (num_blocks + scales_per_row - 1) // scales_per_row

    # Extract elements
    elements = []
    for row_idx in range(element_rows_needed):
        if start_row + row_idx >= len(rows):
            break
        row_val = rows[start_row + row_idx]
        for elem_idx in range(elements_per_row):
            if len(elements) >= num_elements:
                break
            elem = (row_val >> (elem_idx * element_width)) & ((1 << element_width) - 1)
            elements.append(elem)

    # Extract scales (following element rows)
    scales = []
    scale_start_row = start_row + element_rows_needed
    for row_idx in range(scale_rows_needed):
        if scale_start_row + row_idx >= len(rows):
            break
        row_val = rows[scale_start_row + row_idx]
        for scale_idx in range(scales_per_row):
            if len(scales) >= num_blocks:
                break
            scale = (row_val >> (scale_idx * scale_width)) & ((1 << scale_width) - 1)
            scales.append(scale)

    return np.array(elements, dtype=np.uint8), np.array(scales, dtype=np.uint8)


def mxint_to_float(
    elements: np.ndarray,
    scales: np.ndarray,
    int_width: int = 8,
    scale_width: int = 8,
    block_size: int = 8,
) -> np.ndarray:
    """Convert MXINT format to float.

    MXINT: [sign(1)][magnitude(int_width-1)] with biased scale.
    Value = (-1)^sign * (magnitude / 2^(int_width-1)) * 2^(scale - bias)
    """
    values = []
    magnitude_bits = int_width - 1
    magnitude_mask = (1 << magnitude_bits) - 1
    scale_bias = (1 << (scale_width - 1)) - 1  # 127 for 8-bit

    for i, elem in enumerate(elements):
        block_idx = i // block_size
        scale = scales[block_idx] if block_idx < len(scales) else scale_bias

        sign = (elem >> magnitude_bits) & 1
        magnitude = elem & magnitude_mask

        normalized = magnitude / (1 << magnitude_bits)
        scale_val = 2 ** (int(scale) - scale_bias)

        fp_val = normalized * scale_val
        if sign:
            fp_val = -fp_val
        values.append(fp_val)

    return np.array(values, dtype=np.float32)


def mxfp_to_float(
    elements: np.ndarray,
    scales: np.ndarray,
    exp_width: int = 4,
    man_width: int = 3,
    scale_width: int = 8,
    block_size: int = 8,
) -> np.ndarray:
    """Convert MXFP format to float.

    MXFP: [sign(1)][exp(exp_width)][man(man_width)] with biased scale.
    """
    values = []
    bias = (1 << (exp_width - 1)) - 1
    scale_bias = 127

    for i, elem in enumerate(elements):
        block_idx = i // block_size
        scale = scales[block_idx] if block_idx < len(scales) else scale_bias

        sign = (elem >> (exp_width + man_width)) & 1
        exp = (elem >> man_width) & ((1 << exp_width) - 1)
        man = elem & ((1 << man_width) - 1)

        if exp == 0:
            if man == 0:
                elem_val = 0.0
            else:
                elem_val = ((-1) ** sign) * (man / (2**man_width)) * (2 ** (1 - bias))
        elif exp == (1 << exp_width) - 1:
            elem_val = float("inf") if man == 0 else float("nan")
            if sign:
                elem_val = -elem_val
        else:
            elem_val = ((-1) ** sign) * (1 + man / (2**man_width)) * (2 ** (exp - bias))

        scale_val = 2 ** (scale - scale_bias)
        values.append(elem_val * scale_val)

    return np.array(values, dtype=np.float32)


def print_tensor_info(name: str, tensor: torch.Tensor, num_per_row: int = 8):
    """Print tensor information and values."""
    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")
    print(f"Shape: {list(tensor.shape)}")
    print(f"Dtype: {tensor.dtype}")

    flat = tensor.flatten().float()
    print(f"Range: [{flat.min().item():.6f}, {flat.max().item():.6f}]")
    print(f"Mean: {flat.mean().item():.6f}")
    print(f"Total elements: {flat.numel()}")

    print("\nFirst 64 values:")
    for i in range(0, min(64, len(flat)), num_per_row):
        vals = flat[i : i + num_per_row].tolist()
        idx_str = f"[{i:4d}]"
        val_str = " ".join(f"{v:10.4f}" for v in vals)
        print(f"  {idx_str} {val_str}")


def print_mx_data(
    name: str,
    elements: np.ndarray,
    scales: np.ndarray,
    fp_values: np.ndarray,
    block_size: int = 8,
    num_blocks_to_show: int = 8,
):
    """Print MX format data with raw values and converted FP."""
    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"{'=' * 70}")
    print(f"Total elements: {len(elements)}")
    print(f"Total scales: {len(scales)}")
    print(f"Block size: {block_size}")

    scale_bias = 127

    print(f"\nRaw MX Data (first {num_blocks_to_show} blocks):")
    for blk in range(min(num_blocks_to_show, len(scales))):
        start = blk * block_size
        end = min(start + block_size, len(elements))
        blk_elems = elements[start:end]
        scale = scales[blk]
        exp = int(scale) - scale_bias

        elem_hex = " ".join(f"0x{e:02X}" for e in blk_elems)
        print(f"  Block {blk}: scale=0x{scale:02X} ({scale}, exp={exp:+d})")
        print(f"    Elements: [{elem_hex}]")

        blk_fp = fp_values[start:end]
        fp_str = " ".join(f"{v:8.4f}" for v in blk_fp)
        print(f"    FP vals:  [{fp_str}]")

    print("\nConverted FP values (first 64):")
    for i in range(0, min(64, len(fp_values)), 8):
        vals = fp_values[i : i + 8]
        idx_str = f"[{i:4d}]"
        val_str = " ".join(f"{v:10.4f}" for v in vals)
        print(f"  {idx_str} {val_str}")


def main():
    parser = argparse.ArgumentParser(description="View HBM data in floating point format")
    parser.add_argument("--workload-dir", "-w", type=str, required=True, help="Path to workload build directory")
    parser.add_argument("--num-blocks", "-n", type=int, default=8, help="Number of blocks to show in detail")
    args = parser.parse_args()

    workload_dir = Path(args.workload_dir)

    # Load verification params
    params_file = workload_dir / "verification_params.json"
    if not params_file.exists():
        params_file = workload_dir / "comparison_params.json"

    if params_file.exists():
        with open(params_file) as f:
            params = json.load(f)
    else:
        params = {}

    mx_format = params.get("mx_format", "mxint").lower()
    num_elements = params.get("num_elements", 1024)
    block_size = params.get("block_size", 8)
    scale_width = params.get("scale_width", 8)

    if mx_format == "mxint":
        element_width = params.get("man_width", 8)
    else:
        exp_width = params.get("exp_width", 4)
        man_width = params.get("man_width", 3)
        element_width = 1 + exp_width + man_width

    print(f"Workload: {workload_dir}")
    print(f"MX Format: {mx_format.upper()}")
    print(f"Element width: {element_width}, Scale width: {scale_width}, Block size: {block_size}")

    # =========================================================================
    # 1. Print original tensors (act_tensor.pt, weights if exists)
    # =========================================================================
    act_file = workload_dir / "act_tensor.pt"
    if act_file.exists():
        act_tensor = torch.load(act_file, weights_only=True)
        print_tensor_info("ORIGINAL ACTIVATION (act_tensor.pt)", act_tensor)

    weight_file = workload_dir / "weights.pt"
    if weight_file.exists():
        weight_tensor = torch.load(weight_file, weights_only=True)
        print_tensor_info("ORIGINAL WEIGHTS (weights.pt)", weight_tensor)

    # =========================================================================
    # 2. Print initial HBM (hbm.mem) converted to FP
    # =========================================================================
    hbm_init_file = workload_dir / "hbm.mem"
    if hbm_init_file.exists():
        rows = parse_hbm_mem_file(hbm_init_file)
        elements, scales = extract_mx_data(
            rows,
            start_row=0,
            num_elements=num_elements,
            element_width=element_width,
            scale_width=scale_width,
            block_size=block_size,
        )

        if mx_format == "mxint":
            fp_values = mxint_to_float(
                elements, scales, int_width=element_width, scale_width=scale_width, block_size=block_size
            )
        else:
            fp_values = mxfp_to_float(
                elements,
                scales,
                exp_width=exp_width,
                man_width=man_width,
                scale_width=scale_width,
                block_size=block_size,
            )

        print_mx_data(
            "INITIAL HBM (hbm.mem) - Activation Region", elements, scales, fp_values, block_size, args.num_blocks
        )

    # =========================================================================
    # 3. Print result HBM (hbm_result.mem) converted to FP
    # =========================================================================
    hbm_result_file = workload_dir / "hbm_result.mem"
    if hbm_result_file.exists():
        rows = parse_hbm_mem_file(hbm_result_file)
        elements, scales = extract_mx_data(
            rows,
            start_row=0,
            num_elements=num_elements,
            element_width=element_width,
            scale_width=scale_width,
            block_size=block_size,
        )

        if mx_format == "mxint":
            fp_values = mxint_to_float(
                elements, scales, int_width=element_width, scale_width=scale_width, block_size=block_size
            )
        else:
            fp_values = mxfp_to_float(
                elements,
                scales,
                exp_width=exp_width,
                man_width=man_width,
                scale_width=scale_width,
                block_size=block_size,
            )

        print_mx_data(
            "RESULT HBM (hbm_result.mem) - After Simulation", elements, scales, fp_values, block_size, args.num_blocks
        )

    # =========================================================================
    # 4. Print golden result
    # =========================================================================
    golden_file = workload_dir / "golden_result.pt"
    if golden_file.exists():
        golden_tensor = torch.load(golden_file, weights_only=True)
        print_tensor_info("GOLDEN RESULT (golden_result.pt)", golden_tensor)

    # =========================================================================
    # 5. Summary comparison
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    if hbm_init_file.exists() and hbm_result_file.exists():
        init_rows = parse_hbm_mem_file(hbm_init_file)
        result_rows = parse_hbm_mem_file(hbm_result_file)

        diff_count = sum(1 for i, r in zip(init_rows, result_rows) if i != r)
        print(f"HBM rows that changed: {diff_count} / {len(init_rows)}")

        if diff_count > 0:
            print("First 5 changed rows:")
            shown = 0
            for idx, (i, r) in enumerate(zip(init_rows, result_rows)):
                if i != r and shown < 5:
                    print(f"  Row {idx}: init != result")
                    shown += 1


if __name__ == "__main__":
    main()
