import numpy as np
import re
import os
import torch


def parse_golden_output(golden_file_path):
    """
    Parse the "Original Output" section from golden_result.txt.

    Args:
        golden_file_path: Path to the golden_result.txt file

    Returns:
        numpy array: Flattened 1D array of all values from Original Output
    """
    with open(golden_file_path) as f:
        content = f.read()

    # Find the "Original Output:" section. Matches from "Original Output: ["
    # through the last "]" in the file (handles nested brackets for 2D+ tensors).
    match = re.search(r"Original Output:\s*(.*)", content, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'Original Output' section in golden file")

    # Extract the values section. Nested brackets and commas are stripped by the
    # numeric parser below, so we keep everything after "Original Output:" (may
    # include trailing sections — those will fail float() and be skipped).
    values_text = match.group(1)

    # Parse all floating point numbers (handles negative, positive, scientific notation).
    # Use a regex that finds numeric tokens anywhere — this handles values adjacent to
    # brackets or commas (e.g. "[-0.45" at row start or "1.23]" at row end) which
    # whitespace-split + float() would silently drop.
    num_re = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?|-?\d+(?:[eE][-+]?\d+)?")
    values = [float(m) for m in num_re.findall(values_text)]

    return np.array(values, dtype=np.float32)


def read_bin_file_as_array(
    bin_file, exp_width, man_width, row_dim, num_bytes_per_val=2, start_row_idx=0, num_rows=None
):
    """
    Read binary file and convert to numpy array (similar to view_bin_file_by_row but returns array).
    Uses the same row-based indexing logic as view_bin_file_by_row.

    Args:
        bin_file: Path to binary file
        exp_width: Number of bits for exponent
        man_width: Number of bits for mantissa
        row_dim: Number of values per row (must match view_bin_file_by_row)
        num_bytes_per_val: Number of bytes per value (default 2 for BF16)
        start_row_idx: Starting row index
        num_rows: Number of rows to read (None = read all remaining rows)

    Returns:
        numpy array: Flattened 1D array of values, respecting row boundaries
    """
    sign_width = 1
    total_width = sign_width + exp_width + man_width
    if total_width > num_bytes_per_val * 8:
        raise ValueError("num_bytes_per_val is too small for given bit widths.")

    def raw_to_fp(bits_val):
        """Convert raw bits to floating point value."""
        sign = (bits_val >> (exp_width + man_width)) & 0x1
        exponent = (bits_val >> man_width) & ((1 << exp_width) - 1)
        mantissa = bits_val & ((1 << man_width) - 1)
        bias = (1 << (exp_width - 1)) - 1 if exp_width > 0 else 0

        if exp_width == 0:
            base = float(mantissa)
        else:
            if exponent == 0:
                if mantissa == 0:
                    return 0.0 if sign == 0 else -0.0
                base = mantissa / (2**man_width)
                exp_val = 1 - bias
                return ((-1) ** sign) * base * (2**exp_val)
            elif exponent == (1 << exp_width) - 1:
                if mantissa == 0:
                    return float("-inf") if sign else float("inf")
                else:
                    return float("nan")
            else:
                base = 1 + mantissa / (2**man_width)
                exp_val = exponent - bias
                return ((-1) ** sign) * base * (2**exp_val)
        return ((-1) ** sign) * base

    with open(bin_file, "rb") as f:
        data = f.read()

    num_vals = len(data) // num_bytes_per_val
    total_rows = (num_vals + row_dim - 1) // row_dim

    # Calculate which rows to read
    end_row_idx = total_rows if num_rows is None else start_row_idx + num_rows

    values = []
    # Iterate through rows, matching the logic of view_bin_file_by_row
    for row_idx in range(start_row_idx, end_row_idx):
        for col_idx in range(row_dim):
            val_idx = row_idx * row_dim + col_idx
            if val_idx >= num_vals:
                # Reached end of data, pad with None or break
                break
            chunk = data[val_idx * num_bytes_per_val : (val_idx + 1) * num_bytes_per_val]
            if not chunk or len(chunk) < num_bytes_per_val:
                break
            # Use little-endian byte order to match Rust's byte packing
            bits_val = int.from_bytes(chunk, byteorder="little")
            float_val = raw_to_fp(bits_val)
            values.append(float_val)

    return np.array(values, dtype=np.float32)


def reorder_stride_mode(data, num_batches=4, elements_per_batch=128, stride=64):
    """
    Reorder stride-mode data to batch-wise layout.

    Stride mode layout (how data is stored in VRAM):
        For elements_per_batch=128: 2 chunks per batch
        [Batch0[0:64], Batch1[0:64], Batch2[0:64], Batch3[0:64],
         Batch0[64:128], Batch1[64:128], Batch2[64:128], Batch3[64:128]]

        For elements_per_batch=256: 4 chunks per batch
        [Batch0[0:64], Batch1[0:64], Batch2[0:64], Batch3[0:64],
         Batch0[64:128], Batch1[64:128], Batch2[64:128], Batch3[64:128],
         Batch0[128:192], Batch1[128:192], Batch2[128:192], Batch3[128:192],
         Batch0[192:256], Batch1[192:256], Batch2[192:256], Batch3[192:256]]

    Batch-wise layout (how golden data is organized):
        [Batch0[0:elements_per_batch], Batch1[...], ...]

    Args:
        data: 1D numpy array in stride mode
        num_batches: Number of batches (default 4)
        elements_per_batch: Elements per batch (default 128)
        stride: Stride size in stride mode (default 64, typically mlen)

    Returns:
        Reordered 1D numpy array in batch-wise layout
    """
    chunk_size = stride  # Stride mode uses chunks of 'stride' elements (typically mlen=64)
    chunks_per_batch = elements_per_batch // stride
    total_chunks = len(data) // chunk_size
    expected_chunks = num_batches * chunks_per_batch

    if total_chunks != expected_chunks:
        print(f"Warning: Expected {expected_chunks} chunks, got {total_chunks}")

    # Reshape into chunks: [chunk0, chunk1, ..., chunk_n]
    chunks = data.reshape(total_chunks, chunk_size)
    print(f"chunks: {chunks}")

    # Reorder: group all chunks for each batch together
    # For 4 batches with 4 chunks each (256 elements):
    #   batch0: chunks 0, 4, 8, 12
    #   batch1: chunks 1, 5, 9, 13
    #   batch2: chunks 2, 6, 10, 14
    #   batch3: chunks 3, 7, 11, 15
    reordered_chunks = []
    print(f"chunks shape: {chunks.shape}")
    print(f"num_batches: {num_batches}")
    print(f"chunks_per_batch: {chunks_per_batch}")
    for batch_idx in range(num_batches):
        for chunk_group in range(chunks_per_batch):
            chunk_idx = chunk_group * num_batches + batch_idx
            reordered_chunks.append(chunks[chunk_idx])

    return np.concatenate(reordered_chunks)


def reorder_chunk_major(data, seq_len, hidden_dim, mlen):
    """Reorder chunk-major flat [chunks, seq, mlen] into seq-major [seq, hidden]."""
    if hidden_dim % mlen != 0:
        raise ValueError(f"hidden_dim {hidden_dim} must be divisible by mlen {mlen}")
    expected = seq_len * hidden_dim
    if len(data) != expected:
        raise ValueError(f"Chunk-major reorder expected {expected} elements, got {len(data)}")
    chunks = hidden_dim // mlen
    return data.reshape(chunks, seq_len, mlen).transpose(1, 0, 2).reshape(-1)


def slice_rows(data, row_dim, slice_per_row, num_rows):
    """
    Extract the first slice_per_row elements from each row.

    Args:
        data: 1D numpy array with num_rows * row_dim elements
        row_dim: Number of elements per row (e.g., 64)
        slice_per_row: Number of elements to extract from each row (e.g., 16)
        num_rows: Number of rows

    Returns:
        1D numpy array with num_rows * slice_per_row elements
    """
    if len(data) < num_rows * row_dim:
        raise ValueError(f"Data length {len(data)} < expected {num_rows * row_dim}")

    # Reshape to (num_rows, row_dim)
    data_2d = data[: num_rows * row_dim].reshape(num_rows, row_dim)
    # Slice first slice_per_row elements from each row
    sliced = data_2d[:, :slice_per_row]
    # Flatten back to 1D
    return sliced.flatten()


def compare_vram_with_golden(
    bin_file,
    golden_file,
    exp_width=8,
    man_width=7,
    num_bytes_per_val=2,
    row_dim=64,
    start_row_idx=0,
    num_batches=4,
    num_rows=None,
    use_stride_mode=True,
    elements_per_batch=128,
    stride=None,
    atol=0.2,
    rtol=0.2,
    use_slice_mode=False,
    slice_per_row=None,
    use_chunk_major_mode=False,
    seq_len=None,
    hidden_dim=None,
    mlen=None,
    chunk_major_valid_seq_len=None,
    visible_lane_positions=None,
):
    """
    Compare VRAM binary file output with golden reference from golden_result.txt.

    Uses torch.allclose-style comparison: |golden - sim| <= atol + rtol * |golden|

    Args:
        bin_file: Path to binary file to compare
        golden_file: Path to golden_result.txt file
        exp_width: Exponent width for binary file parsing
        man_width: Mantissa width for binary file parsing
        num_bytes_per_val: Bytes per value in binary file
        row_dim: Row dimension (for determining which rows to compare)
        start_row_idx: Starting row index to compare
        num_rows: Number of rows to compare (None = compare all)
        tolerance: Legacy tolerance for relative error reporting
        use_stride_mode: Whether to reorder data from stride mode to batch-wise layout
        stride: Stride chunk size used in stride mode. Defaults to row_dim when not provided.
        atol: Absolute tolerance for allclose comparison (default 0.01 for BF16)
        rtol: Relative tolerance for allclose comparison (default 0.01 = 1%)
        use_slice_mode: Whether to extract first slice_per_row elements from each row (default False)
        slice_per_row: Number of elements to extract per row when use_slice_mode=True

    Returns:
        dict: Dictionary containing comparison metrics:
            - 'mse': Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'max_error': Maximum absolute error
            - 'relative_error': Mean relative error
            - 'match_rate': Percentage of values passing allclose test
            - 'allclose_pass': Boolean indicating if all values pass
            - 'golden_shape': Shape of golden array
            - 'simulated_shape': Shape of simulated array
            - 'errors': Array of absolute errors
    """
    # Parse golden output and quantize to bfloat16 for fair comparison with hardware
    # PLENA uses bfloat16 (8 exp, 7 mantissa), not IEEE float16 (5 exp, 10 mantissa)
    golden_np = parse_golden_output(golden_file)
    golden_values = torch.tensor(golden_np, dtype=torch.bfloat16)

    # Read binary file (now properly handles row-based indexing)
    simulated_np = read_bin_file_as_array(
        bin_file, exp_width, man_width, row_dim, num_bytes_per_val, start_row_idx, num_rows
    )

    # Apply slice mode: extract first slice_per_row elements from each row
    print(f"use_slice_mode: {use_slice_mode}")
    print(f"slice_per_row: {slice_per_row}")
    if use_slice_mode and slice_per_row is not None:
        simulated_np = slice_rows(simulated_np, row_dim, slice_per_row, num_rows)
        # Golden may already be compact (batch-wise visible width), e.g. when
        # simulator output is sliced from padded-head rows while reference is
        # generated directly at visible hidden width.
        expected_expanded_len = num_rows * row_dim
        expected_compact_len = num_batches * elements_per_batch
        if len(golden_np) == expected_expanded_len:
            golden_np = slice_rows(golden_np, row_dim, slice_per_row, num_rows)
            golden_values = torch.tensor(golden_np, dtype=torch.bfloat16)
        elif len(golden_np) == expected_compact_len:
            # Already compact; keep as-is.
            pass
        else:
            raise ValueError(
                f"Unexpected golden length for slice mode: {len(golden_np)} "
                f"(expected expanded={expected_expanded_len} or compact={expected_compact_len})"
            )
        print(f"After slicing: simulated={len(simulated_np)} elements, golden={len(golden_np)} elements")

    # Reorder stride-mode data to match batch-wise golden layout
    print(f"use_stride_mode: {use_stride_mode}")
    print(f"num_batches: {num_batches}")
    print(f"elements_per_batch: {elements_per_batch}")
    effective_stride = row_dim if stride is None else stride
    print(f"stride: {effective_stride}")
    if use_stride_mode:
        simulated_np = reorder_stride_mode(simulated_np, num_batches, elements_per_batch, stride=effective_stride)

    if use_chunk_major_mode:
        if seq_len is None or hidden_dim is None or mlen is None:
            raise ValueError("Chunk-major mode requires seq_len, hidden_dim, and mlen")
        simulated_np = reorder_chunk_major(simulated_np, seq_len=seq_len, hidden_dim=hidden_dim, mlen=mlen)
        if chunk_major_valid_seq_len is not None:
            valid = int(chunk_major_valid_seq_len)
            if valid < 0 or valid > int(seq_len):
                raise ValueError(
                    f"Invalid chunk_major_valid_seq_len={valid}; expected 0 <= valid <= seq_len ({seq_len})"
                )
            simulated_np = simulated_np.reshape(int(seq_len), int(hidden_dim))[:valid, :].reshape(-1)

    simulated_values = torch.tensor(simulated_np, dtype=torch.bfloat16)

    # Ensure dimensions match by truncating to the smaller size
    min_len = min(len(golden_values), len(simulated_values))
    golden_values = golden_values[:min_len]
    simulated_values = simulated_values[:min_len]
    print(f"golden_values: {golden_values}")
    print(f"simulated_values: {simulated_values}")

    if len(golden_values) == 0:
        raise ValueError("No values to compare")

    # Compute errors in bfloat16
    errors = torch.abs(golden_values - simulated_values)

    # Compute metrics
    mse = torch.mean((golden_values - simulated_values) ** 2).item()
    mae = torch.mean(errors).item()
    max_error = torch.max(errors).item()

    # Relative error (avoid division by zero)
    abs_golden = torch.abs(golden_values)
    relative_errors = torch.where(abs_golden > 1e-10, errors / abs_golden, errors)
    mean_relative_error = torch.mean(relative_errors).item()

    # Old match rate (relative error only): |err| / |golden| <= rtol
    within_relative_tolerance = relative_errors <= rtol
    relative_match_rate = torch.sum(within_relative_tolerance).item() / len(relative_errors) * 100.0

    # New match rate using torch.allclose formula: |a - b| <= atol + rtol * |b|
    # This is more appropriate for floating point comparison as it handles
    # both small values (where atol dominates) and large values (where rtol dominates)
    tolerance_threshold = atol + rtol * abs_golden
    within_tolerance = errors <= tolerance_threshold
    allclose_match_rate = torch.sum(within_tolerance).item() / len(errors) * 100.0

    # Pass if at least 90% of values are within tolerance
    allclose_pass = allclose_match_rate >= 90.0

    visible_lane_metrics = None
    padded_lane_metrics = None
    lane_partition_source = None

    if use_chunk_major_mode and hidden_dim is not None and int(hidden_dim) > 0:
        hidden_dim_i = int(hidden_dim)
        row_count = len(errors) // hidden_dim_i
        if row_count > 0:
            usable = row_count * hidden_dim_i
            errors_matrix = errors[:usable].reshape(row_count, hidden_dim_i)
            abs_golden_matrix = abs_golden[:usable].reshape(row_count, hidden_dim_i)
            sim_abs_matrix = torch.abs(simulated_values[:usable]).reshape(row_count, hidden_dim_i)

            visible_idx = None
            if visible_lane_positions is not None:
                idx = torch.tensor(visible_lane_positions, dtype=torch.long)
                idx = idx[(idx >= 0) & (idx < hidden_dim_i)]
                visible_idx = torch.unique(idx, sorted=True)
                lane_partition_source = "provided_visible_lane_positions"
            elif elements_per_batch is not None and int(elements_per_batch) < hidden_dim_i:
                # Legacy fallback: assumes visible lanes are a contiguous prefix.
                visible_idx = torch.arange(int(elements_per_batch), dtype=torch.long)
                lane_partition_source = "contiguous_prefix_heuristic"

            if visible_idx is not None and visible_idx.numel() > 0:
                padded_mask = torch.ones(hidden_dim_i, dtype=torch.bool)
                padded_mask[visible_idx] = False
                padded_idx = torch.where(padded_mask)[0]

                def _lane_stats(idx_tensor):
                    lane_errors = errors_matrix[:, idx_tensor].reshape(-1)
                    lane_abs_golden = abs_golden_matrix[:, idx_tensor].reshape(-1)
                    lane_tolerance = atol + rtol * lane_abs_golden
                    lane_match_rate = torch.mean((lane_errors <= lane_tolerance).float()).item() * 100.0
                    lane_p99 = (
                        torch.quantile(lane_errors.float(), 0.99).item()
                        if lane_errors.numel() > 1
                        else lane_errors.max().item()
                    )
                    return {
                        "count": int(lane_errors.numel()),
                        "mae": float(torch.mean(lane_errors).item()),
                        "max_error": float(torch.max(lane_errors).item()),
                        "p99_error": float(lane_p99),
                        "allclose_match_rate": float(lane_match_rate),
                    }

                visible_lane_metrics = _lane_stats(visible_idx)
                if padded_idx.numel() > 0:
                    padded_lane_metrics = _lane_stats(padded_idx)
                    padded_nonzero_rate = torch.mean((sim_abs_matrix[:, padded_idx] > 0).float()).item() * 100.0
                    padded_lane_metrics["nonzero_rate"] = float(padded_nonzero_rate)

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "relative_error": mean_relative_error,
        "relative_match_rate": relative_match_rate,
        "allclose_match_rate": allclose_match_rate,
        "match_rate": allclose_match_rate,  # Keep for backwards compatibility
        "allclose_pass": allclose_pass,
        "atol": atol,
        "rtol": rtol,
        "golden_shape": tuple(golden_values.shape),
        "simulated_shape": tuple(simulated_values.shape),
        "errors": errors,
        "tolerance_threshold": tolerance_threshold,
        "golden_values": golden_values,
        "simulated_values": simulated_values,
        "visible_lane_metrics": visible_lane_metrics,
        "padded_lane_metrics": padded_lane_metrics,
        "lane_partition_source": lane_partition_source,
    }


def print_comparison_results(results, verbose=False, comparison_params=None):
    """
    Print comparison results in a readable format.

    Args:
        results: Dictionary returned by compare_vram_with_golden
        verbose: If True, print detailed error statistics
        comparison_params: Optional dict with comparison parameters to print
    """
    if comparison_params:
        print("\nComparison Configuration:")
        print(f"  Start Row: {comparison_params.get('start_row_idx', 'N/A')}")
        print(f"  Num Rows: {comparison_params.get('num_rows', 'N/A')}")
        print(f"  Num Batches: {comparison_params.get('num_batches', 'N/A')}")
        print(f"  Elements per Batch: {comparison_params.get('elements_per_batch', 'N/A')}\n")

    print("=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print("Error Metrics:")
    print(f"  Mean Squared Error (MSE):     {results['mse']:.6e}")
    print(f"  Mean Absolute Error (MAE):    {results['mae']:.6e}")
    print(f"  Max Absolute Error:           {results['max_error']:.6f}")
    print(f"  Mean Relative Error:          {results['relative_error']:.6f}")
    print()
    rtol = results.get("rtol", 0.1)
    print(f"Relative Error Check (|err|/|golden| <= {rtol}):")
    relative_match_rate = results.get("relative_match_rate")
    if relative_match_rate is not None:
        print(f"  Match Rate:                   {relative_match_rate:.2f}%")
    else:
        print("  Match Rate:                   N/A")
    print()
    print("Allclose Check (|err| <= atol + rtol * |golden|):")
    atol = results.get("atol")
    if atol is not None:
        print(f"  atol={atol}, rtol={rtol}")
    else:
        print(f"  atol=N/A, rtol={rtol}")
    allclose_match_rate = results.get("allclose_match_rate")
    if allclose_match_rate is not None:
        print(f"  Match Rate:                   {allclose_match_rate:.2f}%")
    else:
        print("  Match Rate:                   N/A")
    allclose_status = "PASS" if results.get("allclose_pass", False) else "FAIL"
    print(f"  All Values Pass:              {allclose_status}")
    print()

    visible_lane_metrics = results.get("visible_lane_metrics")
    if visible_lane_metrics is not None:
        print("Visible Lane Metrics:")
        print(f"  Source:                       {results.get('lane_partition_source', 'unknown')}")
        print(f"  Elements:                     {visible_lane_metrics['count']}")
        print(f"  MAE:                          {visible_lane_metrics['mae']:.6e}")
        print(f"  Max Absolute Error:           {visible_lane_metrics['max_error']:.6f}")
        print(f"  P99 Absolute Error:           {visible_lane_metrics['p99_error']:.6f}")
        print(f"  Allclose Match Rate:          {visible_lane_metrics['allclose_match_rate']:.2f}%")

        padded_lane_metrics = results.get("padded_lane_metrics")
        if padded_lane_metrics is not None:
            print("Padded Lane Metrics:")
            print(f"  Elements:                     {padded_lane_metrics['count']}")
            print(f"  MAE:                          {padded_lane_metrics['mae']:.6e}")
            print(f"  Max Absolute Error:           {padded_lane_metrics['max_error']:.6f}")
            print(f"  P99 Absolute Error:           {padded_lane_metrics['p99_error']:.6f}")
            print(f"  Allclose Match Rate:          {padded_lane_metrics['allclose_match_rate']:.2f}%")
            if "nonzero_rate" in padded_lane_metrics:
                print(f"  Sim Nonzero Rate:             {padded_lane_metrics['nonzero_rate']:.2f}%")
        print()

    if verbose:
        errors = results["errors"]
        print("Error Statistics:")
        print(f"  Min error:                  {torch.min(errors).item():.6f}")
        print(f"  Max error:                  {torch.max(errors).item():.6f}")
        print(f"  Median error:               {torch.median(errors).item():.6f}")
        print(f"  Std deviation:             {torch.std(errors).item():.6f}")
        print()

        top_5_indices = torch.argsort(errors, descending=True)[:5]
        print("Top 5 Largest Errors:")
        for idx in top_5_indices:
            print(
                f"  Index {idx.item():4d}: Golden={results['golden_values'][idx].item():8.4f}, "
                f"Simulated={results['simulated_values'][idx].item():8.4f}, "
                f"Error={errors[idx].item():.6f}"
            )
        print()


def read_hbm_bin_file_as_array(
    bin_file,
    exp_width,
    man_width,
    start_byte_offset=0,
    num_elements=None,
    element_bytes=1,
    scale_width=None,
    block_size=None,
    scale_offset=None,
):
    """
    Read HBM binary file and convert mx data type to numpy array.

    Args:
        bin_file: Path to HBM binary file
        exp_width: Exponent width for mx element data type
        man_width: Mantissa width for mx element data type
        start_byte_offset: Starting byte offset in HBM for elements
        num_elements: Number of elements to read (None = read all remaining)
        element_bytes: Bytes per element (for mx data type)
        scale_width: Bits per scale (required for mx data type)
        block_size: Number of elements per block (required for mx data type)
        scale_offset: Byte offset from element start to scale start (required for mx data type)

    Returns:
        numpy array: Flattened 1D array of FP32 values
    """
    sign_width = 1
    total_width = sign_width + exp_width + man_width
    if total_width > element_bytes * 8:
        raise ValueError("element_bytes is too small for given bit widths.")
    print("read settings:")
    print(f"  bin_file: {bin_file}")
    print(f"  start_byte_offset: {start_byte_offset}")
    print(f"  num_elements: {num_elements}")
    print(f"  element_bytes: {element_bytes}")
    print(f"  scale_width: {scale_width}")
    print(f"  block_size: {block_size}")
    print(f"  scale_offset: {scale_offset}")
    # Print out the bin file contents (raw bytes) as hex for debugging
    with open(bin_file, "rb") as f:
        contents = f.read()
        print(f"\n========== Hexdump of {bin_file} ==========")
        for i in range(0, 2000, 16):
            hex_chunk = " ".join(f"{b:02x}" for b in contents[i : i + 16])
            ascii_chunk = "".join(chr(b) if 32 <= b <= 126 else "." for b in contents[i : i + 16])
            print(f"{i:08x}: {hex_chunk:<47}  {ascii_chunk}")
        print("========== End of hexdump ==========\n")

    def raw_to_fp(bits_val, exp_w, man_w):
        """Convert raw bits to floating point value."""
        sign = (bits_val >> (exp_w + man_w)) & 0x1
        exponent = (bits_val >> man_w) & ((1 << exp_w) - 1)
        mantissa = bits_val & ((1 << man_w) - 1)
        bias = (1 << (exp_w - 1)) - 1 if exp_w > 0 else 0

        if exp_w == 0:
            base = float(mantissa)
        else:
            if exponent == 0:
                if mantissa == 0:
                    return 0.0 if sign == 0 else -0.0
                base = mantissa / (2**man_w)
                exp_val = 1 - bias
                return ((-1) ** sign) * base * (2**exp_val)
            elif exponent == (1 << exp_w) - 1:
                if mantissa == 0:
                    return float("-inf") if sign else float("inf")
                else:
                    return float("nan")
            else:
                base = 1 + mantissa / (2**man_w)
                exp_val = exponent - bias
                return ((-1) ** sign) * base * (2**exp_val)
        return ((-1) ** sign) * base

    # Read element bytes - load num_elements elements
    with open(bin_file, "rb") as f:
        f.seek(start_byte_offset)
        if num_elements is None:
            # Read all remaining data
            element_data = f.read()
            num_elements = len(element_data) // element_bytes
        else:
            # Read exactly num_elements elements
            element_bytes_to_read = num_elements * element_bytes
            element_data = f.read(element_bytes_to_read)

    # Check if this is mx data type
    if scale_width is not None and block_size is not None and scale_offset is not None:
        # MX data type: need to read scales and apply them
        scale_bytes_per_scale = (scale_width + 7) // 8  # Round up to bytes
        num_blocks = (num_elements + block_size - 1) // block_size  # Round up

        # Calculate block index from offset (assuming offset = 0 for now)
        # TODO: If offset != 0, we need to know the base address to calculate offset
        element_offset = 0  # Assume offset = 0 (as in h_store_test.py)
        block_index = element_offset // (element_bytes * block_size)

        # Scale start address = start_byte_offset + scale_reg + block_index
        # scale_reg is in bytes (from SCALE_REG, which is batch * hidden_size in elements = bytes for element_bytes=1)
        scale_start_offset = start_byte_offset + scale_offset + block_index * scale_bytes_per_scale

        print("  Scale calculation:")
        print(f"    start_byte_offset: {start_byte_offset} (element start address)")
        print(f"    scale_offset: {scale_offset} (scale_reg value, relative to element base)")
        print(f"    element_offset: {element_offset} (assumed 0, as in test)")
        print(
            f"    block_index: {block_index} (element_offset / (element_bytes={element_bytes} * block_size={block_size}))"
        )
        print(
            f"    scale_start_offset: {scale_start_offset} (start_byte_offset + scale_offset + block_index * scale_bytes_per_scale)"
        )
        print(f"    num_blocks: {num_blocks}, scale_bytes_per_scale: {scale_bytes_per_scale}")

        # Read scale bytes
        with open(bin_file, "rb") as f:
            f.seek(scale_start_offset)
            scale_data = f.read(num_blocks * scale_bytes_per_scale)

        # Convert scales to FP32
        scales = []
        for i in range(num_blocks):
            scale_bytes = scale_data[i * scale_bytes_per_scale : (i + 1) * scale_bytes_per_scale]
            if len(scale_bytes) < scale_bytes_per_scale:
                scale_bytes = scale_bytes + b"\x00" * (scale_bytes_per_scale - len(scale_bytes))
            scale_bits = int.from_bytes(scale_bytes, byteorder="little")
            # Scale format for activation mx: exp_width=8, man_width=0 (just exponent)
            # Scale is stored as a minifloat: scale_value = 2^(scale_exp - bias)
            # For scale_width=8: exp_width=8, man_width=0
            scale_exp_width = 8  # Scale exponent width for activation mx
            scale_man_width = 0  # Scale mantissa width for activation mx
            scale_fp = raw_to_fp(scale_bits, scale_exp_width, scale_man_width)
            scales.append(scale_fp)
        # Convert elements to FP32 and apply scales
        values = []
        for i in range(num_elements):
            chunk = element_data[i * element_bytes : (i + 1) * element_bytes]
            if len(chunk) < element_bytes:
                break
            bits_val = int.from_bytes(chunk, byteorder="little")
            element_fp = raw_to_fp(bits_val, exp_width, man_width)
            # Apply scale from the block this element belongs to
            block_idx = i // block_size
            if block_idx < len(scales):
                element_fp *= scales[block_idx]

            values.append(element_fp)
    else:
        # Plain FP data type: no scales
        values = []
        for i in range(num_elements):
            chunk = element_data[i * element_bytes : (i + 1) * element_bytes]
            if len(chunk) < element_bytes:
                break
            bits_val = int.from_bytes(chunk, byteorder="little")
            float_val = raw_to_fp(bits_val, exp_width, man_width)
            values.append(float_val)

    return np.array(values, dtype=np.float32)


def compare_hbm_with_golden(
    hbm_file,
    golden_file,
    exp_width=4,
    man_width=3,
    element_bytes=1,
    start_byte_offset=0,
    num_elements=None,
    num_batches=4,
    elements_per_batch=128,
    tolerance=0.2,
    atol=0.03,
    rtol=0.1,
    scale_width=8,
    block_size=8,
    scale_offset=None,
):
    """
    Compare HBM binary file output with golden reference.

    Args:
        hbm_file: Path to HBM dump binary file
        golden_file: Path to golden_result.txt file
        exp_width: Exponent width for mx element data type
        man_width: Mantissa width for mx element data type
        element_bytes: Bytes per element
        start_byte_offset: Starting byte offset in HBM for elements
        num_elements: Number of elements to compare (None = use golden output length)
        num_batches: Number of batches
        elements_per_batch: Elements per batch
        tolerance: Legacy tolerance
        atol: Absolute tolerance
        rtol: Relative tolerance
        scale_width: Bits per scale (required for mx data type)
        block_size: Number of elements per block (required for mx data type)
        scale_offset: Byte offset from element start to scale start (required for mx data type)

    Returns:
        dict: Comparison results
    """
    # Parse golden output
    golden_np = parse_golden_output(golden_file)
    golden_values = torch.tensor(golden_np, dtype=torch.bfloat16)

    # If num_elements not specified, use the number from golden output
    if num_elements is None:
        num_elements = len(golden_np)

    # Read HBM binary file with mx data type conversion
    simulated_np = read_hbm_bin_file_as_array(
        hbm_file,
        exp_width,
        man_width,
        start_byte_offset,
        num_elements,
        element_bytes,
        scale_width=scale_width,
        block_size=block_size,
        scale_offset=scale_offset,
    )

    print("simulated_np: ", simulated_np)

    # Reshape to match expected layout (considering mx format with blocks)
    # For mx format: elements are stored with scales, need to account for block structure
    # For simplicity, compare flattened arrays
    simulated_values = torch.tensor(simulated_np, dtype=torch.bfloat16)

    # Ensure dimensions match
    min_len = min(len(golden_values), len(simulated_values))
    golden_values = golden_values[:min_len]
    simulated_values = simulated_values[:min_len]

    if len(golden_values) == 0:
        raise ValueError("No values to compare")

    # Compute errors
    errors = torch.abs(golden_values - simulated_values)

    # Compute metrics
    mse = torch.mean((golden_values - simulated_values) ** 2).item()
    mae = torch.mean(errors).item()
    max_error = torch.max(errors).item()

    # Relative error
    abs_golden = torch.abs(golden_values)
    relative_errors = torch.where(abs_golden > 1e-10, errors / abs_golden, errors)
    mean_relative_error = torch.mean(relative_errors).item()

    # Match rate using torch.allclose formula
    tolerance_threshold = atol + rtol * abs_golden
    within_tolerance = errors <= tolerance_threshold
    allclose_match_rate = torch.sum(within_tolerance).item() / len(errors) * 100.0
    allclose_pass = torch.all(within_tolerance).item()

    # Relative error match rate (legacy)
    within_relative_tolerance = relative_errors <= rtol
    relative_match_rate = torch.sum(within_relative_tolerance).item() / len(relative_errors) * 100.0

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "relative_error": mean_relative_error,
        "relative_match_rate": relative_match_rate,
        "allclose_match_rate": allclose_match_rate,
        "match_rate": allclose_match_rate,
        "allclose_pass": allclose_pass,
        "atol": atol,
        "rtol": rtol,
        "golden_shape": tuple(golden_values.shape),
        "simulated_shape": tuple(simulated_values.shape),
        "errors": errors,
        "tolerance_threshold": tolerance_threshold,
        "golden_values": golden_values,
        "simulated_values": simulated_values,
    }


def read_fpsram_bin_file_as_array(bin_file, start_idx=0, num_elements=None):
    """
    Read FPSRAM binary file (f16/half-precision) and convert to numpy array.

    Args:
        bin_file: Path to fpsram_dump.bin file
        start_idx: Starting element index (each element is 2 bytes)
        num_elements: Number of elements to read (None = read all remaining)

    Returns:
        numpy array: Flattened 1D array of FP32 values
    """
    with open(bin_file, "rb") as f:
        data = f.read()

    # FPSRAM is stored as f16 (half-precision float, 2 bytes per value)
    total_elements = len(data) // 2

    if num_elements is None:
        num_elements = total_elements - start_idx

    end_idx = min(start_idx + num_elements, total_elements)

    # Convert bytes to numpy array of float16
    all_values = np.frombuffer(data, dtype=np.float16)
    values = all_values[start_idx:end_idx]

    return values.astype(np.float32)


def compare_fpsram_with_golden(fpsram_file, golden_values, start_idx=0, num_elements=None, atol=0.2, rtol=0.2):
    """
    Compare FPSRAM binary file output with golden reference values.

    Args:
        fpsram_file: Path to fpsram_dump.bin file
        golden_values: Golden reference values (numpy array or torch tensor)
        start_idx: Starting element index in FPSRAM
        num_elements: Number of elements to compare (None = use golden length)
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        dict: Comparison results
    """
    # Convert golden values to numpy if needed
    if isinstance(golden_values, torch.Tensor):
        golden_np = golden_values.float().numpy()
    else:
        golden_np = np.array(golden_values, dtype=np.float32)

    # If num_elements not specified, use golden length
    if num_elements is None:
        num_elements = len(golden_np)

    # Read FPSRAM binary file
    simulated_np = read_fpsram_bin_file_as_array(fpsram_file, start_idx, num_elements)

    # Convert to torch for comparison
    golden_tensor = torch.tensor(golden_np, dtype=torch.float32)
    simulated_tensor = torch.tensor(simulated_np, dtype=torch.float32)

    # Ensure dimensions match
    min_len = min(len(golden_tensor), len(simulated_tensor))
    golden_tensor = golden_tensor[:min_len]
    simulated_tensor = simulated_tensor[:min_len]

    if len(golden_tensor) == 0:
        raise ValueError("No values to compare")

    # Compute errors
    errors = torch.abs(golden_tensor - simulated_tensor)

    # Compute metrics
    mse = torch.mean((golden_tensor - simulated_tensor) ** 2).item()
    mae = torch.mean(errors).item()
    max_error = torch.max(errors).item()

    # Relative error
    abs_golden = torch.abs(golden_tensor)
    relative_errors = torch.where(abs_golden > 1e-10, errors / abs_golden, errors)
    mean_relative_error = torch.mean(relative_errors).item()

    # Match rate using torch.allclose formula
    tolerance_threshold = atol + rtol * abs_golden
    within_tolerance = errors <= tolerance_threshold
    allclose_match_rate = torch.sum(within_tolerance).item() / len(errors) * 100.0
    allclose_pass = allclose_match_rate >= 90.0

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "relative_error": mean_relative_error,
        "allclose_match_rate": allclose_match_rate,
        "match_rate": allclose_match_rate,
        "allclose_pass": allclose_pass,
        "atol": atol,
        "rtol": rtol,
        "golden_shape": tuple(golden_tensor.shape),
        "simulated_shape": tuple(simulated_tensor.shape),
        "errors": errors,
        "golden_values": golden_tensor,
        "simulated_values": simulated_tensor,
    }


if __name__ == "__main__":
    # Example usage
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    golden_file = os.path.join(script_dir, "transactional_emulator", "testbench", "build", "golden_result.txt")
    vram_file = os.path.join(script_dir, "transactional_emulator", "vram_dump.bin")

    if os.path.exists(golden_file) and os.path.exists(vram_file):
        results = compare_vram_with_golden(
            vram_file,
            golden_file,
            exp_width=8,
            man_width=7,
            num_bytes_per_val=2,
            row_dim=64,
            start_row_idx=0,
            num_rows=4,
            use_stride_mode=True,
        )
        print_comparison_results(results, verbose=True)
    else:
        print("Files not found:")
        print(f"  Golden: {golden_file}")
        print(f"  VRAM:   {vram_file}")
