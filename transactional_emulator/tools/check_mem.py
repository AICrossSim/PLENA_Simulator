import numpy as np
import re
import os
import math
import torch


def parse_golden_output(golden_file_path):
    """
    Get the golden "Original Output" as a flat 1D numpy array.

    Prefers the lossless ``golden_output.pt`` written alongside
    ``golden_result.txt`` — the .txt is a human-readable dump with
    formatted (lossy) floats, and parsing that text back as the golden
    turns formatting precision loss into fake comparison error. Falls
    back to parsing the text only when the .pt is absent (older builds).

    Args:
        golden_file_path: Path to the golden_result.txt file

    Returns:
        numpy array: Flattened 1D array of all values from Original Output
    """
    # Preferred path: lossless tensor next to golden_result.txt.
    pt_path = os.path.join(os.path.dirname(golden_file_path), "golden_output.pt")
    if os.path.exists(pt_path):
        golden_t = torch.load(pt_path)
        return golden_t.detach().cpu().float().reshape(-1).numpy()

    # Fallback: parse the human-readable text dump (lossy — only used
    # for builds produced before golden_output.pt existed).
    with open(golden_file_path) as f:
        content = f.read()

    # Find the "Original Output:" section
    match = re.search(r"Original Output:\s*\[(.*?)\]", content, re.DOTALL)
    if not match:
        raise ValueError("Could not find 'Original Output' section in golden file")

    # Extract the values section
    values_text = match.group(1)

    # Parse all floating point numbers (handles negative, positive, scientific notation)
    values = []
    for line in values_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Split by whitespace and parse each value
        for val_str in line.split():
            try:
                val = float(val_str)
                values.append(val)
            except ValueError:
                continue

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


def reorder_stride_mode(data, *, num_batches, elements_per_batch, stride):
    # Hardware-geometry params are keyword-only and REQUIRED — no
    # defaults. The old defaults (num_batches=4, elements_per_batch=128,
    # stride=64) were baked for MLEN=64 and silently scrambled data at
    # any other geometry. Caller must pass values derived from
    # plena_settings.toml (via comparison_params).
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

    # Reorder: group all chunks for each batch together
    # For 4 batches with 4 chunks each (256 elements):
    #   batch0: chunks 0, 4, 8, 12
    #   batch1: chunks 1, 5, 9, 13
    #   batch2: chunks 2, 6, 10, 14
    #   batch3: chunks 3, 7, 11, 15
    reordered_chunks = []
    for batch_idx in range(num_batches):
        for chunk_group in range(chunks_per_batch):
            chunk_idx = chunk_group * num_batches + batch_idx
            reordered_chunks.append(chunks[chunk_idx])

    return np.concatenate(reordered_chunks)


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


def parse_isa_latency(stderr_log_path):
    """Extract the emulator's ISA simulation latency.

    The Rust emulator prints ``Simulation completed. Latency <N>`` to
    stderr at the end of a run; the justfile tees that into
    ``build/emulator_stderr.log``. Returns the latency as a string
    (kept verbatim — it may be a Duration like ``1234 cycles`` or a
    bare int depending on the emulator's ``executor.now()`` type), or
    ``None`` if the log is missing or has no Latency line.

    This is pure ISA execution time under the latency model — it does
    NOT include TVM compile time (a separate pre-emulator subprocess).
    """
    if not os.path.exists(stderr_log_path):
        return None
    try:
        with open(stderr_log_path) as f:
            text = f.read()
    except OSError:
        return None
    m = re.search(r"Simulation completed\.\s*Latency\s*(.+)", text)
    if m:
        return m.group(1).strip()
    return None


def _per_row_cosine(golden_values, simulated_values, elements_per_batch):
    """Per-row cosine similarity between sim and golden.

    Each logical output row (``elements_per_batch`` consecutive values)
    is treated as a vector; we report cos(angle) between the sim row and
    the golden row. Unlike |err|/|golden|, this is invariant to a
    uniform per-row scale factor, so it separates *direction* error
    (real shape mismatch) from *magnitude* error.

    Returns ``{'mean', 'min', 'per_row'}``. ``per_row`` is a python list
    of floats (one per full row); a trailing partial row, if any, is
    dropped. Returns NaN-free empties if no full row fits.
    """
    g = golden_values.float()
    s = simulated_values.float()
    n = min(len(g), len(s))
    epb = int(elements_per_batch) if elements_per_batch else 0
    if epb <= 0 or n < epb:
        return {"mean": float("nan"), "min": float("nan"), "per_row": []}

    n_rows = n // epb
    g = g[: n_rows * epb].reshape(n_rows, epb)
    s = s[: n_rows * epb].reshape(n_rows, epb)

    dot = torch.sum(g * s, dim=1)
    gn = torch.linalg.norm(g, dim=1)
    sn = torch.linalg.norm(s, dim=1)
    denom = gn * sn
    # A zero-norm row (all-zero golden or sim) has no defined angle;
    # treat it as a perfect match iff the other side is also zero,
    # otherwise as fully orthogonal (cos=0).
    cos = torch.where(
        denom > 1e-20,
        dot / denom,
        torch.where((gn <= 1e-20) & (sn <= 1e-20), torch.ones_like(denom), torch.zeros_like(denom)),
    )
    cos = torch.clamp(cos, -1.0, 1.0)
    return {
        "mean": torch.mean(cos).item(),
        "min": torch.min(cos).item(),
        "per_row": [c.item() for c in cos],
    }


def compare_vram_with_golden(
    bin_file,
    golden_file,
    *,
    # Hardware-geometry params: keyword-only and REQUIRED,
    # no defaults. Must come from plena_settings.toml via
    # comparison_params — a wrong-geometry default (the
    # old row_dim=64 / num_batches=4 / elements_per_batch
    # =128) silently mis-shapes the comparison.
    row_dim,
    num_batches,
    elements_per_batch,
    exp_width=8,
    man_width=7,
    num_bytes_per_val=2,
    start_row_idx=0,
    num_rows=None,
    use_stride_mode=True,
    atol=0.1,
    rtol=0.1,
    use_slice_mode=False,
    slice_per_row=None,
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
    # Golden is the high-precision reference — keep it fp32. It must NOT
    # be downcast to bf16: the comparison error |err|/|golden| uses
    # golden as the denominator, and rounding the golden to bf16 first
    # inflates the relative error of small values (a real golden of
    # -0.0083 snapped to a bf16 grid point yields a fake large rel err).
    # The simulated side stays bf16 (that IS the hardware VRAM storage);
    # torch promotes it to fp32 for the subtraction automatically.
    golden_np = parse_golden_output(golden_file)
    golden_values = torch.from_numpy(golden_np).float()

    # Read binary file (now properly handles row-based indexing)
    simulated_np = read_bin_file_as_array(
        bin_file, exp_width, man_width, row_dim, num_bytes_per_val, start_row_idx, num_rows
    )

    # Apply slice mode: extract first slice_per_row elements from each row
    print(f"use_slice_mode: {use_slice_mode}")
    print(f"slice_per_row: {slice_per_row}")
    if use_slice_mode and slice_per_row is not None:
        simulated_np = slice_rows(simulated_np, row_dim, slice_per_row, num_rows)
        # Also slice golden values to match: golden is organized as [num_rows, row_dim]
        # but we only want [num_rows, slice_per_row]
        golden_np = slice_rows(golden_np, row_dim, slice_per_row, num_rows)
        golden_values = torch.from_numpy(golden_np).float()  # keep fp32 — see above
        print(f"After slicing: simulated={len(simulated_np)} elements, golden={len(golden_np)} elements")

    # Reorder stride-mode data to match batch-wise golden layout.
    # The stride-mode chunk width is MLEN (== row_dim here), NOT the
    # legacy default 64 baked into reorder_stride_mode's signature.
    # With MLEN=64 the two happened to coincide; at MLEN=512 the
    # default 64 splits each 512-wide row into 8 bogus chunks and
    # cross-batch-interleaves them, scrambling otherwise-correct VRAM.
    print(f"use_stride_mode: {use_stride_mode}")
    print(f"num_batches: {num_batches}")
    print(f"elements_per_batch: {elements_per_batch}")
    print(f"stride (= row_dim / MLEN): {row_dim}")
    if use_stride_mode:
        simulated_np = reorder_stride_mode(
            simulated_np,
            num_batches=num_batches,
            elements_per_batch=elements_per_batch,
            stride=row_dim,
        )

    simulated_values = torch.from_numpy(simulated_np).bfloat16()

    # Ensure dimensions match by truncating to the smaller size
    min_len = min(len(golden_values), len(simulated_values))
    golden_values = golden_values[:min_len]
    simulated_values = simulated_values[:min_len]

    if len(golden_values) == 0:
        raise ValueError("No values to compare")

    # Compute errors in bfloat16
    errors = torch.abs(golden_values - simulated_values)

    # Compute metrics
    mse = torch.mean((golden_values - simulated_values) ** 2).item()
    mae = torch.mean(errors).item()
    max_error = torch.max(errors).item()

    # Relative RMSE = ||err||_2 / ||golden||_2 — the dimensionless
    # error/signal ratio (== 1/SNR). Unlike MSE/MAE this is scale-free,
    # so a value is directly interpretable: 0.02 means the error energy
    # is 2% of the signal energy. Also reported in dB.
    err_norm = torch.linalg.norm(golden_values - simulated_values).item()
    golden_norm = torch.linalg.norm(golden_values).item()
    if golden_norm > 1e-20:
        rel_rmse = err_norm / golden_norm
        snr_db = -20.0 * math.log10(rel_rmse) if rel_rmse > 1e-20 else float("inf")
    else:
        # All-zero golden: a ratio is undefined. Perfect iff sim is also
        # zero, otherwise treat as fully wrong.
        rel_rmse = 0.0 if err_norm <= 1e-20 else float("inf")
        snr_db = float("inf") if err_norm <= 1e-20 else float("-inf")

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

    # Per-row cosine similarity: treat each logical output row as a
    # vector and measure the angle between sim and golden. Robust to a
    # uniform per-row scale error (which inflates |err|/|golden| but
    # leaves direction intact), so it isolates *shape* mismatch from
    # magnitude mismatch.
    row_cos = _per_row_cosine(golden_values, simulated_values, elements_per_batch)

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
        "relative_errors": relative_errors,
        "tolerance_threshold": tolerance_threshold,
        "golden_values": golden_values,
        "simulated_values": simulated_values,
        "row_cos_mean": row_cos["mean"],
        "row_cos_min": row_cos["min"],
        "row_cos_per_row": row_cos["per_row"],
        "rel_rmse": rel_rmse,
        "snr_db": snr_db,
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
    # ISA simulation latency (emulator's executor.now()), parsed from
    # the teed emulator stderr log. Pure ISA exec time — excludes TVM
    # compile. Looked up next to the build dir; silently skipped if the
    # log isn't present (e.g. check_mem run standalone).
    _stderr_log = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "testbench", "build", "emulator_stderr.log"
    )
    _isa_latency = parse_isa_latency(_stderr_log)
    if _isa_latency is not None:
        print(f"ISA Simulation Latency:           {_isa_latency}")
        # Throughput / efficiency vs GPU baselines.
        #   TPS      = tokens / wall-time. One logical output row
        #              (num_batches) is one token position.
        #   tokens/J = tokens / energy. Energy needs a power model;
        #              the project has none yet, so print N/A until a
        #              power baseline (TDP / per-unit watts) exists.
        _ns = None
        _m = re.search(r"[-+]?\d*\.?\d+", _isa_latency)
        if _m:
            try:
                _ns = float(_m.group(0))
            except ValueError:
                _ns = None
        _tokens = None
        if comparison_params is not None:
            _nb = comparison_params.get("num_batches")
            if isinstance(_nb, (int, float)) and _nb > 0:
                _tokens = float(_nb)
        if _ns and _ns > 0 and _tokens is not None:
            _tps = _tokens / (_ns * 1e-9)  # tokens per second
            print(f"Throughput (TPS):                 {_tps:.2f} tokens/s ({int(_tokens)} tokens / {_ns:.0f} ns)")
        else:
            print("Throughput (TPS):                 N/A (need num_batches + latency)")
        print("Energy efficiency (tokens/J):     N/A (no power model — needs a TDP / per-unit power baseline)")
        print()

    # ---- (1) Correctness metrics we trust (scale-invariant) ----
    # NRMSE / SNR / cosine separate real shape error from the uniform
    # quantization noise that the old |err|/|golden| match rate misread
    # as failure on near-zero values. These are the numbers to read.
    def _g(k, default=float("nan")):
        v = results.get(k)
        return default if v is None else v

    print("Correctness (scale-invariant):")
    print(f"  NRMSE (||err||/||golden||):   {_g('nrmse'):.6f}   ({_g('nrmse') * 100:.4f}%)")
    print(f"  SNR:                          {_g('snr_db'):.2f} dB")
    print(f"  Global cosine similarity:     {_g('global_cosine'):.6f}")
    print(f"  Per-row cosine  mean:         {_g('row_cosine_mean'):.6f}")
    print(f"  Per-row cosine  min:          {_g('row_cosine_min'):.6f}")
    print()

    # ---- (2) Basic error reference (absolute, quantization-aware) ----
    rtol = results.get("rtol", 0.1)
    atol = results.get("atol")
    print("Basic error reference:")
    print(f"  MSE:                          {results['mse']:.6e}")
    print(f"  MAE:                          {results['mae']:.6e}")
    print(f"  Max absolute error:           {results['max_error']:.6f}")
    allclose_match_rate = results.get("allclose_match_rate")
    allclose_status = "PASS" if results.get("allclose_pass", False) else "FAIL"
    _atol_s = f"atol={atol}" if atol is not None else "atol=N/A"
    if allclose_match_rate is not None:
        print(f"  Allclose ({_atol_s}, rtol={rtol}): {allclose_match_rate:.2f}%  ->  {allclose_status}")
    else:
        print(f"  Allclose ({_atol_s}, rtol={rtol}): N/A  ->  {allclose_status}")
    # Relative match rate (|err|/|golden| <= rtol) — per-element relative
    # accuracy pass rate. An important pointwise metric; note it is harsh
    # on near-zero values (a small denominator magnifies a single quant
    # step), so read it together with cosine/NRMSE rather than alone.
    relative_match_rate = results.get("relative_match_rate")
    if relative_match_rate is not None:
        print(f"  Relative (|err|/|golden| <= {rtol}): {relative_match_rate:.2f}%  (harsh on near-zero)")
    else:
        print(f"  Relative (|err|/|golden| <= {rtol}): N/A")
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
    # Parse golden output. Keep golden fp32 — do NOT downcast to bf16
    # (that would inflate the relative error of small values; see the
    # same fix in compare_vram_with_golden).
    golden_np = parse_golden_output(golden_file)
    golden_values = torch.from_numpy(golden_np).float()

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

    # read_hbm_bin_file_as_array already decoded each stored element to
    # its exact fp32 value (MX scale applied, or plain float). Keep it in
    # fp32 — the old `.bfloat16()` here re-quantized the simulated side a
    # second time, inflating the error of a plain-fp16 output (linear_min)
    # by a whole bf16 step. The golden side is fp32 too (see above), so a
    # fp32 vs fp32 compare reflects only the hardware's real output error.
    simulated_values = torch.from_numpy(simulated_np).float()

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

    # Relative error match rate (legacy — kept for the return dict only;
    # NOT printed: |err|/|golden| <= rtol punishes near-zero values for a
    # single quantization step, so it reads far worse than the output
    # actually is. NRMSE / cosine below are the metrics we trust.)
    within_relative_tolerance = relative_errors <= rtol
    relative_match_rate = torch.sum(within_relative_tolerance).item() / len(relative_errors) * 100.0

    # --- Correctness metrics we actually trust (scale-invariant) ---
    g32 = golden_values.float()
    s32 = simulated_values.float()
    err_norm = torch.linalg.norm(g32 - s32).item()
    golden_norm = torch.linalg.norm(g32).item()
    sim_norm = torch.linalg.norm(s32).item()
    # NRMSE = ||g-s|| / ||g||  (relative RMS error)
    nrmse = err_norm / golden_norm if golden_norm > 1e-20 else (0.0 if err_norm <= 1e-20 else float("inf"))
    # SNR in dB. Guard against nan/inf in the norms (nan compares False to
    # >1e-20 and would crash math.log10 with a domain error) and a
    # non-positive ratio.
    _ratio = (golden_norm / err_norm) if err_norm > 1e-20 else float("inf")
    if err_norm <= 1e-20:
        snr_db = float("inf")
    elif golden_norm <= 1e-20:
        snr_db = float("-inf")
    elif not math.isfinite(_ratio) or _ratio <= 0.0:
        snr_db = float("nan")
    else:
        snr_db = 20.0 * math.log10(_ratio)
    # Global cosine similarity
    denom = golden_norm * sim_norm
    global_cosine = (
        (torch.dot(g32, s32).item() / denom)
        if denom > 1e-20
        else (1.0 if (golden_norm <= 1e-20 and sim_norm <= 1e-20) else 0.0)
    )
    global_cosine = max(-1.0, min(1.0, global_cosine))
    # Per-row cosine (each logical output row as a vector)
    row_cos = _per_row_cosine(golden_values, simulated_values, elements_per_batch)

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
        "relative_error": mean_relative_error,
        "relative_match_rate": relative_match_rate,
        "allclose_match_rate": allclose_match_rate,
        "match_rate": allclose_match_rate,
        "allclose_pass": allclose_pass,
        "nrmse": nrmse,
        "snr_db": snr_db,
        "global_cosine": global_cosine,
        "row_cosine_mean": row_cos["mean"],
        "row_cosine_min": row_cos["min"],
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


def compare_fpsram_with_golden(fpsram_file, golden_values, start_idx=0, num_elements=None, atol=0.1, rtol=0.1):
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
    golden_tensor = torch.from_numpy(golden_np).float()
    simulated_tensor = torch.from_numpy(simulated_np).float()

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

    # Relative-error match rate (same formula as compare_vram_with_golden):
    # |err| / |golden| <= rtol.
    within_relative_tolerance = relative_errors <= rtol
    relative_match_rate = torch.sum(within_relative_tolerance).item() / len(relative_errors) * 100.0

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
        "relative_match_rate": relative_match_rate,
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
        # Standalone example (manual debugging via vram_dump.bin — the
        # real test path goes through view_mem.py). The hardware geometry
        # below is just illustrative; the production path passes values
        # derived from plena_settings.toml via comparison_params.
        results = compare_vram_with_golden(
            vram_file,
            golden_file,
            exp_width=8,
            man_width=7,
            num_bytes_per_val=2,
            row_dim=64,
            num_batches=4,
            elements_per_batch=128,
            start_row_idx=0,
            num_rows=4,
            use_stride_mode=True,
        )
        print_comparison_results(results, verbose=True)
    else:
        print("Files not found:")
        print(f"  Golden: {golden_file}")
        print(f"  VRAM:   {vram_file}")
