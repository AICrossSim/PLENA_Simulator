"""PLENA backend stubs for linear projection operators."""

def linear_plena(prog, input_var, weight_var):
    """PLENA backend: linear projection via PLENAProgram sub-matrix operations.

    Supports M > mlen via row-block iteration and K_col > 4*mlen via K-split
    partial sums accumulated in VRAM.

    MRAM capacity: 4 tiles × mlen² = 4 × 4096 = 16384 elements (MAX_K_TILES=4).
    When K tiles > MAX_K_TILES, we split into chunks and accumulate partial sums.

      output[r][c] = sum_k input[r][k] @ weight[k][c]
    for r in 0..num_row_blocks-1, c in 0..num_col_blocks-1,
    k split into chunks of at most MAX_K_TILES tiles.
    """
    import math
    mlen = prog.mlen
    MAX_K_TILES = 4  # MRAM capacity: 4 × mlen² elements

    rows, k_total = input_var.shape
    _, out_features = weight_var.shape
    num_row_blocks = math.ceil(rows / mlen)
    num_col_blocks = out_features // mlen
    num_k_tiles = math.ceil(k_total / mlen)

    output = prog.alloc("linear_out", rows, out_features)

    if num_k_tiles <= MAX_K_TILES:
        # Single pass: all K tiles fit in MRAM
        for col_idx in range(num_col_blocks):
            for row_idx in range(num_row_blocks):
                prog.vram_sub_projection_to(
                    input_var, row_idx, weight_var, col_idx, output, row_idx, col_idx,
                )
    else:
        # K-split: chunk K tiles into groups of MAX_K_TILES, accumulate partial sums
        k_chunks = []
        k_start = 0
        while k_start < num_k_tiles:
            k_end = min(k_start + MAX_K_TILES, num_k_tiles)
            k_chunks.append((k_start, k_end - k_start))
            k_start = k_end

        # Temp buffer for partial sums (same shape as output)
        temp = prog.alloc("linear_out_temp", rows, out_features)

        for k_chunk_idx, (k_block_start, k_block_count) in enumerate(k_chunks):
            for col_idx in range(num_col_blocks):
                for row_idx in range(num_row_blocks):
                    if k_chunk_idx == 0:
                        # First chunk: write directly to output
                        prog.vram_sub_projection_to(
                            input_var, row_idx, weight_var, col_idx,
                            output, row_idx, col_idx,
                            k_block_start=k_block_start, k_block_count=k_block_count,
                        )
                    else:
                        # Subsequent chunks: write to temp, then accumulate into output
                        prog.vram_sub_projection_to(
                            input_var, row_idx, weight_var, col_idx,
                            temp, row_idx, col_idx,
                            k_block_start=k_block_start, k_block_count=k_block_count,
                        )
                        prog.vram_block_add_to(
                            output, row_idx, col_idx,
                            temp, row_idx, col_idx,
                            output, row_idx, col_idx,
                        )

    return output
