"""PLENA backend stubs for linear projection operators."""

def linear_plena(prog, input_var, weight_var):
    """PLENA backend: linear projection via PLENAProgram sub-matrix operations."""
    mlen = prog.mlen
    _, out_features = weight_var.shape
    num_col_blocks = out_features // mlen
    output = prog.alloc("linear_out", input_var.shape[0], out_features)
    for col_idx in range(num_col_blocks):
        prog.vram_sub_projection_to(input_var, 0, weight_var, col_idx, output, 0, col_idx)
    return output
