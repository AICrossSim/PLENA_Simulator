"""PLENA backend stubs for linear projection operators."""

def linear_plena(prog, input_var, weight_var):
    """PLENA backend: linear projection via PLENAProgram sub-matrix operations."""
    # Delegate to PLENAProgram's sub-matrix projection
    w_sub = prog.register_sub_matrix(weight_var)
    mlen = prog.mlen
    _, in_features = input_var.shape
    _, out_features = weight_var.shape
    num_col_blocks = out_features // mlen
    output = prog.alloc("linear_out", input_var.shape[0], out_features)
    input_vsub = prog.register_vram_sub_matrix(input_var)
    for col_idx in range(num_col_blocks):
        w_sub.load_col(col_idx)
        prog.vram_sub_projection_to(input_vsub.row(0), w_sub.col(col_idx), output, 0, col_idx)
        prog.reset_mram()
    return output
