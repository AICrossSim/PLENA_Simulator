"""PLENA backend for conv2d: CPU im2col + PLENA systolic matmul."""

from plena.ops.plena.linear_ops import linear_plena


def conv2d_plena(prog, input_col_var, weight_2d_var):
    """
    PLENA backend: im2col is computed on CPU; PLENA runs the systolic matmul.

    The im2col transformation (patch extraction) is pre-computed on the host
    and placed in HBM as a 2-D matrix.  PLENA then executes the matmul using
    H_PREFETCH_V / H_PREFETCH_M and the systolic array — all documented ISA
    instructions.

    Args:
        prog:           PLENAProgram instance.
        input_col_var:  InputVar, im2col-transformed input in HBM,
                        shape = (B*OH*OW, C_in*K*K).
        weight_2d_var:  InputVar, reshaped weight in HBM,
                        shape = (C_in*K*K, C_out).

    Returns:
        VRAMMatrixVar for the output, shape (B*OH*OW, C_out).
    """
    # Load im2col matrix from HBM into VRAM (same as linear_aten_test does for X)
    input_col_batch = prog.load_batch(input_col_var, name="input_col")
    return linear_plena(prog, input_col_batch, weight_2d_var)
