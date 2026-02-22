"""PLENA backend for conv2d via im2col + linear projection."""

from plena.ops.plena.linear_ops import linear_plena


def conv2d_plena(prog, input_col_var, weight_2d_var):
    """PLENA backend: conv2d as im2col + linear projection.

    The caller pre-transforms the tensors before declaring them in the program:
      - input_col_var: BatchVar, shape (B*OH*OW, C_in*K*K)  [im2col result, in VRAM]
      - weight_2d_var: InputVar, shape (C_in*K*K, C_out)    [reshaped weight, in HBM]

    This reduces conv2d to a single linear projection handled natively by PLENA.

    Args:
        prog:           PLENAProgram instance
        input_col_var:  BatchVar — im2col-transformed input loaded in VRAM
        weight_2d_var:  InputVar — reshaped weight in HBM

    Returns:
        VRAMMatrixVar for the output, shape (B*OH*OW, C_out).
    """
    return linear_plena(prog, input_col_var, weight_2d_var)
