"""PLENA backend implementations for positional encoding operators."""


def embedding_add_plena(prog, input_var, pos_weight_var):
    """PLENA backend: add learned position embeddings to input in-place.

    Both input_var and pos_weight_var must be VRAMMatrixVar with the same shape.
    Uses prog.vram_add() which emits V_ADD_VV row-by-row.
    """
    prog.vram_add(input_var, pos_weight_var)
    return input_var


def rope_plena(prog, x_var, x_rot_var, cos_var, sin_var):
    """PLENA backend: apply RoPE in-place.

    x_var is updated in-place: x = x * cos + rotate_half(x) * sin

    x_rot_var must be a VRAMMatrixVar holding rotate_half(x), preloaded from HBM
    by the caller before dispatching this op.
    """
    prog.rope(x_var, x_rot_var, cos_var, sin_var)
    return x_var
