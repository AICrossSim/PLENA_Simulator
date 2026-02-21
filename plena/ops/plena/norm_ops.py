"""PLENA backend implementations for normalization operators."""


def rms_norm_plena(prog, input_var, eps: float = 1e-6):
    """PLENA backend: RMS normalization via PLENAProgram."""
    return prog.rms_norm(input_var, eps_offset=1, reci_hid_offset=2)


def layer_norm_plena(prog, input_var, eps: float = 1e-6):
    """PLENA backend: Layer normalization via PLENAProgram."""
    return prog.layer_norm(input_var, eps_offset=1, reci_hid_offset=2)
