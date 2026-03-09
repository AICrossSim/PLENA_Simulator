"""PLENA backend implementations for normalization operators."""


def rms_norm_plena(prog, input_var, eps: float = 1e-6,
                   eps_offset: int = 1, reci_hid_offset: int = 2):
    """PLENA backend: RMS normalization via PLENAProgram.

    Args:
        prog:            PLENAProgram instance
        input_var:       BatchVar/VRAMMatrixVar — activation in VRAM
        eps:             epsilon value (unused by hardware, kept for API compat)
        eps_offset:      FPRAM slot index for eps value (default 1)
        reci_hid_offset: FPRAM slot index for 1/hidden value (default 2)
    """
    return prog.rms_norm(input_var, eps_offset=eps_offset, reci_hid_offset=reci_hid_offset)


def layer_norm_plena(prog, input_var, eps: float = 1e-6,
                     eps_offset: int = 1, reci_hid_offset: int = 2):
    """PLENA backend: Layer normalization via PLENAProgram.

    Args:
        prog:            PLENAProgram instance
        input_var:       BatchVar/VRAMMatrixVar — activation in VRAM
        eps:             epsilon value (unused by hardware, kept for API compat)
        eps_offset:      FPRAM slot index for eps value (default 1)
        reci_hid_offset: FPRAM slot index for 1/hidden value (default 2)
    """
    return prog.layer_norm(input_var, eps_offset=eps_offset, reci_hid_offset=reci_hid_offset)
