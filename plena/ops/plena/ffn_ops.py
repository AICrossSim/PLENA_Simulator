"""PLENA backend implementation for FFN operator."""

from compiler.asm_templates import ffn_asm, preload_addr_reg_asm, reset_reg_asm


def ffn_plena(prog, input_var, w_gate, w_up, w_down):
    """PLENA backend: FFN with SiLU gate via ffn_asm.

    Generates ISA code for: w_down @ (silu(w_gate @ x) * (w_up @ x))

    Args:
        prog:       PLENAProgram instance
        input_var:  BatchVar — activation in VRAM, shape (batch, hidden)
        w_gate:     InputVar — gate weight in HBM, shape (hidden, inter_dim)
        w_up:       InputVar — up-projection weight in HBM, shape (hidden, inter_dim)
        w_down:     InputVar — down-projection weight in HBM, shape (inter_dim, hidden)

    Returns:
        VRAMMatrixVar for the FFN output (stored at activation_base_address in VRAM).
    """
    batch_size, hidden_size = input_var.shape
    _, inter_dim = w_up.shape
    mlen = prog.mlen
    blen = prog.blen
    vlen = prog.mlen

    # Retrieve VRAM address of the loaded activation
    activation_base_address = prog._compiler.get_vram_addr(input_var.name)

    # Set HBM address registers for each weight matrix
    isa_code = preload_addr_reg_asm(
        addr_reg_to_set=[1, 2, 3],
        available_registers=[1, 2, 3],
        addr_reg_val=[w_gate.hbm_addr, w_up.hbm_addr, w_down.hbm_addr],
    )

    # Reset registers before FFN kernel
    isa_code += reset_reg_asm(alive_registers=[1, 2, 3])

    # Generate FFN ISA kernel
    isa_code += ffn_asm(
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        batch=batch_size,
        seq_len=1,
        hidden_size=hidden_size,
        intermediate_size=inter_dim,
        alive_registers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        gate_weight_hbm_offset_reg=1,
        up_weight_hbm_offset_reg=2,
        down_weight_hbm_offset_reg=3,
        const_one_fp_address=5,
        activation_base_address=activation_base_address,
        use_loop_instructions=True,
    )

    prog._compiler.generated_code += isa_code

    # FFN result is written back to the activation area in VRAM (in-place overwrite)
    return input_var
