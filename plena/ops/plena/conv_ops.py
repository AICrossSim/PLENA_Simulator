"""PLENA backend for conv2d: TRUE on-chip im2col (documented ISA only) + systolic matmul.

No V_SHFT_V used.  im2col is performed entirely on-chip using only formally
documented instructions:
  H_PREFETCH_V  — fetch one patch row from HBM → VRAM scratch
  V_MUL_VV      — isolate one element via a basis vector
  V_RED_SUM     — reduce to scalar (V_RED_SUM accumulates; zeroed with S_ADD_FP first)
  S_ST_FP       — write extracted element to FP_SRAM at its im2col column position
  S_MAP_V_FP    — flush FP_SRAM[0..K_col-1] into the output VRAM row

After im2col the systolic matmul uses the standard linear_plena path.

HBM layout convention (caller must arrange data accordingly):
  input_raw shape  = (C_in * H, W_padded)   — each row is one spatial row of one channel
  weight_2d shape  = (K_col, C_out)          — standard im2col weight layout

Alignment requirement:
  H_PREFETCH_V requires the HBM element address to be a multiple of 64.
  With W_padded=64 and ow=0 (OW=1): offset = (c*H + oh+kr) * 64 → always aligned.
"""

from plena.ops.plena.linear_ops import linear_plena


_PREFETCH_V_AMOUNT = 4   # H_PREFETCH_V always loads this many VRAM rows


def conv2d_plena(
    prog,
    input_raw_var,
    weight_2d_var,
    C_in: int,
    H: int,
    W: int,
    K: int,
    OH: int,
    OW: int,
    M: int,
    W_padded: int = None,
    fp_one_reg: int = 1,
):
    """
    PLENA backend: hardware im2col (no V_SHFT_V) + systolic matmul.

    Args:
        prog:           PLENAProgram instance.
        input_raw_var:  InputVar, raw NCHW input in HBM,
                        shape = (C_in*H, W_padded).
                        Element [c, h, w] is at HBM offset (c*H+h)*W_padded+w.
        weight_2d_var:  InputVar, reshaped weight in HBM,
                        shape = (K_col, C_out)  where K_col = C_in*K*K.
        C_in:   Number of input channels.
        H:      Input spatial height.
        W:      Input spatial width (real, not padded).
        K:      Kernel size (K×K square kernel).
        OH:     Output height  = (H - K) // stride + 1.
        OW:     Output width   = (W - K) // stride + 1.
        M:      Total output positions = OH * OW  (for batch=1).
        W_padded:
            Padded HBM row width for 64-element alignment.
            Must satisfy W_padded % 64 == 0 and W_padded >= W.
            Defaults to next multiple of 64 >= W.

    Returns:
        VRAMMatrixVar for the output, shape (M, C_out).
    """
    # Lazy imports to avoid circular dependencies at module load time
    from compiler.asm_templates.im2col_asm_no_shift import (
        im2col_asm_no_shift,
        PREFETCH_V_AMOUNT,
    )

    vlen  = prog.mlen
    K_col = C_in * K * K
    # Pad K_col to next multiple of vlen so column-block-major tiles don't overflow VRAM
    K_col_padded = ((K_col + vlen - 1) // vlen) * vlen

    if W_padded is None:
        W_padded = ((W + 63) // 64) * 64   # next multiple of 64

    assert W_padded % 64 == 0, f"W_padded={W_padded} must be a multiple of 64"

    # ------------------------------------------------------------------
    # Allocate VRAM regions
    # ------------------------------------------------------------------
    # K basis vectors  (e_kc has 1.0 at position kc, zeros elsewhere)
    basis_mat   = prog.alloc("im2col_basis",   K,                   vlen, strict=False)
    # Scratch area for H_PREFETCH_V landing (needs PREFETCH_V_AMOUNT rows)
    scratch_mat = prog.alloc("im2col_scratch",  PREFETCH_V_AMOUNT,  vlen, strict=False)
    # Temp row for V_MUL_VV result
    temp_mat    = prog.alloc("im2col_temp",     1,                  vlen, strict=False)
    # Output im2col matrix: M rows × K_col_padded cols (padded for tile alignment)
    output_mat  = prog.alloc("im2col_out",      M,                  K_col_padded, strict=False)

    # ------------------------------------------------------------------
    # Look up VRAM base addresses from the symbol table
    # ------------------------------------------------------------------
    basis_vram_base   = prog._compiler.get_vram_addr(basis_mat.name)
    scratch_vram_addr = prog._compiler.get_vram_addr(scratch_mat.name)
    temp_vram_addr    = prog._compiler.get_vram_addr(temp_mat.name)
    output_vram_base  = prog._compiler.get_vram_addr(output_mat.name)

    # ------------------------------------------------------------------
    # GP register allocation
    # alive_registers: [scratch_reg, temp_reg, off_reg, out_reg, basis_reg]
    # setup_gp: used once to load HBM base into the 'a' address register
    # ------------------------------------------------------------------
    alive_registers = [1, 2, 3, 4, 5]
    setup_gp        = 6
    addr_reg_idx    = 0   # use a0 for input HBM base

    # ------------------------------------------------------------------
    # Emit: set address register a0 = input_raw HBM base address
    # ------------------------------------------------------------------
    hbm_base = input_raw_var.hbm_addr
    setup_lines = []
    if hbm_base <= 262143:
        setup_lines.append(f"S_ADDI_INT gp{setup_gp}, gp0, {hbm_base}")
    else:
        setup_lines.append(f"S_LUI_INT gp{setup_gp}, {hbm_base >> 12}")
        setup_lines.append(
            f"S_ADDI_INT gp{setup_gp}, gp{setup_gp}, {hbm_base & 0xFFF}"
        )
    setup_lines.append(f"C_SET_ADDR_REG a{addr_reg_idx}, gp0, gp{setup_gp}")
    prog._compiler.generated_code += "\n".join(setup_lines) + "\n"

    # ------------------------------------------------------------------
    # Emit: im2col assembly (no V_SHFT_V)
    # ------------------------------------------------------------------
    asm_code = im2col_asm_no_shift(
        mlen=vlen,
        vlen=vlen,
        C_in=C_in,
        H=H,
        W=W,
        K=K,
        OH=OH,
        OW=OW,
        M=M,
        alive_registers=alive_registers,
        input_hbm_base_addr_reg=addr_reg_idx,
        basis_vram_base=basis_vram_base,
        scratch_vram_addr=scratch_vram_addr,
        temp_vram_addr=temp_vram_addr,
        output_vram_base=output_vram_base,
        W_padded=W_padded,
        fp_one_reg=fp_one_reg,   # f1 = 1.0 by default (must be in fp_preload[fp_one_reg])
        fp_ex_reg=2,    # f2 = V_RED_SUM accumulator
    )
    prog._compiler.generated_code += asm_code

    # ------------------------------------------------------------------
    # Systolic matmul: im2col_out @ weight_2d  -> (M, C_out)
    # ------------------------------------------------------------------
    return linear_plena(prog, output_mat, weight_2d_var)
