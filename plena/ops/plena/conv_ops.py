"""PLENA backend for conv2d via on-chip im2col + linear projection."""

import sys

from plena.ops.plena.linear_ops import linear_plena


def conv2d_plena(prog, input_raw_var, mask_vec_var, weight_2d_var, conv_params):
    """
    PLENA backend: conv2d via TRUE on-chip im2col + linear projection.

    The host passes the raw NCHW input tensor; all im2col work is done on PLENA
    using H_PREFETCH_V + V_MUL_VV (masking) + V_SHFT_V (shift) + V_ADD_VV
    (accumulate) before handing the assembled rows to the systolic matmul.

    Args:
        prog:            PLENAProgram instance.
        input_raw_var:   InputVar, raw NCHW input in HBM,
                         shape = (C_in, H*W).
        mask_vec_var:    InputVar, mask vector [1..1, 0..0] in HBM,
                         shape = (1, vlen).  First K elements are 1.0, rest 0.0.
        weight_2d_var:   InputVar, reshaped weight in HBM,
                         shape = (K_col, C_out).
        conv_params:     dict with keys:
                           C_in, H, W, K  (kernel size),
                           OH, OW, M (=B*OH*OW), K_col (=C_in*K*K),
                           vlen (optional, defaults to prog.mlen).

    Returns:
        VRAMMatrixVar for the output, shape (M, C_out).
    """
    # ── BatchVar: fetched lazily from sys.modules so we don't need to add
    #    plena_program to sys.path inside this package file.
    plena_program_mod = sys.modules.get("plena_program")
    if plena_program_mod is None:
        raise ImportError(
            "plena_program module must be imported before calling conv2d_plena "
            "(add its directory to sys.path and import PLENAProgram first)."
        )
    BatchVar = plena_program_mod.BatchVar

    # ── im2col ASM template ──────────────────────────────────────────────
    from compiler.asm_templates.im2col_asm import im2col_asm

    C_in    = conv_params["C_in"]
    H       = conv_params["H"]
    W       = conv_params["W"]
    K       = conv_params["K"]
    OH      = conv_params["OH"]
    OW      = conv_params["OW"]
    M       = conv_params["M"]
    K_col   = conv_params["K_col"]
    vlen    = conv_params.get("vlen", prog.mlen)
    W_padded = conv_params.get("W_padded", W)

    # ── 1. Load mask vector into VRAM ────────────────────────────────────
    mask_batch = prog.load_batch(mask_vec_var, name="mask_vec")

    # ── 2. Allocate scratch VRAM (PREFETCH_V_AMOUNT=4 rows) ──────────────
    scratch = prog.alloc("im2col_scratch", 4, vlen)

    # ── 3. Allocate im2col output VRAM (M rows × K_col cols) ─────────────
    im2col_out = prog.alloc("im2col_out", M, K_col)

    # ── 4. Look up VRAM addresses ────────────────────────────────────────
    mask_vram_addr    = prog._compiler.symbol_table[mask_batch.name].vram_addr
    scratch_vram_addr = prog._compiler.symbol_table[scratch.name].vram_addr
    output_vram_base  = prog._compiler.symbol_table[im2col_out.name].vram_addr

    # ── 5. Preamble: point a0 at input_raw HBM base ──────────────────────
    hbm_base = input_raw_var.hbm_addr
    preamble = (
        f"; === im2col preamble: set a0 -> input_raw HBM base {hbm_base} ===\n"
        f"S_ADDI_INT gp1, gp0, {hbm_base}\n"
        f"C_SET_ADDR_REG a0, gp0, gp1\n"
    )

    # ── 6. Generate and inject im2col ASM ────────────────────────────────
    asm_code = im2col_asm(
        mlen=prog.mlen,
        vlen=vlen,
        C_in=C_in,
        H=H,
        W=W,
        K=K,
        OH=OH,
        OW=OW,
        M=M,
        alive_registers=[1, 2, 3, 4, 5, 6],
        input_hbm_base_addr_reg=0,
        mask_vec_vram_addr=mask_vram_addr,
        scratch_vram_addr=scratch_vram_addr,
        output_vram_base=output_vram_base,
        W_padded=W_padded,
    )
    prog._compiler.generated_code += preamble + asm_code

    # ── 7. Wrap im2col_out as a BatchVar so linear_plena can use it ──────
    #    im2col_out is already registered in the symbol table under its
    #    internal_name; constructing a BatchVar with the same name reuses
    #    that allocation without re-allocating VRAM.
    synthetic_batch = BatchVar(prog, im2col_out.internal_name, (M, K_col))
    prog._tensors[im2col_out.internal_name] = synthetic_batch

    # ── 8. Linear projection: im2col rows @ weight_2d ────────────────────
    return linear_plena(prog, synthetic_batch, weight_2d_var)
