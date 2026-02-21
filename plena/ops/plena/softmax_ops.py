"""
PLENA backend implementation for softmax operator.

This encapsulates the online softmax algorithm that was previously
written inline in fpvar_softmax_test.py (Tilelang-style).

ATen-style: the function receives a PLENAProgram context and TensorVar
arguments, orchestrates PLENAProgram calls, and returns the result TensorVar.
"""

from __future__ import annotations


def softmax_plena(prog, input_var, scale: float = 1.0):
    """
    PLENA backend: online softmax (numerically stable, row-wise).

    Algorithm (matches fpvar_softmax_test.py):
        Initialize: m_old[row] = -inf, l_old[row] = 0
        for row in range(mlen):
            m_old_saved = m_old[row]
            S[row] *= scale
            row_max = max(S[row])
            m_old[row] = max(m_old[row], row_max)  # m_curr
            m_res[row] = exp(m_old_saved - m_curr)
            S[row] -= m_curr
            P[row] = exp(S[row])
            sum_p = sum(P[row])
            l_old[row] = l_old[row] * m_res[row] + sum_p
        P[row] /= l_old[row]  # final normalization

    Args:
        prog:      PLENAProgram instance (compilation context)
        input_var: BatchVar or VRAMMatrixVar — the input matrix in VRAM
                   Shape: (mlen, mlen)
        scale:     Multiplicative scale applied before softmax (default 1.0)

    Returns:
        The same input_var (in-place modification) after softmax is applied.
        The result is stored in S (VRAM), which is also returned.

    Note:
        fp_preload layout expected: [0]=0.0, [1]=scale, [2]=-inf
        The caller must set fp_preload = [0.0, scale, float('-inf'), ...]
    """
    mlen = prog.mlen

    # Allocate output matrix in VRAM (S holds the softmax result)
    S = prog.alloc("S", mlen, mlen)

    # Reserve fp_preload region (addresses 0-2): 0=0.0, 1=scale, 2=-inf
    # This ensures FPRAM variable allocation starts after these reserved slots
    prog._compiler.sub_matrix_manager.fpram_allocator.next_free = 3

    # Allocate FPRAM variables
    scale_fp    = prog.fp_var("scale_fp",    size=1)     # scale factor
    m_old       = prog.fp_var("m_old",       size=mlen)  # per-row running max
    m_res       = prog.fp_var("m_res",       size=mlen)  # per-row decay factor
    l_old       = prog.fp_var("l_old",       size=mlen)  # per-row accumulated sum
    row_max_tmp = prog.fp_var("row_max_tmp", size=1)     # temp: current row max
    m_old_saved = prog.fp_var("m_old_saved", size=1)     # temp: saved m_old
    sum_p_tmp   = prog.fp_var("sum_p_tmp",   size=1)     # temp: current row sum
    inv_l       = prog.fp_var("inv_l",       size=mlen)  # 1/l for normalization

    # Step 0: Initialize S = input, load constants from fp_preload
    prog.vram_add(S, input_var)
    prog.fpvar_fill_from_fpram(scale_fp, src_fpram_addr=1)   # load scale
    prog.fpvar_fill_from_fpram(m_old,    src_fpram_addr=2)   # load -inf
    prog.fpvar_fill_from_fpram(l_old,    src_fpram_addr=0)   # load 0.0

    # Row-by-row online softmax
    compiler = prog._compiler
    for row in range(mlen):
        # 1. Save old max for this row
        compiler.fpvar_copy_asm(m_old.address + row, m_old_saved.address, 1)

        # 2. Scale: S[row] *= scale
        prog.tile_row_mul_fp_broadcast(S, scale_fp.address, row)

        # 3. Find row maximum
        prog.tile_row_max(row_max_tmp.address, S, row)

        # 4. Update running max: m_old[row] = max(m_old[row], row_max)
        compiler.fpvar_max_asm(
            m_old.address + row, row_max_tmp.address, m_old.address + row, 1
        )

        # 5. Decay factor: m_res[row] = exp(m_old_saved - m_curr)
        compiler.fpvar_sub_asm(
            m_old_saved.address, m_old.address + row, m_old_saved.address, 1
        )
        compiler.fpvar_exp_asm(m_old_saved.address, m_res.address + row, 1)

        # 6. Subtract max: S[row] -= m_curr
        prog.tile_row_sub_fp(S, m_old.address + row, row)

        # 7. Exponentiate: P[row] = exp(S[row])
        prog.tile_row_exp(S, row)

        # 8. Sum probabilities
        prog.tile_row_sum(sum_p_tmp.address, S, row)

        # 9. Update accumulated sum: l_old[row] = l_old[row]*m_res[row] + sum_p
        compiler.fpvar_mul_asm(
            l_old.address + row, m_res.address + row, l_old.address + row, 1
        )
        compiler.fpvar_add_asm(
            l_old.address + row, sum_p_tmp.address, l_old.address + row, 1
        )

    # Final normalization: P[row] /= l_old[row]
    prog.fpvar_reci(l_old, inv_l)
    for row in range(mlen):
        prog.tile_row_mul_fp(S, inv_l.address + row, row)

    return S
