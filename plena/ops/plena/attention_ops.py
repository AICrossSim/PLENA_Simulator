"""PLENA backend implementation for Flash Attention operator."""


def flash_attention_plena(prog, Q, K, V, scale=None):
    """PLENA backend: Flash Attention via PLENAProgram.

    Implements the Online Softmax flash attention algorithm:
        for q_block in Q_blocks:
            init m_old=-inf, l_old=0, O_row=0
            for k_block in K_blocks:
                S = Q[q_block] @ K[k_block].T * scale
                Online Softmax: update m, l, P
                PV = P @ V[k_block]
                O[q_block] = O[q_block] * m_res + PV
            O[q_block] /= l_old

    Args:
        prog:   PLENAProgram instance
        Q:      VRAMMatrixVar — Q matrix loaded in VRAM, shape (seq_len, head_dim)
        K:      InputVar — K matrix in HBM, shape (seq_len, head_dim)
        V:      InputVar — V matrix in HBM, shape (seq_len, head_dim)
        scale:  Attention scale factor (default: 1/sqrt(head_dim))

    Returns:
        VRAMMatrixVar for the output O, shape (seq_len, head_dim).
    """
    import math

    seq_len, head_dim = Q.shape
    mlen = prog.mlen

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    num_q_blocks = seq_len // mlen
    num_k_blocks = seq_len // mlen

    # Allocate working buffers
    S_block = prog.alloc("S", mlen, mlen)           # Q[q] @ K[k].T result
    PV = prog.alloc("PV", mlen, head_dim)            # P @ V partial result
    O = prog.alloc("O", seq_len, head_dim)           # final output

    # Flash Attention main loop
    for q_idx in range(num_q_blocks):
        # Initialize online softmax state: m=-inf, l=0, O_row=0
        prog.init_online_softmax(q_idx, O)

        for k_idx in range(num_k_blocks):
            # S = Q[q_idx] @ K[k_idx].T
            # auto_reset_mram=True handles reset+load internally
            prog.vram_sub_projection_T_to(
                Q,
                q_idx,
                K,
                k_idx,
                S_block,
                target_row_idx=0,
                target_col_idx=0,
            )

            # Online softmax: scale S, update running max m and sum l
            prog.online_softmax_block(S_block, scale)

            # PV = P @ V[k_idx]
            prog.compute_pv(S_block, V, k_idx, PV, head_dim)

            # O[q_idx] = O[q_idx] * m_res  (online softmax correction)
            prog.scale_o_row(O, q_idx)

            # O[q_idx] += PV
            prog.vram_add(O, PV, dst_row_offset=q_idx * mlen)

        # Final normalization: O[q_idx] /= l
        prog.final_scale_o(q_idx, O)

    return O
