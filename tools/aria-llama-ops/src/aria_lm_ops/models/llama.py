import math

import torch
from ..utils import check_shape
from ..utils.int_arith import ceil_div


@torch.no_grad()
def rms_norm(x: torch.Tensor, w: torch.Tensor, s: int, h: int, var_eps: float):
    """
    Args:
        x: (b, s, h), where b = batch size, s = sequence length, h = hidden size
        w: (h,)
        s: sequence length
        h: hidden size
        var_eps: epsilon for variance

    Returns:
        out: (b, s, h)
    """
    b = x.size(0)
    square_bsh = x.pow(2)  # pow = x * x
    sum_bs = square_bsh.sum(dim=2)  # sum over hidden_dim
    var_bs = sum_bs / h  # average over hidden_dim
    sqrt_bs = torch.sqrt(var_bs + var_eps)  # add epsilon to avoid division by zero
    rsqrt_bs = 1.0 / sqrt_bs  # reciprocal of sqrt
    rsqrt_bs1 = rsqrt_bs.unsqueeze(2)  # broadcast to hidden_dim
    x_norm_bsh = x * rsqrt_bs1  # broadcast multiplication
    out_bsh = w * x_norm_bsh  # elementwise scale by weight

    check_shape(x, (b, s, h))
    check_shape(w, (h,))
    check_shape(square_bsh, (b, s, h))
    check_shape(sum_bs, (b, s))
    check_shape(var_bs, (b, s))
    check_shape(sqrt_bs, (b, s))
    check_shape(rsqrt_bs, (b, s))
    check_shape(rsqrt_bs1, (b, s, 1))
    check_shape(x_norm_bsh, (b, s, h))
    check_shape(out_bsh, (b, s, h))
    return out_bsh


@torch.no_grad()
def flash_attn2_head_gemv(q, k, v, qk_scale, s_q, s_kv, h_qkv, Bc, Tc, Br, Tr, debug=False, return_intermediates=False):
    """
    Args:
        q: [b, s_q, h_qkv], query tensor
        k: [b, s_kv, h_qkv], key tensor
        v: [b, s_kv, h_qkv], value tensor
        qk_scale: float, scale factor for qk, usually 1 / sqrt(h_qkv)
        s_q: int, query sequence length, should be 1 for vector-matrix hardware
        s_kv: int, key-value sequence length, increases with generation step (KV cache)
        h_qkv: int, hidden size per head
        Bc: int, tile size for key-value matrix
        Tc: int, temporal tile count
        debug: bool, print debug info
        return_intermediates: bool, return intermediate values for comparison

    Returns:
        o: [b, s_q, h_qkv], attention output
        intermediates (optional): dict with intermediate values per (batch, q_tile, kv_tile)
    """
    b = q.size(0)
    o = torch.zeros(b, s_q, h_qkv, device=q.device)  # output

    # Storage for intermediate values if requested
    intermediates = {} if return_intermediates else None

    for b_i in range(b):
        for i in range(Tr):
            q_i = q[b_i, i * Br : (i + 1) * Br, :].squeeze(0)  # [Br, h_qkv]
            m = torch.full((Br,), float("-inf"), device=q.device)
            l = torch.zeros((Br,), device=q.device)
            o_i = torch.zeros((Br, h_qkv), device=q.device)
            for j in range(Tc):
                k_j = k[b_i, j * Bc : (j + 1) * Bc, :].squeeze(0)  # [Bc, h_qkv]
                v_j = v[b_i, j * Bc : (j + 1) * Bc, :].squeeze(0)  # [Bc, h_qkv]

                # Step 1: QKT multiplication (before scaling)
                s_j_unscaled = q_i @ k_j.transpose(0, 1)  # Q @ Kj^T, [Br, Bc]
                s_j = s_j_unscaled * qk_scale  # scaled QKT

                # Step 2: Online softmax - find row max
                rowmax_s_j = s_j.max(dim=1).values  # [Br]
                m_old = m.clone()  # save old m for intermediate output
                m_new = torch.maximum(m, rowmax_s_j)  # [Br]

                # Step 3: Shift and exp
                s_j_shifted = s_j - m_new.unsqueeze(1)
                p = torch.exp(s_j_shifted)  # exp(Sj - m_new), shape: [Br, Bc]
                p = p.to(torch.bfloat16)

                # Step 4: Compute m_res and exp(m_res)
                m_res = m - m_new  # [Br]
                exp_m_res = torch.exp(m_res)  # [Br]
                m = m_new

                # Step 5: Update l
                l_old = l.clone()
                p_row_sum = p.sum(dim=1)
                l = exp_m_res * l + p_row_sum  # [Br]

                # Step 6: Compute PV = P @ V
                pv = torch.matmul(p, v_j)  # [Br, h_qkv]

                # Step 7: Update O_i = diag(exp(m_res)) @ O_old + PV
                o_scale_diag = torch.diag(exp_m_res)  # [Br, Br]
                o_old = o_i.clone()
                o_i = torch.matmul(o_scale_diag, o_i) + pv  # [Br, h_qkv]

                # Store intermediate values
                if return_intermediates:
                    key = (b_i, i, j)
                    intermediates[key] = {
                        # Inputs
                        "q_i": q_i.clone(),                    # [Br, h_qkv]
                        "k_j": k_j.clone(),                    # [Bc, h_qkv]
                        "v_j": v_j.clone(),                    # [Bc, h_qkv]
                        # QKT stage
                        "s_j_unscaled": s_j_unscaled.clone(),  # [Br, Bc] - QKT before scaling
                        "s_j": s_j.clone(),                    # [Br, Bc] - QKT after scaling
                        # Online softmax stage
                        "rowmax_s_j": rowmax_s_j.clone(),      # [Br]
                        "m_old": m_old.clone(),                # [Br] - m before update
                        "m_new": m_new.clone(),                # [Br] - m after update
                        "m_res": m_res.clone(),                # [Br] - m_old - m_new
                        "exp_m_res": exp_m_res.clone(),        # [Br] - exp(m_res)
                        "s_j_shifted": s_j_shifted.clone(),    # [Br, Bc] - S - m_new
                        "p": p.clone(),                        # [Br, Bc] - softmax scores (P)
                        "p_row_sum": p_row_sum.clone(),        # [Br] - sum of P per row
                        "l_old": l_old.clone(),                # [Br] - l before update
                        "l_new": l.clone(),                    # [Br] - l after update
                        # PV stage
                        "pv": pv.clone(),                      # [Br, h_qkv] - P @ V
                        # Output accumulation stage
                        "o_old": o_old.clone(),                # [Br, h_qkv] - O before update
                        "o_scaled": torch.matmul(o_scale_diag, o_old).clone(),  # [Br, h_qkv] - diag(exp(m_res)) @ O_old
                        "o_i": o_i.clone(),                    # [Br, h_qkv] - O after update (before final 1/l scaling)
                    }

                if debug and b_i == 0 and i == 0 and j == 0:
                    print(f"b_i={b_i}, i={i}, j={j}")
                    print("s_j_unscaled", s_j_unscaled)
                    print("s_j (scaled)", s_j)
                    print("rowmax_s_j shape", rowmax_s_j.shape)
                    print("m_new shape", m_new.shape)
                    print("m_new", m_new)
                    print("s_j_shifted", s_j_shifted)
                    print("s_j_shifted shape", s_j_shifted.shape)
                    print("p shape", p.shape)
                    print("m_res shape", m_res.shape)
                    print("m_res", m_res)
                    print("exp_m_res", exp_m_res)
                    print("p shape", p.shape)
                    print("p: ", p)
                    print("p_row_sum shape", p_row_sum.shape)
                    print("p_row_sum: ", p_row_sum)
                    print("l_old", l_old)
                    print("l_new", l)
                    print("v_j shape", v_j.shape)
                    print("v_j", v_j)
                    print("pv shape", pv.shape)
                    print("pv value ", pv)
                    print("o_old", o_old)
                    print("o_i (accumulated)", o_i)

            # Final scaling by 1/l (row-wise: O[r][k] = O_acc[r][k] * (1/l[r]))
            inv_l = 1.0 / l  # [Br]
            o_final = o_i * inv_l.unsqueeze(1)  # [Br, h_qkv] * [Br, 1] -> [Br, h_qkv]
            print("o_final", o_final)
            o[b_i, i * Br : (i + 1) * Br, :] = o_final

            # Store final output info
            if return_intermediates:
                final_key = (b_i, i, "final")
                intermediates[final_key] = {
                    "l_final": l.clone(),                  # [Br] - final l value
                    "inv_l": (1.0 / l).clone(),           # [Br] - 1/l for scaling
                    "o_before_scaling": o_i.clone(),      # [Br, h_qkv] - O before 1/l scaling
                    "o_final": o_final.clone(),           # [Br, h_qkv] - final output
                }

    if return_intermediates:
        return o, intermediates
    return o


@torch.no_grad()
def flash_attn2_gemv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qk_scale: float,
    s_q: int,
    s_kv: int,
    h_qkv: int,
    num_q_heads: int,
    num_kv_heads: int,
    Bc: int,
    Br: int,
    debug: bool = False,
    return_intermediates: bool = False,
):
    """
    Args:
        q: [b, s_q, num_q_heads, h_qkv], query tensor
        k: [b, s_kv, num_kv_heads, h_qkv], key tensor
        v: [b, s_kv, num_kv_heads, h_qkv], value tensor
        qk_scale: float, scale factor for qk, usually 1 / sqrt(h_qkv)
        s_q: int, query sequence length, should be 1 for vector-matrix hardware
        s_kv: int, key-value sequence length, increases with generation step (KV cache)
        h_qkv: int, hidden size per head
        num_q_heads: int, number of query heads
        num_kv_heads: int, number of key-value heads
        Bc: int, tile size for key-value matrix
        debug: bool, print debug info for head 0
        return_intermediates: bool, return intermediate values for all heads

    Returns:
        o: [b, s_q, num_q_heads, h_qkv], attention output
        all_intermediates (optional): dict mapping head_idx -> intermediates dict
            Each intermediates dict has keys (batch_idx, q_tile_idx, kv_tile_idx) -> values dict
            Values dict contains:
                - q_i, k_j, v_j: input tiles
                - s_j_unscaled, s_j: QKT before/after scaling
                - rowmax_s_j, m_old, m_new, m_res, exp_m_res: online softmax values
                - s_j_shifted, p, p_row_sum: softmax scores
                - l_old, l_new: running sum of exp
                - pv: P @ V result
                - o_old, o_scaled, o_i: output accumulation stages
            Final output has key (batch_idx, q_tile_idx, "final") with l_final, inv_l, o_before_scaling, o_final
    """
    b = q.size(0)  # batch size

    Tc = ceil_div(s_kv, Bc)  # temporal tile count
    Tr = ceil_div(s_q, Br)  # temporal tile count
    num_head_groups = num_q_heads // num_kv_heads

    o = torch.zeros(b, s_q, h_qkv * num_q_heads, device=q.device)  # [b, s_q, h]
    all_intermediates = {} if return_intermediates else None

    for head_idx in range(num_q_heads):
        q_head = q[:, :, head_idx, :]
        kv_head_idx = head_idx // num_head_groups
        k_head = k[:, :, kv_head_idx, :]
        v_head = v[:, :, kv_head_idx, :]

        head_debug = debug and (head_idx == 0)

        if return_intermediates:
            o_head, head_intermediates = flash_attn2_head_gemv(
                q_head, k_head, v_head,
                qk_scale=qk_scale, s_q=s_q, s_kv=s_kv, h_qkv=h_qkv,
                Bc=Bc, Tc=Tc, Br=Br, Tr=Tr,
                debug=head_debug, return_intermediates=True
            )
            all_intermediates[head_idx] = {
                "kv_head_idx": kv_head_idx,
                "intermediates": head_intermediates,
            }
        else:
            o_head = flash_attn2_head_gemv(
                q_head, k_head, v_head,
                qk_scale=qk_scale, s_q=s_q, s_kv=s_kv, h_qkv=h_qkv,
                Bc=Bc, Tc=Tc, Br=Br, Tr=Tr,
                debug=head_debug, return_intermediates=False
            )

        o[:, :, head_idx * h_qkv : (head_idx + 1) * h_qkv] = o_head

    o = o.reshape(b, s_q, num_q_heads, h_qkv)
    check_shape(q, (b, s_q, num_q_heads, h_qkv))
    check_shape(k, (b, s_kv, num_kv_heads, h_qkv))
    check_shape(v, (b, s_kv, num_kv_heads, h_qkv))
    check_shape(q_head, (b, s_q, h_qkv))
    check_shape(k_head, (b, s_kv, h_qkv))
    check_shape(v_head, (b, s_kv, h_qkv))
    check_shape(o_head, (b, s_q, h_qkv))

    if return_intermediates:
        return o, all_intermediates
    return o
