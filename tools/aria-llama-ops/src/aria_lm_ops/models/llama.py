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
def flash_attn2_head_gemv(q, k, v, qk_scale, s_q, s_kv, h_qkv, Bc, Tc, Br, Tr, debug = False):
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

    Returns:
        o: [b, s_q, h_qkv], attention output
    """
    b = q.size(0)
    o = torch.zeros(b, s_q, h_qkv, device=q.device)  # output

    for b_i in range(b):
        for i in range(Tr):
            q_i = q[b_i, i * Br : (i + 1) * Br, :].squeeze(0)  # [Br, h_qkv]
            m = torch.full((Br,), float("-inf"), device=q.device)
            l = torch.zeros((Br,), device=q.device)
            o_i = torch.zeros((Br, h_qkv), device=q.device)
            for j in range(Tc):
                k_j = k[b_i, j * Bc : (j + 1) * Bc, :].squeeze(0)  # [Bc, h_qkv]
                v_j = v[b_i, j * Bc : (j + 1) * Bc, :].squeeze(0)  # [Bc, h_qkv]
                s_j = q_i @ k_j.transpose(0, 1) * qk_scale  # Q @ Kj^T, [Br, Bc]
                
                rowmax_s_j = s_j.max(dim=1).values  # [Br]
                m_new = torch.maximum(m, rowmax_s_j)  # shape: [b, s_q, 1]
                # Subtract each element of m_new from the corresponding row of s_j
                # s_j: [Br, Bc], m_new: [Br]
                # We want: s_j_shifted[i, :] = s_j[i, :] - m_new[i]
                s_j_shifted = s_j - m_new.unsqueeze(1)
                p = torch.exp(s_j_shifted)  # exp(Sj - rowmax(Sj)), shape: [Br, Bc]
                p = p.to(torch.bfloat16)
                m_res = m - m_new  # [Br]
                m = m_new
                l_scale = torch.exp(m_res)  # [Br]
                p_row_sum = p.sum(dim=1)

                l = l_scale * l + p_row_sum  # [Br]
                o_scale = torch.exp(m_res)  # [Br]
                
                # Format o_scale ([Br]) into a diagonal matrix of shape [Br, Br] with o_scale as diagonal entries
                o_scale_diag = torch.diag(o_scale)  # [Br, Br]
                o_i = torch.matmul(o_scale_diag, o_i) + torch.matmul(p, v_j)  # [Br, h_qkv]  
                if debug and b_i == 0 and i == 0 and j == 0: 
                    print(f"b_i={b_i}, i={i}")
                    print("s_j", s_j)
                    print("rowmax_s_j shape", rowmax_s_j.shape)
                    print("m_new shape", m_new.shape)
                    print("m_new", m_new)
                    print("s_j_shifted", s_j_shifted)
                    print("s_j_shifted shape", s_j_shifted.shape)
                    print("p shape", p.shape)
                    print("m_res shape", m_res.shape)
                    print("m_res", m_res)
                    print("p shape", p.shape)
                    print("p: ", p)
                    # print("p_row_sum shape", p_row_sum.shape)
                    # print("p_row_sum: ", p_row_sum)
                    # print("l_scale shape", l_scale.shape)
                    # print("l shape", l.shape)
                    # print("new l : ", l)
                    # print("o_scale shape", o_scale.shape)
                    # print("o_scale", o_scale)
                    # print("o scale diag", o_scale_diag)
                    # print("o scale diag shape", o_scale_diag.shape)
                    # print("torch.matmul(o_scale_diag, o_i).shape", torch.matmul(o_scale_diag, o_i).shape)
                    print("v_j shape", v_j.shape)
                    print("v_j", v_j)
                    print("torch.matmul(p, v_j).shape", torch.matmul(p, v_j).shape)
                    print("torch.matmul(p, v_j) value ", torch.matmul(p, v_j))
                    print("o_i", o_i)
                    print("torch.diag(1.0 / l).shape", torch.diag(1.0 / l).shape)
            o[b_i, i * Br : (i + 1) * Br, :] = torch.matmul(torch.diag(1.0 / l), o_i)

    # assert b == 1, "b must be 1"
    # assert s_q == 1, "s_q must be 1"
    # check_shape(q, (b, s_q, h_qkv))
    # check_shape(k, (b, s_kv, h_qkv))
    # check_shape(v, (b, s_kv, h_qkv))
    # assert b == 1, "b must be 1"
    # assert s_q == 1, "s_q must be 1"
    # assert ceil_div(s_kv, Bc) == Tc, f"s_kv={s_kv}, Bc={Bc}, Tc={Tc}"
    # check_shape(k_j, (b, 0, h_qkv))  # [1, Bc, h_qkv]
    # check_shape(v_j, (b, 0, h_qkv))  # [1, Bc, h_qkv]
    # check_shape(s_j, (b, s_q, 0))
    # check_shape(o, (b, s_q, h_qkv))
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

    Returns:
        o: [b, s_q, num_q_heads, h_qkv], attention output
    """
    b = q.size(0)  # batch size

    Tc = ceil_div(s_kv, Bc)  # temporal tile count
    Tr = ceil_div(s_q, Br)  # temporal tile count
    num_head_groups = num_q_heads // num_kv_heads

    o = torch.zeros(b, s_q, h_qkv * num_q_heads, device=q.device)  # [b, s_q, h]
    for head_idx in range(num_q_heads):
        q_head = q[:, :, head_idx, :]
        kv_head_idx = head_idx // num_head_groups
        k_head = k[:, :, kv_head_idx, :]
        v_head = v[:, :, kv_head_idx, :]
        if head_idx == 0:
            o_head = flash_attn2_head_gemv(
                q_head, k_head, v_head, qk_scale=qk_scale, s_q=s_q, s_kv=s_kv, h_qkv=h_qkv, Bc=Bc, Tc=Tc, Br=Br, Tr =Tr, debug=True
            )
        else:
            o_head = flash_attn2_head_gemv(
                q_head, k_head, v_head, qk_scale=qk_scale, s_q=s_q, s_kv=s_kv, h_qkv=h_qkv, Bc=Bc, Tc=Tc, Br=Br, Tr =Tr, debug=False
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

    return o
