"""
Reference implementation of Flash Attention 2 GeMV for LLaMA-style GQA models.

flash_attn2_gemv computes grouped-query attention using tiled flash attention
with online softmax, returning both the output and per-tile intermediates for
hardware simulator verification.
"""

import math
import torch


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
    Flash Attention 2 GeMV reference implementation with GQA support.

    Args:
        q:               (batch, s_q, num_q_heads, h_qkv)
        k:               (batch, s_kv, num_kv_heads, h_qkv)
        v:               (batch, s_kv, num_kv_heads, h_qkv)
        qk_scale:        scalar multiplier applied to QK^T
        s_q:             query sequence length
        s_kv:            key/value sequence length
        h_qkv:           per-head dimension
        num_q_heads:     total query heads
        num_kv_heads:    total KV heads (num_q_heads must be divisible by this)
        Bc:              KV tile size (columns)
        Br:              Q tile size (rows)
        debug:           if True print tile-level debug info
        return_intermediates: if True, also return per-tile intermediate tensors

    Returns:
        output: (batch, s_q, num_q_heads, h_qkv) bfloat16/float32 matching q dtype
        all_intermediates (if return_intermediates):
            list of length num_q_heads, each element is a dict:
            {
                "intermediates": {
                    (batch_idx, q_tile_idx, kv_tile_idx): {
                        "l_new":      Tensor[Br]  -- normalisation sum after this tile
                        "exp_m_res":  Tensor[Br]  -- exp(m_old - m_new), the rescale factor
                        "m_new":      Tensor[Br]  -- updated running max
                    }
                }
            }
    """
    assert num_q_heads % num_kv_heads == 0, (
        f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    )
    q_per_kv = num_q_heads // num_kv_heads

    batch = q.shape[0]
    orig_dtype = q.dtype

    # Use float32 internally for numerical stability
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()

    output = torch.zeros(batch, s_q, num_q_heads, h_qkv, dtype=torch.float32)

    num_q_tiles = math.ceil(s_q / Br)
    num_kv_tiles = math.ceil(s_kv / Bc)

    if return_intermediates:
        all_intermediates = [{"intermediates": {}} for _ in range(num_q_heads)]

    for b in range(batch):
        for q_head in range(num_q_heads):
            kv_head = q_head // q_per_kv

            for q_tile in range(num_q_tiles):
                q_start = q_tile * Br
                q_end = min(q_start + Br, s_q)
                actual_Br = q_end - q_start

                Q_tile = q_f[b, q_start:q_end, q_head, :]  # (actual_Br, h_qkv)

                # Flash attention running state
                m = torch.full((actual_Br,), float("-inf"), dtype=torch.float32)
                l = torch.zeros(actual_Br, dtype=torch.float32)
                O = torch.zeros(actual_Br, h_qkv, dtype=torch.float32)

                for kv_tile in range(num_kv_tiles):
                    kv_start = kv_tile * Bc
                    kv_end = min(kv_start + Bc, s_kv)

                    K_tile = k_f[b, kv_start:kv_end, kv_head, :]  # (actual_Bc, h_qkv)
                    V_tile = v_f[b, kv_start:kv_end, kv_head, :]  # (actual_Bc, h_qkv)

                    # Attention scores
                    S = torch.matmul(Q_tile, K_tile.T) * qk_scale  # (actual_Br, actual_Bc)

                    # Online softmax update
                    m_new = torch.maximum(m, S.max(dim=-1).values)   # (actual_Br,)
                    exp_m_res = torch.exp(m - m_new)                  # rescale factor for old state
                    P = torch.exp(S - m_new.unsqueeze(-1))            # (actual_Br, actual_Bc)
                    l_new = exp_m_res * l + P.sum(dim=-1)             # (actual_Br,)

                    # Accumulate output
                    O = exp_m_res.unsqueeze(-1) * O + torch.matmul(P, V_tile)

                    if debug:
                        print(
                            f"  b={b} q_head={q_head} q_tile={q_tile} kv_tile={kv_tile} "
                            f"m_new[:4]={m_new[:4].tolist()} l_new[:4]={l_new[:4].tolist()}"
                        )

                    if return_intermediates:
                        all_intermediates[q_head]["intermediates"][(b, q_tile, kv_tile)] = {
                            "l_new": l_new.clone(),
                            "exp_m_res": exp_m_res.clone(),
                            "m_new": m_new.clone(),
                        }

                    m = m_new
                    l = l_new

                # Normalise
                O = O / l.unsqueeze(-1)
                output[b, q_start:q_end, q_head, :] = O

    output = output.to(orig_dtype)

    if return_intermediates:
        return output, all_intermediates
    return output
