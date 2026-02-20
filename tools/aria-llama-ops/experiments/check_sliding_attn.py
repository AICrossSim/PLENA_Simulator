import torch


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    """
    Standard sliding window attention.
    sliding_window == 0 means no sliding window.
    """
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


def sliding_attn(Q, K, V, S, sm_scale, sliding_window=0):
    """
    Sliding window attention with shape checking.
    sliding_window == 0 means no sliding window.
    """
    print("=" * 60)
    print("SLIDING ATTENTION SHAPE CHECK")
    print("=" * 60)

    # Input shapes
    n_tokens, n_heads, q_mult, d_head = Q.shape
    print(f"\n[INPUTS]")
    print(f"  Q: {Q.shape}  (n_tokens={n_tokens}, n_heads={n_heads}, q_mult={q_mult}, d_head={d_head})")
    print(f"  K: {K.shape}  (expected: {(n_tokens, n_heads, d_head)})")
    print(f"  V: {V.shape}  (expected: {(n_tokens, n_heads, d_head)})")
    print(f"  S: {S.shape}  (will reshape to: {(n_heads, q_mult, 1, 1)})")
    print(f"  sm_scale: {sm_scale}")
    print(f"  sliding_window: {sliding_window}")

    # Assertions
    assert K.shape == (n_tokens, n_heads, d_head), f"K shape mismatch: {K.shape} vs {(n_tokens, n_heads, d_head)}"
    assert V.shape == (n_tokens, n_heads, d_head), f"V shape mismatch: {V.shape} vs {(n_tokens, n_heads, d_head)}"

    # Expand K, V
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    print(f"\n[AFTER EXPAND K, V]")
    print(f"  K: {K.shape}  (n_tokens, n_heads, q_mult, d_head)")
    print(f"  V: {V.shape}  (n_tokens, n_heads, q_mult, d_head)")

    # Reshape S (sink tokens)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    print(f"\n[AFTER RESHAPE/EXPAND S]")
    print(f"  S: {S.shape}  (n_heads, q_mult, n_tokens, 1)")

    # Create causal mask
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    print(f"\n[CAUSAL MASK]")
    print(f"  mask: {mask.shape}  (n_tokens, n_tokens)")

    # Apply sliding window
    if sliding_window > 0:
        sliding_mask = torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
        mask += sliding_mask
        print(f"  sliding mask applied (window={sliding_window})")

    # QK matmul via einsum
    # Q: (q, h, m, d) -> "qhmd"
    # K: (k, h, m, d) -> "khmd"
    # Out: (h, m, q, k) -> "hmqk"
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    print(f"\n[QK = einsum('qhmd,khmd->hmqk', Q, K)]")
    print(f"  QK: {QK.shape}  (n_heads, q_mult, n_tokens_q, n_tokens_k)")

    # Scale
    QK *= sm_scale
    print(f"\n[AFTER SCALING]")
    print(f"  QK: {QK.shape}  (unchanged)")

    # Add mask
    QK += mask[None, None, :, :]
    print(f"\n[AFTER ADDING MASK]")
    print(f"  mask broadcast: {mask[None, None, :, :].shape}")
    print(f"  QK: {QK.shape}  (unchanged)")

    # Concatenate with sinks
    QK = torch.cat([QK, S], dim=-1)
    print(f"\n[AFTER CAT WITH SINKS]")
    print(f"  QK: {QK.shape}  (n_heads, q_mult, n_tokens, n_tokens+1)")

    # Softmax
    W = torch.softmax(QK, dim=-1)
    print(f"\n[AFTER SOFTMAX]")
    print(f"  W: {W.shape}  (unchanged)")

    # Remove sink dimension
    W = W[..., :-1]
    print(f"\n[AFTER REMOVING SINK DIM]")
    print(f"  W: {W.shape}  (n_heads, q_mult, n_tokens, n_tokens)")

    # Attention output via einsum
    # W: (h, m, q, k) -> "hmqk"
    # V: (k, h, m, d) -> "khmd"
    # Out: (q, h, m, d) -> "qhmd"
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    print(f"\n[ATTN = einsum('hmqk,khmd->qhmd', W, V)]")
    print(f"  attn: {attn.shape}  (n_tokens, n_heads, q_mult, d_head)")

    # Final reshape
    output = attn.reshape(n_tokens, -1)
    print(f"\n[FINAL RESHAPE]")
    print(f"  output: {output.shape}  (n_tokens, n_heads * q_mult * d_head = {n_heads * q_mult * d_head})")
    print("=" * 60)

    return output


@torch.no_grad()
def check_sliding_attn():
    # Example dimensions
    n_tokens = 8
    n_heads = 4       # num_key_value_heads
    q_mult = 2        # num_attention_heads // num_key_value_heads (GQA ratio)
    d_head = 64

    Q = torch.randn(n_tokens, n_heads, q_mult, d_head)
    K = torch.randn(n_tokens, n_heads, d_head)
    V = torch.randn(n_tokens, n_heads, d_head)
    S = torch.randn(n_heads * q_mult)  # sinks parameter
    sm_scale = 1.0 / (d_head ** 0.5)

    print("\n>>> Testing with sliding_window=0 (no sliding window)")
    out = sliding_attn(Q, K, V, S, sm_scale, sliding_window=0)

    # Verify against reference
    out_ref = sdpa(Q, K, V, S, sm_scale, sliding_window=0)
    assert torch.allclose(out, out_ref, atol=1e-5), "Mismatch with reference sdpa"
    print("\n[PASSED] Output matches reference sdpa")

    print("\n\n>>> Testing with sliding_window=4")
    out = sliding_attn(Q, K, V, S, sm_scale, sliding_window=4)

    # Verify against reference
    out_ref = sdpa(Q, K, V, S, sm_scale, sliding_window=4)
    assert torch.allclose(out, out_ref, atol=1e-5), "Mismatch with reference sdpa"
    print("\n[PASSED] Output matches reference sdpa")


if __name__ == "__main__":
    check_sliding_attn()
