import torch


def sdpa(q, k, v, s, sm_scale, sliding_window=0):
    """
    Standard sliding window attention.
    sliding_window == 0 means no sliding window.
    """
    n_tokens, n_heads, q_mult, d_head = q.shape
    assert k.shape == (n_tokens, n_heads, d_head)
    assert v.shape == (n_tokens, n_heads, d_head)
    k = k[:, :, None, :].expand(-1, -1, q_mult, -1)
    v = v[:, :, None, :].expand(-1, -1, q_mult, -1)
    s = s.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window)
    qk = torch.einsum("qhmd,khmd->hmqk", q, k)
    qk *= sm_scale
    qk += mask[None, None, :, :]
    qk = torch.cat([qk, s], dim=-1)
    w = torch.softmax(qk, dim=-1)
    w = w[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", w, v)
    return attn.reshape(n_tokens, -1)


def sliding_attn(q, k, v, s, sm_scale, sliding_window=0):
    """
    Sliding window attention with shape checking.
    sliding_window == 0 means no sliding window.
    """
    print("=" * 60)
    print("SLIDING ATTENTION SHAPE CHECK")
    print("=" * 60)

    # Input shapes
    n_tokens, n_heads, q_mult, d_head = q.shape
    print("\n[INPUTS]")
    print(f"  Q: {q.shape}  (n_tokens={n_tokens}, n_heads={n_heads}, q_mult={q_mult}, d_head={d_head})")
    print(f"  K: {k.shape}  (expected: {(n_tokens, n_heads, d_head)})")
    print(f"  V: {v.shape}  (expected: {(n_tokens, n_heads, d_head)})")
    print(f"  S: {s.shape}  (will reshape to: {(n_heads, q_mult, 1, 1)})")
    print(f"  sm_scale: {sm_scale}")
    print(f"  sliding_window: {sliding_window}")

    # Assertions
    assert k.shape == (n_tokens, n_heads, d_head), f"K shape mismatch: {k.shape} vs {(n_tokens, n_heads, d_head)}"
    assert v.shape == (n_tokens, n_heads, d_head), f"V shape mismatch: {v.shape} vs {(n_tokens, n_heads, d_head)}"

    # Expand K, V
    k = k[:, :, None, :].expand(-1, -1, q_mult, -1)
    v = v[:, :, None, :].expand(-1, -1, q_mult, -1)
    print("\n[AFTER EXPAND K, V]")
    print(f"  K: {k.shape}  (n_tokens, n_heads, q_mult, d_head)")
    print(f"  V: {v.shape}  (n_tokens, n_heads, q_mult, d_head)")

    # Reshape S (sink tokens)
    s = s.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    print("\n[AFTER RESHAPE/EXPAND S]")
    print(f"  S: {s.shape}  (n_heads, q_mult, n_tokens, 1)")

    # Create causal mask
    mask = torch.triu(q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    print("\n[CAUSAL MASK]")
    print(f"  mask: {mask.shape}  (n_tokens, n_tokens)")

    # Apply sliding window
    if sliding_window > 0:
        sliding_mask = torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window)
        mask += sliding_mask
        print(f"  sliding mask applied (window={sliding_window})")

    # QK matmul via einsum
    # Q: (q, h, m, d) -> "qhmd"
    # K: (k, h, m, d) -> "khmd"
    # Out: (h, m, q, k) -> "hmqk"
    qk = torch.einsum("qhmd,khmd->hmqk", q, k)
    print("\n[QK = einsum('qhmd,khmd->hmqk', Q, K)]")
    print(f"  QK: {qk.shape}  (n_heads, q_mult, n_tokens_q, n_tokens_k)")

    # Scale
    qk *= sm_scale
    print("\n[AFTER SCALING]")
    print(f"  QK: {qk.shape}  (unchanged)")

    # Add mask
    qk += mask[None, None, :, :]
    print("\n[AFTER ADDING MASK]")
    print(f"  mask broadcast: {mask[None, None, :, :].shape}")
    print(f"  QK: {qk.shape}  (unchanged)")

    # Concatenate with sinks
    qk = torch.cat([qk, s], dim=-1)
    print("\n[AFTER CAT WITH SINKS]")
    print(f"  QK: {qk.shape}  (n_heads, q_mult, n_tokens, n_tokens+1)")

    # Softmax
    w = torch.softmax(qk, dim=-1)
    print("\n[AFTER SOFTMAX]")
    print(f"  W: {w.shape}  (unchanged)")

    # Remove sink dimension
    w = w[..., :-1]
    print("\n[AFTER REMOVING SINK DIM]")
    print(f"  W: {w.shape}  (n_heads, q_mult, n_tokens, n_tokens)")

    # Attention output via einsum
    # W: (h, m, q, k) -> "hmqk"
    # V: (k, h, m, d) -> "khmd"
    # Out: (q, h, m, d) -> "qhmd"
    attn = torch.einsum("hmqk,khmd->qhmd", w, v)
    print("\n[ATTN = einsum('hmqk,khmd->qhmd', W, V)]")
    print(f"  attn: {attn.shape}  (n_tokens, n_heads, q_mult, d_head)")

    # Final reshape
    output = attn.reshape(n_tokens, -1)
    print("\n[FINAL RESHAPE]")
    print(f"  output: {output.shape}  (n_tokens, n_heads * q_mult * d_head = {n_heads * q_mult * d_head})")
    print("=" * 60)

    return output


@torch.no_grad()
def check_sliding_attn():
    # Example dimensions
    n_tokens = 8
    n_heads = 4  # num_key_value_heads
    q_mult = 2  # num_attention_heads // num_key_value_heads (GQA ratio)
    d_head = 64

    q = torch.randn(n_tokens, n_heads, q_mult, d_head)
    k = torch.randn(n_tokens, n_heads, d_head)
    v = torch.randn(n_tokens, n_heads, d_head)
    s = torch.randn(n_heads * q_mult)  # sinks parameter
    sm_scale = 1.0 / (d_head**0.5)

    print("\n>>> Testing with sliding_window=0 (no sliding window)")
    out = sliding_attn(q, k, v, s, sm_scale, sliding_window=0)

    # Verify against reference
    out_ref = sdpa(q, k, v, s, sm_scale, sliding_window=0)
    assert torch.allclose(out, out_ref, atol=1e-5), "Mismatch with reference sdpa"
    print("\n[PASSED] Output matches reference sdpa")

    print("\n\n>>> Testing with sliding_window=4")
    out = sliding_attn(q, k, v, s, sm_scale, sliding_window=4)

    # Verify against reference
    out_ref = sdpa(q, k, v, s, sm_scale, sliding_window=4)
    assert torch.allclose(out, out_ref, atol=1e-5), "Mismatch with reference sdpa"
    print("\n[PASSED] Output matches reference sdpa")


if __name__ == "__main__":
    check_sliding_attn()
