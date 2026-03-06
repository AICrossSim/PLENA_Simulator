#!/usr/bin/env python3
"""
Quantization Memory Study for LLaMA 3.3 70B.

Computes:
- Peak Bandwidth: max(W, KV) * MLEN + A * MLEN @ 1GHz
- KV Footprint: KV cache memory for batch_size=8, context=90k+8k
- Weight: Model weight memory at different precisions
"""

# =============================================================================
# Model Configuration (LLaMA 3.3 70B)
# =============================================================================
MODEL_CONFIG = {
    "hidden_size": 8192,
    "num_attention_heads": 64,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 28672,
    "num_hidden_layers": 80,
    "vocab_size": 128256,
}

# =============================================================================
# Experiment Configuration
# =============================================================================
BATCH_SIZE = 8
INPUT_SEQ_LEN = 90_000   # 90k prefill
OUTPUT_SEQ_LEN = 8_000   # 8k output
TOTAL_CONTEXT = INPUT_SEQ_LEN + OUTPUT_SEQ_LEN  # 98k

MLEN = 2048              # Memory lane width
FREQ_GHZ = 1.0           # 1 GHz

# Quantization configurations: (W, A, KV) in bits
QUANT_CONFIGS = [
    (16, 16, 16),
    (4, 16, 16),
    (4, 4, 16),
    (4, 4, 4),
]


def compute_peak_bandwidth(w_bits: int, a_bits: int, kv_bits: int) -> float:
    """
    Compute peak bandwidth in GB/s.

    Formula: (max(W, KV) * MLEN + A * MLEN) bits/cycle @ 1GHz
    Convert to GB/s: bits * 1e9 / 8 / 1e9 = bits / 8
    """
    peak_bw_bits_per_cycle = max(w_bits, kv_bits) * MLEN + a_bits * MLEN
    # At 1GHz, cycles/s = 1e9
    # GB/s = bits_per_cycle * 1e9 / 8 / 1e9 = bits_per_cycle / 8
    peak_bw_gbps = peak_bw_bits_per_cycle / 8
    return peak_bw_gbps


def compute_kv_footprint(kv_bits: int) -> float:
    """
    Compute KV cache footprint in GB.

    KV cache = 2 (K+V) * batch_size * context_len * num_kv_heads * head_dim * num_layers * bytes
    """
    num_kv_heads = MODEL_CONFIG["num_key_value_heads"]
    head_dim = MODEL_CONFIG["head_dim"]
    num_layers = MODEL_CONFIG["num_hidden_layers"]

    # 2 for K and V
    kv_elements = 2 * BATCH_SIZE * TOTAL_CONTEXT * num_kv_heads * head_dim * num_layers
    kv_bytes = kv_elements * (kv_bits / 8)
    kv_gb = kv_bytes / (1024**3)
    return kv_gb


def compute_model_weight(w_bits: int) -> float:
    """
    Compute model weight size in GB.

    For LLaMA 3.3 70B, count the actual parameters:
    Per layer:
      - Q projection: hidden_size * (num_attention_heads * head_dim)
      - K projection: hidden_size * (num_kv_heads * head_dim)
      - V projection: hidden_size * (num_kv_heads * head_dim)
      - O projection: (num_attention_heads * head_dim) * hidden_size
      - Gate projection: hidden_size * intermediate_size
      - Up projection: hidden_size * intermediate_size
      - Down projection: intermediate_size * hidden_size
      - RMS norms: 2 * hidden_size

    Embeddings:
      - Input embedding: vocab_size * hidden_size
      - Output embedding (tied or separate): vocab_size * hidden_size
      - Final RMS norm: hidden_size
    """
    hidden = MODEL_CONFIG["hidden_size"]
    num_heads = MODEL_CONFIG["num_attention_heads"]
    num_kv_heads = MODEL_CONFIG["num_key_value_heads"]
    head_dim = MODEL_CONFIG["head_dim"]
    intermediate = MODEL_CONFIG["intermediate_size"]
    num_layers = MODEL_CONFIG["num_hidden_layers"]
    vocab = MODEL_CONFIG["vocab_size"]

    # Per-layer parameters
    q_proj = hidden * (num_heads * head_dim)           # 8192 * 8192 = 67M
    k_proj = hidden * (num_kv_heads * head_dim)        # 8192 * 1024 = 8.4M
    v_proj = hidden * (num_kv_heads * head_dim)        # 8192 * 1024 = 8.4M
    o_proj = (num_heads * head_dim) * hidden           # 8192 * 8192 = 67M

    gate_proj = hidden * intermediate                   # 8192 * 28672 = 235M
    up_proj = hidden * intermediate                     # 8192 * 28672 = 235M
    down_proj = intermediate * hidden                   # 28672 * 8192 = 235M

    rms_norms_per_layer = 2 * hidden                   # 2 * 8192 = 16K

    params_per_layer = (q_proj + k_proj + v_proj + o_proj +
                        gate_proj + up_proj + down_proj + rms_norms_per_layer)

    total_layer_params = params_per_layer * num_layers

    # Embedding parameters
    input_embed = vocab * hidden                        # 128256 * 8192 = 1.05B
    output_embed = vocab * hidden                       # Usually tied, but count separately for completeness
    final_rms = hidden

    # LLaMA usually ties embeddings, so we count input_embed only
    embedding_params = input_embed + final_rms

    total_params = total_layer_params + embedding_params

    # Convert to bytes then GB
    weight_bytes = total_params * (w_bits / 8)
    weight_gb = weight_bytes / (1024**3)

    return weight_gb


def main():
    results = {}

    print("=" * 70)
    print("Quantization Memory Study for LLaMA 3.3 70B")
    print("=" * 70)
    print(f"Batch Size:     {BATCH_SIZE}")
    print(f"Input Tokens:   {INPUT_SEQ_LEN:,} (90k)")
    print(f"Output Tokens:  {OUTPUT_SEQ_LEN:,} (8k)")
    print(f"Total Context:  {TOTAL_CONTEXT:,} (98k)")
    print(f"MLEN:           {MLEN}")
    print(f"Frequency:      {FREQ_GHZ} GHz")
    print("=" * 70)
    print()

    for w_bits, a_bits, kv_bits in QUANT_CONFIGS:
        config_name = f"{w_bits}/{a_bits}/{kv_bits}"

        peak_bw = compute_peak_bandwidth(w_bits, a_bits, kv_bits)
        kv_footprint = compute_kv_footprint(kv_bits)
        weight = compute_model_weight(w_bits)

        results[config_name] = {
            "Peak Bandwidth (GB/s)": peak_bw,
            "KV Footprint (GB)": round(kv_footprint, 2),
            "Weight (GB)": round(weight, 2),
        }

        print(f"W/A/KV = {config_name}")
        print(f"  Peak Bandwidth: {peak_bw:.0f} GB/s")
        print(f"  KV Footprint:   {kv_footprint:.2f} GB")
        print(f"  Weight:         {weight:.2f} GB")
        print()

    print("=" * 70)
    print("Results Dictionary:")
    print("=" * 70)
    print(results)

    return results


if __name__ == "__main__":
    results = main()
