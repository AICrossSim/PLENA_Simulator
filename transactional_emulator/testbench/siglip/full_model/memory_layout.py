"""Memory layout computation for full 27-layer SigLIP model.

Extends existing layout utilities to handle all layers' activation and weight storage.
"""

def pad_to(x: int, m: int) -> int:
    """Pad x up to nearest multiple of m."""
    return ((x + m - 1) // m) * m


def compute_embedding_vram_layout(
    seq_len: int,
    hidden_size: int,
    mlen: int,
    vlen: int,
    blen: int,
    vector_sram_base: int = 0,
) -> dict:
    """Compute VRAM layout for embedding stage output.

    Args:
        seq_len: Sequence length (num_patches)
        hidden_size: Hidden dimension
        mlen, vlen, blen: Hardware parameters
        vector_sram_base: Starting VRAM address

    Returns:
        dict with:
            - embedding_base: Start address of embedding output
            - embedding_size: Size in elements
            - next_free_addr: Next available VRAM address after embedding
    """
    # Embedding output: [seq_len, hidden_size] in chunk-major layout
    embedding_size = seq_len * hidden_size

    embedding_base = vector_sram_base
    next_free = embedding_base + embedding_size

    return {
        "embedding_base": embedding_base,
        "embedding_size": embedding_size,
        "next_free_addr": next_free,
    }


def compute_full_model_vram_layout(
    config: dict,
    mlen: int = 64,
    vlen: int = 64,
    blen: int = 4,
    max_layers: int = 27,
    vector_sram_base: int = 0,
) -> dict:
    """Compute VRAM layout for full model: embedding + all encoder layers.

    Each layer needs:
    - Input activation: [seq_len, hidden_size]
    - Output activation: [seq_len, hidden_size]
    - Flash attention intermediates: S matrix, PV matrix (per flash-attn call)

    Strategy: Ping-pong between layer input/output buffers to minimize VRAM.
    Simplified: Store all layer outputs sequentially (conservative estimate).

    Args:
        config: Config dict from load_siglip_config()
        mlen, vlen, blen: Hardware parameters
        max_layers: Number of encoder layers to allocate for
        vector_sram_base: Starting VRAM address

    Returns:
        dict with per-layer bases and sizes
    """
    seq_len = (config["image_size"] // config["patch_size"]) ** 2
    hidden_size = config["hidden_size"]

    # Embedding stage
    embed_info = compute_embedding_vram_layout(seq_len, hidden_size, mlen, vlen, blen, vector_sram_base)

    layout = {
        "seq_len": seq_len,
        "hidden_size": hidden_size,
        "embedding_base": embed_info["embedding_base"],
        "embedding_size": embed_info["embedding_size"],
        "layer_bases": {},
        "layer_sizes": {},
    }

    # Allocate per-layer output storage
    cur_addr = embed_info["next_free_addr"]
    per_layer_activation_size = seq_len * hidden_size

    for layer_idx in range(max_layers):
        layout["layer_bases"][layer_idx] = cur_addr
        layout["layer_sizes"][layer_idx] = per_layer_activation_size
        cur_addr += per_layer_activation_size

    layout["total_vram_elements"] = cur_addr - vector_sram_base
    layout["total_vram_mb"] = layout["total_vram_elements"] * 2 / (1024 * 1024)  # 2 bytes per BF16

    return layout


def compute_embedding_weights_hbm_size(config: dict, real_data_ratio: float = 1.125) -> int:
    """Compute HBM size needed for embedding weights.

    Returns size in elements (before real_data_ratio expansion).
    """
    image_size = config["image_size"]
    patch_size = config["patch_size"]
    num_patches = (image_size // patch_size) ** 2
    hidden_size = config["hidden_size"]
    num_channels = config["num_channels"]

    # Patch embedding weight: [hidden_size, num_channels*patch_size*patch_size]
    # Usually stored transposed: [num_channels*K*K, hidden_size]
    patch_weight_elems = hidden_size * num_channels * patch_size * patch_size

    # Position embedding: [num_patches, hidden_size]
    position_weight_elems = num_patches * hidden_size

    # Bias terms
    patch_bias_elems = hidden_size
    position_bias_elems = 0  # Usually no bias on position embeddings

    total = patch_weight_elems + position_weight_elems + patch_bias_elems + position_bias_elems
    return int(total * real_data_ratio)


def compute_layer_weights_hbm_size(config: dict, real_data_ratio: float = 1.125) -> int:
    """Compute HBM size needed for one encoder layer's weights.

    Includes: LN1, Q/K/V projections, output projection, LN2, MLP (fc1, fc2).

    Returns size in elements.
    """
    hidden_size = config["hidden_size"]
    inter_size = config["intermediate_size"]

    # Layer norm 1
    ln1_size = 2 * hidden_size  # weight + bias

    # Q, K, V, Out projections
    # Q: [hidden_size, hidden_size]
    # K: [hidden_size, hidden_size]
    # V: [hidden_size, hidden_size]
    # Out: [hidden_size, hidden_size]
    qkv_out_size = 4 * (hidden_size * hidden_size + hidden_size)  # weights + biases

    # Layer norm 2
    ln2_size = 2 * hidden_size

    # MLP: fc1 (up), fc2 (down)
    # fc1: [hidden_size, inter_size]
    # fc2: [inter_size, hidden_size]
    mlp_size = (hidden_size * inter_size + inter_size) + (inter_size * hidden_size + hidden_size)

    total = ln1_size + qkv_out_size + ln2_size + mlp_size
    return int(total * real_data_ratio)


def compute_full_model_hbm_layout(
    config: dict,
    embedding_weights: dict,
    layer_weights_list: list,
    real_data_ratio: float = 1.125,
    align_elems: int = 64,
) -> tuple[dict, int]:
    """Compute HBM layout for all model weights (embedding + 27 layers).

    Args:
        config: Config dict
        embedding_weights: Embedding weight dict
        layer_weights_list: List of per-layer weight dicts
        real_data_ratio: Expansion factor for quantization overhead
        align_elems: Alignment boundary

    Returns:
        (layout_dict, total_hbm_elements)
    """
    def _compute_tensor_hbm_size(tensor, real_data_ratio, align_elems):
        """Compute HBM size for a single tensor."""
        if tensor is None:
            return 0
        num_elems = tensor.numel()
        size = int(num_elems * real_data_ratio)
        size = pad_to(size, align_elems)
        return size

    layout = {}
    cur_offset = 0

    # ========== Embedding weights ==========
    embedding_offsets = {}

    # Patch weight
    embedding_offsets["patch_weight"] = cur_offset
    patch_weight_size = _compute_tensor_hbm_size(
        embedding_weights["patch_weight"], real_data_ratio, align_elems
    )
    cur_offset += patch_weight_size

    # Patch bias
    embedding_offsets["patch_bias"] = cur_offset
    patch_bias_size = _compute_tensor_hbm_size(
        embedding_weights["patch_bias"], real_data_ratio, align_elems
    )
    cur_offset += patch_bias_size

    # Position table
    embedding_offsets["position_table"] = cur_offset
    position_size = _compute_tensor_hbm_size(
        embedding_weights["position_table"], real_data_ratio, align_elems
    )
    cur_offset += position_size

    layout["embedding"] = embedding_offsets

    # ========== Layer weights ==========
    layer_offsets = {}

    for layer_idx, layer_weights in enumerate(layer_weights_list):
        layer_offsets[layer_idx] = {}

        # LN1
        layer_offsets[layer_idx]["ln1_weight"] = cur_offset
        ln1_weight_size = _compute_tensor_hbm_size(
            layer_weights["ln1_weight"], real_data_ratio, align_elems
        )
        cur_offset += ln1_weight_size

        layer_offsets[layer_idx]["ln1_bias"] = cur_offset
        ln1_bias_size = _compute_tensor_hbm_size(
            layer_weights["ln1_bias"], real_data_ratio, align_elems
        )
        cur_offset += ln1_bias_size

        # Q, K, V projections
        for proj_name in ["q_proj", "k_proj", "v_proj"]:
            layer_offsets[layer_idx][f"{proj_name}_weight"] = cur_offset
            weight_size = _compute_tensor_hbm_size(
                layer_weights[f"{proj_name}_weight"], real_data_ratio, align_elems
            )
            cur_offset += weight_size

            layer_offsets[layer_idx][f"{proj_name}_bias"] = cur_offset
            bias_size = _compute_tensor_hbm_size(
                layer_weights[f"{proj_name}_bias"], real_data_ratio, align_elems
            )
            cur_offset += bias_size

        # Output projection
        layer_offsets[layer_idx]["out_proj_weight"] = cur_offset
        out_weight_size = _compute_tensor_hbm_size(
            layer_weights["out_proj_weight"], real_data_ratio, align_elems
        )
        cur_offset += out_weight_size

        layer_offsets[layer_idx]["out_proj_bias"] = cur_offset
        out_bias_size = _compute_tensor_hbm_size(
            layer_weights["out_proj_bias"], real_data_ratio, align_elems
        )
        cur_offset += out_bias_size

        # LN2
        layer_offsets[layer_idx]["ln2_weight"] = cur_offset
        ln2_weight_size = _compute_tensor_hbm_size(
            layer_weights["ln2_weight"], real_data_ratio, align_elems
        )
        cur_offset += ln2_weight_size

        layer_offsets[layer_idx]["ln2_bias"] = cur_offset
        ln2_bias_size = _compute_tensor_hbm_size(
            layer_weights["ln2_bias"], real_data_ratio, align_elems
        )
        cur_offset += ln2_bias_size

        # MLP weights (fc1, fc2, and optionally fc3)
        for fc_name in ["fc1", "fc2", "fc3"]:
            if f"{fc_name}_weight" in layer_weights and layer_weights[f"{fc_name}_weight"] is not None:
                layer_offsets[layer_idx][f"{fc_name}_weight"] = cur_offset
                fc_weight_size = _compute_tensor_hbm_size(
                    layer_weights[f"{fc_name}_weight"], real_data_ratio, align_elems
                )
                cur_offset += fc_weight_size

                layer_offsets[layer_idx][f"{fc_name}_bias"] = cur_offset
                fc_bias_size = _compute_tensor_hbm_size(
                    layer_weights[f"{fc_name}_bias"], real_data_ratio, align_elems
                )
                cur_offset += fc_bias_size

    layout["layers"] = layer_offsets
    layout["total_hbm_elements"] = cur_offset
    layout["total_hbm_mb"] = cur_offset * 2 / (1024 * 1024)

    return layout, cur_offset


def validate_memory_layout(
    vram_layout: dict,
    hbm_layout: dict,
    max_vram_mb: int = 512,
    max_hbm_mb: int = 1024,
) -> bool:
    """Validate that layouts fit within hardware constraints.

    Args:
        vram_layout: From compute_full_model_vram_layout()
        hbm_layout: Tuple (layout_dict, total_elems) from compute_full_model_hbm_layout()
        max_vram_mb, max_hbm_mb: Memory limits (default 512MB VRAM, 1GB HBM for 27-layer model)

    Returns:
        True if layout is valid, raises ValueError otherwise
    """
    vram_mb = vram_layout.get("total_vram_mb", 0)
    hbm_mb = hbm_layout[1] * 2 / (1024 * 1024) if isinstance(hbm_layout, tuple) else 0

    print("Memory Layout Validation:")
    print(f"  VRAM: {vram_mb:.2f} MB / {max_vram_mb} MB")
    print(f"  HBM:  {hbm_mb:.2f} MB / {max_hbm_mb} MB")

    if vram_mb > max_vram_mb:
        raise ValueError(f"VRAM overflow: {vram_mb:.2f} MB > {max_vram_mb} MB")

    if hbm_mb > max_hbm_mb:
        raise ValueError(f"HBM overflow: {hbm_mb:.2f} MB > {max_hbm_mb} MB")

    print("✓ Memory layout valid")
    return True


if __name__ == "__main__":
    """Quick test of memory layout computation."""
    from transactional_emulator.testbench.siglip.model_loader import (
        extract_embedding_weights,
        extract_layer_weights,
        load_siglip_config,
        load_siglip_vision_model,
    )

    print("=" * 80)
    print("SigLIP Memory Layout Test")
    print("=" * 80)

    # Load config and weights
    config = load_siglip_config("compiler/doc/Model_Lib/siglip-so400m-patch14-384.json")
    model = load_siglip_vision_model()
    embedding_weights = extract_embedding_weights(model, config)
    layer_weights_list = [extract_layer_weights(model, i, config["hidden_size"]) for i in range(27)]

    # Test VRAM layout
    print("\n--- VRAM Layout ---")
    vram_layout = compute_full_model_vram_layout(config)
    print(f"Embedding base: {vram_layout['embedding_base']}")
    print(f"Embedding size: {vram_layout['embedding_size']} elements")
    print(f"Num layers: {len(vram_layout['layer_bases'])}")
    print(f"Total VRAM: {vram_layout['total_vram_mb']:.2f} MB")

    # Test HBM layout
    print("\n--- HBM Layout ---")
    hbm_layout, total_hbm = compute_full_model_hbm_layout(config, embedding_weights, layer_weights_list)
    print(f"Total HBM elements: {total_hbm}")
    print(f"Total HBM: {total_hbm * 2 / (1024 * 1024):.2f} MB")

    # Validate
    print("\n--- Validation ---")
    validate_memory_layout(vram_layout, (hbm_layout, total_hbm))

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
