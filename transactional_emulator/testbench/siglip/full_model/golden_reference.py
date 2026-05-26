"""Hardware-faithful golden reference computation for SigLIP full model.

Computes embedding and per-layer forward pass using MXFP quantization to
match hardware behavior closely.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.mlp.siglip_mlp_test import quantize_to_mxfp
from transactional_emulator.testbench.siglip.utils.math import gqa_sdpa


def _infer_source_num_heads(source_hidden: int, target_heads: int) -> int:
    if source_hidden % 16 == 0 and target_heads <= 16:
        return 16
    if source_hidden % target_heads == 0:
        return target_heads
    for h in range(min(source_hidden, 64), 0, -1):
        if source_hidden % h == 0 and h >= target_heads:
            return h
    return 1


def _build_hidden_index_map(source_hidden: int, target_hidden: int, target_heads: int) -> torch.Tensor:
    source_heads = _infer_source_num_heads(source_hidden, target_heads)
    source_head_dim = source_hidden // source_heads
    target_head_dim = target_hidden // target_heads

    heads_to_copy = min(source_heads, target_heads)
    dims_to_copy = min(source_head_dim, target_head_dim)

    indices: list[int] = []
    for h in range(heads_to_copy):
        base = h * source_head_dim
        for d in range(dims_to_copy):
            indices.append(base + d)

    if len(indices) < target_hidden:
        selected = set(indices)
        for i in range(source_hidden):
            if i in selected:
                continue
            indices.append(i)
            if len(indices) >= target_hidden:
                break

    return torch.tensor(indices[:target_hidden], dtype=torch.long)


def _gelu_hardware_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Hardware GELU approximation used by gelu_asm: x * sigmoid(1.702 * x)."""
    return x * torch.sigmoid(1.702 * x)


def compute_golden_embedding(
    patches: torch.Tensor,
    embedding_weights: dict,
    use_mxfp: bool = True,
) -> torch.Tensor:
    """Compute golden output for embedding stage.

    Args:
        patches: [num_patches, C*K*K]
        embedding_weights: Dict with patch projection and position table
        use_mxfp: Whether to apply MXFP quantization at hardware-visible boundaries

    Returns:
        [num_patches, hidden_size] embedding output
    """
    patch_weight = embedding_weights["patch_weight"].detach().float()  # [in_features, hidden]
    patch_bias = embedding_weights.get("patch_bias")
    patch_bias_f = patch_bias.detach().float() if patch_bias is not None else None

    patches_f = patches.float()
    in_features = int(patch_weight.shape[0])
    if patches_f.shape[1] < in_features:
        patches_f = F.pad(patches_f, (0, in_features - patches_f.shape[1]))
    elif patches_f.shape[1] > in_features:
        patches_f = patches_f[:, :in_features]

    proj = F.linear(patches_f, patch_weight.T.contiguous(), patch_bias_f)
    if use_mxfp:
        proj = quantize_to_mxfp(proj).to(torch.bfloat16).float()
    else:
        proj = proj.to(torch.bfloat16).float()

    hidden_size = int(proj.shape[1])
    position_table = embedding_weights["position_table"].detach().float()
    pos = torch.zeros(proj.shape[0], hidden_size, dtype=torch.float32)
    valid_hidden = min(hidden_size, position_table.shape[1])
    pos[:, :valid_hidden] = position_table[: proj.shape[0], :valid_hidden]
    if use_mxfp:
        pos = quantize_to_mxfp(pos).to(torch.bfloat16).float()
    else:
        pos = pos.to(torch.bfloat16).float()

    return (proj + pos).to(torch.bfloat16).float()


def compute_golden_layer(
    x: torch.Tensor,
    layer_weights: dict,
    config: dict,
    use_mxfp: bool = True,
) -> torch.Tensor:
    """Compute golden output for one encoder layer (hardware-faithful)."""
    seq_len, hidden_size = x.shape
    inter_size = int(config["intermediate_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    head_dim = hidden_size // num_heads

    eps = float(config.get("layer_norm_eps", 1e-2))
    scale = 1.0 / math.sqrt(head_dim)

    mlen = 64
    hidden_padded = ((hidden_size + mlen - 1) // mlen) * mlen

    x_padded = torch.zeros(seq_len, hidden_padded, dtype=torch.float32)
    x_padded[:, :hidden_size] = x.float()

    source_hidden = int(layer_weights["q_proj_weight"].shape[1])
    if source_hidden == hidden_size:
        hidden_index = torch.arange(hidden_size, dtype=torch.long)
    else:
        hidden_index = _build_hidden_index_map(source_hidden, hidden_size, num_heads)

    ln1_weight = layer_weights.get("ln1_weight")
    ln1_bias = layer_weights.get("ln1_bias")
    if ln1_weight is not None:
        ln1_weight_padded = torch.ones(hidden_padded, dtype=torch.float32)
        ln1_weight_f = ln1_weight.detach().float().index_select(0, hidden_index)
        ln1_weight_padded[:hidden_size] = ln1_weight_f
    else:
        ln1_weight_padded = None
    if ln1_bias is not None:
        ln1_bias_padded = torch.zeros(hidden_padded, dtype=torch.float32)
        ln1_bias_f = ln1_bias.detach().float().index_select(0, hidden_index)
        ln1_bias_padded[:hidden_size] = ln1_bias_f
    else:
        ln1_bias_padded = None

    x_ln1 = F.layer_norm(
        x_padded,
        (hidden_padded,),
        weight=ln1_weight_padded,
        bias=ln1_bias_padded,
        eps=eps,
    ).to(torch.bfloat16).float()

    wq = (
        layer_weights["q_proj_weight"]
        .detach()
        .float()
        .index_select(0, hidden_index)
        .index_select(1, hidden_index)
    )
    bq = layer_weights.get("q_proj_bias")
    bq_f = bq.detach().float().index_select(0, hidden_index) if bq is not None else None
    q_out = F.linear(x_ln1[:, :hidden_size], wq, bq_f).float()
    q_out = q_out.reshape(1, seq_len, num_heads, head_dim).to(torch.bfloat16).float()

    wk = (
        layer_weights["k_proj_weight"]
        .detach()
        .float()
        .index_select(0, hidden_index)
        .index_select(1, hidden_index)
    )
    bk = layer_weights.get("k_proj_bias")
    bk_f = bk.detach().float().index_select(0, hidden_index) if bk is not None else None
    k_out = F.linear(x_ln1[:, :hidden_size], wk, bk_f).float()
    k_out = k_out.reshape(1, seq_len, num_kv_heads, head_dim).to(torch.bfloat16).float()

    wv = (
        layer_weights["v_proj_weight"]
        .detach()
        .float()
        .index_select(0, hidden_index)
        .index_select(1, hidden_index)
    )
    bv = layer_weights.get("v_proj_bias")
    bv_f = bv.detach().float().index_select(0, hidden_index) if bv is not None else None
    v_out = F.linear(x_ln1[:, :hidden_size], wv, bv_f).float()
    v_out = v_out.reshape(1, seq_len, num_kv_heads, head_dim).to(torch.bfloat16).float()

    attn_out = gqa_sdpa(
        q_out,
        k_out,
        v_out,
        scale=scale,
        hq=num_heads,
        hkv=num_kv_heads,
        kv_valid_len=seq_len,
    )
    attn_out = attn_out.reshape(seq_len, hidden_size).to(torch.bfloat16).float()

    wout = (
        layer_weights["out_proj_weight"]
        .detach()
        .float()
        .index_select(0, hidden_index)
        .index_select(1, hidden_index)
    )
    bout = layer_weights.get("out_proj_bias")
    bout_f = bout.detach().float().index_select(0, hidden_index) if bout is not None else None
    attn_final = F.linear(attn_out, wout, bout_f).float().to(torch.bfloat16).float()

    x_res1 = (x_padded[:, :hidden_size].to(torch.bfloat16) + attn_final).to(torch.bfloat16).float()

    x_res1_padded = torch.zeros(seq_len, hidden_padded, dtype=torch.float32)
    x_res1_padded[:, :hidden_size] = x_res1

    ln2_weight = layer_weights.get("ln2_weight")
    ln2_bias = layer_weights.get("ln2_bias")
    if ln2_weight is not None:
        ln2_weight_padded = torch.ones(hidden_padded, dtype=torch.float32)
        ln2_weight_f = ln2_weight.detach().float().index_select(0, hidden_index)
        ln2_weight_padded[:hidden_size] = ln2_weight_f
    else:
        ln2_weight_padded = None
    if ln2_bias is not None:
        ln2_bias_padded = torch.zeros(hidden_padded, dtype=torch.float32)
        ln2_bias_f = ln2_bias.detach().float().index_select(0, hidden_index)
        ln2_bias_padded[:hidden_size] = ln2_bias_f
    else:
        ln2_bias_padded = None

    x_ln2 = F.layer_norm(
        x_res1_padded,
        (hidden_padded,),
        weight=ln2_weight_padded,
        bias=ln2_bias_padded,
        eps=eps,
    ).to(torch.bfloat16).float()

    wfc1_src = layer_weights["fc1_weight"].detach().float()
    if wfc1_src.shape[1] >= hidden_index.numel():
        # Source layout: [inter, hidden]
        wfc1_full = wfc1_src.index_select(1, hidden_index)
    elif wfc1_src.shape[0] >= hidden_index.numel():
        # Runtime-repacked layout: [hidden_runtime, inter]
        wfc1_full = wfc1_src.index_select(0, hidden_index).T
    else:
        raise ValueError(
            f"Unsupported fc1_weight shape {tuple(wfc1_src.shape)} for hidden index size {hidden_index.numel()}"
        )

    wfc2_src = layer_weights["fc2_weight"].detach().float()
    if wfc2_src.shape[0] >= hidden_index.numel():
        # Source layout: [hidden, inter]
        wfc2_full = wfc2_src.index_select(0, hidden_index)
    elif wfc2_src.shape[1] >= hidden_index.numel():
        # Runtime-repacked layout: [inter, hidden_runtime]
        wfc2_full = wfc2_src.index_select(1, hidden_index).T
    else:
        raise ValueError(
            f"Unsupported fc2_weight shape {tuple(wfc2_src.shape)} for hidden index size {hidden_index.numel()}"
        )
    inter_size_effective = min(inter_size, wfc1_full.shape[0], wfc2_full.shape[1])
    inter_padded_effective = ((inter_size_effective + mlen - 1) // mlen) * mlen
    wfc1 = wfc1_full[:inter_size_effective, :]
    bfc1 = layer_weights.get("fc1_bias")
    bfc1_f = bfc1.detach().float()[:inter_size_effective] if bfc1 is not None else None
    fc1_out = F.linear(x_ln2[:, :hidden_size], wfc1, bfc1_f).float()

    if use_mxfp:
        fc1_out_q = quantize_to_mxfp(fc1_out).to(torch.bfloat16).float()
    else:
        fc1_out_q = fc1_out.to(torch.bfloat16).float()

    mlp_activated = _gelu_hardware_sigmoid(fc1_out_q).float()
    if use_mxfp:
        mlp_activated_q = quantize_to_mxfp(mlp_activated).to(torch.bfloat16).float()
    else:
        mlp_activated_q = mlp_activated.to(torch.bfloat16).float()

    wfc2 = wfc2_full[:, :inter_size_effective]
    bfc2 = layer_weights.get("fc2_bias")
    bfc2_f = bfc2.detach().float().index_select(0, hidden_index) if bfc2 is not None else None

    mlp_activated_padded = torch.zeros(seq_len, inter_padded_effective, dtype=torch.float32)
    mlp_activated_padded[:, :inter_size_effective] = mlp_activated_q

    wfc2_padded = torch.zeros(inter_padded_effective, hidden_size, dtype=torch.float32)
    wfc2_padded[:inter_size_effective, :] = wfc2.T

    fc2_out = F.linear(mlp_activated_padded, wfc2_padded.T, bfc2_f).float()
    if use_mxfp:
        fc2_out_q = quantize_to_mxfp(fc2_out).to(torch.bfloat16).float()
    else:
        fc2_out_q = fc2_out.to(torch.bfloat16).float()

    x_final = (x_res1.to(torch.bfloat16) + fc2_out_q[:, :hidden_size].to(torch.bfloat16)).to(torch.bfloat16).float()
    return x_final


def compute_golden_full_model(
    patches: torch.Tensor,
    embedding_weights: dict,
    layer_weights_list: list,
    config: dict,
    use_mxfp: bool = True,
    max_layers: int = 27,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute golden output for full model (embedding + encoder layers)."""
    x = compute_golden_embedding(patches, embedding_weights, use_mxfp=use_mxfp)
    layer_outputs = [x.detach()]

    num_layers_to_run = min(max_layers, len(layer_weights_list))
    for layer_idx in range(num_layers_to_run):
        x = compute_golden_layer(
            x,
            layer_weights_list[layer_idx],
            config,
            use_mxfp=use_mxfp,
        )
        layer_outputs.append(x.detach())

    return x, layer_outputs


if __name__ == "__main__":
    """Quick test of golden reference computation."""
    from transactional_emulator.testbench.siglip.model_loader import (
        extract_embedding_weights,
        extract_layer_weights,
        load_siglip_config,
        load_siglip_vision_model,
    )

    print("=" * 80)
    print("SigLIP Golden Reference Test")
    print("=" * 80)

    config = load_siglip_config("compiler/doc/Model_Lib/siglip-so400m-patch14-384.json")
    model = load_siglip_vision_model()
    embed_weights = extract_embedding_weights(model, config)
    layer_weights = [extract_layer_weights(model, i) for i in range(2)]

    image_size = config["image_size"]
    patch_size = config["patch_size"]
    num_patches = (image_size // patch_size) ** 2
    in_features = 3 * patch_size * patch_size

    torch.manual_seed(42)
    patches = torch.randn(num_patches, in_features, dtype=torch.bfloat16)

    print("\n--- Testing Embedding ---")
    embedding = compute_golden_embedding(patches, embed_weights)
    print(f"Embedding output shape: {embedding.shape}")
    print(f"Embedding output range: [{embedding.min():.4f}, {embedding.max():.4f}]")

    print("\n--- Testing Layer 0 ---")
    layer0_out = compute_golden_layer(embedding, layer_weights[0], config)
    print(f"Layer 0 output shape: {layer0_out.shape}")
    print(f"Layer 0 output range: [{layer0_out.min():.4f}, {layer0_out.max():.4f}]")

    print("\n--- Testing Full Model (2 layers) ---")
    final_out, all_outputs = compute_golden_full_model(
        patches, embed_weights, layer_weights, config, use_mxfp=True, max_layers=2
    )
    print(f"Number of outputs (embed + 2 layers): {len(all_outputs)}")
    print(f"Final output shape: {final_out.shape}")
    print(f"Final output range: [{final_out.min():.4f}, {final_out.max():.4f}]")

    print("\n" + "=" * 80)
    print("Golden reference test complete")
    print("=" * 80)
