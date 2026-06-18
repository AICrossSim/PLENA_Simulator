"""Hardware-faithful golden reference computation for SigLIP full model.

Computes embedding and per-layer forward pass using MXFP quantization to
match hardware behavior closely.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.full_model.runtime_prep import (
    build_hidden_index_map as _build_hidden_index_map,
    infer_source_num_heads as _infer_source_num_heads,
)
from transactional_emulator.testbench.siglip.mlp.siglip_mlp_test import quantize_to_mxfp
from transactional_emulator.testbench.siglip.utils.math import mha_sdpa

# Note: Set to True when comparing with model, False when comparing with hardware
# Since the current encoder layer implementation uses the sigmoid GELU approximation.
USE_GELU_TANH_APPROX = False


def _gelu_hardware_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Hardware GELU approximation used by gelu_asm: x * sigmoid(1.702 * x)."""
    return x * torch.sigmoid(1.702 * x)

def _gelu_hardware_tanh(x: torch.Tensor) -> torch.Tensor:
    """Hardware GELU approximation used by gelu_tanh_asm."""
    x_f32 = x.float()

    x2 = (x_f32 * x_f32).to(torch.bfloat16)
    x3 = (x2.float() * x_f32).to(torch.bfloat16)
    cubic_term = (0.044715 * x3.float()).to(torch.bfloat16)
    poly = (x_f32 + cubic_term.float()).to(torch.bfloat16)

    z = (0.7978845608028654 * poly.float()).to(torch.bfloat16)  # sqrt(2/pi)
    two_z = (z.float() + z.float()).to(torch.bfloat16)
    exp_2z = torch.exp(two_z.float()).to(torch.bfloat16)

    # Stable tanh reconstruction from exp(2z): tanh(z) = 1 - 2/(exp(2z)+1)
    # This avoids inf*0 when exp(2z) overflows in BF16.
    den = (exp_2z.float() + 1.0).to(torch.bfloat16)
    rec = (1.0 / den.float()).to(torch.bfloat16)
    two_rec = (rec.float() + rec.float()).to(torch.bfloat16)
    tanh_z = (1.0 - two_rec.float()).to(torch.bfloat16)

    one_plus_tanh = (1.0 + tanh_z.float()).to(torch.bfloat16)
    half_x = (0.5 * x_f32).to(torch.bfloat16)
    return (half_x.float() * one_plus_tanh.float()).to(torch.bfloat16)


def _to_bf16_visible(x: torch.Tensor) -> torch.Tensor:
    """Apply BF16-visible truncation used by on-chip intermediate tensors."""
    return x.to(torch.bfloat16).float()


def _quantize_mxfp_preserve_shape(x: torch.Tensor) -> torch.Tensor:
    """Apply MXFP quantization for tensors that may be 1D in software view."""
    if x.dim() >= 2:
        return quantize_to_mxfp(x).float()
    if x.dim() == 1:
        return quantize_to_mxfp(x.unsqueeze(0)).squeeze(0).float()
    return quantize_to_mxfp(x.reshape(1, 1)).reshape_as(x).float()


def _apply_mlp_activation(x: torch.Tensor, hidden_act: str, use_mxfp: bool) -> torch.Tensor:
    """Apply activation matching comparison mode.

    In hardware-faithful mode (use_mxfp=True), preserve the hardware GELU
    approximation. In model-faithful mode, follow the model config activation.
    """
    if use_mxfp:
        if USE_GELU_TANH_APPROX:
            return _gelu_hardware_tanh(x)
        else:
            return _gelu_hardware_sigmoid(x)

    act = hidden_act.lower()
    if act == "gelu_pytorch_tanh":
        # return _gelu_hardware_tanh(x)
        return F.gelu(x, approximate="tanh")
    if act == "gelu":
        return F.gelu(x)
    if act == "relu":
        return F.relu(x)
    raise ValueError(f"Unsupported hidden_act '{hidden_act}' for golden layer MLP")


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
    if use_mxfp:
        patch_weight = _quantize_mxfp_preserve_shape(patch_weight)
    patch_bias = embedding_weights.get("patch_bias")
    patch_bias_f = patch_bias.detach().float() if patch_bias is not None else None
    if use_mxfp and patch_bias_f is not None:
        patch_bias_f = _quantize_mxfp_preserve_shape(patch_bias_f)

    patches_f = patches.to(torch.bfloat16).float()
    in_features = int(patch_weight.shape[0])
    if patches_f.shape[1] < in_features:
        patches_f = F.pad(patches_f, (0, in_features - patches_f.shape[1]))
    elif patches_f.shape[1] > in_features:
        patches_f = patches_f[:, :in_features]

    proj = F.linear(patches_f, patch_weight.T.contiguous(), patch_bias_f)
    proj = _to_bf16_visible(proj)

    hidden_size = int(proj.shape[1])
    position_table = embedding_weights["position_table"].detach().float()
    if use_mxfp:
        position_table = _quantize_mxfp_preserve_shape(position_table)
    pos = torch.zeros(proj.shape[0], hidden_size, dtype=torch.float32)
    valid_hidden = min(hidden_size, position_table.shape[1])
    pos[:, :valid_hidden] = position_table[: proj.shape[0], :valid_hidden]
    pos = _to_bf16_visible(pos)

    return (proj + pos).to(torch.bfloat16).float()


def compute_golden_layer(
    x: torch.Tensor,
    layer_weights: dict,
    config: dict,
    use_mxfp: bool = True,
    mlen: int = 64,
) -> torch.Tensor:
    """Compute golden output for one encoder layer (hardware-faithful).

    Args:
        x: Input activations in visible [seq, hidden] shape.
        layer_weights: Per-layer weights in software/runtime view.
        config: Model/runtime config dict.
        use_mxfp: Whether to quantize hardware-visible boundaries.
        mlen: Hardware MLEN used for hidden/intermediate padding.
    """
    seq_len, hidden_size = x.shape
    inter_size = int(config["intermediate_size"])
    num_heads = int(config["num_attention_heads"])
    num_kv_heads = int(config["num_key_value_heads"])
    hidden_act = str(config.get("hidden_act", "gelu_pytorch_tanh"))
    head_dim = hidden_size // num_heads

    eps = float(config.get("layer_norm_eps", 1e-6))
    scale = 1.0 / math.sqrt(head_dim)

    hidden_padded = ((hidden_size + mlen - 1) // mlen) * mlen

    x_padded = torch.zeros(seq_len, hidden_padded, dtype=torch.float32)
    x_padded[:, :hidden_size] = x.to(torch.bfloat16).float()

    source_hidden = int(layer_weights["q_proj_weight"].shape[1])
    if source_hidden == hidden_size:
        hidden_index = torch.arange(hidden_size, dtype=torch.long)
    else:
        hidden_index = _build_hidden_index_map(source_hidden, hidden_size, num_heads)

    ln1_weight = layer_weights.get("ln1_weight")
    ln1_bias = layer_weights.get("ln1_bias")
    if ln1_weight is not None:
        ln1_weight_padded = torch.zeros(hidden_padded, dtype=torch.float32)
        ln1_weight_f = ln1_weight.detach().float().index_select(0, hidden_index)
        if use_mxfp:
            ln1_weight_f = _quantize_mxfp_preserve_shape(ln1_weight_f)
        ln1_weight_padded[:hidden_size] = ln1_weight_f
    else:
        ln1_weight_padded = None
    if ln1_bias is not None:
        ln1_bias_padded = torch.zeros(hidden_padded, dtype=torch.float32)
        ln1_bias_f = ln1_bias.detach().float().index_select(0, hidden_index)
        if use_mxfp:
            ln1_bias_f = _quantize_mxfp_preserve_shape(ln1_bias_f)
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
    if use_mxfp:
        wq = _quantize_mxfp_preserve_shape(wq)
    bq = layer_weights.get("q_proj_bias")
    bq_f = bq.detach().float().index_select(0, hidden_index) if bq is not None else None
    if use_mxfp and bq_f is not None:
        bq_f = _quantize_mxfp_preserve_shape(bq_f)
    q_out = F.linear(x_ln1[:, :hidden_size], wq, bq_f).float()
    q_out = q_out.reshape(1, seq_len, num_heads, head_dim).to(torch.bfloat16).float()

    wk = (
        layer_weights["k_proj_weight"]
        .detach()
        .float()
        .index_select(0, hidden_index)
        .index_select(1, hidden_index)
    )
    if use_mxfp:
        wk = _quantize_mxfp_preserve_shape(wk)
    bk = layer_weights.get("k_proj_bias")
    bk_f = bk.detach().float().index_select(0, hidden_index) if bk is not None else None
    if use_mxfp and bk_f is not None:
        bk_f = _quantize_mxfp_preserve_shape(bk_f)
    k_out = F.linear(x_ln1[:, :hidden_size], wk, bk_f).float()
    k_out = k_out.reshape(1, seq_len, num_kv_heads, head_dim).to(torch.bfloat16).float()

    wv = (
        layer_weights["v_proj_weight"]
        .detach()
        .float()
        .index_select(0, hidden_index)
        .index_select(1, hidden_index)
    )
    if use_mxfp:
        wv = _quantize_mxfp_preserve_shape(wv)
    bv = layer_weights.get("v_proj_bias")
    bv_f = bv.detach().float().index_select(0, hidden_index) if bv is not None else None
    if use_mxfp and bv_f is not None:
        bv_f = _quantize_mxfp_preserve_shape(bv_f)
    v_out = F.linear(x_ln1[:, :hidden_size], wv, bv_f).float()
    v_out = v_out.reshape(1, seq_len, num_kv_heads, head_dim).to(torch.bfloat16).float()

    attn_out = mha_sdpa(
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
    if use_mxfp:
        wout = _quantize_mxfp_preserve_shape(wout)
    bout = layer_weights.get("out_proj_bias")
    bout_f = bout.detach().float().index_select(0, hidden_index) if bout is not None else None
    if use_mxfp and bout_f is not None:
        bout_f = _quantize_mxfp_preserve_shape(bout_f)
    attn_final = F.linear(attn_out, wout, bout_f).float().to(torch.bfloat16).float()

    x_res1 = (x_padded[:, :hidden_size].to(torch.bfloat16) + attn_final).to(torch.bfloat16).float()

    x_res1_padded = torch.zeros(seq_len, hidden_padded, dtype=torch.float32)
    x_res1_padded[:, :hidden_size] = x_res1

    ln2_weight = layer_weights.get("ln2_weight")
    ln2_bias = layer_weights.get("ln2_bias")
    if ln2_weight is not None:
        ln2_weight_padded = torch.zeros(hidden_padded, dtype=torch.float32)
        ln2_weight_f = ln2_weight.detach().float().index_select(0, hidden_index)
        if use_mxfp:
            ln2_weight_f = _quantize_mxfp_preserve_shape(ln2_weight_f)
        ln2_weight_padded[:hidden_size] = ln2_weight_f
    else:
        ln2_weight_padded = None
    if ln2_bias is not None:
        ln2_bias_padded = torch.zeros(hidden_padded, dtype=torch.float32)
        ln2_bias_f = ln2_bias.detach().float().index_select(0, hidden_index)
        if use_mxfp:
            ln2_bias_f = _quantize_mxfp_preserve_shape(ln2_bias_f)
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
    if use_mxfp:
        wfc1 = _quantize_mxfp_preserve_shape(wfc1)
    bfc1 = layer_weights.get("fc1_bias")
    bfc1_f = bfc1.detach().float()[:inter_size_effective] if bfc1 is not None else None
    if use_mxfp and bfc1_f is not None:
        bfc1_f = _quantize_mxfp_preserve_shape(bfc1_f)
    fc1_out = F.linear(x_ln2[:, :hidden_size], wfc1, bfc1_f).float()

    fc1_out_q = _to_bf16_visible(fc1_out)

    mlp_activated = _apply_mlp_activation(fc1_out_q, hidden_act, use_mxfp).float()
    mlp_activated_q = _to_bf16_visible(mlp_activated)

    wfc2 = wfc2_full[:, :inter_size_effective]
    if use_mxfp:
        wfc2 = _quantize_mxfp_preserve_shape(wfc2)
    bfc2 = layer_weights.get("fc2_bias")
    bfc2_f = bfc2.detach().float().index_select(0, hidden_index) if bfc2 is not None else None
    if use_mxfp and bfc2_f is not None:
        bfc2_f = _quantize_mxfp_preserve_shape(bfc2_f)

    mlp_activated_padded = torch.zeros(seq_len, inter_padded_effective, dtype=torch.float32)
    mlp_activated_padded[:, :inter_size_effective] = mlp_activated_q

    wfc2_padded = torch.zeros(inter_padded_effective, hidden_size, dtype=torch.float32)
    wfc2_padded[:inter_size_effective, :] = wfc2.T

    fc2_out = F.linear(mlp_activated_padded, wfc2_padded.T, bfc2_f).float()
    fc2_out_q = _to_bf16_visible(fc2_out)

    x_final = (x_res1.to(torch.bfloat16) + fc2_out_q[:, :hidden_size].to(torch.bfloat16)).to(torch.bfloat16).float()
    return x_final


def compute_golden_full_model(
    patches: torch.Tensor,
    embedding_weights: dict,
    layer_weights_list: list,
    config: dict,
    use_mxfp: bool = True,
    max_layers: int = 27,
    mlen: int = 64,
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
            mlen=mlen,
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
        resolve_siglip_model_spec,
    )

    print("=" * 80)
    print("SigLIP Golden Reference Test")
    print("=" * 80)

    config_path, model_id, variant = resolve_siglip_model_spec()
    print(f"Variant: {variant}")
    print(f"Config: {config_path}")
    print(f"Model ID: {model_id}")
    config = load_siglip_config(config_path)
    model = load_siglip_vision_model(model_id=model_id)
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
