"""Runtime model repacking and VRAM layout prep for full-model SigLIP harness."""

import torch

from transactional_emulator.testbench.siglip.full_model.memory_layout import compute_runtime_vram_layout
from transactional_emulator.testbench.siglip.utils.core import align_up
from transactional_emulator.testbench.siglip.utils.siglip_tensors import (
    build_runtime_hidden_positions,
    _scatter_hidden_vector,
    _scatter_hidden_square_matrix,
    _scatter_seq_hidden_tensor,
    _scatter_hidden_to_inter_matrix,
    _scatter_inter_to_hidden_matrix,
)

__all__ = [
    "build_hidden_index_map",
    "build_runtime_repacked_model",
    "infer_source_num_heads",
    "prepare_runtime_model_and_vram_layout",
]


def infer_source_num_heads(source_hidden: int, target_heads: int) -> int:
    """Infer source checkpoint head count for head-aware hidden remapping."""
    # SigLIP So400m checkpoints use 16 heads. Prefer this when compatible.
    if source_hidden % 16 == 0 and target_heads <= 16:
        return 16
    if source_hidden % target_heads == 0:
        return target_heads
    for h in range(min(source_hidden, 64), 0, -1):
        if source_hidden % h == 0 and h >= target_heads:
            return h
    return 1


def build_hidden_index_map(source_hidden: int, target_hidden: int, target_heads: int) -> torch.Tensor:
    """Build source->target hidden index map preserving head-block locality."""
    source_heads = infer_source_num_heads(source_hidden, target_heads)
    source_head_dim = source_hidden // source_heads
    target_head_dim = target_hidden // target_heads

    heads_to_copy = min(source_heads, target_heads)
    dims_to_copy = min(source_head_dim, target_head_dim)

    indices: list[int] = []
    for h in range(heads_to_copy):
        base = h * source_head_dim
        for d in range(dims_to_copy):
            indices.append(base + d)

    # Fallback: if target_hidden is still not covered, append remaining indices sequentially.
    if len(indices) < target_hidden:
        selected = set(indices)
        for i in range(source_hidden):
            if i in selected:
                continue
            indices.append(i)
            if len(indices) >= target_hidden:
                break

    return torch.tensor(indices[:target_hidden], dtype=torch.long)


def build_runtime_repacked_model(
    config: dict,
    embedding_weights: dict,
    layer_weights_list: list,
    mlen: int,
    seq_len_kernel: int,
    final_ln_weights: dict | None = None,
) -> tuple[dict, dict, list[dict]]:
    """Build runtime-expanded tensors matching flash-attn padded head geometry."""
    runtime_config = dict(config)

    hidden = int(config["hidden_size"])
    inter = int(config["intermediate_size"])
    hq = int(config["num_attention_heads"])
    head_dim = hidden // hq
    d_padded = align_up(head_dim, mlen)
    hidden_runtime = hq * d_padded
    inter_padded = align_up(inter, mlen)

    runtime_config["hidden_size"] = hidden_runtime
    runtime_config["intermediate_size"] = inter_padded
    runtime_config["seq_len_valid"] = int((config["image_size"] // config["patch_size"]) ** 2)

    patch_size = int(config["patch_size"])
    num_channels = int(config["num_channels"])
    in_features = num_channels * patch_size * patch_size
    aligned_in_features = align_up(in_features, mlen)

    source_hidden = int(embedding_weights["position_table"].shape[1])
    # For embeddings, use direct sequential indexing to preserve all information.
    # Embeddings have no head structure, so head-aware selection is unnecessary.
    embedding_hidden_index = torch.arange(min(hidden, source_hidden), dtype=torch.long)
    runtime_hidden_positions = build_runtime_hidden_positions(hidden, hq, d_padded)

    patch_weight = embedding_weights["patch_weight"].detach().float()
    patch_weight_visible = patch_weight[:, embedding_hidden_index]
    patch_weight_runtime = torch.zeros(aligned_in_features, hidden_runtime, dtype=torch.float32)
    patch_weight_runtime[:patch_weight_visible.shape[0], :] = _scatter_seq_hidden_tensor(
        patch_weight_visible, hidden_runtime, runtime_hidden_positions
    ).contiguous()

    patch_bias_src = embedding_weights.get("patch_bias")
    patch_bias_runtime = torch.zeros(hidden_runtime, dtype=torch.float32)
    if patch_bias_src is not None:
        patch_bias_f = patch_bias_src.detach().float()
        patch_bias_visible = patch_bias_f[embedding_hidden_index]
        patch_bias_runtime = _scatter_hidden_vector(
            patch_bias_visible, hidden_runtime, runtime_hidden_positions
        )

    position_src = embedding_weights["position_table"].detach().float()
    position_visible = position_src[:, embedding_hidden_index]
    position_runtime = _scatter_seq_hidden_tensor(
        position_visible, hidden_runtime, runtime_hidden_positions
    )

    runtime_embedding = {
        "patch_weight": patch_weight_runtime,
        "patch_bias": patch_bias_runtime,
        "position_table": position_runtime,
    }

    # For layer weights (attention/MLP), use head-aware mapping to preserve head locality.
    hidden_index = build_hidden_index_map(source_hidden, hidden, hq)

    final_ln_weight_runtime = None
    final_ln_bias_runtime = None
    if final_ln_weights is not None:
        ln_w_src = final_ln_weights.get("ln_weight")
        if ln_w_src is not None:
            ln_w_f = ln_w_src.detach().float()
            ln_w_visible = ln_w_f[hidden_index[:hidden]]
            final_ln_weight_runtime = _scatter_hidden_vector(
                ln_w_visible, hidden_runtime, runtime_hidden_positions
            )

        ln_b_src = final_ln_weights.get("ln_bias")
        if ln_b_src is not None:
            ln_b_f = ln_b_src.detach().float()
            ln_b_visible = ln_b_f[hidden_index[:hidden]]
            final_ln_bias_runtime = _scatter_hidden_vector(
                ln_b_visible, hidden_runtime, runtime_hidden_positions
            )

    runtime_embedding["final_ln_weight"] = final_ln_weight_runtime
    runtime_embedding["final_ln_bias"] = final_ln_bias_runtime

    runtime_layers: list[dict] = []
    for src in layer_weights_list:
        layer_rt: dict[str, torch.Tensor | None] = {}

        ln1_src = src["ln1_weight"].detach().float()
        ln1_visible = ln1_src[hidden_index[:hidden]]
        layer_rt["ln1_weight"] = _scatter_hidden_vector(
            ln1_visible, hidden_runtime, runtime_hidden_positions
        )
        ln1_b = src.get("ln1_bias")
        if ln1_b is not None:
            ln1_b_src = ln1_b.detach().float()
            ln1_b_visible = ln1_b_src[hidden_index[:hidden]]
            layer_rt["ln1_bias"] = _scatter_hidden_vector(
                ln1_b_visible, hidden_runtime, runtime_hidden_positions
            )
        else:
            layer_rt["ln1_bias"] = None

        wq_src = src["q_proj_weight"].detach().float()
        wq_visible = wq_src[hidden_index[:hidden]][:, hidden_index[:hidden]].T
        layer_rt["q_proj_weight"] = _scatter_hidden_square_matrix(
            wq_visible, hidden_runtime, runtime_hidden_positions
        )

        q_bias = src.get("q_proj_bias")
        if q_bias is not None:
            q_bias_src = q_bias.detach().float()
            q_bias_visible = q_bias_src[hidden_index[:hidden]]
            layer_rt["q_proj_bias"] = _scatter_hidden_vector(
                q_bias_visible, hidden_runtime, runtime_hidden_positions
            )
        else:
            layer_rt["q_proj_bias"] = None

        wk_src = src["k_proj_weight"].detach().float()
        wk_visible = wk_src[hidden_index[:hidden]][:, hidden_index[:hidden]].T
        layer_rt["k_proj_weight"] = _scatter_hidden_square_matrix(
            wk_visible, hidden_runtime, runtime_hidden_positions
        )

        wv_src = src["v_proj_weight"].detach().float()
        wv_visible = wv_src[hidden_index[:hidden]][:, hidden_index[:hidden]].T
        layer_rt["v_proj_weight"] = _scatter_hidden_square_matrix(
            wv_visible, hidden_runtime, runtime_hidden_positions
        )

        k_bias = src.get("k_proj_bias")
        if k_bias is not None:
            k_bias_src = k_bias.detach().float()
            k_bias_visible = k_bias_src[hidden_index[:hidden]]
            layer_rt["k_proj_bias"] = _scatter_hidden_vector(
                k_bias_visible, hidden_runtime, runtime_hidden_positions
            )
        else:
            layer_rt["k_proj_bias"] = None

        v_bias = src.get("v_proj_bias")
        if v_bias is not None:
            v_bias_src = v_bias.detach().float()
            v_bias_visible = v_bias_src[hidden_index[:hidden]]
            layer_rt["v_proj_bias"] = _scatter_hidden_vector(
                v_bias_visible, hidden_runtime, runtime_hidden_positions
            )
        else:
            layer_rt["v_proj_bias"] = None

        out_w = src.get("out_proj_weight")
        if out_w is not None:
            out_w_src = out_w.detach().float()
            out_w_visible = out_w_src[hidden_index[:hidden]][:, hidden_index[:hidden]].T
            layer_rt["out_proj_weight"] = _scatter_hidden_square_matrix(
                out_w_visible, hidden_runtime, runtime_hidden_positions
            )
        else:
            layer_rt["out_proj_weight"] = None

        out_b = src.get("out_proj_bias")
        if out_b is not None:
            out_b_src = out_b.detach().float()
            out_b_visible = out_b_src[hidden_index[:hidden]]
            layer_rt["out_proj_bias"] = _scatter_hidden_vector(
                out_b_visible, hidden_runtime, runtime_hidden_positions
            )
        else:
            layer_rt["out_proj_bias"] = None

        ln2_src = src["ln2_weight"].detach().float()
        ln2_visible = ln2_src[hidden_index[:hidden]]
        layer_rt["ln2_weight"] = _scatter_hidden_vector(
            ln2_visible, hidden_runtime, runtime_hidden_positions
        )
        ln2_b = src.get("ln2_bias")
        if ln2_b is not None:
            ln2_b_src = ln2_b.detach().float()
            ln2_b_visible = ln2_b_src[hidden_index[:hidden]]
            layer_rt["ln2_bias"] = _scatter_hidden_vector(
                ln2_b_visible, hidden_runtime, runtime_hidden_positions
            )
        else:
            layer_rt["ln2_bias"] = None

        fc1_src = src["fc1_weight"].detach().float().index_select(1, hidden_index[:hidden])[:inter, :].T
        fc1_weight = torch.zeros(hidden_runtime, inter_padded, dtype=torch.float32)
        fc1_weight[:, :inter] = _scatter_hidden_to_inter_matrix(
            fc1_src, hidden_runtime, runtime_hidden_positions
        )
        layer_rt["fc1_weight"] = fc1_weight
        fc1_b_src = src.get("fc1_bias")
        if fc1_b_src is not None:
            fc1_b_padded = torch.zeros(inter_padded, dtype=torch.float32)
            fc1_b_padded[:min(inter, fc1_b_src.numel())] = fc1_b_src.detach().float()[:inter]
            layer_rt["fc1_bias"] = fc1_b_padded
        else:
            layer_rt["fc1_bias"] = None

        fc2_src = src["fc2_weight"].detach().float().index_select(0, hidden_index[:hidden])[:hidden, :inter].T
        fc2_weight = torch.zeros(inter_padded, hidden_runtime, dtype=torch.float32)
        fc2_weight[:inter, :] = _scatter_inter_to_hidden_matrix(
            fc2_src, hidden_runtime, runtime_hidden_positions
        )
        layer_rt["fc2_weight"] = fc2_weight
        fc2_b_src = src.get("fc2_bias")
        if fc2_b_src is not None:
            fc2_b_visible = fc2_b_src.detach().float()[hidden_index[:hidden]]
            layer_rt["fc2_bias"] = _scatter_hidden_vector(
                fc2_b_visible, hidden_runtime, runtime_hidden_positions
            )
        else:
            layer_rt["fc2_bias"] = None

        runtime_layers.append(layer_rt)

    return runtime_config, runtime_embedding, runtime_layers


def prepare_runtime_model_and_vram_layout(
    *,
    config: dict,
    embedding_weights: dict,
    layer_weights_list: list,
    mlen: int,
    layout_layers: int,
    final_ln_weights: dict | None = None,
    include_out_proj_bias_buffers: bool = False,
    include_mlp_bias_buffers: bool = False,
) -> tuple[int, int, dict, dict, list[dict], dict]:
    """Build runtime-repacked tensors and the matching VRAM layout."""
    seq_len_valid = int((config["image_size"] // config["patch_size"]) ** 2)
    seq_len_kernel = align_up(seq_len_valid, mlen)

    runtime_config, runtime_embed_weights, runtime_layer_weights = build_runtime_repacked_model(
        config=config,
        embedding_weights=embedding_weights,
        layer_weights_list=layer_weights_list,
        mlen=mlen,
        seq_len_kernel=seq_len_kernel,
        final_ln_weights=final_ln_weights,
    )
    runtime_config["seq_len_valid"] = seq_len_valid

    vram_layout = compute_runtime_vram_layout(
        runtime_config,
        seq_len_kernel=seq_len_kernel,
        max_layers=layout_layers,
        mlen=mlen,
        include_out_proj_bias_buffers=include_out_proj_bias_buffers,
        include_mlp_bias_buffers=include_mlp_bias_buffers,
        include_final_post_layernorm=bool(config.get("apply_post_layernorm", False)),
    )

    return (
        seq_len_valid,
        seq_len_kernel,
        runtime_config,
        runtime_embed_weights,
        runtime_layer_weights,
        vram_layout,
    )
