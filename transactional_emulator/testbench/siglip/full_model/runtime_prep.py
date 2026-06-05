"""Runtime model repacking and VRAM layout prep for full-model SigLIP harness."""

import torch

from transactional_emulator.testbench.siglip.full_model.memory_layout import compute_runtime_vram_layout
from transactional_emulator.testbench.siglip.utils.core import align_up

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
    hidden_index = build_hidden_index_map(source_hidden, hidden, hq)

    patch_weight = embedding_weights["patch_weight"].detach().float()
    patch_weight_runtime = torch.zeros(aligned_in_features, hidden_runtime, dtype=torch.float32)
    copy_rows = min(patch_weight.shape[0], aligned_in_features)
    copy_hidden = min(hidden, patch_weight.shape[1], hidden_runtime, hidden_index.numel())
    patch_weight_runtime[:copy_rows, :copy_hidden] = patch_weight[:copy_rows, :].index_select(
        1, hidden_index[:copy_hidden]
    )

    patch_bias_src = embedding_weights.get("patch_bias")
    patch_bias_runtime = torch.zeros(hidden_runtime, dtype=torch.float32)
    if patch_bias_src is not None:
        patch_bias_f = patch_bias_src.detach().float()
        copy_hidden = min(hidden, patch_bias_f.numel(), hidden_runtime, hidden_index.numel())
        patch_bias_runtime[:copy_hidden] = patch_bias_f.index_select(0, hidden_index[:copy_hidden])

    position_src = embedding_weights["position_table"].detach().float()
    position_runtime = torch.zeros(seq_len_kernel, hidden_runtime, dtype=torch.float32)
    valid_rows = min(position_src.shape[0], seq_len_kernel)
    copy_hidden = min(hidden, position_src.shape[1], hidden_runtime, hidden_index.numel())
    position_runtime[:valid_rows, :copy_hidden] = position_src[:valid_rows, :].index_select(
        1, hidden_index[:copy_hidden]
    )

    runtime_embedding = {
        "patch_weight": patch_weight_runtime,
        "patch_bias": patch_bias_runtime,
        "position_table": position_runtime,
    }

    final_ln_weight_runtime = None
    final_ln_bias_runtime = None
    if final_ln_weights is not None:
        ln_w_src = final_ln_weights.get("ln_weight")
        if ln_w_src is not None:
            final_ln_weight_runtime = torch.ones(hidden_runtime, dtype=torch.float32)
            ln_w_f = ln_w_src.detach().float()
            copy_hidden = min(hidden, ln_w_f.numel(), hidden_runtime, hidden_index.numel())
            final_ln_weight_runtime[:copy_hidden] = ln_w_f.index_select(0, hidden_index[:copy_hidden])

        ln_b_src = final_ln_weights.get("ln_bias")
        if ln_b_src is not None:
            final_ln_bias_runtime = torch.zeros(hidden_runtime, dtype=torch.float32)
            ln_b_f = ln_b_src.detach().float()
            copy_hidden = min(hidden, ln_b_f.numel(), hidden_runtime, hidden_index.numel())
            final_ln_bias_runtime[:copy_hidden] = ln_b_f.index_select(0, hidden_index[:copy_hidden])

    runtime_embedding["final_ln_weight"] = final_ln_weight_runtime
    runtime_embedding["final_ln_bias"] = final_ln_bias_runtime

    runtime_layers: list[dict] = []
    for src in layer_weights_list:
        layer_rt: dict[str, torch.Tensor | None] = {}

        ln1_w = torch.zeros(hidden_runtime, dtype=torch.float32)
        ln1_src = src["ln1_weight"].detach().float()
        copy_hidden = min(hidden, ln1_src.numel(), hidden_runtime, hidden_index.numel())
        ln1_w[:copy_hidden] = ln1_src.index_select(0, hidden_index[:copy_hidden])
        layer_rt["ln1_weight"] = ln1_w
        ln1_b = src.get("ln1_bias")
        if ln1_b is not None:
            b = torch.zeros(hidden_runtime, dtype=torch.float32)
            ln1_b_src = ln1_b.detach().float()
            copy_hidden = min(hidden, ln1_b_src.numel(), hidden_runtime, hidden_index.numel())
            b[:copy_hidden] = ln1_b_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["ln1_bias"] = b
        else:
            layer_rt["ln1_bias"] = None

        wq_src = src["q_proj_weight"].detach().float()
        wq = wq_src.index_select(0, hidden_index).index_select(1, hidden_index).T
        wq_rt = torch.zeros(hidden_runtime, hidden_runtime, dtype=torch.float32)
        copy_h = min(hidden, wq.shape[0], wq.shape[1], hidden_runtime, hidden_index.numel())
        wq_rt[:copy_h, :copy_h] = wq[:copy_h, :copy_h]
        layer_rt["q_proj_weight"] = wq_rt

        q_bias = src.get("q_proj_bias")
        if q_bias is not None:
            qb = torch.zeros(hidden_runtime, dtype=torch.float32)
            q_bias_src = q_bias.detach().float()
            copy_hidden = min(hidden, q_bias_src.numel(), hidden_runtime, hidden_index.numel())
            qb[:copy_hidden] = q_bias_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["q_proj_bias"] = qb
        else:
            layer_rt["q_proj_bias"] = None

        wk_src = src["k_proj_weight"].detach().float()
        wk = wk_src.index_select(0, hidden_index).index_select(1, hidden_index).T
        wk_rt = torch.zeros(hidden_runtime, hidden_runtime, dtype=torch.float32)
        copy_h = min(hidden, wk.shape[0], wk.shape[1], hidden_runtime, hidden_index.numel())
        wk_rt[:copy_h, :copy_h] = wk[:copy_h, :copy_h]
        layer_rt["k_proj_weight"] = wk_rt

        wv_src = src["v_proj_weight"].detach().float()
        wv = wv_src.index_select(0, hidden_index).index_select(1, hidden_index).T
        wv_rt = torch.zeros(hidden_runtime, hidden_runtime, dtype=torch.float32)
        copy_h = min(hidden, wv.shape[0], wv.shape[1], hidden_runtime, hidden_index.numel())
        wv_rt[:copy_h, :copy_h] = wv[:copy_h, :copy_h]
        layer_rt["v_proj_weight"] = wv_rt

        k_bias = src.get("k_proj_bias")
        if k_bias is not None:
            kb = torch.zeros(hidden_runtime, dtype=torch.float32)
            k_bias_src = k_bias.detach().float()
            copy_hidden = min(hidden, k_bias_src.numel(), hidden_runtime, hidden_index.numel())
            kb[:copy_hidden] = k_bias_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["k_proj_bias"] = kb
        else:
            layer_rt["k_proj_bias"] = None

        v_bias = src.get("v_proj_bias")
        if v_bias is not None:
            vb = torch.zeros(hidden_runtime, dtype=torch.float32)
            v_bias_src = v_bias.detach().float()
            copy_hidden = min(hidden, v_bias_src.numel(), hidden_runtime, hidden_index.numel())
            vb[:copy_hidden] = v_bias_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["v_proj_bias"] = vb
        else:
            layer_rt["v_proj_bias"] = None

        out_w = src.get("out_proj_weight")
        if out_w is not None:
            ow = torch.zeros(hidden_runtime, hidden_runtime, dtype=torch.float32)
            out_w_src = out_w.detach().float().index_select(0, hidden_index).index_select(1, hidden_index).T
            copy_h = min(hidden, out_w_src.shape[0], out_w_src.shape[1], hidden_runtime, hidden_index.numel())
            ow[:copy_h, :copy_h] = out_w_src[:copy_h, :copy_h]
            layer_rt["out_proj_weight"] = ow
        else:
            layer_rt["out_proj_weight"] = None

        out_b = src.get("out_proj_bias")
        if out_b is not None:
            ob = torch.zeros(hidden_runtime, dtype=torch.float32)
            out_b_src = out_b.detach().float()
            copy_hidden = min(hidden, out_b_src.numel(), hidden_runtime, hidden_index.numel())
            ob[:copy_hidden] = out_b_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["out_proj_bias"] = ob
        else:
            layer_rt["out_proj_bias"] = None

        ln2_w = torch.zeros(hidden_runtime, dtype=torch.float32)
        ln2_src = src["ln2_weight"].detach().float()
        copy_hidden = min(hidden, ln2_src.numel(), hidden_runtime, hidden_index.numel())
        ln2_w[:copy_hidden] = ln2_src.index_select(0, hidden_index[:copy_hidden])
        layer_rt["ln2_weight"] = ln2_w
        ln2_b = src.get("ln2_bias")
        if ln2_b is not None:
            b = torch.zeros(hidden_runtime, dtype=torch.float32)
            ln2_b_src = ln2_b.detach().float()
            copy_hidden = min(hidden, ln2_b_src.numel(), hidden_runtime, hidden_index.numel())
            b[:copy_hidden] = ln2_b_src.index_select(0, hidden_index[:copy_hidden])
            layer_rt["ln2_bias"] = b
        else:
            layer_rt["ln2_bias"] = None

        fc1_src = src["fc1_weight"].detach().float()[:, :].index_select(1, hidden_index)[:inter, :].T
        fc1_rt = torch.zeros(hidden_runtime, inter_padded, dtype=torch.float32)
        fc1_rt[:hidden, :inter] = fc1_src
        layer_rt["fc1_weight"] = fc1_rt
        fc1_b_src = src.get("fc1_bias")
        if fc1_b_src is not None:
            b = torch.zeros(inter_padded, dtype=torch.float32)
            b[:inter] = fc1_b_src.detach().float()[:inter]
            layer_rt["fc1_bias"] = b
        else:
            layer_rt["fc1_bias"] = None

        fc2_src = src["fc2_weight"].detach().float().index_select(0, hidden_index)[:hidden, :inter].T
        fc2_rt = torch.zeros(inter_padded, hidden_runtime, dtype=torch.float32)
        fc2_rt[:inter, :hidden] = fc2_src
        layer_rt["fc2_weight"] = fc2_rt
        fc2_b_src = src.get("fc2_bias")
        if fc2_b_src is not None:
            b = torch.zeros(hidden_runtime, dtype=torch.float32)
            b[:hidden] = fc2_b_src.detach().float().index_select(0, hidden_index)
            layer_rt["fc2_bias"] = b
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
