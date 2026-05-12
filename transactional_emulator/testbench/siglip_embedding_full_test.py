import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware
from compiler.asm_templates import (
    elementwise_add_vram_asm,
    preload_act_asm,
    preload_addr_reg_asm,
    projection_asm,
    reset_reg_asm,
)
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


def quantize_to_mxfp(tensor):
    """Quantize tensor to MXFP format matching hardware."""
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor,
        width=8,
        exponent_width=4,
        exponent_bias_width=8,
        block_size=[8],
    )
    return bm_x.reshape(orig_shape)


def _resolve_vision_root(model):
    return getattr(model, "vision_model", model)


def _resolve_position_embedding(vision_root):
    embeddings = vision_root.embeddings
    for attr_name in ("position_embedding", "position_embeddings", "position_embed"):
        if hasattr(embeddings, attr_name):
            position_embedding = getattr(embeddings, attr_name)
            if hasattr(position_embedding, "weight"):
                return position_embedding.weight.detach()
            return position_embedding.detach()
    raise AttributeError("Could not locate SigLIP position embedding table")


if __name__ == "__main__":
    print("=" * 80)
    print("SigLIP So400M Patch14-384 Full-Config Embedding Test")
    print("  patch projection -> learned position add")
    print("=" * 80)

    model_id = "google/siglip-so400m-patch14-384"
    batch_size = 1
    slice_len = 64
    vlen = 64
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(42)

    print(f"\nLoading {model_id} ...")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    vision_root = _resolve_vision_root(model)
    embeddings = vision_root.embeddings

    image_size = int(getattr(vision_root.config, "image_size", 384))
    patch_size = int(getattr(vision_root.config, "patch_size", 14))
    num_channels = int(getattr(vision_root.config, "num_channels", 3))
    hidden_size = int(getattr(vision_root.config, "hidden_size", 1152))
    num_positions = (image_size // patch_size) ** 2
    in_features = num_channels * patch_size * patch_size

    patch_weight = embeddings.patch_embedding.weight.detach().contiguous()
    patch_weight_2d = patch_weight.reshape(patch_weight.shape[0], -1).T.contiguous()
    position_table = _resolve_position_embedding(vision_root)
    position_tensor = position_table[:slice_len, :hidden_size].contiguous()

    print(
        f"\nTrue config: image_size={image_size}, patch_size={patch_size}, num_channels={num_channels}, "
        f"hidden_size={hidden_size}, num_positions={num_positions}"
    )
    print(f"Using a hardware-friendly slice of {slice_len} tokens from the real position table.")
    print(f"Patch weight shape: {patch_weight.shape}")
    print(f"Position table shape: {position_table.shape}")

    patches = torch.randn(slice_len, num_channels, patch_size, patch_size, dtype=torch.bfloat16)
    act_tensor = patches.reshape(slice_len, in_features)
    weights_tensor = patch_weight_2d

    aligned_in_features = ((in_features + mlen - 1) // mlen) * mlen
    if aligned_in_features != in_features:
        act_tensor = torch.nn.functional.pad(act_tensor, (0, aligned_in_features - in_features))
        weights_tensor = torch.nn.functional.pad(weights_tensor, (0, 0, 0, aligned_in_features - in_features))
        in_features = aligned_in_features

    act_mxfp = quantize_to_mxfp(act_tensor).to(torch.bfloat16)
    weights_mxfp = quantize_to_mxfp(weights_tensor).to(torch.bfloat16)
    position_mxfp = quantize_to_mxfp(position_tensor).to(torch.bfloat16)

    projection_golden = torch.mm(act_mxfp, weights_mxfp)
    final_golden = projection_golden + position_mxfp

    print(f"\nProjection: ({slice_len}, {in_features}) @ ({in_features}, {hidden_size}) -> ({slice_len}, {hidden_size})")
    print(f"Projection golden shape: {projection_golden.shape}")
    print(f"Final golden shape: {final_golden.shape}")

    input_tensor = {
        "act_tensor": act_mxfp,
        "weights": weights_mxfp,
        "position_tensor": position_mxfp,
    }
    golden_result = {
        "input_tensor": input_tensor,
        "original_output": final_golden,
    }

    fp_preload = [0.0, 1e-6, 1 / in_features]

    act_hbm_size = int(in_features * slice_len * real_data_ratio)
    weight_hbm_offset = act_hbm_size
    weight_hbm_end = int((in_features * slice_len + in_features * hidden_size) * real_data_ratio)

    gen_assembly_code = "; SigLIP So400M Patch14-384 Full-Config Embedding Test\n"
    gen_assembly_code += f"; Shape: ({slice_len}, {in_features}) @ ({in_features}, {hidden_size})\n"

    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[weight_hbm_offset, weight_hbm_end],
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3])

    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=4,
        batch=slice_len,
        hidden_size=in_features,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=in_features,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4])

    result_vram_offset = in_features * slice_len

    gen_assembly_code += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=slice_len,
        hidden_size=in_features,
        out_features=hidden_size,
        alive_registers=[1, 2, 3, 4, 5, 6],
        w_base_hbm_offset_reg=1,
        activation_base_address=0,
        result_base_address=result_vram_offset,
        rope_enabled=False,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5, 6])

    position_vram_offset = result_vram_offset + slice_len * hidden_size

    gen_assembly_code += preload_act_asm(
        vlen=vlen,
        preload_len=4,
        batch=slice_len,
        hidden_size=hidden_size,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=position_vram_offset,
        activation_offset_reg=2,
        stride_size=hidden_size,
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

    num_result_vectors = (slice_len * hidden_size) // vlen
    gen_assembly_code += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=num_result_vectors,
        alive_registers=[1, 2],
        dst_base_address=result_vram_offset,
        src_base_address=position_vram_offset,
    )

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=None,
        data=None,
        specified_data_order=["act_tensor", "weights", "position_tensor"],
        build_path=build_dir,
    )

    result_start_row = result_vram_offset // vlen
    num_result_rows = (slice_len * hidden_size) // vlen
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": slice_len,
        "elements_per_batch": hidden_size,
        "use_stride_mode": hidden_size > vlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_assembly_code)

    print("================================================")
    print("Finished generating full-config SigLIP embedding test")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("================================================")
