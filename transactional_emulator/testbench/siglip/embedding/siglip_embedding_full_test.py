from pathlib import Path

import numpy as np
import torch
from transactional_emulator.testbench.siglip.local_asm_templates.embedding_blocks import (
    append_position_add_asm,
    build_embedding_projection_asm,
)

from transactional_emulator.testbench.siglip.utils.math import (
    MXFP_REAL_DATA_RATIO,
    compute_hbm_size_aligned,
    quantize_to_mxfp,
)
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.siglip.utils.core import (
    pad_to_alignment,
    pad_to_batch_boundary,
    resolve_position_embedding,
)
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    prepare_case_artifacts,
    write_comparison_params,
)
from transactional_emulator.testbench.siglip.model_loader import resolve_siglip_model_spec

if __name__ == "__main__":
    print("=" * 80)
    print("SigLIP Full-Config Embedding Test")
    print("  patch projection -> learned position add")
    print("=" * 80)

    _config_path, model_id, variant = resolve_siglip_model_spec()
    batch_size = 1
    vlen = 64
    mlen = 64
    blen = 4
    real_data_ratio = MXFP_REAL_DATA_RATIO

    torch.manual_seed(42)

    print(f"\nResolved variant: {variant}")
    print(f"Loading {model_id} ...")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    vision_root = getattr(model, "vision_model", model)
    embeddings = vision_root.embeddings

    image_size = int(getattr(vision_root.config, "image_size", 384))
    patch_size = int(getattr(vision_root.config, "patch_size", 14))
    num_channels = int(getattr(vision_root.config, "num_channels", 3))
    hidden_size = int(getattr(vision_root.config, "hidden_size", 1152))
    num_patches = (image_size // patch_size) ** 2
    padded_num_patches = pad_to_batch_boundary(num_patches, blen)
    in_features = num_channels * patch_size * patch_size

    patch_weight = embeddings.patch_embedding.weight.detach().contiguous()
    patch_weight_2d = patch_weight.reshape(patch_weight.shape[0], -1).T.contiguous()
    position_table = resolve_position_embedding(vision_root)
    position_tensor = position_table[:num_patches, :hidden_size].contiguous()

    print(
        f"\nTrue config: image_size={image_size}, patch_size={patch_size}, num_channels={num_channels}, "
        f"hidden_size={hidden_size}, num_patches={num_patches}"
    )
    print(f"Using full patch coverage with {num_patches} patches from the real position table.")
    print(f"Patch weight shape: {patch_weight.shape}")
    print(f"Position table shape: {position_table.shape}")

    patches = torch.randn(num_patches, num_channels, patch_size, patch_size, dtype=torch.bfloat16)
    act_tensor = patches.reshape(num_patches, in_features)
    weights_tensor = patch_weight_2d

    if padded_num_patches != num_patches:
        act_tensor = torch.nn.functional.pad(act_tensor, (0, 0, 0, padded_num_patches - num_patches))
        position_tensor = torch.nn.functional.pad(position_tensor, (0, 0, 0, padded_num_patches - num_patches))

    aligned_in_features = pad_to_alignment(in_features, mlen)
    if aligned_in_features != in_features:
        act_tensor = torch.nn.functional.pad(act_tensor, (0, aligned_in_features - in_features))
        weights_tensor = torch.nn.functional.pad(weights_tensor, (0, 0, 0, aligned_in_features - in_features))
        in_features = aligned_in_features

    act_mxfp = quantize_to_mxfp(act_tensor).to(torch.bfloat16)
    weights_mxfp = quantize_to_mxfp(weights_tensor).to(torch.bfloat16)
    position_mxfp = quantize_to_mxfp(position_tensor).to(torch.bfloat16)

    projection_golden = torch.mm(act_mxfp, weights_mxfp)
    final_golden = projection_golden + position_mxfp

    if padded_num_patches != num_patches:
        projection_golden = projection_golden[:num_patches]
        final_golden = final_golden[:num_patches]

    print(
        f"\nProjection: ({padded_num_patches}, {in_features}) @ ({in_features}, {hidden_size}) -> "
        f"({padded_num_patches}, {hidden_size})"
    )
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

    act_hbm_size = compute_hbm_size_aligned(in_features * padded_num_patches)
    weight_hbm_offset = act_hbm_size
    weight_hbm_end = compute_hbm_size_aligned(in_features * padded_num_patches + in_features * hidden_size)

    gen_assembly_code, result_vram_offset = build_embedding_projection_asm(
        title="SigLIP So400M Patch14-384 Full-Config Embedding Test",
        shape_batch=num_patches,
        in_features=in_features,
        out_features=hidden_size,
        effective_batch=padded_num_patches,
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        weight_hbm_offset=weight_hbm_offset,
        weight_hbm_end=weight_hbm_end,
    )

    position_vram_offset = result_vram_offset + padded_num_patches * hidden_size

    gen_assembly_code = append_position_add_asm(
        gen_assembly_code=gen_assembly_code,
        result_vram_offset=result_vram_offset,
        position_vram_offset=position_vram_offset,
        batch=padded_num_patches,
        out_features=hidden_size,
        vlen=vlen,
    )

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Calculate required HBM size: act_tensor + weights + position_tensor (all MXFP8 quantized)
    act_tensor_size_mb = int(np.ceil((in_features * num_patches * real_data_ratio) / (1024 * 1024)))
    weight_tensor_size_mb = int(np.ceil((in_features * hidden_size * real_data_ratio) / (1024 * 1024)))
    position_tensor_size_mb = int(np.ceil((num_patches * hidden_size * real_data_ratio) / (1024 * 1024)))
    total_hbm_size_mb = act_tensor_size_mb + weight_tensor_size_mb + position_tensor_size_mb + 10  # +10 for margin

    print("\nHBM Memory Allocation:")
    print(f"  Act tensor: {act_tensor_size_mb} MB ({in_features} × {num_patches})")
    print(f"  Weight tensor: {weight_tensor_size_mb} MB ({in_features} × {hidden_size})")
    print(f"  Position tensor: {position_tensor_size_mb} MB ({num_patches} × {hidden_size})")
    print(f"  Total HBM size: {total_hbm_size_mb} MB")

    prepare_case_artifacts(
        case_build_dir=build_dir,
        input_tensor=input_tensor,
        asm_code=gen_assembly_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=None,
        hbm_mb=total_hbm_size_mb,
        data_order=["act_tensor", "weights", "position_tensor"],
    )

    result_start_row = result_vram_offset // vlen
    num_result_rows = (padded_num_patches * hidden_size) // vlen
    comparison_params = write_comparison_params(
        build_dir,
        start_row_idx=result_start_row,
        num_rows=num_result_rows,
        num_batches=padded_num_patches,
        elements_per_batch=hidden_size,
        use_stride_mode=True,
    )
    print("================================================")
    print("Finished generating full-config SigLIP embedding test")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("================================================")

    run_and_assert(build_dir, "siglip_embedding_full", mlen=mlen, blen=blen)
