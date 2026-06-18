from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import dataclass
from transactional_emulator.testbench.siglip.local_asm_templates.embedding_blocks import (
    append_position_add_asm,
    build_embedding_projection_asm,
)

from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    prepare_case_artifacts,
    write_comparison_params,
)
from transactional_emulator.testbench.siglip.utils.math import (
    MXFP_REAL_DATA_RATIO,
    quantize_to_mxfp,
)


@dataclass
class SiglipVisionConfig:
    hidden_size: int = 128
    image_size: int = 16
    patch_size: int = 8
    num_channels: int = 1


def extract_non_overlapping_patches(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Return flattened non-overlapping patches as [batch*num_patches, in_features]."""
    batch, channels, _, _ = pixel_values.shape
    patches = pixel_values.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch, channels, -1, patch_size * patch_size)
    patches = patches.transpose(1, 2).flatten(2)
    return patches.view(batch * patches.shape[1], -1)


def pad_features_to_alignment(act_tensor: torch.Tensor, weights_tensor: torch.Tensor, aligned_in_features: int):
    """Pad activation/weight feature dimensions to match hardware alignment requirements."""
    in_features = act_tensor.shape[1]
    if in_features < aligned_in_features:
        pad_cols = aligned_in_features - in_features
        act_tensor = torch.nn.functional.pad(act_tensor, (0, pad_cols))
        weights_tensor = torch.nn.functional.pad(weights_tensor, (0, pad_cols))
        in_features = aligned_in_features
    return act_tensor, weights_tensor, in_features


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
            bias=False, # Remapped as purely mm without bias for simpler PLENA simulation
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


if __name__ == "__main__":
    """
    Test the SiglipVisionEmbeddings component and generate hardware simulator testing.
    This folds the Conv2d into an equivalent matrix multiplication to map to the projection_asm.
    """
    print("Testing SiglipVisionEmbeddings in PLENA Hardware Simulator...")

    config = SiglipVisionConfig(
        hidden_size=64,
        image_size=128,
        patch_size=16,
        num_channels=3
    )

    mlen = 64
    blen = 4
    vlen = 64
    aligned_in_features = 768

    batch_size = 1
    in_features = config.num_channels * config.patch_size * config.patch_size
    out_features = config.hidden_size
    num_patches = (config.image_size // config.patch_size) ** 2

    torch.manual_seed(42)
    model = SiglipVisionEmbeddings(config).bfloat16()
    model.eval()

    pixel_values = torch.randn(batch_size, config.num_channels, config.image_size, config.image_size, dtype=torch.bfloat16)

    # 1. Compute Expected output from full PyTorch model (including Position Embeddings)
    with torch.no_grad():
        expected_output = model(pixel_values)

    # 2. Extract and Quantize for hardware simulation (im2col equivalence)
    # The Conv2d can be viewed as a Linear layer over non-overlapping patches.
    act_tensor = extract_non_overlapping_patches(pixel_values, config.patch_size)
    effective_batch = act_tensor.shape[0]

    # Extract weights from patch_embedding and reshape them to (out_features, in_features)
    weights_tensor = model.patch_embedding.weight.detach().view(out_features, in_features)

    # Hardware alignment: pad feature dimension to simulator-required width.
    act_tensor, weights_tensor, in_features = pad_features_to_alignment(
        act_tensor,
        weights_tensor,
        aligned_in_features,
    )

    # Quantize to E4M3
    act_mxfp = quantize_to_mxfp(act_tensor).to(torch.bfloat16)
    weights_mxfp = quantize_to_mxfp(weights_tensor.t()).to(torch.bfloat16) # (in_features, out_features)
    print(f"Length of weights row (in_features): {weights_mxfp.shape[0]}, should be aligned to 256 for hardware.")

    # Build the positional embedding tensor separately so it can be loaded and added in hardware.
    position_tensor = model.position_embedding(model.position_ids).expand(batch_size, -1, -1)
    position_tensor = position_tensor.reshape(effective_batch, out_features)

    # Quantize the positional embeddings to the same hardware format as the projection output.
    position_mxfp = quantize_to_mxfp(position_tensor).to(torch.bfloat16)

    # Compute HW golden outputs for the patch projection and the final positional embedding sum.
    projection_golden = torch.mm(act_mxfp, weights_mxfp) # (batch_size*num_patches, out_features)
    final_golden = projection_golden + position_mxfp

    # Note: the projection operates on each patch flattened into a row.
    # effective_batch = batch_size * num_patches is the number of rows fed to the projection.
    print(f"Siglip Patch Embedding: ({effective_batch}, {in_features}) @ ({in_features}, {out_features}) -> ({effective_batch}, {out_features})")
    print("expected_output shape (PyTorch full model):", expected_output.shape)
    print("projection_golden shape (per-patch projection):", projection_golden.shape)


    # Set up memory layout for PLENA Simulator
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
    real_data_ratio = MXFP_REAL_DATA_RATIO

    act_hbm_size = int(in_features * effective_batch * real_data_ratio)
    weight_hbm_offset = act_hbm_size
    weight_hbm_end = int((in_features * effective_batch + in_features * out_features) * real_data_ratio)

    gen_assembly_code, result_vram_offset = build_embedding_projection_asm(
        title="Siglip Patch Embedding Test (Conv2d -> Linear Flattened)",
        shape_batch=effective_batch,
        in_features=in_features,
        out_features=out_features,
        effective_batch=effective_batch,
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        weight_hbm_offset=weight_hbm_offset,
        weight_hbm_end=weight_hbm_end,
    )

    position_vram_offset = result_vram_offset + effective_batch * out_features

    gen_assembly_code = append_position_add_asm(
        gen_assembly_code=gen_assembly_code,
        result_vram_offset=result_vram_offset,
        position_vram_offset=position_vram_offset,
        batch=effective_batch,
        out_features=out_features,
        vlen=vlen,
    )

    build_path = Path(__file__).parent / "build"
    prepare_case_artifacts(
        case_build_dir=build_path,
        input_tensor=input_tensor,
        asm_code=gen_assembly_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=None,
        hbm_mb=act_hbm_size + int((in_features * out_features) * real_data_ratio) + int((effective_batch * out_features) * real_data_ratio),
        data_order=["act_tensor", "weights", "position_tensor"],
    )

    result_start_row = result_vram_offset // mlen
    num_result_rows = (effective_batch * out_features) // mlen
    write_comparison_params(
        build_path,
        start_row_idx=result_start_row,
        num_rows=num_result_rows,
        num_batches=effective_batch,
        elements_per_batch=out_features,
    )

    print("================================================")
    print("Finished generating HW assembly code.")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Expected shape from PyTorch (full HW unquantized w/ PosEmb included): {expected_output.shape}")
    print(f"Intermediate Simulation check shape (Patch Conv Projection): {projection_golden.shape}")
    print(f"Final HW golden shape (Projection + PosEmb): {final_golden.shape}")
    print("================================================")

    run_and_assert(build_path, "siglip_embedding_positional", mlen=mlen, blen=blen)
