import sys
from pathlib import Path
import json
import torch
import torch.nn as nn
from dataclasses import dataclass

from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.siglip.local_asm_templates.embedding_blocks import (
    build_embedding_projection_asm,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


@dataclass
class SiglipVisionConfig:
    hidden_size: int = 128
    image_size: int = 16
    patch_size: int = 8
    num_channels: int = 1


def quantize_to_mxfp(tensor):
    """
    Quantize tensor to MXFP format matching hardware (E4M3 with 8-bit scale per block of 8).
    Uses the same quantizer as the transactional emulator's memory loader.
    Returns the dequantized tensor (what hardware sees after HBM->VRAM load).
    """
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8]) # E4M3
    return bm_x.reshape(orig_shape)


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
        image_size=64,
        patch_size=8,
        num_channels=3
    )
    
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
    # The Conv2d can be viewed as applying a Linear layer on non-overlapping patches
    # Format pixel values into non-overlapping patches (B, num_patches, in_features)
    patches = pixel_values.unfold(2, config.patch_size, config.patch_size).unfold(3, config.patch_size, config.patch_size)
    patches = patches.contiguous().view(batch_size, config.num_channels, num_patches, -1)
    patches = patches.transpose(1, 2).flatten(2) # (batch, num_patches, in_features)
    
    # Reshape to 2D for linear projection ASM (batch * num_patches, in_features)
    effective_batch = batch_size * num_patches
    act_tensor = patches.view(effective_batch, in_features)
    
    # Extract weights from patch_embedding and reshape them to (out_features, in_features)
    weights_tensor = model.patch_embedding.weight.data.view(out_features, in_features)

    # Hardware alignment: Pad in_features to a multiple of 256 (mlen=64 * default blen=4)
    aligned_in_features = 256
    if in_features < aligned_in_features:
        act_tensor = torch.nn.functional.pad(act_tensor, (0, aligned_in_features - in_features))
        weights_tensor = torch.nn.functional.pad(weights_tensor, (0, aligned_in_features - in_features))
        in_features = aligned_in_features
    
    # Quantize to E4M3
    act_mxfp = quantize_to_mxfp(act_tensor).to(torch.bfloat16) 
    weights_mxfp = quantize_to_mxfp(weights_tensor.t()).to(torch.bfloat16) # (in_features, out_features)

    # Compute HW Golden output for the patch projection (ignoring pos embeddings for the raw projection test part)
    projection_golden = torch.mm(act_mxfp, weights_mxfp) # (batch_size*num_patches, out_features)
    
    # Note: the projection operates on each patch flattened into a row.
    # effective_batch = batch_size * num_patches is the number of rows fed to the projection.
    print(f"Siglip Patch Embedding: ({effective_batch}, {in_features}) @ ({in_features}, {out_features}) -> ({effective_batch}, {out_features})")
    print("expected_output shape (PyTorch full model):", expected_output.shape)
    print("projection_golden shape (per-patch projection):", projection_golden.shape)

    
    # Set up memory layout for PLENA Simulator
    input_tensor = {
        "act_tensor": act_mxfp,
        "weights": weights_mxfp,
    }
    
    golden_result = {
        "input_tensor": input_tensor,
        "original_output": projection_golden, # The simulator will only check this intermediate projection
    }
    
    fp_preload = [0.0, 1e-6, 1 / in_features]
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    
    act_hbm_size = int(in_features * effective_batch * real_data_ratio)
    weight_hbm_offset = act_hbm_size
    weight_hbm_end = int((in_features * effective_batch + in_features * out_features) * real_data_ratio)

    gen_assembly_code, result_vram_offset = build_embedding_projection_asm(
        title="Siglip Patch Embedding Test (Conv2d -> Linear Flattened)",
        shape_batch=effective_batch,
        in_features=in_features,
        out_features=out_features,
        effective_batch=effective_batch,
        mlen=64,
        blen=4,
        vlen=64,
        weight_hbm_offset=weight_hbm_offset,
        weight_hbm_end=weight_hbm_end,
    )

    build_path = Path(__file__).parent / "build"
    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload, build_dir=build_path)
    
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=None, 
        data=None,
        specified_data_order=["act_tensor", "weights"],
        build_path=build_path,
    )

    result_start_row = result_vram_offset // 64
    num_result_rows = (effective_batch * out_features) // 64
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": effective_batch,
        "elements_per_batch": out_features,
    }
    
    with open(build_path / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating HW assembly code.")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Expected shape from PyTorch (full HW unquantized w/ PosEmb included): {expected_output.shape}")
    print(f"Intermediate Simulation check shape (Patch Conv Projection): {projection_golden.shape}")
    print("================================================")

    run_and_assert(build_path, "siglip_embedding_nopositional", mlen=64, blen=4)