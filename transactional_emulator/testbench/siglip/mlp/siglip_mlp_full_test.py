#!/usr/bin/env python3
from pathlib import Path
import os

import json
import torch
from torch import nn

from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware
from transactional_emulator.testbench.siglip.local_asm_templates.mlp_blocks import build_mlp_pipeline_asm
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.siglip.utils.harness_utils import prepare_case_artifacts


def quantize_to_mxfp(tensor: torch.Tensor) -> torch.Tensor:
    quantized, _, _, _ = _mx_fp_quantize_hardware(
        tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8]
    )
    return quantized.reshape(tensor.shape)


def gelu_with_bf16_intermediates(tensor: torch.Tensor) -> torch.Tensor:
    tensor_f32 = tensor.float()
    step1 = (1.702 * tensor_f32).to(torch.bfloat16)
    step2 = (-step1.float()).to(torch.bfloat16)
    step3 = torch.exp(step2.float()).to(torch.bfloat16)
    step4 = (1.0 + step3.float()).to(torch.bfloat16)
    step5 = (1.0 / step4.float()).to(torch.bfloat16)
    return (tensor_f32 * step5.float()).to(torch.bfloat16)


class SiglipMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x


def _resolve_vision_mlp(model, layer_idx: int = 0):
    vision_root = getattr(model, "vision_model", model)

    # SigLIP models typically expose vision layers as vision_model.encoder.layers.
    encoder = getattr(vision_root, "encoder", None)
    if encoder is not None and hasattr(encoder, "layers"):
        return encoder.layers[layer_idx].mlp

    # Fallback path for model variants that expose vision_model.layers.
    if hasattr(vision_root, "layers"):
        return vision_root.layers[layer_idx].mlp

    raise AttributeError("Could not locate SigLIP vision MLP module")


if __name__ == "__main__":
    repo_root = Path(__file__).parents[3]
    config_path = repo_root / "compiler" / "doc" / "Model_Lib" / "siglip-so400m-patch14-384.json"
    model_id = "google/siglip-so400m-patch14-384"

    with open(config_path) as f:
        siglip_config = json.load(f)

    hidden_size = int(siglip_config["vision_config"]["hidden_size"])
    intermediate_size = int(siglip_config["vision_config"]["intermediate_size"])
    hidden_size_override = int(os.environ.get("SIGLIP_MLP_HIDDEN_SIZE", "0"))
    intermediate_override = int(os.environ.get("SIGLIP_MLP_INTERMEDIATE_SIZE", "0"))
    if hidden_size_override > 0:
        hidden_size = hidden_size_override
    if intermediate_override > 0:
        intermediate_size = intermediate_override

    batch_size = int(os.environ.get("SIGLIP_MLP_BATCH_SIZE", "4"))
    seq_len = int(os.environ.get("SIGLIP_MLP_SEQ_LEN", "1"))
    effective_batch = batch_size * seq_len

    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    fp_preload = [0.0, 1.0, 0.0, 0.0, 1.702]

    mlen = int(os.environ.get("SIGLIP_MLP_MLEN", "64"))
    blen = 4
    vlen = int(os.environ.get("SIGLIP_MLP_VLEN", str(mlen)))

    torch.manual_seed(42)

    act_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    act_matrix = act_tensor.reshape(effective_batch, hidden_size)
    act_hbm = quantize_to_mxfp(act_matrix).to(torch.bfloat16)

    from transformers import AutoModel

    cache_dir = Path(os.environ.get("SIGLIP_MLP_CACHE_DIR", str(Path(__file__).parent / "build")))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "siglip_mlp_weights.pt"
    use_weight_cache = os.environ.get("SIGLIP_MLP_USE_WEIGHT_CACHE", "1") != "0"

    if use_weight_cache and cache_file.exists():
        print(f"Loading cached MLP weights from {cache_file} ...")
        cached = torch.load(cache_file, map_location="cpu")
        weight_up = cached["weight_up"]
        weight_down = cached["weight_down"]
    else:
        print(f"Loading SigLIP weights from {model_id} ...")
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
        mlp = _resolve_vision_mlp(model, layer_idx=0)

        # Keep only weight matrices for hardware parity; ASM path does not add bias.
        weight_up = mlp.fc1.weight.detach().to(torch.bfloat16).contiguous()
        weight_down = mlp.fc2.weight.detach().to(torch.bfloat16).contiguous()
        if use_weight_cache:
            torch.save({"weight_up": weight_up, "weight_down": weight_down}, cache_file)
            print(f"Saved MLP weight cache to {cache_file}")

    weight_up_hbm = quantize_to_mxfp(weight_up).to(torch.bfloat16)
    weight_down_hbm = quantize_to_mxfp(weight_down).to(torch.bfloat16)

    aligned_intermediate = ((intermediate_size + mlen - 1) // mlen) * mlen
    pad_rows = aligned_intermediate - intermediate_size
    weight_up_padded = torch.nn.functional.pad(weight_up_hbm.t(), (0, pad_rows)).contiguous()
    weight_down_padded = torch.nn.functional.pad(weight_down_hbm.t(), (0, 0, 0, pad_rows)).contiguous()

    up_proj = torch.matmul(act_hbm.float(), weight_up_padded.float())
    activated = gelu_with_bf16_intermediates(up_proj)
    final_output = torch.matmul(activated.float(), weight_down_padded.float())
    final_output = final_output[:, :hidden_size].contiguous()

    input_tensor = {
        "act_tensor": act_hbm,
        "weight_up_layer": weight_up_padded,
        "weight_down_layer": weight_down_padded,
    }
    golden_result = {"input_tensor": input_tensor, "original_output": final_output.flatten()}

    gen_assembly_code, final_vram_offset = build_mlp_pipeline_asm(
        title="SigLIP MLP Full-Config Test",
        effective_batch=effective_batch,
        hidden_size=hidden_size,
        aligned_intermediate=aligned_intermediate,
        real_data_ratio=real_data_ratio,
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        weight_up_numel=weight_up_padded.numel(),
        weight_hbm_reg=1,
        weight_down_hbm_reg=2,
        gelu_one_fp_slot=1,
        gelu_1702_fp_slot=4,
    )

    build_path = Path(__file__).parent / "build"

    act_hbm_mb = int((hidden_size * effective_batch * real_data_ratio + (1024 * 1024 - 1)) // (1024 * 1024))
    weight_up_mb = int((weight_up_padded.numel() * real_data_ratio + (1024 * 1024 - 1)) // (1024 * 1024))
    weight_down_mb = int((weight_down_padded.numel() * real_data_ratio + (1024 * 1024 - 1)) // (1024 * 1024))
    total_hbm_size_mb = act_hbm_mb + weight_up_mb + weight_down_mb + 10

    prepare_case_artifacts(
        case_build_dir=build_path,
        input_tensor=input_tensor,
        asm_code=gen_assembly_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=None,
        hbm_mb=total_hbm_size_mb,
        data_order=["act_tensor", "weight_up_layer", "weight_down_layer"],
    )

    result_start_row = final_vram_offset // vlen
    num_result_rows = (effective_batch * hidden_size) // vlen
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": effective_batch,
        "elements_per_batch": hidden_size,
        "use_stride_mode": True,
    }
    with open(build_path / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating SigLIP MLP full-config test")
    print(f"Config: hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Expected shape: ({effective_batch}, {hidden_size})")
    print("================================================")

    run_and_assert(build_path, "siglip_mlp_full", mlen=mlen, blen=blen)
