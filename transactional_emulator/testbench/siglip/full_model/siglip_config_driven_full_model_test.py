"""Config-driven full-model end-to-end test for SigLIP.

Ties together config loading, weight extraction, golden computation, and emulator
execution for a complete end-to-end test that can scale from 1 to 27 encoder layers.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.model_loader import (
    load_siglip_config,
    load_siglip_vision_model,
    extract_embedding_weights,
    extract_layer_weights,
)
from transactional_emulator.testbench.siglip.full_model.golden_reference import (
    compute_golden_embedding,
    compute_golden_layer,
)
from transactional_emulator.testbench.siglip.full_model.memory_layout import (
    compute_full_model_vram_layout,
    compute_full_model_hbm_layout,
    validate_memory_layout,
)


def extract_patches(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Extract patches from image tensor.

    Args:
        pixel_values: [batch, channels, height, width]
        patch_size: Kernel size for patch extraction

    Returns:
        [batch*num_patches, channels*patch_size*patch_size]
    """
    patches = F.unfold(pixel_values.float(), kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).contiguous()


def run_full_model_test(
    config_path: str = "compiler/doc/Model_Lib/siglip-so400m-patch14-384.json",
    output_dir: Path | None = None,
    max_layers: int = 27,
    use_mxfp: bool = True,
    model_dtype: str = "float32",
    seed: int = 42,
) -> dict:
    """Run full end-to-end config-driven test.

    Args:
        config_path: Path to SigLIP config JSON
        output_dir: Directory for outputs (defaults to ./build)
        max_layers: Number of encoder layers to test
        use_mxfp: Whether to use MXFP quantization
        model_dtype: Model load dtype for real-model path (float32 or bfloat16)
        seed: Random seed

    Returns:
        dict with test results and metrics
    """
    torch.manual_seed(seed)

    if output_dir is None:
        output_dir = Path("./build") / "full_model_test"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SigLIP Full-Model Config-Driven End-to-End Test")
    print("=" * 80)
    print(f"\nConfig: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Max layers: {max_layers}")
    print(f"Use MXFP quantization: {use_mxfp}")
    print(f"Model dtype: {model_dtype}")

    # ========== Step 1: Load Configuration ==========
    print("\n--- Step 1: Loading Configuration ---")
    config = load_siglip_config(config_path)

    # ========== Step 2: Load Model and Extract Weights ==========
    print("\n--- Step 2: Loading Model and Extracting Weights ---")
    model = load_siglip_vision_model(model_dtype=model_dtype)
    embedding_weights = extract_embedding_weights(model, config)

    layer_weights_list = []
    for layer_idx in range(max_layers):
        layer_weights = extract_layer_weights(model, layer_idx, config["hidden_size"])
        layer_weights_list.append(layer_weights)

    print(f"✓ Extracted weights for embedding + {len(layer_weights_list)} layers")

    # ========== Step 3: Create Synthetic Input ==========
    print("\n--- Step 3: Creating Synthetic Input ---")
    image_size = config["image_size"]
    patch_size = config["patch_size"]
    num_channels = config["num_channels"]

    pixel_values = torch.randn(1, num_channels, image_size, image_size, dtype=torch.float32)
    patches = extract_patches(pixel_values, patch_size=patch_size)[0]  # [num_patches, in_features]

    print(f"✓ Input shape: {pixel_values.shape}")
    print(f"✓ Patches shape: {patches.shape}")

    # ========== Step 4: Compute Golden Outputs ==========
    print("\n--- Step 4: Computing Golden Outputs (Hardware-Faithful) ---")

    golden_embedding = compute_golden_embedding(patches, embedding_weights, config, use_mxfp=use_mxfp)
    print(f"✓ Embedding output: {golden_embedding.shape} range [{golden_embedding.min():.3f}, {golden_embedding.max():.3f}]")

    layer_outputs_golden = [golden_embedding.detach()]
    x_golden = golden_embedding

    for layer_idx in range(max_layers):
        x_golden = compute_golden_layer(
            x_golden,
            layer_weights_list[layer_idx],
            config,
            layer_idx=layer_idx,
            use_mxfp=use_mxfp,
        )
        layer_outputs_golden.append(x_golden.detach())
        print(f"✓ Layer {layer_idx} output: {x_golden.shape} range [{x_golden.min():.3f}, {x_golden.max():.3f}]")

    # ========== Step 5: Compute Real Model Outputs ==========
    print("\n--- Step 5: Computing Real Model Outputs ---")

    with torch.no_grad():
        model.eval()

        # Extract patches for real model
        real_input = pixel_values.float()

        # Get embedding from real model
        vision_root = model.vision_model if hasattr(model, "vision_model") else model
        real_embedding_output = vision_root.embeddings(real_input)
        print(f"✓ Real embedding output: {real_embedding_output.shape}")

        # Run through encoder layers
        real_layer_outputs = [real_embedding_output.detach()]
        x_real = real_embedding_output

        for layer_idx in range(max_layers):
            x_real = vision_root.encoder.layers[layer_idx](x_real, attention_mask=None)
            real_layer_outputs.append(x_real.detach())
            print(f"✓ Real layer {layer_idx} output: {x_real.shape}")

    # ========== Step 6: Compute Memory Layouts ==========
    print("\n--- Step 6: Computing Memory Layouts ---")
    vram_layout = compute_full_model_vram_layout(config)
    hbm_layout = compute_full_model_hbm_layout(config, embedding_weights, layer_weights_list)

    print(f"✓ VRAM layout: {vram_layout['total_vram_mb']:.2f} MB")
    print(f"✓ HBM layout: {hbm_layout[1] * 2 / (1024 * 1024):.2f} MB")

    validate_memory_layout(vram_layout, hbm_layout)

    # ========== Step 7: Compare Layer Outputs ==========
    print("\n--- Step 7: Comparing Layer Outputs ---")

    layer_reports = []

    # Compare embedding
    embed_golden = layer_outputs_golden[0]
    embed_real = real_layer_outputs[0][0] if real_layer_outputs[0].shape[0] > 1 else real_layer_outputs[0]

    embed_mae = torch.abs(embed_golden - embed_real.float()).mean().item()
    embed_match_rate = (torch.isclose(embed_golden, embed_real.float(), atol=1e-2, rtol=1e-2).float().mean() * 100).item()

    print(f"Embedding: MAE={embed_mae:.6f}, Match Rate={embed_match_rate:.2f}%")

    # Compare layers
    for layer_idx in range(max_layers):
        layer_golden = layer_outputs_golden[layer_idx + 1]
        layer_real = real_layer_outputs[layer_idx + 1][0] if real_layer_outputs[layer_idx + 1].shape[0] > 1 else real_layer_outputs[layer_idx + 1]

        mae = torch.abs(layer_golden - layer_real.float()).mean().item()
        match_rate = (torch.isclose(layer_golden, layer_real.float(), atol=1e-2, rtol=1e-2).float().mean() * 100).item()
        max_error = torch.abs(layer_golden - layer_real.float()).max().item()

        report = {
            "layer_idx": layer_idx,
            "mae": float(mae),
            "match_rate": float(match_rate),
            "max_error": float(max_error),
            "shape": list(layer_golden.shape),
        }
        layer_reports.append(report)

        print(f"Layer {layer_idx}: MAE={mae:.6f}, Match Rate={match_rate:.2f}%, Max Error={max_error:.6f}")

    # ========== Step 8: Generate Report ==========
    print("\n--- Step 8: Generating Report ---")

    summary = {
        "config": config,
        "num_layers": max_layers,
        "use_mxfp": use_mxfp,
        "model_dtype": model_dtype,
        "embedding_metrics": {
            "mae": float(embed_mae),
            "match_rate": float(embed_match_rate),
        },
        "layer_metrics": layer_reports,
        "final_output_shape": list(x_golden.shape),
        "vram_mb": float(vram_layout['total_vram_mb']),
        "hbm_mb": float(hbm_layout[1] * 2 / (1024 * 1024)),
    }

    # Save report
    report_path = output_dir / "summary.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Report saved to {report_path}")

    print("\n" + "=" * 80)
    print("✓ Test Completed Successfully!")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    """Run the full model test."""
    parser = argparse.ArgumentParser(description="SigLIP config-driven full-model comparator")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="compiler/doc/Model_Lib/siglip-so400m-patch14-384.json",
        help="Path to SigLIP config JSON",
    )
    parser.add_argument(
        "max_layers",
        nargs="?",
        type=int,
        default=27,
        help="Number of encoder layers to compare",
    )
    parser.add_argument(
        "--output-dir",
        default="./build/full_model_test",
        help="Directory for summary output",
    )
    parser.add_argument(
        "--use-mxfp",
        dest="use_mxfp",
        action="store_true",
        help="Enable MXFP quantization in golden path (hardware-faithful mode)",
    )
    parser.add_argument(
        "--no-mxfp",
        dest="use_mxfp",
        action="store_false",
        help="Disable MXFP quantization for model-faithful comparison",
    )
    parser.add_argument(
        "--model-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="Dtype used to load the Hugging Face model",
    )
    parser.set_defaults(use_mxfp=True)
    args = parser.parse_args()

    result = run_full_model_test(
        config_path=args.config_path,
        output_dir=Path(args.output_dir),
        max_layers=args.max_layers,
        use_mxfp=args.use_mxfp,
        model_dtype=args.model_dtype,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Embedding match rate: {result['embedding_metrics']['match_rate']:.2f}%")
    print(f"Num layers tested: {result['num_layers']}")

    if result['layer_metrics']:
        avg_layer_match = np.mean([r['match_rate'] for r in result['layer_metrics']])
        print(f"Average layer match rate: {avg_layer_match:.2f}%")
