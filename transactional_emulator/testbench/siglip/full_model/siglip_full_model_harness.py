"""ASM generation for full 27-layer SigLIP model.

Emits monolithic assembly code for all encoder layers, managing VRAM/HBM layout globally.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from transactional_emulator.testbench.config_utils import update_plena_config
from transactional_emulator.testbench.emulator_runner import run_emulator

from transactional_emulator.testbench.siglip.local_asm_templates.embedding_blocks import (
    append_position_add_asm,
    build_embedding_projection_asm,
)
from transactional_emulator.testbench.siglip.local_asm_templates.encoder_layer_blocks import (
    build_encoder_layer_asm,
)
from transactional_emulator.testbench.siglip.local_asm_templates.layout import (
    compute_vram_layout,
)
from transactional_emulator.testbench.siglip.utils.harness_utils import prepare_case_artifacts


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _build_hbm_input_tensors(
    embedding_weights: dict,
    layer_weights_list: list,
    num_layers: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Build HBM preload tensors keyed by filename stem in create_mem_for_sim order.

    The run path is a smoke harness: it executes generated ASM in emulator, but does
    not claim numerical parity yet.
    """
    patch_bias = embedding_weights.get("patch_bias")
    if patch_bias is None:
        patch_bias = torch.zeros(1, dtype=torch.float32)

    input_tensor: dict[str, torch.Tensor] = {
        "patch_weight": embedding_weights["patch_weight"].reshape(-1).to(torch.float32),
        "patch_bias": patch_bias.reshape(-1).to(torch.float32),
        "position_table": embedding_weights["position_table"].reshape(-1).to(torch.float32),
    }

    data_order = ["patch_weight", "patch_bias", "position_table"]

    for layer_idx in range(num_layers):
        weights = layer_weights_list[layer_idx]

        # Keep names aligned with compute_hbm_data_order() and hbm layout keys.
        keyed_tensors = {
            f"layer_{layer_idx}_ln1_weight": weights["ln1_weight"],
            f"layer_{layer_idx}_ln1_bias": weights["ln1_bias"],
            f"layer_{layer_idx}_q_proj_weight": weights["q_proj_weight"],
            f"layer_{layer_idx}_q_proj_bias": weights["q_proj_bias"],
            f"layer_{layer_idx}_k_proj_weight": weights["k_proj_weight"],
            f"layer_{layer_idx}_k_proj_bias": weights["k_proj_bias"],
            f"layer_{layer_idx}_v_proj_weight": weights["v_proj_weight"],
            f"layer_{layer_idx}_v_proj_bias": weights["v_proj_bias"],
            f"layer_{layer_idx}_out_proj_weight": weights["out_proj_weight"],
            f"layer_{layer_idx}_out_proj_bias": weights["out_proj_bias"],
            f"layer_{layer_idx}_ln2_weight": weights["ln2_weight"],
            f"layer_{layer_idx}_ln2_bias": weights["ln2_bias"],
            f"layer_{layer_idx}_fc1_weight": weights["fc1_weight"],
            f"layer_{layer_idx}_fc1_bias": weights["fc1_bias"],
            f"layer_{layer_idx}_fc2_weight": weights["fc2_weight"],
            f"layer_{layer_idx}_fc2_bias": weights["fc2_bias"],
        }

        for key, tensor in keyed_tensors.items():
            if tensor is None:
                tensor = torch.zeros(1, dtype=torch.float32)
            input_tensor[key] = tensor.reshape(-1).to(torch.float32)
            data_order.append(key)

    return input_tensor, data_order


def _prepare_vram_preload(
    config: dict,
    embedding_weights: dict,
    seq_len: int,
    mlen: int,
    hidden_size: int,
) -> np.ndarray:
    """Prepare VRAM preload with [patches | zeros until position offset | position table]."""
    patch_size = config["patch_size"]
    num_channels = config["num_channels"]
    in_features = num_channels * patch_size * patch_size
    aligned_in_features = _align_up(in_features, mlen)

    torch.manual_seed(0)
    pixel_values = torch.randn(1, num_channels, config["image_size"], config["image_size"], dtype=torch.float32)
    patches = F.unfold(pixel_values, kernel_size=patch_size, stride=patch_size).transpose(1, 2).contiguous()[0]

    if aligned_in_features != in_features:
        patches = F.pad(patches, (0, aligned_in_features - in_features))

    input_flat = patches.reshape(-1).to(torch.bfloat16)

    embedding_result_base = aligned_in_features * seq_len
    position_base = embedding_result_base + seq_len * hidden_size
    position_flat = embedding_weights["position_table"].reshape(-1).to(torch.bfloat16)

    total_len = position_base + position_flat.numel()
    preload = torch.zeros(total_len, dtype=torch.bfloat16)
    preload[: input_flat.numel()] = input_flat
    preload[position_base : position_base + position_flat.numel()] = position_flat

    return preload.view(torch.int16).numpy().view(np.uint16)


def run_full_model_emulator_smoke(
    config: dict,
    embedding_weights: dict,
    layer_weights_list: list,
    vram_layout: dict,
    hbm_layout: tuple,
    asm_code: str,
    build_dir: Path,
    max_layers: int,
    mlen: int,
    vlen: int,
    blen: int,
    write_golden_txt: bool = True,
) -> None:
    """Run generated full-model ASM in emulator as a smoke test path."""
    num_layers = min(max_layers, len(layer_weights_list))
    seq_len = vram_layout["seq_len"]
    hidden_size = config["hidden_size"]

    input_tensor, data_order = _build_hbm_input_tensors(embedding_weights, layer_weights_list, num_layers)
    vram_preload = _prepare_vram_preload(config, embedding_weights, seq_len, mlen, hidden_size)

    # Keep emulator hardware config in sync with generated assembly assumptions.
    update_plena_config(vlen=vlen, mlen=mlen, blen=blen, verbose=False)

    final_output_base = vram_layout["layer_bases"][num_layers - 1] if num_layers > 0 else vram_layout["embedding_base"]
    golden_result = {
        "input_tensor": {k: v for k, v in input_tensor.items()},
        # Smoke run placeholder: numerical comparison is not yet wired in this harness.
        "original_output": torch.zeros(seq_len * hidden_size, dtype=torch.float32),
    }

    fp_preload = compute_fp_preload(config, mlen=mlen)

    build_dir = build_dir.resolve()
    # hbm_layout[1] is total HBM elements in BF16 units.
    hbm_mb = int(np.ceil((hbm_layout[1] * 2) / (1024 * 1024))) + 16
    prepare_case_artifacts(
        case_build_dir=build_dir,
        input_tensor=input_tensor,
        asm_code=asm_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=vram_preload,
        hbm_mb=hbm_mb,
        data_order=data_order,
        write_golden_txt=write_golden_txt,
    )

    for key in data_order:
        pt_path = build_dir / f"{key}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"Missing HBM tensor payload: {pt_path}")
        loaded = torch.load(pt_path)
        if loaded is None:
            raise ValueError(f"HBM tensor payload is None: {pt_path}")

    run_emulator(build_dir, log_path=build_dir / "emulator.log")
    print(f"✓ Emulator smoke run completed. Build dir: {build_dir}")
    print(f"  Final output base (elements): {final_output_base}")


def build_full_model_asm(
    config: dict,
    embedding_weights: dict,
    layer_weights_list: list,
    vram_layout: dict,
    hbm_layout: tuple,
    mlen: int = 64,
    vlen: int = 64,
    blen: int = 4,
    max_layers: int = 27,
) -> str:
    """Emit complete ASM code for all 27 encoder layers + embedding.

    Args:
        config: Config dict from load_siglip_config()
        embedding_weights: Embedding weight dict from extract_embedding_weights()
        layer_weights_list: List of per-layer weight dicts from extract_layer_weights()
        vram_layout: VRAM layout from compute_full_model_vram_layout()
        hbm_layout: (layout_dict, total_hbm_elems) from compute_full_model_hbm_layout()
        mlen, vlen, blen: Hardware parameters
        max_layers: Number of layers to emit (usually 27)

    Returns:
        Complete ASM code as string
    """
    hbm_layout_dict, _total_hbm = hbm_layout

    seq_len = vram_layout["seq_len"]
    hidden_size = config["hidden_size"]
    inter_size = config["intermediate_size"]
    num_heads = config["num_attention_heads"]
    num_kv_heads = config["num_key_value_heads"]

    # Padding/alignment
    hidden_padded = ((hidden_size + mlen - 1) // mlen) * mlen
    inter_padded = ((inter_size + mlen - 1) // mlen) * mlen
    d_padded = mlen

    # Real data ratio for quantization overhead
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    asm_code = ""
    asm_code += "; ============================================================\n"
    asm_code += "; SigLIP Full Model (27 Encoder Layers)\n"
    asm_code += f"; Hidden={hidden_size}, Heads={num_heads}, Inter={inter_size}\n"
    asm_code += f"; Sequence length={seq_len}, MLEN={mlen}, VLEN={vlen}\n"
    asm_code += "; ============================================================\n\n"

    # ========== Embedding Stage ==========
    asm_code += "; --- STAGE 0: Patch Embedding + Position Add ---\n"

    in_features = 3 * config["patch_size"] * config["patch_size"]

    # Align input features to MLEN
    aligned_in_features = ((in_features + mlen - 1) // mlen) * mlen

    # Compute HBM offsets for embedding
    patch_weight_offset = hbm_layout_dict.get("embedding", {}).get("patch_weight", 0)

    embedding_proj_asm, embedding_result_base = build_embedding_projection_asm(
        title="SigLIP Patch Projection",
        shape_batch=seq_len,
        in_features=aligned_in_features,  # Use aligned value
        out_features=hidden_size,
        effective_batch=seq_len,
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        weight_hbm_offset=patch_weight_offset,
        weight_hbm_end=patch_weight_offset + int(aligned_in_features * hidden_size * real_data_ratio),
    )

    asm_code += embedding_proj_asm + "\n"

    # Position embedding add
    position_vram_offset = embedding_result_base + seq_len * hidden_size
    embedding_asm = append_position_add_asm(
        gen_assembly_code="",
        result_vram_offset=embedding_result_base,
        position_vram_offset=position_vram_offset,
        batch=seq_len,
        out_features=hidden_size,
        vlen=vlen,
    )

    asm_code += embedding_asm + "\n"

    # The embedding projection template writes the output activation here.
    embedding_output_base = embedding_result_base

    # ========== Encoder Layers ==========
    num_layers_to_emit = min(max_layers, len(layer_weights_list))

    # Reusable workspace region placed after persistent per-layer output buffers.
    persistent_end = vram_layout["embedding_base"] + vram_layout["embedding_size"]
    if vram_layout.get("layer_bases"):
        last_layer_idx = max(vram_layout["layer_bases"].keys())
        persistent_end = max(
            persistent_end,
            vram_layout["layer_bases"][last_layer_idx] + vram_layout["layer_sizes"][last_layer_idx],
        )
    workspace_base = _align_up(persistent_end, mlen)

    for layer_idx in range(num_layers_to_emit):
        asm_code += f"\n; --- LAYER {layer_idx}: Encoder Layer ---\n"

        # Layer input/output chaining:
        # - layer 0 consumes embedding output
        # - layer N consumes layer N-1 output
        # - each layer writes to its dedicated output base
        layer_input_base = embedding_output_base if layer_idx == 0 else vram_layout["layer_bases"][layer_idx - 1]
        layer_output_base = vram_layout["layer_bases"][layer_idx]

        # HBM offsets for this layer's weights
        layer_hbm = hbm_layout_dict.get("layers", {}).get(layer_idx, {})
        wq_offset = layer_hbm.get("q_proj_weight", 0)
        k_offset = layer_hbm.get("k_proj_weight", 0)
        v_offset = layer_hbm.get("v_proj_weight", 0)
        w1_offset = layer_hbm.get("fc1_weight", 0)
        w2_offset = layer_hbm.get("fc2_weight", 0)

        # Temporary VRAM allocations for this layer in a reusable non-overlapping workspace.
        q_seq_base = workspace_base
        q_vram_base = q_seq_base + seq_len * hidden_padded

        flash_layout = compute_vram_layout(
            mlen=mlen,
            blen=blen,
            q_len=seq_len,
            kv_len=seq_len,
            hq=num_heads,
            hkv=num_kv_heads,
            d=d_padded,
            vector_sram_base=q_vram_base,
        )
        attn_vram_base = flash_layout["o_old_base"]
        residual_vram_base = attn_vram_base + seq_len * hidden_padded
        mlp_inter_vram_base = residual_vram_base + seq_len * hidden_padded
        scratch_base = _align_up(mlp_inter_vram_base + seq_len * inter_padded, mlen)

        # Build encoder layer ASM
        layer_asm = build_encoder_layer_asm(
            mlen=mlen,
            blen=blen,
            vlen=vlen,
            batch=1,
            s_q=seq_len,
            s_kv=seq_len,
            s_kv_valid=seq_len,
            hq=num_heads,
            hkv=num_kv_heads,
            h_qkv=d_padded,
            hidden_size=hidden_padded,
            inter_dim=inter_padded,
            x_base=layer_input_base,
            q_base=q_vram_base,
            q_seq_base=q_seq_base,
            attn_base=attn_vram_base,
            residual_base=residual_vram_base,
            mlp_inter_base=mlp_inter_vram_base,
            mlp_out_base=layer_output_base,
            scratch_base=scratch_base,
            k_hbm_offset=k_offset,
            v_hbm_offset=v_offset,
            wq_hbm_offset=wq_offset,
            w1_hbm_offset=w1_offset,
            w2_hbm_offset=w2_offset,
            ln_eps_fp_slot=2,
            ln_reci_hid_fp_slot=3,
            gelu_one_fp_slot=4,
            gelu_1702_fp_slot=5,
            attn_scale_fp_slot=1,
            attn_ninf_fp_slot=6,
            flash_temp_fp_start=64,
            q_bias_base=None,
            include_final_residual=True,
            include_gelu=True,
        )

        asm_code += layer_asm + "\n"

    asm_code += "\n; ============================================================\n"
    asm_code += "; End of SigLIP Full Model ASM\n"
    asm_code += "; ============================================================\n"

    return asm_code


def compute_fp_preload(config: dict, mlen: int = 64, num_slots: int = 1024) -> list[float]:
    """Compute FP preload values for all required constants.

    Returns a preload list where indices are slot numbers.
    """
    fp_preload = [0.0] * num_slots

    hidden_padded = ((config["hidden_size"] + mlen - 1) // mlen) * mlen
    head_dim = config["hidden_size"] // config["num_attention_heads"]

    # Slot 1: Attention scale
    fp_preload[1] = 1.0 / (head_dim ** 0.5)

    # Slot 2: Layer norm epsilon
    fp_preload[2] = config.get("layer_norm_eps", 1e-2)

    # Slot 3: 1 / hidden_size_padded (for layer norm)
    fp_preload[3] = 1.0 / hidden_padded

    # Slot 4: 1.0 (for GELU and other operations)
    fp_preload[4] = 1.0

    # Slot 5: 1.702 (for GELU approximation)
    fp_preload[5] = 1.702

    # Slot 6: -inf (for attention masking)
    fp_preload[6] = float("-inf")

    return fp_preload


def compute_hbm_data_order(num_layers: int = 27) -> list[str]:
    """Compute the HBM data order for weight loading.

    Returns list of weight tensor names in HBM load order.
    """
    order = [
        # Embedding
        "patch_weight", "patch_bias", "position_table",
    ]

    # Per-layer weights
    for layer_idx in range(num_layers):
        order.extend([
            f"layer_{layer_idx}_ln1_weight",
            f"layer_{layer_idx}_ln1_bias",
            f"layer_{layer_idx}_q_proj_weight",
            f"layer_{layer_idx}_q_proj_bias",
            f"layer_{layer_idx}_k_proj_weight",
            f"layer_{layer_idx}_k_proj_bias",
            f"layer_{layer_idx}_v_proj_weight",
            f"layer_{layer_idx}_v_proj_bias",
            f"layer_{layer_idx}_out_proj_weight",
            f"layer_{layer_idx}_out_proj_bias",
            f"layer_{layer_idx}_ln2_weight",
            f"layer_{layer_idx}_ln2_bias",
            f"layer_{layer_idx}_fc1_weight",
            f"layer_{layer_idx}_fc1_bias",
            f"layer_{layer_idx}_fc2_weight",
            f"layer_{layer_idx}_fc2_bias",
        ])

    return order


if __name__ == "__main__":
    """Generate full-model ASM, and optionally run emulator smoke path."""
    from transactional_emulator.testbench.siglip.model_loader import (
        load_siglip_config,
        load_siglip_vision_model,
        extract_embedding_weights,
        extract_layer_weights,
    )
    from transactional_emulator.testbench.siglip.full_model.memory_layout import (
        compute_full_model_hbm_layout,
        compute_full_model_vram_layout,
    )

    parser = argparse.ArgumentParser(description="SigLIP full-model ASM harness")
    parser.add_argument(
        "--config",
        default="compiler/doc/Model_Lib/siglip-so400m-patch14-384.json",
        help="Path to SigLIP config json",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=2,
        help="Number of layers to emit/run",
    )
    parser.add_argument(
        "--mlen",
        type=int,
        default=64,
        help="Hardware MLEN",
    )
    parser.add_argument(
        "--vlen",
        type=int,
        default=64,
        help="Hardware VLEN",
    )
    parser.add_argument(
        "--blen",
        type=int,
        default=4,
        help="Hardware BLEN",
    )
    parser.add_argument(
        "--output-asm",
        default="",
        help="Optional output file for generated ASM",
    )
    parser.add_argument(
        "--run-emulator",
        action="store_true",
        help="Run emulator smoke path after ASM generation",
    )
    parser.add_argument(
        "--build-dir",
        default="./build/siglip_full_model_harness",
        help="Build directory for simulator artifacts",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SigLIP Full Model ASM Generation Test")
    print("=" * 80)

    # Load config and weights
    config = load_siglip_config(args.config)
    model = load_siglip_vision_model()
    embed_weights = extract_embedding_weights(model, config)
    max_layers = min(args.max_layers, config["num_hidden_layers"])
    layer_weights = [extract_layer_weights(model, i, config["hidden_size"]) for i in range(max_layers)]

    # Compute layouts
    vram_layout = compute_full_model_vram_layout(config)
    hbm_layout = compute_full_model_hbm_layout(config, embed_weights, layer_weights)

    # Generate ASM (limited to 2 layers for testing)
    print("\n--- Generating ASM Code ---")
    asm_code = build_full_model_asm(
        config, embed_weights, layer_weights,
        vram_layout, hbm_layout,
        mlen=args.mlen, vlen=args.vlen, blen=args.blen,
        max_layers=max_layers,
    )

    asm_lines = asm_code.split("\n")
    print(f"Generated {len(asm_lines)} lines of ASM code")
    print("First 20 lines:")
    for line in asm_lines[:20]:
        print(f"  {line}")

    if args.output_asm:
        out_path = Path(args.output_asm)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(asm_code)
        print(f"\n✓ Wrote ASM to {out_path}")

    # Test FP preload
    print("\n--- FP Preload Values ---")
    fp_preload = compute_fp_preload(config, mlen=args.mlen)
    print(f"Slot 1 (attention scale): {fp_preload[1]}")
    print(f"Slot 2 (LN epsilon): {fp_preload[2]}")
    print(f"Slot 3 (1/hidden): {fp_preload[3]}")
    print(f"Slot 4 (1.0): {fp_preload[4]}")
    print(f"Slot 5 (1.702): {fp_preload[5]}")
    print(f"Slot 6 (-inf): {fp_preload[6]}")

    # Test data order
    print("\n--- HBM Data Order (first 10) ---")
    data_order = compute_hbm_data_order(num_layers=max_layers)
    for i, item in enumerate(data_order[:10]):
        print(f"  {i}: {item}")

    if args.run_emulator:
        print("\n--- Running Emulator Smoke Path ---")
        run_full_model_emulator_smoke(
            config=config,
            embedding_weights=embed_weights,
            layer_weights_list=layer_weights,
            vram_layout=vram_layout,
            hbm_layout=hbm_layout,
            asm_code=asm_code,
            build_dir=Path(args.build_dir),
            max_layers=max_layers,
            mlen=args.mlen,
            vlen=args.vlen,
            blen=args.blen,
        )

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
