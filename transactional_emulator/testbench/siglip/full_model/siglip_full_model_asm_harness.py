"""Generate and optionally run the monolithic SigLIP full-model ASM harness.

This is the single-run ASM entry point for the full-model flow. It builds one
monolithic program for the requested encoder depth, prepares the corresponding
runtime layout, and can optionally execute the emulator smoke path.
"""

import argparse
from pathlib import Path
import time

from transactional_emulator.testbench.config_utils import update_plena_config
from transactional_emulator.testbench.siglip.full_model.diagnostics import print_layer0_stage_diagnostics
from transactional_emulator.testbench.siglip.full_model.runtime_prep import (
    prepare_runtime_model_and_vram_layout,
)
from transactional_emulator.testbench.siglip.full_model.smoke_pipeline import (
    prepare_smoke_runtime_inputs,
    prepare_smoke_case_artifacts_and_compare_params,
    validate_payloads_and_execute_smoke,
)
from compiler.asm_templates.siglip import (
    build_full_model_asm,
    build_full_model_streaming_asm,
    compute_hbm_data_order,
)


def run_full_model_emulator_smoke(
    config: dict,
    runtime_config: dict,
    runtime_embedding_weights: dict,
    layer_weights_list: list,
    runtime_layer_weights_list: list,
    vram_layout: dict,
    hbm_layout: tuple,
    asm_code: str,
    build_dir: Path,
    max_layers: int,
    mlen: int,
    vlen: int,
    blen: int,
    write_golden_txt: bool = True,
    enforce_numerical_parity: bool = False,
    embedding_mode: str = "bypass",
    skip_numerical_compare: bool = False,
) -> None:
    """Run one generated full-model ASM program in the emulator."""
    smoke_t0 = time.perf_counter()

    prep = prepare_smoke_runtime_inputs(
        config=config,
        runtime_config=runtime_config,
        runtime_embedding_weights=runtime_embedding_weights,
        layer_weights_list=layer_weights_list,
        runtime_layer_weights_list=runtime_layer_weights_list,
        vram_layout=vram_layout,
        max_layers=max_layers,
        mlen=mlen,
        skip_numerical_compare=skip_numerical_compare,
        embedding_mode=embedding_mode,
    )

    num_layers = prep.num_layers
    seq_len = prep.seq_len
    seq_len_valid = prep.seq_len_valid
    hidden_size = prep.hidden_size
    runtime_hidden_size = prep.runtime_hidden_size
    layer_outputs = prep.layer_outputs
    final_compare_golden = prep.final_compare_golden
    input_tensor = prep.input_tensor
    data_order = prep.data_order
    vram_preload = prep.vram_preload

    # Keep emulator hardware config in sync with generated assembly assumptions.
    update_plena_config(vlen=vlen, mlen=mlen, blen=blen, verbose=False)

    fp_preload = compute_fp_preload(config, mlen=mlen)
    # Runtime LN stages operate on expanded hidden width.
    fp_preload[3] = 1.0 / float(runtime_hidden_size)

    build_dir, final_output_base = prepare_smoke_case_artifacts_and_compare_params(
        build_dir=build_dir,
        vram_layout=vram_layout,
        num_layers=num_layers,
        hbm_layout=hbm_layout,
        asm_code=asm_code,
        final_compare_golden=final_compare_golden,
        fp_preload=fp_preload,
        mlen=mlen,
        vlen=vlen,
        seq_len=seq_len,
        seq_len_valid=seq_len_valid,
        hidden_size=hidden_size,
        runtime_hidden_size=runtime_hidden_size,
        input_tensor=input_tensor,
        data_order=data_order,
        vram_preload=vram_preload,
        write_golden_txt=write_golden_txt,
    )

    results = validate_payloads_and_execute_smoke(
        build_dir=build_dir,
        data_order=data_order,
        skip_numerical_compare=skip_numerical_compare,
        config=config,
        runtime_config=runtime_config,
        vram_layout=vram_layout,
        runtime_layer_weights_list=runtime_layer_weights_list,
        layer_outputs=layer_outputs,
        mlen=mlen,
        blen=blen,
        diagnostics_fn=print_layer0_stage_diagnostics,
    )
    print(f"  Build dir: {build_dir}")
    print(f"  Final output base (elements): {final_output_base}")
    if (not skip_numerical_compare) and enforce_numerical_parity and not results["allclose_pass"]:
        raise RuntimeError(
            "Full-model harness numerical comparison failed: "
            f"match_rate={results['match_rate']:.3f}% max_error={results['max_error']:.6f}"
        )

    print(f"[smoke] total_runtime={time.perf_counter() - smoke_t0:.2f}s")


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


if __name__ == "__main__":
    """Generate the full-model ASM program and optionally run one smoke pass."""
    from transactional_emulator.testbench.siglip.model_loader import (
        load_siglip_config,
        load_siglip_vision_model,
        extract_embedding_weights,
        extract_layer_weights,
    )
    from transactional_emulator.testbench.siglip.full_model.memory_layout import (
        compute_full_model_hbm_layout,
    )

    parser = argparse.ArgumentParser(description="SigLIP monolithic ASM harness for a single full-model run")
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
        default=128,
        help="Hardware MLEN",
    )
    parser.add_argument(
        "--vlen",
        type=int,
        default=128,
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
        "--streaming",
        action="store_true",
        help="Use streaming full-model generator (non-persistent inter-layer activations).",
    )
    parser.add_argument(
        "--full-flow-embedding",
        action="store_true",
        help="Run embedding stage in ASM (patch proj + pos add) instead of preload bypass.",
    )
    parser.add_argument(
        "--build-dir",
        default="./build/siglip_full_model_asm_harness",
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
    layer_weights = [extract_layer_weights(model, i) for i in range(max_layers)]
    (
        seq_len_valid,
        seq_len_kernel,
        runtime_config,
        runtime_embed_weights,
        runtime_layer_weights,
        vram_layout,
    ) = prepare_runtime_model_and_vram_layout(
        config=config,
        embedding_weights=embed_weights,
        layer_weights_list=layer_weights,
        mlen=args.mlen,
        layout_layers=max_layers,
        include_out_proj_bias_buffers=False,
        include_mlp_bias_buffers=False,
    )
    hbm_layout = compute_full_model_hbm_layout(runtime_embed_weights, runtime_layer_weights)

    # Generate ASM (limited to 2 layers for testing)
    print("\n--- Generating ASM Code ---")
    embedding_mode = "asm" if args.full_flow_embedding else "bypass"
    print(f"Embedding mode: {embedding_mode}")
    build_fn = build_full_model_streaming_asm if args.streaming else build_full_model_asm
    asm_code = build_fn(
        runtime_config,
        runtime_layer_weights,
        vram_layout,
        hbm_layout,
        mlen=args.mlen,
        vlen=args.vlen,
        blen=args.blen,
        max_layers=max_layers,
        embedding_mode=embedding_mode,
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
            runtime_config=runtime_config,
            runtime_embedding_weights=runtime_embed_weights,
            layer_weights_list=layer_weights,
            runtime_layer_weights_list=runtime_layer_weights,
            vram_layout=vram_layout,
            hbm_layout=hbm_layout,
            asm_code=asm_code,
            build_dir=Path(args.build_dir),
            max_layers=max_layers,
            mlen=args.mlen,
            vlen=args.vlen,
            blen=args.blen,
            enforce_numerical_parity=True,
            embedding_mode=embedding_mode,
        )

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
