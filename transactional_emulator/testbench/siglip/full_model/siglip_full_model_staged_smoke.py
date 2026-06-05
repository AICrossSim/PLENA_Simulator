"""Run staged SigLIP full-model smoke passes across several encoder depths.

This is a multi-run wrapper around the monolithic ASM harness. It reuses the
same full-model build and smoke machinery, but executes several requested layer
depths in sequence and writes a consolidated staged summary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
import time
from statistics import mean

from transactional_emulator.testbench.siglip.model_loader import (
    load_siglip_config,
    load_siglip_vision_model,
    extract_embedding_weights,
    extract_layer_weights,
    extract_final_ln_weights,
)
from transactional_emulator.testbench.siglip.full_model.runtime_prep import (
    build_runtime_repacked_model,
)
from transactional_emulator.testbench.siglip.full_model.siglip_full_model_asm_harness import (
    build_full_model_asm,
    build_full_model_streaming_asm,
    prepare_runtime_model_and_vram_layout,
    run_full_model_emulator_smoke,
)
from transactional_emulator.testbench.siglip.full_model.memory_layout import (
    compute_full_model_hbm_layout,
)


@dataclass(frozen=True)
class StageResult:
    layers: int
    build_dir: str
    asm_path: str
    emulator_log: str
    timing_json: str


def _parse_stages(stages_raw: str) -> list[int]:
    stages: list[int] = []
    for token in stages_raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError(f"Stage depth must be >= 0, got {value}")
        stages.append(value)

    if not stages:
        raise ValueError("At least one stage depth must be provided")

    return sorted(set(stages))


def _avg_layer_count(stage_depths: list[int]) -> float | None:
    if not stage_depths:
        return None
    return float(mean(stage_depths))


def _prepare_runtime_layer_weights_for_layout(
    runtime_layer_weights: list,
    *,
    layout_layers: int,
    model,
    config: dict,
    embedding_weights: dict,
    mlen: int,
    seq_len_kernel: int,
) -> list:
    """Return runtime layer weights sized for layout allocation.

    HBM layout allocation may target more layers than the staged run depth
    when fixed-memory mode is enabled. In that case we extend the runtime
    layer list by repeating a compatible template runtime layer.
    """
    runtime_layer_weights_for_layout = list(runtime_layer_weights)
    if layout_layers <= len(runtime_layer_weights_for_layout):
        return runtime_layer_weights_for_layout

    if runtime_layer_weights_for_layout:
        template_layer = runtime_layer_weights_for_layout[-1]
    elif layout_layers > 0:
        template_src = extract_layer_weights(model, 0)
        _, _, template_runtime_layers = build_runtime_repacked_model(
            config=config,
            embedding_weights=embedding_weights,
            layer_weights_list=[template_src],
            mlen=mlen,
            seq_len_kernel=seq_len_kernel,
        )
        template_layer = template_runtime_layers[0]
    else:
        return runtime_layer_weights_for_layout

    missing_layers = layout_layers - len(runtime_layer_weights_for_layout)
    runtime_layer_weights_for_layout.extend([template_layer] * missing_layers)
    return runtime_layer_weights_for_layout


def run_staged_harness_smoke(
    config_path: str,
    output_dir: Path,
    stages: list[int],
    mlen: int,
    vlen: int,
    blen: int,
    fixed_memory_max_layers: int | None = None,
    compact_artifacts: bool = True,
    full_flow_embedding: bool = False,
    streaming: bool = False,
    apply_post_layernorm: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SigLIP Staged Harness Smoke")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Stages: {stages}")
    print(f"MLEN/VLEN/BLEN: {mlen}/{vlen}/{blen}")

    config = load_siglip_config(config_path)
    config["apply_post_layernorm"] = bool(apply_post_layernorm)
    model = load_siglip_vision_model()
    embedding_weights = extract_embedding_weights(model, config)
    final_ln_weights = extract_final_ln_weights(model) if apply_post_layernorm else None
    if apply_post_layernorm and final_ln_weights is None:
        raise RuntimeError("apply_post_layernorm was requested but the loaded model has no final layer norm")

    max_stage = max(stages) if stages else 0
    layout_layers = max_stage
    if fixed_memory_max_layers is not None:
        if fixed_memory_max_layers < max_stage:
            raise ValueError(
                "fixed_memory_max_layers must be >= largest stage depth "
                f"({max_stage}), got {fixed_memory_max_layers}"
            )
        layout_layers = fixed_memory_max_layers

    print(f"Memory layout layers: {layout_layers}")
    print(f"Compact artifacts: {compact_artifacts}")
    print(f"Streaming mode: {streaming}")

    run_layers = max_stage
    layer_weights_list = [extract_layer_weights(model, idx) for idx in range(run_layers)]

    (
        _seq_len_valid,
        seq_len_kernel,
        runtime_config,
        runtime_embed_weights,
        runtime_layer_weights,
        vram_layout,
    ) = prepare_runtime_model_and_vram_layout(
        config=config,
        embedding_weights=embedding_weights,
        layer_weights_list=layer_weights_list,
        mlen=mlen,
        layout_layers=layout_layers,
        final_ln_weights=final_ln_weights,
        include_out_proj_bias_buffers=True,
        include_mlp_bias_buffers=True,
    )
    runtime_layer_weights_for_layout = _prepare_runtime_layer_weights_for_layout(
        runtime_layer_weights,
        layout_layers=layout_layers,
        model=model,
        config=config,
        embedding_weights=embedding_weights,
        mlen=mlen,
        seq_len_kernel=seq_len_kernel,
    )

    hbm_layout = compute_full_model_hbm_layout(
        runtime_embed_weights,
        runtime_layer_weights_for_layout,
    )

    stage_results: list[StageResult] = []
    stage_summaries: list[dict] = []

    for max_layers in stages:
        stage_dir = output_dir / f"stage_L{max_layers}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        asm_path = stage_dir / "generated_asm_code.asm"
        stage_start = time.perf_counter()

        print("\n" + "-" * 80)
        print(f"Running staged smoke: layers={max_layers}")
        print("-" * 80)

        build_fn = build_full_model_streaming_asm if streaming else build_full_model_asm
        asm_code = build_fn(
            runtime_config,
            runtime_layer_weights,
            vram_layout,
            hbm_layout,
            mlen=mlen,
            vlen=vlen,
            blen=blen,
            max_layers=max_layers,
            embedding_mode=("asm" if full_flow_embedding else "bypass"),
        )
        asm_path.write_text(asm_code, encoding="utf-8")

        stage_passed = True
        stage_error: str | None = None
        try:
            run_summary = run_full_model_emulator_smoke(
                config=config,
                runtime_config=runtime_config,
                runtime_embedding_weights=runtime_embed_weights,
                layer_weights_list=layer_weights_list,
                runtime_layer_weights_list=runtime_layer_weights,
                vram_layout=vram_layout,
                hbm_layout=hbm_layout,
                asm_code=asm_code,
                build_dir=stage_dir,
                max_layers=max_layers,
                mlen=mlen,
                vlen=vlen,
                blen=blen,
                # Numerical comparison is always on for staged smoke, so golden text
                # output is required by compare_emulator_output.
                write_golden_txt=True,
                embedding_mode=("asm" if full_flow_embedding else "bypass"),
            )
        except Exception as exc:
            stage_passed = False
            stage_error = str(exc)
            print(f"✗ Stage L{max_layers} failed: {stage_error}")
            run_summary = None

        elapsed_s = time.perf_counter() - stage_start

        emulator_log = stage_dir / "emulator.log"
        result = StageResult(
            layers=max_layers,
            build_dir=str(stage_dir),
            asm_path=str(asm_path),
            emulator_log=str(emulator_log),
            timing_json=str(stage_dir / "full_model_harness_timing.json"),
        )
        stage_results.append(result)
        stage_summaries.append(
            {
                "layers": max_layers,
                "build_dir": str(stage_dir),
                "asm_path": str(asm_path),
                "emulator_log": str(emulator_log),
                "timing_json": str(stage_dir / "full_model_harness_timing.json"),
                "harness_timing": (run_summary or {}).get("timing"),
                "elapsed_s": elapsed_s,
                "passed": stage_passed,
                "error": stage_error,
            }
        )

        stage_summary_path = stage_dir / "stage_summary.json"
        stage_summary_path.write_text(
            json.dumps(
                {
                    "layers": max_layers,
                    "passed": stage_passed,
                    "elapsed_s": elapsed_s,
                    "compact_artifacts": compact_artifacts,
                    "layout_layers": layout_layers,
                    "embedding_mode": ("asm" if full_flow_embedding else "bypass"),
                    "timing_json": str(stage_dir / "full_model_harness_timing.json"),
                    "harness_timing": (run_summary or {}).get("timing"),
                    "error": stage_error,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": config_path,
        "stages": stages,
        "layout_layers": layout_layers,
        "compact_artifacts": compact_artifacts,
        "embedding_mode": ("asm" if full_flow_embedding else "bypass"),
        "apply_post_layernorm": bool(apply_post_layernorm),
        "mlen": mlen,
        "vlen": vlen,
        "blen": blen,
        "average_stage_depth": _avg_layer_count(stages),
        "stage_results": stage_summaries,
        "all_passed": all(stage["passed"] for stage in stage_summaries),
    }

    summary_path = output_dir / "staged_harness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("STAGED HARNESS SUMMARY")
    print("=" * 80)
    for stage in stage_summaries:
        print(f"L={stage['layers']}: passed={stage['passed']} asm={stage['asm_path']}")
    print(f"Saved: {summary_path}")

    if not summary["all_passed"]:
        raise RuntimeError("One or more staged smoke runs failed; see staged_harness_summary.json for details")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="SigLIP staged smoke runner across multiple full-model depths")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="compiler/doc/Model_Lib/siglip-so400m-patch14-384.json",
        help="Path to SigLIP config JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="./build/siglip_full_model_staged_smoke",
        help="Directory where staged outputs are written",
    )
    parser.add_argument(
        "--stages",
        default="1,3,6",
        help="Comma-separated layer checkpoints to run",
    )
    parser.add_argument("--mlen", type=int, default=64, help="Hardware MLEN")
    parser.add_argument("--vlen", type=int, default=64, help="Hardware VLEN")
    parser.add_argument("--blen", type=int, default=4, help="Hardware BLEN")
    parser.add_argument(
        "--fixed-memory-max-layers",
        type=int,
        default=None,
        help="Pin memory layout to this layer count even when running shallower stages",
    )
    parser.add_argument(
        "--no-compact-artifacts",
        action="store_true",
        help="Write full golden_result.txt tensors (larger, slower).",
    )
    parser.add_argument(
        "--full-flow-embedding",
        action="store_true",
        help="Run embedding stage in ASM before encoder instead of preload bypass.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming full-model generator (non-persistent inter-layer activations).",
    )
    parser.add_argument(
        "--apply-post-layernorm",
        action="store_true",
        help="Emit terminal post-encoder LayerNorm and compare against post-LN golden output.",
    )
    args = parser.parse_args()

    stages = _parse_stages(args.stages)
    run_staged_harness_smoke(
        config_path=args.config_path,
        output_dir=Path(args.output_dir),
        stages=stages,
        mlen=args.mlen,
        vlen=args.vlen,
        blen=args.blen,
        fixed_memory_max_layers=args.fixed_memory_max_layers,
        compact_artifacts=not args.no_compact_artifacts,
        full_flow_embedding=args.full_flow_embedding,
        streaming=args.streaming,
        apply_post_layernorm=args.apply_post_layernorm,
    )


if __name__ == "__main__":
    main()
