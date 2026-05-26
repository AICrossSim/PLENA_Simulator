"""Run staged SigLIP full-model harness smoke tests at increasing depth.

This wrapper keeps MXFP mismatch out of the critical path and focuses on
harness/runtime validation. It runs the existing full-model ASM smoke path at
multiple depths (default 1, 3, 6 layers) and writes a consolidated summary.
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
)
from transactional_emulator.testbench.siglip.full_model.siglip_full_model_harness import (
    build_runtime_repacked_model,
    build_full_model_asm,
    run_full_model_emulator_smoke,
)
from transactional_emulator.testbench.siglip.full_model.memory_layout import (
    compute_full_model_hbm_layout,
)
from transactional_emulator.testbench.siglip.utils.core import align_up


@dataclass(frozen=True)
class StageResult:
    layers: int
    build_dir: str
    asm_path: str
    emulator_log: str


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
    fast_smoke: bool = False,
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
    model = load_siglip_vision_model()
    embedding_weights = extract_embedding_weights(model, config)

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
    print(f"Fast smoke: {fast_smoke}")

    run_layers = max_stage
    layer_weights_list = [extract_layer_weights(model, idx, config["hidden_size"]) for idx in range(run_layers)]

    seq_len_valid = int((config["image_size"] // config["patch_size"]) ** 2)
    seq_len_kernel = align_up(seq_len_valid, mlen)
    runtime_config, runtime_embed_weights, runtime_layer_weights = build_runtime_repacked_model(
        config=config,
        embedding_weights=embedding_weights,
        layer_weights_list=layer_weights_list,
        mlen=mlen,
        seq_len_kernel=seq_len_kernel,
    )
    runtime_config["seq_len_valid"] = seq_len_valid

    hidden_runtime = int(runtime_config["hidden_size"])
    embedding_base = 0
    embedding_size = seq_len_kernel * hidden_runtime
    layer_bases = {}
    layer_sizes = {}
    cur = embedding_base + embedding_size
    for i in range(layout_layers):
        layer_bases[i] = cur
        layer_sizes[i] = seq_len_kernel * hidden_runtime
        cur += layer_sizes[i]
    q_bias_bases = {}
    for i in range(layout_layers):
        q_bias_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime
    ln1_weight_bases = {}
    ln1_bias_bases = {}
    ln2_weight_bases = {}
    ln2_bias_bases = {}
    out_bias_bases = {}
    fc1_bias_bases = {}
    fc2_bias_bases = {}
    for i in range(layout_layers):
        ln1_weight_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime
        ln1_bias_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime
        ln2_weight_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime
        ln2_bias_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime
        out_bias_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime
        fc1_bias_bases[i] = cur
        cur += seq_len_kernel * int(runtime_config["intermediate_size"])
        fc2_bias_bases[i] = cur
        cur += seq_len_kernel * hidden_runtime

    patch_size = int(runtime_config["patch_size"])
    num_channels = int(runtime_config["num_channels"])
    in_features = num_channels * patch_size * patch_size
    aligned_in_features = align_up(in_features, mlen)
    embedding_patch_input_base = cur
    cur += seq_len_kernel * aligned_in_features
    embedding_patch_bias_base = cur
    cur += seq_len_kernel * hidden_runtime
    embedding_position_base = cur
    cur += seq_len_kernel * hidden_runtime

    vram_layout = {
        "seq_len": seq_len_kernel,
        "hidden_size": hidden_runtime,
        "embedding_base": embedding_base,
        "embedding_size": embedding_size,
        "layer_bases": layer_bases,
        "layer_sizes": layer_sizes,
        "q_bias_bases": q_bias_bases,
        "ln1_weight_bases": ln1_weight_bases,
        "ln1_bias_bases": ln1_bias_bases,
        "ln2_weight_bases": ln2_weight_bases,
        "ln2_bias_bases": ln2_bias_bases,
        "out_bias_bases": out_bias_bases,
        "fc1_bias_bases": fc1_bias_bases,
        "fc2_bias_bases": fc2_bias_bases,
        "embedding_patch_input_base": embedding_patch_input_base,
        "embedding_patch_bias_base": embedding_patch_bias_base,
        "embedding_position_base": embedding_position_base,
        "total_vram_elements": cur,
        "total_vram_mb": cur * 2 / (1024 * 1024),
    }
    runtime_layer_weights_for_layout = list(runtime_layer_weights)
    if layout_layers > len(runtime_layer_weights_for_layout):
        if runtime_layer_weights_for_layout:
            template_layer = runtime_layer_weights_for_layout[-1]
            for _ in range(layout_layers - len(runtime_layer_weights_for_layout)):
                runtime_layer_weights_for_layout.append(template_layer)
        elif layout_layers > 0:
            template_src = extract_layer_weights(model, 0, config["hidden_size"])
            _, _, template_runtime_layers = build_runtime_repacked_model(
                config=config,
                embedding_weights=embedding_weights,
                layer_weights_list=[template_src],
                mlen=mlen,
                seq_len_kernel=seq_len_kernel,
            )
            template_layer = template_runtime_layers[0]
            for _ in range(layout_layers):
                runtime_layer_weights_for_layout.append(template_layer)

    hbm_layout = compute_full_model_hbm_layout(
        runtime_config,
        runtime_embed_weights,
        runtime_layer_weights_for_layout,
    )

    stage_results: list[StageResult] = []
    stage_summaries: list[dict] = []

    for stage_idx, max_layers in enumerate(stages):
        stage_dir = output_dir / f"stage_L{max_layers}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        asm_path = stage_dir / "generated_asm_code.asm"
        stage_start = time.perf_counter()

        print("\n" + "-" * 80)
        print(f"Running staged smoke: layers={max_layers}")
        print("-" * 80)

        asm_code = build_full_model_asm(
            runtime_config,
            runtime_embed_weights,
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
            run_full_model_emulator_smoke(
                config=config,
                runtime_config=runtime_config,
                embedding_weights=embedding_weights,
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
                write_golden_txt=(not compact_artifacts) or (not fast_smoke),
                enforce_numerical_parity=(not fast_smoke),
                embedding_mode=("asm" if full_flow_embedding else "bypass"),
                skip_numerical_compare=fast_smoke,
            )
        except Exception as exc:
            stage_passed = False
            stage_error = str(exc)
            print(f"✗ Stage L{max_layers} failed: {stage_error}")

        elapsed_s = time.perf_counter() - stage_start

        emulator_log = stage_dir / "emulator.log"
        result = StageResult(
            layers=max_layers,
            build_dir=str(stage_dir),
            asm_path=str(asm_path),
            emulator_log=str(emulator_log),
        )
        stage_results.append(result)
        stage_summaries.append(
            {
                "layers": max_layers,
                "build_dir": str(stage_dir),
                "asm_path": str(asm_path),
                "emulator_log": str(emulator_log),
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
                    "fast_smoke": fast_smoke,
                    "layout_layers": layout_layers,
                    "embedding_mode": ("asm" if full_flow_embedding else "bypass"),
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
        "fast_smoke": fast_smoke,
        "embedding_mode": ("asm" if full_flow_embedding else "bypass"),
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
    parser = argparse.ArgumentParser(description="SigLIP staged harness smoke runner")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="compiler/doc/Model_Lib/siglip-so400m-patch14-384.json",
        help="Path to SigLIP config JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="./build/siglip_staged_harness",
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
        "--fast-smoke",
        action="store_true",
        help="Skip numerical compare to speed up smoke execution.",
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
        fast_smoke=args.fast_smoke,
    )


if __name__ == "__main__":
    main()
