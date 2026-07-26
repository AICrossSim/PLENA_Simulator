#!/usr/bin/env python3
"""Tiny end-to-end Qwen3-MoE static-index transactional test."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from compiler.aten.plena_frontend import compile_native_hf_decoder
from compiler.aten.tests.test_qwen3_moe_compiler import make_tiny_qwen3_moe
from transactional_emulator.testbench.emulator_runner import emulate_from_result


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "Workspace" / "qwen3_moe_static_tiny",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--no-state-dumps",
        action="store_true",
        help="Skip large MRAM/VRAM state dumps during numerical validation.",
    )
    parser.add_argument(
        "--stage-checkpoints",
        action="store_true",
        help="Copy native decoder stage outputs into persistent VRAM for A/B diagnosis.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=("ideal-ii1", "legacy", "rtl-v1"),
        default="ideal-ii1",
    )
    parser.add_argument(
        "--vector-scalar-schedule",
        choices=("rtl-v4", "rtl-v3", "rtl-v2", "compiler-v1", "legacy"),
        default="rtl-v3",
        help="Select the native Vector/Scalar lowering for numerical A/B tests.",
    )
    parser.add_argument(
        "--selector-schedule",
        choices=("hoisted-v1", "legacy"),
        default="legacy",
    )
    parser.add_argument(
        "--reduction-output-mode",
        choices=("overwrite-v1", "accumulate-v1"),
        default="accumulate-v1",
    )
    args = parser.parse_args()
    args.build_dir = args.build_dir.resolve()

    settings = ROOT / "plena_settings.toml"
    os.environ.setdefault("PLENA_SETTINGS_TOML", str(settings))
    result = compile_native_hf_decoder(
        make_tiny_qwen3_moe(),
        seq_len=8,
        batch_size=1,
        num_layers=1,
        mlen=64,
        blen=4,
        hlen=16,
        broadcast_amount=4,
        mram_tile_capacity=4,
        seed=7,
        reference_backend="scheduled",
        moe_routing_mode="static-indices",
        vector_scalar_schedule=args.vector_scalar_schedule,
        selector_schedule=args.selector_schedule,
        reduction_output_mode=args.reduction_output_mode,
        stage_checkpoints=args.stage_checkpoints,
    )
    args.build_dir.mkdir(parents=True, exist_ok=True)
    (args.build_dir / "compile_info.json").write_text(
        __import__("json").dumps(result["info"], indent=2, sort_keys=True)
    )
    (args.build_dir / "stage_checkpoints.json").write_text(
        __import__("json").dumps(
            result.get("stage_checkpoints", {}), indent=2, sort_keys=True
        )
    )
    if args.compile_only:
        (args.build_dir / "generated_asm_code.asm").write_text(result["isa"])
        return

    emulate_from_result(
        result,
        args.build_dir,
        "qwen3_moe_static_tiny",
        mlen=64,
        blen=4,
        vlen=64,
        threads=1,
        hbm_channels=8,
        timing_mode=args.timing_mode,
        profile_memory=True,
        profile_memory_level="opcode",
        no_state_dumps=args.no_state_dumps,
    )


if __name__ == "__main__":
    main()
