#!/usr/bin/env python3
"""Re-run a compiled Qwen layer for timing-only ``rtl-v1`` validation.

The source build remains immutable. Large compiler inputs are symlinked into a
new run directory, while comparison/config files are copied. The script checks
that decoded BF16 emulator output is bitwise identical to the source run and
records the existing correctness-gate result without changing its thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

import torch

from transactional_emulator.testbench.emulator_runner import (
    _write_comparison_summary,
    compare_emulator_output,
    run_emulator,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT
    / "Workspace/qwen3_32b_transactional_prefetch_sweep/runs"
    / "rtl_v1_validation_20260714/single_point"
)
DEFAULT_OUTPUT = (
    ROOT
    / "Workspace/rtl_v1_latency_validation/qwen3_32b_equal_e4m3_fixed"
)

SYMLINK_INPUTS = (
    "generated_machine_code.mem",
    "generated_asm_code.asm",
    "hbm_for_behave_sim.bin",
    "fp_sram.bin",
    "int_sram.bin",
    "vram_preload.bin",
    "hbm_size.txt",
)
COPY_INPUTS = (
    "plena_settings.toml",
    "comparison_params.json",
    "golden_result.txt",
    "golden_output.pt",
    "tensor_layouts.json",
)


def prepare_run_dir(source: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name in SYMLINK_INPUTS:
        src = source / name
        dst = output / name
        if not src.exists() or dst.exists():
            continue
        dst.symlink_to(src.resolve())
    for name in COPY_INPUTS:
        src = source / name
        dst = output / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    raw = contiguous.view(torch.int16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def scalar_comparison(results: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "mse",
        "mae",
        "max_error",
        "relative_error",
        "relative_match_rate",
        "allclose_match_rate",
        "match_rate",
        "allclose_pass",
        "atol",
        "rtol",
    )
    return {key: results[key] for key in keys if key in results}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-build", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hbm-channels", type=int, default=128)
    parser.add_argument(
        "--keep-vram",
        action="store_true",
        help="Retain the new sparse vram_dump.bin after decoded-output comparison.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source_build.resolve()
    output = args.out_dir.resolve()
    if not (source / "rust_emulator_run_stats.json").exists():
        raise FileNotFoundError(f"source run stats not found under {source}")
    if (output / "rust_emulator_run_stats.json").exists():
        raise FileExistsError(
            f"output run already exists at {output}; choose a fresh --out-dir"
        )
    prepare_run_dir(source, output)

    source_stats = json.loads(
        (source / "rust_emulator_run_stats.json").read_text(encoding="utf-8")
    )
    os.environ["PLENA_SETTINGS_TOML"] = str(output / "plena_settings.toml")
    metrics = run_emulator(
        output,
        hbm_size=int(source_stats["hbm_size_bytes"]),
        hbm_channels=args.hbm_channels,
        threads=args.threads,
        profile_memory=True,
        profile_memory_level="opcode",
        timing_mode="rtl-v1",
        dma_event_trace=output / "dma_event_trace.json",
        require_rtl_validated=False,
    )

    fixed_results, params = compare_emulator_output(output)
    _write_comparison_summary(output, fixed_results, params)
    source_results, _ = compare_emulator_output(source)
    fixed_values = fixed_results["simulated_values"]
    source_values = source_results["simulated_values"]
    output_identical = bool(torch.equal(fixed_values, source_values))
    evidence = {
        "schema_version": 1,
        "scope": "timing-only regression; correctness gate unchanged",
        "source_build": str(source),
        "fixed_build": str(output),
        "source_output_sha256_bf16": tensor_sha256(source_values),
        "fixed_output_sha256_bf16": tensor_sha256(fixed_values),
        "decoded_output_bitwise_identical": output_identical,
        "source_comparison": scalar_comparison(source_results),
        "fixed_comparison": scalar_comparison(fixed_results),
        "comparison_metrics_identical": scalar_comparison(source_results)
        == scalar_comparison(fixed_results),
        "timing_mode": metrics.get("timing_mode"),
        "rtl_validation": metrics.get("rtl_validation"),
        "dma_event_trace": str(output / "dma_event_trace.json"),
    }
    (output / "functional_regression.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    if not args.keep_vram:
        (output / "vram_dump.bin").unlink(missing_ok=True)
        # ``run_emulator`` writes the production dump in its own working
        # directory before copying it into ``output``. The decoded tensor and
        # both hashes above are sufficient timing-regression evidence, so do
        # not leave another multi-GiB copy behind after the comparison.
        for name in ("vram_dump.bin", "mram_dump.bin", "fpsram_dump.bin"):
            (ROOT / "transactional_emulator" / name).unlink(missing_ok=True)
    print(json.dumps(evidence, indent=2))
    return 0 if output_identical and evidence["comparison_metrics_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
