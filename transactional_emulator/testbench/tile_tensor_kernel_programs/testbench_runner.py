"""Shared testbench runner for TileTensorProgram kernel rewrites."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

from compiler.sim_env_utils import create_mem_for_sim
from tile_tensor_test_helper import build_input_feed, stage_input_tensor_for_stride_compare
from transactional_emulator.tools.create_sim_env import create_sim_env


def emit_single_output_testbench(
    *,
    prog,
    out_buf,
    input_tensors: Dict[str, object],
    golden_output,
    asm_name: str,
    artifact_prefix: str,
    build_dir: str | Path,
    fp_preload_min_size: int = 0,
) -> None:
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    comparison_params = stage_input_tensor_for_stride_compare(out_buf)
    staging_isa = comparison_params.pop("staging_isa", "")
    gen_code = prog.compile()
    if staging_isa:
        gen_code = gen_code.rstrip() + "\n\n" + staging_isa

    input_feed, input_order = build_input_feed(prog, input_tensors)
    fp_preload = prog.build_fp_preload(min_size=fp_preload_min_size) if fp_preload_min_size > 0 else None

    create_sim_env(
        input_tensor=input_feed,
        generated_code=gen_code,
        golden_result={"original_output": golden_output},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=asm_name,
        data=None,
        specified_data_order=input_order,
        build_path=build_dir,
    )

    with open(build_dir / "comparison_params.json", "w", encoding="utf-8") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / f"{artifact_prefix}_generated_asm_code.asm", "w", encoding="utf-8") as f:
        f.write(gen_code)

    prog.write_operation_report(build_dir / f"{artifact_prefix}_operation_report.txt")

    if np is not None:
        np.save(
            build_dir / f"{artifact_prefix}_golden_fp32.npy",
            golden_output.detach().cpu().numpy().astype(np.float32),
        )
