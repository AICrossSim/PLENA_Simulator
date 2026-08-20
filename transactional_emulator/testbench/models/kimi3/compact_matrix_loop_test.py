"""Rust numerical proof for the compact row-major decode GEMM.

The shape deliberately exceeds MRAM capacity in K and has several N tiles:
``[1, 320] @ [320, 384]`` at MLEN=64. This exercises the hardware N loop,
MRAM reuse, and BF16 partial-sum accumulation used by full Kimi decode.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import setup_hw
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.gpt_oss_testkit import (
    _comparison_params_for,
    _exact_mxfp8_tensor,
    _linear_projection_golden,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


MLEN = 64
BLEN = 4


def build_and_run(build_dir: Path, *, no_run: bool = False) -> dict:
    build_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(mlen=MLEN, vlen=None, blen=BLEN, hlen=None)
    hw = setup_hw(args, build_dir)

    hidden = 5 * MLEN
    out_features = 6 * MLEN
    x = _exact_mxfp8_tensor((1, hidden), stride=1)
    x_storage = torch.zeros(BLEN, hidden, dtype=torch.bfloat16)
    x_storage[0] = x[0]
    weight = _exact_mxfp8_tensor(
        (hidden, out_features),
        stride=3,
        offset=1,
    )
    golden = _linear_projection_golden(
        x,
        weight,
        mlen=MLEN,
        hbm_input=True,
    )

    prog = PlenaCompiler(
        mlen=MLEN,
        blen=BLEN,
        real_data_ratio=hw.real_data_ratio,
        compact_matrix_loops=True,
    )
    x_input = prog.input(
        "X",
        shape=(1, hidden),
        physical_shape=(BLEN, hidden),
    )
    weight_input = prog.input(
        "W",
        shape=(hidden, out_features),
        physical_shape=(hidden, out_features),
    )
    x_vram = prog.load_batch(x_input, name="X")
    output = prog.linear_projection(x_vram, weight_input, name="compact_output")
    assembly = prog.compile()

    input_tensors = {"X": x_storage, "W": weight}
    create_sim_env(
        input_tensors,
        assembly,
        {"original_output": golden},
        [0.0] * 16,
        build_dir=str(build_dir),
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="kimi3_compact_matrix_loop",
        data=None,
        specified_data_order=["X", "W"],
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs={name: prog.get_hbm_layout(name).hbm_base_addr for name in input_tensors},
    )

    output_addr = prog.get_vram_addr(output.name)
    comparison = _comparison_params_for(
        output,
        rows=1,
        hidden=out_features,
        mlen=MLEN,
        golden=golden,
    )
    comparison.update({"atol": 0.0, "rtol": 0.0})
    (build_dir / "comparison_params.json").write_text(json.dumps(comparison, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(assembly)

    result = {
        "instruction_lines": sum(
            bool(line.strip()) and not line.lstrip().startswith(";") for line in assembly.splitlines()
        ),
        "output_vram_addr": output_addr,
        "ran": not no_run,
    }
    if not no_run:
        result["metrics"] = run_and_assert(
            build_dir,
            "kimi3_compact_matrix_loop",
            mlen=MLEN,
            blen=BLEN,
        )
    (build_dir / "compact_matrix_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "compact_matrix_loop",
    )
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_and_run(args.build_dir, no_run=args.no_run), indent=2))


if __name__ == "__main__":
    main()
