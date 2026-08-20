"""Rust numerical proof for compact BF16 stream-K Matrix accumulation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import tomlkit

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import setup_hw
from transactional_emulator.testbench.aten.golden import quantize_to_mxfp
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.gpt_oss_testkit import (
    _comparison_params_for,
    _exact_mxfp8_tensor,
)
from transactional_emulator.testbench.layout_utils import infer_hbm_tensor_layouts
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


MLEN = 64
BLEN = 4


def _set_matrix_kv_plain_bf16() -> None:
    settings = Path(os.environ["PLENA_SETTINGS_TOML"])
    with settings.open() as stream:
        config = tomlkit.load(stream)
    for mode in ("TRANSACTIONAL", "ANALYTIC"):
        precision = config[mode]["PRECISION"]
        precision["HBM_M_KV_TYPE"] = tomlkit.table()
        precision["HBM_M_KV_TYPE"]["format"] = "Plain"
        precision["HBM_M_KV_TYPE"]["DATA_TYPE"] = tomlkit.table()
        precision["HBM_M_KV_TYPE"]["DATA_TYPE"].update({"type": "Fp", "sign": True, "exponent": 8, "mantissa": 7})
    with settings.open("w") as stream:
        tomlkit.dump(config, stream)


def _bf16_layout(tensor: torch.Tensor) -> dict[str, object]:
    rows, cols = tensor.shape
    return {
        "source_shape": [rows, cols],
        "storage_shape": [rows, cols],
        "source_rows": rows,
        "storage_rows": rows,
        "source_row_elements": cols,
        "storage_row_elements": cols,
        "precision": "HBM_M_KV_TYPE",
    }


def build_and_run(build_dir: Path, *, no_run: bool = False) -> dict:
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(
        argparse.Namespace(mlen=MLEN, vlen=None, blen=BLEN, hlen=None),
        build_dir,
    )
    _set_matrix_kv_plain_bf16()

    hidden = 5 * MLEN
    out_features = 2 * MLEN
    x = _exact_mxfp8_tensor((1, hidden), stride=1)
    x_storage = torch.zeros(BLEN, hidden, dtype=torch.bfloat16)
    x_storage[0] = x[0]
    weight = _exact_mxfp8_tensor(
        (hidden, out_features),
        stride=2,
        offset=1,
    ).to(torch.bfloat16)
    golden = torch.matmul(quantize_to_mxfp(x).float(), weight.float()).to(torch.bfloat16)

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
        "W_BF16",
        shape=(hidden, out_features),
        physical_shape=(hidden, out_features),
        real_data_ratio=2.0,
    )
    x_vram = prog.load_batch(x_input, name="X")
    output = prog.linear_projection_bf16_stream_k_accum(
        x_vram,
        weight_input,
        name="compact_router_logits",
    )
    assembly = prog.compile()

    input_tensors = {"X": x_storage, "W_BF16": weight}
    layouts = infer_hbm_tensor_layouts(input_tensors)
    layouts["W_BF16"] = _bf16_layout(weight)
    create_sim_env(
        input_tensors,
        assembly,
        {"original_output": golden},
        [0.0] * 16,
        build_dir=str(build_dir),
        tensor_layouts=layouts,
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="kimi3_compact_stream_k",
        data=None,
        specified_data_order=["X", "W_BF16"],
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=layouts,
        hbm_addrs={name: prog.get_hbm_layout(name).hbm_base_addr for name in input_tensors},
    )

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
        "ran": not no_run,
    }
    if not no_run:
        result["metrics"] = run_and_assert(
            build_dir,
            "kimi3_compact_stream_k",
            mlen=MLEN,
            blen=BLEN,
        )
    (build_dir / "compact_stream_k_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "compact_stream_k",
    )
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_and_run(args.build_dir, no_run=args.no_run), indent=2))


if __name__ == "__main__":
    main()
