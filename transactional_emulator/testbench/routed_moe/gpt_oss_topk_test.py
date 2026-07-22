# ruff: noqa: E402
"""Instruction-level routed-MoE V_TOPK check.

This validates the Step6 v0 top-k contract in isolation:

    VRAM router logits -> V_TOPK -> INT SRAM expert ids + FP SRAM top-4 weights

    No HBM, expert execution, gather/scatter, or RTL is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Make the `compiler` package importable when this script is run directly.
# Prefer the pinned in-repo submodule (PLENA_Simulator/PLENA_Compiler) over a
# sibling AICrossSim-workspace checkout: a sibling may be on a different branch
# and would otherwise silently shadow the submodule on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _compiler_root in (_REPO_ROOT / "PLENA_Compiler", _REPO_ROOT.parent / "PLENA_Compiler"):
    if (_compiler_root / "aten" / "plena" / "compiler.py").exists():
        sys.path.insert(0, str(_compiler_root))
        break

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.emulator_runner import run_emulator
from transactional_emulator.testbench.layout_utils import prestage_bf16_vram_matrix
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.gpt_oss_testkit import (
    _decode_bf16_dump,
    _decode_u32_dump,
)


def run_topk_instruction(args: argparse.Namespace) -> dict:
    build_dir = args.build_dir.resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(args, build_dir)
    mlen = hw.mlen
    blen = hw.blen

    if args.mode == "gpt_oss":
        num_experts = 32
        top_k = 4
        logits = torch.full((1, num_experts), -100.0, dtype=torch.bfloat16)
        # Top values include an exact tie at experts 3 and 7; the contract
        # requires smaller expert index first.
        top_entries = [(7, 4.0), (3, 4.0), (9, 2.0), (0, 1.0), (31, 0.5)]
    elif args.mode == "qwen3_moe":
        num_experts = 128
        top_k = 8
        if mlen < num_experts:
            raise ValueError(f"Qwen3-MoE V_TOPK requires MLEN >= {num_experts}, got {mlen}")
        logits = torch.full((1, num_experts), -100.0, dtype=torch.bfloat16)
        # Include a tie inside the top-8 to pin the low-index tie contract for
        # the 128-expert policy.
        top_entries = [
            (126, 8.0),
            (2, 7.5),
            (64, 6.0),
            (5, 4.0),
            (70, 4.0),
            (31, 2.0),
            (90, 1.0),
            (0, 0.5),
            (127, 0.25),
        ]
    else:
        raise ValueError(f"unsupported mode {args.mode!r}")

    for expert_idx, value in top_entries:
        logits[0, expert_idx] = torch.tensor(value, dtype=torch.bfloat16)

    ranked = sorted(top_entries, key=lambda item: (-item[1], item[0]))[:top_k]
    expected_indices = torch.tensor([idx for idx, _ in ranked], dtype=torch.int64)
    expected_weights = torch.softmax(torch.tensor([value for _, value in ranked], dtype=torch.float32), dim=0).to(
        torch.bfloat16
    )

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    vram_preload = torch.zeros(mlen, dtype=torch.bfloat16)
    logits_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="TopKLogits",
        tensor=logits,
        vram_addr=0,
        physical_shape=(1, mlen),
        vram_preload=vram_preload,
    )

    weights_fp_base = 32
    indices_int_base = 64
    prog.gpt_oss_router_topk_softmax_v0(
        logits_vram,
        token_idx=0,
        weights_fp_base=weights_fp_base,
        indices_int_base=indices_int_base,
        num_experts=num_experts,
        top_k=top_k,
        name="instruction_level",
    )
    isa = prog.compile()

    create_sim_env(
        {},
        isa,
        {"original_output": torch.zeros(1, dtype=torch.bfloat16)},
        fp_preload=torch.zeros(128, dtype=torch.float16),
        int_preload=torch.zeros(128, dtype=torch.int32),
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts={},
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_topk",
        data=None,
        specified_data_order=[],
        build_path=build_dir,
        input_tensors={},
        tensor_layouts={},
        hbm_addrs={},
    )
    hbm_path = build_dir / "hbm_for_behave_sim.bin"
    if not hbm_path.exists():
        hbm_path.write_bytes(bytes(64))

    metrics = run_emulator(build_dir, hbm_size=64, threads=args.emu_threads)
    got_weights = _decode_bf16_dump(build_dir / "fpsram_dump.bin")[weights_fp_base : weights_fp_base + top_k]
    got_indices = _decode_u32_dump(build_dir / "intsram_dump.bin")[indices_int_base : indices_int_base + top_k]

    weights_close = torch.allclose(got_weights.float(), expected_weights.float(), atol=0.003, rtol=0.0)
    indices_match = torch.equal(got_indices, expected_indices)
    summary = {
        "build_dir": str(build_dir),
        "mode": args.mode,
        "num_experts": num_experts,
        "top_k": top_k,
        "expected_indices": expected_indices.tolist(),
        "got_indices": got_indices.tolist(),
        "expected_weights": [float(v) for v in expected_weights.float().tolist()],
        "got_weights": [float(v) for v in got_weights.float().tolist()],
        "indices_match": bool(indices_match),
        "weights_close": bool(weights_close),
        "tie_break": "lower expert index wins exact logit ties",
        "run_metrics": metrics,
        "passed": bool(indices_match and weights_close),
    }
    (build_dir / "topk_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if not summary["passed"]:
        raise AssertionError(f"V_TOPK instruction-level check failed: {summary}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--mode", choices=["gpt_oss", "qwen3_moe"], default="gpt_oss")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "gpt_oss_topk_instruction",
    )
    parser.add_argument("--emu-threads", type=int, default=None)
    args = parser.parse_args()
    run_topk_instruction(args)


if __name__ == "__main__":
    main()
