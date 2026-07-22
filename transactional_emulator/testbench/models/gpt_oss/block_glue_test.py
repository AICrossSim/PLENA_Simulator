"""GPT-OSS block glue smoke: weighted RMSNorm and residual add.

This is intentionally narrower than a full decoder block.  It verifies the
VRAM-resident BF16 glue pieces that connect attention and MoE:

  * GPT-OSS RMSNorm = unweighted RMSNorm followed by learned scale multiply.
  * residual add = hidden += sublayer output.

No attention, MoE, HBM activation loads, dynamic routing, or RTL is involved.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.aten.golden import (
    _active_precision_settings,
    _rms_norm_vector_ref,
    quantize_to_vector_fp,
)
from transactional_emulator.testbench.routed_moe.gpt_oss_moe_gather_scatter_test import (
    _comparison_params_for,
    _rewrite_compact_golden,
)
from transactional_emulator.testbench.emulator_runner import compare_emulator_output, run_emulator
from transactional_emulator.testbench.layout_utils import prestage_bf16_vram_matrix
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.gpt_oss_testkit import (
    _align_to,
)


def _vram_layout_size(shape: tuple[int, int], *, mlen: int) -> int:
    rows, cols = shape
    return math.ceil(cols / mlen) * rows * mlen


def _bf16_vram(x: torch.Tensor) -> torch.Tensor:
    precision = _active_precision_settings()
    return quantize_to_vector_fp(x.to(torch.bfloat16).float(), precision)


def _load_bf16_tensor(path: Path) -> torch.Tensor:
    obj = torch.load(path.expanduser().resolve(), map_location="cpu")
    if isinstance(obj, dict):
        obj = obj.get("tensor", next(iter(obj.values())))
    if not torch.is_tensor(obj):
        raise TypeError(f"{path} did not contain a tensor")
    return obj.detach().to(torch.bfloat16).cpu().contiguous()


def _weighted_rms_norm_golden(
    x: torch.Tensor,
    weight_rows: torch.Tensor,
    eps: float,
    *,
    active_hidden: int | None = None,
) -> torch.Tensor:
    precision = _active_precision_settings()
    x_vram = _bf16_vram(x)
    w_vram = _bf16_vram(weight_rows)
    if active_hidden is None or active_hidden == x_vram.shape[-1]:
        normed = _rms_norm_vector_ref(x_vram, eps, precision)
    else:
        # DeepSeek MLA has logical widths such as 512 that are physically padded
        # to MLEN. Hardware reduces all physical lanes, but padded tail lanes are
        # zero and FPRAM still holds 1 / logical_hidden.
        mean_sq = x_vram.float().pow(2).sum(dim=-1, keepdim=True) / float(active_hidden)
        rms = quantize_to_vector_fp(torch.rsqrt(mean_sq + eps), precision)
        normed = quantize_to_vector_fp(x_vram.float() * rms.float(), precision)
    return quantize_to_vector_fp(normed.float() * w_vram.float(), precision)


def _residual_add_golden(hidden: torch.Tensor, sublayer: torch.Tensor) -> torch.Tensor:
    precision = _active_precision_settings()
    hidden_vram = _bf16_vram(hidden)
    sublayer_vram = _bf16_vram(sublayer)
    return quantize_to_vector_fp(hidden_vram.float() + sublayer_vram.float(), precision)


def _stats(actual: torch.Tensor, reference: torch.Tensor) -> dict:
    actual_f = actual.float()
    ref_f = reference.float()
    diff = actual_f - ref_f
    denom = torch.linalg.vector_norm(ref_f).clamp_min(1e-12)
    rel_rms = float((torch.linalg.vector_norm(diff) / denom).item())
    atol = float((ref_f.std(unbiased=False) * 0.01).item())
    allowed = atol + 0.02 * ref_f.abs()
    passed = diff.abs() <= allowed
    return {
        "rel_rms": rel_rms,
        "max_abs_error": float(diff.abs().max().item()),
        "atol": atol,
        "rtol": 0.02,
        "pass_rate": float(passed.float().mean().item()),
        "allclose": bool(passed.all().item()),
    }


def _active_tail_stats(actual: torch.Tensor, reference: torch.Tensor, active_hidden: int) -> dict:
    active = _stats(actual[:, :active_hidden], reference[:, :active_hidden])
    tail = actual[:, active_hidden:]
    active["padded_tail_max_abs"] = float(tail.float().abs().max().item()) if tail.numel() else 0.0
    return active


def _run_stage(args: argparse.Namespace, stage: str) -> dict:
    mlen = args.mlen
    blen = args.blen
    rows = args.rows
    hidden = args.hidden_size or 2880
    physical_hidden = _align_to(hidden, mlen) if args.allow_padded_hidden else hidden
    if physical_hidden != hidden and stage != "weighted_rms_norm":
        raise ValueError("--allow-padded-hidden currently applies only to weighted_rms_norm")
    if not args.allow_padded_hidden and hidden % mlen != 0:
        raise ValueError(f"hidden={hidden} must be divisible by MLEN={mlen}")
    if rows % blen != 0:
        raise ValueError(f"rows={rows} must be a multiple of BLEN={blen}")

    torch.manual_seed(args.seed)
    if args.x_pt is not None:
        hidden_x = _load_bf16_tensor(args.x_pt)
        if hidden_x.dim() > 2:
            hidden_x = hidden_x.reshape(-1, hidden_x.shape[-1]).contiguous()
        if tuple(hidden_x.shape) != (rows, hidden):
            raise ValueError(f"--x-pt shape {tuple(hidden_x.shape)} != {(rows, hidden)}")
    else:
        hidden_x = (torch.randn(rows, hidden) * 0.5).to(torch.bfloat16)
    if args.norm_weight_pt is not None:
        norm_weight_raw = _load_bf16_tensor(args.norm_weight_pt)
        if norm_weight_raw.dim() == 1:
            if norm_weight_raw.numel() != hidden:
                raise ValueError(f"--norm-weight-pt length {norm_weight_raw.numel()} != hidden={hidden}")
            norm_weight_rows = norm_weight_raw.reshape(1, hidden).repeat(rows, 1).to(torch.bfloat16)
        elif norm_weight_raw.dim() == 2:
            if tuple(norm_weight_raw.shape) == (1, hidden):
                norm_weight_rows = norm_weight_raw.repeat(rows, 1).to(torch.bfloat16)
            elif tuple(norm_weight_raw.shape) == (rows, hidden):
                norm_weight_rows = norm_weight_raw.to(torch.bfloat16)
            else:
                raise ValueError(
                    f"--norm-weight-pt shape {tuple(norm_weight_raw.shape)} incompatible with {(rows, hidden)}"
                )
        else:
            raise ValueError(f"--norm-weight-pt must be rank 1 or 2, got {tuple(norm_weight_raw.shape)}")
    else:
        norm_weight = (1.0 + torch.randn(hidden) * 0.05).to(torch.bfloat16)
        norm_weight_rows = norm_weight.reshape(1, hidden).repeat(rows, 1).to(torch.bfloat16)

    build_dir = (args.build_dir / stage).expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(args, build_dir)

    physical_shape = (max(blen, rows), physical_hidden)
    alignment = mlen * mlen
    x_base = 0
    y_base = _align_to(x_base + _vram_layout_size(physical_shape, mlen=mlen), alignment)
    w_base = _align_to(y_base + _vram_layout_size(physical_shape, mlen=mlen), alignment)
    preload_size = w_base + _vram_layout_size(physical_shape, mlen=mlen)
    vram_preload = torch.zeros(preload_size, dtype=torch.bfloat16)

    sublayer = (torch.randn(rows, hidden) * 0.25).to(torch.bfloat16)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    x_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="GlueX",
        tensor=hidden_x,
        vram_addr=x_base,
        physical_shape=physical_shape,
        vram_preload=vram_preload,
    )
    y_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="GlueY",
        tensor=sublayer,
        vram_addr=y_base,
        physical_shape=physical_shape,
        vram_preload=vram_preload,
    )
    weight_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="GlueNormWeight",
        tensor=norm_weight_rows,
        vram_addr=w_base,
        physical_shape=physical_shape,
        vram_preload=vram_preload,
    )

    eps = 1e-5
    if stage == "weighted_rms_norm":
        prog.rms_norm(x_vram)
        prog.vram_mul(x_vram, weight_vram, num_rows=rows)
        output = x_vram
        if physical_hidden != hidden:
            hidden_x_physical = torch.zeros(rows, physical_hidden, dtype=torch.bfloat16)
            hidden_x_physical[:, :hidden] = hidden_x
            norm_weight_physical = torch.zeros(rows, physical_hidden, dtype=torch.bfloat16)
            norm_weight_physical[:, :hidden] = norm_weight_rows
            golden = _weighted_rms_norm_golden(
                hidden_x_physical,
                norm_weight_physical,
                eps,
                active_hidden=hidden,
            )
        else:
            golden = _weighted_rms_norm_golden(hidden_x, norm_weight_rows, eps)
    elif stage == "residual_add":
        prog.vram_add(x_vram, y_vram, num_rows=rows)
        output = x_vram
        golden = _residual_add_golden(hidden_x, sublayer)
    else:
        raise ValueError(stage)

    isa = prog.compile()
    fp_preload = [0.0, eps, 1.0 / hidden]
    dummy_input = {"DUMMY": torch.zeros(1, mlen)}
    create_sim_env(
        dummy_input,
        isa,
        {"original_output": golden},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
    )
    create_mem_for_sim(
        data_size=mlen,
        mode="behave_sim",
        asm=f"gpt_oss_block_glue_{stage}",
        build_path=build_dir,
        input_tensors=dummy_input,
        specified_data_order=["DUMMY"],
    )
    _rewrite_compact_golden(build_dir, golden)
    compare_hidden = physical_hidden if physical_hidden != hidden else hidden
    comparison_params = _comparison_params_for(output, rows=rows, hidden=compare_hidden, mlen=mlen, golden=golden)
    (build_dir / "comparison_params.json").write_text(json.dumps(comparison_params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(isa)

    metrics = run_emulator(build_dir, threads=getattr(args, "emu_threads", None))
    results, params = compare_emulator_output(build_dir)
    emu = results["simulated_values"].reshape(rows, compare_hidden).to(torch.bfloat16)
    stage_stats = _active_tail_stats(emu, golden, hidden) if physical_hidden != hidden else _stats(emu, golden)
    summary = {
        "stage": stage,
        "rows": rows,
        "hidden": hidden,
        "physical_hidden": physical_hidden,
        "mlen": mlen,
        "blen": blen,
        "asm_lines": len(isa.splitlines()),
        "run_metrics": metrics,
        "comparison_params": params,
        "stats": stage_stats,
        "passed": stage_stats["allclose"] and stage_stats["rel_rms"] <= 0.02,
    }
    (build_dir / "gpt_oss_block_glue_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise AssertionError(f"{stage} failed: {stage_stats}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument(
        "--stage",
        choices=("weighted_rms_norm", "residual_add", "all"),
        default="all",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "gpt_oss_block_glue",
    )
    parser.add_argument(
        "--allow-padded-hidden",
        action="store_true",
        help="Pad hidden to MLEN physically while keeping 1/logical_hidden in FPRAM.",
    )
    parser.add_argument("--x-pt", type=Path, default=None, help="Optional BF16 input tensor for weighted_rms_norm.")
    parser.add_argument(
        "--norm-weight-pt",
        type=Path,
        default=None,
        help="Optional rank-1 or rank-2 BF16 RMSNorm weight tensor for weighted_rms_norm.",
    )
    args = parser.parse_args()

    stages = ["weighted_rms_norm", "residual_add"] if args.stage == "all" else [args.stage]
    summaries = [_run_stage(args, stage) for stage in stages]
    if len(summaries) > 1:
        args.build_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        (args.build_dir.expanduser().resolve() / "gpt_oss_block_glue_summary.json").write_text(
            json.dumps({"stages": summaries, "passed": all(s["passed"] for s in summaries)}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
