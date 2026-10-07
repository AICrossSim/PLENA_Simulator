# ruff: noqa: E402
"""GPT-OSS layer-0 router GEMM device smoke.

Step 3 scope only:

    X[8, 2880] @ router_W.T[2880, 32] + router_b

Top-k remains host-side.  The device path only proves the narrow-N router logits
GEMM, including the N < MLEN tail block and BF16 bias add.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

_TESTBENCH_ROOT = Path(__file__).resolve().parents[1]
_ATEN_BUILD_DIR = _TESTBENCH_ROOT / "aten" / "build"
# Make the `compiler` package importable when this script is run directly.
# Prefer the pinned in-repo submodule (PLENA_Simulator/PLENA_Compiler) over a
# sibling AICrossSim-workspace checkout: a sibling may be on a different branch
# and would otherwise silently shadow the submodule on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _compiler_root in (_REPO_ROOT / "PLENA_Compiler", _REPO_ROOT.parent / "PLENA_Compiler"):
    if (_compiler_root / "aten" / "plena" / "compiler.py").exists():
        sys.path.insert(0, str(_compiler_root))
        break

from aten.models.gpt_oss.moe_reference import compare_stats
from aten.models.gpt_oss.real_layer_utils import SHARD0, cached_file, load_json
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.emulator_runner import compare_emulator_output, run_emulator
from transactional_emulator.testbench.layout_utils import prestage_bf16_vram_matrix
from transactional_emulator.testbench.routed_moe._reference import ensure_reference
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.sliced_layer_test_builder import quantize_to_mxfp
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.gpt_oss_testkit import (
    _align_to,
    _stats_dict,
)


def _load_router_tensors() -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(cached_file(SHARD0), framework="pt", device="cpu") as f:
        weight = f.get_tensor("model.layers.0.mlp.router.weight").to(torch.bfloat16)
        bias = f.get_tensor("model.layers.0.mlp.router.bias").to(torch.bfloat16)
    return weight, bias


def _router_projection_golden(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    bias: torch.Tensor,
    *,
    mlen: int,
    quantize_x: bool,
    quantize_weight: bool,
) -> torch.Tensor:
    """Router logits with selectable MXFP8 HBM inputs and BF16 K-chunk accumulation."""
    x_q = quantize_to_mxfp(x) if quantize_x else x.to(torch.bfloat16)
    w_q = quantize_to_mxfp(weight_t) if quantize_weight else weight_t.to(torch.bfloat16)
    chunk = mlen * 4
    acc = None
    for k_start in range(0, x_q.shape[1], chunk):
        k_end = min(k_start + chunk, x_q.shape[1])
        partial = torch.matmul(x_q[:, k_start:k_end].float(), w_q[k_start:k_end, :].float()).to(torch.bfloat16)
        acc = partial if acc is None else (acc.float() + partial.float()).to(torch.bfloat16)
    assert acc is not None
    return (acc.float() + bias.reshape(1, -1).float()).to(torch.bfloat16)


def _router_golden_b(x: torch.Tensor, weight_t: torch.Tensor, bias: torch.Tensor, *, mlen: int) -> torch.Tensor:
    """Current PLENA-aware router logits with MXFP8 X/W HBM tensors."""
    return _router_projection_golden(
        x,
        weight_t,
        bias,
        mlen=mlen,
        quantize_x=True,
        quantize_weight=True,
    )


def _router_vector_bf16_golden(
    x: torch.Tensor,
    weight_rows: torch.Tensor,
    bias: torch.Tensor,
    *,
    mlen: int,
) -> torch.Tensor:
    """Golden for the BF16 vector-dot router path.

    Matches the emitted V_MUL_VV + V_RED_SUM sequence: each 64-wide product row
    is quantized back to BF16, then the scalar accumulator is rounded to BF16
    after every tile reduction.
    """
    rows, hidden = x.shape
    num_experts = weight_rows.shape[0]
    out = torch.zeros(rows, num_experts, dtype=torch.bfloat16)
    for row_idx in range(rows):
        for expert_idx in range(num_experts):
            acc = torch.tensor(0.0, dtype=torch.bfloat16)
            for k_start in range(0, hidden, mlen):
                k_end = min(k_start + mlen, hidden)
                product = (x[row_idx, k_start:k_end].float() * weight_rows[expert_idx, k_start:k_end].float()).to(
                    torch.bfloat16
                )
                partial = product.float().sum()
                acc = torch.tensor(float(acc.float().item() + partial.item()), dtype=torch.bfloat16)
            out[row_idx, expert_idx] = (acc.float() + bias[expert_idx].float()).to(torch.bfloat16)
    return out


def _vram_layout_size(physical_shape: tuple[int, int], *, mlen: int) -> int:
    physical_rows, physical_cols = physical_shape
    return math.ceil(physical_cols / mlen) * physical_rows * mlen


def _rank_stability(hf_logits: torch.Tensor, device_logits: torch.Tensor, top_k: int) -> dict:
    hf_sorted_vals, hf_sorted_idx = torch.sort(hf_logits.float(), dim=-1, descending=True)
    del hf_sorted_idx
    hf_top_values, hf_topk = torch.topk(hf_logits.float(), k=top_k, dim=-1)
    device_topk = torch.topk(device_logits.float(), k=top_k, dim=-1).indices
    hf_sets = [set(int(v) for v in row.tolist()) for row in hf_topk.cpu()]
    device_sets = [set(int(v) for v in row.tolist()) for row in device_topk.cpu()]
    set_matches = [hf == device for hf, device in zip(hf_sets, device_sets, strict=True)]
    order_matches = [
        hf == device for hf, device in zip(hf_topk.cpu().tolist(), device_topk.cpu().tolist(), strict=True)
    ]
    per_token_error = (device_logits.float() - hf_logits.float()).abs().max(dim=-1).values
    rank_gap = hf_sorted_vals[:, top_k - 1] - hf_sorted_vals[:, top_k]
    gap_gt_error = rank_gap > per_token_error
    internal_adjacent_gaps = hf_top_values[:, :-1] - hf_top_values[:, 1:]
    min_internal_gap = internal_adjacent_gaps.min(dim=-1).values
    internal_order_gap_gt_error = min_internal_gap > per_token_error
    order_when_internal_gap_safe = [
        bool(order_ok or not order_safe)
        for order_ok, order_safe in zip(order_matches, internal_order_gap_gt_error.cpu().tolist(), strict=True)
    ]
    return {
        "top_k": top_k,
        "hf_topk_indices": hf_topk.cpu().tolist(),
        "device_topk_indices": device_topk.cpu().tolist(),
        "topk_order_matches_hf": bool(torch.equal(hf_topk.cpu(), device_topk.cpu())),
        "topk_order_matches_hf_by_token": order_matches,
        "topk_order_matches_when_internal_gap_safe_by_token": order_when_internal_gap_safe,
        "topk_order_matches_when_internal_gap_safe": all(order_when_internal_gap_safe),
        "topk_set_matches_hf_by_token": set_matches,
        "topk_set_matches_hf": all(set_matches),
        "rank4_rank5_gap": [float(v) for v in rank_gap.cpu().tolist()],
        "min_internal_topk_gap": [float(v) for v in min_internal_gap.cpu().tolist()],
        "device_max_abs_error_by_token": [float(v) for v in per_token_error.cpu().tolist()],
        "gap_gt_measured_error_by_token": [bool(v) for v in gap_gt_error.cpu().tolist()],
        "internal_order_gap_gt_measured_error_by_token": [bool(v) for v in internal_order_gap_gt_error.cpu().tolist()],
        "min_gap": float(rank_gap.min().item()),
        "min_internal_gap": float(min_internal_gap.min().item()),
        "max_device_error": float(per_token_error.max().item()),
        "passed": bool(gap_gt_error.all().item()),
    }


def run_router_gemm(args: argparse.Namespace) -> dict:
    mlen = args.mlen
    blen = args.blen
    build_dir = args.build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(args, build_dir)

    # Shared tok8/layer-0 bundle, generated from the real checkpoint on first
    # use (see routed_moe/_reference.py). Only x is consumed here.
    reference = ensure_reference(args.reference_path, layer_index=0, rows=8, seed=1)
    x = reference["x"].to(torch.bfloat16)
    rows, hidden = x.shape

    config = load_json("config.json")
    num_experts = int(config["num_local_experts"])
    top_k = int(config["num_experts_per_tok"])
    if hidden != int(config["hidden_size"]):
        raise ValueError(f"reference hidden={hidden} does not match config hidden_size={config['hidden_size']}")
    if num_experts > mlen:
        raise ValueError(f"router experts={num_experts} exceeds MLEN={mlen}; this test covers N<MLEN only")

    router_weight, router_bias = _load_router_tensors()
    weight_t = router_weight.t().contiguous()
    weight_rows = router_weight.contiguous()
    hf_logits = F.linear(x, router_weight, router_bias).to(torch.bfloat16)
    hf_top_values, hf_top_indices = torch.topk(hf_logits, k=top_k, dim=-1)
    hf_top_weights = torch.nn.functional.softmax(hf_top_values, dim=1, dtype=hf_top_values.dtype)

    mx_router_golden_b = _router_golden_b(x, weight_t, router_bias, mlen=mlen)
    bf16_router_golden = _router_vector_bf16_golden(x, weight_rows, router_bias, mlen=mlen)
    router_x_mx_w_bf16 = _router_projection_golden(
        x,
        weight_t,
        router_bias,
        mlen=mlen,
        quantize_x=True,
        quantize_weight=False,
    )
    router_x_bf16_w_mx = _router_projection_golden(
        x,
        weight_t,
        router_bias,
        mlen=mlen,
        quantize_x=False,
        quantize_weight=True,
    )
    router_x_bf16_w_bf16 = _router_projection_golden(
        x,
        weight_t,
        router_bias,
        mlen=mlen,
        quantize_x=False,
        quantize_weight=False,
    )
    hf_b_stats = compare_stats(mx_router_golden_b, hf_logits, rtol=0.02)
    hf_bf16_router_stats = compare_stats(bf16_router_golden, hf_logits, rtol=0.02)
    hf_x_mx_w_bf16_stats = compare_stats(router_x_mx_w_bf16, hf_logits, rtol=0.02)
    hf_x_bf16_w_mx_stats = compare_stats(router_x_bf16_w_mx, hf_logits, rtol=0.02)
    hf_x_bf16_w_bf16_stats = compare_stats(router_x_bf16_w_bf16, hf_logits, rtol=0.02)

    print("=" * 80)
    print(f"GPT-OSS router GEMM device check (rows={rows}, hidden={hidden}, experts={num_experts})")
    print("=" * 80)
    print(
        f"tile coverage: M={math.ceil(rows / mlen)}, N=1 (logical {num_experts} < MLEN {mlen}), "
        f"K={math.ceil(hidden / mlen)}, K-split chunks={math.ceil(math.ceil(hidden / mlen) / 4)}"
    )

    physical_rows = max(blen, math.ceil(rows / blen) * blen)
    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)

    x_physical = (physical_rows, hidden)
    w_physical = (num_experts, hidden)
    bias_physical = (physical_rows, mlen)
    alignment = mlen * mlen
    x_base = 0
    w_base = _align_to(x_base + _vram_layout_size(x_physical, mlen=mlen), alignment)
    bias_base = _align_to(w_base + _vram_layout_size(w_physical, mlen=mlen), alignment)
    preload_size = bias_base + _vram_layout_size(bias_physical, mlen=mlen)
    vram_preload = torch.zeros(preload_size, dtype=torch.bfloat16)

    x_input = prestage_bf16_vram_matrix(
        prog=prog,
        name="RouterXBF16",
        tensor=x,
        vram_addr=x_base,
        physical_shape=x_physical,
        vram_preload=vram_preload,
    )
    w_input = prestage_bf16_vram_matrix(
        prog=prog,
        name="RouterWBF16",
        tensor=weight_rows,
        vram_addr=w_base,
        physical_shape=w_physical,
        vram_preload=vram_preload,
    )
    bias = router_bias.reshape(1, -1).repeat(rows, 1).to(torch.bfloat16)
    bias_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="RouterBias",
        tensor=bias,
        vram_addr=bias_base,
        physical_shape=bias_physical,
        vram_preload=vram_preload,
    )
    logits = prog.gpt_oss_router_logits_bf16_v0(
        x_input,
        w_input,
        rows=rows,
        hidden=hidden,
        num_experts=num_experts,
        name="router_logits",
    )
    prog.vram_add(logits, bias_vram, num_rows=rows)

    gen_code = prog.compile()
    scale_reg_count = gen_code.count("C_SET_SCALE_REG")
    if scale_reg_count != 0:
        raise AssertionError(f"BF16 router path must not emit C_SET_SCALE_REG, got {scale_reg_count}")

    input_tensors = {}
    tensor_layouts = {}
    golden_b_padded = torch.zeros(physical_rows, mlen, dtype=torch.bfloat16)
    golden_b_padded[:rows, :num_experts] = bf16_router_golden
    golden_result = {"original_output": golden_b_padded, "tensor_layouts": tensor_layouts}
    fp_preload = [0.0, 1e-6, 1.0 / hidden] + [0.0] * 7
    create_sim_env(
        input_tensors,
        gen_code,
        golden_result,
        fp_preload,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_router_gemm",
        data=None,
        specified_data_order=[],
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs={},
        tensor_layouts=tensor_layouts,
    )
    hbm_path = build_dir / "hbm_for_behave_sim.bin"
    if not hbm_path.exists() or hbm_path.stat().st_size == 0:
        hbm_path.write_bytes(b"\x00" * 64)
        (build_dir / "hbm_size.txt").write_text("64\n")

    logits_addr = prog._compiler.get_vram_addr(logits.name)
    comparison_params = {
        "start_row_idx": logits_addr // mlen,
        "num_rows": (physical_rows * mlen) // mlen,
        "num_batches": rows,
        "elements_per_batch": mlen,
        "row_dim": mlen,
        "physical_rows": physical_rows,
        "atol": float((bf16_router_golden.float().std(unbiased=False) * 0.01).item()),
        "rtol": 0.02,
        "use_stride_mode": False,
    }
    (build_dir / "comparison_params.json").write_text(json.dumps(comparison_params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(gen_code)

    metrics = run_emulator(build_dir, threads=args.emu_threads)
    results, params = compare_emulator_output(build_dir)
    emu_padded = results["simulated_values"].reshape(physical_rows, mlen).to(torch.bfloat16)
    emu_logits = emu_padded[:rows, :num_experts].contiguous()
    padded_tail_max = float(emu_padded[:rows, num_experts:].float().abs().max().item())
    emu_b_stats = compare_stats(emu_logits, bf16_router_golden, rtol=0.02)
    emu_hf_stats = compare_stats(emu_logits, hf_logits, rtol=0.02)
    rank_gate = _rank_stability(hf_logits, emu_logits, top_k)

    summary = {
        "reference_path": str(args.reference_path.expanduser().resolve()),
        "rows": rows,
        "hidden": hidden,
        "num_experts": num_experts,
        "top_k": top_k,
        "mlen": mlen,
        "blen": blen,
        "router_path": "bf16_vector_dot_v0",
        "router_path_contract": {
            "selected": "Vector Machine BF16 dot product (V_MUL_VV + V_RED_SUM)",
            "reason": (
                "M_MM/HBM matrix path is coupled to MX prefetch and C_SET_SCALE_REG. "
                "GPT-OSS router is high precision, so the router smoke uses prestaged BF16 "
                "VRAM operands and emits no MX scale setup. Expert/down paths remain MXFP8."
            ),
            "router_scale_reg_count_expected": 0,
        },
        "tile_coverage": {
            "m_tiles": math.ceil(rows / mlen),
            "n_tiles": 1,
            "logical_n": num_experts,
            "physical_n": mlen,
            "k_tiles": math.ceil(hidden / mlen),
            "k_split_chunks": math.ceil(math.ceil(hidden / mlen) / 4),
        },
        "scale_reg_count": scale_reg_count,
        "hf_topk_indices": hf_top_indices.cpu().tolist(),
        "hf_topk_weights": [[float(v) for v in row] for row in hf_top_weights.cpu().float().tolist()],
        "bf16_router_golden_vs_hf": _stats_dict(hf_bf16_router_stats),
        "golden_b_vs_hf_record_only": _stats_dict(hf_b_stats),
        "router_precision_split_vs_hf_record_only": {
            "x_mxfp8_w_mxfp8_current_b": _stats_dict(hf_b_stats),
            "x_mxfp8_w_bf16": _stats_dict(hf_x_mx_w_bf16_stats),
            "x_bf16_w_mxfp8": _stats_dict(hf_x_bf16_w_mx_stats),
            "x_bf16_w_bf16_chunked": _stats_dict(hf_x_bf16_w_bf16_stats),
            "rank_stability": {
                "x_mxfp8_w_mxfp8_current_b": _rank_stability(hf_logits, mx_router_golden_b, top_k),
                "x_mxfp8_w_bf16": _rank_stability(hf_logits, router_x_mx_w_bf16, top_k),
                "x_bf16_w_mxfp8": _rank_stability(hf_logits, router_x_bf16_w_mx, top_k),
                "x_bf16_w_bf16_chunked": _rank_stability(hf_logits, router_x_bf16_w_bf16, top_k),
            },
            "note": (
                "HF/GptOssMLP router is a high-precision path. This split keeps the BF16 K-chunk "
                "accumulation order fixed and changes only whether X and W_router are fed through "
                "PLENA quantize_to_mxfp. On this tok8 sample, keeping only W_router BF16 does not "
                "collapse the gap; keeping only X BF16 also does not collapse it. The gap collapses "
                "only when both router operands are BF16, leaving the chunk-order BF16 floor. Step-6 "
                "device top-k therefore needs a router precision decision for the whole router path, "
                "not just the router weight tensor."
            ),
        },
        "emulator_vs_golden_b": _stats_dict(emu_b_stats),
        "emulator_vs_hf": _stats_dict(emu_hf_stats),
        "padded_tail_max_abs": padded_tail_max,
        "rank_stability_gate": rank_gate,
        "run_metrics": metrics,
        "comparison_params": params,
        "router_gemm_gate": {
            "emu_vs_golden_b_rel_rms_under_2pct": emu_b_stats.rel_rms <= 0.02,
            "emu_vs_hf_rel_rms_under_1pct": emu_hf_stats.rel_rms <= 0.01,
            "scale_reg_count_zero": scale_reg_count == 0,
            "rank_gap_gt_device_error": rank_gate["passed"],
            "topk_order_matches_when_internal_gap_safe": rank_gate["topk_order_matches_when_internal_gap_safe"],
            "topk_order_matches_hf": rank_gate["topk_order_matches_hf"],
            "topk_set_matches_hf": rank_gate["topk_set_matches_hf"],
            "passed": (
                emu_b_stats.rel_rms <= 0.02
                and emu_hf_stats.rel_rms <= 0.01
                and scale_reg_count == 0
                and rank_gate["topk_set_matches_hf"]
                and rank_gate["topk_order_matches_when_internal_gap_safe"]
            ),
            "note": (
                "Router now uses a BF16 vector-dot path with no MX scale setup. Top-k remains host-side "
                "until the device V_TOPK path exists. Rank4/rank5 gap is used for top-4 set stability; "
                "internal top-k order is only required when adjacent in-top-k HF gaps exceed measured "
                "logit error. Ambiguous order/gap cases still use HF indices and weights."
            ),
        },
    }
    (build_dir / "router_gemm_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["router_gemm_gate"]["passed"]:
        raise AssertionError(f"router BF16 gate failed: {summary['router_gemm_gate']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=_ATEN_BUILD_DIR / "gpt_oss_real_layer0_tok8_l1_stamp" / "hf_layer0_moe_reference.pt",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "gpt_oss_router_gemm_tok8",
    )
    parser.add_argument("--emu-threads", type=int, default=None)
    args = parser.parse_args()
    run_router_gemm(args)


if __name__ == "__main__":
    main()
