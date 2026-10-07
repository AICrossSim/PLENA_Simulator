# ruff: noqa: E402
"""GPT-OSS real layer-0 MoE fixed-routing emulator check.

This is the L1 "external proof" path:

    HF GptOssMLP routing/reference
      -> fixed HF top-k indices/weights
      -> PLENA emulator expert compute + combine
      -> compare against Golden B

HF corresponds to Golden A: MXFP4 checkpoint tensors dequantized to BF16 and
BF16 expert math.  Golden B uses the same fixed routing, but quantizes HBM
activations/weights/route weights through PLENA's current MXFP8 path.  Expert
biases are BF16 VRAM tensors, matching GPT-OSS' non-MX bias semantics.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

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

from aten.models.gpt_oss.moe_reference import compare_stats, split_packed_gate_up
from aten.models.gpt_oss.real_layer_utils import load_json, load_layer0_tensors
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.emulator_runner import compare_emulator_output, run_and_assert
from transactional_emulator.testbench.layout_utils import infer_hbm_tensor_layouts, prestage_bf16_vram_matrix
from transactional_emulator.testbench.routed_moe._reference import ensure_reference
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.sliced_layer_test_builder import quantize_to_mxfp
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.gpt_oss_testkit import (
    _activation_golden,
    _bf16,
    _comparison_params_for,
    _expanded_bias,
    _linear_projection_golden,
    _stats_dict,
)


def _expert_golden_b(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    mlen: int,
    b_gate: torch.Tensor,
    b_up: torch.Tensor,
    b_down: torch.Tensor,
) -> torch.Tensor:
    """Golden B for one selected expert.

    Biases are intentionally not MX-quantized.  They are BF16 VRAM tensors in
    the emulator path and BF16 additions here.
    """
    gate = _linear_projection_golden(x, w_gate, mlen=mlen, hbm_input=True)
    up = _linear_projection_golden(x, w_up, mlen=mlen, hbm_input=True)
    gate = _bf16(gate.float() + b_gate.float())
    up = _bf16(up.float() + b_up.float())
    hidden = _activation_golden(gate, up)
    out = _linear_projection_golden(hidden, w_down, mlen=mlen, hbm_input=False)
    return _bf16(out.float() + b_down.float())


def _make_route_matrix(
    topk_indices: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_id: int,
    hidden: int,
) -> torch.Tensor:
    route = torch.zeros(topk_indices.shape[0], 1, dtype=torch.bfloat16)
    token_idx, topk_pos = torch.where(topk_indices == expert_id)
    if token_idx.numel() > 0:
        route[token_idx, 0] = topk_weights[token_idx, topk_pos].to(torch.bfloat16)
    return route.repeat(1, hidden)


def _strict_elementwise_details(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    rtol: float,
    atol_scale: float = 0.01,
    block_cols: int = 64,
) -> dict:
    """Return the strict elementwise check details requested for real-layer proof."""
    actual_f = actual.float()
    reference_f = reference.float()
    diff = (actual_f - reference_f).abs()
    atol = reference_f.std(unbiased=False) * atol_scale
    allowed = atol + rtol * reference_f.abs()
    failed = diff > allowed
    fail_count = int(failed.sum().item())
    numel = failed.numel()

    details = {
        "fail_count": fail_count,
        "numel": numel,
        "pass_rate": 1.0 - (fail_count / numel if numel else 0.0),
        "atol": float(atol.item()),
        "rtol": rtol,
        "max_abs_error": float(diff.max().item()) if numel else 0.0,
        "max_allowed_ratio": float((diff / allowed.clamp_min(1e-12)).max().item()) if numel else 0.0,
    }

    if failed.ndim == 2:
        _rows, cols = failed.shape
        details["fail_count_by_token"] = [int(v) for v in failed.sum(dim=1).tolist()]
        blocks = []
        for block_idx in range(math.ceil(cols / block_cols)):
            start = block_idx * block_cols
            end = min(start + block_cols, cols)
            count = int(failed[:, start:end].sum().item())
            if count:
                blocks.append({"block": block_idx, "fail_count": count})
        details["fail_count_by_col_block"] = blocks
    return details


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    x_norm = math.sqrt(sum(x * x for x in x_centered))
    y_norm = math.sqrt(sum(y * y for y in y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered, strict=True)) / (x_norm * y_norm)


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


def _tail_gate_details(
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    strict_details: dict,
    clamp_counts: dict,
    rtol: float,
    k_values: tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0),
    block_cols: int = 64,
) -> dict:
    """Classify strict elementwise misses as BF16 tail behavior or mechanism."""
    actual_f = actual.float()
    reference_f = reference.float()
    signed = actual_f - reference_f
    abs_diff = signed.abs()
    sigma = abs_diff.std(unbiased=False)
    numel = int(abs_diff.numel())

    k_sweep = []
    for k in k_values:
        allowed = (k * sigma) + rtol * reference_f.abs()
        failed = abs_diff > allowed
        k_sweep.append({"k": k, "fail_count": int(failed.sum().item())})
    monotonic = all(k_sweep[i]["fail_count"] >= k_sweep[i + 1]["fail_count"] for i in range(len(k_sweep) - 1))
    k4_count = next((entry["fail_count"] for entry in k_sweep if entry["k"] == 4.0), k_sweep[-1]["fail_count"])
    k4_upper_bound = max(8, math.ceil(0.0004 * numel))

    strict_atol = float(strict_details["atol"])
    strict_allowed = strict_atol + rtol * reference_f.abs()
    strict_failed = abs_diff > strict_allowed
    strict_fail_count = int(strict_failed.sum().item())
    if strict_fail_count:
        signed_failed = signed[strict_failed]
        pos = int((signed_failed > 0).sum().item())
        neg = int((signed_failed < 0).sum().item())
        zero = strict_fail_count - pos - neg
        sign_imbalance = abs(pos - neg)
        sign_unbiased = sign_imbalance <= max(4, math.ceil(3.0 * math.sqrt(strict_fail_count)))
    else:
        pos = neg = zero = sign_imbalance = 0
        sign_unbiased = True

    spatial_blocks = strict_details.get("fail_count_by_col_block", [])
    max_block_fail = max((int(block["fail_count"]) for block in spatial_blocks), default=0)
    spatial_upper_bound = max(8, math.ceil(0.25 * strict_fail_count))
    spatial_undominated = max_block_fail <= spatial_upper_bound

    token_fail_counts = [float(v) for v in strict_details.get("fail_count_by_token", [])]
    per_token_clamp = clamp_counts.get("per_token") or []
    token_clamp_counts = [
        float(
            item.get("gate_gt_7", item.get("gate_gt_limit", 0))
            + item.get("up_gt_7", item.get("up_gt_limit", 0))
            + item.get("up_lt_neg_7", item.get("up_lt_neg_limit", 0))
        )
        for item in per_token_clamp
    ]
    pearson = _pearson(token_fail_counts, token_clamp_counts) if token_fail_counts and token_clamp_counts else None
    spearman = _spearman(token_fail_counts, token_clamp_counts) if token_fail_counts and token_clamp_counts else None
    clamp_uncorrelated = True
    if pearson is not None:
        clamp_uncorrelated = clamp_uncorrelated and abs(pearson) <= 0.5
    if spearman is not None:
        clamp_uncorrelated = clamp_uncorrelated and abs(spearman) <= 0.5

    passed = monotonic and k4_count <= k4_upper_bound and sign_unbiased and spatial_undominated and clamp_uncorrelated
    return {
        "classification": "bf16_tail" if passed else "mechanism_suspect",
        "passed": passed,
        "sigma_abs_error": float(sigma.item()),
        "k_sweep": k_sweep,
        "k4_fail_count": k4_count,
        "k4_upper_bound": k4_upper_bound,
        "monotonic_k_sweep": monotonic,
        "strict_fail_count": strict_fail_count,
        "sign": {
            "positive": pos,
            "negative": neg,
            "zero": zero,
            "imbalance": sign_imbalance,
            "unbiased": sign_unbiased,
        },
        "spatial": {
            "max_col_block_fail_count": max_block_fail,
            "upper_bound": spatial_upper_bound,
            "undominated": spatial_undominated,
        },
        "clamp_correlation": {
            "pearson": pearson,
            "spearman": spearman,
            "uncorrelated": clamp_uncorrelated,
            "fail_count_by_token": [int(v) for v in token_fail_counts],
            "clamp_count_by_token": [int(v) for v in token_clamp_counts],
        },
    }


def _run_gate_debug(
    *,
    args: argparse.Namespace,
    build_dir: Path,
    hw,
    x: torch.Tensor,
    w_gate: torch.Tensor,
    b_gate: torch.Tensor,
    golden: torch.Tensor,
    mlen: int,
    blen: int,
    rows: int,
    hidden: int,
    intermediate: int,
) -> dict:
    """Run only ``X @ W_gate + gate_bias`` for the first selected real expert."""
    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    x_input = prog.input("X", shape=(rows, hidden))
    w_gate_input = prog.input("W_0_gate", shape=(hidden, intermediate))
    x_vram = prog.load_batch(x_input, name="X")

    physical_rows = max(blen, math.ceil(rows / blen) * blen)
    bias_physical = (physical_rows, intermediate)
    bias_base = math.ceil(prog.vram_allocator._vmm.next_bump / (mlen * mlen)) * (mlen * mlen)
    bias_size = bias_physical[0] * bias_physical[1]
    bias_aligned = math.ceil(bias_size / (mlen * mlen)) * (mlen * mlen)
    vram_preload = torch.zeros(bias_base + bias_aligned, dtype=torch.bfloat16)
    b_gate_vram = prestage_bf16_vram_matrix(
        prog=prog,
        name="B_0_gate",
        tensor=b_gate,
        vram_addr=bias_base,
        physical_shape=bias_physical,
        vram_preload=vram_preload,
    )

    output_vram = prog.linear_projection(x_vram, w_gate_input, name="real_layer0_debug_gate")
    prog.vram_add(output_vram, b_gate_vram, num_rows=rows)
    isa = prog.compile()

    input_tensors = {"X": x, "W_0_gate": w_gate}
    tensor_layouts = infer_hbm_tensor_layouts(input_tensors)
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden},
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_real_layer0_gate",
        data=None,
        specified_data_order=list(input_tensors),
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
        tensor_layouts=tensor_layouts,
    )

    comparison_params = _comparison_params_for(output_vram, rows=rows, hidden=intermediate, mlen=mlen, golden=golden)
    (build_dir / "comparison_params.json").write_text(json.dumps(comparison_params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(isa)

    manifest = {
        "debug_stage": "gate",
        "rows": rows,
        "hidden": hidden,
        "intermediate": intermediate,
        "mlen": mlen,
        "blen": blen,
        "output_vram_row": prog._compiler.get_vram_addr(output_vram.name) // mlen,
        "asm_lines": len(isa.splitlines()),
        "comparison_params": comparison_params,
    }
    (build_dir / "real_layer0_gate_debug_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if args.no_run:
        return {"build_dir": str(build_dir), "ran": False, "manifest": manifest}

    metrics = run_and_assert(
        build_dir,
        "gpt_oss_real_layer0_gate",
        mlen=mlen,
        blen=blen,
        threads=args.emu_threads,
    )
    results, params = compare_emulator_output(build_dir)
    emu_output = results["simulated_values"].reshape(rows, intermediate).to(torch.bfloat16)
    emu_stats = compare_stats(emu_output, golden, rtol=0.02)
    if emu_stats.rel_rms > 0.02:
        raise AssertionError(f"gate debug emulator rel_rms={emu_stats.rel_rms:.6g} exceeds 2%")
    summary = {
        **manifest,
        "run_metrics": metrics,
        "comparison_params": params,
        "emulator_vs_gate_golden": _stats_dict(emu_stats),
    }
    (build_dir / "real_layer0_gate_debug_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return {"build_dir": str(build_dir), "ran": True, "summary": summary}


def run_real_layer0(args: argparse.Namespace) -> dict:
    mlen = args.mlen
    blen = args.blen
    build_dir = args.build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(args, build_dir)

    reference_path = args.reference_path.expanduser().resolve()
    # The tok1/layer0 bundle is generated from the real gpt-oss-20b checkpoint on
    # first use and cached at reference_path (see routed_moe/_reference.py).
    reference = ensure_reference(reference_path, layer_index=0, rows=1, seed=0)
    x = reference["x"].to(torch.bfloat16)
    topk_indices = reference["topk_indices"].to(torch.long)
    topk_weights = reference["topk_weights"].to(torch.bfloat16)
    hf_output = reference["hf_output"].to(torch.bfloat16)
    golden_a_output = reference["golden_a_output"].to(torch.bfloat16)

    config_dict = load_json("config.json")
    hidden = int(config_dict["hidden_size"])
    intermediate = int(config_dict["intermediate_size"])
    rows = int(x.shape[0])
    if tuple(x.shape) != (rows, hidden):
        raise ValueError(f"reference x shape {tuple(x.shape)} does not match hidden={hidden}")
    if hidden % mlen != 0 or intermediate % mlen != 0:
        raise ValueError(f"hidden/intermediate must be divisible by MLEN={mlen}: {hidden}, {intermediate}")

    tensors = load_layer0_tensors()
    split = split_packed_gate_up(tensors["gate_up_weight"], tensors["gate_up_bias"])
    down_weight = tensors["down_weight"]
    down_bias = tensors["down_bias"]

    selected_experts = sorted({int(v) for v in topk_indices.flatten().tolist()})
    if args.max_selected_experts is not None:
        selected_experts = selected_experts[: args.max_selected_experts]
    if not selected_experts:
        raise ValueError("reference routing selected zero experts")

    print("=" * 80)
    print(
        "GPT-OSS real layer0 fixed-routing emulator check "
        f"(rows={rows}, hidden={hidden}, intermediate={intermediate}, selected_experts={selected_experts})"
    )
    print("=" * 80)

    m_tiles = math.ceil(rows / mlen)
    gate_n_tiles = math.ceil(intermediate / mlen)
    gate_k_tiles = math.ceil(hidden / mlen)
    down_n_tiles = math.ceil(hidden / mlen)
    down_k_tiles = math.ceil(intermediate / mlen)
    print(
        "tile coverage: "
        f"gate/up M={m_tiles}, N={gate_n_tiles}, K={gate_k_tiles}; "
        f"down M={m_tiles}, N={down_n_tiles}, K={down_k_tiles}; "
        f"K-split chunks={math.ceil(gate_k_tiles / 4)}"
    )

    experts = []
    expert_bias_tensors = []
    route_tensors = []
    selected_gate_up = torch.zeros(rows, len(selected_experts), 2 * intermediate, dtype=torch.bfloat16)
    debug_gate_golden = None
    for local_idx, expert_id in enumerate(selected_experts):
        # Clone selected expert views so torch.save does not serialize the
        # original full-expert storage behind a narrow view.
        w_gate = split.gate_weight[expert_id].to(torch.bfloat16).contiguous().clone()
        w_up = split.up_weight[expert_id].to(torch.bfloat16).contiguous().clone()
        w_down = down_weight[expert_id].to(torch.bfloat16).contiguous().clone()
        b_gate = _expanded_bias(split.gate_bias[expert_id], rows)
        b_up = _expanded_bias(split.up_bias[expert_id], rows)
        b_down = _expanded_bias(down_bias[expert_id], rows)
        if args.zero_bias:
            b_gate = torch.zeros_like(b_gate)
            b_up = torch.zeros_like(b_up)
            b_down = torch.zeros_like(b_down)
        route = _make_route_matrix(topk_indices, topk_weights, expert_id, hidden)

        gate = _linear_projection_golden(x, w_gate, mlen=mlen, hbm_input=True)
        up = _linear_projection_golden(x, w_up, mlen=mlen, hbm_input=True)
        gate = _bf16(gate.float() + b_gate.float())
        up = _bf16(up.float() + b_up.float())
        if local_idx == 0:
            debug_gate_golden = gate
        selected_gate_up[:, local_idx, 0::2] = gate
        selected_gate_up[:, local_idx, 1::2] = up

        experts.append((w_gate, w_up, w_down))
        expert_bias_tensors.append((b_gate, b_up, b_down))
        route_tensors.append(route)

    gate = selected_gate_up[..., 0::2].float()
    up = selected_gate_up[..., 1::2].float()
    per_token_clamp = []
    for token_idx in range(rows):
        token_gate = gate[token_idx]
        token_up = up[token_idx]
        per_token_clamp.append(
            {
                "token_index": token_idx,
                "gate_gt_7": int((token_gate > 7.0).sum().item()),
                "gate_lt_neg_7": int((token_gate < -7.0).sum().item()),
                "up_gt_7": int((token_up > 7.0).sum().item()),
                "up_lt_neg_7": int((token_up < -7.0).sum().item()),
            }
        )
    clamp_counts = {
        "gate_gt_7": int((gate > 7.0).sum().item()),
        "gate_lt_neg_7": int((gate < -7.0).sum().item()),
        "up_gt_7": int((up > 7.0).sum().item()),
        "up_lt_neg_7": int((up < -7.0).sum().item()),
        "gate_min": float(gate.min().item()),
        "gate_max": float(gate.max().item()),
        "up_min": float(up.min().item()),
        "up_max": float(up.max().item()),
        "per_token": per_token_clamp,
    }
    print(f"clamp coverage: {json.dumps(clamp_counts, sort_keys=True)}")
    if clamp_counts["gate_gt_7"] + clamp_counts["up_gt_7"] + clamp_counts["up_lt_neg_7"] == 0:
        raise AssertionError("selected real layer0 path did not trigger GPT-OSS clamp")

    if args.debug_stage == "gate":
        assert debug_gate_golden is not None
        return _run_gate_debug(
            args=args,
            build_dir=build_dir,
            hw=hw,
            x=x,
            w_gate=experts[0][0],
            b_gate=expert_bias_tensors[0][0],
            golden=debug_gate_golden,
            mlen=mlen,
            blen=blen,
            rows=rows,
            hidden=hidden,
            intermediate=intermediate,
        )

    golden = torch.zeros(rows, hidden, dtype=torch.bfloat16)
    for (w_gate, w_up, w_down), (b_gate, b_up, b_down), route in zip(
        experts, expert_bias_tensors, route_tensors, strict=True
    ):
        expert_out = _expert_golden_b(
            x,
            w_gate,
            w_up,
            w_down,
            mlen=mlen,
            b_gate=b_gate,
            b_up=b_up,
            b_down=b_down,
        )
        golden = _bf16(golden.float() + _bf16(expert_out.float() * quantize_to_mxfp(route).float()).float())

    a_b_stats = compare_stats(golden, hf_output, rtol=0.15)
    host_a_b_stats = compare_stats(golden, golden_a_output, rtol=0.15)
    # Anchor: the independent high-precision Golden A must track HF within a small
    # relative RMS. Golden A is real MXFP4-dequant expert math over the Golden-A
    # routing, so it is close to — but not bit-identical with — HF (BF16 rounding,
    # and it fails only if the Golden-A routing diverges from HF's for this sample).
    # An elementwise allclose is too strict for BF16 tails, so gate on rel RMS.
    _ANCHOR_REL_RMS_MAX = 0.01
    hf_a_stats = compare_stats(golden_a_output, hf_output, rtol=1e-2)
    if hf_a_stats.rel_rms > _ANCHOR_REL_RMS_MAX:
        raise AssertionError(
            "Golden A (independent high-precision MoE) must track HF within "
            f"{_ANCHOR_REL_RMS_MAX:.1%} rel RMS for this fixed-routing proof: "
            f"rel_rms={hf_a_stats.rel_rms:.6g}, max_abs={hf_a_stats.max_abs_error:.6g}, "
            f"allclose={hf_a_stats.allclose}"
        )

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    x_input = prog.input("X", shape=(rows, hidden))
    x_vram = prog.load_batch(x_input, name="X")

    expert_inputs = []
    route_inputs = []
    input_tensors = {"X": x}
    for local_idx, ((w_gate, w_up, w_down), route) in enumerate(zip(experts, route_tensors, strict=True)):
        w_gate_name = f"W_{local_idx}_gate"
        w_up_name = f"W_{local_idx}_up"
        w_down_name = f"W_{local_idx}_down"
        route_name = f"route{local_idx}"
        w_gate_input = prog.input(w_gate_name, shape=(hidden, intermediate))
        w_up_input = prog.input(w_up_name, shape=(hidden, intermediate))
        w_down_input = prog.input(w_down_name, shape=(intermediate, hidden))
        route_input = prog.input(route_name, shape=(rows, hidden))
        expert_inputs.append((w_gate_input, w_up_input, w_down_input))
        route_inputs.append(route_input)
        input_tensors[w_gate_name] = w_gate
        input_tensors[w_up_name] = w_up
        input_tensors[w_down_name] = w_down
        input_tensors[route_name] = route

    zero = prog.fp_var("zero", size=1)
    limit_pos = prog.fp_var("gpt_oss_limit_pos", size=rows)
    limit_neg = prog.fp_var("gpt_oss_limit_neg", size=rows)
    one = prog.fp_var("one", size=rows)
    neg_alpha = prog.fp_var("neg_alpha", size=rows)

    route_vrams = [prog.load_batch(route_input, name=route_input.display_name) for route_input in route_inputs]

    physical_rows = max(blen, math.ceil(rows / blen) * blen)
    bias_physical_inter = (physical_rows, intermediate)
    bias_physical_hidden = (physical_rows, hidden)
    bias_sizes = []
    for _ in expert_bias_tensors:
        bias_sizes.extend(
            [
                bias_physical_inter[0] * bias_physical_inter[1],
                bias_physical_inter[0] * bias_physical_inter[1],
                bias_physical_hidden[0] * bias_physical_hidden[1],
            ]
        )
    bias_base = 0
    # X and route matrices were already allocated above; put prestaged biases
    # after those regions so mark_used cannot overlap normal loads.
    if prog.vram_allocator._vmm.next_bump > bias_base:
        bias_base = math.ceil(prog.vram_allocator._vmm.next_bump / (mlen * mlen)) * (mlen * mlen)
    bias_total = sum(math.ceil(size / (mlen * mlen)) * (mlen * mlen) for size in bias_sizes)
    vram_preload = torch.zeros(bias_base + bias_total, dtype=torch.bfloat16)
    next_bias_addr = bias_base
    expert_bias_vrams = []
    for local_idx, (b_gate, b_up, b_down) in enumerate(expert_bias_tensors):
        local_bias_vrams = []
        for suffix, tensor, physical_shape in (
            ("gate", b_gate, bias_physical_inter),
            ("up", b_up, bias_physical_inter),
            ("down", b_down, bias_physical_hidden),
        ):
            size = physical_shape[0] * physical_shape[1]
            aligned_size = math.ceil(size / (mlen * mlen)) * (mlen * mlen)
            name = f"B{local_idx}_{suffix}"
            local_bias_vrams.append(
                prestage_bf16_vram_matrix(
                    prog=prog,
                    name=name,
                    tensor=tensor,
                    vram_addr=next_bias_addr,
                    physical_shape=physical_shape,
                    vram_preload=vram_preload,
                )
            )
            next_bias_addr += aligned_size
        expert_bias_vrams.append(tuple(local_bias_vrams))

    output_vram = prog.gpt_oss_moe_fixed_routing_v0(
        x_vram,
        experts=expert_inputs,
        route_weights=route_vrams,
        expert_biases=expert_bias_vrams,
        rows=rows,
        intermediate=intermediate,
        constants=(zero, limit_pos, limit_neg, one, neg_alpha),
        name="real_layer0_fixed_routing",
    )
    isa = prog.compile()

    fp_preload = [0.0] + [7.0] * rows + [-7.0] * rows + [1.0] * rows + [-1.702] * rows + [0.0] * 8
    tensor_layouts = infer_hbm_tensor_layouts(input_tensors)
    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=tensor_layouts,
    )

    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_real_layer0",
        data=None,
        specified_data_order=list(input_tensors),
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
        tensor_layouts=tensor_layouts,
    )

    output_vram_addr = prog._compiler.get_vram_addr(output_vram.name)
    comparison_params = _comparison_params_for(output_vram, rows=rows, hidden=hidden, mlen=mlen, golden=golden)
    (build_dir / "comparison_params.json").write_text(json.dumps(comparison_params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(isa)

    manifest = {
        "reference_path": str(reference_path),
        "selected_experts": selected_experts,
        "rows": rows,
        "hidden": hidden,
        "intermediate": intermediate,
        "mlen": mlen,
        "blen": blen,
        "tile_coverage": {
            "gate_up_m_tiles": m_tiles,
            "gate_up_n_tiles": gate_n_tiles,
            "gate_up_k_tiles": gate_k_tiles,
            "down_m_tiles": m_tiles,
            "down_n_tiles": down_n_tiles,
            "down_k_tiles": down_k_tiles,
            "k_split_chunks": math.ceil(gate_k_tiles / 4),
        },
        "clamp": clamp_counts,
        "hf_vs_golden_a": _stats_dict(hf_a_stats),
        "golden_a_vs_golden_b": _stats_dict(a_b_stats),
        "host_fixed_routing_a_vs_golden_b": _stats_dict(host_a_b_stats),
        "output_vram_row": output_vram_addr // mlen,
        "asm_lines": len(isa.splitlines()),
    }
    (build_dir / "real_layer0_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    if args.no_run:
        return {"build_dir": str(build_dir), "ran": False, "manifest": manifest}

    metrics = run_and_assert(
        build_dir,
        "gpt_oss_real_layer0",
        mlen=mlen,
        blen=blen,
        threads=args.emu_threads,
    )
    results, params = compare_emulator_output(build_dir)
    emu_output = results["simulated_values"].reshape(rows, hidden).to(torch.bfloat16)
    emu_b_stats = compare_stats(emu_output, golden, rtol=0.02)
    emu_hf_stats = compare_stats(emu_output, hf_output, rtol=0.15)
    emu_b_elementwise = _strict_elementwise_details(emu_output, golden, rtol=0.02)
    emu_b_tail_gate = _tail_gate_details(
        emu_output,
        golden,
        strict_details=emu_b_elementwise,
        clamp_counts=clamp_counts,
        rtol=0.02,
    )

    summary = {
        **manifest,
        "run_metrics": metrics,
        "comparison_params": params,
        "emulator_compare_raw": {
            key: results[key]
            for key in (
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
                "golden_shape",
                "simulated_shape",
            )
            if key in results
        },
        "emulator_vs_golden_b": _stats_dict(emu_b_stats),
        "emulator_vs_golden_b_elementwise": emu_b_elementwise,
        "emulator_vs_golden_b_tail_gate": emu_b_tail_gate,
        "emulator_vs_hf_record_only": _stats_dict(emu_hf_stats),
        "l1_external_proof_gate": {
            "hf_vs_golden_a_exact": hf_a_stats.max_abs_error == 0.0 and hf_a_stats.allclose,
            "emu_vs_golden_b_rel_rms_under_2pct": emu_b_stats.rel_rms <= 0.02,
            "emu_vs_golden_b_strict_allclose": emu_b_stats.allclose,
            "emu_vs_golden_b_tail_gate_passed": emu_b_tail_gate["passed"],
            "passed": emu_b_stats.rel_rms <= 0.02 and (emu_b_stats.allclose or emu_b_tail_gate["passed"]),
        },
    }
    (build_dir / "real_layer0_results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if emu_b_stats.rel_rms > 0.02 or (not emu_b_stats.allclose and not emu_b_tail_gate["passed"]):
        raise AssertionError(
            f"emulator vs Golden B failed: rel_rms={emu_b_stats.rel_rms:.6g} "
            f"(limit 2%), allclose={emu_b_stats.allclose}, tail_gate={emu_b_tail_gate['passed']}, "
            f"pass_rate={emu_b_stats.pass_rate:.2%}, max_abs={emu_b_stats.max_abs_error:.6g}"
        )
    return {"build_dir": str(build_dir), "ran": True, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=_ATEN_BUILD_DIR / "gpt_oss_real_layer0_tok1" / "hf_layer0_moe_reference.pt",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "gpt_oss_real_layer0_emu",
    )
    parser.add_argument("--max-selected-experts", type=int, default=None)
    parser.add_argument("--emu-threads", type=int, default=None)
    parser.add_argument("--debug-stage", choices=("full", "gate"), default="full")
    parser.add_argument("--zero-bias", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    run_real_layer0(args)


if __name__ == "__main__":
    main()
