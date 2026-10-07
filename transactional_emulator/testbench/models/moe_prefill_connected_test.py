"""Transactional multi-token MoE proofs for Nemotron 3 and Kimi K3.

The fixture keeps the complete routing data path while compacting each expert
to 64-wide deterministic matrices: router logits, correction-bias Top-K,
dynamic expert table lookup, expert activation, route weighting, scatter-add,
shared expert, combine and residual.  S16 is suitable for CI; S128 exercises
the full 256-entry Top-2 routing lifetime without requiring a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from compiler.aten.kimi3.blocks import (
    KimiLatentMoeConstants,
    KimiLatentMoeShape,
    KimiLatentMoeWeights,
    emit_kimi_latent_moe_residual_block,
)
from compiler.aten.nemotron3.blocks import (
    NemotronMoeConstants,
    emit_nemotron_moe_block,
)
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_routed_moe import KimiSituFPConstants
from transactional_emulator.testbench.aten.configurable import setup_hw
from transactional_emulator.testbench.aten.golden import (
    _active_precision_settings,
)
from transactional_emulator.testbench.emulator_runner import (
    compare_emulator_output,
    run_emulator,
)
from transactional_emulator.testbench.gpt_oss_testkit import (
    _comparison_params_for,
)
from transactional_emulator.testbench.layout_utils import (
    infer_hbm_tensor_layouts,
    prestage_bf16_vram_matrix,
)
from transactional_emulator.testbench.models.kimi3.connected_blocks_test import (
    BETA,
    BLEN,
    EPS,
    LINEAR_BETA,
    MLEN,
    TensorSet,
    _bf16,
    _bf16_layout,
    _exact,
    _linear,
    _moe_golden,
    _register_expert_table,
    _register_weight,
    _rms,
    _set_matrix_kv_plain_bf16,
)
from transactional_emulator.testbench.models.nemotron3.mamba_connected_test import (
    _nemotron_moe_golden,
    _register_moe as _register_nemotron_moe,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


HIDDEN = MLEN
TOP_K = 2
EXPERTS = 4
MOE_CHUNK = 16


def _align(value: int, alignment: int) -> int:
    return math.ceil(value / alignment) * alignment


def _prompt(tokens: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(91)
    random = torch.randn(tokens, HIDDEN, generator=generator) * 0.2
    token_bias = torch.arange(tokens, dtype=torch.float32).unsqueeze(1) / 128.0
    feature_bias = torch.arange(HIDDEN, dtype=torch.float32).unsqueeze(0) / 2048.0
    return (random + token_bias + feature_bias).to(torch.bfloat16)


def _correction() -> torch.Tensor:
    return torch.tensor(
        [[0.0, 0.125, 0.25, -0.125] + [0.0] * (MLEN - EXPERTS)],
        dtype=torch.bfloat16,
    )


def _allocate_kimi_moe_constants(
    prog: PlenaCompiler,
    *,
    rows: int,
) -> tuple[KimiLatentMoeConstants, list[float]]:
    lanes = max(BLEN, rows)
    zero = prog.fp_var("zero", 1)
    one = prog.fp_var("one", lanes)
    neg_one = prog.fp_var("neg_one", lanes)
    beta = prog.fp_var("beta", lanes)
    neg_two_beta = prog.fp_var("neg_two_beta", lanes)
    linear_beta = prog.fp_var("linear_beta", lanes)
    neg_two_linear_beta = prog.fp_var("neg_two_linear_beta", lanes)
    zero_row = prog.fp_var("zero_row", MLEN)
    norm_eps = prog.fp_var("norm_eps", 1)
    norm_reciprocal = prog.fp_var("norm_reciprocal", 1)
    routed_eps = prog.fp_var("routed_eps", 1)
    routed_reciprocal = prog.fp_var("routed_reciprocal", 1)
    variables = (
        zero,
        one,
        neg_one,
        beta,
        neg_two_beta,
        linear_beta,
        neg_two_linear_beta,
        zero_row,
        norm_eps,
        norm_reciprocal,
        routed_eps,
        routed_reciprocal,
    )
    preload = [0.0] * max(var.address + var.size for var in variables)

    def fill(var, value: float) -> None:
        for index in range(var.size):
            preload[var.address + index] = value

    fill(one, 1.0)
    fill(neg_one, -1.0)
    fill(beta, BETA)
    fill(neg_two_beta, -2.0 / BETA)
    fill(linear_beta, LINEAR_BETA)
    fill(neg_two_linear_beta, -2.0 / LINEAR_BETA)
    fill(norm_eps, EPS)
    fill(norm_reciprocal, 1.0 / HIDDEN)
    fill(routed_eps, EPS)
    fill(routed_reciprocal, 1.0 / HIDDEN)
    return (
        KimiLatentMoeConstants(
            situ=KimiSituFPConstants(
                zero=zero,
                one=one,
                neg_one=neg_one,
                beta=beta,
                neg_two_over_beta=neg_two_beta,
                linear_beta=linear_beta,
                neg_two_over_linear_beta=neg_two_linear_beta,
            ),
            zero_row=zero_row,
            norm_eps=norm_eps.address,
            norm_reciprocal_hidden=norm_reciprocal.address,
            routed_norm_eps=routed_eps.address,
            routed_norm_reciprocal_hidden=routed_reciprocal.address,
        ),
        preload,
    )


def _register_kimi_moe(
    prog: PlenaCompiler,
    tensors: TensorSet,
) -> tuple[KimiLatentMoeShape, KimiLatentMoeWeights]:
    router = torch.zeros(HIDDEN, EXPERTS, dtype=torch.bfloat16)
    for expert in range(EXPERTS):
        router[:, expert] = _exact(
            (HIDDEN,), expert + 1, expert, scale=1 / 32
        )
    gate_values = [
        _exact((HIDDEN, HIDDEN), expert + 1, expert, 1 / 32)
        for expert in range(EXPERTS)
    ]
    up_values = [
        _exact((HIDDEN, HIDDEN), expert + 2, expert + 1, 1 / 32)
        for expert in range(EXPERTS)
    ]
    down_values = [
        _exact((HIDDEN, HIDDEN), expert + 3, expert + 2, 1 / 32)
        for expert in range(EXPERTS)
    ]
    weights = KimiLatentMoeWeights(
        router=_register_weight(
            prog, tensors, "W_moe_router", router, bf16=True
        ),
        routed_down=_register_weight(
            prog,
            tensors,
            "W_moe_latent_down",
            _exact((HIDDEN, HIDDEN), 2, 2, 1 / 32),
        ),
        routed_up=_register_weight(
            prog,
            tensors,
            "W_moe_latent_up",
            _exact((HIDDEN, HIDDEN), 3, 3, 1 / 32),
        ),
        routed_gate=_register_expert_table(
            prog, tensors, prefix="W_expert_gate", values=gate_values
        ),
        routed_up_expert=_register_expert_table(
            prog, tensors, prefix="W_expert_up", values=up_values
        ),
        routed_down_expert=_register_expert_table(
            prog, tensors, prefix="W_expert_down", values=down_values
        ),
        shared=(
            _register_weight(
                prog,
                tensors,
                "W_shared_gate",
                _exact((HIDDEN, HIDDEN), 4, 1, 1 / 32),
            ),
            _register_weight(
                prog,
                tensors,
                "W_shared_up",
                _exact((HIDDEN, HIDDEN), 2, 3, 1 / 32),
            ),
            _register_weight(
                prog,
                tensors,
                "W_shared_down",
                _exact((HIDDEN, HIDDEN), 3, 4, 1 / 32),
            ),
        ),
    )
    return (
        KimiLatentMoeShape(
            hidden=HIDDEN,
            routed_hidden=HIDDEN,
            intermediate=HIDDEN,
            shared_intermediate=HIDDEN,
            num_experts=EXPERTS,
            top_k=TOP_K,
        ),
        weights,
    )


def _expected_routes(
    model: str,
    normalized: torch.Tensor,
    tensors: TensorSet,
    correction: torch.Tensor,
) -> list[list[int]]:
    router_name = "W_NEMOTRON_ROUTER" if model == "nemotron" else "W_moe_router"
    logits = _linear(normalized, tensors.values[router_name]).float()
    bias = correction[:, :EXPERTS].float()
    routes: list[list[int]] = []
    for token in range(normalized.shape[0]):
        ranking = logits[token] + bias[0]
        routes.append(
            sorted(
                range(EXPERTS),
                key=lambda expert: (-float(ranking[expert]), expert),
            )[:TOP_K]
        )
    return routes


def _read_and_check_routes(
    path: Path,
    expected: list[list[int]],
) -> dict[str, object]:
    raw = np.fromfile(path, dtype="<i4", count=len(expected) * TOP_K)
    if raw.size != len(expected) * TOP_K:
        raise AssertionError("INT SRAM dump ended before all Top-K routes")
    actual = raw.reshape(len(expected), TOP_K).tolist()
    for token, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if sorted(got) != sorted(want):
            raise AssertionError(
                f"token {token} route mismatch: actual={got}, expected={want}"
            )
    unique_experts = sorted({expert for route in actual for expert in route})
    unique_patterns = len({tuple(sorted(route)) for route in actual})
    if len(expected) > 1 and unique_patterns < 2:
        raise AssertionError("multi-token fixture did not exercise distinct routes")
    return {
        "route_entries": len(expected) * TOP_K,
        "unique_experts": unique_experts,
        "unique_route_patterns": unique_patterns,
        "routes": actual,
    }


def build_and_run(
    model: str,
    build_dir: Path,
    *,
    tokens: int,
) -> dict[str, object]:
    if model not in {"nemotron", "kimi"}:
        raise ValueError(f"unsupported model {model!r}")
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(
        argparse.Namespace(mlen=MLEN, vlen=None, blen=BLEN, hlen=None),
        build_dir,
    )
    _set_matrix_kv_plain_bf16()
    prog = PlenaCompiler(
        mlen=MLEN,
        blen=BLEN,
        real_data_ratio=hw.real_data_ratio,
        compact_matrix_loops=True,
    )
    tensors = TensorSet(values={}, bf16_names=set())
    physical_rows = _align(tokens, MLEN)
    hidden_value = _prompt(tokens)
    correction_value = _correction()
    vram_preload = torch.zeros(
        (physical_rows + BLEN) * HIDDEN, dtype=torch.bfloat16
    )
    hidden = prestage_bf16_vram_matrix(
        prog=prog,
        name=f"{model.upper()}_MOE_PROMPT",
        tensor=hidden_value,
        vram_addr=0,
        physical_shape=(physical_rows, HIDDEN),
        vram_preload=vram_preload,
    )
    correction = prestage_bf16_vram_matrix(
        prog=prog,
        name=f"{model.upper()}_MOE_CORRECTION",
        tensor=correction_value,
        vram_addr=physical_rows * HIDDEN,
        physical_shape=(BLEN, MLEN),
        vram_preload=vram_preload,
    )
    tensors.add("MOE_CORRECTION", correction_value)
    precision = _active_precision_settings()

    if model == "nemotron":
        prog.fp_var("zero", 1)
        zero_row = prog.fp_var("zero_row", max(MLEN, tokens))
        norm_eps = prog.fp_var("norm_eps", 1)
        norm_reciprocal = prog.fp_var("norm_reciprocal", 1)
        route_scale = prog.fp_var("route_scale", TOP_K)
        fp_preload = [0.0] * (route_scale.address + route_scale.size)
        fp_preload[norm_eps.address] = EPS
        fp_preload[norm_reciprocal.address] = 1.0 / HIDDEN
        for index in range(route_scale.size):
            fp_preload[route_scale.address + index] = 2.5
        residual = prog.vram_copy(
            hidden, name="nemotron_moe_residual", num_rows=tokens
        )
        moe_input = prog.vram_copy(
            hidden, name="nemotron_moe_input", num_rows=tokens
        )
        prog.rms_norm(
            moe_input,
            eps_offset=norm_eps.address,
            reci_hid_offset=norm_reciprocal.address,
        )
        shape, weights = _register_nemotron_moe(prog, tensors)
        mixer_out = emit_nemotron_moe_block(
            prog,
            moe_input,
            shape=shape,
            weights=weights,
            correction_bias=correction,
            constants=NemotronMoeConstants(
                zero_row=zero_row,
                routed_scale=route_scale,
            ),
            rows=tokens,
            name="nemotron_moe_prefill",
        )
        prog.vram_add(residual, mixer_out, num_rows=tokens)
        output = residual
        normalized = _rms(hidden_value, precision=precision)
        golden_mixer = torch.cat(
            [
                _nemotron_moe_golden(
                    normalized[token : token + 1],
                    tensors,
                    correction_value,
                    precision=precision,
                )
                for token in range(tokens)
            ],
            dim=0,
        )
        golden = _bf16(
            hidden_value.float() + golden_mixer.float(), precision=precision
        )
    else:
        constants, fp_preload = _allocate_kimi_moe_constants(
            prog, rows=min(tokens, MOE_CHUNK)
        )
        shape, weights = _register_kimi_moe(prog, tensors)
        if tokens <= MOE_CHUNK:
            output = emit_kimi_latent_moe_residual_block(
                prog,
                hidden,
                shape=shape,
                weights=weights,
                correction_bias=correction,
                constants=constants,
                rows=tokens,
                name="kimi_moe_prefill",
                loop_topk=False,
            )
        else:
            output = prog.alloc(
                "kimi_moe_prefill_output",
                rows=tokens,
                cols=HIDDEN,
                strict=False,
                physical_shape=(physical_rows, HIDDEN),
            )
            hidden_base = prog.get_vram_addr(hidden.name)
            for start in range(0, tokens, MOE_CHUNK):
                chunk_rows = min(MOE_CHUNK, tokens - start)
                chunk_input = prog.alloc_at(
                    f"kimi_moe_prefill_input_{start}",
                    rows=chunk_rows,
                    cols=HIDDEN,
                    vram_addr=hidden_base + start * HIDDEN,
                    physical_shape=(chunk_rows, HIDDEN),
                )
                chunk_output = emit_kimi_latent_moe_residual_block(
                    prog,
                    chunk_input,
                    shape=shape,
                    weights=weights,
                    correction_bias=correction,
                    constants=constants,
                    rows=chunk_rows,
                    int_sram_base=start * TOP_K,
                    name=f"kimi_moe_prefill_chunk_{start}",
                    loop_topk=False,
                )
                prog.vram_copy_region(
                    output,
                    chunk_output,
                    num_rows=chunk_rows,
                    num_cols=HIDDEN,
                    dst_row_offset=start,
                )
                prog.free_tensor(chunk_output)
        normalized = _rms(hidden_value, precision=precision)
        golden = torch.cat(
            [
                _moe_golden(
                    hidden_value[token : token + 1],
                    tensors,
                    precision=precision,
                )
                for token in range(tokens)
            ],
            dim=0,
        )

    expected_routes = _expected_routes(
        model, normalized, tensors, correction_value
    )
    assembly = prog.compile()
    input_tensors = {
        name: value
        for name, value in tensors.values.items()
        if name != "MOE_CORRECTION"
    }
    layouts = infer_hbm_tensor_layouts(input_tensors)
    for name in tensors.bf16_names:
        layouts[name] = _bf16_layout(input_tensors[name])
    hbm_addrs = {
        name: prog._compiler.get_hbm_layout(name).hbm_base_addr
        for name in input_tensors
    }
    create_sim_env(
        input_tensors,
        assembly,
        {"original_output": golden},
        fp_preload=fp_preload,
        int_preload=[0] * 1024,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=layouts,
    )
    create_mem_for_sim(
        data_size=MLEN,
        mode="behave_sim",
        asm=f"{model}_moe_prefill_s{tokens}",
        specified_data_order=sorted(input_tensors, key=hbm_addrs.__getitem__),
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=layouts,
        hbm_addrs=hbm_addrs,
    )
    params = _comparison_params_for(
        output,
        rows=tokens,
        hidden=HIDDEN,
        mlen=MLEN,
        golden=golden,
    )
    params.update(
        {"atol": 0.03, "rtol": 0.04, "min_allclose_match_rate": 100.0}
    )
    (build_dir / "comparison_params.json").write_text(
        json.dumps(params, indent=2) + "\n"
    )
    (build_dir / "generated_asm_code.asm").write_text(assembly)
    hbm_size = _align(prog._next_hbm_addr, 64)
    (build_dir / "hbm_size.txt").write_text(f"{hbm_size}\n")

    metrics = run_emulator(build_dir, stage_profile=True, dump_cwd=build_dir)
    results, _ = compare_emulator_output(build_dir, verbose=False)
    if float(results.get("allclose_match_rate", 0.0)) < 100.0:
        raise AssertionError(f"{model} S{tokens} MoE output mismatch: {results}")
    routing = _read_and_check_routes(
        build_dir / "intsram_dump.bin", expected_routes
    )
    summary = {
        "model": model,
        "phase": "prefill",
        "tokens": tokens,
        "experts": EXPERTS,
        "top_k": TOP_K,
        "sim_latency_ns": metrics.get("sim_latency_ns"),
        "sim_cycles": metrics.get("sim_latency_cycles"),
        "max_abs_error": results.get("max_error"),
        "allclose_match_rate": results.get("allclose_match_rate"),
        "routing": routing,
        "hbm_bytes": hbm_size,
    }
    (build_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=("nemotron", "kimi", "all"), default="all"
    )
    parser.add_argument("--tokens", type=int, choices=(16, 128), default=16)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(
            "transactional_emulator/testbench/build/moe_prefill_connected"
        ),
    )
    args = parser.parse_args()
    root = args.build_dir.expanduser().resolve()
    summaries = []
    if args.model in {"nemotron", "all"}:
        summaries.append(
            build_and_run("nemotron", root / "nemotron", tokens=args.tokens)
        )
    if args.model in {"kimi", "all"}:
        summaries.append(build_and_run("kimi", root / "kimi", tokens=args.tokens))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
