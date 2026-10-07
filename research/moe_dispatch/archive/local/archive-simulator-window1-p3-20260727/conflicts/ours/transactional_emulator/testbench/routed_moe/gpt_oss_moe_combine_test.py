"""GPT-OSS MoE fixed-routing two-expert combine emulator smoke.

This is the first end-to-end fixed-routing device-side smoke:

    X -> expert0 -> route_weight0 * out0
    X -> expert1 -> route_weight1 * out1
    output = weighted_out0 + weighted_out1

Router/top-k/dispatch remain host-fixed. The routing weights are loaded as
expanded [rows, hidden] matrices so this test exercises PLENA vram_mul/vram_add
without introducing gather/scatter or scalar broadcast machinery.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import add_hw_args, resolve_rows, setup_hw
from transactional_emulator.testbench.aten.golden import quantize_to_mxfp
from transactional_emulator.testbench.emulator_runner import compare_emulator_output, run_and_assert
from transactional_emulator.testbench.layout_utils import infer_hbm_tensor_layouts
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.gpt_oss_testkit import (
    _activation_golden,
    _bf16,
    _exact_mxfp8_tensor,
    _linear_projection_golden,
)


def _expert_golden(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    mlen: int,
    b_gate: torch.Tensor | None = None,
    b_up: torch.Tensor | None = None,
    b_down: torch.Tensor | None = None,
) -> torch.Tensor:
    gate = _linear_projection_golden(x, w_gate, mlen=mlen, hbm_input=True)
    up = _linear_projection_golden(x, w_up, mlen=mlen, hbm_input=True)
    if b_gate is not None:
        gate = _bf16(gate.float() + quantize_to_mxfp(b_gate).float())
    if b_up is not None:
        up = _bf16(up.float() + quantize_to_mxfp(b_up).float())
    hidden = _activation_golden(gate, up)
    out = _linear_projection_golden(hidden, w_down, mlen=mlen, hbm_input=False)
    if b_down is not None:
        out = _bf16(out.float() + quantize_to_mxfp(b_down).float())
    return out


def _route_rows(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(rows, dtype=torch.float32)
    route0 = torch.tensor(0.25, dtype=torch.float32) + (idx % 4) * torch.tensor(0.25, dtype=torch.float32)
    return route0.unsqueeze(1), (1.0 - route0).unsqueeze(1)


def _print_tile_and_clamp_stats(
    *,
    mlen: int,
    mram_tile_capacity: int,
    rows: int,
    hidden: int,
    intermediate: int,
    projections: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    m_tiles = math.ceil(rows / mlen)
    gate_n_tiles = math.ceil(intermediate / mlen)
    gate_k_tiles = math.ceil(hidden / mlen)
    down_n_tiles = math.ceil(hidden / mlen)
    down_k_tiles = math.ceil(intermediate / mlen)
    print("\n--- v0 stress coverage ---")
    print(
        "gate/up projection tiles: "
        f"M={m_tiles}, N={gate_n_tiles}, K={gate_k_tiles}, "
        f"K-split chunks={math.ceil(gate_k_tiles / mram_tile_capacity)}"
    )
    print(
        "down projection tiles: "
        f"M={m_tiles}, N={down_n_tiles}, K={down_k_tiles}, "
        f"K-split chunks={math.ceil(down_k_tiles / mram_tile_capacity)}"
    )
    for name, (gate, up) in projections.items():
        gate_hi = int((gate > 7).sum().item())
        gate_lo = int((gate < -7).sum().item())
        up_hi = int((up > 7).sum().item())
        up_lo = int((up < -7).sum().item())
        total = gate.numel()
        print(
            f"{name}: gate range=[{gate.min().item():.4g}, {gate.max().item():.4g}], "
            f"gate >7={gate_hi}/{total}, gate <-7={gate_lo}/{total}; "
            f"up range=[{up.min().item():.4g}, {up.max().item():.4g}], "
            f"up >7={up_hi}/{total}, up <-7={up_lo}/{total}"
        )


def run_combine_smoke(args: argparse.Namespace) -> dict:
    mlen = args.mlen
    blen = args.blen
    rows, batch_size, seq_len = resolve_rows(args, default_seq=blen)
    hidden = args.hidden_size or mlen
    intermediate = args.intermediate_size or mlen

    if hidden % mlen != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by MLEN ({mlen})")
    if intermediate % mlen != 0:
        raise ValueError(f"intermediate ({intermediate}) must be divisible by MLEN ({mlen})")

    build_dir = args.build_dir
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(
        "GPT-OSS MoE fixed-routing combine emulator smoke "
        f"(mlen={mlen}, blen={blen}, batch={batch_size}, seq={seq_len}, rows={rows})"
    )
    print("=" * 80)

    x = _exact_mxfp8_tensor((rows, hidden), stride=1)
    w0_gate = _exact_mxfp8_tensor((hidden, intermediate), stride=2, offset=1)
    w0_up = _exact_mxfp8_tensor((hidden, intermediate), stride=3, offset=2)
    w0_down = _exact_mxfp8_tensor((intermediate, hidden), stride=4, offset=3)
    w1_gate = _exact_mxfp8_tensor((hidden, intermediate), stride=3, offset=4)
    w1_up = _exact_mxfp8_tensor((hidden, intermediate), stride=4, offset=0)
    w1_down = _exact_mxfp8_tensor((intermediate, hidden), stride=2, offset=2)

    b0_gate = b0_up = b0_down = None
    b1_gate = b1_up = b1_down = None
    if args.include_bias:
        b0_gate = _exact_mxfp8_tensor((1, intermediate), stride=2, offset=1).repeat(rows, 1)
        b0_up = _exact_mxfp8_tensor((1, intermediate), stride=3, offset=2).repeat(rows, 1)
        b0_down = _exact_mxfp8_tensor((1, hidden), stride=4, offset=3).repeat(rows, 1)
        b1_gate = _exact_mxfp8_tensor((1, intermediate), stride=3, offset=4).repeat(rows, 1)
        b1_up = _exact_mxfp8_tensor((1, intermediate), stride=4, offset=0).repeat(rows, 1)
        b1_down = _exact_mxfp8_tensor((1, hidden), stride=2, offset=2).repeat(rows, 1)

    route0_row, route1_row = _route_rows(rows)
    route0 = route0_row.repeat(1, hidden)
    route1 = route1_row.repeat(1, hidden)

    gate0 = _linear_projection_golden(x, w0_gate, mlen=mlen, hbm_input=True)
    up0 = _linear_projection_golden(x, w0_up, mlen=mlen, hbm_input=True)
    gate1 = _linear_projection_golden(x, w1_gate, mlen=mlen, hbm_input=True)
    up1 = _linear_projection_golden(x, w1_up, mlen=mlen, hbm_input=True)
    _print_tile_and_clamp_stats(
        mlen=mlen,
        mram_tile_capacity=4,
        rows=rows,
        hidden=hidden,
        intermediate=intermediate,
        projections={"expert0": (gate0, up0), "expert1": (gate1, up1)},
    )

    out0 = _expert_golden(
        x,
        w0_gate,
        w0_up,
        w0_down,
        mlen=mlen,
        b_gate=b0_gate,
        b_up=b0_up,
        b_down=b0_down,
    )
    out1 = _expert_golden(
        x,
        w1_gate,
        w1_up,
        w1_down,
        mlen=mlen,
        b_gate=b1_gate,
        b_up=b1_up,
        b_down=b1_down,
    )
    route0_q = quantize_to_mxfp(route0)
    route1_q = quantize_to_mxfp(route1)
    weighted0 = _bf16(out0.float() * route0_q.float())
    weighted1 = _bf16(out1.float() * route1_q.float())
    golden = _bf16(weighted0.float() + weighted1.float())
    if args.include_bias:
        no_bias0 = _expert_golden(x, w0_gate, w0_up, w0_down, mlen=mlen)
        no_bias1 = _expert_golden(x, w1_gate, w1_up, w1_down, mlen=mlen)
        no_bias_golden = _bf16(
            _bf16(no_bias0.float() * route0_q.float()).float() + _bf16(no_bias1.float() * route1_q.float()).float()
        )
        bias_delta = (golden.float() - no_bias_golden.float()).abs().max().item()
        print(f"bias changes output: max_abs_delta={bias_delta:.6g}")
        if bias_delta == 0.0:
            raise AssertionError("--include-bias did not change the fixed-routing golden output")

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    x_input = prog.input("X", shape=(rows, hidden))
    route0_input = prog.input("route0", shape=(rows, hidden))
    route1_input = prog.input("route1", shape=(rows, hidden))
    w0_gate_input = prog.input("W0_gate", shape=(hidden, intermediate))
    w0_up_input = prog.input("W0_up", shape=(hidden, intermediate))
    w0_down_input = prog.input("W0_down", shape=(intermediate, hidden))
    w1_gate_input = prog.input("W1_gate", shape=(hidden, intermediate))
    w1_up_input = prog.input("W1_up", shape=(hidden, intermediate))
    w1_down_input = prog.input("W1_down", shape=(intermediate, hidden))
    if args.include_bias:
        b0_gate_input = prog.input("B0_gate", shape=(rows, intermediate))
        b0_up_input = prog.input("B0_up", shape=(rows, intermediate))
        b0_down_input = prog.input("B0_down", shape=(rows, hidden))
        b1_gate_input = prog.input("B1_gate", shape=(rows, intermediate))
        b1_up_input = prog.input("B1_up", shape=(rows, intermediate))
        b1_down_input = prog.input("B1_down", shape=(rows, hidden))

    zero = prog.fp_var("zero", size=1)
    limit_pos = prog.fp_var("gpt_oss_limit_pos", size=rows)
    limit_neg = prog.fp_var("gpt_oss_limit_neg", size=rows)
    one = prog.fp_var("one", size=rows)
    neg_alpha = prog.fp_var("neg_alpha", size=rows)

    x_vram = prog.load_batch(x_input, name="X")
    route0_vram = prog.load_batch(route0_input, name="route0")
    route1_vram = prog.load_batch(route1_input, name="route1")
    expert_biases = None
    if args.include_bias:
        b0_gate_vram = prog.load_batch(b0_gate_input, name="B0_gate")
        b0_up_vram = prog.load_batch(b0_up_input, name="B0_up")
        b0_down_vram = prog.load_batch(b0_down_input, name="B0_down")
        b1_gate_vram = prog.load_batch(b1_gate_input, name="B1_gate")
        b1_up_vram = prog.load_batch(b1_up_input, name="B1_up")
        b1_down_vram = prog.load_batch(b1_down_input, name="B1_down")
        expert_biases = [
            (b0_gate_vram, b0_up_vram, b0_down_vram),
            (b1_gate_vram, b1_up_vram, b1_down_vram),
        ]

    output_vram = prog.gpt_oss_moe_fixed_routing_v0(
        x_vram,
        experts=[
            (w0_gate_input, w0_up_input, w0_down_input),
            (w1_gate_input, w1_up_input, w1_down_input),
        ],
        route_weights=[route0_vram, route1_vram],
        expert_biases=expert_biases,
        rows=rows,
        intermediate=intermediate,
        constants=(zero, limit_pos, limit_neg, one, neg_alpha),
        name="fixed_routing",
    )
    isa = prog.compile()

    input_tensors = {
        "X": x,
        "route0": route0,
        "route1": route1,
        "W0_gate": w0_gate,
        "W0_up": w0_up,
        "W0_down": w0_down,
        "W1_gate": w1_gate,
        "W1_up": w1_up,
        "W1_down": w1_down,
    }
    if args.include_bias:
        input_tensors.update(
            {
                "B0_gate": b0_gate,
                "B0_up": b0_up,
                "B0_down": b0_down,
                "B1_gate": b1_gate,
                "B1_up": b1_up,
                "B1_down": b1_down,
            }
        )
    golden_result = {"original_output": golden}
    fp_preload = [0.0] + [7.0] * rows + [-7.0] * rows + [1.0] * rows + [-1.702] * rows + [0.0] * 8

    tensor_layouts = infer_hbm_tensor_layouts(input_tensors)
    create_sim_env(
        input_tensors,
        isa,
        golden_result,
        fp_preload,
        build_dir=str(build_dir),
        tensor_layouts=tensor_layouts,
    )

    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="gpt_oss_moe_combine",
        data=None,
        specified_data_order=list(input_tensors),
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
        tensor_layouts=tensor_layouts,
    )

    output_vram_addr = prog._compiler.get_vram_addr(output_vram.name)
    comparison_params = {
        "start_row_idx": output_vram_addr // mlen,
        "num_rows": (rows * hidden) // mlen,
        "num_batches": rows,
        "elements_per_batch": hidden,
        "row_dim": mlen,
        "atol": 0.0,
        "rtol": 0.0,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(isa)

    print(f"Generated {len(isa.splitlines())} lines of ISA")
    print(f"combined output VRAM row: {output_vram_addr // mlen}")
    if args.no_run:
        return {"build_dir": str(build_dir), "ran": False}

    metrics = run_and_assert(build_dir, "gpt_oss_moe_combine", mlen=mlen, blen=blen)
    results, params = compare_emulator_output(build_dir)
    comparison_summary = {
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
    }
    baseline_path = build_dir / "gpt_oss_moe_combine_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "name": "gpt_oss_moe_combine_tiny_emu_vs_golden_b",
                "include_bias": args.include_bias,
                "rows": rows,
                "hidden": hidden,
                "intermediate": intermediate,
                "mlen": mlen,
                "blen": blen,
                "comparison": comparison_summary,
                "comparison_params": params,
                "run_metrics": metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"baseline metrics written: {baseline_path}")
    return {"build_dir": str(build_dir), "ran": True, "metrics": metrics, "baseline": str(baseline_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / "gpt_oss_moe_combine",
    )
    parser.add_argument("--include-bias", action="store_true")
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    run_combine_smoke(args)


if __name__ == "__main__":
    main()
