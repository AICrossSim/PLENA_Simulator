"""Shared-expert MoE emulator test.

Drives ``moe_shared_expert_v0`` and compares the emulator's VRAM output against a
torch reference, **bit-exactly**. Two architectures:

``--arch deepseek`` (default)
    ``y = shared(x)``, SwiGLU, no gate. Also covers Llama-4, GLM-4.5 and Kimi K2,
    which combine the shared branch the same way. DeepSeek's ``n_shared_experts``
    is expressed as a single fused MLP of width
    ``moe_intermediate_size * n_shared_experts``; ``--n-shared`` exercises that.

``--arch qwen2``
    Adds Qwen2-MoE's ``sigmoid(x @ w_shared_gate)`` per-token scalar gate.

Exactness comes from ``_exact_mxfp8_tensor``: its values are all representable in
MXFP8, so the compiler's weight quantization is the identity and the only
remaining rounding is the BF16 steps the reference mirrors explicitly. A
tolerance-based comparison would pass just as happily with a transposed weight or
a dropped activation step.

Routed experts are out of scope here; ``--with-routed-accumulator`` covers only
the combine, i.e. that the shared output lands in the routed accumulator
unweighted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_moe_shared import fused_shared_intermediate
from transactional_emulator.testbench.aten.configurable import add_hw_args, resolve_rows, setup_hw
from transactional_emulator.testbench.aten.golden import quantize_to_mxfp
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.gpt_oss_testkit import _exact_mxfp8_tensor
from transactional_emulator.testbench.layout_utils import prestage_bf16_vram_matrix
from transactional_emulator.testbench.routed_moe._shared_moe_reference import (
    combine_shared_and_routed_golden,
    deepseek_shared_expert_golden,
    qwen2_shared_expert_golden,
    shared_gate_golden,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env

ASM_NAME = "moe_shared_expert"

GATE_WEIGHT_SCALE = 0.125


def _project_from_hbm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Projection golden for an HBM-resident activation and weight."""
    return torch.matmul(quantize_to_mxfp(x).float(), quantize_to_mxfp(w).float()).to(torch.bfloat16)


def _project_from_vram(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Projection golden where the activation is already BF16 in VRAM.

    Post-activation hidden state must not be MX-quantized a second time; only the
    weight follows the matrix-weight MX path.
    """
    return torch.matmul(x.float(), quantize_to_mxfp(w).float()).to(torch.bfloat16)


def build_and_run(args: argparse.Namespace) -> dict:
    mlen = args.mlen
    blen = args.blen
    rows, batch_size, seq_len = resolve_rows(args, default_seq=blen)
    hidden = args.hidden_size or mlen
    moe_intermediate = args.intermediate_size or mlen
    intermediate = fused_shared_intermediate(moe_intermediate, args.n_shared)
    gated = args.arch == "qwen2"

    if hidden % mlen != 0:
        raise ValueError(f"hidden ({hidden}) must be divisible by MLEN ({mlen})")
    if intermediate % mlen != 0:
        raise ValueError(
            f"fused shared intermediate ({intermediate} = {moe_intermediate} x {args.n_shared}) "
            f"must be divisible by MLEN ({mlen})"
        )

    # Resolve: the emulator is launched with its own working directory, so a
    # relative --build-dir (as the justfile recipes pass) would be looked up
    # against the wrong root and fail on a missing generated_machine_code.mem.
    build_dir = args.build_dir.expanduser().resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(
        f"Shared-expert MoE test: arch={args.arch}, n_shared={args.n_shared}, "
        f"rows={rows}, hidden={hidden}, shared_intermediate={intermediate} "
        f"(mlen={mlen}, blen={blen}, batch={batch_size}, seq={seq_len})"
    )
    print("=" * 80)

    x = _exact_mxfp8_tensor((rows, hidden), stride=1)
    w_gate = _exact_mxfp8_tensor((hidden, intermediate), stride=2, offset=1)
    w_up = _exact_mxfp8_tensor((hidden, intermediate), stride=3, offset=2)
    w_down = _exact_mxfp8_tensor((intermediate, hidden), stride=4, offset=3)
    # Scaled down so the gate logits land in roughly [-3, 3] and the sigmoid is
    # strictly interior. At full scale the logits reach +-16 and the gate saturates
    # to exactly 1.0 for most rows, which would let a completely unapplied gate
    # still pass. GATE_WEIGHT_SCALE is a power of two, so the scaling is exact in
    # BF16 and the comparison stays bit-exact.
    w_shared_gate = _exact_mxfp8_tensor((1, hidden), stride=3, offset=1) * GATE_WEIGHT_SCALE

    if gated:
        gate_scalars_golden = shared_gate_golden(x, w_shared_gate).float()
        saturated = int(((gate_scalars_golden == 1.0) | (gate_scalars_golden == 0.0)).sum())
        if saturated:
            raise AssertionError(
                f"{saturated}/{rows} shared-gate scalars saturated to exactly 0 or 1 "
                f"({gate_scalars_golden.tolist()}); those rows would compare equal whether or not "
                "the gate is applied at all, so the test would not detect a dropped gate. "
                "Lower GATE_WEIGHT_SCALE."
            )
        print(f"shared-gate scalars: {[round(v, 4) for v in gate_scalars_golden.tolist()]}")

    if gated:
        shared_golden = qwen2_shared_expert_golden(
            x,
            w_gate,
            w_up,
            w_down,
            w_shared_gate,
            project=_project_from_hbm,
            project_from_vram=_project_from_vram,
        )
    else:
        shared_golden = deepseek_shared_expert_golden(
            x,
            w_gate,
            w_up,
            w_down,
            project=_project_from_hbm,
            project_from_vram=_project_from_vram,
        )

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)
    input_tensors: dict[str, torch.Tensor] = {}

    x_input = prog.input("X", shape=(rows, hidden))
    w_gate_input = prog.input("W_shared_gate_proj", shape=(hidden, intermediate))
    w_up_input = prog.input("W_shared_up", shape=(hidden, intermediate))
    w_down_input = prog.input("W_shared_down", shape=(intermediate, hidden))
    input_tensors.update(
        {
            "X": x,
            "W_shared_gate_proj": w_gate,
            "W_shared_up": w_up,
            "W_shared_down": w_down,
        }
    )
    x_vram = prog.load_batch(x_input, name="X")

    physical_rows = max(mlen, math.ceil(rows / blen) * blen)
    # The gate weight is read by V_MUL_VV on the BF16 vector path, not the matrix
    # path, so it has to be resident in VRAM rather than streamed from HBM as an
    # MX-quantized matrix tile.
    vram_preload = torch.zeros((physical_rows + mlen) * hidden * 4, dtype=torch.bfloat16)
    gate_weight_row = None
    if gated:
        gate_weight_row = prestage_bf16_vram_matrix(
            prog=prog,
            name="W_shared_gate_row",
            tensor=w_shared_gate,
            vram_addr=prog.get_vram_addr(x_vram.name) + physical_rows * hidden * 2,
            physical_shape=(1, hidden),
            vram_preload=vram_preload,
        )

    zero = prog.fp_var("shared_zero", size=1)
    one = prog.fp_var("shared_one", size=max(rows, blen))
    neg_one = prog.fp_var("shared_neg_one", size=max(rows, blen))
    limit_pos = prog.fp_var("shared_unused_limit_pos", size=max(rows, blen))
    limit_neg = prog.fp_var("shared_unused_limit_neg", size=max(rows, blen))
    zero_row = prog.fp_var("shared_zero_row", size=mlen)
    gate_scalars = prog.fp_var("shared_gate_scalars", size=max(rows, blen))
    constants = (zero, limit_pos, limit_neg, one, neg_one)

    shared_out = prog.moe_shared_expert_v0(
        x_vram,
        (w_gate_input, w_up_input, w_down_input),
        rows=rows,
        intermediate=intermediate,
        constants=constants,
        gate_weight_row=gate_weight_row,
        gate_fp_scratch=gate_scalars if gated else None,
        zero_row=zero_row,
        activation_policy="standard_swiglu",
        policy_name="qwen2_moe" if gated else "deepseek_moe",
        name="shared_expert",
    )

    golden = shared_golden
    output_var = shared_out
    if args.with_routed_accumulator:
        # Stand-in for the routed branch: a true-zeroed accumulator. Adding the
        # shared output into it must reproduce the shared output exactly, which
        # pins that the combine is an unweighted add and not, say, scaled by a
        # leftover route weight.
        accumulator = prog.alloc(
            "shared_test_accumulator",
            rows=rows,
            cols=hidden,
            strict=False,
            physical_shape=(physical_rows, hidden),
        )
        prog.moe_true_zero_vram_rows_v0(
            accumulator,
            rows=list(range(physical_rows)),
            hidden=hidden,
            zero_row=zero_row,
            policy_name="deepseek_moe",
            stage="accumulator_init",
            name="shared_test_acc_zero",
        )
        output_var = prog.moe_combine_shared_and_routed_v0(
            accumulator,
            shared_out,
            rows=rows,
            policy_name="qwen2_moe" if gated else "deepseek_moe",
            name="shared_test_combine",
        )
        golden = combine_shared_and_routed_golden(torch.zeros_like(shared_golden), shared_golden)

    isa = prog.compile()

    fp_preload_len = max(
        var.address + var.size for var in (zero, one, neg_one, limit_pos, limit_neg, zero_row, gate_scalars)
    )
    fp_preload = [0.0] * fp_preload_len
    for idx in range(one.size):
        fp_preload[one.address + idx] = 1.0
    for idx in range(neg_one.size):
        fp_preload[neg_one.address + idx] = -1.0

    create_sim_env(
        input_tensors,
        isa,
        {"original_output": golden},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
        vram_preload=vram_preload if gated else None,
    )

    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=ASM_NAME,
        data=None,
        specified_data_order=sorted(input_tensors, key=lambda name: hbm_addrs[name]),
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
    )

    output_vram_addr = prog._compiler.get_vram_addr(output_var.name)
    comparison_params = {
        "start_row_idx": output_vram_addr // mlen,
        "num_rows": (rows * hidden) // mlen,
        "num_batches": rows,
        "elements_per_batch": hidden,
        "row_dim": mlen,
        "atol": 0.0,
        "rtol": 0.0,
    }
    (build_dir / "comparison_params.json").write_text(json.dumps(comparison_params, indent=2))
    (build_dir / "generated_asm_code.asm").write_text(isa)

    print(f"Generated {len(isa.splitlines())} lines of ISA")
    print(f"shared expert output VRAM row: {output_vram_addr // mlen}")
    if args.no_run:
        return {"build_dir": str(build_dir), "ran": False}

    metrics = run_and_assert(build_dir, ASM_NAME, mlen=mlen, blen=blen, stage_profile=True)
    _assert_stage_attribution_healthy(build_dir, gated=gated, combined=args.with_routed_accumulator)
    return {"build_dir": str(build_dir), "ran": True, "metrics": metrics}


# Stages a shared-expert-only program cannot contain. There is no router, no
# gather or scatter of routed tokens, no per-expert route weight, no dynamic
# expert-id weight address, and no expert bias.
IMPOSSIBLE_STAGES = (
    # No decoder residual buffer here: this program is the shared expert alone.
    "residual_setup",
    "router_topk",
    "gather",
    "expert_weight_address",
    "expert_bias",
    "expert_route_weight",
    "expert_projection",
    "expert_activation",
)

REQUIRED_STAGES = ("shared_expert_projection", "shared_expert_activation")


def _assert_stage_attribution_healthy(build_dir: Path, *, gated: bool, combined: bool) -> None:
    """Guards on the explicit ``@stage=`` attribution path.

    The routed tests guard the *legacy* substring classifier. These guards cover
    the mode that replaces it, whose failure modes are different: markers cannot
    drift by rewording, but they can be missing, wrong or misplaced.
    """
    profile = json.loads((build_dir / "stage_profile.json").read_text())
    coverage = profile["classification"]

    # Guard 1. Markers were actually used.
    assert coverage["stage_attribution"] == "explicit_stage_markers", (
        f"stage attribution fell back to {coverage['stage_attribution']!r}; the compiler "
        "emitted no @stage= markers, so shared-expert work is being classified by the legacy "
        "substring rules that predate shared experts and cannot distinguish the branches."
    )

    # Guard 2. Every marker resolved. A name the compiler validated against its own
    # MOE_STAGES but the emulator's StageKind lacks means the two drifted; those
    # regions silently inherit the previous stage.
    unresolved = coverage["unresolved_stage_markers"]
    assert not unresolved, (
        f"emulator does not recognise @stage= names the compiler emitted: {unresolved}. "
        "Reconcile StageKind in stage_profile.rs with MOE_STAGES in program_routed_moe.py."
    )

    # Guard 3. The distribution. Shared work must land in shared buckets, and
    # nothing may land in the routed ones -- separating the two is the entire point
    # of the shared stages existing.
    counts = coverage["stage_instruction_counts"]
    impossible = dict(IMPOSSIBLE_STAGES and {s: counts[s] for s in IMPOSSIBLE_STAGES if counts[s]})
    assert not impossible, (
        f"opcodes were attributed to stages this program cannot contain: {impossible}. "
        "A shared-expert-only program has no routing; a non-zero routed stage means a marker "
        "is missing or placed at the wrong boundary, letting the region bill to its neighbour."
    )
    empty = [stage for stage in REQUIRED_STAGES if not counts[stage]]
    assert not empty, f"shared-expert stages that must carry work are empty: {empty} (counts: {counts})"

    gate_count = counts["shared_expert_gate"]
    if gated:
        assert gate_count, "Qwen2-MoE run recorded no shared_expert_gate instructions"
    else:
        assert not gate_count, (
            f"ungated run recorded {gate_count} shared_expert_gate instructions; DeepSeek-style "
            "shared experts have no sigmoid gate, so this stage must stay empty"
        )

    combine_count = counts["scatter_combine"]
    if combined:
        assert combine_count, "combine run recorded no scatter_combine instructions"
    else:
        assert not combine_count, f"shared-expert-only run recorded {combine_count} scatter_combine instructions"

    # Guard 4. The do_ops -> StageProfiler::record unit contract, same as the
    # routed tests: if a caller ever passes cycles where picoseconds are expected,
    # profiled time stops matching simulated time.
    assert profile["cycle_accounting_status"] == "profiled_time_matches_total", (
        f"stage profile does not account for the whole run: {profile['cycle_accounting_status']} "
        f"(profiled {profile['total_profiled_picos']} ps of {profile['total_simulation_picos']} ps)"
    )

    print(
        f"Stage attribution OK ({coverage['stage_attribution']}): "
        f"{{{', '.join(f'{k}={v}' for k, v in counts.items() if v)}}}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--intermediate-size", type=int, default=None, help="moe_intermediate_size")
    parser.add_argument(
        "--arch",
        choices=("deepseek", "qwen2"),
        default="deepseek",
        help="deepseek: ungated shared expert. qwen2: adds the sigmoid shared-expert gate.",
    )
    parser.add_argument(
        "--n-shared",
        type=int,
        default=1,
        help="DeepSeek n_shared_experts; fused into one MLP of n_shared x intermediate width.",
    )
    parser.add_argument("--with-routed-accumulator", action="store_true")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(__file__).parent / "build" / ASM_NAME,
    )
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()
    build_and_run(args)


if __name__ == "__main__":
    main()
