"""Command line for the multi-chip model.

Examples (from the repository root, with the DeepStack submodule checked out)::

    python -m analytic_models.distributed describe-noc \\
        --noc analytic_models/distributed/examples/illustrative_gpu_cluster_32.json

    python -m analytic_models.distributed estimate \\
        --model llama-3.1-70b --model-lib PLENA_Compiler/doc/Model_Lib \\
        --config plena_settings.toml --isa-lib analytic_models/performance/customISA_lib.json \\
        --noc analytic_models/distributed/examples/illustrative_gpu_cluster_32.json \\
        --profile analytic_models/stacked_dram/examples/fictional_stacked_dram.json --tp 8 --pp 4
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import Any

from analytic_models.performance.perf_model import PerfModel, load_hardware_config_from_toml
from analytic_models.stacked_dram.__main__ import _gb, _memory, _model_path, _ms
from analytic_models.stacked_dram.estimate import OVERLAP_POLICIES, HbmStoragePrecision

from .model import COMM_OVERLAP_POLICIES, DistributedEstimate, PipelinePass, estimate_distributed
from .moe import ROUTING_MODES
from .noc import load_noc_profile
from .plan import MOE_TP_MODES, ParallelPlan
from .workload import ModelSpec


def _print_noc(description: dict[str, Any]) -> None:
    print(f"{description['name']}: {description['num_devices']} devices (ranks fill L1 first)")
    for level in description["layers"]:
        shape = "x".join(str(size) for size in level["shape"])
        print(
            f"  {level['level']}: {level['kind']:<10} {shape:>5}  {level['link_bandwidth_gbytes_per_s']:g} GB/s, "
            f"{level['hop_latency_ns']:g} ns/hop"
        )
    energy = description["energy_pj_per_bit"]
    if energy is not None:
        print(f"  energy: L1 {energy['l1']:g}, L2 {energy['l2']:g}, L3 {energy['l3']:g} pJ/bit")


def _describe_noc(args: argparse.Namespace) -> int:
    description = load_noc_profile(args.noc).describe()
    if args.json:
        print(json.dumps(description, indent=2))
    else:
        _print_noc(description)
    return 0


def _pass_row(label: str, result: PipelinePass) -> str:
    stage = result.stages[result.bottleneck_stage]
    comm = sum(stage.comm_seconds.values())
    return (
        f"{label:<14} {_ms(result.latency_seconds):>11} {_ms(result.period_seconds):>11} {stage.stage:>6} "
        f"{_ms(stage.compute_seconds):>11} {_ms(stage.memory_seconds):>11} {_ms(comm):>9} {_ms(result.transfer_seconds):>9}"
    )


def _print_estimate(result: DistributedEstimate) -> None:
    model, plan, memory = result.model, result.plan, result.memory
    capacity = "not modeled" if memory["capacity_bytes"] is None else f"{_gb(memory['capacity_bytes'])} GB"
    print(
        f"Model           {model['name']} [{model['family']}] (batch {result.batch_size}, "
        f"input {result.input_seq_len}, output {result.output_seq_len})"
    )
    parallel = " ".join(f"{dim}={plan[dim]}" for dim in ("tp", "ep", "dp", "pp", "cp"))
    print(f"Plan            {parallel} on {plan['world_size']} devices, NoC {result.noc['name']}")
    print(
        f"Micro-batch     {result.micro_batch} sequences, {result.sequences_per_device} per device "
        f"(attention and dense layers)"
    )
    if result.moe is not None:
        moe = result.moe
        print(
            f"MoE             tp={moe['tp']} ep={moe['ep']} dp={moe['dp']}, {moe['experts_per_rank']} experts per rank, "
            f"routing {result.routing}"
        )
        for phase in ("prefill", "decode"):
            load = moe[phase]
            print(
                f"  {phase:<13} {load['tokens_per_rank']} tokens and {load['token_expert_pairs_on_busiest_rank']} "
                f"token-expert pairs on the busiest rank (imbalance {load['imbalance']:.3f}, {load['source']})"
            )
    print(
        f"Memory          {memory['name']} [{memory['kind']}] per device: usable "
        f"{_gb(memory['usable_bandwidth_bytes_per_s'])} GB/s, capacity {capacity}"
    )
    print(f"Compute clock   {result.compute_frequency_hz / 1e9:.4f} GHz")
    lm_head = " (LM head included)" if result.include_lm_head else ""
    print(f"Overlap         {result.overlap_policy}, collectives overlap {result.comm_overlap}{lm_head}")
    print()
    print(
        f"{'pass':<14} {'latency ms':>11} {'period ms':>11} {'stage':>6} {'compute ms':>11} {'memory ms':>11} "
        f"{'comm ms':>9} {'p2p ms':>9}"
    )
    print(_pass_row("prefill", result.prefill))
    print(_pass_row("first decode", result.first_token_decode))
    print("(stage: the bottleneck pipeline stage, whose compute, memory and collective times are shown)")
    print()
    print(f"TTFT            {_ms(result.ttft_seconds)} ms")
    print(f"TPS             {result.tps:.2f} tokens/s ({result.tps_per_sequence:.2f} per sequence)")
    total = result.weight_bytes_per_device + result.kv_cache_bytes_per_device
    verdict = {None: "capacity not modeled", True: "fits", False: "exceeds capacity"}[result.fits_in_memory]
    print(
        f"Footprint       weights {_gb(result.weight_bytes_per_device)} GB + KV {_gb(result.kv_cache_bytes_per_device)} GB "
        f"= {_gb(total)} GB per device ({verdict})"
    )
    if result.noc_energy_pj_per_decode_token is not None:
        print(f"NoC energy      {result.noc_energy_pj_per_decode_token / 1e6:.3f} uJ per decode token")
    for warning in result.warnings:
        print(f"warning: {warning}")


def _estimate(args: argparse.Namespace) -> int:
    noc = load_noc_profile(args.noc)
    memory = _memory(args)
    model = ModelSpec.from_json(_model_path(args))
    plan = ParallelPlan(tp=args.tp, ep=args.ep, dp=args.dp, pp=args.pp, cp=args.cp, moe_tp_mode=args.moe_tp_mode)
    perf = PerfModel(load_hardware_config_from_toml(args.config), args.isa_lib)
    precision = HbmStoragePrecision.from_settings(args.config)
    result = estimate_distributed(
        model,
        plan,
        noc,
        perf,
        precision,
        memory,
        batch_size=args.batch_size,
        input_seq_len=args.input_seq,
        output_seq_len=args.output_seq,
        frequency_hz=args.frequency_hz,
        overlap_policy=args.overlap,
        comm_overlap=args.comm_overlap,
        routing=args.routing,
        include_lm_head=args.include_lm_head,
    )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        _print_estimate(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m analytic_models.distributed", description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    describe = commands.add_parser("describe-noc", help="levels, bandwidths and device count of a NoC profile")
    describe.add_argument("--noc", required=True, help="NoC profile JSON")
    describe.add_argument("--json", action="store_true", help="print JSON")
    describe.set_defaults(handler=_describe_noc)

    estimate = commands.add_parser("estimate", help="TTFT/TPS of a decoder on a multi-chip PLENA system")
    estimate.add_argument("--noc", required=True, help="NoC profile JSON; its device count must match the plan")
    memory = estimate.add_mutually_exclusive_group(required=True)
    memory.add_argument("--profile", help="per-device memory profile JSON (stacked_dram or fixed_bandwidth)")
    memory.add_argument("--fixed-bandwidth-gbs", type=float, help="ad-hoc fixed-bandwidth memory, decimal GB/s")
    estimate.add_argument("--fixed-capacity-gib", type=float, help="capacity of the ad-hoc fixed-bandwidth memory")
    estimate.add_argument("--total-layers", type=int, help="override the stacked_dram profile's stack height")
    estimate.add_argument("--connected-layers", type=int, help="override the connected layers (default: all)")
    model = estimate.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", help="model name in --model-lib")
    model.add_argument("--model-path", help="HuggingFace-style config JSON")
    estimate.add_argument("--model-lib", help="directory of model config JSONs")
    estimate.add_argument("--config", required=True, help="plena_settings.toml")
    estimate.add_argument("--isa-lib", required=True, help="customISA_lib.json")
    for dim, text in (
        ("tp", "tensor parallel"),
        ("ep", "expert parallel (MoE models)"),
        ("dp", "data parallel"),
        ("pp", "pipeline parallel"),
        ("cp", "context parallel (dense full-attention models)"),
    ):
        estimate.add_argument(f"--{dim}", type=int, default=1, help=f"{text} degree (default 1)")
    estimate.add_argument(
        "--moe-tp-mode",
        choices=MOE_TP_MODES,
        default="replace",
        help="replace: TP ranks become EP ranks in MoE layers (default); keep: experts stay tensor parallel",
    )
    estimate.add_argument("--routing", choices=ROUTING_MODES, default="balanced", help="MoE routing (default balanced)")
    estimate.add_argument("--batch-size", type=int, default=16)
    estimate.add_argument("--input-seq", type=int, default=2048)
    estimate.add_argument("--output-seq", type=int, default=1024)
    estimate.add_argument("--frequency-hz", type=float, default=1e9, help="nominal compute clock (default 1 GHz)")
    estimate.add_argument("--overlap", choices=OVERLAP_POLICIES, default="stage-roofline", help="compute/DRAM overlap")
    estimate.add_argument(
        "--comm-overlap", choices=COMM_OVERLAP_POLICIES, default="none", help="compute/collective overlap"
    )
    estimate.add_argument("--include-lm-head", action="store_true", help="charge the LM head every decode step")
    estimate.add_argument("--json", action="store_true", help="print JSON")
    estimate.set_defaults(handler=_estimate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "fixed_capacity_gib", None) is not None and not getattr(args, "fixed_bandwidth_gbs", None):
        raise SystemExit("--fixed-capacity-gib requires --fixed-bandwidth-gbs")
    try:
        return args.handler(args)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        # Invalid profiles, plans and paths are user input: report them without a traceback.
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    sys.exit(main())
