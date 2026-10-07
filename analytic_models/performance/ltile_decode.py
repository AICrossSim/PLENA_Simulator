"""Decode entry point; use ``unified`` for the shared compiled operator model.

The older R3 composer below is retained for historical reproduction only.

Surrounding operators remain explicitly analytical. Their shared tiled service
model is not promoted to a Rust-validated full-model execution. Capacity failures
are emitted, never repaired by silently adding memory or external offloading.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

from .ltile_cost import Machine, ProgramCost, assembly_cost
from .ltile_dma import DmaBackend, sha
from .ltile_program import build_program
from .ltile_calibration import write_csv
from .agentic_campaign import load_agentic_campaign
from .hybrid_lcompute_campaign import _model
from .nemotron3_workload import InferencePhase, Precision, PrecisionContract, WorkloadScenario


@dataclass(frozen=True)
class Platform:
    clock_hz: int = 1_000_000_000
    hbm_bytes: int = 16 * 1024**3
    matrix_bytes: int = 1024**2
    vector_bytes: int = 256 * 1024
    matrix_rows: int = 4
    matrix_cols: int = 1024
    vector_width: int = 2048
    # A common explicit transfer granularity for all surrounding operators.
    dma_tile_bytes: int = 4096
    exp_latency: int = 2


RECURRENT = {
    "nemotron3": {"mamba_state_update", "mamba_state_output"},
    "kimi_k3": {"kda_state_decay_prediction", "kda_delta_update_output"},
}
VARIANTS = ("old_isa", "row", "fsm")


def category(stage):
    n = stage.name
    if n == "lm_head":
        return "output_head"
    if "norm" in n:
        return "norm"
    if "conv" in n:
        return "conv"
    if "moe" in n or "ffn" in n:
        return "moe_ffn"
    if "projection" in n:
        return "projection"
    if stage.layer_type in ("attention", "mla"):
        return "attention_mla"
    if "gate" in n or "exp" in n:
        return "gate_prepare"
    return "residual_embedding"


def routes(campaign_path):
    campaign = load_agentic_campaign(campaign_path)
    hist = defaultdict(Counter)
    padding = defaultdict(list)
    for group in campaign.groups:
        for step in range(32):
            members = [dict(campaign.decode_routes[s][step]) for s in group.sample_ids]
            for layer in members[0]:
                assignments = Counter(e for member in members for e in member[layer])
                hist[group.batch_size, layer][len(assignments)] += 1
                padding[group.batch_size, layer].append(sum(math.ceil(n / 4) * 4 for n in assignments.values()))
    means = {key: sum(n * c for n, c in counts.items()) / sum(counts.values()) for key, counts in hist.items()}
    pads = {key: sum(values) / len(values) for key, values in padding.items()}
    rows = [
        dict(
            batch=b,
            layer=layer,
            unique_experts=n,
            observations=count,
            mean_unique=means[b, layer],
            mean_padded_expert_tokens=pads[b, layer],
        )
        for (b, layer), histogram in hist.items()
        for n, count in sorted(histogram.items())
    ]
    return (
        means,
        pads,
        rows,
        dict(
            source_sha256=campaign.routing_source_sha256,
            samples=len(campaign.samples),
            groups=len(campaign.groups),
            steps=32,
            scope="B1 observed routing merged by recorded batch membership; reused under context sweep",
        ),
    )


def workloads(compiler):
    models = {}
    for name in RECURRENT:
        default = Precision.NVFP4 if name == "nemotron3" else Precision.MXFP4
        models[name] = _model(
            name,
            compiler,
            activation_precision=Precision.BF16,
            weight_precision=None,
            state_precision=Precision.BF16,
            precision_contract=PrecisionContract.bf16_recurrence(default),
        )
    # User-selected NVFP4 study. Kimi's source checkpoint is MXFP4: only its
    # routed-expert storage is changed here; exclusions remain BF16. This is a
    # storage hypothesis, not a converted/checkpoint-tested NVFP4 Kimi model.
    kimi = models["kimi_k3"]
    kimi.weight_precision_policy = replace(
        kimi.weight_precision_policy,
        name="kimi_k3_nvfp4_routed_experts_storage_hypothesis",
        default_precision=Precision.NVFP4,
        global_stage_precisions=(
            *kimi.weight_precision_policy.global_stage_precisions,
            ("embedding_lookup", Precision.BF16),
        ),
        source="pinned MXFP4 checkpoint exclusions; NVFP4 routed-expert storage scenario",
    )
    kimi.weight_precision = Precision.NVFP4
    kimi.precision_contract = PrecisionContract.bf16_recurrence(Precision.NVFP4)
    return models


def transfer_parameters(backend, service, tile=4096):
    """Repeated finite transfers, not a presumed HBM peak throughput.

    These are memory-model probes for the *uncalibrated surrounding* model.
    Reads and writes are serialized; there is no cross-instruction overlap.
    Recurrent predictions do not use these stationary rates.
    """
    result = {}
    for direction in ("read", "write"):
        cost = ProgramCost()
        for i in range(256):
            cost.memory_trace.append(("r" if direction == "read" else "w", i * tile, tile))
            cost.transfers[direction, tile] += 1
        outcome = backend.price(cost, service)
        result[direction] = outcome["dma_cycles"] / 256
    result["rmw"] = service == "review"
    return result


def unique_prepared_bytes(trace):
    """Count read-only input allocations once, excluding read/write state.

    KDA's key is consumed twice. Charging its producer twice would manufacture
    a bandwidth advantage for the new path. Interval unions avoid that error.
    """

    def union(intervals):
        result = []
        for begin, end in sorted(intervals):
            if result and begin <= result[-1][1]:
                result[-1][1] = max(end, result[-1][1])
            else:
                result.append([begin, end])
        return result

    reads = union([(a, a + n) for op, a, n in trace if op == "r"])
    writes = union([(a, a + n) for op, a, n in trace if op == "w"])
    total = sum(end - begin for begin, end in reads)
    i = j = 0
    while i < len(reads) and j < len(writes):
        a, b = reads[i]
        c, d = writes[j]
        total -= max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def transfer_cost(read, write, rates, tile):
    # Every tail explicitly occupies a full tile; padding is physical traffic.
    rd, wr = math.ceil(read / tile), math.ceil(write / tile)
    return rd * rates["read"] + wr * rates["write"], rd * tile, wr * tile


def stage_cost(stage, workload, batch, platform, rates, expert_pad=None):
    """Bounded tiled operator estimate; shared verbatim by every variant."""
    t = stage.traffic
    mem, rd, wr = transfer_cost(t.logical_hbm_read_bytes, t.logical_hbm_write_bytes, rates, platform.dma_tile_bytes)
    vector_macs = stage.resource in ("conv", "state", "exp")
    matrix = 0
    if stage.macs and not vector_macs:
        # Dense decode has B rows; attention can pack independent heads. Routed
        # expert occupancy uses recorded assignment counts where available.
        factor = 1.0
        if "routed_experts" in stage.name and expert_pad is not None:
            topk = (
                workload.arch.moe.experts_per_token
                if hasattr(workload.arch, "moe")
                else workload.arch.experts_per_token
            )
            factor = expert_pad / (batch * topk)
        elif category(stage) != "attention_mla":
            factor = math.ceil(batch / platform.matrix_rows) * platform.matrix_rows / batch
        matrix = math.ceil(stage.macs * factor / (platform.matrix_rows * platform.matrix_cols))
    operations = stage.elementwise_ops + (2 * stage.macs if vector_macs else 0)
    passes = math.ceil(operations / platform.vector_width) + math.ceil(stage.scan_compositions / platform.vector_width)
    exp_passes = math.ceil(stage.exp_ops / platform.vector_width)
    arithmetic = passes + exp_passes * platform.exp_latency
    vector_ports = 3 * (passes + exp_passes)
    issue = passes + exp_passes + math.ceil(rd / platform.dma_tile_bytes) + math.ceil(wr / platform.dma_tile_bytes)
    # Logical inter-stage SRAM accesses are explicit; no state traffic from the
    # replaced recurrence stages survives here.
    ports = vector_ports + math.ceil((t.on_chip_read_bytes + t.on_chip_write_bytes) / 4096)
    precision = workload.weight_precision_policy.precision_for(stage.layer_id, stage.name)
    fmt = workload.precision_contract.weight_format(precision)
    decoded_weight = t.weight_read_bytes * 2 / (fmt.element_bits / 8 + (1 / fmt.block if fmt.block else 0))
    ports += math.ceil(decoded_weight / 4096)
    return dict(
        matrix=matrix,
        arithmetic=arithmetic,
        sram=ports,
        issue=issue,
        scalar=0,
        dependency=0,
        dma=mem,
        total=matrix + arithmetic + ports + issue + mem,
        hbm_read_bytes=rd + (wr if rates["rmw"] else 0),
        hbm_write_bytes=wr,
    )


def capacity(name, workload, batch, context):
    arch = workload.arch
    experts = arch.moe.num_experts if name == "nemotron3" else arch.num_experts
    report = workload.build(
        WorkloadScenario(InferencePhase.DECODE, batch_size=batch, context_length=context, moe_unique_experts=experts)
    )
    # Sum every expert, not just the active set in a decode step. Embedding
    # lookup traffic is replaced by the entire untied embedding table.
    weights = sum(s.traffic.weight_read_bytes for s in report.stages if s.name != "embedding_lookup")
    weights += 2 * arch.hidden_size * arch.vocab_size
    state = sum(s.traffic.state_write_bytes for s in report.stages)
    kv = sum(s.traffic.kv_read_bytes for s in report.stages)
    workspace = max(
        max(
            0, s.working_set_bytes - s.traffic.weight_read_bytes - s.traffic.state_write_bytes - s.traffic.kv_read_bytes
        )
        for s in report.stages
    )
    return dict(
        weight_lower_bound_bytes=int(weights),
        persistent_state_bytes=state,
        kv_bytes=kv,
        temporary_scenario_estimate_bytes=workspace,
        required_hbm_lower_bound_bytes=int(weights + state + kv),
        planned_hbm_estimate_bytes=int(weights + state + kv + workspace),
        scope="weights/state/KV logical storage lower bound; workspace is a separate scenario estimate; alignment/metadata excluded",
    )


def run(
    evidence,
    output,
    backend,
    campaign_path,
    *,
    gate_path,
    platform=Platform(),
    allow_uncalibrated_surrounding=False,
):
    gate = json.loads(gate_path.read_text())
    if not gate.get("gate_passed"):
        raise RuntimeError("recurrent cycle/speedup calibration gate has not passed")
    if gate.get("model_contract") != "ltile_r3_compositional_v1" or gate.get("memory_backend") != backend.identity:
        raise ValueError("calibration belongs to a different predictor or memory configuration")
    for name, digest in gate["predictor_sources"].items():
        if sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"predictor changed since calibration: {name}")
    if not allow_uncalibrated_surrounding:
        raise RuntimeError(
            "This composer has recurrence-only calibration; coefficient production and "
            "surrounding operators are unvalidated. Explicitly request a conditional "
            "estimate with allow_uncalibrated_surrounding=True. It is not a formal decode result."
        )
    output.mkdir(parents=True, exist_ok=True)
    compiler = evidence / "E/compiler"
    models = workloads(compiler)
    means, pads, route_rows, route_source = routes(campaign_path)
    write_csv(output / "routing_histogram.csv", route_rows)
    # Both source-faithful primary DMA and aligned write sensitivities apply
    # equally to old ISA, the matched row interface, and FSM.
    rates = {
        service: transfer_parameters(backend, service, platform.dma_tile_bytes) for service in ("review", "1", "64")
    }
    recurrent = {}
    rec_rows = []
    jobs = []
    for name in models:
        kind = "mamba" if name == "nemotron3" else "kda"
        for batch in (1, 2, 4, 8, 16):
            for variant in VARIANTS:
                asm, size = build_program(kind, batch, 2, variant, compiler_root=compiler)
                for service in rates:
                    jobs.append((name, batch, variant, service, asm, size))

    def price_recurrence(job):
        name, batch, variant, service, asm, size = job
        cost = assembly_cost(asm, Machine(), trace_memory=True)
        mem = backend.price(cost, service)
        return dict(
            model=name,
            batch=batch,
            variant=variant,
            dma_service=service,
            tokens_in_window=2,
            compiler_image_bytes=size,
            **{k: v / 2 for k, v in cost.components().items()},
            hbm_read_bytes=mem["read_bytes"] / 2,
            hbm_write_bytes=mem["write_bytes"] / 2,
            prepared_input_bytes=unique_prepared_bytes(cost.memory_trace) / 2,
            source="shape compilation + analytical scheduling + Ramulator DMA",
        )

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(price_recurrence, job) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            recurrent[row["model"], row["batch"], row["variant"], row["dma_service"]] = row
            rec_rows.append(row)
            if index % 10 == 0 or index == len(futures):
                print(f"recurrent programs {index}/{len(futures)}", flush=True)
    rec_rows.sort(key=lambda r: (r["model"], r["batch"], r["variant"], r["dma_service"]))
    write_csv(output / "recurrent_layers.csv", rec_rows)
    all_rows = []
    stages = []
    coverage = {}
    capacities = []
    for name, workload in models.items():
        layers = 23 if name == "nemotron3" else 69
        routing_options = ("measured_group_replay",) if name == "nemotron3" else ("max_unique", "min_unique")
        for batch in (1, 2, 4, 8, 16):
            for context in (4096, 32768, 131072):
                cap = capacity(name, workload, batch, context)
                cap.update(
                    model=name,
                    batch=batch,
                    context=context,
                    configured_hbm_bytes=platform.hbm_bytes,
                    hbm_fits_lower_bound=cap["required_hbm_lower_bound_bytes"] <= platform.hbm_bytes,
                )
                capacities.append(cap)
                for routing in routing_options:
                    scenario = WorkloadScenario(
                        InferencePhase.DECODE, batch_size=batch, context_length=context, moe_unique_experts=1
                    )
                    report = workload.build(scenario)
                    seen = set(s.layer_id for s in report.stages if s.layer_id >= 0)
                    if seen != set(range(52 if name == "nemotron3" else 93)):
                        raise AssertionError("incomplete layer timeline")
                    removed = [s for s in report.stages if s.name in RECURRENT[name]]
                    if len(removed) != 2 * layers:
                        raise AssertionError("unexpected recurrence boundary")
                    for service in rates:
                        component = Counter()
                        cats = Counter()
                        read = write = 0
                        for stage in report.stages:
                            if stage.name in RECURRENT[name]:
                                coverage[name, stage.name] = dict(
                                    model=name,
                                    operator=stage.name,
                                    unit="L256 update/product/residual lanes + Vector tree",
                                    storage="BF16 persistent HBM state; 512KiB current Matrix state group; existing Vector tree rows",
                                    validation="R3 compiled recurrent timing and numerical microbenchmarks; full-model quality separate",
                                    max_logical_working_set_bytes=stage.working_set_bytes,
                                    capacity_policy="request-private HBM; compiler checks each Matrix view allocation; no full-batch on-chip residency",
                                )
                                continue
                            expert_pad = None
                            if "routed_experts" in stage.name:
                                if name == "nemotron3":
                                    unique = means[batch, stage.layer_id]
                                    expert_pad = pads[batch, stage.layer_id]
                                else:
                                    k = workload.arch.experts_per_token
                                    unique = min(workload.arch.num_experts, batch * k) if routing == "max_unique" else k
                                    expert_pad = (
                                        batch * k * 4 if routing == "max_unique" else k * math.ceil(batch / 4) * 4
                                    )
                                stage = replace(
                                    stage,
                                    traffic=replace(
                                        stage.traffic,
                                        weight_read_bytes=math.ceil(stage.traffic.weight_read_bytes * unique),
                                    ),
                                )
                            c = stage_cost(stage, workload, batch, platform, rates[service], expert_pad)
                            component.update(
                                {
                                    k: c[k]
                                    for k in (
                                        "matrix",
                                        "arithmetic",
                                        "sram",
                                        "issue",
                                        "scalar",
                                        "dependency",
                                        "dma",
                                        "total",
                                    )
                                }
                            )
                            cats[category(stage)] += c["total"] - c["dma"]
                            read += c["hbm_read_bytes"]
                            write += c["hbm_write_bytes"]
                            if service == "review":
                                stages.append(
                                    dict(
                                        model=name,
                                        batch=batch,
                                        context=context,
                                        routing=routing,
                                        layer=stage.layer_id,
                                        stage=stage.name,
                                        category=category(stage),
                                        **c,
                                    )
                                )
                                coverage[name, stage.name] = dict(
                                    model=name,
                                    operator=stage.name,
                                    unit="Vector"
                                    if stage.resource in ("conv", "state", "exp")
                                    else "Matrix + Vector"
                                    if stage.macs
                                    else "Vector / DMA",
                                    storage="tiled Matrix SRAM / Vector SRAM; weights, KV and persistent state in HBM",
                                    validation="shape/traffic model; complete compiled schedule and operator cycles unvalidated",
                                    max_logical_working_set_bytes=max(
                                        stage.working_set_bytes,
                                        coverage.get((name, stage.name), {}).get("max_logical_working_set_bytes", 0),
                                    ),
                                    capacity_policy="large weights/KV tiled; liveness across all surrounding operators not proven",
                                )
                        # Preserve nonlinear/exp coefficient preparation which used to
                        # be embedded in KDA's recurrence work counters.
                        preparation = batch * (96 * 128 * 6 + 96 * 5) if name == "kimi_k3" else batch * 64 * 2
                        passes = math.ceil(preparation / platform.vector_width) * layers
                        prep_compute = passes * 6
                        component["arithmetic"] += 2 * passes
                        component["sram"] += 3 * passes
                        component["issue"] += passes
                        component["total"] += prep_compute
                        cats["gate_prepare"] += prep_compute
                        for variant in VARIANTS:
                            r = recurrent[name, batch, variant, service]
                            # The prepared inputs are produced here, not assumed to
                            # pre-exist for free. Materialize one matching HBM packet
                            # for every consumed non-state/non-output operand. Old ISA
                            # expansion is implemented as explicit Vector stores.
                            coefficient_bytes = r["prepared_input_bytes"]
                            coeff_dma, _, coeff_write = transfer_cost(
                                0, coefficient_bytes, rates[service], platform.dma_tile_bytes
                            )
                            coeff_passes = math.ceil(coefficient_bytes / 4096)
                            coeff_compute = coeff_passes * 4
                            total = component["total"] + layers * (r["total"] + coeff_dma + coeff_compute)
                            entry = dict(
                                model=name,
                                batch=batch,
                                context=context,
                                routing=routing,
                                variant=variant,
                                dma_service=service,
                                primary_dma=service == "review",
                                cycles_per_batch_step=total,
                                ms_per_step=total / platform.clock_hz * 1000,
                                aggregate_tokens_s=batch * platform.clock_hz / total,
                                request_tokens_s=platform.clock_hz / total,
                                recurrence_cycles=layers * r["total"],
                                recurrence_compute_cycles=layers * (r["total"] - r["dma"]),
                                recurrence_dma_cycles=layers * r["dma"],
                                preparation_cycles=layers * (coeff_dma + coeff_compute) + prep_compute,
                                coefficient_materialization_cycles=layers * (coeff_dma + coeff_compute),
                                surrounding_cycles=component["total"],
                                surrounding_matrix_cycles=component["matrix"],
                                movement_cycles=component["dma"] + layers * (r["dma"] + coeff_dma),
                                hbm_read_bytes=read
                                + layers * r["hbm_read_bytes"]
                                + (layers * coeff_write if service == "review" else 0),
                                hbm_write_bytes=write + layers * (r["hbm_write_bytes"] + coeff_write),
                                capacity_feasible_lower_bound=cap["hbm_fits_lower_bound"],
                                result_status="conditional analytical decode estimate; surrounding operators not cycle-validated",
                                **{f"{cat}_compute_cycles": value for cat, value in cats.items()},
                            )
                            # Exclusive category chart: recurrence compute, surrounding
                            # compute and coefficient preparation compute, movement.
                            entry["coefficient_materialization_compute_cycles"] = layers * coeff_compute
                            exclusive = (
                                sum(cats.values())
                                + entry["recurrence_compute_cycles"]
                                + entry["coefficient_materialization_compute_cycles"]
                                + entry["movement_cycles"]
                            )
                            if not math.isclose(total, exclusive, abs_tol=1e-5):
                                raise AssertionError("decode category ledger does not reconcile")
                            all_rows.append(entry)
    for r in all_rows:
        group = [
            x for x in all_rows if all(x[k] == r[k] for k in ("model", "batch", "context", "routing", "dma_service"))
        ]
        base = next(x for x in group if x["variant"] == "old_isa")
        matched = next(x for x in group if x["variant"] == "row")
        r["speedup_vs_old_isa"] = base["cycles_per_batch_step"] / r["cycles_per_batch_step"]
        r["speedup_vs_same_arithmetic_row"] = matched["cycles_per_batch_step"] / r["cycles_per_batch_step"]
        common_producer = next(x for x in group if x["variant"] == "fsm")["coefficient_materialization_cycles"]
        r["common_producer_cost_speedup_sensitivity"] = (
            base["cycles_per_batch_step"] - base["coefficient_materialization_cycles"] + common_producer
        ) / (r["cycles_per_batch_step"] - r["coefficient_materialization_cycles"] + common_producer)
        f = base["recurrence_cycles"] / base["cycles_per_batch_step"]
        r["baseline_recurrence_fraction"] = f
        r["recurrence_only_amdahl_limit"] = 1 / (1 - f)
        optimized = (base["recurrence_cycles"] + base["coefficient_materialization_cycles"]) / base[
            "cycles_per_batch_step"
        ]
        r["baseline_recurrence_and_materialization_fraction"] = optimized
        r["recurrence_and_materialization_amdahl_limit"] = 1 / (1 - optimized)
    for row in all_rows:
        row["result_status"] = "conditional_estimate"
        row["full_layer_validated"] = False
    write_csv(output / "decode_all.csv", all_rows)
    write_csv(output / "decode_primary.csv", [r for r in all_rows if r["primary_dma"]])
    write_csv(output / "surrounding_stages.csv", stages)
    write_csv(output / "operator_coverage.csv", list(coverage.values()))
    write_csv(output / "capacity.csv", capacities)
    config = dict(
        result_status="conditional_estimate",
        full_layer_validated=False,
        platform=asdict(platform),
        recurrent=asdict(Machine()),
        timeline="serial stage composition; overlap only inside priced recurrent primitive",
        recurrent_window="two tokens with private request addresses; setup amortized over this window",
        surrounding="stationary 4KiB DMA service + tile occupancy/operation counts; not calibrated full operators",
        matrix_profile_caveat="4096 modeled PEs from R3 metadata; MatrixMachine timing itself does not derive from these dimensions; operator validation pending",
        routing=route_source,
        transfer_cycles_per_4kib=rates,
        precision={n: w.precision_contract.to_dict() for n, w in models.items()},
        weight_policies={n: w.weight_precision_policy.to_dict() for n, w in models.items()},
        numerical_scope="old ISA and v2 differ in update/decay rounding; row vs FSM matches arithmetic; full-model task quality not established",
        calibration_sha256=sha(gate_path),
        memory_backend=backend.identity,
        forbidden_claims=[
            "measured full model",
            "TTFT",
            "GPU speedup",
            "energy efficiency",
            "no offload proven",
            "full RTL validated",
        ],
    )
    (output / "configuration.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                rows=len(all_rows),
                primary=sum(r["primary_dma"] for r in all_rows),
                capacity_fits=sum(r["hbm_fits_lower_bound"] for r in capacities),
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "unified":
        from .ltile_model import main

        main(sys.argv[2:])
        raise SystemExit(0)
    p = argparse.ArgumentParser()
    for arg in ("evidence", "output", "memory-binary", "memory-config", "memory-cache", "campaign", "gate"):
        p.add_argument("--" + arg, required=True, type=Path)
    p.add_argument(
        "--allow-uncalibrated-surrounding",
        action="store_true",
        help="export conditional estimates only; does not certify full-layer timing",
    )
    args = p.parse_args()
    run(
        args.evidence.resolve(),
        args.output.resolve(),
        DmaBackend(args.memory_binary, args.memory_config, args.memory_cache),
        args.campaign.resolve(),
        gate_path=args.gate,
        allow_uncalibrated_surrounding=args.allow_uncalibrated_surrounding,
    )
