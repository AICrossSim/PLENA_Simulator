"""Whole decode graph composed from explicit operator mappings.

The numerical sublayer and this composer use Services/price_program with the
same profile. Graph composition is serial, HBM-backed between operators.
Selection/packing budgets remain named uncertainty terms; capacity-invalid
points are never admitted to the single-device table. This is not TTFT/PPA.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from .ltile_dma import sha

from .agentic_campaign import load_agentic_campaign
from .ltile_decode import workloads
from .ltile_platform import ExecutionProfile, DmaResources, CodecResources, memory_geometry
from .ltile_services import Services
from .ltile_dma import DmaBackend
from .ltile_calibration import write_csv
from .nemotron3_workload import WorkloadScenario, InferencePhase
from .weight_codec import packed_matrix_bytes

COMPONENTS = ("issue", "scalar", "sram", "arithmetic", "dependency", "dma")


def route_occupancy(path):
    """Mean number of experts with N tokens; never round the mean occupancy."""
    c = load_agentic_campaign(path)
    hist = defaultdict(Counter)
    steps = Counter()
    for g in c.groups:
        for t in range(32):
            members = [dict(c.decode_routes[s][t]) for s in g.sample_ids]
            for layer in members[0]:
                assignments = Counter(e for m in members for e in m[layer])
                hist[g.batch_size, layer].update(assignments.values())
                steps[g.batch_size, layer] += 1
    result = {k: {n: v / steps[k] for n, v in h.items()} for k, h in hist.items()}
    for (b, _), h in result.items():
        if not math.isclose(sum(n * v for n, v in h.items()), b * 6):
            raise AssertionError("route assignment conservation failed")
    return result, dict(
        sha256=c.routing_source_sha256,
        samples=len(c.samples),
        groups=len(c.groups),
        scope="real B1 routes merged into recorded batches; context sweep reuses routes",
    )


def linear_shapes(name, a):
    """Named tensor shapes. Multiplicity is tensor count, not token count."""
    h = a.hidden_size
    if name == "lm_head":
        return [(h, a.vocab_size, 1)]
    if name == "mamba_in_projection":
        return [(h, 10304, 1)]
    if name == "mamba_out_projection":
        return [(4096, h, 1)]
    if name == "attention_qkv_projection":
        return [(h, (a.num_heads + 2 * a.num_kv_heads) * a.head_dim, 1)]
    if name == "attention_out_projection":
        return [(a.num_heads * a.head_dim, h, 1)]
    if name == "moe_router_topk":
        return [(h, a.moe.num_experts, 1)]
    if name == "moe_routed_experts":
        return [(h, a.moe.intermediate_size, 1), (a.moe.intermediate_size, h, 1)]
    if name == "moe_shared_expert":
        return [
            (h, a.moe.shared_intermediate_size, a.moe.shared_experts),
            (a.moe.shared_intermediate_size, h, a.moe.shared_experts),
        ]
    if name == "kda_qkv_projection":
        return [(h, 12288, 3)]
    if name == "kda_decay_beta_projection":
        return [(h, 128, 1), (128, 12288, 1), (h, 96, 1)]
    if name == "kda_output_gate_projection":
        return [(h, 12288, 1)]
    if name == "kda_out_projection":
        return [(12288, h, 1)]
    if name == "mla_q_low_rank_projection":
        return [(h, a.q_lora_rank, 1), (a.q_lora_rank, a.kda.num_heads * (a.qk_nope_head_dim + a.qk_rope_head_dim), 1)]
    if name == "mla_kv_latent_projection":
        return [(h, a.kv_lora_rank + a.qk_rope_head_dim, 1)]
    if name == "mla_output_gate":
        return [(h, a.mla_projection_size, 1)]
    if name == "mla_out_projection":
        return [(a.mla_projection_size, h, 1)]
    if name == "latent_moe_router_top16":
        return [(h, a.num_experts, 1)]
    if name == "latent_moe_down_projection_norm":
        return [(h, a.routed_expert_hidden_size, 1)]
    if name == "latent_moe_up_projection":
        return [(a.routed_expert_hidden_size, h, 1)]
    if name == "latent_moe_routed_experts":
        return [
            (a.routed_expert_hidden_size, a.moe_intermediate_size, 2),
            (a.moe_intermediate_size, a.routed_expert_hidden_size, 1),
        ]
    if name == "latent_moe_shared_experts":
        mid = a.shared_experts * a.moe_intermediate_size
        return [(h, mid, 2), (mid, h, 1)]
    if name == "dense_situ_ffn":
        return [(h, a.dense_intermediate_size, 2), (a.dense_intermediate_size, h, 1)]
    return []


def tensor_inventory(name, w):
    """Persistent tensor census, independent of per-step DMA or expert activity."""
    a = w.arch
    rows = []

    def add(label, shape, fmt="BF16", copies=1):
        elements = math.prod(shape)
        size = (
            packed_matrix_bytes(shape[-2], shape[-1])["total"] if fmt == "NVFP4" else math.ceil(elements * 2 / 64) * 64
        )
        rows.append(dict(tensor=label, shape=list(shape), format=fmt, copies=copies, bytes=size * copies))

    add("embedding", (a.vocab_size, a.hidden_size))
    report = w.build(WorkloadScenario(InferencePhase.DECODE, batch_size=1, context_length=4096, moe_unique_experts=1))
    for st in report.stages:
        fmt = w.weight_precision_policy.precision_for(st.layer_id, st.name).value.upper()
        specs = linear_shapes(st.name, a)
        for index, (k, n, copies) in enumerate(specs):
            experts = (
                (a.moe.num_experts if name == "nemotron3" else a.num_experts) if "routed_experts" in st.name else 1
            )
            # NVFP4 quantizes K along each output row, including block tails.
            add(f"{st.layer_id}/{st.name}/{index}", (n, k), fmt, copies * experts)
        if specs:
            continue
        if st.name in ("block_rms_norm", "input_rms_norm", "post_attention_rms_norm", "final_rms_norm"):
            add(f"{st.layer_id}/{st.name}", (1, a.hidden_size))
        elif st.name == "mamba_conv1d":
            add(f"{st.layer_id}/conv", (4, 6144))
            add(f"{st.layer_id}/conv_bias", (1, 6144))
            add(f"{st.layer_id}/dt_bias_A_D", (3, 64))
        elif st.name == "mamba_gate_group_rms_norm":
            add(f"{st.layer_id}/mixer_norm", (1, 4096))
        elif st.name == "moe_combine":
            add(f"{st.layer_id}/router_correction_bias", (1, a.moe.num_experts))
        elif st.name == "kda_short_conv":
            add(f"{st.layer_id}/conv", (4, 12288), copies=3)
            add(f"{st.layer_id}/dt_bias", (1, 12288))
            add(f"{st.layer_id}/A_log", (1, 96))
        elif st.name == "kda_output_gate_rmsnorm":
            add(f"{st.layer_id}/o_norm", (1, 128))
        elif st.name == "mla_compressed_kv_attention":
            add(f"{st.layer_id}/key_transform", (a.kv_lora_rank, a.qk_nope_head_dim), copies=a.kda.num_heads)
            add(f"{st.layer_id}/value_transform", (a.kv_lora_rank, a.v_head_dim), copies=a.kda.num_heads)
            add(f"{st.layer_id}/q_norm", (1, a.q_lora_rank))
            add(f"{st.layer_id}/kv_norm", (1, a.kv_lora_rank))
        elif st.name.startswith("attn_res_") and st.name != "attn_res_capture_prefix":
            add(f"{st.layer_id}/{st.name}/query_norm", (2, a.hidden_size))
    return rows


def capacity(name, w, batch, context, weights, geometry):
    a = w.arch
    if name == "nemotron3":
        state = batch * 23 * (64 * 128 * 64 + 4 * 6144) * 2
        kv = batch * 6 * context * 2 * a.num_kv_heads * a.head_dim * 2
    else:
        state = batch * 69 * (96 * 128 * 128 + 3 * 4 * 12288) * 2
        # The selected GEMV mapping owns both packed orientations of the
        # latent cache; this is a storage cost, not free transpose support.
        kv = batch * 24 * context * (2 * a.kv_lora_rank + a.qk_rope_head_dim) * 2
    # Deliberate HBM reserve covers owned padding, group masks, model constants,
    # attention scores/probabilities and prefixes. No unbounded on-chip memory.
    scratch = max(1024**3, batch * context * (32 if name == "nemotron3" else 96) * 2 * 4)
    positional = 0 if name == "nemotron3" else context * 64 * 2 * 2 + 4096
    required = weights + state + kv + scratch + positional
    return dict(
        weight_tensor_bytes=weights,
        state_bytes=state,
        kv_bytes=kv,
        workspace_reserve_bytes=scratch,
        positional_table_bytes=positional,
        required_hbm_bytes=required,
        available_hbm_bytes=geometry["capacity_bytes"],
        capacity_feasible=required <= geometry["capacity_bytes"],
    )


def resource_budget(profile, *, selection=0, kv_words=0):
    """Named conservative budgets for uncalibrated orchestration.

    Selection scans N scores for each winner, retaining K entries in existing
    SRAM. KV append counts a masked read/modify/write per 32-value bank word.
    These terms are swept separately and never called Rust calibrated.
    """
    c = dict.fromkeys(COMPONENTS, 0)
    c.update(
        issue=selection * 4 + kv_words * 4,
        scalar=selection * 3 + kv_words * 2,
        arithmetic=selection * profile.machine.vector_max_cycles,
        sram=selection * 2 + kv_words * 4,
    )
    c["total"] = sum(c.values())
    return dict(
        components=c,
        hbm_read_bytes=0,
        hbm_write_bytes=0,
        metadata=dict(validation="finite scheduling budget; orchestration not machine-code calibrated"),
    )


def compose_peripheral(stage, w, b, context, services, occupancy):
    """Return exclusive serial components; unsupported graph nodes fail closed."""
    a = w.arch
    n = stage.name
    h = a.hidden_size
    terms = []

    def add(category, result, factor=1):
        terms.append((category, result, factor))

    def vec(kind, values, groups=b, factor=1, category="other"):
        add(category, services.vector(kind, values, groups=groups), factor)

    def mat(k, columns, batch=b, factor=1, category="projection", weight=None):
        fmt = weight or w.weight_precision_policy.precision_for(stage.layer_id, n).value.upper()
        add(category, services.projection(batch, k, columns, fmt), factor)

    if "routed_experts" in n:
        for count, experts in occupancy.items():
            for k, columns, copies in linear_shapes(n, a):
                mat(k, columns, count, experts * copies, "moe")
            mid = a.moe.intermediate_size if hasattr(a, "moe") else a.moe_intermediate_size
            vec("relu2" if hasattr(a, "moe") else "silu", mid, count, experts, "moe")
            if not hasattr(a, "moe"):
                vec("mul", mid, count, experts, "moe")
            vec("copy", h if hasattr(a, "moe") else a.routed_expert_hidden_size, count, experts, "moe_dispatch")
        return terms
    specs = linear_shapes(n, a)
    if specs:
        cat = "output_head" if n == "lm_head" else "moe" if "moe" in n or "ffn" in n else "projection"
        for k, columns, copies in specs:
            mat(k, columns, factor=copies, category=cat)
        if "router" in n:
            experts = a.moe.num_experts if hasattr(a, "moe") else a.num_experts
            topk = a.moe.experts_per_token if hasattr(a, "moe") else a.experts_per_token
            add("routing_budget", resource_budget(services.profile, selection=b * experts * topk))
            # Native Nemotron uses sigmoid scores, correction bias for
            # selection, then renormalizes the selected positive scores.
            # Kimi remains a declared graph hypothesis, not native validation.
            vec("sigmoid_mul", experts, category="moe")  # second operand is ones
            vec("add", experts, category="moe")  # selection correction bias
            vec("positive_normalize", topk, category="moe")
            vec("mul", topk, category="moe")  # routed scale
        elif n == "moe_shared_expert":
            vec("relu2", a.moe.shared_intermediate_size, b * a.moe.shared_experts, category="moe")
        elif n in ("latent_moe_shared_experts", "dense_situ_ffn"):
            mid = a.shared_experts * a.moe_intermediate_size if "shared" in n else a.dense_intermediate_size
            vec("silu", mid, category="moe")
            vec("mul", mid, category="moe")
        elif n == "latent_moe_down_projection_norm":
            vec("norm", a.routed_expert_hidden_size, category="norm")
        elif n == "mla_q_low_rank_projection":
            vec("norm", a.q_lora_rank, category="norm")
        elif n == "mla_kv_latent_projection":
            vec("norm", a.kv_lora_rank, category="norm")
        elif n == "mla_output_gate":
            vec("sigmoid_mul", a.mla_projection_size, category="attention")
        if n in ("attention_qkv_projection", "mla_kv_latent_projection"):
            count = 2 * a.num_kv_heads * a.head_dim if hasattr(a, "moe") else a.kv_lora_rank + a.qk_rope_head_dim
            vec("copy", count, category="kv_append")
            add(
                "kv_pack_budget",
                services.kv_append(
                    b,
                    a.num_kv_heads if hasattr(a, "moe") else 1,
                    a.head_dim if hasattr(a, "moe") else a.kv_lora_rank + a.qk_rope_head_dim,
                    a.head_dim if hasattr(a, "moe") else a.kv_lora_rank,
                    context,
                ),
            )
        return terms
    if n == "attention_qk_softmax_pv":
        queries = a.num_heads // a.num_kv_heads
        vec("mul", a.num_heads * a.head_dim, category="attention_prepare")
        # One KV group shared by its queries; requests have private caches.
        mat(a.head_dim, context, queries, b * a.num_kv_heads, "attention", weight="BF16")
        vec("softmax", context, b * a.num_heads, category="attention")
        mat(context, a.head_dim, queries, b * a.num_kv_heads, "attention", weight="BF16")
    elif n == "mla_compressed_kv_attention":
        heads = a.kda.num_heads
        vec("rope64", a.qk_rope_head_dim, b * (heads + 1), category="attention_prepare")
        vec("mul", heads * (a.qk_nope_head_dim + a.qk_rope_head_dim), category="attention_prepare")
        mat(a.qk_nope_head_dim, a.kv_lora_rank, b, heads, "attention", weight="BF16")
        mat(a.kv_lora_rank + a.qk_rope_head_dim, context, heads, b, "attention", weight="BF16")
        vec("softmax", context, b * heads, category="attention")
        mat(context, a.kv_lora_rank, heads, b, "attention", weight="BF16")
        mat(a.kv_lora_rank, a.v_head_dim, b, heads, "attention", weight="BF16")
    elif n in ("block_rms_norm", "input_rms_norm", "post_attention_rms_norm", "final_rms_norm"):
        vec("norm", h, category="norm")
    elif n in ("block_residual", "prefix_sum_after_mixer", "prefix_sum_after_ffn"):
        vec("add", h, category="residual")
    elif n in ("moe_combine", "latent_moe_combine"):
        count = (
            (a.moe.experts_per_token + a.moe.shared_experts)
            if hasattr(a, "moe")
            else a.experts_per_token + a.shared_experts
        )
        vec("mul", h, b * count, category="moe")
        vec("add", h, b * (count - 1), category="moe")
    elif n == "embedding_lookup":
        vec("copy", h, category="embedding")
    elif n == "attn_res_capture_prefix":
        vec("copy", h, category="residual")
    elif n.startswith("attn_res_"):
        count = stage.macs // (2 * b * h)
        if count < 1:
            raise ValueError("unknown AttnRes shape")
        vec("norm", h, b * count, category="attention_residual")
        vec("dot", h, b * count, category="attention_residual")
        vec("softmax", count, category="attention_residual")
        vec("mul", h, b * count, category="attention_residual")
        if count > 1:
            vec("add", h, b * (count - 1), category="attention_residual")
    else:
        raise ValueError(f"unmapped full-model operator {n}")
    return terms


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile = ExecutionProfile(
        hbm_controllers=args.controllers,
        dma=DmaResources(
            read_credits=args.dma_window,
            write_credits=args.dma_window,
            read_response_bytes=max(2048, args.dma_window * 64),
            write_staging_bytes=max(2048, args.dma_window * 64),
        ),
        codec=CodecResources(lanes=args.codec_lanes),
    )
    if args.sfu_scale != 1:
        profile = replace(
            profile,
            machine=replace(
                profile.machine,
                vector_exp_cycles=profile.machine.vector_exp_cycles * args.sfu_scale,
                vector_softplus_cycles=profile.machine.vector_softplus_cycles * args.sfu_scale,
                vector_reciprocal_cycles=profile.machine.vector_reciprocal_cycles * args.sfu_scale,
            ),
        )
    backend = DmaBackend(
        args.memory_root / "ltile_memory", args.memory_root / "ramulator.json", args.memory_root / "cache"
    )
    services = Services(args.compiler, profile, backend, args.service_cache or out / "services")
    geometry = memory_geometry(json.loads(backend.config.read_text()))
    if geometry["controllers"] != profile.hbm_controllers:
        raise ValueError("HBM profile mismatch")
    models = workloads(args.compiler)
    routes, route_meta = route_occupancy(args.campaign)
    (out / "configuration.json").write_text(
        json.dumps(
            dict(
                profile=asdict(profile),
                profile_sha256=profile.identity,
                memory=geometry,
                source_identity=services.identity,
                graph_sources={
                    str(Path(__file__).with_name(f)): sha(Path(__file__).with_name(f))
                    for f in (
                        "ltile_model.py",
                        "ltile_decode.py",
                        "nemotron3_workload.py",
                        "kimi_k3_workload.py",
                        "agentic_campaign.py",
                    )
                },
                routing=route_meta,
                composition="serial HBM-backed operators, no cross-operator overlap; within-layer memory history retained",
                uncertainty="routing/KV packing are conservative finite budgets; sample address/phase sensitivity separately",
                scope="decode analytical predictions; no task quality, complete RTL, TTFT, power or GPU comparison",
            ),
            indent=2,
        )
        + "\n"
    )
    tables = []
    details = []
    capacities = []
    inventories = []
    for name in args.models:
        w = models[name]
        a = w.arch
        inventory = tensor_inventory(name, w)
        inventories.extend(dict(model=name, **t) for t in inventory)
        weight_bytes = sum(t["bytes"] for t in inventory)
        for b in args.batches:
            for context in args.contexts:
                cap = capacity(name, w, b, context, weight_bytes, geometry)
                capacities.append(dict(model=name, batch=b, context=context, **cap))
                # Do not execute a fictitious single-device graph against
                # more resident storage than exists. Conditional timing is
                # explicit opt-in and remains excluded from the main table.
                if not cap["capacity_feasible"] and not args.include_infeasible:
                    print(
                        json.dumps(dict(model=name, batch=b, context=context, status="capacity_infeasible", **cap)),
                        flush=True,
                    )
                    continue
                scenarios = ["measured_group_replay"] if name == "nemotron3" else ["min_unique", "max_unique"]
                report = w.build(
                    WorkloadScenario(InferencePhase.DECODE, batch_size=b, context_length=context, moe_unique_experts=1)
                )
                for routing in scenarios:
                    for control in ("old_isa", "row", "fsm"):
                        ledger = Counter()
                        categories = Counter()
                        read = write = 0
                        replaced = set()
                        for stage in report.stages:
                            if stage.layer_type in ("mamba", "kda") and (
                                stage.name.startswith(("mamba_", "kda_"))
                                or stage.name in ("block_rms_norm", "input_rms_norm")
                            ):
                                if stage.layer_id in replaced:
                                    continue
                                replaced.add(stage.layer_id)
                                kind = "mamba" if name == "nemotron3" else "kda"
                                weight = (
                                    w.weight_precision_policy.precision_for(
                                        stage.layer_id, "mamba_in_projection"
                                    ).value.upper()
                                    if kind == "mamba"
                                    else "BF16"
                                )
                                result = services.layer(kind, b, control, weight)
                                cats = {s["name"]: s["category"] for s in result["metadata"]["stages"]}
                                terms = [
                                    (
                                        cats[s["name"]],
                                        dict(
                                            components=s,
                                            hbm_read_bytes=s["hbm_read_bytes"],
                                            hbm_write_bytes=s["hbm_write_bytes"],
                                        ),
                                        1,
                                    )
                                    for s in result["sections"]
                                ]
                            else:
                                occ = (
                                    routes.get((b, stage.layer_id), {})
                                    if name == "nemotron3"
                                    else (
                                        {b: a.experts_per_token}
                                        if routing == "min_unique"
                                        else {1: b * a.experts_per_token}
                                    )
                                )
                                terms = compose_peripheral(stage, w, b, context, services, occ)
                            for category, result, factor in terms:
                                c = result["components"]
                                cycles = sum(c[k] for k in COMPONENTS)
                                if not math.isclose(cycles, c["total"]):
                                    raise AssertionError("nonexclusive service components")
                                ledger.update({k: c[k] * factor for k in COMPONENTS})
                                categories[category] += cycles * factor
                                read += result["hbm_read_bytes"] * factor
                                write += result["hbm_write_bytes"] * factor
                                details.append(
                                    dict(
                                        model=name,
                                        batch=b,
                                        context=context,
                                        routing=routing,
                                        control=control,
                                        layer=stage.layer_id,
                                        stage=stage.name,
                                        category=category,
                                        multiplicity=factor,
                                        cycles=cycles * factor,
                                        **{k: c[k] * factor for k in COMPONENTS},
                                        hbm_read_bytes=result["hbm_read_bytes"] * factor,
                                        hbm_write_bytes=result["hbm_write_bytes"] * factor,
                                    )
                                )
                        expected = 23 if name == "nemotron3" else 69
                        if len(replaced) != expected:
                            raise AssertionError("missing/double-counted recurrent sublayers")
                        total = sum(ledger.values())
                        if not math.isclose(total, sum(categories.values())):
                            raise AssertionError("category ledger does not reconcile")
                        tables.append(
                            dict(
                                model=name,
                                batch=b,
                                context=context,
                                routing=routing,
                                control=control,
                                cycles_per_batch_step=total,
                                ms_per_step=total / 1e6,
                                aggregate_tokens_s=b * 1e9 / total,
                                request_tokens_s=1e9 / total,
                                hbm_read_bytes=read,
                                hbm_write_bytes=write,
                                **ledger,
                                **{k + "_cycles": v for k, v in categories.items()},
                                **cap,
                                result_status="analytical_candidate"
                                if cap["capacity_feasible"]
                                else "capacity_infeasible_not_single_device_result",
                            )
                        )
                        write_csv(out / "decode_progress.csv", tables)
                        print(
                            json.dumps(
                                dict(
                                    model=name,
                                    batch=b,
                                    context=context,
                                    routing=routing,
                                    control=control,
                                    ms=total / 1e6,
                                )
                            ),
                            flush=True,
                        )
    for row in tables:
        group = [r for r in tables if all(r[k] == row[k] for k in ("model", "batch", "context", "routing"))]
        base = next(r for r in group if r["control"] == "old_isa")
        matched = next(r for r in group if r["control"] == "row")
        row["speedup_vs_old_recurrent_isa"] = base["cycles_per_batch_step"] / row["cycles_per_batch_step"]
        row["speedup_vs_matched_row"] = matched["cycles_per_batch_step"] / row["cycles_per_batch_step"]
        budget = base.get("routing_budget_cycles", 0) + base.get("kv_pack_budget_cycles", 0)
        # Remove/add the SAME uncertain orchestration costs in both arms.
        row["speedup_budget_0x"] = (base["cycles_per_batch_step"] - budget) / (row["cycles_per_batch_step"] - budget)
        row["speedup_budget_2x"] = (base["cycles_per_batch_step"] + budget) / (row["cycles_per_batch_step"] + budget)
    write_csv(out / "decode_all.csv", tables)
    write_csv(out / "decode_capacity_feasible.csv", [r for r in tables if r["capacity_feasible"]])
    write_csv(out / "operators.csv", details)
    write_csv(out / "capacity.csv", capacities)
    write_csv(out / "tensor_inventory.csv", inventories)
    (out / "status.json").write_text(
        json.dumps(
            dict(
                status="completed",
                points=len(tables),
                capacity_feasible_points=sum(r["capacity_feasible"] for r in tables),
                limits=[
                    "independent-operator memory reset requires composition error validation",
                    "KV append and router orchestration are named resource budgets",
                    "Kimi routing is a boundary scenario; its full checkpoint is not executed",
                    "BF16 software updates and fused FP32 updates have different arithmetic",
                    "candidate codec throughput is not RTL-measured",
                    "peripheral primitive timing calibrated; complete attention/MLA/MoE numerical composition and routing quality remain unvalidated",
                ],
            ),
            indent=2,
        )
        + "\n"
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "compiler", "memory-root", "campaign"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--service-cache", type=Path)
    p.add_argument(
        "--include-infeasible", action="store_true", help="diagnostic timing only; not single-device results"
    )
    p.add_argument("--controllers", type=int, default=16)
    p.add_argument("--dma-window", type=int, default=32)
    p.add_argument("--codec-lanes", type=int, choices=(128, 256, 512), default=256)
    p.add_argument("--sfu-scale", type=int, choices=(1, 2), default=1)
    p.add_argument("--models", nargs="+", choices=("nemotron3", "kimi_k3"), default=["nemotron3", "kimi_k3"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--contexts", nargs="+", type=int, default=[4096, 32768, 131072])
    run(p.parse_args(argv))


if __name__ == "__main__":
    main()
