"""Shared program accounting; prediction never consumes measured timing."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .ltile_cost import assembly_cost
from .ltile_platform import ExecutionProfile


def price_program(assembly, profile, backend, *, matrix=True, nvfp4_regions=()):
    configured = len(json.loads(backend.config.read_text())["memory_system"]["controllers"])
    if configured != profile.hbm_controllers or profile.machine.clock_hz != 10**9:
        raise ValueError("execution profile and memory backend topology/clock differ")
    cost = assembly_cost(
        assembly,
        profile.machine,
        trace_memory=True,
        matrix_service=profile.matrix if matrix else None,
    )
    codec = apply_weight_codec(cost, nvfp4_regions, profile.codec) if nvfp4_regions else {}
    memory = backend.price(cost, profile.dma.service)
    return cost, dict(
        components=cost.components(),
        memory=memory,
        profile=asdict(profile),
        profile_sha256=profile.identity,
        assembly_sha256=hashlib.sha256(assembly.encode()).hexdigest(),
        hbm_read_bytes=memory["read_bytes"],
        hbm_write_bytes=memory["write_bytes"],
        output_format="BF16",
        timing_evidence="compiled analytical schedule + Ramulator",
        codec=codec,
    )


def apply_weight_codec(cost, regions, resource):
    """Replace only weight transfers, retaining the instruction/port schedule.

    Finite packed input/output buffers serialize decoding before Matrix SRAM
    placement. Global scale is fetched once per tensor. Addresses use the
    tensor's reserved region; no expanded BF16 tensor is written to HBM.
    """
    from collections import Counter
    from .weight_codec import WeightPacket

    regions = sorted(regions)
    for i, (base, size) in enumerate(regions):
        if base % 64 or size < 2048 or (i and regions[i - 1][0] + regions[i - 1][1] > base):
            raise ValueError("invalid compressed weight region")
    seen = set()
    trace = []
    stage = -1
    extra = Counter()
    decoded = 0
    for op, address, size in cost.memory_trace:
        if op == "m":
            stage += 1
        region = next(((b, n) for b, n in regions if op == "r" and b <= address < b + n), None)
        if region is None:
            trace.append((op, address, size))
            continue
        base, length = region
        if address + size > base + length:
            raise ValueError("weight read crosses tensor allocation")
        packet = WeightPacket(size // 2)
        service = packet.service(resource)
        if base not in seen:
            trace.append(("r", base, 64))
            seen.add(base)
        packed_address = base + 64 + (address - base) * 9 // 32
        if packed_address % 64 or packed_address + packet.transfer_bytes > base + length:
            raise ValueError("compressed packet placement exceeds reservation")
        trace.append(("r", packed_address, packet.transfer_bytes))
        cycles = sum(service[k] for k in ("issue", "arithmetic", "sram"))
        trace.append(("d", cycles, 0))
        decoded += size
        for key in ("issue", "arithmetic", "sram"):
            setattr(cost, key, getattr(cost, key) + service[key])
            extra[key] += service[key]
            if cost.sections:
                if stage < 0:
                    raise ValueError("unmarked weight stage")
                cost.sections[stage][key] += service[key]
                cost.sections[stage]["total"] += service[key]
        if cost.sections:
            cost.sections[stage]["frontend"] += service["issue"]
    cost.memory_trace = trace
    cost.transfers = Counter()
    for op, _, size in trace:
        if op in ("r", "w"):
            cost.transfers["read" if op == "r" else "write", size] += 1
    return dict(
        **extra,
        decoded_bf16_bytes=decoded,
        tensors=len(seen),
        input_buffer_bytes=resource.input_bytes,
        output_buffer_bytes=resource.output_bytes,
        status="finite serialized candidate; throughput sensitivity, no codec RTL claim",
    )


def validate_saved_execution(folder, backend):
    """Reprice a saved program independently, then compare physical counters.

    The observation is read only after prediction. Model parameters are explicit
    inputs saved before the original execution, not fitted to its results.
    """
    from .ltile_cost import Machine
    from .matrix_service import MatrixService
    from .ltile_platform import DmaResources

    folder = Path(folder)
    original = json.loads((folder / "result.json").read_text())
    profile = ExecutionProfile(
        machine=Machine(**original["machine"]),
        matrix=MatrixService(**json.loads((folder / "matrix_profile.json").read_text())),
        dma=DmaResources(),
    )
    _, prediction = price_program((folder / "generated_asm_code.asm").read_text(), profile, backend)
    timing = json.loads((folder / "execution_timing.json").read_text())
    c = timing["counters"]
    observed = dict(
        issue=c["issue_cycles"],
        scalar=timing["scalar_and_control_cycles"],
        sram=c["bank_service_cycles"],
        arithmetic=c["arithmetic_cycles"],
        dependency=c["dependency_cycles"],
        dma=timing["dma_and_memory_wait_picos"] / timing["period_picos"],
        total=timing["total_picos"] / timing["period_picos"],
    )
    errors = {key: prediction["components"][key] - value for key, value in observed.items()}
    if any(errors.values()):
        raise AssertionError(f"saved execution mismatch: {folder.name}: {errors}")
    for key in ("hbm_read_bytes", "hbm_write_bytes"):
        if prediction[key] != timing[key]:
            raise AssertionError(f"physical traffic mismatch: {key}")
    return dict(case=folder.name, prediction=prediction, observed=observed, component_errors=errors)
