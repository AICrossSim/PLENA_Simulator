"""Translate final compiler schedules into structural energy actions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from compiler.aten.program_sink import CostTrace, TraceInstruction

from .schemas import ActionHardwareConfig, EnergyAction


MATRIX_ARRAY = {"M_MM", "M_TMM", "M_BMM", "M_BTMM"}
MATRIX_VECTOR = {"M_MV", "M_TMV", "M_BMV", "M_BTMV"}
MATRIX_WRITEOUT = {"M_MM_WO", "M_BMM_WO", "M_MV_WO", "M_BMV_WO"}
VECTOR_TWO_READ = {"V_ADD_VV", "V_SUB_VV", "V_MUL_VV"}


def _logic_actions(
    item: TraceInstruction,
    hardware: ActionHardwareConfig,
    trace_metadata: Mapping[str, Any],
) -> list[EnergyAction]:
    opcode = item.opcode
    count = item.multiplicity
    active = item.active or {}
    active_lanes = int(active.get("lanes") or hardware.vlen)
    total_lanes = int(active.get("total_lanes") or hardware.vlen)
    fidelity = "compiler-active-shape" if item.active is not None else "physical-full-width-from-main-isa"
    common = {
        "stage": item.stage,
        "count": count,
        "source_opcode": opcode,
        "precision": hardware.fp_format,
    }

    if opcode in MATRIX_ARRAY | MATRIX_VECTOR:
        family = "array_compute" if opcode in MATRIX_ARRAY else "matrix_vector_compute"
        return [
            EnergyAction(
                component="matrix",
                action=family,
                active_instances=hardware.blen * hardware.blen,
                total_instances=hardware.blen * hardware.blen,
                fidelity="main-structural-matrix",
                **common,
            ),
            EnergyAction(
                component="matrix",
                action="cross_k_reduce",
                active_instances=max(0, hardware.mlen // hardware.blen - 1),
                total_instances=max(0, hardware.mlen // hardware.blen - 1),
                fidelity="main-structural-matrix",
                **common,
            ),
        ]
    if opcode in MATRIX_WRITEOUT:
        return [
            EnergyAction(
                component="matrix",
                action="output_conversion",
                active_instances=hardware.blen * hardware.blen,
                total_instances=hardware.blen * hardware.blen,
                fidelity="main-structural-matrix",
                **common,
            )
        ]

    vector_family = {
        "V_ADD_VV": "lane_add_sub_vv",
        "V_SUB_VV": "lane_add_sub_vv",
        "V_ADD_VF": "lane_add_sub_vf",
        "V_SUB_VF": "lane_add_sub_vf",
        "V_MAX_VF": "lane_add_sub_vf",
        "V_MIN_VF": "lane_add_sub_vf",
        "V_MUL_VV": "lane_multiply_vv",
        "V_MUL_VF": "lane_multiply_vf",
        "V_EXP_V": "lane_sfu_exp",
        "V_RECI_V": "lane_sfu_reciprocal",
        "V_SHFT_V": "lane_movement_shift",
        "V_RED_SUM": "reduction_sum_full",
        "V_RED_MAX": "reduction_max_full",
        "V_BASIC": "lane_add_sub_vv",
    }
    if opcode == "V_TOPK":
        variants = dict(item.variant)
        expert_count = int(variants.get("expert_count", 0) or 0)
        if not expert_count:
            histogram = trace_metadata.get("routing_histogram")
            expert_count = len(histogram) if histogram is not None else 0
        if not expert_count:
            expert_count = int((item.active or {}).get("cols") or 0)
        if not expert_count:
            raise ValueError("V_TOPK power action requires expert_count metadata")
        count *= expert_count
        vector_family[opcode] = "reduction_max_full"
    if opcode in vector_family:
        return [
            EnergyAction(
                component="vector",
                action=vector_family[opcode],
                count=count,
                source_opcode=opcode,
                precision=hardware.fp_format,
                active_instances=active_lanes,
                total_instances=total_lanes,
                active_bits=int(active.get("bits") or active_lanes * hardware.fp_width),
                fidelity=fidelity,
                stage=item.stage,
            )
        ]

    scalar_family = {
        "S_ADD_FP": "fp_add_sub_move",
        "S_SUB_FP": "fp_add_sub_move",
        "S_MAX_FP": "fp_add_sub_move",
        "S_BASIC": "fp_add_sub_move",
        "S_MUL_FP": "fp_multiply",
        "S_EXP_FP": "fp_sfu_exp",
        "S_RECI_FP": "fp_sfu_reciprocal",
        "S_SQRT_FP": "fp_sfu_sqrt",
        "S_MUL_INT": "integer_multiply",
        "S_ADD_INT": "integer_alu",
        "S_ADDI_INT": "integer_alu",
        "S_SUB_INT": "integer_alu",
        "S_LUI_INT": "integer_alu",
        "S_LD_FP": "register_or_sram_access",
        "S_ST_FP": "register_or_sram_access",
        "S_LD_INT": "register_or_sram_access",
        "S_ST_INT": "register_or_sram_access",
        "S_MAP_V_FP": "register_or_sram_access",
    }
    if opcode in scalar_family:
        return [
            EnergyAction(
                component="scalar",
                action=scalar_family[opcode],
                fidelity="exact-opcode-family",
                **common,
            )
        ]
    if opcode.startswith("C_"):
        return [
            EnergyAction(
                component="control",
                action="frontend_issue",
                fidelity="exact-opcode-family",
                **common,
            )
        ]
    if opcode.startswith("H_"):
        return []
    raise ValueError(f"no power action family for opcode {opcode!r}")


def _implicit_sram_actions(item: TraceInstruction, hardware: ActionHardwareConfig) -> list[EnergyAction]:
    opcode = item.opcode
    accesses: tuple[tuple[str, str, int], ...]
    if opcode in MATRIX_ARRAY | MATRIX_VECTOR | MATRIX_WRITEOUT:
        accesses = (("matrix_sram", "read", 2), ("matrix_sram", "write", 1))
    elif opcode.startswith("V_"):
        reads = 2 if opcode in VECTOR_TWO_READ else 1
        accesses = (("vector_sram", "read", reads), ("vector_sram", "write", 1))
    elif opcode == "S_LD_FP":
        accesses = (("scalar_fp_sram", "read", 1),)
    elif opcode in {"S_ST_FP", "S_MAP_V_FP"}:
        accesses = (("scalar_fp_sram", "write", 1),)
    elif opcode == "S_LD_INT":
        accesses = (("scalar_int_sram", "read", 1),)
    elif opcode == "S_ST_INT":
        accesses = (("scalar_int_sram", "write", 1),)
    else:
        accesses = ()
    return [
        EnergyAction(
            stage=item.stage,
            component=memory,
            action=direction,
            count=item.multiplicity * access_count,
            source_opcode=opcode,
            precision=hardware.fp_format,
            fidelity="main-isa-implied-access",
        )
        for memory, direction, access_count in accesses
    ]


def _explicit_sram_actions(item: TraceInstruction, hardware: ActionHardwareConfig) -> list[EnergyAction]:
    return [
        EnergyAction(
            stage=item.stage,
            component=str(access["memory"]),
            action=str(access["direction"]),
            count=item.multiplicity * int(access.get("accesses", 1)),
            source_opcode=item.opcode,
            precision=hardware.fp_format,
            bytes=(
                0
                if access.get("bytes_per_access") is None
                else item.multiplicity
                * int(access.get("accesses", 1))
                * int(access["bytes_per_access"])
            ),
            fidelity="compiler-sram-descriptor",
        )
        for access in item.sram
    ]


def _dma_actions(trace: CostTrace, hardware: ActionHardwareConfig) -> list[EnergyAction]:
    result: list[EnergyAction] = []
    for event in trace.dma_events:
        transfer = event.transfer
        action = {
            "H_PREFETCH_M": "matrix_prefetch",
            "H_PREFETCH_V": "vector_prefetch",
            "H_STORE_V": "vector_writeback",
        }.get(transfer.opcode)
        if action is None:
            raise ValueError(f"no HBM-controller action for {transfer.opcode!r}")
        amount = transfer.write_amount if transfer.direction == "write" else transfer.amount
        result.append(
            EnergyAction(
                stage=event.stage,
                component="hbm_controller",
                action=action,
                count=event.multiplicity,
                source_opcode=transfer.opcode,
                precision=transfer.precision,
                active_instances=amount,
                total_instances=transfer.dim,
                fidelity=transfer.geometry_fidelity,
            )
        )
        result.append(
            EnergyAction(
                stage=event.stage,
                component="matrix_sram" if transfer.opcode == "H_PREFETCH_M" else "vector_sram",
                action="read" if transfer.direction == "write" else "write",
                count=event.multiplicity * amount,
                source_opcode=transfer.opcode,
                precision=transfer.precision,
                fidelity="compiler-dma-geometry",
            )
        )
    return result


def _merge(actions: Iterable[EnergyAction]) -> tuple[EnergyAction, ...]:
    grouped: dict[tuple[Any, ...], EnergyAction] = {}
    for action in actions:
        key = tuple(
            value
            for name, value in vars(action).items()
            if name not in {"count", "busy_picos", "bytes"}
        )
        previous = grouped.get(key)
        grouped[key] = action if previous is None else replace(
            previous,
            count=previous.count + action.count,
            busy_picos=previous.busy_picos + action.busy_picos,
            bytes=previous.bytes + action.bytes,
        )
    return tuple(sorted(grouped.values(), key=lambda item: (item.stage, item.component, item.action, item.source_opcode)))


def build_energy_actions(
    trace: CostTrace,
    hardware_config: ActionHardwareConfig,
) -> tuple[EnergyAction, ...]:
    """Build a complete compressed action census from final compiler work."""

    actions: list[EnergyAction] = []
    hbm_instruction_counts: dict[str, int] = defaultdict(int)
    for item in trace.instructions:
        actions.extend(_logic_actions(item, hardware_config, trace.metadata))
        actions.extend(
            _explicit_sram_actions(item, hardware_config)
            if item.sram
            else _implicit_sram_actions(item, hardware_config)
        )
        if item.opcode.startswith("H_"):
            hbm_instruction_counts[item.opcode] += item.multiplicity
    actions.extend(_dma_actions(trace, hardware_config))

    dma_counts: dict[str, int] = defaultdict(int)
    for event in trace.dma_events:
        dma_counts[event.transfer.opcode] += event.multiplicity
    if dict(hbm_instruction_counts) != dict(dma_counts):
        raise ValueError(
            "power actions require exact HBM instruction/DMA parity: "
            f"instructions={dict(hbm_instruction_counts)}, dma={dict(dma_counts)}"
        )
    return _merge(actions)


__all__ = ["build_energy_actions"]
