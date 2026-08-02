"""Timing providers for compiler-emitted PLENA instructions.

The default provider mirrors the current transactional emulator's dispatch
latencies. It intentionally does not use the legacy closed-form model's
``pipelined`` constants, which describe a different abstraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import toml

from compiler.aten.program_sink import TraceInstruction


def _value(section: Mapping[str, Any], name: str) -> int:
    raw = section[name]
    if isinstance(raw, Mapping):
        raw = raw.get("value", raw)
    return int(raw)


def _latency_value(section: Mapping[str, Any], name: str, dc_enabled: bool) -> int:
    raw = section[name]
    if not isinstance(raw, Mapping):
        return int(raw)
    key = "dc_lib_en" if dc_enabled else "dc_lib_dis"
    return int(raw[key])


@dataclass(frozen=True)
class MainTimingConfig:
    """Subset of main's runtime configuration that affects compute timing."""

    mlen: int
    blen: int
    vlen: int
    hlen: int
    broadcast_amount: int
    period_picos: int = 1_000
    systolic_processing_overhead: int = 0
    vector_add_cycles: int = 1
    vector_mul_cycles: int = 1
    vector_exp_cycles: int = 1
    vector_reci_cycles: int = 2
    vector_max_cycles: int = 4
    vector_sum_cycles: int = 8
    scalar_fp_basic_cycles: int = 1
    scalar_fp_exp_cycles: int = 1
    scalar_fp_sqrt_cycles: int = 1
    scalar_fp_reci_cycles: int = 1
    scalar_int_basic_cycles: int = 1
    source: str = "explicit"
    dc_enabled: bool = True

    def __post_init__(self) -> None:
        for name in (
            "mlen",
            "blen",
            "vlen",
            "hlen",
            "broadcast_amount",
            "period_picos",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.mlen % self.blen:
            raise ValueError("MLEN must be divisible by BLEN")

    @classmethod
    def from_toml(
        cls,
        path: str | Path,
        *,
        section: str = "TRANSACTIONAL",
        period_picos: int = 1_000,
    ) -> "MainTimingConfig":
        data = toml.load(path)
        if section not in data:
            raise KeyError(f"missing {section} section in {path}")
        root = data[section]
        config = root["CONFIG"]
        latency = root["LATENCY"]
        dc_enabled = bool(_value(config, "DC_EN"))
        return cls(
            mlen=_value(config, "MLEN"),
            blen=_value(config, "BLEN"),
            vlen=_value(config, "VLEN"),
            hlen=_value(config, "HLEN"),
            broadcast_amount=_value(config, "BROADCAST_AMOUNT"),
            period_picos=period_picos,
            systolic_processing_overhead=_latency_value(
                latency, "SYSTOLIC_PROCESSING_OVERHEAD", dc_enabled
            ),
            vector_add_cycles=_latency_value(latency, "VECTOR_ADD_CYCLES", dc_enabled),
            vector_mul_cycles=_latency_value(latency, "VECTOR_MUL_CYCLES", dc_enabled),
            vector_exp_cycles=_latency_value(latency, "VECTOR_EXP_CYCLES", dc_enabled),
            vector_reci_cycles=_latency_value(latency, "VECTOR_RECI_CYCLES", dc_enabled),
            vector_max_cycles=_latency_value(latency, "VECTOR_MAX_CYCLES", dc_enabled),
            vector_sum_cycles=_latency_value(latency, "VECTOR_SUM_CYCLES", dc_enabled),
            scalar_fp_basic_cycles=_latency_value(
                latency, "SCALAR_FP_BASIC_CYCLES", dc_enabled
            ),
            scalar_fp_exp_cycles=_latency_value(latency, "SCALAR_FP_EXP_CYCLES", dc_enabled),
            scalar_fp_sqrt_cycles=_latency_value(
                latency, "SCALAR_FP_SQRT_CYCLES", dc_enabled
            ),
            scalar_fp_reci_cycles=_latency_value(
                latency, "SCALAR_FP_RECI_CYCLES", dc_enabled
            ),
            scalar_int_basic_cycles=_latency_value(
                latency, "SCALAR_INT_BASIC_CYCLES", dc_enabled
            ),
            source=f"{Path(path).resolve()}#{section}",
            dc_enabled=dc_enabled,
        )


class TimingProvider(Protocol):
    name: str

    def latency_picos(
        self,
        instruction: TraceInstruction,
        trace_metadata: Mapping[str, Any],
    ) -> int: ...

    def provenance(self) -> dict[str, Any]: ...


class MainTimingProvider:
    """Mirror the serial compute durations in main's Rust emulator."""

    name = "main-emulator-v1"

    def __init__(self, config: MainTimingConfig):
        self.config = config

    def _cycles(self, item: TraceInstruction, metadata: Mapping[str, Any]) -> int:
        opcode = item.opcode
        c = self.config

        if opcode in {"M_MM", "M_TMM", "M_BMM", "M_BTMM"}:
            return c.systolic_processing_overhead + c.mlen
        if opcode in {"M_MM_WO", "M_BMM_WO", "M_MV_WO", "M_BMV_WO"}:
            return 1
        if opcode in {"M_MV", "M_TMV"}:
            return c.mlen
        if opcode in {"M_BMV", "M_BTMV"}:
            return c.systolic_processing_overhead + 1

        if opcode in {"V_ADD_VV", "V_ADD_VF", "V_SUB_VV", "V_SUB_VF"}:
            return c.vector_add_cycles
        if opcode in {"V_MUL_VV", "V_MUL_VF", "V_SHFT_V"}:
            return c.vector_mul_cycles
        if opcode in {"V_MAX_VF", "V_MIN_VF", "V_RED_MAX"}:
            return c.vector_max_cycles
        if opcode == "V_RED_SUM":
            return c.vector_sum_cycles
        if opcode == "V_EXP_V":
            return c.vector_exp_cycles
        if opcode == "V_RECI_V":
            return c.vector_reci_cycles
        if opcode == "V_TOPK":
            variants = dict(item.variant)
            expert_count = variants.get("expert_count")
            if expert_count is None:
                histogram = metadata.get("routing_histogram")
                expert_count = len(histogram) if histogram is not None else None
            if expert_count is None:
                raise ValueError("V_TOPK timing requires expert_count variant or routing histogram")
            return c.vector_max_cycles * int(expert_count)

        if opcode in {"S_ADD_FP", "S_SUB_FP", "S_MAX_FP", "S_MUL_FP"}:
            return c.scalar_fp_basic_cycles
        if opcode == "S_EXP_FP":
            return c.scalar_fp_exp_cycles
        if opcode == "S_RECI_FP":
            return c.scalar_fp_reci_cycles
        if opcode == "S_SQRT_FP":
            return c.scalar_fp_sqrt_cycles
        if opcode in {"S_LD_FP", "S_ST_FP"}:
            return 1
        if opcode == "S_MAP_V_FP":
            # vector_transfer_fp and the dispatch-side completion delay each
            # advance VLEN cycles in main.
            return 2 * c.vlen
        if opcode in {
            "S_ADD_INT",
            "S_ADDI_INT",
            "S_SUB_INT",
            "S_MUL_INT",
            "S_LUI_INT",
            "S_LD_INT",
            "S_ST_INT",
        }:
            return c.scalar_int_basic_cycles

        if opcode in {
            "C_SET_ADDR_REG",
            "C_SET_SCALE_REG",
            "C_SET_STRIDE_REG",
            "C_SET_V_MASK_REG",
            "C_LOOP_START",
            "C_LOOP_END",
            "C_BREAK",
        }:
            return 1
        if opcode.startswith("H_"):
            raise ValueError(f"HBM opcode {opcode} belongs to a memory provider")
        raise ValueError(f"main timing provider has no entry for {opcode}")

    def latency_picos(
        self,
        instruction: TraceInstruction,
        trace_metadata: Mapping[str, Any],
    ) -> int:
        return self._cycles(instruction, trace_metadata) * self.config.period_picos

    def provenance(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "source": self.config.source,
            "period_picos": self.config.period_picos,
            "dc_enabled": self.config.dc_enabled,
            "matrix_semantics": "main-row-granular-structural",
            "dependency_model": "serial-resource-work",
        }


class IdealII1TimingProvider(MainTimingProvider):
    """Experimental hazard-free V/S/C model with structural Matrix timing."""

    name = "ideal-ii1-experimental-v1"

    def _cycles(self, item: TraceInstruction, metadata: Mapping[str, Any]) -> int:
        if item.opcode.startswith(("V_", "S_", "C_")):
            return 1
        return super()._cycles(item, metadata)

    def provenance(self) -> dict[str, Any]:
        result = super().provenance()
        result.update(
            {
                "provider": self.name,
                "timing_status": "architectural-ideal-assumption",
                "vector_scalar_control_cycles": 1,
                "hazards_included": False,
            }
        )
        return result


__all__ = [
    "IdealII1TimingProvider",
    "MainTimingConfig",
    "MainTimingProvider",
    "TimingProvider",
]
