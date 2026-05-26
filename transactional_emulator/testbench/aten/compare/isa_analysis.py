"""Static ISA analysis helpers shared by local comparison testbenches."""

from __future__ import annotations

import tomllib
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INSTR_PREFIXES = ("S_", "C_", "H_", "V_", "M_")

MATRIX_COMPUTE_OPS = {
    "M_MM",
    "M_TMM",
    "M_BMM",
    "M_BTMM",
    "M_MV",
    "M_TMV",
    "M_BMV",
    "M_BTMV",
}
MATRIX_WRITE_OPS = {
    "M_MM_WO",
    "M_TMM_WO",
    "M_BMM_WO",
    "M_BTMM_WO",
    "M_MV_WO",
    "M_TMV_WO",
    "M_BMV_WO",
    "M_BTMV_WO",
}
SCALAR_FP_BASIC_OPS = {"S_ADD_FP", "S_SUB_FP", "S_MAX_FP", "S_MUL_FP"}
SCALAR_INT_OPS = {
    "S_ADD_INT",
    "S_ADDI_INT",
    "S_SUB_INT",
    "S_MUL_INT",
    "S_LUI_INT",
    "S_LD_INT",
    "S_ST_INT",
}
CONTROL_ONE_CYCLE_OPS = {
    "C_SET_ADDR_REG",
    "C_SET_SCALE_REG",
    "C_SET_STRIDE_REG",
    "C_SET_V_MASK_REG",
    "C_LOOP_START",
    "C_LOOP_END",
    "C_BREAK",
}


@dataclass(frozen=True)
class SimulatorCycleModel:
    """Opcode cost model matching explicit `cycle!(...)` calls in the behavior simulator."""

    settings_path: Path
    dc_en: int
    latency_profile: str | None
    mlen: int
    vlen: int
    systolic_processing_overhead: int
    vector_add_cycles: int
    vector_mul_cycles: int
    vector_exp_cycles: int
    vector_reci_cycles: int
    vector_max_cycles: int
    vector_sum_cycles: int
    scalar_fp_basic_cycles: int
    scalar_fp_exp_cycles: int
    scalar_fp_sqrt_cycles: int
    scalar_fp_reci_cycles: int
    scalar_int_basic_cycles: int

    @property
    def description(self) -> str:
        return (
            f"behavior simulator constants from {self.settings_path} "
            f"(DC_EN={self.dc_en}, latency_profile={self.latency_profile or 'dc_lib_dis'}; "
            "H_PREFETCH/H_STORE async memory timing not statically charged)"
        )

    def instruction_cycles(self, opcode: str) -> int:
        if opcode in MATRIX_COMPUTE_OPS:
            return self.systolic_processing_overhead + self.mlen
        if opcode in MATRIX_WRITE_OPS:
            return 1
        if opcode.startswith("V_ADD") or opcode.startswith("V_SUB"):
            return self.vector_add_cycles
        if opcode.startswith("V_MUL") or opcode == "V_SHIFT_V":
            return self.vector_mul_cycles
        if opcode == "V_EXP_V":
            return self.vector_exp_cycles
        if opcode == "V_RECI_V":
            return self.vector_reci_cycles
        if opcode == "V_RED_MAX":
            return self.vector_max_cycles
        if opcode == "V_RED_SUM":
            return self.vector_sum_cycles
        if opcode in SCALAR_FP_BASIC_OPS:
            return self.scalar_fp_basic_cycles
        if opcode == "S_EXP_FP":
            return self.scalar_fp_exp_cycles
        if opcode == "S_RECI_FP":
            return self.scalar_fp_reci_cycles
        if opcode == "S_SQRT_FP":
            return self.scalar_fp_sqrt_cycles
        if opcode in {"S_LD_FP", "S_ST_FP"}:
            return 1
        if opcode == "S_MAP_V_FP":
            return self.vlen
        if opcode in SCALAR_INT_OPS:
            return self.scalar_int_basic_cycles
        if opcode in CONTROL_ONE_CYCLE_OPS:
            return 1
        if opcode.startswith("H_PREFETCH") or opcode.startswith("H_STORE"):
            # The simulator starts an async memory transfer here; the wait is
            # paid by a later SRAM read/write if the transfer has not resolved.
            return 0
        return 1


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_latency_profile(profile: str | None) -> str | None:
    if profile is None:
        return None
    profile = profile.strip()
    if not profile or profile in {"default", "dc_lib_dis"}:
        return None
    return profile


def _latency_value(section: dict[str, Any], name: str, *, dc_en: int, latency_profile: str | None) -> int:
    values = section[name]
    if dc_en:
        return int(values["dc_lib_en"])
    latency_profile = _normalize_latency_profile(latency_profile)
    if latency_profile is not None and latency_profile in values:
        return int(values[latency_profile])
    return int(values["dc_lib_dis"])


def load_behavior_cycle_model(
    *,
    settings_path: Path | None = None,
    dc_en: int = 1,
    latency_profile: str | None = None,
) -> SimulatorCycleModel:
    """Load behavior simulator cycle constants from `plena_settings.toml`.

    The default `dc_en=1` intentionally mirrors the local simulator setting
    used for the behavior config, while still allowing callers to ask for the
    disabled-latency column. When `dc_en=0`, `latency_profile` can select a
    named per-FPGA override from each BEHAVIOR.LATENCY entry, falling back to
    `dc_lib_dis` where an override is absent.
    """

    if settings_path is None:
        settings_path = _repo_root_from_here() / "plena_settings.toml"
    settings_path = settings_path.resolve()
    with settings_path.open("rb") as f:
        settings = tomllib.load(f)

    behavior = settings["TRANSACTIONAL"]
    config = behavior["CONFIG"]
    latency = behavior["LATENCY"]
    if latency_profile is None:
        latency_profile = os.environ.get("PLENA_LATENCY_PROFILE")
    if latency_profile is None:
        profile_config = config.get("LATENCY_PROFILE")
        if isinstance(profile_config, dict):
            latency_profile = profile_config.get("value")
    latency_profile = _normalize_latency_profile(latency_profile)
    if dc_en:
        latency_profile = None

    return SimulatorCycleModel(
        settings_path=settings_path,
        dc_en=dc_en,
        latency_profile=latency_profile,
        mlen=int(config["MLEN"]["value"]),
        vlen=int(config["VLEN"]["value"]),
        systolic_processing_overhead=_latency_value(
            latency, "SYSTOLIC_PROCESSING_OVERHEAD", dc_en=dc_en, latency_profile=latency_profile
        ),
        vector_add_cycles=_latency_value(latency, "VECTOR_ADD_CYCLES", dc_en=dc_en, latency_profile=latency_profile),
        vector_mul_cycles=_latency_value(latency, "VECTOR_MUL_CYCLES", dc_en=dc_en, latency_profile=latency_profile),
        vector_exp_cycles=_latency_value(latency, "VECTOR_EXP_CYCLES", dc_en=dc_en, latency_profile=latency_profile),
        vector_reci_cycles=_latency_value(latency, "VECTOR_RECI_CYCLES", dc_en=dc_en, latency_profile=latency_profile),
        vector_max_cycles=_latency_value(latency, "VECTOR_MAX_CYCLES", dc_en=dc_en, latency_profile=latency_profile),
        vector_sum_cycles=_latency_value(latency, "VECTOR_SUM_CYCLES", dc_en=dc_en, latency_profile=latency_profile),
        scalar_fp_basic_cycles=_latency_value(
            latency, "SCALAR_FP_BASIC_CYCLES", dc_en=dc_en, latency_profile=latency_profile
        ),
        scalar_fp_exp_cycles=_latency_value(
            latency, "SCALAR_FP_EXP_CYCLES", dc_en=dc_en, latency_profile=latency_profile
        ),
        scalar_fp_sqrt_cycles=_latency_value(
            latency, "SCALAR_FP_SQRT_CYCLES", dc_en=dc_en, latency_profile=latency_profile
        ),
        scalar_fp_reci_cycles=_latency_value(
            latency, "SCALAR_FP_RECI_CYCLES", dc_en=dc_en, latency_profile=latency_profile
        ),
        scalar_int_basic_cycles=_latency_value(
            latency, "SCALAR_INT_BASIC_CYCLES", dc_en=dc_en, latency_profile=latency_profile
        ),
    )


def loop_count_from(line: str) -> int:
    try:
        return int(line.split(",")[-1].strip())
    except (IndexError, ValueError):
        return 1


def analyze_asm(
    asm: str,
    *,
    cycle_model: SimulatorCycleModel | None = None,
    selected_opcodes: tuple[str, ...] = (
        "C_LOOP_START",
        "C_LOOP_END",
        "H_PREFETCH_M",
        "H_PREFETCH_V",
        "M_TMM",
        "M_BTMM",
        "M_MM",
        "M_MM_WO",
        "M_BMM_WO",
        "V_ADD_VV",
        "V_MUL_VF",
        "V_EXP_V",
        "V_RED_MAX",
        "V_RED_SUM",
        "S_LD_FP",
        "S_ST_FP",
    ),
) -> dict[str, Any]:
    if cycle_model is None:
        cycle_model = load_behavior_cycle_model()

    lines = asm.splitlines()
    loop_stack: list[int] = []
    op_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    static_instr = 0
    dynamic_instr = 0
    estimated_cycles = 0

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        token = line.split()[0]
        if not token.startswith(INSTR_PREFIXES):
            continue

        multiplier = 1
        for loop_count in loop_stack:
            multiplier *= loop_count

        static_instr += 1
        dynamic_instr += multiplier
        estimated_cycles += cycle_model.instruction_cycles(token) * multiplier
        op_counts[token] += multiplier
        type_counts[token[0]] += multiplier

        if token == "C_LOOP_START":
            loop_stack.append(loop_count_from(line))
        elif token == "C_LOOP_END" and loop_stack:
            loop_stack.pop()

    return {
        "source_lines": len(lines),
        "static_instruction_lines": static_instr,
        "comment_or_metadata_lines": len(lines) - static_instr,
        "dynamic_instruction_count": dynamic_instr,
        "estimated_cycles": estimated_cycles,
        "estimated_ms_at_1ghz": estimated_cycles / 1_000_000,
        "cycle_model": cycle_model.description,
        "cycle_model_dc_en": cycle_model.dc_en,
        "cycle_model_latency_profile": cycle_model.latency_profile,
        "loop_start_lines": sum(1 for line in lines if line.strip().startswith("C_LOOP_START")),
        "instruction_types_dynamic": dict(sorted(type_counts.items())),
        "selected_opcodes_dynamic": {op: op_counts[op] for op in selected_opcodes if op_counts[op]},
    }
