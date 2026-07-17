"""RTL-calibrated opcode timing shared with the transactional emulator.

The calibration JSON is the single source of measured cycle coefficients.  The
Rust transactional scheduler and this Python cost evaluator intentionally use
the same measurement boundaries:

* ``resource_cycles`` is the backend occupancy interval;
* ``result_ready_cycles`` is the consumer-visible dependency interval;
* ``initiation_interval_cycles`` is the minimum independent issue spacing.

CostEmitter's primary compute objective sums ``resource_cycles``.  That sum is
resource work, not a scheduled makespan; the distinction is carried explicitly
in the generated report.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RTL_TIMING_CALIBRATION = (
    REPO_ROOT / "transactional_emulator/calibration/rtl_opcode_timing_v1.json"
)

MATRIX_TILE_OPS = {"M_MM", "M_TMM", "M_BMM", "M_BTMM"}
MATRIX_VECTOR_OPS = {"M_MV", "M_TMV", "M_BMV", "M_BTMV"}
MATRIX_GEMM_WRITE_OPS = {"M_MM_WO", "M_BMM_WO"}
MATRIX_GEMV_WRITE_OPS = {"M_MV_WO", "M_BMV_WO"}
MATRIX_BROADCAST_OPS = {"M_BMM", "M_BTMM", "M_BMM_WO", "M_BMV", "M_BTMV", "M_BMV_WO"}
MEMORY_OPS = {"H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V"}
CONTROL_OPS = {
    "C_SET_ADDR_REG",
    "C_SET_SCALE_REG",
    "C_SET_STRIDE_REG",
    "C_SET_V_MASK_REG",
    "C_LOOP_START",
    "C_LOOP_END",
    "C_BREAK",
}
SCALAR_INT_OPS = {
    "S_ADD_INT",
    "S_ADDI_INT",
    "S_SUB_INT",
    "S_MUL_INT",
    "S_LUI_INT",
    "S_LD_INT",
    "S_ST_INT",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ceil_log2(value: int) -> int:
    if value <= 1:
        return 0
    return (value - 1).bit_length()


@dataclass(frozen=True)
class FpFormat:
    exponent: int
    mantissa: int

    @property
    def width(self) -> int:
        return 1 + self.exponent + self.mantissa

    @classmethod
    def parse(cls, value: FpFormat | str | Mapping[str, Any]) -> FpFormat:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            text = value.upper().replace("_", "")
            for prefix in ("FP", "MXFP"):
                if text.startswith(f"{prefix}E") and "M" in text:
                    exponent, mantissa = text.removeprefix(f"{prefix}E").split("M", 1)
                    return cls(int(exponent), int(mantissa))
            raise ValueError(f"unsupported FP format {value!r}")
        if not isinstance(value, Mapping):
            raise TypeError(f"FP format must be a mapping, got {type(value).__name__}")
        return cls(
            exponent=int(value.get("exponent", value.get("exp"))),
            mantissa=int(value.get("mantissa", value.get("mant"))),
        )


@dataclass(frozen=True)
class ComputeFormat:
    family: str
    element_bits: int
    exponent: int | None = None
    mantissa: int | None = None
    scale_bits: int = 8
    block: int = 64

    @property
    def fp_format(self) -> FpFormat | None:
        if self.family != "mxfp":
            return None
        assert self.exponent is not None and self.mantissa is not None
        return FpFormat(self.exponent, self.mantissa)

    @classmethod
    def parse(
        cls,
        value: ComputeFormat | str | Mapping[str, Any],
        *,
        default_scale_bits: int = 8,
        default_block: int = 64,
    ) -> ComputeFormat:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            text = value.upper().replace("_", "")
            if text.startswith("MXINT"):
                width = int(text.removeprefix("MXINT"))
                return cls("mxint", width, scale_bits=default_scale_bits, block=default_block)
            if text.startswith("MXFPE") and "M" in text:
                exponent, mantissa = text.removeprefix("MXFPE").split("M", 1)
                exp = int(exponent)
                mant = int(mantissa)
                return cls(
                    "mxfp",
                    1 + exp + mant,
                    exponent=exp,
                    mantissa=mant,
                    scale_bits=default_scale_bits,
                    block=default_block,
                )
            raise ValueError(f"unsupported compute precision {value!r}")
        if not isinstance(value, Mapping):
            raise TypeError(
                f"compute precision must be a string or mapping, got {type(value).__name__}"
            )
        family = str(value.get("family", value.get("kind", value.get("type", "")))).lower()
        family = family.replace("_", "")
        scale_bits = int(value.get("scale_bits", value.get("scale_width", default_scale_bits)))
        block = int(value.get("block", default_block))
        if family.startswith("mxint"):
            suffix = family.removeprefix("mxint")
            width = int(suffix or value.get("width", value.get("bits")))
            return cls("mxint", width, scale_bits=scale_bits, block=block)
        if family.startswith("mxfp"):
            exponent = int(value.get("exponent", value.get("exp")))
            mantissa = int(value.get("mantissa", value.get("mant")))
            return cls(
                "mxfp",
                1 + exponent + mantissa,
                exponent=exponent,
                mantissa=mantissa,
                scale_bits=scale_bits,
                block=block,
            )
        raise ValueError(f"unsupported compute precision mapping {value!r}")


def _mx_format_from_toml(section: Mapping[str, Any]) -> ComputeFormat:
    fmt = str(section.get("format", ""))
    if fmt != "Mx":
        raise ValueError(f"expected MX precision section, got format={fmt!r}")
    elem = section["ELEM"]
    kind = str(elem.get("type", ""))
    scale = section.get("SCALE", {})
    scale_bits = (
        int(bool(scale.get("sign", False)))
        + int(scale.get("exponent", 0))
        + int(scale.get("mantissa", 0))
        if scale.get("type") == "Fp"
        else int(scale.get("width", 0))
    )
    if kind == "Int":
        return ComputeFormat(
            "mxint",
            int(elem["width"]),
            scale_bits=scale_bits,
            block=int(section["block"]),
        )
    if kind == "Fp":
        exponent = int(elem["exponent"])
        mantissa = int(elem["mantissa"])
        return ComputeFormat(
            "mxfp",
            int(bool(elem.get("sign", True))) + exponent + mantissa,
            exponent=exponent,
            mantissa=mantissa,
            scale_bits=scale_bits,
            block=int(section["block"]),
        )
    raise ValueError(f"unsupported MX element type {kind!r}")


def _plain_fp_from_toml(section: Mapping[str, Any]) -> FpFormat:
    data = section.get("DATA_TYPE", section)
    if data.get("type") != "Fp":
        raise ValueError(f"expected plain FP section, got {data!r}")
    return FpFormat(int(data["exponent"]), int(data["mantissa"]))


@dataclass(frozen=True)
class ComputePrecisionConfig:
    weight: ComputeFormat
    activation: ComputeFormat
    kv: ComputeFormat
    matrix_internal_fp: FpFormat
    vector_internal_fp: FpFormat
    scalar_fp: FpFormat
    integer_bits: int

    @classmethod
    def from_settings(cls, settings: Mapping[str, Any]) -> ComputePrecisionConfig:
        precision = settings["PRECISION"]
        integer = precision["HBM_V_INT_TYPE"]["DATA_TYPE"]
        matrix_kv = _mx_format_from_toml(precision["HBM_M_KV_TYPE"])
        vector_kv = _mx_format_from_toml(precision["HBM_V_KV_TYPE"])
        if matrix_kv != vector_kv:
            raise ValueError(
                "rtl-v1 compute timing requires HBM_M_KV_TYPE and "
                f"HBM_V_KV_TYPE to match, got {matrix_kv!r} and {vector_kv!r}"
            )
        return cls(
            weight=_mx_format_from_toml(precision["HBM_M_WEIGHT_TYPE"]),
            activation=_mx_format_from_toml(precision["HBM_V_ACT_TYPE"]),
            kv=matrix_kv,
            matrix_internal_fp=_plain_fp_from_toml(precision["MATRIX_SRAM_TYPE"]),
            vector_internal_fp=_plain_fp_from_toml(precision["VECTOR_SRAM_TYPE"]),
            scalar_fp=_plain_fp_from_toml(precision["SCALAR_FP"]),
            integer_bits=int(integer["width"]),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        fallback: ComputePrecisionConfig,
    ) -> ComputePrecisionConfig:
        block = int(value.get("block", value.get("mx_block", fallback.weight.block)))
        scale_bits = int(
            value.get("scale_bits", value.get("scale_width", fallback.weight.scale_bits))
        )
        internal = value.get("internal_fp", value.get("fp", value.get("FP_SETTING")))
        matrix_fp = value.get("matrix_internal_fp", internal)
        vector_fp = value.get("vector_internal_fp", internal)
        scalar_fp = value.get("scalar_fp", internal)
        return cls(
            weight=ComputeFormat.parse(
                value.get(
                    "weight",
                    value.get(
                        "weight_precision", value.get("WEIGHT_WIDTH", fallback.weight)
                    ),
                ),
                default_scale_bits=scale_bits,
                default_block=block,
            ),
            activation=ComputeFormat.parse(
                value.get("activation", value.get("ACT_WIDTH", fallback.activation)),
                default_scale_bits=scale_bits,
                default_block=block,
            ),
            kv=ComputeFormat.parse(
                value.get("kv", value.get("KV_WIDTH", fallback.kv)),
                default_scale_bits=scale_bits,
                default_block=block,
            ),
            matrix_internal_fp=(
                fallback.matrix_internal_fp if matrix_fp is None else FpFormat.parse(matrix_fp)
            ),
            vector_internal_fp=(
                fallback.vector_internal_fp if vector_fp is None else FpFormat.parse(vector_fp)
            ),
            scalar_fp=fallback.scalar_fp if scalar_fp is None else FpFormat.parse(scalar_fp),
            integer_bits=int(
                value.get("integer_bits", value.get("int_data_width", fallback.integer_bits))
            ),
        )

    def mismatch_messages(self, other: ComputePrecisionConfig) -> list[str]:
        mismatches = []
        for name in self.__dataclass_fields__:
            expected = getattr(self, name)
            actual = getattr(other, name)
            if expected != actual:
                mismatches.append(f"{name}: settings={expected!r}, explicit={actual!r}")
        return mismatches

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OpcodeTimingEstimate:
    resource_cycles: int
    result_ready_cycles: int
    initiation_interval_cycles: int
    calibration_status: str
    rtl_supported: bool
    calibration_in_domain: bool
    first_result_ready_cycles: int | None = None
    result_cadence_cycles: int | None = None


@dataclass(frozen=True)
class TimingHardware:
    mlen: int
    blen: int
    vlen: int
    hlen: int = 1
    broadcast_amount: int = 1


@dataclass(frozen=True)
class RtlOpcodeTimingCalibration:
    path: Path
    data: Mapping[str, Any]
    sha256: str

    @classmethod
    def load(
        cls, path: str | Path = DEFAULT_RTL_TIMING_CALIBRATION
    ) -> RtlOpcodeTimingCalibration:
        source = Path(path).resolve()
        data = json.loads(source.read_text())
        if int(data.get("schema_version", -1)) != 3:
            raise ValueError(
                f"unsupported RTL opcode timing schema {data.get('schema_version')!r}"
            )
        if data.get("measurement_boundary") != (
            "full_machine_execute_accept_to_consumer_ready_and_backend_idle"
        ):
            raise ValueError(f"unexpected RTL timing measurement boundary in {source}")
        return cls(path=source, data=data, sha256=_sha256(source))

    @property
    def clock_period_assumption_ps(self) -> int:
        return int(self.data["clock_period_assumption_ps"])

    def metadata(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sha256": self.sha256,
            "schema_version": int(self.data["schema_version"]),
            "model": self.data["model"],
            "measurement_boundary": self.data["measurement_boundary"],
            "clock_period_assumption_ps": self.clock_period_assumption_ps,
            "source": dict(self.data["source"]),
        }

    @staticmethod
    def _fixed(
        cycles: int,
        status: str,
        *,
        supported: bool | None = None,
        in_domain: bool | None = None,
    ) -> OpcodeTimingEstimate:
        cycles = max(1, int(cycles))
        if supported is None:
            supported = status != "unsupported_rtl"
        if in_domain is None:
            in_domain = status in {"full_machine_measured", "ramulator_observed"}
        return OpcodeTimingEstimate(
            resource_cycles=cycles,
            result_ready_cycles=cycles,
            initiation_interval_cycles=cycles,
            calibration_status=status,
            rtl_supported=supported,
            calibration_in_domain=in_domain,
        )

    @staticmethod
    def _split(
        resource: int,
        ready: int,
        initiation_interval: int,
        status: str,
        *,
        supported: bool = True,
        in_domain: bool | None = None,
    ) -> OpcodeTimingEstimate:
        if in_domain is None:
            in_domain = status == "full_machine_measured"
        return OpcodeTimingEstimate(
            resource_cycles=max(1, int(resource)),
            result_ready_cycles=max(1, int(ready)),
            initiation_interval_cycles=max(1, int(initiation_interval)),
            calibration_status=status,
            rtl_supported=supported,
            calibration_in_domain=in_domain,
        )

    def estimate(
        self,
        opcode: str,
        hardware: TimingHardware,
        precision: ComputePrecisionConfig,
    ) -> OpcodeTimingEstimate | None:
        if opcode in MEMORY_OPS:
            return None
        if opcode in MATRIX_TILE_OPS | MATRIX_VECTOR_OPS | MATRIX_GEMM_WRITE_OPS | MATRIX_GEMV_WRITE_OPS:
            return self._matrix_estimate(opcode, hardware, precision)
        return self._nonmatrix_estimate(opcode, hardware, precision)

    def _matrix_estimate(
        self,
        opcode: str,
        hardware: TimingHardware,
        precision: ComputePrecisionConfig,
    ) -> OpcodeTimingEstimate:
        gemv = opcode in MATRIX_VECTOR_OPS | MATRIX_GEMV_WRITE_OPS
        writeout = opcode in MATRIX_GEMM_WRITE_OPS | MATRIX_GEMV_WRITE_OPS
        broadcast = opcode in MATRIX_BROADCAST_OPS
        if precision.weight.family == "mxint":
            config = self.data["matrix"]["mxint"]
            if writeout:
                cycles = (
                    int(config["gemm_writeout_blen_coefficient"]) * hardware.blen
                    + int(config["gemm_writeout_fixed_cycles"])
                )
            elif gemv:
                cycles = (
                    hardware.mlen
                    + int(config["gemm_compute_blen_coefficient"]) * hardware.blen
                    + int(config["gemm_compute_fixed_cycles"])
                )
            else:
                cycles = (
                    int(config["gemm_compute_blen_coefficient"]) * hardware.blen
                    + int(config["gemm_compute_fixed_cycles"])
                )
            implemented = bool(config["gemv_implemented"]) if gemv else True
            if broadcast and not bool(config["broadcast_implemented"]):
                implemented = False
            return self._fixed(
                cycles,
                "structural_extrapolation" if implemented else "unsupported_rtl",
                supported=implemented,
                in_domain=False,
            )

        if precision.weight.family != "mxfp":
            raise ValueError(f"unsupported MatrixMachine family {precision.weight.family!r}")
        config = self.data["matrix"]["mxfp"]
        broadcast_supported = bool(config["broadcast_implemented"])
        supported = not broadcast or broadcast_supported
        operand = FpFormat.parse(config["measured_operand_format"])
        internal = FpFormat.parse(config["measured_internal_format"])
        profile_measured = precision.weight.fp_format == operand and precision.matrix_internal_fp == internal

        if writeout and gemv:
            cycles = int(config["gemv_writeout_busy_cycles"])
            ready = int(config["gemv_writeout_ready_cycles"])
            return self._split(
                cycles,
                ready,
                cycles,
                "structural_extrapolation" if supported else "unsupported_rtl",
                supported=supported,
                in_domain=False,
            )
        if writeout:
            busy = (
                int(config["gemm_writeout_busy_blen_coefficient"]) * hardware.blen
                + int(config["gemm_writeout_busy_overhead_coefficient"])
                * int(config["systolic_processing_overhead_cycles"])
                + int(config["gemm_writeout_busy_fixed_cycles"])
            )
            ready = (
                int(config["gemm_writeout_ready_blen_coefficient"]) * hardware.blen
                + int(config["gemm_writeout_ready_overhead_coefficient"])
                * int(config["systolic_processing_overhead_cycles"])
                + int(config["gemm_writeout_ready_fixed_cycles"])
            )
            row_supported = bool(config["gemm_writeout_row_writeback_supported"])
            rtl_supported = bool(config["gemm_writeout_rtl_supported"]) and supported
            return OpcodeTimingEstimate(
                resource_cycles=max(1, busy),
                result_ready_cycles=max(1, ready if row_supported else busy),
                initiation_interval_cycles=max(1, busy),
                calibration_status=(
                    "structural_extrapolation" if rtl_supported else "unsupported_rtl"
                ),
                rtl_supported=rtl_supported,
                calibration_in_domain=False,
            )

        cycles = (
            int(config["gemv_load_mlen_coefficient"]) * hardware.mlen
            + int(config["gemv_load_fixed_cycles"])
            if gemv
            else int(config["gemm_load_blen_coefficient"]) * hardware.blen
            + int(config["gemm_load_fixed_cycles"])
        )
        measured_shapes = {
            (int(point["mlen"]), int(point["blen"])) for point in config["measured_shapes"]
        }
        directly_measured = (
            opcode == "M_MM"
            and profile_measured
            and (hardware.mlen, hardware.blen) in measured_shapes
        )
        in_domain = (
            supported
            and not gemv
            and profile_measured
            and hardware.mlen <= 64
            and hardware.blen <= 16
        )
        return self._split(
            cycles,
            cycles,
            cycles,
            (
                "unsupported_rtl"
                if not supported
                else "full_machine_measured"
                if directly_measured
                else "structural_extrapolation"
            ),
            supported=supported,
            in_domain=in_domain,
        )

    def _nonmatrix_estimate(
        self,
        opcode: str,
        hardware: TimingHardware,
        precision: ComputePrecisionConfig,
    ) -> OpcodeTimingEstimate:
        vector = self.data["vector"]
        scalar = self.data["scalar"]
        vector_format = precision.vector_internal_fp
        scalar_format = precision.scalar_fp

        vector_point_measured = any(
            int(point["vlen"]) == hardware.vlen
            and FpFormat.parse(point) == vector_format
            for point in vector["measured_points"]
        )
        vector_status = (
            "full_machine_measured" if vector_point_measured else "structural_extrapolation"
        )

        vector_fixed = {
            "V_ADD_VV": "add_vv_cycles",
            "V_ADD_VF": "add_vf_cycles",
            "V_SUB_VV": "sub_vv_cycles",
            "V_SUB_VF": "sub_vf_cycles",
            "V_MUL_VV": "mul_vv_cycles",
            "V_MUL_VF": "mul_vf_cycles",
            "V_EXP_V": "exp_cycles",
            "V_RECI_V": "reciprocal_cycles",
        }
        if opcode in vector_fixed:
            cycles = int(vector[vector_fixed[opcode]])
            return self._split(
                cycles,
                cycles,
                int(vector["initiation_interval_cycles"]),
                vector_status,
                in_domain=vector_point_measured,
            )
        if opcode == "V_RED_SUM":
            cycles = int(vector["reduce_sum_base_cycles"]) + int(
                vector["reduce_sum_per_level_cycles"]
            ) * _ceil_log2(hardware.vlen + 1)
            return self._fixed(cycles, vector_status, in_domain=vector_point_measured)
        if opcode == "V_RED_MAX":
            cycles = int(vector["reduce_max_base_cycles"]) + int(
                vector["reduce_max_per_level_cycles"]
            ) * _ceil_log2(hardware.vlen + 1)
            return self._fixed(cycles, vector_status, in_domain=vector_point_measured)
        if opcode == "V_SHIFT_V":
            implemented = bool(vector["shift_implemented"])
            return self._fixed(
                int(vector["shift_conservative_cycles"]),
                "structural_extrapolation" if implemented else "unsupported_rtl",
                supported=implemented,
                in_domain=False,
            )

        scalar_format_measured = any(
            FpFormat.parse(point) == scalar_format for point in scalar["measured_points"]
        )
        scalar_point_measured = any(
            int(point["vlen"]) == hardware.vlen
            and FpFormat.parse(point) == scalar_format
            for point in scalar["measured_points"]
        )
        scalar_status = (
            "full_machine_measured" if scalar_format_measured else "structural_extrapolation"
        )
        scalar_pairs = {
            "S_ADD_FP": ("fp_add_ready_cycles", "fp_add_done_cycles"),
            "S_SUB_FP": ("fp_sub_ready_cycles", "fp_sub_done_cycles"),
            "S_MUL_FP": ("fp_mul_ready_cycles", "fp_mul_done_cycles"),
            "S_EXP_FP": ("fp_exp_ready_cycles", "fp_exp_done_cycles"),
            "S_RECI_FP": ("fp_reciprocal_ready_cycles", "fp_reciprocal_done_cycles"),
            "S_SQRT_FP": ("fp_sqrt_ready_cycles", "fp_sqrt_done_cycles"),
        }
        if opcode in scalar_pairs:
            ready_name, done_name = scalar_pairs[opcode]
            return self._split(
                int(scalar[done_name]),
                int(scalar[ready_name]),
                int(scalar[done_name]),
                scalar_status,
                in_domain=scalar_format_measured,
            )
        if opcode == "S_MAP_V_FP":
            ready = hardware.vlen + int(scalar["map_vector_ready_fixed_cycles"])
            done = hardware.vlen + int(scalar["map_vector_done_fixed_cycles"])
            return self._split(
                done,
                ready,
                done,
                "full_machine_measured" if scalar_point_measured else "structural_extrapolation",
                in_domain=scalar_point_measured,
            )
        if opcode == "S_MAX_FP":
            implemented = bool(scalar["fp_max_implemented"])
            return self._fixed(
                int(scalar["fp_max_conservative_cycles"]),
                "structural_extrapolation" if implemented else "unsupported_rtl",
                supported=implemented,
                in_domain=False,
            )
        if opcode in {"S_LD_FP", "S_ST_FP"}:
            return self._fixed(
                int(scalar["sram_cycles"]),
                "structural_extrapolation",
                supported=True,
                in_domain=False,
            )
        if opcode in SCALAR_INT_OPS:
            return self._fixed(
                int(scalar["int_basic_cycles"]),
                "structural_extrapolation",
                supported=True,
                in_domain=False,
            )
        if opcode in CONTROL_OPS:
            return self._fixed(
                1,
                "structural_extrapolation",
                supported=True,
                in_domain=True,
            )
        raise ValueError(f"RTL opcode timing model has no semantics for opcode {opcode!r}")


@dataclass(frozen=True)
class ComputeWork:
    resource_work_cycles: int
    latency_ns: float
    category_cycles: Mapping[str, int]
    category_latency_ns: Mapping[str, float]
    opcode_cycles: Mapping[str, int]
    validation: Mapping[str, Any]


def aggregate_compute_work(
    counts: Mapping[str, int],
    *,
    calibration: RtlOpcodeTimingCalibration,
    hardware: TimingHardware,
    precision: ComputePrecisionConfig,
    clock_period_ps: int,
    opcode_category,
) -> ComputeWork:
    total = 0
    category_cycles: Counter[str] = Counter()
    opcode_cycles: dict[str, int] = {}
    status_counts: Counter[str] = Counter()
    status_cycles: Counter[str] = Counter()
    unsupported_counts: Counter[str] = Counter()
    unsupported_cycles: Counter[str] = Counter()
    out_of_domain_counts: Counter[str] = Counter()
    out_of_domain_cycles: Counter[str] = Counter()
    total_opcodes = 0

    for opcode, raw_count in counts.items():
        count = int(raw_count)
        if opcode in MEMORY_OPS or count == 0:
            continue
        timing = calibration.estimate(opcode, hardware, precision)
        if timing is None:
            continue
        cycles = count * timing.resource_cycles
        total += cycles
        total_opcodes += count
        opcode_cycles[opcode] = cycles
        category_cycles[opcode_category(opcode)] += cycles
        status_counts[timing.calibration_status] += count
        status_cycles[timing.calibration_status] += cycles
        if not timing.rtl_supported:
            unsupported_counts[opcode] += count
            unsupported_cycles[opcode] += cycles
        elif not timing.calibration_in_domain:
            out_of_domain_counts[opcode] += count
            out_of_domain_cycles[opcode] += cycles

    if unsupported_counts:
        validation_status = "unsupported_opcodes"
    elif out_of_domain_counts:
        validation_status = "out_of_domain"
    else:
        validation_status = "validated"
    cycle_to_ns = clock_period_ps / 1000.0
    validation = {
        "status": validation_status,
        "total_opcodes": total_opcodes,
        "status_opcode_counts": dict(sorted(status_counts.items())),
        "status_resource_cycles": dict(sorted(status_cycles.items())),
        "unsupported_opcode_counts": dict(sorted(unsupported_counts.items())),
        "unsupported_resource_cycles": dict(sorted(unsupported_cycles.items())),
        "out_of_domain_opcode_counts": dict(sorted(out_of_domain_counts.items())),
        "out_of_domain_resource_cycles": dict(sorted(out_of_domain_cycles.items())),
        "calibration_in_domain": not unsupported_counts and not out_of_domain_counts,
    }
    return ComputeWork(
        resource_work_cycles=total,
        latency_ns=total * cycle_to_ns,
        category_cycles=dict(sorted(category_cycles.items())),
        category_latency_ns={
            name: cycles * cycle_to_ns for name, cycles in sorted(category_cycles.items())
        },
        opcode_cycles=dict(sorted(opcode_cycles.items())),
        validation=validation,
    )


__all__ = [
    "ComputeFormat",
    "ComputePrecisionConfig",
    "ComputeWork",
    "DEFAULT_RTL_TIMING_CALIBRATION",
    "FpFormat",
    "OpcodeTimingEstimate",
    "RtlOpcodeTimingCalibration",
    "TimingHardware",
    "aggregate_compute_work",
]
