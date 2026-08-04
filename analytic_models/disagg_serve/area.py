"""Decode-chip area bridge for proxy, full-chip, system, and geometry estimates.

``calibrated`` uses the precision-aware full-chip decomposition in the top-level
``area`` package. ``proxy`` is retained only as an explicitly labelled
MatrixMachine-only fallback for legacy sensitivity studies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

_HERE = Path(__file__).resolve().parent

REF_MULTIPLIERS = 4096
REF_MM2 = 0.237
MM2_PER_MULTIPLIER = REF_MM2 / REF_MULTIPLIERS

DEFAULT_MLEN_CANDIDATES = (64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_BLEN_CANDIDATES = (4, 8, 16, 32, 64)


def _area_package():
    package_root = str(_HERE.parent)
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    import area

    return area


def _is_mxfp_label(label: Any) -> bool:
    token = str(label).upper()
    return "E" in token and "M" in token


def _width_token(element_bits: int, label: str) -> str:
    """Convert a serving precision label to an area-model width token."""

    if _is_mxfp_label(label):
        token = str(label).upper()
        exponent = token[token.index("E") :]
        if not exponent.startswith("E") or "M" not in exponent:
            raise ValueError(f"invalid MXFP label: {label}")
        mantissa_end = exponent.index("M") + 1
        while mantissa_end < len(exponent) and exponent[mantissa_end].isdigit():
            mantissa_end += 1
        return f"MXFP_{exponent[:mantissa_end]}"
    return f"MXINT{int(element_bits)}"


def _hardware_value(hardware: Any, name: str, default: Any) -> Any:
    if isinstance(hardware, Mapping):
        return hardware.get(name, default)
    return getattr(hardware, name, default)


def _activation_token(precision: Mapping[str, Any]) -> str:
    labels = (
        (int(precision["attn_elem"]), str(precision["attn_label"])),
        (int(precision["ffn_elem"]), str(precision["ffn_label"])),
        (int(precision["kv_elem"]), str(precision["kv_label"])),
    )
    mxfp = [item for item in labels if _is_mxfp_label(item[1])]
    if not mxfp:
        return _width_token(min(8, max(int(precision["m_bits"]), 2)), "MXINT")
    if len(mxfp) != len(labels):
        raise ValueError(
            "mixed MXINT/MXFP serving profiles need a dual-family datapath model"
        )
    element_bits, label = max(mxfp, key=lambda item: item[0])
    return _width_token(element_bits, label)


def _area_config(
    mlen: int,
    blen: int,
    vlen: int,
    precision: Mapping[str, Any],
    *,
    hlen: int | None = None,
    fp_setting: str = "FP_E5M6",
    matrix_sram_size: int = 4096,
    vector_sram_size: int = 4096,
    int_sram_depth: int = 32,
    fp_sram_depth: int = 512,
    int_data_width: int = 32,
    hbm_m_prefetch_amount: int = 16,
    hbm_v_prefetch_amount: int = 16,
    hbm_v_writeback_amount: int = 16,
    reduction_segments: int = 1,
    scalar_fp_issue_pipeline: bool = False,
    enable_loop_address_generator: bool = False,
) -> dict[str, Any]:
    weight_element = max(int(precision["attn_elem"]), int(precision["ffn_elem"]))
    weight_label = (
        precision["attn_label"]
        if int(precision["attn_elem"]) >= int(precision["ffn_elem"])
        else precision["ffn_label"]
    )
    return {
        "MLEN": int(mlen),
        "BLEN": int(blen),
        "VLEN": int(vlen),
        "HLEN": int(hlen if hlen is not None else blen),
        "BLOCK_DIM": int(blen),
        "WEIGHT_WIDTH": _width_token(weight_element, str(weight_label)),
        "KV_WIDTH": _width_token(int(precision["kv_elem"]), str(precision["kv_label"])),
        "ACT_WIDTH": _activation_token(precision),
        "FP_SETTING": fp_setting,
        "INT_DATA_WIDTH": int(int_data_width),
        "MATRIX_SRAM_DEPTH": int(matrix_sram_size),
        "VECTOR_SRAM_DEPTH": int(vector_sram_size),
        "INT_SRAM_DEPTH": int(int_sram_depth),
        "FP_SRAM_DEPTH": int(fp_sram_depth),
        "HBM_M_Prefetch_Amount": int(hbm_m_prefetch_amount),
        "HBM_V_Prefetch_Amount": int(hbm_v_prefetch_amount),
        "HBM_V_Writeback_Amount": int(hbm_v_writeback_amount),
        "REDUCTION_SEGMENTS": int(reduction_segments),
        "SCALAR_FP_ISSUE_PIPELINE": bool(scalar_fp_issue_pipeline),
        "ENABLE_LOOP_ADDRESS_GENERATOR": bool(enable_loop_address_generator),
    }


def _config_from_hardware(
    hardware: Any, precision: Mapping[str, Any]
) -> dict[str, Any]:
    mlen = int(_hardware_value(hardware, "MLEN", 0))
    blen = int(_hardware_value(hardware, "BLEN", 0))
    vlen = int(_hardware_value(hardware, "VLEN", mlen))
    return _area_config(
        mlen,
        blen,
        vlen,
        precision,
        hlen=int(_hardware_value(hardware, "HLEN", blen)),
        fp_setting=str(_hardware_value(hardware, "FP_SETTING", "FP_E5M6")),
        matrix_sram_size=int(
            _hardware_value(
                hardware,
                "MATRIX_SRAM_DEPTH",
                _hardware_value(hardware, "MATRIX_SRAM_SIZE", 4096),
            )
        ),
        vector_sram_size=int(
            _hardware_value(
                hardware,
                "VECTOR_SRAM_DEPTH",
                _hardware_value(hardware, "VECTOR_SRAM_SIZE", 4096),
            )
        ),
        int_sram_depth=int(_hardware_value(hardware, "INT_SRAM_DEPTH", 32)),
        fp_sram_depth=int(_hardware_value(hardware, "FP_SRAM_DEPTH", 512)),
        int_data_width=int(_hardware_value(hardware, "INT_DATA_WIDTH", 32)),
        hbm_m_prefetch_amount=int(
            _hardware_value(hardware, "HBM_M_Prefetch_Amount", 16)
        ),
        hbm_v_prefetch_amount=int(
            _hardware_value(hardware, "HBM_V_Prefetch_Amount", 16)
        ),
        hbm_v_writeback_amount=int(
            _hardware_value(hardware, "HBM_V_Writeback_Amount", 16)
        ),
        reduction_segments=int(
            _hardware_value(hardware, "REDUCTION_SEGMENTS", 1)
        ),
        scalar_fp_issue_pipeline=bool(
            _hardware_value(hardware, "SCALAR_FP_ISSUE_PIPELINE", False)
        ),
        enable_loop_address_generator=bool(
            _hardware_value(
                hardware,
                "ENABLE_LOOP_ADDRESS_GENERATOR",
                _hardware_value(hardware, "loop_address_generator", False),
            )
        ),
    )


def proxy_mm2(mlen: int, blen: int) -> float:
    """Return the legacy MatrixMachine-only area proxy in mm^2."""

    return int(mlen) * int(blen) * MM2_PER_MULTIPLIER


def proxy_area(mlen: int, blen: int) -> dict[str, Any]:
    area = proxy_mm2(mlen, blen)
    return {
        "area_mm2": area,
        "area_model": "matrix_multiplier_proxy_fallback",
        "breakdown_mm2": {"MatrixMachineProxy": area},
        "evidence_tier": "declared_proxy",
        "full_chip": False,
    }


def calibrated_area(
    mlen: int,
    blen: int,
    vlen: int,
    precision: Mapping[str, Any],
    **configuration: Any,
) -> dict[str, Any]:
    """Return the precision-aware full-chip estimate and block breakdown."""

    area = _area_package()
    config = _area_config(mlen, blen, vlen, precision, **configuration)
    return area.estimate_area(config)


def calibrated_mm2(
    mlen: int,
    blen: int,
    vlen: int,
    prec: Mapping[str, Any],
    **configuration: Any,
) -> float:
    """Return precision-aware full-chip area in mm^2."""

    return float(calibrated_area(mlen, blen, vlen, prec, **configuration)["area"]) / 1e6


def _millimetre_result(result: Mapping[str, Any]) -> dict[str, Any]:
    converted = dict(result)
    converted["area_mm2"] = float(result["area"]) / 1e6
    converted["breakdown_mm2"] = {
        name: float(value) / 1e6 for name, value in result["breakdown"].items()
    }
    return converted


def area_mm2(
    model: str,
    hw_cfg: Any,
    prec: Mapping[str, Any],
    *,
    return_breakdown: bool = False,
) -> float | dict[str, Any]:
    """Estimate chip area, optionally returning the complete evidence ledger."""

    if model == "proxy":
        result = proxy_area(
            int(_hardware_value(hw_cfg, "MLEN", 0)),
            int(_hardware_value(hw_cfg, "BLEN", 0)),
        )
        return result if return_breakdown else float(result["area_mm2"])
    if model != "calibrated":
        raise ValueError("area model must be 'calibrated' or 'proxy'")
    area = _area_package()
    result = area.estimate_area(_config_from_hardware(hw_cfg, prec))
    converted = _millimetre_result(result)
    return converted if return_breakdown else float(converted["area_mm2"])


def area_breakdown_mm2(
    model: str, hw_cfg: Any, prec: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a per-block mm^2 breakdown for reporting and provenance."""

    result = area_mm2(model, hw_cfg, prec, return_breakdown=True)
    if not isinstance(result, dict):
        raise TypeError("area breakdown dispatch did not return a ledger")
    return result


def system_area_mm2(
    hw_cfg: Any,
    prec: Mapping[str, Any],
    *,
    chip_count: int,
    ports_per_chip: int = 0,
    link_bandwidth_gbps: float = 900.0 * 8.0,
) -> dict[str, Any]:
    """Return aggregate chip plus C2C-PHY silicon area in mm^2."""

    area = _area_package()
    result = area.estimate_system_area(
        _config_from_hardware(hw_cfg, prec),
        chip_count=chip_count,
        ports_per_chip=ports_per_chip,
        link_bandwidth_gbps=link_bandwidth_gbps,
    )
    converted = dict(result)
    converted["area_mm2"] = float(result["area"]) / 1e6
    converted["chip_area_mm2"] = float(result["chip_area"]) / 1e6
    converted["link_phy_area_per_port_mm2"] = (
        float(result["link_phy_area_per_port"]) / 1e6
    )
    converted["breakdown_mm2"] = {
        name: float(value) / 1e6 for name, value in result["breakdown"].items()
    }
    return converted


def solve_area_budget(
    area_budget_mm2: float,
    hw_cfg: Any,
    prec: Mapping[str, Any],
    *,
    mlen_candidates: Iterable[int] = DEFAULT_MLEN_CANDIDATES,
    blen_candidates: Iterable[int] = DEFAULT_BLEN_CANDIDATES,
    vlen_candidates: Iterable[int] | None = None,
    require_vlen_equal_mlen: bool = True,
    hidden_size: int | None = None,
) -> dict[str, Any]:
    """Solve legal precision-dependent geometry at fixed full-chip area.

    This is the production integration point for ``disagg_decode``. It returns
    the chosen MLEN/BLEN/VLEN, full-chip breakdown, utilization, and evidence;
    callers can apply those three fields to their hardware configuration.
    """

    budget = float(area_budget_mm2)
    if budget <= 0.0:
        raise ValueError("area_budget_mm2 must be positive")
    area = _area_package()
    result = area.solve_geometry_for_area(
        _config_from_hardware(hw_cfg, prec),
        budget * 1e6,
        mlen_candidates=mlen_candidates,
        blen_candidates=blen_candidates,
        vlen_candidates=vlen_candidates,
        require_vlen_equal_mlen=require_vlen_equal_mlen,
        hidden_size=hidden_size,
    )
    converted = dict(result)
    converted["area_mm2"] = float(result["area_um2"]) / 1e6
    converted["area_budget_mm2"] = budget
    converted["breakdown_mm2"] = {
        name: float(value) / 1e6 for name, value in result["breakdown"].items()
    }
    return converted
