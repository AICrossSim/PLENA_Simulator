"""Reconstruct PLENA static power for steady-state pipeline accounting."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from analytic_models.area_new import estimate_area


HLEN = 128
FP_CONSTANT_NUM = 10
COMPACT_STATS_LANE_TIERS = (4, 8, 16, 32, 64)
NVLINK_ENDPOINT_AREA_MM2_PER_PORT = 24.7


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _resolve_recorded_path(path: str, *, repo_root: Path) -> Path:
    recorded = Path(path)
    if recorded.is_file():
        return recorded
    parts = recorded.parts
    for anchor in ("Workspace", "analytic_models"):
        if anchor in parts:
            candidate = repo_root.joinpath(*parts[parts.index(anchor) :])
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"cannot resolve recorded artifact path: {path}")


def _parse_mx_precision(spec: Any, *, default_scale_width: int) -> dict[str, int | str]:
    scale_width = default_scale_width
    if isinstance(spec, str):
        text = spec.upper().replace("_", "")
        if text.startswith("MXINT"):
            return {
                "family": "mxint",
                "width": int(text.removeprefix("MXINT")),
                "scale_width": scale_width,
            }
        text = text.removeprefix("MXFP")
        if text.startswith("E") and "M" in text:
            exp_text, mant_text = text[1:].split("M", 1)
            exp = int(exp_text)
            mant = int(mant_text)
            return {
                "family": "mxfp",
                "exp": exp,
                "mant": mant,
                "width": 1 + exp + mant,
                "scale_width": scale_width,
            }
    if isinstance(spec, Mapping):
        kind = str(spec.get("kind", spec.get("type", ""))).upper().replace("_", "")
        scale_width = int(spec.get("scale_width", spec.get("scale", default_scale_width)))
        if "MXINT" in kind or "width" in spec or "bits" in spec:
            width = (
                int(kind.removeprefix("MXINT"))
                if kind.startswith("MXINT") and kind != "MXINT"
                else int(spec.get("width", spec.get("bits")))
            )
            return {"family": "mxint", "width": width, "scale_width": scale_width}
        if "MXFP" in kind or {"exp", "mant"} <= set(spec):
            exp = int(spec["exp"])
            mant = int(spec["mant"])
            return {
                "family": "mxfp",
                "exp": exp,
                "mant": mant,
                "width": 1 + exp + mant,
                "scale_width": scale_width,
            }
    raise ValueError(f"unsupported MX precision specification: {spec!r}")


def _compact_stats_lanes(*, vlen: int, num_attention_heads: int) -> int:
    required = min(num_attention_heads, vlen // HLEN)
    return next((tier for tier in COMPACT_STATS_LANE_TIERS if tier >= required), COMPACT_STATS_LANE_TIERS[-1])


@dataclass
class PrefillIdlePowerEstimator:
    """Recover static power from the exact area configuration of a DSE campaign."""

    model: Mapping[str, Any]
    profiles: Mapping[str, Mapping[str, Any]]
    hbm_capacity_bytes: int
    mx_scale_width: int
    vector_scalar_schedule: str
    address_generation_mode: str
    logic_leakage_mw_per_um2: float
    sram_background_w_per_gb: float
    hbm_background_mw_per_gb: float
    _area_cache: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    @property
    def cached_area_configuration_count(self) -> int:
        return len(self._area_cache)

    @classmethod
    def from_campaign(cls, campaign_dir: Path, *, repo_root: Path) -> PrefillIdlePowerEstimator:
        summary = _read_json(campaign_dir / "run_summary.json")
        model = _read_json(_resolve_recorded_path(str(summary["model_config"]), repo_root=repo_root))
        accuracy = _read_json(
            _resolve_recorded_path(str(summary["accuracy_constraints"]), repo_root=repo_root)
        )
        profiles = {
            str(profile["name"]): dict(profile)
            for profile in accuracy["precision_profiles"]
        }
        logic = _read_json(repo_root / "analytic_models/power/calibration/logic_energy_v2.json")
        sram = _read_json(
            _resolve_recorded_path(
                str(summary["sram_background_energy_artifact"]), repo_root=repo_root
            )
        )
        hbm = _read_json(
            _resolve_recorded_path(
                str(summary["external_memory_energy_artifact"]), repo_root=repo_root
            )
        )
        return cls(
            model=model,
            profiles=profiles,
            hbm_capacity_bytes=int(summary["hbm_capacity_bytes"]),
            mx_scale_width=int(summary["mx_scale_width"]),
            vector_scalar_schedule=str(summary["vector_scalar_schedule"]),
            address_generation_mode=str(summary["address_generation_mode"]),
            logic_leakage_mw_per_um2=float(logic["logic_leakage_mw_per_um2"]),
            sram_background_w_per_gb=float(sram["selected_background_power_w_per_gb"]),
            hbm_background_mw_per_gb=float(
                hbm["coefficients"]["background_power_mw_per_gb"]["p50"]
            ),
        )

    def _area_inputs(self, row: Mapping[str, Any]) -> dict[str, Any]:
        profile = self.profiles[str(row["precision_profile"])]
        mlen = int(row["MLEN"])
        vlen = int(row["VLEN"])
        blen = int(row["BLEN"])
        row_lanes = int(row["softmax_row_lanes"])
        num_attention_heads = int(self.model["num_attention_heads"])
        num_kv_heads = int(self.model["num_key_value_heads"])
        logical_broadcast = num_attention_heads // num_kv_heads
        physical_broadcast = min(logical_broadcast, mlen // HLEN)

        act = _parse_mx_precision(profile["ACT_WIDTH"], default_scale_width=self.mx_scale_width)
        kv = _parse_mx_precision(profile["KV_WIDTH"], default_scale_width=self.mx_scale_width)
        weight = _parse_mx_precision(profile["WEIGHT_WIDTH"], default_scale_width=self.mx_scale_width)
        fp = profile["FP_SETTING"]
        scale_width = max(int(act["scale_width"]), int(kv["scale_width"]), int(weight["scale_width"]))
        matrix_sram_depth = int(row.get("MATRIX_SRAM_SIZE") or row["matrix_sram_depth"])
        config = {
            "MLEN": mlen,
            "BLEN": blen,
            "VLEN": vlen,
            "MATRIX_SRAM_DEPTH": matrix_sram_depth,
            "VECTOR_SRAM_DEPTH": 2 * int(self.model["head_dim"]) + math.ceil(int(self.model["hidden_size"]) / vlen),
            "INT_SRAM_DEPTH": 32,
            "FP_SRAM_DEPTH": FP_CONSTANT_NUM,
            "COMPACT_STATS_LANES": _compact_stats_lanes(
                vlen=vlen,
                num_attention_heads=num_attention_heads,
            ),
            "SOFTMAX_ROW_LANES": row_lanes,
            "VECTOR_SRAM_ROW_BANKS": row_lanes,
            "SOFTMAX_STATE_BANK_ENTRIES": physical_broadcast * mlen,
            "HLEN": HLEN,
            "INT_DATA_WIDTH": int(row["INT_DATA_WIDTH"]),
            "ACT_ELEMENT_WIDTH": int(act["width"]),
            "KV_ELEMENT_WIDTH": int(kv["width"]),
            "FP_EXP_WIDTH": int(fp["exp"]),
            "FP_MANT_WIDTH": int(fp["mant"]),
            "WT_MX_EXP_WIDTH": int(weight.get("exp", 2)),
            "WT_MX_MANT_WIDTH": int(weight.get("mant", 1)),
            "WEIGHT_ELEMENT_BITS": int(weight["width"]),
            "MX_SCALE_WIDTH": scale_width,
            "BLOCK_DIM": blen,
            "HBM_ELE_WIDTH": mlen,
            "HBM_SCALE_WIDTH": (mlen // blen) * scale_width,
            "HBM_M_Prefetch_Amount": mlen,
            "HBM_V_Prefetch_Amount": blen,
            "HBM_V_Writeback_Amount": blen,
            "ACT_WIDTH": profile["ACT_WIDTH"],
            "KV_WIDTH": profile["KV_WIDTH"],
            "WEIGHT_WIDTH": profile["WEIGHT_WIDTH"],
            "FP_SETTING": f"FP_E{int(fp['exp'])}M{int(fp['mant'])}",
            "SRAM_PORT_MODEL": "ideal-dual-port",
            "vector_scalar_area_version": self.vector_scalar_schedule,
            "address_generation_mode": self.address_generation_mode,
        }
        return config

    def estimate(self, row: Mapping[str, Any]) -> dict[str, Any]:
        area_inputs = self._area_inputs(row)
        area_key = json.dumps(area_inputs, sort_keys=True, separators=(",", ":"))
        area = self._area_cache.get(area_key)
        if area is None:
            area = estimate_area(area_inputs)
            self._area_cache[area_key] = area
        chip_count = int(row["physical_chip_count"])
        port_count = int(row["nvlink_port_count"])
        core_area_um2 = float(area["area"])
        core_area_mm2 = core_area_um2 / 1.0e6
        reconstructed_area_mm2 = chip_count * (
            core_area_mm2 + port_count * NVLINK_ENDPOINT_AREA_MM2_PER_PORT
        )
        recorded_area_mm2 = float(row["total_silicon_area_mm2"])
        if not math.isclose(reconstructed_area_mm2, recorded_area_mm2, rel_tol=1.0e-10, abs_tol=1.0e-7):
            raise ValueError(
                "post-hoc area reconstruction does not match the DSE record: "
                f"{reconstructed_area_mm2:.9f} != {recorded_area_mm2:.9f} mm2"
            )

        sram = area["sram"]
        sram_area_um2 = float(sram["area"])
        tiling = sram["area_sram_macro_tiling"]
        sram_capacity_gb_per_chip = math.fsum(
            float(detail.get("covered_bits", 0.0)) / 8.0e9
            for detail in tiling.values()
        )
        logic_leakage_power_w = (
            (core_area_um2 - sram_area_um2)
            * self.logic_leakage_mw_per_um2
            / 1000.0
            * chip_count
        )
        sram_background_power_w = (
            sram_capacity_gb_per_chip
            * self.sram_background_w_per_gb
            * chip_count
        )
        hbm_background_power_w = (
            self.hbm_capacity_bytes
            / 1.0e9
            * self.hbm_background_mw_per_gb
            / 1000.0
        )
        idle_power_w = math.fsum(
            (logic_leakage_power_w, sram_background_power_w, hbm_background_power_w)
        )
        return {
            "idle_power_w": idle_power_w,
            "logic_leakage_power_w": logic_leakage_power_w,
            "sram_background_power_w": sram_background_power_w,
            "hbm_background_power_w": hbm_background_power_w,
            "sram_allocated_capacity_gb": sram_capacity_gb_per_chip * chip_count,
            "reconstructed_core_area_mm2": core_area_mm2,
            "reconstructed_total_area_mm2": reconstructed_area_mm2,
            "recorded_total_area_mm2": recorded_area_mm2,
            "area_reconstruction_error_mm2": reconstructed_area_mm2 - recorded_area_mm2,
            "area_reconstruction_validated": True,
            "idle_power_scope": "logic_leakage+sram_background+hbm_background",
        }
