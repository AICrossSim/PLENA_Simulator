#!/usr/bin/env python3
"""Audit rtl-v6 banking area and summarize the observed VLEN x R design grid."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytic_models.area_new import estimate_area

PROFILE_RE = re.compile(
    r"w_(?P<weight>mxint\d+|mxfp_e\d+m\d+)__"
    r"act_(?P<act>mxint\d+|mxfp_e\d+m\d+)__"
    r"kv_(?P<kv>mxint\d+|mxfp_e\d+m\d+)__"
    r"fp_e(?P<fp_exp>\d+)m(?P<fp_mant>\d+)",
    re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return json.load(handle)
    return json.loads(path.read_text())


def _load_trial_records(run_dir: Path | None) -> dict[int, dict[str, Any]]:
    if run_dir is None:
        return {}
    records: dict[int, dict[str, Any]] = {}
    paths = sorted(run_dir.glob("trial_*/trial_record.json"))
    paths.extend(sorted(run_dir.glob("trial_*/trial_record.json.gz")))
    for path in paths:
        try:
            record = _read_json(path)
            records[int(record["trial"])] = record
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
    return records


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, "", "None"):
        return default
    return float(value)


def _integer(row: dict[str, Any], key: str, default: int = 0) -> int:
    return int(_number(row, key, float(default)))


def _precision(row: dict[str, Any]) -> dict[str, str]:
    match = PROFILE_RE.fullmatch(str(row.get("precision_profile", "")))
    if not match:
        raise ValueError(
            "cannot reconstruct precision from profile "
            f"{row.get('precision_profile')!r}"
        )
    values = match.groupdict()

    def token(name: str) -> str:
        return values[name].upper().replace("MXFP_", "MXFP_")

    return {
        "WEIGHT_WIDTH": token("weight"),
        "ACT_WIDTH": token("act"),
        "KV_WIDTH": token("kv"),
        "FP_SETTING": f"FP_E{values['fp_exp']}M{values['fp_mant']}",
    }


def _compact_lane_tier(vlen: int, hlen: int, num_heads: int = 64) -> int:
    required = max(1, min(num_heads, vlen // max(hlen, 1)))
    return next(tier for tier in (4, 8, 16, 32, 64) if tier >= required)


def _row_lanes(row: dict[str, Any], default: int) -> tuple[int, str]:
    for key in ("softmax_row_lanes", "SOFTMAX_ROW_LANES"):
        if row.get(key) not in (None, "", "None"):
            return int(float(row[key])), key
    return default, "assumed-r1-missing-historical-field" if default == 1 else "cli-assumption"


def _area_config(row: dict[str, Any], row_lanes: int) -> dict[str, Any]:
    precision = _precision(row)
    mlen = _integer(row, "MLEN")
    vlen = _integer(row, "VLEN", mlen)
    blen = _integer(row, "BLEN")
    hlen = _integer(row, "HLEN", 128)
    state_entries = _integer(
        row,
        "softmax_state_entries_required",
        _integer(row, "BROADCAST_AMOUNT", 8) * mlen,
    )
    return {
        **precision,
        "MLEN": mlen,
        "VLEN": vlen,
        "BLEN": blen,
        "HLEN": hlen,
        "MATRIX_SRAM_DEPTH": _integer(
            row,
            "MATRIX_SRAM_SIZE",
            _integer(row, "MATRIX_SRAM_TILES", 2) * mlen,
        ),
        "VECTOR_SRAM_DEPTH": _integer(
            row, "VECTOR_SRAM_SIZE", 2 * hlen + math.ceil(4096 / vlen)
        ),
        "INT_SRAM_DEPTH": _integer(row, "INT_SRAM_DEPTH", 32),
        "FP_SRAM_DEPTH": _integer(row, "FP_SRAM_DEPTH", 10),
        "INT_DATA_WIDTH": _integer(row, "INT_DATA_WIDTH", 32),
        "COMPACT_STATS_LANES": _integer(
            row, "COMPACT_STATS_LANES", _compact_lane_tier(vlen, hlen)
        ),
        "SOFTMAX_ROW_LANES": row_lanes,
        "VECTOR_SRAM_ROW_BANKS": row_lanes,
        "SOFTMAX_STATE_BANK_ENTRIES": state_entries,
        "BLOCK_DIM": blen,
        "HBM_ELE_WIDTH": mlen,
        "HBM_SCALE_WIDTH": max(1, mlen // blen) * 8,
        "HBM_M_Prefetch_Amount": _integer(row, "HBM_M_Prefetch_Amount", mlen),
        "HBM_V_Prefetch_Amount": _integer(row, "HBM_V_Prefetch_Amount", blen),
        "HBM_V_Writeback_Amount": _integer(row, "HBM_V_Writeback_Amount", blen),
        "MX_SCALE_WIDTH": 8,
        "SRAM_PORT_MODEL": "ideal-dual-port",
        "vector_scalar_area_version": "rtl-v6",
        "address_generation_mode": str(
            row.get("address_generation_mode", "loop-agu-v1")
        ),
    }


def audit_row(row: dict[str, Any], *, assumed_row_lanes: int = 1) -> dict[str, Any]:
    lanes, lanes_source = _row_lanes(row, assumed_row_lanes)
    metrics = estimate_area(_area_config(row, lanes))
    sram = metrics["sram"]
    vector = metrics.get("vector_machine") or {}
    banking = sram["vector_sram_banking"]
    sram_breakdown = sram["area_sram_breakdown"]
    state_area = sum(
        float(sram_breakdown.get(name, 0.0))
        for name in (
            "SoftmaxStateBank",
            "SoftmaxStatisticBank",
            "SoftmaxFactorBank",
        )
    )
    logic_delta = float(vector.get("rtl_v6_delta_area", 0.0))
    bank_delta = float(banking["selected_banking_area_delta_um2"])
    incremental = logic_delta + bank_delta + state_area
    chip_count = _integer(row, "chip_count", 1)
    ports = _integer(row, "nvlink_port_count", 1)
    new_core_mm2 = float(metrics["area"]) / 1e6
    new_total_mm2 = chip_count * (new_core_mm2 + ports * 24.7)
    old_total_mm2 = _number(
        row, "total_silicon_area_mm2", _number(row, "area_mm2")
    )
    vlen = _integer(row, "VLEN")
    row_util = row.get("softmax_row_lane_utilization")
    return {
        "trial": row.get("trial"),
        "state": row.get("state"),
        "precision_profile": row.get("precision_profile"),
        "MLEN": _integer(row, "MLEN"),
        "VLEN": vlen,
        "BLEN": _integer(row, "BLEN"),
        "INT_DATA_WIDTH": _integer(row, "INT_DATA_WIDTH", 32),
        "chip_count": _integer(row, "chip_count", 1),
        "tp_degree": _integer(row, "tp_degree", 1),
        "dp_degree": _integer(row, "dp_degree", 1),
        "ep_degree": _integer(row, "ep_degree", 1),
        "nvlink_port_count": _integer(row, "nvlink_port_count", 1),
        "matrix_sram_config_id": row.get("matrix_sram_config_id"),
        "MATRIX_SRAM_TILES": _integer(row, "MATRIX_SRAM_TILES", 2),
        "matrix_sram_policy": row.get("matrix_sram_policy"),
        "COMPACT_STATS_LANES": _integer(
            row, "COMPACT_STATS_LANES", _compact_lane_tier(vlen, 128)
        ),
        "softmax_row_lanes": lanes,
        "softmax_row_lanes_source": lanes_source,
        "softmax_elements_per_cycle": lanes * vlen,
        "softmax_row_lane_utilization": row_util,
        "softmax_bank_utilization": row.get("softmax_bank_utilization"),
        "matrix_utilization_by_stage": row.get("matrix_utilization_by_stage"),
        "vector_utilization_by_stage": row.get("vector_utilization_by_stage"),
        "padding_cycles": row.get("padding_cycles"),
        "latency_ms": row.get("latency_ms"),
        "system_energy_nominal_mj": row.get("system_energy_nominal_mj"),
        "old_core_area_mm2": row.get("core_area_mm2"),
        "new_core_area_mm2": new_core_mm2,
        "old_total_silicon_area_mm2": old_total_mm2,
        "new_total_silicon_area_mm2": new_total_mm2,
        "total_area_delta_mm2": new_total_mm2 - old_total_mm2,
        "total_area_delta_pct": (
            (new_total_mm2 - old_total_mm2) / old_total_mm2 * 100.0
            if old_total_mm2
            else None
        ),
        "vector_sram_logical_bits": banking["logical_bits"],
        "vector_sram_bank_count": banking["physical_bank_count"],
        "vector_sram_macro_rounding_bits": banking["macro_rounding_overhead_bits"],
        "vector_sram_banking_area_delta_mm2": bank_delta / 1e6,
        "softmax_state_stat_factor_area_mm2": state_area / 1e6,
        "rtl_v6_logic_delta_area_mm2": logic_delta / 1e6,
        "softmax_incremental_area_mm2": incremental / 1e6,
        "softmax_area_efficiency_elements_per_cycle_per_mm2": (
            lanes * vlen / (incremental / 1e6) if incremental > 0 else None
        ),
        "area_calibration_status": vector.get("rtl_v6_delta_status"),
        "audit_fidelity": (
            "historical-row-lanes-assumed"
            if lanes_source.startswith("assumed")
            else "recorded-row-lanes-recomputed-current-area-model"
        ),
    }


MATCHED_R_SIGNATURE_FIELDS = (
    "precision_profile",
    "MLEN",
    "VLEN",
    "BLEN",
    "INT_DATA_WIDTH",
    "chip_count",
    "tp_degree",
    "dp_degree",
    "ep_degree",
    "nvlink_port_count",
    "matrix_sram_config_id",
    "MATRIX_SRAM_TILES",
    "matrix_sram_policy",
    "COMPACT_STATS_LANES",
)


def _matched_r_signature(row: dict[str, Any]) -> tuple[str, ...]:
    """Identify trials that differ only in the softmax row-lane tier.

    The signature is intentionally strict.  A missing field remains an empty
    token instead of being inferred, so old studies cannot accidentally pair
    different precision, topology, or SRAM configurations.
    """

    return tuple(str(row.get(field, "")) for field in MATCHED_R_SIGNATURE_FIELDS)


def _matched_r_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_matched_r_signature(row)].append(row)

    output: list[dict[str, Any]] = []
    for signature, members in groups.items():
        by_r = {int(row["softmax_row_lanes"]): row for row in members}
        baseline = by_r.get(1)
        if baseline is None or len(by_r) < 2:
            continue
        signature_hash = hashlib.sha256(
            json.dumps(signature, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        baseline_latency = _number(baseline, "latency_ms")
        baseline_energy = _number(baseline, "system_energy_nominal_mj")
        baseline_area = _number(baseline, "new_total_silicon_area_mm2")
        for lanes, row in sorted(by_r.items()):
            latency = _number(row, "latency_ms")
            energy = _number(row, "system_energy_nominal_mj")
            area = _number(row, "new_total_silicon_area_mm2")
            output.append(
                {
                    "matched_signature": signature_hash,
                    **dict(zip(MATCHED_R_SIGNATURE_FIELDS, signature, strict=True)),
                    "softmax_row_lanes": lanes,
                    "baseline_row_lanes": 1,
                    "latency_ms": latency,
                    "latency_delta_pct_vs_r1": (
                        100.0 * (latency - baseline_latency) / baseline_latency
                        if baseline_latency
                        else None
                    ),
                    "system_energy_nominal_mj": energy,
                    "energy_delta_pct_vs_r1": (
                        100.0 * (energy - baseline_energy) / baseline_energy
                        if baseline_energy
                        else None
                    ),
                    "total_silicon_area_mm2": area,
                    "area_delta_pct_vs_r1": (
                        100.0 * (area - baseline_area) / baseline_area
                        if baseline_area
                        else None
                    ),
                    "softmax_elements_per_cycle": row[
                        "softmax_elements_per_cycle"
                    ],
                    "softmax_row_lane_utilization": row.get(
                        "softmax_row_lane_utilization"
                    ),
                    "audit_fidelity": row["audit_fidelity"],
                }
            )
    return output


def _grid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(int(row["VLEN"]), int(row["softmax_row_lanes"]))].append(row)

    output: list[dict[str, Any]] = []
    for (vlen, lanes), members in sorted(groups.items()):
        latencies = sorted(_number(row, "latency_ms") for row in members)
        energies = sorted(
            _number(row, "system_energy_nominal_mj") for row in members
        )
        areas = sorted(
            _number(row, "new_total_silicon_area_mm2") for row in members
        )
        midpoint = len(members) // 2
        output.append(
            {
                "VLEN": vlen,
                "softmax_row_lanes": lanes,
                "softmax_elements_per_cycle": vlen * lanes,
                "trial_count": len(members),
                "minimum_latency_ms": latencies[0],
                "median_latency_ms": latencies[midpoint],
                "minimum_energy_mj": energies[0],
                "median_energy_mj": energies[midpoint],
                "minimum_total_silicon_area_mm2": areas[0],
                "median_total_silicon_area_mm2": areas[midpoint],
                "fidelity": "descriptive-unmatched-dse-grid",
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--grid-csv", type=Path)
    parser.add_argument("--matched-r-csv", type=Path)
    parser.add_argument("--assume-row-lanes", type=int, default=1, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--include-states",
        default="complete",
        help="comma-separated trial states to audit, or 'all' (default: complete)",
    )
    args = parser.parse_args()

    trial_records = _load_trial_records(args.run_dir)
    included_states = {
        value.strip().lower()
        for value in args.include_states.split(",")
        if value.strip()
    }
    include_all_states = included_states == {"all"}
    output = []
    failures = []
    excluded_state_counts: Counter[str] = Counter()
    for csv_row in _read_csv(args.input_csv):
        row = dict(csv_row)
        state = str(row.get("state", "complete")).strip().lower()
        if not include_all_states and state not in included_states:
            excluded_state_counts[state] += 1
            continue
        try:
            trial = int(float(row.get("trial", -1)))
            row.update(trial_records.get(trial, {}))
            output.append(audit_row(row, assumed_row_lanes=args.assume_row_lanes))
        except (KeyError, ValueError, TypeError) as exc:
            failures.append({"trial": row.get("trial"), "error": repr(exc)})

    _write_csv(args.output_csv, output)
    grid_rows = _grid_rows(output)
    matched_r_rows = _matched_r_rows(output)
    if args.grid_csv is not None:
        _write_csv(args.grid_csv, grid_rows)
    if args.matched_r_csv is not None:
        _write_csv(args.matched_r_csv, matched_r_rows)
    summary = {
        "schema": "rtl_v6_softmax_resource_audit_v1",
        "input_csv": str(args.input_csv),
        "run_dir": str(args.run_dir) if args.run_dir else None,
        "successful_rows": len(output),
        "included_states": "all" if include_all_states else sorted(included_states),
        "excluded_state_counts": dict(excluded_state_counts),
        "failed_rows": failures,
        "fidelity_counts": dict(Counter(row["audit_fidelity"] for row in output)),
        "row_lane_counts": dict(Counter(str(row["softmax_row_lanes"]) for row in output)),
        "area_calibration_status_counts": dict(
            Counter(str(row["area_calibration_status"]) for row in output)
        ),
        "vlen_row_lane_grid_points": len(grid_rows),
        "matched_r_rows": len(matched_r_rows),
        "matched_r_signatures": len(
            {row["matched_signature"] for row in matched_r_rows}
        ),
        "grid_csv": str(args.grid_csv) if args.grid_csv else None,
        "matched_r_csv": (
            str(args.matched_r_csv) if args.matched_r_csv else None
        ),
        "notes": [
            "Latency and energy are retained from the original trial; only area is recomputed.",
            "Missing historical SOFTMAX_ROW_LANES is never inferred from performance and is explicitly labelled.",
            "Area is recomputed with the currently promoted fail-closed rtl-v6 paired-DC artifact; each row retains its calibration status.",
            "The VLEN x R grid is descriptive because its trials may differ in other DSE knobs.",
            "Matched-R deltas are emitted only when precision, topology, SRAM, VLEN, MLEN, and BLEN signatures agree exactly.",
            "A VLEN-only latency comparison is not claimed because the current search enforces MLEN=VLEN.",
        ],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
