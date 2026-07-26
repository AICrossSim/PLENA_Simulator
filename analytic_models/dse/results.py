"""Deterministic DSE selectors and compact multi-chip summaries."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import write_json


def select_area_reference_candidates(
    completed_records: list[dict[str, Any]],
    *,
    target_area_mm2: float,
    area_budget_mm2: float,
    target_area_tolerance_pct: float,
) -> dict[str, Any]:
    feasible = [
        record
        for record in completed_records
        if float(record["area_mm2"]) <= area_budget_mm2
    ]

    def energy(record: Mapping[str, Any]) -> float:
        return float(
            record.get(
                "system_energy_nominal_mj",
                record.get("system_energy_mj", float("inf")),
            )
        )

    def fastest_key(record: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(record["latency_ms"]),
            -float(record["accuracy_score"]),
            float(record["area_mm2"]),
            int(record.get("trial", -1)),
        )

    fastest = min(feasible, key=fastest_key) if feasible else None
    lowest_energy = (
        min(
            feasible,
            key=lambda record: (
                energy(record),
                float(record["latency_ms"]),
                -float(record["accuracy_score"]),
                float(record["area_mm2"]),
                int(record.get("trial", -1)),
            ),
        )
        if feasible
        else None
    )
    fidelity_qualified = [
        record for record in feasible if record.get("candidate_fidelity") == "validated"
    ]
    fastest_fidelity_qualified = (
        min(fidelity_qualified, key=fastest_key) if fidelity_qualified else None
    )
    highest_accuracy = (
        max(
            feasible,
            key=lambda record: (
                float(record["accuracy_score"]),
                -float(record["latency_ms"]),
                -float(record["area_mm2"]),
                -int(record.get("trial", -1)),
            ),
        )
        if feasible
        else None
    )
    closest_to_target = (
        min(
            feasible,
            key=lambda record: (
                abs(float(record["area_mm2"]) - target_area_mm2),
                -float(record["accuracy_score"]),
                float(record["latency_ms"]),
                int(record.get("trial", -1)),
            ),
        )
        if feasible
        else None
    )
    below_target = [
        record for record in feasible if float(record["area_mm2"]) <= target_area_mm2
    ]
    closest_below_target = (
        max(
            below_target,
            key=lambda record: (
                float(record["area_mm2"]),
                float(record["accuracy_score"]),
                -float(record["latency_ms"]),
                -int(record.get("trial", -1)),
            ),
        )
        if below_target
        else None
    )
    tolerance_mm2 = target_area_mm2 * target_area_tolerance_pct / 100.0
    within_tolerance = [
        record
        for record in feasible
        if abs(float(record["area_mm2"]) - target_area_mm2) <= tolerance_mm2
    ]
    fastest_within_tolerance = (
        min(within_tolerance, key=fastest_key) if within_tolerance else None
    )
    lowest_energy_within_tolerance = (
        min(
            within_tolerance,
            key=lambda record: (
                energy(record),
                float(record["latency_ms"]),
                int(record.get("trial", -1)),
            ),
        )
        if within_tolerance
        else None
    )
    best_energy_delay_product = (
        min(
            feasible,
            key=lambda record: (
                float(record["latency_ms"]) * energy(record),
                float(record["area_mm2"]),
                -float(record["accuracy_score"]),
                int(record.get("trial", -1)),
            ),
        )
        if feasible
        else None
    )
    smaller_beating_target_candidate = []
    if closest_to_target is not None:
        smaller_beating_target_candidate = [
            record
            for record in feasible
            if float(record["area_mm2"]) < float(closest_to_target["area_mm2"])
            and float(record["latency_ms"]) <= float(closest_to_target["latency_ms"])
            and energy(record) <= energy(closest_to_target)
            and float(record["accuracy_score"])
            >= float(closest_to_target["accuracy_score"])
        ]
    smallest_design_beating_a100_area_candidate = (
        min(
            smaller_beating_target_candidate,
            key=lambda record: (
                float(record["area_mm2"]),
                float(record["latency_ms"]),
                energy(record),
                int(record.get("trial", -1)),
            ),
        )
        if smaller_beating_target_candidate
        else None
    )
    p90_feasible = [
        record
        for record in completed_records
        if float(record.get("area_uncertainty_p90_mm2", record["area_mm2"]))
        <= area_budget_mm2
    ]

    def p90_fastest_key(record: dict[str, Any]) -> tuple[float, float, float, int]:
        return (
            float(record["latency_ms"]),
            -float(record["accuracy_score"]),
            float(record.get("area_uncertainty_p90_mm2", record["area_mm2"])),
            int(record.get("trial", -1)),
        )

    p90_fastest = min(p90_feasible, key=p90_fastest_key) if p90_feasible else None
    p90_closest_to_target = (
        min(
            p90_feasible,
            key=lambda record: (
                abs(
                    float(
                        record.get("area_uncertainty_p90_mm2", record["area_mm2"])
                    )
                    - target_area_mm2
                ),
                -float(record["accuracy_score"]),
                float(record["latency_ms"]),
                int(record.get("trial", -1)),
            ),
        )
        if p90_feasible
        else None
    )
    return {
        "feasible": feasible,
        "fastest": fastest,
        "lowest_energy": lowest_energy,
        "fidelity_qualified": fidelity_qualified,
        "fastest_fidelity_qualified": fastest_fidelity_qualified,
        "highest_accuracy": highest_accuracy,
        "closest_to_target": closest_to_target,
        "closest_below_target": closest_below_target,
        "within_tolerance": within_tolerance,
        "fastest_within_tolerance": fastest_within_tolerance,
        "lowest_energy_within_tolerance": lowest_energy_within_tolerance,
        "best_energy_delay_product": best_energy_delay_product,
        "smallest_design_beating_a100_area_candidate": (
            smallest_design_beating_a100_area_candidate
        ),
        "p90_feasible": p90_feasible,
        "p90_fastest": p90_fastest,
        "p90_closest_to_target": p90_closest_to_target,
    }


def write_multi_chip_analysis(
    run_dir: Path,
    completed_records: list[dict[str, Any]],
    *,
    target_area_mm2: float,
) -> None:
    sram_group_fields = (
        "precision_profile",
        "MLEN",
        "BLEN",
        "INT_DATA_WIDTH",
        "chip_count",
        "parallel_model",
        "tp_degree",
        "cp_degree",
        "nvlink_port_count",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in completed_records:
        key = tuple(record.get(field) for field in sram_group_fields)
        grouped.setdefault(key, []).append(record)
    marginal_rows: list[dict[str, Any]] = []
    for key, group in grouped.items():
        ordered = sorted(
            group,
            key=lambda record: int(
                record.get("matrix_sram_tiles", record.get("MATRIX_SRAM_TILES", 2))
            ),
        )
        previous = None
        for record in ordered:
            row = {field: value for field, value in zip(sram_group_fields, key)}
            row.update(
                {
                    "trial": record.get("trial"),
                    "matrix_sram_tiles": record.get(
                        "matrix_sram_tiles", record.get("MATRIX_SRAM_TILES")
                    ),
                    "matrix_sram_logical_mb": record.get(
                        "matrix_sram_logical_mb"
                    ),
                    "latency_ms": record.get("latency_ms"),
                    "total_silicon_area_mm2": record.get("area_mm2"),
                    "kv_reload_factor": record.get("kv_reload_factor"),
                    "kv_tile_load_count": record.get("kv_tile_load_count"),
                    "attention_kv_resident": record.get("attention_kv_resident"),
                    "useful_saturation_tiles": record.get(
                        "matrix_sram_useful_saturation_tiles"
                    ),
                    "delta_latency_ms": (
                        None
                        if previous is None
                        else float(record["latency_ms"])
                        - float(previous["latency_ms"])
                    ),
                    "delta_area_mm2": (
                        None
                        if previous is None
                        else float(record["area_mm2"]) - float(previous["area_mm2"])
                    ),
                }
            )
            marginal_rows.append(row)
            previous = record
    if marginal_rows:
        with (run_dir / "matrix_sram_marginals.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(marginal_rows[0]))
            writer.writeheader()
            writer.writerows(marginal_rows)

    pair_fields = (
        "precision_profile",
        "MLEN",
        "BLEN",
        "INT_DATA_WIDTH",
        "chip_count",
        "matrix_sram_tiles",
    )
    pairs: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for record in completed_records:
        key = tuple(record.get(field) for field in pair_fields)
        pairs.setdefault(key, {})[str(record.get("parallel_model"))] = record
    tp_comparisons = []
    for key, modes in pairs.items():
        if not {"tp-sp", "tp-only"} <= set(modes):
            continue
        optimistic = modes["tp-sp"]
        conservative = modes["tp-only"]
        tp_comparisons.append(
            {
                **{field: value for field, value in zip(pair_fields, key)},
                "tp_sp_latency_ms": optimistic["latency_ms"],
                "tp_only_latency_ms": conservative["latency_ms"],
                "tp_only_minus_tp_sp_ms": (
                    float(conservative["latency_ms"])
                    - float(optimistic["latency_ms"])
                ),
                "tp_only_over_tp_sp": (
                    float(conservative["latency_ms"])
                    / float(optimistic["latency_ms"])
                ),
            }
        )

    summary_fields = (
        "trial",
        "latency_ms",
        "area_mm2",
        "system_energy_nominal_mj",
        "accuracy_score",
        "chip_count",
        "tp_degree",
        "cp_degree",
        "nvlink_port_count",
        "MLEN",
        "BLEN",
        "matrix_sram_policy",
        "matrix_sram_tiles",
        "precision_profile",
        "max_token_fraction",
        "max_causal_pair_fraction",
        "tp_collective_latency_ns",
        "cp_kv_ring_latency_ns",
        "weight_replication_factor",
    )

    def summary(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            field: record.get(field) for field in summary_fields if field in record
        }

    by_chip = {}
    for chip_count in sorted(
        {int(record.get("chip_count", 1)) for record in completed_records}
    ):
        subset = [
            record
            for record in completed_records
            if int(record.get("chip_count", 1)) == chip_count
        ]
        by_chip[str(chip_count)] = {
            "trial_count": len(subset),
            "fastest": (
                summary(min(subset, key=lambda record: float(record["latency_ms"])))
                if subset
                else None
            ),
            "closest_to_reference_area": (
                summary(
                    min(
                        subset,
                        key=lambda record: abs(
                            float(record["area_mm2"]) - target_area_mm2
                        ),
                    )
                )
                if subset
                else None
            ),
        }
    by_decomposition: dict[str, list[dict[str, Any]]] = {}
    for record in completed_records:
        key = (
            f"N{int(record.get('chip_count', 1))}"
            f"_TP{int(record.get('tp_degree', 1))}"
            f"_CP{int(record.get('cp_degree', 1))}"
            f"_EP{int(record.get('ep_degree', 1))}"
            f"_P{int(record.get('nvlink_port_count', 1))}"
        )
        by_decomposition.setdefault(key, []).append(record)
    decomposition_summary = {
        key: {
            "trial_count": len(records),
            "fastest": summary(
                min(records, key=lambda record: float(record["latency_ms"]))
            ),
            "lowest_energy": summary(
                min(
                    records,
                    key=lambda record: float(record["system_energy_nominal_mj"]),
                )
            ),
        }
        for key, records in sorted(by_decomposition.items())
    }
    write_json(
        run_dir / "multi_chip_analysis.json",
        {
            "model": (
                "tile_aware_tp_cp_ep_v3"
                if any(
                    record.get("multi_chip_model") == "tile-aware-tp-cp-ep-v3"
                    for record in completed_records
                )
                else "factorized_tp_cp_v2"
                if any(
                    record.get("multi_chip_model") == "factorized-tp-cp-v2"
                    for record in completed_records
                )
                else "stage_level_multi_chip_v1"
            ),
            "target_aggregate_area_mm2": target_area_mm2,
            "by_chip_count": by_chip,
            "by_tp_cp_port": decomposition_summary,
            "tp_sp_vs_tp_only": tp_comparisons,
            "matrix_sram_marginals_csv": "matrix_sram_marginals.csv",
            "notes": [
                "The default tile-aware v3 model searches legal TP x CP x EP "
                "decompositions and reconstructs rank-local padded tiles.",
                "factorized-tp-cp-v2 is a historical fractional A/B baseline.",
                "CP uses an exact zigzag token and causal-pair census.",
                "NVLink ports use 450 GB/s one-way peak per port with no "
                "sustained-bandwidth efficiency discount.",
            ],
        },
    )
