"""CLI for the common Mamba-2/KDA recurrent-state engine model."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from itertools import product
from pathlib import Path

from .state_engine import (
    RecurrentStateEngineModel,
    StateAlgorithm,
    StateEngineDesign,
    StateGeometry,
    StateSramLayout,
    StateStorage,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Common recurrent-state engine DSE")
    parser.add_argument("--algorithm", type=StateAlgorithm, choices=StateAlgorithm, default=StateAlgorithm.MAMBA2)
    parser.add_argument("--layout", type=StateSramLayout, choices=StateSramLayout)
    parser.add_argument("--state-precision", type=StateStorage, choices=StateStorage)
    parser.add_argument("--conv-state-precision", type=StateStorage, choices=StateStorage)
    parser.add_argument("--head-lanes", type=int, default=1)
    parser.add_argument("--row-lanes", type=int, default=4)
    parser.add_argument("--column-lanes", type=int, default=8)
    parser.add_argument("--banks", type=int, default=32)
    parser.add_argument("--fma-lanes", type=int, default=32)
    parser.add_argument("--head-tile-slots", type=int, default=2)
    parser.add_argument("--sweep-head-lanes")
    parser.add_argument("--sweep-banks")
    parser.add_argument("--sweep-head-tile-slots")
    parser.add_argument("--state-resident", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser


def _geometry(
    algorithm: StateAlgorithm,
    precision: StateStorage | None,
    conv_precision: StateStorage | None,
) -> StateGeometry:
    if algorithm == StateAlgorithm.MAMBA2:
        state = precision or StateStorage.FP32
        return StateGeometry.nemotron3_mamba2(state)
    return StateGeometry.kimi_k3_kda(
        precision or StateStorage.FP32,
        conv_precision or StateStorage.BF16,
    )


def build_document(args: argparse.Namespace) -> dict:
    layouts = tuple(StateSramLayout) if args.layout is None else (args.layout,)
    head_lanes = _int_values(args.sweep_head_lanes, args.head_lanes)
    banks = _int_values(args.sweep_banks, args.banks)
    head_tile_slots = _int_values(args.sweep_head_tile_slots, args.head_tile_slots)
    base = StateEngineDesign(
        head_lanes=args.head_lanes,
        row_lanes=args.row_lanes,
        column_lanes=args.column_lanes,
        banks_per_head_lane=args.banks,
        fma_lanes_per_head_lane=args.fma_lanes,
        head_tile_slots=args.head_tile_slots,
    )
    geometry = _geometry(
        args.algorithm,
        args.state_precision,
        args.conv_state_precision,
    )
    results = [
        RecurrentStateEngineModel(
            geometry,
            replace(
                base,
                layout=layout,
                head_lanes=head_lane_count,
                banks_per_head_lane=bank_count,
                head_tile_slots=slot_count,
            ),
        ).evaluate(
            state_resident=args.state_resident
        )
        for layout, head_lane_count, bank_count, slot_count in product(
            layouts, head_lanes, banks, head_tile_slots
        )
    ]
    return {
        "schema_version": 1,
        "calibration": "uncalibrated_no_rtl",
        "resource_proxy": "compare total FMA lanes, state banks, and head-tile SRAM bytes; this is not synthesized area",
        "results": [result.to_dict() for result in results],
    }


def _int_values(raw: str | None, default: int) -> tuple[int, ...]:
    values = (default,) if raw is None else tuple(int(value) for value in raw.split(","))
    if not values or any(value <= 0 for value in values):
        raise ValueError("DSE values must be positive comma-separated integers")
    return values


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_document(args)
    print(f"State Engine | {args.algorithm} | calibration=NO")
    print("layout              heads banks slots  us/layer  SRAM KiB  bank stall  HBM MiB")
    for result in document["results"]:
        metrics = result["metrics"]
        bank = result["bank_stats"]
        design = result["design"]
        print(
            f"{design['layout']:<20} {design['head_lanes']:>5} "
            f"{design['banks_per_head_lane']:>5} {design['head_tile_slots']:>5} "
            f"{metrics['latency_us']:>8.1f} "
            f"{metrics['head_tile_sram_bytes'] / 1024:>9.1f} "
            f"{bank['stall_cycles']:>11} "
            f"{(metrics['hbm_read_bytes'] + metrics['hbm_write_bytes']) / (1024 * 1024):>8.1f}"
        )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(document, indent=2) + "\n")
        print(f"JSON report: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
