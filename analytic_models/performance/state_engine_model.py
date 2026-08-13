"""CLI for the common Mamba-2/KDA recurrent-state engine model."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .state_engine import (
    RecurrentStateEngineModel,
    StateAlgorithm,
    StateEngineDesign,
    StateGeometry,
    StateSramLayout,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Common recurrent-state engine DSE")
    parser.add_argument("--algorithm", type=StateAlgorithm, choices=StateAlgorithm, default=StateAlgorithm.MAMBA2)
    parser.add_argument("--layout", type=StateSramLayout, choices=StateSramLayout)
    parser.add_argument("--head-lanes", type=int, default=1)
    parser.add_argument("--row-lanes", type=int, default=4)
    parser.add_argument("--column-lanes", type=int, default=8)
    parser.add_argument("--banks", type=int, default=32)
    parser.add_argument("--fma-lanes", type=int, default=32)
    parser.add_argument("--head-tile-slots", type=int, default=2)
    parser.add_argument("--state-resident", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser


def _geometry(algorithm: StateAlgorithm) -> StateGeometry:
    if algorithm == StateAlgorithm.MAMBA2:
        return StateGeometry.nemotron3_mamba2()
    return StateGeometry.kimi_k3_kda()


def build_document(args: argparse.Namespace) -> dict:
    layouts = tuple(StateSramLayout) if args.layout is None else (args.layout,)
    base = StateEngineDesign(
        head_lanes=args.head_lanes,
        row_lanes=args.row_lanes,
        column_lanes=args.column_lanes,
        banks_per_head_lane=args.banks,
        fma_lanes_per_head_lane=args.fma_lanes,
        head_tile_slots=args.head_tile_slots,
    )
    geometry = _geometry(args.algorithm)
    results = [
        RecurrentStateEngineModel(geometry, replace(base, layout=layout)).evaluate(
            state_resident=args.state_resident
        )
        for layout in layouts
    ]
    return {
        "schema_version": 1,
        "calibration": "uncalibrated_no_rtl",
        "results": [result.to_dict() for result in results],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = build_document(args)
    print(f"State Engine | {args.algorithm} | calibration=NO")
    print("layout              us/layer  SRAM KiB  bank stall  HBM MiB")
    for result in document["results"]:
        metrics = result["metrics"]
        bank = result["bank_stats"]
        design = result["design"]
        print(
            f"{design['layout']:<20} {metrics['latency_us']:>8.1f} "
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
