"""Run a compact SigLIP regression matrix across functional and hardware lanes.

This script executes a small set of depth checkpoints (embedding-only, 1-layer,
3-layer by default) for two validation lanes:

- Functional lane: model-faithful comparison (no MXFP)
- Hardware lane: hardware-faithful comparison (MXFP enabled)

It writes per-run artifacts and a consolidated summary table under build/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from statistics import mean

from transactional_emulator.testbench.siglip.full_model.siglip_config_driven_full_model_test import run_full_model_test


@dataclass(frozen=True)
class LaneConfig:
    name: str
    use_mxfp: bool
    model_dtype: str


def _parse_cases(cases_raw: str) -> list[int]:
    values: list[int] = []
    for token in cases_raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = int(token)
        if value < 0:
            raise ValueError(f"Layer count must be >= 0, got {value}")
        values.append(value)

    if not values:
        raise ValueError("At least one case must be provided")

    return sorted(set(values))


def _avg_layer_match(summary: dict) -> float | None:
    layer_metrics = summary.get("layer_metrics", [])
    if not layer_metrics:
        return None
    return float(mean(metric["match_rate"] for metric in layer_metrics))


def _layer0_metrics(summary: dict) -> tuple[float | None, float | None]:
    layer_metrics = summary.get("layer_metrics", [])
    if not layer_metrics:
        return None, None
    first = layer_metrics[0]
    return float(first["match_rate"]), float(first["mae"])


def _format_pct(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}%"


def _format_num(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"


def _write_markdown_table(rows: list[dict], out_path: Path) -> None:
    lines = [
        "# SigLIP Regression Matrix",
        "",
        "| Lane | Layers | MXFP | Model DType | Embedding Match | Embedding MAE | Layer0 Match | Layer0 MAE | Avg Layer Match |",
        "| --- | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row['lane']} | "
            f"{row['max_layers']} | "
            f"{'on' if row['use_mxfp'] else 'off'} | "
            f"{row['model_dtype']} | "
            f"{_format_pct(row['embedding_match_rate'])} | "
            f"{_format_num(row['embedding_mae'])} | "
            f"{_format_pct(row['layer0_match_rate'])} | "
            f"{_format_num(row['layer0_mae'])} | "
            f"{_format_pct(row['avg_layer_match_rate'])} |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_regression_matrix(
    config_path: str,
    output_dir: Path,
    cases: list[int],
    seed: int,
    functional_model_dtype: str,
    hardware_model_dtype: str,
) -> dict:
    lanes = [
        LaneConfig(name="functional", use_mxfp=False, model_dtype=functional_model_dtype),
        LaneConfig(name="hardware", use_mxfp=True, model_dtype=hardware_model_dtype),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SigLIP Regression Matrix")
    print("=" * 80)
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Cases: {cases}")
    print(f"Functional lane dtype: {functional_model_dtype}")
    print(f"Hardware lane dtype: {hardware_model_dtype}")

    rows: list[dict] = []

    for lane in lanes:
        for max_layers in cases:
            run_dir = output_dir / f"{lane.name}_L{max_layers}"
            print("\n" + "-" * 80)
            print(
                f"Running lane={lane.name}, layers={max_layers}, "
                f"use_mxfp={lane.use_mxfp}, model_dtype={lane.model_dtype}"
            )
            print("-" * 80)

            summary = run_full_model_test(
                config_path=config_path,
                output_dir=run_dir,
                max_layers=max_layers,
                use_mxfp=lane.use_mxfp,
                model_dtype=lane.model_dtype,
                seed=seed,
            )

            layer0_match, layer0_mae = _layer0_metrics(summary)
            row = {
                "lane": lane.name,
                "max_layers": max_layers,
                "use_mxfp": lane.use_mxfp,
                "model_dtype": lane.model_dtype,
                "embedding_match_rate": float(summary["embedding_metrics"]["match_rate"]),
                "embedding_mae": float(summary["embedding_metrics"]["mae"]),
                "layer0_match_rate": layer0_match,
                "layer0_mae": layer0_mae,
                "avg_layer_match_rate": _avg_layer_match(summary),
                "run_output_dir": str(run_dir),
            }
            rows.append(row)

    matrix = {
        "generated_at": datetime.now(UTC).isoformat(),
        "config_path": config_path,
        "cases": cases,
        "seed": seed,
        "rows": rows,
    }

    json_path = output_dir / "matrix_summary.json"
    md_path = output_dir / "matrix_summary.md"
    json_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    _write_markdown_table(rows, md_path)

    print("\n" + "=" * 80)
    print("MATRIX SUMMARY")
    print("=" * 80)
    for row in rows:
        print(
            f"{row['lane']:>10} L={row['max_layers']:<2} "
            f"emb_match={_format_pct(row['embedding_match_rate']):>8} "
            f"layer0_match={_format_pct(row['layer0_match_rate']):>8} "
            f"avg_layer_match={_format_pct(row['avg_layer_match_rate']):>8}"
        )
    print(f"\nSaved: {json_path}")
    print(f"Saved: {md_path}")

    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="SigLIP quick regression matrix runner")
    parser.add_argument(
        "config_path",
        nargs="?",
        default="compiler/doc/Model_Lib/siglip-so400m-patch14-384.json",
        help="Path to SigLIP config JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="./build/full_model_matrix",
        help="Directory where matrix outputs are written",
    )
    parser.add_argument(
        "--cases",
        default="0,1,3",
        help="Comma-separated max-layer checkpoints (e.g. 0,1,3)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--functional-model-dtype",
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Model dtype for functional lane (use_mxfp=False)",
    )
    parser.add_argument(
        "--hardware-model-dtype",
        default="float32",
        choices=["float32", "bfloat16"],
        help="Model dtype for hardware lane (use_mxfp=True)",
    )
    args = parser.parse_args()

    cases = _parse_cases(args.cases)
    run_regression_matrix(
        config_path=args.config_path,
        output_dir=Path(args.output_dir),
        cases=cases,
        seed=args.seed,
        functional_model_dtype=args.functional_model_dtype,
        hardware_model_dtype=args.hardware_model_dtype,
    )


if __name__ == "__main__":
    main()
