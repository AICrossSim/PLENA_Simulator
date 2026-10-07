"""Compare executable L-Compute layouts before freezing the RTL geometry."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from .projection_scatter import (
    ProjectionFlow,
    ScatterPlan,
    _scatter_write_stats,
    verify_scatter_roundtrip,
)


MIB = 1024 * 1024
_TESTDATA = Path(__file__).parents[2] / "transactional_emulator" / "testdata"


def _service_cycles(packets: list[list[int]], *, banks: int, ports: int) -> tuple[int, int]:
    ideal = service = 0
    for packet in packets:
        counts = [0] * banks
        for bank in packet:
            counts[bank] += 1
        ideal += math.ceil(len(packet) / (banks * ports))
        service += max(math.ceil(count / ports) for count in counts)
    return ideal, service


def _dense_mapping(mode: str, row: int, column: int, rows: int, columns: int, banks: int) -> int:
    if mode == "row_major":
        return (row * columns + column) % banks
    if mode == "transpose":
        return (column * rows + row) % banks
    if mode == "diagonal_custom":
        return (row + column) % banks
    raise ValueError(f"unknown dense layout {mode!r}")


def _dense_column_case(mode: str, *, rows: int, columns: int, banks: int, ports: int, burst: int) -> dict[str, Any]:
    write_packets = []
    for start in range(0, rows * columns, burst):
        write_packets.append(
            [
                _dense_mapping(
                    mode,
                    source // columns,
                    source % columns,
                    rows,
                    columns,
                    banks,
                )
                for source in range(start, min(start + burst, rows * columns))
            ]
        )
    read_packets = [
        [
            _dense_mapping(mode, row, column, rows, columns, banks)
            for row in range(rows)
        ]
        for column in range(columns)
    ]
    write_ideal, write_service = _service_cycles(write_packets, banks=banks, ports=ports)
    read_ideal, read_service = _service_cycles(read_packets, banks=banks, ports=ports)
    return {
        "layout": mode,
        "values": rows * columns,
        "sram_bytes_bf16": rows * columns * 2,
        "hbm_repack_bytes": 0,
        "write_ideal_cycles": write_ideal,
        "write_service_cycles": write_service,
        "write_stall_cycles": write_service - write_ideal,
        "read_ideal_cycles": read_ideal,
        "read_service_cycles": read_service,
        "read_stall_cycles": read_service - read_ideal,
        "total_service_cycles": write_service + read_service,
    }


def _row_major(plan: ScatterPlan) -> ScatterPlan:
    fields = tuple(replace(field, skew_kind="none", skew_stride=0) for field in plan.fields)
    candidate = replace(
        plan,
        layout="row_major",
        flow=ProjectionFlow.BUFFERED,
        fields=fields,
        mapping_sha256="",
    )
    return replace(candidate, mapping_sha256=candidate.compute_mapping_sha256())


def _buffered(plan: ScatterPlan) -> ScatterPlan:
    return replace(plan, flow=ProjectionFlow.BUFFERED)


def _state_case(plan: ScatterPlan, *, layers: int, name: str) -> dict[str, Any]:
    plan = _buffered(plan)
    read = verify_scatter_roundtrip(plan, tokens=1)
    spilled = {0: set(range(plan.source_values_per_token))}
    write = _scatter_write_stats(plan, spilled)
    return {
        "layout": name,
        "layers": layers,
        "values": plan.source_values_per_token * layers,
        "logical_bytes_bf16": plan.source_values_per_token * 2 * layers,
        "physical_sram_bytes_bf16": plan.physical_values_per_token * 2 * layers,
        "layout_descriptor_bytes": 256 * layers,
        "hbm_repack_bytes": 0,
        "write_ideal_cycles": write.ideal_cycles * layers,
        "write_service_cycles": write.service_cycles * layers,
        "write_stall_cycles": write.stall_cycles * layers,
        "read_ideal_cycles": read.ideal_cycles * layers,
        "read_service_cycles": read.service_cycles * layers,
        "read_stall_cycles": read.stall_cycles * layers,
        "total_service_cycles": (write.service_cycles + read.service_cycles) * layers,
        "max_read_bank_multiplicity": read.max_bank_multiplicity,
        "roundtrip_values": read.read_values * layers,
        "roundtrip_ok": read.read_values == plan.source_values_per_token,
    }


def _load_plan(name: str) -> ScatterPlan:
    document = json.loads((_TESTDATA / name).read_text())
    return ScatterPlan.from_dict(document["projection_scatters"][0]["plan"])


def _comparison(row: dict[str, Any], optimized: dict[str, Any]) -> dict[str, float]:
    read_ratio = row["read_service_cycles"] / optimized["read_service_cycles"]
    total_ratio = row["total_service_cycles"] / optimized["total_service_cycles"]
    return {
        "read_service_speedup": read_ratio,
        "read_service_reduction_percent": 100 * (1 - 1 / read_ratio),
        "read_write_service_speedup": total_ratio,
        "read_write_service_reduction_percent": 100 * (1 - 1 / total_ratio),
    }


def build_report() -> dict[str, Any]:
    mamba = _load_plan("projection_scatter_v1_nemotron_decode.json")
    kda = _load_plan("projection_scatter_v1_kimi_k3_decode.json")
    mamba_row = _state_case(_row_major(mamba), layers=23, name="row_major")
    mamba_skew = _state_case(mamba, layers=23, name="mamba_skew")
    kda_row = _state_case(_row_major(kda), layers=69, name="row_major")
    kda_skew = _state_case(kda, layers=69, name="kda_k8_skew")
    dense = [
        _dense_column_case(
            mode,
            rows=16,
            columns=128,
            banks=16,
            ports=1,
            burst=64,
        )
        for mode in ("row_major", "transpose", "diagonal_custom")
    ]
    return {
        "schema_version": 1,
        "contract": "plena-l-scatter-m-v1-dse",
        "hardware": {
            "banks": 16,
            "ports_per_bank": 1,
            "producer_burst_values": 64,
            "activation_precision": "bf16",
        },
        "dense_column_ablation": {
            "shape": [16, 128],
            "cases": dense,
            "finding": (
                "Pure transpose exchanges row-write conflicts for column-read conflicts; "
                "the diagonal CUSTOM mapping balances both directions."
            ),
        },
        "nemotron3_mamba_decode": {
            "layer_count": 23,
            "cases": [mamba_row, mamba_skew],
            "comparison": _comparison(mamba_row, mamba_skew),
        },
        "kimi_k3_kda_decode": {
            "layer_count": 69,
            "cases": [kda_row, kda_skew],
            "comparison": _comparison(kda_row, kda_skew),
        },
        "limits": [
            "These are deterministic bank-service cycles, not full-layer or chip speedups.",
            "HBM repack bytes are zero because L_SCATTER_M stays on chip; Matrix weight traffic is unchanged.",
            "The model includes read and write conflicts but not RTL mux delay, area, or timing closure.",
            "The dense transpose case is a generic column-read microbenchmark, not an X_STATE packet.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(build_report(), indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
