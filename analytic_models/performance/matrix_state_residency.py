"""State precision and Matrix SRAM residency contract for Matrix L-Compute.

This module prevents a layout experiment from silently assuming that an
official FP32 recurrent state fits in the existing BF16 Matrix SRAM.  It reports
the shipped SRAM geometry, the packed lower-bound number of capacity windows,
and the explicit HBM read/write traffic when state is not resident.

The report is a capacity contract, not an RTL claim.  In particular, an FP32
or MX8 state is not natively representable by the shipped BF16 Matrix SRAM.
Those rows require either conversion to BF16 or a different SRAM data format;
neither is hidden in the cycle count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = ROOT / "plena_settings.toml"
MAMBA_PRECISION_CSV = (
    Path(__file__).with_name("profiles") / "b200_supplemental" / "mamba_precision.csv"
)
KDA_PRECISION_JSON = Path(__file__).with_name("profiles") / "kda_state_precision.json"

KIMI_K3_STATE_ELEMENTS = 96 * 128 * 128
KIMI_K3_KDA_LAYERS = 69
# Official Nemotron-3 Nano Mamba-2 state is
# [num_heads=64, head_dim=64, state_dim=128]. ``n_groups=8`` controls B/C
# sharing; it does not replace the head dimension in the recurrent state.
NEMOTRON_STATE_ELEMENTS = 64 * 64 * 128
NEMOTRON_MAMBA_LAYERS = 23


class StateFormat(StrEnum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    MX8_B128 = "mx8_b128"


def storage_bytes(elements: int, storage: StateFormat, *, block_size: int = 128) -> int:
    if elements < 0:
        raise ValueError("elements must be non-negative")
    if storage == StateFormat.FP32:
        return 4 * elements
    if storage in {StateFormat.BF16, StateFormat.FP16}:
        return 2 * elements
    if storage == StateFormat.MX8_B128:
        return elements + math.ceil(elements / block_size)
    raise ValueError(f"unsupported state format {storage}")


def _plain_width_bits(precision: dict[str, Any]) -> int:
    if precision.get("format") != "Plain":
        raise ValueError("Matrix SRAM capacity report requires a Plain element type")
    dtype = precision.get("DATA_TYPE", {})
    if dtype.get("type") == "Fp":
        return int(dtype["sign"]) + int(dtype["exponent"]) + int(dtype["mantissa"])
    if dtype.get("type") == "Int":
        return int(dtype["width"])
    raise ValueError(f"unsupported Matrix SRAM data type: {dtype}")


def _plain_format_name(precision: dict[str, Any]) -> str:
    """Return the exact plain format; equal width alone is not compatibility."""

    dtype = precision.get("DATA_TYPE", {})
    if precision.get("format") != "Plain" or dtype.get("type") != "Fp":
        return "unsupported"
    signature = (
        int(dtype["sign"]),
        int(dtype["exponent"]),
        int(dtype["mantissa"]),
    )
    return {
        (1, 8, 23): "fp32",
        (1, 8, 7): "bf16",
        (1, 5, 10): "fp16",
    }.get(signature, f"fp_s{signature[0]}e{signature[1]}m{signature[2]}")


@dataclass(frozen=True)
class MatrixSramGeometry:
    mode: str
    mlen: int
    depth_rows: int
    element_bits: int
    element_format: str = "unknown"

    @property
    def elements(self) -> int:
        return self.mlen * self.depth_rows

    @property
    def physical_bytes(self) -> int:
        return math.ceil(self.elements * self.element_bits / 8)

    @property
    def whole_square_tiles(self) -> int:
        return self.depth_rows // self.mlen

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            elements=self.elements,
            physical_bytes=self.physical_bytes,
            whole_square_tiles=self.whole_square_tiles,
            legacy_square_tile_api_valid=self.whole_square_tiles > 0,
            compact_matrix_view_api_valid=self.depth_rows > 0,
            size_unit="MLEN-wide logical rows",
        )
        return result


def load_geometries(path: Path = DEFAULT_SETTINGS) -> dict[str, MatrixSramGeometry]:
    settings = tomllib.loads(path.read_text())
    result = {}
    for mode in ("ANALYTIC", "TRANSACTIONAL"):
        section = settings[mode]
        result[mode.lower()] = MatrixSramGeometry(
            mode=mode.lower(),
            mlen=int(section["CONFIG"]["MLEN"]["value"]),
            depth_rows=int(section["CONFIG"]["MATRIX_SRAM_SIZE"]["value"]),
            element_bits=_plain_width_bits(section["PRECISION"]["MATRIX_SRAM_TYPE"]),
            element_format=_plain_format_name(section["PRECISION"]["MATRIX_SRAM_TYPE"]),
        )
    return result


@dataclass(frozen=True)
class ResidencyCase:
    model: str
    layers: int
    state_elements_per_layer: int
    state_format: StateFormat
    state_bytes_per_layer: int
    matrix_capacity_bytes: int
    capacity_windows: int
    final_window_occupancy: float
    hbm_read_write_bytes_per_layer_token: int
    hbm_read_write_bytes_all_layers_token: int
    natively_representable: bool


def residency_case(
    *,
    model: str,
    layers: int,
    state_elements: int,
    storage: StateFormat,
    geometry: MatrixSramGeometry,
) -> ResidencyCase:
    state_bytes = storage_bytes(state_elements, storage)
    windows = math.ceil(state_bytes / geometry.physical_bytes)
    final_bytes = state_bytes - (windows - 1) * geometry.physical_bytes
    native = storage.value == geometry.element_format
    return ResidencyCase(
        model=model,
        layers=layers,
        state_elements_per_layer=state_elements,
        state_format=storage,
        state_bytes_per_layer=state_bytes,
        matrix_capacity_bytes=geometry.physical_bytes,
        capacity_windows=windows,
        final_window_occupancy=final_bytes / geometry.physical_bytes,
        hbm_read_write_bytes_per_layer_token=2 * state_bytes,
        hbm_read_write_bytes_all_layers_token=2 * state_bytes * layers,
        natively_representable=native,
    )


def _load_mamba_accuracy() -> dict[str, Any]:
    wanted = {
        "fp32": "fp32",
        "bf16_chunk128": "bf16",
        "fp16_chunk128": "fp16",
        "mx8_chunk128": "mx8_b128",
    }
    result = {}
    with MAMBA_PRECISION_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["case"] != "prefill_s32768" or row["variant"] not in wanted:
                continue
            result[wanted[row["variant"]]] = {
                "schedule": row["schedule"],
                "output_relative_l2_mean": float(row["output_relative_l2_mean"]),
                "state_relative_l2_mean": float(row["state_relative_l2_mean"]),
                "total_bytes": int(row["total_bytes"]),
                "source": "B200 Nemotron Mamba, S=32768, three seeds",
            }
    if set(result) != set(wanted.values()):
        raise ValueError("B200 Mamba precision artifact is incomplete")
    return result


def _load_kda_accuracy() -> dict[str, Any] | None:
    if not KDA_PRECISION_JSON.exists():
        return None
    payload = json.loads(KDA_PRECISION_JSON.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported KDA precision artifact")
    return payload


def build_report(settings: Path = DEFAULT_SETTINGS) -> dict[str, Any]:
    geometries = load_geometries(settings)
    cases = []
    for geometry in geometries.values():
        for storage in StateFormat:
            cases.append(
                residency_case(
                    model="kimi_k3",
                    layers=KIMI_K3_KDA_LAYERS,
                    state_elements=KIMI_K3_STATE_ELEMENTS,
                    storage=storage,
                    geometry=geometry,
                )
            )
            cases.append(
                residency_case(
                    model="nemotron3_nano",
                    layers=NEMOTRON_MAMBA_LAYERS,
                    state_elements=NEMOTRON_STATE_ELEMENTS,
                    storage=storage,
                    geometry=geometry,
                )
            )
    paper = geometries["analytic"]
    packet_footprints = {
        "nemotron_mamba_32_heads_x_64": {
            "reserved_rows_one_operand": 32,
            "reserved_bytes_one_operand": 32 * paper.mlen * paper.element_bits // 8,
            "reserved_bytes_two_operands": 64 * paper.mlen * paper.element_bits // 8,
            "fits_one_operand": 32 <= paper.depth_rows,
            "fits_two_operands": 64 <= paper.depth_rows,
        },
        "kimi_kda_16_heads_x_128": {
            "reserved_rows_one_operand": 16,
            "reserved_bytes_one_operand": 16 * paper.mlen * paper.element_bits // 8,
            "reserved_bytes_two_operands": 32 * paper.mlen * paper.element_bits // 8,
            "fits_one_operand": 16 <= paper.depth_rows,
            "fits_two_operands": 32 <= paper.depth_rows,
        },
    }
    return {
        "schema_version": 1,
        "matrix_sram": {name: geometry.to_dict() for name, geometry in geometries.items()},
        "residency": [asdict(case) for case in cases],
        "published_packet_footprints": packet_footprints,
        "accuracy": {
            "nemotron_mamba": _load_mamba_accuracy(),
            "kimi_kda": _load_kda_accuracy(),
        },
        "conclusions": [
            "Official Kimi K3 FP32 recurrent state is 6 MiB per layer.",
            "The analytic Matrix SRAM is 1 MiB (2048 x 256 BF16 elements).",
            "The shipped transactional Matrix SRAM is 512 KiB of BF16 storage.",
            "FP32 and MX8 state are not natively represented by that BF16 SRAM.",
            "Matrix L-Compute therefore cannot assume full state residency; it must report explicit tiled traffic.",
            "Compact Mamba/KDA BF16 operand views fit the 256-row analytic SRAM even though one 2048-square legacy tile does not.",
            "BF16 q/k/v/B/C temporaries are the unconditional Matrix L-Compute scope.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    rendered = json.dumps(build_report(args.settings), indent=2) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
