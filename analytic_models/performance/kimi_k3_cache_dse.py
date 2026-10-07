"""Capacity sweep for Kimi K3's exact mixed-precision KDA state."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .kimi_k3_workload import KimiK3Architecture
from .nemotron3_workload import Precision


MIB = 1024 * 1024


@dataclass(frozen=True)
class CachePoint:
    capacity_mib: float
    policy: str
    entry_bytes: int
    resident_layers: int
    accesses: int
    hits: int
    misses: int
    hbm_read_bytes: int
    hbm_write_bytes: int

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    def to_dict(self) -> dict[str, int | float | str]:
        return {**asdict(self), "hit_rate": self.hit_rate}


def evaluate_capacity(
    capacity_mib: float,
    *,
    policy: str,
    batch_size: int = 1,
    decode_tokens: int = 2,
) -> CachePoint:
    if capacity_mib < 0 or batch_size <= 0 or decode_tokens <= 0:
        raise ValueError("capacity must be non-negative and workload dimensions positive")
    if policy not in {"streaming", "pinned", "lru"}:
        raise ValueError(f"unsupported KDA cache policy {policy}")
    arch = KimiK3Architecture()
    layers = len(arch.kda_layer_numbers)
    entry_bytes = (
        arch.recurrent_state_bytes(Precision.FP32)
        + arch.conv_state_bytes(Precision.BF16)
    ) // layers
    states = layers * batch_size
    physical_capacity_entries = min(states, int(capacity_mib * MIB) // entry_bytes)
    capacity_entries = 0 if policy == "streaming" else physical_capacity_entries
    accesses = states * decode_tokens

    if capacity_entries == 0:
        hits = 0
    elif policy == "pinned":
        hits = capacity_entries * decode_tokens
    elif capacity_entries == states:
        hits = accesses
    else:
        # A layer-major pass over more states than the cache capacity evicts
        # every tail entry before the next pass reaches it.
        hits = 0
    misses = accesses - hits
    return CachePoint(
        capacity_mib=capacity_mib,
        policy=policy,
        entry_bytes=entry_bytes,
        resident_layers=capacity_entries,
        accesses=accesses,
        hits=hits,
        misses=misses,
        hbm_read_bytes=misses * entry_bytes,
        hbm_write_bytes=misses * entry_bytes,
    )


def build_report(
    capacities_mib: tuple[float, ...] = (0, 24, 32, 64, 128, 256, 512),
    *,
    batch_size: int = 1,
    decode_tokens: int = 2,
) -> dict[str, object]:
    points = [
        evaluate_capacity(
            capacity,
            policy=policy,
            batch_size=batch_size,
            decode_tokens=decode_tokens,
        )
        for capacity in capacities_mib
        for policy in ("streaming", "lru", "pinned")
    ]
    return {
        "schema_version": 1,
        "model": "moonshotai/Kimi-K3",
        "scope": "69_kda_layer_decode_state_capacity",
        "state_contract": {
            "recurrent": "fp32",
            "conv": "bf16",
            "entry_bytes": points[0].entry_bytes,
            "entry_mib": points[0].entry_bytes / MIB,
            "layers": 69,
            "total_mib_per_request": 69 * points[0].entry_bytes / MIB,
        },
        "workload": {"batch_size": batch_size, "decode_tokens": decode_tokens},
        "points": [point.to_dict() for point in points],
        "interpretation": [
            "Pinned residency gives proportional steady-state hits at sub-model capacity.",
            "Layer-major LRU has zero reuse until all request/layer states fit.",
            "Read/write bytes exclude one-time preload and final commit for pinned entries.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Kimi K3 KDA state-cache capacity DSE")
    parser.add_argument("--capacities-mib", default="0,24,32,64,128,256,512")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=2)
    parser.add_argument("--json-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    capacities = tuple(float(value) for value in args.capacities_mib.split(","))
    report = build_report(
        capacities,
        batch_size=args.batch_size,
        decode_tokens=args.decode_tokens,
    )
    for point in report["points"]:
        if point["policy"] == "pinned":
            print(
                f"{point['capacity_mib']:>6g} MiB: {point['resident_layers']:>2}/69 resident, "
                f"hit={100 * point['hit_rate']:>5.1f}%, "
                f"HBM={2 * point['hbm_read_bytes'] / MIB:>8.1f} MiB"
            )
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
