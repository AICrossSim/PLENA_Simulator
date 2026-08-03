"""Calibrated residual model over exact HBM request statistics."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from compiler.aten.isa_builder import DmaTransfer

from .schema import (
    FEATURE_SEMANTIC_VERSION,
    MANIFEST_HASH_ALGORITHM,
    PHYSICAL_BURST_BYTES,
    REQUEST_BYTES,
    DmaRequestManifest,
    HbmV4Config,
    MemoryFormat,
    mop4clxor_phase_statistics,
    request_manifest_fixture_hash,
)


SCHEMA_VERSION = 4
FEATURE_NAMES = (
    "read_phase_startup",
    "write_phase_startup",
    "read_write_turnaround",
    "read_channel_tail",
    "write_channel_tail",
    "read_bankgroup_serial",
    "write_bankgroup_serial",
    "read_bank_serial",
    "write_bank_serial",
    "read_row_miss",
    "write_row_miss",
    "read_row_conflict",
    "write_row_conflict",
    "sram_dma_drain",
)
WARM_FEATURE_NAMES = (
    "read_phase_startup",
    "write_phase_startup",
    "read_row_conflict",
    "write_row_conflict",
)


@dataclass(frozen=True)
class V4FeatureVector:
    theoretical_phase_floor_ns: float
    values: Mapping[str, float]


@dataclass(frozen=True)
class HbmV4Prediction:
    latency_ns: float
    theoretical_phase_floor_ns: float
    calibration_in_domain: bool
    domain_issues: tuple[str, ...]
    extrapolation_ratio: float
    features: Mapping[str, float]
    row_state_regime: str


def occurrence_features(
    manifest: DmaRequestManifest,
    transfer: DmaTransfer,
    config: HbmV4Config,
    *,
    open_rows: np.ndarray | None = None,
) -> V4FeatureVector:
    if open_rows is None:
        open_rows = np.full(config.channels * 32, -1, dtype=np.int64)
    read = mop4clxor_phase_statistics(manifest.read_lines, config, open_rows)
    write = mop4clxor_phase_statistics(manifest.write_lines, config, open_rows)
    read_max, read_pseudo, read_group, read_bank, read_miss, read_initial, read_conflict = read
    write_max, write_pseudo, write_group, write_bank, write_miss, write_initial, write_conflict = write
    bursts_per_line = REQUEST_BYTES // PHYSICAL_BURST_BYTES
    read_average = len(manifest.read_lines) * bursts_per_line / config.channels
    write_average = len(manifest.write_lines) * bursts_per_line / config.channels
    burst_service_ns = PHYSICAL_BURST_BYTES / config.channel_bandwidth_bytes_per_ns
    floor = burst_service_ns * (max(read_max, 2 * read_pseudo) + max(write_max, 2 * write_pseudo))
    has_read = bool(manifest.read_lines)
    has_write = bool(manifest.write_lines)
    return V4FeatureVector(
        theoretical_phase_floor_ns=floor,
        values={
            "read_phase_startup": float(has_read),
            "write_phase_startup": float(has_write),
            "read_write_turnaround": float(has_read and has_write),
            "read_channel_tail": max(0.0, read_max - read_average),
            "write_channel_tail": max(0.0, write_max - write_average),
            "read_bankgroup_serial": float(read_group),
            "write_bankgroup_serial": float(write_group),
            "read_bank_serial": float(max(0, read_bank - bursts_per_line)),
            "write_bank_serial": float(max(0, write_bank - bursts_per_line)),
            "read_row_miss": float(read_miss),
            "write_row_miss": float(write_miss),
            "read_row_conflict": float(read_conflict),
            "write_row_conflict": float(write_conflict),
            "read_initial_row_conflict": float(read_initial),
            "write_initial_row_conflict": float(write_initial),
            "sram_dma_drain": math.log2(transfer.amount + 1)
            + math.sqrt((len(manifest.read_lines) + len(manifest.write_lines)) / config.channels),
        },
    )


@dataclass(frozen=True)
class HbmServiceModelV4:
    calibration_id: str
    coefficients: Mapping[str, Mapping[str, float]]
    domains: Mapping[str, Mapping[str, Any]]
    warm_coefficients: Mapping[str, Mapping[str, float]]
    compatibility: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "HbmServiceModelV4":
        data = json.loads(Path(path).read_text())
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported HBM V4 schema {data.get('schema_version')!r}")
        if tuple(data.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("HBM V4 feature schema does not match evaluator")
        if tuple(data.get("warm_feature_names", ())) != WARM_FEATURE_NAMES:
            raise ValueError("HBM V4 warm feature schema does not match evaluator")
        compatibility = data.get("compatibility", {})
        expected = {
            "dma_semantic_version": "production-dma-lines-v2",
            "feature_semantic_version": FEATURE_SEMANTIC_VERSION,
            "request_manifest_hash_algorithm": MANIFEST_HASH_ALGORITHM,
            "request_manifest_fixture_hash": request_manifest_fixture_hash(),
            "request_bytes": REQUEST_BYTES,
            "physical_burst_bytes": PHYSICAL_BURST_BYTES,
            "mapper": "MOP4CLXOR",
            "ramulator_preset": "HBM2_2Gbps",
        }
        mismatch = {key: (compatibility.get(key), value) for key, value in expected.items() if compatibility.get(key) != value}
        if mismatch:
            raise ValueError(f"HBM V4 artifact is incompatible with runtime semantics: {mismatch}")
        return cls(
            calibration_id=str(data["calibration_id"]),
            coefficients=data["coefficients"],
            domains=data.get("domains", {}),
            warm_coefficients=data.get("warm_coefficients", {}),
            compatibility=compatibility,
            metadata=data.get("metadata", {}),
        )

    def predict(
        self,
        opcode: str,
        transfer: DmaTransfer,
        fmt: MemoryFormat,
        config: HbmV4Config,
        manifest: DmaRequestManifest,
        *,
        open_rows: np.ndarray | None = None,
    ) -> HbmV4Prediction:
        vector = occurrence_features(manifest, transfer, config, open_rows=open_rows)
        return self.predict_features(opcode, transfer, fmt, config, vector)

    def predict_features(
        self,
        opcode: str,
        transfer: DmaTransfer,
        fmt: MemoryFormat,
        config: HbmV4Config,
        vector: V4FeatureVector,
    ) -> HbmV4Prediction:
        group = f"{opcode}:c{config.channels}"
        if group not in self.coefficients:
            raise ValueError(f"HBM V4 has no calibrated group {group}")
        zero_miss = (
            bool(vector.values["read_phase_startup"] or vector.values["write_phase_startup"])
            and vector.values["read_row_miss"] == 0
            and vector.values["write_row_miss"] == 0
        )
        fully_warm = (
            zero_miss
            and vector.values["read_initial_row_conflict"] == 0
            and vector.values["write_initial_row_conflict"] == 0
            and group in self.warm_coefficients
        )
        coefficients = self.warm_coefficients[group] if fully_warm else self.coefficients[group]
        feature_names = WARM_FEATURE_NAMES if fully_warm else FEATURE_NAMES
        estimate = vector.theoretical_phase_floor_ns + sum(
            float(coefficients.get(name, 0.0)) * vector.values[name] for name in feature_names
        )
        regime = "fully_warm" if fully_warm else "cold_or_mixed"
        group_domain = self.domains.get(group, {})
        domain = group_domain.get("row_state_regimes", {}).get(regime, group_domain)
        issues: list[str] = []
        ratio = 1.0
        for name, value in vector.values.items():
            limits = domain.get("features", {}).get(name)
            if not limits:
                continue
            lower, upper = float(limits["min"]), float(limits["max"])
            scale = max(1.0, upper - lower, abs(lower), abs(upper))
            tolerance = 0.05 * scale
            if value < lower - tolerance:
                issues.append(f"{name}={value:g}<min={lower:g}")
                ratio = max(ratio, 1 + (lower - value) / scale)
            elif value > upper + tolerance:
                issues.append(f"{name}={value:g}>max={upper:g}")
                ratio = max(ratio, 1 + (value - upper) / scale)
        signatures = set(domain.get("request_signatures", ()))
        if signatures and fmt.request_signature() not in signatures:
            issues.append(f"request_signature={fmt.request_signature()}")
        return HbmV4Prediction(
            latency_ns=max(vector.theoretical_phase_floor_ns, estimate),
            theoretical_phase_floor_ns=vector.theoretical_phase_floor_ns,
            calibration_in_domain=not issues,
            domain_issues=tuple(issues),
            extrapolation_ratio=ratio,
            features=vector.values,
            row_state_regime=regime,
        )
