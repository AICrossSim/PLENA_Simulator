"""JSON memory profiles for the latency estimator.

A profile describes one memory system the caller wants to study. Two kinds
exist; all quantities are SI (bytes, bytes/s, Hz, seconds, degC/W, W, pJ/bit):

``stacked_dram``::

    {
      "schema_version": 1,
      "kind": "stacked_dram",
      "name": "...",
      "provenance": {...},                 # free-form: where the numbers came from
      "dram": {...},                       # StackedDramConfig fields
                                           # ("bank_timing" is a nested DramTimingConfig)
      "buffering": {...},                  # optional BufferingPolicy
      "thermal": {...},                    # optional ThermalPolicy
      "energy": {"read_pj_per_bit": ..., "write_pj_per_bit": ...}   # optional
    }

``fixed_bandwidth``::

    {
      "schema_version": 1,
      "kind": "fixed_bandwidth",
      "name": "...",
      "provenance": {...},
      "bandwidth_bytes_per_s": ...,
      "capacity_bytes": ... | null,
      "transaction_bytes": ...,            # optional
      "energy": {...}                      # optional
    }

Unknown keys are rejected so that a misspelt field cannot be silently ignored.
Loading records the profile path and its SHA-256 in ``provenance``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import MISSING, fields, replace
from pathlib import Path
from typing import Any

from .config import BufferingPolicy, DramEnergyConfig, DramTimingConfig, StackedDramConfig, ThermalPolicy
from .memory import FixedBandwidthMemory
from .model import StackedDramModel

SCHEMA_VERSION = 1

MemoryProfile = StackedDramModel | FixedBandwidthMemory


def _check_keys(data: Mapping[str, Any], *, required: set[str], optional: set[str], where: str) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{where} must be a JSON object")
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f"{where} is missing {', '.join(missing)}")
    unknown = sorted(data.keys() - required - optional)
    if unknown:
        raise ValueError(f"{where} has unknown keys {', '.join(unknown)}")


def _dataclass_from(cls: type, data: Mapping[str, Any], *, where: str) -> Any:
    names = {item.name for item in fields(cls)}
    required = {item.name for item in fields(cls) if item.default is MISSING and item.default_factory is MISSING}
    _check_keys(data, required=required, optional=names - required, where=where)
    return cls(**data)


def _optional(cls: type, data: Mapping[str, Any], key: str) -> Any:
    value = data.get(key)
    return None if value is None else _dataclass_from(cls, value, where=key)


def _stacked_dram(data: Mapping[str, Any]) -> StackedDramModel:
    _check_keys(
        data,
        required={"schema_version", "kind", "name", "dram"},
        optional={"provenance", "buffering", "thermal", "energy"},
        where="stacked_dram profile",
    )
    if not isinstance(data["dram"], Mapping):
        raise TypeError("dram must be a JSON object")
    dram = dict(data["dram"])
    if dram.get("bank_timing") is not None:
        dram["bank_timing"] = _dataclass_from(DramTimingConfig, dram["bank_timing"], where="dram.bank_timing")
    return StackedDramModel(
        name=data["name"],
        config=_dataclass_from(StackedDramConfig, dram, where="dram"),
        buffering=_optional(BufferingPolicy, data, "buffering"),
        thermal=_optional(ThermalPolicy, data, "thermal"),
        energy=_optional(DramEnergyConfig, data, "energy"),
        provenance=data.get("provenance") or {},
    )


def _fixed_bandwidth(data: Mapping[str, Any]) -> FixedBandwidthMemory:
    _check_keys(
        data,
        required={"schema_version", "kind", "name", "bandwidth_bytes_per_s", "capacity_bytes"},
        optional={"provenance", "transaction_bytes", "energy"},
        where="fixed_bandwidth profile",
    )
    return FixedBandwidthMemory(
        name=data["name"],
        bandwidth_bytes_per_s=data["bandwidth_bytes_per_s"],
        capacity_bytes=data["capacity_bytes"],
        transaction_bytes=data.get("transaction_bytes"),
        energy=_optional(DramEnergyConfig, data, "energy"),
        provenance=data.get("provenance") or {},
    )


_BUILDERS = {
    StackedDramModel.kind: _stacked_dram,
    FixedBandwidthMemory.kind: _fixed_bandwidth,
}


def memory_from_dict(data: Mapping[str, Any]) -> MemoryProfile:
    """Build a memory system from a parsed profile."""

    if not isinstance(data, Mapping):
        raise TypeError("a memory profile must be a JSON object")
    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version!r}; expected {SCHEMA_VERSION}")
    kind = data.get("kind")
    if kind not in _BUILDERS:
        raise ValueError(f"unknown memory kind {kind!r}; expected one of {', '.join(sorted(_BUILDERS))}")
    provenance = data.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a JSON object")
    return _BUILDERS[kind](data)


def load_memory_profile(path: str | Path) -> MemoryProfile:
    """Load a profile file and stamp its path and SHA-256 into ``provenance``."""

    profile_path = Path(path)
    raw = profile_path.read_bytes()
    memory = memory_from_dict(json.loads(raw))
    provenance = dict(memory.provenance)
    provenance["profile_path"] = str(profile_path)
    provenance["profile_sha256"] = hashlib.sha256(raw).hexdigest()
    return replace(memory, provenance=provenance)
