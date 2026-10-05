"""JSON NoC profiles, built into DeepStack ``Hierarchy`` objects.

A profile is the three-level hierarchy accepted by DeepStack's
``mosaic.noc.custom_profile.make_custom_profile``: ``L3`` (outermost), ``L2``
and ``L1`` (innermost), each with ``kind`` (``switch``, ``ring``, ``chain``,
``mesh2d``, ``torus2d``, ``all_to_all``, ...), ``shape``, ``hop_latency_ns``,
``link_bandwidth_gbytes_per_s`` and, for switches, the optional
``switch_center_in_gbytes_per_s``/``switch_center_out_gbytes_per_s``::

    {
      "schema_version": 1,
      "kind": "noc_hierarchy",
      "name": "...",
      "provenance": {...},
      "layers": {"L3": {...}, "L2": {...}, "L1": {...}},
      "port_spread": "even",                                         # optional
      "energy_pj_per_bit": {"l1": ..., "l2": ..., "l3": ...}         # optional
    }

The device count is the product of the layer sizes; ranks are grouped TP, EP,
SP, CP, DP, PP from the innermost level outwards.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ._deepstack import Hierarchy, NocEnergyConfig, make_custom_profile

SCHEMA_VERSION = 1
_REQUIRED = {"schema_version", "kind", "name", "layers"}
_OPTIONAL = {"provenance", "port_spread", "energy_pj_per_bit"}


@dataclass(frozen=True)
class NocProfile:
    name: str
    hierarchy: Hierarchy
    energy: NocEnergyConfig | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def num_devices(self) -> int:
        return self.hierarchy.num_devices

    def describe(self) -> dict[str, Any]:
        levels = []
        for level, topology in zip(("L3", "L2", "L1"), self.hierarchy.layers, strict=True):
            levels.append(
                {
                    "level": level,
                    "kind": topology.kind.value,
                    "shape": list(topology.shape),
                    "hop_latency_ns": topology.hop_latency * 1e9,
                    "link_bandwidth_gbytes_per_s": topology.link_bandwidth / 1e9,
                }
            )
        energy = None
        if self.energy is not None:
            energy = {"l1": self.energy.l1_pj_per_bit, "l2": self.energy.l2_pj_per_bit, "l3": self.energy.l3_pj_per_bit}
        return {
            "name": self.name,
            "num_devices": self.num_devices,
            "layers": levels,
            "energy_pj_per_bit": energy,
            "provenance": dict(self.provenance),
        }


def noc_from_dict(data: Mapping[str, Any]) -> NocProfile:
    if not isinstance(data, Mapping):
        raise TypeError("a NoC profile must be a JSON object")
    missing = sorted(_REQUIRED - data.keys())
    unknown = sorted(data.keys() - _REQUIRED - _OPTIONAL)
    if missing or unknown:
        raise ValueError(f"NoC profile: missing {missing}, unknown {unknown}")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {data['schema_version']!r}; expected {SCHEMA_VERSION}")
    if data["kind"] != "noc_hierarchy":
        raise ValueError(f"expected kind 'noc_hierarchy', got {data['kind']!r}")
    provenance = data.get("provenance")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a JSON object")

    energy = None
    if data.get("energy_pj_per_bit") is not None:
        coefficients = data["energy_pj_per_bit"]
        if not isinstance(coefficients, Mapping) or set(coefficients) != {"l1", "l2", "l3"}:
            raise ValueError("energy_pj_per_bit must have exactly the keys l1, l2 and l3")
        energy = NocEnergyConfig(
            l1_pj_per_bit=coefficients["l1"],
            l2_pj_per_bit=coefficients["l2"],
            l3_pj_per_bit=coefficients["l3"],
        )
    hierarchy = make_custom_profile(
        layers=data["layers"],
        name=data["name"],
        port_spread=data.get("port_spread", "even"),
        energy_config=energy,
    )
    return NocProfile(name=data["name"], hierarchy=hierarchy, energy=energy, provenance=provenance or {})


def load_noc_profile(path: str | Path) -> NocProfile:
    """Load a NoC profile and stamp its path and SHA-256 into ``provenance``."""

    profile_path = Path(path)
    raw = profile_path.read_bytes()
    profile = noc_from_dict(json.loads(raw))
    provenance = dict(profile.provenance)
    provenance["profile_path"] = str(profile_path)
    provenance["profile_sha256"] = hashlib.sha256(raw).hexdigest()
    return replace(profile, provenance=provenance)
