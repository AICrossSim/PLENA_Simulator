"""ManagedTensor + HbmLayout: per-tensor shape+address bookkeeping.

The manager owns a flat HBM address space. Each tensor is placed at a concrete
byte offset whose stride is the *real* packed size (see binio.packed_byte_size,
which matches what the emulator's packer actually writes — not the compiler's
_hbm_packed_byte_size, which disagrees for non-64-aligned sizes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional

from .geometry import BehaviorSettings
from .binio import packed_byte_size, scale_region_offset


class Role(Enum):
    WEIGHT = "weight"          # read-only, reused across kernels/layers
    ACTIVATION = "activation"  # produced then consumed; producer.addr==consumer.addr
    SCRATCH = "scratch"        # transient; address reserved, no preset data
    IO = "io"                  # pipeline-level input / output


@dataclass
class ManagedTensor:
    name: str
    shape: tuple[int, ...]
    hbm_addr: int
    role: Role
    data: object = None        # torch.Tensor | None (None for SCRATCH)

    @property
    def num_elements(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n

    def packed_bytes(self, settings: BehaviorSettings) -> int:
        return packed_byte_size(self.num_elements, settings)

    def scale_offset(self, settings: BehaviorSettings) -> int:
        """Byte offset from this tensor's start to its scale region."""
        return scale_region_offset(self.num_elements, settings)


class HbmLayout:
    """Bump-allocated flat HBM address space the manager owns.

    Not the compiler's bump cursor — the manager pre-plans every address and
    feeds them back as hbm_address_overrides (MANAGER_DESIGN.md §1.2). Pinned
    and aliased tensors do not advance the cursor.
    """

    def __init__(self, settings: BehaviorSettings, base: int = 0):
        self.settings = settings
        self.base = int(base)
        self.cursor = int(base)
        self.tensors: Dict[str, ManagedTensor] = {}

    def place(self, name: str, shape, role: Role, data=None) -> ManagedTensor:
        """Bump-allocate a new tensor at the current cursor."""
        if name in self.tensors:
            raise ValueError(f"tensor {name!r} already placed")
        t = ManagedTensor(name=name, shape=tuple(int(s) for s in shape),
                          hbm_addr=self.cursor, role=role, data=data)
        self.cursor += t.packed_bytes(self.settings)
        self.tensors[name] = t
        return t

    def pin(self, name: str, shape, role: Role, addr: int, data=None) -> ManagedTensor:
        """Place a tensor at a fixed address; does NOT advance the cursor."""
        if name in self.tensors:
            raise ValueError(f"tensor {name!r} already placed")
        t = ManagedTensor(name=name, shape=tuple(int(s) for s in shape),
                          hbm_addr=int(addr), role=role, data=data)
        self.tensors[name] = t
        return t

    def alias(self, new_name: str, existing: str, *, shape=None,
              role: Optional[Role] = None, data=None) -> ManagedTensor:
        """Make ``new_name`` share an existing tensor's HBM address (reuse)."""
        if existing not in self.tensors:
            raise KeyError(f"cannot alias unknown tensor {existing!r}")
        base = self.tensors[existing]
        t = ManagedTensor(
            name=new_name,
            shape=tuple(int(s) for s in (shape if shape is not None else base.shape)),
            hbm_addr=base.hbm_addr,
            role=role if role is not None else base.role,
            data=data,
        )
        self.tensors[new_name] = t
        return t

    def overrides(self) -> Dict[str, int]:
        """-> AddressAllocConfig.hbm_address_overrides (name -> byte addr)."""
        return {name: t.hbm_addr for name, t in self.tensors.items()}

    def total_bytes(self) -> int:
        """Upper bound of used HBM (max addr + its packed size)."""
        if not self.tensors:
            return self.base
        return max(t.hbm_addr + t.packed_bytes(self.settings)
                  for t in self.tensors.values())
