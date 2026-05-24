"""Global FPRAM constant pool (MANAGER_DESIGN.md §3).

Every kernel auto-hoists its FP literals into ``__const_f16_*`` global.fpram
1-slot buffers (frontend/passes/hoist_float_constants.py). In a chained run,
if each kernel allocated constants from FPRAM_USER_BASE independently, kernel
N's scratch could clobber kernel N+1's preloaded constants
([[feedback_chained_fpram_scratch_const_overlap]]). The manager fixes this by:

  * collecting every kernel's hoisted constants up front,
  * allocating them into ONE global FPRAM slot space (same value -> shared
    slot, deduped),
  * handing each kernel back a {const_name: slot} override map, and
  * placing all kernels' FPRAM scratch ABOVE the const ceiling
    (scratch_base), so scratch can never overlap any kernel's constants.

FPRAM slot addresses are *word* indices (each holds one float16); fp_sram.bin
is a flat float16 array indexed by slot. FPRAM_USER_BASE is 32 (mirrors
address_alloc._FPRAM_BASE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping

FPRAM_USER_BASE = 32   # mirrors address_alloc.FPRAM_USER_BASE


def _value_key(v: float) -> int:
    """Dedup key for a constant value. float16 is the storage type, so two
    values that round to the same float16 bit pattern share a slot."""
    import numpy as np
    return int(np.float16(v).view(np.uint16))


@dataclass
class ConstPool:
    base: int = FPRAM_USER_BASE
    # value-key -> slot
    _slot_of_value: Dict[int, int] = field(default_factory=dict)
    _value_of_slot: Dict[int, float] = field(default_factory=dict)
    # kernel name -> {const buffer name -> slot}
    per_kernel: Dict[str, Dict[str, int]] = field(default_factory=dict)
    _next_slot: int = field(default=FPRAM_USER_BASE)

    def __post_init__(self):
        self._next_slot = self.base

    def collect(self, kernel_name: str, hoisted: Mapping[str, float]) -> None:
        """Register one kernel's hoisted constants, allocating shared slots.

        Same value (by float16 bit pattern) reuses an existing slot; new values
        get the next free slot. Idempotent per (kernel, name).
        """
        mapping = self.per_kernel.setdefault(kernel_name, {})
        for name, value in hoisted.items():
            key = _value_key(value)
            slot = self._slot_of_value.get(key)
            if slot is None:
                slot = self._next_slot
                self._next_slot += 1
                self._slot_of_value[key] = slot
                self._value_of_slot[slot] = float(value)
            mapping[name] = slot

    @property
    def const_ceiling(self) -> int:
        """First slot above all allocated constants. Scratch starts here."""
        return self._next_slot

    @property
    def scratch_base(self) -> int:
        """FPRAM_SCRATCH_BASE: all kernels' FPRAM scratch must sit at/above
        this slot so it never overlaps any kernel's constants."""
        return self._next_slot

    def overrides_for(self, kernel_name: str) -> Dict[str, int]:
        """-> AddressAllocConfig.fpram_address_overrides for this kernel."""
        return dict(self.per_kernel.get(kernel_name, {}))

    def write_fp_sram(self, path: str | Path) -> int:
        """Write fp_sram.bin: a float16 array indexed by slot, every allocated
        slot filled with its constant value. One shared file for the whole
        pipeline. Returns the number of slots written."""
        import numpy as np

        if not self._value_of_slot:
            n = 1
        else:
            n = max(self._value_of_slot) + 1
        arr = np.zeros(n, dtype=np.float16)
        for slot, value in self._value_of_slot.items():
            arr[slot] = np.float16(value)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(arr.tobytes())
        return n
