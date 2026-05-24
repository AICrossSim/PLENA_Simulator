"""PLENA multi-kernel manager.

A host-side orchestration layer that sits *below* the tilelang/TIR frontend
and *above* the PLENA emulator. It owns a flat HBM byte image
(``hbm_for_behave_sim.bin``): every tensor has a shape and a concrete HBM byte
address, and can be written / read / compared independently via byte ``seek``
(no append-order coupling). See ``PLENA_Simulator/MANAGER_DESIGN.md``.

Step 2 (this commit) implements the data/address layer only:
  * ``BehaviorSettings`` / ``addr_cfg_from_toml`` — geometry + MX precision read
    from ``plena_settings.toml`` ``[BEHAVIOR]`` (single source of truth).
  * ``ManagedTensor`` / ``HbmLayout`` — per-tensor shape+addr bookkeeping.
  * ``write_tensor`` / ``read_tensor`` — seek-based MX pack/unpack against the
    bin, byte-compatible with the legacy append packer.
"""

from .geometry import BehaviorSettings, addr_cfg_from_toml, load_behavior_settings
from .tensor import ManagedTensor, Role, HbmLayout
from .binio import write_tensor, read_tensor, packed_byte_size
from .runner import compile_kernel, CompiledKernel
from .const_pool import ConstPool, FPRAM_USER_BASE
from .pipeline import Manager, CompareResult

__all__ = [
    "Manager",
    "CompareResult",
    "ConstPool",
    "FPRAM_USER_BASE",
    "BehaviorSettings",
    "addr_cfg_from_toml",
    "load_behavior_settings",
    "ManagedTensor",
    "Role",
    "HbmLayout",
    "write_tensor",
    "read_tensor",
    "packed_byte_size",
    "compile_kernel",
    "CompiledKernel",
]
