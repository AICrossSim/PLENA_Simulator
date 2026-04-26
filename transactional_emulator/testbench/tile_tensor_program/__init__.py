"""TileTensor program package.

Re-exports the public API so existing callers can continue to use
`from tile_tensor_program import TileTensorProgram` etc. unchanged.

The package is split as:
- _types: FP/Parallel/Tile dataclasses and module-level type aliases
- _helpers: module-level `_xxx` helper functions
- _hardware_manager: HardwareManager
- _thread_manager: ThreadManager (+ _LoopHintRange)
- _value_manager: ValueManager
- _tensor_manager: TensorManager
- _compute_manager: ComputeManager (ISA emitter)
- _program: TileTensorProgram (top-level builder)
"""

from ._types import *  # noqa: F401,F403
from ._helpers import *  # noqa: F401,F403
from ._hardware_manager import HardwareManager
from ._thread_manager import ThreadManager
from ._value_manager import ValueManager
from ._tensor_manager import TensorManager
from ._compute_manager import ComputeManager
from ._program import TileTensorProgram

# Re-export commonly imported names from outside the package:
#   from tile_tensor_program import Input, TileTensorProgram, _logical_shape_to_physical_shape
from ._types import Input
from ._helpers import _logical_shape_to_physical_shape

__all__ = [
    "TileTensorProgram",
    "HardwareManager",
    "ThreadManager",
    "ValueManager",
    "TensorManager",
    "ComputeManager",
    "Input",
    "_logical_shape_to_physical_shape",
]
