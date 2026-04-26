"""HardwareManager: registry for simulated HBM/VRAM/MRAM objects."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ._types import *  # noqa: F401,F403
from ._helpers import *  # noqa: F401,F403


class HardwareManager:
    """Registry for simulated HBM/VRAM/MRAM objects and placement metadata.

    This layer tracks hardware-visible objects only. It does not own tensor
    grouping, value/scatter binding policy, or compute semantics.
    """

    def __init__(self, program: "TileTensorProgram") -> None:
        self.program = program
        self.hbm_objects: Dict[str, Dict[str, object]] = {}
        self.vram_objects: Dict[str, Dict[str, object]] = {}
        self.mram_objects: Dict[str, Dict[str, object]] = {}


