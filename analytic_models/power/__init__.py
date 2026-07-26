"""PLENA on-chip and HBM3E-equivalent system power models.

The on-chip estimator remains available independently.  The system estimator
adds a literature-parameterized external HBM3E layer while retaining explicit
scope and measurement-boundary metadata.
"""

from .external_memory import estimate_external_hbm_power
from .multi_chip import (
    DEFAULT_INTERCONNECT_ENERGY,
    estimate_multi_chip_system_power,
)
from .power_model import estimate_onchip_power
from .system_power import estimate_system_power

__all__ = [
    "DEFAULT_INTERCONNECT_ENERGY",
    "estimate_external_hbm_power",
    "estimate_multi_chip_system_power",
    "estimate_onchip_power",
    "estimate_system_power",
]
