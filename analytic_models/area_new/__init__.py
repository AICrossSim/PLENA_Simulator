"""Public API for the precision-aware PLENA area proxy.

The package estimates synthesized logic and SRAM macro area in square
micrometres (um^2). It is intended for relative DSE ranking and early chip
sizing. It is not a replacement for placed-and-routed signoff area.
"""

from .area_proxy import estimate_area, estimate_matrix_machine_area
from .hbm_model import estimate_hbm_system_area
from .precision import PrecisionError, derive_compute_sides, parse_precision
from .scalar_model import estimate_scalar_machine_area
from .sram_model import estimate_sram_area
from .top_model import estimate_full_chip_top_residual
from .vector_model import estimate_vector_machine_area

__all__ = [
    "PrecisionError",
    "derive_compute_sides",
    "estimate_hbm_system_area",
    "estimate_area",
    "estimate_matrix_machine_area",
    "estimate_scalar_machine_area",
    "estimate_sram_area",
    "estimate_full_chip_top_residual",
    "estimate_vector_machine_area",
    "parse_precision",
]
