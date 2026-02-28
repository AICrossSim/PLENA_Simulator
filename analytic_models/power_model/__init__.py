"""
PLENA Power Model Package

Analytical power and area models for the PLENA accelerator.
Models are fitted from Synopsys DC synthesis results.

Components:
- MCU (Matrix Compute Unit) - Systolic array
- Vector Unit - FP elementwise and reduction operations
- Matrix SRAM - Weight storage with rescaling
- Vector SRAM - Activation storage with MXFP conversion

Usage:
    from power_model import PLENAConfig, estimate_plena_power, estimate_plena_area

    config = PLENAConfig(M=8, K=32, VLEN=64, MLEN=64)
    power = estimate_plena_power(config)  # Returns dict with power in mW
    area = estimate_plena_area(config)    # Returns dict with area in um²
"""

from .plena_power_model import (
    PLENAConfig,
    estimate_plena_power,
    estimate_plena_area,
    estimate_mcu_power,
    estimate_mcu_area,
    estimate_vector_unit_power,
    estimate_vector_unit_area,
    estimate_matrix_sram_power,
    estimate_matrix_sram_area,
    estimate_vector_sram_power,
    estimate_vector_sram_area,
    VALIDATION_DATA,
)

__all__ = [
    "PLENAConfig",
    "estimate_plena_power",
    "estimate_plena_area",
    "estimate_mcu_power",
    "estimate_mcu_area",
    "estimate_vector_unit_power",
    "estimate_vector_unit_area",
    "estimate_matrix_sram_power",
    "estimate_matrix_sram_area",
    "estimate_vector_sram_power",
    "estimate_vector_sram_area",
    "VALIDATION_DATA",
]
