#!/usr/bin/env python3
"""
PLENA Analytical Power and Area Model

This model estimates power and area for the PLENA accelerator by summing
contributions from four main components:

1. Matrix Compute Unit (MCU) - Systolic array for matrix multiplication
2. Vector Unit - FP elementwise and reduction compute units
3. Matrix SRAM - Storage for weight matrices with rescaling
4. Vector SRAM - Storage for activation vectors with MXFP conversion

Models are fitted from Synopsys Design Compiler synthesis results using
polynomial regression on parameterized RTL configurations.

Methodology:
-----------
1. Generate parameterized RTL wrappers for each component
2. Run DC synthesis with consistent constraints (1ns clock period)
3. Extract power and area from synthesis reports
4. Fit polynomial/power-law models to the data points

Limitations:
-----------
- MCU power model uses only M=4 data (M>=8 had RTL bug with inflated switching)
- Vector compute power is under-estimated due to lack of switching activity
- SRAM power assumes default toggle rates from DC estimation
- Models are for combinational/sequential power, not including clock network

Usage:
------
    from plena_power_model import PLENAConfig, estimate_plena_power, estimate_plena_area

    config = PLENAConfig(M=8, K=32, VLEN=64, MLEN=64)
    power = estimate_plena_power(config)
    area = estimate_plena_area(config)
"""

from dataclasses import dataclass
from typing import Dict
import math


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PLENAConfig:
    """PLENA accelerator configuration parameters."""
    # MCU dimensions
    M: int = 8          # Systolic array dimension (M x M PEs)
    K: int = 32         # Reduction dimension (K elements per PE)

    # Vector/Matrix dimensions
    VLEN: int = 64      # Vector length (parallel FP lanes)
    MLEN: int = 64      # Matrix dimension (for SRAM width)

    # SRAM depths
    VSRAM_DEPTH: int = 1024    # Vector SRAM depth (entries)
    MSRAM_DEPTH: int = 1024    # Matrix SRAM depth (entries)

    # Instance counts
    num_mcu: int = 1
    num_vector_unit: int = 1
    num_matrix_sram: int = 1
    num_vector_sram: int = 1


# ============================================================================
# MCU (Matrix Compute Unit) Model
# ============================================================================
# Fitted from synthesis of mx_systolic_mcu with various M, K configurations
# Using only M=4 data due to RTL bug in M>=8 (now fixed, needs re-synthesis)
#
# Power: Scales as M^2 * K (systolic array compute complexity)
# Area: Power law fit with R^2 = 0.999

_MCU_VALID_M_RANGE = (4, 32)
_MCU_VALID_K_RANGE = (8, 128)

# Synthesis data points (M=4 only - reliable data)
# M=4, K=8:  544 mW, 4746 um²
# M=4, K=16: 1080 mW, 10783 um²
# M=4, K=32: 2210 mW, 23183 um²


def estimate_mcu_power(M: float, K: float) -> float:
    """
    Estimate MCU power consumption.

    Model: Scaled from M=4 synthesis data using M^2 * K complexity
    Base power: ~68 mW per K element at M=4

    Args:
        M: Systolic array dimension (M x M PEs)
        K: Reduction dimension

    Returns:
        Power in Watts
    """
    # Base power coefficient derived from M=4 data
    # M=4, K=8:  544 mW  -> 544/(4^2 * 8)  = 4.25 mW per M^2*K
    # M=4, K=16: 1080 mW -> 1080/(4^2 * 16) = 4.22 mW per M^2*K
    # M=4, K=32: 2210 mW -> 2210/(4^2 * 32) = 4.32 mW per M^2*K
    # Average: ~4.26 mW per M^2*K unit
    power_per_unit = 4.26e-3  # W per (M^2 * K) unit
    return power_per_unit * (M ** 2) * K


def estimate_mcu_area(M: float, K: float) -> float:
    """
    Estimate MCU area.

    Model: Power law Z = a * M^b * K^c
    Fitted coefficients: a=154.0, b=0.772, c=1.138
    R² = 0.999

    Args:
        M: Systolic array dimension
        K: Reduction dimension

    Returns:
        Area in μm²
    """
    return 1.5399614657e+02 * (M ** 0.7720723827) * (K ** 1.1375970695)


# ============================================================================
# Vector Unit (Elementwise + Reduction) Model
# ============================================================================
# Fitted from synthesis of vector_compute wrappers (VLEN = 8, 16, 32, 64)
#
# Note: Synthesis power is under-estimated due to lack of switching activity
# annotation. The values below are from synthesis reports but actual power
# will be higher during operation.
#
# Synthesis data:
# VLEN=8:  4.3 mW (synth), 6772 um²
# VLEN=16: 8.6 mW (synth), 14161 um²
# VLEN=32: 17.2 mW (synth), 28016 um²
# VLEN=64: 34.0 mW (synth), 53125 um²

_VECTOR_VALID_VLEN_RANGE = (8, 128)


def estimate_vector_unit_power(VLEN: float, activity_factor: float = 1.0) -> float:
    """
    Estimate Vector Unit power consumption.

    Model: Polynomial fit to synthesis data, with optional activity scaling

    The base model is from synthesis without switching activity annotation.
    Use activity_factor > 1.0 to estimate realistic operational power.
    Typical activity_factor for compute: 2-5x synthesis estimate.

    Args:
        VLEN: Vector length (parallel FP lanes)
        activity_factor: Multiplier for switching activity (default 1.0)

    Returns:
        Power in Watts
    """
    # Polynomial fit: P = a*VLEN^2 + b*VLEN + c
    # R² = 0.999998
    base_power = (-1.7746135753e-07 * VLEN**2 +
                   5.4279213710e-04 * VLEN +
                  -4.9166666667e-05)
    return max(0, base_power * activity_factor)


def estimate_vector_unit_area(VLEN: float) -> float:
    """
    Estimate Vector Unit area.

    Model: Polynomial fit
    R² = 0.999997

    Args:
        VLEN: Vector length

    Returns:
        Area in μm²
    """
    # Polynomial fit: A = a*VLEN^2 + b*VLEN + c
    area = (-1.8077767520e+00 * VLEN**2 +
             9.5733197768e+02 * VLEN +
            -7.4309888100e+02)
    return max(0, area)


# ============================================================================
# Matrix SRAM Model
# ============================================================================
# Fitted from synthesis of matrix_sram_with_rounding (MLEN = 8, 16, 32, 48, 64)
# at DEPTH=128, scaled linearly for other depths
#
# Synthesis data:
# MLEN=8:  20.4 mW, 9528 um²
# MLEN=16: 40.8 mW, 19293 um²
# MLEN=32: 82.4 mW, 40025 um²
# MLEN=48: 123.7 mW, 61776 um²
# MLEN=64: 167.7 mW, 85874 um²

_MSRAM_VALID_MLEN_RANGE = (8, 128)


def estimate_matrix_sram_power(MLEN: float, DEPTH: float = 128) -> float:
    """
    Estimate Matrix SRAM power consumption.

    Model: Polynomial fit at DEPTH=128, linearly scaled for other depths
    R² = 0.999966

    Args:
        MLEN: Matrix dimension (SRAM width)
        DEPTH: SRAM depth (number of entries)

    Returns:
        Power in Watts
    """
    depth_scale = DEPTH / 128.0
    # Polynomial fit: P = a*MLEN^2 + b*MLEN + c
    base_power = (1.9583044211e-06 * MLEN**2 +
                  2.4835249306e-03 * MLEN +
                  5.2254044409e-04)
    return base_power * depth_scale


def estimate_matrix_sram_area(MLEN: float, DEPTH: float = 128) -> float:
    """
    Estimate Matrix SRAM area.

    Model: Polynomial fit at DEPTH=128, linearly scaled for other depths
    R² = 0.999974

    Args:
        MLEN: Matrix dimension
        DEPTH: SRAM depth

    Returns:
        Area in μm²
    """
    depth_scale = DEPTH / 128.0
    # Polynomial fit: A = a*MLEN^2 + b*MLEN + c
    base_area = (3.1301163646e+00 * MLEN**2 +
                 1.1353823744e+03 * MLEN +
                 3.0247391488e+02)
    return base_area * depth_scale


# ============================================================================
# Vector SRAM (FP Vector SRAM) Model
# ============================================================================
# Fitted from synthesis of fp_vector_sram (VLEN = 8, 16, 32, 48, 64)
# at DEPTH=128, scaled linearly for other depths
#
# Note: Power is dominated by switching activity in the SRAM arrays
#
# Synthesis data:
# VLEN=8:  248.0 mW, 2059 um²
# VLEN=16: 250.7 mW, 3984 um²
# VLEN=32: 256.2 mW, 7821 um²
# VLEN=48: 261.7 mW, 11665 um²
# VLEN=64: 267.1 mW, 15492 um²

_VSRAM_VALID_VLEN_RANGE = (8, 128)


def estimate_vector_sram_power(VLEN: float, DEPTH: float = 128) -> float:
    """
    Estimate Vector SRAM power consumption.

    Model: Polynomial fit at DEPTH=128, linearly scaled for other depths
    R² = 0.999996

    Args:
        VLEN: Vector length (SRAM width)
        DEPTH: SRAM depth (number of entries)

    Returns:
        Power in Watts
    """
    depth_scale = DEPTH / 128.0
    # Polynomial fit: P = a*VLEN^2 + b*VLEN + c
    base_power = (-3.1305759318e-08 * VLEN**2 +
                   3.4435611618e-04 * VLEN +
                   2.4520172086e-01)
    return base_power * depth_scale


def estimate_vector_sram_area(VLEN: float, DEPTH: float = 128) -> float:
    """
    Estimate Vector SRAM area.

    Model: Polynomial fit at DEPTH=128, linearly scaled for other depths
    R² = 1.000000

    Args:
        VLEN: Vector length
        DEPTH: SRAM depth

    Returns:
        Area in μm²
    """
    depth_scale = DEPTH / 128.0
    # Polynomial fit: A = a*VLEN^2 + b*VLEN + c
    base_area = (-1.1014309483e-02 * VLEN**2 +
                  2.4067861081e+02 * VLEN +
                  1.3453437699e+02)
    return base_area * depth_scale


# ============================================================================
# Combined PLENA Model
# ============================================================================

def estimate_plena_power(config: PLENAConfig,
                         vector_activity: float = 1.0) -> Dict[str, float]:
    """
    Estimate total PLENA accelerator power consumption.

    Args:
        config: PLENAConfig with all parameters
        vector_activity: Activity factor for vector unit (default 1.0)

    Returns:
        Dictionary with power breakdown in milliWatts:
        - mcu_power_mw: Matrix Compute Unit power
        - vector_unit_power_mw: Vector Unit power
        - matrix_sram_power_mw: Matrix SRAM power
        - vector_sram_power_mw: Vector SRAM power
        - total_power_mw: Sum of all components
    """
    # Component powers (in Watts)
    mcu = estimate_mcu_power(config.M, config.K) * config.num_mcu
    vec = estimate_vector_unit_power(config.VLEN, vector_activity) * config.num_vector_unit
    msram = estimate_matrix_sram_power(config.MLEN, config.MSRAM_DEPTH) * config.num_matrix_sram
    vsram = estimate_vector_sram_power(config.VLEN, config.VSRAM_DEPTH) * config.num_vector_sram

    total = mcu + vec + msram + vsram

    # Convert to milliWatts
    return {
        "mcu_power_mw": mcu * 1000,
        "vector_unit_power_mw": vec * 1000,
        "matrix_sram_power_mw": msram * 1000,
        "vector_sram_power_mw": vsram * 1000,
        "total_power_mw": total * 1000,
    }


def estimate_plena_area(config: PLENAConfig) -> Dict[str, float]:
    """
    Estimate total PLENA accelerator area.

    Args:
        config: PLENAConfig with all parameters

    Returns:
        Dictionary with area breakdown in μm²:
        - mcu_area: Matrix Compute Unit area
        - vector_unit_area: Vector Unit area
        - matrix_sram_area: Matrix SRAM area
        - vector_sram_area: Vector SRAM area
        - total_area: Sum of all components
    """
    mcu = estimate_mcu_area(config.M, config.K) * config.num_mcu
    vec = estimate_vector_unit_area(config.VLEN) * config.num_vector_unit
    msram = estimate_matrix_sram_area(config.MLEN, config.MSRAM_DEPTH) * config.num_matrix_sram
    vsram = estimate_vector_sram_area(config.VLEN, config.VSRAM_DEPTH) * config.num_vector_sram

    total = mcu + vec + msram + vsram

    return {
        "mcu_area": mcu,
        "vector_unit_area": vec,
        "matrix_sram_area": msram,
        "vector_sram_area": vsram,
        "total_area": total,
    }


# ============================================================================
# Validation Data
# ============================================================================

VALIDATION_DATA = {
    "mcu": [
        {"M": 4, "K": 8, "power_mw": 543.754, "area": 4745.51},
        {"M": 4, "K": 16, "power_mw": 1080.0, "area": 10783.47},
        {"M": 4, "K": 32, "power_mw": 2210.0, "area": 23182.83},
    ],
    "vector_unit": [
        {"VLEN": 8, "power_mw": 4.295, "area": 6771.81},
        {"VLEN": 16, "power_mw": 8.567, "area": 14160.50},
        {"VLEN": 32, "power_mw": 17.150, "area": 28015.82},
        {"VLEN": 64, "power_mw": 33.961, "area": 53125.00},
    ],
    "matrix_sram": [
        {"MLEN": 8, "depth": 128, "power_mw": 20.374, "area": 9527.69},
        {"MLEN": 16, "depth": 128, "power_mw": 40.834, "area": 19292.94},
        {"MLEN": 32, "depth": 128, "power_mw": 82.401, "area": 40025.31},
        {"MLEN": 48, "depth": 128, "power_mw": 123.719, "area": 61776.36},
        {"MLEN": 64, "depth": 128, "power_mw": 167.682, "area": 85873.92},
    ],
    "vector_sram": [
        {"VLEN": 8, "depth": 128, "power_mw": 247.956, "area": 2059.02},
        {"VLEN": 16, "depth": 128, "power_mw": 250.708, "area": 3984.23},
        {"VLEN": 32, "depth": 128, "power_mw": 256.169, "area": 7821.05},
        {"VLEN": 48, "depth": 128, "power_mw": 261.680, "area": 11665.44},
        {"VLEN": 64, "depth": 128, "power_mw": 267.105, "area": 15491.64},
    ],
}


# ============================================================================
# Main - Demonstration and Validation
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PLENA Analytical Power/Area Model")
    print("=" * 70)
    print()
    print("Components: MCU + Vector Unit + Matrix SRAM + Vector SRAM")
    print()

    # Validate individual component models
    print("Model Validation (Predicted vs Actual from Synthesis)")
    print("-" * 70)

    print("\n1. MCU (M=4 data only - M>=8 excluded due to RTL bug):")
    print(f"   {'M':>4} {'K':>5} {'Actual':>12} {'Predicted':>12} {'Error':>8}")
    for d in VALIDATION_DATA["mcu"]:
        pred = estimate_mcu_power(d["M"], d["K"]) * 1000
        err = abs(pred - d["power_mw"]) / d["power_mw"] * 100
        print(f"   {d['M']:>4} {d['K']:>5} {d['power_mw']:>10.1f} mW {pred:>10.1f} mW {err:>6.1f}%")

    print("\n2. Vector Unit (synthesis power - no activity annotation):")
    print(f"   {'VLEN':>6} {'Actual':>12} {'Predicted':>12} {'Error':>8}")
    for d in VALIDATION_DATA["vector_unit"]:
        pred = estimate_vector_unit_power(d["VLEN"]) * 1000
        err = abs(pred - d["power_mw"]) / d["power_mw"] * 100
        print(f"   {d['VLEN']:>6} {d['power_mw']:>10.3f} mW {pred:>10.3f} mW {err:>6.2f}%")

    print("\n3. Matrix SRAM (DEPTH=128):")
    print(f"   {'MLEN':>6} {'Actual':>12} {'Predicted':>12} {'Error':>8}")
    for d in VALIDATION_DATA["matrix_sram"]:
        pred = estimate_matrix_sram_power(d["MLEN"], d["depth"]) * 1000
        err = abs(pred - d["power_mw"]) / d["power_mw"] * 100
        print(f"   {d['MLEN']:>6} {d['power_mw']:>10.3f} mW {pred:>10.3f} mW {err:>6.2f}%")

    print("\n4. Vector SRAM (DEPTH=128):")
    print(f"   {'VLEN':>6} {'Actual':>12} {'Predicted':>12} {'Error':>8}")
    for d in VALIDATION_DATA["vector_sram"]:
        pred = estimate_vector_sram_power(d["VLEN"], d["depth"]) * 1000
        err = abs(pred - d["power_mw"]) / d["power_mw"] * 100
        print(f"   {d['VLEN']:>6} {d['power_mw']:>10.3f} mW {pred:>10.3f} mW {err:>6.2f}%")

    # Example configurations
    print("\n" + "=" * 70)
    print("Example PLENA Configurations")
    print("=" * 70)

    configs = [
        PLENAConfig(M=8, K=32, VLEN=64, MLEN=64),
        PLENAConfig(M=16, K=64, VLEN=64, MLEN=64),
        PLENAConfig(M=8, K=32, VLEN=128, MLEN=128, VSRAM_DEPTH=2048, MSRAM_DEPTH=2048),
    ]

    for i, cfg in enumerate(configs):
        print(f"\nConfig {i+1}: M={cfg.M}, K={cfg.K}, VLEN={cfg.VLEN}, MLEN={cfg.MLEN}")
        print(f"          VSRAM_DEPTH={cfg.VSRAM_DEPTH}, MSRAM_DEPTH={cfg.MSRAM_DEPTH}")
        print("-" * 50)

        power = estimate_plena_power(cfg)
        area = estimate_plena_area(cfg)

        print("Power Breakdown:")
        print(f"  MCU:          {power['mcu_power_mw']:>10.2f} mW ({power['mcu_power_mw']/power['total_power_mw']*100:>5.1f}%)")
        print(f"  Vector Unit:  {power['vector_unit_power_mw']:>10.2f} mW ({power['vector_unit_power_mw']/power['total_power_mw']*100:>5.1f}%)")
        print(f"  Matrix SRAM:  {power['matrix_sram_power_mw']:>10.2f} mW ({power['matrix_sram_power_mw']/power['total_power_mw']*100:>5.1f}%)")
        print(f"  Vector SRAM:  {power['vector_sram_power_mw']:>10.2f} mW ({power['vector_sram_power_mw']/power['total_power_mw']*100:>5.1f}%)")
        print(f"  TOTAL:        {power['total_power_mw']:>10.2f} mW ({power['total_power_mw']/1000:.3f} W)")

        print("Area Breakdown:")
        print(f"  MCU:          {area['mcu_area']:>12.0f} um²")
        print(f"  Vector Unit:  {area['vector_unit_area']:>12.0f} um²")
        print(f"  Matrix SRAM:  {area['matrix_sram_area']:>12.0f} um²")
        print(f"  Vector SRAM:  {area['vector_sram_area']:>12.0f} um²")
        print(f"  TOTAL:        {area['total_area']:>12.0f} um² ({area['total_area']/1e6:.4f} mm²)")

    print("\n" + "=" * 70)
    print("Notes:")
    print("- MCU power model scaled from M=4 synthesis data only")
    print("- Vector Unit power is under-estimated (no switching activity)")
    print("- SRAM power scales linearly with depth")
    print("- All models fitted with R² > 0.999")
    print("=" * 70)
