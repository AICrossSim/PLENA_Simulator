"""Versioned calibration artifacts for compiler-action power models."""

from pathlib import Path


CALIBRATION_DIR = Path(__file__).resolve().parent
DEFAULT_EXTERNAL_HBM3E = CALIBRATION_DIR / "external_memory_hbm3e_v1.json"
DEFAULT_LOGIC_ENERGY = CALIBRATION_DIR / "logic_energy_main_v1.json"
DEFAULT_POWER_VALIDATION = CALIBRATION_DIR / "power_validation_main_v1.json"
DEFAULT_SRAM_ENERGY = CALIBRATION_DIR / "sram_energy_asap7_v1.json"


__all__ = [
    "CALIBRATION_DIR",
    "DEFAULT_EXTERNAL_HBM3E",
    "DEFAULT_LOGIC_ENERGY",
    "DEFAULT_POWER_VALIDATION",
    "DEFAULT_SRAM_ENERGY",
]
