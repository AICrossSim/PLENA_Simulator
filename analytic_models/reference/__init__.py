"""CPU numerical references used to validate PLENA analytic and RTL models."""

from .nemotron3_mamba import (
    Mamba2Shape,
    Mamba2State,
    Mamba2Weights,
    affine_scan_chunked,
    gated_group_rms_norm,
    mamba_prefill_sequential,
    mamba_step,
    selective_state_step,
)
from .state_precision import StateStorage, quantize_state

__all__ = [
    "Mamba2Shape",
    "Mamba2State",
    "Mamba2Weights",
    "StateStorage",
    "affine_scan_chunked",
    "gated_group_rms_norm",
    "mamba_prefill_sequential",
    "mamba_step",
    "quantize_state",
    "selective_state_step",
]
