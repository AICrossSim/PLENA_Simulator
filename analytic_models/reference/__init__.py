"""CPU numerical references used to validate PLENA analytic and RTL models."""

from .kimi_k3_kda import KdaShape, KdaState, activate_log_decay, kda_recurrent_sequence, kda_step
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
    "KdaShape",
    "KdaState",
    "Mamba2Shape",
    "Mamba2State",
    "Mamba2Weights",
    "StateStorage",
    "activate_log_decay",
    "affine_scan_chunked",
    "gated_group_rms_norm",
    "kda_recurrent_sequence",
    "kda_step",
    "mamba_prefill_sequential",
    "mamba_step",
    "quantize_state",
    "selective_state_step",
]
