"""Small CLI helpers shared by formal DSE launchers."""

from __future__ import annotations

import argparse
from typing import Any

from .profiles import CURRENT_DSE_PROFILE, RTL_VALIDATION_PROFILE


MODEL_PROFILE_NAMES = (
    CURRENT_DSE_PROFILE.name,
    RTL_VALIDATION_PROFILE.name,
    "custom",
)


def add_model_profile_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-profile",
        choices=MODEL_PROFILE_NAMES,
        default=CURRENT_DSE_PROFILE.name,
        help=(
            "Named compatible model stack. The formal DSE default is "
            "current-dse-v1; rtl-validation-v1 is reserved for detailed "
            "single-chip sensitivity runs."
        ),
    )


def model_profile_consistency(args: Any) -> tuple[bool, tuple[str, ...]]:
    if args.model_profile == "custom":
        return True, ()
    profile = (
        CURRENT_DSE_PROFILE
        if args.model_profile == CURRENT_DSE_PROFILE.name
        else RTL_VALIDATION_PROFILE
    )
    expected = {
        "compiler_compute_timing": profile.compute_timing,
        "compiler_trace_granularity": profile.cost_trace_granularity,
        "multi_chip_model": profile.multi_chip_model,
        "clock_gating_mode": profile.clock_gating_mode,
        "vector_scalar_schedule": profile.vector_scalar_schedule,
        "softmax_vector_schedule": profile.softmax_vector_schedule,
        "softmax_state_schedule": profile.softmax_state_schedule,
        "pv_accumulation_schedule": profile.pv_accumulation_schedule,
    }
    mismatches = tuple(
        f"{name}={getattr(args, name)!r} (expected {value!r})"
        for name, value in expected.items()
        if getattr(args, name) != value
    )
    return not mismatches, mismatches
