"""Canonical conditional hardware-domain helpers."""

from __future__ import annotations

import math
from typing import Any, Callable


DEFAULT_MLEN_VALUES = (256, 512, 1024, 2048, 4096, 8192)
CHIP_COUNT_SCALING_MODES = ("per-a100-reference", "absolute")

LEGAL_BLENS_BY_MLEN = {
    256: (32, 64, 128, 256),
    512: (32, 64, 128, 256, 512),
    1024: (32, 64, 128, 256, 512, 1024),
    2048: (32, 64, 128, 256, 512, 1024),
    4096: (32, 64, 128, 256, 512),
    8192: (32, 64, 128),
}


def scale_chip_counts_for_reference(
    chip_counts: tuple[int, ...],
    *,
    reference_a100_count: int,
    mode: str,
) -> tuple[int, ...]:
    """Resolve normalized search values to physical PLENA chip counts."""

    if reference_a100_count <= 0:
        raise ValueError(
            "reference_a100_count must be positive, got "
            f"{reference_a100_count}"
        )
    if mode not in CHIP_COUNT_SCALING_MODES:
        raise ValueError(
            f"unknown chip-count scaling mode {mode!r}; "
            f"expected one of {CHIP_COUNT_SCALING_MODES}"
        )
    if not chip_counts or any(int(value) <= 0 for value in chip_counts):
        raise ValueError("chip_counts must contain positive integers")
    multiplier = reference_a100_count if mode == "per-a100-reference" else 1
    return tuple(int(value) * multiplier for value in chip_counts)


def valid_blen_values(mlen: int) -> tuple[int, ...]:
    try:
        return LEGAL_BLENS_BY_MLEN[int(mlen)]
    except KeyError as exc:
        raise ValueError(f"MLEN={mlen} is outside the canonical DSE domain") from exc


def valid_blen_log2_values(mlen: int) -> tuple[int, ...]:
    return tuple(int(math.log2(value)) for value in valid_blen_values(mlen))


def valid_mlen_values(chip_count: int) -> tuple[int, ...]:
    chips = int(chip_count)
    if chips <= 0:
        raise ValueError(f"chip_count must be positive, got {chip_count}")
    if chips >= 16:
        maximum = 2048
    elif chips >= 8:
        maximum = 4096
    else:
        maximum = 8192
    return tuple(value for value in DEFAULT_MLEN_VALUES if value <= maximum)


def valid_mlen_log2_values(chip_count: int) -> tuple[int, ...]:
    return tuple(int(math.log2(value)) for value in valid_mlen_values(chip_count))


def conditional_mlen_param_name(chip_count: int) -> str:
    return f"MLEN_LOG2_CHIPS_{int(chip_count)}"


def conditional_blen_param_name(mlen: int) -> str:
    return f"BLEN_LOG2_MLEN_{int(mlen)}"


def conditional_tp_param_name(chip_count: int) -> str:
    return f"TP_DEGREE_N{int(chip_count)}"


def conditional_ep_param_name(tp_degree: int, cp_degree: int) -> str:
    return f"EP_DEGREE_TP{int(tp_degree)}_CP{int(cp_degree)}"


def conditional_sram_param_name(
    mlen: int,
    chip_count: int,
    parallel_model: str,
    *,
    tp_degree: int | None = None,
    cp_degree: int | None = None,
) -> str:
    parallel = parallel_model.upper().replace("-", "_")
    suffix = (
        f"_TP{int(tp_degree)}_CP{int(cp_degree)}"
        if tp_degree is not None and cp_degree is not None
        else ""
    )
    return (
        f"SRAM_CONFIG_INDEX_M{int(mlen)}_N{int(chip_count)}_"
        f"{parallel}{suffix}"
    )


def canonical_sram_choices(
    *,
    policies: tuple[str, ...],
    k_blocks: int,
    mlen: int,
    projection_tiles: int,
    derive_policy: Callable[..., Any],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for policy in policies:
        plan = derive_policy(
            policy=policy,
            k_blocks=k_blocks,
            mlen=mlen,
            projection_tiles=projection_tiles,
        )
        key = (plan.matrix_sram_tiles, plan.resident_prefix_blocks)
        choice = grouped.setdefault(
            key,
            {
                "matrix_sram_tiles": int(plan.matrix_sram_tiles),
                "resident_prefix_blocks": int(plan.resident_prefix_blocks),
                "canonical_policy": policy,
                "policy_aliases": [],
                "plan": plan,
            },
        )
        choice["policy_aliases"].append(policy)
    ordered = sorted(
        grouped.values(),
        key=lambda item: (
            int(item["matrix_sram_tiles"]),
            int(item["resident_prefix_blocks"]),
            str(item["canonical_policy"]),
        ),
    )
    for index, choice in enumerate(ordered):
        choice["index"] = index
        choice["policy_aliases"] = tuple(choice["policy_aliases"])
        choice["config_id"] = (
            f"tiles{choice['matrix_sram_tiles']}_"
            f"resident{choice['resident_prefix_blocks']}"
        )
    return tuple(ordered)
