"""Guards on the shared-vs-routed branch classification.

``_branch_split`` reduces a stage profile to one number, ``shared_branch_fraction``,
which is the headline claim about how much of a MoE layer's cost is the
always-on shared expert. A stage assigned to the wrong side, or to neither side
by accident, moves that number with nothing else changing -- the picoseconds
still add up, the profile still validates, and no other test notices.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from moe_timing.replay.utils import (  # noqa: E402
    ROUTED_BRANCH_STAGES,
    SHARED_BRANCH_STAGES,
    _branch_split,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
STAGE_PROFILE_RS = REPO_ROOT / "transactional_emulator" / "src" / "stage_profile.rs"

#: Stages billed to neither branch: input and combine plumbing both branches
#: share. Named explicitly so that a stage added to the emulator has to be
#: classified on purpose rather than dropped out of the split by silence.
PLUMBING_STAGES = (
    "residual_setup",
    "accumulator_init",
    "gather",
    "scatter_combine",
)


def _emulator_stage_names() -> list[str]:
    """Every `StageKind` name the emulator can bill to, except `other`.

    Read out of the Rust source rather than mirrored, for the same reason the
    emulator parses the compiler's `MOE_STAGES` rather than copying it: a
    mirrored list agrees with the original right up until one of them moves.
    """
    names = re.findall(r"StageKind::\w+ => \"(\w+)\"", STAGE_PROFILE_RS.read_text())
    assert names, f"no StageKind names found in {STAGE_PROFILE_RS}; this guard would pass vacuously"
    return [name for name in names if name != "other"]


def test_router_is_billed_to_the_routed_branch() -> None:
    """The router only exists because the layer routes.

    It does not scale with `top_k`, which is why it was originally left out, but
    it scales with `num_experts` and has no counterpart at all in a dense or
    shared-only layer. Leaving it out understated routing by the whole router
    GEMM plus top-k selection.
    """
    assert "router_topk" in ROUTED_BRANCH_STAGES
    assert "router_topk" not in SHARED_BRANCH_STAGES


def test_every_emulator_stage_is_classified_exactly_once() -> None:
    """No stage may fall out of the split by being forgotten.

    Adding a `StageKind` variant without touching this module silently drops it
    from both branches, which reads as "this work is free" rather than as an
    unclassified stage.
    """
    classified = list(SHARED_BRANCH_STAGES) + list(ROUTED_BRANCH_STAGES) + list(PLUMBING_STAGES)
    assert len(classified) == len(set(classified)), (
        f"a stage is classified into more than one bucket: {sorted(classified)}"
    )

    emulator = set(_emulator_stage_names())
    unclassified = sorted(emulator - set(classified))
    assert not unclassified, (
        "these emulator stages are in neither branch nor the plumbing list, so "
        f"_branch_split silently ignores them: {unclassified}"
    )

    unknown = sorted(set(classified) - emulator)
    assert not unknown, (
        "these classified names are not stages the emulator can bill to, so they "
        f"contribute nothing and the classification is stale: {unknown}"
    )


def test_router_picos_land_on_the_routed_side() -> None:
    """The classification has to reach the arithmetic, not just the tuple."""
    profile = {
        "stages": {
            "shared_expert_projection": {"wall_picos": 300},
            "router_topk": {"wall_picos": 200},
            "expert_projection": {"wall_picos": 500},
            # Plumbing: must land on neither side.
            "accumulator_init": {"wall_picos": 1000},
            "gather": {"wall_picos": 1000},
        }
    }
    split = _branch_split(profile)

    assert split["shared_branch_picos"] == 300
    assert split["routed_branch_picos"] == 700, "router_topk is not reaching the routed total"
    assert split["shared_branch_fraction"] == pytest.approx(300 / 1000)


def test_split_is_null_for_a_profile_that_cannot_express_the_question() -> None:
    """Nulls mean "this profile predates schema v4", not "no shared expert".

    `to_json` serializes every `StageKind` unconditionally, so a v4 profile from
    a program with no shared expert still carries `shared_expert_projection` at
    zero and the split returns `0 / N / 0.0`. The null path is reachable only for
    a profile written before those stages existed. A consumer that reads nulls as
    "no shared branch" would mis-handle a real `shared_branch_fraction == 0.0`.
    """
    split = _branch_split({"stages": {"router_topk": {"wall_picos": 200}}})
    assert split == {
        "shared_branch_picos": None,
        "routed_branch_picos": None,
        "shared_branch_fraction": None,
    }


def test_a_shared_free_v4_profile_reports_zero_not_null() -> None:
    """The case the test above is *not* about, pinned so the distinction holds."""
    split = _branch_split(
        {
            "stages": {
                "shared_expert_projection": {"wall_picos": 0},
                "shared_expert_activation": {"wall_picos": 0},
                "shared_expert_gate": {"wall_picos": 0},
                "router_topk": {"wall_picos": 200},
                "expert_projection": {"wall_picos": 800},
            }
        }
    )
    assert split == {
        "shared_branch_picos": 0,
        "routed_branch_picos": 1000,
        "shared_branch_fraction": 0.0,
    }
