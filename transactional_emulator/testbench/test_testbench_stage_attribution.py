"""The MoE ``@stage=`` attribution contract, applied to the testbenches.

The compiler carries this lint over its own emitters in
``PLENA_Compiler/aten/tests/test_moe_stage_attribution.py``. That scan stops at
the repository boundary, and the testbenches here are on the other side of it
while calling the same stage-polymorphic emitters -- so a bad stage name in a
testbench is exactly as invisible as one in the compiler used to be, and there
was nothing checking for it.

The pin bump in 878a341 is the concrete case. Making ``stage`` required in the
compiler turned six testbench call sites into errors, all of which had been
silently inheriting ``moe_true_zero_vram_rows_v0``'s ``accumulator_init``
default. They were caught because the signature change made them fail to run at
all. A *wrong but declared* stage name would not have been caught by anything.

The matcher is imported from the compiler's guard rather than copied. A second
copy of "which callees take a MoE stage argument" is the same mirroring problem
these tests exist to catch: it would agree with the original right up until the
compiler's rule moved.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTBENCH_DIR = pathlib.Path(__file__).resolve().parent
COMPILER_GUARD = (
    REPO_ROOT / "PLENA_Compiler" / "aten" / "tests" / "test_moe_stage_attribution.py"
)

#: MoE ``stage=`` arguments each testbench file is expected to carry.
#:
#: Six of these are the call sites 878a341 had to fix; the seventh, in
#: ``moe_shared_expert_test.py``, predates it. Pinning the counts is the
#: anti-vacuity check: a moved directory or a narrowed glob would otherwise
#: leave the scan reaching nothing and passing.
#:
#: Counts, not stage names -- reattributing a site is ordinary work, while
#: losing sight of one is the failure this notices.
_EXPECTED_STAGE_ARGUMENT_SITES = {
    "attention_semantics_test.py": 2,
    "qwen3_trace_replay.py": 1,
    "gpt_oss_moe_gather_scatter_test.py": 3,
    "moe_shared_expert_test.py": 1,
}


def _load_compiler_guard() -> types.ModuleType:
    """Import the compiler's guard module by path.

    By path rather than by package: ``aten/tests`` is a package, so importing it
    normally executes ``aten/__init__.py``, which pulls in the op registry and
    with it yaml and torch. Nothing here needs any of that.
    """
    spec = importlib.util.spec_from_file_location("_compiler_stage_guard", COMPILER_GUARD)
    assert spec is not None and spec.loader is not None, f"cannot load {COMPILER_GUARD}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _guard_or_skip() -> types.ModuleType:
    """The compiler guard, or a skip when the submodule is not checked out.

    Skipping is allowed locally, where a clone without ``--recursive`` is an
    ordinary state. In CI it is not: the workflow checks out with
    ``submodules: recursive``, so an absent submodule means a broken pipeline,
    and a guard that skips itself there reads green while checking nothing.
    """
    if not COMPILER_GUARD.is_file():
        reason = (
            f"{COMPILER_GUARD} is not readable; run "
            "`git submodule update --init PLENA_Compiler` from the repository root"
        )
        assert not os.environ.get("CI"), f"the stage-attribution guard cannot run in CI: {reason}"
        pytest.skip(reason)
    return _load_compiler_guard()


def _testbench_sources() -> list[pathlib.Path]:
    sources = sorted(
        path
        for path in TESTBENCH_DIR.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    assert sources, f"no Python sources found under {TESTBENCH_DIR}; the guard would pass vacuously"
    return sources


def test_testbench_stage_arguments_are_declared_moe_stages() -> None:
    """Every literal ``stage=`` a testbench passes must name a real stage.

    A name outside ``MOE_STAGES`` reaches the emulator, lands in
    ``unresolved_stage_markers`` and leaves that region billed to whatever
    marker came before it -- with instruction counts, cycle totals and numerical
    results all unchanged, so nothing else fails.
    """
    guard = _guard_or_skip()
    declared = guard._declared_moe_stages()

    offenders: list[str] = []
    for path in _testbench_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for callee, keyword in guard._moe_stage_arguments(tree):
            if keyword.value.value not in declared:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{keyword.value.lineno} "
                    f"{callee}(stage={keyword.value.value!r})"
                )

    assert not offenders, (
        "these testbench call sites pass a stage name that is not in the compiler's "
        "MOE_STAGES:\n  " + "\n  ".join(offenders)
    )


def test_testbench_stage_parameters_have_no_default() -> None:
    """A testbench helper wrapping an emitter must not default ``stage`` either.

    Same failure as on the compiler side: the wrapper is called from inside a
    marked region, inherits the enclosing marker for everything it emits, and a
    default silently supplies a wrong answer instead of a missing one.
    """
    guard = _guard_or_skip()

    offenders: list[str] = []
    for path in _testbench_sources():
        for func in guard._functions(path):
            for _arg, default in guard._defaulted_stage_params(func):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{func.lineno} "
                    f"{func.name}(stage={ast.unparse(default)})"
                )

    assert not offenders, (
        "these testbench helpers give `stage` a default, so a caller that forgets "
        "it is silently billed to the wrong stage:\n  " + "\n  ".join(offenders)
    )


def test_the_known_call_sites_are_actually_scanned() -> None:
    """Prove the scan reaches the call sites it is supposed to cover.

    Both lints above pass by finding nothing. If the glob stopped matching --
    a renamed directory, a testbench moved out of the tree -- they would keep
    passing over an empty scan, which is the same green as being correct.
    """
    guard = _guard_or_skip()

    found: dict[str, int] = {}
    for path in _testbench_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        count = sum(1 for _ in guard._moe_stage_arguments(tree))
        if count:
            found[path.name] = count

    assert found == _EXPECTED_STAGE_ARGUMENT_SITES, (
        "the set of testbench files passing MoE `stage=` arguments changed.\n"
        f"  scanned:  {dict(sorted(found.items()))}\n"
        f"  expected: {dict(sorted(_EXPECTED_STAGE_ARGUMENT_SITES.items()))}\n"
        "If a call site was added or removed on purpose, update "
        "_EXPECTED_STAGE_ARGUMENT_SITES. If it was not, the scan has stopped "
        "reaching code it is meant to cover."
    )


def test_non_moe_stage_arguments_are_not_flagged() -> None:
    """``qkt_multiply(stage="decode")`` lives in this tree and is not a MoE stage.

    ``direct_emit/flashattn_qkt_test.py`` passes ``stage="decode"`` to select
    prefill vs decode. It is the reason the matcher keys on the callee: a scan
    over every ``stage=`` keyword would report correct attention code as a bad
    MoE stage name.
    """
    guard = _guard_or_skip()

    tree = ast.parse(
        'qkt_multiply(d=16, stage="decode", mlen=4)\n'
        'moe_true_zero_vram_rows_v0(builder, stage="accumulator_init")\n'
    )
    matched = [(callee, keyword.value.value) for callee, keyword in guard._moe_stage_arguments(tree)]

    assert matched == [("moe_true_zero_vram_rows_v0", "accumulator_init")], (
        f"the stage-argument matcher is not scoped to MoE callees; it picked up {matched}"
    )
