"""The MoE ``@stage=`` attribution contract, applied to the testbenches.

The compiler carries this lint over its own emitters in
``PLENA_Compiler/aten/tests/test_moe_stage_attribution.py``. That scan stops at
the repository boundary, and the testbenches here are on the other side of it
while calling the same stage-polymorphic emitters -- so a bad stage name in a
testbench is exactly as invisible as one in the compiler used to be, and there
was nothing checking for it.


The matcher is imported from the compiler's guard rather than copied: a second
copy would agree with the original right up until the compiler's rule moved.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import pathlib
import re
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TESTBENCH_DIR = pathlib.Path(__file__).resolve().parent
COMPILER_GUARD = REPO_ROOT / "PLENA_Compiler" / "aten" / "tests" / "test_moe_stage_attribution.py"

#: MoE ``stage=`` arguments each testbench file is expected to carry.

#: Counts, not stage names -- reattributing a site is ordinary work, while
#: losing sight of one is the failure this notices.
#:
#: Keyed by path relative to the testbench root, not by basename: two files with
#: the same name in different directories would otherwise collapse into one key
#: and silently drop a call site from the count this exists to pin.
_EXPECTED_STAGE_ARGUMENT_SITES = {
    "models/gpt_oss/attention_semantics_test.py": 2,
    "moe_timing/qwen/qwen3_trace_replay.py": 1,
    "routed_moe/gpt_oss_moe_gather_scatter_test.py": 3,
    "routed_moe/moe_shared_expert_test.py": 1,
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
    sources = sorted(path for path in TESTBENCH_DIR.rglob("*.py") if "__pycache__" not in path.parts)
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
        for callee, constant in guard._moe_stage_arguments(tree):
            if constant.value not in declared:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{constant.lineno} {callee}({constant.value!r})")

    assert not offenders, (
        "these testbench call sites pass a stage name that is not in the compiler's "
        "MOE_STAGES:\n  " + "\n  ".join(offenders)
    )


def test_testbench_stage_parameters_have_no_default() -> None:
    """A testbench helper wrapping an emitter must not default ``stage`` either."""
    guard = _guard_or_skip()

    offenders: list[str] = []
    for path in _testbench_sources():
        for func in guard._functions(path):
            for _arg, default in guard._defaulted_stage_params(func):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{func.lineno} {func.name}(stage={ast.unparse(default)})"
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
            found[str(path.relative_to(TESTBENCH_DIR))] = count

    assert found == _EXPECTED_STAGE_ARGUMENT_SITES, (
        "the set of testbench files passing MoE `stage=` arguments changed.\n"
        f"  scanned:  {dict(sorted(found.items()))}\n"
        f"  expected: {dict(sorted(_EXPECTED_STAGE_ARGUMENT_SITES.items()))}\n"
        "If a call site was added or removed on purpose, update "
        "_EXPECTED_STAGE_ARGUMENT_SITES. If it was not, the scan has stopped "
        "reaching code it is meant to cover."
    )


def _ci_workflow_text() -> str:
    """The text of every workflow file under ``.github/workflows``, concatenated.

    Reading only ``transactional_emulator.yml`` assumed it was the only workflow
    that runs guards from this tree. That was true when written and is not a
    property anything enforces: a second workflow running them is a wiring these
    checks cannot see, so a correctly wired guard reads as unwired and the check
    against silence becomes a source of it. Concatenating is deliberately
    permissive -- the question is whether *some* job names the file, and which
    one is not this check's business.
    """
    workflows = sorted((REPO_ROOT / ".github" / "workflows").glob("*.y*ml"))
    assert workflows, f"no workflow files under {REPO_ROOT / '.github' / 'workflows'}; this would pass vacuously"
    return "\n".join(path.read_text() for path in workflows)


def test_the_workflow_scan_reads_every_workflow_file() -> None:
    """Pins the scan to the whole directory, not one file in it.

    Both checks below pass by finding nothing unwired, so narrowing what they
    read makes them *more* likely to pass -- except for the guards named only in
    the file that was dropped, which start reading as never run by CI. That is
    not hypothetical: it is what reading only ``transactional_emulator.yml`` did
    to two guards correctly wired into ``ci.yml``.
    """
    directory = REPO_ROOT / ".github" / "workflows"
    present = sorted(directory.glob("*.y*ml"))
    assert len(present) > 1, f"only {[path.name for path in present]} under {directory}; this would pass vacuously"

    text = _ci_workflow_text()
    unread = [path.name for path in present if path.read_text() not in text]
    assert not unread, f"_ci_workflow_text() leaves these workflows unread: {unread}"


def test_every_guard_file_is_wired_into_ci() -> None:
    """The torch-free guards under ``testbench/`` must all be named by some job.

    The workflows list files individually, because most of ``testbench/`` needs
    torch and a built emulator and cannot be collected. That makes it possible
    to add a guard here that CI never runs -- indistinguishable from not having
    written it.

    Being *named* in a workflow is not the same as being *run* by it: this
    matches the workflow text, and a job whose trigger never fires would still
    satisfy it. And "does not import torch" is a proxy for "collectable in the
    pytest-only job" -- a guard that needs a built emulator binary but no torch
    would be forced into that job and fail there.
    """
    workflow = _ci_workflow_text()

    torch_free: list[str] = []
    for path in sorted(TESTBENCH_DIR.rglob("test_*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text()
        if re.search(r"^\s*(import torch|from torch)", source, re.MULTILINE):
            continue
        torch_free.append(path.name)

    assert torch_free, f"no torch-free guards found under {TESTBENCH_DIR}; this would pass vacuously"

    unwired = sorted(name for name in torch_free if name not in workflow)
    assert not unwired, (
        "these torch-free guards are in the tree but the workflow does not name "
        "them, so CI never runs them:\n  " + "\n  ".join(unwired)
    )


def _justfile_recipes(text: str) -> dict[str, str]:
    """Recipe name -> body, for the repo justfile.

    A recipe starts at a non-indented ``name args...:`` line and runs to the next
    one. Blank lines stay inside the current body, which just permits.
    """
    recipes: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line[:1] not in ("", " ", "\t", "#") and ":" in line:
            current = line.split(":", 1)[0].split()[0]
            recipes.setdefault(current, [])
        elif current is not None and (line.startswith((" ", "\t")) or not line.strip()):
            recipes[current].append(line)
        elif line.strip():
            current = None
    return {name: "\n".join(body) for name, body in recipes.items()}


def test_every_guard_is_reachable_from_ci() -> None:
    """Every guard file must be run by a job, directly or through a recipe.

    The check above only covers torch-free guards, because those are the ones
    that belong in the pytest-only job. A guard that *does* import torch is
    exempt from it and is typically run through a justfile recipe instead -- so
    nothing asserted those recipes were still invoked. Dropping one line from the
    workflow silently retired every guard behind it, with all checks green.

    Reachability, not execution: a recipe named in a job whose trigger never
    fires still satisfies this. It closes the gap between "a file exists" and
    "something in CI mentions a way to run it", which is where guards go quiet.
    """
    workflow = _ci_workflow_text()
    recipes = _justfile_recipes((REPO_ROOT / "justfile").read_text())
    assert recipes, "no justfile recipes parsed; this guard would pass vacuously"

    def named_in(text: str, token: str) -> bool:
        """`token` appears as a whole word, not as the prefix of a longer one.

        A plain substring test passes on `just test-sliced-layer-builder-renamed`
        for the recipe `test-sliced-layer-builder`, so renaming an invocation
        without renaming the recipe would satisfy this guard while retiring the
        job. `\\b` does not help -- a hyphen is a non-word character, so the
        boundary matches mid-token.
        """
        return re.search(rf"{re.escape(token)}(?![\w-])", text) is not None

    unreachable: list[str] = []
    for path in sorted(TESTBENCH_DIR.rglob("test_*.py")):
        if "__pycache__" in path.parts:
            continue
        if named_in(workflow, path.name):
            continue
        owning = [name for name, body in recipes.items() if named_in(body, path.name)]
        if any(named_in(workflow, name) for name in owning):
            continue
        how = f" (named by recipe(s) {owning}, none of which the workflow invokes)" if owning else ""
        unreachable.append(f"{path.name}{how}")

    assert not unreachable, (
        "these guards are in the tree but no CI job names them or a recipe that "
        "runs them:\n  " + "\n  ".join(unreachable)
    )


def test_non_moe_stage_arguments_are_not_flagged() -> None:
    """``qkt_multiply(stage="decode")`` lives in this tree and is not a MoE stage.

    ``direct_emit/flashattn_qkt_test.py`` passes ``stage="decode"`` to select
    prefill vs decode. It is why the matcher keys on the callee.
    """
    guard = _guard_or_skip()

    tree = ast.parse(
        'qkt_multiply(d=16, stage="decode", mlen=4)\n'
        'moe_true_zero_vram_rows_v0(builder, stage="accumulator_init")\n'
        'moe_stage_marker("gather", "detail")\n'
    )
    matched = sorted((callee, constant.value) for callee, constant in guard._moe_stage_arguments(tree))

    assert matched == [
        ("moe_stage_marker", "gather"),
        ("moe_true_zero_vram_rows_v0", "accumulator_init"),
    ], f"the stage-argument matcher is wrong; it picked up {matched}"
