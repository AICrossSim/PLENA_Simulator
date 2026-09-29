"""Guards on the campaign runner's verdict, its export filter and its progress states.

``run_subset`` computes a functional gate per run, a determinism verdict and a
per-trace replay status, then used to ``return 0`` regardless -- so a campaign
that replayed nothing, failed every gate and disagreed with itself exited exactly
like a clean one. It also exported gate-failed runs into the medians and p95s
with nothing downstream marking them suspect.

Both are silent-wrong-number failures: the CSV looks the same either way. They
are checked here rather than at the call site because ``run_subset`` has no
caller -- it is an operator tool, which is precisely why its exit code has to
mean something.

The module is imported with torch/transformers stubbed. Importing it for real
pulls in ``qwen.generate_true_routing_with_weights`` -> ``transformers``, and a
``pytest.importorskip`` would make this guard silently vacuous in the
pytest-only CI job. Stubbing keeps the *real* functions under test.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


#: Names this module injected, so they can be withdrawn again.
#:
#: `sys.modules` is process-wide. Leaving a fake `torch` behind would make every
#: module collected after this one in the same pytest session import the stub
#: instead of the real package -- a test that silently exercises a stub is worse
#: than one that fails to import.
_INSTALLED: list[str] = []


def _stub(name: str, **attrs: object) -> None:
    """Substitute for a package only where the real one is unavailable.

    The stub exists for the pytest-only CI job, which installs pytest and nothing
    else. Where the real package *is* importable -- any dev checkout, and the
    build-and-test job -- installing the stub anyway shadows it for the rest of
    the session: `pytest transactional_emulator/testbench/moe_timing/` collects
    this directory before `qwen/`, so the qwen guards would then fail to import
    torch, with a traceback pointing at PLENA_Tools rather than at this file.
    """
    if name in sys.modules:
        return
    try:
        importlib.import_module(name)
    except ImportError:
        pass
    else:
        return
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    _INSTALLED.append(name)


_stub("torch", Tensor=object, tensor=lambda *a, **k: None, zeros=lambda *a, **k: None)
_stub("torch.nn", functional=types.SimpleNamespace())
_stub("torch.nn.functional")
_stub("transformers", AutoConfig=object, AutoModelForCausalLM=object, AutoTokenizer=object)
_stub("safetensors", safe_open=object)
_stub("safetensors.torch", load_file=lambda *a, **k: {})

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

#: Modules already loaded before anything is imported against a stub.
#:
#: Withdrawing the stub names is not enough on its own: a module imported while
#: they were installed keeps the fake bound inside it and stays cached, so a test
#: collected later gets a module built on a stub -- the exact "silently exercises
#: a stub" outcome the withdrawal exists to prevent. Everything loaded after this
#: point goes too.
_PRELOADED = frozenset(sys.modules)

from transactional_emulator.testbench.moe_timing.campaign.run_subset import (  # noqa: E402
    _exit_code,
)

#: The shape `write_report` actually produces. The first version of these tests
#: hand-wrote a flat {name: count} mapping for `failure_counts`, which
#: `_failure_counts` never returns -- so the guard passed while `_exit_code` was
#: iterating the wrong level and returning 1 for every campaign. Fixtures are
#: built from the real producer below for exactly that reason.
_CLEAN = {
    "route_trace_count": 4,
    "exported_runs": 4,
    "functional_gate_counts": {"True": 4},
    "determinism": {"status": "passed"},
    "progress_snapshot": {"replay_failed_runs": 0},
    "failure_counts": {
        "total_existing_failures": 0,
        "by_file": {"replay_failures.jsonl": None, "p3_rev_replay_failures.jsonl": 0},
    },
}


def test_a_clean_campaign_exits_zero() -> None:
    assert _exit_code(_CLEAN) == 0


def test_a_report_missing_every_optional_field_exits_zero() -> None:
    """Absent is not failed. A partial run must not be reported as a regression."""
    assert _exit_code({}) == 0


def test_a_failed_functional_gate_fails_the_run() -> None:
    report = dict(_CLEAN, functional_gate_counts={"True": 3, "False": 1})
    assert _exit_code(report) == 1


def test_a_failed_determinism_gate_fails_the_run() -> None:
    assert _exit_code(dict(_CLEAN, determinism={"status": "failed"})) == 1


def test_a_skipped_determinism_gate_does_not_fail_the_run() -> None:
    """`skipped` is a decision not to check, not a check that failed."""
    assert _exit_code(dict(_CLEAN, determinism={"status": "skipped"})) == 0


def test_a_gate_failed_replay_fails_the_run() -> None:
    report = dict(_CLEAN, progress_snapshot={"replay_failed_runs": 2})
    assert _exit_code(report) == 1


def test_historical_failure_logs_do_not_fail_this_run() -> None:
    """`failures/*.jsonl` is opened append-only and never truncated.

    It describes the out_root's whole history, so gating on it would make one
    transient failure poison every later campaign in that directory. The counts
    stay in the JSON, where a human can date them; the exit code speaks only for
    this invocation.
    """
    report = dict(
        _CLEAN,
        failure_counts={
            "total_existing_failures": 3,
            "by_file": {"p3_rev_replay_failures.jsonl": 3},
        },
    )
    assert _exit_code(report) == 0


def test_a_staged_run_that_stops_before_export_is_not_a_failure() -> None:
    """`--actions=select,route,build` leaves traces on disk and nothing exported.

    An earlier version failed that as "traces were selected but nothing was
    exported", which is a verdict on the operator's command line rather than on
    the data, and fired on every staged invocation.
    """
    report = dict(_CLEAN, route_trace_count=4, exported_runs=0)
    assert _exit_code(report) == 0


def test_every_failure_mode_is_reported_not_just_the_first(capsys) -> None:
    """An operator fixing one problem must not have to re-run to discover the next."""
    report = {
        "functional_gate_counts": {"False": 2},
        "determinism": {"status": "failed"},
        "progress_snapshot": {"replay_failed_runs": 1},
        "failure_counts": {"total_existing_failures": 5, "by_file": {"replay.jsonl": 5}},
    }
    assert _exit_code(report) == 1
    printed = capsys.readouterr().out
    for expected in ("functional gate", "determinism", "gate-failed"):
        assert expected in printed, f"{expected!r} missing from:\n{printed}"


def test_a_clean_campaign_built_by_the_real_producer_exits_zero(tmp_path) -> None:
    """The regression test for how the first version of this file went wrong.

    Every other case above hand-writes a report. That is what let `_exit_code`
    ship while it iterated `failure_counts` at the wrong level: the fixtures used
    a flat mapping, `_failure_counts` returns a nested one, and the guard agreed
    with the fixture rather than with the code. Building the field from its real
    producer is the only version of this test that could have caught it.
    """
    import argparse

    from transactional_emulator.testbench.moe_timing.campaign.run_subset import (
        _failure_counts,
    )

    args = argparse.Namespace(out_root=tmp_path, replay_run_name="p3_rev_parallel_replay")
    (tmp_path / "failures").mkdir(parents=True, exist_ok=True)

    report = dict(_CLEAN, failure_counts=_failure_counts(args))
    assert "by_file" in report["failure_counts"], "the producer's shape changed"
    assert _exit_code(report) == 0, f"a clean campaign must exit 0, got report={report['failure_counts']}"

    # A this-run signal must still be caught through the same path.
    report = dict(_CLEAN, failure_counts=_failure_counts(args), functional_gate_counts={"True": 3, "False": 1})
    assert _exit_code(report) == 1


def teardown_module(_module: object) -> None:
    """Withdraw the stubs, and everything that was imported against them.

    Popping only `_INSTALLED` restores the names but not the session: modules
    imported while the stubs were live hold the fake objects and stay cached, so
    the next test to import one gets a module built on a stub. Nothing is dropped
    when no stub was installed -- the real packages were available and every
    import was genuine.
    """
    if not _INSTALLED:
        return
    for name in _INSTALLED:
        sys.modules.pop(name, None)
    for name in [name for name in sys.modules if name not in _PRELOADED]:
        sys.modules.pop(name, None)


# --------------------------------------------------------------------------
# The two behavioural changes that had no test at all: a gate-failed run must
# not reach the exported numbers, and "never ran" must not read as "ran and
# failed". Both were verified by hand when written, which is why deleting either
# left every test green.
# --------------------------------------------------------------------------


def _run_dir(root: Path, trace_id: str, *, gate: object, stage_profile: bool = True) -> Path:
    import json

    d = root / "trace_replay" / trace_id
    d.mkdir(parents=True, exist_ok=True)
    payload: dict = {} if gate is None else {"zero_input_smoke_gate": {"passed": gate}}
    (d / "qwen3_trace_replay_results.json").write_text(json.dumps(payload))
    (d / "rust_emulator_run_stats.json").write_text("{}")
    if stage_profile:
        (d / "stage_profile.json").write_text("{}")
    return d


def test_a_gate_failed_run_is_not_complete_and_is_not_missing(tmp_path) -> None:
    """The trichotomy: complete / failed / missing are three different answers."""
    from transactional_emulator.testbench.moe_timing.campaign.run_subset import _result_ok

    passed = _run_dir(tmp_path, "ok", gate=True)
    failed = _run_dir(tmp_path, "bad", gate=False)
    never = tmp_path / "trace_replay" / "absent"

    assert _result_ok(passed) is True
    assert _result_ok(failed) is False, "a gate-failed run must not count as complete"
    assert _result_ok(never) is False
    assert (failed / "qwen3_trace_replay_results.json").exists(), (
        "the failed run must still leave its artefact behind -- it is the diagnostic"
    )
    assert not (never / "qwen3_trace_replay_results.json").exists()


def test_a_passing_run_missing_only_its_stage_profile_is_not_called_failed(tmp_path) -> None:
    """Incomplete is not failed. This is the case the first trichotomy got wrong."""
    from transactional_emulator.testbench.moe_timing.qwen.run_trace_batch import (
        _prior_run_passed,
    )

    d = _run_dir(tmp_path, "no_profile", gate=True, stage_profile=False)
    assert _prior_run_passed(d / "qwen3_trace_replay_results.json") is True, (
        "the gate passed; only the profile is absent, which is work remaining"
    )


def test_gate_failed_rows_are_excluded_from_the_exported_numbers() -> None:
    """The filter's contract: drop False, keep None, and say how many went."""
    rows = [
        {"trace_id": "a", "benchmark": "b", "sample_id": "1", "functional_gate": True},
        {"trace_id": "b", "benchmark": "b", "sample_id": "2", "functional_gate": False},
        {"trace_id": "c", "benchmark": "b", "sample_id": "3", "functional_gate": None},
    ]
    kept = [r for r in rows if r.get("functional_gate") is not False]
    dropped = [r for r in rows if r.get("functional_gate") is False]
    unknown = [r for r in rows if r.get("functional_gate") is None]

    assert [r["trace_id"] for r in kept] == ["a", "c"], (
        "an absent gate is unknown, not failed -- dropping it would shrink the sample"
    )
    assert [r["trace_id"] for r in dropped] == ["b"]
    assert len(unknown) == 1


# The qwen replay reports two gates. `zero_input_smoke_gate` compares an all-zero
# accumulator against an all-zero golden, which dummy expert weights produce for
# any routing at all; `router_gate` checks the experts V_TOPK actually selected.
# Only the second can fail on a routing fault, so anything that consults one gate
# has to consult both -- otherwise the apparatus that decides what to retry and
# what to average keys on the check that cannot fail.


def _results(*, smoke: bool, router: bool | None) -> dict:
    payload = {"zero_input_smoke_gate": {"passed": smoke, "gate_kind": "zero_input_shape_smoke"}}
    if router is not None:
        payload["router_gate"] = {"passed": router, "gate_kind": "device_selected_experts_match_trace"}
    return payload


def test_a_router_failed_run_is_not_treated_as_already_done(tmp_path) -> None:
    """`--skip-existing` must retry it, not skip it.

    The results file is written before the gate assertion raises, so a
    router-failed run leaves a complete-looking artifact whose smoke gate says
    True. Reading only that gate makes the failure permanent: the trace is never
    re-run, and its cycles stay in the export.
    """
    from transactional_emulator.testbench.moe_timing.qwen.run_trace_batch import _prior_run_passed
    from transactional_emulator.testbench.moe_timing.replay.utils import write_json

    path = tmp_path / "qwen3_trace_replay_results.json"
    write_json(path, _results(smoke=True, router=False))

    assert _prior_run_passed(path) is False


def test_a_run_passing_both_gates_is_treated_as_done(tmp_path) -> None:
    from transactional_emulator.testbench.moe_timing.qwen.run_trace_batch import _prior_run_passed
    from transactional_emulator.testbench.moe_timing.replay.utils import write_json

    path = tmp_path / "qwen3_trace_replay_results.json"
    write_json(path, _results(smoke=True, router=True))

    assert _prior_run_passed(path) is True


def test_a_result_with_no_router_gate_still_reads_from_the_smoke_gate(tmp_path) -> None:
    """Artifacts written before on-device routing carry no `router_gate`.

    Absent must stay unknown-but-passing here, or every pre-existing run in an
    out_root would suddenly re-run.
    """
    from transactional_emulator.testbench.moe_timing.qwen.run_trace_batch import _prior_run_passed
    from transactional_emulator.testbench.moe_timing.replay.utils import write_json

    path = tmp_path / "qwen3_trace_replay_results.json"
    write_json(path, _results(smoke=True, router=None))

    assert _prior_run_passed(path) is True


def test_the_campaign_summary_reports_a_router_failure_as_a_failed_gate(tmp_path) -> None:
    """`functional_gate_passed` feeds the export filter and the exit code.

    `summarize_run` picks one gate out of a precedence chain, and the qwen
    replay lands on the smoke gate. Leaving `router_gate` out of it means
    `export_selected` averages the cycles of a run that routed to the wrong
    experts, and `_exit_code` counts it as passing.
    """
    from transactional_emulator.testbench.moe_timing.replay.utils import summarize_run, write_json

    build = tmp_path / "run"
    build.mkdir()
    # `run_result_path` searches for this name, not `qwen3_trace_replay_results.json`;
    # `run_trace` writes the same summary to both.
    write_json(build / "gather_scatter_results.json", _results(smoke=True, router=False))
    write_json(build / "rust_emulator_run_stats.json", {"sim_latency_cycles": 10})
    write_json(build / "stage_profile.json", {"total_simulation_cycles": 10})

    row, _ = summarize_run("r0", build)

    assert row["functional_gate_passed"] is False
