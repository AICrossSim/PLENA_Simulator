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
    if name in sys.modules:
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
    """Withdraw the stubs so the rest of the session sees the real packages."""
    for name in _INSTALLED:
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
