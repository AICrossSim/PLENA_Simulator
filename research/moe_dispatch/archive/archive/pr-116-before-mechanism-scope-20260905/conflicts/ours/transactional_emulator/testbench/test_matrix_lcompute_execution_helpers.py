from itertools import pairwise

import pytest
import torch

from transactional_emulator.testbench.aten.matrix_lcompute_execution_compare import (
    Arena,
    NEMOTRON_MAMBA,
    _bf16_bytes,
    pack_vector_state,
    unpack_vector_state,
)


def test_request_guard_detects_a_single_byte_write_outside_outputs():
    arena = Arena()
    first = arena.reserve(64, writable=True)
    second = arena.reserve(64, writable=True)
    post = bytearray(arena.image)
    post[first] = 1
    post[second] = 2
    arena.check_immutable(post)
    post[second - 1] ^= 1
    with pytest.raises(AssertionError, match="escaped"):
        arena.check_immutable(post)


def test_vector_state_layout_preserves_heads_rows_and_request_bases():
    spec = NEMOTRON_MAMBA
    # Exactly representable small integer sentinels encode head/row/lane variation.
    state = (
        torch.arange(spec.heads * spec.recurrence_rows * spec.row_elements)
        .remainder(127)
        .float()
        .reshape(spec.heads, spec.recurrence_rows, spec.row_elements)
    )
    packed = _bf16_bytes(pack_vector_state(state, 32))
    post = bytes(128) + packed + bytes(64) + _bf16_bytes(pack_vector_state(-state, 32))
    assert torch.equal(unpack_vector_state(post, 128, spec, 32), state)
    assert torch.equal(unpack_vector_state(post, 128 + len(packed) + 64, spec, 32), -state)


def test_executable_artifacts_require_common_accuracy_before_publishing_a_speedup():
    import csv
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "artifacts"
    kimi = root / "matrix_lcompute_execution_kda_diagnostic_v1"
    results = json.loads((kimi / "summary.json").read_text())
    assert all(result["own_rounding_reference_exact"] for result in results)
    assert not next(result for result in results if result["variant"] == "B")["common_budget_passed"]
    rows = list(csv.DictReader((kimi / "qualified_comparison.csv").open()))
    assert rows[0]["D_speedup_vs_B_qualified"] == ""


def test_every_executed_batch_has_private_state_and_identical_logical_inputs():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "artifacts/matrix_lcompute_execution_v1"
    results = json.loads((root / "summary.json").read_text())
    assert len(results) == 15
    for batch in (1, 2, 4, 8, 16):
        variants = {r["variant"]: r for r in results if r["batch"] == batch}
        assert set(variants) == {"A", "B", "D"}
        for result in variants.values():
            assert result["private_state_and_immutable_guards_passed"]
            assert result["own_rounding_reference_exact"] and result["common_budget_passed"]
            assert result["operand_sha256"] == variants["D"]["operand_sha256"]
            assert result["initial_state_sha256_by_request"] == variants["D"]["initial_state_sha256_by_request"]
            ranges = result["state_arenas"]
            assert len(ranges) == batch
            assert all(a["end"] < b["begin"] for a, b in pairwise(ranges))
        assert variants["A"]["output_sha256"] == variants["B"]["output_sha256"]
        assert variants["A"]["state_sha256"] == variants["B"]["state_sha256"]


def test_snapshot_timing_and_mixed_contracts_never_publish_a_ratio(tmp_path):
    import copy
    import csv
    import json
    from pathlib import Path
    from transactional_emulator.testbench.aten.matrix_lcompute_execution_compare import write_execution_tables
    source = Path(__file__).resolve().parents[2] / 'artifacts/matrix_lcompute_execution_v1/summary.json'
    results = [r for r in json.loads(source.read_text()) if r['batch'] == 1]
    snapshots = copy.deepcopy(results)
    for r in snapshots:
        r['diagnostic_state_snapshots'] = True
    write_execution_tables(snapshots, tmp_path)
    row = next(csv.DictReader((tmp_path/'qualified_comparison.csv').open()))
    assert row['common_numeric_budget_passed'] == 'True'
    assert row['D_speedup_vs_B_qualified'] == ''
    other = tmp_path  # Reusing an output directory must clear a stale ratio table.
    results[0]['experimental_fp32_dot'] = True
    write_execution_tables(results, other)
    assert not (other/'qualified_comparison.csv').exists()


def test_pairwise_control_rejects_hardware_extensions_before_execution(tmp_path):
    from transactional_emulator.testbench.aten.matrix_lcompute_execution_compare import run_variant, KIMI_KDA
    with pytest.raises(ValueError, match="no experimental FP32 or L_TILE"):
        run_variant(KIMI_KDA, 'D', 1, 1, tmp_path, pairwise_bf16_dot=True)
    with pytest.raises(ValueError, match="no experimental FP32 or L_TILE"):
        run_variant(KIMI_KDA, 'B', 1, 1, tmp_path, pairwise_bf16_dot=True, experimental_fp32_dot=True)
    assert not list(tmp_path.iterdir())


def test_pairwise_oracle_observes_each_bf16_rounding_boundary():
    from transactional_emulator.testbench.aten.matrix_lcompute_execution_compare import pairwise_bf16_reference
    # BF16 sequential sum loses +1 after256; the explicit tree preserves it here.
    values = torch.zeros(1, 128, 1)
    values[0, :4, 0] = torch.tensor([256., 1., 1., -256.])
    assert pairwise_bf16_reference(values).item() == 1.
    values[0, :4, 0] = torch.tensor([256., 1., -256., 0.])
    assert pairwise_bf16_reference(values).item() == 0.  # Pairwise is not magic FP32.
