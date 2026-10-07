import copy
import csv

import pytest
import torch

from transactional_emulator.testbench.aten.matrix_lcompute_execution_compare import (
    Arena,
    NEMOTRON_MAMBA,
    _bf16_bytes,
    pack_vector_state,
    unpack_vector_state,
    write_execution_tables,
)


def _control_results():
    """Small report inputs; execution correctness is checked by Rust runs."""
    results = []
    for variant, cycles in (("A", 30), ("B", 24), ("D", 12)):
        results.append(
            {
                "model": "test_recurrence",
                "variant": variant,
                "batch": 1,
                "tokens": 2,
                "seed": 17,
                "own_rounding_reference_exact": True,
                "common_budget_passed": True,
                "common_output_errors": [{"relative_l2": 0.0}],
                "timing": {
                    "period_picos": 1000,
                    "total_picos": cycles * 1000,
                    "dma_and_memory_wait_picos": (cycles - 10) * 1000,
                    "scalar_and_control_cycles": 1,
                    "counters": {
                        "issued_instructions": 2,
                        "issue_cycles": 2,
                        "bank_service_cycles": 3,
                        "arithmetic_cycles": 4,
                    },
                },
            }
        )
    return results


def _comparison_row(output_dir):
    with (output_dir / "qualified_comparison.csv").open() as source:
        return next(csv.DictReader(source))


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


def test_common_accuracy_is_required_before_publishing_a_speedup(tmp_path):
    results = _control_results()
    write_execution_tables(results, tmp_path)
    assert float(_comparison_row(tmp_path)["D_speedup_vs_B_qualified"]) == 2.0

    # A control can exactly reproduce its own rounding and still fail the
    # shared error budget. Reusing the directory must remove its old ratio.
    results[1]["common_budget_passed"] = False
    results[1]["common_output_errors"] = [{"relative_l2": 0.0135}]
    write_execution_tables(results, tmp_path)
    row = _comparison_row(tmp_path)
    assert row["common_numeric_budget_passed"] == "False"
    assert row["timing_qualified"] == "False"
    assert row["D_speedup_vs_B_qualified"] == ""


@pytest.mark.parametrize("contract_field", ["experimental_fp32_dot", "pairwise_bf16_dot"])
def test_snapshot_timing_and_mixed_contracts_never_publish_a_ratio(tmp_path, contract_field):
    results = _control_results()
    snapshots = copy.deepcopy(results)
    for r in snapshots:
        r["diagnostic_state_snapshots"] = True
    write_execution_tables(snapshots, tmp_path)
    row = _comparison_row(tmp_path)
    assert row["common_numeric_budget_passed"] == "True"
    assert row["D_speedup_vs_B_qualified"] == ""
    results[0][contract_field] = True
    write_execution_tables(results, tmp_path)
    assert not (tmp_path / "qualified_comparison.csv").exists()


def test_pairwise_control_rejects_hardware_extensions_before_execution(tmp_path):
    from transactional_emulator.testbench.aten.matrix_lcompute_execution_compare import run_variant, KIMI_KDA

    with pytest.raises(ValueError, match="no experimental FP32 or L_TILE"):
        run_variant(KIMI_KDA, "D", 1, 1, tmp_path, pairwise_bf16_dot=True)
    with pytest.raises(ValueError, match="no experimental FP32 or L_TILE"):
        run_variant(KIMI_KDA, "B", 1, 1, tmp_path, pairwise_bf16_dot=True, experimental_fp32_dot=True)
    assert not list(tmp_path.iterdir())


def test_pairwise_oracle_observes_each_bf16_rounding_boundary():
    from transactional_emulator.testbench.aten.matrix_lcompute_execution_compare import pairwise_bf16_reference

    # BF16 sequential sum loses +1 after256; the explicit tree preserves it here.
    values = torch.zeros(1, 128, 1)
    values[0, :4, 0] = torch.tensor([256.0, 1.0, 1.0, -256.0])
    assert pairwise_bf16_reference(values).item() == 1.0
    values[0, :4, 0] = torch.tensor([256.0, 1.0, -256.0, 0.0])
    assert pairwise_bf16_reference(values).item() == 0.0  # Pairwise is not magic FP32.
