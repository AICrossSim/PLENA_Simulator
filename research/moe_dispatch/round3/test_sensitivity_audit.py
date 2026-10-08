import pytest

from research.moe_dispatch.round3.sensitivity_audit import audit_flip_csv


def test_empty_flip_csv_preserves_schema(tmp_path):
    path = tmp_path / "flip_points.csv"
    path.write_text("sample_index,certified_reversal\n")
    result = audit_flip_csv(path, [{"sample_index": "0", "delta": "0.01"}])
    assert result == {"flip_rows": 0, "flip_header_present": True,
                      "zero_row_flip_csv_is_valid": True}


@pytest.mark.parametrize("contents", [
    "sample_index,certified_reversal\n0,False\n0,False\n",
    "sample_index,certified_reversal\n99,False\n",
])
def test_flip_csv_rejects_duplicate_or_unknown_indices(tmp_path, contents):
    path = tmp_path / "flip_points.csv"
    path.write_text(contents)
    with pytest.raises(AssertionError):
        audit_flip_csv(path, [{"sample_index": "0", "delta": "-0.01"}])


def test_flip_csv_rejects_missing_header(tmp_path):
    path = tmp_path / "flip_points.csv"
    path.write_text("")
    with pytest.raises(AssertionError):
        audit_flip_csv(path, [])
