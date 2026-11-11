"""Run-length helpers of the event timeline."""

from complementary_arrays.mech.timeline import merge, runs


def test_merge_joins_short_gaps():
    assert merge([1, 2, 3, 10, 11, 50], 6) == [[1, 11], [50, 50]]
    assert merge([], 3) == []


def test_runs_split_on_label_or_gap():
    assert runs(["a", "a", "b", "b", "b"], [5, 6, 7, 8, 10]) == [["a", 5, 6], ["b", 7, 8], ["b", 10, 10]]
