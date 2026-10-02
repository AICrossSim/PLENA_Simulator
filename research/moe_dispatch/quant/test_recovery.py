"""Host-failure recovery guards use synthetic verifier rows, not accuracy data."""
import pytest
import recover_campaign as r
from test_layer_shards import one_candidate


def fixture_rows():
    return [dict(row,relative_error=.2,cosine=.98,factor_bytes=32) for row in one_candidate()]


def test_recovery_keeps_complete_candidate_and_discards_unfinished_tail():
    rows=fixture_rows();kept,coverage=r.partial_q1(rows+rows[:12],13)
    assert kept==rows and coverage['completed_candidates']==1
    assert not coverage['complete']


def test_recovery_rejects_duplicate_or_incomplete_expert_coverage():
    rows=fixture_rows()
    with pytest.raises(ValueError,match='duplicate'):r.partial_q1(rows+rows,13)
    damaged=[dict(row) for row in rows];damaged[0]['projection']='u'
    with pytest.raises(ValueError,match='coverage'):r.partial_q1(damaged,13)
