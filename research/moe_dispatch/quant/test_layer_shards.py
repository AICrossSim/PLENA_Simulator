"""Independent merge guards; fixtures are synthetic verifier inputs only."""
import pytest
import run_layer_shards as s


def one_candidate():
    common=dict(layer=13,rank_lanes=8,bits=4,method='qera_approx',factor_a='mxint4',factor_b='bf16')
    rows=[]
    for e in range(-1,64):
        rows += [dict(common,scope='projection',expert=e,projection=p,rank=24) for p in ('g','u','d')]
        rows += [dict(common,scope='expert',expert=e,projection='ffn',rank=32)]
    rows += [dict(common,scope='layer',expert='all',projection='moe',rank=32)]
    return rows


def test_candidate_merge_checks_every_expert_projection(monkeypatch):
    monkeypatch.setattr(s.q,'q1_coverage',lambda *a,**k:{'complete':True})
    rows=one_candidate()
    assert s.verify_q1(rows,[13])['complete']
    with pytest.raises(ValueError,match='missing'):
        s.verify_q1(rows[1:],[13])
    damaged=[dict(r) for r in rows];damaged[0]['projection']='u'
    with pytest.raises(ValueError,match='missing'):
        s.verify_q1(damaged,[13])


def test_candidate_merge_rejects_cross_candidate_metadata(monkeypatch):
    monkeypatch.setattr(s.q,'q1_coverage',lambda *a,**k:{'complete':True})
    rows=one_candidate();rows[0]['factor_b']='mxint8'
    with pytest.raises(ValueError,match='provenance'):
        s.verify_q1(rows,[13])


def q3_rows():
    return [dict(layer=13,rank_lanes=L,bits=b,method=m,uniform_rank=u,window_start=0,strategy=p)
        for L in (8,16) for b in (4,3) for m in ('qera_approx','qera_exact','lqer','l2qer') for u in (16,32)
        for p in ('uniform','frequency_static','gate_weighted_budget_oracle','gate_weighted_causal')]


def test_q3_merge_checks_complete_independent_sequences():
    rows=q3_rows();assert s.verify_q3(rows,[13],tokens=16)['actual_ffn_rows']==128
    for broken in (rows[1:],rows+rows[:1]):
        with pytest.raises(ValueError,match='incomplete/duplicate'):
            s.verify_q3(broken,[13],tokens=16)
