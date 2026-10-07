"""Physical, causal same-plan replay; no synthetic zero-error output shortcuts."""
from dataclasses import replace
from copy import deepcopy
import pytest
from .model import Core,Design,Parameters,simulate
from .predictors import Predictor
from .oracle_replay import same_schedule_oracle


def workload(batch=16):
    return {'id':'conditional_oracle_fixture','batch':batch,'hidden':2048,'top_k':2,
            'experts':[{'id':0,'Me':batch,'H':2048,'F':1408},
                       {'id':1,'Me':batch,'H':2048,'F':1408},
                       {'id':2,'Me':batch,'H':2048,'F':2816,'is_shared':True}]}


@pytest.mark.parametrize('mode',['pipelined','port_tight','fixed_issue'])
@pytest.mark.parametrize('prefetch',[False,True])
def test_replay_recomputes_service_under_frozen_admissions(mode,prefetch):
    d=Design((Core(4,4,512),Core(2,4,512)),flows=('WS','IS'))
    p=Parameters(onchip_mode=mode,prefetch=prefetch)
    w=workload();plan=simulate(w,d,p,predictor=Predictor('ours'));before=deepcopy(plan)
    r=same_schedule_oracle(plan,d,p)
    assert plan==before
    assert r['cycles']==pytest.approx(plan['cycles'],rel=1e-10,abs=1e-5)
    assert r['oracle_replay']['physically_replayed']
    assert r['oracle_replay']['max_hbm_difference_bytes']<1
    assert r['oracle_replay']['mae_pct']<1e-8
    assert r['oracle_replay']['max_timing_difference_cycles']<1e-4
    assert r['oracle_replay']['scope'].endswith('not a scheduling upper bound')


def test_replay_rejects_forged_finish_instead_of_forcing_expected_duration():
    d=Design((Core(6,16,128),),flows=('WS',));p=Parameters();plan=simulate(workload(),d,p)
    plan['phases'][0]['finish']+=100
    with pytest.raises(AssertionError,match='physical replay phase timing mismatch'):
        same_schedule_oracle(plan,d,p)


def test_replay_rejects_changed_resource_service():
    d=Design((Core(6,16,128),),flows=('WS',));p=Parameters();plan=simulate(workload(),d,p)
    with pytest.raises(AssertionError):same_schedule_oracle(plan,d,replace(p,credits=512))


def test_global_chunks_replay_individually_and_pay_real_spill():
    d=Design((Core(6,16,128),),flows=('WS',));p=Parameters();w=workload(256)
    plan=simulate(w,d,p,predictor=Predictor('ours'))
    assert plan['storage_chunks']>1 and plan['activation_spill_bytes']>0
    r=same_schedule_oracle(plan,d,p)
    assert r['cycles']==pytest.approx(plan['cycles'],rel=1e-10)
    assert abs(r['oracle_replay']['hbm_replay_bytes']-plan['hbm_bytes'])<1
    assert r['oracle_replay']['mae_pct']<1e-8


def test_replay_rejects_forged_first_ready_after_physical_recalculation():
    d=Design((Core(6,16,128),),flows=('WS',));p=Parameters();plan=simulate(workload(),d,p)
    plan['tasks'][0]['first_weight_ready']+=100
    with pytest.raises(AssertionError,match='physical replay task/prefix timing mismatch'):
        same_schedule_oracle(plan,d,p)


def test_replay_rejects_prefetch_before_binding_or_active_current():
    d=Design((Core(6,16,128),),flows=('WS',));p=Parameters();plan=simulate(workload(),d,p)
    plan['tasks'][0]['prefetch_request']=0.
    with pytest.raises(AssertionError,match='unrequested/unadmitted frozen prefetch'):
        same_schedule_oracle(plan,d,p)


def test_captured_gpqa_queue_waits_for_serialized_binding_lock():
    """Original failure requires the actual preceding predictor warmup sequence."""
    from .common import inputs
    d=Design((Core(4,4,512),Core(2,4,512)),flows=('WS','IS'))
    p=Parameters(onchip_mode='pipelined');pred=Predictor('ours')
    captures=inputs();target='v3_captured_mixed_heldout_gpqa_t64_l26';plan=None
    for w in captures['development']+captures['heldout']:
        current=simulate(w,d,p,predictor=pred)
        if w['id']==target:
            plan=current;break
    assert plan is not None, 'frozen GPQA regression capture is required'
    bind={b['expert_index']:b for b in plan['bindings']}
    assert any(t['expert_index']==45 for t in plan['tasks'])
    # Before the scoreboard fix, task 45 started 2.278 cycles before its
    # bind lock because Current completed during the serialized comparator.
    assert all(t['start']>=bind[t['expert_index']]['bind_cycle']-1e-7 for t in plan['tasks'])
    replay=same_schedule_oracle(plan,d,p)
    assert replay['oracle_replay']['max_timing_difference_cycles']<1e-4


def test_e5_driver_emits_six_primary_rows_and_separate_profile_reference(tmp_path,monkeypatch):
    from argparse import Namespace
    from . import run
    d=Design((Core(4,4,512),Core(2,4,512)),flows=('WS','IS'))
    w=workload(2);w['id']='driver_fixture'
    monkeypatch.setattr(run,'inputs',lambda:{'development':[w],'heldout':[w]})
    monkeypatch.setattr(run,'selection_file',lambda:{})
    monkeypatch.setattr(run,'designs_from_selection',lambda sel,mode:{'best_hetero':d,'fixed_4+2':d})
    monkeypatch.setattr(run,'MODES',('pipelined',));monkeypatch.setattr(run,'BATCHES',(2,))
    monkeypatch.setattr(run,'RESULTS_DIRECTORY',tmp_path)
    monkeypatch.setattr(run,'finalize',lambda *args,**kw:None)
    # Dispatch assignment is outside this changed predictor protocol. This
    # fixture uses real model execution while bypassing the unrelated CP pass.
    def run_many(ws,design,params,kind='runtime',policy='eft',jobs=1):
        return [simulate(x,design,params,policy='eft' if policy=='milp' else policy) for x in ws]
    monkeypatch.setattr(run,'run_many',run_many);monkeypatch.setattr(run,'pack_eval',lambda r,kind:r)
    run.e5(Namespace(jobs=1,onchip_mode='pipelined'))
    table=run.read_csv(tmp_path/'E5/predictor_table.csv')
    assert len(table)==12
    assert {r['predictor'] for r in table}=={'random','static','btb','ema','ours','oracle'}
    assert all(r['oracle_kind']=='conditional_same_actual_schedule' for r in table)
    profile=run.read_csv(tmp_path/'E5/profile_guided_reference.csv')
    assert len(profile)==2 and all(r['predictor']=='oracle_profile_guided' for r in profile)
    replay=run.read_csv(tmp_path/'E5/oracle_replay_validation.csv')
    assert len(replay)==2 and all(r['physically_replayed']=='True' for r in replay)
    for r in table:
        if r['predictor']=='oracle':assert float(r['mae_pct'])<1e-8 and float(r['e2e_ratio_vs_ours'])==pytest.approx(1.,abs=1e-10)
