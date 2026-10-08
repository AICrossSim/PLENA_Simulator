from dataclasses import asdict
import copy
import json

import pytest

from research.moe_dispatch.round3 import robust_heldout as r
from research.moe_dispatch.round3.model import Core, Design
from research.moe_dispatch.round3.config import parameters
from research.moe_dispatch.round3.search import key


def test_development_near_set_keeps_frozen_choice_but_excludes_heldout_irrelevant_candidates():
    designs = [Design((Core(6,4,512),), flows=(f,)) for f in ("OS","WS","IS")]
    rows = [{"design":asdict(d),"score_ms":score,"status":"evaluated"}
            for d,score in zip(designs,(100.,101.,101.0001))]
    certificate={"selected":rows[0],"witnesses":rows}
    chosen=r.near_witnesses(certificate)
    assert {key(r.decode(x['design'])) for x in chosen}=={key(designs[0]),key(designs[1])}
    assert certificate['selected']==rows[0]


def _reusable_fixture(tmp_path,monkeypatch):
    root=tmp_path/'round3';old=tmp_path/'round2'
    monkeypatch.setattr(r,'ROOT',root);monkeypatch.setattr(r,'OLD',old)
    for directory in (root/'E5/dispatch',old/'results/E0'):
        directory.mkdir(parents=True)
    (old/'results/E0/frozen_inputs.json').write_text('{}')
    (root/'E5/dispatch/RUN_RECEIPT.json').write_text('{}')
    (root/'E5/dispatch/repeat_checks.json').write_text('[]')
    d=Design((Core(6,4,512),));p=parameters('pipelined')
    point={'point_id':'test','onchip_mode':'pipelined','physical_key':key(d),'hardware':asdict(d),'parameters':asdict(p)}
    held=[{'id':f'w{i}','batch':2} for i in range(135)]
    result=[{'cycles':100.+i,'hbm_bytes':4096} for i in range(135)]
    raw=root/'old.json.gz';r.save_gzip(raw,result)
    current={str(root/name):f'source-{name}' for name in ('model.py','runtime.py','optimizer.py','config.py')}
    receipt={'dispatch':'milp','onchip_mode':'pipelined','hardware':asdict(d),'parameters':asdict(p),
        'exact_repeat':True,'result_digest':r.digest(result),'repeat_digest':r.digest(result),
        'raw_file':'old.json.gz','raw_sha256':r.sha(raw),'solver_checks':[{'window_id':w['id']} for w in held]}
    run={'source_unchanged':True,'source_sha256':current,
         'input_manifest_sha256':r.sha(old/'results/E0/frozen_inputs.json')}
    return point,held,run,receipt,current


def test_reuse_requires_same_frozen_physical_source_and_full_result_bytes(tmp_path,monkeypatch):
    point,held,run,receipt,current=_reusable_fixture(tmp_path,monkeypatch)
    valid=r.reuse_existing(point,held,run,[receipt],current)
    assert valid['exact_repeat'] and len(valid['rows'])==135
    changed=copy.deepcopy(current);changed[str(r.ROOT/'runtime.py')]='changed'
    with pytest.raises(AssertionError):r.reuse_existing(point,held,run,[receipt],changed)
    changed=copy.deepcopy(receipt);changed['repeat_digest']='not repeated'
    with pytest.raises(AssertionError):r.reuse_existing(point,held,run,[changed],current)
    changed=copy.deepcopy(receipt);changed['result_digest']=changed['repeat_digest']='forged'
    with pytest.raises(AssertionError):r.reuse_existing(point,held,run,[changed],current)


def test_reuse_rejects_wrong_window_order_or_physical_parameters(tmp_path,monkeypatch):
    point,held,run,receipt,current=_reusable_fixture(tmp_path,monkeypatch)
    changed=copy.deepcopy(receipt);changed['solver_checks'][0]['window_id']='other'
    with pytest.raises(AssertionError):r.reuse_existing(point,held,run,[changed],current)
    changed=copy.deepcopy(receipt);changed['parameters']['credits']=256
    with pytest.raises(AssertionError):r.reuse_existing(point,held,run,[changed],current)
