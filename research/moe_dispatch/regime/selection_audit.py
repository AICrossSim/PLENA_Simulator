"""Replay saved development selections; retain full compiler configuration IDs."""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib
import numpy as np
from .search import development,point
from ..geometry3d.study import cores_from,canonical,write_csv,write_json


def runtime_rows(path):
    rows=list(csv.DictReader(path.open()))
    for r in rows:
        r['legal']=r['legal']=='True'
        r['partition']=json.loads(r['partition'])
        r['window_cycles']=json.loads(r['window_cycles'])
        for k in ('prefetch_slots','gu_group_limit','down_group_limit'):
            r[k]=int(r[k])
        r['valid_operand_traffic']=r['valid_operand_traffic']=='True'
        r['score_ms']=float(r['score_ms']) if r['score_ms'] else None
    return rows


def signature(r):
    # The original export omitted group caps. Match aggregated fractions under
    # precisely that old projection, then export full unambiguous config IDs.
    return (r['family'],r['geometry'],json.dumps(r['partition'],sort_keys=True),
            r['flow'],r['policy'],str(r['prefetch_slots']))


def run(root,inputs):
    dev=development(inputs);checks=[]
    selection=json.loads((root/'search/FROZEN_SELECTION.json').read_text())['points']
    for credits,fmt in sorted({(p['credits'],p['weight_format']) for p in selection}):
        sub=root/'search'/f'{fmt}_c{credits}'
        allrows=[]
        for name in ('geometry_screen.json','strong_baseline_candidates.json','independent_allocations.json'):
            allrows+=json.loads((sub/name).read_text())
        allrows+=runtime_rows(sub/'runtime_development.csv')
        exported=[];totals=defaultdict(float)
        for fam in ('single','homogeneous','heterogeneous'):
            unique={canonical({k:r[k] for k in point(cores_from(r['geometry']))}):r
                    for r in allrows if r['legal'] and r['family']==fam}
            legal=list(unique.values());logs=np.log(np.array([r['window_cycles'] for r in legal]))
            rng=np.random.default_rng(20261005);groups=defaultdict(list)
            for i,w in enumerate(dev):groups[w['batch']].append(i)
            scores=np.zeros((len(legal),1000))
            for ids in groups.values():
                drawn=rng.choice(ids,(1000,len(ids)),replace=True)
                scores+=logs[:,drawn].sum(axis=2)
            counts=np.bincount(np.argmin(scores,axis=0),minlength=len(legal))
            for r,n in zip(legal,counts):
                if not n:continue
                p={k:r[k] for k in point(cores_from(r['geometry']))}
                fraction=float(n/1000);totals[signature(r)]+=fraction
                exported.append({'credits':credits,'weight_format':fmt,**p,
                    'configuration_sha256':hashlib.sha256(canonical(p)).hexdigest(),
                    'bootstrap_selected_fraction':fraction,'scope':'development selection,1000 paired batch-stratified draws'})
        old=defaultdict(float)
        for r in csv.DictReader((sub/'selection_stability.csv').open()):
            r['partition']=json.loads(r['partition'])
            old[signature(r)]+=float(r['bootstrap_selected_fraction'])
        assert old.keys()==totals.keys()
        assert all(abs(old[k]-totals[k])<1e-12 for k in totals)
        write_csv(sub/'selection_stability_complete.csv',[
            {**r,'partition':json.dumps(r['partition'],sort_keys=True)} for r in exported])
        checks.append({'credits':credits,'weight_format':fmt,'replay_matches_original':True,
                       'compiler_group_caps_included':True,'complete_configuration_ids':True})
    write_json(root/'audit/SELECTION_EXPORT_CHECK.json',checks)
