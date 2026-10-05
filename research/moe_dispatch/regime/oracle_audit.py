"""Check that component oracles change timing, not the mapped work ledger."""
from pathlib import Path
from dataclasses import replace
import json
from .campaign import evaluate,load_inputs
from .search import settings_for
from ..geometry3d.study import cores_from,write_json,canonical


def run(root,inputs):
    held=load_inputs(inputs)['heldout']
    frozen=json.loads((root/'search/FROZEN_SELECTION.json').read_text())['points']
    checks=[]
    fields=('hbm_bytes','native_unique_bytes','useful_macs','issued_macs',
            'X_sram_bytes','global_activation_bytes','budget','compulsory_traffic')
    phase_fields=('core','expert_index','phase','compute','issues','useful_macs','issued_macs',
        'hbm_bytes','weight_unique','w_port_bytes','x_port_bytes','acc_port_bytes',
        'activation_bytes','peak_w_bytes','peak_x_bytes','peak_accumulator_bytes','decoder_elements')
    for p in frozen:
        sub=root/'search'/f"{p['weight_format']}_c{p['credits']}"
        charged=json.loads((sub/f"heldout_{p['family']}.json").read_text())
        owners=[tuple(b['core'] for b in r['bindings']) for r in charged]
        s=settings_for(p['credits'],p['weight_format'],p);g=cores_from(p['geometry'])
        for name,alt in (('ideal_HBM',replace(s,hbm=False)),('ideal_ports',replace(s,ports=False)),
                         ('zero_control',replace(s,control=False)),
                         ('compute_only',replace(s,hbm=False,ports=False,control=False))):
            results=evaluate(held,g,alt,p['policy'],owners=owners)
            for a,b in zip(results,charged):
                assert all(canonical(a[k])==canonical(b[k]) for k in fields),(p['geometry'],name,a['workload'])
                assert tuple(x['core'] for x in a['bindings'])==tuple(x['core'] for x in b['bindings'])
                key=lambda r:[tuple(x[k] for k in phase_fields) for x in r['phases']]
                # Completion order can change; mapped expert/phase contents may not.
                assert sorted(key(a))==sorted(key(b))
            checks.append({'credits':p['credits'],'weight_format':p['weight_format'],'family':p['family'],
                'oracle':name,'windows':len(results),'repeats':2,'work_traffic_reservations_owners_unchanged':True})
    write_json(root/'audit/ORACLE_TIMING_ONLY_CHECK.json',{'checks':checks,
        'scope':'analytical ledger/owner/capacity verification; not native mutual-exclusion/drain or numerical-output measurement'})
