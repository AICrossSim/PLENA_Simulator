"""Frozen inputs, paired metrics, repeat checking, provenance for round two."""
from __future__ import annotations
from dataclasses import asdict
import csv, hashlib, json, math, subprocess, sys
from pathlib import Path
import numpy as np
from .model import Core, Design, Parameters, task_cost, simulate, micro_cost, ceildiv
BATCHES=(2,4,8,16,64,96,128)
MODES=('pipelined','port_tight','fixed_issue')
ROOT=Path(__file__).resolve().parent

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')

def write_csv(p,rows,fields=None):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    rows=list(rows)
    fields=fields or list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def read_csv(p):
    with Path(p).open() as f: return list(csv.DictReader(f))

def gmean(values):
    vs=list(values)
    return math.exp(sum(math.log(float(v)) for v in vs)/len(vs)) if vs else float('nan')

def decode_design(x):
    if isinstance(x,str): return Design(tuple(Core(*map(int,s.split('x'))) for s in x.split('+')))
    x=dict(x);x['cores']=tuple(Core(**c) if isinstance(c,dict) else Core(*c) for c in x['cores'])
    return Design(**x)

def encode_design(d):return asdict(d)

def inputs():
    f=json.loads((ROOT/'results/E0/frozen_inputs.json').read_text())
    source=Path(f['input_directory'])
    for name,hashval in f['input_sha256'].items():
        if sha(source/name)!=hashval: raise AssertionError('frozen input hash changed: '+name)
    from research.moe_dispatch.regime.campaign import load_inputs
    ws=load_inputs(source)
    assert [w['id'] for w in ws['development']]==f['development_window_ids']
    assert [w['id'] for w in ws['heldout']]==f['heldout_window_ids']
    return ws

def evaluate_repeat(w,d,p,**kw):
    a=simulate(w,d,p,**kw);b=simulate(w,d,p,**kw)
    assert canonical(a)==canonical(b),'two run mismatch'
    return a

def paired_ci(ratios,seed=20261007,draws=2000):
    logs=np.log(np.asarray(ratios,dtype=float));rng=np.random.default_rng(seed)
    v=np.exp(logs[rng.integers(0,len(logs),size=(draws,len(logs)))].mean(axis=1))
    # Report speed reduction, 1-candidate/baseline. Larger is better.
    lo,hi=np.quantile(1-v,[.025,.975])
    return float(lo),float(hi)

def native_bytes(w):
    return sum(3*2*e['H']*e['F'] for e in w['experts'])

def useful_macs(w):return sum(3*e['Me']*e['H']*e['F'] for e in w['experts'])

def bounds(w,d,p,result=None):
    """Installed-design lower bounds; private ownership never inferred from speed."""
    from .optimizer import _costs
    allcost=[]
    for row in _costs(w,d,p):
        costs=[(c,co) for c,co in enumerate(row) if co is not None]
        if not costs:raise ValueError('no physical core for storage-chunked task')
        allcost.append(costs)
    unique=native_bytes(w)/p.hbm_bandwidth
    actual=(result['hbm_bytes'] if result else sum(min(co.hbm_bytes for c,co in cs) for cs in allcost))/p.hbm_bandwidth
    # Minimum legal traffic, divided by the SUM of installed ports, not by a free per-core full fabric.
    W=sum(min(co.w_sram_bytes for c,co in cs) for cs in allcost)
    X=sum(min(co.x_sram_bytes for c,co in cs) for cs in allcost)
    A=sum(min(co.acc_sram_bytes for c,co in cs) for cs in allcost)
    port=max(W/sum(p.w_bandwidth(d,c) for c in range(len(d.cores))),X/(24*p.bank_Bpc),A/(12*p.bank_Bpc))
    def fastest(cs):
        return min(max(co.irreducible_busy,co.dependency_floor,co.hbm_bytes/p.hbm_bandwidth,
            co.w_sram_bytes/p.w_bandwidth(d,c),co.x_sram_bytes/(d.x_banks[c]*p.bank_Bpc),
            co.acc_sram_bytes/(d.acc_banks[c]*p.bank_Bpc),co.vector_elements/(d.vector_lanes[c]*p.vector_scale)) for c,co in cs)
    task=max((fastest(cs) for cs in allcost),default=0)
    fs={'hbm_floor_unique':unique,'hbm_floor_actual':actual,'mac_floor':useful_macs(w)*p.issue_interval/d.total_macs,
        'port_floor':port,'task_floor':task}
    term=max(fs,key=fs.get);bound=fs[term]
    if result and bound>result['cycles']+1e-5:raise AssertionError(('bound invalid',w['id'],d.geometry,term,bound,result['cycles']))
    return {**{k:v/1e6 for k,v in fs.items()},'bound':bound/1e6,'binding_term':term,
            'headroom_pct':100*(result['cycles']/bound-1) if result else None}

def finalize(directory,command,summary,extra=''):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if summary is not None:(directory/'SUMMARY.md').write_text(summary+'\n')
    repo=ROOT.parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    hashes={p.name:sha(p) for p in sorted(ROOT.glob('*.py'))}
    frozen=json.loads((ROOT/'results/E0/frozen_inputs.json').read_text())
    (directory/'README.md').write_text('# Reproducible round-two evidence\n\n'+
        f'Execution commit: `{commit}`. BF16 only. 1 hypothetical cycle = 1 ns; ms = cycles / 1e6.\n\n'+
        'Boundary: post-router MoE Gate/Up, SiLU/Z, Down, combine. Phase-fluid analytical estimates, not RTL/native HBM measurements.\n\n'+
        'Command:\n```sh\n'+command+'\n```\n\nSource hashes:\n```json\n'+json.dumps(hashes,indent=2)+'\n```\n\n'+
        'Frozen input hashes:\n```json\n'+json.dumps(frozen['input_sha256'],indent=2)+'\n```\n\n'+extra+'\n')
    rows=[{'file':str(p.relative_to(directory)),'sha256':sha(p),'execution_commit':commit,
        'input_manifest_sha256':sha(ROOT/'results/E0/frozen_inputs.json')} for p in sorted(directory.rglob('*'))
        if p.is_file() and p.name!='PROVENANCE.csv']
    write_csv(directory/'PROVENANCE.csv',rows)
