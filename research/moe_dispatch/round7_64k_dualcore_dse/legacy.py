"""Re-execute legacy control/timing, then isolate model/task/aggregation changes."""
import math
from .common import ROOT, CFG, read_json,save_csv,save_json,gm,sha
from .model import references,task_cost
from .scheduler import optimal

PUBLISHED={
 'old_single':[.007606,.012895,.017922,.029931,.088412,.124548,.160710],
 'old_homo':[.003834,.007198,.012020,.021830,.077105,.112607,.148366],
 'old_hetero':[.003819,.006108,.010827,.020516,.075633,.111226,.146526]}

def run():
    from research.moe_dispatch.scaling_g64.geometry import Core as LegacyCore
    from research.moe_dispatch.scaling_g64.runtime import schedule
    from research.moe_dispatch.plena65536_dedicated.model import job_cycles
    from research.moe_dispatch.plena65536_shared.timing import PROFILES
    from .trace import load_inputs
    ws,_=load_inputs(False);original=read_json(ROOT/'inputs/original_windows.json.gz')
    rows=[]
    for label,cs in references()[1:]:
        cores=tuple(LegacyCore.from_shape(c.pm,c.pn,c.g) for c in cs)
        for w,nw in zip(original,ws):
            common=dict(profile='pipelined_tree1',policy='mdpu',threshold=2,queue_depth=8,
                        prefetch=True,bypass=True,context_cycles=4,details=False,wait_best=True)
            old=schedule(cores,w,**common);again=schedule(cores,w,**common)
            assert old==again
            p=[[job_cycles(c,e,PROFILES[0]) for c in cores] for e in w['experts']]
            native_opt=optimal(p)
            for model in CFG['models']:
                pmerged=[[task_cost(c,e['Me'],model,0,e['F'])['cycles'] for c in cs] for e in w['experts']]
                psep=[[task_cost(c,t['m'],model,0)['cycles'] for c in cs] for t in nw['tasks']]
                rows.append(dict(label=label,model=model,window=w['id'],batch=w['batch'],
                                 legacy_runtime_us=old['cycles']/1000,
                                 legacy_cost_optimal_us=native_opt['cycles']/1000,
                                 new_cost_merged_shared_us=optimal(pmerged)['cycles']/1000,
                                 new_cost_two_shared_us=optimal(psep)['cycles']/1000))
        print('Legacy replay finished '+label,flush=True)
    save_csv(ROOT/'results/legacy_by_window.csv',rows)
    summary=[]
    for model in CFG['models']:
        for label,cs in references()[1:]:
            for i,b in enumerate(CFG['batches']):
                rs=[r for r in rows if r['model']==model and r['label']==label and r['batch']==b]
                given=PUBLISHED.get(label,[None]*7)[i]
                oldgm=gm([r['legacy_runtime_us'] for r in rs])
                summary.append(dict(model=model,label=label,batch=b,published_us=given*1000 if given else None,
                    legacy_replay_GM_us=oldgm,
                    replay_vs_published_pct=(oldgm/(given*1000)-1)*100 if given else None,
                    **{k:math.fsum(r[k] for r in rs)/len(rs) for k in rows[0] if k.endswith('_us')}))
    save_csv(ROOT/'results/legacy_attribution.csv',summary)
    save_json(ROOT/'results/legacy_receipt.json',dict(points=4*135,repeat=2,all_identical=True,
        source_hashes={str(p.relative_to(ROOT.parents[2])):sha(p) for sub in ('scaling_g64','plena65536_shared','plena65536_dedicated')
                       for p in (ROOT.parent/sub).glob('*.py')}))

if __name__=='__main__':run()
