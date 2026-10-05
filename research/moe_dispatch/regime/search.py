"""Development-only geometry screen, independent quota refinement, frozen test.

Exhaustive geometry, bounded hierarchical allocation search. This does not
claim the global joint geometry/partition/runtime optimum. Hardware and
runtime settings are frozen across every batch within a transport regime.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from collections import defaultdict
import argparse, csv, hashlib, json, math, time
import numpy as np
from ..geometry3d.compute import enumerate_geometries, enumeration_counts, family, geometry_id, TIMING_PROFILES
from ..geometry3d.memory import FabricProfile
from ..geometry3d.study import cores_from, verify_capture, canonical, write_csv, write_json
from .campaign import evaluate, result_rows, aggregate
from .model import Settings, projection_phase, expert_phases
from .resources import Partition, memory_budget
from .metrics import geomean, paired_bootstrap, core_rows

FRACTIONS=(.125,.25,.375,.5,.625,.75,.875)
FIELDS=tuple(Partition.__dataclass_fields__)
POLICIES=('eft','idle','round_robin','threshold_1','threshold_2','threshold_4','threshold_8','threshold_16')
GROUPS=(0,1,2,4,8)
DEV=()


def init_worker(dev):
    global DEV
    DEV=dev


def settings_for(credits,fmt,point=None):
    p=point or {}
    return Settings(weight_format=fmt, fabric=replace(FabricProfile(),hbm_credits=credits),
        valid_operand_traffic=p.get('valid_operand_traffic',True),
        gu_group_limit=p.get('gu_group_limit',2),down_group_limit=p.get('down_group_limit',4),
        allocation=p.get('allocation','equal'),flow=p.get('flow','bounded_ws'),
        partition=Partition(**p['partition']) if p.get('partition') else None,
        prefetch_slots=p.get('prefetch_slots',32))


def point(cores,allocation='equal',partition=None,flow='bounded_ws',policy='eft',slots=32,gu=2,down=4):
    return {'geometry':geometry_id(cores),'family':family(cores),'allocation':allocation,
            'partition':None if partition is None else asdict(partition),
            'flow':flow,'policy':policy,'prefetch_slots':slots,
            'valid_operand_traffic':True,'gu_group_limit':gu,'down_group_limit':down}


def score_point(p,credits,fmt,dev=None):
    s=settings_for(credits,fmt,p)
    try:
        results=evaluate(DEV if dev is None else dev,cores_from(p['geometry']),s,p['policy'],detail=False)
    except ValueError as exc:
        return {**p,'legal':False,'reason':str(exc),'score_ms':None,'window_cycles':[]}
    cycles=[r['cycles'] for r in results]
    return {**p,'legal':True,'reason':'','score_ms':geomean(cycles)/1e6,'window_cycles':cycles}


def screen_one(args):
    cores,credits,fmt=args
    # Both installed allocations are initial probes; final allocations have
    # seven independent axes. Invalid probes do not exclude the other probe.
    rows=[score_point(point(cores,a),credits,fmt) for a in ('equal','proportional')]
    legal=[r for r in rows if r['legal']]
    best=min(legal,key=lambda r:(r['score_ms'],r['allocation'])) if legal else rows[0]
    return {**best,'initial_profiles_tested':2,'initial_legal_profiles':len(legal)}


def baseline_one(args):
    p,credits,fmt=args
    return score_point(p,credits,fmt)


def refine_one(args):
    initial,credits,fmt=args
    cores=cores_from(initial['geometry']); seen={}
    def run(p):
        key=canonical(p)
        if key not in seen: seen[key]=score_point(p,credits,fmt)
        return seen[key]
    def order(r): return (float('inf') if not r['legal'] else r['score_ms'],canonical({k:r[k] for k in point(cores)}))
    for flow in ('bounded_ws','bounded_os'):
        for a in ('equal','proportional'): run(point(cores,a,flow=flow))
        if len(cores)==1: continue
        # Four deterministic multistarts, then independent coordinate sweeps.
        # No allocation is derived from MAC count in these sweeps.
        for f in (.25,.5,.75):
            current=point(cores,partition=Partition(**dict.fromkeys(FIELDS,f)),flow=flow)
            run(current)
            for _ in range(2):
                before=canonical(current)
                for field in FIELDS:
                    options=[]
                    for value in FRACTIONS:
                        part=Partition(**{**current['partition'],field:value})
                        options.append(run(point(cores,partition=part,flow=flow)))
                    winner=min(options,key=order)
                    if winner['legal']:
                        current={k:winner[k] for k in point(cores)}
                if canonical(current)==before: break
        # Probe cross-axis combinations beyond local coordinate optima.
        rng=np.random.default_rng(int(hashlib.sha256(initial['geometry'].encode()).hexdigest()[:8],16))
        for _ in range(32):
            part=Partition(**dict(zip(FIELDS,map(float,rng.choice(FRACTIONS,len(FIELDS))))))
            run(point(cores,partition=part,flow=flow))
    return list(seen.values())


def csv_row(r):
    return {**r,'partition':json.dumps(r['partition'],sort_keys=True,separators=(',',':')),
            'window_cycles':json.dumps(r['window_cycles'],separators=(',',':')),
            'initial_profiles_tested':r.get('initial_profiles_tested',0),
            'initial_legal_profiles':r.get('initial_legal_profiles',0)}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    here=Path(__file__).parent
    # Timing-relevant modules and selection code; prose/report tooling excluded.
    paths=[here/n for n in ('model.py','resources.py','metrics.py','campaign.py','search.py')]
    paths +=[here.parent/'geometry3d'/n for n in ('model.py','compute.py','memory.py','study.py')]
    return {str(p.relative_to(here.parent)):sha(p) for p in paths}


def development(inputs):
    ws=[w for name in ('development.json','mixed_development.json')
        for w in json.loads((inputs/name).read_text())['workloads']]
    verify_capture(ws)
    return ws


def candidates(grid):
    rows=list(csv.DictReader(grid.open()))
    return sorted({(int(r['credits']),r['weight_format']) for r in rows
        if r['split']=='development' and float(r['architecture_headroom_percent'])>=10})


def select(inputs,out,grid,workers=16,limit=None):
    out.mkdir(parents=True,exist_ok=True)
    dev=development(inputs); regimes=candidates(grid)
    geometries=enumerate_geometries()
    if limit: geometries=geometries[:limit]
    contract={'scope':'analytical fluid screening, not native/silicon results',
        'regimes_from_development_space_ge10percent':regimes,'geometry_counts':enumeration_counts(),
        'headroom_grid_sha256':sha(grid),
        'headroom_definition':'optimized single / max(unique-weight HBM,peak MAC,compulsory port traffic)-1; actual mapped port service is conditional attribution, not architecture-independent floor',
        'geometry_domain':'PM1..16, PN1..192, independent PK32/64/128/256/512/1024; exact 12288 main multipliers',
        'objective':'unweighted geomean of paired per-window latency ratios to common fixed reference; equivalent ranking by geomean latency',
        'selection_unit':'one frozen hardware/runtime point per family per credit/format regime, ALL development batches together',
        'screen':'all geometries, equal and MAC-proportional initial quotas; both rerun twice',
        'port_protocol':'physical padded reservations/full issued MACs; actual valid_m/n/k SRAM transfers as in Rust; 16B output-word rounding',
        'compiler_group_search':'GU/Down independently {auto,1,2,4,8}; primary commonGU2Down4; fixed8pairedrecords; no claim of optimal descriptor count or streaming-Z dataflow',
        'refinement':'top8 development geometries per family plus stronger full-single/equal-homogeneous WS/OS baseline seeds; three starts .25/.5/.75, two coordinate sweeps plus32 fixed random cross-axis probes',
        'strong_baselines':'all44single and all41homogeneous shapes xWS/OS xprefetch2/6/32 x25GU/Down group pairs; independent quota refinement additionally applied to homogeneous finalists',
        'independent_axes':FIELDS,'fractions':FRACTIONS,
        'runtime':'after hardware quota selection, policies EFT/idle/RR/threshold1,2,4,8,16 and prefetch2/6/32; dev only',
        'search_limitation':'hierarchical finite search, NOT exhaustive joint allocation/geometry/runtime optimum',
        'repeats':2,'clock':'hypothetical1GHz','SRAM_bytes':2158592,
        'W8_W4':'ideal packed transport/coalescing, ideal decoder primary; BF16 operands still occupy installed slots; no accuracy qualification',
        '512credit':'252.061538GB/s continuous credit cap, fixed8KiB ingress with backpressure;512 extra tag bytes inside16KiBcontrol; not native-implemented credit expansion',
        'input_development_sha256':{p.name:sha(p) for p in inputs.glob('*development.json')},
        'source_sha256':source_hashes(),'test_limit':limit,
        'gates':{'calibration_entry_vs_both_percent':5,'victory_calibrated_percent':10,'bootstrap95_lower_percent':5,'area_energy_saving_percent':20,'area_energy_active':False}}
    path=out/'PREREGISTERED.json'
    if path.exists(): assert path.read_bytes()==canonical(contract),'contract or source changed'
    else: write_json(path,contract)
    reused=out/'REUSED_PRIMARY_RECEIPT.json'
    if reused.exists():
        receipt=json.loads(reused.read_text())
        # Only selection coverage changed; every timing module remains exact.
        now=source_hashes()
        assert all(now[k]==v for k,v in receipt['timing_sources'].items() if k!='regime/search.py')
        for name,digest in receipt['files'].items():assert sha(out/name)==digest
    frozen=[]; began=time.time()
    with ProcessPoolExecutor(max_workers=workers,initializer=init_worker,initargs=(dev,)) as pool:
        for credits,fmt in regimes:
            sub=out/f'{fmt}_c{credits}';sub.mkdir(exist_ok=True)
            screen_path=sub/'geometry_screen.json'
            if screen_path.exists(): screen=json.loads(screen_path.read_text())
            else:
                screen=[]
                for i,r in enumerate(pool.map(screen_one,((g,credits,fmt) for g in geometries),chunksize=16)):
                    screen.append(r)
                    if i%2000==0: print(f'{fmt}/{credits}: geometry {i}/{len(geometries)} elapsed{time.time()-began:.0f}s',flush=True)
                write_json(screen_path,screen)
                write_csv(sub/'geometry_screen.csv',[csv_row(r) for r in screen])
            baseline_path=sub/'strong_baseline_candidates.json'
            if baseline_path.exists():baseline=json.loads(baseline_path.read_text())
            else:
                jobs=((point(g,flow=flow,slots=slots,gu=gu,down=down),credits,fmt) for g in geometries
                      if family(g) in ('single','homogeneous')
                      for flow in ('bounded_ws','bounded_os') for slots in (2,6,32)
                      for gu in GROUPS for down in GROUPS)
                baseline=list(pool.map(baseline_one,jobs,chunksize=8))
                write_json(baseline_path,baseline)
                write_csv(sub/'strong_baseline_candidates.csv',[csv_row(r) for r in baseline])
            seeds=[]
            for fam in ('single','homogeneous','heterogeneous'):
                seeds += sorted((r for r in screen if r['legal'] and r['family']==fam),key=lambda r:(r['score_ms'],r['geometry']))[:8]
            for fam in ('single','homogeneous'):
                best=min((r for r in baseline if r['legal'] and r['family']==fam),key=lambda r:(r['score_ms'],r['geometry']))
                if best['geometry'] not in {r['geometry'] for r in seeds}:seeds.append(best)
            if {r['family'] for r in seeds}!= {'single','homogeneous','heterogeneous'}:
                raise ValueError('screen requires all three baseline families')
            refine_path=sub/'independent_allocations.json'
            if refine_path.exists(): refined=json.loads(refine_path.read_text())
            else:
                refined=[r for rs in pool.map(refine_one,((r,credits,fmt) for r in seeds)) for r in rs]
                write_json(refine_path,refined)
                write_csv(sub/'independent_allocations.csv',[csv_row(r) for r in refined])
            runtime=[]
            for fam in ('single','homogeneous','heterogeneous'):
                best=min((r for r in refined+baseline if r['legal'] and r['family']==fam),key=lambda r:(r['score_ms'],r['geometry'],canonical(r['partition'])))
                p={k:best[k] for k in point(cores_from(best['geometry']))}
                jobs=[]
                for policy in POLICIES:
                    for slots in (2,6,32):
                        for gu in GROUPS:
                            for down in GROUPS:
                                jobs.append(({**p,'policy':policy,'prefetch_slots':slots,'gu_group_limit':gu,'down_group_limit':down},credits,fmt))
                # Host parallelism only: each candidate retains the same two
                # complete runs and pool.map preserves deterministic order.
                runtime.extend(pool.map(baseline_one,jobs,chunksize=8))
                winner=min((r for r in runtime if r['legal'] and r['family']==fam),key=lambda r:(r['score_ms'],r['policy'],r['prefetch_slots']))
                frozen.append({'credits':credits,'weight_format':fmt,**winner})
            write_csv(sub/'runtime_development.csv',[csv_row(r) for r in runtime])
            allrows=screen+baseline+refined+runtime
            near=[]; stability=[]
            for fam in ('single','homogeneous','heterogeneous'):
                legal=[r for r in allrows if r['legal'] and r['family']==fam]
                unique={canonical({k:r[k] for k in point(cores_from(r['geometry']))}):r for r in legal}
                legal=list(unique.values());best=min(r['score_ms'] for r in legal)
                near +=[r for r in legal if r['score_ms']<=1.01*best]
                # Development selection stability, paired batch-stratified resampling.
                logs=np.log(np.array([r['window_cycles'] for r in legal]));rng=np.random.default_rng(20261005)
                groups=defaultdict(list)
                for i,w in enumerate(dev):groups[w['batch']].append(i)
                scores=np.zeros((len(legal),1000))
                for ids in groups.values():
                    drawn=rng.choice(ids,(1000,len(ids)),replace=True)
                    scores+=logs[:,drawn].sum(axis=2)
                counts=np.bincount(np.argmin(scores,axis=0),minlength=len(legal))
                stability +=[{'credits':credits,'weight_format':fmt,'family':fam,'geometry':r['geometry'],
                    'partition':json.dumps(r['partition'],sort_keys=True),'flow':r['flow'],'policy':r['policy'],
                    'prefetch_slots':r['prefetch_slots'],'bootstrap_selected_fraction':float(n/1000),
                    'scope':'development selection stability among evaluated candidates,1000 paired stratified resamples'}
                    for r,n in zip(legal,counts) if n]
            write_csv(sub/'within1percent.csv',[csv_row(r) for r in near])
            write_csv(sub/'selection_stability.csv',stability)
            print(f'{fmt}/{credits}: hardware/runtime selected; {len(refined)} independent quota probes',flush=True)
    write_json(out/'FROZEN_SELECTION.json',{'preregistered_sha256':sha(path),'source_sha256':source_hashes(),
        'heldout_consulted_for_selection':False,'points':frozen,'complete':True})
    print('All development selections frozen.',flush=True)


def final(inputs,out):
    frozen=json.loads((out/'FROZEN_SELECTION.json').read_text())
    assert source_hashes()==frozen['source_sha256'],'timing/selection source changed after freezing'
    held=[w for name in ('heldout.json','mixed_heldout.json') for w in json.loads((inputs/name).read_text())['workloads']]
    verify_capture(held)
    rows=[];cr=[];gates=[];bybatch=[];sensitivity=[];oracles=[];calibration=[]
    grouped=defaultdict(dict)
    for p in frozen['points']:grouped[(p['credits'],p['weight_format'])][p['family']]=p
    for (credits,fmt),points in sorted(grouped.items()):
        details={};settings={}
        for fam,p in points.items():
            s=settings_for(credits,fmt,p);settings[fam]=s
            got=evaluate(held,cores_from(p['geometry']),s,p['policy']);details[fam]=got
            rows+=result_rows(got,s,credits=credits,weight_format=fmt,family=fam,geometry=p['geometry'])
            cr +=[{'credits':credits,'weight_format':fmt,'family':fam,**r} for result in got for r in core_rows(result,s)]
            write_json(out/f'{fmt}_c{credits}'/f'heldout_{fam}.json',got)
        pass_all=True
        for ref in ('single','homogeneous'):
            stat=paired_bootstrap(details['heterogeneous'],details[ref])
            gate={'credits':credits,'weight_format':fmt,'baseline':ref,**stat,
                'entry_5percent':stat['time_reduction_percent']>=5,
                'calibrated':False,'victory':False,
                'reason':'uncalibrated3Dmodel; accuracy pending for W8/W4; area branch inactive'}
            gates.append(gate);pass_all &= gate['entry_5percent']
            for batch in sorted({r['batch'] for r in held}):
                sub=lambda rs:[r for r in rs if r['batch']==batch]
                bybatch.append({'credits':credits,'weight_format':fmt,'baseline':ref,'batch':batch,
                                **paired_bootstrap(sub(details['heterogeneous']),sub(details[ref]))})
        # Same-owner, same-hardware component oracles, never additive savings.
        for fam,p in points.items():
            cores=cores_from(p['geometry']);s=settings[fam]
            owners=[tuple(b['core'] for b in r['bindings']) for r in details[fam]]
            for name,alt in (('ideal_HBM',replace(s,hbm=False)),('ideal_ports',replace(s,ports=False)),
                             ('zero_control',replace(s,control=False)),('compute_only',replace(s,hbm=False,ports=False,control=False))):
                got=evaluate(held,cores,alt,p['policy'],owners=owners,detail=False)
                oracles.append({'credits':credits,'weight_format':fmt,'family':fam,'geometry':p['geometry'],
                     'oracle':name,'geomean_ms':geomean(r['latency_ms'] for r in got),
                     'geomean_charged_ratio':geomean(a['cycles']/b['cycles'] for a,b in zip(got,details[fam])),
                     'total_ms':sum(r['latency_ms'] for r in got),'owner':'charged_fixed'})
            for timing,profile in TIMING_PROFILES.items():
                got=evaluate(held,cores,replace(s,timing=profile),p['policy'],detail=False)
                sensitivity.append({'credits':credits,'weight_format':fmt,'family':fam,'geometry':p['geometry'],
                    'sensitivity':'PKlatency_'+timing,'geomean_ms':geomean(r['latency_ms'] for r in got),
                    'total_ms':sum(r['latency_ms'] for r in got)})
            if fmt!='BF16':
                for decode in (128,512):
                    got=evaluate(held,cores,replace(s,decoder_elements_per_cycle=decode),p['policy'],detail=False)
                    sensitivity.append({'credits':credits,'weight_format':fmt,'family':fam,'geometry':p['geometry'],
                        'sensitivity':f'decoder_{decode}_elements_per_cycle','geomean_ms':geomean(r['latency_ms'] for r in got),
                        'total_ms':sum(r['latency_ms'] for r in got)})
        calibration.append({'credits':credits,'weight_format':fmt,'passes_both_entry_gates':pass_all,
            'action':'requires6frozenpoints_on_matching3Dnative_backend' if pass_all else 'skip_native_candidate_calibration_by_preregistered_gate',
            'native3D_available':False,'area_model_available':False,'claim':'no architecture victory declared'})
        if pass_all:
            # Concrete reviewable calibration points, but no fabricated native
            # results. Current native supports one common PN/PK and separate
            # GEMMs; no matching independent-PN/PK pairedGU/Z/combine adapter.
            points6=[{**p,'label':'selected_'+fam} for fam,p in points.items()]
            for label,gid in (('fixed_6','6x4x512'),('fixed_3+3','3x4x512+3x4x512'),('fixed_4+2','2x4x512+4x4x512')):
                points6.append({'credits':credits,'weight_format':fmt,'label':label,
                               **point(cores_from(gid),'proportional')})
            write_json(out/f'{fmt}_c{credits}'/'SIX_FROZEN_CALIBRATION_POINTS.json',{
                'points':points6,'heldout_input_sha256':{p.name:sha(p) for p in inputs.glob('*heldout.json')},
                'backend_status':'matching3DpostrouterpairedGU_Z_Down_combine_adapter_not_implemented',
                'requirements':['same addresses/packing and row policy','same private arenas and port banks',
                    'independent PM/PN/PK per core','complete fused FFN timing boundary',
                    'same owners for attribution','two repeat-exact runs','native request/finite SRAM drain',
                    'actual per-PK compute latency calibration','BF16 tolerance; no cross-PK bitexact claim']})
    write_csv(out/'heldout_windows.csv',rows)
    write_csv(out/'heldout_by_batch.csv',aggregate(rows,('credits','weight_format','family','geometry','batch')))
    write_csv(out/'heldout_totals.csv',aggregate(rows,('credits','weight_format','family','geometry')))
    write_csv(out/'heldout_core_services.csv',cr)
    write_csv(out/'paired_gates.csv',gates);write_csv(out/'paired_gates_by_batch.csv',bybatch)
    write_csv(out/'timing_decoder_sensitivity.csv',sensitivity);write_csv(out/'fixed_owner_oracles.csv',oracles)
    write_json(out/'CALIBRATION_GATES.json',calibration)
    write_json(out/'COMPLETION.json',{'complete_analytical_search':True,'repeats':2,'heldout_windows_per_point':len(held),
        'frozen_points':len(frozen['points']),'heldout_input_sha256':{p.name:sha(p) for p in inputs.glob('*heldout.json')},
        'frozen_selection_sha256':sha(out/'FROZEN_SELECTION.json'),'source_sha256':source_hashes(),
        'native_calibrated':False,'area_synthesized':False,'trained_model_quant_quality_qualified':False})
    print('Frozen heldout results, paired gates and sensitivities complete.',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--grid',type=Path,required=True);p.add_argument('--stage',choices=('select','final','all'),default='all')
    p.add_argument('--workers',type=int,default=16);p.add_argument('--limit',type=int)
    a=p.parse_args()
    if a.stage in ('select','all'):select(a.inputs,a.out,a.grid,a.workers,a.limit)
    if a.stage in ('final','all'):final(a.inputs,a.out)


if __name__=='__main__':main()
