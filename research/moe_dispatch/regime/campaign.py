"""Common-input bounds, work-regime grid, independent-allocation DSE and gates."""
from dataclasses import asdict, replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse, csv, hashlib, json, math, statistics, time
from collections import Counter, defaultdict
from ..geometry3d.compute import Core, enumerate_geometries, family, geometry_id
from ..geometry3d.memory import FabricProfile
from ..geometry3d.study import cores_from, verify_capture, canonical, write_csv, write_json
from .model import Settings, simulate_layer, projection_phase, expert_phases
from .resources import Partition, memory_budget
from .metrics import bounds, core_rows, geomean, paired_score, paired_bootstrap


def load_inputs(root):
    ans={}
    for split in ('development','heldout'):
        ans[split]=[w for name in (split+'.json','mixed_'+split+'.json')
                    for w in json.loads((root/name).read_text())['workloads']]
        verify_capture(ans[split])
    return ans


def evaluate(workloads,cores,settings,policy='eft',repeats=2,owners=None,detail=True):
    def run():
        return [simulate_layer(w,cores,settings,policy,detail,
                None if owners is None else owners[i]) for i,w in enumerate(workloads)]
    first=run()
    if repeats==2:
        assert first==run(),'repeat mismatch'
    return first


def result_rows(results,settings,**tags):
    return [{**tags,**{k:r[k] for k in ('workload','batch','latency_ms','hbm_bytes','native_unique_bytes','useful_macs','issued_macs','spatial_utilization','X_sram_bytes')},
             **bounds(r,settings)} for r in results]


def aggregate(rows,keys):
    groups=defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in keys)].append(r)
    ans=[]
    for group,rs in sorted(groups.items(),key=lambda x:str(x[0])):
        out=dict(zip(keys,group)); total=sum(r['latency_ms'] for r in rs)
        out.update(windows=len(rs),total_ms=total,mean_ms=total/len(rs),
            hbm_GiB=sum(r['hbm_bytes'] for r in rs)/2**30,
            hbm_lb_ms=sum(r['hbm_lb_ms'] for r in rs),
            unique_weight_hbm_lb_ms=sum(r['unique_weight_hbm_lb_ms'] for r in rs),
            compute_fixed_owner_lb_ms=sum(r['compute_fixed_owner_lb_ms'] for r in rs),
            port_fixed_mapping_lb_ms=sum(r['port_fixed_mapping_lb_ms'] for r in rs),
            max_conditional_lb_ms=sum(r['max_conditional_lb_ms'] for r in rs),
            global_HBM_MAC_lb_ms=sum(r['global_HBM_MAC_lb_ms'] for r in rs),
            architecture_search_lb_ms=sum(r['architecture_search_lb_ms'] for r in rs),
            port_compulsory_lb_ms=sum(r['port_compulsory_lb_ms'] for r in rs),
            compute_peak_lb_ms=sum(r['compute_peak_lb_ms'] for r in rs),
            architecture_headroom_percent=100*(geomean(r['latency_ms']/r['architecture_search_lb_ms'] for r in rs)-1),
            gap_over_conditional_percent=100*(geomean(r['latency_ms']/r['max_conditional_lb_ms'] for r in rs)-1),
            gap_over_global_HBM_MAC_percent=100*(geomean(r['latency_ms']/r['global_HBM_MAC_lb_ms'] for r in rs)-1),
            time_sum_over_hbm_percent=100*(total/sum(r['hbm_lb_ms'] for r in rs)-1),
            bandwidth_GBps=sum(r['hbm_bytes'] for r in rs)/(total*1e6),
            hbm_equivalent_busy_fraction=sum(r['hbm_lb_ms'] for r in rs)/total)
        ans.append(out)
    return ans


def diagnose(inputs,out,old):
    out.mkdir(parents=True,exist_ok=True)
    workloads=load_inputs(inputs)
    stats=[]; hist=[]
    for split,ws in workloads.items():
        for batch in sorted({w['batch'] for w in ws}):
            sub=[w for w in ws if w['batch']==batch]
            rs=[e['Me'] for w in sub for e in w['experts'] if not e['is_shared']]
            stats.append({'split':split,'batch':batch,'windows':len(sub),
                          'routed_experts':len(rs),'shared_experts':len(sub),
                          'routed_Me_min':min(rs),'routed_Me_max':max(rs),
                          'routed_Me_mean':statistics.mean(rs),
                          'tasks_Me_gt2':sum(e['Me']>2 for w in sub for e in w['experts']),
                          'tasks_Me_ge2':sum(e['Me']>=2 for w in sub for e in w['experts'])})
            for shared in (False,True):
                for m,n in sorted(Counter(e['Me'] for w in sub for e in w['experts'] if e['is_shared']==shared).items()):
                    hist.append({'split':split,'batch':batch,'is_shared':shared,'Me':m,'experts':n})
    write_csv(out/'window_counts.csv',stats); write_csv(out/'Me_histogram.csv',hist)
    frozen=json.loads((old/'FROZEN_SELECTION.json').read_text())['points']
    previous=json.loads((old/'heldout_details.json').read_text())
    rows=[]; cr=[]; proof=[]
    for point in frozen:
        s=Settings(allocation=point['allocation'],flow=point['flow'],prefetch_slots=point['prefetch_slots'])
        got=evaluate(workloads['heldout'],cores_from(point['geometry']),s,point['policy'])
        for a,b in zip(got,previous[point['label']]):
            for k in ('cycles','hbm_bytes','native_unique_bytes','useful_macs','issued_macs','X_sram_bytes','core_finish_cycles'):
                assert a[k]==b[k],(point['label'],a['workload'],k)
        proof.append({'label':point['label'],'windows':len(got),'complete_repeats':2,'frozen_v6_metrics_exact_match':True})
        rows+=result_rows(got,s,label=point['label'],geometry=point['geometry'],split='heldout')
        cr +=[{**{'label':point['label']},**r} for result in got for r in core_rows(result,s)]
    write_csv(out/'unified_windows.csv',rows)
    write_csv(out/'unified_by_batch.csv',aggregate(rows,('label','geometry','batch')))
    write_csv(out/'unified_totals.csv',aggregate(rows,('label','geometry')))
    write_csv(out/'core_service_breakdown.csv',cr)
    write_json(out/'v6_reproduction.json',proof)
    grid=[]; gridcores=[]
    for credits in (256,512):
        for fmt in ('BF16','W8','W4'):
            # Geometry remains the optimized prior single for this cheap screen.
            s=Settings(allocation='equal',weight_format=fmt,fabric=replace(FabricProfile(),hbm_credits=credits))
            for split,ws in workloads.items():
                got=evaluate(ws,(Core(6,16,128),),s)
                grid+=result_rows(got,s,credits=credits,weight_format=fmt,split=split,
                                  geometry='6x16x128',decoder='ideal_oracle' if fmt!='BF16' else 'not_needed')
                gridcores +=[{'credits':credits,'weight_format':fmt,'split':split,**r} for result in got for r in core_rows(result,s)]
    write_csv(out/'regime_grid_windows.csv',grid)
    summary=aggregate(grid,('credits','weight_format','split','batch'))
    for r in summary:
        r['search_eligible_development']=r['split']=='development' and r['gap_over_conditional_percent']>=10
        r['space_status']='candidate_space' if r['gap_over_conditional_percent']>=10 else ('below_3pct_screening_margin' if r['gap_over_conditional_percent']<3 else 'limited_space_3_to_10pct')
        r['format_status']='reference' if r['weight_format']=='BF16' else 'transport_oracle; no accuracy qualification'
    write_csv(out/'regime_grid.csv',summary)
    write_csv(out/'regime_core_services.csv',gridcores)
    sensitivity=[]
    for credits in (256,512):
        for fmt in ('W8','W4'):
            for decode in (128,512):
                s=Settings(allocation='equal',weight_format=fmt,decoder_elements_per_cycle=decode,
                           fabric=replace(FabricProfile(),hbm_credits=credits))
                r=result_rows(evaluate(workloads['development'],(Core(6,16,128),),s),s,
                              credits=credits,weight_format=fmt,decoder_elements_per_cycle=decode)
                sensitivity+=aggregate(r,('credits','weight_format','decoder_elements_per_cycle','batch'))
    write_csv(out/'decoder_sensitivity_development.csv',sensitivity)
    write_json(out/'DIAGNOSTIC_RECEIPT.json',{'complete':True,'repeats':2,'v6_windows_exact_match':945,
               'grid_windows':len(grid),'grid_cells':len(summary),'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.glob('*.json')},
               'format_scope':'W8/W4 transport/coalescing and decoder hypotheses, BF16 datapath; not trained-model quality',
               'screening_error_margin_percent':3,'screening_error_margin_calibrated':False,
               'search_space_gate_percent':10,'calibration_entry_both_baselines_percent':5,
               'victory_after_calibration_percent':10,'bootstrap95_lower_required_percent':5,
               'area_or_energy_gate_percent':20,'area_or_energy_gate_active':False})
    print('Diagnostic and regime grid complete.',flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--old',type=Path,required=True); p.add_argument('--stage',choices=['diagnose'],default='diagnose')
    a=p.parse_args(); diagnose(a.inputs,a.out,a.old)


if __name__=='__main__': main()
