"""Plot-reader integration checks using temporary synthetic fixture tables only.

These values verify file schemas and rendering, not architecture performance.
They are never written into the repository's formal results directories.
"""
import csv, importlib.util, itertools, json, sys
from pathlib import Path


def _module():
    spec=importlib.util.spec_from_file_location('round2_draft_figures',Path(__file__).with_name('figures.py'))
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module);return module


def test_all_ten_readers_render_with_explicit_temporary_fixture_tables(tmp_path):
    p=_module();root=tmp_path/'TEST_ONLY_NOT_EXPERIMENT_RESULTS';results=root/'results'
    def write(name,rows):
        dest=results/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def js(name,obj):
        dest=results/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(obj))
    write('E1/bounds_per_window.csv',[dict(design='B1',onchip_mode=m,batch=b,latency_ms=1,hbm_floor_unique=.5,hbm_floor_actual=.6,mac_floor=.1,port_floor=.2,task_floor=.2) for m,b in itertools.product(p.MODES,p.BATCHES)])
    names=('B0','B1','B2','fixed_3+3','fixed_4+2','best_5+1','best_4+2','best_2+4','best_hetero','U1','U2')
    write('E4/heldout_main_table.csv',[dict(entry=n,onchip_mode=m,sched_type='runtime',**{f'B{b}':1+j*.1 for b in p.BATCHES}) for m in p.MODES for j,n in enumerate(names)])
    write('E4/per_window.csv',[dict(entry=n,onchip_mode=m,sched_type='runtime',cycles=1e6,hbm_busy_frac=.6,w_port_busy=3e5,x_port_busy=1e5,acc_port_busy=1e5,core0_compute_busy=2e5,core1_compute_busy=1e5) for m,n in itertools.product(p.MODES,('B1','B2','best_hetero'))])
    for mode in p.MODES:
        js(f'E3/bnb_{mode}_A.json',dict(families={f:dict(declared_lattice_points=1000) for f in p.FAMILIES},certificate=[dict(family=f,status='lower_bound_pruned',depth=2,lattice_points=100) for f in p.FAMILIES],leaves=[dict(family=f,geomean_ms=.9) for f in p.FAMILIES]))
        js(f'E3/seed_points_{mode}.json',[dict(family=f,geomean_ms=1) for f in p.FAMILIES])
        history=results/f'E3/seed_points_{mode}_partial.jsonl'
        history.write_text(''.join(json.dumps(dict(family=f,geomean_ms=1))+'\n' for f in p.FAMILIES))
    write('E3/workload_map.csv',[dict(E=64,topk=6,F=1408,shared_units=2,bw_or_mac_scale=bw,batch=b,concentration=c,delta_vs_single_pct=-c,proof_complete=False) for bw,b,c in itertools.product((126.030769,252.061538,504.123077),(2,4,8,16,32,64,128,256),range(5))])
    write('E3/synthetic_calibration.csv',[dict(window_id=f'fixtureB{b}',batch=b,concentration_level=2,me_hist_KL_real_to_synthetic=.1) for b in p.BATCHES])
    params=('weight_tile_service_cycles','bank_Bpc','dotstagecycles','credits','vector_scale')
    write('E3/sobol.csv',[dict(param=k,S1=.1,ST=.2,S1_ci=.01,ST_ci=.02,all_searches_certified=False) for k in params])
    write('E3/flip_samples.csv',[dict(param=k,value=x,delta=.1-.1*x,proof_complete=False) for k,x in itertools.product(params,(0,1,2))])
    write('E3/flip_boundary.csv',[dict(param=k,value=1) for k in params])
    write('E2/layer_grid.csv',[dict(design=n,onchip_mode=m,batch='all',df_big=a,df_small=b,ratio_vs_OS_OS=1) for m,n in itertools.product(p.MODES,('B1','fixed_4+2','previous_asym','best_hetero')) for a,b in ([(a,'') for a in p.FLOWS] if n=='B1' else itertools.product(p.FLOWS,p.FLOWS))])
    write('E2/micro.csv',[dict(shape=s,dataflow=f,expert_type='routed',onchip_mode=m,Me=me,cycles=me*100) for s,f,m,me in itertools.product(('6x16x128','6x4x512','1x2x64','5x19x128','4x4x512','2x4x512'),p.FLOWS,p.MODES,(1,2,3,4,6,8,12,16,32,64,128))])
    write('E5/predictor_table.csv',[dict(design=n,onchip_mode=m,predictor=pred,mae_pct=10,e2e_ratio_vs_oracle=1.1) for n,m,pred in itertools.product(('best_hetero','fixed_4+2'),p.MODES,('random','static','btb','ema','ours','oracle'))])
    plots=p.Figures(root)
    for name in p.METHODS:getattr(plots,name)()
    plots.manifest()
    assert len(plots.outputs)==20
    assert all(x.stat().st_size>1000 for x in plots.outputs)
    provenance=json.loads((plots.out/'FIGURE_PROVENANCE.json').read_text())
    assert len(provenance['outputs'])==20 and len(provenance['input_sha256'])>=15
    assert all(str(tmp_path) in path for path in provenance['input_sha256'])


def test_missing_input_never_creates_placeholder_or_fabricated_values(tmp_path):
    import pytest
    p=_module();plots=p.Figures(tmp_path)
    with pytest.raises(FileNotFoundError):plots.headroom()
    assert not plots.outputs
