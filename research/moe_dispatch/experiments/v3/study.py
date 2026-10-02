#!/usr/bin/env python3
"""Frozen, resumable v3 experiment matrix; every accepted point repeats twice.

Analytical event simulation, never hardware/Ramulator measurement. Captured
routes are kept verbatim. True mixed inputs must carry ordered prompt routes;
constructed inputs have an explicit provenance category and separate tables.
"""
from __future__ import annotations
import argparse,copy,csv,gzip,hashlib,importlib.util,json,math,os,re,shutil,subprocess,sys,tempfile,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
RESEARCH=HERE.parents[1]
SIM=RESEARCH.parents[1]
PYTHON=Path('/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python')
DEFAULT_ROOT=SIM/'outputs/moe_supply_first_v3'
OLD=Path('/tmp/plena-joint-runtime-20260930')
BATCHES=(2,4,8,16)
LAYERS=(2,13,26)
STEPS=(3,7,11)
OPS={
 'OP0':dict(precision='P0',hbm_bytes_per_ns=256,hbm_latency_ns=64,credits=256),
 'OP1':dict(precision='P0',hbm_bytes_per_ns=256,hbm_latency_ns=64,credits=544),
 'OP2':dict(precision='P2',main_bits=4,hbm_bytes_per_ns=256,hbm_latency_ns=64,credits=544),
 'OP3':dict(precision='P2',main_bits=4,hbm_bytes_per_ns=256,hbm_latency_ns=150,credits=1232),
 'OP4':dict(precision='P2',main_bits=3,hbm_bytes_per_ns=256,hbm_latency_ns=64,credits=544),
 'OP5':dict(precision='P1',main_bits=4,hbm_bytes_per_ns=256,hbm_latency_ns=64,credits=544),
}
DESIGNS={
 'BL0':dict(lanes=[4,2],dataflow=['legacy','legacy'],arch='joint_v1'),
 'BL1':dict(lanes=[6],dataflow=['legacy'],arch='joint_v1'),
 'BL2':dict(lanes=[6],dataflow=['switchable'],contexts_per_core=2),
 'BL3':dict(lanes=[3,3],dataflow=['switchable','switchable']),
 'BL4':dict(lanes=[4,2],dataflow=['ws_group','is_stream']),
 'BL5':dict(lanes=[4,2],dataflow=['switchable','switchable']),
 'M8_single':dict(lanes=[8],dataflow=['switchable'],contexts_per_core=2),
 'M8_homogeneous':dict(lanes=[4,4],dataflow=['switchable','switchable']),
 'M8_asym62':dict(lanes=[6,2],dataflow=['ws_group','is_stream']),
 'M8_asym53':dict(lanes=[5,3],dataflow=['ws_group','is_stream']),
 'M8_flex62':dict(lanes=[6,2],dataflow=['switchable','switchable']),
 'M8_flex53':dict(lanes=[5,3],dataflow=['switchable','switchable']),
}
_LEGACY_LAYOUT_CACHE={}
_LEGACY_RESOURCE_CACHE=None
_LEGACY_CHUNKS_MODULE=None
FROZEN_FORMAT={}
FORMAT_RECEIPT=None
FORMAT_KEYS=('main_bits','factor_a','factor_b','rank_lanes','ranks')
SWITCHES=('credit_release','byte_pool','prefetch_quota','pipeline_supply','x_reuse','w_reuse','wide_ports','inline_silu')
BASE=dict(arch='supply_v3',main_bits=4,factor_a='mxint4',factor_b='bf16',rank_lanes=8,
 ranks=dict(routed=[32,32,24],shared=[32,32,48]),comp_mode='lanes',
 hbm_bytes_per_ns=256,hbm_latency_ns=64,credits=544,credit_release='ingress',
 ingress_bytes=8192,pool_bytes=65536,pool_banks=128,quota_policy='little',quota_margin=64,
 pool_read_per_core=[512,512],x_port=[512,128],wor_tiles=[8,1],dec_bw=1024,vlen=64,
 contexts_per_core=2,context_interleave=True,context_switch_policy='starved',work_steal=True,dot_latency=20,z_mode='auto',placement='supply_ipd',window=8,
 guard_kappa=2,guard_m0=64,control_reserve=16384,
 pipeline_supply=True,x_reuse=True,w_reuse=True,inline_silu=True,byte_pool=True,
 prefetch_quota=True,wide_ports=True,iso_port=True,ideal_hbm=False,ideal_onchip=False,
 record_trace=False,rank_alloc='static',rank_selection='common')

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 # Content-addressed blobs can be first-created by several workers together.
 # Give each writer a unique temporary name before atomic publication.
 with tempfile.NamedTemporaryFile(mode='w',prefix=p.name+'.',suffix='.tmp',dir=p.parent,delete=False) as stream:
  stream.write(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp=Path(stream.name)
 tmp.replace(p)
def frozen(p,x):
 p=Path(p)
 if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(x)),f'Frozen input changed: {p}'
 else:write(p,x)
def csvwrite(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 fields=sorted({k for r in rows for k in r});tmp=p.with_suffix('.csv.tmp')
 with tmp.open('w',newline='') as f:
  writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
 tmp.replace(p)

def load_format(path):
 """Load only the preregistered numerical format, never workload-specific tuning."""
 global FROZEN_FORMAT,FORMAT_RECEIPT
 value=json.loads(Path(path).read_text());fmt=value.get('config',value)
 allowed={'main_bits','factor_a','factor_b','rank_lanes','ranks'}
 assert set(fmt)<=allowed,f'Frozen format contains non-format fields: {set(fmt)-allowed}'
 assert fmt.get('main_bits',4) in (3,4)
 assert fmt.get('rank_lanes',8) in (8,16)
 for key in ('routed','shared'):
  if 'ranks' in fmt:assert len(fmt['ranks'][key])==3 and all(isinstance(x,int) and x>=0 for x in fmt['ranks'][key])
 FROZEN_FORMAT=copy.deepcopy(fmt)
 FORMAT_RECEIPT={'path':str(Path(path).resolve()),'sha256':sha(path),'quality_status':value.get('quality_status','timing_candidate_not_accuracy_validated')}

def physical_format():
 """Actual global format, including defaults; quality metadata is not timing."""
 return {key:copy.deepcopy(FROZEN_FORMAT.get(key,BASE[key])) for key in FORMAT_KEYS}

def planner_artifacts():
 selected=Path(os.environ.get('PLENA_DISPATCH_COMPILER','/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-compiler/research/moe_dispatch')).resolve()
 return {'compiler.py':selected/'compiler.py','legacy_chunks.py':selected/'legacy_chunks.py','frontend.py':RESEARCH/'frontend.py',
  'legacy_frozen_designs.json':Path('/scratch/shared/mcl123/plena/outputs/moe_robust_fixed_20260930/frozen_designs.json')}

def prepare_legacy_workload(workload,lanes,group,resources):
 """Pinned finite-memory outer plan; legacy kernels and original routes remain intact."""
 global _LEGACY_CHUNKS_MODULE
 selected=planner_artifacts()['legacy_chunks.py'].resolve()
 if _LEGACY_CHUNKS_MODULE is None:
  spec=importlib.util.spec_from_file_location('plena_dispatch_legacy_chunks',selected)
  module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
  _LEGACY_CHUNKS_MODULE=module
 assert Path(_LEGACY_CHUNKS_MODULE.__file__).resolve()==selected,'Cached legacy chunk planner differs from the frozen selected source'
 return _LEGACY_CHUNKS_MODULE.prepare_legacy_workload(workload,lanes,group,resources)

def campaign_signature(root,binary):
 root=Path(root)
 fmt=physical_format()
 return dict(binary_sha256=sha(binary),study_sha256=sha(__file__),manifest_sha256=sha(root/'input_manifest.json'),
  input_bundle_sha256={name:sha(root/'inputs'/f'{name}.json') for name in ('development','heldout','constructed','mixed_development','mixed_heldout') if (root/'inputs'/f'{name}.json').exists()},
  planner_artifacts_sha256={name:sha(path) for name,path in planner_artifacts().items()},
  physical_format_sha256=digest(fmt),physical_format=fmt)

def verify_heldout_authorization(root,signature):
 permit=Path(root)/'heldout_authorization.json';assert permit.exists(),'Heldout requires root authorization receipt after committed prereg'
 auth=json.loads(permit.read_text());assert auth.get('prereg_commit') and auth.get('authorized') is True
 assert auth.get('frozen_signature')==signature,'Heldout authorization does not cover this exact binary, runner, physical format and input signature'
 assert 'quality_receipt_sha256' in auth,'Heldout authorization must freeze the final quality receipt SHA separately'
 assert auth['quality_receipt_sha256']==(FORMAT_RECEIPT['sha256'] if FORMAT_RECEIPT else None),'Heldout authorization does not cover this final qualification receipt'
 return auth

def window(a,meta,indices,step,layer,name):
 li=list(a['layer_ids']).index(layer);tokens=[];assign={}
 for t,i in enumerate(indices):
  assert a['valid'][i,step]
  routes=[]
  for slot,(eid,score) in enumerate(zip(a['decode_idx'][i,step,li],a['decode_weight'][i,step,li])):
   eid=int(eid);score=float(score);assign.setdefault(eid,[]).append((t,slot,score))
   routes.append(dict(expert_id=eid,slot=slot,score=score))
  tokens.append(dict(token_index=t,sample_id=str(a['sample_ids'][i]),routes=routes))
 return from_tokens(tokens,meta,name,'captured_decode',dict(layer=layer,step=step))

def from_tokens(tokens,meta,name,provenance,extra=None):
 h=int(meta['hidden_size']);assign={}
 for t in tokens:
  for r in t['routes']:assign.setdefault(int(r['expert_id']),[]).append((t['token_index'],r['slot'],r['score']))
 experts=[]
 for eid in [-1]+sorted(assign):
  shared=eid==-1;f=int(meta['shared_expert_intermediate_size'] if shared else meta['moe_intermediate_size'])
  rows=[(t['token_index'],-1,1.) for t in tokens] if shared else assign[eid]
  weights={}
  for pi,(phase,n,k) in enumerate((('gate',f,h),('up',f,h),('down',h,f))):
   weights[phase]=dict(shape_nk=[n,k],dtype='BF16',hbm_base=((int(meta['routed_experts']) if shared else eid)*3+pi)*33554432,
    row_stride_bytes=k*2,payload_bytes=n*k*2,physical_bytes=n*k*2,payload_scope='capture dimensions only; weights not numeric payload')
  experts.append(dict(id=eid,is_shared=shared,Me=len(rows),H=h,F=f,token_indices=[r[0] for r in rows],route_slots=[r[1] for r in rows],route_scores=[r[2] for r in rows],weights=weights))
 out=dict(id=name,batch=len(tokens),hidden=h,top_k=int(meta['top_k']),tokens=tokens,experts=experts,
  routing_is_input_not_timed=True,route_scores_renormalized=False,descriptor_order='shared_first',provenance=provenance,
  scope='one captured MoE FFN layer, router and attention excluded; shape-only analytical simulation')
 if extra:out.update(extra)
 return out

class CachedArchive(dict):
 @property
 def files(self):return list(self.keys())

def prepare(root):
 root=Path(root);old=json.loads((OLD/'workload_manifest.json').read_text())
 # Both old robust and old Joint requests were exposed; neither is new test data.
 excluded=set(old.get('excluded_request_ids',[]))
 for w in old['windows']:excluded.update(w['sample_ids'])
 dev=json.loads((OLD/'inputs/test_workloads.json').read_text())['workloads']
 for w in dev:
  w.pop('engine_layout',None);w['experts'].sort(key=lambda e:not e['is_shared'])
  w.update(provenance='captured_decode',split='development',descriptor_order='shared_first')
 sources=[];held=[];construct=[];meta0=None;selected=set();dataset_pools={}
 for source in old['sources']:
  ds=source['dataset'];path=Path(source['path']);assert sha(path)==source['sha256']
  archive=np.load(path,allow_pickle=False);a=CachedArchive({k:archive[k] for k in archive.files});meta=json.loads(a['meta'].item());meta0=meta
  available=[]
  for i,sid in enumerate(a['sample_ids']):
   ident=ds+':'+str(sid)
   if ident not in excluded and all(bool(a['valid'][i,s]) for s in STEPS):
    available.append((hashlib.sha256(('plena-v3-holdout:'+ident).encode()).hexdigest(),i,ident))
  available.sort();assert len(available)>=126,(ds,len(available))
  dataset_pools[ds]=[x[2] for x in available]
  cursor=0;groups={}
  for batch in BATCHES:
   picked=available[cursor:cursor+batch];cursor+=batch;groups[batch]=picked;selected.update(x[2] for x in picked)
  for layer in LAYERS:
   for step in STEPS:
    for batch,picked in groups.items():
     name=f'heldout_{ds}_b{batch}_l{layer}_s{step}'
     w=window(a,meta,[x[1] for x in picked],step,layer,name)
     w.update(split='heldout',dataset=ds,request_ids=[x[2] for x in picked],source_sha256=sha(path));held.append(w)
  # Constructed b32/b64 = offline rebatching captured requests, distinct from heldout groups.
  for batch in (32,64):
   picked=available[cursor:cursor+batch];cursor+=batch
   name=f'constructed_{ds}_b{batch}_l13_s7';w=window(a,meta,[x[1] for x in picked],7,13,name)
   w.update(split='constructed',dataset=ds,request_ids=[x[2] for x in picked],provenance='constructed_offline_rebatch',constructed=True);construct.append(w)
  sources.append(dict(dataset=ds,path=str(path),sha256=sha(path),request_count=len(a['sample_ids']),layers=list(map(int,a['layer_ids'])),steps=int(a['valid'].shape[1]),heldout_request_count=30,
   remaining_eligible=len(available),fields={k:dict(shape=list(a[k].shape),dtype=str(a[k].dtype)) for k in a.files},
   prefill_order_available=False,numerical_X_available=False))
 # Synthetic Zipf, clearly separate from continuous true prompt routes.
 for t in (32,64,96,128):
  for alpha in (.6,.8,1.):
   rng=np.random.default_rng(int(t*1000+alpha*100));e=int(meta0['routed_experts']);k=int(meta0['top_k']);p=np.arange(1,e+1,dtype=float)**(-alpha);p/=sum(p)
   tokens=[]
   for i in range(t):
    ids=rng.choice(e,size=k,replace=False,p=p);scores=p[ids]/sum(p[ids])
    tokens.append(dict(token_index=i,sample_id=f'zipf{alpha}-{i}',routes=[dict(expert_id=int(eid),slot=j,score=float(scores[j])) for j,eid in enumerate(ids)]))
   w=from_tokens(tokens,meta0,f'constructed_zipf_T{t}_a{alpha}','constructed_zipf',dict(split='constructed',constructed=True,dataset='synthetic_zipf',mix_description=f'proxy T={t}; not captured continuous prefill'))
   construct.append(w)
 manifest=dict(schema='plena_supply_v3_inputs_1',sources=sources,excluded_request_ids=sorted(excluded),heldout_request_ids=sorted(selected),
  request_disjoint=not bool(selected&excluded),development_windows=len(dev),heldout_windows=len(held),constructed_windows=len(construct),
  heldout_selection='SHA256(plena-v3-holdout:dataset:sample_id); require valid steps3/7/11; first30; fixed B groups reused at layers2/13/26 & steps3/7/11',
  correlation='108 route windows use90 unique requests; groups reused across layers/steps, so windows are correlated',
  real_mixed_status='unavailable: existing NPZ stores aggregate prefill_counts, not ordered per-token prompt routes',
  real_numerical_status='unavailable in route archives: no X or numerical expert payload',
  phases_excluded=['router_compute','attention','full_model'],host_time_not_latency=True)
 for split,ws in [('development',dev),('heldout',held),('constructed',construct)]:frozen(root/'inputs'/f'{split}.json',dict(workloads=ws))
 frozen(root/'input_manifest.json',manifest)
 write(root/'selection_request_ids.json',dict(development=sorted(excluded),heldout=sorted(selected),available_by_dataset=dataset_pools))
 print(json.dumps({k:manifest[k] for k in ('development_windows','heldout_windows','constructed_windows','request_disjoint')},sort_keys=True))
 return manifest

def config(design,op,port='iso',changes=None):
 cfg=copy.deepcopy(BASE);cfg.update(copy.deepcopy(DESIGNS[design]));cfg.update(OPS[op]);cfg['iso_port']=port=='iso'
 if cfg['precision']!='P0':
  cfg.update(copy.deepcopy(FROZEN_FORMAT))
  if op=='OP4':cfg['main_bits']=3
 n=len(cfg['lanes']);cfg['pool_read_per_core']=[1024] if n==1 else [512,512]
 if port=='iso':cfg['x_port']=[640] if n==1 else ([320,320] if cfg['lanes'][0]==cfg['lanes'][1] else [512,128])
 else:cfg['x_port']=[512] if n==1 else ([512,512] if cfg['lanes'][0]==cfg['lanes'][1] else [512,128])
 cfg['dec_bw']=2048 if port=='iso' and n==1 else 1024
 cfg['wor_tiles']=[8] if n==1 else ([4,4] if cfg['lanes'][0]==cfg['lanes'][1] else [8,1])
 if port=='iso':cfg['wor_total_slots']=18
 if cfg['precision']=='P0':cfg.update(comp_mode='none',rank_lanes=0)
 if design in ('BL0','BL1'):
  cfg.update(dispatch='joint' if design=='BL0' else 'fifo',group=4,runtime_fsm=True,next_prefetch=True,arbiter='stock',tail_partition=False,window=8)
  if op=='OP1':cfg.update(diagnostic_credit_expansion=True,legacy_diagnostic_overbudget=True)
 if design=='BL1':
  cfg.update(placement='fifo',credit_release='landing',byte_pool=False,prefetch_quota=False,pipeline_supply=False,x_reuse=False,w_reuse=False,wide_ports=False,inline_silu=False)
 if changes:cfg.update(copy.deepcopy(changes))
 cfg['t_chunk']=128 if cfg['precision']=='P2' and cfg['rank_lanes']==8 and cfg['hbm_bytes_per_ns']!=512 else 96
 return cfg

def legal(w,cfg,design,op):
 if design in ('BL0','BL1') and op not in ('OP0','OP1'):return 'legacy baselines use P0; byte-only precision counterfactual is M0 separately'
 if w['batch']>96 and (cfg['precision']!='P2' or cfg['rank_lanes']==16 or cfg['hbm_bytes_per_ns']==512):return 'frozen capacity excludes this T/precision/pool working point'
 if cfg['precision']=='P2' and cfg['comp_mode']=='kext':return 'K extension only P1'
 if cfg['precision']=='P2' and cfg['comp_mode'] in ('separate','offload') and cfg['factor_b']!='mxint4':return 'P2 short-K path requires INT4 B'
 return ''

def point(w,d,op,port='iso',variant='default',changes=None,suite='main'):
 cfg=config(d,op,port,changes)
 if cfg['arch']=='joint_v1':
  w=copy.deepcopy(w);w.pop('engine_layout',None)
  os.environ.setdefault('PLENA_DISPATCH_COMPILER','/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-compiler/research/moe_dispatch')
  if str(RESEARCH) not in sys.path:sys.path.insert(0,str(RESEARCH))
  from frontend import compiler
  assert Path(compiler.__file__).resolve()==planner_artifacts()['compiler.py'],'Cached frontend planner differs from the frozen selected Compiler source'
  global _LEGACY_RESOURCE_CACHE
  if _LEGACY_RESOURCE_CACHE is None:_LEGACY_RESOURCE_CACHE=json.loads(Path('/scratch/shared/mcl123/plena/outputs/moe_robust_fixed_20260930/frozen_designs.json').read_text())['designs']
  design=next(x for x in _LEGACY_RESOURCE_CACHE if x['lanes']==cfg['lanes']);resources=copy.deepcopy(design['resources'])
  resources.update(joint_state_bytes=256,control_bytes=[4352//len(cfg['lanes'])]*len(cfg['lanes']))
  cache_key=(w['id'],tuple(cfg['lanes']))
  # The cache now retains the complete prepared workload, including any finite
  # outer token-chunk plan; storing only engine_layout would discard that plan.
  if cache_key not in _LEGACY_LAYOUT_CACHE:_LEGACY_LAYOUT_CACHE[cache_key]=prepare_legacy_workload(w,cfg['lanes'],design['group'],resources)
  w=_LEGACY_LAYOUT_CACHE[cache_key]
  cfg['group']=design['group']
 key='__'.join((suite,w['id'],d,op,port,variant))
 return dict(key=key,suite=suite,workload=w,design=d,op=op,port=port,variant=variant,config=cfg,
             unavailable_reason=legal(w,cfg,d,op))

def points(root,split,suite):
 ws=json.loads((Path(root)/'inputs'/f'{split}.json').read_text())['workloads'];out=[]
 if suite in ('main','all'):
  out += [point(w,d,op,port) for w in ws for d in DESIGNS for op in OPS for port in ('iso','demand')]
 if suite in ('dev_comparator',):
  out += [point(w,d,'OP2','iso',suite=suite) for w in ws for d in ('BL2','BL3','BL5','M8_single','M8_homogeneous','M8_flex62','M8_flex53')]
 if suite in ('policies','all'):
  out += [point(w,'BL4',op,variant=p,changes=dict(placement=p),suite='policies') for w in ws for op in ('OP1','OP2') for p in ('fifo','earliest_finish','joint','supply_ipd')]
 if suite in ('compensation','all'):
  for w in ws:
   for op in ('OP5','OP2'):
    for comp in ('none','lanes','separate','kext','offload'):
     changes=dict(comp_mode=comp,comp_equal_bytes=True)
     # Preserve the full quantized-B supplement as a separate comparison.
     if op=='OP2':changes['factor_b']='mxint4'
     out.append(point(w,'BL4',op,variant=('b4_'+comp if op=='OP2' else comp),changes=changes,suite='compensation'))
   # The primary P2 pair keeps the actual frozen default factor-B format.
   for comp in ('none','lanes'):
    out.append(point(w,'BL4','OP2',variant=comp,changes=dict(comp_mode=comp,comp_equal_bytes=True),suite='compensation'))
 if suite in ('ablation','all'):
  for w in ws:
   for d in ('BL2','BL3','BL4'):
    for op in ('OP0','OP1','OP2'):
     for prefix in range(len(SWITCHES)+1):
      changes={s:('ingress' if prefix>0 else 'landing') if s=='credit_release' else i<prefix for i,s in enumerate(SWITCHES)}
      # Credit count remains working-point-specific; release and depth separate.
      out.append(point(w,d,op,variant=f'forward{prefix}',changes=changes,suite='ablation'))
     for s in SWITCHES:out.append(point(w,d,op,variant='leave_out_'+s,changes={s:'landing' if s=='credit_release' else False},suite='ablation'))
 if suite in ('sensitivity','all'):
  for w in ws:
   for d in ('BL2','BL3','BL4','BL5'):
    for bw,lat in ((128,64),(512,64),(128,150),(256,150)):
     out.append(point(w,d,'OP2',variant=f'bw{bw}_lat{lat}',changes=dict(hbm_bytes_per_ns=bw,hbm_latency_ns=lat,credits=math.ceil(bw*(lat+4)/32),pool_bytes=131072 if bw==512 else 65536),suite='sensitivity'))
 if suite in ('m0_baseline','all'):
  for w in ws:
   for d in ('BL0','BL1'):
    for variant,change in [('normal',{}),('byte_only',dict(weight_bytes_scale=1/3.481,diagnostic_credit_expansion=True))]:
     out.append(point(w,d,'OP0',variant=variant,changes=change,suite='m0_baseline'))
 if suite in ('m0','all'):
  for w in ws:
   for org,lanes in (('6',[6]),('33',[3,3]),('42',[4,2])):
    for variant,change in [('normal',{}),('ideal_hbm',dict(ideal_hbm=True)),('ideal_onchip',dict(ideal_onchip=True)),('both',dict(ideal_hbm=True,ideal_onchip=True)),('byte_only',dict(weight_bytes_scale=1/3.481)),('credits544',dict(credits=544))]:
     c=dict(lanes=lanes,**change)
     if variant in ('byte_only','credits544'):c.update(diagnostic_credit_expansion=True)
     out.append(point(w,'BL0','OP0',variant=f'{org}_{variant}',changes=c,suite='m0'))
 # The prereg matrix includes unsupported points as explicit exclusion records.
 return out

def validate(report,cfg):
 if not isinstance(report,dict):raise AssertionError('Non-object report')
 if report.get('status') in ('unavailable','unsupported','invalid_configuration'):return False
 inv=report.get('invariants',{})
 if isinstance(inv,dict):
  failed={k:v for k,v in inv.items() if v is False}
  assert not failed,f'Invariant failure: {failed}'
 if cfg.get('arch')=='joint_v1':
  cycles=report['cycles'];assert type(cycles) is int and cycles>0
  assert report.get('drained') is True,'Legacy requests/contexts did not drain'
  assert report.get('ownership_k_order_capacity_checks') is True,'Legacy ownership/K/capacity checks missing'
  accepted=report['dma_transactions_accepted'];landed=report['dma_transactions_landed']
  assert accepted==landed,'Legacy accepted DMA did not all land'
  if cfg.get('runtime_fsm',True):assert accepted*32==report['weight_bytes'],'Legacy DMA wire bytes do not match accepted 32B requests'
  scale=cfg.get('weight_bytes_scale',1.0);effective_credits=cfg['credits']
  if scale<1:
   assert 0<scale and (cfg.get('diagnostic_credit_expansion') or cfg.get('diagnostic_profile')),'Compression credit expansion requires explicit diagnostic scope'
   effective_credits=math.ceil(cfg['credits']/scale)
   assert report['config']['credits']==effective_credits and report['config']['diagnostic_credit_expansion'] is True
  assert report['credit_peak']<=effective_credits,'Legacy credit peak exceeds installed diagnostic/physical credits'
  cores=report['cores'];assert len(cores)==len(cfg['lanes'])
  hardware=report['physical_budget']
  assert sum(c['stats']['weight_bytes'] for c in cores)==report['weight_bytes']
  assert sum(c['stats']['useful_macs'] for c in cores)==report['useful_macs']
  for i,core in enumerate(cores):
   stats=core['stats'];assert core['m']==cfg['lanes'][i]
   assert core['capacity']==hardware['acc_bytes'][i]
   assert 0<=core['reserved_input_result_control']<=core['capacity']
   # The original observer only updates workspace_peak_bytes on a binding;
   # an inactive core can still own a fully resident persistent arena.
   assert 0<=stats['workspace_peak_bytes'] and max(core['reserved_input_result_control'],stats['workspace_peak_bytes'])<=core['capacity'],'Legacy private result/workspace peak exceeds capacity'
   assert stats['weight_peak_bytes']<=hardware['weight_slots'][i]*4096
   assert stats['x_peak_bytes']<=cfg['lanes'][i]*512*2*2
   assert stats['dma_accepted']==stats['dma_landed']
  profile=report.get('m0_profile')
  if cfg.get('diagnostic_profile'):assert isinstance(profile,dict),'Requested legacy profile missing'
  if profile is not None:
   assert profile.get('mutually_exclusive') is True
   assert sum(profile['hbm_states'].values())==cycles==profile['hbm_sum']
   assert len(profile['core_states'])==len(cores)
   assert profile['core_sums']==[cycles]*len(cores)
   assert all(sum(states.values())==cycles for states in profile['core_states'])
  wrapper=report.get('legacy_chunks')
  if wrapper is not None:
   assert wrapper.get('schema')=='plena_legacy_token_chunks_v1'
   assert wrapper.get('global_routes_and_scores_preserved') is True
   assert wrapper.get('persistent_addresses_checked') is True
   assert type(wrapper['chunk_size']) is int and wrapper['chunk_size']>0
   parts=wrapper['parts'];assert wrapper['chunks']==len(parts)>1
   assert report['unique_weight_bytes']>0
   assert report['weight_bytes']==report['unique_weight_bytes']+report['refetch_bytes']
   assert report['refetch_bytes']==wrapper['refetch_bytes']>0,'Chunk refetch traffic must be real and reported'
   persistent=wrapper['full_layer_persistent_cores'];assert len(persistent)==len(cores)
   full_batch=persistent[0]['original_x']['shape'][0]
   assert type(full_batch) is int and full_batch>0
   for i,arena in enumerate(persistent):
    assert arena['capacity']==cores[i]['capacity'] and arena['reserved']<=arena['capacity']
    assert arena['original_x']['shape'][0]==arena['combined_output']['shape'][0]==full_batch
   next_row=elapsed=setup_cycles=setup_bytes=total_weight=total_macs=0
   for part in parts:
    start,end=part['token_range'];assert start==next_row and 0<end-start<=wrapper['chunk_size'];next_row=end
    assert part['start_cycle']==elapsed
    assert part['kernel_start_cycle']==elapsed+part['setup_cycles']
    child=part['kernel_report'];assert 'legacy_chunks' not in child
    assert validate(child,cfg),'Legacy child cannot be unsupported'
    elapsed=part['kernel_start_cycle']+child['cycles']
    setup_cycles+=part['setup_cycles'];setup_bytes+=part['setup_sram_bytes']
    total_weight+=child['weight_bytes'];total_macs+=child['useful_macs']
   assert next_row==full_batch,'Legacy chunks must cover the full original token window'
   assert elapsed==cycles and setup_cycles==wrapper['setup_cycles']
   assert setup_bytes==wrapper['setup_sram_bytes'] and total_weight==report['weight_bytes'] and total_macs==report['useful_macs']
 if cfg.get('arch')=='supply_v3':
  cycles=report['cycles'];assert isinstance(cycles,int) and cycles>0
  required=('unique_task_owner','requests_drained','pool_references_released','unique_weight_bytes_exact','main_useful_macs_exact','stall_states_exclusive','rank_capacity_checked_by_plan','storage_fits')
  assert all(inv.get(k) is True for k in required),f'Missing or invalid invariant: {inv}'
  assert report.get('drained') is True
  h=report.get('hbm_states',report.get('h_states',{}));assert h,'Missing exclusive HBM states'
  assert sum(h.values())==cycles,(h,cycles)
  cores=report.get('cores',[]);assert len(cores)==len(cfg['lanes']),'Wrong core count'
  for core in cores:
   states=core.get('states',core.get('c_states',{}));assert states,'Missing exclusive core states'
   assert sum(states.values())==cycles,(states,cycles)
  assert report['dma_transactions_accepted']==report['dma_transactions_landed']
  assert report['actual_storage_peak_bytes']<=2158592
  budget=report.get('budget',{});assert budget.get('fits') is True
  assert budget['total_bytes']<=2158592 and budget['capacity_bytes']==2158592
  assert budget['t_chunk_capacity']==cfg['t_chunk']
  assert report['credit_peak']<=cfg['credits']
  assert report['pool_peak_bytes']<=cfg['pool_bytes']
  assert report['ingress_peak_bytes']<=cfg['ingress_bytes']
  if 'native_expert_bytes' in report:
   assert report['weight_bytes']==report['native_expert_bytes']+report.get('comp_padding_bytes',0)+report.get('work_steal_waste_bytes',0),'Wire byte categories do not sum'
  if 'routed_rank_factor_bytes' in report:
   assert report['rank_factor_bytes']==report['routed_rank_factor_bytes']+report['shared_rank_factor_bytes'],'Rank factor byte categories do not sum'
  movement=report.get('onchip_movement_breakdown_bytes',{})
  assert movement and all(type(v) is int and v>=0 for v in movement.values()),'Missing or invalid physical movement breakdown'
  assert sum(movement.values())==report['onchip_traffic_bytes']==report['all_onchip_traffic_bytes'],'Physical endpoint traffic categories do not sum'
 return True

def rawbytes(dest,i):
 p=dest/f'repeat{i}.json'
 return p.read_bytes() if p.exists() else gzip.decompress((dest/f'repeat{i}.json.gz').read_bytes())

def store_json(root,path,value):
 target=Path(root)/'blobs'/(digest(value)+'.json')
 frozen(target,value)
 if not path.exists():path.symlink_to(os.path.relpath(target,path.parent))

def run_point(p,root,binary,signature,timeout):
 key=p['key'];dest=Path(root)/'raw'/digest(signature)[:16]/key;dest.mkdir(parents=True,exist_ok=True)
 manifest=dict(point_key=key,design=p['design'],op=p['op'],port=p['port'],variant=p['variant'],suite=p['suite'],config_sha256=digest(p['config']),workload_sha256=digest(p['workload']),signature=signature)
 frozen(dest/'point.json',manifest)
 if p['unavailable_reason']:
  write(dest/'unavailable.json',dict(reason=p['unavailable_reason'],kind='predeclared_configuration_exclusion'));return dict(point=p,status='excluded',reason=p['unavailable_reason'])
 store_json(root,dest/'config.json',p['config']);store_json(root,dest/'workload.json',p['workload'])
 receipt=dest/'receipt.json'
 if receipt.exists():
  r=json.loads(receipt.read_text());assert r['signature']==signature
  assert hashlib.sha256(rawbytes(dest,1)).hexdigest()==r['sha256'] and rawbytes(dest,1)==rawbytes(dest,2)
  report=json.loads(rawbytes(dest,1));assert validate(report,p['config'])
  return dict(point=p,status='complete',report=report,receipt=r)
 host_seconds=[]
 for i in (1,2):
  command=[str(binary),str(dest/'workload.json'),str(dest/'config.json'),str(dest/f'repeat{i}.json')]
  start=time.monotonic()
  try:proc=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=timeout)
  except subprocess.TimeoutExpired as err:
   write(dest/'failure.json',dict(kind='host_timeout',command=command,timeout_seconds=timeout,message=str(err)))
   return dict(point=p,status='failed',reason=f'Host timeout after {timeout} seconds')
  host_seconds.append(time.monotonic()-start)
  if proc.stdout:(dest/f'repeat{i}.stdout').write_bytes(proc.stdout)
  if proc.stderr:(dest/f'repeat{i}.stderr').write_bytes(proc.stderr)
  if proc.returncode:
   write(dest/'failure.json',dict(command=command,returncode=proc.returncode,stderr=proc.stderr.decode(errors='replace')[-6000:]));return dict(point=p,status='failed',reason=proc.stderr.decode(errors='replace')[-1000:])
  try:
   valid=validate(json.loads((dest/f'repeat{i}.json').read_text()),p['config'])
  except (AssertionError,KeyError,TypeError,json.JSONDecodeError) as err:
   write(dest/'failure.json',dict(kind='invariant_or_schema_failure',message=str(err)))
   return dict(point=p,status='failed',reason=str(err))
  if not valid:
   write(dest/'unavailable.json',dict(kind='engine_unsupported',report=json.loads((dest/f'repeat{i}.json').read_text())))
   return dict(point=p,status='unsupported',reason='Engine explicitly marked configuration unavailable')
 first=rawbytes(dest,1);second=rawbytes(dest,2)
 if first!=second:
  write(dest/'failure.json',dict(kind='nondeterministic_raw_json',repeat1_sha256=hashlib.sha256(first).hexdigest(),repeat2_sha256=hashlib.sha256(second).hexdigest()))
  return dict(point=p,status='failed',reason='Duplicate original JSON bytes differ')
 r=dict(signature=signature,sha256=hashlib.sha256(first).hexdigest(),repeats=2,bit_identical=True,host_seconds_repeats=host_seconds,storage='two gzip members, original JSON hashes verified after decompression')
 for i,data in ((1,first),(2,second)):
  compressed=gzip.compress(data,mtime=0);(dest/f'repeat{i}.json.gz').write_bytes(compressed)
  assert gzip.decompress(compressed)==data
  (dest/f'repeat{i}.json').unlink()
 write(receipt,r);return dict(point=p,status='complete',report=json.loads(first),receipt=r)

def local_operand_port_ledger(cfg,budget):
 """Full physical-row register delivery demand, not a synthesized RF port."""
 if cfg.get('arch')!='supply_v3':return {}
 slot=budget.get('wor_slot_bytes')
 if slot is None:
  main=(512*4*cfg['main_bits']+7)//8+64
  slot=((main+cfg['rank_lanes']*4*2+31)//32)*32 if cfg['precision']=='P2' else 4096
 extra_b=cfg['rank_lanes']*4*2 if cfg['precision']=='P1' else 0
 weight=[slot+extra_b]*len(cfg['lanes']);x=[m*512*2 for m in cfg['lanes']]
 return dict(local_WOR_broadcast_read_bytes_per_cycle=weight,local_XOR_broadcast_read_bytes_per_cycle=x,
  local_operand_broadcast_total_bytes_per_cycle=sum(weight)+sum(x),
  WOR_fill_source='finite pool read / P1 decoder already in canonical port ledger',
  XOR_fill_source='finite activation/X port already in canonical port ledger',
  definition='installed full physical-row worst-case issue delivery, including broadcast; not SRAM bank port or RTL timing proof',
  canonical_proxy_excludes_local_RF_broadcast_ports=True)

def area_ledger(cfg,budget):
 """Prespecified relative proxy; never a physical area estimate."""
 if cfg.get('arch')!='supply_v3':return {}
 structures=budget.get('structures',{});other_rf=structures.get('WOR',0)+structures.get('XOR',0)+budget.get('combine_lease_register_bytes',0)+budget.get('WOR_rankB_reserved_in_control',0)
 accumulator=structures.get('accumulator',0)
 capacities=budget.get('accumulator_capacity_bytes_per_core')
 if capacities is not None:
  assert len(capacities)==len(cfg['dataflow']) and sum(capacities)==accumulator
  ws_rf=sum(cap for cap,flow in zip(capacities,cfg['dataflow']) if flow!='is_stream')
  hybrid_rf=sum(min(cap,2*cfg['t_chunk']*32*4) for cap,flow in zip(capacities,cfg['dataflow']) if flow!='is_stream')
 else:
  # Archived provisional reports predate the per-core physical-type ledger.
  # This fallback is explicitly not eligible for a robust area-based verdict.
  ws_rf=2*cfg['t_chunk']*32*4 if 'is_stream' in cfg['dataflow'] else accumulator
  hybrid_rf=min(ws_rf,len([f for f in cfg['dataflow'] if f!='is_stream'])*2*cfg['t_chunk']*32*4)
 rf=other_rf+ws_rf
 # The unused storage budget is still implemented SRAM, not free area.
 sram=2158592-rf;total_main=sum(cfg['lanes'])*4*512
 mx=total_main if cfg['precision']=='P2' else 0;bf=total_main-mx
 rank=sum(cfg['lanes'])*4*cfg['rank_lanes'];active_rank=rank if cfg['comp_mode']=='lanes' else 0
 pool_read=sum(cfg['pool_read_per_core']);x_read=sum(cfg['x_port'])
 # Logical R+W widths count both directions of the same physical 1RW port;
 # they do not imply an independent uncharged vector source-read port.
 u_read=sum(budget.get('rank_cache_local_read_bytes_per_cycle',[]));u_write=sum(budget.get('rank_cache_local_write_bytes_per_cycle',[]))
 acc_widths=budget.get('accumulator_port_bytes_per_cycle')
 acc_physical=sum(acc_widths) if acc_widths is not None else sum(cfg['lanes'])*4*4
 ports=pool_read+cfg['hbm_bytes_per_ns']+x_read+2*acc_physical+512+u_read+u_write
 raw=dict(main_mx_multipliers=mx,main_bf16_multipliers=bf,rank_bf16_multipliers=rank,
  rank_bf16_multipliers_active=active_rank,sram_bytes=sram,rf_bytes=rf,read_write_port_bytes_per_cycle=ports,accumulator_physical_1rw_bytes_per_cycle=acc_physical,accumulator_logical_read_write_bytes_per_cycle=2*acc_physical,accumulator_ports_explicitly_reported=acc_widths is not None,rank_cache_local_read_bytes_per_cycle=u_read,rank_cache_local_write_bytes_per_cycle=u_write,physical_storage_ceiling_bytes=2158592)
 def proxy(rf_bytes):return mx/12288+2*bf/12288+2*rank/12288+(2158592-rf_bytes)/2097152+4*rf_bytes/2097152+.1*ports/2048
 raw.update(nonaccumulator_rf_bytes=other_rf,accumulator_rf_bytes=ws_rf,accumulator_sram_bytes=accumulator-ws_rf,accumulator_hybrid_rf_bytes=hybrid_rf,accumulator_types_explicitly_reported=capacities is not None,accumulator_area_type_model='one-cycle WS full arena RF interpretation; specialized IS SRAM; not RTL mapping',proxy_default=proxy(rf),proxy_accumulator_hybrid=proxy(other_rf+hybrid_rf),proxy_accumulator_all_sram=proxy(other_rf),proxy_accumulator_all_rf=proxy(other_rf+accumulator))
 local=local_operand_port_ledger(cfg,budget)
 raw.update(local_operand_broadcast_bytes_per_cycle=local['local_operand_broadcast_total_bytes_per_cycle'],canonical_port_proxy_scope='pool/HBM/X/accumulator/Combine/U-cache service widths; local WOR/XOR broadcast excluded and separately cost-sensitized',proxy_with_local_operand_broadcast_ports=proxy(rf)+.1*local['local_operand_broadcast_total_bytes_per_cycle']/2048)
 return raw

def histogram_summary(histogram):
 hist=sorted((int(ns),int(count)) for ns,count in histogram.items() if count)
 if not hist:return {}
 count=sum(n for ns,n in hist);result=dict(samples=count,mean_cycles=sum(ns*n for ns,n in hist)/count,max_cycles=hist[-1][0])
 for q in (.50,.95,.99):
  threshold=math.ceil(count*q);cumulative=0
  for ns,n in hist:
   cumulative+=n
   if cumulative>=threshold:result[f'p{int(q*100)}_cycles']=ns;break
 return result

def flatten(item):
 p=item['point'];r=item.get('report',{});w=p['workload'];provenance=w.get('provenance','captured_decode');pm=provenance if isinstance(provenance,dict) else {};origin=pm.get('origin',provenance);row=dict(point=p['key'],raw_binary=item.get('receipt',{}).get('signature',{}).get('binary_sha256',''),suite=p['suite'],split=w.get('split',pm.get('split','development')),provenance=origin,workload=w['id'],dataset=w.get('dataset',pm.get('dataset','')),tokens=w['batch'],design=p['design'],op=p['op'],port=p['port'],variant=p['variant'],status=item['status'],reason=item.get('reason',''),precision_quality='not_accuracy_validated_candidate' if p['op']=='OP4' else 'default_format_provisional_until_Q1',rows='+'.join(map(str,p['config']['lanes'])))
 if item.get('receipt',{}).get('signature'):row['raw_campaign']=digest(item['receipt']['signature'])
 if item.get('receipt',{}).get('signature',{}).get('physical_format_sha256'):row['physical_format_sha256']=item['receipt']['signature']['physical_format_sha256']
 if item.get('receipt',{}).get('host_seconds_repeats'):row['host_seconds_two_repeats']=sum(item['receipt']['host_seconds_repeats'])
 row.update(main_bits=p['config']['main_bits'],factor_a=p['config']['factor_a'],factor_b=p['config']['factor_b'],rank_lanes=p['config']['rank_lanes'],comp_mode=p['config']['comp_mode'],precision=p['config']['precision'],clock_ghz=1,hbm_bytes_per_cycle=p['config']['hbm_bytes_per_ns'],hbm_latency_cycles=p['config']['hbm_latency_ns'],credits_installed=p['config']['credits'],physical_t_chunk=p['config'].get('t_chunk',''),dataflow='+'.join(p['config']['dataflow']),main_multiplier_budget=sum(p['config']['lanes'])*4*512)
 if not row['dataset'] and w['id'].startswith('joint_test_'):row['dataset']=w['id'].split('_')[2]
 for key in ('cycles','time_ms_at_1ghz','weight_bytes','unique_weight_bytes','refetch_bytes','useful_macs','issued_macs','supply_efficiency','onchip_bytes','padding_macs','bank_conflict_cycles','control_cycles'):
  if key in r:row[key]=r[key]
 if p['config'].get('arch')=='joint_v1':
  wrapper=r.get('legacy_chunks',{})
  row.update(legacy_batch_chunked=bool(wrapper),legacy_batch_chunk_size=wrapper.get('chunk_size',w['batch']),
   legacy_batch_chunk_count=wrapper.get('chunks',1),legacy_batch_refetch_bytes=r.get('refetch_bytes',0),
   legacy_batch_unique_weight_bytes=r.get('unique_weight_bytes',r.get('weight_bytes',0)),
   legacy_batch_setup_cycles=wrapper.get('setup_cycles',0),legacy_batch_setup_sram_bytes=wrapper.get('setup_sram_bytes',0),
   legacy_batch_reset_policy=wrapper.get('reset_policy','unchanged feasible whole-window legacy kernel'))
 if 'onchip_traffic_bytes' in r:row['onchip_bytes']=r['onchip_traffic_bytes']
 for key in ('onchip_bytes_per_useful_mac','all_onchip_traffic_bytes','legacy_onchip_subset_bytes','onchip_traffic_definition','hbm_bytes_per_token','cross_core_bytes','pool_read_bytes','pool_bank_conflicts','activation_bank_conflicts','prediction_mean_absolute_error_cycles','prediction_worst_underestimate_cycles','dma_transactions_accepted','pool_peak_bytes','ingress_peak_bytes','credit_peak','comp_padding_bytes','work_steal_waste_bytes','work_steal_attempts','work_steal_successes','actual_storage_peak_bytes','baseline_bf16_weight_bytes','offload_helper_cycles','rank_factor_bytes','weight_retake_bytes','native_expert_bytes','pool_write_bytes','combine_bank_conflicts','quota_updates','quota_denials','pool_capacity_denials','latency_ewma_cycles','issued_main_macs','issued_aux_macs','useful_aux_macs','padding_main_macs','padding_aux_macs'):
  if key in r:row[key]=r[key]
 for key,value in r.get('onchip_movement_breakdown_bytes',{}).items():row['movement_'+key]=value
 bindings=r.get('bindings',[])
 if bindings:
  dist={}
  for binding in bindings:
   legal_count=binding.get('legal_core_count')
   if legal_count is not None:dist[legal_count]=dist.get(legal_count,0)+1
  for key,value in dist.items():row[f'bindings_legal_cores_{key}']=value
  errors=[b['actual_completion_cycle']-b['predicted_completion_cycle'] for b in bindings if 'actual_completion_cycle' in b and 'predicted_completion_cycle' in b]
  if errors:
   row.update(binding_completion_mean_absolute_error_cycles=sum(map(abs,errors))/len(errors),binding_completion_worst_underestimate_cycles=max(0,max(errors)))
  cold=[b for b in bindings if b.get('expert_id',-1)>=0 and b.get('Me',0)<=2 and 'actual_completion_cycle' in b]
  if cold:
   span=max(b['actual_completion_cycle'] for b in cold)-min(b['cycle'] for b in cold)
   row.update(cold_descriptors=len(cold),cold_token_rows=sum(b['Me'] for b in cold),cold_completion_span_cycles=span,cold_token_rows_per_cycle=sum(b['Me'] for b in cold)/max(span,1))
 row['shared_me']=max((e['Me'] for e in w['experts'] if e['is_shared']),default=0)
 for key in ('routed_rank_factor_bytes','shared_rank_factor_bytes','rank_budget_scope'):
  if key in r:row[key]=r[key]
 for key in ('pool_allocation_mode','pool_reservation_granularity_bytes','quota_progress_borrows'):
  if key in r:row[key]=r[key]
 for i,value in enumerate(r.get('pool_partition_bytes_per_core',[])):row[f'pool_partition_bytes_core{i}']=value
 if FORMAT_RECEIPT:row.update(frozen_format_sha256=FORMAT_RECEIPT['sha256'],precision_quality=FORMAT_RECEIPT['quality_status'] if p['op']!='OP4' else 'W3_requires_separate_Q1_qualification')
 if p['config']['precision']=='P0':row['precision_quality']='BF16_format_baseline'
 elif p['suite']=='compensation' and p['variant'].startswith('b4_'):row['precision_quality']='P2_compensation_B_MXINT4_supplement_requires_independent_quality_validation'
 if isinstance(r.get('compression_oracle'),dict):
  row.update(compression_oracle=True,scaled_wire_bytes=r['compression_oracle']['scaled_wire_bytes'],logical_weight_bytes=r['compression_oracle']['logical_weight_bytes'])
 if isinstance(r.get('budget'),dict):
  for section in ('structures','ports','multipliers'):
   for key,value in r['budget'].get(section,{}).items():
    if isinstance(value,(int,float,bool)):row[f'budget_{section}_{key}']=value
  ledger=area_ledger(p['config'],r['budget'])
  row.update({f'area_{k}':v for k,v in ledger.items()})
  local=local_operand_port_ledger(p['config'],r['budget'])
  for key,value in local.items():
   if isinstance(value,list):
    for i,width in enumerate(value):row[f'local_{key}_core{i}']=width
   else:row[f'local_{key}']=value
  for key in ('accumulator_port_bytes_per_cycle','accumulator_capacity_bytes_per_core','rank_cache_local_read_bytes_per_cycle','rank_cache_local_write_bytes_per_cycle','rank_cache_bank_counts'):
   for i,value in enumerate(r['budget'].get(key,[])):row[f'budget_{key}_core{i}']=value
 if 'cycles' in r:
  row.update(ms=r['cycles']/1e6,us=r['cycles']/1e3,weight_gbps=r.get('weight_bytes',0)/r['cycles'])
 for key in ('traffic','budget','hbm_states','h_states','prediction'):
  if isinstance(r.get(key),dict):
   for k,v in r[key].items():
    if isinstance(v,(int,float,bool,str)):row[key+'_'+k]=v
 cores=r.get('cores',[])
 if cores and p['config'].get('arch')=='supply_v3':
  main_issued=sum(c.get('main_issues',0)*c['m']*4*512 for c in cores)
  row.update(issued_main_macs=main_issued,issued_aux_macs=r.get('issued_macs',0)-main_issued)
  if p['config']['comp_mode']!='kext' or 'padding_main_macs' in r:
   padding=r.get('padding_main_macs',main_issued-r.get('useful_macs',0));row.update(padding_main_macs=padding,main_padding_fraction=padding/max(main_issued,1))
  row['rank_lane_utilization']=sum(c.get('rank_macs',0) for c in cores)/max(sum(c.get('rank_capacity_macs',0) for c in cores),1) if p['config']['comp_mode']=='lanes' else 0
 for i,c in enumerate(cores):
  for key,v in c.items():
   if isinstance(v,(int,float,bool)):row[f'core{i}_{key}']=v
  for key in ('states','c_states','stats','traffic'):
   if isinstance(c.get(key),dict):
    for k,v in c[key].items():
     if isinstance(v,(int,float,bool)):row[f'core{i}_{key}_{k}']=v
  for field,label in (('group_wall_elapsed_histogram','group_wall_service'),('issue_to_accumulator_completion_histogram','issue_to_acc_commit'),('pool_to_wor_elapsed_histogram','pool_to_wor')):
   hist=c.get(field,c.get('service_histogram',{}) if field=='group_wall_elapsed_histogram' else {})
   for key,value in histogram_summary(hist).items():row[f'core{i}_{label}_{key}']=value
 return row

def execute(args):
 root=Path(args.root);binary=Path(args.binary).resolve();assert binary.is_file()
 signature=campaign_signature(root,binary)
 if args.split not in ('development','mixed_development'):
  verify_heldout_authorization(root,signature)
 snapshot=root/'snapshots'/digest(signature)[:16];snapshot.mkdir(parents=True,exist_ok=True)
 frozen_binary=snapshot/'moe-dispatch-analytical-v1'
 if not frozen_binary.exists():shutil.copy2(binary,frozen_binary)
 assert sha(frozen_binary)==signature['binary_sha256']
 shutil.copy2(__file__,snapshot/'study.py')
 source_dir=snapshot/'rust_src'
 if not source_dir.exists():shutil.copytree(RESEARCH/'rust/src',source_dir)
 write(snapshot/'manifest.json',signature)
 planner_dir=snapshot/'planner_sources';planner_dir.mkdir(exist_ok=True)
 for name,path in planner_artifacts().items():
  assert sha(path)==signature['planner_artifacts_sha256'][name],'Planner changed during campaign preparation'
  target=planner_dir/name
  if not target.exists():shutil.copy2(path,target)
  assert sha(target)==signature['planner_artifacts_sha256'][name]
 frozen(planner_dir/'source_paths.json',{name:str(path) for name,path in planner_artifacts().items()})
 frozen(snapshot/'physical_format.json',physical_format())
 if FORMAT_RECEIPT:
  qualification_dir=snapshot/'qualification_receipts';qualification_dir.mkdir(exist_ok=True)
  target=qualification_dir/(FORMAT_RECEIPT['sha256']+'.json')
  if not target.exists():shutil.copy2(FORMAT_RECEIPT['path'],target)
  assert sha(target)==FORMAT_RECEIPT['sha256']
  write(snapshot/'latest_qualification_receipt.json',FORMAT_RECEIPT)
 write(root/'active_result_signature.json',signature)
 binary=frozen_binary
 ps=points(root,args.split,args.suite)
 if args.filter:ps=[p for p in ps if args.filter in p['key']]
 if args.filter_regex:ps=[p for p in ps if re.search(args.filter_regex,p['key'])]
 if args.limit:ps=ps[:args.limit]
 assert ps
 selector=args.filter+'|'+args.filter_regex
 stage=f'{args.split}_{args.suite}'+('_'+hashlib.sha256(selector.encode()).hexdigest()[:8] if args.filter or args.filter_regex else '')
 write(root/f'{stage}_matrix.json',dict(count=len(ps),keys=[p['key'] for p in ps],repeats=2,signature=signature))
 rows=[];started=time.monotonic();last_checkpoint=started
 checkpoint_points=32 if len(ps)<=500 else 1000
 with ThreadPoolExecutor(max_workers=args.workers) as executor:
  jobs=[executor.submit(run_point,p,root,binary,signature,args.timeout) for p in ps]
  for fut in as_completed(jobs):
   item=fut.result();row=flatten(item);row['evaluation_split']=args.split;rows.append(row)
   now=time.monotonic()
   if len(rows)%checkpoint_points==0 or now-last_checkpoint>=120 or len(rows)==len(ps):
    csvwrite(root/f'{stage}.csv',sorted(rows,key=lambda r:r['point']))
    last_checkpoint=now
   if len(rows)%32==0 or len(rows)==len(ps):
    print(f'{stage} {len(rows)}/{len(ps)} host_s={time.monotonic()-started:.1f} complete={sum(r["status"]=="complete" for r in rows)}',flush=True)
 write(root/f'{stage}_receipt.json',dict(points=len(rows),complete=sum(r['status']=='complete' for r in rows),excluded=sum(r['status']=='excluded' for r in rows),unsupported=sum(r['status']=='unsupported' for r in rows),failed=sum(r['status']=='failed' for r in rows),repeats_equal=True,host_elapsed_seconds=time.monotonic()-started,workers=args.workers,signature=signature))

def comparator(root):
 root=Path(root);path=root/'development_dev_comparator.csv';rows=list(csv.DictReader(path.open()));by={}
 receipt=json.loads((root/'development_dev_comparator_receipt.json').read_text())
 assert receipt['points']==84 and receipt['complete']==84 and not any(receipt[k] for k in ('failed','unsupported','excluded')),'Comparator requires all 84 planned points, each valid and duplicated'
 signature=receipt['signature'];assert signature==json.loads((root/'active_result_signature.json').read_text()),'Comparator CSV belongs to an inactive campaign'
 expected=points(root,'development','dev_comparator')
 assert {r['point'] for r in rows}=={p['key'] for p in expected} and len(rows)==84
 assert all(r['status']=='complete' and r['raw_binary']==signature['binary_sha256'] for r in rows)
 original_receipts=[]
 for p in expected:
  dest=root/'raw'/digest(signature)[:16]/p['key'];raw_receipt=json.loads((dest/'receipt.json').read_text())
  assert raw_receipt['signature']==signature and raw_receipt['bit_identical'] and raw_receipt['repeats']==2
  original_receipts.append(dict(point=p['key'],sha256=raw_receipt['sha256']))
 for row in rows:
  if row['status']=='complete':by.setdefault(row['design'],[]).append(float(row['cycles']))
 groups={'M6':['BL2','BL3','BL5'],'M8':['M8_single','M8_homogeneous','M8_flex62','M8_flex53']}
 assert set(by)==set(sum(groups.values(),[])) and all(len(v)==12 for v in by.values())
 times={d:math.exp(sum(map(math.log,v))/len(v)) for d,v in by.items()}
 selected={g:min(ds,key=lambda d:times[d]) for g,ds in groups.items()}
 out=dict(selected=selected['M6'],selected_by_budget=selected,working_point='OP2',port='iso',development_windows=12,geometric_mean_cycles=times,signature=signature,criterion='minimum development geometric mean within each MAC budget, fixed before heldout; no pointwise alternative selection')
 write(root/'development_fixed_comparator.json',out)
 ready=dict(status='development_comparator_ready_for_root_prereg_commit',heldout_authorized=False,signature=signature,points=84,runs=168,all_valid_and_bit_identical=True,selected_by_budget=selected,comparator_sha256=sha(root/'development_fixed_comparator.json'),development_csv_sha256=sha(path),format_status=FORMAT_RECEIPT or dict(quality_status='timing_candidate_not_accuracy_validated'),original_json_receipts=original_receipts)
 write(root/'development_freeze_ready_receipt.json',ready);print(json.dumps(out,indent=2))

def cli():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','matrix','run','comparator']);p.add_argument('--root',default=str(DEFAULT_ROOT));p.add_argument('--split',default='development',choices=['development','heldout','constructed','mixed_development','mixed_heldout']);p.add_argument('--suite',default='main',choices=['main','all','m0','m0_baseline','ablation','policies','compensation','sensitivity','dev_comparator']);p.add_argument('--binary',default=str(RESEARCH/'rust/target/release/moe-dispatch-analytical-v1'));p.add_argument('--workers',type=int,default=12);p.add_argument('--timeout',type=int,default=3600);p.add_argument('--filter',default='');p.add_argument('--filter-regex',default='');p.add_argument('--limit',type=int,default=0);p.add_argument('--format-config',default='');a=p.parse_args()
 format_path=Path(a.format_config) if a.format_config else Path(a.root)/'frozen_format.json'
 if a.format_config or format_path.exists():load_format(format_path)
 if a.action=='prepare':prepare(a.root)
 elif a.action=='run':execute(a)
 elif a.action=='comparator':comparator(a.root)
 else:
  ps=points(a.root,a.split,a.suite);summary=dict(points=len(ps),legal=sum(not p['unavailable_reason'] for p in ps),excluded=sum(bool(p['unavailable_reason']) for p in ps),repeats=2)
  write(Path(a.root)/f'{a.split}_{a.suite}_declared_matrix.json',dict(summary=summary,points=[{k:v for k,v in p.items() if k not in ('workload','config')} for p in ps]));print(json.dumps(summary))
if __name__=='__main__':cli()
