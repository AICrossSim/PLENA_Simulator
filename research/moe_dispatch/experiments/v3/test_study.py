import importlib.util,sys
from pathlib import Path
p=Path(__file__).with_name('study.py');spec=importlib.util.spec_from_file_location('v3study',p);s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)

def test_precision_and_legacy_conditions_are_explicit():
 w={'batch':16}
 assert s.config('BL4','OP2')['precision']=='P2'
 assert s.config('BL4','OP3')['credits']==1232
 assert s.legal(w,s.config('BL0','OP2'),'BL0','OP2')
 assert not s.legal(w,s.config('BL4','OP2'),'BL4','OP2')

def test_capacity_exclusion_cannot_be_silently_skipped():
 assert s.legal({'batch':128},s.config('BL4','OP5'),'BL4','OP5')
 assert not s.legal({'batch':128},s.config('BL4','OP2'),'BL4','OP2')

def test_main_mac_and_equal_port_budgets():
 for d in ('BL2','BL3','BL4','BL5'):
  c=s.config(d,'OP2');assert sum(c['lanes'])*4*512==12288
  assert sum(c['pool_read_per_core'])==1024
  assert sum(c['x_port'])==640

def test_streamed_and_scalar_routed_ownership():
 meta={'hidden_size':2048,'moe_intermediate_size':1408,'shared_expert_intermediate_size':2816,'routed_experts':64,'top_k':1}
 tokens=[{'token_index':0,'sample_id':'a','routes':[{'expert_id':7,'slot':0,'score':.75}]}]
 w=s.from_tokens(tokens,meta,'x','captured_decode')
 assert w['experts'][0]['is_shared'] and w['experts'][0]['F']==2816
 assert w['experts'][1]['id']==7 and w['experts'][1]['route_scores']==[.75]
 assert w['experts'][1]['token_indices']==[0]
 assert w['route_scores_renormalized'] is False

def test_comparable_decoder_and_fixed_hardware_capacity():
 for d in ('BL2','BL3','BL4','BL5'):
  c=s.config(d,'OP5');assert c['dec_bw']*len(c['lanes'])==2048
  assert c['t_chunk']==96 and c['wor_total_slots']==18
  assert s.config(d,'OP2')['t_chunk']==128
  assert s.config(d,'OP2',changes={'hbm_bytes_per_ns':512})['t_chunk']==96

def test_invalid_reports_cannot_be_counted_as_success():
 import pytest
 assert s.validate({'status':'unsupported'}, {}) is False
 with pytest.raises(AssertionError):s.validate({'invariants':{'requests_drained':False}}, {})
 with pytest.raises(AssertionError):s.validate({'cycles':10,'hbm_states':{'H0':9}}, {'arch':'supply_v3'})

def test_compressed_original_json_retains_byte_identity(tmp_path):
 import gzip
 data=b'{"cycles":12,"invariants":{"requests_drained":true}}\n'
 for i in (1,2):(tmp_path/f'repeat{i}.json.gz').write_bytes(gzip.compress(data,mtime=0))
 assert s.rawbytes(tmp_path,1)==data==s.rawbytes(tmp_path,2)


def test_area_proxy_charges_unused_storage_and_rank_modes():
 c=s.config('BL4','OP2');b={'structures':{'WOR':20736,'XOR':16384,'accumulator':77824}}
 a=s.area_ledger(c,b)
 assert a['sram_bytes']+a['rf_bytes']==2158592
 assert a['rank_bf16_multipliers']==192
 c['comp_mode']='none';assert s.area_ledger(c,b)['rank_bf16_multipliers']==192
 assert s.area_ledger(c,b)['rank_bf16_multipliers_active']==0

def test_area_proxy_bills_actual_shared_accumulator_source_port():
 c=s.config('BL4','OP2');b={'structures':{'WOR':20736,'XOR':16384,'accumulator':77824},'accumulator_port_bytes_per_cycle':[128,128]}
 ledger=s.area_ledger(c,b)
 assert ledger['accumulator_physical_1rw_bytes_per_cycle']==256
 assert ledger['accumulator_logical_read_write_bytes_per_cycle']==512
 assert ledger['accumulator_ports_explicitly_reported'] is True

def test_local_operand_ports_preserve_canonical_proxy_and_expose_delivery():
 import math
 c=s.config('BL4','OP2');b={'wor_slot_bytes':1152}
 p=s.local_operand_port_ledger(c,b)
 assert p['local_WOR_broadcast_read_bytes_per_cycle']==[1152,1152]
 assert p['local_XOR_broadcast_read_bytes_per_cycle']==[4096,2048]
 assert p['local_operand_broadcast_total_bytes_per_cycle']==8448
 a=s.area_ledger(c,b)
 assert math.isclose(a['proxy_with_local_operand_broadcast_ports']-a['proxy_default'],.1*8448/2048)
 assert s.local_operand_port_ledger(s.config('BL2','OP5'),{})['local_WOR_broadcast_read_bytes_per_cycle']==[4160]

def test_area_accumulator_bounds_do_not_claim_hybrid_mapping():
 c=s.config('BL5','OP2');b={'structures':{'WOR':20736,'XOR':12288,'accumulator':77824},'accumulator_capacity_bytes_per_core':[45056,32768],'accumulator_port_bytes_per_cycle':[128,128]}
 a=s.area_ledger(c,b)
 assert a['accumulator_rf_bytes']==77824 and a['accumulator_hybrid_rf_bytes']==65536
 assert a['proxy_accumulator_all_sram']<a['proxy_accumulator_hybrid']<a['proxy_default']==a['proxy_accumulator_all_rf']
 assert 'not RTL' in a['accumulator_area_type_model']

def test_report_preserves_boolean_area_ledger_flags(tmp_path):
 sys.path.insert(0,str(p.parent));from report import rows
 s.csvwrite(tmp_path/'development_main.csv',[{'point':'a','status':'complete','cycles':100,'rows':'4+2','area_accumulator_types_explicitly_reported':True}])
 assert rows(tmp_path)[0]['area_accumulator_types_explicitly_reported'] is True

def test_report_uses_physical_signature_not_quality_metadata(tmp_path):
 sys.path.insert(0,str(p.parent));from report import rows
 signature={'binary_sha256':'engine','physical_format_sha256':'actual-format'}
 s.write(tmp_path/'active_result_signature.json',signature)
 source=[{'point':name,'status':'complete','cycles':100,'rows':'4+2','raw_binary':'engine','raw_campaign':s.digest(signature),'physical_format_sha256':'actual-format','frozen_format_sha256':qual} for name,qual in (('before','unqualified-file'),('after','qualified-file'))]
 source.append({**source[0],'point':'wrong-format','physical_format_sha256':'changed-format'})
 s.csvwrite(tmp_path/'development_main.csv',source)
 assert {r['point'] for r in rows(tmp_path)}=={'before','after'}

def test_service_histogram_quantiles_use_sample_weights():
 x=s.histogram_summary({'1':90,'3':9,'100':1})
 assert x['samples']==100 and x['mean_cycles']==2.17
 assert x['p50_cycles']==1 and x['p95_cycles']==3 and x['p99_cycles']==3 and x['max_cycles']==100

def test_missing_invariants_are_not_accepted():
 import pytest
 with pytest.raises(AssertionError):s.validate({'cycles':10,'hbm_states':{'H0':10},'cores':[]},{'arch':'supply_v3','lanes':[6]})

def test_frozen_format_changes_all_non_bf16_points_without_batch_tuning(tmp_path):
 import json
 old=s.FROZEN_FORMAT.copy();receipt=s.FORMAT_RECEIPT
 path=tmp_path/'format.json';path.write_text(json.dumps({'config':{'main_bits':3,'rank_lanes':16,'ranks':{'routed':[64,64,48],'shared':[64,64,96]}},'quality_status':'qualified_by_Q1'}))
 try:
  s.load_format(path)
  assert s.config('BL4','OP2')['main_bits']==3
  assert s.config('BL2','OP5')['rank_lanes']==16
  assert s.config('BL3','OP0')['rank_lanes']==0
  assert s.config('BL4','OP2')['t_chunk']==96
 finally:s.FROZEN_FORMAT=old;s.FORMAT_RECEIPT=receipt

def test_physical_format_signature_ignores_qualification_metadata(tmp_path):
 import json
 old=s.FROZEN_FORMAT.copy();receipt=s.FORMAT_RECEIPT
 binary=tmp_path/'engine';binary.write_bytes(b'frozen-engine');s.write(tmp_path/'input_manifest.json',{'immutable':True})
 path=tmp_path/'format.json'
 try:
  s.FROZEN_FORMAT={};s.FORMAT_RECEIPT=None
  without_file=s.campaign_signature(tmp_path,binary)
  s.write(path,{'config':s.physical_format(),'quality_status':'timing_candidate','source_sha256':'first'})
  s.load_format(path);first=s.campaign_signature(tmp_path,binary);first_quality=s.FORMAT_RECEIPT['sha256']
  assert first==without_file
  s.write(path,{'config':s.physical_format(),'quality_status':'qualified_by_Q1','source_sha256':'second'})
  s.load_format(path);assert s.campaign_signature(tmp_path,binary)==first
  assert s.FORMAT_RECEIPT['sha256']!=first_quality
 finally:s.FROZEN_FORMAT=old;s.FORMAT_RECEIPT=receipt

def test_every_actual_format_key_changes_timing_signature(tmp_path):
 import copy
 old=s.FROZEN_FORMAT.copy();receipt=s.FORMAT_RECEIPT
 binary=tmp_path/'engine';binary.write_bytes(b'frozen-engine');s.write(tmp_path/'input_manifest.json',{'immutable':True})
 try:
  s.FROZEN_FORMAT={};base=s.physical_format();signature=s.campaign_signature(tmp_path,binary)
  alternatives={'main_bits':3,'factor_a':'mxint8s','factor_b':'mxint4','rank_lanes':16,'ranks':{'routed':[16,16,16],'shared':[32,32,48]}}
  for key,value in alternatives.items():
   s.FROZEN_FORMAT=copy.deepcopy(base);s.FROZEN_FORMAT[key]=value
   assert s.campaign_signature(tmp_path,binary)!=signature,key
 finally:s.FROZEN_FORMAT=old;s.FORMAT_RECEIPT=receipt

def test_external_planner_source_changes_timing_signature(tmp_path,monkeypatch):
 binary=tmp_path/'engine';binary.write_bytes(b'frozen-engine');s.write(tmp_path/'input_manifest.json',{'immutable':True})
 sources={name:tmp_path/name for name in ('compiler.py','legacy_chunks.py','frontend.py','legacy_frozen_designs.json')}
 for name,path in sources.items():path.write_text(name)
 monkeypatch.setattr(s,'planner_artifacts',lambda:sources)
 baseline=s.campaign_signature(tmp_path,binary)
 for name,path in sources.items():
  before=path.read_text();path.write_text(before+'changed')
  assert s.campaign_signature(tmp_path,binary)!=baseline,name
  path.write_text(before)

def test_legacy_outer_plan_survives_point_cache_without_changing_routes(monkeypatch):
 import copy
 workload={'id':'large_legacy_plan_cache_case','batch':96,'tokens':[{'token_index':0,'sample_id':'unchanged','routes':[]}],'experts':[]}
 original=copy.deepcopy(workload);calls=[]
 def prepare(value,lanes,group,resources):
  calls.append((copy.deepcopy(value),lanes[:],group,copy.deepcopy(resources)))
  return {**copy.deepcopy(value),'engine_layout':{'frozen_arena':True},'legacy_batch_execution':{'schema':'plena_legacy_token_chunks_v1','chunk_size':11,'chunks':[{'token_range':[0,11],'workload':{'batch':11}}]}}
 monkeypatch.setattr(s,'prepare_legacy_workload',prepare)
 monkeypatch.setattr(s,'_LEGACY_LAYOUT_CACHE',{})
 monkeypatch.setattr(s,'_LEGACY_RESOURCE_CACHE',[{'lanes':[6],'group':4,'resources':{'acc_bytes':[2097152]}}])
 first=s.point(workload,'BL1','OP0');second=s.point(workload,'BL1','OP1')
 assert len(calls)==1 and calls[0][0]==original
 assert first['workload']['tokens']==original['tokens'] and workload==original
 assert first['workload']['legacy_batch_execution']==second['workload']['legacy_batch_execution']
 assert first['workload']['legacy_batch_execution']['chunks'][0]['workload']['batch']==11
 assert first['config']['dispatch']=='fifo' and first['config']['lanes']==[6]
 assert second['config']['credits']==544 and second['config']['legacy_diagnostic_overbudget'] is True

def legacy_validation_fixture():
 cfg={'arch':'joint_v1','lanes':[6],'credits':256,'runtime_fsm':True,'diagnostic_profile':True}
 report={'cycles':10,'drained':True,'ownership_k_order_capacity_checks':True,
  'dma_transactions_accepted':2,'dma_transactions_landed':2,'weight_bytes':64,'useful_macs':10,'credit_peak':2,
  'physical_budget':{'acc_bytes':[1000],'weight_slots':[10]},
  'cores':[{'m':6,'capacity':1000,'reserved_input_result_control':100,
   'stats':{'workspace_peak_bytes':200,'weight_peak_bytes':4096,'x_peak_bytes':6144,
    'weight_bytes':64,'useful_macs':10,'dma_accepted':2,'dma_landed':2}}],
  'm0_profile':{'mutually_exclusive':True,'hbm_states':{'H0':8,'H4':2},'hbm_sum':10,
   'core_states':[{'C0':8,'C8':2}],'core_sums':[10]}}
 return report,cfg

def test_legacy_validation_requires_drain_and_true_dma_bytes():
 import copy,pytest
 report,cfg=legacy_validation_fixture();assert s.validate(report,cfg)
 for key,value in [('drained',False),('dma_transactions_landed',1),('weight_bytes',32)]:
  bad=copy.deepcopy(report);bad[key]=value
  with pytest.raises(AssertionError):s.validate(bad,cfg)

def test_legacy_validation_checks_physical_private_peaks_and_profile():
 import copy,pytest
 report,cfg=legacy_validation_fixture()
 for key,value in [('workspace_peak_bytes',1001),('weight_peak_bytes',40961),('x_peak_bytes',12289)]:
  bad=copy.deepcopy(report);bad['cores'][0]['stats'][key]=value
  with pytest.raises(AssertionError):s.validate(bad,cfg)
 bad=copy.deepcopy(report);bad['m0_profile']['core_states'][0]['C0']=7
 with pytest.raises(AssertionError):s.validate(bad,cfg)

def test_legacy_inactive_core_still_reserves_persistent_arena():
 import copy,pytest
 report,cfg=legacy_validation_fixture();cfg.update(lanes=[4,2])
 report['cores'][0]['m']=4;report['physical_budget'].update(acc_bytes=[1000,500],weight_slots=[5,5])
 inactive=copy.deepcopy(report['cores'][0]);inactive.update(m=2,capacity=500,reserved_input_result_control=300)
 inactive['stats'].update(workspace_peak_bytes=0,weight_peak_bytes=0,x_peak_bytes=0,weight_bytes=0,useful_macs=0,dma_accepted=0,dma_landed=0)
 report['cores'].append(inactive)
 report['m0_profile'].update(core_states=[{'C0':8,'C8':2},{'C0':10}],core_sums=[10,10])
 assert s.validate(report,cfg)
 bad=copy.deepcopy(report);del bad['m0_profile']
 with pytest.raises(AssertionError):s.validate(bad,cfg)

def test_legacy_compression_credit_expansion_is_explicit_counterfactual():
 import copy,pytest
 report,cfg=legacy_validation_fixture();cfg['weight_bytes_scale']=.25
 report.update(credit_peak=300,config={'credits':1024,'diagnostic_credit_expansion':True})
 assert s.validate(report,cfg)
 bad=copy.deepcopy(report);bad['config']['credits']=256
 with pytest.raises(AssertionError):s.validate(bad,cfg)
 cfg['diagnostic_profile']=False
 with pytest.raises(AssertionError):s.validate(report,cfg)

def test_legacy_chunk_validation_checks_aliases_refetch_and_serial_time():
 import copy,pytest
 child,cfg=legacy_validation_fixture();report=copy.deepcopy(child)
 report.update(cycles=24,weight_bytes=128,useful_macs=20,dma_transactions_accepted=4,dma_transactions_landed=4,unique_weight_bytes=64,refetch_bytes=64)
 report['cores'][0]['stats'].update(weight_bytes=128,useful_macs=20,dma_accepted=4,dma_landed=4)
 report['m0_profile'].update(hbm_states={'H0':16,'H4':8},hbm_sum=24,core_states=[{'C0':16,'C8':8}],core_sums=[24])
 report['legacy_chunks']={'schema':'plena_legacy_token_chunks_v1','chunk_size':2,'chunks':2,
  'global_routes_and_scores_preserved':True,'persistent_addresses_checked':True,'refetch_bytes':64,
  'full_layer_persistent_cores':[{'capacity':1000,'reserved':100,'original_x':{'shape':[4,2]},'combined_output':{'shape':[4,2]}}], 'setup_cycles':4,'setup_sram_bytes':128,
  'parts':[{'token_range':[i*2,i*2+2],'start_cycle':i*12,'setup_cycles':2,'setup_sram_bytes':64,
   'kernel_start_cycle':i*12+2,'kernel_report':copy.deepcopy(child)} for i in range(2)]}
 assert s.validate(report,cfg)
 for mutate in ('alias','refetch','time','route','truncated'):
  bad=copy.deepcopy(report)
  if mutate=='alias':bad['legacy_chunks']['persistent_addresses_checked']=False
  elif mutate=='refetch':bad['refetch_bytes']=0
  elif mutate=='time':bad['legacy_chunks']['parts'][1]['kernel_start_cycle']=13
  elif mutate=='route':bad['legacy_chunks']['parts'][1]['token_range']=[1,3]
  else:
   bad['legacy_chunks']['full_layer_persistent_cores'][0]['original_x']['shape'][0]=5
   bad['legacy_chunks']['full_layer_persistent_cores'][0]['combined_output']['shape'][0]=5
  with pytest.raises(AssertionError):s.validate(bad,cfg)

def test_legacy_flatten_exposes_capacity_extension():
 report,cfg=legacy_validation_fixture();cfg=s.config('BL1','OP0')
 report.update(unique_weight_bytes=64,refetch_bytes=128)
 report['legacy_chunks']={'chunk_size':11,'chunks':9,'setup_cycles':77,'setup_sram_bytes':512,'reset_policy':'full drain and refetch'}
 workload={'id':'large','batch':96,'experts':[]}
 point={'key':'p','suite':'main','workload':workload,'design':'BL1','op':'OP0','port':'iso','variant':'default','config':cfg}
 row=s.flatten({'point':point,'report':report,'status':'complete'})
 assert row['legacy_batch_chunked'] and row['legacy_batch_chunk_size']==11 and row['legacy_batch_chunk_count']==9
 assert row['legacy_batch_refetch_bytes']==128 and row['legacy_batch_unique_weight_bytes']==64
 assert row['legacy_batch_setup_cycles']==77 and row['legacy_batch_setup_sram_bytes']==512

def test_report_legacy_capacity_table_separates_current_real_mixed_baseline():
 import math
 sys.path.insert(0,str(p.parent));from render_report import legacy_capacity_groups
 base={'evaluation_split':'mixed_heldout','suite':'main','op':'OP0','port':'iso','design':'BL1','tokens':96,
  'legacy_batch_chunk_size':11,'legacy_batch_chunk_count':9,'legacy_batch_unique_weight_bytes':1048576,
  'legacy_batch_refetch_bytes':2097152,'legacy_batch_setup_cycles':2000,'ms':20.0}
 values=[base,{**base,'legacy_batch_chunk_size':10,'ms':5.0},{**base,'evaluation_split':'mixed_development'},{**base,'op':'OP1'}]
 result=legacy_capacity_groups(values);assert len(result)==1
 assert result[0]['windows']==2 and result[0]['chunk_size_range']=='10–11'
 assert result[0]['unique_weight_mib_mean']==1 and result[0]['refetch_mib_mean']==2
 assert result[0]['setup_ms_mean']==.002 and math.isclose(result[0]['ms_geomean'],10)

def test_heldout_qualification_sha_is_independently_frozen(tmp_path):
 import pytest
 old=s.FROZEN_FORMAT.copy();receipt=s.FORMAT_RECEIPT
 try:
  s.FORMAT_RECEIPT={'sha256':'quality-new'};signature={'physical_format_sha256':'hardware-fixed'}
  auth={'authorized':True,'prereg_commit':'committed','frozen_signature':signature,'quality_receipt_sha256':'quality-old'}
  s.write(tmp_path/'heldout_authorization.json',auth)
  with pytest.raises(AssertionError,match='qualification receipt'):s.verify_heldout_authorization(tmp_path,signature)
  auth['quality_receipt_sha256']='quality-new';s.write(tmp_path/'heldout_authorization.json',auth)
  assert s.verify_heldout_authorization(tmp_path,signature)==auth
 finally:s.FROZEN_FORMAT=old;s.FORMAT_RECEIPT=receipt

def test_n2_complete_evidence_produces_pass_and_failure():
 sys.path.insert(0,str(p.parent));from assess import compensation_assessment
 evidence=[]
 for op,modes in {'OP5':('lanes','separate','kext','offload'),'OP2':('lanes',)}.items():
  for mode in modes:
   for i in range(8):
    evidence.append({'mode':mode,'op':op,'workload':f'{i}','tokens':96 if i<6 else 16,'shared_me':96 if i<6 else 16,'provenance':'captured_mixed' if i<6 else 'captured_decode','sum_core_busy_overhead':.02 if mode=='lanes' else .11,'layer_overhead':.02 if mode=='lanes' else .06,'equal_wire_bytes':True,'rank_multiplier_overhead':.015625})
 assert compensation_assessment(evidence,8)['passed'] is True
 evidence[0]['sum_core_busy_overhead']=.04
 assert compensation_assessment(evidence,8)['passed'] is False
 assert compensation_assessment(evidence[:-1],8)['passed'] is None

def test_n4_keeps_original_threshold_and_labels_type_dependence():
 sys.path.insert(0,str(p.parent));from assess import n4_verdict
 nominal=n4_verdict(.99,.80,.03,False)
 assert nominal['passed'] is True and nominal['area_based_claim_type_conditional'] is True
 assert nominal['unconditional_organization_evidence'] is False
 assert n4_verdict(.95,1.0,.20,False)['passed'] is True
 assert n4_verdict(.96,.86,.03,True)['passed'] is False

def test_compensation_primary_keeps_default_B_and_supplement_is_separate(tmp_path):
 import json
 (tmp_path/'inputs').mkdir();(tmp_path/'inputs/development.json').write_text(json.dumps({'workloads':[{'id':'tiny','batch':2,'experts':[]}]}))
 points=s.points(tmp_path,'development','compensation')
 assert len(points)==12 and sum(not q['unavailable_reason'] for q in points)==11
 primary=[q for q in points if q['op']=='OP2' and not q['variant'].startswith('b4_')]
 assert {q['variant'] for q in primary}=={'none','lanes'}
 assert all(q['config']['factor_b']=='bf16' and q['config']['comp_equal_bytes'] for q in primary)
 assert all(q['config']['factor_b']=='mxint4' for q in points if q['variant'].startswith('b4_'))

def test_m5_physical_energy_storage_uses_bf16_rne():
 sys.path.insert(0,str(p.parent));from m5_causal import bf16_rne
 assert bf16_rne(1.0)==1.0
 assert bf16_rne(1.00390625)==1.0
 assert bf16_rne(1.01171875)==1.015625

def test_concurrent_identical_blob_publication_is_atomic(tmp_path):
 import json,threading
 from concurrent.futures import ThreadPoolExecutor
 barrier=threading.Barrier(8);target=tmp_path/'shared.json';value={'weights':[1,2,3],'capacity':2158592}
 def publish(_):barrier.wait();s.write(target,value)
 with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(publish,range(8)))
 assert json.loads(target.read_text())==value
 assert not list(tmp_path.glob('*.tmp'))

def test_comparator_receipt_is_bound_to_complete_active_campaign(tmp_path):
 import csv,json,pytest
 (tmp_path/'inputs').mkdir();s.write(tmp_path/'inputs/development.json',{'workloads':[{'id':f'w{i}','batch':2,'experts':[]} for i in range(12)]})
 signature={'binary_sha256':'frozen-engine'};s.write(tmp_path/'active_result_signature.json',signature)
 planned=s.points(tmp_path,'development','dev_comparator');rows=[]
 times={'BL2':100,'BL3':80,'BL5':90,'M8_single':100,'M8_homogeneous':90,'M8_flex62':70,'M8_flex53':75}
 for point in planned:
  rows.append({'point':point['key'],'design':point['design'],'cycles':times[point['design']],'status':'complete','raw_binary':'frozen-engine'})
  s.write(tmp_path/'raw'/s.digest(signature)[:16]/point['key']/'receipt.json',{'signature':signature,'bit_identical':True,'repeats':2,'sha256':'original-json'})
 s.csvwrite(tmp_path/'development_dev_comparator.csv',rows)
 receipt={'points':84,'complete':84,'failed':0,'unsupported':0,'excluded':0,'signature':signature};s.write(tmp_path/'development_dev_comparator_receipt.json',receipt)
 s.comparator(tmp_path)
 ready=json.loads((tmp_path/'development_freeze_ready_receipt.json').read_text())
 assert ready['selected_by_budget']=={'M6':'BL3','M8':'M8_flex62'} and ready['heldout_authorized'] is False
 s.write(tmp_path/'active_result_signature.json',{'binary_sha256':'changed-engine'})
 with pytest.raises(AssertionError):s.comparator(tmp_path)

def test_development_launcher_stops_on_unsupported_legal_point(tmp_path):
 import json,pytest
 sys.path.insert(0,str(p.parent));from run_development import verify_stage
 signature={'binary_sha256':'engine'};s.write(tmp_path/'active_result_signature.json',signature)
 receipt={'points':2,'complete':1,'excluded':0,'failed':0,'unsupported':1,'signature':signature,'host_elapsed_seconds':1}
 s.write(tmp_path/'development_all_receipt.json',receipt)
 with pytest.raises(AssertionError):verify_stage(tmp_path,'development','all')
 receipt.update(unsupported=0,excluded=1);s.write(tmp_path/'development_all_receipt.json',receipt)
 assert verify_stage(tmp_path,'development','all')['excluded']==1

def test_heldout_requires_exact_frozen_signature(tmp_path):
 import argparse,pytest
 binary=tmp_path/'engine';binary.write_bytes(b'not executed')
 s.write(tmp_path/'input_manifest.json',{'frozen':True})
 s.write(tmp_path/'heldout_authorization.json',{'authorized':True,'prereg_commit':'existing-commit','frozen_signature':{'binary_sha256':'obsolete'}})
 with pytest.raises(AssertionError,match='exact binary'):
  s.execute(argparse.Namespace(root=str(tmp_path),binary=str(binary),split='heldout'))

def test_prereg_generator_preserves_nominal_and_quality_thresholds():
 sys.path.insert(0,str(p.parent));from preregister import THRESHOLDS,QUALITY_THRESHOLDS
 assert THRESHOLDS['N4_area_difference_max']==.05
 assert THRESHOLDS['N4_mixed_time_ratio_max']==.95 and THRESHOLDS['N4_traffic_ratio_max']==.85
 assert QUALITY_THRESHOLDS=={'layer_error_over_mxint4_rtn_max':.5,'layer_cosine_min':.999,'q4_perplexity_increase_max':.02}

def test_prereg_generator_cannot_finalize_without_matching_comparator(tmp_path,monkeypatch):
 import pytest
 sys.path.insert(0,str(p.parent));import preregister as pr
 s.write(tmp_path/'prereg_payload.json',{'thresholds':pr.THRESHOLDS,'inputs':{}})
 s.write(tmp_path/'area_coefficients.json',{'relative_only':True})
 monkeypatch.setattr(pr,'matrix_inventory',lambda root:[])
 with pytest.raises(AssertionError,match='actual approved binary'):pr.freeze_manifest(tmp_path,None,draft=False)
 draft=pr.freeze_manifest(tmp_path,None,draft=True)
 assert draft['heldout_authorized'] is False and draft['status']=='draft_not_authorized'

def test_native_wire_sensitivity_only_changes_matching_flag(tmp_path):
 sys.path.insert(0,str(p.parent));from compensation_native import plans
 ws=[{'id':f'joint_test_{d}_b{b}_l13_s7','batch':b,'experts':[]} for d in ('bfcl','gpqa','swe') for b in (2,4,8,16)]
 s.write(tmp_path/'inputs/development.json',{'workloads':ws});pairs=plans(tmp_path)
 assert len(pairs)==42
 assert sum(native['op']=='OP5' for original,native in pairs)==30
 assert sum(native['op']=='OP2' for original,native in pairs)==12
 for original,native in pairs:
  changed={k for k in original['config'] if original['config'][k]!=native['config'][k]}
  assert changed=={'comp_equal_bytes'} and native['config']['comp_equal_bytes'] is False
  assert original['workload']==native['workload']

def test_report_n5_never_infers_quality_from_timing_or_completion():
 sys.path.insert(0,str(p.parent));from render_report import n5_verdict
 assert n5_verdict(True,{'complete':True}) is None
 assert n5_verdict(True,{'complete':False,'N5_quality_pass':True}) is None
 assert n5_verdict(None,{'complete':True,'N5_quality_pass':True}) is None
 assert n5_verdict(True,{'complete':True,'N5_quality_pass':False}) is False
 assert n5_verdict(False,{'complete':True,'N5_quality_pass':True}) is False
 assert n5_verdict(True,{'complete':True,'N5_quality_pass':True}) is True
 assert n5_verdict(True,{'complete':True,'N5_quality_pass':True,'physical_format_sha256':'different'},'actual') is None
 assert n5_verdict(True,{'complete':True,'N5_quality_pass':True,'physical_format_sha256':'actual'},'actual') is True

def test_report_organization_table_uses_identical_actual_windows():
 import pytest
 sys.path.insert(0,str(p.parent));from render_report import organization_groups,DESIGN_ORDER
 rows=[dict(evaluation_split='heldout',suite='main',op='OP2',port='iso',design=d,provenance='captured_decode',tokens=2,workload='common',ms=1+i) for i,d in enumerate(DESIGN_ORDER)]
 rows.append({**rows[0],'workload':'unpaired_slow_window','ms':1000})
 provenance,tokens,count,values=organization_groups(rows)[0]
 assert count==1 and values==pytest.approx({'BL2':1,'BL3':2,'BL4':3,'BL5':4})
 assert organization_groups(rows[:-2]+rows[-1:])[0][2]==0

def test_historical_stall_panel_requires_development_pair_and_exclusive_states(tmp_path):
 sys.path.insert(0,str(p.parent));from report import historical_supply_pairs
 path=tmp_path/'m0'/'window'/'4+2'/'profile_on'/'repeat0.json'
 s.write(path,dict(cycles=100,m0_profile=dict(mutually_exclusive=True,core_states=[dict(C1=25,C2=75),dict(C0=100)])))
 row=dict(workload='window',point='actual',suite='main',design='BL4',port='iso',op='OP2',split='development',tokens=2,cycles=80,core0_states_C1=40,core0_states_C2=40,core1_states_C0=80)
 pairs=historical_supply_pairs(tmp_path,[row,{**row,'split':'heldout'},{**row,'workload':'missing'}])
 assert len(pairs)==2 and pairs[0]['legacy_C2']==.75 and pairs[0]['v3_C2']==.5
 assert 'Not heldout' in pairs[0]['scope'] and len(pairs[0]['legacy_raw_sha256'])==64

def test_numerical_pareto_keeps_q1_norms_and_actual_all_expert_bytes(tmp_path):
 sys.path.insert(0,str(p.parent));from report import numerical_pareto_rows
 row=dict(scope='layer',layer=13,rank_lanes=8,bits=4,method='qera_approx',rank=32,factor_a='mxint4',factor_b='bf16',relative_error=.02,cosine=.999,factor_bytes=123456)
 s.csvwrite(tmp_path/'quant/full_numerics/q1_metrics.csv',[row,{**row,'scope':'projection'}])
 s.write(tmp_path/'quant/full_numerics/q1_coverage.json',dict(complete=False,expected_candidates=2178))
 points,receipt=numerical_pareto_rows(tmp_path)
 assert len(points)==1 and points[0]['total_physical_weight_bytes']==points[0]['main_bytes']+123456
 assert points[0]['main_bytes']>64*2048*1408*.5 and 'full8192' in points[0]['scope']
 assert receipt['quality_campaign_complete'] is False and receipt['candidates']==1
