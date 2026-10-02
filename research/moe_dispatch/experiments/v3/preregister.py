#!/usr/bin/env python3
"""Generate a reviewable preregistration; never commit or authorize heldout.

Draft generation is read-only with respect to simulator timing. Final generation
requires the complete active 84-point development comparator and a binary whose
physical-format/input/runner signature matches its original-JSON receipts.
"""
from __future__ import annotations
import argparse,copy,csv,json,time
from pathlib import Path
import study

THRESHOLDS={
 'N1_Rlegacy_max':.40,'N1_Rv3_min':.85,
 'N2_busy_overhead_max':.03,'N2_onchip_layer_overhead_max':.03,
 'N2_other_busy_overhead_min':.10,'N2_other_layer_overhead_min':.05,
 'N2_cold_throughput_loss_min':.10,'N2_multiplier_overhead_max':.032,
 'N3_OP0_b16_eta_min':.95,'N3_OP1_eta_min':.90,'N3_leave_one_loss_min':.03,
 'N4_mixed_time_ratio_max':.95,'N4_traffic_ratio_max':.85,'N4_area_difference_max':.05,
 'N5_time_ratio_max':.97,'N5_p95_ratio_max':.95,
 'N5_Q3_error_ratio_max':.90,'N5_Q3_bytes_ratio_max':.80,
}
QUALITY_THRESHOLDS={'layer_error_over_mxint4_rtn_max':.50,'layer_cosine_min':.999,'q4_perplexity_increase_max':.02}
SPLITS=('development','heldout','constructed','mixed_development','mixed_heldout')

def matrix_inventory(root):
 rows=[]
 for split in SPLITS:
  ps=study.points(root,split,'all');by={}
  for point in ps:by.setdefault(point['suite'],[]).append(point)
  if split=='development':by['dev_comparator']=study.points(root,split,'dev_comparator')
  for suite,points in sorted(by.items()):
   excluded=sum(bool(p['unavailable_reason']) for p in points)
   rows.append(dict(split=split,suite=suite,points=len(points),legal=len(points)-excluded,excluded=excluded,repeats=2))
 return rows

def freeze_manifest(root,binary=None,draft=False):
 root=Path(root)
 payload=json.loads((root/'prereg_payload.json').read_text())
 assert payload['thresholds']==THRESHOLDS,'Changing a nominal acceptance threshold is prohibited'
 artifacts={key:study.sha(root/key) for key in payload['inputs']}
 for key,value in artifacts.items():assert value==payload['inputs'][key],f'Prepared input changed: {key}'
 inventory=matrix_inventory(root)
 fmt=study.physical_format()
 result=dict(schema='plena_supply_v3_prereg_2',status='draft_not_authorized' if draft else 'ready_for_root_commit_not_authorized',
  heldout_authorized=False,created_unix=time.time(),physical_format=fmt,physical_format_sha256=study.digest(fmt),
  quality_receipt=copy.deepcopy(study.FORMAT_RECEIPT),quality_receipt_sha256=study.FORMAT_RECEIPT['sha256'] if study.FORMAT_RECEIPT else None,
  thresholds=THRESHOLDS,quality_thresholds=QUALITY_THRESHOLDS,input_artifacts_sha256=artifacts,
  base=copy.deepcopy(study.BASE),designs=copy.deepcopy(study.DESIGNS),working_points=copy.deepcopy(study.OPS),
  matrix=inventory,legal_points=sum(r['legal'] for r in inventory),executions=2*sum(r['legal'] for r in inventory),
  source_sha256={name:study.sha(study.HERE/name) for name in ('study.py','report.py','assess.py','render_report.py','run_development.py','run_final.py','m5_causal.py','compensation_native.py','preregister.py')},
  area_coefficients=json.loads((root/'area_coefficients.json').read_text()),protocol=payload)
 result['actual_planner_artifacts']={name:{'path':str(path),'sha256':study.sha(path)} for name,path in study.planner_artifacts().items()}
 v3_compiler=study.planner_artifacts()['compiler.py'].with_name('compiler_v3.py')
 result['compiler_v3_non_timing_source']={'path':str(v3_compiler),'sha256':study.sha(v3_compiler)}
 numerical_paths=list((study.RESEARCH/'quant').glob('*.py'))+list((study.RESEARCH/'v3_reference').glob('*.py'))
 result['numerical_source_sha256']={str(path.relative_to(study.SIM)):study.sha(path) for path in sorted(numerical_paths)}
 numerical_inputs=('quant/real_captures/development/manifest.json','quant/real_captures/heldout/manifest.json','quant/q0_weight_inventory.json')
 result['numerical_input_artifacts']={name:{'path':str(root/name),'sha256':study.sha(root/name)} for name in numerical_inputs if (root/name).is_file()}
 if study.FORMAT_RECEIPT:
  qualification=json.loads(Path(study.FORMAT_RECEIPT['path']).read_text())
  result['frozen_numerical_qualification_content']=qualification
  if qualification.get('quality_receipt_path'):
   proof=Path(qualification['quality_receipt_path'])
   assert proof.is_file() and study.sha(proof)==qualification['quality_receipt_sha256'],'Independent qualification proof changed'
   result['independent_quality_proof']={'path':str(proof),'sha256':study.sha(proof)}
 rust_paths=list((study.RESEARCH/'rust/src').rglob('*.rs'))+[study.RESEARCH/'rust/Cargo.toml',study.RESEARCH/'rust/Cargo.lock']
 assert all(path.is_file() for path in rust_paths),'The validated Rust dependency manifest and lockfile are required'
 result['actual_rust_source_sha256']={str(path.relative_to(study.SIM)):study.sha(path) for path in sorted(rust_paths)}
 if binary is not None:result['frozen_signature']=study.campaign_signature(root,binary)
 if not draft:
  assert binary is not None,'Final freeze requires the actual approved binary'
  ready=json.loads((root/'development_freeze_ready_receipt.json').read_text())
  assert ready['signature']==result['frozen_signature'],'Development selection does not match final physical campaign'
  assert ready['points']==84 and ready['runs']==168 and ready['all_valid_and_bit_identical']
  assert ready['heldout_authorized'] is False
  assert study.sha(root/'development_fixed_comparator.json')==ready['comparator_sha256']
  selected=json.loads((root/'development_fixed_comparator.json').read_text())
  assert selected['signature']==result['frozen_signature'] and selected['selected_by_budget']==ready['selected_by_budget']
  result.update(selected_by_budget=ready['selected_by_budget'],development_geometric_mean_cycles=selected['geometric_mean_cycles'],
   comparator_sha256=ready['comparator_sha256'],development_ready_sha256=study.sha(root/'development_freeze_ready_receipt.json'),
   development_original_json_sha256=ready['original_json_receipts'])
  source_manifest=Path(binary).parent/'source_manifest.json';assert source_manifest.exists(),'Final release requires its actual Rust source manifest'
  compiled=json.loads(source_manifest.read_text());assert compiled['binary_sha256']==result['frozen_signature']['binary_sha256']
  assert compiled['source_hashes']==result['actual_rust_source_sha256'],'Current Rust sources differ from the validated immutable release'
  result['compiled_source_manifest']={'path':str(source_manifest),'sha256':study.sha(source_manifest),'content':compiled}
  validation={}
  for path in sorted((root/'tests').rglob('*receipt.json')):
   receipt=json.loads(path.read_text())
   if receipt.get('binary_sha256')==result['frozen_signature']['binary_sha256'] or path.name=='compiler_reference_receipt.json':validation[str(path.relative_to(root))]={'sha256':study.sha(path),'content':receipt}
  assert any(r['content'].get('pytest_passed_cases',0)>=55 and r['content'].get('exit_code')==0 for r in validation.values()),'Missing matching 55-case timed numerical validation'
  assert any(r['content'].get('pytest_passed_cases',0)>=47 and r['content'].get('pytest_passed_subtests',0)>=271 and r['content'].get('exit_code')==0 for r in validation.values()),'Missing matching 47-case/271-subtest legacy validation'
  assert any(r['content'].get('rust_unit_tests_passed',0)>=71 and (r['content'].get('all_passed') is True or (r['content'].get('full_raw_byteexact') is True and r['content'].get('total_runs',0)>=46)) for r in validation.values()),'Missing matching Rust unit and complete duplicate smoke validation'
  assert 'tests/compiler_reference_receipt.json' in validation,'Missing paired Compiler correctness receipt'
  result['correctness_receipts']=validation
 return result

def markdown(m):
 status='DRAFT: final engine and development selection pending' if m['status'].startswith('draft') else 'Frozen for root commit; heldout timing remains unauthorized'
 selected=m.get('selected_by_budget',{'M6':'PENDING: minimum development GM of BL2/BL3/BL5','M8':'PENDING: minimum development GM of single/homogeneous/flex62/flex53'})
 quality=m['quality_receipt'] or {'quality_status':'timing_candidate_not_accuracy_validated','sha256':None}
 designs='\n'.join(f"| {name} | {' + '.join(str(x)+'×4×512' for x in cfg['lanes'])} | {' / '.join(cfg['dataflow'])} |" for name,cfg in m['designs'].items())
 matrix='\n'.join(f"| {r['split']} | {r['suite']} | {r['legal']} | {r['excluded']} |" for r in m['matrix'])
 hashes='\n'.join(f"| `{path}` | `{value}` |" for path,value in m['input_artifacts_sha256'].items())
 return f'''# PLENA-MoE Supply-first v3 preregistration

Status: **{status}**. This file does not authorize evaluation. The root researcher
must commit this file and the adjacent frozen manifest, then record that commit,
the exact frozen timing signature, and the independent quality-receipt SHA in
`heldout_authorization.json`. No heldout timing is used to select the design,
format, comparator, or thresholds. All existing provisional runs remain separate.

## Scope and input populations

Latency times one complete routed SwiGLU MoE FFN layer after routing. Router,
attention and full-model inference are excluded. This is deterministic analytical
Rust event simulation with finite HBM/credit/ingress/pool/ports; it is not native
Ramulator, RTL or silicon measurement. At 1 GHz, 1 cycle = 1 ns and 10⁶ cycles =
1 ms. Host execution time is only campaign progress.

Twelve previously exposed windows are development data. Architectural decode
holdout comprises 108 windows from 90 request identities, at layers 2/13/26,
decode steps 3/7/11 and B2/B4/B8/B16. Identities are SHA-selected and disjoint
from every exposed development request. Reused groups are correlated. Genuine
mixed holdout comprises 27 BFCL/GPQA/SWE windows at those three layers and
T64/96/128, drawn from only three independent BF16-CPU prefill prompt captures
plus distinct B16 decode requests. N4/N5 use the 18 T64/96 windows; T128 is a
separate stress population. SWE canonical input was recovered and byte-matched
to its original prepared-file SHA. Six real mixed development windows and 18
separately labeled constructed/rebatched/Zipf cases are not mixed into holdout
conclusions. Original archive-inventory missing-prefill status is historical;
`mixed_capture_manifest.json` records the subsequent real captures.

## Frozen computation, storage and supply

Physical dimensions are **M×N×K**, with N=4 and K=512. M6 organizations all contain
12,288 main multipliers. M8 organizations share a separate 16,384 multiplier
budget; no M6/M8 comparison is claimed to be equal-compute.

| Design | Main-core dimensions | Dataflow |
|---|---|---|
{designs}

Hardware remains fixed across workloads. Physical storage ceiling is 2,158,592 B,
including unused allocated SRAM and 16 KiB control reserve. Physical t_chunk=128
only for P2/L8 at 128/256 B/ns; P1, L16 and bandwidth512 use fixed t_chunk=96.
T128 combinations that violate this allocation are explicit exclusions.
ISO fixes task-specified aggregate pool read=1,024 B/cycle, X=640 B/cycle,
P1 decoder=2,048 B/cycle and installed WOR=18 slots. Group sizes are single/dense8,
homogeneous4 per core, stream1. Demand-port variants report their own costs.
**Private accumulator/source shared 1RW ports are max(128,32×M_c) B/cycle:**
single6=192; homogeneous128+128; asymmetric128+128; single8=256; D6=192.
They are not equal across organizations, and their actual installed width is
included in the component ledger. Vector/UStore reads and RMW share those ports.
U-cache capacity is inside the fixed control reserve; its additional local 1R1W
read/write bandwidth is explicitly costed.

One shared HBM path uses 32 B requests. OP0/OP1: BF16 main weights, 256 B/ns,
64 ns, 256/544 credits. OP2: P2/W4, 256 B/ns, 64 ns,544 credits. OP3: 150 ns,
1,232 credits. OP4: W3 conditional numerical candidate. OP5: P1. Latency/BW
sensitivity also uses the prespecified full inventory. Legacy BL0/BL1 credits544
are labeled overbudget diagnostics; oracle byte shrink retains BF16 local moves.

Default runtime uses window8, two contexts, supply_ipd, bounded prefetch/drop
work stealing, starved-only context switching and dot dependency latency20.
Context age=max(working-point HBM latency, current group tile count×ceil(Me/M));
this is one fixed shape-dependent hardware rule, not per-batch tuning.
Pipeline/quota/byte-pool/reuse/ports/SiLU off-switches must execute actual finite
alternative mechanisms; no fixed issue-gap proxy is accepted.

## Physical numerical format and separate qualification

Frozen actual format: `{json.dumps(m['physical_format'],sort_keys=True)}`.
Physical-format SHA256: `{m['physical_format_sha256']}`.
Current qualification status: **{quality['quality_status']}**.
Final independent qualification-receipt SHA256: `{m['quality_receipt_sha256']}`.

P2 uses MX low-bit weights with BF16 X/U/Z. `factor_a` describes the low-rank
factor A, not input activation precision; these results are not W4A4. Main input
scaling and mixed-input PE area are not RTL characterized. Numerical quantities
have independent real-weight receipts. Timing signature always hashes the five
normalized actual format fields, including defaults. Changing only qualification
metadata does not invalidate identical physical timing; every bit/factor/L/rank
change creates a new physical campaign and comparator. The final quality file
SHA is separately fixed in preregistration and authorization. No legacy timing
without this explicit physical-format proof is retroactively merged.

Quality selection uses only calibration/development numerical data: ≥32k
calibration and ≥8k request-disjoint evaluation tokens at layer13, plus layers2/26.
When only layer-level results exist, required MoE output error is ≤0.50× W4-RTN
and cosine ≥0.999. If mandatory GPU Q4 is available, perplexity increase is ≤2%.
Choose minimum bytes among qualifying hardware-legal candidates. If none pass,
retain the default only as an explicitly unqualified timing candidate; do not
relax any threshold. W3, B-MXINT4 compensation supplements, and policies whose
rank vectors differ require their own accuracy evidence.

## Comparator frozen from development

M6 comparator: **{selected['M6']}**. M8 comparator: **{selected['M8']}**.
Select minimum geometric mean cycles over all 12 development windows at OP2 ISO
from BL2/BL3/BL5 or M8 single/homogeneous/flex62/flex53, respectively. All 84
candidate points require two original identical valid raw JSONs. Never choose a
new best comparator per heldout point or change hardware for a workload.

## Original acceptance criteria (unchanged)

| Claim | Evidence and frozen criterion |
|---|---|
| N1 bottleneck transfer | BL1 R_legacy≤0.40 and BL4 R_v3≥0.85 on decode B2–16 plus real mixedT64/96. R uses equal organization/input/port comparisons, unique useful weight bytes, and simulated time. |
| N2 rank channel | P1 five equal-wire modes; P2 frozen-default none/lanes primary. Lanes sum-core busy overhead≤3%, dense SharedMe≥16/realT96 layer overhead≤3%, installed multipliers≤3.2%. Each of separate/kext/offload must show busy overhead≥10%, or focal layer overhead≥5%, or cold-throughput loss≥10%. Per-core ownership changes are diagnostic. |
| N3 supply mechanisms | All eight forward-add and leave-one-out controls at OP0/OP1/OP2; each leave-one control has time or η loss≥3% in at least one working point. Full OP0 B16 η≥0.95 and OP1 B2–16 η≥0.90. Mechanisms without evidence lose novelty claims. |
| N4 organization | Real mixed BL4 time GM≤0.95× fixed comparator, **or** complete onchip bytes/useful-MAC GM≤0.85 with nominal area-proxy difference≤5%. Decode±3% equivalence reported. ISO/demand and M8 reported separately. |
| N5 runtime | BL4 supply_ipd versus actual v3 joint: real mixed time GM≤0.97 or cross-window p95 ratio≤0.95. All policies share window8/resources/supply path. Physical-BF16 gate_weighted Q3 error≤0.90× uniform at equal bytes, or bytes≤0.80 at equal error. |

N2 B-MXINT4 modes remain supplemental conditional-quality rows, not substitutes
for frozen-default primary comparisons. Equal-wire compensation includes real
32 B padding through credit/ingress/pool; installed rank lanes remain charged
when disabled. Busy=sum exclusive C1+C4+C5+C6+C7 across cores, with matched
work populations. Padding matches total expert-tail wire bytes, not identical
per-tile layouts, A-factor locations or phase consumption order; this is not
pure arithmetic isolation. An additional six development B2/B16 windows×five
P1 modes/two P2 modes×two raw repeats disables wire padding (84 additional
runs), reports native-wire compensation and matching-induced overhead shifts,
and does not substitute or modify any primary N2 gate or main inventory.
For N5 rank integration, common expert-rank choice precedes
projection-cap clipping. Use BF16-RNE energy LUT and its independently calibrated
λ0; routed-only budget changes reset λ0, same budget carries causal feedback.
Two actual reset sequences are repeated; software float-LUT quality is separate.

## Nominal area proxy and additional sensitivity

Default coefficients: main MX1, main/rank BF16 2, divided by12,288; storage SRAM1
and RF4, divided by2MiB; ports0.1 per2,048 logical R+W B/cycle. Full allocated
storage remains charged. WS-capable accumulator arena is interpreted as RF;
specialized IS arena as SRAM, reflecting one-cycle WS RMW. This is a physical-type
proxy interpretation, not mapped RTL. Also report hybrid RF=min(arena,
2×t_chunk×32×4) plus SRAM overflow, all-accumulator-SRAM, and all-accumulator-RF
bounds at **unchanged measured timing**, and coefficient multipliers0.5/1/2.

**N4 nominal criterion remains the original frozen default-proxy criterion.**
`area_type_robust_pass` and coefficient sensitivity are separate diagnostics,
not new hidden thresholds. A nominal area/traffic pass that depends on type is
reported as conditional and cannot support an unconditional hardware-area claim.
No synthesis PPA, power, joules or absolute area is inferred from this proxy.
Canonical port costs cover finite pool/HBM/X/accumulator/Combine/U-cache service
widths. The full-row local WOR/XOR register broadcast demand is separately
reported; a supplementary proxy adds its width with the same port coefficient.
This sensitivity does not silently revise the nominal N4 area gate. Local RF
fanout is not free, and its physical implementation still requires synthesis.

## Full declared matrix and replication

{m['legal_points']:,} legal points ×2 = **{m['executions']:,} raw simulator runs**;
all explicit exclusions remain in the inventory. Historical M0 144×2 is separately
preserved, in addition to full-window M0 extras below. No point is silently removed.

| Split | Suite | Legal points | Declared exclusions |
|---|---|---:|---:|
{matrix}

Each point archives both original JSON byte strings in gzip, verifies exact
identity and hashes after decompression, and checks all eight invariants,
accepted/landed/drained requests, fixed resource peaks, exclusive core/HBM state
sums, complete onchip movement-category sum and wire-byte decomposition.
Failures/unsupported legal points block completion; a missing claim produces
unavailable, and complete evidence produces pass/fail without changed thresholds.
Aggregate figures separate measured event-model results, closed-form reference
estimates, constructed inputs, and conditional numerical candidates.

## Input hashes

| Artifact | SHA256 |
|---|---|
{hashes}

The adjacent frozen manifest additionally records the final executable, timing
runner, analysis scripts, actual format, final quality receipt, comparator and
84 original development JSON hashes. Root commit and heldout authorization must
match those exact records. Any future substantive correction requires a new
campaign and explicit disclosure; older failed or provisional artifacts remain.
'''

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=study.DEFAULT_ROOT);p.add_argument('--binary',type=Path);p.add_argument('--format-config',default='');p.add_argument('--draft',action='store_true');p.add_argument('--markdown',type=Path,default=study.RESEARCH/'PREREG_V3.md');p.add_argument('--manifest',type=Path,default=study.RESEARCH/'FROZEN_MANIFEST_V3.json');a=p.parse_args()
 fmt=Path(a.format_config) if a.format_config else a.root/'frozen_format.json'
 if a.format_config or fmt.exists():study.load_format(fmt)
 manifest=freeze_manifest(a.root,a.binary,a.draft)
 study.write(a.manifest,manifest);a.markdown.parent.mkdir(parents=True,exist_ok=True);a.markdown.write_text(markdown(manifest))
 print(json.dumps({'status':manifest['status'],'manifest':str(a.manifest),'markdown':str(a.markdown),'legal_points':manifest['legal_points'],'executions':manifest['executions'],'heldout_authorized':False},sort_keys=True))
if __name__=='__main__':main()
