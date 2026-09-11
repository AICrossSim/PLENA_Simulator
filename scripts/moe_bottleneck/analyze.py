#!/usr/bin/env python3
"""Aggregate validated repeats and retain per-core overlapping service counters."""
import csv,json
from pathlib import Path
OUT=Path('/scratch/shared/mcl123/plena/outputs/moe_bottleneck_20260911')

def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    assert m['status']=='passed',m['status']
    measured=list(csv.DictReader((OUT/'measurements.csv').open()))
    assert len(measured)==m['planned_runs']
    rows=[];cores=[];by={};port_service_checks=0
    for c in m['cases']:
        for profile in m['profiles']:
            r=json.loads((OUT/c['id']/f'{profile}.rep1.json').read_text())['result']
            arch=json.loads((OUT/c['id']/f'{profile}.arch.json').read_text())
            row={'case':c['id'],'window':c['window']['name'],'organization':c['organization'],'mode':c['mode'],
                 'profile':profile,'total_us':r['total_ps']/1e6,'speedup':c['prior_total_ps']/r['total_ps'],
                 'latency_reduction_pct':100*(1-r['total_ps']/c['prior_total_ps']),
                 'useful_macs':r['useful_macs'],'issued_macs':r['issued_macs'],'hbm_read_bytes':r['hbm_read_bytes'],
                 'nominal_compute_lower_bound_us':r['issued_macs']/4096/1000}
            rows.append(row);by[(c['id'],profile)]=row
            base=json.loads((OUT/c['id']/'baseline.rep1.json').read_text())['result']
            for i,core in enumerate(r['cores']):
                b=base['cores'][i]
                assert b['useful_macs']==core['useful_macs'] and b['issued_macs']==core['issued_macs']
                for field,key in [('weight_port_busy_ps','weight_port_speedup'),('accumulator_port_busy_ps','accumulator_speedup')]:
                    factor=arch['diagnostic'].get(key,1)
                    if factor<64:
                        assert b['refinement'][field]==core['refinement'][field]*factor,(c['id'],profile,field)
                        port_service_checks+=1
                ref=core['refinement'];pool=ref.get('output_pool') or {}
                cores.append({'case':c['id'],'profile':profile,'core':core['id'],'jobs':core['jobs'],
                    'useful_macs':core['useful_macs'],'issued_macs':core['issued_macs'],
                    'issue_and_activation_us':core['compute_busy_ps']/1e6,
                    'weight_wait_us':core['weight_ready_wait_ps']/1e6,
                    'feedback_wait_us':core['accumulator_dependency_stall_ps']/1e6,
                    'weight_port_service_us':ref['weight_port_busy_ps']/1e6,
                    'accumulator_service_us':ref['accumulator_port_busy_ps']/1e6,
                    'scheduler_service_us':pool.get('scheduler_busy_ps',0)/1e6,
                    'issued_mac_ceiling_fraction':core['issued_macs']/(core['multipliers']*r['total_ps']/arch['clock_period_ps']*arch['diagnostic'].get('mac_speedup',1))})
    write_csv(OUT/'sensitivity.csv',rows);write_csv(OUT/'core_services.csv',cores)
    comparison=[]
    for window in ('qwen_full_decode_b8','qwen_full_decode_b32'):
        for p in m['profiles']:
            single=by[(window+'_single_legacy_n3',p)]
            single_pool=by[(window+'_single_pool_q32',p)]
            best_single=min(single['total_us'],single_pool['total_us'])
            for mode in ('legacy_n2','pool_q32'):
                dual=by[(window+'_heterogeneous_'+mode,p)]
                comparison.append({'window':window,'profile':p,'dual_mode':mode,
                    'single_n3_us':single['total_us'],'dual_us':dual['total_us'],
                    'dual_speedup_vs_single_n3':single['total_us']/dual['total_us'],
                    'single_q32_us':single_pool['total_us'],'best_single_us':best_single,
                    'dual_speedup_vs_best_single':best_single/dual['total_us']})
    write_csv(OUT/'organization_comparison.csv',comparison)
    summary={'status':'passed','runs':len(measured),'unique_points':len(rows),'numerical_exact_and_repeated':True,'independent_port_service_checks':port_service_checks,
             'hbm_bytes_changed_runs':sum(x['hbm_bytes_changed']=='True' for x in measured),
             'organization_comparisons':comparison}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PASS',len(rows),'points',len(measured),'runs')
    for row in rows:
        if row['profile'] in ('baseline','mac2','accumulator2','hbm_clock2','supply2','all2','ideal_hbm','ideal_supply64','ideal_supply64_mac2'):
            print(row['case'],row['profile'],round(row['total_us'],3),round(row['speedup'],3))

if __name__=='__main__':main()
