#!/usr/bin/env python3
"""Export audited uniform-budget lambda variants from calibration-only spectra."""
import argparse,copy,json
from pathlib import Path
import numpy as np
import evaluate as q


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--table',type=Path,required=True)
    ap.add_argument('--calibration',type=Path,required=True);ap.add_argument('--uniform-ranks',default='16,32');ap.add_argument('--hardware-bf16',action='store_true');args=ap.parse_args()
    table=json.loads(args.table.read_text());manifest=json.loads(args.calibration.read_text());layer=int(table['layer']);routes=[];gates=[]
    for s in manifest['shards']:
        p=Path(s['path']);p=p if p.is_absolute() else args.calibration.parent/p
        if q.sha(p)!=s['sha256']:raise ValueError('calibration payload hash mismatch')
        with np.load(p) as z:routes.append(z[f'routes_l{layer}']);gates.append(z[f'gates_l{layer}'])
    rr,gg=np.concatenate(routes),np.concatenate(gates);cal={'x':np.empty((len(rr),0)), 'routes':rr,'gates':gg}
    if len(rr)!=table['calibration_tokens']:raise ValueError('table/capture calibration count mismatch')
    software_te,rb=np.asarray(table['tail_energy']),np.asarray(table['factor_bytes']);te=software_te
    physical_projection={k:q.rn.bf16_round(np.asarray(v,np.float32)).tolist() for k,v in table['tail_energy_per_projection'].items()}
    if args.hardware_bf16:te=sum(np.asarray(v,np.float64) for v in physical_projection.values())
    for uniform in map(int,args.uniform_ranks.split(',')):
        index=table['rank_candidates'].index(uniform);lam,fit,_=q.fit_lambda_window_budget(cal,te,rb,index)
        value=copy.deepcopy(table);value.update(lambda0=lam,lambda_bounds=[lam/100,lam*100],lambda_fit=fit,
            uniform_reference_rank=uniform,uniform_factor_budget_bytes=float(rb[:,index].sum()),
            initial_control_state={'lambda':lam,'windows_observed':0},
            variant_provenance={'spectra_table_sha256':q.sha(args.table),'calibration_manifest_sha256':q.sha(args.calibration),
                                'source':'calibration only; no validation gates or simulator timing'})
        if args.hardware_bf16:
            software_lam,software_fit,_=q.fit_lambda_window_budget(cal,software_te,rb,index)
            weights=[];active=[]
            for lo in range(0,len(rr),16):
                w=np.zeros(64)
                for ids,gs in zip(rr[lo:lo+16],gg[lo:lo+16]):
                    for e,g in zip(ids,gs):w[e]+=float(g)**2
                weights.append(w);active.append(np.isin(np.arange(64),np.unique(rr[lo:lo+16])))
            idx=np.argmin(np.asarray(weights)[:,:,None]*te[None,:,:]+software_lam*rb[None,:,:],axis=2)
            diagnostic=float((rb[np.arange(64)[None,:],idx]*np.asarray(active)).sum(axis=1).mean())
            value.update(tail_energy=te.tolist(),tail_energy_per_projection=physical_projection,
                software_float_lambda0=software_lam,software_float_lambda_fit=software_fit,
                float_lambda0_on_hardware_diagnostic={'mean_allocated_window_bytes':diagnostic,
                    'mean_uniform_window_bytes':fit['mean_uniform_window_bytes'],'byte_delta':diagnostic-fit['mean_uniform_window_bytes']},
                physical_energy_storage='BF16 RNE perprojection; decoded values summed; exact uint32 bytecost',
                origin='physical BF16-energy LUT and actual development-window routed factor bytes; validation never used to calibrate')
            value['variant_provenance']['evaluator_source_sha256']=q.sha(Path(q.__file__))
            value['variant_provenance']['export_source_sha256']=q.sha(Path(__file__))
            stem=args.table.stem.split('_uniform')[0].replace('rank_table_','rank_table_hw_bf16_',1)
        else:stem=args.table.stem.split('_uniform')[0]
        path=args.table.with_name(stem+f'_uniform{uniform}.json');path.write_text(json.dumps(value,indent=2)+'\n')
        print(json.dumps({'path':str(path),'lambda0':lam,'fit':fit}))


if __name__=='__main__':main()
