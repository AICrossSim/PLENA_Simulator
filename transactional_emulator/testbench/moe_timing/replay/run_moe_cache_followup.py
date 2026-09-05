#!/usr/bin/env python3
"""Extend cache-disabled evidence after identifying cache-port domination.

This is an explicitly adaptive diagnostic, not a predeclared primary campaign.
It adds all three missing windows; none is selected on a favorable speedup.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from compare_moe_normal import digest, read_json, run_comparison
from run_moe_full_campaign import save


def run(root,binary):
    manifest=read_json(root/'campaign.json')
    names=[n for n in manifest['fixtures'] if n!='deepseek_full_decode_b32']
    paths=[root/'sensitivity_architectures/cache_disabled'/(n+'.json')
           for n in manifest['architectures']]
    declaration=dict(reason='Qwen primary reports show single-core cache-port domination and zero hits on K512/K1024. Test bypass for every remaining window.',
        scope='adaptive diagnostic covering all previously missing windows, not selective positive examples',
        fixtures=names,executable_sha256=digest(binary),driver_sha256=digest(__file__),
        comparison_driver_sha256=digest(Path(__file__).with_name('compare_moe_normal.py')),
        architectures=[dict(path=str(p),sha256=digest(p)) for p in paths],
        repeats=manifest['repeats'],hbm_channels=manifest['hbm_channels'],tolerance=manifest['numerical_tolerance'])
    save(root/'cache_followup_plan.json',declaration)
    outcomes={}
    save(root/'cache_followup_status.json',dict(status='running',all_gates_passed=False,outcomes=outcomes))
    def execute(name):
        print('Starting cache bypass '+name,flush=True)
        try:
            run_comparison(binary,root/name/'workload.json',root/name/'golden.json',paths,
                root/'comparisons'/(name+'_cache_disabled'),repeats=manifest['repeats'],
                hbm_channels=manifest['hbm_channels'],workers=2,timeout=900,**manifest['numerical_tolerance'])
            return dict(status='passed')
        except Exception as error:return dict(status='failed',error=str(error))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(execute,name):name for name in names}
        for future in as_completed(futures):
            name=futures[future];outcomes[name]=future.result()
            print(name+': '+outcomes[name]['status'],flush=True)
            save(root/'cache_followup_status.json',dict(status='running',all_gates_passed=False,outcomes=outcomes))
    passed=len(outcomes)==len(names) and all(o['status']=='passed' for o in outcomes.values())
    save(root/'cache_followup_status.json',dict(status='passed' if passed else 'failed',all_gates_passed=passed,
        execution_plan_sha256=digest(root/'cache_followup_plan.json'),outcomes=outcomes))
    if not passed:raise RuntimeError('cache follow-up failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--binary',type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.binary.resolve())
