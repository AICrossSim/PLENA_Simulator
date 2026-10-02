#!/usr/bin/env python3
"""Analyze observed Joint timing without adding overlapping stage times to latency."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path

def write_csv(path,rows):
    keys=sorted({k for row in rows for k in row})
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)

def analyze(root):
    root=Path(root);groups=defaultdict(dict);stages=[];intervals=[];fits=[]
    for path in sorted(root.glob('*/*/*/repeat0.json')):
        workload,org,condition=path.relative_to(root).parts[:3]
        r=json.loads(path.read_text());groups[(workload,org)][condition]=r
        p=r.get('m0_profile')
        if not p:continue
        for key,(count,operand,mac,rmw) in p['stage_totals_count_operand_mac_rmw'].items():
            stages.append(dict(workload=workload,organization=org,condition=condition,phase=key,
                issues=count,operand_latency_mean_ns=operand/count,mac_latency_mean_ns=mac/count,
                rmw_latency_mean_ns=rmw/count,issue_to_commit_latency_mean_ns=(operand+mac+rmw)/count,
                interpretation='per-issue critical path latency; overlaps other issues; not layer-time decomposition'))
        for c,hist in enumerate(p['issue_interval_histograms']):
            count=sum(hist.values());mean=sum(int(k)*v for k,v in hist.items())/max(count,1)
            intervals.append(dict(workload=workload,organization=org,condition=condition,core=c,
                intervals=count,mean_issue_interval_ns=mean,layer_cycles=r['cycles'],
                interpretation='includes starvation and phase boundaries; not pure MAC service'))
    comparison=[]
    for (workload,org),g in sorted(groups.items()):
        old=g['frozen_old'];issues=sum(c['stats']['issues'] for c in old['cores'])
        wp=root/workload/org/'frozen_old'/'workload.json';w=json.loads(wp.read_text())
        tiles=sum(2*math.ceil(e['F']/4)*math.ceil(e['H']/512)+math.ceil(e['H']/4)*math.ceil(e['F']/512) for e in w['experts'])
        estimate=max(1.026*old['weight_bytes']/128,30.4*issues)
        # This empirical fit was supplied for single-core only.
        if org=='6':fits.append(dict(workload=workload,observed_ns=old['cycles'],
            supplied_fit_ns=estimate,fit_relative_error=estimate/old['cycles']-1,
            issues=issues,tiles=tiles,issues_per_tile=issues/tiles,ns_per_tile=old['cycles']/tiles,
            weight_bytes=old['weight_bytes']))
        p=g['profile_on']['m0_profile'];count=sum(v[0] for v in p['stage_totals_count_operand_mac_rmw'].values())
        sums=[sum(v[i] for v in p['stage_totals_count_operand_mac_rmw'].values()) for i in range(1,4)]
        comparison.append(dict(workload=workload,organization=org,legacy_ms=old['cycles']/1e6,
            ideal_hbm_ms=g['ideal_hbm']['cycles']/1e6,ideal_onchip_ms=g['ideal_onchip']['cycles']/1e6,
            both_ideal_ms=g['both_ideal']['cycles']/1e6,compression_only_ms=g['weight_scale']['cycles']/1e6,
            compression_speedup=old['cycles']/g['weight_scale']['cycles'],
            compression_realization_R=old['cycles']/g['weight_scale']['cycles']/3.481,
            credits544_ms=g['credits544']['cycles']/1e6,credit_speedup=old['cycles']/g['credits544']['cycles'],
            per_issue_operand_mean_ns=sums[0]/count,per_issue_mac_mean_ns=sums[1]/count,
            per_issue_rmw_mean_ns=sums[2]/count,per_issue_total_latency_ns=sum(sums)/count,
            profile_timing_unchanged=old['cycles']==g['profile_on']['cycles']==g['profile_off']['cycles'],
            exclusive=p['mutually_exclusive']))
    write_csv(root/'m0_stage_latencies.csv',stages);write_csv(root/'m0_issue_intervals.csv',intervals)
    write_csv(root/'m0_comparison.csv',comparison);write_csv(root/'m0_legacy_fit.csv',fits)
    note={
        'model':'preserved Joint analytical finite-port model; ns = cycles at 1GHz',
        'latency_vs_throughput':'Issue→operand→MAC→RMW is a latency path. Independent issues overlap, so its sum must not be compared as an additive wall-clock split.',
        'hypothesis_A':'Not the stated fully serial hypothesis: W SRAM and X SRAM operand reads run concurrently via max(W_end,X_end), and multiple independent output contexts overlap. These dependencies are visible in main.rs.',
        'hypothesis_B':'No fixed 30-cycle descriptor handshake exists. MAC latency is configured at20cycles; operand reads and ordered accumulator feedback are separately modeled. A constant31cycle issue duration would be an assumption, not this measurement.',
        'observer':'All old timing and counter fields are checked by m0_profile.py, excluding only added config keys.',
        'prediction_scope':'Compare supplied fit and oracle predictions against these observations; differences are reported, not used to retune acceptance numbers.',
        'single_fit_mean_abs_error':sum(abs(r['fit_relative_error']) for r in fits)/max(len(fits),1),
        'single_fit_max_abs_error':max((abs(r['fit_relative_error']) for r in fits),default=0),
    }
    (root/'m0_analysis.json').write_text(json.dumps(note,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    axes[0].scatter([r['observed_ns']/1e6 for r in fits],[r['supplied_fit_ns']/1e6 for r in fits])
    lim=max([r['observed_ns']/1e6 for r in fits]+[1]);axes[0].plot([0,lim],[0,lim],'k--')
    axes[0].set(xlabel='Observed single-core layer latency(ms)',ylabel='Supplied empirical fit(ms)')
    xs=[r['issues_per_tile'] for r in fits]
    axes[1].scatter(xs,[r['ns_per_tile'] for r in fits],label='Observed single6')
    xmin,xmax=min(xs),max(xs);xx=[xmin+(xmax-xmin)*i/99 for i in range(100)]
    wire=1.026*sum(r['weight_bytes']/r['tiles']/128 for r in fits)/len(fits)
    axes[1].plot(xx,[max(wire,30.4*x) for x in xx],label='Supplied max model')
    axes[1].set(xlabel='Issues per unique weight tile',ylabel='Layer ns per tile');axes[1].legend()
    by=defaultdict(list)
    for r in stages:
        if r['condition']=='profile_on':by[r['organization']].append(r)
    orgs=sorted(by);bottom=[0.]*len(orgs)
    for key,label in [('operand_latency_mean_ns','Operand'),('mac_latency_mean_ns','MAC latency'),('rmw_latency_mean_ns','RMW')]:
        values=[sum(r[key]*r['issues'] for r in by[o])/sum(r['issues'] for r in by[o]) for o in orgs]
        axes[2].bar(orgs,values,bottom=bottom,label=label);bottom=[a+b for a,b in zip(bottom,values)]
    axes[2].set(xlabel='Organization',ylabel='Per-issue latency(ns); paths overlap');axes[2].legend()
    fig.tight_layout();dest=root.parent/'figures';dest.mkdir(exist_ok=True)
    fig.savefig(dest/'legacy_fit.svg');fig.savefig(dest/'legacy_fit.png',dpi=160);plt.close(fig)
    return note

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(analyze(a.root),indent=2))
