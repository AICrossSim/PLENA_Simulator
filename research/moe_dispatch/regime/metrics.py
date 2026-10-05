"""Conditional service bounds and paired statistics, never additive stall time."""
from collections import defaultdict
import math
import numpy as np


def geomean(values):
    vals=list(values)
    if not vals or any(x<=0 for x in vals):
        raise ValueError('positive paired ratios required')
    return math.exp(sum(math.log(v) for v in vals)/len(vals))


def bounds(result,settings):
    b=result['budget']; ps=result['phases']; total=result['cycles']
    comp=defaultdict(float); w=defaultdict(float); x=defaultdict(float); acc=defaultdict(float)
    for p in ps:
        c=p['core']; comp[c]+=p['compute']; w[c]+=p['w_port_bytes']; x[c]+=p['x_port_bytes']; acc[c]+=p['acc_port_bytes']
    hbw=settings.fabric.landing_credit_bandwidth_upper_bound
    hbm=result['hbm_bytes']/hbw
    unique=result['native_unique_bytes']/hbw
    port=max(sum(w.values())/(64*16),sum(x.values())/(24*16),sum(acc.values())/(12*16),result['global_activation_bytes']/384,
             *(w[i]/m['w_banks']/16 for i,m in enumerate(b['cores'])),
             *(x[i]/m['x_banks']/16 for i,m in enumerate(b['cores'])),
             *(acc[i]/m['accumulator_banks']/16 for i,m in enumerate(b['cores'])))
    compute=max(comp.values())
    conditional=max(hbm,compute,port)
    # Architecture-independent arithmetic lower bound; true minimum port
    # traffic depends on the mapped protocol, so it is not silently inferred.
    absolute_compute=result['useful_macs']/12288
    universal=max(unique,absolute_compute)
    # Necessary traffic for the FIXED staging protocol, independent of core
    # shapes, owners, quotas, padded lanes and reloads. Using actual mapped
    # traffic as a geometry-search floor would incorrectly rule out the very
    # traffic reduction that geometry/dataflow changes are meant to obtain.
    # BF16 valid W is staged and read at least once; valid X/Z is filled/read;
    # final GU/Down FP32 records are written and read by their consumers.
    if 'compulsory_traffic' not in result:
        raise ValueError('workload compulsory-traffic ledger required for search bound')
    minimum=result['compulsory_traffic']
    port_min=max((result['native_unique_bytes']+12*minimum['sum_HF'])/(64*16),
                 4*minimum['sum_MH_plus_MF']/(24*16),
                 minimum['accumulator_bytes']/(12*16),
                 minimum['global_activation_bytes']/384)
    search_lb=max(unique,absolute_compute,port_min)
    assert search_lb<=conditional*(1+1e-9)
    assert conditional <= total*(1+1e-9), (conditional,total)
    return {'hbm_lb_ms':hbm/1e6,'unique_weight_hbm_lb_ms':unique/1e6,
            'compute_fixed_owner_lb_ms':compute/1e6,'port_fixed_mapping_lb_ms':port/1e6,
            'max_conditional_lb_ms':conditional/1e6,
            'gap_over_conditional_percent':100*(total/conditional-1),
            'global_HBM_MAC_lb_ms':universal/1e6,
            'gap_over_global_HBM_MAC_percent':100*(total/universal-1),
            'compute_peak_lb_ms':absolute_compute/1e6,
            'port_compulsory_lb_ms':port_min/1e6,
            'architecture_search_lb_ms':search_lb/1e6,
            'architecture_headroom_percent':100*(total/search_lb-1),
            'effective_hbm_GBps':result['hbm_bytes']/total,
            'hbm_equivalent_busy_fraction':hbm/total,
            'hbm_cap_GBps':hbw,'bound_scope':'fixed traffic/owner/mapping; global column omits geometry-dependent ports'}


def core_rows(result,settings):
    out=[]; wall=result['cycles']
    for c,m in enumerate(result['budget']['cores']):
        ps=[p for p in result['phases'] if p['core']==c]
        compute=sum(p['compute'] for p in ps)
        w=sum(p['w_port_bytes'] for p in ps)/m['w_banks']/16
        x=sum(p['x_port_bytes'] for p in ps)/m['x_banks']/16
        acc=sum(p['acc_port_bytes'] for p in ps)/m['accumulator_banks']/16
        stream=sum(p['stream_completed']-p['started'] for p in ps)
        finish=result['core_finish_cycles'][c]
        reason=max(result['exclusive_stream_counters'][c],key=lambda k:result['exclusive_stream_counters'][c][k] if k not in ('idle','hbm_startup') else -1)
        out.append({'workload':result['workload'],'batch':result['batch'],'core':c,
          'core_shape':f"{m['pm']}x{m['pn']}x{m['pk']}",
          'compute_dependency_service_ms':compute/1e6,'W_port_service_ms':w/1e6,
          'X_port_service_ms':x/1e6,'acc_port_service_ms':acc/1e6,
          'stream_interval_ms':stream/1e6,'finish_ms':finish/1e6,
          'tail_idle_ms':(wall-finish)/1e6,'dominant_fluid_limiter':reason,
          'scope':'analytical service demands; overlap; not measured array busy/native stall breakdown'})
    return out


def paired_score(candidate,reference):
    if len(candidate)!=len(reference) or any(a['workload']!=b['workload'] for a,b in zip(candidate,reference)):
        raise ValueError('identical ordered windows required')
    return geomean(a['cycles']/b['cycles'] for a,b in zip(candidate,reference))


def paired_bootstrap(candidate,reference,seed=20261005,samples=4000):
    paired_score(candidate,reference)
    ratios=np.array([a['cycles']/b['cycles'] for a,b in zip(candidate,reference)],dtype=float)
    assert len(ratios)>0 and len(candidate)==len(reference)
    logs=np.log(ratios); rng=np.random.default_rng(seed)
    # Preserve each batch's sample count, and always pair both architectures.
    groups=defaultdict(list)
    for i,r in enumerate(reference): groups[r['batch']].append(i)
    totals=np.zeros(samples)
    for ids in groups.values():
        ids=np.array(ids)
        drawn=rng.choice(ids,size=(samples,len(ids)),replace=True)
        totals+=logs[drawn].sum(axis=1)
    savings=100*(1-np.exp(totals/len(ratios)))
    return {'geomean_latency_ratio':float(np.exp(logs.mean())),
            'time_reduction_percent':float(100*(1-np.exp(logs.mean()))),
            'bootstrap95_low_percent':float(np.quantile(savings,.025)),
            'bootstrap95_high_percent':float(np.quantile(savings,.975)),
            'bootstrap_samples':samples,'seed':seed,
            'bootstrap_scope':'paired stratified windows; correlated/historically exposed captures are not independent model trials'}
