"""Feasible ready-record scheduling across distinct expert GEMMs on one core.

Every issue still belongs to one expert, projection and output tile: this
does not co-pack rows or use different weights in the same matrix invocation.
Only independent invocations from different experts may be interleaved.
The queue policy is deterministic round-robin over ready output records,
not an optimal instruction-scheduling algorithm.
"""
from __future__ import annotations

from collections import deque
from heapq import heappop, heappush
from operator import index

from .search import scalar_expert, timing


def _integer(value, label, minimum=1):
    if isinstance(value,bool):
        raise ValueError(f"{label} must be an integer")
    try:
        value = index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def schedule_ready_pool(experts, core, profile="flat20", *, record_trace=False):
    """Return an exact event replay of a feasible interleaved issue policy.

    All experts are available at cycle zero. GU records are initially ready;
    an expert's Down records become ready only after every Gate and Up final
    K segment commits (SiLU/product are free in this compute-only scope).
    The next K segment of any output record waits its own previous commit.
    Every installed record/operand is unlimited and resident for this oracle.

    ``cycles`` describes the actual ready-pool policy. A separately reported
    ``best_feasible_cycles`` may choose the inherited serial-expert schedule
    when round-robin is slower. Neither result is claimed globally optimal.
    The trace is optional because large cases can contain millions of issues.
    """
    t = timing(profile) if isinstance(profile,str) else profile
    latency = _integer(t.completion_latency(core),"completion latency")
    if _integer(t.initiation_interval,"initiation interval") != 1:
        raise ValueError("this ready-pool experiment requires inherited II=1")
    pm,pn,pk = (core.pm,core.pn,core.pk)
    if min(pm,pn,pk) < 1:
        raise ValueError("positive core dimensions required")
    jobs = []
    for original in experts:
        e = dict(original)
        e['Me'] = _integer(e['Me'],"Me",0)
        e['H'] = _integer(e['H'],"H")
        e['F'] = _integer(e['F'],"F")
        if e['Me']:
            jobs.append(e)
    if len({e['id'] for e in jobs}) != len(jobs):
        raise ValueError("expert ids must be unique")
    # The same logical-work order applies to every compute geometry.
    jobs.sort(key=lambda e: (-3*e['Me']*e['H']*e['F'],repr(e['id'])))
    records = []
    ready = deque()
    gu_count = []
    down_records = []
    useful = total_issues = 0
    baseline_cycles = baseline_peak_records = 0
    for ei,e in enumerate(jobs):
        m,h,f = e['Me'],e['H'],e['F']
        nm,nf,nh = (m+pm-1)//pm,(f+pn-1)//pn,(h+pn-1)//pn
        kh,kf = (h+pk-1)//pk,(f+pk-1)//pk
        gu_count.append(2*nm*nf)
        downs = []
        # Records contain [expert, phase, K segments, issued segments,
        # last commit, allocated partial, pending invocation].
        for phase,nn,nk in ((0,nf,kh),(1,nf,kh),(2,nh,kf)):
            for _n in range(nn):
                for _m in range(nm):
                    r = len(records)
                    records.append([ei,phase,nk,0,0,False,False])
                    (downs if phase == 2 else ready).append(r)
        down_records.append(downs)
        total_issues += 2*nm*nf*kh+nm*nh*kf
        useful += 3*m*h*f
        if isinstance(profile,str):
            scalar = scalar_expert(e,core,profile,0)
            baseline_cycles += scalar['cycles']
            baseline_peak_records = max(baseline_peak_records,
                                        scalar['required_live_FP32_output_bytes']//core.record_bytes)
        else:
            # Independent closed form matching the inherited serial-expert
            # stream; custom timing is useful for tiny hand-calculated tests.
            def stage(q,s):
                return (s-1)*max(q,latency)+q-1+latency
            baseline_cycles += stage(2*nm*nf,kh)+stage(nm*nh,kf)
            baseline_peak_records = max(baseline_peak_records,2*nm*nf,nm*nh)

    remaining_gu = gu_count.copy()
    remaining_down = [len(r) for r in down_records]
    gu_finish = [None]*len(jobs)
    down_first_issue = [None]*len(jobs)
    expert_finish = [None]*len(jobs)
    events = []
    issue_trace = [] if record_trace else None
    now = serial = issues = live = peak_live = peak_ready = peak_inflight = 0
    last_issue = -1
    previous_expert = None
    expert_switches = idle = 0
    max_done = 0
    while ready or events:
        while events and events[0][0] <= now:
            done,_,r = heappop(events)
            state = records[r]
            ei,phase,nk,issued,_,allocated,pending = state
            assert pending and allocated
            state[4],state[6] = done,False
            max_done = max(max_done,done)
            if issued < nk:
                ready.append(r)
            elif phase < 2:
                remaining_gu[ei] -= 1
                if remaining_gu[ei] == 0:
                    gu_finish[ei] = done
                    # The completed GUI values feed free SiLU/product into Z.
                    # No completed GU output is dropped before this barrier.
                    live -= gu_count[ei]
                    ready.extend(down_records[ei])
            else:
                state[5] = False
                live -= 1
                remaining_down[ei] -= 1
                if remaining_down[ei] == 0:
                    expert_finish[ei] = done
        peak_ready = max(peak_ready,len(ready))
        if ready:
            r = ready.popleft()
            state = records[r]
            ei,phase,nk,kseg,committed,allocated,pending = state
            assert not pending and kseg < nk and now >= committed
            assert now >= last_issue+1
            if phase == 2:
                assert gu_finish[ei] is not None and now >= gu_finish[ei]
                if down_first_issue[ei] is None:
                    down_first_issue[ei] = now
            if not allocated:
                assert kseg == 0
                state[5] = True
                live += 1
                peak_live = max(peak_live,live)
            done = now+latency
            state[3],state[6] = kseg+1,True
            heappush(events,(done,serial,r))
            serial += 1
            issues += 1
            peak_inflight = max(peak_inflight,len(events))
            if previous_expert is not None and previous_expert != ei:
                expert_switches += 1
            previous_expert,last_issue = ei,now
            if record_trace:
                issue_trace.append({'expert_id':jobs[ei]['id'],'phase':('Gate','Up','Down')[phase],
                                    'record':r,'k_segment':kseg,'issue_cycle':now,
                                    'previous_commit_cycle':committed,'commit_cycle':done})
            now += 1
        elif events:
            next_event = max(now,events[0][0])
            idle += next_event-now
            now = next_event
    assert issues == total_issues and live == 0
    assert all(v is not None for v in expert_finish)
    actual = max_done
    issued_macs = issues*pm*pn*pk
    assert issued_macs >= useful
    assert actual >= issues
    assert actual == max(expert_finish,default=0)
    use_pool = actual <= baseline_cycles
    output = {
        'scope':'feasible ready-record round-robin; no optimal instruction-scheduling claim',
        'timing_profile':getattr(t,'name',str(profile)),
        'cycles':actual,'issues':issues,'useful_macs':useful,'issued_macs':issued_macs,
        'padding_macs':issued_macs-useful,
        'spatial_utilization':useful/issued_macs if issued_macs else 1.,
        'peak_live_partial_records':peak_live,
        'peak_live_partial_bytes':peak_live*pm*pn*4,
        'installed_record_descriptors_if_precreated':len(records),
        'peak_ready_records':peak_ready,'peak_inflight_dot_results':peak_inflight,
        'completion_latency_cycles':latency,'initiation_interval_cycles':1,
        'idle_issue_cycles_including_final_drain':idle,
        'issue_gap_cycles_before_last_issue':last_issue+1-issues if issues else 0,
        'final_drain_cycles':max_done-last_issue-1 if issues else 0,
        'expert_switches_between_issues':expert_switches,
        'whole_expert_baseline_cycles':baseline_cycles,
        'whole_expert_baseline_peak_partial_records':baseline_peak_records,
        'best_feasible_cycles':min(actual,baseline_cycles),
        'best_feasible_source':'ready_pool' if use_pool else 'whole_expert_baseline',
        'best_feasible_peak_partial_records':peak_live if use_pool else baseline_peak_records,
        'expert_timelines':[{'expert_id':e['id'],'Me':e['Me'],
                             'GU_all_committed_cycle':gu_finish[i],
                             'Down_first_issue_cycle':down_first_issue[i],
                             'Down_all_committed_cycle':expert_finish[i]}
                            for i,e in enumerate(jobs)],
        'assumptions':['all expert weights/X and all output contexts resident/unlimited',
                       'Gate/Up final outputs retained until this expert GU barrier',
                       'zero SiLU/product/combine/dispatch/ports/NoC/DMA service',
                       'one expert and one weight tile per issue; no mixed-expert row packing',
                       'completed Down tiles may stream to untimed combined-output storage'],
    }
    if record_trace:
        output['issue_trace'] = issue_trace
    return output
