"""Describe captured workload populations without hardware selection."""
from .common import ROOT, inputs, native_bytes, useful_macs, write_csv, write_json, sha
from .model import Parameters


def main():
    rows=[]
    for split, windows in inputs().items():
        if split not in ('development','heldout'):
            continue
        for w in windows:
            routed=[e for e in w['experts'] if not e['is_shared']]
            shared=[e for e in w['experts'] if e['is_shared']]
            rows.append({'split':split,'window_id':w['id'],'batch':w['batch'],
                'active_routed_experts':len(routed),'routed_Me_le_2':sum(e['Me']<=2 for e in routed),
                'routed_Me_gt_2':sum(e['Me']>2 for e in routed),'shared_tasks':len(shared),
                'shared_Me':','.join(str(e['Me']) for e in shared),
                'routed_token_assignments':sum(e['Me'] for e in routed),
                'unique_weight_bytes':native_bytes(w),'useful_MACs':useful_macs(w),
                'unique_weight_HBM_floor_ms':native_bytes(w)/Parameters().hbm_bandwidth/1e6})
    write_csv(ROOT/'results/E0/workload_profile.csv',rows)
    write_json(ROOT/'results/E0/workload_profile_protocol.json',{'windows':len(rows),
        'formula':'Active means Me>0; Me<=2 is a low-token expert, not an inactive expert. Shared is separate. Unique BF16 weights /126.030769GB/s gives the HBM floor.',
        'command':'/tmp/plena-round2-venv/bin/python -m research.moe_dispatch.round2.profile_inputs',
        'input_manifest_sha256':sha(ROOT/'results/E0/frozen_inputs.json')})
    print(len(rows),'captured workload profiles')


if __name__=='__main__':
    main()
