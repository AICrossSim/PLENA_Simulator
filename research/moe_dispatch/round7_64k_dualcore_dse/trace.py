"""Read the 135 original windows without recropping, resampling or fabrication."""
from collections import Counter
import json
from pathlib import Path
from .common import CFG, REPO, ROOT, save_json, save_csv, sha

def validate_window(w):
    b = int(w['batch'])
    if b not in CFG['batches'] or w['top_k'] != CFG['top_k']:
        raise ValueError('Unexpected batch or top-k')
    counts = [0]*CFG['routed_experts']
    if len(w['tokens']) != b:
        raise ValueError('Missing real per-token routing')
    for j, token in enumerate(w['tokens']):
        ids = [int(x['expert_id']) for x in token['routes']]
        if token['token_index'] != j or len(ids) != 6 or len(set(ids)) != 6:
            raise ValueError('Invalid real top-six routing')
        for eid in ids:
            if not 0 <= eid < 64:
                raise ValueError('Invalid expert ID')
            counts[eid] += 1
    if sum(counts) != 6*b:
        raise ValueError('sum(Me) != 6B')
    shared = []
    for e in w['experts']:
        if e['H'] != 2048:
            raise ValueError('Hidden dimension mismatch')
        if e['is_shared']:
            shared.append(e)
        elif e['F'] != 1408 or counts[e['id']] != e['Me']:
            raise ValueError('Token/expert histogram mismatch')
    if len(shared) != 1 or shared[0]['Me'] != b or shared[0]['F'] != 2816:
        raise ValueError('Expected the archived fused Shared descriptor')
    if sum(not e['is_shared'] for e in w['experts']) != sum(m>0 for m in counts):
        raise ValueError('Active expert descriptor count mismatch')
    return counts

def load_inputs(write=True):
    manifest_path = REPO / CFG['trace_manifest']
    manifest = json.loads(manifest_path.read_text())
    config_path = Path(CFG['hf_config'])
    hf = json.loads(config_path.read_text())
    expected = dict(hidden_size=2048, moe_intermediate_size=1408,
                    n_routed_experts=64, num_experts_per_tok=6, n_shared_experts=2)
    if any(hf.get(k) != v for k, v in expected.items()):
        raise ValueError('Local Hugging Face configuration mismatch')
    ws, sources, original = [], [], []
    for filename in ('heldout.json', 'mixed_heldout.json'):
        path = Path(manifest['input_directory']) / filename
        if sha(path) != manifest['input_sha256'][filename]:
            raise ValueError('Frozen trace hash mismatch: '+str(path))
        sources.append(dict(path=str(path), sha256=sha(path)))
        for index, w in enumerate(json.loads(path.read_text())['workloads']):
            counts = validate_window(w)
            provenance = w['provenance']
            original.append(w)
            ws.append(dict(id=w['id'], batch=w['batch'], dataset=w['dataset'],
                           layer=w['layer'], step=w['step'], counts=counts,
                           tasks=[dict(id=-1, m=w['batch'], shared=True),
                                  dict(id=-2, m=w['batch'], shared=True)] +
                                 [dict(id=e, m=m, shared=False) for e, m in enumerate(counts) if m],
                           source=str(path), source_index=index, provenance=provenance,
                           scope=w['scope'], request_ids=w.get('request_ids', [])))
    if len(ws) != 135 or [w['id'] for w in ws] != manifest['heldout_window_ids']:
        raise ValueError('Original 135-window identity/order changed')
    counts = Counter(w['batch'] for w in ws)
    if counts != {2:27, 4:27, 8:27, 16:27, 64:9, 96:9, 128:9}:
        raise ValueError('Original batch grouping changed')
    meta = dict(sources=sources, source_manifest=str(manifest_path),
                source_manifest_sha256=sha(manifest_path), hf_config=str(config_path),
                hf_config_sha256=sha(config_path), checked_hf_fields=expected,
                windows=135, batch_counts=dict(counts), all_top6_checks_pass=True,
                transformation='Original token sets unchanged; fused F2816 Shared becomes two F1408 tasks.',
                selection_scope='Exhaustive optimization on the requested historical heldout windows; no independent test split.',
                decode_scope='B2/4/8/16 captured decode; B64/96/128 = 16 decode + 48/80/112 continuous prefill tokens.')
    if write:
        save_json(ROOT/'inputs/windows.json.gz', ws)
        save_json(ROOT/'inputs/original_windows.json.gz', original)
        save_json(ROOT/'inputs/manifest.json', meta)
        save_json(ROOT/'inputs/hf_config.json', hf)
        save_csv(ROOT/'inputs/windows.csv', [dict(window=w['id'], batch=w['batch'], dataset=w['dataset'],
                 layer=w['layer'], decode_step=w['step'], active=sum(m>0 for m in w['counts']),
                 sum_Me=sum(w['counts']), counts_64=json.dumps(w['counts']),
                 provenance=json.dumps(w['provenance'], ensure_ascii=False)) for w in ws])
    return ws, meta
