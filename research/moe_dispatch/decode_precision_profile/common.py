from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

BATCHES = (1, 2, 4, 8, 16, 32, 64, 96, 128, 256)
LAYERS = tuple(range(1, 27))
PROFILE_STEPS = (1, 4, 8, 16, 32)
H, F, SHARED_F, EXPERTS, TOPK = 2048, 1408, 2816, 64, 6

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write_json(path, value):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(p)

def write_csv(path, rows):
    if not rows: return
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def validate_routes(ids, scores):
    import numpy as np
    if ids.shape != scores.shape or ids.shape[-1] != TOPK:
        raise ValueError('route/score shape mismatch')
    if (ids < 0).any() or (ids >= EXPERTS).any(): raise ValueError('invalid expert')
    if not (np.diff(np.sort(ids, axis=-1), axis=-1) > 0).all(): raise ValueError('duplicate Top-6')
    if not np.isfinite(scores).all() or (scores < 0).any(): raise ValueError('invalid score')

def population(ids):
    import numpy as np
    b = len(ids); c = np.bincount(ids.reshape(-1), minlength=64)
    if c.sum() != 6*b: raise ValueError('6B conservation failed')
    cold = (c > 0) & (c <= 2); hot = c > 2
    hc, cc = int(c[hot].sum()), int(c[cold].sum())
    return dict(hot_experts=int(hot.sum()), cold_experts=int(cold.sum()),
        inactive_experts=int((c == 0).sum()), hot_assignments=hc, cold_assignments=cc,
        shared_flops=6*b*H*SHARED_F, hot_flops=6*hc*H*F, cold_flops=6*cc*H*F,
        shared_flops_pct=25., hot_flops_pct=100*hc/(8*b), cold_flops_pct=100*cc/(8*b),
        unique_bf16_weight_MiB=33+16.5*int((c > 0).sum()))
