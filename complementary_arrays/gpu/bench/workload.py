"""Serving workload of the B200 measurements: the evaluation requests of the routing capture at their captured lengths.

Each request is forced to produce exactly its remaining captured tokens (EOS ignored). B slots start in steady state:
requests drawn with probability proportional to (1 / requests of its source) x captured length, each started at a
uniform position of its generation with the captured prefix appended to the prompt; a freed slot takes the next
request of a fixed source-interleaved order from position 0.
"""

import pickle
from pathlib import Path

import numpy as np


def load(pool, capture, kind, seed=48):
    """(requests, refill order, start weights, rng). pool: the model's pool file (capture/pool.py); capture: directory
    with the reference sampled capture <kind>_ref_sample_chunk*.npz (capture/capture.py with all steps kept)."""
    rows = pickle.loads(Path(pool).read_bytes())["rows"]
    evals = {r["request"]: r for r in rows if r["split"] == "eval"}
    out = {}
    for f in sorted(Path(capture).glob(f"{kind}_ref_sample_chunk*.npz")):
        with np.load(f) as z:
            for i, rq in enumerate(z["request"]):
                if str(rq) in evals:
                    o0, o1 = z["offsets"][i], z["offsets"][i + 1]
                    out[str(rq)] = z["out_tokens"][o0:o1].tolist()
    reqs = [dict(evals[k], out=out[k]) for k in sorted(out)]
    if len(reqs) != len(evals):
        raise ValueError(f"{len(evals) - len(reqs)} evaluation requests have no captured generation")
    rng = np.random.default_rng(seed)
    by_source = {}
    for r in reqs:
        by_source.setdefault(r["source"], []).append(r)
    for v in by_source.values():
        rng.shuffle(v)
    order = [v[i] for i in range(max(map(len, by_source.values()))) for v in by_source.values() if i < len(v)]
    w = np.array([len(r["out"]) / len(by_source[r["source"]]) for r in reqs], float)
    return reqs, order, w / w.sum(), rng
