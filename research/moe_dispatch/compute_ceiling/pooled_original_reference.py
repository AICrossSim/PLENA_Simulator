"""Give the fixed PLENA single core the same cross-expert issue capability."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .pooled_search import pooled_layer, summarize
from .search import Shape, canonical, csv_out, dump
from .trace_inputs import load_primary_inputs


def run(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    windows = load_primary_inputs('heldout')['heldout']
    cores = (Shape(6, 4, 512),)
    rows = []
    began = time.monotonic()
    for wi, w in enumerate(windows):
        first = pooled_layer(w, cores, cached=False)
        second = pooled_layer(w, cores, cached=False)
        assert canonical(first) == canonical(second)
        rows.append(first)
        if wi % 27 == 0:
            print(canonical({'done': wi + 1, 'total': len(windows),
                             'seconds': time.monotonic() - began}), flush=True)
    dump(out / 'heldout_original_single.json', rows)
    csv_out(out / 'original_single_by_batch.csv', summarize('original_single', rows, windows))
    dump(out / 'ORIGINAL_REFERENCE.json', {
        'configs': {'original_single': [{'pm': 6, 'pn': 4, 'pk': 512}]},
        'fixed_reference_not_searched': True,
        'same_cross_expert_issue_and_fallback_as_searched_hardware': True,
        'repeats': 2,
    })
    dump(out / 'COMPLETION_REFERENCE.json', {
        'status': 'PASS', 'windows': len(windows), 'repeats': 2,
        'all_repeated_results_identical': True,
        'elapsed_seconds': time.monotonic() - began,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
