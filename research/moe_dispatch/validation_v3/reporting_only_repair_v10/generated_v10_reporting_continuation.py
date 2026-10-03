#!/usr/bin/env python3
"""Resume only frozen reporting after the completed v10 timing matrix.

CSV writes absent optional cold-throughput observations as empty cells.  The
frozen reader preserves these cells as strings.  In memory only, remove that
one absent optional field so the unchanged get(field, 0) guards see absence.
No raw CSV, timing, layout, binary, source, threshold or authorization is edited.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

SIM = Path('/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-simulator')
EXPERIMENTS = SIM / 'research/moe_dispatch/experiments/v3'
sys.path.insert(0, str(EXPERIMENTS))
import report
import study
from run_development import verify_stage

EXPECTED_BINARY = 'cb86a53a63bd15acbd027da4941f8fa496e1dfe4eb3f16e71f167ccf2aa057d5'
EXPECTED_SIGNATURE = 'dcfaacff72cd6fb8'
EXPECTED_COUNTS = {
    ('development', 'all'): (4260, 4056, 204),
    ('development', 'dev_comparator'): (84, 84, 0),
    ('mixed_development', 'all'): (2130, 1618, 512),
    ('heldout', 'all'): (38340, 36504, 1836),
    ('mixed_heldout', 'all'): (9585, 7281, 2304),
    ('constructed', 'all'): (6390, 5469, 921),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    study.write(path, value)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def main(root):
    root = Path(root).resolve()
    started = time.monotonic()
    receipt_path = root / 'reporting_only_amendment_v10_receipt.json'
    require(not receipt_path.exists(), 'Preserve the prior amendment receipt; do not overwrite it')
    active = json.loads((root / 'active_result_signature.json').read_text())
    auth = json.loads((root / 'heldout_authorization.json').read_text())
    require(active['binary_sha256'] == EXPECTED_BINARY and
            study.digest(active)[:16] == EXPECTED_SIGNATURE,
            'Wrong frozen timing signature')
    require(auth.get('authorized') is True and auth.get('prereg_commit') and
            auth.get('frozen_signature') == active, 'Missing original heldout authorization')
    identity = json.loads((root / 'generated_v10_completion_identity.json').read_text())
    require(sha(identity['actual_test_binary']) == EXPECTED_BINARY, 'Frozen binary changed')
    source_manifest = root / 'tests/repair_freeze_candidate_v10/source_manifest.json'
    manifest = json.loads(source_manifest.read_text())
    require(sha(source_manifest) == identity['source_manifest_sha256'], 'Source pin changed')
    source_paths = [SIM / relative for relative in manifest['source_hashes']]
    source_paths += list(EXPERIMENTS.glob('*.py'))
    source_pins = {str(path): sha(path) for path in source_paths}
    require(all(sha(SIM / relative) == value for relative, value in manifest['source_hashes'].items()),
            'Frozen engine source changed')
    require(sha(EXPERIMENTS / 'study.py') == active['study_sha256'], 'Frozen study changed')
    csv_pins = {path.name: sha(path) for path in root.glob('*.csv')
                if not path.name.startswith(('table_', 'analytical_', 'claim_'))}
    stages = []
    for (split, suite), counts in EXPECTED_COUNTS.items():
        stage = verify_stage(root, split, suite)
        full = json.loads(Path(stage['path']).read_text())
        require((stage['points'], stage['complete'], stage['excluded']) == counts,
                'Predeclared stage count mismatch: ' + str((split, suite)))
        require(full.get('repeats_equal') is True and stage['signature'] == active,
                'Current stage lacks identical duplicate runs')
        stages.append(stage)
    require(sum(row['complete'] for row in stages) == 55012, 'Full legal population incomplete')
    old_progress = json.loads((root / 'final_pipeline_progress.json').read_text())
    evaluation_stages = [stage for stage in stages if
                         Path(stage['path']).name in ('heldout_all_receipt.json',
                           'mixed_heldout_all_receipt.json', 'constructed_all_receipt.json')]
    require(len(old_progress['receipts']) == len(evaluation_stages), 'Evaluation-stage count changed')
    for original, checked in zip(old_progress['receipts'], evaluation_stages):
        require({k: v for k, v in original.items() if k != 'path'} ==
                {k: v for k, v in checked.items() if k != 'path'} and
                Path(original['path']).resolve() == Path(checked['path']).resolve(),
                'Original evaluation-stage hashes changed')
    evaluation_stages = old_progress['receipts']
    original_coverage = report.coverage(root)
    require(all(int(row['declared_points']) == row['observed_points'] and
                int(row['declared_legal']) == row['complete'] and
                not any(row[key] for key in ('pending_points', 'unsupported', 'failed'))
                for row in original_coverage), 'Declared coverage incomplete')
    require(sum(row['complete'] for row in original_coverage) == 55012,
            'Coverage differs from full matrix')

    history = root / 'history/v10_reporting_failure_before_continuation'
    require(not history.exists(), 'Preserve the original reporting-failure history')
    history.mkdir(parents=True)
    preserved = []
    preserve_paths = [root / 'final_pipeline_v10.log', root / 'v10_root_completion_driver_state.json',
                      root / 'final_pipeline_progress.json']
    preserve_paths += list(root.glob('table_*.csv'))
    preserve_paths += [root / name for name in ('figure_receipt.json', 'analysis_receipt.json',
                       'claim_timing_evidence.json', 'REPORT_V3.md') if (root / name).is_file()]
    for path in preserve_paths:
        if not path.is_file():
            continue
        target = history / path.name
        shutil.copy2(path, target)
        preserved.append(dict(name=path.name, sha256=sha(path), copy_sha256=sha(target)))
    write(history / 'MANIFEST.json', dict(schema='report_failure_preserved_v10_v1', files=preserved))

    original_reader = report.rows
    normalization = []

    def normalized_rows(*args, **kwargs):
        values = original_reader(*args, **kwargs)
        for row in values:
            if row.get('cold_token_rows_per_cycle') == '':
                require(row.get('cold_descriptors') in ('', None) and
                        row.get('cold_token_rows') in ('', None),
                        'Blank throughput must represent an absent cold population')
                before = dict(row)
                del row['cold_token_rows_per_cycle']
                require({**row, 'cold_token_rows_per_cycle': ''} == before,
                        'Normalization changed another field')
        return values

    values = original_reader(root)
    require(len(values) == 55012, 'Current measured row count incomplete')
    for row in values:
        if row.get('cold_token_rows_per_cycle') == '':
            normalization.append(dict(point=row['point'], suite=row['suite'],
                provenance=row['provenance'], original_value='',
                action='remove absent optional metric in memory; unchanged get(..., 0) guard'))
    require(normalization, 'Expected reporting failure fixture not present')
    report.rows = normalized_rows
    values = normalized_rows(root)
    write(root / 'reporting_only_v10_normalized_missing_metrics.json', normalization)
    progress = dict(schema='plena_v3_reporting_only_continuation_v1', status='running',
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(), signature=active,
        normalization_count=len(normalization), normalized_field='cold_token_rows_per_cycle',
        reason='Frozen CSV reader retains absent optional metric as empty string; >0 guard rejects it',
        normalization_semantics='Missing metric stays unavailable, never manufactured as an observed zero',
        frozen_source_pins=source_pins, original_raw_csv_pins=csv_pins,
        stages=stages, original_coverage=original_coverage,
        helper_sha256=sha(__file__), no_core_remeasurement=True,
        no_source_or_threshold_changes=True)
    write(root / 'reporting_only_v10_progress.json', progress)
    report.summary(root, values)
    print('FROZEN SUMMARY COMPLETE', flush=True)
    report.figures(root, values)
    print('FROZEN FIGURES COMPLETE', flush=True)
    declared = report.coverage(root)
    write(root / 'analysis_receipt.json', dict(rows=len(values),
        simulator_scope='analytical discrete-event FFN; not native HBM/RTL/full model',
        duplicated_raw_json_verification='See point receipts',
        input_manifest_sha256=sha(root / 'input_manifest.json')))
    import assess
    assess.assess(root)
    print('FROZEN ASSESS COMPLETE', flush=True)
    require(declared == original_coverage, 'Reporting changed declared coverage')
    final = dict(status='complete', receipts=evaluation_stages, coverage=declared,
        host_elapsed_seconds=old_progress['host_elapsed_seconds'] + time.monotonic() - started,
        prereg_commit=auth['prereg_commit'], frozen_signature=active,
        launcher_sha256=sha(EXPERIMENTS / 'run_final.py'),
        reporting_continuation_helper_sha256=sha(__file__),
        reporting_amendment_receipt=str(receipt_path))
    write(root / 'final_pipeline_receipt.json', final)
    import render_report
    print(render_report.render(root), flush=True)
    require(all(sha(path) == value for path, value in source_pins.items()),
            'Frozen source changed during continuation')
    require(all(sha(root / path) == value for path, value in csv_pins.items()),
            'Original measured CSV changed during continuation')
    require(json.loads((root / 'active_result_signature.json').read_text()) == active and
            json.loads((root / 'heldout_authorization.json').read_text()) == auth,
            'Signature or authorization changed')
    progress.update(status='complete', completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        host_elapsed_seconds=time.monotonic() - started,
        full_matrix_complete=55012, original_raw_runs=110024,
        measured_csv_files_unchanged=len(csv_pins), frozen_sources_unchanged=len(source_pins),
        final_pipeline_receipt_sha256=sha(root / 'final_pipeline_receipt.json'),
        analysis_receipt_sha256=sha(root / 'analysis_receipt.json'),
        claim_timing_evidence_sha256=sha(root / 'claim_timing_evidence.json'),
        normalization_index_sha256=sha(root / 'reporting_only_v10_normalized_missing_metrics.json'),
        original_failure_manifest_sha256=sha(history / 'MANIFEST.json'),
        renderer_source_sha256=sha(EXPERIMENTS / 'render_report.py'),
        report_sha256=sha(root / 'REPORT_V3.md'))
    write(receipt_path, progress)
    print('COMPLETE: reporting-only continuation; all original 55,012x2 results retained', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    main(args.root)
