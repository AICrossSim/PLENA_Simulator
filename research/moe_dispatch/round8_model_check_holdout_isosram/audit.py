"""Final artifact/integrity audit; does not run additional experiments."""
from .study import ROOT, CFG, snapshot, read_json, save_json, sha

def run():
    out=ROOT/'results/study'
    receipt=read_json(out/'VALIDATION.json')
    assert receipt['passed'] and receipt['frozen_file_sha256']==snapshot()
    for name,digest in receipt['source_sha256'].items():
        assert sha(ROOT/name)==digest, f'Experiment source modified since execution: {name}'
    assert receipt['experiment_config_sha256']==sha(ROOT/'config.json')
    for name,digest in receipt['result_sha256'].items():
        if name.endswith('TESTS_FULL.txt') or name.endswith('.md'):continue
        assert sha(ROOT/name)==digest,name
    summary=read_json(out/'summary.json')
    for key,s in summary.items():
        for name in ('2A','2B','2C','3A','3B','3C'):
            assert (out/key/f'table{name}.csv').is_file()
            assert (out/key/f'table{name}.md').is_file()
        # Independently verify no reported Pareto point is dominated by any
        # candidate, using integer hardware bytes/cycle as the coordinate.
        import csv
        candidates=[r for r in csv.DictReader((out/'all_candidate_scores.csv').open())
                    if key==r['model']+'_cap'+r['pn_cap']]
        for r in s['3B']:
            rp=round(r['peak']*1000);rt=r['gm']
            for q in candidates:
                if q['org']!=r['org']:continue
                qp=round(float(q['peak'])*1000);qt=float(q['gm'])
                assert not (qp<=rp and qt<=rt-1e-10 or qp<rp and qt<=rt+1e-10)
    for cap in (256,512):
        s=summary[f'issue_formula_cap{cap}']
        assert all(r['bound']=='memory-bound' for r in s['3C'] if r['sram_cap_TBps']!='8.2')
        assert all(r['count']==50 for r in s['2A'])
    tests=(out/'TESTS_FULL.txt').read_text()
    assert '15 passed' in tests and 'failed' not in tests.lower()
    assert (ROOT/'results.md').exists()
    assert len(list(out.glob('*/table*.csv')))==24
    assert len(list(out.glob('*/table*.md')))==24
    receipt.update(final_artifact_audit_passed=True,tests_passed=15,
        markdown_csv_table_pairs=24,
        source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')},
        result_sha256={str(p.relative_to(ROOT)):sha(p) for p in out.rglob('*')
                       if p.is_file() and p.name!='VALIDATION.json'},
        report_sha256={name:sha(ROOT/name) for name in
                       ('results.md','README.md','diagnosis.md','diagnosis.pdf','evidence.md')})
    save_json(out/'VALIDATION.json',receipt)
    print('PASS: 24 CSV/Markdown pairs, 15 tests, Pareto and HBM checks, frozen rounds unchanged.')

if __name__=='__main__':run()

