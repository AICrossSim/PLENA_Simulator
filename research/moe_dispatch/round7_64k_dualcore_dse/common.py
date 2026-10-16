import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
CFG = json.loads((ROOT / 'config.json').read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

def gm(values):
    return math.exp(math.fsum(math.log(float(v)) for v in values) / len(values))

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.gz':
        with path.open('wb') as out:
            with gzip.GzipFile(fileobj=out, mode='wb', mtime=0, filename='') as z:
                z.write((canonical(value)+'\n').encode())
    else:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')

def read_json(path):
    path = Path(path)
    with (gzip.open(path, 'rt') if path.suffix == '.gz' else path.open()) as f:
        return json.load(f)

def save_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot silently write an empty result table')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader(); w.writerows(rows)

def markdown(rows):
    def fmt(v):
        if isinstance(v, float):
            return f'{v:.6f}'
        return str(v).replace('|', '/')
    keys = list(rows[0])
    return '\n'.join(['| '+' | '.join(keys)+' |', '| '+' | '.join(['---']*len(keys))+' |'] +
                     ['| '+' | '.join(fmt(r[k]) for k in keys)+' |' for r in rows])

def table_files(directory, name, rows):
    save_csv(directory / (name+'.csv'), rows)
    (directory / (name+'.md')).write_text(markdown(rows)+'\n')
