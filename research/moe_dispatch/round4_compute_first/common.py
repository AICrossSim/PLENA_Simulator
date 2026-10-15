import csv, gzip, hashlib, json, math
from pathlib import Path
from research.moe_dispatch.round2.common import inputs as _historical_inputs

ROOT=Path(__file__).resolve().parent
def inputs():
    snapshot=ROOT/'inputs_snapshot.json.gz'
    if not snapshot.exists():return _historical_inputs()
    raw=gzip.decompress(snapshot.read_bytes())
    manifest=json.loads((ROOT/'INPUT_MANIFEST.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==manifest['uncompressed_sha256']
    return json.loads(raw)
def canonical(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def write_json(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n')
def write_csv(p,rs):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); rs=list(rs)
    fields=list(dict.fromkeys(k for r in rs for k in r))
    with p.open('w',newline='') as f:
        wr=csv.DictWriter(f,fields,lineterminator='\n'); wr.writeheader(); wr.writerows(rs)
def read_csv(p):
    with Path(p).open() as f: return list(csv.DictReader(f))
def gm(vs):
    vs=list(vs)
    if not vs: return 0.0
    if min(vs)==0: return 0.0
    return math.exp(math.fsum(math.log(float(v)) for v in vs)/len(vs))
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])
