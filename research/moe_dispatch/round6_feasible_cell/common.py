import csv,gzip,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
R5=ROOT.parent/'round5_largebatch_compute'
CACHE=Path('/tmp/plena-round6-feasible-cache')
BATCHES=(64,128,256,512,1024,2048,4096)
BUDGETS=(65536,131072,262144,524288,1048576)
BLOCK=262144
SHARED_GU=22*2**20
SHARED_DOWN=11*2**20
W_TBPS=10
SEED=20261016
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 text=json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
 if p.suffix=='.gz':
  with gzip.open(p,'wt') as f:f.write(text)
 else:p.write_text(text)
def read_json(p):
 p=Path(p)
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return json.load(f)
def write_csv(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);rows=list(rows)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (gzip.open(p,'wt',newline='') if p.suffix=='.gz' else p.open('w',newline='')) as f:
  w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def read_csv(p):
 p=Path(p)
 with (gzip.open(p,'rt') if p.suffix=='.gz' else p.open()) as f:return list(csv.DictReader(f))
def gm(xs):return math.exp(math.fsum(math.log(float(x)) for x in xs)/len(xs))
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])
