import csv, hashlib, json, math, gzip
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BATCHES=(64,128,256,512,1024,2048)
BUDGETS=(65536,131072,262144,524288,1048576)
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write_csv(p,rows):
    rows=list(rows);p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (gzip.open(p,'wt',newline='') if p.suffix=='.gz' else p.open('w',newline='')) as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
def gm(xs):return math.exp(math.fsum(math.log(x) for x in xs)/len(xs))
def table(h,rs):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rs])
