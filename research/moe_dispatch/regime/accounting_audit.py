"""Isolate valid-lane accounting from compiler tuning on frozen windows."""
from dataclasses import replace
from pathlib import Path
import argparse,json
from .campaign import evaluate,result_rows,aggregate,load_inputs
from .model import Settings
from .search import settings_for
from ..geometry3d.compute import Core
from ..geometry3d.study import cores_from,write_csv
from ..geometry3d.memory import FabricProfile


def run(root,inputs):
    held=load_inputs(inputs)['heldout'];rows=[]
    points=json.loads((root/'optimized_grid/FROZEN_OPTIMIZED_SINGLE.json').read_text())['points']
    for p in points:
        old=Settings(weight_format=p['weight_format'],allocation='equal',
                     fabric=replace(FabricProfile(),hbm_credits=p['credits']))
        for name,g,s in (
            ('padded_frozen_program',(Core(6,16,128),),old),
            ('valid_same_program',(Core(6,16,128),),replace(old,valid_operand_traffic=True)),
            ('valid_optimized_program',cores_from(p['geometry']),settings_for(p['credits'],p['weight_format'],p))):
            got=evaluate(held,g,s)
            rows+=result_rows(got,s,credits=p['credits'],weight_format=p['weight_format'],
                             accounting_program=name,geometry=p['geometry'])
    out=root/'audit';out.mkdir(exist_ok=True)
    write_csv(out/'ACCOUNTING_CORRECTION_WINDOWS.csv',rows)
    write_csv(out/'ACCOUNTING_CORRECTION_BY_BATCH.csv',aggregate(rows,
        ('credits','weight_format','accounting_program','batch')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    a=p.parse_args();run(a.root,a.inputs)
