"""Sequential, reviewable reproduction of the preregistered evaluation."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
from .campaign import diagnose
from .optimized_grid import run as optimize_grid
from .search import select,final,candidates
from .postprocess import run as postprocess
from .accounting_audit import run as accounting_audit
from ..geometry3d.study import write_json


def snapshot(inputs,out):
    target=out/'inputs';target.mkdir()
    for name in ('development.json','mixed_development.json','heldout.json','mixed_heldout.json'):
        shutil.copy2(inputs/name,target/name)
    source=out/'source';source.mkdir()
    here=Path(__file__).parent
    for directory in (here,here.parent/'geometry3d'):
        dest=source/directory.name;dest.mkdir()
        for file in directory.iterdir():
            if file.suffix in ('.py','.md'):shutil.copy2(file,dest/file.name)
    return target


def manifest(out):
    write_json(out/'MANIFEST.json',{
        str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json'})


def run(inputs,old,out,weights=None,workers=16):
    if out.exists() and any(out.iterdir()):raise ValueError('use an empty new output directory')
    out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','pytest','research/moe_dispatch/regime',
                    'research/moe_dispatch/geometry3d','-q'],check=True)
    frozen_inputs=snapshot(inputs,out)
    diagnose(frozen_inputs,out/'diagnosis',old)
    optimize_grid(frozen_inputs,out/'optimized_grid',workers)
    accounting_audit(out,frozen_inputs)
    grid=out/'optimized_grid/optimized_grid.csv'
    if not candidates(grid):
        write_json(out/'STOPPED_AT_SPACE_GATE.json',{'complete':True,
            'reason':'no development batch/regime has >=10percent protocol-scoped headroom',
            'geometry_search_run':False,'victory':False,'calibration_run':False})
        (out/'REPORT_ZH.md').write_text('开发集所有区间均未达到10%的搜索空间门槛；本轮按规则停止。\n'
            '下限与每batch数据见optimized_grid/optimized_grid.csv。未宣布架构胜出。\n')
        manifest(out);return
    select(frozen_inputs,out/'search',grid,workers)
    final(frozen_inputs,out/'search')
    if weights is not None:
        # Optional numerical probe uses actual model matrices, never substitutes
        # synthetic task accuracy for heldout trained-model validation.
        from .quant_probe import run as quant_probe
        quant_probe(weights,out/'quant')
    else:
        (out/'quant').mkdir()
        write_json(out/'quant/QUANT_QUALITY_STATUS.json',{'trained_model_task_quality_qualified':False,
                   'numerical_probe_run':False,'reason':'no pretrained model directory provided'})
    postprocess(out,frozen_inputs)
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--old-system',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--weights',type=Path)
    p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();run(a.inputs,a.old_system,a.out,a.weights,a.workers)
