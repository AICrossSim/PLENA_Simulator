#!/usr/bin/env python3
"""Matched one-factor comparisons inside the completed grid, without extra runs."""
import argparse
import copy
from pathlib import Path
from compare_moe_normal import read_json,require
from run_moe_dma_campaign import save
from run_moe_joint_dse import FIXTURES


def analyze(root):
    require(read_json(root/'status.json')['status']=='passed','grid incomplete')
    ranking=read_json(root/'ranking.json');comparisons=[]
    for category,winner in ranking['winners'].items():
        axes=winner['point']['axes']
        variants=[('reference_all',dict(slots=2,credits=64,layout='raw',threshold=8))]
        variants += [(key+'_reference',{key:value}) for key,value in
                     [('slots',2),('credits',64),('layout','raw'),('threshold',8)]]
        for label,override in variants:
            target=dict(axes,**override)
            matches=[r for r in ranking['ranked'] if r['point']['axes']==target]
            require(len(matches)==1,'missing/ambiguous matched point: '+category+'/'+label)
            reference=matches[0]
            comparisons.append(dict(category=category,label=label,winner=winner['point']['id'],
                reference=reference['point']['id'],changed_axes=override,
                reference_geomean_ps=reference['geomean_ps'],winner_geomean_ps=winner['geomean_ps'],
                speedup_winner_over_reference=reference['geomean_ps']/winner['geomean_ps'],
                per_fixture={f:dict(reference_ps=reference['measurements'][f]['total_ps'],
                    winner_ps=winner['measurements'][f]['total_ps'],
                    reference_hbm_bytes=reference['measurements'][f]['hbm_read_bytes'],
                    winner_hbm_bytes=winner['measurements'][f]['hbm_read_bytes']) for f in FIXTURES}))
    save(root/'matched_effects.json',dict(status='passed',comparisons=comparisons,
        scope='Fixed winning core geometry and SRAM allocation. One factor reset at a time around the joint optimum; factors interact, so these ratios must not be multiplied as independent contributions. reference_all resets slots/credits/layout/threshold together.'))
    lines=['# 联合 DSE：相同核心形状的参数对照','',
        '全部取自已完成的数值搜索，没有用成本公式补造运行点。固定各类胜者的核心形状和 SRAM 分配，把一项参数恢复为参考值；reference_all 同时恢复所有参考值。单项效应存在交互，不能相乘当成独立贡献。','',
        '| 类别 | 恢复的参考项 | 参考点 | 胜者点 | 胜者相对参考加速 |','|---|---|---|---|---:|']
    for row in comparisons:
        lines.append('| {} | {} | {} | {} | {:.4f}× |'.format(row['category'],row['label'],row['reference'],row['winner'],row['speedup_winner_over_reference']))
    lines+=['','参考参数为 2 个预取槽、64 个 DMA credits、原始布局、M=8 调度阈值。HBM、PE、SRAM 总预算和共享功能单元资源保持相同。','']
    (root/'MATCHED_EFFECTS_ZH.md').write_text('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);analyze(p.parse_args().root.resolve())
