"""Render requested tables only; keep uncalibrated OS results in the appendix."""
from .study import ROOT, CFG, MODELS, read_json

ORG={'single':'单核','homo':'同构双核','hetero':'异构双核'}
SCHED={'optimal':'精确最优','greedy':'贪心'}
def f(x,n=6):return '—' if x is None else f'{float(x):.{n}f}'
def shape(s):return '无' if s=='none' else s.replace('x','×').replace('+',' + ')
def batch(b):return f'B{b}'+('（decode + prefill 混合窗口）' if int(b)>=64 else '')
def limit(x):return '不限' if x=='unlimited' else x
def table(head,rows):
    return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+
        ['| '+' | '.join(str(x).replace('|',' / ') for x in r)+' |' for r in rows])

def display(name,rows):
    if name=='2A':
        return table(['组织','评估调度','最多选中形状','次数/50','评估GM中位数 (µs)','5%分位 (µs)','95%分位 (µs)'],
            [[ORG[r['org']],SCHED[r['scheduler']],shape(r['most_selected']),r['count'],f(r['median_us']),f(r['p05_us']),f(r['p95_us'])] for r in rows])
    if name=='2B':
        return table(['调度','同构/异构 中位数','5%分位','95%分位','异构更快的划分占比 (%)'],
            [[SCHED[r['scheduler']],f(r['median']),f(r['p05']),f(r['p95']),f(r['hetero_faster_pct'],1)] for r in rows])
    if name=='2C':
        return table(['Batch','n','平均差：同构−异构 (µs)','95% t区间 (µs)','显著性（未多重校正）'],
            [[batch(r['batch']),r['n'],f(r['mean_homo_minus_hetero_us']),f"[{f(r['ci95_low_us'])}, {f(r['ci95_high_us'])}]",
              '异构较快' if r['direction']=='hetero' else '同构较快' if r['direction']=='homo' else '不显著／相等'] for r in rows if r['scheduler']=='optimal'])
    if name=='3A':
        return table(['SRAM上限 (TB/s)','组织','最优形状 PM×PN×PK','峰值读宽 (TB/s)','GM Compute (µs)','同构/异构','单核/异构'],
            [[limit(r['sram_cap_TBps']),ORG[r['org']],shape(r['geometry']),f(r['sram_peak_TBps'],3),f(r['gm_compute_us']),
              f(r['speedup_vs_homo']) if r['org']=='hetero' else '—',f(r['speedup_vs_single']) if r['org']=='hetero' else '—'] for r in rows])
    if name=='3B':
        return table(['组织','Pareto形状 PM×PN×PK','峰值读宽 (TB/s)','GM Compute (µs)'],
            [[ORG[r['org']],shape(r['geometry']),f(r['peak'],3),f(r['gm'])] for r in rows])
    if name=='3C':
        return table(['SRAM上限 (TB/s)','Batch','组织','Compute (µs)','HBM需带宽 (TB/s)','Fetch @4 TB/s (µs)','Roofline (µs)','Bound'],
            [[limit(r['sram_cap_TBps']),batch(r['batch']),ORG[r['org']],f(r['compute_us']),f(r['hbm_req_TBps']),
              f(r['hbm_fetch_us']),f(r['roofline_us']),
              'HBM' if r['bound']=='memory-bound' else '计算' if r['bound']=='compute-bound' else f"混合（计算{r['compute_bound_windows']}/{r['windows']}）"] for r in rows])
    raise ValueError(name)

def build():
    summary=read_json(ROOT/'results/study/summary.json')
    check=read_json(ROOT/'results/study/VALIDATION.json')
    x=summary['issue_formula_cap256']['2B'];y=summary['issue_formula_cap512']['2B']
    text=[
      '第8轮完整结果：模型核对、50次留出验证、同SRAM峰值读宽DSE',
      '主模型采用用户确认的ceil理想发射公式，原OS递推只作未校准敏感性，二者均不是RTL测量。',
      f"50次内部留出中，PN≤256时异构对同构的精确最优调度加速比中位数为{x[0]['median']:.6f}×，贪心为{x[1]['median']:.6f}×；PN≤512时分别为{y[0]['median']:.6f}×和{y[1]['median']:.6f}×，因此约3%的收益在同池留出中保留，但不抵抗调度和列宽变化。",
      '相同SRAM峰值上限下，ceil在16.6和50.0 TB/s档由异构最快，33.3 TB/s档由单核略快，8.2 TB/s档仅单核可行；不限带宽时PN≤256由异构最快、PN≤512由同构最快。',
      '按每层唯一权重读取一次、4 TB/s HBM的Roofline下界，SRAM上限≥16.6 TB/s的所有入选配置在全部135窗均为HBM受限，计算差异不转化为此下界中的延迟差；8.2 TB/s档单核在B64/B96/B128计算受限，但该档无双核可比。',
      'OS敏感性会改变PN≤512及33.3 TB/s档的胜负和部分HBM瓶颈判断，所以结论必须附模型条件，不能宣布物理架构胜出。',
      '所有候选按两个模型逐窗口重新精确求解并各重复两次，贪心也各重复两次，结果逐位一致，第2～7轮文件未变。',
      '**口径与解释边界**',
      '主公式：每GEMM周期=⌈M/PM⌉⌈N/PN⌉⌈K/PK⌉，三个GEMM相加；G=PK=64，1 GHz，overhead=0，统一OS，BF16/FP32，65536个乘法器。两个Shared任务各F=1408，另加所有激活Routed；专家不拆核，不含向量、路由和真实存储时序。异构集合沿用第7轮，允许两个32768 MAC核具有不同几何，不能把所有入选异构都称为算力不等的一大一小。',
      '所有逐batch结果均为窗口算术平均，GM为7个batch算术平均的等权几何平均；不使用135窗直接GM。搜索空间在PN≤256下有单核5、同构5、异构116种；PN≤512下分别为6、6、144种。PM未限制为2的幂，PN以16递增，PK=64，小核预算8K/16K/24K/32K，总MAC严格相等。',
      '第2步：种子20261016，50次按batch分层划分；每次71搜索窗（小batch各14、大batch各5）、64评估窗（小batch各13、大batch各4）。同一份划分用于所有模型、cap和调度；每次只用搜索窗的精确最优时间选型，再用该形状评估最优/贪心两种调度。贪心按任务最短核服务时间降序排队，再选预计完成最早的核；不是测量过的在线DPU。',
      '这些是同一历史样本池的重复留出，划分相互重叠，且窗口可能共享请求/层；不是50份独立新数据集。5%/95%为50次分布的经验分位，不是95%置信区间。2C按要求用窗口配对t区间，未做多重检验校正，相关窗口会削弱其推断力度。2C形状取第7轮全135窗的原选型，故它是描述性配对分析，不作为独立留出证据。',
      '第3步按要求使用全135窗重新选型，不声称其收益经过第2步的留出验证。SRAM约束是整片每周期读宽总和≤给定上限：Σ(PN+PM)×64×2 B×1 GHz；实际读宽可低于预算，不代表各设计实际端口数相等。这是硬件宽度过滤，不是有限端口、容量、布线、面积和能耗的仿真。',
      'HBM需求=每窗口唯一有效BF16权重字节/计算时间，Fetch=字节/4 TB/s；Roofline先逐窗取max(Compute,Fetch)，再按batch平均。Roofline假设完美重叠，是下界而非实测E2E。SRAM内部操作数流量另保留在CSV，不能用HBM唯一字节替代。',
      'SRAM分档结果有离散边界效应：同构的4×128×64双核需要33.792 TB/s，略高于33.3，因而该档只能退回18.432 TB/s的8×64×64双核；16.6档异构实际使用15.360 TB/s、同构12.288 TB/s。表3B完整列出这一搜索空间的Pareto阶梯，不能将42%～45%的档内优势泛化到任意相近带宽。',
      '**诊断表1.2：来源、任务计数、单GEMM和B2对账**',
      (ROOT/'diagnosis.md').read_text().replace('<div style="page-break-before: always;"></div>',''),
    ]
    labels={'2A':'50次划分的配置选择及评估延迟','2B':'评估集异构相对同构加速比',
        '2C':'原选型的逐batch配对t区间','3A':'相同SRAM峰值上限的最优配置',
        '3B':'各组织全部Pareto非支配候选','3C':'各入选配置逐batch计算与HBM Roofline'}
    for model in MODELS:
        if model=='plena_os':
            text.extend(['**附录A：原OS敏感性（不是另一份硬件测量）**',
                '关键结论会改变：无SRAM限制时，OS两种PN上限均约1.0304×最优调度收益，但贪心约0.9546×；ceil在PN≤512已由同构胜出。OS异构在50次选型中，4×192×64+1×256×64出现36次、3×256×64+1×256×64出现14次；ceil主结果各组织均50次选择同一几何。',
                '同SRAM的16.6、50.0 TB/s档“异构最快”方向不变；33.3 TB/s档从ceil的单核略快变成OS的异构快约4.53%。HBM判断也变：OS在部分大batch中计算受限，而ceil除8.2档外均是HBM受限。这源于未校准的8输出上下文/15周期累加依赖递推，不应当作更真实模型自动覆盖ceil。'])
        for cap in CFG['pn_caps']:
            s=summary[f'{model}_cap{cap}']
            for name in labels:
                caption=f"表{name} — {'ceil主模型' if model=='issue_formula' else '原OS敏感性'}；PN≤{cap}；{labels[name]}"
                method=('按搜索集精确最优选型、评估集分别最优/贪心；50次、每次64评估窗' if name in ('2A','2B') else
                        '精确最优分配；原135窗，按batch分组，B64～B128为decode + prefill混合窗口')
                md=caption+'\n\n口径：'+method+'。\n\n'+display(name,s[name])
                text.append(md)
                (ROOT/'results/study'/f'{model}_cap{cap}'/f'table{name}.md').write_text(md+'\n')
            if model=='issue_formula':
                text.append('2C读法：正差为异构更快；PN≤256的B2同构更快，其余batch异构更快；PN≤512的B2～B16同构更快，B64～B128逐窗完全相同。主表2C只列最优调度，贪心配对数据也在同名CSV。')
    text.extend(['**附录B：旧表9形状定义注释（原第7轮表不修改）**',
        '旧形状的第三维G=MLEN/BLEN，是子阵列组数，旧映射BLEN=PM、MLEN=PM×G；本轮第三维G=PK，是一次发射的归约宽度。因此旧4×256×64对应BLEN4、MLEN256、G64，本轮同名形状对应PM4、PN256、PK64；不能仅凭形状名字相同就认为时序契约相同。旧表还有1个合并Shared、Gate/Up融合、runtime控制及GM/AM差别。',
        '带原表9全部列及逐行定义注释的112行数据：[table9_annotated.csv](results/study/table9_annotated.csv)。下表给出每种旧形状的映射；它是口径说明，不新增实验。'])
    text.append(table(['旧形状','旧G定义','旧BLEN','旧MLEN','R7/R8第三维定义'],[
        ['4×256×64','MLEN/BLEN=64','4','256','PK=64'],
        ['2×256×64 + 2×256×64','各核MLEN/BLEN=64','2 / 2','128 / 128','各核PK=64'],
        ['4×128×64 + 1×512×64','各核MLEN/BLEN=64','4 / 1','256 / 64','各核PK=64'],
        ['1×512×64 + 1×512×64','各核MLEN/BLEN=64','1 / 1','64 / 64','各核PK=64']]))
    text.extend(['**可复现性与交付检查**',
        f"重新验证{sum(v['points'] for v in check['model_assignment_checks'].values()):,}个模型×形状×窗口点，精确分配84,240次、贪心84,240次，两次完全一致；50次划分×2模型×2cap全部重跑选择/汇总两遍并逐字节核对。156个不同硬件配置覆盖两个cap，各点重新核对任务归属、载荷、周期和原R7结果。",
        '运行命令见[README.md](README.md)，测试输出见[TESTS_FULL.txt](results/study/TESTS_FULL.txt)，完整校验及源文件哈希见[VALIDATION.json](results/study/VALIDATION.json)。第1步历史校验保留在results/VALIDATION.json；其中stage2/3=false仅指当时诊断脚本的范围，当前完成状态以results/study/VALIDATION.json为准。'])
    (ROOT/'results.md').write_text('\n\n'.join(text)+'\n')
    print(f'Wrote results.md: {len(text)} blocks, all requested tables for both models and both caps.')

if __name__=='__main__':build()

