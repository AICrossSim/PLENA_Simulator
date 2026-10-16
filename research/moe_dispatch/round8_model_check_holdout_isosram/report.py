"""Compact two-page diagnosis; complete raw details are kept in CSV/JSON."""
import csv
import json
import re
from pathlib import Path
from .diagnose import ROOT, REPO, R7, CFG

def read_csv(name):
    with (ROOT/'results'/name).open() as f:return list(csv.DictReader(f))

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])

def filelink(label,relative,line=None):
    return f'[{label}]({REPO/relative}'+(f':{line}' if line else '')+')'

def build():
    blocks=[[],[]]
    def text(page,content):blocks[page].append(('text',content))
    def tab(page,title,headers,rows):blocks[page].append(('table',title,headers,rows))
    text(0,'第8轮／第1步：耗时模型核对（2026-10-16）')
    text(0,'状态：第1步诊断完成；用户已确认第2、3步采用ceil主模型、原OS敏感性，均检查PN≤256/512。时间均为µs（1周期=0.001µs），overhead=0。')
    text(0,'三套都是分析模型，均非RTL校准测量。“已有OS”仅表示复用研究函数，不能视为原生PLENA时序。')
    tab(0,'1.2(a,b) 来源与任务口径；单核串行，无PN搜索；同一135窗',
        ['模型','函数与首次可追溯阶段','G／Shared／Gate-Up口径'],[
        ['旧表', 'shared/timing.phase → dedicated/model.job_cycles → scaling_g64/runtime.schedule',
         'G=MLEN/BLEN；1×F2816；Gate/Up合并phase'],
        ['第7轮OS','round7/model.gemm → round5/model.projection；同递推见第2/4轮',
         'G=PK；2×F1408；Gate/Up各自排空'],
        ['ceil','round7/model.gemm(issue_formula)；本实现第7轮',
         'G=PK；2×F1408；三GEMM分别计数']])
    text(0,'路径前缀均为research/moe_dispatch/；上表缩写shared=plena65536_shared、dedicated=plena65536_dedicated、'
         'round7=round7_64k_dualcore_dse、round5=round5_largebatch_compute。旧phase最早可见于未编号Shared-only实验；文件未提交，无法证实轮次/创建时间。'
         'R5函数首次提交b095a7de；8记录/归约延迟假设前身见geometry3d(1fcc764a)、R2(2a68dfdb)。全部路径/行号/哈希见model_sources.json。'
         '135窗逐专家核对：Routed集合完全相同，只有Shared任务计数改变。')
    text(0,'1.2(c) 单GEMM拆解：三表均为点测，无调度/窗口；PN最大512（含主cap256外参照）。'
         '每格=C/R/A/O→合计，分别为纯ceil乘加／归约排空／累加依赖／其他；属于公式记账，不是实测停顿。'
         'G1=(M,K,N)=(1,2048,1408)，G2=(4,2048,1408)，G3=(1,1408,2048)。')
    rows=read_csv('gemm_breakdown.csv')
    for model,label in [('legacy_phase','旧phase'),('round7_os','第7轮OS'),('ceil','ceil')]:
        tabrows=[]
        for shape in CFG['shapes']:
            name='x'.join(map(str,shape));cells=[]
            for m,k,n in CFG['gemms_M_K_N']:
                r=next(r for r in rows if r['model']==model and r['shape']==name and int(r['M'])==m and int(r['K'])==k and int(r['N'])==n)
                vals=[float(r[x]) for x in ['pure_us','reduction_drain_us','accumulation_dependency_us','other_us','total_us']]
                cells.append('/'.join(f'{v:.3f}' for v in vals[:4])+' → '+f'{vals[4]:.3f}')
            tabrows.append([name]+cells)
        tab(0,label+'；单位µs',['PM×PN×G','G1：C/R/A/O→合计','G2：C/R/A/O→合计','G3：C/R/A/O→合计'],tabrows)
    text(0,'1.2(d) 公式：令 a=⌈M/PM⌉，b=⌈N/PN⌉，k=⌈K/G⌉，C=abk。以下公式单位为周期。')
    text(0,'旧phase：BLEN=PM，MLEN=PM·G；T=ab·PM·⌈K/(PM·G)⌉ + PM+11+log₂G。'
         'R=log₂G，A=0，O=ab[PM⌈K/(PM·G)⌉−k]+PM+11。'
         'Gate/Up合并时records翻倍但只收一次排空，所以整专家不能简单加三个独立phase。')
    text(0,'第7轮OS：L=3+2log₂G=15，按N分组q=min(8,剩余列tile)。'
         'T=aΣq{qk+(k−1)max(0,L−q)+(L−1)}；A=aΣq(k−1)max(0,L−q)，'
         'R=2log₂G·a·组数，O=2a·组数。R/O只是L−1的代数拆分；源函数没有独立归约/累加计数器。ceil：T=C，R=A=O=0。')

    text(1,'第8轮／第1步：证据、逐任务对账与主模型建议')
    tab(1,'1.2(d) 硬件依据核验；静态源码审查，无额外硬件实验',
        ['依据','能支持什么','不能支持什么'],[
        ['PLENA §III-B／图5、6','OS部分和驻PE；子阵列间M_SUM；连续tile流水化','没有8上下文、每K段15周期依赖的依据'],
        ['customISA_lib.json:2–5','M_MM独立2·BLEN+9，流水间隔BLEN','没有旧模型完整M_SUM/shadow-bank重叠依据'],
        ['fp_sum_across_sa／fp_default_pe RTL','跨阵列加法树及PE局部累加存在','没有本实验8记录、2log₂G+3的校准依据'],
        ['Rust matrix_machine.rs:143/512；matrix_service.rs:1/120',
         '原路径另收overhead+MLEN；候选service可配MAC/tree延迟','均非R7函数；候选service自身声明未综合']])
    text(1,'为什么宽核变慢：N=1408时，PN256有6个独立输出tile，PN512只有3个。'
         'R7模型强制回访同一输出间隔≥15周期；因此G1的纯乘加从0.192降为0.096µs，'
         '依赖等待却从0.279增为0.372µs，合计只从0.485降到0.482µs。'
         '这是有限上下文和发射顺序假设的后果，不是单行核必然物理缓慢；8记录本身也不等于必须按小组排空。')
    text(1,'1.2(e) 窗口 heldout_bfcl_b2_l2_s3：4×256×64（PN≤256），单核顺序执行；'
         '2请求同一decode步3、层2。12个Routed均Me=1、Shared Me=2；路由总数12=6B。')
    taskrows=read_csv('b2_case_tasks.csv')
    rows2=[]
    for r in taskrows:
        if r['task'].startswith('Shared'):
            rows2.extend([['Shared A，F1408','2','合并共1.098','1.307','0.560'],
                          ['Shared B，F1408','2','已含上格','1.307','0.560']])
        else:rows2.append([r['task'],r['M']]+[f'{float(r[k]):.3f}' for k in ['legacy_us','r7_os_us','ceil_us']])
    rows2.extend([['专家合计','—','8.514','18.298','7.840'],['控制开销／窗口总计','—','0.005／8.519','0／18.298','0／7.840']])
    tab(1,'逐任务时长（µs）；旧Shared为一个F2816任务，不虚构拆分归属',
        ['任务','Me','旧表模型','第7轮OS','ceil'],rows2)
    text(1,'任务核对：普通专家旧模型=0.405(Gate+Up)+0.213(Down)=0.618；'
         'R7 OS=0.485+0.485+0.337=1.307；ceil=0.192+0.192+0.176=0.560µs。'
         '旧合并Shared=0.725+0.373=1.098µs；新增控制只在旧重放收0.005µs。')
    tab(1,'27个B2窗口；形状4×256×64／PN≤256／单核；统一算术平均与原表对照',
        ['模型','原表采用统计','纯乘加','依赖等待','其余算术/控制','总计（AM）'],[
        ['旧表','GM=7.605794','—','单独收费0','全部服务+控制','7.672111'],
        ['第7轮OS','AM=16.506926','7.072593','8.903889','0.530444','16.506926'],
        ['ceil','AM=7.072593','7.072593','0','0','7.072593']])
    text(1,'B2活跃数直方图 A:n={7:2,8:2,9:1,10:4,11:8,12:10}，E[A]=10.629630。'
         '旧每窗总计=(618A+1098+5)/1000；ceil=(560A+1120)/1000；R7 OS=(1307A+2614)/1000。'
         '换成合并Shared，R7 OS的AM仅降0.009µs、ceil降0.064µs；Shared拆分不是2倍差距主因。')
    text(1,'第4项只读检查：纯decode trace存在：final_artifacts/moe_regime_20261005_v1/payload/inputs/heldout.json，'
         '108窗，B2/4/8/16各27，层2/13/26、步3/7/11，BFCL/GPQA/SWE，每窗来自B个不同请求。'
         '另27窗在mixed_heldout.json：B64/96/128各9，层相同、decode步7，16个decode token＋同一请求连续prefill前缀48/80/112。'
         '后者必须标注decode+prefill混合窗口；18窗还带capture_prefix_smoke=true。逐窗来源保留原始数据，不重采样。')
    text(1,'建议：纯计算几何研究暂以ceil为主模型（理想发射下界，非真实PLENA延迟）；原8记录/15周期模型只保留敏感性。'
         '旧模型虽更接近论文拓扑，G语义、合法形状、融合及排空假设不同，也未完成BF16校准，不能直接替代。'
         '若要求真实硬件结论，需要先确定并校准BF16数据通路、累加反馈和上下文资源。用户已确认ceil主模型、原OS敏感性及PN≤256/512；完整后续结果见results.md。')

    md=[]
    for i,page in enumerate(blocks):
        if i:md.append('<div style="page-break-before: always;"></div>')
        for b in page:
            if b[0]=='text':md.append(b[1])
            else:md.extend([b[1],table(b[2],b[3])])
    md.append('完整源码路径、行号与证据： [evidence.md](evidence.md)；45项逐GEMM计数： [CSV](results/gemm_breakdown.csv)；'
              '27窗逐项数据：[CSV](results/b2_windows.csv)。论文：[PLENA §III-B](https://arxiv.org/html/2509.09505v3#S3.SS2)。')
    (ROOT/'diagnosis.md').write_text('\n\n'.join(md)+'\n')
    render_pdf(blocks)

def render_pdf(blocks):
    """Two landscape A4 pages with readable, measured wrapped table/text."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.font_manager import FontProperties, fontManager
    fontManager.addfont('/usr/share/fonts/google-droid/DroidSansFallback.ttf')
    font=FontProperties(family=['DejaVu Sans','Droid Sans Fallback'],size=8)
    # Width estimation uses East Asian full/half-width classes; conservative.
    import unicodedata
    def wrap(s,width):
        lines=[];line='';units=0
        for ch in s:
            w=1 if unicodedata.east_asian_width(ch) in ('F','W') else .58
            if ch=='\n' or (units+w>width and line):
                lines.append(line);line='';units=0
            if ch!='\n':line+=ch;units+=w
        if line:lines.append(line)
        return '\n'.join(lines)
    W,H=842,595
    with PdfPages(ROOT/'diagnosis.pdf',metadata={'Title':'Round8 Step1 Timing Diagnosis','CreationDate':None,'ModDate':None}) as pdf:
        for page_i,page in enumerate(blocks):
            fig=plt.figure(figsize=(W/72,H/72));ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,W);ax.set_ylim(0,H);ax.axis('off')
            # Pre-layout all blocks, then uniformly fit within the page.
            layout=[];height=0
            for b in page:
                if b[0]=='text':
                    content=wrap(b[1],(W-44)/8)
                    h=(content.count('\n')+1)*10+5
                    layout.append(('text',content,h));height+=h
                else:
                    _,title,heads,rows=b;cnt=len(heads)
                    weights=([.12,.293,.293,.294] if cnt==4 else [.2]+[.8/(cnt-1)]*(cnt-1))
                    if cnt==3:weights=[.20,.43,.37]
                    width=[(W-44)*x for x in weights]
                    tr=[]
                    for row in [heads]+rows:
                        rr=[wrap(str(s),max(4,(w-7)/7.3)) for s,w in zip(row,width)]
                        rh=max(x.count('\n')+1 for x in rr)*9+5
                        tr.append((rr,rh))
                    h=14+sum(rh for _,rh in tr)+5
                    layout.append(('table',title,width,tr,h));height+=h
            scale=min(1,(H-42)/height)
            assert scale>.68, f'Page {page_i+1} too dense: {height}'
            y=H-18
            for b in layout:
                if b[0]=='text':
                    ax.text(22,y,b[1],fontproperties=font,fontsize=8*scale,va='top',linespacing=1.25)
                    y-=b[2]*scale
                else:
                    _,title,width,tr,h=b
                    ax.text(22,y,title,fontproperties=font,fontsize=8*scale,va='top',color='#174363');y-=14*scale
                    for ri,(rr,rh) in enumerate(tr):
                        x=22
                        for s,w in zip(rr,width):
                            ax.add_patch(plt.Rectangle((x,y-rh*scale),w,rh*scale,fc='#eaf0f4' if ri==0 else 'white',ec='#ccd6dd',lw=.3))
                            ax.text(x+3,y-2*scale,s,fontproperties=font,fontsize=7.3*scale,va='top',linespacing=1.23)
                            x+=w
                        y-=rh*scale
                    y-=5*scale
            assert y>=18
            ax.text(W-22,10,f'{page_i+1} / 2',fontproperties=font,fontsize=8,ha='right')
            pdf.savefig(fig);plt.close(fig)

if __name__=='__main__':build()
