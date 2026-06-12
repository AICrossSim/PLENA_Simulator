#!/usr/bin/env python3
"""
PLENA 细致能耗估算 — 数据用 George,结构用真实 .isa。

设计:复刻 George analytic 的两条腿(cycle 公式 + 功耗系数),但把
George "用模型维度猜指令数" 换成 "读真实 .isa 数指令(含 cloop 展开)"。
比 George 精确在:
  - 指令条数真实(不是维度公式猜)
  - 按 opcode 前缀分桶 → 能拆 matmul/vector/scalar 各占多少(George 糊成一坨)
  - MCU 活跃时间 = 真实 M_* 指令 cycle(George 按段近似)
  - memory/compute 靠 opcode 前缀区分(H_ vs M_/V_/S_),不需要两套维度公式

数据源(全 George):
  - cycle 公式: analytic_models/performance/customISA_lib.json (新版, M_BMM=(MLEN//BLEN)²×BLEN)
  - 功耗系数: analytic_models/power_model/plena_power_model.py (origin/experiment)
口径(全 George):
  - total_time = max(compute_time, memory_time)  [重叠]
  - 能耗 = 组件功耗 × 组件活跃时间;systolic util prefill=100%

用法:
  python tools/power/plena_isa_energy.py <ir_dir> [--mcu-power-w 100] [--customisa path]
  例: python tools/power/plena_isa_energy.py managerbuild_DSB/ir
"""
import argparse, re, json, math, glob, os, sys

# ---------------------------------------------------------------------------
# 硬件配置 — 从 plena_settings.toml [BEHAVIOR] 实读, 不硬编码 (BLEN 等填错会
# 让 matmul cycle 全错: M_MM=BLEN, M_BTMM=BLEN×(MLEN//BLEN)²)。
# ---------------------------------------------------------------------------
def load_hw(toml_path="plena_settings.toml"):
    import toml as _toml
    d = _toml.load(toml_path)
    b = d["BEHAVIOR"]; cfg = b["CONFIG"]; lat = b["LATENCY"]
    dc = int(cfg.get("DC_EN", {}).get("value", 1))
    col = "dc_lib_en" if dc else "dc_lib_dis"
    def L(name, default=1):
        return int(lat[name][col]) if name in lat else default
    return dict(
        MLEN=int(cfg["MLEN"]["value"]), HLEN=int(cfg["HLEN"]["value"]),
        BLEN=int(cfg["BLEN"]["value"]), VLEN=int(cfg["VLEN"]["value"]),
        VECTOR_BASIC_CYCLES=L("VECTOR_BASIC_CYCLES"),
        VECTOR_ADD_CYCLES=L("VECTOR_ADD_CYCLES"),
        VECTOR_MUL_CYCLES=L("VECTOR_MUL_CYCLES"),
        VECTOR_EXP_CYCLES=L("VECTOR_EXP_CYCLES"),
        VECTOR_SUM_CYCLES=L("VECTOR_SUM_CYCLES"),
        VECTOR_MAX_CYCLES=L("VECTOR_MAX_CYCLES"),
        VECTOR_RECI_CYCLES=L("VECTOR_RECI_CYCLES"),
        SCALAR_FP_BASIC_CYCLES=L("SCALAR_FP_BASIC_CYCLES"),
        SCALAR_FP_EXP_CYCLES=L("SCALAR_FP_EXP_CYCLES"),
        SCALAR_FP_RECI_CYCLES=L("SCALAR_FP_RECI_CYCLES"),
        SCALAR_FP_SQRT_CYCLES=L("SCALAR_FP_SQRT_CYCLES"),
        SCALAR_INT_BASIC_CYCLES=L("SCALAR_INT_BASIC_CYCLES"),
    )

HW = load_hw()
FREQ_HZ = 1e9
NS = 1e-9

# ---------------------------------------------------------------------------
# 路 B — memory 解析模型 (George "bytes/带宽" 口径, 不从 emulator report 抓)
# 数据源: plena_settings.SSB_build_1.toml [ANALYTIC]
#   每条 H_ 搬运元素 = dim × amount;  字节 = 元素 × elem_bits/8 × mx_factor
#   mx_factor = 1 + scale_bits/(elem_bits×block)  (MX 的 scale 额外开销)
#   带宽口径 (George): M 端口=MLEN 字节/cycle, V 端口=VLEN 字节/cycle
#                      (HBM_WIDTH 只用于带宽利用率, 不用于算时间; 同 sys_model._bytes_to_us)
# ---------------------------------------------------------------------------
MEM = dict(
    M_PREFETCH_AMOUNT=64, V_PREFETCH_AMOUNT=4, V_WRITEBACK_AMOUNT=4,
    # HBM_*_TYPE: MX-E4M3 = 8-bit elem (1 sign + 4 exp + 3 mant) + E8M0 8-bit
    # scale per block of 8. (Earlier this said ELEM_BITS=4, which only counted
    # the exponent and halved the HBM byte count — fixed to 8.)
    ELEM_BITS=8, SCALE_BITS=8, MX_BLOCK=8,
)
# HBM datatype selector: forward inference uses MX-E4M3 (8-bit elem + E8M0
# scale); backward (training) uses plain fp16 for accuracy — 16-bit elements,
# no per-block scale (mx_factor = 1).
HBM_DTYPE = "mx_e4m3"           # precision tier; see _PRECISION_TABLE below
def _elem_bits():
    # element bit-width per the precision tier (drives HBM bytes / memory time)
    return _PRECISION_TABLE.get(HBM_DTYPE, (1.0, 1.0, 16, False))[2]
def _mx_factor():
    # MX block-scale overhead only for the MX tiers; plain fp* carry no scale.
    has_scale = _PRECISION_TABLE.get(HBM_DTYPE, (1.0, 1.0, 16, False))[3]
    if not has_scale:
        return 1.0
    return 1.0 + MEM['SCALE_BITS'] / (MEM['ELEM_BITS'] * MEM['MX_BLOCK'])
def h_instr_bytes(op):
    """一条 H_ 指令搬运的 HBM 字节数 (含 MX scale 开销, 或 fp16 无 scale)。"""
    f = _mx_factor()
    eb = _elem_bits() / 8.0
    if op == 'H_PREFETCH_M': return HW['MLEN'] * MEM['M_PREFETCH_AMOUNT'] * eb * f
    if op == 'H_PREFETCH_V': return HW['VLEN'] * MEM['V_PREFETCH_AMOUNT'] * eb * f
    if op == 'H_STORE_V':    return HW['VLEN'] * MEM['V_WRITEBACK_AMOUNT'] * eb * f
    return 0.0
def h_port_bw(op):
    """George 端口带宽 (字节/cycle): M 走矩阵端口 MLEN, V 走向量端口 VLEN。"""
    return HW['MLEN'] if op == 'H_PREFETCH_M' else HW['VLEN']
def memory_us_pathB(hcnt):
    """memory_time (us) = Σ_op  条数×每条字节 / (端口带宽 × freq/1e6)。"""
    us = 0.0
    for op, n in hcnt.items():
        if n <= 0: continue
        bw_per_us = h_port_bw(op) * FREQ_HZ / 1e6
        us += h_instr_bytes(op) * n / bw_per_us
    return us

# ---------------------------------------------------------------------------
# 功耗系数 — 新标准 (bottom-up, GPU 对齐口径: 存储 BF16 / 累加 FP32, 7nm/1GHz)。
# 锚点 (RTL 综合 / 用户给定): 单个 BF16→FP32 MAC = 1 mW @ 7nm 1GHz (= 1 pJ/op)。
#   交叉验证: IBM ISSCC2021 7nm FP8 MAC ≈0.57pJ(含整芯), BF16 更宽→~1pJ 合理;
#             Horowitz45nm FP32 MAC 4.6pJ → 7nm ~0.5pJ. 1mW 落在区间上沿(保守)。
#   FLOP 口径: 1 MAC = 2 FLOP (NVIDIA 惯例), 1mW/MAC = 1pJ/MAC @1GHz.
# datatype 只进功耗腿; latency(cycle)与 type 无关 (流水线级数固定), 不受影响。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 精度档位 (datatype) → 功耗倍率 + 存储位宽。
#   基准锚点对应 ~FP8 (8-bit elem) 的低精度算子功耗 = 1×。
#   **基准 = FP16** (MAC=1mW, vector lane=0.5mW, SRAM=0.055pJ/bit 都是 FP16 值)。
#   两条腿分开缩放 (不是单一倍率):
#     - MCU(matmul): 看**累加精度**。FP8×FP8→FP16 累加, MAC 阵列仍是 FP16 级 →
#       FP8 的 MCU **不砍半** (=1×, 同 FP16)。只有 FP32 累加才 2×。
#     - Vector / SRAM / 带宽: 看**元素位宽** (逐元素 ALU / 存储行宽 / HBM 字节)。
#       FP8 = 8-bit → 0.5×; FP16 = 1×; FP32 = 2×。
# ---------------------------------------------------------------------------
_PRECISION_TABLE = {
    # dtype          mcu×   vec×   elem_bits   has_mx_scale
    'mx_e4m3':      (1.0,   0.5,    8,          True),   # 推理: 8-bit elem(+E8M0 scale), 累加 FP16
    'fp8':          (1.0,   0.5,    8,          False),  # 纯 FP8, 累加 FP16
    'fp16':         (1.0,   1.0,    16,         False),  # 基准
    'bf16':         (1.0,   1.0,    16,         False),
    'fp32':         (2.0,   2.0,    32,         False),  # 全精度 (训练反向用)
}
def _mcu_scale():   # matmul 阵列功耗倍率 (看累加精度)
    return _PRECISION_TABLE.get(HBM_DTYPE, (1.0, 1.0, 16, False))[0]
def _vec_scale():   # vector / SRAM 功耗倍率 (看元素位宽)
    return _PRECISION_TABLE.get(HBM_DTYPE, (1.0, 1.0, 16, False))[1]

MAC_POWER_MW   = 1.0     # 单个 FP16 MAC 满载动态功耗 (mW @7nm/1GHz); ×_mcu_scale()
# Vector lane: elementwise ALU(add/mul/exp/reci) + reduce 树(adder/max). RTL:
#   src/vector_machine/rtl/{fp_vector_element_alu, fp_reduction_compute_unit}.sv
#   每 lane 一个 ALU(BF16); reduce 是 log2(VLEN) 层二叉树, 等效 ~1 add/lane.
#   单个 BF16 FP 算子 ≈ 0.5mW (从 1mW MAC = 乘+加 两算子拆). 一条 vector 指令
#   只激活一类单元 → 典型 elementwise(mul/add)/reduce(sum/max) ≈ 0.5mW/lane;
#   EXP ~3× (~1.5mW), RECI ~2× (~1mW). 用典型值, 区间在报告标注。
VEC_LANE_POWER_MW = 0.5  # 每 lane 满载典型 (elementwise/reduce, BF16)

def mcu_power_w(BLEN, MLEN):                 # PE(MAC)总数 = BLEN×MLEN (RTL: matrix_machine.sv)
    return MAC_POWER_MW * 1e-3 * BLEN * MLEN * _mcu_scale()    # 看累加精度
def vec_power_w(VLEN, act=1.0):              # VLEN lane × 每 lane 典型; act 已含在锚点, 默认1
    return VEC_LANE_POWER_MW * 1e-3 * VLEN * act * _vec_scale()  # 看元素位宽

# --- 旧 George 拟合 (origin/experiment power_model.py), 用 --power-model george 调回 ---
def mcu_power_w_george(M_batch, K_mlen):
    return 17.044e-3 * M_batch * K_mlen
def vec_power_w_george(VLEN, act=3.0):
    b = -1.7746135753e-07*VLEN**2 + 5.4279213710e-04*VLEN - 4.9166666667e-05
    return max(0.0, b)*act

# 片上 SRAM — 新标准 bottom-up (Horowitz/MIT 锚点, 45nm→7nm 换算)。
#   Horowitz/MIT 45nm: 32KB SRAM 读 32b = 10pJ → ~0.31 pJ/bit @45nm.
#   45nm→7nm 动态能耗 ~5-6× 降 → ~0.055 pJ/bit @7nm (与 7nm 实测 0.7pJ/bit 含互连同量级).
#   满载功耗 = pJ/bit × 每拍读 bit 数 × freq. 每拍读一整行:
#     Matrix SRAM 行 = MLEN × 16b (BF16 存储口径);  Vector SRAM 行 = VLEN × 16b.
SRAM_PJ_PER_BIT = 0.055   # @7nm 1GHz, Horowitz 32KB→7nm 换算
STORE_BITS      = 16      # FP16 基准存储行宽 (1×); FP8 半之, FP32 倍之, 由 _vec_scale()
def msram_power_w(MLEN, depth=None):        # depth 兼容旧签名, bottom-up 不用
    return SRAM_PJ_PER_BIT * 1e-12 * (MLEN * STORE_BITS) * FREQ_HZ * _vec_scale()  # W
def vsram_power_w(VLEN, depth=None):
    return SRAM_PJ_PER_BIT * 1e-12 * (VLEN * STORE_BITS) * FREQ_HZ * _vec_scale()  # W

# --- 旧 George SRAM 拟合, 用 --power-model george 调回 ---
def msram_power_w_george(MLEN, depth):
    b = 1.9583044211e-06*MLEN**2 + 2.4835249306e-03*MLEN + 5.2254044409e-04
    return b*(depth/128.0)
def vsram_power_w_george(VLEN, depth):
    b = -3.1305759318e-08*VLEN**2 + 3.4435611618e-04*VLEN + 2.4520172086e-01
    return b*(depth/128.0)

# ---------------------------------------------------------------------------
# 指令 → 组件桶 (按 opcode 前缀)
# ---------------------------------------------------------------------------
# load/store 类: 数据搬运/寄存器进出, 不是"计算", George 口径一律不算。
LDST_OPS = {'S_LD_FP','S_ST_FP','S_LD_INT','S_ST_INT','S_MAP_V_FP','S_MAP_FP_V'}
def bucket(op):
    if op.startswith('M_'): return 'matmul'    # MCU + Matrix SRAM
    if op.startswith('V_') or op == 'C_HADAMARD_TRANSFORM': return 'vector'  # Vector Unit + Vector SRAM
    if op in LDST_OPS: return 'ldst'           # 标量 load/store, 搬运, 不计入 compute
    if op.startswith('S_'):
        # 粗略估计口径: 整数标量(地址/计数/移位)是辅助开销, 不计入 compute_time;
        # 只有真正的 FP 标量算子(EXP/RECI/SQRT/ADD_FP...)算。
        return 'scalar_int' if op.endswith('_INT') else 'scalar'
    if op.startswith('H_'): return 'memory'    # HBM DMA
    return 'control'                           # C_* loop/setreg

# DSB 逐 kernel transmission (ns) — emulator 实测 HBM 搬运时间 (KERNEL_REPORT_DSB §1.3)。
# George memory_time 口径本是 bytes/带宽 静态除法;此处用 emulator 实测 transmission 替代(更准)。
DSB_TRANSMISSION_NS = {
    'I_norm1':451689,'I_mod1':451727,'I_linq':599839,'I_link':599839,'I_linv':599839,
    'I_qknq':377293,'I_qknk':377293,'I_ropeq':525938,'I_ropek':525938,'T_norm1':451669,
    'T_mod1':451727,'T_linq':599839,'T_link':599839,'T_linv':599839,'T_qknq':377293,
    'T_qknk':377293,'T_ropeq':525938,'T_ropek':525938,'concat_q':2423040,'concat_k':2423040,
    'concat_v':2423040,'flash':1790101,'split_attn':2422955,'I_proj':599839,'I_res1':451727,
    'I_norm2':451669,'I_mod2':451727,'I_mlpin':599839,'I_gelu':302849,'I_mlpout':599839,
    'I_res2':451727,'T_proj':599839,'T_res1':451727,'T_norm2':451669,'T_mod2':451727,
    'T_mlpin':599839,'T_gelu':302849,'T_mlpout':599839,'T_res2':451727,
}

# SSB (single_stream_block) 逐 kernel transmission (ns) — emulator 实测
# (KERNEL_REPORT_SSB §1.3, MLEN=1024)。键名 = managerbuild_SSB_2/ir/<kernel>。
SSB_TRANSMISSION_NS = {
    'layernorm':903415,'modulate':903443,'linear_q':1199527,'linear_k':1199527,
    'linear_v':1199527,'linear_mlp':1199527,'qknorm_q':754520,'qknorm_k':754520,
    'rope_q':1052001,'rope_k':1052001,'gelu':605691,'flash_attention':1790101,
    'concat':1163754,'linear2':1841987,'residual_gate':903567,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ir_dir', help='managerbuild_*/ir 目录')
    ap.add_argument('--mcu-power-w', type=float, default=None,
                    help='直接指定 MCU 满负荷功耗(W),覆盖 power_model 外推值')
    ap.add_argument('--customisa', default='analytic_models/performance/customISA_lib.json',
                    help='George cycle 公式 json (performance 新版)')
    ap.add_argument('--settings', default='plena_settings.toml',
                    help='HW 几何+cycle 常量来源 toml (读 [BEHAVIOR])。'
                         'managerbuild→plena_settings.toml(BLEN=32); '
                         'managerbuild_SSB_2→plena_settings.SSB_build_1.toml(BLEN=128)')
    ap.add_argument('--power-model', choices=['bottomup','george'], default='bottomup',
                    help='功耗系数: bottomup=新标准(BF16→FP32, 1mW/MAC + 0.5mW/lane, 默认); '
                         'george=旧 power_model.py 拟合(17mW/PE + 二次 vec×act)')
    ap.add_argument('--vec-activity', type=float, default=None,
                    help='vector activity 系数; bottomup 默认1(已含锚点), george 默认3')
    ap.add_argument('--simt-threads', type=int, default=1,
                    help='SIMT 每 warp 线程数 (默认1=纯SIMD关闭). 设16: vector+scalar_fp '
                         'cycle ÷16 (16线程并行), Vector Unit 功耗 ×16 (SRAM不变); '
                         'matmul 不受益. 能耗不变(cyc÷16×功耗×16), 时间缩短16×')
    ap.add_argument('--device-count', type=int, default=1,
                    help='并行 device 数 (理想线性). 设 N: 整链时间÷N, 总能量不变, '
                         '平均功率×N. 在 SIMT 之上的又一层 (SIMT 缩 vector 时间, device '
                         '缩整链). 默认1.')
    ap.add_argument('--mem-source', choices=['analytic','report'], default='analytic',
                    help='memory_time 来源: analytic=路B解析(George bytes/带宽, 默认), '
                         'report=emulator 实测 transmission 表(旧口径)')
    ap.add_argument('--trans', choices=['auto','dsb','ssb','none'], default='auto',
                    help='--mem-source report 时的实测 transmission 表 (auto 按 ir_dir 名推断)')
    ap.add_argument('--hbm-dtype', '--precision',
                    choices=['mx_e4m3','fp8','fp16','bf16','fp32'], default='mx_e4m3',
                    help='精度档 (datatype): 同时驱动 (a) HBM 字节/memory_time, '
                         '(b) 元器件功耗倍率. mx_e4m3/fp8=低精度 1× (8bit, 默认推理); '
                         'fp16/bf16=高精度 2× (16bit, 训练/高精度推理); fp32=4× (32bit). '
                         'mx_e4m3 额外含 E8M0 block-scale 开销.')
    a = ap.parse_args()
    globals()['HBM_DTYPE'] = a.hbm_dtype

    # 按 --settings 重新读 HW (BLEN 等几何 + cycle 常量), 覆盖模块默认。
    global HW
    HW = load_hw(a.settings)
    print(f"HW: {a.settings} → MLEN={HW['MLEN']} HLEN={HW['HLEN']} BLEN={HW['BLEN']} VLEN={HW['VLEN']}")

    # 选 transmission 表 (emulator 实测搬运, KERNEL_REPORT §1.3)
    if a.trans == 'auto':
        u = a.ir_dir.upper()
        sel = 'ssb' if 'SSB' in u else ('dsb' if 'DSB' in u else 'none')
    else:
        sel = a.trans
    TRANS = {'dsb':DSB_TRANSMISSION_NS, 'ssb':SSB_TRANSMISSION_NS, 'none':{}}[sel]

    isa_def = json.load(open(a.customisa))
    PIPE = {k: int(eval(v['pipelined'], {}, HW)) for k, v in isa_def.items() if 'pipelined' in v}
    def cyc(op):
        if op in PIPE: return PIPE[op]
        al = {'V_SHFT_V':'V_BASIC','V_PS_V':'V_BASIC','C_HADAMARD_TRANSFORM':'V_BASIC',
              'M_TMV':'M_MV','M_BTMM_WO':'M_BMM_WO'}
        return PIPE.get(al.get(op, ''), 1)

    # --- 解析一个 .isa: 展 cloop, 按桶累加 cycle + 数 H_ 条数 ---
    def parse(path):
        toks = []
        for ln in open(path):
            s = ln.split(';')[0].strip()
            if not s: continue
            op = s.split()[0]
            if op == 'C_LOOP_START': toks.append(('LS', int(re.search(r',\s*(\d+)', s).group(1)), op))
            elif op == 'C_LOOP_END': toks.append(('LE', 0, op))
            else: toks.append(('OP', 0, op))
        pos = [0]
        buckets = {'matmul':0,'vector':0,'scalar':0,'scalar_int':0,'ldst':0,'memory':0,'control':0}
        hops = [0]
        # 路 B: 按 H_ 指令类型分别累计条数, 用于解析算 memory 字节
        hcnt = {'H_PREFETCH_M':0, 'H_PREFETCH_V':0, 'H_STORE_V':0}
        def walk(m):
            while pos[0] < len(toks):
                k, v, op = toks[pos[0]]
                if k == 'LS': pos[0]+=1; walk(m*v)
                elif k == 'LE': pos[0]+=1; return
                else:
                    pos[0]+=1
                    b = bucket(op); buckets[b] += cyc(op)*m
                    if b == 'memory':
                        hops[0] += m
                        if op in hcnt: hcnt[op] += m
        walk(1)
        return buckets, hops[0], hcnt

    # --- 功耗 (W) ---
    if a.power_model == 'george':
        _act = a.vec_activity if a.vec_activity is not None else 3.0
        P_mcu_default = mcu_power_w_george(HW['BLEN'], HW['MLEN'])
        P_vec = vec_power_w_george(HW['VLEN'], _act)
    else:  # bottomup 新标准 (BF16→FP32, 7nm/1GHz)
        _act = a.vec_activity if a.vec_activity is not None else 1.0
        P_mcu_default = mcu_power_w(HW['BLEN'], HW['MLEN'])
        P_vec = vec_power_w(HW['VLEN'], _act)
    P_mcu  = a.mcu_power_w if a.mcu_power_w is not None else P_mcu_default
    P_msram= msram_power_w(HW['MLEN'], 32)    # MATRIX_SRAM_SIZE/MLEN
    P_vsram= vsram_power_w(HW['VLEN'], 128)   # VECTOR_SRAM_SIZE/VLEN

    rows=[]
    for d in sorted(glob.glob(os.path.join(a.ir_dir, '*/'))):
        name = os.path.basename(d.rstrip('/'))
        isa = os.path.join(d, name+'.isa')
        if not os.path.exists(isa): continue
        bk, hops, hcnt = parse(isa)
        # 最终标准口径 (用户拍板):
        #   compute = matmul(不÷) + vector÷N + scalar_fp÷N
        #   scalar_int / ldst / control / memory  —— 永不记录。
        #   SIMT: 1 warp × N 线程, 只 vector + scalar_fp ÷N; matmul 不受益。
        N = max(1, a.simt_threads)
        vec_cyc_eff = bk['vector'] / N
        scl_cyc_eff = bk['scalar'] / N          # scalar = scalar_fp 桶 (只真 FP 算术)
        compute_cyc = bk['matmul'] + vec_cyc_eff + scl_cyc_eff
        compute_us = compute_cyc/FREQ_HZ*1e6
        # 诉求2: memory_time 解析算(路 B, George bytes/带宽 口径), 不从 emulator report 抓。
        if a.mem_source == 'report':
            memory_us = TRANS.get(name, 0)/1e3
        else:
            memory_us = memory_us_pathB(hcnt)
        # George 重叠口径: total = max(compute, memory)
        total_us = max(compute_us, memory_us)
        bound = 'mem' if memory_us > compute_us else 'comp'
        # 能耗: MCU 活跃=matmul cycle; SRAM 全程通电(用 total_us, 含搬运段漏电)。
        # SIMT: Vector Unit 功耗 ×N, 活跃时间 = vector_cyc/N → 能耗 = (P_vec×N)×(vec_cyc/N) 不变。
        #       VSRAM 功耗不变(随 total_us, 已含 SIMT 缩短后的时间)。
        t_mcu = bk['matmul']*NS
        t_vec_eff = vec_cyc_eff*NS              # SIMT 缩短后的 vector 活跃时间
        E_mcu = P_mcu*t_mcu; E_vec = (P_vec*N)*t_vec_eff
        E_msram = P_msram*t_mcu; E_vsram = P_vsram*(total_us/1e6)
        E = E_mcu+E_vec+E_msram+E_vsram
        rows.append(dict(name=name, **bk, hops=hops, compute_us=compute_us,
                         memory_us=memory_us, total_us=total_us, bound=bound,
                         E_mcu=E_mcu,E_vec=E_vec,E_msram=E_msram,E_vsram=E_vsram,E=E))

    rows.sort(key=lambda r:-r['E'])
    TE=sum(r['E'] for r in rows)
    Tc=sum(r['compute_us'] for r in rows); Tm=sum(r['memory_us'] for r in rows)
    Tt=sum(r['total_us'] for r in rows)
    N = max(1, a.simt_threads)
    P_vec_eff = P_vec * N
    if N > 1:
        print(f"SIMT: 1 warp × {N} 线程 → vector+scalar_fp cycle ÷{N}, "
              f"Vector Unit 功耗 ×{N} ({P_vec:.3f}→{P_vec_eff:.3f} W); "
              f"matmul 不受益; scalar 单元功耗未建模(=0, George 缺 scalar 综合数据); "
              f"能耗不变, 时间缩短~{N}×")
    else:
        print(f"SIMT: 关闭 (纯 SIMD, --simt-threads 1; compute 不含 scalar_int)")
    _pm = 'bottomup(FP16 基准 1mW/MAC, 0.5mW/lane @7nm)' if a.power_model=='bottomup' else 'george(拟合)'
    print(f"功耗模型: {_pm}")
    print(f"精度档: {HBM_DTYPE} → MCU×{_mcu_scale():.1f}(累加精度) / Vector·SRAM×{_vec_scale():.1f}(位宽), "
          f"elem={_elem_bits()}bit, mx_factor={_mx_factor():.3f}")
    print(f"功耗(W): MCU={P_mcu:.1f}  Vector={P_vec_eff:.3f}  MSRAM={P_msram:.3f}  VSRAM={P_vsram:.3f}")
    print(f"cycle公式: {os.path.basename(a.customisa)} (M_BMM={PIPE.get('M_BMM')}, M_MM={PIPE.get('M_MM')}); total=max(compute,memory)")
    if a.mem_source == 'analytic':
        print(f"memory 口径: analytic 路B (George bytes/带宽; M端口={HW['MLEN']}B/cyc, V端口={HW['VLEN']}B/cyc, "
              f"mx_factor={_mx_factor():.3f}); compute 不含 scalar_int")
    else:
        print(f"memory 口径: report 实测 transmission 表={sel} "
              f"({'emulator KERNEL_REPORT §1.3' if TRANS else 'none → memory=0'}); compute 不含 scalar_int")
    print("="*120)
    print(f"{'kernel':<16}{'matmul':>12}{'vector':>11}{'scalarFP':>11}{'sclrINT':>12}{'#H':>5}{'comp_us':>10}{'mem_us':>9}{'max_us':>10}{'bnd':>5}{'E(mJ)':>9}")
    print("-"*120)
    for r in rows:
        print(f"{r['name']:<16}{r['matmul']:>12,}{r['vector']:>11,}{r['scalar']:>11,}{r['scalar_int']:>12,}{r['hops']:>5}"
              f"{r['compute_us']:>10.1f}{r['memory_us']:>9.2f}{r['total_us']:>10.1f}{r['bound']:>5}{r['E']*1e3:>9.2f}")
    print("-"*100)
    print(f"\n整链 compute = {Tc/1e3:.2f} ms   memory = {Tm/1e3:.2f} ms   total(Σmax) = {Tt/1e3:.2f} ms")
    print(f"整链能耗     = {TE*1e3:.1f} mJ = {TE:.4f} J   等效平均功率 = {TE/(Tt/1e6):.1f} W")
    for comp,key in [("MCU",'E_mcu'),("Vector",'E_vec'),("MSRAM",'E_msram'),("VSRAM",'E_vsram')]:
        s=sum(r[key] for r in rows)
        print(f"  {comp:<7}{s*1e3:>8.1f} mJ ({s/TE*100:>4.1f}%)")
    # device-count: 理想线性 N device 并行 → 时间÷D, 总能量不变, 平均功率×D。
    D = max(1, a.device_count)
    if D > 1:
        Tt_d = Tt / D
        P_avg_d = TE / (Tt_d/1e6)            # = 原功率 × D
        print(f"\ndevice_count={D} (理想线性并行): "
              f"total = {Tt_d/1e3:.2f} ms (÷{D})   "
              f"能耗 = {TE*1e3:.1f} mJ (不变)   "
              f"等效平均功率 = {P_avg_d:.1f} W (×{D})")

if __name__ == '__main__':
    main()
