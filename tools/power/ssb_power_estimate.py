#!/usr/bin/env python3
"""
SSB 功耗/能耗估计 — 基于 DC 综合拟合的组件稳态功耗 + 报告实测活跃时间。

设计动机
========
现有 analytic_models/power_model/plena_power_model.py 用
    P_mcu = 4.26e-3 * M^2 * K
把 M 当成单个脉动阵列的物理边长。当 MLEN=1024 时 M^2 放大 6.5 万倍 ->
~57 万瓦,物理荒谬。根因:真机不是 1024x1024 物理阵列,而是一个固定的小阵列
(MLEN_PHYS=128)反复跑 (MLEN/MLEN_PHYS)^2 次 tile。功耗取决于**物理阵列**,
延迟/能耗才取决于跑了多少次。

本模型
======
1. 物理阵列固定在 MLEN_PHYS=128 (sys_model_experiment.py 的真机锚点),
   各组件稳态功耗用 DC 拟合公式在 *物理* 尺寸上求值 —— 全部落在拟合范围
   (MLEN/VLEN <= 128) 内,不外推、不爆。
2. 能耗 = Σ_component ( P_component[W] * t_active_component[s] )
   其中 t_active 来自 SSB 报告 §1.5 的 per-class ISA 静态 cycle (1 cycle=1ns)
   + transmission 时间。这是"哪个单元忙了多久"的实测代理。
3. 每个组件按它实际承担的指令类映射活跃时间,不活跃时只计漏电底盘。

口径与局限都打印出来,可直接进报告。
"""

import argparse

# ---------------------------------------------------------------------------
# DC 综合拟合公式 (摘自 analytic_models/power_model/plena_power_model.py, commit 7630947)
# 全部在物理尺寸 (<=128) 上求值 —— 拟合有效区内。
# ---------------------------------------------------------------------------

def p_mcu(M, K):
    """脉动阵列功耗 [W]. 4.26e-3 * M^2 * K (M=4 综合点反推)."""
    return 4.26e-3 * (M ** 2) * K

def p_vector_unit(VLEN, activity=1.0):
    """向量计算单元 [W]. 二次拟合 VLEN=8..64, R^2=0.999998.
    注:综合未标 switching,真实运行应乘 activity 2~5x."""
    base = -1.7746135753e-07 * VLEN**2 + 5.4279213710e-04 * VLEN - 4.9166666667e-05
    return max(0.0, base) * activity

def p_matrix_sram(MLEN, depth):
    """矩阵 SRAM [W]. 二次拟合 @depth=128, 线性缩放. R^2=0.99997."""
    base = 1.9583044211e-06 * MLEN**2 + 2.4835249306e-03 * MLEN + 5.2254044409e-04
    return base * (depth / 128.0)

def p_vector_sram(VLEN, depth):
    """向量 SRAM [W]. 二次拟合 @depth=128, 线性缩放. R^2=0.999996.
    注意 +0.2452W 常数底盘 (SRAM array 漏电主导)."""
    base = -3.1305759318e-08 * VLEN**2 + 3.4435611618e-04 * VLEN + 2.4520172086e-01
    return base * (depth / 128.0)


# ---------------------------------------------------------------------------
# SSB 报告 §1.5 整链数据 (MLEN=1024, 2026-05-28, DC_EN=1, 1cycle=1ns)
# ---------------------------------------------------------------------------
NS = 1e-9
ISA_STATIC_NS = {          # 整链 ISA 静态层各类 cycle (= ns)
    "matmul":      2_246_744,   # M_*  -> MCU + Matrix SRAM 读
    "vector":      7_290_880,   # V_*  -> Vector Unit + Vector SRAM
    "scalar_fp":  11_116_544,   # S_*_FP (含 FPRAM LD/ST) -> Vector SRAM/标量
    "scalar_int": 11_331_999,   # S_*_INT 地址计算 -> 标量/控制 (低功耗)
    "control":     4_450_423,   # C_* loop/setreg -> 控制 (低功耗)
}
ISA_STATIC_TOTAL_NS = sum(ISA_STATIC_NS.values())      # 36,436,590
TRANSMISSION_NS     = 16_523_108                        # emu - isa, HBM 搬运
EMU_TOTAL_NS        = ISA_STATIC_TOTAL_NS + TRANSMISSION_NS  # 52,959,698


def estimate(mlen_phys, k_phys, vlen_phys, msram_depth, vsram_depth,
             vector_activity, hbm_power_w):
    # --- 组件稳态功耗 (物理尺寸, 不爆) ---
    P_mcu   = p_mcu(mlen_phys, k_phys)
    P_vec   = p_vector_unit(vlen_phys, vector_activity)
    P_msram = p_matrix_sram(mlen_phys, msram_depth)
    P_vsram = p_vector_sram(vlen_phys, vsram_depth)

    # --- 各组件活跃时间 (s) ---
    # MCU + Matrix SRAM 在 matmul 期活跃
    t_mcu   = ISA_STATIC_NS["matmul"] * NS
    t_msram = ISA_STATIC_NS["matmul"] * NS
    # Vector Unit 在 vector 期活跃
    t_vec   = ISA_STATIC_NS["vector"] * NS
    # Vector SRAM 在 vector + scalar_fp 期活跃 (FPRAM LD/ST 走它); 全程漏电底盘另算
    t_vsram_active = (ISA_STATIC_NS["vector"] + ISA_STATIC_NS["scalar_fp"]) * NS

    # --- 能耗: 动态(活跃) + 漏电底盘(全程) ---
    # 简化: 计算单元只在活跃期耗能(无活动近似 0); SRAM 全程有漏电底盘 + 活跃增量。
    # Vector SRAM 那 0.2452W 常数项本质是 always-on 漏电,所以 vsram 全程计。
    E_mcu   = P_mcu   * t_mcu
    E_vec   = P_vec   * t_vec
    E_msram = P_msram * t_msram
    E_vsram = P_vsram * (EMU_TOTAL_NS * NS)   # SRAM 全程通电 (漏电底盘主导)
    E_hbm   = hbm_power_w * (TRANSMISSION_NS * NS)

    E_total = E_mcu + E_vec + E_msram + E_vsram + E_hbm

    rows = [
        ("MCU (脉动阵列)",      P_mcu,   t_mcu,                 E_mcu),
        ("Vector Unit",         P_vec,   t_vec,                 E_vec),
        ("Matrix SRAM",         P_msram, t_msram,               E_msram),
        ("Vector SRAM (全程)",  P_vsram, EMU_TOTAL_NS*NS,       E_vsram),
        ("HBM 搬运",            hbm_power_w, TRANSMISSION_NS*NS, E_hbm),
    ]
    return rows, E_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlen-phys", type=int, default=128,
                    help="物理脉动阵列边长 (真机, 默认128; DC拟合有效)")
    ap.add_argument("--k-phys", type=int, default=128, help="reduction 深度 K")
    ap.add_argument("--vlen-phys", type=int, default=64,
                    help="物理向量单元 VLEN (DC拟合到64; 注意外推风险)")
    ap.add_argument("--msram-depth", type=int, default=1024)
    ap.add_argument("--vsram-depth", type=int, default=1024)
    ap.add_argument("--vector-activity", type=float, default=3.0,
                    help="向量单元 switching 修正 (综合低估, 典型2~5, 默认3)")
    ap.add_argument("--hbm-power-w", type=float, default=0.0,
                    help="HBM 搬运功耗基线 [W]; 默认0 (无数据, 需外部TDP)")
    a = ap.parse_args()

    rows, E_total = estimate(a.mlen_phys, a.k_phys, a.vlen_phys,
                             a.msram_depth, a.vsram_depth,
                             a.vector_activity, a.hbm_power_w)

    print("=" * 78)
    print("SSB 功耗/能耗估计 (基于 DC 综合 + 报告实测活跃时间)")
    print(f"物理几何: MLEN={a.mlen_phys} K={a.k_phys} VLEN={a.vlen_phys} "
          f"MSRAM_D={a.msram_depth} VSRAM_D={a.vsram_depth} "
          f"vec_activity={a.vector_activity}x")
    print(f"链总时间 (emulator): {EMU_TOTAL_NS*1e6:.1f} us = {EMU_TOTAL_NS*NS*1e3:.2f} ms")
    print("=" * 78)
    print(f"{'组件':<20}{'稳态功耗(mW)':>16}{'活跃时间(ms)':>16}{'能耗(mJ)':>14}{'占比':>8}")
    print("-" * 78)
    for name, P, t, E in rows:
        pct = E / E_total * 100 if E_total else 0
        print(f"{name:<20}{P*1e3:>16.2f}{t*1e3:>16.3f}{E*1e3:>14.3f}{pct:>7.1f}%")
    print("-" * 78)
    print(f"{'总能耗':<20}{'':>16}{'':>16}{E_total*1e3:>14.3f}{100.0:>7.1f}%")
    print(f"\n等效平均功耗 (总能耗/链时间): {E_total/(EMU_TOTAL_NS*NS):.2f} W")
    print()
    print("口径与局限:")
    print("  - 功耗按物理阵列(MLEN_PHYS)求值,落在DC拟合有效区(<=128),不外推不爆。")
    print("  - 能耗 = 组件功耗 x 该组件活跃时间(报告§1.5 per-class cycle)。")
    print("  - MCU功耗仍是M=4单点外推到MLEN_PHYS,有不确定度(注释标M>=8 RTL bug)。")
    print("  - Vector Unit综合未标switching,已乘activity修正(默认3x);可调。")
    print("  - HBM搬运功耗无DC数据,默认0;有TDP基线时用--hbm-power-w传入。")
    print("  - scalar_int/control(地址/循环)未单列功耗:无对应综合组件,视为标量逻辑")
    print("    并入控制路径,功耗远低于MCU/SRAM,此处忽略(保守低估)。")


if __name__ == "__main__":
    main()
