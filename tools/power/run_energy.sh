#!/usr/bin/env bash
# 一键跑 PLENA 能耗估算 (数据用 George, 结构用真实 .isa)。
#
# 用法:
#   tools/power/run_energy.sh                          # 默认跑 managerbuild_SSB_2/ir, MCU=100W
#   tools/power/run_energy.sh <ir_dir>                 # 指定 ir 目录
#   tools/power/run_energy.sh <ir_dir> <mcu_power_w>   # 指定 MCU 功耗(W)
#   tools/power/run_energy.sh managerbuild_SSB_2/ir formula # MCU 用 M×K 公式而非固定值
#
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

IR_DIR="${1:-managerbuild_SSB_2/ir}"
MCU="${2:-100}"     # 默认 100W; 传 "formula" 则用 M×K 公式

TOOL="tools/power/plena_isa_energy.py"
# George performance 版 cycle 公式 (M_BMM=8192); 工作树无, 从 experiment 分支导出到临时文件
CUSTOMISA="$(mktemp /tmp/customISA_perf.XXXX.json)"
trap 'rm -f "$CUSTOMISA"' EXIT
git show origin/experiment:analytic_models/performance/customISA_lib.json > "$CUSTOMISA" 2>/dev/null \
  || { echo "无法导出 George customISA_lib.json (需要 origin/experiment 分支)"; exit 1; }

echo "========================================================================"
echo "PLENA 能耗估算  |  ir=$IR_DIR  |  cycle=George(M_BMM=8192)  |  MCU=$MCU"
echo "========================================================================"

if [ "$MCU" = "formula" ]; then
  python3 "$TOOL" "$IR_DIR" --customisa "$CUSTOMISA"
else
  python3 "$TOOL" "$IR_DIR" --customisa "$CUSTOMISA" --mcu-power-w "$MCU"
fi
