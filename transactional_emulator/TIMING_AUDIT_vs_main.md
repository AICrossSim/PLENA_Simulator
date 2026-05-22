# 计时模型核实:tilelang-tvm-compiler 分支 vs main

**日期**: 2026-05-21
**结论**: 本分支相对 `main`,对所有 main 已有指令的 cycle 成本和计时逻辑**零改动**。
此前算出的 fa_min cycle profile(73.8M cycle、matmul-bound 90.9%)与 main 的
计时模型完全可比,未被本分支污染。

## 逐层核实证据

| 检查项 | 结果 |
|---|---|
| `lib/runtime/src/time.rs`(计时核心) | 仅**新增**只读 `to_picos()`;已有计时逻辑(时间基准/Duration/to_secs)零改动 |
| `src/main.rs` 的 `cycle!` 行 | 新增 5、删除 **0**、修改 **0** |
| 那 5 个新增 cycle! 的归属 | 全是 main 中**不存在**的新 opcode(见下表),不是给老指令补计时 |
| `cycle!` 宏定义 | 未改 |
| 成本常量定义(`*_CYCLES` / `VLEN` 的 static/helper) | main.rs 内未改 |
| `src/load_config.rs`(成本数值真正来源) | 唯一改动 `max_loop_instructions 10000→1000000`,**与 cycle 无关**;所有 cycle/vlen/cost 数值行零改动 |
| 删除老 opcode match arm | **无**(删 arm 会改老指令行为/计时)→ 一个都没删 |
| `lib/vector_sram/src/lib.rs` | 5 增 1 删,无 cycle/time/latency 改动 |

## 5 个新增 cycle! 明细

这 5 个 opcode 在 main 的 main.rs 里 grep 出现 **0 次**,确认是本分支新增指令:

| 新增 cycle! | 所属 opcode | 成本 | 与 cost-model 记忆一致 |
|---|---|---|---|
| `cycle!(*VLEN)` | S_MAP_FP_V | VLEN(1024) | ✓ S_MAP_* = VLEN |
| `cycle!(*SCALAR_INT_BASIC_CYCLES)` | S_SLL_INT | 1 | ✓ 移位=1 |
| `cycle!(*SCALAR_INT_BASIC_CYCLES)` | S_SLLI_INT | 1 | ✓ |
| `cycle!(*SCALAR_INT_BASIC_CYCLES)` | S_SRL_INT | 1 | ✓ |
| `cycle!(*SCALAR_INT_BASIC_CYCLES)` | S_SRLI_INT | 1 | ✓ |

## 复核命令(可重跑)

```bash
cd transactional_emulator
# cycle! 增删统计
git diff main -- src/main.rs | grep -cE "^\+.*cycle!"   # 5
git diff main -- src/main.rs | grep -cE "^-.*cycle!"    # 0
# 新 opcode 在 main 是否存在
for op in S_MAP_FP_V S_SLL_INT S_SLLI_INT S_SRL_INT S_SRLI_INT; do
  echo -n "$op: "; git show main:src/main.rs | grep -c "Opcode::$op "
done                                                     # 全 0
# 删除的老 opcode arm
git diff main -- src/main.rs | grep -E "^-\s*op::Opcode::"   # 空
# load_config 计时数值
git diff main -- src/load_config.rs | grep -iE "^[-+].*(cycle|vlen|cost)"  # 空
```
