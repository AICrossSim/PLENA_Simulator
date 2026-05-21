# 多-AI 协调板 (同 branch: tilelang-tvm-compiler / compiler: plena-compiler)

> 两个 Claude 实例在不同 GCP、**同一个 git branch** 上工作。无实时通信,
> 唯一共享媒介是这个文件 + git。**改前先 `git pull`,改后立刻 commit+push,
> push 前先读本文件认领区。** 别碰对方认领的文件。

## 文件边界(认领区 —— 改前在这登记)

| 区域 | 负责方 | 说明 |
|---|---|---|
| `compiler/tilelang_tvm_compiler/` 的 **v2 后端 / 寄存器分配** | **GCP-A** | mir.py, mir_passes.py, mir_to_isa.py, pre_isa_ir_v2.py, pre_isa_pass_v2.py, pre_isa_to_mir.py, pipeline.py(v2 路径) |
| `transactional_emulator/tools/` staging/比较 | **GCP-B** | create_sim_env.py, check_mem.py, view_mem.py |
| `tools/memory_mapping/rand_gen.py` | **GCP-B** | quantize/打包 |
| `plena_settings.toml` | 共享(改前喊一声/登记) | 配置 |
| testbench `*_test.py` | GCP-B | |

(GCP-A = 当前这台,负责 regalloc;GCP-B = 另一台,负责 staging/golden。
如分工不符,在这里改。)

## 已 push 状态(谁推了什么)

- **GCP-A**: compiler `plena-compiler` 顶端 `6e40b6d`
  "v2 backend: scope-recursive register allocation"。
  - mir_to_isa.py 单一 spill-to-IntRAM 机制(scope-entry spill carried
    值在 C_LOOP_START 前 store、循环外;body 只 reload;use-and-return)。
  - C_SET_ADDR_REG 3 操作数;FORCE_SERIAL_LOOPS=True。
  - 勿回退这些。
- **GCP-B**(本地工作区,未提交): `create_sim_env.py` golden 精度修复
  (`.2f` → `.9g`,提前 dump golden_output.pt)。**这个修复是好的,GCP-B
  请 commit+push**,GCP-A 也需要它来排除"小值假误差"。
  - check_mem.py / view_mem.py(GCP-A 改过:把比对详情写到
    build/compare_summary.txt、终端只留 match_rate;去掉刷屏 print)。
    **注意:check_mem/view_mem 两边都动过,小心冲突。**

## 当前正在查的问题

**fa_min 大配置(mlen=1024)正确率不对,小配置(mlen=64)正确**:
- 已排除:算法结构(HLIR 对)、VRAM 越界、HBM scale 截断、BROADCAST
  不一致(已对齐 8)、stride-mode reorder(GCP-A 逻辑验证过:这次几何
  4096 chunk 全对位,**不是 reorder bug**)。
- 强嫌疑:数值层(量化/精度)。GCP-B 的 `.9g` golden 修复直接相关——
  之前 golden_result.txt 用 `.2f` 截断,小值产生假误差。**先让这个修复
  生效再看 match_rate**。
- 见 `HANDOFF_fa_min_large_config.md`(GCP-A 写的详细分析)。

## 已知坑

- **V_hbm.pt 缺失崩溃**:`rm -rf build` 后第一次跑偶发 V_hbm.pt 不存在
  → `quantize_tensor(None)` 崩。疑似中间态/并发写 build/。**两台别同时
  写同一个 build/。** 重跑(`rm -rf build` + 完整 run)通常自愈。
- **不要两个 GCP 同时改同一文件** —— 后 push 的会让先 push 的需要 rebase/
  覆盖。check_mem/view_mem 已是双方都碰过的雷区。

## 约定

1. 改前 `git pull`(主仓库 + `cd compiler && git pull`)。
2. 改后立刻 commit + push,且更新本文件的"已 push 状态"。
3. 只改自己认领区的文件;要碰对方的,先在本文件登记+说明。
4. compiler 是 submodule:改 compiler 要先在 compiler 里 push,再(若需)
   在主仓库 bump submodule 指针。
