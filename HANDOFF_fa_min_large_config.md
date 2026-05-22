# 交接:fa_min 大配置正确率问题

## 状态总览

- **寄存器分配重写已完成且工作**:v2 后端 `mir_to_isa.py` 现在用
  scope-recursive spill(carried 值在 `C_LOOP_START` 前 store、循环外;
  body 内只 reload)。编译通过、模拟器跑完、无崩溃、无对齐错。
- **小配置(mlen=64)正确率 = 对 ✅**(用户确认)。
- **大配置(mlen=1024)数值范围正常,但正确率不对 ❌** ← 当前要查的。
- 大配置模拟器跑完:`Latency 143399218 ns`(≈142.7M cycle,M_MM 占 94%)。

关键推论:**寄存器/地址/算法结构在小配置全对**,否则小配置也会错。
大配置才错 = **只在大尺寸暴露的、与尺寸相关的问题**。这不是 regalloc
bug(那会在小配置也崩/错)。

## 当前配置(plena_settings.toml)

- `active = "analytic"`,ANALYTIC 区:mlen=1024 blen=8 hlen=128 vlen=1024
  broadcast=8。
- **⚠️ 已知不一致(待确认是否影响)**:Rust 模拟器
  `transactional_emulator/src/load_config.rs:417` **硬读 `BEHAVIOR` 区**
  (`Ok(settings.behavior)`),不看 `active`。Python 编译侧读 `active`
  (=analytic)。两区现在 mlen 都=1024,但 **BEHAVIOR.BROADCAST_AMOUNT=4
  vs ANALYTIC=8**(line 30 vs 228)——编译按 8、模拟按 4。
  flash-attention 用 BTMM,broadcast 不一致**可能**是正确率问题之一,优先
  核这个。

## ★ 寄存器分配已用 trace_gp 工具双重排除(2026-05-21)

`transactional_emulator/tools/trace_gp.py`(justfile 自动跑,写
`build/trace_gp_report.txt`)符号化追踪每个 GP/slot 持有的 MIR %value,
检测任何"读 GP 时持有的 % ≠ 期望 %"的腐败。结果:
- 小配置(通过):**0 mismatch**
- 大配置 HEAD=8(stride_mode 关):**0 mismatch**
→ **寄存器分配/地址在所有配置自洽,bug 不在 regalloc。** 之前那个 AI 说
的"O store 丢 bx 项"经核实不成立(bx 项在 dst gp5,line 425-427)。

## 当前最强方向:多 head-group(HEAD=16)

HEAD_COUNT 公式现为 `MLEN//HLEN`(=8,单 head-group,stride_mode 关)。
之前失败的备份是 `*2`(=16,2 个 head-group,stride_mode 开)。
- HEAD=8:elements_per_batch=1024=mlen → chunks_per_batch=1 → stride_mode OFF
- HEAD=16:elements_per_batch=2048>mlen → chunks_per_batch=2 → stride_mode ON
**唯一区别是 stride-mode reorder。** 正在验证 HEAD=8 是否通过;若通过,
bug 锁定在 HEAD=16 的 reorder 或硬件多-group 写回 vs reorder 假设。
reorder 逻辑用正确几何验证过 0 错位,但**依赖"stage 物理顺序=
`[h_group][s_tile][行]`"的假设,需对硬件实际 stage-output 写回核实**
(失败备份 ASM 的 stage tile 顺序是 (h,s)=(0,0)(0,1)(1,0)(1,1))。

## 已排除 / 已确认(不跑 sim,只读 build 数据)

- **算法结构正确**:`build/post_to_plena.hlir.txt` 的 online-softmax 完整且对
  —— [42] O_loc ×= M_RES(running rescale)、[40] L_NEW += P_SUM、[55]
  O_loc ×= L_INV(最终归一化)都在。小配置用同一 HLIR 跑对 → 算法层无 bug。
- **VRAM 不越界**:S_loc @3145728 + 8M = 11.5M < 容量(behavior depth16384 ×
  vlen1024 = 16.7M)。
- **HBM scale 截断已修**(main.rs:1216 多 chunk)。
- **BROADCAST_AMOUNT 已对齐**(两区都 8)。

## 最强嫌疑(精确到此):O_loc packed-head 物理 stride

O_loc shape `1×1024×8×128`,被两种索引混用:
- per-head:`(0, row, head_phase, 0) ext=(1,1,1,128)`(op[26]/[35] row_mul_fp)
- all-head:`(0, row1, 0, 0) ext=(1,1,8,128)`(op[4] v_zero / op[31] v_add)
- 写回:`dma_v2h_slice O_loc -> O_hbm (0,bx*1024,head*8,0) ext=(1,1024,8,128)`

两种索引的物理地址必须落在同一 O_loc 布局、且 head_phase 维 stride 一致
(应 = hlen = 128)。**大配置 head=8/hlen=128 下,如果 per-head 的 head_phase
stride 算错(非 128),per-head 的 rescale 写 与 all-head 的 v_add/写回 错位**
→ 输出数值范围正常但位置错乱(== 观察到的症状)。小配置 head 维更小,可能
恰好不暴露,或 stride 凑巧对。

**查法(GCP)**:
1. 跑比对拿差异模式。**整块/行错位 = 印证 stride/layout**;均匀偏尺度 = 别处。
   注意:这次 `vector_result.mem` 是空的、`stage_output=O_hbm` 但
   `check_hbm=False` —— **比对链路这次没产出**,需确认 view_mem 配置或直接解
   `hbm_for_behave_sim.bin` O 区(MX 编码,O 写回 4718592 字节 = 4194304 elem
   ×1B + 524288 scale×1B,block=8)对 golden_result.txt 的 Original Output 段。
2. 在 ISA(`generated_asm_code.asm`)里查 op[26]/[35](O_loc per-head row_mul)
   的 VRAM 地址 stride,确认 head_phase 维 = 128。对照 op[31] v_add 的
   all-head stride。参考记忆:Pinned global layout / Matmul dst_row_stride
   (packed-head dst 需物理 s_inner stride,不是 extents 连乘)。
3. **A/B**:小配置同样 dump O_loc 的两种索引地址,对比哪个 stride 从对变错。

## 嫌疑排序(从最可能)

1. **BROADCAST_AMOUNT 编译/模拟不一致**(8 vs 4)。先把 BEHAVIOR 区的
   BROADCAST_AMOUNT 也改成 8,或确认 broadcast 对 BTMM/BMM 的影响。这是
   "大配置才暴露"的有力候选(小配置 toml 当时两区可能一致)。

2. **输出 layout 在大配置下重排错**。比对走
   `transactional_emulator/tools/check_mem.py: compare_vram_with_golden`,
   line 381-386 有 stride-mode 的 batch-wise 重排("cross-batch-
   interleaves ... scrambling otherwise-correct VRAM")。
   comparison_params:`num_rows=4096 num_batches=2048 elements_per_batch=2048
   row_dim=1024 use_stride_mode=true`。若数据本身对、只是重排错 → match_rate
   会呈现规律性错位(整块对/错)。参考记忆 [Testbench output layout unified]。

3. **online-softmax 跨 kv_block 合并 / rescale**。大配置 row=1024、
   head_phase=8、kv_block=2。若 running max/sum 的块间合并或 O 的
   rescale(乘 exp(M_old-M_new))在大尺寸下边界错 → 数值范围对、值错。
   参考记忆 [SSB attention diagnostic](attention chain ~45% 是量化非 bug)。

4. **已排除**:HBM scale chunk 截断 —— `transfer_mx_from_hbm` 的 scale
   读取已是多 chunk 循环(main.rs:1216 `while copied < span`),不是这个。

## 下一步:先拿到差异模式(关键)

在 GCP 跑比对,拿 `match_rate` 和**错在哪**(整块错 vs 散布错 vs 尺度偏移):

```bash
cd PLENA_Simulator
# 跑完模拟器后:
python3 transactional_emulator/tools/view_mem.py
# 看 match_rate(%)、relative_match_rate、error/signal ratio
```

差异模式直接区分嫌疑:
- **整块对/整块错、或行列错位** → layout 重排(嫌疑 2)
- **全体均匀偏一个尺度** → broadcast/scale(嫌疑 1)或归一化
- **前面的 kv_block 对、后面错** → 块间合并/rescale(嫌疑 3)

## A/B 利器:小配置是 known-good

小配置(mlen=64)正确。可以**同一套比对逻辑跑小配置**,对比大配置,看
哪个量从"对"变"错"。如果某个中间 buffer(S_loc / O_loc / M_OLD / L_OLD)
在大配置下 dump 出来和 golden 对,但最终 O 错 → 缩小到最后的写出/layout。

## 已改动的源文件(本次寄存器重写,勿回退)

- `compiler/tilelang_tvm_compiler/mir_to_isa.py` — scope-recursive 单一
  spill 机制(spill_carried_at_entry / _make_room / use-and-return reload)。
- `compiler/tilelang_tvm_compiler/mir.py` — C_SET_ADDR_REG operand_kinds
  改 ("verbatim_str","i32")(早先修的汇编器崩溃)。
- `compiler/tilelang_tvm_compiler/pre_isa_pass_v2.py` — C_SET_ADDR_REG
  三处 operands 加 "gp0";FORCE_SERIAL（pre_isa_ir_v2.FORCE_SERIAL_LOOPS）。
- `compiler/tilelang_tvm_compiler/pre_isa_ir_v2.py` — FORCE_SERIAL_LOOPS=True
  (emit-time unroll 已废,全 serial)。
- `compiler/tilelang_tvm_compiler/pipeline.py` — MIR per-pass dump 到
  build/mir_passes/(调试用)。

## main.rs 临时诊断(跑通后删)

`transactional_emulator/src/main.rs` 里有 `[H_PREFETCH_M BAD_DST]` /
`[M_MM BADADDR]` / `[V_MUL_VF MISALIGN]` 的 eprintln + 对齐断言带 ctx。
这些是调试加的,正确性查完后清掉。

## 设计文档(已写,供参考)

- `compiler/tilelang_tvm_compiler/REGALLOC_SCOPE_DESIGN.md` — 现行 scope
  递归 spill 的设计。
- `compiler/tilelang_tvm_compiler/REGALLOC_B_DESIGN.md` — 早期方案,已被
  上者取代。
