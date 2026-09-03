# Nemotron Agentic Workload：Matrix L-Compute DSE

## 1. 这次新增了什么

这轮没有增加 ISA、硬件或 cache。它把真实 B200 campaign 中的 48 条
Nemotron Agentic workload 接入现有 52 层 Matrix L-Compute 时间线：

- BFCL v3、GPQA-Diamond、SWE-bench Verified 各 16 条；
- 真实官方 NVFP4 checkpoint，固定 revision
  `ce1b118ae66ec705d02c241525192832eb045fd3`；
- GPU 按 prompt 长度组成的 `B1/B2/B4/B8/B16` 分组；
- 23 个 MoE 层、Top-6、连续 32 个 decode step 的真实 eager 路由；
- 优化 vLLM 路径测得的 TTFT、ITL、吞吐、显存和能耗。

归档 SHA256 为
`18ff9fa81a8993cc9d5406e1f1db0d3d73357bcfc935751f160ed5a636827435`，
包内 38 个校验项全部通过。

## 2. 三种数据不能混在一起

| 数据 | 含义 |
|---|---|
| GPU timing/power | 真实 B200 NVFP4 测量 |
| GPU eager routing | 真实 checkpoint 路由，供 workload replay 使用 |
| PLENA cycles | 官方尺寸、Compiler recurrence 和 symbolic weights 的 pre-RTL 公式时间线 |

优化 timing 与 eager routing 的生成 token 只有 3/48 条完全相同。因此路由回放
只使用 eager run 自己的 token trace；timing run 只作 GPU baseline，不能把两者的
token ID 拼成一条“实测执行”。

GPU routing 是 48 条独立 B1 trace。Simulator 按 GPU timing 的 length-sorted
batch membership 合并同一组请求的 active-expert union。这比均匀路由或最坏边界
更真实，但仍不是直接挂 hook 采集的 B2/B4/B8/B16 路由，报告中明确保留这个限制。

## 3. 路由修正

旧 DSE 的最大分散边界假定每一步每层触碰 `min(128, 6B)` 个专家。真实 Agentic
trace 显示专家有明显复用：

| Batch | 最大分散假定 | 真实 active experts 中位数 |
|---:|---:|---:|
| 1 | 6 | 6 |
| 2 | 12 | 11 |
| 4 | 24 | 19 |
| 8 | 48 | 32 |
| 16 | 96 | 49 |

所以 B16 的 routed-expert weight traffic 不能继续按 96 个专家估算。新结果对
93 个 length-sorted group 分别建模，再汇总中位数和 P95，而不是只选一个 prompt。

## 4. GPU baseline

下表是原 campaign 对三个 benchmark 全部正式 measurement 的汇总：

| Batch | TTFT median | ITL median | 吞吐 median | Batch 能耗 median |
|---:|---:|---:|---:|---:|
| 1 | 81.79 ms | 4.454 ms | 144.80 tok/s | 65.30 J |
| 2 | 80.82 ms | 5.086 ms | 267.54 tok/s | 75.78 J |
| 4 | 83.73 ms | 5.614 ms | 494.06 tok/s | 97.90 J |
| 8 | 86.44 ms | 5.766 ms | 969.99 tok/s | 129.02 J |
| 16 | 122.05 ms | 6.738 ms | 1569.63 tok/s | 186.69 J |

这些值不是 PLENA 模拟结果，也没有被拿来校准某条 ISA 的周期。
它们来自原 campaign 的全局 `all` 汇总；生成的 `summary.csv` 为了保留
workload group 权重，另外报告各 group 中位数的中位数，两种统计口径不要混用。

## 5. 真实路由下的 PLENA 消融

硬件假设与主 Matrix L-Compute campaign 相同：`MLEN=2048`、`BLEN=32`、
64 banks、1 MiB BF16 Matrix SRAM、1 GHz、1560 B/cycle HBM。每组连续模拟
32 个 decode step。下表是三个 benchmark 所有 group 的中位数：

| Batch | D/A：相对原 PLENA | D/B：相对 Arlo | D/C | D TPOT 代理 | D 聚合吞吐代理 |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.016x | 1.545x | 1.098x | 2.009 ms | 497.8 tok/s |
| 2 | 2.580x | 1.848x | 1.152x | 2.583 ms | 774.4 tok/s |
| 4 | 3.286x | 2.227x | 1.220x | 3.571 ms | 1120.1 tok/s |
| 8 | 4.100x | 2.664x | 1.298x | 5.267 ms | 1519.1 tok/s |
| 16 | 5.082x | 3.191x | 1.393x | 7.999 ms | 2000.3 tok/s |

`A/B` 是逐条动态指令的发行代理；`C/D` 才包含 Matrix service、递推算术和
HBM 项。因此 `D/A` 和 `D/B` 是多行执行、紧凑 descriptor、减少 chunk/issue
等组合收益，不是“可编程 skew 单项加速”。

批量越大，Matrix 权重被更多请求摊薄，而每个请求的 recurrent state 仍需独立
更新，所以 L-Compute 覆盖的工作占比上升。这解释了 D/B 从 B1 的 1.545x
增至 B16 的 3.191x；它不是 GPU batch scaling 推导出来的数字。

## 6. Bank conflict 的准确结论

在 93 个 group 中：

- C 的累计 bank-stall 范围为 `188,416` 到 `3,014,656` cycles；
- D 的 bank-stall 全部为 0；
- 但公平的 D′ 对照使用固定对角接线和普通 per-tile base phase，也能达到 0 stall；
- 因此 D/D′ 的纯 bank-service speedup 仍是 `1.00x`。

这意味着当前结果证明了 `L_TILE` 多行执行和 compact view 有效，但没有证明新增
可编程 `row_skew/tile_skew` 加法器有性能必要性。若论文主张“可编程斜存消除
冲突”，必须找到 D′ 无法通过普通 base placement 实现的真实 packet；Agentic
MoE routing 本身不会改变 Mamba state packet 的物理 bank 坐标。

## 7. 不能声称什么

- 不能用 PLENA 的 1 GHz 代理 TPOT 除以 B200 ITL，称为硅片加速比；
- 不能报告 PLENA Token/J，当前没有 RTL 功耗或综合结果；
- 不能称为真实 30B 权重在 Rust 第一层到最后一层执行；
- 不能称为直接测得的 B2/B4/B8/B16 routing；
- 不能把 D/B 全部归因给斜存或 bank conflict。

## 8. 复现

原始 613 MiB routing JSON 保存在外部 artifact，不进入 Git。运行：

```bash
SIMULATOR_ROOT=/absolute/path/to/PLENA_Simulator
COMPILER_ROOT=/absolute/path/to/PLENA_Compiler
CAMPAIGN_ROOT=/absolute/path/to/AGENTIC_NEMOTRON_B200_20260903

cd "$SIMULATOR_ROOT"

nix develop --no-write-lock-file --command \
  just matrix-lcompute-agentic \
  "$CAMPAIGN_ROOT" \
  "$COMPILER_ROOT"
```

输出位于 `artifacts/matrix_lcompute_agentic_v1/`：

- `campaign.json`：完整证据和 claim boundary；
- `group_results.csv`：93 个 group 的逐组结果；
- `summary.csv`：按 benchmark/batch 的中位数和 P95。
