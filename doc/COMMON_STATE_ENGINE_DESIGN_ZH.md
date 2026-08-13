# PLENA Common State Engine: Mamba-2 + KDA

## 当前范围

这一版只验证一个问题：Nemotron 3 Mamba-2 和 Kimi K3 KDA 能否共用相同的
recurrent-state datapath 和 banked head-tile SRAM。它不是 RTL，也没有声称
支持完整 Kimi K3；MLA、LatentMoE、AttnRes 和 vision tower 尚未加入。

## 为什么可以共用

| Workload | 每个 head 的 state | 默认存储 | 每 head 容量 |
|---|---:|---:|---:|
| Nemotron 3 Mamba-2 | 64 x 128 | FP32 | 32 KiB |
| Kimi K3 KDA | 128 x 128 | BF16 | 32 KiB |

两种算法的公式不同，但都需要：state decay、state-vector reduction、rank-1
outer-product update 和高精度累加。因此候选硬件使用两个 32 KiB buffer
ping-pong：计算当前 head 时预取下一个 head。

Mamba 可以在一遍 state traversal 中完成 update 和 C reduction。KDA 必须先
完成 `prediction = S @ k` 才知道 error，再执行 update/output，所以在片上
head tile 上读两遍；state 从 HBM 只读一次、最终只写一次。

## SRAM 排布修正

简单的 `bank=(row+column)%banks` 只能保证一条 row 或 column 分散，不能保证
一次读取完整 `4 x 8` tile 时没有冲突。当前候选采用二维 cyclic mapping：

```text
local_bank = (row % 4) * 8 + (column % 8)
```

32 个元素恰好进入 32 个 bank。粗粒度 row/column tile 决定每个 bank 内的
offset，因此没有复制 state。Compiler 仍看到正常二维 state；mapping 对 ISA
透明。

## 第一版未校准结果

候选参数：1 GHz、64 B/cycle HBM、1 head lane、32 FMA lanes、4 x 8 state
tile、32 个 single-port banks、两个 head-tile slots。

| Workload | State source | Layout | us/layer | Bank stall |
|---|---|---|---:|---:|
| Mamba-2 | HBM stream | row-major | 65.5 | 49,152 |
| Mamba-2 | HBM stream | dual-axis cyclic | 65.5 | 0 |
| Mamba-2 | resident | row-major | 65.5 | 49,152 |
| Mamba-2 | resident | dual-axis cyclic | 32.8 | 0 |
| KDA | HBM stream | row-major | 393.2 | 294,912 |
| KDA | HBM stream | dual-axis cyclic | 147.5 | 0 |

Mamba streaming 时 layout 没改变总时间，因为 2 MiB read + 2 MiB write 已经
成为瓶颈。KDA 的两遍片上 traversal 使 bank conflict 更严重，所以 layout
直接影响总时间。

## 目前不能得出的结论

1. 32 banks、4 x 8 lanes、1 head lane 还不是冻结参数。
2. DSE 没有包含 bank mux、地址生成、布线和跨时钟 SRAM 的面积/频率代价。
3. Mamba B/C group broadcast 属于 projection/input path，不等同于 state SRAM
   mapping。
4. KDA chunked prefill 的 tile-16 matrix path尚未映射到 PLENA Matrix Engine。
5. 完整模型仍可能被 projection/MoE 权重流量限制，不能把 state-core speedup
   当成端到端 speedup。

## 下一步

1. 把 Mamba/KDA 描述成同一个 `X_STATE` descriptor contract，算法字段选择
   `MAMBA2` 或 `KDA`。
2. Transactional Simulator 模拟 head-tile PRELOAD、两遍 KDA、COMMIT、cache
   hit/miss 和 bank stall。
3. Compiler 生成 Mamba single-pass 与 KDA two-pass trace，并验证 state 生命周期。
4. 扫描 1/2/4 head lanes、16/32/64 banks、1/2 tile slots，确定 RTL 参数。
5. 参数冻结后才从 RTL main 创建 `feat/common-state-engine`。
