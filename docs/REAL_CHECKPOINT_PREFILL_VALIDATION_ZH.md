# 真实权重链路与 Transactional Prefill 验证

这组实验补的是两个不同的证据缺口：真实权重是否能连续经过多层，以及
prefill 的全部 chunk 是否真的在 transactional emulator 中更新并传递 state。
它们不是 Nemotron/Kimi 全算子整模性能结论。

## 1. 统一数值契约

- HBM 输入、临时 spill、recurrent state、Matrix SRAM、Vector SRAM：BF16；
- Matrix/归约累加：FP32；
- 没有 cache、`X_STATE`、私有 state SRAM 或运行时调度器；
- Compiler 保存每个 HBM tensor 的元素字节数，BF16 DMA 的地址和 stride 均按
  2 bytes/element 生成。回归测试覆盖声明、存回和重新加载。

## 2. Transactional S128 prefill

执行链均为：Compiler assembly -> 32-bit machine words -> Rust emulator ->
读回全部 128 个输出和最终 state -> 与独立 PyTorch reference 比较。

| Workload | 几何与 chunk | ISA 行 / 机器字 | Rust cycles | HBM read / write | 最大误差 | Bank stall |
|---|---|---:|---:|---:|---:|---:|
| Mamba-2 | B1, 1 head, state/head dim 64, 2 x 64 token | 8,587 / 6,310 | 188,638 | 181,248 / 32,768 B | 0.0009765625 | 0 |
| KDA | B1, 1 head, key/value dim 64, 8 x 16 token | 17,513 / 13,656 | 1,346,121 | 1,146,880 / 393,216 B | 0.0009765625 | 0 |

两项都比较 12,288 个 BF16 值，allclose 通过率均为 100%。这证明完整
chunk 计算、跨 chunk state、输出打包和 HBM 写回；它不等于真实尺寸整模
prefill，也没有 row-major 对照，不能从这些周期计算 TTFT 加速比。

机器可读结果：
[`artifacts/transactional_prefill_bf16/summary.json`](../artifacts/transactional_prefill_bf16/summary.json)。

## 3. 公开 checkpoint 的第一层到最后一层链路

模型为 `AntonV/mamba2-130m-hf`，固定 snapshot
`05e8773fc4ac1cd067e8a18a5c45372ce5178405`，共 24 层。测试先用官方实现
对 4-token prompt 建立初始 state，然后让一个 decode token 连续经过 24 层：

1. 真实 checkpoint 权重执行 BF16 norm/projection/conv 和系数准备；
2. 每层 Mamba recurrence 由 Compiler 生成 `L_TILE`，汇编并在 Rust 执行；
3. Rust 写出的 output/state 直接进入该层后处理和下一层，不回填 golden；
4. 最终执行 norm 与语言模型输出头，并同时比较独立 BF16 链和官方 FP32 链。

结果：24/24 层完成；递推累计 `5,198,064` Rust cycles；bank stall 总数 0；
每层 recurrence output/state、最终 hidden 和 BF16 logits 的最大误差均为 0；
相对官方 FP32 logits 的 relative-L2 为 `0.00787939`，Top-1/Top-5 均一致。

机器可读结果：
[`artifacts/mamba2_130m_real_checkpoint_lcompute/summary.json`](../artifacts/mamba2_130m_real_checkpoint_lcompute/summary.json)。

## 4. 结论边界

已经证明真实权重可以形成第一层到最后一层的连续 recurrence 数据链，也证明
Mamba/KDA chunked prefill 能在 Rust 中完整执行。尚未证明的是：Nemotron 52 层
或 Kimi 93 层真实 checkpoint 的所有 Matrix/Attention/MoE/normalization 操作都
在 Rust 执行，以及完整模型 TTFT/TPOT 或相对 GPU 的硬件加速比。

## 5. 复现

```bash
nix develop --no-write-lock-file --command \
  just test-transactional-prefill-full PLENA_Compiler \
  artifacts/transactional_prefill_bf16/summary.json

# python_bin 需要 Python >=3.11、torch、transformers 和 safetensors。
just test-mamba2-real-checkpoint \
  /path/to/python PLENA_Compiler /path/to/mamba2-130m-hf/snapshot \
  artifacts/mamba2_130m_real_checkpoint_lcompute
```
