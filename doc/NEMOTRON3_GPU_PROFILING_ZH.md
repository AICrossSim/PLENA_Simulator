# Nemotron 3 GPU Profiling Request

## 目的

GPU profiling 要回答三类问题：

1. 真实 Nemotron 3 中 Mamba、Attention、MoE 各占多少时间。
2. Mamba fused kernel 实际从 DRAM 读写多少数据，尤其是 FP32 recurrent state。
3. Analytic workload 的 stage、MoE routing 和流量假设是否正确。

GPU kernel latency 不是 PLENA state engine latency，不能直接填入 PLENA 周期模型。

## 模型与硬件

- 模型：`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- GPU：4 x RTX A6000 48 GiB
- dtype：BF16 weights/activations，记录运行时实际 `mamba_ssm_cache_dtype`
- 不使用 NVFP4 作为这组 GPU timing baseline；A6000 没有原生 NVFP4 Tensor Core

Hugging Face checkpoint 约 63.2 GB。无需 git clone 模型仓库，
`from_pretrained`、vLLM 或 SGLang 会下载权重；需要足够磁盘和 Hugging Face
访问权限。完整模型建议用 4-way tensor parallel。单 Mamba layer microbenchmark
可以只下载 config/custom code，并使用随机权重。

## 必跑阶段 1：单个真实 shape Mamba layer

使用官方 Nemotron implementation 和真实 shape：hidden 2688、projection 10304、
64 Mamba heads、head dim 64、state dim 128、8 groups、conv kernel 4、chunk 128。

Prefill cases：

```text
batch=1, sequence={128, 512, 2048, 8192}
```

Decode cases：

```text
batch={1, 4, 8, 16}, one-token STEP, warm resident state
```

每个 case 先 warmup，再用 CUDA events 或 PyTorch profiler 测 50 次；报告
median/p95。用 Nsight Systems 获取 kernel 顺序和融合边界，再选代表 case 用
Nsight Compute 收集：

```text
dram__bytes_read.sum
dram__bytes_write.sum
sm__cycles_elapsed.avg
sm__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
```

必须分出：in projection、conv1d、dt/exp、state update/output、gate/norm、
out projection。若 state update 和 output 已融合，标为
`mamba_state_update_output_fused`。

## 必跑阶段 2：完整模型 baseline

先用官方模型卡支持的 vLLM 或 SGLang 跑通 4 GPU BF16。固定输入 token，关闭
随机采样对 timing 的影响。至少运行：

| Phase | Batch | Context/input | Generated |
|---|---:|---:|---:|
| prefill | 1 | 512 | 1 |
| prefill | 1 | 2048 | 1 |
| prefill | 1 | 8192 | 1 |
| decode | 1 | 512 | 32 |
| decode | 1 | 2048 | 32 |
| decode | 1 | 8192 | 32 |
| decode | 4 | 2048 | 32/request |

每个 case 报告 TTFT、mean/median/p95 TPOT、tokens/s、峰值显存。先用 Nsight
Systems 或运行时 profiler 得到完整时间线；不要一开始对完整四卡 generation
运行全量 Nsight Compute，开销会非常大。

## MoE 必须额外记录

对每个 MoE layer 和 decode step 保存 top-6 expert IDs，至少汇总：

```text
unique routed experts/layer/step: min, mean, p95, max
token assignments per expert
shared-expert time
routed-expert time
all-to-all or cross-GPU communication time
```

否则 analytic model 只能猜每步到底读 6 个、24 个还是更多 expert weights，
整机 HBM 结论会不可靠。

## 输出文件

交付以下文件，不要只发一张 profiler 截图：

```text
environment.txt
full_model_latency.csv
mamba_layer_latency.csv
moe_routing.csv
nsys_report.nsys-rep
ncu_mamba_prefill.csv
ncu_mamba_decode.csv
nemotron3_gpu_profile.json
```

`nemotron3_gpu_profile.json` 按 `doc/nemotron3_gpu_profile.schema.json` 生成，之后运行：

```bash
uv run python -m analytic_models.performance.nemotron3_profile \
  nemotron3_gpu_profile.json
```
