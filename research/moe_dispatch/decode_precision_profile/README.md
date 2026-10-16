# 全层真实 decode 与多精度 GPU 采集（2026-10-16）

新增实验目录；不改动第2–8轮结果。不提交/推送本轮代码，除非用户另外要求。

## 负载与范围

- 模型：DeepSeek-V2-Lite-Chat，本地实际预训练权重；H=2048，64选6，Routed宽1408，两个Shared合并宽2816。
- 26个MoE层：1–26，层0为Dense。
- B=1/2/4/8/16/32/64/96/128/256，B是同一解码步中不同请求的数量。
- BFCL、GPQA、SWE三个请求来源，每个数据集每档最多两个不重叠请求组，共57组。GPQA仅198请求，B128仅一组、B256无数据；不重复请求补齐。
- 每组真实自回归32个decode步（使用KV cache），每步保留完整Top-6 IDs与scores。Prefill单独记录，禁止当作decode窗口。
- 选取第1/4/8/16/32步保存真实层输入/输出，GPU性能回放使用同一份BF16输入和路由比较六种精度。
- 提示词最多1024 token，超长时保留前1/4与后3/4，逐请求记载截断；固定长度生成在EOS后继续，报告EOS后请求比例。不是自然停止服务SLA。

## 精度含义

| 名称 | 权重 | FFN激活 | 量化 |
|---|---|---|---|
| BF16 | BF16 | BF16 | 不量化 |
| FP16 | FP16 | FP16 | BF16检查点转换 |
| FP8 W8A8 | E4M3FN | 动态FP8 | 每输出通道权重缩放、每token激活缩放 |
| INT8 W8A8 | INT8 | 动态INT8 | 每输出通道权重缩放、每token激活缩放 |
| INT8 W8A16 | INT8 | BF16 | K方向group128，GEMM内解量化 |
| INT4 W4A16 | packed INT4 | BF16 | K方向group128，GEMM内解量化 |

六种精度均进行独立全模型生成以观察路由变化，但FP8/INT8/INT4仅量化MoE FFN，attention等仍为BF16，router为FP32。这些RTN量化并非校准后的AWQ/GPTQ模型。自由生成的路由变化同时包含生成前缀分歧；局部FFN误差和生成一致率不能当作任务准确率。

## GPU性能测量

使用真实层权重、真实输入、真实Top-6，不用随机权重代替。vLLM fused_experts_impl运行Shared和Routed，Shared是E=1的同精度路径。各精度单独占用一张B200，10次warmup、50次CUDA Event测量、两次全量重复；CUDA Graph结构复用时更新全部输入/路由/掩码设备buffer。

- **grouped**：Shared + 全部Routed + combine，作为整体性能主口径。Router/attention不计入。
- **split**：Shared + Hot + Cold + combine，以同一次调用的阶段event进行闭合计时。Hot/Cold各调用一次会重复公共工作，因此不等于原生grouped调用的成本分解。
- `e2e_us`是不插内部阶段标记的CUDA Graph耗时。`instrumented_us`是插入阶段标记后的耗时，时间占比以此为分母。
- `relative_l2`与捕获的原始BF16模型输出比较；保存均值、95分位与最大值。
- WNA16在初始核对中出现小幅图重放输出变化，独立解量化Torch参考核对通过。记录变化并以明确容差检查，不宣称逐位一致。
- GPU计时两遍不会逐位一致，保存全部样本和稳定性统计。未锁频，默认内核配置可能非最优；没有硬件counter证据不判定Memory/Compute-bound。
- Shared权重合计33MiB，每个Routed 16.5MiB（BF16）。有效FLOPs按三个GEMM计，Shared恒占25%，不包含SiLU等标量操作。

## 环境与输出

运行主机：`ee-tromokratis`，8张NVIDIA B200。本地镜像（避免占满scratch）：

`/home/mcl123/plena-outputs/decode_precision_20261016/`

远端运行到中途触及home个人配额，已将完整实验迁到数据盘：

`/data/models/mcl123-home-models-20260905/experiments/decode_precision_20261016/`

完成的分片经SHA校验后复用，未完成的层/请求组重跑。远端与本地通过rsync同步；原始记录保留在`capture_*`及`profiles/<precision>/repeat{1,2}`，报告在`report/`。

主要交付：

- `report/SUMMARY_ZH.md`：简明全部batch/精度主表。
- `report/Decode_Precision_Profile.xlsx`：可筛选的完整逐层/逐batch/逐步数据。
- `report/decode_windows.csv`：每窗口64专家token计数，可直接给后续DSE使用。
- `report/profile_points.csv`：全部测量点归纳；原始50次样本在profiles的jsonl.gz。
- `report/precision_route_comparison.csv`：真实生成的跨精度路由变化。
- `report/AUDIT.json`、`GPU_TESTS.json`、`ENVIRONMENT_MODEL.json`：完整性、GPU核对和模型来源证据。

## 复现

在包含research目录的工作目录执行；以下ROOT指远端实验目录：

```bash
ROOT=/data/models/mcl123-home-models-20260905/experiments/decode_precision_20261016
export TRITON_CACHE_DIR="$ROOT/runtime_cache/triton"
CAPTURE_PY=/home/mcl123/envs/plena-moe-capture/bin/python
CUDA_VISIBLE_DEVICES=0 "$CAPTURE_PY" -m research.moe_dispatch.decode_precision_profile.capture \
  --model /home/mcl123/models/deepseek-v2-lite-chat \
  --sources /home/mcl123/plena-inputs/prepared --out "$ROOT/capture_bf16" \
  --batches 2,4,8,16,1,32,64,96,128,256
```

FP16加`--dtype fp16`并更换输出目录。四种量化采集分别加`--expert-precision fp8_w8a8/int8_w8a8/int8_w8a16/int4_w4a16`。已完成分片通过SHA校验后可`--resume`。

```bash
CUDA_VISIBLE_DEVICES=2 bash research/moe_dispatch/decode_precision_profile/run_profile.sh bf16 "$ROOT"
```

依次把bf16替换为其余五种精度；也可分配不同GPU并行，每张GPU一个测量进程。脚本自动跑repeat1/2。默认所有26层、57请求组、5个采样步。

```bash
python -m unittest research.moe_dispatch.decode_precision_profile.test_contract -v
python -m research.moe_dispatch.decode_precision_profile.test_gpu \
  --model /home/mcl123/models/deepseek-v2-lite-chat \
  --capture "$ROOT/capture_bf16" --out "$ROOT/GPU_TESTS.json"
python -m research.moe_dispatch.decode_precision_profile.report --root "$ROOT" --out "$ROOT/report"
python -m research.moe_dispatch.decode_precision_profile.publish --report "$ROOT/report"
```

CPU报告环境需要numpy、matplotlib、openpyxl；GPU环境依赖本机固定的Torch/vLLM/Transformers版本，具体版本写入各份metadata。发布器只接受六精度采集及十二次GPU回放全部完成的结果。

额外的`kernel_profile.py`在第13层、BFCL第0组第1步，对B2/16/128/256和六精度保存真实CUDA kernel trace。它仅提供kernel名称与时长，不冒充Tensor Core占空比或HBM利用率；profiler注入开销下的时长不替换主测量值。
