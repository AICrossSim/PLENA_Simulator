# PLENA Pre-RTL GPU Follow-up Prompt

把下面整段交给 GPU 服务器上的独立 Codex session。该 session 不应假设知道本项目的
历史；所有结论必须由实际文件、命令和 profiler 报告支撑。

```text
你在一台有空闲 NVIDIA B200/GB200 GPU 的服务器上工作。请完成 PLENA pre-RTL
最后一轮 GPU 数据采集。先只读检查 GPU、磁盘、现有代码和已有结果；只使用一张
完全空闲的 GPU（utilization=0、无 compute PID）。若实验过程中出现其他进程，丢弃
受污染结果并等待，不要 reset GPU，也不要杀别人的进程。

已有 campaign 已经完成，禁止无意义重跑：
- Nemotron NVFP4 完整模型 latency、routing、4 份 NSYS；
- Nemotron Prefill/Decode 的 Mamba、Attention、MoE 共 6 份 NCU；
- KDA B1/S2048、Decode B1/B8 六阶段 NCU、数值对拍和 projection layout。

如果本机已有上一轮 campaign，先把它的绝对路径记为 `PROFILE_ROOT`，读取其中的
manifest/REPRODUCE，并复用环境、checkpoint 与源码。不要假设某个用户名或 home 路径。

任务 A（必须）：Nemotron Mamba persistent-state mixed precision
1. 固定官方模型 revision ce1b118ae66ec705d02c241525192832eb045fd3；优先复用已经
   下载的真实 NVFP4 checkpoint 和真实 Mamba layer 权重/输入。若只能使用固定随机
   BF16 权重，必须在结论中明确标为 numerical microbenchmark，不能冒充模型精度。
2. 以 FP32 recurrent state 为 golden，比较 FP16、BF16、MX8-B128。MX8-B128 表示
   每 128 个 state values 一组独立 scale；记录 scale dtype、舍入、饱和和 zero policy。
3. 至少测 S=2048、8192、32768；每个 prefill 后继续 decode 512 step。使用 3 个固定
   seed，保存每个 chunk/每 32 个 decode step 的误差轨迹，而不只保存最终一个数。
4. 每个 case 报告 output/state max abs、mean abs、relative L2、cosine、NaN/Inf、state
   动态范围、饱和率，以及 state storage 和读写 bytes。FP32 golden 必须逐次重建，不能
   被前一个低精度 case 污染。
5. 若 S32768 OOM，可用相同 recurrent state 连续喂 chunk，但必须证明与一次性 FP32
   路径等价，并记录 chunk size。不要悄悄缩小 heads/state_dim。

任务 B（必须）：Kimi K3 的 MLA 与 LatentMoE 代表层 profiling
1. 固定 moonshotai/Kimi-K3 revision
   9f62e4e9fffbd0a83ddd60e1c209d828994b3569，只下载 config/custom code/tokenizer；不要
   下载完整 Kimi K3 权重。按官方 config 实例化真实 shape、固定随机 BF16 权重。
2. 分别跑一个官方 MLA mixer 和一个官方 LatentMoE/FFN layer。先做小 shape CPU/
   PyTorch reference 对拍，再跑真实 shape；输出必须 finite，记录输入输出 shape/dtype。
3. Cases：Prefill B1/S128、B1/S2048；Decode context=2048 的 B1 和 B8。普通 latency
   使用 20 warmup + 100 measurements，报告 median/P95/peak memory。
4. 加 NVTX 分段。MLA 至少分 q_a、q_norm、q_b、kv_a、kv_norm、kv_b、RoPE、attention
   core、KV-cache read/write、out projection；LatentMoE 至少分 router/top-k、shared
   expert、routed expert projection、activation、down projection、weighted combine。
5. 每类代表 case 采 NSYS；对 Prefill B1/S2048、Decode B1、Decode B8 采分阶段 NCU。
   保存 kernel calls/time、DRAM read/write、L2 sectors、SM/memory throughput、achieved/
   theoretical occupancy、registers/thread、grid/block。unsupported metric 写 N/A，不能写0。
6. 明确说明使用的是随机单层权重，因此这些数据可校准 shape/traffic/timing，不能测
   Kimi 语言质量或真实 routing。不要声称完整 Kimi 已运行。

任务 C（先离线，必要时才重跑）：Nemotron Prefill Mamba 分阶段流量
1. 先从已有 Nemotron Prefill Mamba .ncu-rep/raw CSV 按 kernel 名和 NVTX 离线拆分：
   in projection、conv1d、dt/exp、state update/output、gate/norm、out projection。
2. 对每段汇总 kernel count/time/DRAM R/W/L2 sectors。只有现有报告无法可靠归类时，
   才在空闲 GPU 上重跑一个 Prefill S128 Mamba NCU；写明为什么必须重跑。
3. 目标是解释 PLENA logical model 与 B200 Prefill Mamba physical read 的 1.678x 差距，
   不要把 GPU physical/logical ratio直接乘到 PLENA 周期上。

通用要求：
- 固定并记录 GPU UUID、driver、CUDA、PyTorch、Transformers、Triton、FLA、NCU、NSYS、
  模型和源码 commit、完整命令、环境变量和随机 seed。
- 采集前后保存 nvidia-smi；任何并发污染、fallback、OOM、NaN 或不支持的 metric 必须
  明确写出，不能删除失败证据或用估算值代替实测值。
- 不修改 PLENA Compiler/Simulator/RTL 仓库；采集脚本和结果放在新的时间戳目录：
  `$HOME/plena-profiles/formal-runs/pre_rtl_followup_<UTC timestamp>/`
- 输出 machine-readable CSV/JSON、原始 .ncu-rep/.nsys-rep、README/REPRODUCE、manifest、
  SHA256SUMS。最后生成 tar.gz 和独立 .sha256，并执行 sha256sum -c、gzip -t。
- 最终回答先给完成/缺失表，再给关键数字、限制、结果绝对路径、归档大小和 SHA256。
```
