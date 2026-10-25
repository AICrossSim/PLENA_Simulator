# Multi-chip PLENA model

`analytic_models.distributed` estimates TTFT and decode throughput of a decoder
served by several PLENA chips. Every device is one PLENA chip: `PerfModel`
compute plus a memory system from [`analytic_models.stacked_dram`](../stacked_dram/README.md)
(a fixed bandwidth such as an HBM baseline, or a 3D DRAM stack). The devices
are connected by a hierarchical network and run a TP/EP/DP/PP/CP plan.

## Provenance

The network side is DeepStack (Mo et al., *DeepStack: Facilitating Co-Design
Exploration of 3D DRAM-Stacked Accelerators for Distributed LLM Inference*,
[arXiv:2604.04750](https://arxiv.org/abs/2604.04750), MICRO 2026;
[tile-ai/DeepStack](https://github.com/tile-ai/DeepStack)), vendored as the
`DeepStack` git submodule and imported, not copied:

| DeepStack module | used for |
| --- | --- |
| `mosaic.noc.custom_profile`, `mosaic.noc.noc_topo` | the three-level NoC hierarchy (switch, ring, chain, mesh, torus, all-to-all levels), routing and the two-stage time: longest hop latency plus busiest link's bytes over its bandwidth |
| `mosaic.collectives` | all-reduce (best of recursive doubling, ring, Rabenseifner, double tree and all-to-all; ring for non-power-of-two groups), all-gather and the MoE token all-to-all |
| `mosaic.noc.traffic_matrix` | rank grouping (TP, EP, SP, CP, DP, PP from the innermost level out) and pipeline transfers |
| `mosaic.noc.route_stats` | NoC energy from per-level pJ/bit coefficients |
| `mosaic.utils` | MoE routing statistics: busiest-rank expert counts and the closed-form imbalance estimate |
| `mosaic.data.routing` | the `qwen3_235b` and `deepseek_v3` routing traces |

`_deepstack.py` is the only place that touches `sys.path`; it imports source
modules only and fails if any of DeepStack's bundled reference binaries gets
loaded. Run `git submodule update --init DeepStack` once, or point
`PLENA_DEEPSTACK_ROOT` at another DeepStack checkout. DeepStack's extra
dependency, SciPy, is in `pyproject.toml`.

The compute side is PLENA's: the `PerfModel` calls of `llama_model.py` and
`gpt_oss_model.py` with per-device shapes, priced against DRAM traffic by
`stacked_dram.price_stage`.

## What one layer costs

On each device, a transformer layer is one roofline stage (compute against
its DRAM traffic, under `--overlap stage-roofline` or `serial`) followed by
its collectives (`--comm-overlap none`) or overlapped with them (`full`).

| part | per-device work | collective |
| --- | --- | --- |
| attention | heads and KV heads split over TP (KV heads replicated when TP exceeds them), KV sequence split over CP; QKV projection cycles split evenly over TP | TP all-reduce of the output; with CP, all-reduce of the partial outputs (decode) or all-gather of K and V (prefill) |
| dense FFN | intermediate size split over TP | TP all-reduce of the output |
| routed experts | the busiest EP rank's tokens and token-expert pairs (see below) | token dispatch and combine across each EP group; TP all-reduce of the expert outputs when experts stay tensor parallel |
| RMSNorm, residual | replicated across TP | none |
| embedding, LM head | first stage (prefill lookup) and last stage; vocabulary split over TP | none: sampling reduces the logits locally |

The layer compositions are PLENA's: `"llama"` follows `llama_model.py` (dense
full-attention layers, no LM head) and `"moe"` follows `gpt_oss_model.py`
(full or sliding-window attention per layer, a residual after the MLP, routed
experts or a dense FFN per layer, the LM head in prefill). A config with routed
experts, sliding-window layers or `model_type == "gpt_oss"` uses `"moe"`.
`--include-lm-head` adds the LM head to every decode step.

## Parallel plan

`ParallelPlan(tp, ep, dp, pp, cp, moe_tp_mode)` uses `tp * ep * dp * pp * cp`
devices, which must equal the NoC profile's device count. Following DeepStack's
decode driver:

* attention, dense FFN, norms and pipeline transfers see `dp * ep`
  data-parallel ranks;
* MoE layers see `ep` expert-parallel ranks. With `moe_tp_mode="replace"`
  (default, DeepStack's `"replace_only"`) the TP ranks become EP ranks too,
  `ep' = tp * ep` (capped at the expert count, the rest staying tensor
  parallel), and the TP all-reduce after attention stands in for the
  reduce-scatter and all-gather around the experts. With `"keep"` the experts
  stay tensor parallel inside each EP rank;
* pipeline stages hold `ceil(layers / pp)` contiguous layers.

## MoE routing

`--routing` selects how tokens reach experts:

* `balanced`: `PerfModel.mlp_moe`'s assumption, every expert serves
  `ceil(tokens * top_k / experts)` tokens. One device reproduces `mlp_moe`
  exactly. For the dispatch, a token's choices are spread over EP ranks.
* `random`: experts drawn uniformly per token (fixed seed).
* `qwen3_235b`, `deepseek_v3`: DeepStack's packaged traces (prefill trace for
  prefill, decode trace for decode); the model's expert count and top-k must
  match the trace.

Below 4096 micro-batch tokens the busiest rank's expert counts come from the
routed tokens; above, from DeepStack's closed-form imbalance estimate. The
all-to-all switches to the estimate at 8192 tokens. Both thresholds are
DeepStack's. Shared experts are not modelled.

## Pipelining and outputs

As in DeepStack, the batch is split into `pp` micro-batches of
`ceil(batch / pp)` sequences. A pass of one micro-batch through the stages has
a *period* (slowest stage plus one stage-to-stage transfer) and a *latency*
(all stages plus `pp - 1` transfers). The estimate reports:

* `ttft_seconds`: one micro-batch's prefill latency plus its first decode step;
* `tps`: one micro-batch of tokens per decode period, summed over the output;
* `tps_per_sequence`: one token every `pp` periods;
* the bottleneck stage of the prefill pass and of the first decode step, with
  its compute, memory and per-collective times;
* weights and KV cache of the fullest stage's devices against the memory's
  capacity (sliding-window layers keep only their window);
* NoC energy per decode token: every collective's traffic spans all stages
  and each stage runs its own layers once per period.

With one device every collective is free and the estimate equals the
single-chip one: `stacked_dram.estimate_decoder_latency` for the llama
composition and, with unbounded bandwidth, `llama_model.py` and
`gpt_oss_model.py`.

## NoC profiles

A profile is DeepStack's three-level `make_custom_profile` input in JSON:

```json
{
  "schema_version": 1,
  "kind": "noc_hierarchy",
  "name": "...",
  "provenance": {"source": "..."},
  "layers": {
    "L3": {"kind": "switch", "shape": [1, 1], "hop_latency_ns": 0, "link_bandwidth_gbytes_per_s": 50},
    "L2": {"kind": "ring", "shape": [1, 4], "hop_latency_ns": 1000, "link_bandwidth_gbytes_per_s": 50},
    "L1": {"kind": "switch", "shape": [1, 8], "hop_latency_ns": 200, "link_bandwidth_gbytes_per_s": 100,
           "switch_center_in_gbytes_per_s": 800, "switch_center_out_gbytes_per_s": 800}
  },
  "port_spread": "even",
  "energy_pj_per_bit": {"l1": 1.0, "l2": 2.0, "l3": 3.0}
}
```

`L1` is the innermost level; ranks fill it first, so TP groups land on the
fastest links. The profiles in `examples/` are illustrative only: two follow
DeepStack's public `h200x32` GPU preset, two its `examples/custom_noc.py`, and
all use DeepStack's example energy coefficients. None describes PLENA hardware.

## Command line

```bash
# Levels, bandwidths and device count of a NoC profile
python -m analytic_models.distributed describe-noc \
    --noc analytic_models/distributed/examples/illustrative_ring_switch_32.json

# Llama-3.1-70B on 32 chips: TP inside a node, PP across nodes, 3D DRAM per chip
just distributed-estimate llama-3.1-70b analytic_models/distributed/examples/illustrative_gpu_cluster_32.json \
    --profile analytic_models/stacked_dram/examples/fictional_stacked_dram.json --tp 8 --pp 4

# gpt-oss-20b on 8 chips with expert parallelism and random routing
just distributed-estimate gpt-oss-20b analytic_models/distributed/examples/illustrative_gpu_node_8.json \
    --fixed-bandwidth-gbs 1000 --tp 2 --ep 4 --routing random --json

# Tests
just test-distributed
```

## Python

```python
from analytic_models.distributed import ModelSpec, ParallelPlan, estimate_distributed, load_noc_profile
from analytic_models.performance.perf_model import PerfModel, load_hardware_config_from_toml
from analytic_models.stacked_dram import HbmStoragePrecision, load_memory_profile

perf = PerfModel(load_hardware_config_from_toml("plena_settings.toml"), "analytic_models/performance/customISA_lib.json")
result = estimate_distributed(
    ModelSpec.from_json("PLENA_Compiler/doc/Model_Lib/llama-3.1-70b.json"),
    ParallelPlan(tp=8, pp=4),
    load_noc_profile("analytic_models/distributed/examples/illustrative_gpu_cluster_32.json"),
    perf,
    HbmStoragePrecision.from_settings("plena_settings.toml"),
    load_memory_profile("analytic_models/stacked_dram/examples/fictional_stacked_dram.json"),
    batch_size=16,
    input_seq_len=2048,
    output_seq_len=1024,
)
print(result.ttft_seconds, result.tps, result.warnings)
```

## Limitations

* Decode runs one micro-batch per stage at a time; there is no continuous
  batching and no prefill/decode overlap.
* Every MoE layer routes the same tokens (the first rows of a trace), as in
  DeepStack.
* Pipeline transfers send the full activation from every TP rank, as in
  DeepStack's decode driver.
* Context parallelism is modelled for dense full-attention models only.

## Citation

```bibtex
@inproceedings{mo2026deepstack,
  title     = {DeepStack: Facilitating Co-Design Exploration of 3D DRAM-Stacked Accelerators for Distributed LLM Inference},
  author    = {Zhiwen Mo and Guoyu Li and Hao (Mark) Chen and Yu Cheng and Zhengju Tang and Qianzhou Wang and Lei Wang and Shuang Liang and Lingxiao Ma and Yuxiao Guo and Wayne Luk and Jilong Xue and Hongxiang Fan},
  booktitle = {59th IEEE/ACM International Symposium on Microarchitecture (MICRO)},
  year      = {2026},
  note      = {arXiv:2604.04750}
}
```
