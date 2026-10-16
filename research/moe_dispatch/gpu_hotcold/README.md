# DeepSeek-V2-Lite GPU Shared / Hot / Cold profiling

This is a real CUDA **post-router layer replay**, using the 135 captured held-out
routing windows already used by the MoE DSE. It does not use simulator latency,
generated routing histograms, or a new pretrained-model forward pass.

The route exporter preserves each token's six distinct expert IDs and captured
routing scores. BF16 weight and activation values are seeded random tensors,
with numerical verification against an independent PyTorch expert-loop
reference. Precision, GPU name, software, input hashes, numerical tolerances,
raw timings, and the original mixed-window provenance are recorded.

## Workload and execution scope

- Hidden dimension: 2048; 64 routed experts; Top-6.
- Shared FFN width: 2816. Routed FFN width: 1408.
- Gate/Up, SiLU product, Down, GPU token gather/alignment, routing weights, and
  output combination are included. Router GEMM/softmax/Top-K are excluded:
  the captured IDs and scores are supplied.
- Batch sizes: 2, 4, 8, 16, 64, 96, 128. Window counts: 27, 27, 27, 27, 9, 9, 9.
- Cold means `1 <= Me <= 2`; Hot means `Me > 2`; `Me == 0` is inactive.
- Shared accounts for 25% of useful GEMM operations by shape, **not necessarily
  25% of elapsed GPU time**.

The four measured implementations are:

| Path | Purpose | Timed work / limitations |
|---|---|---|
| `vllm_grouped` | Primary production-library baseline | Unquantized vLLM fused experts; runtime expert alignment included |
| `vllm_split` | Controlled Hot/Cold diagnostic | Two category-specific calls using expert maps; alignment and some common work repeated; not an attribution of the grouped path |
| `triton_grouped` | Independent grouped prototype | Expert grouping/tile maps prepared on CPU outside timing; GPU gather, grouped GEMMs, activation, combination included |
| `triton_split` | Prototype Hot/Cold diagnostic | Same numerical computation with changed launch/packing plan; not grouped-path attribution |

The Shared FFN is separate. Hot/Cold stage time contains category dispatch;
dispatch is not measured independently and must not be counted a second time.

## Timing contract

1. JIT compilation, allocation, and initialization occur outside measurements.
2. At least 10 warmup iterations and 50 raw samples per point.
3. Eager and CUDA Graph are separate modes. Two complete campaigns are retained.
4. Primary E2E latency has only outer CUDA events, with no internal markers.
5. A separate instrumented complete invocation has adjacent stage events. Its
   components and total are measured in the same replay, never isolated loops.
6. Empty categories measure exactly zero. Stage sums must close within event
   rounding. Instrumentation effects are reported against the plain invocation.
7. Repeated full-layer warm execution is used. There is no cache flush between
   components. These are not cold-cache HBM-only timings.
8. Profilers are separate from timing samples. Event timings cannot reveal
   Tensor Core occupancy or achieved DRAM bandwidth. Nsight replay duration
   cannot be substituted for unprofiled E2E latency.

## Run

Use Python 3.10+ with a BF16-capable CUDA GPU, matching PyTorch/Triton, and vLLM.
The validated local campaign records the exact installed versions in metadata;
there is no claim that every future library version has an identical API.

From the repository root:

```bash
python -m research.moe_dispatch.gpu_hotcold.traces \
  --output OUTPUT/input/heldout_routes.json

python -m research.moe_dispatch.gpu_hotcold.benchmark \
  --routes OUTPUT/input/heldout_routes.json --output OUTPUT/repeat1 \
  --warmup 10 --samples 50 --modes eager,graph \
  --paths vllm_grouped,vllm_split,triton_grouped,triton_split \
  --correctness all --torch-profiler

python -m research.moe_dispatch.gpu_hotcold.benchmark \
  --routes OUTPUT/input/heldout_routes.json --output OUTPUT/repeat2 \
  --warmup 10 --samples 50 --modes eager,graph \
  --paths vllm_grouped,vllm_split,triton_grouped,triton_split \
  --correctness all

python -m research.moe_dispatch.gpu_hotcold.report \
  --input OUTPUT/repeat1 OUTPUT/repeat2 \
  --traces OUTPUT/input/heldout_routes.json --out OUTPUT/report
```

The report has no CUDA dependency. Install `openpyxl` in its environment to
also produce `GPU_HotCold_完整结果.xlsx`. Markdown and CSV output do not depend
on Excel support. The trace exporter needs the original repository captures;
the exported input JSON and GPU benchmark are portable without the DSE loader.

Existing measurement directories are immutable: choose fresh destinations for
reruns. Report generation refuses incomplete repeats, altered route hashes,
missing numeric checks, changed populations, duplicate points, and missing
execution modes. `--allow-partial` is for integration debugging and marks the
report `PARTIAL`; it is not acceptable for the complete experiment claim.

## Output and aggregation

Each repeat writes:

- `metadata.json`: GPU, software, precision, cache policy, source hashes, scope.
- `measurements.jsonl`: all raw timing arrays and per-point summaries.
- `correctness.json`: every window/path numerical result and tolerance.
- `COMPLETE.json`: campaign completion receipt.
- Optional `torch_profiler/`: separate kernel/operator traces, not hardware
  utilization counters or primary latency samples.

The report writes:

- `main_latency.csv`: uninstrumented E2E means; pooled median and population
  standard deviation (including different-window variation).
- `split_breakdown.csv`: same-invocation Shared/Hot/Cold/combine arithmetic
  means and fractions. Grouped-path stage times remain in `grouped_points.csv`.
- `controlled_ratios.csv`: paired split/grouped and prototype/vLLM ratios,
  always based on **uninstrumented** full-invocation times.
- `instrumentation.csv`: timing change caused by the instrumented execution.
- `repeat_stability.csv`: actual differences between the two campaigns.
- `workload.csv`: captured expert populations and useful operation fractions.
- `point_summary.csv`, `split_points.csv`, `correctness.csv`, `environment.csv`.
- `REPORT.md`, optional Excel workbook, and `REPORT_AUDIT.json`.
- When separate profiler CSVs are present in the report directory, Excel also
  includes `counter_summary`, `routed_gemm_summary`, `kineto_launch_summary`,
  and `counter_kernels` sheets. They retain separate replay semantics and their
  source CSV hashes are audited; report generation does not rewrite them.

Each Batch mean equally weights its windows and repeats. Main latency's
`7-Batch GM` equally weights the seven Batch means, independent of the unequal
window counts. Latency ratios pair the same window/repeat, take within-Batch
GM, then equal-Batch GM. Component tables use arithmetic means so timings add.
The final component row uses equal-Batch AM of times and, separately,
equal-Batch AM of percentages; its percentages are **not** the ratio of those
AM times. No geometric mean is taken over zero-valued category timings.

Do not label Hot domination as Compute-bound or Cold domination as saturated
HBM. A hardware bottleneck classification requires separately collected and
interpreted counters. The report deliberately makes no automatic Mem/Compute
classification or guessed crossover Batch.

## Verification

```bash
python -m unittest research.moe_dispatch.gpu_hotcold.test_traces \
  research.moe_dispatch.gpu_hotcold.test_report -v
```

Unit-test synthetic fixtures are not benchmark workloads. GPU numerical
checks and raw measurement coverage are separately audited before the complete
report is emitted. GPU timings are expected to fluctuate; repeat agreement is
reported quantitatively, never claimed bitwise-identical.
