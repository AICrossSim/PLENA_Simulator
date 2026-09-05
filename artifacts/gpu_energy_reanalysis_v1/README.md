# Legacy GPU energy reanalysis

`summary.json` is an **offline approximation**, not a new B200 measurement or
an exact corrected batch-energy baseline. The original archive remains unchanged:

```
/scratch/shared/mcl123/plena/artifacts/gpu/AGENTIC_NEMOTRON_B200_20260903
```

The legacy recorder integrated unsorted samples without clipping the trial
window. It also omitted the batch start/end timestamps and the time of the
actual NVML power query. This reanalysis sorts valid readings, interpolates
power at the window boundaries, and integrates only from the earliest observed
request arrival to the latest request finish in each batch. It requires power
samples on both sides of the window and never extrapolates. Missing batch/query
timestamps cannot be recovered by sorting.

For B16, across 60 measured batch trials:

| Quantity | Value |
| --- | ---: |
| Original archived median batch integral | 186.694929 J |
| Approximate observed-request-window median | 180.794972 J |
| Trials with nonmonotonic archived sample order | 12 / 60 |

The second number must remain labelled approximate; it must not replace the
first as a purported newly measured, exact GPU energy result. Agentic campaign
exports retain the original number with status
`archived_legacy_integration_requires_recapture`. Neither quantity is PLENA
chip power or energy.

## Reproduce the offline artifact

Run from the Simulator repository root; no GPU is needed:

```bash
nix develop --no-write-lock-file --command python -m analytic_models.performance.gpu_energy \
  --campaign-root /scratch/shared/mcl123/plena/artifacts/gpu/AGENTIC_NEMOTRON_B200_20260903 \
  --output artifacts/gpu_energy_reanalysis_v1/summary.json
```

Equivalent recipe:

```bash
nix develop --no-write-lock-file --command just gpu-energy-reanalysis \
  /scratch/shared/mcl123/plena/artifacts/gpu/AGENTIC_NEMOTRON_B200_20260903
```

The reader verifies `latency_raw.jsonl` and `power_raw.csv` against the original
`SHA256SUMS` before analysis, records their SHA-256 values in the result, and
refuses to write inside the raw archive. The artifact covers 1,860 measured
batch-sweep trials; summary rows provide global and benchmark/group aggregates.

## Capture a new baseline with the corrected recorder

This command is a reproduction procedure, **not evidence of a completed new
capture**. Run on the B200 host from a checkout of this Simulator revision, with
the same checkpoint and archived `samples.json`. Use a fresh output directory.

Keep the GPU software stack recorded in the raw archive's `environment.txt`
and `scripts/run_timing.sh`: Python 3.11.15, PyTorch 2.9.0 with CUDA 12.8,
vLLM 0.12.0, FlashInfer 0.5.3, and `nvidia-ml-py==13.610.43` (the distribution
providing the `pynvml` import). The archived `package.pynvml=N/A` field refers
to the distribution name; the installed provider is `nvidia-ml-py`.

The maintained module also imports this repository's analytic package, which
needs `toml` and Pydantic. The archived environment includes Pydantic 2.13.4 but
does not list `toml`; add `toml==0.10.2` to that environment. Do not replace the
GPU stack with the CPU-only Nix test environment for capture.

Adjust these host paths to the mounted checkpoint/archive and an unused output
directory. Preserve the original launcher's runtime/cache environment if its
packages are supplied through the runtime `PYTHONPATH` overlay:

```bash
CAPTURE_PYTHON=/home/mcl123/envs/plena-nemotron-vllm/bin/python
CAPTURE_RAW=/scratch/shared/mcl123/plena/artifacts/gpu/AGENTIC_NEMOTRON_B200_20260903
CAPTURE_MODEL=/run/user/1704375/plena-agentic-hf/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/snapshots/ce1b118ae66ec705d02c241525192832eb045fd3
CAPTURE_OUTPUT=/path/to/new-b200-energy-capture
CAPTURE_GPU=0

"$CAPTURE_PYTHON" -m pip install 'toml==0.10.2'
export VLLM_USE_FLASHINFER_MOE_FP4=1
export VLLM_FLASHINFER_MOE_BACKEND=throughput
export VLLM_ENABLE_V1_MULTIPROCESSING=0

"$CAPTURE_PYTHON" -m analytic_models.performance.gpu_timing_campaign \
  --model "$CAPTURE_MODEL" \
  --samples "$CAPTURE_RAW/samples.json" \
  --output-dir "$CAPTURE_OUTPUT" \
  --physical-gpu "$CAPTURE_GPU" \
  --warmup 5 --measurements 20 --power-interval-ms 20
```

The new recorder resolves the physical NVML GPU to its UUID, binds CUDA to that
UUID before CUDA/vLLM initialization, and verifies CUDA/NVML identity before
and after model loading. It uses one background power reader, records each
power query's start/end and midpoint, waits for a sample after batch completion,
and integrates only over the recorded batch boundaries. Trial labels do not
select the integration window. If CUDA was already initialized on an incompatible
device selection, start a new process.

Missing power remains `N/A`; a short batch with no internal memory sample has
an `N/A` observed memory peak. Captured joules remain sampled NVML estimates,
not an external power-meter reading. The new raw directory must be archived
with its metadata, environment, scripts/revision and checksums before it can
serve as a replacement baseline. This repair did not perform that GPU run.

## CPU-only regression checks

```bash
nix develop --no-write-lock-file --command python -m pytest -q \
  analytic_models/performance/test_gpu_energy.py
```

These tests cover sorting, clipping, bracketing, conflicting readings,
single-thread sampling, NVML failures, CUDA UUID binding, startup cleanup,
missing telemetry, short windows, and immutable legacy reanalysis. They use
mock CUDA/NVML interfaces and do not measure GPU performance.
