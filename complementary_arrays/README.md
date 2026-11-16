# Complementary systolic arrays for MoE decode

Cycle-level simulator, design-space search and measurement code for MoE decode layers on one or two systolic arrays
that share HBM, on-chip SRAM and a weight port. The RTL of the dispatchers, arrays, weight port and engine control
plane is in [PLENA_RTL_Internal](https://github.com/AICrossSim/PLENA_RTL_Internal) (branch `moe-rtl`, folder `moe/`;
the repository may be private).

| Path | Contents |
|---|---|
| `sim/` | Simulator (`engine.py`, C++ sources in `src/`, built with g++ on first use into `$SIM_BUILD_DIR`) and the workload model (`workload.py`: layer dimensions, weight formats) |
| `dse/` | Decode-step snapshots from the routing captures, design-space search per cell and organisation, chip selection, per-batch designs, evaluation, bootstrap intervals, result table |
| `mech/` | Mechanism data of fixed chips: multiplier-cycle accounting, single-layer timeline, ablations |
| `ppa/` | Area and energy per MoE layer from the block model and the simulator's activity counters |
| `xcheck/` | Closed-loop HBM cross-check of sampled windows with Ramulator 2 |
| `gpu/` | GPU side (one NVIDIA B200): routing captures, B200 expert-device measurements, accuracy of the weight formats |
| `scripts/` | `prepare_data.sh`, `run_search.sh`, `run_final.sh`, `run_mech.sh` |
| `tests/` | Unit tests and simulator reference values (`tests/data/`); `test_reproduction.py` checks the pipeline against reference results |

## Simulator

`simulate()` runs layer windows (the routed experts of B tokens, up to 1024) and returns one row of counters per
window; `counters()` names them and `trace()` gives a per-cycle record of one window. The chip is described by its
array shapes and an `Environment`:

- Dispatch policy (`policy`): 2 regret list scheduling, 13 two-ended dispatch (hot and cold end of the task list),
  14 band dispatch, 15 supply-aware two-ended dispatch (each context weighs both end tasks by HBM and weight-port load,
  tie tolerance `p15_tie`), 16 band dispatch that re-plans whenever a task finishes, with costs scaled by each array's
  measured progress (`p14_steal` sets the steal rule of 14 and 16).
- Weight records: `scale_group` 32 gives NVFP4-style records (W4G32P); -1 gives 8-bit records with block scales kept
  outside (FP8). `redeliv_packed=1` keeps re-read panels in stored form, re-reading and decoding them on every token
  pass (needs `port_mode=1`).
- Converter (`converter`): one chip-wide converter (`shared`), one per array (`per_array` at `converter_rate`, or
  `matched` to the array's intake), or `unlimited`.
- Shared expert: `shared_expert=False` removes it; `shared_place` puts it on array 0 or 1, or splits it into two
  halves, one per array.
- Ablations: `prefetch=False` (no cross-phase HBM prefetch) and `port_arbitration="fcfs"` (first-come first-served
  weight port).

`counters()` reports per array what it does every cycle (`array_state_cycles` over `ARRAY_STATES`), idle-array cycles
(`idle_while_other_works*`), pass and MAC statistics, and per context the HBM, port, decode and packed re-read
traffic.

HBM is an ideal stream by default. With `ram_config` (a Ramulator 2 configuration; `xcheck/hbm2_2134.json` is the one
used) `simulate()` serves every HBM request through Ramulator instead. This uses a separate build linked against
libramulator in `$RAMULATOR_LIB`; the default build does not need Ramulator. `trace()` and a split shared expert are
not available in this mode.

## Pipeline

`dse/snapshots.py` turns the per-request routing captures into decode-step snapshots: 64 per batch size, requests drawn
in proportion to their length at uniform positions, split 40/40/20 into development, evaluation and sealed requests.
It also serves the windows that the search and the evaluation simulate.

`dse/search.py` searches one organisation of one cell in stages:

- `top10`: rule candidates;
- `hetopt` and the sharded `hetx<i>` (pair only; the second starts from packed re-reads);
- `prior`: earlier designs converted to the cell's format and promoted to the whole multiplier budget;
- `retune<i>`.

Each stage probes the candidates, sweeps the dispatch policies, tunes the SRAM split, prefetch, slots and re-read
format, and measures on the development snapshots. `select.py` keeps the fastest valid chip on the geometric mean over
batch 32–128, and `perbatch.py` designs each organisation for one batch size. `evaluate.py` runs chip sets on the
evaluation, sealed and batch-256/512 snapshots. `bootstrap.py` gives two-stage intervals (requests resampled within
source, snapshots redrawn and re-simulated), and `export.py` writes the result table.

`mech/stall.py` splits multiplier-cycles into useful work, empty rows, waits and idle time. `mech/timeline.py` traces
the median layer of the median batch-128 snapshot, and `mech/ablate.py` changes one mechanism on fixed chips.

`ppa/` computes chip area and energy per MoE layer from the block constants (`model.py`: logic areas after synthesis in
ASAP7, SRAM and HBM energies from the literature) and the activity counters that `mech/` records. `xcheck/` samples
evaluation windows (`sample.py`), runs them through Ramulator HBM2 (`run.sh`: 512 channels at 2,134 MT/s) and
summarizes cycles against the ideal stream (`summary.py`).

## GPU side

The GPU side runs on one NVIDIA B200 in separate environments (`requirements*.txt`).

- `capture/run.sh` builds the request pool (six sources × 250 requests from public datasets, split 40/40/20 into
  development, evaluation and sealed sets). It records in vLLM the routed experts of every generated token with
  reference and INT4 weights, checks this against a transformers capture path, and builds the 27,000-request pool for
  batch 256 and 512.
- `bench/run.sh` measures the B200 as an expert device on the same requests: MoE-layer wall time from decode-only
  streaming steps with the MoE or the shared expert zeroed, in vLLM and SGLang with FP8 and NVFP4 expert weights,
  kernel times, board energy and the read bandwidth. `bench/report.py` gives the fastest B200 time per format.
- `accuracy/run.sh` measures accuracy with quantized expert weights: emulated formats on standard tasks
  (transformers), GPU kernels on GPQA, MATH-500, AIME, LiveCodeBench, BFCL and SWE-bench Verified through a vLLM
  server, and emulated 8- and 4-bit KV caches. `accuracy/summarize.py` writes the tables.

Model weights, request texts, generations and captured routes are not included.

## Running

Python 3.11 with numpy and pytest, and g++ with OpenMP, for everything outside `gpu/`. Run from the repository root:

```bash
python -m pytest complementary_arrays/tests -q
bash complementary_arrays/scripts/prepare_data.sh DATA CAPTURES
bash complementary_arrays/scripts/run_search.sh DATA OUT      # CELLS, JOBS, THREADS, HETX_SHARDS, SEEDS
bash complementary_arrays/scripts/run_final.sh DATA OUT CAPTURES
bash complementary_arrays/scripts/run_mech.sh DATA OUT
```

`prepare_data.sh` builds the snapshot files from the captures of `gpu/capture/run.sh` and checks them against
`scripts/snapshots.sha256`. `SEEDS` names directories of earlier designs for the `prior` stage. The reproduction tests
run when `COMPLEMENTARY_ARRAYS_DATA` (snapshot data root) and `COMPLEMENTARY_ARRAYS_REFERENCE` (reference results with
`final/`, `search/` and `eval/final/`) are set, and are skipped otherwise. The closed-loop simulator test runs when
`RAMULATOR_LIB` is set.
