# Complementary systolic arrays for MoE decode

Cycle-level simulator, design-space search and measurement code for MoE decode layers on one or two systolic arrays
that share HBM, on-chip SRAM and a weight port. The RTL of the dispatchers, arrays, weight port and engine control
plane is in [PLENA_RTL_Internal](https://github.com/AICrossSim/PLENA_RTL_Internal) (branch `moe-rtl`, folder `moe/`;
the repository may be private).

| Path | Contents |
|---|---|
| `sim/` | Simulator (`engine.py`, C++ sources in `src/`, built with g++ on first use into `$SIM_BUILD_DIR`) and the workload model (`workload.py`: layer dimensions, weight formats, routing traces, evaluation windows) |
| `dse/` | Design-space search per cell and organisation (`search.py`, `candidates.py`, `shapes.py`, `tune.py`, `organisations.py`), chip selection, frozen-chip and sealed-group evaluation, bootstrap intervals, result table, dispatch-policy oracle |
| `mech/` | Mechanism data of the final chips: stall partitions, weight-port ladders, event timelines |
| `gpu/` | GPU side (one NVIDIA B200, `gpu/requirements.txt`): routing captures, MoE-layer and decode-step baselines, accuracy of the weight formats |
| `scripts/` | `prepare_data.sh` (data root from the captured traces, checked against `traces.sha256`), `run_search.sh`, `run_final.sh`, `run_mech.sh` |
| `tests/` | Unit tests and simulator reference values (`tests/data/`); `test_reproduction.py` checks the pipeline against reference results |

## Running

Python 3.11 with numpy and pytest, and g++ with OpenMP, for everything outside `gpu/`. Run from the repository root:

```bash
python -m pytest complementary_arrays/tests -q
bash complementary_arrays/scripts/prepare_data.sh DATA TRACES SEALED
bash complementary_arrays/scripts/run_search.sh DATA OUT      # JOBS, THREADS, ENGINE_OPTIONS
bash complementary_arrays/scripts/run_final.sh DATA OUT
bash complementary_arrays/scripts/run_mech.sh DATA OUT
```

The reproduction tests run when `COMPLEMENTARY_ARRAYS_DATA` (data root) and `COMPLEMENTARY_ARRAYS_REFERENCE` (reference
results in the output layout) are set, and are skipped otherwise. The routing traces come from `gpu/capture/run.sh`;
model weights and the request datasets are not included.
