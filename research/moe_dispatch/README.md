# Heterogeneous MoE dispatch: analytical execution model

Research branch: `research/moe-heterogeneous-dispatch`.

This prototype compares `6`, `3+3`, and `4+2` M-lane organizations under equal
aggregate arithmetic, SRAM, bank and supply budgets. It consumes explicit
private-memory plans from the **pinned Compiler submodule** and executes complete
expert gate/up/activation/down/retirement/combine flows on one event clock.

Implemented: bounded dynamic admission before weight fetching; whole-expert
ownership; optional paired-column splitting with charged Z exchange; finite
prefetch slots, shared HBM credits and round-robin arbitration; X reuse across N
bands; private SRAM/accumulator banks and ordered K updates. Descriptor selection
and copies are charged. The current predictor is a shape/service heuristic,
**not an online-trained predictor**.

This is a standalone analytical crate, not the native transactional-emulator
backend. Router/attention latency, pretrained weight execution, Ramulator,
hardware calibration and RTL are outside this version. A separate Python audit
checks small numerical workloads against byte-addressed SRAM; timed Rust events
do not execute those tensors. See [METHODS.md](METHODS.md).

## Reproduce from the Simulator repository root

Use Python 3.10+ and Rust supporting edition 2024 (1.85+).
The planner/runner need only Python's standard library; NumPy is for numerical
auditing. Initialize only the required Compiler submodule:

```bash
git submodule update --init PLENA_Compiler
python3 -m venv /tmp/moe-dispatch-venv
. /tmp/moe-dispatch-venv/bin/activate
python -m pip install -r research/moe_dispatch/requirements.txt
export CARGO_TARGET_DIR=/tmp/moe-dispatch-target
cargo test --locked --manifest-path research/moe_dispatch/rust/Cargo.toml
cargo build --locked --release --manifest-path research/moe_dispatch/rust/Cargo.toml
python -m unittest discover -s PLENA_Compiler/research/moe_dispatch -p 'test_*.py' -v
python -m unittest discover -s research/moe_dispatch -p 'test_*.py' -v
python research/moe_dispatch/numerical.py --output /tmp/moe-dispatch-numerical.json
```

Run the 12 default-policy points, twice each, and check **entire raw reports**
against the frozen references (not just rounded latency):

```bash
python research/moe_dispatch/run_experiments.py --suite matrix \
  --filter '__g4__dynamic__base$' --workers 2 --output /tmp/moe-dispatch-replay
python research/moe_dispatch/verify_reference.py /tmp/moe-dispatch-replay
```

For the full search, omit `--filter`; then run `--suite toy` and `--suite
sensitivity` to separate synthetic shape examples and memory sensitivity. Every
point requires two identical repetitions and checks useful MACs, weight bytes,
finite capacity/credits and final drain. `--list` previews the selected cases.
Results include CSV summaries, per-core front-end observations, frozen source
hashes, plans/configs and complete JSON reports. Build artifacts and raw campaigns
are not committed.

The default workload bundle comes from the Compiler: captured DeepSeek-V2-Lite
BFCL last-token prefill routes, B2/B4/B8/B16. No model weights are downloaded.
Those batches are nested prefixes of one capture, not independent benchmarks.

During paired-repository development, `PLENA_DISPATCH_COMPILER` can select a
different `research/moe_dispatch` directory. The runner records that planner's
hash and snapshots its source; default review/reproduction uses the gitlink.

## Review map

| File | Responsibility |
|---|---|
| `rust/src/main.rs` | Admission, finite resources, DMA, bank service, execution and reports |
| `rust/src/plan.rs` | Physical tiling, X reuse and exact work/traffic accounting |
| `frontend.py` | Load the pinned Compiler without duplicating its implementation |
| `run_experiments.py` | Equal-budget matrix/toy/sensitivity sweeps and repeat checks |
| `numerical.py`, `test_*.py` | Separate payload audit and timing integration tests |
| `results/RESULTS.md` | Small result table, configuration, counterexamples and scope |

No existing native Simulator defaults, timing implementation or frozen reports
are changed by this branch. Earlier experimental worktrees remain separate.
