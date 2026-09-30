# Heterogeneous MoE dispatch: analytical execution model

Research branch: `research/moe-heterogeneous-dispatch`.

This prototype compares `6`, `3+3`, and `4+2` M-lane organizations under equal
aggregate arithmetic, SRAM, bank and supply budgets. It consumes explicit
private-memory plans from the **pinned Compiler submodule** and executes complete
expert gate/up/activation/down/retirement/combine flows on one event clock.

Default runtime: an 8-entry pending FIFO, one Current + one Next per core,
atomic whole-expert ownership, one prefetched tile per Next, stable DMA requests
under backpressure, shared credits released after SRAM landing, and ordered K
updates. X reuse, private SRAM, accumulator banks and result inboxes are retained.
Historical paired-column splitting remains under `runtime_fsm=false` only.
Descriptor selection and copies are charged. The predictor is a shape/service heuristic,
**not an online-trained predictor**.

This is a standalone analytical crate, not the native transactional-emulator
backend. Router/attention latency, pretrained weight execution, Ramulator,
hardware calibration and RTL are outside this version. A separate Python audit
checks small numerical workloads against byte-addressed SRAM; a new replay also
uses actual Rust DMA addresses, issue events and K commits to execute seeded
payloads. Large performance runs remain timing-only. See
[RUNTIME_FSM.md](RUNTIME_FSM.md) for the current contract;
[RUNTIME_POLICY_ABLATION.md](RUNTIME_POLICY_ABLATION.md) for the policy and
residual-capacity admission experiments;
[METHODS.md](METHODS.md) describes the historical controller.

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

Run B2/B4/B8/B16 × three organizations × four control modes, twice each.
The complete raw report must match between repetitions:

```bash
python research/moe_dispatch/run_experiments.py --suite runtime \
  --workers 4 --output /tmp/moe-runtime-replay
```

The `matrix`, `toy` and `sensitivity` suites retain the old controller explicitly;
exact reproduction of archived references additionally requires their pinned
source/compiler versions. Do not overwrite or re-golden those reports. Every
point requires two identical repetitions and checks useful MACs, weight bytes,
finite capacity/credits and final drain. `--list` previews the selected cases.
Results include CSV summaries, per-core front-end observations, frozen source
hashes, plans/configs and complete JSON reports. Build artifacts and raw campaigns
are not committed.

The default workload bundle comes from the Compiler: captured DeepSeek-V2-Lite
BFCL last-token prefill routes, B2/B4/B8/B16. No model weights are downloaded.
Those batches are nested prefixes of one capture, not independent benchmarks.

## Policy and residual-capacity experiments

The runtime now defaults to stock-cycle arbitration. Optional controls cover
tail-only Next prefetch, late binding, Shared pinning, Compiler-supplied Current
depth protection, four-tier arbitration and one next-projection tile. The new
admission rules remain experimental; the results do not support enabling all
four by default. Historical suites select round-robin explicitly.

```bash
python research/moe_dispatch/run_experiments.py --suite runtime_policy \
  --workers 4 --repeats 2 --output /tmp/moe-runtime-policy
python research/moe_dispatch/run_experiments.py --suite surplus \
  --workers 4 --repeats 2 --output /tmp/moe-surplus
python research/moe_dispatch/audit_runtime_policies.py \
  --root /tmp/moe-surplus --expected-points 132
```

`credit_diagnostic` (36 points) and `surplus_credit_diagnostic` (132 points)
explicitly label expanded return-buffer/credit-tag budgets. They are separate
from the 256-credit same-SRAM ablation. See the policy methods for details.
`check_large_batches.py --capture CAPTURE.npz --output OUTPUT` rebatches real
decode routes and checks capacity first. All B32–B256 plans in the current
capture fail the existing all-results-resident storage contract, so no large
batch latency is reported. The results summary is
[RUNTIME_POLICIES_20260930.md](results/RUNTIME_POLICIES_20260930.md).

During paired-repository development, `PLENA_DISPATCH_COMPILER` can select a
different `research/moe_dispatch` directory. The runner records that planner's
hash and snapshots its source; default review/reproduction uses the gitlink.

## Review map

| File | Responsibility |
|---|---|
| `rust/src/main.rs` | Admission, finite resources, DMA, bank service, execution and reports |
| `rust/src/runtime.rs` | Current/Next, bounded prefetch, stable DMA protocol and response identity |
| `rust/src/surplus.rs` | Depth reservation, deadline admission, next-phase prefetch and supply observations |
| `rust/src/plan.rs` | Physical tiling, X reuse and exact work/traffic accounting |
| `frontend.py` | Load the pinned Compiler without duplicating its implementation |
| `run_experiments.py`, `audit_runtime_policies.py` | Immutable campaigns, ablations, repeat and resource checks |
| `trace_payload.py`, `test_runtime.py` | Actual timed-event numerical replay and boundary cases |
| `numerical.py`, `test_*.py` | Separate payload audit and timing integration tests |
| `results/RUNTIME_FSM_20260930.md` | Current repair acceptance, all four modes and measured model outcomes |
| `results/runtime_fsm_20260930.csv` | Compact 48-point acceptance table with raw-report hashes |
| `results/RESULTS.md` | Historical prototype results, retained unchanged |

No existing native Simulator defaults, timing implementation or frozen reports
are changed by this branch. Earlier experimental worktrees remain separate.
