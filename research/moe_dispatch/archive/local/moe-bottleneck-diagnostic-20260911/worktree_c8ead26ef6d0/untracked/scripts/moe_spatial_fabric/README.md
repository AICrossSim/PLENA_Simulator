# Finite spatial-M operand fabric

Read `../../doc/moe_spatial_fabric_contract.md` before interpreting results.
This is an independent decoded-BF16 **interface experiment**. It does not modify
the normal MoE engine, Compiler lowering, MX format or native HBM implementation.
It extends the physical-M experiment, not the old temporal-M hardware budget.

From `transactional_emulator`:

```bash
source /scratch/shared/mcl123/plena/outputs/moe_stream_ctrl_20260914/5cec1f6917944c7d8d96cb5ac7f79bc7/step0/repro/build-env.sh
cargo test --workspace
cargo clippy --bin moe_spatial_fabric -- -D warnings
cargo build --release --bin moe_spatial_fabric
```

From the repository root, using a fresh output directory:

```bash
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python scripts/moe_spatial_fabric/run_study.py \
  --binary /tmp/plena-moe-dual-core-target/release/moe_spatial_fabric \
  --output /tmp/plena-fabric-rerun/evaluation \
  --input-manifest /scratch/shared/mcl123/plena/outputs/moe_spatial_m_20260919/07844dcc0d3c467ca14e5fa84255eaa0/results/input_manifest.json
```

Generate selections before the extra numerical headline audit, then regenerate
the report to include that audit. Use the same frozen binary for both runners:

```bash
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python scripts/moe_spatial_fabric/summarize.py \
  --evaluation /tmp/plena-fabric-rerun/evaluation
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python scripts/moe_spatial_fabric/verify_headlines.py \
  --evaluation /tmp/plena-fabric-rerun/evaluation \
  --binary /tmp/plena-moe-dual-core-target/release/moe_spatial_fabric \
  --output /tmp/plena-fabric-rerun/headline_numeric
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python scripts/moe_spatial_fabric/summarize.py \
  --evaluation /tmp/plena-fabric-rerun/evaluation
/scratch/shared/mcl123/plena/venvs/plena-py311/bin/python scripts/moe_spatial_fabric/plot_results.py \
  --evaluation /tmp/plena-fabric-rerun/evaluation
```

The completed study contains 546 points repeated twice. Another 19 numerical
points repeated twice cover every headline architecture/window/interface and
the control-granularity comparison. Their cycles, counters and timeline hashes
must exactly match the corresponding shape-only runs, and all architectures
must produce the same FP32 and BF16 output bits for a given window.

The runner executes positive/negative controls, tails, archived routed shapes,
isolated Me1/Me32, and resource/control sensitivities. Every point runs twice.
It compares all eight ordered partitions of the six-row budget used in the
previous study, both pinned and stealing policies, and three control modes.
Packet-serialized operand ports and optimistic packed/banked operand ports are
reported separately. A shape/policy is chosen on B8 and kept fixed on disjoint
B32 tokens 8–31. Winner follow-ups include the best homogeneous and heterogeneous
shapes, rather than only the hand-picked 3+3 and 4+2 examples.

The three control modes are:

* `invocation`: each invocation issues and completes through the global port.
* `cohort`: same-round invocations with the same weight key share a descriptor.
* `tile_cohort`: one persistent descriptor covers all Me rows for an expert/N/K
  tile; later M blocks use local sequencers, with one final descriptor retirement.

Persistent headers may use at most half the metadata pool. Open cohorts have
resume priority and invocation progress credit; every admission also reserves
result credit. Tests include the long-N/small-credit case that otherwise deadlocks.

The source weight port, activation port, shared backing accumulator port and
control ports actually gate execution. Weight data are retained in finite
private holding slots and numerically consumed from those slots. Input and
weight values are generated BF16 fixtures; outputs are checked against both
Rust scalar and independent Python integer references. No real model accuracy
or native codec claim follows from these tests.

All runs audit event ordering, port/byte reservations, MAC coverage, budgets,
lifetimes and drain inside Rust, then hash full event and service timelines.
Selected runs retain full traces for a second independent Python audit; all
others retain counters and both timeline hashes. Every numerical case has an
independent integer check. Reports explicitly distinguish numeric/shape-only
and full-trace/compact cases.

Busy service counters overlap. They must not be added to elapsed cycles.
Packed-port occupancy is the union of occupied cycles; byte ranges, rather
than rounded request intervals, are non-overlapping. Each local commit actor
still processes at most one invocation per cycle. Control cycle assumptions
(2/3/2) and bandwidths are sensitivity parameters, not synthesis measurements.

The shared accumulator backing is a strong idealized access model: RMW bytes
and service are charged, but bank conflicts, distributed private-store migration
and NoC topology are absent. Holding-slot read implementation, gather/broadcast
routing, area, power, upstream memory response and independently scheduled
weight prefetch remain outside this experiment. Extra capacity alone is not a
standalone prefetch actor.

Oracles keep the policy and functional/storage rules while removing selected
timing. Online scheduling and cache decisions may therefore change traffic;
inspect both byte and time columns. They are not fixed-trace causal bounds.

The final report should cite the frozen binary and source archive accompanying
`evaluation/`, not an intermediate trial or a subsequently rebuilt shared binary.
