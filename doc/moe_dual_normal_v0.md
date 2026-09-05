# MoE dual normal-buffer V0

This experiment executes a complete fixed-route expert FFN with nonzero values:
gather, Gate/Up GEMMs, SwiGLU, Down GEMM, and deterministic weighted combine.
It adds a compiler manifest exporter and an independent simulator executable.
The executable shares the repository's runtime, memory, MX decoder, and one
Ramulator HBM instance. It does not yet lower these jobs to the legacy ISA.

## Meeting requirement and implemented boundary

- One or two independent square matrix cores, configured separately through
  `BLEN` and `MLEN`; multipliers per core are `BLEN * MLEN`.
- Both cores use normal buffers. Each has private activation SRAM, FP32
  accumulator storage and two weight slots. The slots retain packed MX ingress
  and decoded BF16 values until consumed. No transpose buffer in V0.
- One shared HBM model, finite global 64-byte request credits and staging;
  each core prefetches its next weight tile while consuming its current tile.
- Routes are grouped by expert. An expert group is assigned using its row
  count and a fixed threshold. Independent core queues execute concurrently.
- Route outputs occupy bounded reorder SRAM. Combine follows token/slot order
  regardless of completion order, with an optional constant-weight shared
  expert contribution last.
- A grouped single core uses exactly the same arithmetic and timing rules.
  The comparison tool requires equal total multipliers and shared resources.

The first campaign also keeps total configured private SRAM equal, but this is
**not an area-equivalent comparison**: input port widths, duplicated controls,
banking, physical arithmetic pipelines and routing area are not synthesized.
Neither the historical 1024-elements/cycle limit nor the 16-slot setting is
treated as a requirement from the meeting.

## Source entry points

- Compiler: `PLENA_Compiler/aten/plena/moe_normal_export.py` exports actual
  PLENA E4M3/E8M0 block-8 bytes, separate payload/scales in output-major `[N,K]`
  rows, source hashes, inputs/routes and an independent scalar golden.
- Simulator: `transactional_emulator/src/bin/moe_dual_normal.rs` validates the
  manifest/image, creates HBM and writes a report with input and binary hashes.
- Engine: `transactional_emulator/src/moe_normal/`; see
  [engine contract](moe_dual_normal_engine.md) for resource and timing formulas.
- Fixtures: `transactional_emulator/testbench/moe_timing/replay/prepare_moe_normal_fixtures.py`.
- Comparison: `transactional_emulator/testbench/moe_timing/replay/compare_moe_normal.py`.

The schema fixes PLENA's local codec semantics, including its subnormal/zero
behavior; it is not a claim of general OCP MX format compliance. Every projection
accumulates ascending K with separate FP32 multiply/add, then rounds to BF16.
SwiGLU and final output also round to BF16. The Python oracle decodes the actual
exported bytes independently from the Rust implementation.

## Validation and reproducibility

Use the repository's Rust/libtorch/Ramulator environment (the pinned libtorch
version is 2.7.0 for `tch 0.20`). Place build output on a filesystem with room.
The local campaign archives its exact environment exports, fixture source
hashes, configuration files, input bytes, command logs and binary SHA-256 in
`outputs/moe_dual_normal_20260905` at the workspace root.

From `transactional_emulator`, in that build environment:

```bash
cargo test --offline -j4 --bin moe_dual_normal
cargo build --offline -j4 --bin moe_dual_normal
```

Run `prepare_moe_normal_fixtures.py --help` for the fixture-generation arguments.
It needs a Python interpreter with torch and `PLENA_Tools` on `PYTHONPATH`.
The comparison command takes `--binary`, `--workload`, `--golden`, repeated
`--architecture` arguments (baseline first), and `--output-dir`. It requires at
least two runs per configuration. Numerical mismatch, stale reports, changed
input/binary identities, resource overflow or nondeterministic results fail the
campaign before publishing a speedup. BF16 bit equality is reported separately
from the selected numeric tolerance; the local campaign uses zero tolerance.

The declared configurations have 96 multipliers: single `B4/K24`, alternate
single `B2/K48`, and dual `B4/K16 + B2/K16`. These are demonstrators, not DSE
optima. Both single-core results must be reported to avoid attributing a weak
choice of single-core geometry to an inherent dual-core advantage.

## What these results can establish

The four fixtures cover tails, unused experts, constant-weight shared expert,
and archived Qwen/DeepSeek route IDs and weights. Archived cases use synthetic
ready inputs and weights at reduced dimensions. They do not run those models
or their runtime routers/shared gates. Timing starts with input and routes
already available in bounded shared SRAM and ends with the BF16 combined
output there; input/output HBM transfers are outside this boundary.

The engine performs numerical arithmetic, not merely request-count replay.
HBM timing comes from Ramulator; matrix/vector timing is an explicit analytical
model. It has accumulator dependency stalls and finite slots, but no SRAM bank
conflicts, calibrated exp/div unit, full physical pipeline or RTL verification.
Each whole expert group must fit; capacity failure does not silently spill.

## Follow-up work, in dependency order

1. Use the observed traffic and utilization to improve normal-buffer loading
   and dispatch. Add homogeneous multicore controls and sensitivity to group
   threshold, HBM bandwidth, vector throughput and SRAM ports before selecting
   an architecture. Keep the equal-multiplier grouped single-core baseline.
2. Integrate the job contract with compiler lowering and the simulator's ISA
   execution path. Add bounded group splitting/spilling and runtime dispatch,
   then full-size inputs/weights and model-specific shared expert/gate semantics.
3. Validate SRAM banking, accumulator and arithmetic pipeline ports/area before
   making hardware PPA claims; measure realistic layer and request boundaries.
4. Add transpose-capable buffering with an explicit payload/scale mapping and
   numerical tests. Then extend to multiple small cores, optional sharing,
   attention and RTL. These remain tracked work, not features completed by V0.
