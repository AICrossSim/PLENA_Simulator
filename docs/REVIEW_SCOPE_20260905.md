# Simulator review against main: support, precision and evidence

**Draft for code/design review only. Do not merge or enable auto-merge.**
RTL and hardware decisions follow this review. No GPU task or new architecture
experiment is part of preparing this PR.

This snapshot starts from Simulator `ad201c3953e15df0b49e1e8358af3937e96e1009`
and is compared with main `117154b19bece8cf1f8691ca7cf1e1ea5328b512`.
The original feature branch and archived measurements remain available. This
review branch consolidates its net changes, makes targeted correctness fixes,
and clarifies support boundaries; it does not discard earlier measurements.

Paired [Compiler Draft PR #79](https://github.com/AICrossSim/PLENA_Compiler/pull/79)
is pinned by the submodule at `1955abba267e4c9d9874221f53a9c16db984c8cf`.

## What actually works, and what is not established

| Area | Implemented support | Evidence and boundary |
|---|---|---|
| Static Mamba/KDA | Additional instruction forms, scalar SRAM, recurrent stage profiling, Python testbenches and analytic operator models | Stage/reference/interpreter tests and selected Rust execution; not proof of every official layer with real checkpoint weights |
| Matrix-SRAM L_TILE | Descriptor validation, banked cells, explicit viewed DMA, packet order/lane restoration, three recurrence primitives and output/state stores | Compiler-generated machine code executes at official recurrence sizes; both fixed and phased numerical controls are retained |
| Matrix projection | Existing Matrix accumulation retained, optional Matrix views/writeback added | Focused projection tests; existing accumulators were not introduced by this PR |
| Controlled A/B/D | Prepared BF16 operand arenas, private requests, exact per-path oracle, common error budget, optional intermediate-state snapshots, serial timing ledger | Actual recurrent-core execution; no checkpoint weights or upstream coefficient generation in this harness |
| Full-model timelines | 52/93-layer analytic coverage, weight/storage policies and workload/routing replay | Formula/timing-model results; full nonrecurrent layers are not numerically executed by the recurrence schedule |
| GPU evidence | Imported measured Nemotron NVFP4 baseline, routing and component evidence; energy reanalysis with explicit limitations | Existing collection and provenance, not new GPU runs or a full real Kimi checkpoint baseline |
| Optional numeric controls | FP32-dot subforms and compiler-only BF16 pairwise mode | Both default off; separate experiments, not selected replacement architectures |

Mamba/KDA convenience wrappers have explicit B1/geometry limits; the hybrid
schedule and controlled private-request harness are separate APIs. Batch support
in one must not be advertised as support in all. Unsupported wide direct
projection is rejected; generic/view support is not automatic integration into
all model lowering paths.

## Precision in plain language

权重格式决定文件占用和搬运量；state 格式决定递推记忆怎么存；累加精度决定
计算过程中保留多少有效数字。此次工作没有把全模型改成 BF16，也没有把原
Matrix 的 accumulator 统一改成 FP32。新增的独立 BF16 state DMA 契约修正了
递推数据的实际字节解释；权重敏感度表则是另外一项流量计数。

| Path | Stored data | Arithmetic / meaning |
|---|---|---|
| Weight sensitivity model | NVFP4 block16, MX8 block8 or BF16, with explicit scales, padding and BF16 exclusions | Logical traffic accounting. This does not numerically decode checkpoint NVFP4/MX8 weights in the Rust recurrence kernel. Legacy block128 results remain historical. |
| Ordinary prepared A/B | Actual BF16 state, coefficients and SRAM/HBM bytes | Each ordinary VV result rounds when written back to BF16 SRAM. These controls were added by this project, not supplied as KDA lowering by the original paper. |
| D / L_TILE | BF16 state, fields and output | DOT retains FP32 across logical rows; update primitives keep intermediate products until primitive writeback. Numerical contract differs from separate BF16 MUL/ADD instructions. Physical resource reuse is unresolved. |
| Experimental FP32 dot | BF16 persistent data | Only the two KDA dots retain FP32 products/partial sums. Requires the explicit compiler option and `PLENA_EXPERIMENTAL_FP32_DOT=1`; modeled extra storage is 8 KiB at VLEN=2048. |
| Experimental BF16 tree | BF16 persistent data and partial rows | Compiler emits ordinary MUL/ADD with BF16 boundaries; reserves 15 rows/60 KiB within existing Vector SRAM. No V_DOT/L_TILE. Opt-in diagnostic, not the chosen architecture. |
| GPU model | Actual GPU checkpoint/runtime formats | Measured GPU execution follows that implementation's precision; it is not silently substituted for the PLENA storage/rounding contract. |

`plena_settings.toml` adds independent `HBM_STATE_TYPE` entries. Original global
W/A/KV settings are not changed by that addition. `QuantTensor` may use host f32
as its representation; Vector SRAM's byte encoding is the actual stored-format
rounding boundary. Original Matrix f32 host accumulators predate L-Compute and
do not, by themselves, specify physical register widths or capacity.

## Read the code in this order

1. `PLENA_Compiler` submodule and the paired Compiler review guide: emitted ISA
   and precision contracts must match this decoder.
2. `transactional_emulator/src/op.rs`: instruction forms and reserved bits;
   `accelerator/mview.rs`, `lstream.rs`: descriptor and packet contracts.
3. `transactional_emulator/lib/sram/src/matrix.rs`: mapping, ownership, bank
   service and physical cell reads/writes; `vector.rs`: byte encoding.
4. `transactional_emulator/src/accelerator/dispatch.rs` and `vector_machine.rs`:
   recurrence arithmetic, state lifetime and explicit timing charges.
5. `transactional_emulator/src/timing.rs`, `dma.rs`, `runtime_config.rs`:
   serial execution ledger and its relationship to main's timing modes.
6. `transactional_emulator/testbench/aten/matrix_lcompute_execution_compare.py`:
   actual machine-code control runs, exact and common-reference gates, and
   suppression of unqualified or snapshot-contaminated performance ratios.
7. `analytic_models/performance/nemotron3_workload.py` and
   `matrix_lcompute_campaign.py`: storage bytes, full-layer timing and sources.

## Review findings and unresolved decisions

- **Corrected input boundary:** L_TILE execution rejects logical source lines
  wider than the configured Vector width. Previously it accepted those lines
  but charged only one Vector arithmetic pass. Ordinary official 64/128-column
  cases at VLEN=2048 are unaffected. Wide views remain usable by other legal
  consumers; no new column-splitting hardware or algorithm is introduced.
- **Legacy binary compatibility needs review:** Matrix-view writeback uses
  bit17 of the former 18-bit `M_MM_WO` immediate as a view marker. A pre-extension
  word with that bit set can be reinterpreted as the viewed form. The assembler's
  new rejection of large ordinary immediates does not preserve existing binaries.
  This draft does not reassign the encoding or claim full old-binary compatibility.
  The concrete legacy word `0x80000046` (`M_MM_WO gp1, gp0, 131072`) now
  selects view0; with no configured view it traps, rather than performing the
  former Vector writeback. Separately, ordinary DMA selector2 now means State,
  so callers that relied on the older nonzero-selector-as-KV behavior need
  selector1. Both changes must be reviewed explicitly.
- **Hardware mapping is open:** L_TILE descriptors, traversal, route/restore
  logic and FP32 intermediate state need resource/precision mapping. No new MAC
  array or SRAM payload does not prove zero added hardware.
- **Timing scope is open:** historical analytic A/B issue proxies and C/D service
  models remain asymmetric. Controlled Rust A/B/D use explicit common accounting,
  but are different programs with differing coefficient traffic and rounding.
  Their ratio is not a drop-in replacement for old Arlo/full-model speedups.
- **Numerical scope is open:** exact matching of a chosen arithmetic oracle is
  necessary but does not prove checkpoint quality. Standard long-chain BF16-tree
  cases pass; stronger synthetic long-memory cases still fail, including some
  experimental FP32-dot cases. Do not loosen thresholds to publish a ratio.
- **Coverage is open:** full real Nemotron/Kimi checkpoint execution from first
  to last layer in Rust and an original-resource complete KDA B1 mapping remain
  uncompleted. Prepared-field execution does not cover coefficient generation.
  Other routed-MoE opcodes present in the broader Compiler remain outside this
  recurrence decoder's supported subset; this is not complete branch-wide ISA
  closure.

These are review boundaries, not requests to implement RTL in this PR.

## Validation and artifacts

Fresh PR-preparation results are recorded in `REVIEW_VALIDATION_20260905.md`.
Historical gate counts elsewhere refer to their original commits and must not
be presented as a fresh run of this review snapshot.

The eight `artifacts/**/campaign.json` files have only their JSON whitespace
compacted. Every parsed value is identical to the source snapshot. All CSVs,
summary JSON, GPU profiles and their byte hashes are untouched. See
`artifacts/REVIEW_PACKING_MANIFEST.json` for source commit, old/new byte hashes
and canonical content hashes. Older freeze-document campaign byte hashes refer
to the original pretty-printed files, not these compact review copies.

The compact objects remain regression inputs; no artifact check is skipped.
GitHub marks only those generated payloads as generated. Review the source,
CSV summaries and this guide first, then expand payloads when needed.

Reproduce the complete connected gate in the configured dev environment:

```bash
nix develop --no-write-lock-file --command just test-matrix-lcompute PLENA_Compiler
```

This includes Python contracts, Compiler guards, Rust workspace tests, Matrix
projection and official-shape recurrence/control execution. It does not run
real GPU workloads, full-model checkpoint inference, RTL synthesis or PPA.
