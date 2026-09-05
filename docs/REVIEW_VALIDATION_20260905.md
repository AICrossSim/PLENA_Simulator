# Fresh Simulator review validation — 2026-09-05

Draft review evidence, not merge acceptance or a hardware claim. The paired
Compiler is `1955abba267e4c9d9874221f53a9c16db984c8cf`; this Simulator snapshot
starts from `ad201c3953e15df0b49e1e8358af3937e96e1009` and includes the review
fix that rejects L_TILE execution wider than the configured Vector width.

## Checks executed for this review

| Check | Result |
|---|---|
| Compiler Matrix-L-Compute gate | 231 passed, exit 0; one missing-NumPy warning |
| Simulator Matrix-L-Compute Python gate plus `test_hybrid_lcompute_campaign.py` | 187 passed, exit 0 |
| Rust `cargo fmt --all -- --check` | Passed |
| Rust `cargo test --workspace --release -- --test-threads=1` | 308 passed, 0 failed, 0 ignored |
| Rust `cargo clippy --workspace --all-targets --release -- -D warnings` | Passed |
| Compiler-generated Matrix-view projection executed by Rust | Passed; zero output error |
| Official-shape Mamba/KDA, fixed and affine, four-token Rust recurrence | All four pass their declared layout-specific gates |
| Mamba prepared A/B/D, B1, two tokens | All three exact against their own rounding references; all pass the common budget |
| KDA prepared A/B/D, B1, two tokens | All three exact against their own rounding references; **A/B fail the common output budget**, D passes |

The Python suite includes artifact regressions; no assertion was removed for
the eight compacted JSON payloads. The Compiler fallback checks exercise its
Python 3.10 enum compatibility implementation on Python 3.11; a real Python
3.10 interpreter was unavailable. Counts are for these named gates, not every
Python test in either repository. The focused Compiler 112-test run overlaps
the 231-test gate and must not be added to it.

## Actual numerical results

Projection observed zero output error; its acceptance check uses
`atol=rtol=0.2`, not an exact-comparison gate.

Four-token recurrence inputs are deterministic synthetic BF16 values at the
official recurrent-state dimensions. Affine/phased cases require exact
comparison; fixed cases retain the declared 0.01 relative-L2 and elementwise
budget. This fresh run does not repeat the archived multi-seed/long-token sweep.
It checks each token's output and the final state; intermediate-state snapshots
were not enabled in these fresh connected runs.

| Model / layout | State relative L2 | Output relative L2 | Gate |
|---|---:|---:|---|
| Nemotron Mamba fixed | 0 | 0.005383845 | Pass |
| Nemotron Mamba affine | 0 | 0 | Exact pass |
| Kimi KDA fixed | 0.000462562 | 0.007075169 | Pass |
| Kimi KDA affine | 0 | 0 | Exact pass |

The KDA fixed output still has only about 1.41x margin to its relative-L2
threshold. Exact affine results on these inputs do not establish exactness
for every input or equivalence to a GPU checkpoint implementation.

The separate B1/two-token controls use seed 20260903, prepared coefficients,
request-private BF16 state, and the common serial accounting contract. Both
optional FP32-dot and BF16-tree flags are off.

| Controls | Own rounding oracle | Worst common output relative L2 | Common 0.01 budget |
|---|---|---:|---|
| Mamba A/B | Exact | 0.007733444 | Pass |
| Mamba D | Exact | 0 | Pass |
| KDA A/B | Exact | 0.013485170 | **Fail** |
| KDA D | Exact | 0 | Pass |

KDA A/B output errors are 0.013485170 and 0.013422726 across the two tokens.
Their state error is 0.000736433. The diagnostic CLI exits 0 after recording
these failures; that exit code does **not** mean the common numerical gate
passed. Its `qualified_comparison.csv` correctly sets both qualification flags
false and leaves the speedup blank. Do not publish a KDA speedup from this run.
Mamba's qualified D/B value is 1.1105195x for this controlled core only; it is
not the historical Arlo/full-model comparison or a B200 speedup.

Small unmodified result summaries and their source hashes are checked in at
`artifacts/review_20260905/`. No large HBM dump or checkpoint is added.

## Reproduction and environment

The repository entry point is:

```bash
nix develop --no-write-lock-file --command just test-matrix-lcompute PLENA_Compiler
```

This review ran its components separately with the checked-out review Compiler,
the existing Python environments, CUDA hidden, and CPU thread counts limited.
The additional artifact regression was
`analytic_models/performance/test_hybrid_lcompute_campaign.py`.
Rust used Nix dependencies and a warm Cargo target directory. Formatting and
clippy were run in addition to the repository gate.

The connected commands, run from the Simulator root with
`PLENA_COMPILER_ROOT` pointing to the pinned Compiler, were:

```bash
python3 transactional_emulator/testbench/aten/matrix_view_projection_test.py
python3 -m transactional_emulator.testbench.aten.matrix_lcompute_recurrence_test --output-dir /tmp/review-recurrences
python3 -m transactional_emulator.testbench.aten.matrix_lcompute_execution_compare --model mamba --batches 1 --tokens 2 --output-dir /tmp/review-mamba
python3 -m transactional_emulator.testbench.aten.matrix_lcompute_execution_compare --model kda --batches 1 --tokens 2 --output-dir /tmp/review-kda
```

An initial attempt to run the host Python environment inside Nix failed before
projection execution because Nix's injected libstdc++ required GLIBC_2.38.
Running host Python outside that injected library environment and using
`PLENA_USE_NIX_BUILD=1` for the Rust build resolved it. The subsequent projection
and all recurrence/control invocations completed. The failed environment log
is retained with the local review evidence, not counted as a numerical run.

No GPU collection, full real Nemotron/Kimi checkpoint inference, new batch
sweep, RTL synthesis or PPA run was performed for this review.

## Open correctness boundary

The legacy `M_MM_WO` word `0x80000046` was independently executed. It formerly
encoded ordinary Vector writeback at immediate 131072; this decoder treats its
old immediate bit17 as a Matrix-view marker. Without view0 configured, it traps.
This compatibility issue remains open in the Draft PR; passing tests do not
make this snapshot safe to merge. See `REVIEW_SCOPE_20260905.md` for the
separate DMA selector compatibility, precision and coverage boundaries.
