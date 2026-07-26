# DSE Artifact Compaction and FFN Address Lowering v1

> **Mixed-status implementation milestone, audited 2026-07-26.** Compact
> artifact retention remains current. The first FFN live-stride fix was
> superseded by the capacity-independent `affine-loop-v2` lowering documented
> in
> [`../compiler/unified_affine_ffn_loop_lowering_v2.md`](../compiler/unified_affine_ffn_loop_lowering_v2.md).

## Scope

This change addresses two independent DSE problems:

1. Completed 2,048-point studies retained repeated compiler reports and
   per-worker recovery files, producing multi-gigabyte run directories.
2. Large-MLEN FFN lowering repeatedly legalized dead or invariant pointer
   increments into long `S_ADDI_INT` sequences. The worst observed point,
   `M8192/B32/N4`, spent 67.1% of aggregate compute work in `S_ADDI_INT`.

No RTL or ISA opcode was added. Matrix timing, HBM V4, area coefficients,
power coefficients, AGU-v1, and the numerical correctness gate are unchanged.

## Compact Artifact Retention

The DSE now defaults to:

```text
--artifact-retention compact
```

During execution, each trial keeps a resume-safe compact record and a
gzip-compressed detail record. Compiler reports and generated settings are
stored once by canonical content hash. Successful finalization retains full
detail only for Pareto and named selector trials, then removes worker JSONL
files, heartbeats, locks, worker-local area caches, and transient compiler
caches. `--artifact-retention full` preserves the historical behavior.

An offline compactor supports `--dry-run` and `--apply`. It verifies the study
objectives, Pareto set, and selector inputs before deleting redundant files.

### Existing-run validation

| Run | Before | After | Objective fingerprint |
|---|---:|---:|---|
| Short, `seq=482, batch=16`, 2,048 points | about 12 GiB | 247.7 MB | unchanged (`3b4c...9056`) |
| Long, `seq=32768, batch=1` | about 2.7 GiB | 169.2 MB | unchanged (`f48d...4a0`) |

The short result satisfies the 500 MiB migration target. The long result
satisfies the 250 MiB target. A new 64-point compact smoke occupied only
8.2 MiB after finalization.

## FFN Live-Stride Lowering

The new default is:

```text
ffn_address_schedule = live-stride-v1
```

The compatibility mode remains:

```text
ffn_address_schedule = legacy
```

For each FFN projection, the compiler now builds a shared pointer-liveness
plan used by both ASM emission and CostEmitter:

- When `k_tile_count == 1`, post-`M_MM` weight and activation pointer updates
  are dead and are removed.
- The final prefetch pointer increments are removed because the pointers are
  reset before their next use.
- The final output-column increment is removed.
- Live large strides are materialized once in `gp8`-`gp10` and applied with
  one `S_ADD_INT` instead of hundreds or thousands of legal `S_ADDI_INT`
  chunks.
- Multi-K-tile loop-carried immediates remain visible to the existing AGU-v1
  loop pass.

Metadata reports the selected schedule, dead updates, stride loads, avoided
immediate chunks, and residual address opcodes. These fields are present in
compact trial JSON and the DSE CSV outputs.

## Fixed-Point Results

All results use Qwen3-32B, `seq=32768`, `batch=1`, four-chip optimistic TP+SP,
`BLEN=32`, streaming Matrix SRAM, and the same representative MXFP precision
profile.

| MLEN | Legacy latency | Live-stride latency | Reduction |
|---:|---:|---:|---:|
| 2,048 | 37.259 s | 28.927 s | 22.36% |
| 4,096 | 23.937 s | 21.512 s | 10.13% |
| 8,192 | 64.855 s | 21.796 s | 66.39% |

For the anomalous `M8192/B32/N4` point:

| Work item | Legacy cycles | Live-stride cycles | Delta |
|---|---:|---:|---:|
| `S_ADDI_INT` | 173,946,511,288 | 1,543,314,424 | -99.11% |
| `S_ADD_INT` | 171,983,872 | 339,592,064 | +167,608,192 |
| Matrix compute | 76,093,641,728 | 76,093,641,728 | 0 |
| Total compute work | 259,309,909,351 | 87,074,319,847 | -66.42% |

The FFN stage fell from 193.297B to 21.062B cycles. Attention remained
66.012B cycles, so the corrected point is no longer dominated by compiler
address arithmetic.

## Invariant Evidence

Across the M2048, M4096, and M8192 fixed-point A/B runs:

- `M_MM` and `M_MM_WO` work is exactly unchanged.
- HBM physical read/write bytes are exactly unchanged.
- HBM read/write request counts are exactly unchanged.
- The same V4 memory model and precision profile are used.
- CostEmitter receives the compiler-derived schedule; no cycles are
  subtracted after lowering.

For M8192 specifically:

```text
HBM read bytes     1,055,208,243,200 -> unchanged
HBM write bytes      279,172,874,240 -> unchanged
HBM read requests         16,487,628,800 -> unchanged
HBM write requests         4,362,076,160 -> unchanged
```

Focused tests cover one-, two-, and four-K-tile liveness, Matrix/DMA parity,
large-immediate lowering, gzip canonical hashing, compact trial retention,
and CSV metadata preservation.

### Transactional numerical A/B

A tiny real K-split FFN was compiled and executed through the transactional
emulator in both address modes:

```text
MLEN=VLEN=64, BLEN=4
batch=4, seq=2
hidden=64, intermediate=256
Matrix SRAM tiles=2
```

The effective 1,024-byte BF16 result region was bitwise identical:

```text
legacy SHA-256:
a852362efefb58534e36c24e417487099f373c7c34fde118f9de49fee304cf1c

live-stride-v1 SHA-256:
a852362efefb58534e36c24e417487099f373c7c34fde118f9de49fee304cf1c
```

Both paths also passed the unchanged comparison gate. The live-stride run
reported 99.61% relative-error matches and 100% `allclose` matches. This
directly checks the multi-K-tile path for which loop-carried updates remain
live, rather than relying only on static opcode accounting.

## Parallel DSE Smoke

The latest long-context infrastructure completed:

```text
64 COMPLETE / 64 requested workers
0 PRUNED
0 FAIL
SQLite WAL storage
```

Observed resource data:

```text
maximum dynamic concurrency: 60
peak worker RSS:              1.18 GiB
peak active process-tree RSS: 34.43 GiB
minimum MemAvailable:         79.08 GiB
final compact run size:        8.2 MiB
```

This validates the new schema, compact retention, shared report cache,
live-stride default, and multi-process finalization before the 2,048-point
long-context study.

## Formal Long-Context Run

The new study was launched after the 64-point acceptance smoke:

```text
tmux session: qwen3_dse_long_ffn_live_v1
run directory: /tmp/qwen3_32b_dse_long_32768x1_ffn_live_v1
log: /tmp/qwen3_dse_long_ffn_live_v1.log
target: 2,048 COMPLETE
attempt ceiling: 2,560
requested workers: 64
worker trial-count recycle cap: disabled
storage: SQLite WAL
artifact retention: compact
```

The worker pool remains governed by the RSS and system-memory safety limits;
disabling count-based recycling does not disable memory-pressure protection.

## Interpretation and Limits

The prior M8192 result was not evidence that a larger MLEN is inherently slow.
It was a compiler lowering artifact amplified by the 18-bit immediate limit.
The fix removes provably dead arithmetic and replaces repeated constant
materialization with ordinary register-register adds.

The remaining M8192 result is still an architectural extrapolation. It does
not validate an 8,192-wide RTL implementation, timing closure, or area model.
The formal DSE also continues to use ideal-II1 compute timing and optimistic
TP+SP multi-chip scaling, which must be disclosed independently.
