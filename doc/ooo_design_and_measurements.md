# PLENA OOO Dispatcher — Design & Measurement Provenance

Branch: `yw/ooo_arch`. Companion artifacts: `tools/gen_stream_ubench.py`
(generator + measured-results comments), `tools/regression_suite.sh`
(numerical-invariance gate). All commits referenced below are on this branch.

---

## 1. Out-of-order control logic

**Principle: single in-order issue, out-of-order execution; ordering is
enforced by three independent mechanisms, each owning one hazard class.**

```
do_ops (the only dispatcher task; decodes in program order)
  │
  ├─ S_* / C_* scalar+control ───── executed inline by the dispatcher
  ├─ H_PREFETCH ─── tracker write-claim → install Err(rx) → returns
  │                 immediately (data arrives asynchronously)
  ├─ H_STORE_V / V_RED_* ────────── drain barrier, then inline
  └─ M_* / V_* compute ops (23 opcodes):
        1. snapshot reg-file operands + compute AccessRange at decode
        2. PendingTracker.acquire(reads, write)   ← the only stall point
        3. Executor::spawn as an independent task; dispatcher moves on
```

### Layer 1 — register dependences
Operands are snapshotted into values at issue. Later scalar instructions
mutating registers cannot affect already-issued ops. The reverse direction
(V_RED_SUM/V_RED_MAX write `fp_reg`, which later decode reads) is handled
by a full drain barrier before those ops.

### Layer 2 — intra-unit ordering
Matrix / vector units are each wrapped in one `Arc<tokio::Mutex>` (FIFO).
Ops on the same unit therefore execute serially in issue order — the
unit-internal accumulators (`m_accum`/`hm_accum`/`v_accum`) need no extra
protection. Ops on *different* units hold different mutexes: true
concurrency.

### Layer 3 — cross-unit memory dependences (`PendingTracker`, one per SRAM)
* At decode, the dispatcher derives the op's read/write **byte-range
  bounding boxes** over VRAM/MRAM from the reg-file.
* `acquire(reads, write)` is atomic all-or-nothing: every read range must
  be free of in-flight writers, the write range free of in-flight
  readers+writers, checked **before inserting anything** — which makes
  in-place ops (`V_ADD_VV y, y, bias`) structurally unable to deadlock
  against their own slots (a real bug found via the deadlock detector:
  piecewise acquire froze every linear kernel at its bias-add).
* Conflict ⇒ the dispatcher parks (program order preserved). No conflict
  ⇒ overlap. Net semantics: RAW/WAR/WAW in program order,
  read-after-read concurrent.
* Spawned tasks release their slots on completion (`notify_waiters`).

### Layer 4 — prefetch data dependences (Step 1 per-tile state machine)
Each SRAM tile is `Mutex<Result<QuantTensor, oneshot::Receiver>>`. A
prefetch installs `Err(rx)` and returns; the HBM fanout task sends data
later; any consumer that reads the tile awaits the channel in place. The
install itself is bracketed by a tracker write-claim (released right
after install) so it cannot clobber a tile an already-issued-but-not-yet-
executed reader still needs (WAR).

### Determinism & failure containment
* Executor timers tie-break by global insertion sequence number, not
  pointer order (commit `bbe1b51`) — reproducible across processes,
  machines, rebuilds.
* `execute_batch` deadlock detector: the discrete-event executor's
  `enter()` is a non-blocking loop that returns when no task is ready and
  no timer pending; if `do_ops` hasn't completed at that point, remaining
  tasks are parked on wakers only other parked tasks could fire. Instead
  of hanging at 0% CPU, the run fails fast with dispatcher pc,
  `unit_in_flight`, and both trackers' live ranges.

### Why numerics are bit-identical, not merely "within tolerance"
Every path that could alter fp16 accumulation order — intra-unit op order
and cross-unit memory order — is pinned to program order. Reordering only
happens between mathematically independent instructions. Verified: the
full regression (flash_per_head_v3 + linear + 6-case regime_sweep) is
bit-identical to the sequential baseline, twice in a row
(`tools/regression_suite.sh --diff` ⇒ "identical — no numerical drift").

---

## 2. Measured speedups & provenance

**Quantity measured**: simulated latency (`Simulation completed. Latency`,
the discrete-event clock — *not* wall time). Single-run deterministic;
sanity reruns reproduce to the nanosecond.

**Binaries** (same artifacts run through all):

| label | built from | semantics |
|---|---|---|
| baseline | sibling worktree `yw/online_emulator` (rebuilt with the config-driven DRAM WIP for the c2_4090 rows) | blocking prefetch + in-order dispatch |
| step1-only | this repo @ `bbe1b51` | async per-tile prefetch + in-order dispatch |
| ooo | `yw/ooo_arch` HEAD | Step 1 + 2.2c dispatch-ahead |

**Artifacts** (one `.mem` + HBM image per row, CLI one-shot mode,
`PLENA_CONFIG` absolute path):

| measurement | artifact source | archived |
|---|---|---|
| attention +0.5% (77,096,408 → 76,713,238 ns) | tilelang `flash_per_head_v3_verify --s-q 1024 --s-kv 1024` build dir (113,698 instr; HBM-verify PASS first) | session log |
| streaming GEMM 1.41× (428,180 → 303,973 ns) | `tools/gen_stream_ubench.py` (16×1 MiB MXFP8 chunks through 4 MRAM tiles, zero-filled HBM) | commit `16ffdf4` |
| memory-bound 1.055× (313,157 → 296,805 ns) | same, compute cut to 1 M_MM/chunk | commit `0075729` |
| V-stream 1.14× @8ch (336,376 → 295,582 ns) | 2048 × 8 KB `H_PREFETCH_V`, contiguous | session log |
| c2_4090 table (below) | same four artifacts + `configs/config_2_4090.toml` | commit `26860f3` |

**Configs**: `config_2.toml` = 8-channel HBM2 2 Gbps (~128 GB/s peak),
1 GHz core. `config_2_4090.toml` = 64-channel (~1 TB/s), 2.52 GHz core
(ported from the online worktree WIP in commit `3a9566b`; defaults verified
bit-exact against the previously hard-coded behaviour).

### Results

config_2 (8ch, 1 GHz):

| workload | blocking | ooo | speedup |
|---|---|---|---|
| 1024-seq text attention (compute-bound) | 77,096,408 ns | 76,713,238 ns | 1.005× |
| streaming GEMM, compute-rich | 428,180 ns | 303,973 ns | **1.41×** |
| streaming GEMM, memory-bound | 313,157 ns | 296,805 ns | 1.055× |
| V-stream 8 KB bursts | 336,376 ns | 295,582 ns | 1.14× |

config_2_4090 (64ch, 2.52 GHz):

| workload | blocking | ooo | speedup |
|---|---|---|---|
| M-stream (memory-bound) | 44,271 ns | 37,590 ns | 1.18× |
| M-stream + compute | 89,887 ns | 54,438 ns | **1.65×** |
| V-stream 8 KB bursts | 176,784 ns | 57,724 ns | **3.06×** |
| V-stride 4 KiB | 181,221 ns | 116,558 ns | 1.55× |

### Interpretation

* Gain ≈ `(T_mem + T_compute) / max(T_mem, T_compute)`: zero at the
  compute-bound ceiling (attention: the single matrix unit is >99% of
  simulated time; OOO recovers the dispatch bubbles and hits the
  unit-bound limit), maximal when memory and compute are balanced.
* **The OOO advantage grows with memory bandwidth** (V-stream 1.14× @8ch
  → 3.06× @64ch): a blocking dispatcher is round-trip-bound per prefetch
  instruction (~95 GB/s, 9.4% of the 1 TB/s peak — extra channels cannot
  help), while dispatch-ahead keeps multiple prefetches outstanding and
  reaches ~291 GB/s (29%); M-path 1 MiB bursts reach ~528 GB/s (52%).
  This is the mechanism behind the earlier observation that "8→64
  channels only bought 2.5×" on the blocking simulator.
* Ablation (step1-only ≈ baseline on every row): the async per-tile
  channel mechanism produces **no** overlap by itself — the in-order
  dispatcher immediately blocks on the first consumer's read. Mechanism
  (Step 1) × scheduling (2.2c) — neither suffices alone.
* OOO is latency *hiding*, not bandwidth: with compute cut 8×, gain
  collapses to 1.055× — the HBM drain itself is untouched. Remaining gap
  to peak is V-burst granularity (`HBM_V_Prefetch_Amount`, a config knob)
  and DRAM row efficiency (data layout) — neither is a dispatcher
  problem.

### Honest footnotes

1. The ooo binary also contains the per-channel command-issue-port change
   (`2068110`), measured **neutral** (±1%) at 8ch — speedups are
   attributable to dispatch semantics.
2. Attention is a production kernel; the four ubenches are synthetic
   microbenchmarks isolating M-path / V-path / burst granularity /
   stride, standard practice for architecture evaluation.
3. The c2_4090 baseline was rebuilt from the sibling worktree *with* the
   config-driven DRAM WIP, so both sides of every comparison ran
   identical hardware parameters.

---

## 3. Addendum — 2.2e (Tomasulo) measurements

Reservation-station issue (commit `b64cdde`): the dispatcher never
blocks; ops wait for ticket-ordered tracker grants inside their spawned
tasks; fp operands resolve post-grant / pre-unit-lock; the matrix unit
keeps program order via a ticket turnstile (accumulator semantics).
Regression: bit-identical.

Masked-softmax ubench (flash S→P shape, 256 rows × 8 masked heads) —
the scoreboard's flat 11,969 ns at any unit count becomes:

| NUM_VECTOR_UNITS | latency | speedup |
|---|---|---|
| 1 | 11,012 ns | 1.09× |
| 2 | 6,416 ns | 1.87× |
| 4 | 5,092 ns | 2.35× |
| 8 / 16 | ~5,085 ns | saturated |

Saturation = the ~5K-op scalar/control stream at 1 op/cycle: with the
back-end unblocked, FRONT-END issue bandwidth is the next wall
(multi-issue decode = future work), reached before fp-register
pressure binds.

Flash attention per_head_v3, SQ=1024 (same artifact as §2):

| regime | blocking | tomasulo | gain |
|---|---|---|---|
| config_2 (8ch, 1 GHz) | 77,096,836 ns | 76,287,490 ns | 1.05% |
| config_2_4090 (64ch, 2.52 GHz) | 30,516,571 ns | 30,107,841 ns | 1.34% |

Attention stays single-matrix-unit-bound in both regimes; Tomasulo
doubles the recoverable stall fraction vs 2.2c (0.5% → 1.05%) by
eliminating the V_RED / H_STORE_V machine-wide drains. NUM_VECTOR_UNITS=8
is a no-op on this kernel (12 vector ops total) — the clean control
group for the softmax ubench's 2.35×: unit replication pays exactly
where vector work sits on the critical path.
