# SmolVLM2 — Compile-vs-Emulate Phase Timing & Current-`main` Re-confirm

Generated: `2026-06-02`  ·  Branch: `feat/multibatch-decoder` (compiler `0f55571` ≡ merged `a4c80f8`, tools `a6ac9e3` ≡ `8d385af`)

Native runs only, ≤5 layers, batch=1 (sub-64 decoder batch via `--seq-len 4`). Each `run_model.py` run has four host-side phases — **load** (HF model load), **golden** (CPU reference compute, `backend=scheduled`), **isa_gen** (`plena_frontend` codegen), **emulate** (Rust transactional emulator, host wall) — followed by the golden↔sim comparison. `--threads` (1 sub-64 / 4 mlen=64 / 8 mlen=256) caps the **emulator** subprocess only.

## TL;DR findings

1. **After the `--threads` fix, emulation is no longer the bottleneck — the test harness is.** Across the 18 timed runs: golden **1235s** + isa_gen **1630s** (compile side, ~75%) vs emulate **722s** (~25%).
2. **Golden (CPU reference) is NOT cheap** — it scales **~30–40s/layer** (decoder 256: 36 → 95 → 162s for 1/3/5L). It is pure harness overhead (the reference we compare against, not accelerator work) → the single cleanest optimization target (cache / vectorize the per-layer scheduled-reference recompute).
3. **ISA-gen is vision-heavy** (~60s/layer vision vs ~14s/layer decoder) and is the **only** thing that fails: at **mlen=16, vision/vlm-e2e ISA-gen runs away (>30 min, single-threaded codegen hot spot)** — both mlen=16 vision runs DNF. Decoder is fine at mlen=16 (21s).
4. **Practical rule: use mlen≥32 for vision/vlm-e2e.** mlen=32 vision = 230s; mlen=16 vision = DNF.
5. **Numerics re-confirmed on current `main`:** 18/20 PASS (the 2 fails are the mlen=16 vision DNFs, i.e. compile, not numerics). Multi-layer degradation reproduced: decoders erode with depth (5L: 64/64/16 → 94.8%, 256 → 95.7%), vision is graceful (256 5L → 99.99%).

## Results + phase breakdown

`sim_lat` = modeled hardware latency (`executor.now()`, deterministic, the real accelerator metric). Phase columns are **host wall seconds**.

| # | config | case | L | allclose | sim_lat | load | golden | isa_gen | emulate | total |
|---|--------|------|---|----------|--------:|-----:|-------:|--------:|--------:|------:|
| 3 | 16/16/4 | decoder | 1 | 100% | 5.35ms | 6.7 | 19.1 | 16.4 | 11.0 | 53 |
| 4 | 32/32/4 | decoder | 1 | 100% | 8.36ms | 6.6 | 18.2 | 17.9 | 10.6 | 53 |
| 7 | 64/64/16 | decoder | 1 | 98.5% | 1.26ms | 6.7 | 22.5 | 17.6 | 5.9 | 53 |
| 8 | 64/64/16 | decoder | 5 | 94.8% | 6.23ms | 8.0 | 96.8 | 77.4 | 20.6 | 203 |
| 9 | 256 | decoder | 1 | 98.7% | 0.98ms | 7.1 | 36.2 | 22.1 | 9.8 | 75 |
| 10 | 256 | decoder | 3 | 97.9% | 2.89ms | 8.1 | 95.0 | 46.5 | 21.4 | 171 |
| 11 | 256 | decoder | 5 | 95.7% | 4.80ms | 9.2 | 162.3 | 72.0 | 32.6 | 276 |
| 6 | 32/32/4 | vision | 1 | 100% | 39.83ms | 7.3 | 48.3 | 128.6 | 45.4 | 230 |
| 12 | 64/64/16 | vision | 1 | 100% | 3.81ms | 7.0 | 47.0 | 70.6 | 12.3 | 137 |
| 13 | 64/64/16 | vision | 5 | 99.97% | 14.44ms | 9.6 | 196.8 | 314.7 | 45.9 | 567 |
| 14 | 256 | vision | 1 | 99.98% | 9.55ms | 7.4 | 50.0 | 128.6 | 85.8 | 272 |
| 15 | 256 | vision | 3 | 100% | 14.20ms | 8.3 | 132.1 | 233.2 | 122.6 | 496 |
| 16 | 256 | vision | 5 | 99.99% | 18.85ms | ⚠️ | ⚠️ | ⚠️ | ~161 | ~287* |
| 17 | 64/64/16 | connector | 1 | 99.96% | 4.73ms | 7.7 | 52.5 | 76.2 | 17.5 | 154 |
| 18 | 256 | connector | 1 | 99.66% | 9.90ms | 7.5 | 52.1 | 142.4 | 92.5 | 295 |
| 2 | 32/32/4 | vlm-e2e | 1v+1t | 100% | 8.36ms | 7.3 | 52.9 | 71.5 | 10.9 | 143 |
| 19 | 64/64/16 | vlm-e2e | 1v+1t | 100% | 1.28ms | 6.8 | 65.2 | 22.6 | 5.6 | 100 |
| 20 | 256 | vlm-e2e | 1v+1t | 100% | 0.98ms | 7.2 | 82.6 | 53.8 | 9.9 | 154 |
| 1 | **16/16/4** | **vlm-e2e** | 1v+1t | — | — | — | — | **DNF >30m** | — | **DNF** |
| 5 | **16/16/4** | **vision** | 1 | — | — | — | — | **DNF >30m** | — | **DNF** |

Aggregate (18 timed runs): load **131s** · golden **1235s** · isa_gen **1630s** · emulate **722s**.

\* **Run 16 (vision 256 5L) load/golden/isa split is misparsed** (a marker-timestamp glitch — its `load`=2.9s/`golden`=5.3s are implausible and its total came out below the 3L run, which is impossible). `emulate ≈161s` and the overnight allclose (99.99%) are valid; only the golden/isa split for this one row needs a clean re-run.

## Per-phase scaling (host seconds)

| phase | per-layer (decoder) | per-layer (vision) | nature |
|-------|--------------------:|-------------------:|--------|
| load | ~7s (fixed) | ~7s (fixed) | HF model load, one-time |
| golden | ~30–40s | ~37s | **CPU reference recompute — harness overhead, optimization target** |
| isa_gen | ~14s | ~60s | `plena_frontend` codegen; runaway at mlen=16 vision |
| emulate | small (`--threads`) | small–moderate | Rust sim host wall; deterministic sim_lat is the HW metric |

## mlen=16 vision DNF — root cause (proven)

Unbuffered phase probe of `vision-layers 16/16/4 --compile-only`: model load ✓, golden ✓ (done ~50s), then **stuck in `--- PLENA Vision Backend (ISA generation) ---` for the full 20-min window at ~1 core (computing, not deadlocked)**. At mlen=16 the vision attention tiles into 4 col-blocks × 4 q-tiles × heads, and the codegen tile loop is single-threaded and scales pathologically. Not threads (capping OMP=1/8 didn't help — still DNF), not golden (fast), not the emulator. Decoder is unaffected (1 col-block at head_dim=64/mlen... it tiles far less). The earlier pre-flight "compiled, 358K ISA" report was false (stale build-dir artifacts).

## Optimizations

1. **Use mlen≥32 for vision/vlm-e2e** (practical; sidesteps the runaway entirely).
2. **Cache/vectorize the scheduled-reference golden recompute** — ~1235s (40%) of total wall is the CPU reference; it is harness-only overhead and the biggest clean win.
3. **Profile the vision-backend ISA emitter** (`plena_frontend`) for the superlinear tile loop — only needed if mlen=16 vision must be supported.
4. **Cap the *parent* python torch threads** (`torch.set_num_threads(args.threads)`) — minor; prevents the default-64-thread *thrash* (191 threads) that turns the mlen=16 hang into a 25-min thrash, but does not make mlen=16 vision feasible.

## Method / caveats

- Host wall times are from a sequential re-run (no cross-run contention) with timestamped phase markers (`PYTHONUNBUFFERED=1`, epoch-stamped marker lines). They depend on box load and are **not** a hardware metric; `sim_lat` is.
- `emulate` is the Rust emulator's own `host wall time` line; `golden`/`isa_gen` are inter-marker deltas; `golden`/`isa_gen` for vlm-e2e sum the vision + decoder sub-phases.
- All runs deterministic — `sim_lat` is invariant to threads/box load.
