# SmolVLM2 — Per-Phase Host Timing (post parent-thread-cap fix)

Re-measured: `2026-06-02`, **after** the parent-thread-cap fix landed on `main` (`#82`, `d0d8b7f`). Submodules: compiler `a4c80f8`, tools `8d385af` (the merged multibatch decoder). The earlier numbers in this doc's history were taken at the default 64 parent threads (pre-fix); the table below is the post-fix reality and the prior PRE→POST A/B is summarised in its own section.

Native runs only, **≤5 layers**, batch=1 (sub-64 decoder via `--seq-len 4`). Each `run_model.py` run has five host-side phases:

| phase | span (stdout markers) | what it is |
|-------|-----------------------|------------|
| **load** | `Loading model` → `Computing CPU golden` | HF model load (one-time) |
| **golden** | `Computing CPU golden` → `Backend (ISA generation)` | CPU reference compute (`backend=scheduled`); summed over vision+decoder for vlm-e2e |
| **isa_gen** | `Backend (ISA generation)` → `ASM written` | `plena_frontend` codegen |
| **sim_env** | `ASM written` → `Running Rust` | write the HBM + SRAM binaries (`create_sim_env` / `create_mem_for_sim`) |
| **emulate** | `Running Rust` → `host wall time` | Rust transactional emulator (host wall) |

`--threads` (1 sub-64 / 4 mlen=64 / 8 mlen=256) caps the emulator subprocess **and now the parent python** (the landed fix ties parent OMP/MKL/OpenBLAS to `--threads`). `sim_lat` below is the modeled-HW latency (`executor.now()`, deterministic, the real accelerator metric). **Phase columns are host wall seconds and depend on box load — they are NOT a hardware metric.**

## TL;DR

1. **The fix flipped the profile.** Pre-fix, harness overhead (`golden` + `sim_env` + `load`) was **73%** of wall time; post-fix it is **~12%**. The real work — `emulate` (cycle-accurate Rust sim) **64%** + `isa_gen` (accelerator codegen) **24%** = **~88%** — now dominates, which is where the time *should* go.
2. **`golden` collapsed 48× (1451s → 31s)** and **`sim_env` collapsed 21× (1788s → 86s).** Both were pure parent-python OpenMP/MKL thread oversubscription (default 64 threads thrashing on many tiny per-op quantization ops), not inherent cost. Per-run the golden speedup is 30–85×.
3. **Aggregate host wall: ~4426s (~74 min) → ~1433s (~24 min), 3.1× faster** for the 18-run sweep, despite `emulate` reading *higher* purely from box load (see point 5).
4. **`isa_gen` is unchanged (334s → 342s)** — it is single-threaded codegen, thread-independent. So the two **mlen=16 vision runs still DNF** (codegen runaway, >30 min): the cap cannot help them. **Practical rule: use mlen≥32 for vision/vlm-e2e** (decoder is fine at mlen=16: 21s total).
5. **`emulate` rose 715s → 914s, but that is box-load noise, not a regression.** Every `sim_lat` is byte-identical pre↔post (e.g. 32/32/4 vlm-e2e `8364759ns` both runs; vision 256 5L `18.85ms` both) — identical `sim_lat` proves the emulator did identical work; the wall-time delta is the shared box being busier during the re-run. `emulate` is explicitly a host-wall metric.
6. **Numerics unchanged: 18/20 PASS** (the 2 fails are the mlen=16 vision *compile* DNFs, not numerics; `allclose` is deterministic and thread-independent, carried from the prior run). Multi-layer degradation reproduced: decoders erode with depth (5L: 64/64/16 → 94.8%, 256 → 95.7%), vision is graceful (256 5L → 99.99%).

## ⚠️ "vlm-e2e" here is DECODER-ONLY — NOT true end-to-end

Two load-bearing caveats — these rows are labelled **`vlm-e2e‡`** in the table:

1. **Only the decoder is emulated.** In the vlm-e2e path the vision encoder + connector run as **CPU golden** (`vision_result["padded_golden_output"]`) to produce the decoder's input embeds; only the decoder is compiled + emulated. Proof: vlm-e2e and the standalone decoder at the same config are **byte-identical** (32/32/4: both `isa=583396`, `sim=8.364759ms`). So the `vlm-e2e‡` rows' `sim_lat` and phase times are **decoder-only** — the vision encoder's emulated cost (≈39ms at mlen=32; see the vision rows) is NOT included. Full-pipeline latency ≈ vision + connector + decoder. The "100%" validates the **decoder consuming the connector embeds (the handoff)**; vision/connector accuracy is validated by their own standalone rows.
2. **1 layer each side, not full depth.** Runs used `--vision-layers 1 --text-layers 1`; the full model is **12 vision (SigLIP) + 30 text-decoder** layers. A real full-depth run would degrade (dominated by the 30-layer decoder accumulation — see the per-layer numbers). Full-depth, fully-emulated e2e is **untested**.

## Results + 5-phase breakdown (post-fix host seconds)

| # | config | case | L | allclose | sim_lat | load | golden | isa_gen | sim_env | emulate | total |
|---|--------|------|---|----------|--------:|-----:|-------:|--------:|--------:|--------:|------:|
| 3 | 16/16/4 | decoder | 1 | 100% | 5.35ms | 3.2 | 0.3 | 1.6 | 3.4 | 12.0 | 21 |
| 4 | 32/32/4 | decoder | 1 | 100% | 8.36ms | 3.1 | 0.3 | 0.8 | 3.1 | 11.8 | 19 |
| 7 | 64/64/16 | decoder | 1 | 98.5% | 1.26ms | 3.1 | 0.4 | 0.2 | 0.9 | 7.9 | 12 |
| 8 | 64/64/16 | decoder | 5 | 94.8% | 6.23ms | 3.3 | 2.1 | 1.4 | 3.8 | 30.1 | 41 |
| 9 | 256/256/64 | decoder | 1 | 98.7% | 0.98ms | 3.1 | 0.8 | 0.1 | 0.8 | 13.1 | 18 |
| 10 | 256/256/64 | decoder | 3 | 97.9% | 2.89ms | 3.2 | 2.2 | 0.2 | 2.1 | 29.7 | 37 |
| 11 | 256/256/64 | decoder | 5 | 95.7% | 4.80ms | 3.4 | 3.4 | 0.4 | 3.1 | 46.9 | 57 |
| 6 | 32/32/4 | vision | 1 | 100% | 39.83ms | 3.5 | 0.6 | 59.2 | 3.8 | 52.9 | 120 |
| 12 | 64/64/16 | vision | 1 | 100% | 3.81ms | 3.2 | 0.6 | 1.9 | 3.0 | 17.6 | 26 |
| 13 | 64/64/16 | vision | 5 | 99.97% | 14.44ms | 3.3 | 3.9 | 15.9 | 8.2 | 71.9 | 103 |
| 14 | 256/256/64 | vision | 1 | 99.98% | 9.55ms | 3.3 | 0.8 | 17.6 | 7.7 | 102.1 | 132 |
| 15 | 256/256/64 | vision | 3 | 100% | 14.20ms | 3.1 | 2.6 | 51.6 | 11.9 | 151.5 | 221 |
| 16 | 256/256/64 | vision | 5 | 99.99% | 18.85ms | 3.5 | 3.9 | 87.5 | 17.2 | 203.9 | 316 |
| 17 | 64/64/16 | connector | 1 | 99.96% | 4.73ms | 3.3 | 1.7 | 5.0 | 3.8 | 24.4 | 38 |
| 18 | 256/256/64 | connector | 1 | 99.66% | 9.90ms | 3.4 | 1.5 | 12.7 | 8.0 | 107.0 | 132 |
| 2 | 32/32/4 | vlm-e2e‡ | 1v+1t | 100% | 8.36ms | 3.9 | 1.5 | 65.5 | 3.1 | 11.4 | 86 |
| 19 | 64/64/16 | vlm-e2e‡ | 1v+1t | 100% | 1.28ms | 3.5 | 2.0 | 5.8 | 0.9 | 7.9 | 20 |
| 20 | 256/256/64 | vlm-e2e‡ | 1v+1t | 100% | 0.98ms | 3.4 | 2.0 | 14.4 | 0.7 | 12.0 | 33 |
| 1 | **16/16/4** | **vlm-e2e‡** | 1v+1t | — | — | — | — | **DNF >30m** | — | — | **DNF** |
| 5 | **16/16/4** | **vision** | 1 | — | — | — | — | **DNF >30m** | — | — | **DNF** |

**Aggregate (18 timed runs):** load **60s** · golden **31s** · isa_gen **342s** · sim_env **86s** · emulate **914s** · total **~1433s**.

> **Post-fix split:** emulate **64%** · isa_gen **24%** · sim_env **6%** · load **4%** · golden **2%**. Harness overhead (load+golden+sim_env) is now **~12%** of wall time vs **73%** pre-fix. `emulate` + `isa_gen` (the real work) are the residual. `sim_lat` (the HW metric) is unchanged by the cap.

**‡ `vlm-e2e‡` is NOT true end-to-end — decoder-only.** The vision encoder + connector run as CPU golden (not emulated) to produce the decoder's input embeds; only the decoder is emulated. So every `vlm-e2e‡` row is byte-identical to the standalone decoder at the same config (32/32/4: `isa=583396`, `sim=8.364759ms`), and its `sim_lat`/phases exclude the vision encoder (≈39ms). The decoder here sees only 4 image tokens (64 patches → connector ÷16). See the caveat section above.

## PRE → POST A/B (the fix)

Both phases that collapsed were parent-python OpenMP/MKL thread oversubscription, measured by re-running the identical sweep through `run_model.py` before and after the cap. `sim_lat` is byte-identical in every pair (the emulator does identical work), so this is pure host-side harness speedup.

| phase | pre-fix (64 threads) | post-fix (capped) | factor |
|-------|---------------------:|------------------:|-------:|
| golden | 1451s | 31s | **48×** |
| sim_env | 1788s | 86s | **21×** |
| isa_gen | 334s | 342s | ~1× (thread-independent codegen) |
| emulate | 715s | 914s | host-load noise; `sim_lat` identical |
| load | 138s | 60s | 2.3× (warm disk cache) |

Per-run, the biggest absolute wins are the deep vision configs: vision 256 5L golden 221.6s → 3.9s and sim_env 526.5s → 17.2s (total 995s → 316s); vision 64 5L golden 196.8s → 3.9s and sim_env 300.1s → 8.2s (total 567s → 103s).

## Per-phase scaling (post-fix)

| phase | per-layer | notes |
|-------|-----------|-------|
| load | ~3s fixed | model load (warm cache) |
| golden | <1s/layer | CPU reference; was the #1 cost pre-fix, now negligible after the cap |
| isa_gen | ≤1.5s decoder, ~15–90s vision | the real codegen cost; **runaway at mlen=16 vision (DNF)** |
| sim_env | ~2–17s, grows with tensor-size × layers | binary writing; collapsed 21× with the cap |
| emulate | small (decoder) → moderate (vision 256) | `--threads`-capped; deterministic `sim_lat` is the HW metric |

## mlen=16 vision DNF — root cause (proven, unchanged)

Unbuffered probe of `vision-layers 16/16/4 --compile-only`: load ✓, golden ✓ (~50s), then **stuck in `--- PLENA Vision Backend (ISA generation) ---` for the full 20-min window at ~1 core (computing, not deadlocked)**. At mlen=16 the vision attention tiles into 4 col-blocks × 4 q-tiles × heads, and the codegen tile loop is single-threaded and superlinear in tile count. Not threads (OMP=1 and OMP=8 both still DNF — the default-64 case additionally *thrashes* to 191 threads), not golden (fast), not the emulator. The parent-thread-cap fix is confirmed not to help (isa_gen unchanged at 342s aggregate). Decoder is unaffected.

## Optimizations

1. **Cap the parent python's OMP/MKL threads — LANDED (`run_model.py`, `#82`).** golden + sim_env were thread oversubscription in the *parent* (default 64 threads thrashing on the many tiny per-op quantization ops). Tying parent threads to `--threads`: measured **golden 1451s → 31s (48×)**, **sim_env 1788s → 86s (21×)** across the full sweep. By far the biggest lever — confirmed in this re-measurement. Does NOT help the mlen=16 isa_gen runaway (thread-independent codegen).
2. **Kill the ASM text→binary re-parse in sim_env.** Post-cap, the residual sim_env (~2–17s) is dominated by `parse_asm_file` re-parsing the emitted ASM *text* back into binary (2.6M `parse_reg_or_int`, 10M `strip`…). Emit binary directly from the ISA structures, or optimize the parser hot loop. Files: `assembler/parser.py`, `assembler/assembly_to_binary.py`. Now the largest remaining harness cost.
3. **`isa_gen` / `_emit` (`isa_emit.py:25`)** — now the #2 phase overall (24% of wall) and the *only* thing that fails; thread-independent, this is where the mlen=16 vision runaway lives. Buffer/batch the per-instruction emission. For now, **use mlen≥32 for vision** to sidestep the runaway.

## Method / caveats

- Host times: sequential re-run (no cross-run contention), `PYTHONUNBUFFERED=1`, epoch-stamped phase-marker lines; inter-marker deltas. `emulate` is the emulator's own `host wall time`. golden/isa_gen sum vision+decoder sub-phases for vlm-e2e.
- All runs deterministic — `sim_lat` is invariant to threads/box load (verified byte-identical pre↔post); the host-second columns are not.
- Re-run artifacts: `/tmp/smolvlm2_phase_postfix/` (post-fix logs + `parse.py`), `/tmp/smolvlm2_phase/` (pre-fix logs).
