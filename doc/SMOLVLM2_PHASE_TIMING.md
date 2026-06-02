# SmolVLM2 — Per-Phase Host Timing & Current-`main` Re-confirm

Generated: `2026-06-02`  ·  Branch: `feat/multibatch-decoder` (compiler `0f55571` ≡ merged `a4c80f8`, tools `a6ac9e3` ≡ `8d385af`)

Native runs only, **≤5 layers**, batch=1 (sub-64 decoder via `--seq-len 4`). Each `run_model.py` run has five host-side phases:

| phase | span (stdout markers) | what it is |
|-------|-----------------------|------------|
| **load** | `Loading model` → `Computing CPU golden` | HF model load (one-time) |
| **golden** | `Computing CPU golden` → `Backend (ISA generation)` | CPU reference compute (`backend=scheduled`); summed over vision+decoder for vlm-e2e |
| **isa_gen** | `Backend (ISA generation)` → `ASM written` | `plena_frontend` codegen |
| **sim_env** | `ASM written` → `Running Rust` | write the HBM + SRAM binaries (`create_sim_env` / `create_mem_for_sim`) |
| **emulate** | `Running Rust` → `host wall time` | Rust transactional emulator (host wall) |

`--threads` (1 sub-64 / 4 mlen=64 / 8 mlen=256) caps the **emulator** subprocess only. `sim_lat` below is the modeled-HW latency (`executor.now()`, deterministic, the real accelerator metric). **Phase columns are host wall seconds and depend on box load — they are NOT a hardware metric.**

## TL;DR

1. **The cycle-accurate emulation is only ~16% of wall time. The test harness dominates:** `sim_env` (mem-binary writing) **1788s (~40%)** + `golden` (CPU reference) **1451s (~33%)** = **73% pure harness overhead**. `emulate` 715s (~16%). **`isa_gen` (the actual accelerator codegen) is cheap — only 334s (~7%).**
2. **Biggest, cleanest optimization = `sim_env` (HBM/SRAM binary generation).** It scales with tensor-size × layers and explodes for big vision (vision 256 5L = **526s** just writing binaries). Pure serialization overhead.
3. **#2 = the `golden` recompute** (~30–40s/layer); also harness-only.
4. **`isa_gen` is the *only* thing that fails — at mlen=16 vision it runs away (>30 min, single-threaded codegen hot spot)** → both mlen=16 vision runs DNF. Everywhere else codegen is ≤83s. **Practical rule: use mlen≥32 for vision/vlm-e2e** (decoder is fine at mlen=16: 53s).
5. **Numerics re-confirmed on current `main`: 18/20 PASS** (the 2 fails are the mlen=16 vision *compile* DNFs, not numerics). Multi-layer degradation reproduced: decoders erode with depth (5L: 64/64/16 → 94.8%, 256 → 95.7%), vision is graceful (256 5L → 99.99%).

## ⚠️ "vlm-e2e" here is DECODER-ONLY — NOT true end-to-end

Two load-bearing caveats — these rows are labelled **`vlm-e2e‡`** in the table:

1. **Only the decoder is emulated.** In the vlm-e2e path the vision encoder + connector run as **CPU golden** (`vision_result["padded_golden_output"]`) to produce the decoder's input embeds; only the decoder is compiled + emulated. Proof: vlm-e2e and the standalone decoder at the same config are **byte-identical** (32/32/4: both `isa=583396`, `sim=8.364759ms`). So the `vlm-e2e‡` rows' `sim_lat` and phase times are **decoder-only** — the vision encoder's emulated cost (≈39ms at mlen=32; see the vision rows) is NOT included. Full-pipeline latency ≈ vision + connector + decoder. The "100%" validates the **decoder consuming the connector embeds (the handoff)**; vision/connector accuracy is validated by their own standalone rows.
2. **1 layer each side, not full depth.** Runs used `--vision-layers 1 --text-layers 1`; the full model is **12 vision (SigLIP) + 30 text-decoder** layers. A real full-depth run would degrade (dominated by the 30-layer decoder accumulation — see the per-layer numbers). Full-depth, fully-emulated e2e is **untested**.

## Results + 5-phase breakdown (host seconds)

| # | config | case | L | allclose | sim_lat | load | golden | isa_gen | sim_env | emulate | total |
|---|--------|------|---|----------|--------:|-----:|-------:|--------:|--------:|--------:|------:|
| 3 | 16/16/4 | decoder | 1 | 100% | 5.35ms | 6.7 | 19.1 | 1.4 | 15.0 | 11.0 | 53 |
| 4 | 32/32/4 | decoder | 1 | 100% | 8.36ms | 6.6 | 18.2 | 0.8 | 17.1 | 10.6 | 53 |
| 7 | 64/64/16 | decoder | 1 | 98.5% | 1.26ms | 6.7 | 22.5 | 0.1 | 17.4 | 5.9 | 53 |
| 8 | 64/64/16 | decoder | 5 | 94.8% | 6.23ms | 8.0 | 96.8 | 1.4 | 76.0 | 20.6 | 203 |
| 9 | 256 | decoder | 1 | 98.7% | 0.98ms | 7.1 | 36.2 | 0.1 | 22.0 | 9.8 | 75 |
| 10 | 256 | decoder | 3 | 97.9% | 2.89ms | 8.1 | 95.0 | 0.2 | 46.3 | 21.4 | 171 |
| 11 | 256 | decoder | 5 | 95.7% | 4.80ms | 9.2 | 162.3 | 0.3 | 71.7 | 32.6 | 276 |
| 6 | 32/32/4 | vision | 1 | 100% | 39.83ms | 7.3 | 48.3 | 57.1 | 71.5 | 45.4 | 230 |
| 12 | 64/64/16 | vision | 1 | 100% | 3.81ms | 7.0 | 47.0 | 1.5 | 69.1 | 12.3 | 137 |
| 13 | 64/64/16 | vision | 5 | 99.97% | 14.44ms | 9.6 | 196.8 | 14.5 | 300.1 | 45.9 | 567 |
| 14 | 256 | vision | 1 | 99.98% | 9.55ms | 7.4 | 50.0 | 16.8 | 111.8 | 85.8 | 272 |
| 15 | 256 | vision | 3 | 100% | 14.20ms | 8.3 | 132.1 | 45.6 | 187.6 | 122.6 | 496 |
| 16 | 256 | vision | 5 | 99.99% | 18.85ms | 9.8 | 221.6 | 82.9 | 526.5 | 154.3 | 995 |
| 17 | 64/64/16 | connector | 1 | 99.96% | 4.73ms | 7.7 | 52.5 | 6.1 | 70.0 | 17.5 | 154 |
| 18 | 256 | connector | 1 | 99.66% | 9.90ms | 7.5 | 52.1 | 13.1 | 129.3 | 92.5 | 295 |
| 2 | 32/32/4 | vlm-e2e‡ | 1v+1t | 100% | 8.36ms | 7.3 | 52.9 | 54.1 | 17.5 | 10.9 | 143 |
| 19 | 64/64/16 | vlm-e2e‡ | 1v+1t | 100% | 1.28ms | 6.8 | 65.2 | 5.2 | 17.4 | 5.6 | 100 |
| 20 | 256 | vlm-e2e‡ | 1v+1t | 100% | 0.98ms | 7.2 | 82.6 | 32.2 | 21.6 | 9.9 | 154 |
| 1 | **16/16/4** | **vlm-e2e‡** | 1v+1t | — | — | — | — | **DNF >30m** | — | — | **DNF** |
| 5 | **16/16/4** | **vision** | 1 | — | — | — | — | **DNF >30m** | — | — | **DNF** |

**Aggregate (18 timed runs):** load **138s** · golden **1451s** · isa_gen **334s** · **sim_env 1788s** · emulate **715s**.

> **NB — these host-second columns are at the default 64 parent threads (PRE the thread-cap fix in this PR).** golden + sim_env are dominated by parent-python thread oversubscription, not inherent cost: capping parent threads (this PR) cuts **golden ~180× and sim_env ~25×** (vision 256 3L: golden 132→0.7s, sim_env 188→7.6s). Post-fix, **`isa_gen` (codegen) + `emulate` are the real residual.** `sim_lat` (HW metric) is unaffected.

**‡ `vlm-e2e‡` is NOT true end-to-end — decoder-only.** The vision encoder + connector run as CPU golden (not emulated) to produce the decoder's input embeds; only the decoder is emulated. So every `vlm-e2e‡` row is byte-identical to the standalone decoder at the same config (32/32/4: `isa=583396`, `sim=8.364759ms`), and its `sim_lat`/phases exclude the vision encoder (≈39ms). The decoder here sees only 4 image tokens (64 patches → connector ÷16). See the caveat section above.

## Per-phase scaling

| phase | per-layer | notes |
|-------|-----------|-------|
| load | ~7s fixed | model load |
| golden | ~30–40s/layer | CPU reference; harness overhead, optimization target |
| isa_gen | ≤1s decoder, ~15–80s vision | cheap normally; **runaway at mlen=16 vision (DNF)** |
| sim_env | grows with tensor-size × layers | decoder 256 5L = 72s; **vision 256 5L = 526s** — biggest single cost |
| emulate | small (decoder) → moderate (vision 256) | `--threads` win; deterministic sim_lat is the HW metric |

## mlen=16 vision DNF — root cause (proven)

Unbuffered probe of `vision-layers 16/16/4 --compile-only`: load ✓, golden ✓ (~50s), then **stuck in `--- PLENA Vision Backend (ISA generation) ---` for the full 20-min window at ~1 core (computing, not deadlocked)**. At mlen=16 the vision attention tiles into 4 col-blocks × 4 q-tiles × heads, and the codegen tile loop is single-threaded and superlinear in tile count. Not threads (OMP=1 and OMP=8 both still DNF — the default-64 case additionally *thrashes* to 191 threads), not golden (fast), not the emulator. Decoder is unaffected.

## Optimizations

1. **Cap the parent python's OMP/MKL threads — LANDED in this PR (`run_model.py`).** golden + sim_env were thread oversubscription in the *parent* (default 64 threads thrashing on the many tiny per-op quantization ops). Tying parent threads to `--threads` gives, on vision 256 3L: **golden 132s → 0.7s (~180×), sim_env 188s → 7.6s (~25×)**. By far the biggest lever — **the host-second columns above are PRE-fix (default 64 threads)** and collapse with the cap. Does NOT help the mlen=16 isa_gen runaway (thread-independent codegen).
2. **Kill the ASM text→binary re-parse in sim_env.** Post-cap, the residual sim_env (~7s) is dominated by `parse_asm_file` re-parsing the emitted ASM *text* back into binary (2.6M `parse_reg_or_int`, 10M `strip`…). Emit binary directly from the ISA structures, or optimize the parser hot loop. Files: `assembler/parser.py`, `assembler/assembly_to_binary.py`.
3. **`isa_gen` / `_emit` (`isa_emit.py:25`)** — ~47s self, ~30ms/call, thread-independent; this is where the mlen=16 vision runaway lives. Buffer/batch the per-instruction emission. For now, **use mlen≥32 for vision** to sidestep the runaway.

## Method / caveats

- Host times: sequential re-run (no cross-run contention), `PYTHONUNBUFFERED=1`, epoch-stamped phase-marker lines; inter-marker deltas. `emulate` is the emulator's own `host wall time`. golden/isa_gen sum vision+decoder sub-phases for vlm-e2e.
- All runs deterministic — `sim_lat` is invariant to threads/box load; the host-second columns are not.
