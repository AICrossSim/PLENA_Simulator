# Rollable Ops — Survey & Rolled-vs-Unrolled Experiment Plan

Generated: `2026-06-02`. Scope: which PLENA codegen ops (beyond attention) emit fully-unrolled instruction streams that could instead use hardware loops (`C_LOOP_START gp{r}, {count} … C_LOOP_END`), to cut emitted instruction count. Companion to `SMOLVLM2_PHASE_TIMING.md` (this is optimization-track #1 from there: reduce the instruction count that drives `isa_gen` emit, `sim_env` re-parse, and `emulate` host time). Survey method: multi-agent sweep of every op-lowering file (70 unique codegen sites, 64 rollable, 24 verified for HW-loop feasibility).

## The one load-bearing fact (CORRECTED after shipping rope+norm — see note)

**Rolling is mostly a host-time win, but it is NOT exactly `sim_lat`-neutral.** The static cycle model (`transactional_emulator/testbench/aten/compare/isa_analysis.py:265-278`) multiplies each loop body by the product of its enclosing loop counts — so cycles would be loop-count-invariant *only if the rolled and unrolled bodies executed the same instructions*. They do not: the unrolled body **bakes absolute operand addresses** (`load_large_int`), while the rolled body keeps **running pointers advanced by `S_ADDI_INT` each iteration** plus `C_LOOP` control. That different instruction mix shifts `dynamic_instruction_count` and `estimated_cycles` (the `sim_lat` proxy). So rolling primarily shrinks emitted text (faster `isa_gen`, `sim_env` re-parse, emulator host wall), but it also perturbs modeled latency by a small amount (loop overhead).

> **MEASURED (rope + normalization shipped on `ATEN_OPS_UNROLL`, decoder 16/16/4):** rolling those two ops cut emitted lines 638,919 → 625,491 (**−2.1%**) and shifted `sim_lat` 5,354,236 → 5,358,616 ns (**+0.08%**), with `allclose` 100% PASS. Per-config the *line* win ranges from **−54%** (decoder 256, where rope/rmsnorm are a large fraction) to **−5%** (vision, layernorm only — SigLIP has no RoPE). The earlier "zero effect on HW latency" claim below was an oversimplification; the effect is small but real. The original convention's default is *rolled* (smaller code, slight loop overhead); `ATEN_OPS_UNROLL=1` is *unrolled* (more code, fewer cycles). The `affects_sim_lat: no` cells in the table below should be read as "small / loop-overhead," not "zero."

Corollary: rolling is only a *text* win when `body × count` is large. At big tiles (mlen=256) the C_LOOP + `S_ADDI_INT` stride overhead can exceed the saved `load_large_int` address bakes, making a rolled op emit **more** static lines than unrolled (confirmed for `vram_sub_projection`). So rolling must be applied selectively, not globally.

## Current roll state

One master boolean `unroll_loops` (env **`ATEN_OPS_UNROLL`**, `compiler.py:95`, default `0` = rolled) threads via `MemoryStateMixin` into `self._unroll` / `self.unroll_attention`. It covers exactly three families today:
- `isa_fp_ops` FPVar/FPRAM vector ops (`isa_fp_ops.py`, 5 sites) — rolled by default.
- `vram_sub_projection` 3-level GEMM nest (`vram_sub_projection_asm.py:104-127`) — rolled by default.
- ALL attention-internal ops (online-softmax, pv-multiply, scale-o, final-scaling, reset) in `isa_attention.py` — rolled by default, each with an `_unrolled` twin selected when `ATEN_OPS_UNROLL=1`.

Separately: `PLENA_ROLL_KV_GROUPS` (`plena_frontend.py:940`, default on) rolls the attention KV-group axis, but only when `batch_size==1` and `num_kv_heads>1`. FFN has its own selector `use_loop_instructions` (auto-set at `program_tensors.py:404` by `max_k_tiles<=mram_tile_capacity`, **not** on `ATEN_OPS_UNROLL`), and even its rolled path leaves the innermost K-accumulation unrolled.

**SHIPPED:** `rope_asm` and `normalization_asm` are now on `ATEN_OPS_UNROLL` too (rolled by default — items #2/#3 below). **Still NOT reached: `projection_asm`, `gemv_asm`, `embedding_asm`, `batched_matmul_asm`, `im2col`, `ffn_asm`** — the remaining opportunity.

## Rollable ops beyond attention (ranked by leverage)

| # | op | file:lines | state | est. text reduction | risk | lever |
|---|----|-----------|-------|--------------------|------|-------|
| 1 | **Projection** QKV/O (weight-row × act-col × K nest) | `projection_asm.py:184-215, 322-351; _emit_projection_chunk 53-91` | unrolled, no lever | **70–95%/proj** (×4 proj/layer; biggest driver after RoPE) | med (inner-K + act-col are clean strides; outer weight-row has conditional prefetch every mlen/blen rows) | needs code; precedent `vram_sub_projection_asm.py:104-127` |
| 2 | **RoPE** per-chunk per-position apply ✅ SHIPPED | `rope_asm.py` (single linear C_LOOP) | **rolled by default** (`ATEN_OPS_UNROLL`) | measured −54% lines on decoder 256 1L (rope+rmsnorm); 0 on vision (no RoPE) | **low** (allclose 100%; sim_lat +0.08%) | `ATEN_OPS_UNROLL` |
| 3 | **Normalization** RMS/LayerNorm hidden-dim loops ✅ SHIPPED | `normalization_asm.py` | **rolled by default** (`ATEN_OPS_UNROLL`) | measured −5% lines on vision 32 (layernorm) | **low** (allclose 100%) | `ATEN_OPS_UNROLL` |
| 4 | **FFN innermost K-accumulation** | `ffn_asm.py:1133-1140, 718-724; K-split _emit_ffn_projection_chunk 542-602` | outer rolled, **inner-K unrolled**; K-split path fully unrolled | med-high | med (guarded last-iter stride skip) | partial (`use_loop_instructions`) + needs code |
| 5 | **GEMV** (decode-path matmul) | `gemv_asm.py:72-105` | unrolled, no lever | high in decode/batch=1 | med (same conditional-prefetch as proj) | needs code (mirror proj) |
| 6 | **batched_matmul** QK/PV-style nest | `batched_matmul_asm.py:54-117` | unrolled, no lever | high where used | med — **verify it's on the SmolVLM2 path first** (attention now uses `M_BTMM` single-ops, may be cold); also has a stray debug print at 41-42 | needs code |
| 7 | **Embedding** inner rows_per_token chunk | `embedding_asm.py:82-93` | unrolled, no lever | low-med (only inner; outer token loop is data-dependent → can't roll) | low-med | needs code (inner only) |
| 8 | **im2col** patch/kernel nest (ViT patch-embed) | `im2col_asm.py:216, 223, 243-244` | unrolled, no lever | potentially large for vision | **high** (per-iter addresses not a single stride; 64-alignment shifts; f-reg save/restore) | needs code |

**Already rolled (use as references/controls):** `isa_fp_ops` (toggle exemplar), `vram_sub_projection` (the control op proving rolling is *not* always a text win), `isa_matrix.vram_block_add`. `fpvar_shift_asm` (`isa_fp_ops.py:219`) is the one always-unrolled vector op — `src_idx=i-shift` is not a constant stride.

## Attention's three roll axes (the user's hypothesis, confirmed)

All in `program_attention.py` + `plena_frontend.py`:
- **Axis 1 — per-KV-head / KV-group:** ROLLED *conditionally* via `PLENA_ROLL_KV_GROUPS` (default on) → `flash_attention_packed_groups_looped` (`program_attention.py:700`, emits `C_LOOP_START gp_kv_loop, num_kv_heads` at 798) — but **only** when `batch_size==1` AND `num_kv_heads>1`; otherwise Python-unrolled (and the batch>1 path is doubly-unrolled over batch×kv_head).
- **Axis 2 — per-attention-head-within-group:** `program_attention.py:655` `for head, s_head in enumerate(s_views)` — **NOT ROLLED, no lever at all.** Each head emits a full softmax+PV+scale+pack pipeline. This is the one attention axis with no roll switch.
- **Axis 3 — attention-internal ops** (softmax rows, pv_multiply, scale_o, final_scaling, reset): ROLLED by default via `unroll_attention` (= `ATEN_OPS_UNROLL`), each with an `_unrolled` twin.

So "currently batch/kv rolled" = axis 1 (kv-groups, conditional) + axis 3 (internals, default); the gap is **axis 2 (per-head)**.

## Lever inventory (experiment control panel)

- **`ATEN_OPS_UNROLL`** (env, `compiler.py:95-99`): `1`→unrolled, `0`/unset→rolled. Scope: `isa_fp_ops`, all attention-internal ops, `vram_sub_projection`, `vram_block_add`, `isa_tile_rows._arith_progression`. Does **not** reach projection/gemv/rope/norm/embedding/batched_matmul/im2col/ffn.
- **`PLENA_ROLL_KV_GROUPS`** (env, `plena_frontend.py:940`, default `1`): rolls attention axis 1; only active `batch_size==1` & `num_kv_heads>1`.
- **`use_loop_instructions`** (param, `ffn_asm.py:24`; auto at `program_tensors.py:404`): FFN up/gate/down C_LOOP vs fully unrolled; innermost K still unrolled either way; not tied to `ATEN_OPS_UNROLL`.
- **`use_fused_up_gate`** (param, `ffn_asm.py:25`, default False): fused path mixing C_LOOP + interleaved prefetch unrolling (12 regs).
- `PLENA_LATENCY_PROFILE` / `dc_en`: select cycle-cost column; affects `estimated_cycles` magnitude but NOT the rolled-vs-unrolled comparison.
- **New levers to add** (don't exist yet): `PLENA_ROLL_ROPE`, `PLENA_ROLL_PROJ_INNER`, `PLENA_ROLL_NORM`, `PLENA_ROLL_GEMV`, `PLENA_ROLL_EMBEDDING_INNER`, and routing `ffn_asm` onto `ATEN_OPS_UNROLL`. Recommended: gate the new ones on `ATEN_OPS_UNROLL` so the whole transformer flips with one env var.

## Experiment plan (rolled vs unrolled)

**Goal:** prove rolling shrinks emitted text + host-time (isa_gen / sim_env / emulate) while keeping `allclose` correct, and **quantify the small `sim_lat` shift** (loop overhead) — and locate where rolling is net-negative on text.

**Step 0 — characterize the cycle delta first.** Run `analyze_asm` (`isa_analysis.py:223`) on rolled vs unrolled output of an already-gated op (`vram_sub_projection` or any `isa_fp_ops`) and record both `static_instruction_lines` (drops when rolled) and `estimated_cycles` (shifts slightly — the rolled body's `S_ADDI_INT` advances + `C_LOOP` control replace the unrolled body's baked address loads). Do NOT assume cycles are equal; measure the magnitude (for rope+norm it was +0.08% on decoder 16/16/4).

**Configs (span the line-count spectrum):**
- C1: vision encoder mlen=16 (the 2.4M-line case) — max text, rolling pays most.
- C2: vision encoder mlen=32 (~600K lines).
- C3: text decoder / mlen=256 — large tiles, where per-op rolling can go net-**negative** on static lines (the crossover; include `vram_sub_projection` as the demonstrator).
- Use compile-only (`--compile-only`) for the line/host-time metrics; use the smallest viable sub-64 slice for the emulate+allclose leg (full emulate is hours/run).

**Toggle matrix (per op group, A=unrolled vs B=rolled, all else default-rolled):**
1. `ATEN_OPS_UNROLL=1` vs `0` (flips isa_fp_ops + attention-internal + vram_sub_projection + vram_block_add together).
2. `PLENA_ROLL_KV_GROUPS=0` vs `1` (attention axis 1; batch==1 multi-KV config only).
3. FFN `use_loop_instructions` False vs True (hard-set at `program_tensors.py:404` for the run, or pick a dim where the auto-rule flips).
4. NEW per-op flags (add then toggle, highest leverage first): RoPE (`rope_asm.py:48-58`), projection inner-K + act-col (`projection_asm.py:208-214 / 344-349`), normalization hidden-dim (`normalization_asm.py:38-44, 59-64`). Add an `unroll` param gated on `ATEN_OPS_UNROLL`, emit `C_LOOP_START/END` + `S_ADDI_INT`-stride bodies modeled on `isa_fp_ops` / `vram_sub_projection`.

**Measure per (config × toggle):**
- `static_instruction_lines`, `dynamic_instruction_count`, `estimated_cycles` (= sim_lat proxy) from `analyze_asm`. Expect: static lines DROP when rolled (except mlen=256 small-tile ops which may RISE); dynamic + estimated_cycles shift slightly (loop overhead from the running-pointer advances), NOT unchanged.
- `C_LOOP_START` count (sanity that rolling fired).
- Host wall-time of each stage separately: `isa_gen` (Python emit), `sim_env` re-parse (line-count dominated), `emulate` (Rust; parse-bound for unrolled → net win when rolled).
- `allclose` vs torch reference on the smallest config (correctness gate: rolled == unrolled == reference).

**Expected outcome:** `allclose` unchanged in every cell (correctness holds). `sim_lat`/`estimated_cycles` shift by a *small* amount rolled vs unrolled (loop overhead — measured +0.08% for rope+norm; sign/magnitude depend on how many baked address-loads the rolled stride-advances replace). Static lines + isa_gen+sim_env+emulate host-seconds fall when rolling RoPE + projection + norm — sharply on configs where those ops are a large fraction (decoder), modestly on vision (norm only, since SigLIP has no RoPE). At mlen=256, per-op rolling of small-tile ops can increase static lines — apply selectively.

**Recommendation:** ship **RoPE + normalization** rolling first (low risk, clean single-stride bodies, no conditional-body problem), then **projection inner-K / act-col**, all gated on `ATEN_OPS_UNROLL` so the whole transformer flips with one env var. Defer the outer weight-row / im2col / batched_matmul rolls (conditional-body or irregular-stride risk) until the easy wins are measured.
