# im2col: shift vs no-shift benchmark (SmolVLM2 vision patch embedding)

Generated: `2026-06-03`. Compares the two on-chip im2col code paths for the SigLIP patch-embedding Conv2d (stride = patch_size = 16), across native hardware tile sizes.

## The two paths

`aten/ops/plena/conv_ops.py` (`conv2d_plena`) selects between two im2col implementations, toggled by `use_shift` / env `CONV_USE_SHIFT`:

- **no-shift** (default, `CONV_USE_SHIFT=0`, `asm_templates/im2col_asm_no_shift.py`) — extracts each patch element scalar-by-scalar through FP_SRAM (`H_PREFETCH_V` + `V_MUL_VV` + `V_RED_SUM` + `S_ST_FP` + `S_MAP_V_FP`). Flexible (any alignment) but verbose.
- **shift** (`CONV_USE_SHIFT=1`, `asm_templates/im2col_asm.py`) — places a whole patch with one `V_SHIFT_V` per K-group. Loads each patch at its exact `pixel_col` (patch lands at `[0:K]`), masks, right-shifts to the target column.

The shift path was unusable for SmolVLM2 vision until **PLENA_Compiler #57** removed a stale compiler-side 64-alignment assertion. The patch columns are stride-16 (not 64-aligned); the emulator's byte-block HBM gather (`dma::transfer_mx_from_hbm`, #62) walks each load one 64-byte block at a time and clamps to block boundaries, so a non-64-aligned `H_PREFETCH_V` start is loaded correctly — no emulator change needed.

## Method

1-layer SmolVLM2 vision encoder (`run_model.py smolvlm2 --case vision-layers --layers 1 --seq-len 64`), native configs (mlen=16/32/64/256), both modes. Captured: `static ISA lines` (codegen size, drives host isa_gen + sim_env re-parse), `sim_lat` (modeled HW latency, `executor.now()`, deterministic), and `allclose` vs the MXFP8+BF16 golden reference. Reproduce with `CONV_USE_SHIFT={0,1}` and the configs below; artifacts in `/tmp/im2col_cmp/`.

## Results

| config (mlen) | mode | ISA lines | sim_lat | allclose |
|---------------|------|----------:|--------:|:--------:|
| 16/16/4 | no-shift | 2,369,505 | 54.855ms | PASS |
| 16/16/4 | shift | 2,092,822 (−12%) | 54.138ms (−1.3%) | PASS |
| 32/32/4 | no-shift | 599,202 | 39.826ms | PASS |
| 32/32/4 | shift | 335,688 (−44%) | 39.120ms (−1.8%) | PASS |
| 64/64/16 | no-shift | 321,664 | 3.812ms | PASS |
| 64/64/16 | shift | 84,047 (−74%) | 3.165ms (−17%) | PASS |
| 256/256/64 | no-shift | 317,805 | 5.649ms | PASS |
| 256/256/64 | shift | 79,804 (−75%) | 4.996ms (−12%) | PASS |

shift is **correct (allclose 100% everywhere) and strictly better** on both axes — but the *percentage* win grows sharply with mlen (−12% → −75% lines). That percentage is misleading; see below.

## Key finding: the shift win is essentially mlen-INDEPENDENT in absolute terms

The percentage swings only because the *denominator* (total program) collapses as mlen grows. The **absolute** saving (no-shift − shift) is nearly constant across a 16× range of mlen:

| mlen | lines saved | sim_lat saved | total program (no-shift) |
|------|------------:|--------------:|-------------------------:|
| 16 | 276,683 | 0.72ms | 2,369,505 / 54.9ms |
| 32 | 263,514 | 0.71ms | 599,202 / 39.8ms |
| 64 | 237,617 | 0.65ms | 321,664 / 3.8ms |
| 256 | 238,001 | 0.65ms | 317,805 / 5.6ms |

≈ **240–277K lines and ≈0.65–0.72ms saved, regardless of mlen.** shift is exactly as useful in absolute terms at mlen=16 as at mlen=256; it only *shows up* as a big percentage once the surrounding compute gets cheap.

### Why the im2col cost (and the shift saving) is mlen-independent

The patch embed extracts the **same image** — `C_in · K · K · num_patches` patch elements — independent of the hardware tile size. no-shift emits work proportional to that element count (scalar gather); shift emits work proportional to `num_patches · C_in · K` (one placement per K-group). Both are set by the **patch geometry**, not mlen, so their difference barely moves with mlen.

### Why the *rest* of the program shrinks with mlen (which is what changes the %)

1. **Coarser tiling** — larger mlen ⇒ fewer, larger matmul tiles ⇒ far fewer instructions/cycles for the projections, FFN, and attention matmuls.
2. **The attention cliff at mlen=64** — SmolVLM2 head_dim=64. At mlen<64 (16/32), head_dim > mlen, so attention takes the **non-packed, multi-col-block** path (2 col-blocks at mlen=32, 4 at mlen=16) — very expensive. At mlen≥64, head_dim == mlen ⇒ the **single-col-block/packed** path, ~10× cheaper (note sim_lat drops 39.8ms → 3.8ms from mlen=32 → 64). That cliff is why the im2col suddenly *dominates* the program at mlen=64+.

So at mlen=16/32 the ~0.7ms / ~250K-line im2col win is buried under a 2M-line, 40–55ms attention/projection program; at mlen=64/256 the same win is most of a tiny program.

## Takeaways

- The shift im2col is strictly better — smaller code, lower modeled latency, identical numerics — so prefer it wherever the patch geometry allows (it stays opt-in via `CONV_USE_SHIFT=1`; default remains `im2col_asm_no_shift`).
- Its *value as a fraction of the program* is large only at mlen≥64 (where the patch embed dominates). At mlen<64 it still saves the same absolute amount, but the non-packed attention + fine-tile matmuls dwarf it — so don't expect a big headline % there.
- Corollary: the im2col is **not** the lever for shrinking the giant mlen=16 vision program (that's attention/projection); the im2col is the lever at mlen≥64.
- mlen=16's row needs the O(n²) ISA-emit fix (PLENA_Compiler #56) to compile in the first place; without it the no-shift path runs away (>30 min).
