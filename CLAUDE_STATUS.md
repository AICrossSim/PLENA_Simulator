# CLAUDE Session Status — PLENA ATen Migration

**Active branch:** `kev/aten-on-main` (rebased on top of `main`)
**Also exists:** `kev/aten` (original branch, pre-rebase)
**Last Updated:** 2026-02-23
**Task:** Migrate friend's (Ziqian Gao) testbench work to ATen-style operator dispatch

---

## Project Overview

**PLENA** = Programmable Long-context Efficient Neural Accelerator (hardware simulator for LLM inference).

- Root repo: `/home/khl22/new_plena/`
- Submodule with all active work: `/home/khl22/new_plena/PLENA_Simulator/`
- Friend's original work is on branch: `feature/add-testbench-files` (committed, same HEAD as `kev/aten`)
- User's aten migration is **all uncommitted** on `kev/aten`

---

## What the ATen Migration Is

Friend wrote tests that called `PLENAProgram` methods **inline** directly.
User (kev) is wrapping them in an **ATen-style operator dispatch system** (`plena/ops/`):

```
plena/
  __init__.py
  native_ops.yaml          # Op declarations (like PyTorch's native_functions.yaml)
  ops/
    __init__.py            # Module-level dispatch fns (ops.softmax, ops.linear, ...)
    registry.py            # OpRegistry class with Backend.CPU / Backend.PLENA
    cpu/                   # PyTorch golden reference implementations
      softmax_ops.py       # softmax_cpu
      linear_ops.py        # linear_cpu
      norm_ops.py          # rms_norm_cpu, layer_norm_cpu
      ffn_ops.py           # ffn_cpu
      attention_ops.py     # flash_attention_cpu
    plena/                 # PLENA ISA-generation implementations
      softmax_ops.py       # softmax_plena (online softmax algorithm)
      linear_ops.py        # linear_plena (sub-matrix projection)
      norm_ops.py          # rms_norm_plena, layer_norm_plena
      ffn_ops.py           # ffn_plena (uses ffn_asm template)
      attention_ops.py     # flash_attention_plena (online softmax flash attn)
      conv_ops.py          # conv2d_plena (TRUE on-chip im2col via V_SHFT_V)
```

**Pattern:**
```python
registry = OpRegistry.load()
registry.set_backend(Backend.CPU)
golden = ops.softmax(X, scale=1.0)       # PyTorch reference

registry.set_backend(Backend.PLENA)
result = ops.softmax(prog, X_batch, scale=1.0)  # Generates ISA
```

---

## Registered Operators (native_ops.yaml)

| Op | Category | CPU impl | PLENA impl | Test file |
|----|----------|----------|------------|-----------|
| `softmax` | primitive | `softmax_cpu` | `softmax_plena` | `fpvar_softmax_aten_test.py` |
| `linear` | primitive | `linear_cpu` | `linear_plena` | `linear_aten_test.py` |
| `rms_norm` | primitive | `rms_norm_cpu` | `rms_norm_plena` | `rms_norm_aten_test.py` |
| `layer_norm` | primitive | `layer_norm_cpu` | `layer_norm_plena` | `layer_norm_aten_test.py` |
| `ffn` | composite | `ffn_cpu` | `ffn_plena` | `ffn_aten_test.py` |
| `flash_attention` | composite | `flash_attention_cpu` | `flash_attention_plena` | `flash_attention_aten_test.py` |
| `conv2d` | composite | `conv2d_cpu` (matmul on im2col) | `conv2d_plena` (TRUE on-chip im2col) | `conv2d_aten_test.py` |

---

## Session 8 Progress (2026-02-24) — im2col Without V_SHFT_V (Documented ISA Only)

### Goal
Replace V_SHFT_V (opcode 0x32, not formally documented per PhD lead) with documented
instructions only: V_MUL_VV + V_RED_SUM + S_ST_FP + S_MAP_V_FP for element extraction.

### New Files

| File | Change |
|------|--------|
| `compiler/asm_templates/im2col_asm_no_shift.py` | New ASM generator — basis vectors + extract-per-element approach |
| `compiler/asm_templates/__init__.py` | Export `im2col_asm_no_shift` |
| `plena/ops/plena/conv_ops.py` | Rewrote `conv2d_plena` to use `im2col_asm_no_shift` |
| `transactional_emulator/testbench/conv2d_aten_test.py` | New test — H=67, W=4, W_padded=64, OW=1 |

### Algorithm (per output row m)

```
oh = m // OW,  ow = m % OW
for c in 0..C_in-1:
    for kr in 0..K-1:
        scratch = H_PREFETCH_V(hbm_off)        # load K real elements at positions 0..K-1
        for kc in 0..K-1:
            temp = V_MUL_VV(scratch, e_kc)     # zero everywhere except position kc
            S_ADD_FP f_ex, f0, f0              # zero accumulator
            V_RED_SUM f_ex, temp               # f_ex = scratch[kc]
            S_ST_FP f_ex, fpsram[c*K*K+kr*K+kc]  # store element
VRAM[m] = S_MAP_V_FP fpsram[0..K_col-1]       # flush row
```

### Critical Bug Fixed: fp_preload goes into fpsram, not fp_reg

`fp_reg` (FP registers f0-f7) initialises to all-zeros (Rust: `[f16::ZERO; 8]`).
`fp_preload` values are copied into `fpsram` (FP_SRAM scratchpad), NOT into fp_reg.

**Bug:** Assumed `fp_preload[1] = 1.0` set `fp_reg[1] = 1.0`. It only set `fpsram[1] = 1.0`.
`S_ST_FP f1, gp0, 0` wrote 0.0 (not 1.0) to basis vector positions → basis all-zeros → zero output.

**Fix:** Added `S_LD_FP f{fp_one_reg}, gp0, {fp_one_reg}` as first instruction of im2col ASM
to explicitly load 1.0 from fpsram into fp_reg[1] before the basis construction.

### Test Result

```
conv2d_aten_test: [ATen-style conv2d test PASSED]
  allclose (atol=0.2, rtol=0.2): 95.58% match
  Max error: 1.75 (expected MX8 quantization noise)
  Latency: 105613 ns   (~2× V_SHFT_V due to extra instructions per element)
  HBM read: 532480 bytes
```

---

## Session 7 Progress (2026-02-23) — TRUE On-Chip im2col via V_SHFT_V

### Goal
Replace the CPU-side im2col pre-processing with TRUE hardware im2col on PLENA:
raw NCHW input in HBM → im2col matrix in VRAM entirely on-chip, before the systolic matmul.

### New ISA: V_SHFT_V (opcode 0x32)

**Semantics:**
```
V_SHFT_V rd, rs1, rs2
  VRAM[rd][i] = VRAM[rs1][i - shift] if i >= shift else 0.0
  where shift = GP[rs2]
```
Used to place K loaded elements at their target column position in the im2col row.

### Files Changed

| File | Change |
|------|--------|
| `transactional_emulator/src/op.rs` | Added `V_SHFT_V { rd, rs1, rs2 }` enum variant + decode for opcode `0x32` |
| `transactional_emulator/src/main.rs` | Implemented `shift()` method on `VectorMachine` |
| `compiler/asm_templates/im2col_asm.py` | New ASM generator (H_PREFETCH_V → V_MUL_VV → V_SHFT_V → V_ADD_VV) |
| `compiler/asm_templates/__init__.py` | Exported `im2col_asm` |
| `plena/ops/plena/conv_ops.py` | `conv2d_plena` calls `im2col_asm` before `linear_plena`; supports `W_padded` |
| `transactional_emulator/testbench/conv2d_aten_test.py` | End-to-end test with H=67, W=4, W_padded=64, OW=1 |

### HBM Alignment Fix

H_PREFETCH_V requires element addresses to be multiples of 64 (`assert!(addr.is_multiple_of(64))`).

**Solution:** store input row-padded with `W_padded=64`:
- HBM layout: `(C_in*H, W_padded)` shape
- HBM offset = `(c*H + oh+kr) * W_padded + ow`
- With OW=1 (ow=0 always): offset = `(c*H + oh+kr) * 64` → always a multiple of 64 ✓
- Test params chosen: H=67, W=4 → OH=64, OW=1, M=64, K_col=64, N=64 (all = mlen)

### im2col Algorithm (per output row m)

```
oh = m // OW,  ow = m % OW
accum = zeros(VLEN)
for c in 0..C_in-1:
    for kr in 0..K-1:
        hbm_off = (c*H + oh+kr)*W_padded + ow   # 64-aligned iff ow=0
        scratch = H_PREFETCH_V(hbm_off)          # load VLEN elements
        scratch = scratch * mask_vec              # zero positions K..63
        shift   = c*K*K + kr*K
        scratch = V_SHFT_V(scratch, shift)        # place at target column
        accum  += scratch                          # V_ADD_VV
store accum -> im2col_out[m]
```

### Commits

```
c39c0db  feat: implement TRUE on-chip im2col on PLENA via V_SHFT_V  (parent repo)
4da66b8  feat: add im2col_asm template for on-chip im2col via V_SHFT_V  (compiler submodule)
```

### Test Result

```
conv2d_aten_test: [ATen-style conv2d test PASSED]
  allclose (atol=0.2, rtol=0.2): 95.58% match
  Max error: 1.75 (expected MX8 quantization noise)
  Latency: 51886 ns
  HBM read: 532992 bytes
```

---

## All Test Status (as of Session 7)

### Layer 3 — ATen-style `plena.ops`

| Test | Status |
|------|--------|
| `./run.sh test-softmax` | ✅ PASS |
| `./run.sh test-linear` | ✅ PASS |
| `./run.sh test-rms-norm` | ✅ PASS |
| `./run.sh test-layer-norm` | ✅ PASS |
| `./run.sh test-ffn` | ✅ PASS |
| `./run.sh test-flash-attention` | ✅ PASS |
| `./run.sh test-conv2d` | ✅ PASS |

### Layer 2 — DeveloperCompiler tests

| Test | Status |
|------|--------|
| `rms` | ✅ PASS |
| `layer_norm` | ✅ PASS |
| `linear` | ✅ PASS |
| `ffn` | ✅ PASS |
| `flashattn_qkt` | ✅ PASS |
| `btmm_bmmwo` | ✅ PASS |
| `projection_T` | ✅ PASS |
| `bmm` | ✅ PASS |
| `s_map_v` | ✅ PASS |
| `two_input` | ✅ PASS |

---

## ISA Opcode Reference (operation.svh vs Notion doc)

Complete opcode table from `compiler/doc/operation.svh`. Items marked ⚠️ are **missing from the Notion ISA doc**.

| Opcode | Value | Status |
|--------|-------|--------|
| `M_MM` | `6'h01` | ✅ documented |
| `M_TMM` | `6'h02` | ✅ documented |
| `M_BMM` | `6'h03` | ✅ documented |
| `M_BTMM` | `6'h04` | ⚠️ not in doc |
| `M_BMM_WO` | `6'h05` | ⚠️ not in doc |
| `M_MM_WO` | `6'h06` | ✅ documented |
| `M_MV` | `6'h07` | ✅ documented |
| `M_TMV` | `6'h08` | ✅ documented |
| `M_BMV` | `6'h09` | ⚠️ not in doc |
| `M_BTMV` | `6'h0A` | ⚠️ not in doc |
| `M_MV_WO` | `6'h0B` | ✅ documented |
| `M_BMV_WO` | `6'h0C` | ⚠️ not in doc |
| `V_ADD_VV` | `6'h0D` | ✅ documented |
| `V_ADD_VF` | `6'h0E` | ✅ documented |
| `V_SUB_VV` | `6'h0F` | ✅ documented |
| `V_SUB_VF` | `6'h10` | ✅ documented |
| `V_MUL_VV` | `6'h11` | ✅ documented |
| `V_MUL_VF` | `6'h12` | ✅ documented |
| `V_EXP_V` | `6'h13` | ✅ documented |
| `V_RECI_V` | `6'h14` | ✅ documented (doc calls it `V_REC_V`) |
| `V_RED_SUM` | `6'h15` | ✅ documented |
| `V_RED_MAX` | `6'h16` | ✅ documented |
| `S_ADD_FP` | `6'h17` | ✅ documented |
| `S_SUB_FP` | `6'h18` | ✅ documented |
| `S_MAX_FP` | `6'h19` | ✅ documented |
| `S_MUL_FP` | `6'h1A` | ✅ documented |
| `S_EXP_FP` | `6'h1B` | ✅ documented |
| `S_RECI_FP` | `6'h1C` | ⚠️ not in doc |
| `S_SQRT_FP` | `6'h1D` | ⚠️ not in doc |
| `S_LD_FP` | `6'h1E` | ✅ documented |
| `S_ST_FP` | `6'h1F` | ✅ documented |
| `S_MAP_V_FP` | `6'h20` | ✅ documented |
| `S_ADD_INT` | `6'h21` | ✅ documented |
| `S_ADDI_INT` | `6'h22` | ✅ documented |
| `S_SUB_INT` | `6'h23` | ✅ documented |
| `S_MUL_INT` | `6'h24` | ✅ documented |
| `S_LUI_INT` | `6'h25` | ✅ documented |
| `S_LD_INT` | `6'h26` | ✅ documented |
| `S_ST_INT` | `6'h27` | ✅ documented |
| `S_DIV_INT` | — | ⚠️ in doc but **not in svh** (removed?) |
| `H_PREFETCH_M` | `6'h28` | ✅ documented |
| `H_PREFETCH_V` | `6'h29` | ✅ documented |
| `H_STORE_V` | `6'h2A` | ✅ documented |
| `C_SET_ADDR_REG` | `6'h2B` | ✅ documented |
| `C_SET_SCALE_REG` | `6'h2C` | ✅ documented |
| `C_SET_STRIDE_REG` | `6'h2D` | ✅ documented |
| `C_SET_V_MASK_REG` | `6'h2E` | ⚠️ not in doc |
| `C_LOOP_START` | `6'h2F` | ⚠️ not in doc |
| `C_LOOP_END` | `6'h30` | ⚠️ not in doc |
| `V_PS_V` | `6'h31` | ⚠️ not in doc — prefix scan, added Aug 22 2025 by Ali |
| `V_SHFT_V` | `6'h32` | ⚠️ not in doc — barrel right-shift, added Aug 22 2025 by Ali |
| `C_HADAMARD_TRANSFORM` | `6'h33` | ⚠️ not in doc |
| `C_BREAK` | `6'h34` | ✅ in doc (no opcode value given) |

**Source:** `compiler/doc/operation.svh`. Extensions (`0x31`–`0x34`) traced to commit `020d4b3 "ali new instructions"` (ali-r5, Aug 22 2025) in the old `/home/khl22/plena` repo.

---

## Key Architecture Notes

### Hardware Constants (plena_settings.toml, BEHAVIOR config)
- VLEN = MLEN = 64
- HLEN = 16
- BLEN = 4
- PREFETCH_V_AMOUNT = 4
- HBM element format: MX8 (8-bit E4M3, block=8, real_data_ratio=1.125)

### `use_stride_mode` rule
- `True` when `hidden_size > mlen` — VRAM rows are stride-interleaved
- `False` when `hidden_size == mlen` — batch-contiguous layout
- Default in `emulator_runner.py` is `True` → always set explicitly in `comparison_params`

### FFN hardware operation order
- "Upsize Linear" = W_up → gp4 (SiLU input)
- "Gate Projection" = W_gate → gp6
- `output = W_down @ (silu(W_up @ x) * (W_gate @ x))`

### How to run tests
```bash
cd /home/khl22/new_plena/PLENA_Simulator

# Layer 3 ATen tests (via run.sh which handles nix+conda+PYTHONPATH):
./run.sh test-softmax
./run.sh test-linear
./run.sh test-rms-norm
./run.sh test-layer-norm
./run.sh test-ffn
./run.sh test-flash-attention

# conv2d test (direct, needs PYTHONPATH):
PYTHONPATH=/home/khl22/new_plena/PLENA_Simulator:/home/khl22/new_plena/PLENA_Simulator/tools \
  conda run -n plena python3 transactional_emulator/testbench/conv2d_aten_test.py

# Layer 2 tests:
./run.sh build <test_name>   # e.g. ./run.sh build bmm
```

### Push Status
**Blocked** — `booth-algo` SSH key lacks write access to `AICrossSim/PLENA_Simulator`.
Waiting for collaborator access to be granted.

---

## Key File Locations

| Purpose | Path |
|---------|------|
| ATen registry | `plena/ops/registry.py` |
| Op declarations | `plena/native_ops.yaml` |
| CPU backends | `plena/ops/cpu/*.py` |
| PLENA backends | `plena/ops/plena/*.py` |
| im2col ASM template | `compiler/asm_templates/im2col_asm.py` |
| High-level API | `transactional_emulator/testbench/plena_program.py` |
| Low-level compiler | `transactional_emulator/testbench/developer_compiler.py` |
| Memory manager | `transactional_emulator/testbench/sub_matrix_manager.py` |
| ATen tests | `transactional_emulator/testbench/*_aten_test.py` |
| Build commands | `justfile` / `run.sh` |
