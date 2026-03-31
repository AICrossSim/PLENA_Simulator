# PLENA Simulator — Project Context

**PLENA** = Programmable Long-context Efficient Neural Accelerator (hardware LLM inference simulator).

---

## CI Checks (run before every commit)

```bash
# Python formatting (must pass — CI blocks on this)
source /home/khl22/miniconda3/etc/profile.d/conda.sh && conda activate plena
ruff format .
ruff check --fix .

# Rust formatting (must pass — CI blocks on this)
nix develop --command bash -c "cd transactional_emulator && cargo fmt --all"
```

Both `ruff format --check` and `cargo fmt --all -- --check` are enforced in CI.
`ruff check` has ~838 pre-existing naming convention errors (N803, RUF001, etc.) that are not yet enforced.

---

## Environment

- Working dir: `/home/khl22/new_plena/PLENA_Simulator/`
- Run tests: `bash run.sh <recipe>` — auto-enters `nix develop`, activates conda `plena` env, sets `PYTHONPATH` to project root + `tools/`
- Direct python (if already in nix+conda): `/home/khl22/.conda/envs/plena/bin/python`
- PYTHONPATH must include: `$(pwd):$(pwd)/tools`

---

## Hardware Constants

| Constant | Value | Note |
|----------|-------|------|
| MLEN / VLEN | 64 | Vector/matrix lane width |
| BLEN | 4 | Batch lanes |
| HLEN | 16 | |
| MAX_K_TILES | 4 | MRAM limit: 4 × 64² = 16384 elements; max single-pass K_col = 256 |
| PREFETCH_V_AMOUNT | 4 | H_PREFETCH_V loads 4 VRAM rows |
| HBM format | MXFP8 | E4M3, block=8 |

---

## Project Structure

ATen-style operator dispatch system (`plena/ops/`):

```
plena/
  native_ops.yaml          # Op declarations
  ops/
    __init__.py            # Dispatch fns: ops.softmax, ops.linear, ...
    registry.py            # OpRegistry: Backend.CPU / Backend.PLENA
    cpu/                   # PyTorch golden reference implementations
    plena/                 # PLENA ISA-generation backends
      softmax_ops.py / linear_ops.py / norm_ops.py
      ffn_ops.py / attention_ops.py / conv_ops.py / embedding_ops.py

compiler/asm_templates/    # ISA code generators (im2col_asm_no_shift.py, rope_asm.py, ...)
transactional_emulator/testbench/
  plena_program.py         # High-level API
  developer_compiler.py   # Low-level compiler
  sub_matrix_manager.py   # Memory manager
  model_layer_test_builder.py  # Shared HF model test infra
  *_aten_test.py           # Per-op tests
justfile / run.sh          # Build recipes
```

Call chain: `plena_program.py` → `developer_compiler.py` → `sub_matrix_manager.py`

### Registered Operators

`softmax`, `linear`, `rms_norm`, `layer_norm`, `ffn`, `flash_attention`, `conv2d`, `embedding_add`, `rope`

---

## Critical Conventions

### FPRAM Slot Layout (pipeline-wide)

| Slot | Value | Used by |
|------|-------|---------|
| 0 | 0.0 | reserved |
| 1 | attn_scale | flash_attention |
| 2 | -inf | flash_attention online softmax |
| 3 | eps (1e-6) | rms_norm / layer_norm |
| 4 | 1/hidden_size | rms_norm / layer_norm |
| 5 | 1.0 | FFN SiLU sigmoid denominator |

**rms_norm in decoder pipeline** MUST use:
```python
prog.rms_norm(X, eps_offset=3, reci_hid_offset=4)
```
NOT `ops.rms_norm` (uses default slots 1,2 — conflicts with flash_attn).

### HBM / VRAM Quantization

- `create_mem_for_sim` applies MXFP8 quantization when writing to HBM.
- Golden reference MUST call `quantize_to_mxfp()` on all HBM-stored tensors (weights, K, V).
- VRAM activations are NOT quantized.

### K_col Padding (conv2d / im2col)

`conv_ops.py` allocates `im2col_out` with `K_col_padded = ceil(K_col/64)*64`.
Tests must zero-pad weight tensors to match.

### K-split (K_col > 256)

`linear_plena` auto-splits K into chunks of ≤ MAX_K_TILES=4 with `vram_block_add_to` accumulation.
- First chunk: write to output directly; subsequent: write to temp, accumulate.
- `k_block_start, k_block_count` threaded through: `linear_ops.py` → `plena_program.py` → `developer_compiler.py` → `sub_matrix_manager.py`.

### Stride Mode

`use_stride_mode=True` when `hidden_size > mlen=64`. Always set explicitly in `comparison_params`.

### FFN Hardware Operation Order

`output = W_down @ (silu(W_up @ x) * (W_gate @ x))`
- W_up → gp4 (SiLU input); W_gate → gp6 (multiplied as-is)

### im2col fp_one_reg

`conv2d_plena(..., fp_one_reg=5)` — slot 5 for 1.0 (slots 1-4 reserved for pipeline).

---

## VRAM Layout (decoder pipeline, seq=64, hidden=64)

```
X      rows 0-63    addr 0-4095
POS    rows 64-127  addr 4096-8191
QROT   rows 128-191 addr 8192-12287
COS    rows 192-255 addr 12288-16383
SIN    rows 256-319 addr 16384-20479
S      rows 320-383 addr 20480-24575   (flash_attn scratch)
PV     rows 384-447 addr 24576-28671   (flash_attn scratch)
O      rows 448-511 addr 28672-32767   (flash_attn output, actual start addr 28736)
```

---

## Known Bugs (Fixed)

**FFN VRAM conflict in decoder pipeline**
`_ffn_asm_with_loops` gate projection at `gate_result = batch*hidden + inter*batch`.
With hidden=64, batch=64: `gate_result = 4096 + inter*64`. Max safe `inter_dim=192`; use `inter_dim=128` for safety (gate_result_max=20539 < 28736).

**im2col corrupts fp_sram**
im2col zeros fp_sram[0..K-1] for basis construction, destroying pipeline slots.
Fix: save precious slots [1,2,3,4,5] into f_regs [1,3,4,6,7] before im2col, restore after.
Parameter: `fp_sram_precious_slots` (default [1,2,3,4,5]).

**NumPy 2.x / PyTorch compat**
- `np.array(torch_tensor)` → must call `.numpy()` first
- `torch.from_numpy(x)` → use `torch.tensor(x, dtype=...)`

**LlamaRMSNorm eps attribute**
Use: `getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5))` — NOT `norm.eps`.

---

## HuggingFace Model Loading

`model_layer_test_builder.py` provides:
- `get_model_dims(model_id)` — probe HF config without loading weights
- `slice_dims_for_sim(dims)` — clip to sim limits (hidden≤128, inter≤256)
- `load_ffn_weights(model_id, layer_idx)` — tries `AutoModelForCausalLM` first, falls back to `AutoModel`
- `quantize_to_mxfp(tensor)` — MXFP8 round-trip
- `build_and_run_ffn_test(...)` / `build_and_run_decoder_test(...)` — e2e runners

LlamaForCausalLM path: `model.model.layers[i].mlp`
SmolVLM2 path: `model.text_model.layers[i].mlp`

---

## Test Suite

```bash
# Primitives
bash run.sh test-softmax
bash run.sh test-linear
bash run.sh test-rms-norm
bash run.sh test-layer-norm
bash run.sh test-ffn
bash run.sh test-flash-attention
bash run.sh test-bmm
bash run.sh test-embedding-add
bash run.sh test-rope

# Real-model FFN
bash run.sh test-ffn-smollm2-135m     # hidden=128, inter=256, 100% allclose
bash run.sh test-ffn-clm60m           # hidden=128, inter=256, 100% allclose

# Full decoder pipeline
bash run.sh test-decoder-smollm2-135m # seq=64, hidden=64, inter=128, ~99% allclose

# Unit tests (no HF download)
bash run.sh test-model-builder        # 8/8 pass

# Conv2d / im2col
bash run.sh test-conv2d               # baseline K_col=64
bash run.sh test-conv2d-tiled         # K_col=128 (2 tiles)
bash run.sh test-conv2d-siglip        # K_col=192 (3 tiles)
bash run.sh test-conv2d-siglip-real   # K_col=588 K-split (10 tiles, 3 chunks), 90.33% allclose

# Vision encoder pipeline
python3 transactional_emulator/testbench/smolvlm2_vision_encoder_test.py  # 99.95% allclose
```
