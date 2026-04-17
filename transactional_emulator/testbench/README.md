# PLENA Testbench

The Python-side testbench for the PLENA transactional emulator. Every file here either **is** the compiler (flat root), provides shared test infrastructure (flat root), or is a test driver (inside a subdirectory).

## Directory layout

```
testbench/
├── plena_compiler.py         # single-file PLENA compiler (PlenaCompiler class)
├── plena_program.py          # back-compat shim → re-exports from plena_compiler
├── tile_compiler.py          # back-compat shim → re-exports from plena_compiler
│
├── model_layer_test_builder.py   # HF weight loading + golden runners
├── test_model_layer_builder.py   # unit tests for the builder above
├── test_data_gen.py              # deterministic tensor generators
├── emulator_runner.py            # Rust emulator driver
├── config_utils.py               # test config helpers
├── check_mem.py                  # memory-layout sanity tool
│
├── aten/                          # ATen-style DSL operator tests (9)
├── conv/                          # conv2d / vision encoder tests (5)
├── models/                        # full-model decoder / FFN tests (9)
├── misc/                          # ad-hoc experiments + asm generators (3)
└── direct_emit/                   # pre-ATen asm_template regressions (17)
```

`plena_compiler.py`, `plena_program.py`, and `tile_compiler.py` stay flat because every test imports from them via `from plena_compiler import PlenaCompiler`. The testbench root is on `PYTHONPATH` (set by `run.sh`) so subdir tests can still resolve those modules.

## Flat-root files

| File | Role |
|---|---|
| `plena_compiler.py` | Single class `PlenaCompiler(DeveloperCompiler → TileCompiler)` that owns the DSL API, ISA emission, memory allocators, and interrupt machinery. This is the post-Phase-2b.4 collapsed version. |
| `plena_program.py` | Thin re-export shim. Legacy imports like `from plena_program import PLENAProgram` still work; `PLENAProgram = PlenaCompiler` is aliased. |
| `tile_compiler.py` | Thin re-export shim. Same pattern for legacy `from tile_compiler import TileCompiler, MRAMAllocator, …`. |
| `model_layer_test_builder.py` | Loads a HuggingFace model layer (FFN / decoder / vision-encoder), applies MXFP8 quantization, and drives `build_and_run_*_test` helpers. Used by every full-model test. |
| `test_model_layer_builder.py` | Unit tests for the builder — runs standalone with no HF download. |
| `test_data_gen.py` | Seeded tensor generators so tests are deterministic across runs. |
| `emulator_runner.py` | Invokes the Rust `transactional_emulator` binary on generated ISA. |
| `config_utils.py` | Shared comparison tolerances, `comparison_params` helper. |
| `check_mem.py` | Small diagnostic script for inspecting VRAM/HBM usage snapshots. |

## Subdirectory contents

### `aten/` — operator correctness tests (9 files)

Each test drives an operator via the `PlenaCompiler` Python DSL + `OpRegistry` dispatch. Since the `compile_module` driver tests were retired, every remaining ATen test is a single-driver DSL test, so the `_dsl_` suffix has been dropped — all files use the `<op>_test.py` form.

| Operator | Test |
|---|---|
| rms_norm | `rms_norm_test.py` |
| layer_norm | `layer_norm_test.py` |
| linear | `linear_test.py` |
| ffn | `ffn_test.py` |
| flash attention (MHA) | `flash_attention_test.py` |
| flash attention (GQA) | `flash_attention_gqa_test.py` |
| embedding_add | `embedding_add_test.py` |
| rope | `rope_test.py` |
| fpvar_softmax | `fpvar_softmax_test.py` |

The `compile_module` (`torch.export` → IR → ISA) driver tests (`rms_norm_test.py`, `layer_norm_vlen128_test.py`, `linear_test.py`, `ffn_matched_test.py`, `decoder_test.py`) were retired; `compile_module` coverage now lives in the `compiler/aten` submodule. The pre-existing `bmm_test.py` and `v_shift_v_test.py` were moved into `direct_emit/` because they call `compiler/asm_templates/*` directly rather than going through the DSL.

### `conv/` — conv2d / vision encoder (5 files)

| File | Purpose |
|---|---|
| `conv2d_test.py` | Primitive conv2d correctness, baseline K_col=64. |
| `conv2d_tiled_im2col_test.py` | Multi-tile im2col (K_col=128). Pre-existing flaky at `ATEN_UNROLL=1` — see top of file. |
| `conv2d_siglip_ksize14_test.py` | K_col=588, tests 3-chunk K-split (`MAX_K_TILES=4`). |
| `conv2d_siglip_real_k14_test.py` | Same, with real SigLIP weights. |
| `smolvlm2_vision_encoder_test.py` | End-to-end vision encoder pipeline (conv + norm + attention). |

### `models/` — full-model tests and profilers (9 files)

| File | Purpose |
|---|---|
| `clm60m_ffn_test.py` | AICrossSim/clm-60m FFN layer. |
| `smollm2_135m_ffn_test.py` | HuggingFaceTB/SmolLM2-135M FFN. |
| `smolvlm2_256m_ffn_test.py` | SmolVLM2 text-model FFN. |
| `smollm2_135m_decoder_test.py` | SmolLM2 decoder pipeline (norm → QKV → flash attn → O → FFN). |
| `llada_8b_decoder_test.py` | LLaDA-8B decoder (KV-cache path, longer seq). |
| `ffn_test.py` | Hand-written FFN reference (pre-ATen). |
| `llada_multilayer_decoder_profile.py` | Profiler — latency-per-layer report. Not a pass/fail test. |
| `smolvlm2_multilayer_decoder_profile.py` | Same, for SmolVLM2. |
| `llada_lm_head_asm_gen.py` | Generates lm_head ASM only. Used as input to offline perf analysis. |

### `misc/` — one-offs (3 files)

| File | Purpose |
|---|---|
| `decoder_asm_gen.py` | Legacy decoder ASM dump (pre-ATen path). |
| `flash_attention_gqa_fused_test.py` | Pre-ATen fused GQA driver. |
| `flash_attention_gqa_naive_test.py` | Pre-ATen naive GQA driver. |

### `direct_emit/` — pre-ATen asm_template regressions (17 files)

Direct `compiler/asm_templates/*` ISA-emitter regression tests preserved from the pre-ATen era; cover low-level ISA emission paths (loop mode, `h_store`, `batched_matmul`, two-input ops, flash-attn prefill/decode/qkt, `s_map_v`, `silu`, `gelu`, `rms`, `layer_norm`, `linear`/`linear_loop`, `dllm1`, `ffn_intermediate`, `v_shift_v`) that the ATen-level suite does not currently exercise. See `direct_emit/README.md`.

## Running tests

All tests are invoked through `run.sh` (handles nix + conda + PYTHONPATH):

```bash
bash run.sh test-rms-norm       # -> aten/rms_norm_test.py
bash run.sh test-layer-norm     # -> aten/layer_norm_test.py
bash run.sh test-flash-attention
bash run.sh test-conv2d-siglip
bash run.sh test-decoder-smollm2-135m
```

The full recipe list lives in `../../justfile`. `compile_module` (`torch.export` → IR → ISA) regression coverage lives in the `compiler/aten` submodule.

## Naming conventions

- `<op>_test.py` — single-driver test exercising operator `op` via the DSL + `OpRegistry`. With `compile_module` retired, every ATen test now follows this form.
- `<op>_vlen128_test.py` / `<op>_matched_test.py` — explicit dim variants when multiple are kept.
- Deprecated: `aten_compiler_<op>_*_test.py`, `<op>_aten_test.py`, and the transitional `<op>_dsl_test.py` form — all collapsed to `<op>_test.py` once the `compile_module` driver tests were retired (Stage 1 of the compile_module retirement; that coverage lives in the `compiler/aten` submodule).

## Relationship to the `compiler/` submodule

`testbench/plena_compiler.py` is **not** the PLENA production compiler. The production toolchain lives in the `../../compiler/` git submodule (`compiler/generator/runner.py`, `compiler/asm_templates/`, `compiler/assembler/`). The submodule emits `.asm` files directly from HuggingFace models.

Both compilers consume the **same ISA templates** from `compiler/asm_templates/` (imported in `plena_compiler.py`). Their roles:

| Tool | Input | Output |
|---|---|---|
| `compiler/generator/runner.py` | HF model path + hardware spec | `.asm` file |
| `testbench/plena_compiler.py` | Python DSL calls | ISA string emitted in-process, fed to Rust emulator |

Name them carefully in commits and issues — "compiler" usually means the submodule, "PlenaCompiler" means the testbench class.
