# direct_emit/ — pre-ATen asm_template regression tests

These 17 tests were preserved from the pre-ATen era (restored from `origin/main` after the ATen migration squash). They drive `compiler/asm_templates/*` ISA emitters **directly** — no `PlenaCompiler` DSL, no `OpRegistry` dispatch, no `compile_module` path.

They are kept alongside the ATen-level suite because they exercise lower-level hardware primitives (loop mode, `h_store`, `batched_matmul`, two-input ops, `linear_loop`, flash-attn prefill/decode/qkt splits, `s_map_v`, `dllm1`, etc.) that the ATen tests don't currently cover.

`bmm_test.py` and `v_shift_v_test.py` were moved in from `testbench/aten/` because they call `compiler/asm_templates/*` directly (same layer as the rest of `direct_emit/`) rather than going through the DSL.

Because they bypass the DSL, they may drift from evolving compiler/DSL conventions over time — treat stale tests as documentation of the raw ISA surface rather than a maintained correctness suite.
