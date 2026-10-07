# Compiler phase-1 validation

Date: 2026-10-07. Compiler branch `research/moe-supply-first-v3`. This validates source integration and numerical/encoding contracts; it does not provide new MoE performance numbers or synthesized PPA.

| Check | Result | Evidence |
|---|---|---|
| Ordinary assembler/ATen/templates/generator suite, excluding separately run `slow` tests | 1,143 passed, 29 explicit historical-profile skips, 5 slow deselected; 11 subtests passed | `tests/compiler_complete_tests_supported.log` |
| Latest router resource-lifetime and >4-GiB loader regressions plus compiler helper tests | 41 passed | `tests/compiler_resource_lifetime_final.log` |
| Affected MLA, Kimi/Nemotron hybrid, grouped MoE and Matrix prefill integration | 45 passed | `tests/compiler_lifetime_affected_tests.log` |
| Current TVM frontend/backend test suite | 143 passed; 5 historical demo modules explicitly skipped | `tests/compiler_tilelang_complete_final.log` |
| Standalone TVM test entrypoints, including tests returning assertion counts | 13 entrypoints, every exit code 0 | `tests/compiler_tvm_main_results.json` |
| MoE research compiler suites | 33 passed, 3 subtests passed | `tests/compiler_moe_research_tests_final.log` |
| Four-mode real-checkpoint precision ablation | 1 passed after replacing unsupported sole-gap claim with independent measurements | `tests/compiler_slow_quantization_measured.log` |
| Slow full connected-program construction | 3 passed; 1 explicit retired X_STATE assembler profile skipped | `tests/compiler_slow_connected_lifetime_fixed.log` |

The ordinary full run predates the final two added resource-lifetime regressions; those are covered by the 41-test helper rerun and affected 45-test integration rerun. The loader adds a second GP only for nonzero HBM high words. Router local scratch is freed after its last emitted consumer; emitted arithmetic is unchanged.

The slow connected run constructs the complete 93-layer Kimi K3 program and both supported Nemotron full-program forms, validates symbolic HBM bindings, and checks encoded machine-code budgets. It completed in 489.16 seconds. The previously failing >4-GiB address scratch and score-router resource leaks are fixed; no capacity bound was relaxed.

## Active and archived profiles

The primary opcode map remains Matrix/L-TILE plus routed MoE, as recorded in `active_isa_profile.json`. Tests that demand incompatible X_MAMBA/X_STATE/L_SCATTER_M encodings through that map report explicit retired-profile skips; their original versions remain in `archive/retired_isa_tests`. Their standalone codecs, memory layouts and CPU references remain tested. Two historical no-state-directory deletion-policy guards are retired because the user specifically requested preserved research history and code; the active opcode conflict guard remains enabled. The X_STATE generated Python contract is still checked against its archived schema independently of active opcode ownership.

Five removed legacy graph-IR/demo modules are explicitly skipped. The removal is recorded by historical commit `0259583`; originals are in `archive/retired_frontend_tests`. Current mid-IR fold/view/fuse/lower and Matrix tests remain enabled. Tests use valid post-split VarRef identities, lane metadata and mathematically consistent GEMM fixtures.

The explicit `plena.*` intrinsic interface is preserved through a narrowly selected frontend into the same addressed typed-HLIR/ISA backend. It is not routed through removed graph IR. Mixed explicit-intrinsic and TileLang-store profiles are rejected, and unknown Evaluate calls now fail instead of silently disappearing.

## Isolated TVM environment

The shared production Python environment was not modified. The isolated interpreter is `/tmp/plena-round2-tvm-env/bin/python`, with TileLang `0.1.7.post3`, `apache-tvm-ffi==0.1.6`, CUDA runtime `12.9.79`, NVRTC `12.9.86`, `torch-c-dlpack-ext==0.1.5` and `z3-solver==5.1.0.0`. It imports the wheel's TVM `0.23.dev0`. A `.pth` supplies the existing shared Torch/numerical packages. Dependency installation logs remain under `outputs/round2_preflight_20261007`.

TVM commands use:

```sh
LD_LIBRARY_PATH=/scratch/shared/mcl123/plena/venvs/plena-tvm-runtime/prefix/lib:/tmp/plena-round2-tvm-env/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib:/tmp/plena-round2-tvm-env/lib/python3.11/site-packages/nvidia/cuda_runtime/lib
PYTHONPATH=/tmp/plena-round2-compiler-testpath:/scratch/shared/mcl123/plena/worktrees/moe-supply-first-v3-compiler
/tmp/plena-round2-tvm-env/bin/python -m pytest tilelang_tvm_compiler/tests -q
```

Ordinary tests use the existing `venvs/plena-py311/bin/python` and the same namespace root. Research tests run separately from `research/moe_dispatch` to avoid the research module named `compiler` shadowing the top-level compiler package. These are dependency/profile boundaries, not omission of failed current features.

## Precision-ablation correction

Same `AICrossSim/clm-60m` checkpoint, seed 42, sequence length 64 and five layers:

| Mode | Within 0.01 absolute tolerance | MSE against independent float32 reference |
|---|---:|---:|
| MXFP8 weights + BF16 intermediates | 34.4076% | 9.72442e-4 |
| Float32 weights + BF16 intermediates | 89.7746% | 4.14078e-5 |
| MXFP8 weights + float32 intermediates | 34.6395% | 9.31313e-4 |
| Float32 weights + float32 intermediates | 100.0000% | 2.48857e-8 |

The original test asserted >95% for the BF16-intermediate mode and claimed BF16 caused no gap; it failed at 89.8%. Numerical outputs were not changed to fit that assertion. Its original claim is archived, and the revised test checks finite measurements, the float32 baseline, separate error costs and the actual effect of removing weight quantization. This historical precision experiment is not part of the new BF16-only MoE performance claim.
