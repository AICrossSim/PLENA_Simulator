# Testbench

## Quick Start

```bash
just aten-compile <nickname> [--config <preset>]   # codegen only
just aten-emulate <nickname> [--config <preset>]   # codegen + Rust emulator
```

Examples:
```bash
just aten-emulate smollm2 --config sliced_64x64x16_b1
just aten-compile llada-8b --config native_256x256x64_b1
just aten-emulate smolvlm2 --case vision-layers --layers 5
```

Model nicknames and hardware presets are defined in `model_configs/*.yaml`.

## Entry Point

`run_model.py` is the unified runner. It loads a YAML model config by
nickname, reads the hardware preset's `mode` field (sliced/native), and
routes to the appropriate compile path.

## Directory Layout

```
testbench/
├── aten/              ATen compiler op tests (linear, attention, norm, ...)
│   ├── compare/       Codegen comparison harnesses + ISA analysis
│   └── vision/        Conv2d + VLM pipeline tests
├── direct_emit/       Raw asm_template tests (no ATen compiler)
├── misc/              One-off tests
├── model_configs/     YAML model configs + loader
├── models/            Model-level validation harnesses and profiling scripts
│   └── gpt_oss/       GPT-OSS attention/block/decoder-chain semantics checks
├── routed_moe/        Routed-MoE router/top-k/expert/gather-scatter bring-up
├── build_paths.py     Shared build directory constant
├── config_utils.py    Hardware config utilities
├── emulator_runner.py Shared Rust emulator runner + emulate_from_result()
├── run_model.py       Unified CLI entry point
├── sim_env_utils.py   HBM binary writers + memory setup
└── sliced_layer_test_builder.py  Sliced model test framework
```

`aten/` is reserved for reusable ATen operator checks.  Model-specific routed
MoE and decoder-block bring-up lives under `routed_moe/` and `models/` so that
reviewers can distinguish reusable operator coverage from model semantics
harnesses.
