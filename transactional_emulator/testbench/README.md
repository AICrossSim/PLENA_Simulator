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

## Connected recurrent sublayers

`models/recurrent_layer_test.py` executes a B1 Nemotron Mamba mixer or the first
Kimi KDA attention sublayer, including input norm, projections, convolution,
coefficient production and packing, recurrence, output norm/gate, and output
projection. Every intermediate is produced by machine code. Outer transformer
residual/MoE blocks are outside this boundary.

The checkpoints are decoded **offline to BF16 weights**. Source NVFP4/FP8
formats are retained as provenance; the reported transfers charge BF16 weights.
These diagnostics do not measure a compressed-weight hardware decoder. Mamba's
FP32 formula comparison and KDA's captured native output are different references.

With the repository's Python environment installed, prepare an output directory
outside the checkout, then build the memory reference and emulator:

```bash
export LAYER_OUT=/path/to/layer-results
mkdir -p "$LAYER_OUT"
nix develop -c python3 analytic_models/performance/ltile_dma.py --prepare "$LAYER_OUT/memory"
nix develop -c cargo build --locked --manifest-path transactional_emulator/Cargo.toml \
  --bin transactional_emulator --config 'profile.dev.package.transactional_emulator.opt-level=2'
```

The default Cargo executable is
`transactional_emulator/target/debug/transactional_emulator`; adjust that path
if `CARGO_TARGET_DIR` is set. A checkpoint-free projection check is:

```bash
python -m transactional_emulator.testbench.aten.matrix_projection_test \
  --k 289 --n 2049 --k-tile 256 --runtime /path/to/transactional_emulator \
  --memory-root "$LAYER_OUT/memory" --output "$LAYER_OUT/projection"
```

The connected driver accepts `--kind mamba|kda`, `--checkpoint`, `--runtime`,
`--memory-root`, and a fresh `--output`. KDA additionally needs `--capture`
containing the continuous-input NPZ and initial state. Mamba's subset includes
`input_hidden` and layer-0 checkpoint tensors. Large checkpoints and captures
are evidence inputs, not Git fixtures.

`--control row|fsm` holds the extended arithmetic and access path fixed; `row`
is **not** an unchanged-old-ISA baseline. `--state-input state_handoff.npz`
carries an earlier executed state. KDA `--token N` without a handoff requires a
captured checkpoint at N−1; Mamba follow-ups repeat the first embedding.
`--recheck-only` verifies that code, initial HBM, runtime and service contracts
match a saved execution before recomputing reference metrics. It accepts
losslessly compressed `.bin.gz` HBM images and does not claim a new execution.

Results include physical HBM counters, seven cycle-accounting fields, numerical
comparisons and SHA-256 provenance. Matrix mini-array timing, scalar/SFU timing,
and the conservative packing schedule remain explicit candidate contracts.
Functional agreement and cycle-accounting agreement do not certify RTL timing
or authorize formal full-model speedup results.
