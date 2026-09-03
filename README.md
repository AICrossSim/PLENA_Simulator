# PLENA Simulation System

<div align="center">
  <img src="doc/plena_logo.png" alt="PLENA Logo" width="300"/>
</div>

This repository contains the multi-level simulator system for **PLENA (Programmable Long-context Efficient Neural Accelerator)**.

## Overview

The PLENA Simulator provides three main components:

- **Transaction-level Simulator**: Models PLENA's architectural behavior at a high level, enabling rapid exploration of design choices, memory hierarchies, and long-context LLM inference workflows without the overhead of cycle-accurate RTL simulation.
- **Analytical Latency Model**: Provides fast estimation of PLENA's performance characteristics (TTFT, TPS) based on architectural parameters and instruction latencies for specified workloads.
- **Utilization Model**: Analyzes the utilization of the systolic array based on architectural parameters and instruction latencies, computing attainable vs theoretical FLOPS.

## Matrix SRAM L-Compute branch

This branch executes the prepared-coefficient Nemotron Mamba-2 and Kimi KDA
decode recurrence through physical Matrix-SRAM banks. State, prepared fields
and outputs use a uniform BF16 PLENA contract. Official GPU FP32 state remains
profiling and accuracy metadata; it is not used to widen the Matrix port. There
is no cache, private state SRAM, `X_STATE`, command queue, runtime scheduler or
new MAC array.

The Rust simulator stores real `banks x rows x bank_width` cells. It checks
aliasing, one-port bank service, row/column addressing, lane restoration,
segment broadcast, reductions, explicit output writeback and recurrent state
carried across four tokens. The Compiler emits the same view contract and
canonical 32-bit instructions.

```text
L_TILE_CFG   slot, shape_reg, map_reg
L_TILE_EXEC  dst, src, scale, primitive[, axis_mask]
```

One opcode (`0x3f`) serves both forms while preserving legacy `L_CFG` at
`funct1=0`. The primitives are generic scale-accumulate, dot-reduce and
outer-update; model names do not appear in the encoding or decoder. Viewed
`H_PREFETCH_V`/`H_STORE_V` words use bit 31 plus a two-bit view slot. Legacy DMA
words and their KV precision interpretation are unchanged.

At `MLEN=2048`, `BLEN=32`, 64 banks, a 1 MiB BF16 Matrix SRAM and 1560 HBM
bytes/cycle, the fresh formula-based B1 decode timeline is:

| Model | Original A | Arlo B | Fixed single-base C | Affine D | D/A | D/B |
|---|---:|---:|---:|---:|---:|---:|
| Nemotron 3 | 4,055,091 | 3,110,067 | 2,210,882 | 2,014,554 | 2.0129x | 1.5438x |
| Kimi K3 | 103,816,704 | 97,013,856 | 93,286,200 | 91,178,043 | 1.1386x | 1.0640x |

`A` and `B` are one-cycle-per-issued-instruction proxies, not transactional
Rust timings. `C` and `D` include explicit Matrix service, arithmetic and HBM
terms. Consequently `D/A` and `D/B` are mainly multi-row utilization plus issue
compression; they are not programmable-skew speedups.

KDA decay/beta preparation is not hidden: B1 includes 5,107,104 ordinary
elementwise operations and 1,702,368 exponentials across 69 layers, charged as
the same 4,485 Vector cycles in every variant. This implements the official
`decay = exp(lower_bound * sigmoid(rate * (gate + dt_bias)))` and
`beta = sigmoid(beta_logit)` preparation before `L_TILE`.

`C` is a constrained single-base executable descriptor, not the fair bank-only
baseline. The fair `D'` control uses PLENA's fixed diagonal wiring and ordinary
Compiler-selected per-tile base phases. It occupies the same physical cells as
`D`, reaches zero bank stalls, and gives `D/D' = 1.00x` for both official BF16
state packets. This branch therefore does **not** claim a programmable-skew
bank speedup. `C -> D` is descriptor/chunk/issue and KDA-spill improvement.
Ordinary Attention/MLA/MoE row and column service is unchanged at all base
phases.

Compiler-generated recurrence programs run through the assembler and Rust
decoder for four consecutive tokens at official recurrence geometry. The test
compares 524,288 Nemotron and 1,572,864 Kimi state values plus every head-group
output. Fixed and affine cases all pass; the largest relative-L2 error is
0.0071 under BF16. Every output group has a distinct HBM destination.

The separate long-sequence storage study reports BF16 output relative-L2 error
of 0.000312 for Nemotron at 32K tokens and 0.017061 for Kimi at 2K tokens versus
FP32 state. These are synthetic recurrence errors, not checkpoint-level
language-quality results.

All 23 Nemotron Mamba layers and all 69 Kimi KDA layers emit legal `L_TILE`
instructions in the official 52/93-layer order. Whole-model cycles still come
from an analytic timeline with official dimensions, GPU calibration and
symbolic weights; ordinary layers are schedule markers. Real checkpoints have
not run numerically from first to final layer in Rust. Coefficient generation
is an upstream Vector stage. Prefill is modeled but not accelerated by
`L_TILE`.

The BF16 bank word is 512 bits, matching the reference port. Static overlap
receives no credit: the one-MiB point is short by 45,312 bytes for a second
Nemotron state group and 28,736 bytes for Kimi. No RTL or synthesis means no
PPA, frequency, power, Token/J or silicon claim. The timing scoreboard still
uses conservative logical extents for Matrix views; physical `Cell::Pending`
state enforces correctness, but exact bank-word overlap timing is not claimed.

See [the full result report](docs/MATRIX_LCOMPUTE_E2E_RESULTS_ZH.md) and
`artifacts/matrix_lcompute_e2e_v5/`. Run:

```bash
nix develop --no-write-lock-file --command \
  just test-matrix-lcompute /absolute/path/to/PLENA_Compiler
```

![Figure 1: Diagram of the PLENA](doc/PLENA_Sys.png)

---

## PLENA Publication

If you use this simulator in your research, please cite the following paper:

**Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference**  
[arXiv:2509.09505](https://arxiv.org/abs/2509.09505)

```bibtex
@misc{wu2025combatingmemorywallsoptimization,
  title        = {Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference},
  author       = {Haoran Wu and Can Xiao and Jiayi Nie and Xuan Guo and Binglei Lou and Jeffrey T. H. Wong and Zhiwen Mo and Cheng Zhang and Przemyslaw Forys and Wayne Luk and Hongxiang Fan and Jianyi Cheng and Timothy M. Jones and Rika Antonova and Robert Mullins and Aaron Zhao},
  year         = {2025},
  eprint       = {2509.09505},
  archivePrefix= {arXiv},
  primaryClass = {cs.AR},
  url          = {https://arxiv.org/abs/2509.09505}
}
```

---

## Setup

There are two ways to get a working environment. **Option A (Docker)** is the
recommended path — you only need Docker installed, and it wraps the full toolchain
in a reproducible container. **Option B (Nix)** runs directly on your machine if you
prefer native development.

### Option A — Docker (recommended)

You only need Docker installed (no Nix or direnv on the host). All commands run from
the repository root. Your working tree is bind-mounted into the container at
`/workspace`, so edits on the host are picked up live and build artifacts persist
on the host.

**Prerequisites:**

- Docker Engine with the Compose plugin (`docker compose`)
- (Optional) NVIDIA Container Toolkit for CUDA support

**Build the image and open a shell:**

```bash
git submodule update --init --recursive   # once, on the host
just docker-dev
```

**Run a test directly (no interactive shell needed):**

```bash
just docker-test test-aten-linear            # run a just recipe in Docker
just docker-test test-aten-linear --mlen 128 # ...with args
```

The first emulator test compiles the Rust binary automatically (one-time, a few
minutes); it persists on the host and later runs reuse it.

**Common Docker commands** (see [`docker/README.md`](docker/README.md) for the full list):

| Command | Description |
|---------|-------------|
| `just docker-dev` | Build, start, and enter the dev container |
| `just docker-run <cmd>` | Run a command in the dev environment |
| `just docker-test <recipe> [args...]` | Run a `just` recipe in Docker |
| `just docker-down` | Stop containers |

**CUDA support:**

```bash
docker compose -f docker/docker-compose.yml --profile cuda up -d dev-cuda
docker compose -f docker/docker-compose.yml exec dev-cuda bash
```

> **Note:** The repository is bind-mounted from the host (owned by your host user)
> while the container runs as `root`. The image marks `/workspace` as a git
> `safe.directory` so Nix's flake evaluation doesn't fail with a dubious-ownership
> error. If you build a custom image, preserve that setting.

### Option B — Nix (native)

**Prerequisites:**

- `nix` package manager (with flakes enabled)
- `direnv` for environment management

```bash
# Install direnv hook in your shell
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
```

**Installation:**

```bash
# Allow direnv to load the environment
direnv allow

# Enter the development environment
nix develop

# Update git submodules
git submodule update --remote --merge
```

You are now in a shell with the full toolchain (Rust, Python 3.12, clang, cmake,
etc.) and can run any of the `just` commands below directly.

---

## Configuration

The simulator and emulator both use `plena_settings.toml` as the main configuration file for hardware parameters. This file contains:

- Hardware dimensions (MLEN, BLEN, VLEN, HLEN)
- Memory configuration (HBM, SRAM sizes)
- Instruction latencies
- Prefetch/writeback amounts

The configuration file supports two modes:
- `analytic`: Used by analytical models (latency and utilization)
- `transactional`: Used by the transaction-level emulator

Set the active mode in the `[MODE]` section of `plena_settings.toml`.

---

## Transaction-level Emulation

The transaction-level emulator executes machine code instructions sequentially, modeling PLENA's behavior at a high abstraction level. It includes:

- HBM/DRAM off-chip memory simulation
- Handwritten assembly templates for every operator in PLENA ISA for LLaMA
- Test scripts to verify correctness of assembly templates

The emulator reads hardware configuration from `plena_settings.toml` (using the `behavior` mode).

### Running Simulations

**Standard mode:**
```bash
just build-emulator [task]
# Example: just build-behave-sim linear
```

**Debug mode:**
```bash
just build-emulator-debug [task]
# Example: just build-behave-sim-debug linear
```

**Run pre-generated assembly:**
```bash
just run-generated-asm
```

**Quiet mode (latency and error metrics only):**
```bash
just run-generated-asm-quiet
```

---

## Analytical Models

### Latency Model

The latency model provides fast performance estimation for PLENA workloads. It computes:
- **TTFT (Time To First Token)**: Latency for the prefill phase
- **TPS (Tokens Per Second)**: Throughput for the decode phase

#### Available Commands

**List available models:**
```bash
just latency-list-models
```

**Run with default settings** (llama-3.1-8b, batch=4, input=2048, output=1024):
```bash
just latency llama-3.1-8b
```

**Run with custom batch size:**
```bash
just latency-batch llama-3.1-8b 8
```

**Run with full custom parameters:**
```bash
just latency-full llama-3.1-8b 4 2048 1024
# Format: just latency-full {model} {batch} {input_seq} {output_seq}
```

**Get JSON output:**
```bash
just latency-json llama-3.1-8b
```

## Project Structure

```
PLENA_Simulator/
├── transactional_emulator/    # Transaction-level simulator (Rust)
├── analytic_models/          # Analytical models (Python)
│   ├── latency/             # Latency estimation model
│   └── utilisation/         # Utilization analysis model
├── compiler/                # Compiler and model definitions
├── PLENA_Tools/             # Supporting tools and utilities (submodule)
├── doc/                     # Documentation and diagrams
├── plena_settings.toml      # Main configuration file
└── justfile                 # Command shortcuts
```
