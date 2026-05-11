# PLENA Simulation System

<div align="center">
  <img src="doc/plena_logo.png" alt="PLENA Logo" width="300"/>
</div>

This repository contains the multi-level simulator system for **PLENA (Programmable Long-context Efficient Neural Accelerator)**.

## Overview

The PLENA Simulator provides three main components:

- **Transaction-level Simulator**: Models PLENA's architectural behavior at a high level, enabling rapid exploration of design choices, memory hierarchies, and long-context LLM inference workflows without the overhead of cycle-accurate RTL simulation. It now also supports a long-lived online emulator service with C++, Python, and Flask-based interactive clients.
- **Analytical Latency Model**: Provides fast estimation of PLENA's performance characteristics (TTFT, TPS) based on architectural parameters and instruction latencies for specified workloads.
- **Utilization Model**: Analyzes the utilization of the systolic array based on architectural parameters and instruction latencies, computing attainable vs theoretical FLOPS.

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

### Prerequisites

Install the following prerequisites:

- `nix` package manager
- `direnv` for environment management

```bash
# Install direnv hook in your shell
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
source ~/.bashrc
```

### Installation

```bash
# Allow direnv to load the environment
direnv allow

# Enter the development environment
nix develop

# Initialize the compiler submodule used by the assembler and Web GUI ASM flow
git submodule update --init compiler
```

If your local Nix installation has not enabled flakes yet, use:

```bash
nix --extra-experimental-features "nix-command flakes" develop
```

If the `compiler` submodule cannot be fetched over SSH, switch it to HTTPS first:

```bash
git config submodule.compiler.url https://github.com/AICrossSim/PLENA_Compiler.git
git submodule update --init compiler
```

---

## Configuration

The simulator and emulator both use `plena_settings.toml` as the main configuration file for hardware parameters. This file contains:

- Hardware dimensions (MLEN, BLEN, VLEN, HLEN)
- Memory configuration (HBM, SRAM sizes)
- Instruction latencies
- Prefetch/writeback amounts

In practice:
- the analytical models read the analytic-oriented configuration entries
- the transactional emulator currently reads the `BEHAVIOR` section directly

If you are tuning the online or batch emulator, update `BEHAVIOR.CONFIG.*` in `plena_settings.toml`.

---

## Transaction-level Emulation

The transaction-level emulator executes machine code instructions sequentially, modeling PLENA's behavior at a high abstraction level. It includes:

- HBM/DRAM off-chip memory simulation
- Handwritten assembly templates for every operator in PLENA ISA for LLaMA
- Test scripts to verify correctness of assembly templates
- A long-lived TCP service mode for interactive execution and inspection
- Demo clients in C++ and Python
- A Flask Web GUI with register and memory heatmaps

The emulator reads hardware configuration from the `BEHAVIOR` section in `plena_settings.toml`.

### Running Simulations

**Standard mode:**
```bash
just build-emulator [task]
# Example: just build-emulator linear
```

**Debug mode:**
```bash
just build-emulator-debug [task]
# Example: just build-emulator-debug linear
```

To run a pre-generated machine-code file directly without the `just` wrappers:

```bash
cd transactional_emulator
cargo run --release -- \
  --opcode /abs/path/generated_machine_code.mem \
  --hbm /abs/path/hbm_for_behave_sim.bin \
  --fpsram /abs/path/fp_sram.bin \
  --intsram /abs/path/int_sram.bin \
  --quiet
```

### Interactive Online Emulator

The Rust emulator can also run as a persistent TCP service:

```bash
cd transactional_emulator
cargo run --release -- --serve --bind 127.0.0.1:7878
```

The online service accepts newline-delimited JSON commands and can be driven from:

- `transactional_emulator/demo/cpp_client_demo.cpp`
- `transactional_emulator/demo/python_client_demo.py`
- `transactional_emulator/demo/webgui.py`

The Flask Web GUI provides:

- tabbed endpoint, preload, execute, and memory-probe controls
- a `State Snapshot` panel with adaptive register heatmaps
- `VRAM`, `MRAM`, and `HBM` heatmap tabs
- inline opcode execution
- inline PLENA ASM execution via `Compile & Execute ASM`

Start the GUI with:

```bash
cd transactional_emulator
python3 demo/webgui.py \
  --listen-host 127.0.0.1 \
  --listen-port 5000 \
  --emulator-host 127.0.0.1 \
  --emulator-port 7878
```

Then open `http://127.0.0.1:5000`.

The inline ASM path uses the checked-out `compiler` submodule. Decimal immediates are currently the
safest choice for pasted ASM snippets.

For more details, see `transactional_emulator/readme.md`.

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

#### Configuration

The latency model reads hardware configuration from `plena_settings.toml` (using the `analytic` mode). Key parameters include:
- Hardware dimensions (MLEN, BLEN, VLEN)
- Instruction latencies
- Memory bandwidth and prefetch amounts

### Utilization Model

The utilization model analyzes systolic array utilization, computing attainable vs theoretical FLOPS for different operations. It provides utilization metrics for:
- Attention operations (projection + flash attention)
- FFN (Feed-Forward Network) operations
- Overall inference utilization (prefill + decode phases)

#### Available Commands

**List available models:**
```bash
just util-list-models
```

**Run with default settings:**
```bash
just util llama-3.1-8b
```

**Run with custom batch size:**
```bash
just util-batch llama-3.1-8b 8
```

**Run with full custom parameters:**
```bash
just util-full llama-3.1-8b 4 2048 1024
# Format: just util-full {model} {batch} {input_seq} {output_seq}
```

**Get JSON output:**
```bash
just util-json llama-3.1-8b
```

**Run without partitioned matrix optimization:**
```bash
just util-no-partition llama-3.1-8b
```

#### Configuration

The utilization model also reads hardware configuration from `plena_settings.toml` (using the `analytic` mode). It uses the same configuration as the latency model.

---

## Project Structure

```
PLENA_Simulator/
├── transactional_emulator/    # Transaction-level simulator (Rust)
├── analytic_models/          # Analytical models (Python)
│   ├── latency/             # Latency estimation model
│   └── utilisation/         # Utilization analysis model
├── compiler/                # Compiler and model definitions
├── tools/                   # Supporting tools and utilities
├── doc/                     # Documentation and diagrams
├── plena_settings.toml      # Main configuration file
└── justfile                 # Command shortcuts
```
