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

# Update git submodules
git submodule update --remote --merge
```

---

## Configuration

The simulator uses `plena_settings.toml` as the main configuration file for hardware parameters. This file contains:

- Hardware dimensions (MLEN, BLEN, VLEN, HLEN)
- Memory configuration (HBM, SRAM sizes)
- Instruction latencies
- Prefetch/writeback amounts

The configuration file supports two modes:
- `analytic`: Used by analytical models (latency and utilization)
- `behavior`: Used by the transaction-level simulator

Set the active mode in the `[MODE]` section of `plena_settings.toml`.

---

## Transaction-level Simulation

The transaction-level simulator executes machine code instructions sequentially, modeling PLENA's behavior at a high abstraction level. It includes:

- HBM/DRAM off-chip memory simulation
- Handwritten assembly templates for every operator in PLENA ISA for LLaMA
- Test scripts to verify correctness of assembly templates

The simulator reads hardware configuration from `plena_settings.toml` (using the `behavior` mode).

### Running Simulations

**Standard mode:**
```bash
just build-behave-sim [task]
# Example: just build-behave-sim linear
```

**Debug mode:**
```bash
just build-behave-sim-debug [task]
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
├── behavioral_simulator/    # Transaction-level simulator (Rust)
├── analytic_models/          # Analytical models (Python)
│   ├── latency/             # Latency estimation model
│   └── utilisation/         # Utilization analysis model
├── compiler/                # Compiler and model definitions
├── tools/                   # Supporting tools and utilities
├── doc/                     # Documentation and diagrams
├── plena_settings.toml      # Main configuration file
└── justfile                 # Command shortcuts
```
