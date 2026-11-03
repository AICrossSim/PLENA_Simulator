# Transactional Emulator

This simulator was primarily developed by **Dr. Gary Guo**.

## Features

- **Configurable**: Reads settings from `plena_settings.toml` file located in `src/definitions/plena_settings.toml`
- **Cycle-Accurate Simulation**: Provides precise timing simulation at the cycle level
- **HBM Integration**: Enabled with Ramulator 2 for high-bandwidth memory modeling
- **Instruction-Based Execution**: Takes machine code as input and executes instructions sequentially. Each instruction triggers a function call that simulates hardware behavior

## Running Simulations

### Debug Mode

To run a simulation in debug mode from the `Coprocessor_for_Llama` directory:

```bash
just build-emulator-debug [task]
```

Where `[task]` is one of: `linear`, `rms`, or `attn`

## Building the Simulator

Please refer to the [Root README.md](../README.md) for detailed build instructions. Starting the Nix environment is required before building.


## HBM Memory Model

The simulator integrates **Ramulator 2** for High-Bandwidth Memory (HBM) modeling.

### Generation and channel count

Two `TRANSACTIONAL.CONFIG` settings in `plena_settings.toml` select the Ramulator model; each has a command-line override that takes precedence:

| Setting | CLI override | Default | Meaning |
|---|---|---|---|
| `HBM_GEN` | `--hbm-gen` | `"hbm2"` | `hbm2`: `Ramulator::hbm2_preset` (HBM2, `HBM2_8Gb` organisation, 2.0 Gb/s per pin, 64-bit channels). `hbm3`: `Ramulator::hbm3_preset` (HBM3, `HBM3_8Gb_8hi` organisation at the 6.4 Gb/s JESD238 speed bin, 32-bit channels, BL8, on Ramulator's dual-command-bus `HBM34` controller). |
| `HBM_CHANNELS` | `--hbm-channels` | `8` | Number of independent channels, one Ramulator controller each. Consecutive transfers interleave across channels. |

Both keys are optional: a settings file without them behaves exactly as before (HBM2, 8 channels). The HBM3 model is the one ramulator 2.1 ships (`python/ramulator/dram/hbm3.py`), reproduced in `lib/ramulator/src/config/hbm3.rs` so no Python is needed at run time. Note that the per-channel peak bandwidth differs between the presets (16 GB/s for HBM2, 25.6 GB/s for HBM3), so compare generations at equal channel counts or scale `HBM_CHANNELS` to the stack being modeled.

### MX Data Type Address Patterns

- **Element Address**:  
  ```
  element_addr[Onchip] + hbm_offset
  ```

- **Scale Address**:  
  ```
  Scale_offset + (element_addr[Onchip] >> element_2_scale_ratio)
  ```



## Matrix Operations

### MM_WO (Matrix Multiply - Write Out)

Writes a (BLEN, BLEN) accumulator matrix (`m_accum`) to the Vector SRAM. This operation loads a (BLEN, VLEN) matrix from HBM and uses a mask to write to the Vector SRAM.


## Notes

- Currently, MLEN and VLEN are assumed to be equal in this simulator.




## Supported Experiments

- **Linear Projection Testing** (`linear`)
- **RMSNorm Testing** (`rms`)
- **Attention Testing** (`attn`)
- **FFN Testing** (`ffn`)
