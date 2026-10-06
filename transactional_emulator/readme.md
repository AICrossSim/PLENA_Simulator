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

### Accumulate latency model

`TRANSACTIONAL.CONFIG.MATRIX_LATENCY_MODEL` in `plena_settings.toml` (command-line override `--matrix-latency-model`) selects the cycles charged for each matrix-matrix accumulate (`M_MM`, `M_TMM`, `M_BMM`, `M_BTMM`):

| Value | Cycles per accumulate | Notes |
|---|---|---|
| `"mlen"` (default) | `SYSTOLIC_PROCESSING_OVERHEAD + MLEN` | The historical charge; `SYSTOLIC_PROCESSING_OVERHEAD` comes from `TRANSACTIONAL.LATENCY`. |
| `"rtl_measured"` | `3 * BLEN + 11` | Issue-to-issue cost of back-to-back accumulates on the PLENA_RTL matrix machine (`mxint_systolic_mcu`, KLEN = BLEN), measured with Verilator 5.034 on SimTop at PLENA_RTL 783ee48: 23 cycles at BLEN=4 (MLEN 8, 16, 32, 64), 35 at BLEN=8 (MLEN 16, 32) and 59 at BLEN=16 (MLEN 32), identical for `M_MM`, `M_TMM`, `M_BMM` and `M_BTMM`. The RTL drains each accumulate before starting the next and splits the MLEN-deep reduction across MLEN / BLEN parallel mini-arrays, so the cost depends on BLEN only. |

The key is optional: a settings file without it charges the `mlen` model, so existing runs are unchanged. Matrix-vector ops and write-outs are not affected by the setting.


## Notes

- Currently, MLEN and VLEN are assumed to be equal in this simulator.




## Supported Experiments

- **Linear Projection Testing** (`linear`)
- **RMSNorm Testing** (`rms`)
- **Attention Testing** (`attn`)
- **FFN Testing** (`ffn`)
