# Transactional Emulator

This simulator was primarily developed by **Dr. Gary Guo**.

## What This Directory Contains

The transactional emulator is the execution-oriented part of the PLENA
simulator. It consumes instruction streams and models hardware-visible state
changes such as tile movement, residency, and operator execution.

For the current local TileTensor workflow, the most relevant area is:

- `testbench/`
  - Python-side program construction, test scaffolding, generated assembly, and
    compare helpers

Important files for current work:

- `testbench/tile_tensor_program.py`
  - logical/value/compute runtime used by local TileTensor tests
- `testbench/tile_tensor_group_head_linear_test.py`
- `testbench/tile_tensor_group_head_add_test.py`
- `testbench/tile_tensor_group_head_layer_norm_test.py`

## Features

- **Configurable**: Reads settings from `plena_settings.toml` file located in `src/definitions/plena_settings.toml`
- **Cycle-Accurate Simulation**: Provides precise timing simulation at the cycle level
- **HBM Integration**: Enabled with Ramulator 2 for high-bandwidth memory modeling
- **Instruction-Based Execution**: Takes machine code as input and executes instructions sequentially. Each instruction triggers a function call that simulates hardware behavior

## Running Simulations

For local TileTensor program work, the common flow is:

1. write one Python test in `testbench/`
2. build logical tensors and operators through `TileTensorProgram`
3. compile to generated assembly
4. stage inputs / golden output
5. run the behavioral simulator or downstream compare tooling

### Debug Mode

To run a simulation in debug mode from the `Coprocessor_for_Llama` directory:

```bash
just build-emulator-debug [task]
```

Where `[task]` is one of: `linear`, `rms`, or `attn`

## Building the Simulator

Please refer to the [Root README.md](../README.md) for detailed build instructions. Starting the Nix environment is required before building.

## Current TileTensor Runtime Model

The current TileTensor runtime in `testbench/tile_tensor_program.py` follows
this pipeline:

1. `mapt`
   - group logical tensor tiles for one compute packet
2. `mapv`
   - resolve wide tiles or scatter-backed narrow tiles into value-layer objects
3. compute
   - run copy, atomic ops, matmul, BTMM, or write-out logic
4. `mapv_back`
   - bind produced values back to logical destination tiles
5. `mapt_back`
   - return the final logical tensor/input handle

Important implementation rule:

- logical grouping belongs to the tensor layer
- value/scatter creation and residency decisions belong to the value layer

### BTMM/QKT Write-Out

The current BTMM/QKT path uses late destination allocation for `BTMM_WO`.

The effective sequence is:

1. `mapt`
   - produce one logical thread with `dst_tiles`
2. `BTMM`
   - compute using already resolved source values
3. `BTMM_WO`
   - allocate one contiguous VRAM window sized to the thread's output tile count
   - materialize one fresh output `ValueTile` per lane/tile inside that window
4. bind back
   - rebind each produced output value tile to its matching logical destination
     tile

This means destination logical tiles are known early, but their final writable
backing value tiles are chosen only at write-out time.

## FP-domain Objects

Besides tensor tiles, the local TileTensor runtime also exposes one FP-domain
for scalar and small structured working values.

The main objects are:

- `FPVar`
  - one scalar FP slot
- `FPFragment`
  - one collection of scalar FP slots
- `FPFragmentSlice`
  - one sliced view over an FPFragment

These are typically used for:

- constants such as reciprocal scale or epsilon
- reduction outputs such as rowwise mean/variance
- small FP working buffers used by normalization-style operators

Important difference from tensor tiles:

- FP-domain objects are mapped through `mapf`
- tensor tiles are mapped through `mapt`
- FP-domain objects do not participate in `ValueTile` / `Scatter` binding

The current maintained example that uses this path most clearly is:

- `testbench/tile_tensor_group_head_layer_norm_test.py`

## Group-Head Tests

The maintained executable examples in `testbench/` currently focus on
group-head operators over BSHD tensors:

- linear
- add
- layer norm

These tests are valuable because they exercise:

- packed narrow-head tile metadata
- scatter/scatter-group mapping
- alias-safe destination updates such as `A + B -> B`
- matmul routing beyond the simplest wide-tile path


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


## Notes

- Currently, MLEN and VLEN are assumed to be equal in this simulator.




## Supported Experiments

- **Linear Projection Testing** (`linear`)
- **RMSNorm Testing** (`rms`)
- **Attention Testing** (`attn`)
- **FFN Testing** (`ffn`)

For current local development, the group-head TileTensor tests in `testbench/`
are a more representative reference than many older historical experiments.
