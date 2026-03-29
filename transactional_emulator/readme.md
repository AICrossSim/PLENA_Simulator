# Transactional Emulator

This simulator was primarily developed by **Dr. Gary Guo**.

## Features

- **Configurable**: Reads runtime settings from the repository-level `plena_settings.toml` and currently uses the `BEHAVIOR` section
- **Cycle-Accurate Simulation**: Provides precise timing simulation at the cycle level
- **HBM Integration**: Enabled with Ramulator 2 for high-bandwidth memory modeling
- **Instruction-Based Execution**: Takes machine code as input and executes instructions sequentially. Each instruction triggers a function call that simulates hardware behavior
- **Online Service Mode**: Can run as a long-lived TCP service with newline-delimited JSON commands
- **Developer Tooling**: Includes C++, Python, and Flask-based clients for interactive control and inspection

## Running Simulations

### Debug Mode

To run a simulation in debug mode from the repository root:

```bash
just build-emulator-debug [task]
```

Where `[task]` is one of: `linear`, `rms`, or `attn`

### Online Service Mode

The emulator can also run as a long-lived TCP service:

```bash
cargo run --release -- --serve --bind 127.0.0.1:7878
```

Optional preload files can still be passed with the existing flags:

```bash
cargo run --release -- --serve --bind 127.0.0.1:7878 \
  --hbm ./testbench/build/hbm_for_behave_sim.bin \
  --fpsram ./testbench/build/fp_sram.bin \
  --intsram ./testbench/build/int_sram.bin \
  --vram ./testbench/build/vector_sram.bin
```

The service accepts newline-delimited JSON requests such as:

```json
{"cmd":"get_config"}
{"cmd":"execute_file","path":"/abs/path/generated_machine_code.mem"}
{"cmd":"get_state"}
{"cmd":"read_vram","addr":0}
```

### Service Startup Notes

The online service constructs the emulator state eagerly at startup. On memory-constrained machines,
the default `BEHAVIOR.CONFIG.HBM_SIZE` in `../plena_settings.toml` may be too large and the process
can exit with `Killed: 9` before the TCP listener comes up.

If that happens, temporarily reduce the HBM size before starting the service. For example:

```toml
[BEHAVIOR.CONFIG.HBM_SIZE]
value = 1073741824
```

`1073741824` is `1 GiB`. If that is still too large for your machine, try `268435456` (`256 MiB`).

### Native Linking Notes

The `ramulator` Rust crate links against a native `libramulator` library.

For the expected environment, enter the project shell first:

```bash
nix develop
```

If your local Nix installation has not enabled flakes yet, use:

```bash
nix --extra-experimental-features "nix-command flakes" develop
```

If you build from an IDE or from a shell that does not inherit the Nix environment, set the
native library path explicitly before running Cargo:

```bash
export RAMULATOR_LIB_DIR=/path/to/directory/containing/libramulator
export LIBRARY_PATH="$RAMULATOR_LIB_DIR:$LIBRARY_PATH"
export DYLD_LIBRARY_PATH="$RAMULATOR_LIB_DIR:$DYLD_LIBRARY_PATH"
```

On macOS the library may be `libramulator.dylib`; on Linux it is typically `libramulator.so`.

### C++ Demo Client

A minimal C++ socket client is provided at `demo/cpp_client_demo.cpp`.

Compile it with:

```bash
c++ -std=c++17 -O2 demo/cpp_client_demo.cpp -o demo/cpp_client_demo
```

Run it against the online emulator:

```bash
./demo/cpp_client_demo 127.0.0.1 7878 \
  /abs/path/generated_machine_code.mem \
  /abs/path/hbm_for_behave_sim.bin \
  /abs/path/fp_sram.bin \
  /abs/path/int_sram.bin
```

### Python Demo Client

A matching Python client is provided at `demo/python_client_demo.py`.

Run it against the online emulator with a pre-generated opcode file:

```bash
python3 demo/python_client_demo.py \
  --host 127.0.0.1 \
  --port 7878 \
  --opcode-file /abs/path/generated_machine_code.mem \
  --hbm /abs/path/hbm_for_behave_sim.bin \
  --fpsram /abs/path/fp_sram.bin \
  --intsram /abs/path/int_sram.bin
```

It can also send a small inline batch directly:

```bash
python3 demo/python_client_demo.py \
  --host 127.0.0.1 \
  --port 7878 \
  --opcode 0x12345678 \
  --opcode 0x9abcdef0
```

### Flask Web GUI

A browser-based control surface is provided at `demo/webgui.py`.

Install Flask first if your Python environment does not already have it:

```bash
python3 -m pip install flask
```

Start the online emulator first:

```bash
cargo run --release -- --serve --bind 127.0.0.1:7878
```

Then launch the Flask app:

```bash
python3 demo/webgui.py \
  --listen-host 127.0.0.1 \
  --listen-port 5000 \
  --emulator-host 127.0.0.1 \
  --emulator-port 7878
```

Open `http://127.0.0.1:5000` in your browser. The GUI supports:

- a tabbed `Connection & Control` panel for endpoint setup, preload actions, execution, and memory probing
- a large `State Snapshot` panel with adaptive register heatmaps and hover-to-reveal register values
- loading HBM, FP SRAM, INT SRAM, and VRAM files
- executing a machine-code file
- executing an inline opcode batch
- pasting a PLENA assembly snippet, compiling it in-process, and sending the generated opcodes to the online emulator
- viewing `Last Request`, `Last Response`, and `Configuration Snapshot` as tabs
- windowed `VRAM Heatmap`, `MRAM Heatmap`, and `HBM Heatmap` views for interactive memory inspection

### Inline ASM in the Web GUI

The `Compile & Execute ASM` action uses the checked-out `compiler` submodule and assembles the
snippet with:

- `compiler/assembler/assembly_to_binary.py`
- `compiler/doc/operation.svh`
- `compiler/doc/configuration.svh`

Initialize the submodule before using this feature:

```bash
git submodule update --init compiler
```

If your clone cannot fetch submodules over SSH, reconfigure that submodule to use HTTPS first:

```bash
git config submodule.compiler.url https://github.com/AICrossSim/PLENA_Compiler.git
git submodule update --init compiler
```

The current assembler accepts PLENA assembly syntax. Decimal immediates are the safest choice for
inline snippets because not every parser path currently treats `0x...` literals uniformly.

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


## Notes

- Currently, MLEN and VLEN are assumed to be equal in this simulator.




## Supported Experiments

- **Linear Projection Testing** (`linear`)
- **RMSNorm Testing** (`rms`)
- **Attention Testing** (`attn`)
- **FFN Testing** (`ffn`)
