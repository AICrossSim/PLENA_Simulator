
build-behave-sim arg:
    # 1) Build env for the given target
    rm -rf behavioral_simulator/testbench/build
    python3 behavioral_simulator/testbench/{{arg}}_test.py
    # 2) Compute absolute paths (so they still work after cd)
    asm_path="$(pwd)/behavioral_simulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/behavioral_simulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/behavioral_simulator/testbench/build/fp_sram.bin" && \
    cd behavioral_simulator && \
    cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path"

build-behave-sim-debug arg:
    # 1) Build env for the given target
    rm -rf behavioral_simulator/testbench/build
    python3 behavioral_simulator/testbench/{{arg}}_test.py
    # # 2) Compute absolute paths (so they still work after cd)
    asm_path="$(pwd)/behavioral_simulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/behavioral_simulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/behavioral_simulator/testbench/build/fp_sram.bin" && \
    int_sram_path="$(pwd)/behavioral_simulator/testbench/build/int_sram.bin" && \
    cd behavioral_simulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path"
    python3 behavioral_simulator/tools/view_mem.py

run-generated-asm:
    asm_path="$(pwd)/behavioral_simulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/behavioral_simulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/behavioral_simulator/testbench/build/fp_sram.bin" && \
    int_sram_path="$(pwd)/behavioral_simulator/testbench/build/int_sram.bin" && \
    cd behavioral_simulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path"
    python3 behavioral_simulator/tools/view_mem.py

# Quiet mode: only output latency and error metrics
run-generated-asm-quiet:
    asm_path="$(pwd)/behavioral_simulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/behavioral_simulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/behavioral_simulator/testbench/build/fp_sram.bin" && \
    cd behavioral_simulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --quiet
    python3 behavioral_simulator/tools/view_mem.py