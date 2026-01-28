alias ts := test-sw
alias th := test-hw

test-hw:
    python3 src/basic_components/fp_operation/test/fp_ieee_partition_tb.py
    python3 src/basic_components/fp_operation/test/fp_ieee_normalize_tb.py
    # python3 src/basic_components/fp_operation/test/fp_ieee_casting_tb.py
    python3 src/basic_components/fp_operation/test/fp_cp_adder_tb.py
    python3 src/basic_components/fp_operation/test/fp_cp_mult_tb.py
    # python3 src/basic_components/fp_operation/test/fp_cp_asym_mult_tb.py

    # python3 src/basic_components/fp_operation/test/fp_reciprocal_tb.py
    # python3 src/basic_components/fp_operation/test/fp_exp_tb.py
    # python3 src/basic_components/fp_operation/test/fp_cp_reciprocal_tb.py
    # python3 src/basic_components/fp_operation/test/fp_cp_exp_tb.py

    python3 src/basic_components/fp_operation/test/fp_fix_reciprocal_tb.py
    python3 src/basic_components/fp_operation/test/fp_fix_exp_tb.py
    python3 src/basic_components/fp_operation/test/fp_fix_adder_tb.py
    python3 src/basic_components/fp_operation/test/fp_fix_mult_tb.py

test-sw:
    python3 tools/quant/quant_operations/sqrt.py
    python3 tools/quant/quant_operations/reciprocal.py

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
    python3 behavioral_simulator/testbench/view_mem.py

run-generated-asm:
    asm_path="$(pwd)/behavioral_simulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/behavioral_simulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/behavioral_simulator/testbench/build/fp_sram.bin" && \
    int_sram_path="$(pwd)/behavioral_simulator/testbench/build/int_sram.bin" && \
    cd behavioral_simulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path"
    python3 behavioral_simulator/testbench/view_mem.py

# Quiet mode: only output latency and error metrics
run-generated-asm-quiet:
    asm_path="$(pwd)/behavioral_simulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/behavioral_simulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/behavioral_simulator/testbench/build/fp_sram.bin" && \
    cd behavioral_simulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --quiet
    python3 behavioral_simulator/testbench/view_mem.py


build-rtl-sim arg:
    rm -rf test/Instr_Level_Benchmark/build/{{arg}}
    python3 src/system/sys_utils/build_env.py --asm {{arg}}


reformat:
    black *.py
    black src/chop
    black src/mase_components
    black src/mase_cocotb
    black test
    # find src/mase_components -name '*.sv' -exec verible-verilog-format --inplace {} +;