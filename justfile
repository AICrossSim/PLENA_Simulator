
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

# ==================== Latency Model ====================

# List available models for latency estimation
latency-list-models:
    python3 analytic_models/latency/latency_model.py --list-models

# Run latency model with default settings (llama-3.1-8b, batch=4, input=2048, output=1024)
latency model="llama-3.1-8b":
    python3 analytic_models/latency/latency_model.py --model {{model}}

# Run latency model with custom batch size
latency-batch model batch:
    python3 analytic_models/latency/latency_model.py --model {{model}} --batch-size {{batch}}

# Run latency model with full custom parameters
latency-full model batch input_seq output_seq:
    python3 analytic_models/latency/latency_model.py --model {{model}} --batch-size {{batch}} --input-seq {{input_seq}} --output-seq {{output_seq}}

# Run latency model with JSON output
latency-json model="llama-3.1-8b":
    python3 analytic_models/latency/latency_model.py --model {{model}} --json

# ==================== Utilization Model ====================

# List available models for utilization estimation
util-list-models:
    python3 analytic_models/utilisation/utilisation_model.py --list-models

# Run utilization model with default settings
util model="llama-3.1-8b":
    python3 analytic_models/utilisation/utilisation_model.py --model {{model}}

# Run utilization model with custom batch size
util-batch model batch:
    python3 analytic_models/utilisation/utilisation_model.py --model {{model}} --batch-size {{batch}}

# Run utilization model with full custom parameters
util-full model batch input_seq output_seq:
    python3 analytic_models/utilisation/utilisation_model.py --model {{model}} --batch-size {{batch}} --input-seq {{input_seq}} --output-seq {{output_seq}}

# Run utilization model with JSON output
util-json model="llama-3.1-8b":
    python3 analytic_models/utilisation/utilisation_model.py --model {{model}} --json

# Run utilization model without partitioned matrix optimization
util-no-partition model="llama-3.1-8b":
    python3 analytic_models/utilisation/utilisation_model.py --model {{model}} --no-partition