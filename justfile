build-emulator arg:
    # 1) Build env for the given target
    rm -rf transactional_emulator/testbench/build
    python3 transactional_emulator/testbench/{{arg}}_test.py
    # # 2) Compute absolute paths (so they still work after cd)
    asm_path="$(pwd)/transactional_emulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/transactional_emulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/transactional_emulator/testbench/build/fp_sram.bin" && \
    int_sram_path="$(pwd)/transactional_emulator/testbench/build/int_sram.bin" && \
    cd transactional_emulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path" --quiet
    python3 transactional_emulator/tools/view_mem.py


build-emulator-debug arg:
    # 1) Build env for the given target
    rm -rf transactional_emulator/testbench/build
    python3 transactional_emulator/testbench/{{arg}}_test.py
    # # 2) Compute absolute paths (so they still work after cd)
    asm_path="$(pwd)/transactional_emulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/transactional_emulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/transactional_emulator/testbench/build/fp_sram.bin" && \
    int_sram_path="$(pwd)/transactional_emulator/testbench/build/int_sram.bin" && \
    cd transactional_emulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path"
    python3 transactional_emulator/tools/view_mem.py

# ==================== Performance Model ====================

# Common paths for performance model
_perf_model_lib := "$(pwd)/compiler/doc/Model_Lib"
_perf_config := "$(pwd)/plena_settings.toml"
_perf_isa_lib := "$(pwd)/analytic_models/performance/customISA_lib.json"

# List available models for performance estimation
perf-list-models:
    python3 analytic_models/performance/llama_model.py --list-models --model-lib {{_perf_model_lib}}

# Run performance model with default settings (llama-3.1-8b, batch=4, input=2048, output=1024)
perf model="llama-3.1-8b":
    python3 analytic_models/performance/llama_model.py --model {{model}} \
        --model-lib {{_perf_model_lib}} \
        --config {{_perf_config}} \
        --isa-lib {{_perf_isa_lib}}

# Run performance model with full custom parameters
perf-full model batch input_seq output_seq:
    python3 analytic_models/performance/llama_model.py --model {{model}} --batch-size {{batch}} --input-seq {{input_seq}} --output-seq {{output_seq}} \
        --model-lib {{_perf_model_lib}} \
        --config {{_perf_config}} \
        --isa-lib {{_perf_isa_lib}}

# Run performance model from task file (JSON input specifying model, batch, input_seq, output_seq)
perf-task task_file:
    python3 analytic_models/performance/llama_model.py --task-file {{task_file}} \
        --model-lib {{_perf_model_lib}} \
        --config {{_perf_config}} \
        --isa-lib {{_perf_isa_lib}}

# ==================== Memory Model ====================

# Common paths for memory model (reuses perf model paths)
_mem_model_lib := "$(pwd)/compiler/doc/Model_Lib"
_mem_config := "$(pwd)/plena_settings.toml"

# List available models for memory analysis
mem-list-models:
    python3 analytic_models/memory/llm_memory_model.py --list-models --model-lib {{_mem_model_lib}}

# Run memory model with default settings (llama-3.1-8b, batch=1, input=2048, output=128)
mem model="llama-3-8b":
    python3 analytic_models/memory/llm_memory_model.py --model {{model}} \
        --model-lib {{_mem_model_lib}} \
        --config {{_mem_config}}

# Run memory model with full custom parameters
mem-full model batch input_seq output_seq:
    python3 analytic_models/memory/llm_memory_model.py --model {{model}} \
        --batch-size {{batch}} --input-seq {{input_seq}} --output-seq {{output_seq}} \
        --model-lib {{_mem_model_lib}} \
        --config {{_mem_config}}

# Run memory model with JSON output
mem-json model="llama-3.1-8b":
    python3 analytic_models/memory/llm_memory_model.py --model {{model}} --json \
        --model-lib {{_mem_model_lib}} \
        --config {{_mem_config}}

# Run memory model quietly (suppress config output)
mem-quiet model="llama-3.1-8b":
    python3 analytic_models/memory/llm_memory_model.py --model {{model}} --quiet \
        --model-lib {{_mem_model_lib}} \
        --config {{_mem_config}}

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