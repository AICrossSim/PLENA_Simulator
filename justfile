# ==================== Docker ====================

# Docker compose file location
docker_compose := "docker/docker-compose.yml"

# Build development Docker image
docker-build-dev:
    docker compose -f {{docker_compose}} build dev

# Build all Docker images
docker-build-all:
    docker compose -f {{docker_compose}} build

# Start development container
docker-dev:
    docker compose -f {{docker_compose}} up -d dev && docker compose -f {{docker_compose}} exec dev bash

# Run a command in the Docker dev environment
docker-run *args:
    docker compose -f {{docker_compose}} run --rm dev {{args}}

# Run a just recipe in Docker, e.g. `just docker-test test-aten-linear`
docker-test *args:
    docker compose -f {{docker_compose}} run --rm dev just {{args}}

# Stop all containers
docker-down:
    docker compose -f {{docker_compose}} down

# Clean Docker volumes (warning: removes caches)
docker-clean:
    docker compose -f {{docker_compose}} down -v
    docker volume rm plena-nix-store plena-cargo-cache plena-venv-cache 2>/dev/null || true

# Build runtime image with transactional emulator
docker-build-runtime:
    docker compose -f {{docker_compose}} build runtime

# ==================== Emulator ====================

build-emulator arg:
    # 1) Build env for the given target (writes to the shared transactional_emulator/build)
    rm -rf transactional_emulator/build
    python3 transactional_emulator/testbench/{{arg}}_test.py
    # 2) Compute absolute paths (so they still work after cd)
    build_dir="$(pwd)/transactional_emulator/build" && \
    asm_path="$build_dir/generated_machine_code.mem" && \
    data_path="$build_dir/hbm_for_behave_sim.bin" && \
    fp_sram_path="$build_dir/fp_sram.bin" && \
    int_sram_path="$build_dir/int_sram.bin" && \
    cd transactional_emulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path" --quiet
    python3 PLENA_Tools/verification/view_mem.py


build-emulator-debug arg:
    # 1) Build env for the given target (writes to the shared transactional_emulator/build)
    rm -rf transactional_emulator/build
    python3 transactional_emulator/testbench/{{arg}}_test.py
    # 2) Compute absolute paths (so they still work after cd)
    build_dir="$(pwd)/transactional_emulator/build" && \
    asm_path="$build_dir/generated_machine_code.mem" && \
    data_path="$build_dir/hbm_for_behave_sim.bin" && \
    fp_sram_path="$build_dir/fp_sram.bin" && \
    int_sram_path="$build_dir/int_sram.bin" && \
    cd transactional_emulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path"
    python3 PLENA_Tools/verification/view_mem.py

# ==================== Performance Model ====================

# Run performance model: just build-perf-model <model> [batch] [input_seq] [output_seq]
build-perf-model model batch="4" input_seq="2048" output_seq="1024":
    python3 analytic_models/performance/llama_model.py \
        --model {{model}} \
        --batch-size {{batch}} \
        --input-seq {{input_seq}} \
        --output-seq {{output_seq}} \
        --model-lib "$(pwd)/PLENA_Compiler/doc/Model_Lib" \
        --config "$(pwd)/plena_settings.toml" \
        --isa-lib "$(pwd)/analytic_models/performance/customISA_lib.json"

# ==================== ATen-style Operator Tests ====================

# Ensure plena.ops and PLENA_Tools/ are importable
export PYTHONPATH := justfile_directory() + ":" + justfile_directory() + "/PLENA_Compiler" + ":" + justfile_directory() + "/PLENA_Tools" + ":" + justfile_directory() + "/transactional_emulator/testbench" + ":" + env_var_or_default("PYTHONPATH", "")

alias ts := test-sw
alias th := test-hw

test-hw:
    python3 src/basic_components/fp_operation/test/fp_ieee_partition_tb.py
    python3 src/basic_components/fp_operation/test/fp_ieee_normalize_tb.py
    python3 src/basic_components/fp_operation/test/fp_cp_adder_tb.py
    python3 src/basic_components/fp_operation/test/fp_cp_mult_tb.py
    python3 src/basic_components/fp_operation/test/fp_fix_reciprocal_tb.py
    python3 src/basic_components/fp_operation/test/fp_fix_exp_tb.py
    python3 src/basic_components/fp_operation/test/fp_fix_adder_tb.py
    python3 src/basic_components/fp_operation/test/fp_fix_mult_tb.py

test-sw:
    python3 PLENA_Tools/plena_quant/quant_operations/sqrt.py
    python3 PLENA_Tools/plena_quant/quant_operations/reciprocal.py

test-aten-softmax *args:
    python3 transactional_emulator/testbench/aten/fpvar_softmax_test.py {{args}}

test-aten-linear *args:
    python3 transactional_emulator/testbench/aten/linear_test.py {{args}}

test-aten-rms-norm *args:
    python3 transactional_emulator/testbench/aten/rms_norm_test.py {{args}}

test-aten-layer-norm *args:
    python3 transactional_emulator/testbench/aten/layer_norm_test.py {{args}}

test-aten-ffn *args:
    python3 transactional_emulator/testbench/aten/ffn_test.py {{args}}

# Unified model compile/emulate (use model nickname from YAML configs)
# Examples:
#   just aten-compile smollm2 --config sliced_64x64x16_b1
#   just aten-emulate llada-8b --config native_256x256x64_b1
#   just aten-emulate smolvlm2 --case vision-layers --layers 5
aten-compile nickname *args:
    python3 transactional_emulator/testbench/run_model.py {{nickname}} --compile-only {{args}}

aten-emulate nickname *args:
    python3 transactional_emulator/testbench/run_model.py {{nickname}} {{args}}

# Unit tests for sliced_layer_test_builder (no HF download required)
test-sliced-layer-builder:
    python3 transactional_emulator/testbench/test_sliced_layer_builder.py

test-qwen3-config:
    python3 transactional_emulator/testbench/test_qwen3_config.py


# Unit tests for LUI+ADDI large immediate fix in ASM templates
test-large-immediate:
    cd PLENA_Compiler && PYTHONPATH=. python3 asm_templates/tests/test_large_immediate.py

# ASM profiler: section + cycle breakdown of last generated ASM
asm-profile asm_path="":
    python3 analytic_models/roofline/asm_profiler.py {{asm_path}}

test-aten-flash-attention *args:
    python3 transactional_emulator/testbench/aten/flash_attention_gqa_test.py {{args}}

test-aten-bmm:
    python3 transactional_emulator/testbench/direct_emit/bmm_test.py

test-aten-conv2d preset="all":
    @if [ "{{preset}}" = "all" ]; then \
        for p in baseline tiled siglip ksplit; do \
            echo "=== conv2d preset: $$p ===" && \
            python3 transactional_emulator/testbench/aten/vision/conv2d_test.py --preset $$p || exit 1; \
        done; \
    else \
        python3 transactional_emulator/testbench/aten/vision/conv2d_test.py --preset {{preset}}; \
    fi

test-aten-embedding-add *args:
    python3 transactional_emulator/testbench/aten/embedding_add_test.py {{args}}

test-aten-rope *args:
    python3 transactional_emulator/testbench/aten/rope_test.py {{args}}

# Generate and profile multi-layer decoder ASM (smolvlm2: 30 layers, 1 step; llada: 32 layers x 64 denoising steps + LM head)
multilayer-decoder-profile model="smolvlm2":
    python3 transactional_emulator/testbench/models/multi_model_multilayer_decoder_profile.py --model {{model}}


# ATen-backed sliced emulator check: PlenaCompiler + ops.* -> emulator -> numerical check
test-sliced-aten-emulator model="AICrossSim/clm-60m" seq_len="64" num_layers="1":
    cd PLENA_Compiler && PYTHONPATH=".:../PLENA_Tools:../transactional_emulator/testbench:..:" python3 -m compiler.aten.sliced_emulator_runner {{model}} --seq-len {{seq_len}} --num-layers {{num_layers}}
