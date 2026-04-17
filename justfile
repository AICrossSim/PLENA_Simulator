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

# Run LLaDA diffusion performance model (T denoising steps over full sequence, no AR decode)
perf-llada model="llada-8b" steps="64":
    python3 analytic_models/performance/llama_model.py --model {{model}} --llada --diffusion-steps {{steps}} \
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
# Extra args can be passed, e.g.: just mem llama-3-8b --no-flash-attn
mem model="llama-3-8b" *args:
    python3 analytic_models/memory/llm_memory_model.py --model {{model}} \
        --model-lib {{_mem_model_lib}} \
        --config {{_mem_config}} {{args}}

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
# ==================== ATen-style Operator Tests ====================

# Ensure plena.ops and tools/ are importable
export PYTHONPATH := justfile_directory() + ":" + justfile_directory() + "/tools" + ":" + justfile_directory() + "/transactional_emulator/testbench" + ":" + env_var_or_default("PYTHONPATH", "")

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
    python3 tools/quant/quant_operations/sqrt.py
    python3 tools/quant/quant_operations/reciprocal.py

test-softmax:
    python3 transactional_emulator/testbench/aten/fpvar_softmax_test.py

test-linear:
    python3 transactional_emulator/testbench/aten/linear_test.py

test-rms-norm:
    python3 transactional_emulator/testbench/aten/rms_norm_test.py

test-layer-norm:
    python3 transactional_emulator/testbench/aten/layer_norm_test.py

test-ffn:
    python3 transactional_emulator/testbench/aten/ffn_test.py

# Real-model FFN tests (requires HuggingFace model download on first run)
test-ffn-smolvlm2:
    python3 transactional_emulator/testbench/models/smolvlm2_256m_ffn_test.py

test-ffn-smollm2-135m:
    python3 transactional_emulator/testbench/models/smollm2_135m_ffn_test.py

test-ffn-clm60m:
    python3 transactional_emulator/testbench/models/clm60m_ffn_test.py

test-decoder-smollm2-135m:
    python3 transactional_emulator/testbench/models/smollm2_135m_decoder_test.py

test-decoder-llada-8b:
    python3 transactional_emulator/testbench/models/llada_8b_decoder_test.py

test-vision-encoder-smolvlm2:
    python3 transactional_emulator/testbench/conv/smolvlm2_vision_encoder_test.py

# Unit tests for model_layer_test_builder (no HF download required)
test-model-builder:
    python3 transactional_emulator/testbench/test_model_layer_builder.py

# Unit tests for LUI+ADDI large immediate fix in ASM templates
test-large-immediate:
    python3 compiler/asm_templates/tests/test_large_immediate.py

# ASM profiler: section + cycle breakdown of last generated ASM
asm-profile asm_path="":
    python3 analytic_models/roofline/asm_profiler.py {{asm_path}}

test-flash-attention:
    python3 transactional_emulator/testbench/aten/flash_attention_gqa_test.py

test-bmm:
    python3 transactional_emulator/testbench/direct_emit/bmm_test.py

test-conv2d:
    python3 transactional_emulator/testbench/conv/conv2d_test.py

test-conv2d-tiled:
    python3 transactional_emulator/testbench/conv/conv2d_tiled_im2col_test.py

test-conv2d-siglip:
    python3 transactional_emulator/testbench/conv/conv2d_siglip_ksize14_test.py

test-conv2d-siglip-real:
    python3 transactional_emulator/testbench/conv/conv2d_siglip_real_k14_test.py

test-embedding-add:
    python3 transactional_emulator/testbench/aten/embedding_add_test.py

test-rope:
    python3 transactional_emulator/testbench/aten/rope_test.py

multilayer-decoder-profile layers="30":
    python3 transactional_emulator/testbench/models/smolvlm2_multilayer_decoder_profile.py --layers {{layers}}

# Generate and profile LLaDA decoder ASM (N transformer layers × T denoising steps + LM head)
asm-profile-llada layers="32" steps="64":
    python3 transactional_emulator/testbench/models/llada_multilayer_decoder_profile.py --layers {{layers}} --steps {{steps}}
