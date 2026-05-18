# ==================== Emulator ====================

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
test-ffn-multi-model:
    python3 transactional_emulator/testbench/models/multi_model_ffn_test.py

test-ffn-smolvlm2:
    python3 transactional_emulator/testbench/models/multi_model_ffn_test.py smolvlm2

test-ffn-smollm2-135m:
    python3 transactional_emulator/testbench/models/multi_model_ffn_test.py smollm2-135m

test-ffn-clm60m:
    python3 transactional_emulator/testbench/models/multi_model_ffn_test.py clm60m

test-decoder-multi-model:
    python3 transactional_emulator/testbench/models/multi_model_decoder_test.py

test-decoder-smollm2-135m:
    python3 transactional_emulator/testbench/models/multi_model_decoder_test.py smollm2-135m

test-decoder-llada-8b:
    python3 transactional_emulator/testbench/models/multi_model_decoder_test.py llada-8b

test-vision-encoder-smolvlm2:
    python3 transactional_emulator/testbench/conv/smolvlm2_vision_encoder_test.py

# Unit tests for sliced_layer_test_builder (no HF download required)
test-sliced-layer-builder:
    python3 transactional_emulator/testbench/test_sliced_layer_builder.py

# Deprecated alias kept for existing scripts.
test-model-builder:
    just test-sliced-layer-builder

# Unit tests for LUI+ADDI large immediate fix in ASM templates
test-large-immediate:
    cd PLENA_Compiler && PYTHONPATH=. python3 asm_templates/tests/test_large_immediate.py

# ASM profiler: section + cycle breakdown of last generated ASM
asm-profile asm_path="":
    python3 analytic_models/roofline/asm_profiler.py {{asm_path}}

test-flash-attention:
    python3 transactional_emulator/testbench/aten/flash_attention_gqa_test.py

test-bmm:
    python3 transactional_emulator/testbench/direct_emit/bmm_test.py

test-conv2d:
    python3 transactional_emulator/testbench/conv/conv2d_baseline_test.py

test-conv2d-tiled:
    python3 transactional_emulator/testbench/conv/conv2d_k4_tiled_test.py

test-conv2d-siglip:
    python3 transactional_emulator/testbench/conv/conv2d_k8_siglip_test.py

test-conv2d-siglip-real:
    python3 transactional_emulator/testbench/conv/conv2d_k14_siglip_ksplit_test.py

test-embedding-add:
    python3 transactional_emulator/testbench/aten/embedding_add_test.py

test-rope:
    python3 transactional_emulator/testbench/aten/rope_test.py

# Generate and profile multi-layer decoder ASM (smolvlm2: 30 layers, 1 step; llada: 32 layers x 64 denoising steps + LM head)
multilayer-decoder-profile model="smolvlm2":
    python3 transactional_emulator/testbench/models/multi_model_multilayer_decoder_profile.py --model {{model}}

asm-profile-llada layers="32" steps="64":
    python3 transactional_emulator/testbench/models/multi_model_multilayer_decoder_profile.py --model llada --layers {{layers}} --steps {{steps}}

# ATen-backed sliced emulator check: PlenaCompiler + ops.* -> emulator -> numerical check
test-sliced-aten-emulator model="AICrossSim/clm-60m" seq_len="64" num_layers="1":
    cd PLENA_Compiler && PYTHONPATH=".:tools:../tools:../transactional_emulator/testbench:..:" python3 -m compiler.aten.sliced_emulator_runner {{model}} --seq-len {{seq_len}} --num-layers {{num_layers}}

# Deprecated alias kept for existing scripts.
test-aten-e2e model="AICrossSim/clm-60m" seq_len="64" num_layers="1":
    just test-sliced-aten-emulator "{{model}}" "{{seq_len}}" "{{num_layers}}"

# Deprecated alias kept for existing scripts.
test-generator-aten model="AICrossSim/clm-60m" seq_len="64" num_layers="1":
    just test-sliced-aten-emulator "{{model}}" "{{seq_len}}" "{{num_layers}}"
