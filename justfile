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

# Routed-MoE (GPT-OSS) substrate integration tests. Self-contained (no HF
# download, no HF libs): synthetic tensors exercise the V_TOPK router path, the
# V_MIN_VF/V_MAX_VF clamp path, and the gate-up / activation / expert / combine
# MoE stages. The model-backed tests (real_layer0, router_gemm, gather_scatter)
# are NOT here: they need the gpt-oss-20b checkpoint in the HF cache plus the
# huggingface_hub/safetensors libs, so they run only on a warmed developer box.
test-routed-moe-topk *args:
    python3 transactional_emulator/testbench/routed_moe/gpt_oss_topk_test.py {{args}}

test-routed-moe-clamp *args:
    python3 transactional_emulator/testbench/routed_moe/gpt_oss_moe_clamp_test.py {{args}}

test-routed-moe-activation *args:
    python3 transactional_emulator/testbench/routed_moe/gpt_oss_moe_activation_test.py {{args}}

test-routed-moe-gate-up *args:
    python3 transactional_emulator/testbench/routed_moe/gpt_oss_moe_gate_up_test.py {{args}}

test-routed-moe-expert *args:
    python3 transactional_emulator/testbench/routed_moe/gpt_oss_moe_expert_test.py {{args}}

test-routed-moe-combine *args:
    python3 transactional_emulator/testbench/routed_moe/gpt_oss_moe_combine_test.py {{args}}

# Shared-expert MoE (DeepSeek / Qwen2-MoE / Llama-4 / GLM). Like the tests above
# this is fully synthetic and needs no checkpoint. Bit-exact: MXFP8-representable
# inputs make weight quantization the identity, so atol=rtol=0.
test-shared-moe *args:
    python3 transactional_emulator/testbench/routed_moe/moe_shared_expert_test.py {{args}}

# Qwen2-MoE variant: adds the sigmoid shared-expert gate, the one shared-expert
# architecture that scales its shared branch.
test-shared-moe-gated *args:
    python3 transactional_emulator/testbench/routed_moe/moe_shared_expert_test.py \
        --arch qwen2 --build-dir transactional_emulator/testbench/routed_moe/build/moe_shared_expert_gated {{args}}

# DeepSeek n_shared_experts=2 fused into one wider MLP, plus the routed-accumulator
# combine. Pins that the shared branch is added unweighted.
test-shared-moe-deepseek-fused *args:
    python3 transactional_emulator/testbench/routed_moe/moe_shared_expert_test.py \
        --n-shared 2 --with-routed-accumulator \
        --build-dir transactional_emulator/testbench/routed_moe/build/moe_shared_expert_fused {{args}}

# V_TOPK at a given (num_experts, top_k). Policies outside the two hardwired
# rmask values route through C_SET_TOPK_REG; the test asserts which encoding was
# taken, so a silent fallback fails instead of passing on the untested path.
# Policies: gpt_oss qwen3_moe llama4_scout qwen2_moe deepseek_v2_lite deepseek_v3
test-router-policy policy="deepseek_v2_lite" *args:
    python3 transactional_emulator/testbench/routed_moe/moe_router_policy_test.py \
        --policy {{policy}} {{args}}

# Every routing shape, both encodings. deepseek_v3 is the widest at 256 experts
# spanning four MLEN-wide logit blocks.
test-router-policy-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for policy in gpt_oss qwen3_moe llama4_scout qwen2_moe deepseek_v2_lite deepseek_v3; do
        echo "=== V_TOPK policy: $policy ==="
        just test-router-policy "$policy"
    done

# Emulator timing gates: ramulator address sensitivity, prefetch/compute overlap,
# the matrix cycle formula, and stage-profile accounting. Exits non-zero when a
# required gate fails, so it is the one check that catches a timing-model
# regression. Needs a built emulator; no checkpoint.
#
# Invoked with `-m`, not by path. Run as a script under the PYTHONPATH this
# justfile exports, `moe_timing/replay/` lands on sys.path[0] and its own
# `utils.py` shadows the compiler's `utils` package, so `assembly_to_binary`
# fails to import. (Without that PYTHONPATH it fails earlier, on
# `transactional_emulator` itself -- either way the path form does not run.) The
# other moe_timing entry points that reach into the compiler use `-m` too.
test-timing-gates *args:
    python3 -m transactional_emulator.testbench.moe_timing.replay.timing_validation_gates {{args}}

# Full shared-expert + routing-policy suite. Synthetic, no checkpoint needed.
test-moe-shared-all:
    #!/usr/bin/env bash
    set -euo pipefail
    just test-shared-moe
    just test-shared-moe-gated
    just test-shared-moe-deepseek-fused
    just test-router-policy-all

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

# Matrix SRAM views and prepared BF16 recurrence through Compiler -> Rust.
test-matrix-lcompute *args:
    python3 transactional_emulator/testbench/aten/matrix_lcompute_test.py {{args}}
