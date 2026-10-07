# ==================== Docker ====================

# Docker compose file location
docker_compose := "docker/docker-compose.yml"
mamba_rtl_root := env_var_or_default("PLENA_RTL_ROOT", justfile_directory() + "/../PLENA_RTL")
mamba_compiler_root := env_var_or_default("PLENA_COMPILER_ROOT", justfile_directory() + "/../PLENA_Compiler")

# ==================== Mamba-2 Development Checkpoint ====================

# Verify that the pinned shell supplies every tool used by the three repos.
mamba-env-check:
    nix develop .#mamba --command bash -c 'set -euo pipefail; python3.12 -c "import bitstring, cocotb, toml, torch, pytest, yaml; print(\"Python/Torch\", torch.__version__, \"pytest\", pytest.__version__, \"cocotb\", cocotb.__version__)"; rustc --version; cargo --version; verilator --version; iverilog -V 2>&1 | sed -n "1p"; yosys -V; sv2v --version; cmake --version | sed -n "1p"; just --version; /bin/ps -p 1 >/dev/null'

# Report repository layout, profile drift, and dependency hazards without mutation.
mamba-env-audit:
    nix develop .#mamba --command python3.12 {{mamba_rtl_root}}/tools/contract/audit_environment.py --compiler {{mamba_compiler_root}} --simulator {{justfile_directory()}}

# Check the generated cross-repository contract against all three source maps.
mamba-contract-check:
    nix develop .#mamba --command python3.12 {{mamba_rtl_root}}/tools/contract/sync_contract.py --compiler {{mamba_compiler_root}} --simulator {{justfile_directory()}} --check

# Run official-shape FP32/BF16 scan/state equivalence tests.
mamba-golden-check:
    nix develop .#mamba --command bash -c 'set -euo pipefail; cd "$1"; PYTHONPATH=".:${PYTHONPATH:-}" python3.12 -m pytest -q aten/tests/test_nemotron3_mamba2_reference.py' -- {{mamba_compiler_root}}

# Check ABI packing, assembler encoding, descriptor lowering, and persistent state allocation.
mamba-compiler-check:
    nix develop .#mamba --command bash -c 'set -euo pipefail; cd "$1"; PYTHONPATH=".:${PYTHONPATH:-}" python3.12 -m pytest -q assembler/tests/test_mamba_abi.py aten/tests/test_nemotron3_mamba2_extract.py aten/tests/test_nemotron3_mamba2_lowering.py' -- {{mamba_compiler_root}}

# Run the complete Rust unit/regression suite in the pinned environment.
mamba-simulator-check:
    nix develop .#mamba --command bash -c 'set -euo pipefail; export CARGO_BUILD_JOBS=4; cargo test --manifest-path transactional_emulator/Cargo.toml'

# Compile real Mamba command images, execute prefill and a separate-process step,
# and compare FP32/BF16 output plus persistent state against PyTorch.
mamba-simulator-e2e:
    nix develop .#mamba --command bash -c 'set -euo pipefail; export CARGO_BUILD_JOBS=4; cargo build --manifest-path transactional_emulator/Cargo.toml; python3.12 tools/mamba_cli_e2e.py --compiler-root "$1" --simulator-root "$2" --emulator "$2/transactional_emulator/target/debug/transactional_emulator"' -- {{mamba_compiler_root}} {{justfile_directory()}}

# Lint standalone and composed Mamba control/data-movement RTL, then run Cocotb tests.
mamba-rtl-validator-check:
    nix develop .#mamba --command bash -c 'set -euo pipefail; cd "$1"; for source in src/mamba/rtl/mamba_descriptor_fetch.sv src/mamba/rtl/mamba_checked_mul_u64_u32.sv src/mamba/rtl/mamba_checked_divisible_u32.sv src/mamba/rtl/mamba_hazard_tracker.sv src/mamba/rtl/mamba_memory_arbiter.sv src/mamba/rtl/mamba_raw_dma.sv src/mamba/rtl/mamba_state_reset.sv src/mamba/rtl/mamba_completion_writer.sv src/mamba/rtl/mamba_completion_guard.sv src/memory/HBM/rtl/hbm_address_register_file.sv; do verilator --lint-only --sv -Wno-fatal -I./src/definitions "$source"; done; verilator --lint-only --sv -Wno-fatal -I./src/definitions --top-module mamba_descriptor_validator src/mamba/rtl/mamba_checked_mul_u64_u32.sv src/mamba/rtl/mamba_checked_divisible_u32.sv src/mamba/rtl/mamba_descriptor_validator.sv; verilator --lint-only --sv -Wno-fatal -I./src/definitions --top-module mamba_span_size_calculator src/mamba/rtl/mamba_checked_mul_u64_u32.sv src/mamba/rtl/mamba_span_size_calculator.sv; verilator --lint-only --sv -Wno-fatal -I./src/definitions --top-module mamba_span_validator src/mamba/rtl/mamba_checked_mul_u64_u32.sv src/mamba/rtl/mamba_span_size_calculator.sv src/mamba/rtl/mamba_span_validator.sv; verilator --lint-only --sv -Wno-fatal -I./src/definitions --top-module mamba_command_controller src/mamba/rtl/mamba_checked_mul_u64_u32.sv src/mamba/rtl/mamba_checked_divisible_u32.sv src/mamba/rtl/mamba_span_size_calculator.sv src/mamba/rtl/mamba_span_validator.sv src/mamba/rtl/mamba_descriptor_validator.sv src/mamba/rtl/mamba_completion_guard.sv src/mamba/rtl/mamba_hazard_tracker.sv src/mamba/rtl/mamba_command_controller.sv; verilator --lint-only --sv -Wno-fatal -I./src/definitions --top-module mamba_descriptor_dma_path src/mamba/rtl/mamba_descriptor_fetch.sv src/mamba/rtl/mamba_raw_dma.sv src/mamba/rtl/mamba_descriptor_dma_path.sv; verilator --lint-only --sv -Wno-fatal -I./src/definitions --top-module mamba_state_reset_dma_path src/mamba/rtl/mamba_state_reset.sv src/mamba/rtl/mamba_raw_dma.sv src/mamba/rtl/mamba_state_reset_dma_path.sv; verilator --lint-only --sv -Wno-fatal -I./src/definitions --top-module mamba_completion_dma_path src/mamba/rtl/mamba_completion_writer.sv src/mamba/rtl/mamba_raw_dma.sv src/mamba/rtl/mamba_completion_dma_path.sv; verilator --lint-only --sv -Wno-fatal -I./src/definitions --top-module mamba_command_dma_frontend src/mamba/rtl/mamba_checked_mul_u64_u32.sv src/mamba/rtl/mamba_checked_divisible_u32.sv src/mamba/rtl/mamba_span_size_calculator.sv src/mamba/rtl/mamba_span_validator.sv src/mamba/rtl/mamba_descriptor_validator.sv src/mamba/rtl/mamba_completion_guard.sv src/mamba/rtl/mamba_hazard_tracker.sv src/mamba/rtl/mamba_command_controller.sv src/mamba/rtl/mamba_raw_dma.sv src/mamba/rtl/mamba_descriptor_fetch.sv src/mamba/rtl/mamba_descriptor_dma_path.sv src/mamba/rtl/mamba_state_reset.sv src/mamba/rtl/mamba_state_reset_dma_path.sv src/mamba/rtl/mamba_completion_writer.sv src/mamba/rtl/mamba_completion_dma_path.sv src/mamba/rtl/mamba_memory_arbiter.sv src/mamba/rtl/mamba_command_dma_frontend.sv; export PYTHONPATH="tools:${PYTHONPATH:-}"; python3.12 src/mamba/test/mamba_descriptor_fetch_tb.py; python3.12 src/mamba/test/mamba_descriptor_validator_tb.py; python3.12 src/mamba/test/mamba_checked_mul_u64_u32_tb.py; python3.12 src/mamba/test/mamba_checked_divisible_u32_tb.py; python3.12 src/mamba/test/mamba_span_size_calculator_tb.py; python3.12 src/mamba/test/mamba_span_validator_tb.py; python3.12 src/mamba/test/mamba_hazard_tracker_tb.py; python3.12 src/mamba/test/mamba_command_controller_tb.py; python3.12 src/mamba/test/mamba_memory_arbiter_tb.py; python3.12 src/mamba/test/mamba_raw_dma_tb.py; python3.12 src/mamba/test/mamba_state_reset_tb.py; python3.12 src/mamba/test/mamba_completion_guard_tb.py; python3.12 src/mamba/test/mamba_descriptor_dma_path_tb.py; python3.12 src/mamba/test/mamba_state_reset_dma_path_tb.py; python3.12 src/mamba/test/mamba_completion_dma_path_tb.py; python3.12 src/mamba/test/mamba_command_dma_frontend_tb.py; python3.12 src/memory/HBM/test/hbm_address_register_file_tb.py' -- {{mamba_rtl_root}}

# Fast full-top RTL syntax/elaboration gate; executable build remains a release gate.
mamba-rtl-lint-check:
    nix develop .#mamba --command bash -c 'set -euo pipefail; cd "$1"; just rtl-lint' -- {{mamba_rtl_root}}

mamba-check: mamba-env-check mamba-env-audit mamba-contract-check mamba-golden-check mamba-compiler-check mamba-simulator-check mamba-simulator-e2e mamba-rtl-validator-check mamba-rtl-lint-check

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
