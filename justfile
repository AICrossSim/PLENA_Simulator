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

# Run the Mamba-2 (selective SSM) performance model.
# just build-perf-model-mamba2 <model> [batch] [input_seq] [output_seq]
build-perf-model-mamba2 model="mamba2-2.7b" batch="4" input_seq="2048" output_seq="1024":
    python3 analytic_models/performance/mamba2_model.py \
        --model {{model}} \
        --batch-size {{batch}} \
        --input-seq {{input_seq}} \
        --output-seq {{output_seq}} \
        --model-lib "$(pwd)/PLENA_Compiler/doc/Model_Lib" \
        --config "$(pwd)/plena_settings.toml" \
        --isa-lib "$(pwd)/analytic_models/performance/customISA_lib.json"

# Same, but against a model config outside Model_Lib.
# just build-perf-model-mamba2-path /path/to/mamba2-2.7b.json 1 2048 128
build-perf-model-mamba2-path path batch="4" input_seq="2048" output_seq="1024":
    python3 analytic_models/performance/mamba2_model.py \
        --model-path {{path}} \
        --batch-size {{batch}} \
        --input-seq {{input_seq}} \
        --output-seq {{output_seq}} \
        --config "$(pwd)/plena_settings.toml" \
        --isa-lib "$(pwd)/analytic_models/performance/customISA_lib.json"

# TTFT/TPS sweep for one model across several context lengths, JSON per point.
# just latency-sweep llama-3.1-8b 1 128 "512 2048 8192"
latency-sweep model batch="1" output_seq="128" contexts="512 2048 8192":
    #!/usr/bin/env bash
    set -euo pipefail
    script=analytic_models/performance/llama_model.py
    case "{{model}}" in
        mamba2*) script=analytic_models/performance/mamba2_model.py ;;
        gpt-oss*) script=analytic_models/performance/gpt_oss_model.py ;;
    esac
    for ctx in {{contexts}}; do
        python3 "$script" \
            --model {{model}} \
            --batch-size {{batch}} \
            --input-seq "$ctx" \
            --output-seq {{output_seq}} \
            --model-lib "$(pwd)/PLENA_Compiler/doc/Model_Lib" \
            --config "$(pwd)/plena_settings.toml" \
            --isa-lib "$(pwd)/analytic_models/performance/customISA_lib.json" \
            --json --quiet
    done

# Bandwidth-model + KV-store-bugfix regression tests for the analytic perf model.
test-perf-model:
    python3 analytic_models/test_perf_model_bandwidth.py

# Validate the checked-in GPU workload/baseline contracts. Raw Nsight reports
# are optional and are only needed by the importer-reproduction test.
test-gpu-evidence:
    python3 -m pytest \
        analytic_models/performance/test_b200_formal_campaign.py \
        analytic_models/performance/test_gpu_evidence.py \
        -v

gpu-evidence-report:
    python3 -m analytic_models.performance.gpu_evidence

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

# Compiler-generated Matrix writeback -> affine bank placement -> Vector lane restore.
test-lcompute-affine-projection:
    python3 transactional_emulator/testbench/aten/affine_projection_test.py

# Compiler projection -> M_MM_WO consumer-shaped Matrix view -> existing
# V_ADD_VV.MV consumer, checked numerically by the Rust emulator.
test-matrix-view-projection compiler_root="PLENA_Compiler":
    PLENA_SETTINGS_TOML="$PWD/plena_settings.toml" \
      PLENA_COMPILER_ROOT={{compiler_root}} \
      python3 transactional_emulator/testbench/aten/matrix_view_projection_test.py

# Matrix-SRAM-only L-Compute Python gate. This deliberately excludes the older
# Vector-SRAM L_CFG campaign so its speedups cannot be attributed to Matrix
# affine co-layout.
test-matrix-lcompute-python compiler_root="PLENA_Compiler":
    PLENA_COMPILER_ROOT={{compiler_root}} python3 -m pytest -q \
        analytic_models/performance/test_matrix_sram_layout.py \
        analytic_models/performance/test_matrix_state_residency.py \
        analytic_models/performance/test_matrix_lcompute_campaign.py \
        transactional_emulator/testbench/test_matrix_lcompute_recurrence_helpers.py

# Compiler encoding, dominance, packet extraction, physical writeback and
# official-shape workload guards used by the Matrix L-Compute campaign.
test-matrix-lcompute-compiler compiler_root="PLENA_Compiler":
    env -u LD_LIBRARY_PATH -u LIBRARY_PATH -u NIX_LDFLAGS -u PYTHONPATH \
      PLENA_SETTINGS_TOML="$PWD/plena_settings.toml" \
      PYTHONPATH={{compiler_root}} \
      uv run --directory {{compiler_root}} python -m pytest -q \
        assembler/tests/test_l_mview.py \
        aten/tests/test_affine_layout.py \
        aten/tests/test_hybrid_compile_report.py \
        aten/tests/test_hybrid_l_tile_schedule.py \
        aten/tests/test_hybrid_workloads.py \
        aten/tests/test_kda_precision_campaign.py \
        aten/tests/test_kda_official_layer.py \
        aten/tests/test_l_stream_cfg.py \
        aten/tests/test_layout_planner.py \
        aten/tests/test_lstream_lowering.py \
        aten/tests/test_lstream_packet_lowering.py \
        aten/tests/test_matrix_access_packets.py \
        aten/tests/test_matrix_packet_report.py \
        aten/tests/test_matrix_prefill_handoff.py \
        aten/tests/test_matrix_recurrence_lowering.py \
        aten/tests/test_mview_contract.py \
        aten/tests/test_projection_affine_writeback.py

# Physical banks, lane restoration, recurrence numerics and all existing Rust
# emulator regressions. Nix supplies ramulator and libtorch to the linker.
test-matrix-lcompute-rust:
    nix develop --no-write-lock-file --command bash -c \
        'cd transactional_emulator && cargo test --workspace --release -- --test-threads=1'

# Internal form used after the caller has already entered `nix develop`.
_test-matrix-lcompute-rust-in-dev-shell:
    # Ramulator2 owns process-global native state; parallel Rust test binaries
    # can otherwise race and intermittently SIGSEGV despite each test passing.
    cd transactional_emulator && cargo test --workspace --release -- --test-threads=1

# Official recurrence geometry, four consecutive tokens, Compiler assembly and
# machine words executed by the Rust emulator. Temporary HBM dumps are removed;
# the command fails on state/output mismatch or lane/head permutation.
test-matrix-lcompute-recurrence compiler_root="PLENA_Compiler":
    tmp_dir="$(mktemp -d)"; trap 'rm -rf "$tmp_dir"' EXIT; \
      PLENA_COMPILER_ROOT={{compiler_root}} python3 \
      "$PWD/transactional_emulator/testbench/aten/matrix_lcompute_recurrence_test.py" \
      --output-dir "$tmp_dir"

# Complete pre-RTL gate: Compiler contract + analytic campaign + physical Rust
# simulator + Compiler-generated recurrence binaries executed by Rust. Invoke this recipe
# through `nix develop` as shown in README.md; entering Nix once keeps Cargo's
# build fingerprint stable across both Rust checks.
test-matrix-lcompute compiler_root="PLENA_Compiler":
    just test-matrix-lcompute-python {{compiler_root}}
    just test-matrix-lcompute-compiler {{compiler_root}}
    just _test-matrix-lcompute-rust-in-dev-shell
    just test-matrix-view-projection {{compiler_root}}
    just test-matrix-lcompute-recurrence {{compiler_root}}

# Write A/B/C/D/E tables plus state capacity, precision and overlap contracts.
matrix-lcompute-campaign compiler_root="PLENA_Compiler":
    python3 -m analytic_models.performance.matrix_lcompute_campaign \
        --compiler-root {{compiler_root}} \
        --output-dir artifacts/matrix_lcompute_e2e_v5

# ISA/layout unit tests plus reproducibility checks for both checked campaigns.
test-hybrid-lcompute:
    python3 -m pytest -q \
        analytic_models/performance/test_lcompute_layout.py \
        analytic_models/performance/test_hybrid_lcompute_campaign.py \
        analytic_models/performance/test_hybrid_routing.py \
        analytic_models/performance/test_hybrid_connected_evidence.py \
        transactional_emulator/testbench/test_emulator_runner_metrics.py

# Slow executable evidence: Matrix affine writeback, S128 prefill-to-decode
# handoff, and request-private recurrent state at B=1/2/4/8/16. This uses only
# deterministic synthetic values and the Rust emulator; no GPU/checkpoint is
# required. The JSON keeps cycles, bank counters, numerical error and hashes.
hybrid-connected-evidence compiler_root="PLENA_Compiler":
    python3 -m analytic_models.performance.hybrid_connected_evidence \
        --compiler-root {{compiler_root}} \
        --json-out artifacts/hybrid_lcompute_connected_v1/evidence.json

# Official 52/93-layer timelines, A-J ablation, bandwidth/bank/FIFO DSE and
# exact lane recompilation at the PLENA paper's 2048-wide system point.
hybrid-paper2048-campaign compiler_root="PLENA_Compiler":
    python3 -m analytic_models.performance.hybrid_lcompute_campaign \
        --compiler-root {{compiler_root}} --hardware-profile paper2048 \
        --long --lane-sweep \
        --json-out artifacts/hybrid_lcompute_paper2048_v1/campaign.json \
        --csv-dir artifacts/hybrid_lcompute_paper2048_v1/tables

# B=1/2/4/8/16 full-model bounds plus replay of the pinned B200 Nemotron
# routing trace. Kimi remains explicitly bounded until its real trace arrives.
hybrid-paper2048-batch-campaign compiler_root="PLENA_Compiler":
    python3 -m analytic_models.performance.hybrid_lcompute_campaign \
        --compiler-root {{compiler_root}} --hardware-profile paper2048 \
        --batch-sweep --measured-routing \
        --json-out artifacts/hybrid_lcompute_paper2048_batch_v1/campaign.json \
        --csv-dir artifacts/hybrid_lcompute_paper2048_batch_v1/tables

# Faster focused gate when only the S128 handoff needs to be rechecked.
test-hybrid-prefill-handoff:
    python3 transactional_emulator/testbench/mamba2/mamba2_stage_test.py \
        --case prefill_s128_decode_handoff
    python3 transactional_emulator/testbench/kda/kda_stage_test.py \
        --case prefill_s128_decode_handoff --chunk 16

# Full transactional S128 prefill: all token outputs and final state are read
# back and compared. Large SRAM dumps live only in a temporary directory.
test-transactional-prefill-full compiler_root="PLENA_Compiler" output="artifacts/transactional_prefill_bf16/summary.json":
    python3 transactional_emulator/testbench/aten/transactional_prefill_evidence.py \
        --compiler-root {{compiler_root}} --output {{output}}

# Published 24-layer Mamba-2 checkpoint: host BF16 perimeter with every
# recurrent core compiled, assembled and executed by the Rust L_TILE path.
test-mamba2-real-checkpoint python_bin="python3" compiler_root="PLENA_Compiler" checkpoint="/scratch/shared/mcl123/plena/model_cache/huggingface/hub/models--AntonV--mamba2-130m-hf/snapshots/05e8773fc4ac1cd067e8a18a5c45372ce5178405" output_dir="artifacts/mamba2_130m_real_checkpoint_lcompute":
    PLENA_COMPILER_ROOT={{compiler_root}} PLENA_USE_NIX_BUILD=1 {{python_bin}} \
        transactional_emulator/testbench/aten/mamba2_real_checkpoint_lcompute_test.py \
        --checkpoint {{checkpoint}} --output-dir {{output_dir}}

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

# ==================== Mamba-2 / selective SSM ====================

# Per-stage numerical checks of the Mamba-2 lowering against a float32 torch
# golden. Fully synthetic -- no checkpoint, no HF libs. Cases:
#   dt      softplus + clamp        (exercises V_SOFTPLUS_V)
#   cumsum  a @ lower-tri ones      (the prefix-scan substitute; f32 accumulate)
#   decay   exp(min(cs_i-cs_j,0))   (exercises S_MAP_FP_V and V_SUB_VF rorder=1)
#   conv1d  causal depthwise k=4
#   decode_batch  four request-private recurrent states in one Rust program
# NOTE: the emulator's V_EXP_V is libtorch's exact exp, not the RTL's fixed-point
# model, so passing here bounds the lowering and not the silicon. See the module
# docstring of mamba2_stage_test.py.
test-mamba2-stage case="dt" *args:
    python3 transactional_emulator/testbench/mamba2/mamba2_stage_test.py --case {{case}} {{args}}

# Every Mamba-2 stage.
test-mamba2-all:
    #!/usr/bin/env bash
    set -euo pipefail
    for case in dt cumsum decay conv1d decode_batch; do
        echo "=== Mamba-2 stage: $case ==="
        just test-mamba2-stage "$case"
    done

# The chunked-SSD reference vs the plain recurrence. Pure torch, no emulator:
# this is what makes the chunked form usable as an intermediate golden at all.
test-mamba2-reference:
    python3 -m unittest compiler.aten.tests.test_mamba2_reference compiler.aten.tests.test_mamba_stage_contract -v

# ==================== KDA / gated delta attention ====================

test-kda-stage case="cumprod" *args:
    python3 transactional_emulator/testbench/kda/kda_stage_test.py --case {{case}} {{args}}

# Every KDA stage at the transactional machine width, followed by the cases
# that cross two 64-lane blocks at Kimi's 128x128 head geometry.
test-kda-all:
    #!/usr/bin/env bash
    set -euo pipefail
    build_root=transactional_emulator/testbench/kda/build
    cleanup() {
        if [[ -d "$build_root" ]]; then
            find "$build_root" -depth -delete
        fi
        find transactional_emulator -maxdepth 1 -type f \
            \( -name 'vram_dump.bin' -o -name 'mram_dump.bin' \
               -o -name 'fpsram_dump.bin' -o -name 'intsram_dump.bin' \) \
            -delete
    }
    trap cleanup EXIT
    for case in cumprod ut prefill_out prefill_state prefill_chain_out \
        prefill_chain_state state_transpose layer layer_chain recurrent_batch; do
        python3 transactional_emulator/testbench/kda/kda_stage_test.py --case "$case"
    done
    for case in prefill_out prefill_state prefill_chain_out \
        prefill_chain_state state_transpose layer layer_chain; do
        python3 transactional_emulator/testbench/kda/kda_stage_test.py \
            --case "$case" --key-dim 128 --value-dim 128
    done
    # The complete official decode order: eight Matrix projections, three
    # convolutions, recurrent KDA, gated RMSNorm, and output projection.
    python3 transactional_emulator/testbench/kda/kda_stage_test.py \
        --case official_layer --mlen 8 --blen 2 --num-heads 2 \
        --key-dim 8 --value-dim 8

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
