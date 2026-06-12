set shell := ["bash", "-cu"]

build-emulator arg:
    # 1) Build env for the given target
    rm -rf transactional_emulator/testbench/build
    case "{{arg}}" in \
      activations|elementwise|linear|layernorm|rmsnorm|attention|rope|hbm_copy|single_stream_block|tvm_online_softmax_min) script_path="transactional_emulator/testbench/tile_tensor_kernel_programs/{{arg}}.py" ;; \
      *) script_path="transactional_emulator/testbench/{{arg}}_test.py" ;; \
    esac && \
    PYTHONPATH="$(pwd)/compiler/tilelang_runtime_compier:$(pwd)/transactional_emulator/testbench${PYTHONPATH:+:$PYTHONPATH}" python3 "$script_path"
    # 2) Compute absolute paths (so they still work after cd)
    asm_path="$(pwd)/transactional_emulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/transactional_emulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/transactional_emulator/testbench/build/fp_sram.bin" && \
    cd transactional_emulator && \
    cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --quiet
    python3 transactional_emulator/tools/view_mem.py

build-emulator-debug arg:
    # 1) Build env for the given target
    rm -rf transactional_emulator/testbench/build
    # TVM testbenches (tvm_<arg>_test.py) run under the default .venv
    # (Python 3.12): it has torch AND tilelang's bundled tvm. The nix
    # gcc libstdc++ must be on LD_LIBRARY_PATH for torch's C extensions
    # (DEFAULT_LD_LIBRARY_PATH in test_helper.py). The legacy .venv-tvm
    # (3.11, tvm-only, no torch) is NOT used here. Non-TVM targets keep
    # the plain python3 path.
    case "{{arg}}" in \
      activations|elementwise|linear|layernorm|rmsnorm|attention|rope|hbm_copy|single_stream_block|tvm_online_softmax_min) \
        PYTHONPATH="$(pwd)/compiler/tilelang_runtime_compier:$(pwd)/transactional_emulator/testbench${PYTHONPATH:+:$PYTHONPATH}" python3 "transactional_emulator/testbench/tile_tensor_kernel_programs/{{arg}}.py" ;; \
      flash_attention_min|flash_attention_gemm_only|flash_attention_offset|flash_decode_min|flash_decode_min_gemm_only|linear_min|linear_min_no_transpose|linear_min_offset|conv2d_min|gelu_min|gelu_offset|layernorm_min|rmsnorm_min|rope_min|silu_min|modulate_min|modulation_gen_min|residual_gate_min|copy_offset|single_stream_block_min|ssb_staged) \
        LD_LIBRARY_PATH="/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib" PYTHONPATH="$(pwd)/compiler:$(pwd)/transactional_emulator/testbench${PYTHONPATH:+:$PYTHONPATH}" "$(pwd)/.venv/bin/python" "transactional_emulator/testbench/tvm_{{arg}}_test.py" ;; \
      *) \
        PYTHONPATH="$(pwd)/compiler/tilelang_runtime_compier:$(pwd)/transactional_emulator/testbench${PYTHONPATH:+:$PYTHONPATH}" python3 "transactional_emulator/testbench/{{arg}}_test.py" ;; \
    esac
    # 1.5) GP/slot consistency trace -> build/trace_gp_report.txt
    # (static check on the v2-emitted ISA; terminal only gets the headline)
    python3 transactional_emulator/tools/trace_gp.py \
      transactional_emulator/testbench/build/generated_asm_code.asm \
      > transactional_emulator/testbench/build/trace_gp_report.txt 2>&1 \
      && head -1 transactional_emulator/testbench/build/trace_gp_report.txt \
      || echo "[trace_gp] skipped (no asm / error)"
    # # 2) Compute absolute paths (so they still work after cd)
    asm_path="$(pwd)/transactional_emulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/transactional_emulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/transactional_emulator/testbench/build/fp_sram.bin" && \
    int_sram_path="$(pwd)/transactional_emulator/testbench/build/int_sram.bin" && \
    cd transactional_emulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path" --quiet 2> >(tee testbench/build/emulator_stderr.log >&2)
    python3 transactional_emulator/tools/view_mem.py

# Re-run the emulator on the LAST build's artifacts — no rm -rf, no
# TVM compile, no data/HBM regeneration. Use this when only the Rust
# emulator (src/*.rs) changed: cargo rebuilds incrementally and the
# existing build/ (asm + HBM bin + fp/int sram) is reused as-is.
rerun-emulator-debug:
    asm_path="$(pwd)/transactional_emulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/transactional_emulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/transactional_emulator/testbench/build/fp_sram.bin" && \
    int_sram_path="$(pwd)/transactional_emulator/testbench/build/int_sram.bin" && \
    cd transactional_emulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path" --quiet 2> >(tee testbench/build/emulator_stderr.log >&2)
    python3 transactional_emulator/tools/view_mem.py

# Same as build-emulator-debug but WITHOUT --quiet, so the emulator
# prints every btmm / mul_scalar / intermediate tensor to the terminal.
build-emulator-verbose arg:
    # 1) Build env for the given target
    rm -rf transactional_emulator/testbench/build
    case "{{arg}}" in \
      activations|elementwise|linear|layernorm|rmsnorm|attention|rope|hbm_copy|single_stream_block|tvm_online_softmax_min) script_path="transactional_emulator/testbench/tile_tensor_kernel_programs/{{arg}}.py" ;; \
      *) script_path="transactional_emulator/testbench/{{arg}}_test.py" ;; \
    esac && \
    PYTHONPATH="$(pwd)/compiler/tilelang_runtime_compier:$(pwd)/transactional_emulator/testbench${PYTHONPATH:+:$PYTHONPATH}" python3 "$script_path"
    # 2) Compute absolute paths (so they still work after cd)
    asm_path="$(pwd)/transactional_emulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/transactional_emulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/transactional_emulator/testbench/build/fp_sram.bin" && \
    int_sram_path="$(pwd)/transactional_emulator/testbench/build/int_sram.bin" && \
    cd transactional_emulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path" 2> >(tee testbench/build/emulator_stderr.log >&2)
    python3 transactional_emulator/tools/view_mem.py

# ==================== TVM-based compiler (skeleton) ====================

# Run the minimal TVM->PLENA pseudo-ISA end-to-end test.
# Default target compiles the minimal_btmm kernel and writes pseudo-ISA
# to transactional_emulator/testbench/build/tvm_minimal_btmm.plena.s.
tvm-compile arg="btmm":
    # Uses the dedicated .venv-tvm (Python 3.11) where apache-tvm is installed.
    # The main project venv is 3.12 and has no apache-tvm wheel available.
    # LD_LIBRARY_PATH is cleared because the Nix-provided libstdc++/glibc on
    # PATH conflicts with TVM's manylinux wheel. The TVM compiler test only
    # needs tvm + numpy, so dropping nix lib paths is safe here.
    case "{{arg}}" in \
      btmm) script_path="transactional_emulator/testbench/tvm_btmm_test.py" ;; \
      *) script_path="transactional_emulator/testbench/tvm_{{arg}}_test.py" ;; \
    esac && \
    LD_LIBRARY_PATH="" \
    PYTHONPATH="$(pwd)/compiler:$(pwd)/transactional_emulator/testbench${PYTHONPATH:+:$PYTHONPATH}" \
        "$(pwd)/.venv-tvm/bin/python" "$script_path"

run-generated-asm:
    asm_path="$(pwd)/transactional_emulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/transactional_emulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/transactional_emulator/testbench/build/fp_sram.bin" && \
    int_sram_path="$(pwd)/transactional_emulator/testbench/build/int_sram.bin" && \
    cd transactional_emulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --intsram "$int_sram_path" 2> >(tee testbench/build/emulator_stderr.log >&2)
    python3 transactional_emulator/tools/view_mem.py

# Quiet mode: only output latency and error metrics
run-generated-asm-quiet:
    asm_path="$(pwd)/transactional_emulator/testbench/build/generated_machine_code.mem" && \
    data_path="$(pwd)/transactional_emulator/testbench/build/hbm_for_behave_sim.bin" && \
    fp_sram_path="$(pwd)/transactional_emulator/testbench/build/fp_sram.bin" && \
    cd transactional_emulator && \
    RUST_BACKTRACE=1 cargo run --release -- --opcode "$asm_path" --hbm "$data_path" --fpsram "$fp_sram_path" --quiet
    python3 transactional_emulator/tools/view_mem.py

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
