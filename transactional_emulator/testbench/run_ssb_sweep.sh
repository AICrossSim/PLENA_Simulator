#!/usr/bin/env bash
# Sweep the SingleStreamBlock chain step by step.
#
# For each CHAIN_UPTO step: build the truncated chain, run the
# transactional emulator, compare the staged VRAM output to golden,
# record BOTH match rates (Relative Error = primary, Allclose =
# secondary). Prints one summary table at the end.
#
# Run from the repo root (PLENA_Simulator/):
#     bash transactional_emulator/testbench/run_ssb_sweep.sh
#
# Optional: pass a subset of steps as args, e.g.
#     bash .../run_ssb_sweep.sh linear_q linear_k flash_attention
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TB="transactional_emulator/testbench"
BUILD="$TB/build"

# Chain order — must match STEP_ORDER in tvm_ssb_staged_test.py.
# flash_attention and gelu each write their OWN compact output tensor
# now (no shared wide concat), so both are independently verifiable;
# a dedicated concat step joins them before linear2.
ALL_STEPS=(
    layernorm modulate
    linear_q linear_k linear_v linear_mlp
    qknorm_q qknorm_k
    rope_q rope_k
    flash_attention gelu concat
    linear2 residual_gate
)
if [ "$#" -gt 0 ]; then
    STEPS=("$@")
else
    STEPS=("${ALL_STEPS[@]}")
fi

declare -A REL_RATE
declare -A ALL_RATE
declare -A STATUS

for step in "${STEPS[@]}"; do
    echo "============================================================"
    echo ">>> CHAIN_UPTO = $step"
    echo "============================================================"

    rm -rf "$BUILD"

    # 1) build the truncated chain (SSB_UPTO overrides CHAIN_UPTO).
    if ! SSB_UPTO="$step" \
         PYTHONPATH="$REPO_ROOT/compiler/tilelang_runtime_compier:$REPO_ROOT/$TB${PYTHONPATH:+:$PYTHONPATH}" \
         python3 "$TB/tvm_ssb_staged_test.py"; then
        echo "  !! build FAILED for $step"
        STATUS[$step]="BUILD-FAIL"
        REL_RATE[$step]="-"; ALL_RATE[$step]="-"
        continue
    fi

    # 2) run the emulator.
    asm="$REPO_ROOT/$BUILD/generated_machine_code.mem"
    hbm="$REPO_ROOT/$BUILD/hbm_for_behave_sim.bin"
    fps="$REPO_ROOT/$BUILD/fp_sram.bin"
    ints="$REPO_ROOT/$BUILD/int_sram.bin"
    if ! ( cd transactional_emulator && \
           RUST_BACKTRACE=1 cargo run --release -- \
             --opcode "$asm" --hbm "$hbm" --fpsram "$fps" \
             --intsram "$ints" --quiet ); then
        echo "  !! emulator FAILED for $step"
        STATUS[$step]="RUN-FAIL"
        REL_RATE[$step]="-"; ALL_RATE[$step]="-"
        continue
    fi

    # 3) compare. view_mem dumps thousands of memory rows; redirect to
    # a FILE (never a shell variable — that overflows ARG_MAX).
    viewlog="$BUILD/_sweep_view_mem.log"
    python3 transactional_emulator/tools/view_mem.py > "$viewlog" 2>&1
    grep -E 'Match Rate|Error Check|All Values Pass' "$viewlog" || true

    rel="$(grep -A2 'Relative Error Check' "$viewlog" | grep 'Match Rate' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
    acc="$(grep -A3 'Allclose Check'       "$viewlog" | grep 'Match Rate' | grep -oE '[0-9]+\.[0-9]+' | head -1)"
    REL_RATE[$step]="${rel:-?}"
    ALL_RATE[$step]="${acc:-?}"
    STATUS[$step]="ok"
done

# Summary table -> stdout AND a file next to this script.
SUMMARY="$(dirname "${BASH_SOURCE[0]}")/run_ssb_sweep_results.txt"
{
    echo "============================================================"
    echo " SingleStreamBlock staged sweep -- summary"
    echo " (Relative Error = primary metric; Allclose = secondary)"
    echo " generated: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================================"
    printf " %-18s %16s %16s   %s\n" "CHAIN_UPTO" "rel.err<=0.2" "allclose<=0.2" "status"
    printf " %-18s %16s %16s   %s\n" "------------------" "------------" "-------------" "------"
    for step in "${STEPS[@]}"; do
        printf " %-18s %15s%% %15s%%   %s\n" \
            "$step" "${REL_RATE[$step]:-?}" "${ALL_RATE[$step]:-?}" "${STATUS[$step]:-?}"
    done
    echo "============================================================"
} | tee "$SUMMARY"
echo
echo "summary written to: $SUMMARY"
