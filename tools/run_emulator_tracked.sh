#!/bin/bash
# Tracked emulator run with progress estimation + crash detection
# Usage: bash tools/run_emulator_tracked.sh <build_dir> [hbm_size]
#
# Outputs progress to <build_dir>/emulator_progress.log
# Detects panics/crashes and writes to <build_dir>/emulator_error.log

set -euo pipefail

BUILD_DIR="${1:?Usage: $0 <build_dir> [hbm_size]}"
HBM_SIZE="${2:-}"

EMU_DIR="$(cd "$(dirname "$0")/../transactional_emulator" && pwd)"
BINARY="$EMU_DIR/target/release/transactional_emulator"
PROGRESS_LOG="$BUILD_DIR/emulator_progress.log"
ERROR_LOG="$BUILD_DIR/emulator_error.log"
PID_FILE="$BUILD_DIR/emulator.pid"

# Setup LD_LIBRARY_PATH for libtorch
LIBTORCH=$(ls -d "$EMU_DIR/target/release/build/torch-sys-"*/out/libtorch/libtorch/lib 2>/dev/null | head -1)
[ -n "$LIBTORCH" ] && export LD_LIBRARY_PATH="$LIBTORCH:${LD_LIBRARY_PATH:-}"

# Build command
CMD=("$BINARY"
    --opcode "$BUILD_DIR/generated_machine_code.mem"
    --hbm "$BUILD_DIR/hbm_for_behave_sim.bin"
    --fpsram "$BUILD_DIR/fp_sram.bin"
    --intsram "$BUILD_DIR/int_sram.bin"
    --quiet)

# HBM size
if [ -n "$HBM_SIZE" ]; then
    CMD+=(--hbm-size "$HBM_SIZE")
elif [ -f "$BUILD_DIR/hbm_size.txt" ]; then
    CMD+=(--hbm-size "$(cat "$BUILD_DIR/hbm_size.txt")")
else
    PRELOAD_SIZE=$(stat -c%s "$BUILD_DIR/hbm_for_behave_sim.bin" 2>/dev/null || echo 0)
    HBM_CALC=$(( (PRELOAD_SIZE * 2 + 63) / 64 * 64 ))
    [ "$HBM_CALC" -gt 0 ] && CMD+=(--hbm-size "$HBM_CALC")
fi

# Total instructions (rough: .mem file bytes / 4)
MEM_SIZE=$(stat -c%s "$BUILD_DIR/generated_machine_code.mem" 2>/dev/null || echo 0)
TOTAL_INSTR=$((MEM_SIZE / 4))

# ISA lines
ASM_LINES=$(wc -l < "$BUILD_DIR/generated_asm_code.asm" 2>/dev/null || echo 0)

echo "=== Emulator Tracked Run ===" | tee "$PROGRESS_LOG"
echo "Build dir:    $BUILD_DIR" | tee -a "$PROGRESS_LOG"
echo "ASM lines:    $ASM_LINES" | tee -a "$PROGRESS_LOG"
echo "Machine code: $TOTAL_INSTR instructions (~$((MEM_SIZE/1024)) KB)" | tee -a "$PROGRESS_LOG"
echo "Started:      $(date)" | tee -a "$PROGRESS_LOG"
echo "" > "$ERROR_LOG"

# Launch emulator in background
cd "$EMU_DIR"
"${CMD[@]}" 2>"$ERROR_LOG" &
EMU_PID=$!
echo "$EMU_PID" > "$PID_FILE"
echo "PID:          $EMU_PID" | tee -a "$PROGRESS_LOG"
echo "---" | tee -a "$PROGRESS_LOG"

# Monitor loop
START_TIME=$(date +%s)
while kill -0 "$EMU_PID" 2>/dev/null; do
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_TIME))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))

    # Get CPU time from /proc
    CPU_TIME=$(ps -p "$EMU_PID" -o cputime= 2>/dev/null | tr -d ' ' || echo "?")
    CPU_PCT=$(ps -p "$EMU_PID" -o %cpu= 2>/dev/null | tr -d ' ' || echo "?")

    # Check for errors
    if [ -s "$ERROR_LOG" ]; then
        ERROR_PEEK=$(head -1 "$ERROR_LOG")
        if echo "$ERROR_PEEK" | grep -qi "panic\|error\|abort"; then
            echo "[${MINS}m${SECS}s] CRASH DETECTED: $ERROR_PEEK" | tee -a "$PROGRESS_LOG"
            break
        fi
    fi

    echo "[${MINS}m${SECS}s] Running... CPU: ${CPU_PCT}%, CPU-time: ${CPU_TIME}" | tee -a "$PROGRESS_LOG"
    sleep 60
done

# Wait for exit
wait "$EMU_PID" 2>/dev/null
EXIT_CODE=$?
END_TIME=$(date +%s)
TOTAL_SECS=$((END_TIME - START_TIME))

echo "---" | tee -a "$PROGRESS_LOG"
echo "Finished:     $(date)" | tee -a "$PROGRESS_LOG"
echo "Wall time:    $((TOTAL_SECS/60))m$((TOTAL_SECS%60))s" | tee -a "$PROGRESS_LOG"
echo "Exit code:    $EXIT_CODE" | tee -a "$PROGRESS_LOG"

# Check for errors
if [ -s "$ERROR_LOG" ]; then
    echo "STDERR:" | tee -a "$PROGRESS_LOG"
    cat "$ERROR_LOG" | tee -a "$PROGRESS_LOG"
fi

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "STATUS: SUCCESS" | tee -a "$PROGRESS_LOG"
else
    echo "STATUS: FAILED (exit $EXIT_CODE)" | tee -a "$PROGRESS_LOG"
fi

rm -f "$PID_FILE"
exit "$EXIT_CODE"
