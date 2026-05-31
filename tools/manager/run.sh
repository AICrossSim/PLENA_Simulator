#!/usr/bin/env bash
# Run a manager script with the full environment (gcc libstdc++ + zlib + venv
# libtorch on LD_LIBRARY_PATH, and tools/compiler/testbench on PYTHONPATH).
#
# Usage:
#   tools/manager/run.sh                          # no arg -> _validate_double_block (default)
#   tools/manager/run.sh <script.py> [args...]
#   tools/manager/run.sh _validate_block          # single_stream_block
#   tools/manager/run.sh _validate_double_block   # MMDiT double_stream_block (default)
#   tools/manager/run.sh tools/manager/_validate_qkv_mlp.py
#   tools/manager/run.sh _validate_ln_mod        # .py + tools/manager/ prefix optional
#
# The noisy per-read check_mem prints are filtered out; everything else passes
# through.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

GCC=/nix/store/si4q3zks5mn5jhzzyri9hhd3cv789vlm-gcc-15.2.0-lib/lib
ZLIB=/nix/store/ri9paa3mri4kqakljak8ldvbcp7lpmif-zlib-1.3.1/lib
TORCH="$REPO/.venv/lib/python3.12/site-packages/torch/lib"

# Resolve the target script: accept a full path, a bare name, or a name without
# .py, defaulting into tools/manager/. With no argument, default to
# _validate_double_block (the full MMDiT double_stream_block end-to-end run).
arg="${1:-_validate_double_block}"; [ $# -gt 0 ] && shift || true
if   [ -f "$arg" ];                         then script="$arg"
elif [ -f "$REPO/$arg" ];                   then script="$REPO/$arg"
elif [ -f "$REPO/tools/manager/$arg" ];     then script="$REPO/tools/manager/$arg"
elif [ -f "$REPO/tools/manager/$arg.py" ];  then script="$REPO/tools/manager/$arg.py"
else echo "run.sh: cannot find script '$arg'" >&2; exit 2
fi

cd "$REPO"
# Default allocator for manager runs: gp_only_spill (GP-priority with
# pre-decided IntRAM overflow for low-frequency values; 3 dedicated
# scratch GPs at the top of the file handle reloads/stbacks). Override
# by exporting PLENA_ALLOC_MODE before invoking run.sh, e.g.:
#   PLENA_ALLOC_MODE=stable tools/manager/run.sh _validate_block
: "${PLENA_ALLOC_MODE:=gp_only_spill}"
export PLENA_ALLOC_MODE
LD_LIBRARY_PATH="$GCC:$ZLIB:$TORCH" \
PYTHONPATH="tools:compiler:transactional_emulator/testbench" \
  ./.venv/bin/python3 "$script" "$@" 2>&1 | grep -vE \
  "read settings:|  bin_file:|  start_byte|  num_elements:|  element_bytes:|  scale_width:|  block_size:|  scale_offset:|Scale calculation|element_offset:|block_index:|scale_start_offset:|num_blocks:"
