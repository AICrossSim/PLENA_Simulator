#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

nix develop "$repo_root" -c bash -lc "cd '$repo_root/transactional_emulator' && cargo test --workspace"

cd "$repo_root"
source /scratch/shared/mcl123/plena/venvs/plena-py311/bin/activate
export PYTHONPATH="$repo_root:$repo_root/PLENA_Compiler:$repo_root/PLENA_Tools:$repo_root/transactional_emulator/testbench:${PYTHONPATH:-}"
python transactional_emulator/testbench/timing_goldens/check_timing_goldens.py
