#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

nix develop "$repo_root" -c bash -lc "cd '$repo_root/transactional_emulator' && cargo test --workspace"

cd "$repo_root"
python_bin="${PLENA_PYTHON:-python3}"
if [[ -n "${PLENA_VENV:-}" ]]; then
  source "$PLENA_VENV/bin/activate"
  python_bin="${PLENA_PYTHON:-python}"
else
  default_venv="$repo_root/../../venvs/plena-py311"
  if [[ -z "${PLENA_PYTHON:-}" && -d "$default_venv" ]]; then
    source "$default_venv/bin/activate"
    python_bin="python"
  fi
fi
export PYTHONPATH="$repo_root:$repo_root/PLENA_Compiler:$repo_root/PLENA_Tools:$repo_root/transactional_emulator/testbench:${PYTHONPATH:-}"
"$python_bin" transactional_emulator/testbench/timing_goldens/check_timing_goldens.py
