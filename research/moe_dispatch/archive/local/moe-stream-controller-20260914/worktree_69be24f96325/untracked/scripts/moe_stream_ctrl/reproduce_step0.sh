#!/usr/bin/env bash
# Fresh UUID, read-only frozen inputs, no controller changes or PR writes.
set -euo pipefail

task_workspace=/scratch/shared/mcl123/plena
task_source="$task_workspace/review_20260914/simulator-moe-stream-controller"
task_reference="$task_workspace/outputs/moe_stream_ctrl_20260914/5cec1f6917944c7d8d96cb5ac7f79bc7/step0"
task_python="$task_workspace/venvs/plena-py311/bin/python"
task_uuid=$("$task_python" -c 'import uuid; print(uuid.uuid4().hex)')
task_output="$task_workspace/outputs/moe_stream_ctrl_20260914/$task_uuid"

"$task_python" - "$task_output" "$task_source" <<'PY'
import datetime, json, pathlib, subprocess, sys
output, source = map(pathlib.Path, sys.argv[1:])
output.mkdir(exist_ok=False)
(output / 'step0/repro').mkdir(parents=True)
record = dict(run_id=output.name, created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              worktree=str(source), output_root=str(output), status='step0_in_progress',
              base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip(),
              implementation_gate='Await explicit confirmation of proposed resource fields after step0.')
(output / 'run.json').write_text(json.dumps(record, indent=2) + '\n')
print('Fresh reproduction output:', output)
PY

cp "$task_reference/repro/build-env.sh" "$task_output/step0/repro/build-env.sh"
cp "$task_reference/RESOURCE_CONTRACT_PROPOSAL.md" "$task_output/step0/RESOURCE_CONTRACT_PROPOSAL.md"
source "$task_output/step0/repro/build-env.sh"
cd "$task_source/transactional_emulator"
cargo test --workspace --locked --offline > "$task_output/step0/repro/cargo-test-workspace.log" 2>&1
cargo clippy --workspace --all-targets --locked --offline -- -D warnings > "$task_output/step0/repro/clippy-workspace.log" 2>&1
cargo build --release --bin moe_dual_normal --locked --offline > "$task_output/step0/repro/release.log" 2>&1

"$task_python" "$task_source/scripts/moe_stream_ctrl/step0.py" \
    --output "$task_output/step0" \
    --binary "$CARGO_TARGET_DIR/release/moe_dual_normal" \
    --library "$task_reference/repro/libramulator.so" \
    --workers 4 > "$task_output/step0/repro/regression.log" 2>&1
"$task_python" "$task_source/scripts/moe_stream_ctrl/summarize_step0.py" --output "$task_output/step0"
printf 'Report: %s/step0/REPORT.md\n' "$task_output"
