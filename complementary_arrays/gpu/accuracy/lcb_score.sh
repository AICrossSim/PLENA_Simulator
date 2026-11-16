#!/usr/bin/env bash
# Score LiveCodeBench generations in a rootless podman container: no network, read-only root, 8 worker processes.
# usage: lcb_score.sh <LiveCodeBench repo at commit 28fef95> <task dir> <run dir> [tag]
#   reads <run dir>/lcb[_<tag>].jsonl (generate.py), writes <run dir>/lcb[_<tag>]_scored.jsonl
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: $0 <lcb repo> <task dir> <run dir> [tag]" >&2; exit 2; }
repo=$(readlink -f "$1") data=$(readlink -f "$2") run=$(readlink -f "$3") name=lcb${4:+_$4}
image=localhost/lcb-score:1
here=$(dirname "$(readlink -f "$0")")
if ! podman image exists "$image"; then
  printf 'FROM docker.io/library/python:3.11-slim\nRUN pip install --no-cache-dir numpy==2.2.6 tqdm==4.67.1\n' |
    podman build -t "$image" -f - "$here"
fi
nice -n 10 podman run --rm --network none --cgroups disabled --read-only --tmpfs /tmp:size=2g \
  -v "$repo":/lcb:ro -v "$data":/data:ro -v "$run":/run_dir -v "$here/lcb_score.py":/w/lcb_score.py:ro \
  "$image" python /w/lcb_score.py /data/lcb.jsonl "/run_dir/$name.jsonl" "/run_dir/${name}_scored.jsonl" 8
