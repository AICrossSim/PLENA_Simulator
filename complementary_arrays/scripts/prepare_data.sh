#!/usr/bin/env bash
# Assemble the data root from the captured routing traces and the sealed request group, and check the traces.
# usage: prepare_data.sh <data root> <trace directory> <sealed group directory (holds qwen35_bf16/ and deepseek_bf16/)>
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <data root> <trace directory> <sealed group directory>" >&2; exit 2; }
root=$1 traces=$2 sealed=$3
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$root/traces" "$root/sealed"
for model in qwen deepseek qwen3next; do
  for batch in 2 4 8 16 32 64 96 128; do
    for split in dev heldout; do
      cp -p "$traces/${model}_b${batch}_${split}.npz" "$root/traces/"
    done
  done
done
for group in qwen35_bf16 deepseek_bf16; do
  cp -rp "$sealed/$group" "$root/sealed/"
done
(cd "$root/traces" && sha256sum --quiet -c "$here/traces.sha256")
echo "data root ready: $root"
