from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

from transactional_emulator.testbench.window1_p1.p1_utils import (
    PLENA_ROOT,
    REPO_ROOT,
    TESTBENCH_ROOT,
    gini_from_counts,
    write_csv,
    write_json,
)


OUT_ROOT = PLENA_ROOT / "outputs" / "window1_p2"
PAPER_ARTIFACTS = PLENA_ROOT / "paper_artifacts"
WEIGHTS_ROOT = PLENA_ROOT / "weights"
COMPILER_ROOT = PLENA_ROOT / "repos" / "PLENA_Compiler"
TOOLS_ROOT = PLENA_ROOT / "repos" / "PLENA_Tools"

MODEL_CONFIGS = {
    "qwen3": {
        "name": "Qwen3-30B-A3B",
        "path": WEIGHTS_ROOT / "qwen3-30b-a3b",
        "hidden_size": 2048,
        "intermediate_size": 768,
        "num_experts": 128,
        "top_k": 8,
        "layers": 48,
        "policy_name": "qwen3_moe",
        "activation_policy": "standard_swiglu",
    },
}

INPUT_FILES = {
    "gpqa_diamond": PAPER_ARTIFACTS / "gpqa_diamond_model_inputs.jsonl.gz",
    "bfcl_v3": PAPER_ARTIFACTS / "bfcl_v3_model_inputs.jsonl.gz",
}


def ensure_paths() -> None:
    for path in (REPO_ROOT, COMPILER_ROOT, TOOLS_ROOT, TESTBENCH_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ["PYTHONPATH"] = ":".join(
        [
            str(REPO_ROOT),
            str(COMPILER_ROOT),
            str(TOOLS_ROOT),
            str(TESTBENCH_ROOT),
            os.environ.get("PYTHONPATH", ""),
        ]
    )


def open_text(path: Path, mode: str = "rt"):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with open_text(path, "rt") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "at") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def routing_stats(indices: list[list[int]], num_experts: int) -> dict[str, Any]:
    counts = [0 for _ in range(num_experts)]
    for row in indices:
        for expert_id in row:
            counts[int(expert_id)] += 1
    nonzero = [idx for idx, value in enumerate(counts) if value > 0]
    total = sum(counts)
    return {
        "expert_counts": counts,
        "active_experts": nonzero,
        "hot_experts": sorted(nonzero, key=lambda idx: (-counts[idx], idx)),
        "cold_experts": [idx for idx, value in enumerate(counts) if value == 0],
        "duplicate_factor": float(total) / float(len(nonzero)) if nonzero else 0.0,
        "gini": gini_from_counts(counts),
    }


def stable_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)


__all__ = [
    "INPUT_FILES",
    "MODEL_CONFIGS",
    "OUT_ROOT",
    "PAPER_ARTIFACTS",
    "append_jsonl",
    "ensure_paths",
    "iter_jsonl",
    "load_json",
    "routing_stats",
    "stable_id",
    "write_csv",
    "write_json",
]

