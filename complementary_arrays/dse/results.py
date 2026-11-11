"""Result file layout under the output root."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Row order of the result tables.
MODEL_ORDER = ("qwen", "deepseek", "qwen3next")
FORMAT_ORDER = ("w8g128", "bf16", "w4g32")
BUDGETS = (98304, 196608)


def cell_name(model: str, fmt: str, P: int, hbm_tbps: int = 8) -> str:
    return f"{model}_{fmt}_H{hbm_tbps}_P{P}"


def search_path(out: Path, stage: str, cell: str, org: str) -> Path:
    return Path(out) / "search" / stage / cell / f"{org}.json"


def final_path(out: Path, cell: str, org: str) -> Path:
    return Path(out) / "final" / cell / f"{org}.json"


def sealed_path(out: Path, cell: str) -> Path:
    return Path(out) / "sealed" / f"{cell}.json"


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, obj, indent: int | None = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=indent, ensure_ascii=False) + "\n")
    tmp.replace(path)
