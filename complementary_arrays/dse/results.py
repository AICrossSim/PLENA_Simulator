"""Result file layout under the output root.

search/<stage>/<cell>/<org>.json      search stages (top10, hetopt, prior, hetx<i>, retune<i>)
<name>/<cell>/<org>.json              selected chips (final by default)
perbatch/<cell>/B<B>/<org>.json       designs for one batch size
eval/<version>/<cell>/<set>_<org>.*   evaluation of a chip set on a window set
bootstrap/<version>/<cell>[...]       bootstrap replicates and intervals
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MODEL_ORDER = ("qwen", "deepseek")
BUDGETS = (98304, 196608)


def cell_name(model: str, fmt: str, P: int, hbm_tbps: int = 8) -> str:
    return f"{model}_{fmt}_H{hbm_tbps}_P{P}"


def search_path(out: Path, stage: str, cell: str, org: str) -> Path:
    return Path(out) / "search" / stage / cell / f"{org}.json"


def final_path(out: Path, cell: str, org: str, name: str = "final") -> Path:
    return Path(out) / name / cell / f"{org}.json"


def perbatch_path(out: Path, cell: str, batch: int, org: str) -> Path:
    return Path(out) / "perbatch" / cell / f"B{batch}" / f"{org}.json"


def eval_dir(out: Path, version: str, cell: str) -> Path:
    return Path(out) / "eval" / version / cell


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def write_json(path: Path, obj, indent: int | None = 1) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=indent, ensure_ascii=False) + "\n")
    tmp.replace(path)
