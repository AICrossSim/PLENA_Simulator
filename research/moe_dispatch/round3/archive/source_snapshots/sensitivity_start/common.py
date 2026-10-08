"""Read-only historical evidence, frozen inputs, and round-three receipts."""
from __future__ import annotations
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from ..round2.common import inputs, paired_ci
from .config import BATCHES, MODES, SEED, parameters

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
OLD = ROOT.parent / "round2"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f))


def gm(values):
    values = [float(v) for v in values]
    if not values or min(values) <= 0:
        raise ValueError("A geometric mean needs nonempty positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


gmean = gm


def encode_design(design):
    return asdict(design)


def decode_design(value):
    from .model import Design, Core
    if isinstance(value, str):
        value = json.loads(value) if value.startswith("{") else {
            "cores": [Core(*map(int, c.split("x"))) for c in value.split("+")]}
    value = dict(value)
    value["cores"] = tuple(Core(**c) if isinstance(c, dict) else
                           c if isinstance(c, Core) else Core(*c) for c in value["cores"])
    return Design(**value)


def frozen_designs(mode, *, common_ws=False):
    from dataclasses import replace
    value = json.loads((OLD / "dispatch_fix/frozen_designs.json").read_text())
    designs = {name: decode_design(d) for name, d in value["modes"][mode].items()}
    if common_ws:
        designs = {name: replace(d, flows=("WS",) * len(d.cores)) for name, d in designs.items()}
    return designs


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def check_round2_readonly():
    result = subprocess.run(["git", "diff", "HEAD", "--exit-code", "--", str(OLD)],
                            cwd=REPO, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError("Committed round-two evidence changed: " + result.stdout[:2000])
    return subprocess.check_output(["git", "rev-parse", "HEAD:research/moe_dispatch/round2"],
                                   cwd=REPO, text=True).strip()


def source_manifest():
    return {str(p.relative_to(REPO)): sha(p)
            for p in sorted([*ROOT.glob("*.py"),*ROOT.glob("*.c")])}


def metadata(extra=None):
    return {"execution_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO,
                                                        text=True).strip(),
            "source_sha256": source_manifest(), "round2_tree": check_round2_readonly(),
            "input_manifest_sha256": sha(OLD / "results/E0/frozen_inputs.json"),
            "input_hashes": json.loads((OLD / "results/E0/frozen_inputs.json").read_text()),
            "scope": "BF16 post-router phase-fluid analytical estimate; not native HBM/RTL/full-model timing",
            "clock_ns": 1.0, "main_parameters": asdict(parameters()), **(extra or {})}
