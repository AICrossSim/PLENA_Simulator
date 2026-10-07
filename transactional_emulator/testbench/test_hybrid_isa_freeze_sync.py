"""Cross-repo guard for the hybrid opcode allocation and freeze status."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
COMPILER = Path(os.environ.get("PLENA_COMPILER_ROOT", ROOT / "PLENA_Compiler"))
FREEZE_PATH = COMPILER / "spec" / "hybrid_isa_freeze_v1.json"


def _freeze() -> dict:
    if not FREEZE_PATH.exists():
        pytest.skip("pinned Compiler predates hybrid_isa_freeze_v1.json")
    return json.loads(FREEZE_PATH.read_text())


def _decode_body() -> str:
    rust = (ROOT / "transactional_emulator" / "src" / "op.rs").read_text()
    match = re.search(
        r"pub\s+fn\s+decode\s*\([^)]*\)\s*->\s*Self\s*\{(?P<body>.*?)\n\s*\}\n\}",
        rust,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("body")


def test_simulator_implements_only_the_frozen_hybrid_opcodes() -> None:
    freeze = _freeze()
    body = _decode_body()
    for name in ("C_SET_TOPK_REG", "X_STATE", "L_SCATTER_M"):
        opcode = freeze["implemented_opcodes"][name]
        assert re.search(rf"0x{opcode:02x}\s*=>", body, flags=re.IGNORECASE)
        assert f"Self::{name}" in body

    for name in freeze["reserved_not_implemented"]:
        assert f"Self::{name}" not in body
    for opcode in [
        *freeze["reserved_not_implemented"].values(),
        *freeze["unallocated_opcodes"],
    ]:
        assert not re.search(rf"0x{opcode:02x}\s*=>", body, flags=re.IGNORECASE)


def test_freeze_remains_pre_rtl_and_route_reservation_only() -> None:
    freeze = _freeze()
    assert freeze["scope"]["compiler"] == "implemented"
    assert freeze["scope"]["transactional_simulator"] == "implemented"
    assert freeze["scope"]["rtl"] == "not_started"
    assert freeze["scope"]["route_extension"] == "opcode_space_reserved_only"
