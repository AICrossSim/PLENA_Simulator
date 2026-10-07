"""Guard: this repo's generated X_STATE contract must match the Compiler's spec.

``tools/state_contract.py --check`` run inside PLENA_Compiler only compares that
repo's generated Python. The Rust ``generated_contract.rs`` and the descriptor
golden that live here are compared only when ``--simulator-root`` is passed, and
until this guard existed nothing passed it from either side -- the Compiler CI
cannot, because it has no Simulator checkout.

That left every wire-format change relying on human discipline across two repos.
It has already mattered twice: ``conv_state_precision`` claimed descriptor byte
61, which the Compiler previously spent on ``reserved0``, and the ``group_stride``
skew kind had to be taught to both sides in lockstep. Both landed correctly, but
a divergence would not have surfaced until a descriptor silently decoded with the
wrong field offsets at runtime.

This repo owns the check because PLENA_Compiler is a submodule here, so both
trees are present in one checkout.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
COMPILER_ROOT = pathlib.Path(
    os.environ.get("PLENA_COMPILER_ROOT", REPO_ROOT / "PLENA_Compiler")
).resolve()
CHECKER = COMPILER_ROOT / "tools" / "state_contract.py"
LAYOUT_GOLDEN = COMPILER_ROOT / "spec" / "l_scatter_m_v1_golden.json"
LOCAL_LAYOUT_GOLDEN = (
    REPO_ROOT / "transactional_emulator" / "testdata" / "l_scatter_m_v1_golden.json"
)


def test_l_scatter_m_golden_matches_the_compiler_spec_copy() -> None:
    if not LAYOUT_GOLDEN.exists():
        pytest.skip(
            "pinned PLENA_Compiler predates the executable L_SCATTER_M golden; "
            "bump the submodule to arm this guard"
        )
    assert LOCAL_LAYOUT_GOLDEN.read_bytes() == LAYOUT_GOLDEN.read_bytes(), (
        "Compiler and Simulator L_SCATTER_M golden files differ; regenerate the "
        "Compiler golden and copy it byte-for-byte into transactional_emulator/testdata"
    )


def test_generated_state_contract_matches_the_compiler_spec() -> None:
    if not CHECKER.exists():
        pytest.skip(
            f"{CHECKER.relative_to(REPO_ROOT)} is absent, so the pinned "
            "PLENA_Compiler predates the X_STATE contract generator. Bump the "
            "submodule to a commit carrying tools/state_contract.py to arm this "
            "guard; it enforces nothing until then."
        )
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--check",
            "--simulator-root",
            str(REPO_ROOT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "the Compiler X_STATE spec and this repo's generated contract disagree.\n"
        "Regenerate with: python3 PLENA_Compiler/tools/state_contract.py "
        "--simulator-root . --sync-simulator\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_the_checker_rejects_a_tampered_generated_contract(tmp_path: pathlib.Path) -> None:
    """A passing check must be evidence, not a no-op on a missing file.

    Copies the repo into a scratch tree, corrupts one generated Rust constant,
    and requires the checker to fail. Without this, an argument-handling change
    that quietly stopped comparing the Rust output would leave the guard above
    green forever.
    """
    if not CHECKER.exists():
        pytest.skip("pinned PLENA_Compiler predates the X_STATE contract generator")
    generated = (
        REPO_ROOT
        / "transactional_emulator"
        / "src"
        / "state_engine"
        / "generated_contract.rs"
    )
    tampered_root = tmp_path / "PLENA_Simulator"
    target = tampered_root / generated.relative_to(REPO_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        generated.read_text(encoding="utf-8").replace(
            "DESCRIPTOR_SIZE: usize = 256", "DESCRIPTOR_SIZE: usize = 128", 1
        ),
        encoding="utf-8",
    )
    for extra in ("transactional_emulator/src/op.rs", "transactional_emulator/testdata/x_state_v2_golden.json"):
        source = REPO_ROOT / extra
        destination = tampered_root / extra
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())

    result = subprocess.run(
        [sys.executable, str(CHECKER), "--check", "--simulator-root", str(tampered_root)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, (
        "the checker accepted a generated_contract.rs with a wrong descriptor "
        "size, so a green run from the guard above would prove nothing"
    )
