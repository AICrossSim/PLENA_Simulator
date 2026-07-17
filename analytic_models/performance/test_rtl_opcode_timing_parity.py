"""Cross-language parity for CostEmitter and transactional opcode timing."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
import toml

from analytic_models.performance.compiler_cost_model import TransactionalCycleModel
from analytic_models.performance.rtl_opcode_timing import (
    ComputePrecisionConfig,
    RtlOpcodeTimingCalibration,
    TimingHardware,
)


ROOT = Path(__file__).resolve().parents[2]
EMULATOR = ROOT / "transactional_emulator"
SETTINGS_TEMPLATE = ROOT / "plena_settings.toml"
CALIBRATION = RtlOpcodeTimingCalibration.load()
RUST_BINARY = EMULATOR / "target/debug/rtl_opcode_timing_dump"


def _plain_fp(exponent: int, mantissa: int) -> dict:
    return {
        "format": "Plain",
        "DATA_TYPE": {
            "type": "Fp",
            "sign": True,
            "exponent": exponent,
            "mantissa": mantissa,
        },
    }


def _mx(family: str) -> dict:
    if family == "mxfp":
        element = {
            "type": "Fp",
            "sign": True,
            "exponent": 4,
            "mantissa": 3,
        }
    elif family == "mxint":
        element = {"type": "Int", "width": 4}
    else:  # pragma: no cover - test fixture programming error
        raise ValueError(family)
    return {
        "format": "Mx",
        "block": 8,
        "ELEM": element,
        "SCALE": {
            "type": "Fp",
            "sign": False,
            "exponent": 8,
            "mantissa": 0,
        },
    }


def _write_settings(
    path: Path,
    *,
    family: str,
    mlen: int,
    blen: int,
    vlen: int,
    internal_fp: tuple[int, int] = (8, 7),
) -> None:
    data = toml.load(SETTINGS_TEMPLATE)
    config = data["TRANSACTIONAL"]["CONFIG"]
    for name, value in {
        "MLEN": mlen,
        "BLEN": blen,
        "VLEN": vlen,
        "HLEN": min(128, mlen),
        "BROADCAST_AMOUNT": 1,
        "MATRIX_SRAM_SIZE": max(mlen, 2 * mlen),
        "VECTOR_SRAM_SIZE": max(vlen, 2 * vlen),
        "HBM_M_Prefetch_Amount": mlen,
        "HBM_V_Prefetch_Amount": blen,
        "HBM_V_Writeback_Amount": blen,
        "CLOCK_PERIOD_PS": 1000,
    }.items():
        config[name] = {"value": value}

    precision = data["TRANSACTIONAL"]["PRECISION"]
    mx = _mx(family)
    precision["HBM_M_WEIGHT_TYPE"] = mx
    precision["HBM_M_KV_TYPE"] = mx
    precision["HBM_V_ACT_TYPE"] = mx
    precision["HBM_V_KV_TYPE"] = mx
    plain = _plain_fp(*internal_fp)
    precision["MATRIX_SRAM_TYPE"] = plain
    precision["VECTOR_SRAM_TYPE"] = plain
    precision["SCALAR_FP"] = dict(plain["DATA_TYPE"])
    path.write_text(toml.dumps(data))


@pytest.fixture(scope="module", autouse=True)
def _build_rust_timing_dump() -> None:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is required for Python/Rust opcode timing parity")
    subprocess.run(
        ["cargo", "build", "--quiet", "--bin", "rtl_opcode_timing_dump"],
        cwd=EMULATOR,
        check=True,
    )


@pytest.mark.parametrize(
    ("family", "mlen", "blen", "vlen", "internal_fp"),
    [
        ("mxfp", 16, 4, 8, (8, 7)),
        ("mxfp", 32, 4, 16, (8, 7)),
        ("mxfp", 32, 8, 32, (8, 7)),
        ("mxfp", 64, 4, 64, (8, 7)),
        ("mxfp", 64, 16, 32, (6, 5)),
        ("mxfp", 512, 64, 512, (8, 7)),
        ("mxint", 16, 4, 8, (8, 7)),
        ("mxint", 32, 4, 16, (8, 7)),
        ("mxint", 32, 8, 32, (8, 7)),
        ("mxint", 512, 64, 512, (8, 7)),
    ],
)
def test_python_and_rust_opcode_timing_match_exactly(
    tmp_path: Path,
    family: str,
    mlen: int,
    blen: int,
    vlen: int,
    internal_fp: tuple[int, int],
) -> None:
    settings_path = tmp_path / "settings.toml"
    _write_settings(
        settings_path,
        family=family,
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        internal_fp=internal_fp,
    )
    environment = dict(os.environ, PLENA_SETTINGS_TOML=str(settings_path))
    rust = json.loads(
        subprocess.run(
            [str(RUST_BINARY)],
            cwd=EMULATOR,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )

    model = TransactionalCycleModel.load(settings_path)
    hardware = TimingHardware(
        model.mlen,
        model.blen,
        model.vlen,
        model.hlen,
        model.broadcast_amount,
    )
    precision = ComputePrecisionConfig.from_settings(model.raw_settings)
    assert rust["hardware"] == {
        "mlen": mlen,
        "blen": blen,
        "vlen": vlen,
        "hlen": min(128, mlen),
        "broadcast_amount": 1,
    }
    for opcode, rust_estimate in rust["opcodes"].items():
        python_estimate = CALIBRATION.estimate(opcode, hardware, precision)
        assert rust_estimate == (
            None if python_estimate is None else asdict(python_estimate)
        ), f"timing mismatch for {family} {mlen=} {blen=} {vlen=} {opcode=}"
