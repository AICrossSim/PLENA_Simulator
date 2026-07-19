"""Tests for compiler/runtime DMA transfer contract validation."""

from __future__ import annotations

import tomlkit

from transactional_emulator.testbench.emulator_runner import (
    _validate_compile_runtime_transfer_contract,
)


def _write_settings(path, *, prefetch: int, writeback: int) -> None:
    document = tomlkit.document()
    document["TRANSACTIONAL"] = {
        "CONFIG": {
            "HBM_V_Prefetch_Amount": {"value": prefetch},
            "HBM_V_Writeback_Amount": {"value": writeback},
        }
    }
    path.write_text(tomlkit.dumps(document))


def test_matching_dma_transfer_contract_passes(tmp_path, monkeypatch):
    settings = tmp_path / "plena_settings.toml"
    _write_settings(settings, prefetch=8, writeback=8)
    monkeypatch.setenv("PLENA_SETTINGS_TOML", str(settings))

    _validate_compile_runtime_transfer_contract(
        {
            "info": {
                "hbm_v_prefetch_amount": 8,
                "hbm_v_writeback_amount": 8,
            }
        }
    )


def test_mismatched_dma_transfer_contract_is_rejected(tmp_path, monkeypatch):
    settings = tmp_path / "plena_settings.toml"
    _write_settings(settings, prefetch=8, writeback=8)
    monkeypatch.setenv("PLENA_SETTINGS_TOML", str(settings))

    try:
        _validate_compile_runtime_transfer_contract(
            {
                "info": {
                    "hbm_v_prefetch_amount": 4,
                    "hbm_v_writeback_amount": 8,
                }
            }
        )
    except ValueError as exc:
        message = str(exc)
        assert "HBM_V_Prefetch_Amount" in message
        assert "compiler=4, emulator=8" in message
    else:
        raise AssertionError("DMA transfer mismatch was not rejected")
