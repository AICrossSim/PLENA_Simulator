"""Run the RTL production-path TopK artifact in the transactional emulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import tomlkit

from transactional_emulator.testbench.emulator_runner import run_emulator


REPO_ROOT = Path(__file__).resolve().parents[3]


def _replace_table(table, values: dict) -> None:
    for key in list(table):
        del table[key]
    for key, value in values.items():
        table[key] = value


def _write_rtl_settings(path: Path) -> None:
    settings = tomlkit.parse((REPO_ROOT / "plena_settings.toml").read_text())
    transactional = settings["TRANSACTIONAL"]
    config = transactional["CONFIG"]
    for name, value in {
        "BLEN": 4,
        "HLEN": 8,
        "MLEN": 8,
        "VLEN": 8,
        "BROADCAST_AMOUNT": 4,
        "HBM_SIZE": 65536,
        "HBM_M_Prefetch_Amount": 4,
        "HBM_V_Prefetch_Amount": 4,
        "HBM_V_Writeback_Amount": 4,
    }.items():
        config[name]["value"] = value

    precision = transactional["PRECISION"]
    precision["VECTOR_SRAM_TYPE"]["format"] = "Plain"
    _replace_table(
        precision["VECTOR_SRAM_TYPE"]["DATA_TYPE"],
        {"type": "Fp", "sign": True, "exponent": 6, "mantissa": 5},
    )
    precision["HBM_V_ACT_TYPE"]["format"] = "Mx"
    precision["HBM_V_ACT_TYPE"]["block"] = 8
    _replace_table(
        precision["HBM_V_ACT_TYPE"]["ELEM"],
        {"type": "MxInt", "width": 8},
    )
    _replace_table(
        precision["HBM_V_ACT_TYPE"]["SCALE"],
        {"type": "Fp", "sign": False, "exponent": 8, "mantissa": 0},
    )
    _replace_table(
        precision["SCALAR_FP"],
        {"type": "Fp", "sign": True, "exponent": 6, "mantissa": 5},
    )
    path.write_text(tomlkit.dumps(settings))


def _u32_dump(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 4:
        raise ValueError(f"INT SRAM dump has a partial word: {path}")
    return [int.from_bytes(data[offset : offset + 4], "little") for offset in range(0, len(data), 4)]


def _u16_dump(path: Path) -> list[int]:
    data = path.read_bytes()
    if len(data) % 2:
        raise ValueError(f"FP SRAM dump has a partial word: {path}")
    return [int.from_bytes(data[offset : offset + 2], "little") for offset in range(0, len(data), 2)]


def run(artifact_dir: Path) -> dict:
    artifact_dir = artifact_dir.resolve()
    metadata = json.loads((artifact_dir / "router_topk_expected.json").read_text())
    required = (
        "generated_machine_code.mem",
        "hbm_for_behave_sim.bin",
        "fp_sram.bin",
        "int_sram.bin",
    )
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"RTL TopK artifact is incomplete: {missing}")
    if (artifact_dir / "vram_preload.bin").exists():
        raise AssertionError("RTL TopK integration artifact must not preload VRAM")

    words = [int(token, 16) for token in (artifact_dir / "generated_machine_code.mem").read_text().split()]
    topk_words = [word for word in words if word & 0x3F == 0x37]
    policies = [(word >> 18) & 0xF for word in topk_words]
    if policies != [0, 1]:
        raise AssertionError(f"expected Compiler V_TOPK policies [0, 1], got {policies}")

    hbm_bytes = (artifact_dir / "hbm_for_behave_sim.bin").read_bytes()
    hbm_sha256 = hashlib.sha256(hbm_bytes).hexdigest()
    if hbm_sha256 != metadata["unified_hbm_sha256"]:
        raise AssertionError("Simulator HBM binary is not the byte-identical RTL HBM image")

    settings_path = artifact_dir / "simulator_rtl_settings.toml"
    _write_rtl_settings(settings_path)
    previous_settings = os.environ.get("PLENA_SETTINGS_TOML")
    os.environ["PLENA_SETTINGS_TOML"] = str(settings_path)
    try:
        metrics = run_emulator(artifact_dir, threads=1)
    finally:
        if previous_settings is None:
            os.environ.pop("PLENA_SETTINGS_TOML", None)
        else:
            os.environ["PLENA_SETTINGS_TOML"] = previous_settings

    int_values = _u32_dump(artifact_dir / "intsram_dump.bin")
    fp_bits = _u16_dump(artifact_dir / "fpsram_dump.bin")
    checks = {}
    for policy_name, expected_fp_bits in (("gpt_oss", 0x3E80), ("qwen3", 0x3E00)):
        policy = metadata[policy_name]
        count = len(policy["indices"])
        index_base = policy["indices_base"]
        weight_base = policy["weights_base"]
        got_indices = int_values[index_base : index_base + count]
        got_weights = fp_bits[weight_base : weight_base + count]
        if got_indices != policy["indices"]:
            raise AssertionError(f"{policy_name} indices: got {got_indices}, expected {policy['indices']}")
        if got_weights != [expected_fp_bits] * count:
            raise AssertionError(
                f"{policy_name} BF16 weights: got {[hex(value) for value in got_weights]}, "
                f"expected {hex(expected_fp_bits)}"
            )
        checks[policy_name] = {
            "indices": got_indices,
            "weight_bits": [f"0x{value:04X}" for value in got_weights],
        }

    result = {
        "artifact_dir": str(artifact_dir),
        "hbm_sha256": hbm_sha256,
        "policies": policies,
        "checks": checks,
        "metrics": metrics,
    }
    (artifact_dir / "simulator_topk_results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
