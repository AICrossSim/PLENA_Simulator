"""Run one RTL Router -> TopK -> dynamic expert FFN artifact in the emulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from pathlib import Path

from transactional_emulator.testbench.emulator_runner import run_emulator
from transactional_emulator.testbench.routed_moe.rtl_topk_artifact_test import (
    _portable_metrics,
    _u16_dump,
    _u32_dump,
    _write_rtl_settings,
)


# The transactional matrix/vector datapaths currently execute in f32 and only
# quantize when SRAM is serialized. Keep this semantic cross-check bounded by
# observed FP12 code distance; RTL bit accuracy is checked separately by the
# artifact's Python golden and SimTop test.
MAX_RTL_CODE_DISTANCE = {
    "activation": 1,
    "gate": 2,
    "hidden": 4,
    "output": 4,
    "combined": 4,
}
MAX_TOPK_WEIGHT_ABS_ERROR = 0.01


def _read_packed_fp12_vector(
    dump: bytes, element_addr: int, *, vlen: int
) -> list[int]:
    if element_addr % vlen:
        raise ValueError(f"VRAM address {element_addr} is not VLEN-aligned")
    row_bytes = (vlen * 12 + 7) // 8
    start = (element_addr // vlen) * row_bytes
    packed = int.from_bytes(dump[start : start + row_bytes], "little")
    return [(packed >> (12 * lane)) & 0xFFF for lane in range(vlen)]


def _bf16_bits(value: float) -> int:
    raw = struct.unpack("<I", struct.pack("<f", value))[0]
    rounded = raw + 0x7FFF + ((raw >> 16) & 1)
    return (rounded >> 16) & 0xFFFF


def _decode_fp12(bits: int) -> float:
    sign = -1.0 if bits & 0x800 else 1.0
    exponent = (bits >> 5) & 0x3F
    mantissa = bits & 0x1F
    if exponent == 0:
        value = mantissa / 32.0 * 2.0**-30
    elif exponent == 0x3F:
        return sign * math.inf if mantissa == 0 else math.nan
    else:
        value = (1.0 + mantissa / 32.0) * 2.0 ** (exponent - 31)
    return sign * value


def _mxint_row_bf16_bits(elements: list[int], scale_bits: int) -> list[int]:
    scale = 0.0 if scale_bits == 0 else 2.0 ** (scale_bits - 127)
    values = []
    for bits in elements:
        sign = -1.0 if bits & 0x80 else 1.0
        values.append(_bf16_bits(sign * (bits & 0x7F) / 128.0 * scale))
    return values


def run(artifact_dir: Path) -> dict:
    artifact_dir = artifact_dir.resolve()
    metadata_path = artifact_dir / "router_topk_expert_ffn_expected.json"
    metadata = json.loads(metadata_path.read_text())
    required = (
        "generated_machine_code.mem",
        "hbm_for_behave_sim.bin",
        "fp_sram.bin",
        "int_sram.bin",
    )
    missing = [name for name in required if not (artifact_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"RTL expert FFN artifact is incomplete: {missing}")
    if (artifact_dir / "vram_preload.bin").exists():
        raise AssertionError("expert FFN proof must not preload VRAM")

    words = [
        int(token, 16)
        for token in (artifact_dir / "generated_machine_code.mem").read_text().split()
    ]
    topk_words = [word for word in words if word & 0x3F == 0x37]
    policies = [(word >> 18) & 0xF for word in topk_words]
    if policies != [metadata["topk"]["policy"]]:
        raise AssertionError(
            f"Compiler V_TOPK policy mismatch: got {policies}, "
            f"expected {[metadata['topk']['policy']]}"
        )

    hbm_bytes = (artifact_dir / "hbm_for_behave_sim.bin").read_bytes()
    hbm_sha256 = hashlib.sha256(hbm_bytes).hexdigest()
    if hbm_sha256 != metadata["unified_hbm_sha256"]:
        raise AssertionError("Simulator and RTL did not consume byte-identical HBM")

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
    expected_indices = metadata["topk"]["indices"]
    index_base = metadata["topk"]["indices_base"]
    weight_base = metadata["topk"]["weights_base"]
    got_indices = int_values[index_base : index_base + len(expected_indices)]
    if got_indices != expected_indices:
        raise AssertionError(
            f"TopK indices: got {got_indices}, expected {expected_indices}"
        )
    got_weights = fp_bits[weight_base : weight_base + len(expected_indices)]
    expected_weights = metadata["topk"]["weights_fp12"]
    weight_distances = [
        abs(got_value - expected_value)
        for got_value, expected_value in zip(
            got_weights, expected_weights, strict=True
        )
    ]
    weight_abs_errors = [
        abs(_decode_fp12(got_value) - _decode_fp12(expected_value))
        for got_value, expected_value in zip(
            got_weights, expected_weights, strict=True
        )
    ]
    if max(weight_abs_errors) > MAX_TOPK_WEIGHT_ABS_ERROR:
        raise AssertionError(
            f"TopK FP12 weights: got {[hex(value) for value in got_weights]}, "
            f"expected {[hex(value) for value in expected_weights]}, "
            f"code distances={weight_distances}, "
            f"absolute errors={weight_abs_errors}, "
            f"limit={MAX_TOPK_WEIGHT_ABS_ERROR}"
        )

    vram_dump = (artifact_dir / "vram_dump.bin").read_bytes()
    activation = _read_packed_fp12_vector(
        vram_dump,
        metadata["activation_vram_address"],
        vlen=metadata["contract"]["vlen"],
    )
    activation_distances = [
        abs(got_value - expected_value)
        for got_value, expected_value in zip(
            activation, metadata["activation_fp12"], strict=True
        )
    ]
    if max(activation_distances) > MAX_RTL_CODE_DISTANCE["activation"]:
        raise AssertionError(
            f"input activation was corrupted: "
            f"got {[hex(value) for value in activation]}, "
            f"expected {[hex(value) for value in metadata['activation_fp12']]}, "
            f"code distances={activation_distances}"
        )

    vram_checks = {}
    for stage, address in metadata["vram_addresses"].items():
        if stage not in MAX_RTL_CODE_DISTANCE:
            raise AssertionError(f"missing RTL code-distance limit for {stage!r}")
        got = _read_packed_fp12_vector(
            vram_dump,
            address,
            vlen=metadata["contract"]["vlen"],
        )
        expected = metadata["fp12_golden"][stage]
        code_distances = [
            abs(got_value - expected_value)
            for got_value, expected_value in zip(got, expected, strict=True)
        ]
        max_distance = MAX_RTL_CODE_DISTANCE[stage]
        if max(code_distances) > max_distance:
            raise AssertionError(
                f"{stage} FP12 mismatch: got {[hex(value) for value in got]}, "
                f"expected {[hex(value) for value in expected]}, "
                f"code distances={code_distances}, limit={max_distance}"
            )
        vram_checks[stage] = {
            "bits": [f"0x{value:03X}" for value in got],
            "values": [_decode_fp12(value) for value in got],
            "rtl_code_distance": code_distances,
            "max_rtl_code_distance": max_distance,
            "bit_exact": got == expected,
        }

    final_prefetch = metadata["dynamic_prefetches"][-1]
    expected_mram = []
    for elements, scales in zip(
        final_prefetch["element_rows"], final_prefetch["scale_rows"], strict=True
    ):
        if len(scales) != 1:
            raise AssertionError("fixture expects one E8M0 scale per MLEN row")
        expected_mram.extend(_mxint_row_bf16_bits(elements, scales[0]))
    got_mram = _u16_dump(artifact_dir / "mram_dump.bin")[: len(expected_mram)]
    if got_mram != expected_mram:
        raise AssertionError(
            f"final selected down-weight MRAM payload mismatch: "
            f"got {[hex(value) for value in got_mram]}, "
            f"expected {[hex(value) for value in expected_mram]}"
        )

    selected_pair = metadata["topk"].get("selected_pair")
    selected_experts = (
        [got_indices[selected_pair]] if selected_pair is not None else got_indices
    )
    result = {
        "artifact_name": artifact_dir.name,
        "policy_name": metadata["policy_name"],
        "hbm_sha256": hbm_sha256,
        "policy": policies[0],
        "selected_experts": selected_experts,
        "weights_fp12": [f"0x{value:03X}" for value in got_weights],
        "weights_rtl_code_distance": weight_distances,
        "weights_rtl_abs_error": weight_abs_errors,
        "activation_fp12": [f"0x{value:03X}" for value in activation],
        "activation_rtl_code_distance": activation_distances,
        "vram_checks": vram_checks,
        "final_mram_words": len(got_mram),
        "metrics": _portable_metrics(metrics),
    }
    (artifact_dir / "simulator_expert_ffn_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.artifact_dir), indent=2))


if __name__ == "__main__":
    main()
