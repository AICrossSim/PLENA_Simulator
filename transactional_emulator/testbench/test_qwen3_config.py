"""
Qwen3 model config and hardware-constraint checks.

Run with: python transactional_emulator/testbench/test_qwen3_config.py
No HuggingFace model downloads are required.
"""

from pathlib import Path
import sys

import tomlkit

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transactional_emulator.testbench.model_configs.loader import (  # noqa: E402
    load_model_config_by_nickname,
    validate_hardware_constraints,
)


def _transactional_sram_depths() -> tuple[int, int]:
    with (REPO_ROOT / "plena_settings.toml").open() as f:
        config = tomlkit.load(f)
    txn = config["TRANSACTIONAL"]["CONFIG"]
    return int(txn["MATRIX_SRAM_SIZE"]["value"]), int(txn["VECTOR_SRAM_SIZE"]["value"])


def test_qwen3_8b_arch_config():
    mc = load_model_config_by_nickname("qwen3-8b")
    arch = mc.arch
    assert arch.hidden_size == 4096
    assert arch.inter_dim == 12288
    assert arch.num_heads == 32
    assert arch.num_kv_heads == 8
    assert arch.head_dim == 128
    assert arch.gqa_ratio == 4
    assert arch.num_layers == 36
    assert arch.rope_theta == 1000000
    assert arch.rms_norm_eps == 1.0e-6
    print("  PASS test_qwen3_8b_arch_config")


def _assert_preset_constraints(preset_name: str):
    mc = load_model_config_by_nickname("qwen3-8b")
    preset = mc.get_preset(preset_name)
    _, vector_depth = _transactional_sram_depths()
    matrix_depth = preset.mram_tile_capacity * preset.mlen
    issues = validate_hardware_constraints(
        mc.arch,
        preset,
        matrix_sram_depth=matrix_depth,
        vector_sram_depth=vector_depth,
        int_sram_depth=1024,
        fp_sram_depth=2048,
        fp_constant_num=10,
    )
    assert issues == [], f"Qwen3-8B preset {preset_name} violates constraints:\n" + "\n".join(issues)


def test_qwen3_8b_sliced_preset_constraints():
    _assert_preset_constraints("sliced_512x256x64_b1")
    print("  PASS test_qwen3_8b_sliced_preset_constraints")


def test_qwen3_8b_sliced_vlen_mlen_preset_constraints():
    _assert_preset_constraints("sliced_512x512x64_b1")
    print("  PASS test_qwen3_8b_sliced_vlen_mlen_preset_constraints")


def test_qwen3_8b_native_vlen_mlen_preset_constraints():
    _assert_preset_constraints("native_512x512x64_b1")
    print("  PASS test_qwen3_8b_native_vlen_mlen_preset_constraints")


def main() -> int:
    print("=" * 60)
    print("Qwen3-8B config tests")
    print("=" * 60)

    tests = [
        test_qwen3_8b_arch_config,
        test_qwen3_8b_sliced_preset_constraints,
        test_qwen3_8b_sliced_vlen_mlen_preset_constraints,
        test_qwen3_8b_native_vlen_mlen_preset_constraints,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"  FAIL {test.__name__}: {exc}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if failed:
        return 1
    print("All Qwen3-8B config tests PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
