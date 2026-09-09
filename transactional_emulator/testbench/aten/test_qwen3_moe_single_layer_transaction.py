from __future__ import annotations

import os
from pathlib import Path

import pytest

from transactional_emulator.testbench.aten.qwen3_moe_single_layer_transaction import (
    RUNTIME_DEPENDENCY_SCHEMA,
    SCHEMA,
    _seal_runtime_dependencies,
    _validate_runtime_dependencies,
    build_fixture,
    execute_fixture,
    validate_execution,
    validate_execution_receipt,
    validate_fixture_receipt,
)


@pytest.fixture(scope="module")
def sealed_fixture(tmp_path_factory):
    build_dir = tmp_path_factory.mktemp("qwen3_single_layer_fixture")
    receipt = build_fixture(build_dir)
    return build_dir, receipt


def test_fixture_is_deterministic_ordered_and_fail_closed(sealed_fixture, tmp_path):
    build_dir, first = sealed_fixture
    second = build_fixture(tmp_path / "second")

    assert first["schema"] == SCHEMA
    assert first["semantic_contract_sha256"] == second["semantic_contract_sha256"]
    assert first["artifacts"] == second["artifacts"]
    assert first["fixture_complete"] is True
    assert first["execution_complete"] is False
    assert first["tiny_single_layer_transaction_parity"] is False
    assert first["full_model_compiler_valid"] is False
    assert first["target_geometry_valid"] is False
    assert first["publication_rankable"] is False
    assert first["transformers_semantics_source"] == "custom_pinned_semantic_fixture"
    assert first["installed_transformers_api_invoked"] is False
    assert first["python_oracle_runtime"]["transformers_runtime_api_invoked"] is False
    assert len(first["expected_boundary_sha256"]) == 17
    assert len(set(first["expected_expert_ids_topk_order"])) == 8

    stages = first["compiler_receipt"]["stages_executed"]
    assert stages == [
        "attention_rmsnorm",
        "qkv_projection",
        "qk_rmsnorm",
        "rope",
        "kv_cache_append",
        "packedkv_decode_attention",
        "attention_output_projection",
        "attention_residual",
        "post_attention_rmsnorm",
        "router_bf16",
        "topk8_runtime",
        "dynamic_fused_experts",
        "fp32_route_combine",
        "moe_residual",
    ]
    assert validate_fixture_receipt(build_dir)["receipt_sha256"] == first[
        "receipt_sha256"
    ]


def test_fixture_rejects_program_and_oracle_tampering(sealed_fixture):
    build_dir, _ = sealed_fixture
    for relative in (
        Path("generated_asm_code.asm"),
        Path("oracle/route_scores.f32.bin"),
        Path("oracle/k_cache_post.mx.bin"),
    ):
        path = build_dir / relative
        original = path.read_bytes()
        path.write_bytes(original + b"tamper")
        with pytest.raises(RuntimeError, match="hash mismatch"):
            validate_fixture_receipt(build_dir)
        path.write_bytes(original)
    validate_fixture_receipt(build_dir)


def test_partial_execution_never_promotes_parity(sealed_fixture):
    build_dir, _ = sealed_fixture
    with pytest.raises(RuntimeError, match="execution is incomplete"):
        validate_execution(build_dir)
    with pytest.raises(RuntimeError, match="execution receipt is absent"):
        validate_execution_receipt(build_dir)


def test_runtime_dependency_receipt_seals_and_rechecks_libtorch_bundle(tmp_path):
    emulator = tmp_path / "emulator"
    emulator.write_bytes(b"emulator")
    bundle = tmp_path / "libtorch" / "lib"
    bundle.mkdir(parents=True)
    libraries = {
        name: bundle / name
        for name in ("libtorch_cpu.so", "libtorch.so", "libc10.so")
    }
    for index, path in enumerate(libraries.values(), start=1):
        path.write_bytes(bytes([index]) * 64)
    build_version = bundle.parent / "build-version"
    build_version.write_text("2.7.0+cpu\n", encoding="utf-8")
    ldd_stdout = (
        f"libtorch_cpu.so => {libraries['libtorch_cpu.so']} (0x00000001)\n"
        f"libc10.so => {libraries['libc10.so']} (0x00000002)\n"
    )

    receipt = _seal_runtime_dependencies(emulator, ldd_stdout=ldd_stdout)
    assert receipt["schema"] == RUNTIME_DEPENDENCY_SCHEMA
    assert receipt["libtorch_build_version"]["value"] == "2.7.0+cpu"
    assert receipt["libraries"]["libtorch_cpu.so"][
        "dynamic_loader_dependency"
    ] is True
    assert receipt["libraries"]["libtorch.so"][
        "dynamic_loader_dependency"
    ] is False
    assert (
        _validate_runtime_dependencies(
            receipt,
            emulator,
            ldd_stdout=ldd_stdout,
        )["content_sha256"]
        == receipt["content_sha256"]
    )

    original = libraries["libtorch_cpu.so"].read_bytes()
    libraries["libtorch_cpu.so"].write_bytes(original + b"tamper")
    with pytest.raises(RuntimeError, match="differ from the sealed receipt"):
        _validate_runtime_dependencies(
            receipt,
            emulator,
            ldd_stdout=ldd_stdout,
        )
    libraries["libtorch_cpu.so"].write_bytes(original)

    wrong_schema = {**receipt, "schema": "wrong-schema"}
    with pytest.raises(RuntimeError, match="wrong schema"):
        _validate_runtime_dependencies(
            wrong_schema,
            emulator,
            ldd_stdout=ldd_stdout,
        )


@pytest.mark.skipif(
    os.environ.get("PLENA_RUN_QWEN3_SINGLE_LAYER_TRANSACTION") != "1",
    reason="set PLENA_RUN_QWEN3_SINGLE_LAYER_TRANSACTION=1 for the Rust proof",
)
def test_compiler_generated_binary_matches_every_oracle_boundary(tmp_path):
    emulator = Path(
        os.environ.get(
            "PLENA_TRANSACTIONAL_EMULATOR",
            Path(__file__).resolve().parents[2]
            / "target/release/transactional_emulator",
        )
    )
    build_dir = tmp_path / "execution"
    build_fixture(build_dir)
    receipt = execute_fixture(build_dir, emulator)

    assert receipt["all_bf16_boundaries_byte_exact"] is True
    assert receipt["route_scores_fp32_byte_exact"] is True
    assert receipt["route_assignment_conserved"] is True
    assert receipt["kv_cache_post_bytes_exact"] is True
    assert receipt["kv_untouched_hbm_bytes_conserved"] is True
    assert receipt["compiler_generated_binary_emulator_parity"] is True
    assert receipt["tiny_single_layer_transaction_parity"] is True
    assert receipt["full_model_compiler_valid"] is False
    assert receipt["target_geometry_valid"] is False
    assert receipt["publication_rankable"] is False
    assert receipt["transformers_semantics_source"] == "custom_pinned_semantic_fixture"
    assert receipt["installed_transformers_api_invoked"] is False
    assert receipt["runtime_dependencies"]["schema"] == RUNTIME_DEPENDENCY_SCHEMA
    assert receipt["runtime_dependencies"]["libtorch_build_version"][
        "value"
    ] == "2.7.0+cpu"
    assert validate_execution_receipt(build_dir)["receipt_sha256"] == receipt[
        "receipt_sha256"
    ]

    route_path = build_dir / "route_f32_sram_dump.bin"
    original_route = route_path.read_bytes()
    route_path.write_bytes(b"\x00" * 32 + original_route[32:])
    with pytest.raises(RuntimeError, match="route-score bytes"):
        validate_execution(build_dir)
    route_path.write_bytes(original_route)

    post_hbm = build_dir / "post_hbm.bin"
    original_hbm = post_hbm.read_bytes()
    altered = bytearray(original_hbm)
    altered[-1] ^= 1
    post_hbm.write_bytes(altered)
    with pytest.raises(RuntimeError, match="outside its exact"):
        validate_execution(build_dir)
    post_hbm.write_bytes(original_hbm)
    validate_execution(build_dir)
