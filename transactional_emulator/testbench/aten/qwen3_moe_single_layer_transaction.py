"""Content-addressed hidden-64 Qwen3-MoE single-layer decode proof.

The transaction is deliberately tiny but complete: q_len=1 PackedKV attention
including Q/K normalization and cache append is followed by the exact BF16/FP32
routed-MoE tail.  It is a validation fixture, never a target-shape or timing
claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import struct
import subprocess
import tempfile
from typing import Any

import numpy as np
import torch

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.packed_kv import PackedKVLayout
from compiler.aten.qwen3_moe_runtime import (
    QWEN3_MOE_MODEL_ID,
    QWEN3_MOE_MODEL_REVISION,
    QWEN3_MOE_TRANSFORMERS_ABI,
    Qwen3RawBf16ExpertBankLayout,
    pack_qwen3_raw_bf16_expert_hbm,
    qwen3_post_attention_moe_cpu_reference,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim


SIMULATOR_ROOT = Path(__file__).resolve().parents[3]
COMPILER_ROOT = SIMULATOR_ROOT / "compiler"
EMULATOR_ROOT = SIMULATOR_ROOT / "transactional_emulator"
SCHEMA = "plena-qwen3-tiny-single-layer-decode-transaction-v1"
RUNTIME_DEPENDENCY_SCHEMA = (
    "plena-qwen3-tiny-single-layer-runtime-dependencies-v1"
)
DESCRIPTOR_BASE = 0x1_0000
HBM_SIZE = 4 * 1024 * 1024
VRAM_ELEMENTS = 65_536
SOURCE_FILES = (
    Path(__file__).resolve(),
    COMPILER_ROOT / "aten/plena/program_routed_moe.py",
    COMPILER_ROOT / "aten/plena/program_attention.py",
    COMPILER_ROOT / "aten/plena/packed_kv.py",
    COMPILER_ROOT / "aten/qwen3_moe_runtime.py",
    COMPILER_ROOT / "doc/operation.svh",
    EMULATOR_ROOT / "src/accelerator/dispatch.rs",
    EMULATOR_ROOT / "src/accelerator/qwen3_moe.rs",
    EMULATOR_ROOT / "src/vector_machine.rs",
    EMULATOR_ROOT / "src/op.rs",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


@contextlib.contextmanager
def _settings_environment(settings: Path) -> Iterator[None]:
    previous = os.environ.get("PLENA_SETTINGS_TOML")
    os.environ["PLENA_SETTINGS_TOML"] = str(settings)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PLENA_SETTINGS_TOML", None)
        else:
            os.environ["PLENA_SETTINGS_TOML"] = previous


def _replace_table_value(text: str, table: str, value: Any) -> str:
    pattern = re.compile(
        rf"(^\[{re.escape(table)}\]\s*$.*?^value\s*=\s*)[^\n#]+",
        re.MULTILINE | re.DOTALL,
    )
    rendered = json.dumps(value) if isinstance(value, str) else str(value)
    updated, count = pattern.subn(rf"\g<1>{rendered} ", text, count=1)
    if count != 1:
        raise RuntimeError(f"settings table {table!r} is absent or ambiguous")
    return updated


def write_tiny_settings(path: Path) -> None:
    text = (SIMULATOR_ROOT / "plena_settings.toml").read_text(encoding="utf-8")
    overrides = {
        "TRANSACTIONAL.CONFIG.MLEN": 64,
        "TRANSACTIONAL.CONFIG.VLEN": 64,
        "TRANSACTIONAL.CONFIG.HLEN": 64,
        "TRANSACTIONAL.CONFIG.BLEN": 4,
        "TRANSACTIONAL.CONFIG.BROADCAST_AMOUNT": 1,
        "TRANSACTIONAL.CONFIG.HBM_SIZE": HBM_SIZE,
        "TRANSACTIONAL.CONFIG.HBM_M_Prefetch_Amount": 64,
        "TRANSACTIONAL.CONFIG.HBM_V_Prefetch_Amount": 4,
        "TRANSACTIONAL.CONFIG.HBM_V_Writeback_Amount": 4,
    }
    for table, value in overrides.items():
        text = _replace_table_value(text, table, value)
    _atomic_write(path, text.encode("utf-8"))


def _bf16_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .cpu()
        .to(torch.bfloat16)
        .contiguous()
        .view(torch.uint16)
        .numpy()
        .astype("<u2", copy=False)
        .tobytes()
    )


def _tail_fixture() -> dict[str, torch.Tensor]:
    hidden = intermediate = 64
    experts = 128
    hidden_index = torch.arange(hidden, dtype=torch.int64)
    expert_index = torch.arange(experts, dtype=torch.int64)[:, None, None]
    output_index = torch.arange(intermediate, dtype=torch.int64)[None, :, None]
    input_index = hidden_index[None, None, :]
    norm_weight = (0.75 + ((hidden_index * 11) % 17).float() / 64).to(
        torch.bfloat16
    )
    router_weight = (
        (
            (
                torch.arange(experts, dtype=torch.int64)[:, None] * 97
                + hidden_index[None, :] * 53
                + 19
            )
            % 2001
            - 1000
        ).float()
        / 263
    ).to(torch.bfloat16)
    gate = (
        ((expert_index * 13 + output_index * 17 + input_index * 19 + 5) % 257 - 128).float()
        / 509
    ).to(torch.bfloat16)
    up = (
        ((expert_index * 23 + output_index * 29 + input_index * 31 + 7) % 257 - 128).float()
        / 521
    ).to(torch.bfloat16)
    down = (
        (
            (
                expert_index * 37
                + hidden_index[None, :, None] * 41
                + torch.arange(intermediate, dtype=torch.int64)[None, None, :] * 43
                + 11
            )
            % 257
            - 128
        ).float()
        / 523
    ).to(torch.bfloat16)
    return {
        "post_attention_norm": norm_weight,
        "router": router_weight,
        "fused_gate_up": torch.cat((gate, up), dim=1),
        "down": down,
    }


def _physical(tensor: torch.Tensor, rows: int, cols: int = 64) -> torch.Tensor:
    output = torch.zeros((rows, cols), dtype=torch.bfloat16)
    logical = tensor.to(torch.bfloat16)
    if logical.ndim == 1:
        logical = logical.unsqueeze(0)
    output[: logical.shape[0], : logical.shape[1]] = logical
    return output


def _put_vram(
    image: torch.Tensor, compiler: PlenaCompiler, variable, value: torch.Tensor
) -> None:
    base = compiler.get_vram_addr(variable.name)
    physical = _physical(value, *variable.physical_shape)
    end = base + physical.numel()
    if end > image.numel():
        raise RuntimeError(f"VRAM fixture for {variable.name} exceeds sealed capacity")
    image[base:end] = physical.flatten()


def _input_layout(compiler: PlenaCompiler, tensor) -> dict[str, Any]:
    layout = compiler.get_hbm_layout(tensor.name)
    return {
        "logical_shape": list(tensor.shape),
        "physical_shape": list(tensor.physical_shape),
        "source_rows": tensor.shape[0],
        "storage_rows": tensor.physical_shape[0],
        "source_row_elements": tensor.shape[1],
        "storage_row_elements": tensor.physical_shape[1],
        "precision_role": layout.precision_role,
        "element_bits": layout.hbm_element_width,
        "block_size": layout.hbm_block_size,
        "scale_bits": layout.hbm_scale_width,
        "hbm_size": layout.hbm_size,
    }


def _source_receipt() -> dict[str, str]:
    return {
        str(path.relative_to(SIMULATOR_ROOT)): _sha256_file(path)
        for path in SOURCE_FILES
    }


def _python_oracle_runtime() -> dict[str, Any]:
    """Record the runtime used by the custom pinned semantic fixture."""

    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "torch_distribution_version": importlib.metadata.version("torch"),
        "torch_runtime_version": str(torch.__version__),
        "transformers_distribution_version": importlib.metadata.version(
            "transformers"
        ),
        "transformers_semantic_fixture_abi": QWEN3_MOE_TRANSFORMERS_ABI,
        "transformers_runtime_api_invoked": False,
        "oracle_implementation": "custom_torch_semantic_fixture",
    }


def _ldd_output(emulator_binary: Path) -> str:
    result = subprocess.run(
        ["ldd", str(emulator_binary)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ldd failed for emulator binary with exit {result.returncode}"
        )
    return result.stdout


def _parse_ldd_paths(output: str) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    pattern = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)\s*$")
    for line in output.splitlines():
        match = pattern.match(line)
        if match is None:
            continue
        soname, raw_path = match.groups()
        if raw_path == "not":
            raise RuntimeError(f"dynamic library {soname} is unresolved")
        path = Path(raw_path).resolve()
        if soname in resolved and resolved[soname] != path:
            raise RuntimeError(f"dynamic library {soname} resolves ambiguously")
        resolved[soname] = path
    return resolved


def _seal_runtime_dependencies(
    emulator_binary: Path,
    *,
    ldd_stdout: str | None = None,
) -> dict[str, Any]:
    """Seal the libtorch bundle selected for the emulator process."""

    emulator_binary = Path(emulator_binary).resolve()
    output = _ldd_output(emulator_binary) if ldd_stdout is None else ldd_stdout
    resolved = _parse_ldd_paths(output)
    dynamically_required = ("libtorch_cpu.so", "libc10.so")
    missing = [name for name in dynamically_required if name not in resolved]
    if missing:
        raise RuntimeError(
            f"emulator dynamic dependency resolution is missing {missing}"
        )
    libtorch_dir = resolved["libtorch_cpu.so"].parent
    libraries: dict[str, dict[str, Any]] = {}
    for name in (*dynamically_required, "libtorch.so"):
        loaded = name in resolved
        path = resolved[name] if loaded else (libtorch_dir / name).resolve()
        if not path.is_file():
            raise RuntimeError(f"sealed libtorch bundle member is missing: {path}")
        if path.parent != libtorch_dir:
            raise RuntimeError("emulator resolves a mixed-directory libtorch bundle")
        libraries[name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "dynamic_loader_dependency": loaded,
            "resolution": (
                "dynamic_loader"
                if loaded
                else "same_bundle_member_not_needed_by_dynamic_loader"
            ),
        }
    build_version = (libtorch_dir.parent / "build-version").resolve()
    if not build_version.is_file():
        raise RuntimeError(f"libtorch build-version is missing: {build_version}")
    version = build_version.read_text(encoding="utf-8").strip()
    if not version or any(character.isspace() for character in version):
        raise RuntimeError("libtorch build-version is empty or malformed")
    body = {
        "schema": RUNTIME_DEPENDENCY_SCHEMA,
        "resolution_tool": "ldd",
        "emulator_binary": str(emulator_binary),
        "libtorch_bundle_directory": str(libtorch_dir),
        "libraries": libraries,
        "libtorch_build_version": {
            "path": str(build_version),
            "value": version,
            "sha256": _sha256_file(build_version),
        },
        "required_dynamic_libraries_resolved": True,
        "single_libtorch_bundle": True,
    }
    return {
        **body,
        "content_sha256": _sha256_bytes(_canonical_bytes(body)),
    }


def _validate_runtime_dependencies(
    receipt: dict[str, Any],
    emulator_binary: Path,
    *,
    ldd_stdout: str | None = None,
) -> dict[str, Any]:
    if receipt.get("schema") != RUNTIME_DEPENDENCY_SCHEMA:
        raise RuntimeError("runtime dependency receipt has the wrong schema")
    recorded_hash = receipt.get("content_sha256")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if recorded_hash != _sha256_bytes(_canonical_bytes(body)):
        raise RuntimeError("runtime dependency receipt content hash mismatch")
    current = _seal_runtime_dependencies(
        emulator_binary,
        ldd_stdout=ldd_stdout,
    )
    if receipt != current:
        raise RuntimeError(
            "resolved emulator runtime dependencies differ from the sealed receipt"
        )
    return receipt


def _validate_execution_invocation(build_dir: Path) -> dict[str, Any]:
    path = build_dir / "execution_invocation.json"
    try:
        invocation = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("execution invocation is absent or unreadable") from error
    recorded_hash = invocation.pop("invocation_sha256", None)
    if recorded_hash != _sha256_bytes(_canonical_bytes(invocation)):
        raise RuntimeError("execution invocation content hash mismatch")
    invocation["invocation_sha256"] = recorded_hash
    if invocation.get("schema") != SCHEMA:
        raise RuntimeError("execution invocation has the wrong schema")
    emulator_binary = Path(invocation.get("emulator_binary", "")).resolve()
    if not emulator_binary.is_file():
        raise RuntimeError("execution invocation emulator binary is missing")
    if _sha256_file(emulator_binary) != invocation.get("emulator_binary_sha256"):
        raise RuntimeError("execution invocation emulator binary hash mismatch")
    dependencies = invocation.get("runtime_dependencies")
    if not isinstance(dependencies, dict):
        raise RuntimeError("execution invocation runtime dependencies are absent")
    _validate_runtime_dependencies(dependencies, emulator_binary)
    return invocation


def build_fixture(build_dir: Path) -> dict[str, Any]:
    """Generate the compiler program and all exact input/oracle artifacts."""

    build_dir = Path(build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    settings = build_dir / "plena_settings.toml"
    write_tiny_settings(settings)
    tail = _tail_fixture()
    packed = PackedKVLayout(
        kv_heads=1,
        head_dim=64,
        mlen=64,
        block_size=8,
        element_bits=8,
        scale_bits=8,
    )
    expert_layout = Qwen3RawBf16ExpertBankLayout.canonical(
        DESCRIPTOR_BASE, hidden=64, intermediate=64
    )

    with _settings_environment(settings):
        compiler = PlenaCompiler(
            mlen=64,
            blen=4,
            hbm_v_prefetch_amount=4,
            hbm_v_writeback_amount=4,
        )

        def bf16_vram(name: str, rows: int = 1):
            physical_rows = 4 if rows == 1 else rows
            return compiler.alloc(
                name,
                rows,
                64,
                strict=False,
                physical_shape=(physical_rows, 64),
            )

        layer_input = bf16_vram("layer_input")
        attention_norm = bf16_vram("attention_norm")
        q_norm = bf16_vram("q_norm")
        k_norm = bf16_vram("k_norm")
        rope_cos = bf16_vram("rope_cos")
        rope_sin = bf16_vram("rope_sin")
        post_attention_norm = bf16_vram("post_attention_norm")
        router = bf16_vram("router", rows=128)

        def mx_input(name: str, role: str):
            return compiler.input(
                name,
                (64, 64),
                physical_shape=(64, 64),
                hbm_element_width=8,
                hbm_block_size=8,
                hbm_scale_width=8,
                precision_role=role,
            )

        q_weight = mx_input("q_weight", "weight")
        k_weight = mx_input("k_weight", "weight")
        v_weight = mx_input("v_weight", "weight")
        rotate_weight = mx_input("rotate_half_weight", "weight")
        o_weight = mx_input("o_weight", "weight")
        k_cache = mx_input("k_cache", "key")
        v_cache = mx_input("v_cache", "value")
        output, compiler_receipt = (
            compiler.qwen3_tiny_single_layer_decode_transaction_v0(
                layer_input,
                attention_norm,
                q_norm,
                k_norm,
                rope_cos,
                rope_sin,
                post_attention_norm,
                router,
                q_weight,
                k_weight,
                v_weight,
                rotate_weight,
                o_weight,
                k_cache,
                v_cache,
                packed_layout=packed,
                descriptor_layout=expert_layout,
                cache_position=3,
            )
        )
        compiler.emit("C_BREAK\n")
        assembly = compiler.compile()

        ones = torch.ones((1, 64), dtype=torch.bfloat16)
        half = torch.full((1, 64), 0.5, dtype=torch.bfloat16)
        identity = torch.eye(64, dtype=torch.bfloat16)
        rotate = torch.zeros((64, 64), dtype=torch.bfloat16)
        index = torch.arange(32)
        rotate[index + 32, index] = -1
        rotate[index, index + 32] = 1
        rope = torch.cat(
            (
                torch.zeros((1, 32), dtype=torch.bfloat16),
                torch.ones((1, 32), dtype=torch.bfloat16),
            ),
            dim=1,
        )
        k_initial = torch.zeros((64, 64), dtype=torch.bfloat16)
        k_initial[:3] = rope
        v_initial = torch.zeros((64, 64), dtype=torch.bfloat16)
        v_initial[1] = 0.5
        v_initial[2] = 1.0
        hbm_tensors = {
            "q_weight": identity,
            "k_weight": identity,
            "v_weight": identity,
            "rotate_half_weight": rotate,
            "o_weight": identity,
            "k_cache": k_initial,
            "v_cache": v_initial,
        }
        input_vars = {
            variable.name: variable
            for variable in (
                q_weight,
                k_weight,
                v_weight,
                rotate_weight,
                o_weight,
                k_cache,
                v_cache,
            )
        }
        data_order = sorted(
            hbm_tensors,
            key=lambda key: compiler.get_hbm_layout(key).hbm_base_addr,
        )
        tensor_layouts = {
            name: _input_layout(compiler, input_vars[name]) for name in data_order
        }
        hbm_addrs = {
            name: compiler.get_hbm_layout(name).hbm_base_addr for name in data_order
        }

        vram = torch.zeros(VRAM_ELEMENTS, dtype=torch.bfloat16)
        for variable, value in (
            (layer_input, ones),
            (attention_norm, ones),
            (q_norm, ones),
            (k_norm, ones),
            (rope_cos, half),
            (rope_sin, half),
            (post_attention_norm, tail["post_attention_norm"]),
            (router, tail["router"]),
        ):
            _put_vram(vram, compiler, variable, value)

        attention_output = torch.full((1, 64), 0.625, dtype=torch.bfloat16)
        tail_oracle = qwen3_post_attention_moe_cpu_reference(
            attention_output,
            ones,
            tail["post_attention_norm"],
            tail["router"],
            tail["fused_gate_up"],
            tail["down"],
        )
        expected_values = {
            "layer_input": ones,
            "attention_normalized": ones,
            "attention_normalized_padded": ones,
            "q_projected": ones,
            "k_projected": ones,
            "v_projected": ones,
            "q_normalized": ones,
            "k_normalized": ones,
            "q_rope": rope,
            "k_rope": rope,
            "attention_output": attention_output,
            "o_projected": attention_output,
            "attention_residual": tail_oracle.attention_residual,
            "post_attention_normalized": tail_oracle.normalized,
            "router_logits": tail_oracle.moe.router.raw_logits.reshape(2, 64),
            "expert_combined": tail_oracle.moe.output,
            "output": tail_oracle.output,
        }
        expected_hashes: dict[str, str] = {}
        for boundary, metadata in compiler_receipt["vram_boundaries"].items():
            physical_rows, physical_cols = metadata["physical_shape"]
            payload = _bf16_bytes(
                _physical(
                    expected_values[boundary], physical_rows, physical_cols
                )
            )
            boundary_path = build_dir / "oracle" / f"{boundary}.bf16.bin"
            _atomic_write(boundary_path, payload)
            expected_hashes[boundary] = _sha256_bytes(payload)

        expert_image = pack_qwen3_raw_bf16_expert_hbm(
            expert_layout, tail["fused_gate_up"], tail["down"]
        )
        assembly_path = build_dir / "generated_asm_code.asm"
        _atomic_write(assembly_path, assembly.encode("utf-8"))
        _atomic_write(build_dir / "vram_preload.bin", _bf16_bytes(vram))
        fp_values = np.zeros(512, dtype="<f2")
        fp_values[:3] = [0.0, 0.125, -np.inf]
        _atomic_write(build_dir / "fp_sram.bin", fp_values.tobytes())
        _atomic_write(
            build_dir / "int_sram.bin", np.zeros(1024, dtype="<u4").tobytes()
        )
        create_mem_for_sim(
            mode="behave_sim",
            specified_data_order=data_order,
            build_path=build_dir,
            input_tensors=hbm_tensors,
            tensor_layouts=tensor_layouts,
            hbm_addrs=hbm_addrs,
        )
        hbm_path = build_dir / "hbm_for_behave_sim.bin"
        pre_hbm = bytearray(hbm_path.read_bytes())
        if len(pre_hbm) < expert_layout.end_address:
            pre_hbm.extend(b"\x00" * (expert_layout.end_address - len(pre_hbm)))
        pre_hbm[
            expert_layout.descriptor_base : expert_layout.end_address
        ] = expert_image
        _atomic_write(hbm_path, bytes(pre_hbm))
        _atomic_write(build_dir / "hbm_size.txt", f"{HBM_SIZE}\n".encode())

        expected_k = k_initial.clone()
        expected_k[3] = rope[0]
        expected_v = v_initial.clone()
        expected_v[3] = 1.0
        cache_oracle_dir = build_dir / "cache_oracle_stage"
        _atomic_write(cache_oracle_dir / "generated_asm_code.asm", b"C_BREAK\n")
        create_mem_for_sim(
            mode="behave_sim",
            specified_data_order=["k_cache", "v_cache"],
            build_path=cache_oracle_dir,
            input_tensors={"k_cache": expected_k, "v_cache": expected_v},
            tensor_layouts={
                key: tensor_layouts[key] for key in ("k_cache", "v_cache")
            },
            hbm_addrs={
                "k_cache": 0,
                "v_cache": compiler.get_hbm_layout("k_cache").hbm_size,
            },
        )
        cache_oracle = (cache_oracle_dir / "hbm_for_behave_sim.bin").read_bytes()
        k_size = compiler.get_hbm_layout("k_cache").hbm_size
        v_size = compiler.get_hbm_layout("v_cache").hbm_size
        _atomic_write(build_dir / "oracle/k_cache_post.mx.bin", cache_oracle[:k_size])
        _atomic_write(
            build_dir / "oracle/v_cache_post.mx.bin",
            cache_oracle[k_size : k_size + v_size],
        )

        route_scores = tail_oracle.moe.router.route_scores.flatten().float()
        route_ids = tail_oracle.moe.router.expert_indices.flatten().to(torch.int64)
        route_bytes = route_scores.numpy().astype("<f4", copy=False).tobytes()
        id_bytes = np.asarray(route_ids.tolist(), dtype="<u4").tobytes()
        _atomic_write(build_dir / "oracle/route_scores.f32.bin", route_bytes)
        _atomic_write(build_dir / "oracle/expert_ids.u32.bin", id_bytes)

        python_runtime = _python_oracle_runtime()
        semantic_contract = {
            "schema": SCHEMA,
            "model_id": QWEN3_MOE_MODEL_ID,
            "model_revision": QWEN3_MOE_MODEL_REVISION,
            "transformers_abi": QWEN3_MOE_TRANSFORMERS_ABI,
            "transformers_semantics_source": "custom_pinned_semantic_fixture",
            "installed_transformers_api_invoked": False,
            "installed_torch_version": python_runtime[
                "torch_distribution_version"
            ],
            "installed_transformers_version": python_runtime[
                "transformers_distribution_version"
            ],
            "python_oracle_runtime": python_runtime,
            "geometry": {
                "batch": 1,
                "q_len": 1,
                "hidden": 64,
                "intermediate": 64,
                "experts": 128,
                "top_k": 8,
                "cache_position": 3,
            },
            "precision": {
                "vram": "BF16",
                "weights": "MXFP8_E4M3_block8_scaleE8M0",
                "kv": "MXFP8_E4M3_block8_scaleE8M0",
                "route": "FP32",
                "expert_hbm": "raw_BF16",
            },
            "source_sha256": _source_receipt(),
            "compiler_receipt": compiler_receipt,
            "expected_boundary_sha256": expected_hashes,
            "route_scores_sha256": _sha256_bytes(route_bytes),
            "expert_ids_sha256": _sha256_bytes(id_bytes),
            "cache_post_sha256": {
                cache_name: _sha256_file(
                    build_dir / "oracle" / f"{cache_name}_post.mx.bin"
                )
                for cache_name in ("k_cache", "v_cache")
            },
            "expected_expert_ids_topk_order": route_ids.tolist(),
        }
        fixture_receipt = {
            **semantic_contract,
            "semantic_contract_sha256": _sha256_bytes(
                _canonical_bytes(semantic_contract)
            ),
            "artifacts": {
                path.name: _sha256_file(path)
                for path in (
                    settings,
                    assembly_path,
                    build_dir / "generated_machine_code.mem",
                    hbm_path,
                    build_dir / "vram_preload.bin",
                    build_dir / "fp_sram.bin",
                    build_dir / "int_sram.bin",
                )
            },
            "output_vram_address": compiler.get_vram_addr(output.name),
            "hbm_inputs": {
                name: {
                    "base": compiler.get_hbm_layout(name).hbm_base_addr,
                    "size": compiler.get_hbm_layout(name).hbm_size,
                }
                for name in data_order
            },
            "expert_hbm": expert_layout.as_dict(),
            "assignment_expected": 8,
            "fixture_complete": True,
            "execution_complete": False,
            "tiny_single_layer_transaction_parity": False,
            "full_model_compiler_valid": False,
            "target_geometry_valid": False,
            "rtl_valid": False,
            "timing_calibrated": False,
            "publication_rankable": False,
        }
        fixture_receipt["receipt_sha256"] = _sha256_bytes(
            _canonical_bytes(fixture_receipt)
        )
        _atomic_json(build_dir / "fixture_receipt.json", fixture_receipt)
        return fixture_receipt


def _read_boundary(vram: bytes, metadata: dict[str, Any]) -> bytes:
    start = int(metadata["base_element_address"]) * 2
    end = int(metadata["end_element_address_exclusive"]) * 2
    if start < 0 or end <= start or end > len(vram):
        raise RuntimeError("VRAM boundary lies outside the emulator dump")
    return vram[start:end]


def _contiguous_ranges(addresses: list[dict[str, int]], cache_base: int):
    for address in addresses:
        yield (
            cache_base + address["element_offset_bytes"],
            cache_base
            + address["element_offset_bytes"]
            + address["element_transfer_bytes"],
        )
        yield (
            cache_base
            + address["element_plane_bytes"]
            + address["scale_offset_bytes"],
            cache_base
            + address["element_plane_bytes"]
            + address["scale_offset_bytes"]
            + address["scale_transfer_bytes"],
        )


def validate_execution(build_dir: Path) -> dict[str, Any]:
    """Validate every byte boundary and write the final sealed receipt."""

    build_dir = Path(build_dir).resolve()
    fixture = validate_fixture_receipt(build_dir)
    required = {
        "vram_dump.bin",
        "route_f32_sram_dump.bin",
        "intsram_dump.bin",
        "post_hbm.bin",
        "op_stats.jsonl",
        "emulator_stdout.log",
        "execution_invocation.json",
    }
    missing = sorted(name for name in required if not (build_dir / name).is_file())
    if missing:
        raise RuntimeError(f"emulator execution is incomplete; missing {missing}")
    invocation = _validate_execution_invocation(build_dir)
    if invocation.get("fixture_receipt_sha256") != _sha256_file(
        build_dir / "fixture_receipt.json"
    ):
        raise RuntimeError("execution invocation is bound to a different fixture")

    vram = (build_dir / "vram_dump.bin").read_bytes()
    boundary_hashes: dict[str, str] = {}
    for name, metadata in fixture["compiler_receipt"]["vram_boundaries"].items():
        actual = _read_boundary(vram, metadata)
        expected = (build_dir / "oracle" / f"{name}.bf16.bin").read_bytes()
        if actual != expected:
            mismatch = next(
                index
                for index, (left, right) in enumerate(zip(actual, expected))
                if left != right
            )
            raise RuntimeError(
                f"BF16 boundary {name!r} differs at byte {mismatch}: "
                f"emulator={actual[mismatch]:#04x}, oracle={expected[mismatch]:#04x}"
            )
        boundary_hashes[name] = _sha256_bytes(actual)

    route_dump = (build_dir / "route_f32_sram_dump.bin").read_bytes()
    int_dump = (build_dir / "intsram_dump.bin").read_bytes()
    expected_route = (build_dir / "oracle/route_scores.f32.bin").read_bytes()
    expected_ids = (build_dir / "oracle/expert_ids.u32.bin").read_bytes()
    if route_dump[: len(expected_route)] != expected_route:
        raise RuntimeError("FP32 route-score bytes differ from the PyTorch oracle")
    if int_dump[: len(expected_ids)] != expected_ids:
        raise RuntimeError("runtime expert-ID bytes differ from the PyTorch oracle")
    routes = struct.unpack("<8f", route_dump[:32])
    ids = struct.unpack("<8I", int_dump[:32])
    if len(set(ids)) != 8 or any(index >= 128 for index in ids):
        raise RuntimeError("route assignment IDs are not eight unique experts in [0,128)")
    if not all(math.isfinite(value) and value >= 0 for value in routes):
        raise RuntimeError("route weights are non-finite or negative")
    if abs(sum(routes) - 1.0) > 2e-6:
        raise RuntimeError("route weights do not conserve unit probability")

    pre_hbm = bytearray((build_dir / "hbm_for_behave_sim.bin").read_bytes())
    pre_hbm.extend(b"\x00" * (HBM_SIZE - len(pre_hbm)))
    post_hbm = (build_dir / "post_hbm.bin").read_bytes()
    if len(post_hbm) != HBM_SIZE:
        raise RuntimeError("post-run HBM dump has the wrong sealed capacity")
    touched: set[int] = set()
    for role, cache_name in (("key", "k_cache"), ("value", "v_cache")):
        base = fixture["hbm_inputs"][cache_name]["base"]
        for start, end in _contiguous_ranges(
            fixture["compiler_receipt"]["append"][role], base
        ):
            touched.update(range(start, end))
    first_untouched_mismatch = next(
        (
            index
            for index, (before, after) in enumerate(zip(pre_hbm, post_hbm))
            if index not in touched and before != after
        ),
        None,
    )
    if first_untouched_mismatch is not None:
        raise RuntimeError(
            "KV append modified HBM outside its exact element/scale ranges at "
            f"byte {first_untouched_mismatch}"
        )
    for cache_name in ("k_cache", "v_cache"):
        entry = fixture["hbm_inputs"][cache_name]
        actual = post_hbm[entry["base"] : entry["base"] + entry["size"]]
        expected = (build_dir / "oracle" / f"{cache_name}_post.mx.bin").read_bytes()
        if actual != expected:
            raise RuntimeError(f"post-append {cache_name} bytes differ from oracle")

    with (build_dir / "op_stats.jsonl").open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    aggregate = [row for row in records if row.get("aggregate") is True]
    if len(aggregate) != 1:
        raise RuntimeError("op trace must contain exactly one aggregate record")
    summary = aggregate[0]
    if summary.get("total_hbm_rd") != summary.get(
        "total_hbm_issue_rd"
    ) or summary.get("total_hbm_wr") != summary.get("total_hbm_issue_wr"):
        raise RuntimeError("op trace issue-origin HBM traffic does not reconcile")
    assignment_count = len(ids)
    if assignment_count != fixture["assignment_expected"]:
        raise RuntimeError("route assignment conservation failed")

    artifacts = {
        name: _sha256_file(build_dir / name)
        for name in sorted(required)
    }
    execution = {
        "schema": SCHEMA,
        "fixture_receipt_sha256": _sha256_file(build_dir / "fixture_receipt.json"),
        "semantic_contract_sha256": fixture["semantic_contract_sha256"],
        "artifacts": artifacts,
        "boundary_sha256": boundary_hashes,
        "all_bf16_boundaries_byte_exact": True,
        "route_scores_fp32_byte_exact": True,
        "expert_ids_byte_exact": True,
        "route_probability_conserved": True,
        "route_assignment_conserved": True,
        "assignment_count": assignment_count,
        "expert_ids_topk_order": list(ids),
        "kv_cache_post_bytes_exact": True,
        "kv_untouched_hbm_bytes_conserved": True,
        "packedkv_attention_prefix_executed": True,
        "pytorch_oracle_valid": True,
        "python_oracle_runtime": fixture["python_oracle_runtime"],
        "transformers_abi": QWEN3_MOE_TRANSFORMERS_ABI,
        "transformers_semantics_source": "custom_pinned_semantic_fixture",
        "installed_transformers_api_invoked": False,
        "runtime_dependencies": invocation["runtime_dependencies"],
        "compiler_generated_binary_emulator_parity": True,
        "tiny_single_layer_transaction_parity": True,
        "execution_complete": True,
        "full_model_compiler_valid": False,
        "target_geometry_valid": False,
        "rtl_valid": False,
        "timing_calibrated": False,
        "publication_rankable": False,
    }
    execution["receipt_sha256"] = _sha256_bytes(_canonical_bytes(execution))
    _atomic_json(build_dir / "execution_receipt.json", execution)
    return execution


def validate_fixture_receipt(build_dir: Path) -> dict[str, Any]:
    """Reject altered/incomplete programs, inputs, sources, or oracle files."""

    build_dir = Path(build_dir).resolve()
    path = build_dir / "fixture_receipt.json"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("fixture receipt is absent or unreadable") from error
    recorded_hash = receipt.pop("receipt_sha256", None)
    if recorded_hash != _sha256_bytes(_canonical_bytes(receipt)):
        raise RuntimeError("fixture receipt content hash mismatch")
    receipt["receipt_sha256"] = recorded_hash
    if receipt.get("schema") != SCHEMA or receipt.get("fixture_complete") is not True:
        raise RuntimeError("fixture receipt is incomplete or has the wrong schema")
    if receipt.get("source_sha256") != _source_receipt():
        raise RuntimeError("compiler/emulator source identity differs from the fixture")
    if receipt.get("python_oracle_runtime") != _python_oracle_runtime():
        raise RuntimeError("Python oracle runtime identity differs from the fixture")
    if (
        receipt.get("transformers_semantics_source")
        != "custom_pinned_semantic_fixture"
        or receipt.get("installed_transformers_api_invoked") is not False
    ):
        raise RuntimeError("Transformers semantic-fixture provenance is invalid")
    for name, expected in receipt.get("artifacts", {}).items():
        artifact = build_dir / name
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise RuntimeError(f"fixture artifact hash mismatch for {name}")
    for boundary, expected in receipt["expected_boundary_sha256"].items():
        artifact = build_dir / "oracle" / f"{boundary}.bf16.bin"
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise RuntimeError(f"oracle boundary hash mismatch for {boundary}")
    if _sha256_file(build_dir / "oracle/route_scores.f32.bin") != receipt[
        "route_scores_sha256"
    ]:
        raise RuntimeError("route-score oracle hash mismatch")
    if _sha256_file(build_dir / "oracle/expert_ids.u32.bin") != receipt[
        "expert_ids_sha256"
    ]:
        raise RuntimeError("expert-ID oracle hash mismatch")
    for cache_name in ("k_cache", "v_cache"):
        artifact = build_dir / "oracle" / f"{cache_name}_post.mx.bin"
        if _sha256_file(artifact) != receipt["cache_post_sha256"][cache_name]:
            raise RuntimeError(f"cache oracle hash mismatch for {cache_name}")
    return receipt


def validate_execution_receipt(build_dir: Path) -> dict[str, Any]:
    """Revalidate a completed receipt and every output/trace digest."""

    build_dir = Path(build_dir).resolve()
    path = build_dir / "execution_receipt.json"
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("execution receipt is absent or unreadable") from error
    recorded_hash = receipt.pop("receipt_sha256", None)
    if recorded_hash != _sha256_bytes(_canonical_bytes(receipt)):
        raise RuntimeError("execution receipt content hash mismatch")
    receipt["receipt_sha256"] = recorded_hash
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("execution_complete") is not True
        or receipt.get("tiny_single_layer_transaction_parity") is not True
    ):
        raise RuntimeError("execution receipt is incomplete or has the wrong schema")
    fixture = validate_fixture_receipt(build_dir)
    invocation = _validate_execution_invocation(build_dir)
    if receipt.get("semantic_contract_sha256") != fixture.get(
        "semantic_contract_sha256"
    ):
        raise RuntimeError("execution receipt is bound to a different fixture")
    if receipt.get("runtime_dependencies") != invocation.get(
        "runtime_dependencies"
    ):
        raise RuntimeError("execution receipt runtime dependencies differ")
    for name, expected in receipt.get("artifacts", {}).items():
        artifact = build_dir / name
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise RuntimeError(f"execution artifact hash mismatch for {name}")
    return receipt


def execute_fixture(build_dir: Path, emulator_binary: Path) -> dict[str, Any]:
    build_dir = Path(build_dir).resolve()
    validate_fixture_receipt(build_dir)
    emulator_binary = Path(emulator_binary).resolve()
    if not emulator_binary.is_file():
        raise FileNotFoundError(f"emulator binary is missing: {emulator_binary}")
    command = [
        str(emulator_binary),
        "--opcode",
        str(build_dir / "generated_machine_code.mem"),
        "--hbm",
        str(build_dir / "hbm_for_behave_sim.bin"),
        "--fpsram",
        str(build_dir / "fp_sram.bin"),
        "--intsram",
        str(build_dir / "int_sram.bin"),
        "--vram",
        str(build_dir / "vram_preload.bin"),
        "--settings",
        str(build_dir / "plena_settings.toml"),
        "--hbm-size",
        str(HBM_SIZE),
        "--hbm-dump",
        str(build_dir / "post_hbm.bin"),
        "--op-stats",
        str(build_dir / "op_stats.jsonl"),
        "--log-level",
        "warn",
    ]
    invocation = {
        "schema": SCHEMA,
        "command": command,
        "emulator_binary": str(emulator_binary),
        "emulator_binary_sha256": _sha256_file(emulator_binary),
        "runtime_dependencies": _seal_runtime_dependencies(emulator_binary),
        "fixture_receipt_sha256": _sha256_file(
            build_dir / "fixture_receipt.json"
        ),
    }
    invocation["invocation_sha256"] = _sha256_bytes(_canonical_bytes(invocation))
    _atomic_json(build_dir / "execution_invocation.json", invocation)
    for name in (
        "vram_dump.bin",
        "route_f32_sram_dump.bin",
        "intsram_dump.bin",
        "post_hbm.bin",
        "op_stats.jsonl",
    ):
        (build_dir / name).unlink(missing_ok=True)
    result = subprocess.run(
        command,
        cwd=build_dir,
        env={
            **os.environ,
            "RUST_BACKTRACE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_WAIT_POLICY": "PASSIVE",
        },
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    _atomic_write(build_dir / "emulator_stdout.log", result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"transactional emulator failed with exit {result.returncode}; "
            f"see {build_dir / 'emulator_stdout.log'}"
        )
    return validate_execution(build_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--emulator", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        receipt = validate_execution(args.build_dir)
    else:
        if args.emulator is None:
            parser.error("--emulator is required unless --validate-only is used")
        build_fixture(args.build_dir)
        receipt = execute_fixture(args.build_dir, args.emulator)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
