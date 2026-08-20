"""Four-token Rust proof for Kimi K3 compressed-MLA caching.

The default uses all 96 MLA heads and Kimi's real compressed/head dimensions.
Only the projection input rank is compacted to 64.  Persistent HBM stores 512
latent values plus 64 rotated-key values per token.  Reconstructed K/V reuse a
single-head scratch pair; allocating a persistent expanded 96-head cache is an
explicit test failure.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from compiler.aten.kimi3.blocks import (
    MlaBlockShape,
    MlaBlockWeights,
    allocate_mla_decode_cache,
    emit_mla_residual_block,
)
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import setup_hw
from transactional_emulator.testbench.aten.golden import (
    golden_flash_attention_mha_single_block,
)
from transactional_emulator.testbench.emulator_runner import (
    compare_emulator_output,
    run_emulator,
)
from transactional_emulator.testbench.gpt_oss_testkit import (
    _comparison_params_for,
)
from transactional_emulator.testbench.layout_utils import (
    infer_hbm_tensor_layouts,
    prestage_bf16_vram_matrix,
    read_bf16_vram_matrix,
)
from transactional_emulator.testbench.models.kimi3.connected_blocks_test import (
    TensorSet,
    _allocate_fp_constants,
    _bf16,
    _bf16_layout,
    _exact,
    _linear,
    _register_weight,
    _rms,
    _set_matrix_kv_plain_bf16,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


MLEN = 64
BLEN = 4
TOKENS = 4
HIDDEN = 64
Q_LORA = 64
KV_LORA = 512
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128
DEFAULT_HEADS = 96


def _golden_step(
    hidden: torch.Tensor,
    tensors: TensorSet,
    *,
    heads: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    compressed_history: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    mixer = _rms(hidden)
    q_latent = _rms(_linear(mixer, tensors.values["W_MLA_Q_A"]))
    q_all = _linear(q_latent, tensors.values["W_MLA_Q_B"])
    compressed = _linear(mixer, tensors.values["W_MLA_KV_A"])
    kv_latent = _rms(compressed[:, :KV_LORA])
    k_rope = compressed[:, KV_LORA:]
    k_rope_rot = _linear(k_rope, tensors.values["W_MLA_K_ROTATE"])
    k_rope = _bf16(_bf16(k_rope * cos) + _bf16(k_rope_rot * sin))
    compressed_row = torch.cat((kv_latent, k_rope), dim=-1).to(torch.bfloat16)
    compressed_history.append(compressed_row)
    history = torch.cat(compressed_history, dim=0)
    history_latent = history[:, :KV_LORA]
    history_rope = history[:, KV_LORA:]

    outputs = []
    kv_weight = tensors.values["W_MLA_KV_B"]
    for head in range(heads):
        q_start = head * (QK_NOPE + QK_ROPE)
        q_head = q_all[:, q_start : q_start + QK_NOPE + QK_ROPE]
        q_rope = q_head[:, QK_NOPE:]
        q_rope_rot = _linear(q_rope, tensors.values["W_MLA_Q_ROTATE"])
        q_rope = _bf16(_bf16(q_rope * cos) + _bf16(q_rope_rot * sin))
        q_head = torch.cat((q_head[:, :QK_NOPE], q_rope), dim=-1)
        kv_start = head * (QK_NOPE + V_HEAD)
        kv_head = _linear(
            history_latent,
            kv_weight[:, kv_start : kv_start + QK_NOPE + V_HEAD],
        )
        key = torch.cat((kv_head[:, :QK_NOPE], history_rope), dim=-1)
        value = kv_head[:, QK_NOPE:]
        outputs.append(
            golden_flash_attention_mha_single_block(
                q_head,
                key,
                value.float(),
                (QK_NOPE + QK_ROPE) ** -0.5,
            )
        )
    attention = torch.cat(outputs, dim=-1)
    return _linear(attention, tensors.values["W_MLA_OUT"]), compressed_row


def _rotate_half_matrix(width: int) -> torch.Tensor:
    if width <= 0 or width % 2:
        raise ValueError(f"rotate-half width must be a positive even number, got {width}")
    rotate = torch.zeros(width, width, dtype=torch.bfloat16)
    half = width // 2
    for index in range(half):
        rotate[index + half, index] = -1.0
        rotate[index, index + half] = 1.0
    return rotate


def _rope_tables(tokens: int, width: int, theta: float = 10_000.0) -> tuple[torch.Tensor, torch.Tensor]:
    half = width // 2
    frequencies = 1.0 / (theta ** (torch.arange(half).float() / half))
    angles = torch.outer(torch.arange(tokens).float(), frequencies)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    return (
        torch.cat((cos_half, cos_half), dim=-1).to(torch.bfloat16),
        torch.cat((sin_half, sin_half), dim=-1).to(torch.bfloat16),
    )


def build_and_run(
    build_dir: Path,
    *,
    heads: int = DEFAULT_HEADS,
    tokens: int = TOKENS,
    seed: int = 47,
) -> dict[str, object]:
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    build_dir.mkdir(parents=True, exist_ok=True)
    hw = setup_hw(
        argparse.Namespace(mlen=MLEN, vlen=None, blen=BLEN, hlen=None),
        build_dir,
    )
    _set_matrix_kv_plain_bf16()
    torch.manual_seed(seed)

    prog = PlenaCompiler(
        mlen=MLEN,
        blen=BLEN,
        real_data_ratio=hw.real_data_ratio,
        compact_matrix_loops=True,
    )
    shape = MlaBlockShape(
        hidden=HIDDEN,
        q_lora=Q_LORA,
        kv_lora=KV_LORA,
        qk_nope=QK_NOPE,
        qk_rope=QK_ROPE,
        v_head=V_HEAD,
        heads=heads,
    )
    cache = allocate_mla_decode_cache(
        prog,
        shape=shape,
        max_tokens=tokens,
    )
    cache.assert_hbm_contract(prog)
    if len(cache.persistent_backings) != 1:
        raise AssertionError("MLA must have exactly one persistent compressed backing")
    if any("head" in backing.name for backing in cache.persistent_backings):
        raise AssertionError("per-head persistent MLA cache allocation is forbidden")
    scratch_hbm_bytes = sum(backing.hbm_size for backing in cache.scratch_backings)
    expected_scratch_hbm_bytes = cache.reconstructed_k.byte_capacity + cache.reconstructed_v.byte_capacity
    if scratch_hbm_bytes != expected_scratch_hbm_bytes:
        raise AssertionError("MLA scratch HBM accounting must cover exactly one K/V head")

    tensors = TensorSet(values={}, bf16_names=set())
    rotate_half = _rotate_half_matrix(QK_ROPE)
    weights = MlaBlockWeights(
        q_a=_register_weight(
            prog,
            tensors,
            "W_MLA_Q_A",
            _exact((HIDDEN, Q_LORA), 1, 1, 1 / 64),
        ),
        q_b=_register_weight(
            prog,
            tensors,
            "W_MLA_Q_B",
            _exact((Q_LORA, heads * (QK_NOPE + QK_ROPE)), 2, 3, 1 / 64),
        ),
        kv_a=_register_weight(
            prog,
            tensors,
            "W_MLA_KV_A",
            _exact((HIDDEN, KV_LORA + QK_ROPE), 3, 5, 1 / 64),
        ),
        kv_b=_register_weight(
            prog,
            tensors,
            "W_MLA_KV_B",
            _exact((KV_LORA, heads * (QK_NOPE + V_HEAD)), 4, 7, 1 / 64),
        ),
        out=_register_weight(
            prog,
            tensors,
            "W_MLA_OUT",
            _exact((heads * V_HEAD, HIDDEN), 5, 9, 1 / 64),
        ),
        q_rope_rotate=_register_weight(
            prog,
            tensors,
            "W_MLA_Q_ROTATE",
            rotate_half,
            bf16=True,
        ),
        k_rope_rotate=_register_weight(
            prog,
            tensors,
            "W_MLA_K_ROTATE",
            rotate_half,
            bf16=True,
        ),
        gate=None,
    )
    for backing in cache.all_backings:
        tensors.add(
            backing.name,
            torch.zeros(backing.physical_shape, dtype=torch.bfloat16),
            bf16=True,
        )

    norms, _moe_constants, fp_preload = _allocate_fp_constants(prog)
    # The shared connected fixture uses MLEN-wide toy MLA dimensions.  This
    # test keeps Kimi's real 512-wide latent RMSNorm and 192-wide Q/K heads, so
    # overwrite the two shape-dependent constants with their real values. MLA
    # uses ordinary M_TMM, which has no packed-BMM 0.25 factor; unlike fused GQA,
    # its softmax constant is the direct 1/sqrt(d) scale.
    fp_preload[norms.kv_reciprocal_hidden] = 1.0 / KV_LORA
    fp_preload[1] = (QK_NOPE + QK_ROPE) ** -0.5
    hidden_values = [(torch.randn(1, HIDDEN) * 0.1 + token / 64.0).to(torch.bfloat16) for token in range(tokens)]
    rope_cos_values, rope_sin_values = _rope_tables(tokens, QK_ROPE)
    preload_tiles = tokens * 3
    vram_preload = torch.zeros(preload_tiles * MLEN * HIDDEN, dtype=torch.bfloat16)
    hidden_vars = [
        prestage_bf16_vram_matrix(
            prog=prog,
            name=f"MLA_HIDDEN_TOKEN_{token}",
            tensor=value,
            vram_addr=token * MLEN * HIDDEN,
            physical_shape=(MLEN, HIDDEN),
            vram_preload=vram_preload,
        )
        for token, value in enumerate(hidden_values)
    ]
    cos_vars = [
        prestage_bf16_vram_matrix(
            prog=prog,
            name=f"MLA_ROPE_COS_TOKEN_{token}",
            tensor=rope_cos_values[token : token + 1],
            vram_addr=(tokens + token) * MLEN * HIDDEN,
            physical_shape=(MLEN, QK_ROPE),
            vram_preload=vram_preload,
        )
        for token in range(tokens)
    ]
    sin_vars = [
        prestage_bf16_vram_matrix(
            prog=prog,
            name=f"MLA_ROPE_SIN_TOKEN_{token}",
            tensor=rope_sin_values[token : token + 1],
            vram_addr=(2 * tokens + token) * MLEN * HIDDEN,
            physical_shape=(MLEN, QK_ROPE),
            vram_preload=vram_preload,
        )
        for token in range(tokens)
    ]

    outputs = prog.alloc(
        "MLA_FOUR_TOKEN_OUTPUTS",
        rows=tokens,
        cols=HIDDEN,
        strict=False,
        physical_shape=(MLEN, HIDDEN),
    )
    golden_outputs = []
    compressed_history: list[torch.Tensor] = []
    for token, hidden in enumerate(hidden_vars):
        result = emit_mla_residual_block(
            prog,
            hidden,
            shape=shape,
            weights=weights,
            cos=cos_vars[token],
            sin=sin_vars[token],
            norms=norms,
            rows=1,
            name=f"kimi_mla_token{token}",
            add_residual=False,
            cache=cache,
            token_index=token,
        )
        prog.vram_copy_region(
            outputs,
            result,
            num_rows=1,
            num_cols=HIDDEN,
            dst_row_offset=token,
        )
        prog.free_tensor(result)
        golden_output, _ = _golden_step(
            hidden_values[token],
            tensors,
            heads=heads,
            cos=rope_cos_values[token : token + 1],
            sin=rope_sin_values[token : token + 1],
            compressed_history=compressed_history,
        )
        golden_outputs.append(golden_output)
    golden = torch.cat(golden_outputs, dim=0)

    compressed_readback = prog.load_batch(
        cache.compressed.prefix(tokens),
        name="MLA_COMPRESSED_CACHE_READBACK",
        storage_precision=2,
        hbm_precision=1,
    )
    scratch_k_readback = prog.load_batch(
        cache.reconstructed_k.prefix(tokens),
        name="MLA_RECONSTRUCTED_K_SCRATCH_READBACK",
        storage_precision=2,
        hbm_precision=1,
    )
    scratch_v_readback = prog.load_batch(
        cache.reconstructed_v.prefix(tokens),
        name="MLA_RECONSTRUCTED_V_SCRATCH_READBACK",
        storage_precision=2,
        hbm_precision=1,
    )
    assembly = prog.compile()
    cache.assert_hbm_contract(prog)
    if assembly.count("MLA_RECONSTRUCTED_HEAD_TILE") != tokens * heads:
        raise AssertionError("each token/head must reconstruct exactly one temporary tile")
    if "DECODE_CACHE_APPEND kimi_mla_cache_reconstructed" in assembly:
        raise AssertionError("reconstructed K/V must never be appended as persistent history")

    cache_like_hbm_objects = {
        name: backing
        for name, backing in prog._compiler._inputs.items()
        if "cache" in name.lower() or "scratch" in name.lower()
    }
    expected_cache_objects = {backing.name for backing in cache.all_backings}
    if set(cache_like_hbm_objects) != expected_cache_objects:
        raise AssertionError(
            "unexpected cache/scratch HBM allocation: "
            f"expected={sorted(expected_cache_objects)}, "
            f"actual={sorted(cache_like_hbm_objects)}"
        )
    expanded_widths = {
        heads * shape.qk_head,
        heads * shape.v_head,
        heads * (shape.qk_head + shape.v_head),
    }
    forbidden_expanded_hbm_objects = [
        name
        for name, backing in cache_like_hbm_objects.items()
        if heads > 1 and name != cache.compressed.backing.name and backing.shape[1] in expanded_widths
    ]
    if forbidden_expanded_hbm_objects:
        raise AssertionError(f"expanded all-head K/V cache found in HBM: {forbidden_expanded_hbm_objects}")

    input_tensors = tensors.values
    layouts = infer_hbm_tensor_layouts(input_tensors)
    for name in tensors.bf16_names:
        layouts[name] = _bf16_layout(input_tensors[name])
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    create_sim_env(
        input_tensors,
        assembly,
        {"original_output": golden},
        fp_preload=fp_preload,
        int_preload=[0] * 16,
        build_dir=str(build_dir),
        vram_preload=vram_preload,
        tensor_layouts=layouts,
    )
    create_mem_for_sim(
        data_size=MLEN,
        mode="behave_sim",
        asm="kimi3_mla_cache_four_token",
        specified_data_order=sorted(input_tensors, key=hbm_addrs.__getitem__),
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=layouts,
        hbm_addrs=hbm_addrs,
    )
    params = _comparison_params_for(
        outputs,
        rows=tokens,
        hidden=HIDDEN,
        mlen=MLEN,
        golden=golden,
    )
    params.update({"atol": 0.03, "rtol": 0.04, "min_allclose_match_rate": 100.0})
    (build_dir / "comparison_params.json").write_text(json.dumps(params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(assembly)
    hbm_size = math.ceil(prog._next_hbm_addr / 64) * 64
    (build_dir / "hbm_size.txt").write_text(f"{hbm_size}\n")

    metrics = run_emulator(build_dir, stage_profile=True, dump_cwd=build_dir)
    results, _ = compare_emulator_output(build_dir, verbose=False)
    if float(results.get("allclose_match_rate", 0.0)) < 100.0:
        raise AssertionError(f"four-token compressed MLA output mismatch: {results}")

    actual_cache = read_bf16_vram_matrix(
        build_dir / "vram_dump.bin",
        address=prog.get_vram_addr(compressed_readback.name),
        rows=tokens,
        width=shape.kv_a_width,
        physical_rows=compressed_readback.physical_shape[0],
        mlen=MLEN,
    )
    expected_cache = torch.cat(compressed_history, dim=0)
    cache_max_abs = float((actual_cache.float() - expected_cache.float()).abs().max())
    cache_close = torch.isclose(actual_cache, expected_cache, atol=0.01, rtol=0.01)
    cache_match_rate = float(cache_close.float().mean().item() * 100.0)
    if cache_match_rate < 100.0:
        raise AssertionError(f"compressed MLA cache mismatch: max_abs={cache_max_abs}, allclose={cache_match_rate}%")

    history = expected_cache
    last_head_offset = (heads - 1) * (QK_NOPE + V_HEAD)
    last_head_kv = _linear(
        history[:, :KV_LORA],
        tensors.values["W_MLA_KV_B"][:, last_head_offset : last_head_offset + QK_NOPE + V_HEAD],
    )
    expected_scratch_k = torch.cat((last_head_kv[:, :QK_NOPE], history[:, KV_LORA:]), dim=-1)
    expected_scratch_v = last_head_kv[:, QK_NOPE:]
    scratch_checks = {}
    for label, readback, expected in (
        ("k", scratch_k_readback, expected_scratch_k),
        ("v", scratch_v_readback, expected_scratch_v),
    ):
        actual = read_bf16_vram_matrix(
            build_dir / "vram_dump.bin",
            address=prog.get_vram_addr(readback.name),
            rows=tokens,
            width=expected.shape[1],
            physical_rows=readback.physical_shape[0],
            mlen=MLEN,
        )
        max_abs = float((actual.float() - expected.float()).abs().max())
        match_rate = float(torch.isclose(actual, expected, atol=0.02, rtol=0.02).float().mean().item() * 100.0)
        if match_rate < 100.0:
            raise AssertionError(
                f"reconstructed {label.upper()} scratch mismatch: max_abs={max_abs}, allclose={match_rate}%"
            )
        scratch_checks[label] = {
            "max_abs_error": max_abs,
            "allclose_match_rate": match_rate,
        }

    expansion_ratio = cache.theoretical_expanded_cache_bytes / cache.logical_persistent_bytes
    summary = {
        "tokens": tokens,
        "heads": heads,
        "compressed_width": shape.kv_a_width,
        "qk_head_width": shape.qk_head,
        "value_head_width": shape.v_head,
        "sim_latency_ns": metrics.get("sim_latency_ns"),
        "output_max_abs_error": results.get("max_error"),
        "allclose_match_rate": results.get("allclose_match_rate"),
        "compressed_cache_max_abs_error": cache_max_abs,
        "compressed_cache_allclose_match_rate": cache_match_rate,
        "reconstructed_scratch_checks": scratch_checks,
        "logical_persistent_cache_bytes": cache.logical_persistent_bytes,
        "theoretical_expanded_all_head_cache_bytes": cache.theoretical_expanded_cache_bytes,
        "expanded_to_compressed_ratio": expansion_ratio,
        "one_head_scratch_hbm_bytes": scratch_hbm_bytes,
        "persistent_hbm_objects": [backing.name for backing in cache.persistent_backings],
        "reused_scratch_hbm_objects": [backing.name for backing in cache.scratch_backings],
        "forbidden_expanded_hbm_objects": forbidden_expanded_hbm_objects,
        "hbm_cache_contract": "compressed-latent-plus-one-head-scratch-only",
        "hbm_bytes": hbm_size,
    }
    (build_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("transactional_emulator/testbench/build/kimi3_mla_cache_four_token"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_and_run(
                args.build_dir.expanduser().resolve(),
                heads=args.heads,
                tokens=args.tokens,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
