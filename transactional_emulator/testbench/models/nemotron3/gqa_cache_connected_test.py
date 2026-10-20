"""Rust proof for Nemotron 3's persistent GQA K/V cache.

The projection side is compact (hidden=64) to keep the test reproducible, but
the attention geometry is Nemotron's real 32 query heads, two K/V heads and
128-wide heads.  The same fixture executes either incremental decode or one
causal multi-row prefill in a single Rust emulator invocation.  Every token
output and every final cache row is checked against the same sequential CPU
reference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from compiler.aten.nemotron3.blocks import (
    NemotronAttentionShape,
    NemotronAttentionWeights,
    allocate_nemotron_gqa_decode_cache,
    emit_nemotron_attention_block,
)
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.configurable import setup_hw
from transactional_emulator.testbench.sliced_layer_test_builder import (
    _active_precision_settings,
    _flash_attn_ref,
)
from transactional_emulator.testbench.emulator_runner import (
    compare_emulator_output,
    run_emulator,
)
from transactional_emulator.testbench.layout_utils import (
    infer_hbm_tensor_layouts,
    prestage_bf16_vram_matrix,
    read_bf16_vram_matrix,
)
from transactional_emulator.testbench.models.kimi3.connected_blocks_test import (
    TensorSet,
    _bf16_layout,
    _exact,
    _linear,
    _register_weight,
    _set_matrix_kv_plain_bf16,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


MLEN = 64
BLEN = 4
TOKENS = 4
HIDDEN = 64
QUERY_HEADS = 32
KV_HEADS = 2
HEAD_DIM = 128


def _golden_step(
    hidden: torch.Tensor,
    tensors: TensorSet,
    key_history: list[list[torch.Tensor]],
    value_history: list[list[torch.Tensor]],
    *,
    precision,
) -> torch.Tensor:
    q_all = _linear(hidden, tensors.values["W_GQA_Q"])
    k_all = _linear(hidden, tensors.values["W_GQA_K"])
    v_all = _linear(hidden, tensors.values["W_GQA_V"])
    for head in range(KV_HEADS):
        start = head * HEAD_DIM
        key_history[head].append(k_all[:, start : start + HEAD_DIM])
        value_history[head].append(v_all[:, start : start + HEAD_DIM])

    heads_per_kv = QUERY_HEADS // KV_HEADS
    outputs = []
    for q_head in range(QUERY_HEADS):
        q_start = q_head * HEAD_DIM
        kv_head = q_head // heads_per_kv
        outputs.append(
            _flash_attn_ref(
                q_all[:, q_start : q_start + HEAD_DIM],
                torch.cat(key_history[kv_head], dim=0),
                torch.cat(value_history[kv_head], dim=0).float(),
                HEAD_DIM**-0.5,
                precision=precision,
            )
        )
    return _linear(torch.cat(outputs, dim=-1), tensors.values["W_GQA_OUT"])


def build_and_run(
    build_dir: Path,
    *,
    tokens: int = TOKENS,
    mode: str = "decode",
    seed: int = 31,
) -> dict[str, object]:
    if tokens <= 0:
        raise ValueError(f"tokens must be positive, got {tokens}")
    if mode not in {"decode", "prefill"}:
        raise ValueError(f"mode must be 'decode' or 'prefill', got {mode!r}")
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
    prog.fp_var("zero", 1)
    prog.fp_var("attention_scale", 1)
    prog.fp_var("negative_infinity", 1)
    prog.fp_var("online_softmax_workspace", 253)
    fp_preload = [0.0] * 256
    fp_preload[1] = (HEAD_DIM**-0.5) / 0.25
    fp_preload[2] = float("-inf")

    shape = NemotronAttentionShape(
        hidden=HIDDEN,
        query_heads=QUERY_HEADS,
        kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )
    cache = allocate_nemotron_gqa_decode_cache(
        prog,
        shape=shape,
        max_tokens=tokens,
    )
    tensors = TensorSet(values={}, bf16_names=set())
    weights = NemotronAttentionWeights(
        q=_register_weight(
            prog,
            tensors,
            "W_GQA_Q",
            _exact((HIDDEN, QUERY_HEADS * HEAD_DIM), 1, 1, 1 / 64),
        ),
        k=_register_weight(
            prog,
            tensors,
            "W_GQA_K",
            _exact((HIDDEN, KV_HEADS * HEAD_DIM), 2, 3, 1 / 64),
        ),
        v=_register_weight(
            prog,
            tensors,
            "W_GQA_V",
            _exact((HIDDEN, KV_HEADS * HEAD_DIM), 3, 5, 1 / 64),
        ),
        out=_register_weight(
            prog,
            tensors,
            "W_GQA_OUT",
            _exact((QUERY_HEADS * HEAD_DIM, HIDDEN), 4, 7, 1 / 64),
        ),
    )
    for backing in cache.backings:
        tensors.add(
            backing.name,
            torch.zeros(backing.physical_shape, dtype=torch.bfloat16),
            bf16=True,
        )

    hidden_values = [(torch.randn(1, HIDDEN) * 0.125 + token / 32.0).to(torch.bfloat16) for token in range(tokens)]
    physical_rows = math.ceil(tokens / MLEN) * MLEN
    if mode == "decode":
        vram_preload = torch.zeros(tokens * MLEN * HIDDEN, dtype=torch.bfloat16)
        hidden_vars = [
            prestage_bf16_vram_matrix(
                prog=prog,
                name=f"HIDDEN_TOKEN_{token}",
                tensor=value,
                vram_addr=token * MLEN * HIDDEN,
                physical_shape=(MLEN, HIDDEN),
                vram_preload=vram_preload,
            )
            for token, value in enumerate(hidden_values)
        ]
    else:
        vram_preload = torch.zeros(physical_rows * HIDDEN, dtype=torch.bfloat16)
        hidden_vars = [
            prestage_bf16_vram_matrix(
                prog=prog,
                name="HIDDEN_PROMPT",
                tensor=torch.cat(hidden_values, dim=0),
                vram_addr=0,
                physical_shape=(physical_rows, HIDDEN),
                vram_preload=vram_preload,
            )
        ]

    outputs = prog.alloc(
        "GQA_TOKEN_OUTPUTS",
        rows=tokens,
        cols=HIDDEN,
        strict=False,
        physical_shape=(physical_rows, HIDDEN),
    )
    golden_outputs = []
    key_history: list[list[torch.Tensor]] = [[] for _ in range(KV_HEADS)]
    value_history: list[list[torch.Tensor]] = [[] for _ in range(KV_HEADS)]
    if mode == "decode":
        for token, hidden in enumerate(hidden_vars):
            result = emit_nemotron_attention_block(
                prog,
                hidden,
                shape=shape,
                weights=weights,
                rows=1,
                name=f"nemotron_gqa_token{token}",
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
    else:
        result = emit_nemotron_attention_block(
            prog,
            hidden_vars[0],
            shape=shape,
            weights=weights,
            rows=tokens,
            name="nemotron_gqa_prefill",
            cache=cache,
            token_index=0,
            causal=True,
        )
        prog.vram_copy_region(
            outputs,
            result,
            num_rows=tokens,
            num_cols=HIDDEN,
        )
        prog.free_tensor(result)
    precision = _active_precision_settings()
    for token in range(tokens):
        golden_outputs.append(
            _golden_step(
                hidden_values[token],
                tensors,
                key_history,
                value_history,
                precision=precision,
            )
        )
    golden = torch.cat(golden_outputs, dim=0)

    readbacks = []
    for cache_tensor in (*cache.keys, *cache.values):
        readbacks.append(
            (
                cache_tensor,
                prog.load_batch(
                    cache_tensor.prefix(tokens),
                    name=f"{cache_tensor.backing.name}_readback",
                    storage_precision=2,
                    hbm_precision=1,
                ),
            )
        )

    assembly = prog.compile()
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
        asm=f"nemotron3_gqa_cache_{mode}_{tokens}",
        specified_data_order=sorted(input_tensors, key=hbm_addrs.__getitem__),
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=layouts,
        hbm_addrs=hbm_addrs,
    )
    from transactional_emulator.testbench.gpt_oss_testkit import (
        _comparison_params_for,
    )

    params = _comparison_params_for(
        outputs,
        rows=tokens,
        hidden=HIDDEN,
        mlen=MLEN,
        golden=golden,
    )
    params.update({"atol": 0.02, "rtol": 0.03, "min_allclose_match_rate": 100.0})
    (build_dir / "comparison_params.json").write_text(json.dumps(params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(assembly)
    hbm_size = math.ceil(prog._next_hbm_addr / 64) * 64
    (build_dir / "hbm_size.txt").write_text(f"{hbm_size}\n")

    metrics = run_emulator(build_dir, stage_profile=True, dump_cwd=build_dir)
    results, _ = compare_emulator_output(build_dir, verbose=False)
    if float(results.get("allclose_match_rate", 0.0)) < 100.0:
        raise AssertionError(f"{mode} GQA output mismatch: {results}")

    cache_errors = {}
    vram_dump = build_dir / "vram_dump.bin"
    expected_histories = [torch.cat(history, dim=0) for history in (*key_history, *value_history)]
    for (cache_tensor, readback), expected in zip(
        readbacks,
        expected_histories,
        strict=True,
    ):
        actual = read_bf16_vram_matrix(
            vram_dump,
            address=prog.get_vram_addr(readback.name),
            rows=tokens,
            width=cache_tensor.width,
            physical_rows=readback.physical_shape[0],
            mlen=MLEN,
        )
        error = float((actual.float() - expected.float()).abs().max())
        cache_errors[cache_tensor.backing.name] = error
        if not torch.equal(actual, expected):
            raise AssertionError(f"GQA cache mismatch for {cache_tensor.backing.name}: max_abs={error}")

    summary = {
        "mode": mode,
        "tokens": tokens,
        "query_heads": QUERY_HEADS,
        "kv_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "sim_latency_ns": metrics.get("sim_latency_ns"),
        "output_max_abs_error": results.get("max_error"),
        "allclose_match_rate": results.get("allclose_match_rate"),
        "cache_max_abs_errors": cache_errors,
        "logical_cache_bytes": tokens * 2 * KV_HEADS * HEAD_DIM * 2,
        "allocated_cache_bytes_with_guard_rows": cache.persistent_bytes,
        "hbm_bytes": hbm_size,
    }
    (build_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("transactional_emulator/testbench/build/nemotron3_gqa_cache_four_token"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            build_and_run(
                args.build_dir.expanduser().resolve(),
                tokens=args.tokens,
                mode=args.mode,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
