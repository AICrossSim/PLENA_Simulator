"""TileTensorProgram rewrite of `tilelang_kernels.attention`."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_TESTBENCH_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_TESTBENCH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tile_tensor_program import TileTensorProgram
from tile_tensor_kernel_programs.testbench_runner import emit_single_output_testbench


def build_flashattention_program(
    *,
    mlen: int,
    blen: int,
    batch_size: int,
    hlen: int,
    seq_len: int,
    head_count: int,
    causal: bool = True,
) -> tuple[TileTensorProgram, object, object]:
    group_heads = 4
    if mlen % hlen != 0:
        raise ValueError(f"Require mlen divisible by hlen, got mlen={mlen}, hlen={hlen}")
    if head_count % group_heads != 0:
        raise ValueError(f"Require head_count divisible by {group_heads}, got {head_count}")

    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=hlen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=4096,
    )

    q_in = prog.input("Q_IN", (batch_size, seq_len, head_count, hlen))
    k_in = prog.input("K_IN", (batch_size, seq_len, head_count, hlen))
    v_in = prog.input("V_IN", (batch_size, seq_len, head_count, hlen))
    out_buf = prog.input("OUT", (batch_size, seq_len, head_count, hlen))

    scale_scalar = prog.constant(prog._auto_name("flash_scale"), 1.0 / math.sqrt(hlen))
    neg_inf_scalar = prog.constant(prog._auto_name("neg_inf"), -1.0e4)
    zero_scalar = prog.constant(prog._auto_name("zero"), 0.0)
    scores_max = prog.alloc_fragment(prog._auto_name("scores_max"), (1, group_heads, mlen))
    logsum = prog.alloc_fragment(prog._auto_name("logsum"), (1, group_heads, mlen))
    scores_max_prev = prog.alloc_fragment(prog._auto_name("scores_max_prev"), (1, group_heads, mlen))
    scores_scale = prog.alloc_fragment(prog._auto_name("scores_scale"), (1, 1, mlen))
    scores_sum = prog.alloc_fragment(prog._auto_name("scores_sum"), (1, 1, mlen))
    mask_head = prog.alloc_fragment(prog._auto_name("mask_head"), (1, mlen, 1, mlen))

    if causal:
        with prog.parallel_region3d((mlen, 1, mlen), name="causal_mask") as (q_local, head_local, k_local):
            mask_head[0, q_local, head_local, k_local] = prog.if_then_else(
                q_local < k_local,
                neg_inf_scalar,
                zero_scalar,
            )

    q_block_count = seq_len // mlen
    group_block_count = head_count // group_heads
    for batch_index in range(batch_size):
        for q_block in prog.pipelined(q_block_count, num_stages=2):
            q_start = q_block * mlen
            q_end = q_start + mlen
            for group_block in prog.parallel(group_block_count):
                group_start = group_block * group_heads
                q_group = prog.alloc_fragment(prog._auto_name("Q_GROUP"), (1, mlen, group_heads, hlen))
                score_group = prog.alloc_fragment(prog._auto_name("S_GROUP"), (1, mlen, group_heads, mlen))
                out_group = prog.alloc_fragment(prog._auto_name("O_GROUP"), (1, mlen, group_heads, hlen))
                pv_group = prog.alloc_fragment(prog._auto_name("PV_GROUP"), (1, mlen, group_heads, hlen))

                prog.copy(
                    q_in[batch_index, q_start:q_end, group_start : group_start + group_heads, :],
                    q_group,
                )
                prog.clear(out_group)
                for local_head in prog.parallel(group_heads):
                    prog.fill(logsum[0, local_head, :], 0.0)
                    prog.fill(scores_max[0, local_head, :], neg_inf_scalar)

                for kv_block in prog.pipelined(q_block_count, num_stages=2):
                    kv_start = kv_block * mlen
                    kv_end = kv_start + mlen
                    if causal and kv_start >= q_end:
                        continue

                    k_group = prog.alloc_fragment(prog._auto_name("K_GROUP"), (1, mlen, group_heads, hlen))
                    v_group = prog.alloc_fragment(prog._auto_name("V_GROUP"), (1, mlen, group_heads, hlen))
                    prog.copy(
                        k_in[batch_index, kv_start:kv_end, group_start : group_start + group_heads, :],
                        k_group,
                    )
                    prog.copy(
                        v_in[batch_index, kv_start:kv_end, group_start : group_start + group_heads, :],
                        v_group,
                    )

                    prog.matmul(q_group, k_group.T, score_group)
                    prog.free_tensor_tile(k_group)
                    for local_head in prog.parallel(group_heads):
                        score_head = score_group[0, :, local_head : local_head + 1, :]
                        out_head = out_group[0, :, local_head : local_head + 1, :]
                        prog.row_op(score_head, scale_scalar, "mul", dim=-1)

                        if causal and kv_start < q_end and q_start < kv_end:
                            prog.atomic_add(score_head, mask_head, score_head)

                        prog.copy(scores_max[0, local_head, :], scores_max_prev[0, local_head, :])
                        prog.fill(scores_max[0, local_head, :], neg_inf_scalar)
                        prog.row_op(score_head, op="reduce_max", out=scores_max[0, local_head, :], dim=-1)

                    with prog.parallel_region2d((group_heads, mlen)) as (h, s):
                        scores_max[0, h, s] = prog.max(scores_max[0, h, s], scores_max_prev[0, h, s])

                    for local_head in prog.parallel(group_heads):
                        score_head = score_group[0, :, local_head : local_head + 1, :]
                        out_head = out_group[0, :, local_head : local_head + 1, :]
                        with prog.parallel_region2d((1, mlen)) as (_, s):
                            scores_scale[0, 0, s] = scores_max_prev[0, local_head, s] - scores_max[0, local_head, s]
                        with prog.parallel_region2d((1, mlen)) as (_, s):
                            scores_scale[0, 0, s] = prog.exp(scores_scale[0, 0, s])
                        prog.row_op(score_head, scores_max[0, local_head, :], "sub", dim=-1)
                        prog.row_op(score_head, op="exp", dim=-1)
                        prog.fill(scores_sum[0, 0, :], 0.0)
                        prog.row_op(score_head, op="reduce_sum", out=scores_sum[0, 0, :], dim=-1)
                        with prog.parallel_region2d((1, mlen)) as (_, s):
                            logsum[0, local_head, s] = logsum[0, local_head, s] * scores_scale[0, 0, s]+scores_sum[0, 0, s]
 
                        prog.row_op(out_head, scores_scale[0, 0, :], "mul", dim=-1)

                    prog.matmul(score_group, v_group, pv_group)
                    prog.free_tensor_tile(score_group)
                    prog.free_tensor_tile(v_group)
                    prog.atomic_add(out_group, pv_group, out_group)
                    prog.free_tensor_tile(pv_group)

                for local_head in prog.parallel(group_heads):
                    out_head = out_group[0, :, local_head : local_head + 1, :]
                    with prog.parallel_region2d((1, mlen)) as (_, s):
                        scores_scale[0, 0, s] = prog.reci(logsum[0, local_head, s])
                    prog.row_op(out_head, scores_scale[0, 0, :], "mul", dim=-1)

                prog.copy(
                    out_group,
                    out_buf[batch_index, q_start:q_end, group_start : group_start + group_heads, :],
                )
                prog.free_tensor_tile(q_group)
                prog.free_tensor_tile(out_group)

    if causal:
        prog.free_tensor_tile(mask_head)
    return prog, out_buf, out_buf


def build_flashattention_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.einsum("bshd,bthd->bsht", q, k) * scale
    if causal:
        q_len = q.shape[1]
        k_len = k.shape[1]
        mask = torch.triu(torch.ones((q_len, k_len), device=scores.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.view(1, q_len, 1, k_len), -1.0e4)
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bsht,bthd->bshd", probs, v)


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    batch_size = 2
    hlen = 16
    seq_len = 128*4
    head_count = 12
    causal = True

    prog, _, out_buf = build_flashattention_program(
        mlen=mlen,
        blen=blen,
        batch_size=batch_size,
        hlen=hlen,
        seq_len=seq_len,
        head_count=head_count,
        causal=causal,
    )
    q_data = torch.randn(batch_size, seq_len, head_count, hlen, dtype=torch.float32) * 0.5
    k_data = torch.randn(batch_size, seq_len, head_count, hlen, dtype=torch.float32) * 0.5
    v_data = torch.randn(batch_size, seq_len, head_count, hlen, dtype=torch.float32) * 0.5
    golden = build_flashattention_golden(q_data, k_data, v_data, causal=causal).reshape(
        batch_size * seq_len,
        head_count * hlen,
    )

    emit_single_output_testbench(
        prog=prog,
        out_buf=out_buf,
        input_tensors={
            "Q_IN": q_data,
            "K_IN": k_data,
            "V_IN": v_data,
            "OUT": torch.zeros(batch_size, seq_len, head_count, hlen, dtype=torch.float32),
        },
        golden_output=golden,
        asm_name="tile_tensor_kernel_attention",
        artifact_prefix="tile_tensor_kernel_attention",
        build_dir=_TESTBENCH_DIR / "build",
        fp_preload_min_size=32,
    )
