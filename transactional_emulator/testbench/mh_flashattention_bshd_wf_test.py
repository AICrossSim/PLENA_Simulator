import json
import math
import sys
from pathlib import Path

import torch

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from compiler.sim_env_utils import create_mem_for_sim
from tile_tensor_program import TileTensorProgram
from tile_tensor_test_helper import build_input_feed, stage_input_tensor_for_stride_compare
from transactional_emulator.tools.create_sim_env import create_sim_env


def _first_scatter_group(prog: TileTensorProgram, tensor_like: object) -> object:
    tiles = getattr(tensor_like, "tiles", None)
    if not isinstance(tiles, dict) or not tiles:
        raise RuntimeError(f"Expected tensor-like object with tiles, got {type(tensor_like).__name__}")
    first_tile = tiles[sorted(tiles.keys())[0]]
    return prog.value_manager.map_tile_to_scatter_group(first_tile)


def build_program(
    mlen: int,
    blen: int,
    hlen: int,
    seq_len: int,
    head_count: int,
) -> tuple[TileTensorProgram, object, object]:
    group_heads = 4
    if mlen % hlen != 0:
        raise ValueError(f"Require mlen divisible by hlen, got mlen={mlen}, hlen={hlen}")
    if head_count <= 0:
        raise ValueError(f"head_count must be positive, got {head_count}")
    if head_count % group_heads != 0:
        raise ValueError(f"Current draft requires head_count divisible by {group_heads}, got {head_count}")

    batch_size = 1
    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=hlen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=4096,
    )

    in_q = prog.input("Q_IN", (batch_size, seq_len, head_count, hlen))
    in_k = prog.input("K_IN", (batch_size, seq_len, head_count, hlen))
    in_v = prog.input("V_IN", (batch_size, seq_len, head_count, hlen))
    out_buf = prog.input("OUT", (batch_size, seq_len, head_count, hlen))

    q = prog.tensor("Q", (batch_size, seq_len, head_count, hlen))
    k = prog.tensor("K", (batch_size, seq_len, head_count, hlen))
    v = prog.tensor("V", (batch_size, seq_len, head_count, hlen))
    o = prog.tensor("O", (batch_size, seq_len, head_count, hlen))
    prog.copy(in_q, q)
    prog.copy(in_k, k)
    prog.copy(in_v, v)

    scale_scalar = prog.constant(prog._auto_name("flash_scale_scalar"), 1.0 / math.sqrt(hlen))
    neg_inf_scalar = prog.constant(prog._auto_name("flash_neg_inf"), -1.0e4)
    scores_max = prog.alloc_fragment(prog._auto_name("scores_max"), (batch_size, group_heads, mlen))
    logsum = prog.alloc_fragment(prog._auto_name("logsum"), (batch_size, group_heads, mlen))
    scores_max_prev = prog.alloc_fragment(prog._auto_name("scores_max_prev"), (batch_size, 1, mlen))
    scores_scale = prog.alloc_fragment(prog._auto_name("scores_scale"), (batch_size, 1, mlen))
    scores_sum = prog.alloc_fragment(prog._auto_name("scores_sum"), (batch_size, 1, mlen))
    inv_l = prog.alloc_fragment(prog._auto_name("inv_l"), (batch_size, 1, mlen))
    for q_start in range(0, seq_len, mlen):
        q_end = q_start + mlen
        for group_start in range(0, head_count, group_heads):
            q_group = prog.alloc_fragment(prog._auto_name("Q_GROUP"), (batch_size, mlen, group_heads, hlen))
            score_group = prog.alloc_fragment(prog._auto_name("S_GROUP"), (batch_size, mlen, group_heads, mlen))
            out_group = prog.alloc_fragment(prog._auto_name("O_GROUP"), (batch_size, mlen, group_heads, hlen))
            pv_group = prog.alloc_fragment(prog._auto_name("PV_GROUP"), (batch_size, mlen, group_heads, hlen))

            prog.copy(q[0, q_start:q_end, group_start : group_start + group_heads, :], q_group)
            prog.clear(out_group)
            for local_head in range(group_heads):
                prog.fill(logsum[0, local_head, :], 0.0)
                prog.fill(scores_max[0, local_head, :], neg_inf_scalar)

            for kv_start in range(0, seq_len, mlen):
                kv_end = kv_start + mlen
                k_group = prog.alloc_fragment(prog._auto_name("K_GROUP"), (batch_size, mlen, group_heads, hlen))
                v_group = prog.alloc_fragment(prog._auto_name("V_GROUP"), (batch_size, mlen, group_heads, hlen))

                prog.copy(k[0, kv_start:kv_end, group_start : group_start + group_heads, :], k_group)
                prog.copy(v[0, kv_start:kv_end, group_start : group_start + group_heads, :], v_group)

                prog.matmul(q_group, k_group.T, score_group)
                for local_head in range(group_heads):
                    score_head = score_group[0, :, local_head : local_head + 1, :]
                    out_head = out_group[0, :, local_head : local_head + 1, :]

                    prog.row_op(score_head, scale_scalar, "mul", dim=-1)
                    prog.copy(scores_max[0, local_head, :], scores_max_prev[0, 0, :])
                    prog.fill(scores_max[0, local_head, :], neg_inf_scalar)
                    prog.row_op(score_head, op="reduce_max", out=scores_max[0, local_head, :], dim=-1)
                    prog.pure_fp_compute(
                        scores_max[0, local_head, :],
                        scores_max[0, local_head, :],
                        src2=scores_max_prev[0, 0, :],
                        control="max",
                        task_id=f"row_max_merge.q{q_start}.k{kv_start}.g{group_start}.h{local_head}",
                    )
                    prog.copy(scores_max_prev[0, 0, :], scores_scale[0, 0, :])
                    prog.pure_fp_compute(
                        scores_scale[0, 0, :],
                        scores_scale[0, 0, :],
                        src2=scores_max[0, local_head, :],
                        control="sub",
                        task_id=f"scores_scale_sub.q{q_start}.k{kv_start}.g{group_start}.h{local_head}",
                    )
                    prog.pure_fp_compute(
                        scores_scale[0, 0, :],
                        scores_scale[0, 0, :],
                        control="exp",
                        task_id=f"scores_scale_exp.q{q_start}.k{kv_start}.g{group_start}.h{local_head}",
                    )
                    prog.row_op(score_head, scores_max[0, local_head, :], "sub", dim=-1)
                    prog.row_op(score_head, op="exp", dim=-1)
                    prog.fill(scores_sum[0, 0, :], 0.0)
                    prog.row_op(score_head, op="reduce_sum", out=scores_sum[0, 0, :], dim=-1)
                    prog.pure_fp_compute(
                        logsum[0, local_head, :],
                        logsum[0, local_head, :],
                        src2=scores_scale[0, 0, :],
                        control="mul",
                        task_id=f"logsum_rescale.q{q_start}.k{kv_start}.g{group_start}.h{local_head}",
                    )
                    prog.pure_fp_compute(
                        logsum[0, local_head, :],
                        logsum[0, local_head, :],
                        src2=scores_sum[0, 0, :],
                        control="add",
                        task_id=f"logsum_add.q{q_start}.k{kv_start}.g{group_start}.h{local_head}",
                    )
                    prog.row_op(out_head, scores_scale[0, 0, :], "mul", dim=-1)

                prog.matmul(score_group, v_group, pv_group)
                prog.atomic_add(out_group, pv_group, out_group)

            for local_head in range(group_heads):
                out_head = out_group[0, :, local_head : local_head + 1, :]
                prog.pure_fp_compute(
                    logsum[0, local_head, :],
                    inv_l[0, 0, :],
                    control="reci",
                    task_id=f"inv_l.q{q_start}.g{group_start}.h{local_head}",
                )
                prog.row_op(out_head, inv_l[0, 0, :], "mul", dim=-1)

            prog.copy(out_group, o[0, q_start:q_end, group_start : group_start + group_heads, :])

    prog.copy(o, out_buf)
    return prog, o, out_buf


def build_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.einsum("bshd,bthd->bsht", q, k) * scale
    p = torch.softmax(s, dim=-1)
    o = torch.einsum("bsht,bthd->bshd", p, v)
    return o


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    hlen = 16
    seq_len = 128
    head_count = 8

    prog, out, out_buf = build_program(
        mlen=mlen,
        blen=blen,
        hlen=hlen,
        seq_len=seq_len,
        head_count=head_count,
    )
    comparison_params = stage_input_tensor_for_stride_compare(out_buf)
    staging_isa = comparison_params.pop("staging_isa", "")
    gen_code = prog.compile()
    if staging_isa:
        gen_code = gen_code.rstrip() + "\n\n" + staging_isa
    tensor_table = prog.get_tensor_table()

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    q_data = torch.randn(1, seq_len, head_count, hlen, dtype=torch.float32) * 0.5
    k_data = torch.randn(1, seq_len, head_count, hlen, dtype=torch.float32) * 0.5
    v_data = torch.randn(1, seq_len, head_count, hlen, dtype=torch.float32) * 0.5
    golden_o = build_golden(q=q_data, k=k_data, v=v_data)

    input_feed, input_order = build_input_feed(
        prog,
        {
            "Q_IN": q_data,
            "K_IN": k_data,
            "V_IN": v_data,
            "OUT": torch.zeros(1, seq_len, head_count, hlen, dtype=torch.float32),
        }
    )
    fp_preload = prog.build_fp_preload(min_size=32)
    golden_result = {"original_output": golden_o.reshape(seq_len, head_count * hlen)}

    create_sim_env(
        input_tensor=input_feed,
        generated_code=gen_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="mh_flashattention_bshd_wf_test",
        data=None,
        specified_data_order=input_order,
        build_path=build_dir,
    )

    with open(build_dir / "comparison_params.json", "w", encoding="utf-8") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "flashattention_syntax_draft_generated_asm_code.asm", "w", encoding="utf-8") as f:
        f.write(gen_code)

    if np is not None:
        np.save(
            build_dir / "flashattention_syntax_draft_golden_fp32.npy",
            golden_o.reshape(seq_len, head_count * hlen).detach().cpu().numpy().astype(np.float32),
        )
        np.save(
            build_dir / "flashattention_syntax_draft_golden_bf16.npy",
            golden_o.reshape(seq_len, head_count * hlen)
            .detach()
            .cpu()
            .to(torch.bfloat16)
            .to(torch.float32)
            .numpy()
            .astype(np.float32),
        )

    with open(build_dir / "flashattention_syntax_draft_fp_table.json", "w", encoding="utf-8") as f:
        json.dump(prog.get_fp_table(), f, indent=2)

    with open(build_dir / "flashattention_syntax_draft_table.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                name: {
                    "kind": meta["kind"],
                    "shape": list(meta["shape"]),
                    "tiles": {
                        f"{i},{j}": {
                            "tile_id": tile.tile_id,
                            "binding": prog.value_manager.bindings.get(tile.tile_id),
                            "scatter_binding": prog.value_manager.tile_scatter_bindings.get(tile.tile_id),
                            "tile_shape": list(tile.tile_shape),
                            "metadata": dict(tile.metadata),
                        }
                        for (i, j), tile in meta["tiles"].items()
                    },
                }
                for name, meta in tensor_table.items()
            },
            f,
            indent=2,
        )

    prog.write_operation_report(build_dir / "flashattention_syntax_draft_ops.txt")
    prog.write_tile_distribution_report(build_dir / "flashattention_syntax_draft_tile_report.txt")
