import json
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


def build_program(
    mlen: int,
    blen: int,
    hlen: int,
    seq_len: int,
    head_count: int,
    eps: float,
) -> tuple[TileTensorProgram, object, object]:
    group_heads = 4
    if head_count % group_heads != 0:
        raise ValueError(f"Current test requires head_count divisible by {group_heads}, got {head_count}")

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

    x_in = prog.input("X_IN", (batch_size, seq_len, head_count, hlen))
    weight_in = prog.input("WEIGHT_IN", (batch_size, seq_len, head_count, hlen))
    bias_in = prog.input("BIAS_IN", (batch_size, seq_len, head_count, hlen))
    out_buf = prog.input("OUT", (batch_size, seq_len, head_count, hlen))

    x = prog.tensor("X", (batch_size, seq_len, head_count, hlen))
    weight = prog.tensor("WEIGHT", (batch_size, seq_len, head_count, hlen))
    bias = prog.tensor("BIAS", (batch_size, seq_len, head_count, hlen))
    y = prog.tensor("Y", (batch_size, seq_len, head_count, hlen))
    prog.copy(x_in, x)
    prog.copy(weight_in, weight)
    prog.copy(bias_in, bias)

    recip_hlen = prog.constant(prog._auto_name("recip_hlen"), 1.0 / float(hlen), size=seq_len)
    eps_vec = prog.constant(prog._auto_name("eps"), float(eps), size=seq_len)

    mean = prog.alloc_fragment(prog._auto_name("mean"), (batch_size, 1, seq_len))
    var = prog.alloc_fragment(prog._auto_name("var"), (batch_size, 1, seq_len))
    inv_std = prog.alloc_fragment(prog._auto_name("inv_std"), (batch_size, 1, seq_len))

    for group_start in range(0, head_count, group_heads):
        x_group = prog.alloc_fragment(prog._auto_name("X_GROUP"), (batch_size, seq_len, group_heads, hlen))
        centered_group = prog.alloc_fragment(prog._auto_name("CENTERED_GROUP"), (batch_size, seq_len, group_heads, hlen))
        sq_group = prog.alloc_fragment(prog._auto_name("SQ_GROUP"), (batch_size, seq_len, group_heads, hlen))
        weight_group = prog.alloc_fragment(prog._auto_name("WEIGHT_GROUP"), (batch_size, seq_len, group_heads, hlen))
        bias_group = prog.alloc_fragment(prog._auto_name("BIAS_GROUP"), (batch_size, seq_len, group_heads, hlen))

        prog.copy(x[0, :, group_start : group_start + group_heads, :], x_group)
        prog.copy(weight[0, :, group_start : group_start + group_heads, :], weight_group)
        prog.copy(bias[0, :, group_start : group_start + group_heads, :], bias_group)
        prog.copy(x_group, centered_group)

        for local_head in range(group_heads):
            x_head = x_group[0, :, local_head : local_head + 1, :]
            centered_head = centered_group[0, :, local_head : local_head + 1, :]
            sq_head = sq_group[0, :, local_head : local_head + 1, :]

            prog.fill(mean[0, 0, :], 0.0)
            prog.fill(var[0, 0, :], 0.0)

            prog.row_op(x_head, op="reduce_sum", out=mean[0, 0, :], dim=-1)
            prog.pure_fp_compute(
                mean[0, 0, :],
                mean[0, 0, :],
                src2=recip_hlen,
                control="mul",
                task_id=f"ln_mean_scale.g{group_start}.h{local_head}",
            )
            prog.row_op(centered_head, mean[0, 0, :], "sub", dim=-1)

            prog.atomic_mul(centered_group, centered_group, sq_group)
            prog.row_op(sq_head, op="reduce_sum", out=var[0, 0, :], dim=-1)
            prog.pure_fp_compute(
                var[0, 0, :],
                var[0, 0, :],
                src2=recip_hlen,
                control="mul",
                task_id=f"ln_var_scale.g{group_start}.h{local_head}",
            )
            prog.pure_fp_compute(
                var[0, 0, :],
                var[0, 0, :],
                src2=eps_vec,
                control="add",
                task_id=f"ln_var_eps.g{group_start}.h{local_head}",
            )
            prog.fp_sqrt(var[0, 0, :], inv_std[0, 0, :])
            prog.fp_reci(inv_std[0, 0, :], inv_std[0, 0, :])
            prog.row_op(centered_head, inv_std[0, 0, :], "mul", dim=-1)

        prog.atomic_mul(centered_group, weight_group, centered_group)
        prog.atomic_add(centered_group, bias_group, centered_group)
        prog.copy(centered_group, y[0, :, group_start : group_start + group_heads, :])

    prog.copy(y, out_buf)
    return prog, y, out_buf


def build_golden(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    y = (x - mean) / torch.sqrt(var + eps)
    return y * weight + bias


if __name__ == "__main__":
    torch.manual_seed(0)

    mlen = 64
    blen = 4
    hlen = 16
    seq_len = 128
    head_count = 8
    eps = 1.0e-4

    x_data = torch.randn(1, seq_len, head_count, hlen, dtype=torch.float32) * 0.25
    weight_head = torch.randn(1, 1, head_count, hlen, dtype=torch.float32) * 0.2 + 1.0
    bias_head = torch.randn(1, 1, head_count, hlen, dtype=torch.float32) * 0.05
    weight_data = weight_head.expand(1, seq_len, head_count, hlen).contiguous()
    bias_data = bias_head.expand(1, seq_len, head_count, hlen).contiguous()
    golden_y = build_golden(x_data, weight_data, bias_data, eps)

    prog, out, out_buf = build_program(
        mlen=mlen,
        blen=blen,
        hlen=hlen,
        seq_len=seq_len,
        head_count=head_count,
        eps=eps,
    )
    comparison_params = stage_input_tensor_for_stride_compare(out_buf)
    staging_isa = comparison_params.pop("staging_isa", "")
    fp_preload = prog.build_fp_preload(min_size=32)
    gen_code = prog.compile()
    if staging_isa:
        gen_code = gen_code.rstrip() + "\n\n" + staging_isa

    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_feed, input_order = build_input_feed(
        prog,
        {
            "X_IN": x_data,
            "WEIGHT_IN": weight_data,
            "BIAS_IN": bias_data,
            "OUT": torch.zeros(1, seq_len, head_count, hlen, dtype=torch.float32),
        },
    )
    golden_result = {"original_output": golden_y.reshape(seq_len, head_count * hlen)}

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
        asm="tile_tensor_group_head_layer_norm_test",
        data=None,
        specified_data_order=input_order,
        build_path=build_dir,
    )

    with open(build_dir / "comparison_params.json", "w", encoding="utf-8") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "tile_tensor_group_head_layer_norm_generated_asm_code.asm", "w", encoding="utf-8") as f:
        f.write(gen_code)

    if np is not None:
        np.save(
            build_dir / "tile_tensor_group_head_layer_norm_golden_fp32.npy",
            golden_y.reshape(seq_len, head_count * hlen).detach().cpu().numpy().astype(np.float32),
        )

    ops_report_path = build_dir / "tile_tensor_group_head_layer_norm_ops.txt"
    tile_report_path = build_dir / "tile_tensor_group_head_layer_norm_tile_report.txt"
    prog.write_operation_report(ops_report_path)
    prog.write_tile_distribution_report(tile_report_path)
    print(f"operation report: {ops_report_path}")
    print(f"tile report: {tile_report_path}")
