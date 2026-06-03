#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compiler.asm_templates.elementwise_add_vram_asm import elementwise_add_vram_asm
from compiler.asm_templates.preload_addr_reg import preload_addr_reg_asm
from compiler.asm_templates.projection_asm import projection_asm
from compiler.asm_templates.reset_reg_asm import reset_reg_asm
from transactional_emulator.testbench.config_utils import update_plena_config
from transactional_emulator.testbench.siglip.utils.core import tensor_metrics
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    format_siglip_extended_run_config,
    load_siglip_harness_run_config,
    prepare_case_and_run_emulator,
    resolve_siglip_vram_dump_path,
)
from transactional_emulator.testbench.siglip.utils.math import (
    MXFP_REAL_DATA_RATIO,
    projection_matmul_k_split_visible,
    quantize_flattened_like_hbm,
)
from transactional_emulator.testbench.siglip.utils.siglip_tensors import prepare_reduced_siglip_tensors
from transactional_emulator.testbench.siglip.utils.vram import (
    load_vram_bf16,
    pack_seq_to_chunk_major,
    unpack_chunk_major_to_seq,
)


def emit_and_run_q_projection_isolation(build_dir: Path) -> None:
    blen = 4
    run_cfg = load_siglip_harness_run_config(
        build_dir=build_dir,
        mlen_default=64,
        vlen_default=64,
        q_chunk_default=729,
        inter_dim_default=256,
    )
    mlen = run_cfg.mlen
    vlen = run_cfg.vlen
    max_q_chunk = run_cfg.max_q_chunk

    update_plena_config(vlen=vlen, mlen=mlen, blen=blen, verbose=False)
    build_dir.mkdir(parents=True, exist_ok=True)
    print(format_siglip_extended_run_config(run_cfg))

    tensors = prepare_reduced_siglip_tensors(mlen=mlen)
    s_full = int(tensors["s_full"])
    hidden_size = int(tensors["hidden_size"])
    hq = int(tensors["hq"])
    h_qkv = int(tensors["h_qkv"])
    x_in_full = tensors["x_in_full"]
    wq_padded = tensors["wq_padded"]
    q_bias_padded = tensors["q_bias_padded"]

    d_padded = mlen
    hidden_size_padded = hq * d_padded
    if hidden_size_padded != hidden_size:
        raise ValueError(
            f"Expected reduced hidden size to match padded width ({hidden_size_padded}), got {hidden_size}"
        )

    s_q_actual = min(max_q_chunk, s_full)
    s_q_kernel = ((s_q_actual + blen - 1) // blen) * blen

    eps = 1e-2
    real_data_ratio = MXFP_REAL_DATA_RATIO

    x_chunk_actual = x_in_full[:s_q_actual].contiguous()
    x_chunk_padded = torch.zeros(s_q_kernel, hidden_size_padded, dtype=x_chunk_actual.dtype)
    x_chunk_padded[:s_q_actual, :hidden_size] = x_chunk_actual

    x_ln1 = F.layer_norm(x_chunk_padded.float(), (hidden_size_padded,), eps=eps).to(torch.bfloat16).float()

    wq_hbm = quantize_flattened_like_hbm(wq_padded)
    q_seq_golden = projection_matmul_k_split_visible(
        x_ln1,
        wq_hbm[:hidden_size_padded, :hidden_size_padded],
        mlen=mlen,
    )
    q_seq_golden = (q_seq_golden + q_bias_padded.unsqueeze(0)).to(torch.bfloat16).float()

    x_chunk_packed = pack_seq_to_chunk_major(x_ln1, mlen=mlen)
    x_vram_flat = x_chunk_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
    q_bias_tile = q_bias_padded.unsqueeze(0).repeat(s_q_kernel, 1)
    q_bias_packed = pack_seq_to_chunk_major(q_bias_tile, mlen=mlen)
    q_bias_flat = q_bias_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)

    x_base = 0
    q_bias_base = int(x_vram_flat.size)
    q_seq_base = int(q_bias_base + q_bias_flat.size)
    vram_preload = np.concatenate([x_vram_flat, q_bias_flat])

    wq_flat = wq_padded.reshape(-1).to(torch.float32)
    wq_elems = int(wq_flat.numel())
    hbm_mb = int(np.ceil((wq_elems * real_data_ratio) / (1024 * 1024))) + 8

    asm = "; SigLIP Q Projection Isolation Test\n"
    asm += preload_addr_reg_asm(
        addr_reg_to_set=[1],
        available_registers=[1],
        addr_reg_val=[0],
    )
    asm += reset_reg_asm(alive_registers=[1])
    asm += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=s_q_kernel,
        hidden_size=hidden_size_padded,
        vlen=vlen,
        alive_registers=[4, 5, 6, 7, 8, 9],
        w_base_hbm_offset_reg=1,
        activation_base_address=x_base,
        result_base_address=q_seq_base,
        out_features=hidden_size_padded,
        scratch_base_address=q_seq_base + s_q_kernel * hidden_size_padded,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[4, 5, 6, 7, 8, 9])
    asm += elementwise_add_vram_asm(
        vlen=vlen,
        num_vectors=(s_q_kernel * hidden_size_padded) // vlen,
        alive_registers=[10, 11],
        dst_base_address=q_seq_base,
        src_base_address=q_bias_base,
    )
    asm += reset_reg_asm(alive_registers=[10, 11])

    input_tensor = {"WQ": wq_flat.reshape(1, -1)}
    golden_result = {
        "input_tensor": {"WQ": wq_flat.reshape(-1)},
        "original_output": q_seq_golden.reshape(-1),
    }

    prepare_case_and_run_emulator(
        case_build_dir=build_dir,
        input_tensor=input_tensor,
        asm_code=asm,
        golden_result=golden_result,
        fp_preload=None,
        vram_preload=vram_preload,
        hbm_mb=hbm_mb,
        data_order=["WQ"],
    )

    vram_file = resolve_siglip_vram_dump_path()
    q_seq_sim_flat = load_vram_bf16(
        vram_file,
        num_elements=int(s_q_kernel * hidden_size_padded),
        start_elem=int(q_seq_base),
    )
    q_seq_sim = unpack_chunk_major_to_seq(
        q_seq_sim_flat,
        seq_len=int(s_q_kernel),
        hidden_dim=int(hidden_size_padded),
        mlen=int(mlen),
    )

    packed_metrics = tensor_metrics(
        q_seq_sim_flat,
        pack_seq_to_chunk_major(q_seq_golden, mlen=mlen),
    )
    full_metrics = tensor_metrics(q_seq_sim, q_seq_golden)
    actual_metrics = tensor_metrics(q_seq_sim[:s_q_actual], q_seq_golden[:s_q_actual])

    report = {
        "run_config": {
            "SIGLIP_Q_CHUNK": int(max_q_chunk),
            "s_q_actual": int(s_q_actual),
            "s_q_kernel": int(s_q_kernel),
            "hidden_size_padded": int(hidden_size_padded),
            "h_qkv": int(h_qkv),
            "mlen": int(mlen),
            "vlen": int(vlen),
            "eps": float(eps),
        },
        "projection_packed_layout": packed_metrics,
        "projection_full_kernel": full_metrics,
        "projection_actual_tokens": actual_metrics,
        "sample": {
            "golden": q_seq_golden.reshape(-1)[:8].tolist(),
            "sim": q_seq_sim.reshape(-1)[:8].tolist(),
        },
    }

    with open(build_dir / "q_projection_isolation_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(
        "Q projection isolation: "
        f"actual={s_q_actual} kernel={s_q_kernel} hidden={hidden_size_padded} "
        f"packed_match={packed_metrics['match_rate']:.3f}% "
        f"full_match={full_metrics['match_rate']:.3f}% "
        f"actual_match={actual_metrics['match_rate']:.3f}% "
        f"max_err={actual_metrics['max_error']:.6f}"
    )


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "build" / "siglip_q_projection_isolation"
    emit_and_run_q_projection_isolation(out_dir)
