import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compiler.asm_templates.elementwise_add_vram_asm import elementwise_add_vram_asm
from compiler.asm_templates.normalization_asm import layer_norm_asm
from compiler.asm_templates.preload_addr_reg import preload_addr_reg_asm
from compiler.asm_templates.projection_asm import projection_asm
from compiler.asm_templates.reset_reg_asm import reset_reg_asm
from transactional_emulator.testbench.config_utils import update_plena_config
from transactional_emulator.testbench.siglip.utils.siglip_tensors import (
    load_or_prepare_full_siglip_tensors,
)
from transactional_emulator.testbench.siglip.utils.core import tensor_metrics
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    load_siglip_harness_run_config,
    prepare_case_and_run_emulator,
)
from transactional_emulator.testbench.siglip.utils.math import (
    MXFP_REAL_DATA_RATIO,
    projection_matmul_k_split_visible,
    quantize_flattened_like_hbm,
)
from transactional_emulator.testbench.siglip.utils.vram import (
    load_vram_bf16,
    pack_seq_to_chunk_major,
    unpack_chunk_major_to_seq,
)


def emit_and_run_projection_test(build_dir: Path) -> None:
    run_cfg = load_siglip_harness_run_config(
        build_dir=build_dir,
        mlen_default=128,
        vlen_default=128,
        q_chunk_default=128,
        max_chunks_default=1,
        inter_dim_default=1024,
    )
    mlen = run_cfg.mlen
    blen = 4
    vlen = run_cfg.vlen

    s_q_kernel = run_cfg.max_q_chunk
    include_ln1_asm = os.environ.get("SIGLIP_PROJECTION_INCLUDE_LN1", "1") != "0"

    update_plena_config(vlen=vlen, mlen=mlen, blen=blen, verbose=False)

    build_dir.mkdir(parents=True, exist_ok=True)
    cache_path = run_cfg.cache_path

    tensors = load_or_prepare_full_siglip_tensors(cache_path=cache_path)
    s_full = int(tensors["s_full"])
    hidden_size = int(tensors["hidden_size"])
    hq = int(tensors["hq"])
    h_qkv = int(tensors["h_qkv"])
    hidden_size_padded = hq * mlen

    # Keep this focused on projection validation only.
    start = 0
    end = min(start + s_q_kernel, s_full)
    s_q_actual = end - start

    x_in_full = tensors["x_in_full"]
    wq_padded = tensors["wq_padded"]
    q_bias_padded = tensors["q_bias_padded"]

    eps = 1e-2
    real_data_ratio = MXFP_REAL_DATA_RATIO

    x_chunk_actual = x_in_full[start:end].contiguous()
    x_chunk_padded = torch.zeros(s_q_kernel, hidden_size_padded, dtype=x_chunk_actual.dtype)
    x_chunk_padded[:s_q_actual, :hidden_size] = x_chunk_actual

    x_ln1 = F.layer_norm(x_chunk_padded.float(), (hidden_size_padded,), eps=eps)
    asm_input = x_chunk_padded if include_ln1_asm else x_ln1
    wq_hbm = quantize_flattened_like_hbm(wq_padded)
    x_ln1_hw = x_ln1.to(torch.bfloat16).float()
    q_seq_golden = projection_matmul_k_split_visible(
        x_ln1_hw,
        wq_hbm[:hidden_size_padded, :hidden_size_padded],
        mlen=mlen,
    )
    q_seq_golden = (q_seq_golden + q_bias_padded.unsqueeze(0)).to(torch.bfloat16).float()

    # VRAM default layout is chunk-major [hidden//vlen, batch, vlen].
    x_chunk_packed = pack_seq_to_chunk_major(asm_input, mlen=mlen)
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
    wq_hbm_offset = 0
    hbm_mb = int(np.ceil((wq_elems * real_data_ratio) / (1024 * 1024))) + 8

    ln_eps_fp_slot = 2
    ln_reci_hid_fp_slot = 3
    fp_preload = [0.0] * 64
    fp_preload[ln_eps_fp_slot] = eps
    fp_preload[ln_reci_hid_fp_slot] = 1.0 / hidden_size_padded

    asm = "; SigLIP Projection-only Test\n"
    asm += preload_addr_reg_asm(
        addr_reg_to_set=[1],
        available_registers=[1],
        addr_reg_val=[wq_hbm_offset],
    )
    asm += reset_reg_asm(alive_registers=[1])

    if include_ln1_asm:
        asm += layer_norm_asm(
            _eps_offset=ln_eps_fp_slot,
            reci_hid_offset=ln_reci_hid_fp_slot,
            alive_registers=[5, 6, 7],
            activation_base_address=x_base,
            scratchpad_base_address=q_seq_base + s_q_kernel * hidden_size_padded,
            vlen=vlen,
            batch_size=s_q_kernel,
            hidden_dim=hidden_size_padded,
        )
        asm += reset_reg_asm(alive_registers=[5, 6, 7])

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
        fp_preload=fp_preload,
        vram_preload=vram_preload,
        hbm_mb=hbm_mb,
        data_order=["WQ"],
    )

    emulator_dir = Path(__file__).parents[2]
    vram_file = emulator_dir / "vram_dump.bin"
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

    q_seq_golden_t = projection_matmul_k_split_visible(
        x_ln1_hw,
        wq_hbm[:hidden_size_padded, :hidden_size_padded].t(),
        mlen=mlen,
    )
    q_seq_golden_t = (q_seq_golden_t + q_bias_padded.unsqueeze(0)).to(torch.bfloat16).float()
    q_seq_golden_no_bias = projection_matmul_k_split_visible(
        x_ln1_hw,
        wq_hbm[:hidden_size_padded, :hidden_size_padded],
        mlen=mlen,
    )

    q_seq_golden_packed = pack_seq_to_chunk_major(q_seq_golden, mlen=mlen)
    packed_metrics = tensor_metrics(q_seq_sim_flat, q_seq_golden_packed)

    full_metrics = tensor_metrics(q_seq_sim, q_seq_golden)
    actual_metrics = tensor_metrics(q_seq_sim[:s_q_actual], q_seq_golden[:s_q_actual])
    transposed_metrics = tensor_metrics(q_seq_sim[:s_q_actual], q_seq_golden_t[:s_q_actual])
    no_bias_metrics = tensor_metrics(q_seq_sim[:s_q_actual], q_seq_golden_no_bias[:s_q_actual])

    report = {
        "run_config": {
            "SIGLIP_Q_CHUNK": s_q_kernel,
            "SIGLIP_PROJECTION_INCLUDE_LN1": include_ln1_asm,
            "hidden_size": hidden_size,
            "hidden_size_padded": hidden_size_padded,
            "h_qkv": h_qkv,
            "s_q_actual": s_q_actual,
        },
        "projection_packed_layout": packed_metrics,
        "projection_full_kernel": full_metrics,
        "projection_actual_tokens": actual_metrics,
        "projection_transposed_w_probe": transposed_metrics,
        "projection_no_bias_probe": no_bias_metrics,
        "sample": {
            "golden": q_seq_golden.reshape(-1)[:8].tolist(),
            "sim": q_seq_sim.reshape(-1)[:8].tolist(),
        },
    }

    with open(build_dir / "projection_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(
        "Projection metrics: "
        f"include_ln1_asm={include_ln1_asm} "
        f"packed_match={packed_metrics['match_rate']:.3f}% "
        f"full_match={full_metrics['match_rate']:.3f}% "
        f"actual_match={actual_metrics['match_rate']:.3f}% "
        f"transposed_probe={transposed_metrics['match_rate']:.3f}% "
        f"no_bias_probe={no_bias_metrics['match_rate']:.3f}% "
        f"max_err={actual_metrics['max_error']:.6f}"
    )


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "build" / "siglip_projection_test"
    emit_and_run_projection_test(out_dir)
