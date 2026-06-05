#!/usr/bin/env python3
"""
Dedicated SigLIP Flash Attention Test
Isolates attention computation to diagnose accuracy issues without full encoder layer
"""
from pathlib import Path
import argparse
import json
import os

import numpy as np
import torch

from transactional_emulator.testbench.siglip.local_asm_templates.layout import (
    compute_hbm_offsets,
    compute_vram_layout,
)
from compiler.asm_templates.flashattn.overall import flash_attn_asm
from compiler.asm_templates.flashattn.encoder_mha import flash_attn_encoder_mha_asm
from compiler.asm_templates.projection_asm import projection_asm
from compiler.asm_templates.preload_addr_reg import preload_addr_reg_asm
from compiler.asm_templates.reset_reg_asm import reset_reg_asm
from transactional_emulator.testbench.config_utils import update_plena_config
from transactional_emulator.testbench.siglip.utils.harness_utils import (
    prepare_case_and_run_emulator,
    resolve_siglip_vram_dump_path,
    write_comparison_params,
)
from transactional_emulator.testbench.siglip.utils.vram import (
    load_vram_bf16,
    pack_seq_to_chunk_major,
)
from transactional_emulator.testbench.siglip.utils.math import (
    MXFP_REAL_DATA_RATIO,
    gqa_sdpa,
    quantize_flattened_like_hbm,
)


def _write_golden_values_file(file_path: Path, values: torch.Tensor) -> None:
    """Write golden values as hex for comparison."""
    values_bf16 = values.to(torch.bfloat16)
    int16_view = values_bf16.view(torch.int16).numpy()
    uint16_view = int16_view.astype(np.uint16)
    hex_strings = [f"{val:04x}" for val in uint16_view]
    with open(file_path, "w") as f:
        f.write("\n".join(hex_strings))


def _simple_compare_vram_with_golden(
    vram_bin_file: Path, golden_hex_file: Path, num_elements: int, start_elem: int = 0
) -> dict:
    """Compare a slice of the binary VRAM dump against a hex golden file.

    start_elem: element offset (in uint16 units) into the VRAM dump where the
                region of interest begins.
    """
    if not vram_bin_file.exists() or not golden_hex_file.exists():
        return {"allclose_match_rate": 0.0, "mse": float('inf')}

    with open(vram_bin_file, "rb") as f:
        vram_bytes = f.read()

    vram_uint16 = np.frombuffer(vram_bytes, dtype=np.uint16)
    vram_region  = vram_uint16[start_elem : start_elem + num_elements]
    vram_bf16    = torch.from_numpy(vram_region.copy()).view(torch.bfloat16).float()

    with open(golden_hex_file) as f:
        golden_hex = [line.strip() for line in f if line.strip()]

    golden_uint16 = np.array([int(h, 16) for h in golden_hex[:num_elements]], dtype=np.uint16)
    golden_bf16   = torch.from_numpy(golden_uint16).view(torch.bfloat16).float()

    match_rate = float(torch.isclose(vram_bf16, golden_bf16, atol=1e-2, rtol=1e-2).sum()) / len(vram_bf16)
    mse = torch.mean((vram_bf16 - golden_bf16) ** 2).item()

    return {"allclose_match_rate": match_rate, "mse": mse}


def _load_golden_bf16(golden_hex_file: Path, num_elements: int) -> torch.Tensor:
    with open(golden_hex_file) as f:
        golden_hex = [line.strip() for line in f if line.strip()]
    golden_uint16 = np.array([int(h, 16) for h in golden_hex[:num_elements]], dtype=np.uint16)
    return torch.from_numpy(golden_uint16).view(torch.bfloat16).float()


def _error_diagnostics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    abs_err = (pred - target).abs()
    rmse = torch.sqrt(torch.mean((pred - target) ** 2)).item()
    return {
        "rmse": float(rmse),
        "p50_abs": float(torch.quantile(abs_err, 0.50).item()),
        "p90_abs": float(torch.quantile(abs_err, 0.90).item()),
        "p99_abs": float(torch.quantile(abs_err, 0.99).item()),
        "under_1e2": float((abs_err <= 1e-2).float().mean().item()),
        "under_2e2": float((abs_err <= 2e-2).float().mean().item()),
        "under_5e2": float((abs_err <= 5e-2).float().mean().item()),
        "under_1e1": float((abs_err <= 1e-1).float().mean().item()),
    }


def _match_rate(pred: torch.Tensor, target: torch.Tensor, atol: float, rtol: float | None = None) -> float:
    if rtol is None:
        rtol = atol
    return float(torch.isclose(pred, target, atol=atol, rtol=rtol).float().mean().item())


def _match_and_mse(pred: torch.Tensor, target: torch.Tensor, atol: float = 1e-2, rtol: float = 1e-2) -> tuple[float, float]:
    match = float(torch.isclose(pred, target, atol=atol, rtol=rtol).float().mean().item())
    mse = float(torch.mean((pred - target) ** 2).item())
    return match, mse


def _seq_to_head_major_flat(q_seq: torch.Tensor, s_q: int, hq: int, d_padded: int) -> torch.Tensor:
    return q_seq.reshape(s_q, hq, d_padded).permute(1, 0, 2).contiguous().reshape(-1).float()


def _build_q_only_asm(
    *,
    mlen: int,
    blen: int,
    vlen: int,
    s_q: int,
    hidden_size_padded: int,
    x_base_vram: int,
    q_seq_base_vram: int,
    q_head_base_vram: int,
    scratch_base_vram: int,
    hq: int,
    d_padded: int,
    wq_hbm_offset: int,
) -> str:
    asm = "; SigLIP Q-only Probe\n"
    asm += preload_addr_reg_asm(
        addr_reg_to_set=[3],
        available_registers=[3],
        addr_reg_val=[int(wq_hbm_offset)],
    )
    asm += projection_asm(
        mlen=mlen,
        blen=blen,
        batch=s_q,
        hidden_size=hidden_size_padded,
        vlen=vlen,
        alive_registers=[4, 5, 6, 7, 8, 9],
        w_base_hbm_offset_reg=3,
        activation_base_address=x_base_vram,
        result_base_address=q_seq_base_vram,
        out_features=hidden_size_padded,
        scratch_base_address=scratch_base_vram,
        rope_enabled=False,
    )
    asm += reset_reg_asm(alive_registers=[4, 5, 6, 7, 8, 9])
    # projection_asm already writes chunk-major [hidden//vlen, s_q, vlen].
    # In this test config hidden//vlen == hq and vlen == d_padded, so this is
    # exactly the head-major Q layout consumed by flash_attn_asm.
    return asm


def emit_and_run_asm_test(
    build_path: Path,
    tiny_debug: bool = False,
    hq_override: int | None = None,
    deterministic: bool = False,
    s_q_override: int | None = None,
    s_kv_override: int | None = None,
    mlen_override: int | None = None,
    vlen_override: int | None = None,
    blen_override: int | None = None,
    force_head_dim_to_mlen: bool = False,
    kv_valid_len_override: int | None = None,
    use_encoder_mha_path: bool = False,
) -> dict:
    """Run flash attention in isolation with full SigLIP Layer-0 dimensions.

    Q is computed on-chip from X via projection_asm, then repacked to head-major.
    K, V are MXFP-quantized and fetched from HBM by the flash attention kernel.
    """

    # ---- Hardware config ----
    mlen = int(mlen_override) if mlen_override is not None else 128
    vlen = int(vlen_override) if vlen_override is not None else mlen
    blen = int(blen_override) if blen_override is not None else 4
    if mlen <= 0 or vlen <= 0 or blen <= 0:
        raise ValueError("mlen/vlen/blen must be > 0")
    update_plena_config(vlen=vlen, mlen=mlen, blen=blen, verbose=False)

    # ---- SigLIP Layer 0 dimensions (single tile) ----
    if hq_override is not None:
        s_q = 128
        s_kv = 128
        hq = int(hq_override)
        hkv = int(hq_override)
        d_padded = 128
    elif tiny_debug:
        s_q = 128
        s_kv = 128
        hq = 2
        hkv = 2
        d_padded = 128
    else:
        s_q = 128
        s_kv = 128
        hq = 16
        hkv = 16
        d_padded = ((72 + mlen - 1) // mlen) * mlen  # head dim (72) padded to MLEN

    if s_q_override is not None:
        s_q = int(s_q_override)
    if s_kv_override is not None:
        s_kv = int(s_kv_override)
    if s_q <= 0 or s_kv <= 0:
        raise ValueError("s_q/s_kv must be > 0")

    if force_head_dim_to_mlen and d_padded > mlen:
        print(
            f"Warning: forcing head dim from {d_padded} to MLEN={mlen} "
            "for experimental non-faithful run."
        )
        d_padded = mlen

    s_kv_override = os.environ.get("SIGLIP_ATTN_S_KV", "").strip()
    if s_kv_override:
        s_kv = int(s_kv_override)
        if s_kv <= 0:
            raise ValueError("SIGLIP_ATTN_S_KV must be > 0")

    kv_valid_len_env = os.environ.get("SIGLIP_ATTN_KV_VALID_LEN", "").strip()
    if kv_valid_len_override is not None:
        kv_valid_len = int(kv_valid_len_override)
    elif kv_valid_len_env:
        kv_valid_len = int(kv_valid_len_env)
    else:
        kv_valid_len = None
    if kv_valid_len is not None and (kv_valid_len <= 0 or kv_valid_len > s_kv):
        raise ValueError(f"SIGLIP_ATTN_KV_VALID_LEN must be in [1, {s_kv}]")
    hidden_size_padded = hq * d_padded  # 2048
    real_data_ratio = MXFP_REAL_DATA_RATIO

    scale = 1.0 / np.sqrt(d_padded)

    # FP SRAM slots — match encoder layer convention
    attn_scale_fp_slot  = 1
    attn_ninf_fp_slot   = 6
    flash_temp_fp_start = 64

    torch.manual_seed(42)

    # ---- X and WQ (Q projection) ----
    # X stays in VRAM at x_base=0. WQ is stored in HBM and used by projection_asm.
    # Use realistic small magnitudes to keep attention logits numerically stable.
    if deterministic or tiny_debug or hq_override is not None:
        x_grid = torch.arange(s_q * hidden_size_padded, dtype=torch.float32).reshape(s_q, hidden_size_padded)
        wq_grid = torch.arange(hidden_size_padded * hidden_size_padded, dtype=torch.float32).reshape(
            hidden_size_padded, hidden_size_padded
        )
        x_seq = (((x_grid % 17) - 8) / 512.0).to(torch.bfloat16)
        wq = (((wq_grid % 19) - 9) / 512.0).to(torch.bfloat16)
    else:
        x_seq = (0.02 * torch.randn(s_q, hidden_size_padded)).to(torch.bfloat16)  # [s_q, hidden]
        wq = (0.02 * torch.randn(hidden_size_padded, hidden_size_padded)).to(torch.bfloat16)  # [in, out]
    # Keep HBM inputs unquantized here; create_mem_for_sim applies the MXFP pass
    # to the flattened [1, N] payload. Golden must match that exact quantization shape.
    wq_mxfp = quantize_flattened_like_hbm(wq)
    wq_flat_f32 = wq.reshape(-1).to(torch.float32)

    # Golden Q from same projected path (BF16-visible result).
    q_seq_proj = torch.matmul(x_seq.float(), wq_mxfp.float()).to(torch.bfloat16)  # [s_q, hidden]
    q_seq_proj_heads = q_seq_proj.reshape(s_q, hq, d_padded).contiguous()

    # ---- K, V: MXFP-quantized, seq-major [s_kv, hkv, d_padded] ----
    # ---- K, V: head-major [hkv, s_kv, d_padded] in HBM (required by per-head kernel) ----
    # The flash attn HBM offset formula is kv_head * kv_len * d, which assumes head-major.
    if deterministic or tiny_debug or hq_override is not None:
        kv_grid = torch.arange(s_kv * hkv * d_padded, dtype=torch.float32).reshape(s_kv, hkv, d_padded)
        k_seq = (((kv_grid % 23) - 11) / 128.0).to(torch.bfloat16)  # [s_kv, hkv, d]
        v_seq = ((((kv_grid + 7) % 29) - 14) / 128.0).to(torch.bfloat16)
    else:
        k_seq = torch.randn(s_kv, hkv, d_padded, dtype=torch.bfloat16)   # [s_kv, hkv, d]
        v_seq = torch.randn(s_kv, hkv, d_padded, dtype=torch.bfloat16)
    k_head = k_seq.permute(1, 0, 2).contiguous()                      # [hkv, s_kv, d]
    v_head = v_seq.permute(1, 0, 2).contiguous()
    k_mxfp = quantize_flattened_like_hbm(k_head)      # emulator-visible K
    v_mxfp = quantize_flattened_like_hbm(v_head)      # emulator-visible V
    k_flat_f32 = k_head.reshape(-1).to(torch.float32)
    v_flat_f32 = v_head.reshape(-1).to(torch.float32)
    # For golden: seq-major [s_kv, hkv, d] (permute back)
    k_mxfp_seq = k_mxfp.permute(1, 0, 2).contiguous()
    v_mxfp_seq = v_mxfp.permute(1, 0, 2).contiguous()

    # ---- HBM layout: WQ first, then K, then V ----
    wq_elems = wq_flat_f32.numel()
    k_elems = k_flat_f32.numel()
    v_elems = v_flat_f32.numel()
    (wq_hbm_offset, k_hbm_offset, v_hbm_offset), _ = compute_hbm_offsets(
        [wq_elems, k_elems, v_elems], real_data_ratio=real_data_ratio, align_elems=64
    )
    hbm_mb_base = int(np.ceil(((wq_elems + k_elems + v_elems) * real_data_ratio) / (1024 * 1024))) + 8
    # With batch=s_q, flash-attn KV prefetch can walk a much larger MX index space
    # than the raw Q/K/V tensor footprint implies. Use a conservative floor to
    # avoid emulator HBM scale-array OOB during isolated-kernel stress testing.
    hbm_mb = max(hbm_mb_base, 256)

    # ---- VRAM layout ----
    # [x_base .. x_base + s_q*hidden) holds X in seq-major.
    # [q_seq_base ..) receives projection output in seq-major.
    # [q_head_base ..) receives repacked head-major Q used by flash attention.
    x_base_vram = 0
    x_elems = s_q * hidden_size_padded
    q_seq_base_vram = x_base_vram + x_elems
    q_head_base_vram = q_seq_base_vram
    layout = compute_vram_layout(
        mlen=mlen, blen=blen,
        q_len=s_q,
        hq=hq, hkv=hkv, d=d_padded,
        vector_sram_base=q_head_base_vram,
    )
    attn_out_base = layout["o_old_base"]
    scratch_base_vram = attn_out_base + (s_q * hidden_size_padded)

    # ---- Golden: SDPA with projected Q and HBM K/V ----
    q_for_sdpa = q_seq_proj_heads.unsqueeze(0).float()  # [1, s_q, hq, d]
    k_for_sdpa = k_mxfp_seq.unsqueeze(0).float()    # [1, s_kv, hkv, d]
    v_for_sdpa = v_mxfp_seq.unsqueeze(0).float()    # [1, s_kv, hkv, d]
    attn_golden = gqa_sdpa(q_for_sdpa, k_for_sdpa, v_for_sdpa, scale, hq, hkv, kv_valid_len=kv_valid_len)
    # attn_golden: [s_q, hidden_size_padded]

    # Packed probe: row-chunk major [num_chunks, s_q, mlen]
    num_chunks = hidden_size_padded // mlen
    attn_packed_golden = (
        attn_golden.reshape(s_q, num_chunks, mlen)
        .permute(1, 0, 2).contiguous().reshape(-1)
    )

    # ---- Assembly ----
    # a1 = K HBM offset, a2 = V HBM offset, a3 = WQ HBM offset
    asm = _build_q_only_asm(
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        s_q=s_q,
        hidden_size_padded=hidden_size_padded,
        x_base_vram=x_base_vram,
        q_seq_base_vram=q_seq_base_vram,
        q_head_base_vram=q_head_base_vram,
        scratch_base_vram=scratch_base_vram,
        hq=hq,
        d_padded=d_padded,
        wq_hbm_offset=wq_hbm_offset,
    )
    asm = asm.replace("; SigLIP Q-only Probe", "; SigLIP Isolated Flash Attention Test")
    asm = asm[:0] + preload_addr_reg_asm(
        addr_reg_to_set=[1, 2, 3],
        available_registers=[1, 2, 3],
        addr_reg_val=[int(k_hbm_offset), int(v_hbm_offset), int(wq_hbm_offset)],
    ) + asm.split("\n", 2)[2]
    if use_encoder_mha_path:
        if hq != hkv:
            raise ValueError("--use-encoder-mha-path requires hq == hkv")
        asm += flash_attn_encoder_mha_asm(
            mlen=mlen,
            vlen=vlen,
            blen=blen,
            batch=1,
            hq=hq,
            hkv=hkv,
            d=d_padded,
            q_len=s_q,
            kv_len=s_kv,
            alive_registers_int=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            alive_registers_fp=[1, 2, 3, 4, 5, 6],
            vector_sram_base_address=q_head_base_vram,
            fp_sram_start_address=flash_temp_fp_start,
            k_base_hbm_offset_reg=1,
            v_base_hbm_offset_reg=2,
            attn_scale_fp_address=attn_scale_fp_slot,
            inf_fp_address=attn_ninf_fp_slot,
            causal_mask=False,
            kv_valid_len=kv_valid_len,
        )
    else:
        asm += flash_attn_asm(
            mlen=mlen, vlen=vlen, blen=blen,
            batch=1, hq=hq, hkv=hkv, d=d_padded,
            q_len=s_q, kv_len=s_kv,
            alive_registers_int=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            alive_registers_fp=[1, 2, 3, 4, 5, 6],
            vector_sram_base_address=q_head_base_vram,
            fp_sram_start_address=flash_temp_fp_start,
            k_base_hbm_offset_reg=1,
            v_base_hbm_offset_reg=2,
            attn_scale_fp_address=attn_scale_fp_slot,
            inf_fp_address=attn_ninf_fp_slot,
            causal_mask=False,
            kv_valid_len=kv_valid_len,
        )

    fp_preload = [0.0] * 1024
    fp_preload[attn_scale_fp_slot] = float(scale)
    fp_preload[attn_ninf_fp_slot]  = float("-inf")

    # ---- Sim env setup ----
    input_tensor = {
        "WQ": wq_flat_f32.reshape(1, -1),
        "K": k_flat_f32.reshape(1, -1),
        "V": v_flat_f32.reshape(1, -1),
    }
    golden_result = {
        "input_tensor": input_tensor,
        "original_output": attn_golden.reshape(s_q, hidden_size_padded),
    }

    build_path.mkdir(parents=True, exist_ok=True)

    # Q-only probe: projection + repack without flash attention.
    q_only_build_path = build_path / "q_only_probe"
    q_only_build_path.mkdir(parents=True, exist_ok=True)
    q_only_asm = _build_q_only_asm(
        mlen=mlen,
        blen=blen,
        vlen=vlen,
        s_q=s_q,
        hidden_size_padded=hidden_size_padded,
        x_base_vram=x_base_vram,
        q_seq_base_vram=q_seq_base_vram,
        q_head_base_vram=q_head_base_vram,
        scratch_base_vram=scratch_base_vram,
        hq=hq,
        d_padded=d_padded,
        wq_hbm_offset=wq_hbm_offset,
    )
    x_chunk_packed = pack_seq_to_chunk_major(x_seq.float(), mlen=mlen)
    x_vram_preload = x_chunk_packed.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)

    prepare_case_and_run_emulator(
        case_build_dir=q_only_build_path,
        input_tensor=input_tensor,
        asm_code=q_only_asm,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=x_vram_preload,
        hbm_mb=hbm_mb,
        data_order=["WQ", "K", "V"],
        emulator_hbm_size=64 * 1024 * 1024,
        emulator_log_path=q_only_build_path / "emulator.log",
    )
    q_only_vram_file = resolve_siglip_vram_dump_path()
    q_num_elements = hq * s_q * d_padded
    q_seq_num_elements = s_q * hidden_size_padded
    q_only_q_seq_vram = load_vram_bf16(q_only_vram_file, q_seq_num_elements, start_elem=q_seq_base_vram)
    q_only_q_head_vram = load_vram_bf16(q_only_vram_file, q_num_elements, start_elem=q_head_base_vram)
    q_seq_golden = pack_seq_to_chunk_major(q_seq_proj.float(), mlen=mlen)
    q_head_golden = _seq_to_head_major_flat(q_seq_proj, s_q, hq, d_padded)
    q_seq_match, q_seq_mse = _match_and_mse(q_only_q_seq_vram, q_seq_golden)
    q_only_match, q_only_mse = _match_and_mse(q_only_q_head_vram, q_head_golden)

    prepare_case_and_run_emulator(
        case_build_dir=build_path,
        input_tensor=input_tensor,
        asm_code=asm,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=x_vram_preload,
        hbm_mb=hbm_mb,
        data_order=["WQ", "K", "V"],
        emulator_hbm_size=64 * 1024 * 1024,
        emulator_log_path=build_path / "emulator.log",
    )

    # Golden probe files
    golden_unpacked = build_path / "attn_golden_unpacked.txt"
    golden_packed   = build_path / "attn_golden_packed.txt"
    _write_golden_values_file(golden_unpacked, attn_golden.reshape(-1))
    _write_golden_values_file(golden_packed,   attn_packed_golden)

    write_comparison_params(
        build_path,
        start_row_idx=int(attn_out_base // mlen),
        num_rows=int((s_q * hidden_size_padded) // mlen),
        num_batches=int(s_q),
        elements_per_batch=int(hidden_size_padded),
        use_stride_mode=True,
        extra_params={"row_dim": int(mlen)},
    )

    print("=" * 70)
    print("SigLIP Flash Attention Test — Isolated (Q via projection_asm)")
    print("=" * 70)
    if hq_override is not None:
        mode_name = f"hq-sweep(hq={hq_override})"
    elif tiny_debug:
        mode_name = "tiny-debug"
    elif deterministic:
        mode_name = "deterministic"
    else:
        mode_name = "full"
    print(f"  mode={mode_name}")
    print(f"  kernel_path={'encoder_mha' if use_encoder_mha_path else 'overall'}")
    print(f"  s_q={s_q}, s_kv={s_kv}, hq={hq}, hkv={hkv}, d={d_padded}")
    print(f"  hidden={hidden_size_padded}, mlen={mlen}, vlen={vlen}, blen={blen}")
    print(f"  scale={scale:.6f}")
    print(f"  wq_hbm_offset={wq_hbm_offset}, k_hbm_offset={k_hbm_offset}, v_hbm_offset={v_hbm_offset}")
    print(f"  attn_out_base (VSRAM)={attn_out_base}  start_row={attn_out_base // mlen}")

    # ---- Run emulator ----
    print("\n--- Running Rust transactional emulator ---")
    # Emulator already ran via prepare_case_and_run_emulator with the required HBM override.

    # ---- Probing ----
    vram_file = resolve_siglip_vram_dump_path()
    num_elements = s_q * hidden_size_padded

    # Verify Q path before evaluating attention quality.
    q_num_elements = hq * s_q * d_padded
    q_head_vram = load_vram_bf16(vram_file, q_num_elements, start_elem=q_head_base_vram)
    q_match, q_mse = _match_and_mse(q_head_vram, q_head_golden)
    print(f"  Q-seq probe:    match={q_seq_match:.2%}  mse={q_seq_mse:.4e}")
    print(f"  Q-only probe:   match={q_only_match:.2%}  mse={q_only_mse:.4e}")
    print(f"  Q VRAM check:   match={q_match:.2%}  mse={q_mse:.4e}")

    unpacked_results = _simple_compare_vram_with_golden(
        vram_file, golden_unpacked, num_elements, start_elem=attn_out_base,
    )
    packed_results = _simple_compare_vram_with_golden(
        vram_file, golden_packed, num_elements, start_elem=attn_out_base,
    )

    print("\n" + "=" * 70)
    print("PROBING RESULTS")
    print("=" * 70)
    print(f"  Unpacked order: match={unpacked_results['allclose_match_rate']:.2%}  mse={unpacked_results['mse']:.4e}")
    print(f"  Packed order:   match={packed_results['allclose_match_rate']:.2%}  mse={packed_results['mse']:.4e}")

    # Attention-only metric: use actual Q read from VRAM as reference input.
    # This isolates flash-attn correctness from projection/repack drift.
    q_from_vram_seq = q_head_vram.reshape(hq, s_q, d_padded).permute(1, 0, 2).contiguous()
    attn_golden_q_from_vram = gqa_sdpa(
        q_from_vram_seq.unsqueeze(0).float(),
        k_for_sdpa,
        v_for_sdpa,
        scale,
        hq,
        hkv,
        kv_valid_len=kv_valid_len,
    ).reshape(-1)
    pred_out = load_vram_bf16(vram_file, num_elements, start_elem=attn_out_base)
    attn_only_match, attn_only_mse = _match_and_mse(pred_out, attn_golden_q_from_vram.float())
    print(f"  Attn-only (Q from VRAM): match={attn_only_match:.2%}  mse={attn_only_mse:.4e}")

    q_relaxed_5e2 = _match_rate(q_head_vram, q_head_golden, atol=5e-2)
    q_relaxed_1e1 = _match_rate(q_head_vram, q_head_golden, atol=1e-1)
    attn_relaxed_5e2 = _match_rate(pred_out, attn_golden_q_from_vram.float(), atol=5e-2)
    attn_relaxed_1e1 = _match_rate(pred_out, attn_golden_q_from_vram.float(), atol=1e-1)
    print(
        "  Relaxed match: "
        f"Q<=5e-2 {q_relaxed_5e2:.2%}  Q<=1e-1 {q_relaxed_1e1:.2%}  "
        f"Attn<=5e-2 {attn_relaxed_5e2:.2%}  Attn<=1e-1 {attn_relaxed_1e1:.2%}"
    )

    # Per-head diagnostics to localize structural mismatch.
    pred_head = pred_out.reshape(s_q, hq, d_padded)
    gold_head = attn_golden_q_from_vram.float().reshape(s_q, hq, d_padded)
    head_match = []
    head_mse = []
    for h in range(hq):
        p = pred_head[:, h, :].reshape(-1)
        g = gold_head[:, h, :].reshape(-1)
        m = float(torch.isclose(p, g, atol=1e-2, rtol=1e-2).float().mean().item())
        e = float(torch.mean((p - g) ** 2).item())
        head_match.append(m)
        head_mse.append(e)
    worst_by_match = np.argsort(np.array(head_match))[: min(4, hq)].tolist()
    print(
        "  Per-head match(min/mean/max): "
        f"{min(head_match):.2%}/{float(np.mean(head_match)):.2%}/{max(head_match):.2%}"
    )
    print(
        "  Worst heads by match: "
        + ", ".join([f"h{h}={head_match[h]:.2%}(mse={head_mse[h]:.3e})" for h in worst_by_match])
    )

    # Error distribution to distinguish small drift vs structural mismatch.
    pred_unpacked = load_vram_bf16(vram_file, num_elements, start_elem=attn_out_base)
    gold_unpacked = _load_golden_bf16(golden_unpacked, num_elements)
    diag = _error_diagnostics(pred_unpacked, gold_unpacked)
    print("\nERROR DIAGNOSTICS")
    print("=" * 70)
    print(f"  rmse={diag['rmse']:.4e}")
    print(
        f"  abs_err percentiles: p50={diag['p50_abs']:.4e}  p90={diag['p90_abs']:.4e}  p99={diag['p99_abs']:.4e}"
    )
    print(
        "  abs_err hit-rate: "
        f"<=1e-2 {diag['under_1e2']:.2%}  "
        f"<=2e-2 {diag['under_2e2']:.2%}  "
        f"<=5e-2 {diag['under_5e2']:.2%}  "
        f"<=1e-1 {diag['under_1e1']:.2%}"
    )

    return {
        "attn_unpacked": unpacked_results,
        "attn_packed":   packed_results,
        "q_vram_check": {
            "allclose_match_rate": q_match,
            "mse": q_mse,
        },
        "q_relaxed_5e2": {
            "allclose_match_rate": q_relaxed_5e2,
            "mse": q_mse,
        },
        "q_relaxed_1e1": {
            "allclose_match_rate": q_relaxed_1e1,
            "mse": q_mse,
        },
        "q_seq_probe": {
            "allclose_match_rate": q_seq_match,
            "mse": q_seq_mse,
        },
        "q_only_probe": {
            "allclose_match_rate": q_only_match,
            "mse": q_only_mse,
        },
        "attn_only_q_from_vram": {
            "allclose_match_rate": attn_only_match,
            "mse": attn_only_mse,
        },
        "attn_relaxed_5e2": {
            "allclose_match_rate": attn_relaxed_5e2,
            "mse": attn_only_mse,
        },
        "attn_relaxed_1e1": {
            "allclose_match_rate": attn_relaxed_1e1,
            "mse": attn_only_mse,
        },
        "per_head": {
            "match_rate": head_match,
            "mse": head_mse,
        },
        "error_diag": diag,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SigLIP isolated flash-attention test")
    parser.add_argument("--tiny", action="store_true", help="run tiny deterministic debug configuration")
    parser.add_argument("--hq", type=int, default=None, help="override hq/hkv for sweep (uses s_q=s_kv=128)")
    parser.add_argument("--deterministic", action="store_true", help="use deterministic X/WQ/K/V patterns")
    parser.add_argument("--s-q", type=int, default=None, help="override query sequence length")
    parser.add_argument("--s-kv", type=int, default=None, help="override KV sequence length")
    parser.add_argument("--kv-valid-len", type=int, default=None, help="override valid KV length used for masking")
    parser.add_argument(
        "--kv-sweep",
        type=str,
        default=None,
        help="comma-separated KV valid lengths to sweep (e.g. 64,128,256,729)",
    )
    parser.add_argument("--mlen", type=int, default=None, help="override hardware MLEN")
    parser.add_argument("--vlen", type=int, default=None, help="override hardware VLEN")
    parser.add_argument("--blen", type=int, default=None, help="override hardware BLEN")
    parser.add_argument(
        "--use-encoder-mha-path",
        action="store_true",
        help="use flash_attn_encoder_mha_asm (force per-head + chunk-major output)",
    )
    parser.add_argument(
        "--force-head-dim-mlen",
        action="store_true",
        help="experimental: force attention head dim to MLEN (non-faithful for SigLIP when MLEN < 72)",
    )
    args = parser.parse_args()

    build_path = Path(__file__).parent / "build" / "siglip_attention_test"
    if args.kv_sweep:
        kv_values = [int(v.strip()) for v in args.kv_sweep.split(",") if v.strip()]
        if not kv_values:
            raise ValueError("--kv-sweep must contain at least one integer")

        sweep_results = []
        print("\n" + "=" * 70)
        print("KV SWEEP")
        print("=" * 70)
        for kv_valid in kv_values:
            run_s_kv = args.s_kv if args.s_kv is not None else kv_valid
            run_build_path = build_path / f"kv_{kv_valid}"
            print(f"\n--- Running kv_valid_len={kv_valid}, s_kv={run_s_kv} ---")
            metrics = emit_and_run_asm_test(
                run_build_path,
                tiny_debug=args.tiny,
                hq_override=args.hq,
                deterministic=args.deterministic,
                s_q_override=args.s_q,
                s_kv_override=run_s_kv,
                mlen_override=args.mlen,
                vlen_override=args.vlen,
                blen_override=args.blen,
                force_head_dim_to_mlen=args.force_head_dim_mlen,
                kv_valid_len_override=kv_valid,
                use_encoder_mha_path=args.use_encoder_mha_path,
            )
            attn_only = metrics.get("attn_only_q_from_vram", {})
            packed = metrics.get("attn_packed", {})
            sweep_results.append(
                {
                    "kv_valid_len": kv_valid,
                    "s_kv": run_s_kv,
                    "attn_only_match_rate": float(attn_only.get("allclose_match_rate", 0.0)),
                    "attn_only_mse": float(attn_only.get("mse", float("inf"))),
                    "packed_match_rate": float(packed.get("allclose_match_rate", 0.0)),
                    "packed_mse": float(packed.get("mse", float("inf"))),
                }
            )

        summary_path = build_path / "kv_sweep_summary.json"
        with open(summary_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print("\nKV sweep summary written to", summary_path)
        for item in sweep_results:
            print(
                f"kv={item['kv_valid_len']}: "
                f"attn_only_match={item['attn_only_match_rate']:.2%}, "
                f"packed_match={item['packed_match_rate']:.2%}"
            )
    else:
        metrics = emit_and_run_asm_test(
            build_path,
            tiny_debug=args.tiny,
            hq_override=args.hq,
            deterministic=args.deterministic,
            s_q_override=args.s_q,
            s_kv_override=args.s_kv,
            mlen_override=args.mlen,
            vlen_override=args.vlen,
            blen_override=args.blen,
            force_head_dim_to_mlen=args.force_head_dim_mlen,
            kv_valid_len_override=args.kv_valid_len,
            use_encoder_mha_path=args.use_encoder_mha_path,
        )
        print("\n" + "=" * 70)
        print("FINAL METRICS")
        print("=" * 70)
        for name, m in metrics.items():
            if "allclose_match_rate" in m:
                print(f"{name}: match_rate={m['allclose_match_rate']:.2%}, mse={m['mse']:.6e}")
            elif "match_rate" in m and "mse" in m:
                if isinstance(m["match_rate"], list):
                    print(
                        f"{name}: match_rate(min/mean/max)="
                        f"{min(m['match_rate']):.2%}/{sum(m['match_rate']) / len(m['match_rate']):.2%}/{max(m['match_rate']):.2%}"
                    )
                else:
                    print(f"{name}: match_rate={m['match_rate']:.2%}, mse={m['mse']:.6e}")
            else:
                print(f"{name}: {m}")
