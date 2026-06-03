"""Shared harness utilities for SigLIP testbench scripts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import numpy as np
import torch

from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.emulator_runner import run_emulator
from transactional_emulator.tools.create_sim_env import create_sim_env


@dataclass(frozen=True)
class SiglipHarnessRunConfig:
    """Shared runtime config derived from SIGLIP_* environment variables."""

    mlen: int
    vlen: int
    max_q_chunk: int
    inter_dim_raw: str
    kv_valid_len_raw: str


def load_siglip_harness_run_config(
    *,
    build_dir: Path,
    mlen_default: int,
    vlen_default: int | None = None,
    q_chunk_default: int | None = None,
    inter_dim_default: int = 1024,
) -> SiglipHarnessRunConfig:
    """Parse common SIGLIP harness env vars with harness-specific defaults."""
    mlen = int(os.environ.get("SIGLIP_MLEN", str(mlen_default)))
    resolved_vlen_default = mlen if vlen_default is None else vlen_default
    vlen = int(os.environ.get("SIGLIP_VLEN", str(resolved_vlen_default)))
    resolved_q_chunk_default = mlen if q_chunk_default is None else q_chunk_default
    max_q_chunk = int(os.environ.get("SIGLIP_Q_CHUNK", str(resolved_q_chunk_default)))
    if max_q_chunk <= 0:
        raise ValueError("SIGLIP_Q_CHUNK must be > 0")

    inter_dim_raw = os.environ.get("SIGLIP_INTER_DIM", str(inter_dim_default))
    kv_valid_len_raw = os.environ.get("SIGLIP_KV_VALID_LEN", "full")

    return SiglipHarnessRunConfig(
        mlen=mlen,
        vlen=vlen,
        max_q_chunk=max_q_chunk,
        inter_dim_raw=inter_dim_raw,
        kv_valid_len_raw=kv_valid_len_raw,
    )


def warn_q_chunk_mismatch(*, max_q_chunk: int, mlen: int) -> None:
    """Warn when SIGLIP_Q_CHUNK differs from MLEN for mlen-tuned templates."""
    if max_q_chunk == mlen:
        return
    print(
        f"Warning: SIGLIP_Q_CHUNK={max_q_chunk} differs from mlen={mlen}; "
        "current encoder template is tuned for mlen-sized Q tiles and may diverge."
    )


def format_siglip_run_config(run_cfg: SiglipHarnessRunConfig) -> str:
    """Format the common SigLIP run-config log line."""
    return (
        "Run config: "
        f"SIGLIP_Q_CHUNK={run_cfg.max_q_chunk}, "
        f"SIGLIP_INTER_DIM={run_cfg.inter_dim_raw}"
    )


def format_siglip_extended_run_config(run_cfg: SiglipHarnessRunConfig) -> str:
    """Format SigLIP run-config log line with MLEN/VLEN and KV valid length."""
    return (
        "Run config: "
        f"SIGLIP_MLEN={run_cfg.mlen}, "
        f"SIGLIP_VLEN={run_cfg.vlen}, "
        f"SIGLIP_Q_CHUNK={run_cfg.max_q_chunk}, "
        f"SIGLIP_INTER_DIM={run_cfg.inter_dim_raw}, "
        f"SIGLIP_KV_VALID_LEN={run_cfg.kv_valid_len_raw}"
    )


def resolve_siglip_vram_dump_path() -> Path:
    """Return the canonical repo-level SigLIP VRAM dump path."""
    return Path(__file__).resolve().parents[3] / "vram_dump.bin"


def build_encoder_stage_metrics(
    *,
    vram_bin_file: Path,
    mlen: int,
    s_q_actual: int,
    hidden_size_padded: int,
    aligned_inter_dim: int,
    debug_stage0_snapshot_base: int | None,
    x_ln1_base: int | None,
    debug_attn_snapshot_base: int | None,
    q_seq_base: int | None,
    q_base: int | None,
    ln2_base: int,
    residual1_base: int,
    mlp_inter_base: int,
    mlp_out_base: int,
    x_in_padded: torch.Tensor,
    x_ln1_golden: torch.Tensor | None,
    q_seq_proj_golden: torch.Tensor,
    hq: int,
    d_padded: int,
    k_tile_gqa: torch.Tensor,
    v_tile_gqa: torch.Tensor,
    scale: float,
    s_kv_valid: int | None,
    x_res1_golden: torch.Tensor,
    x_ln2_golden: torch.Tensor,
    mlp_mid_golden: torch.Tensor,
    mlp_out_golden: torch.Tensor,
    final_golden: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict:
    """Build standard per-stage encoder metrics from VRAM snapshots."""
    from transactional_emulator.testbench.siglip.utils.core import tensor_metrics
    from transactional_emulator.testbench.siglip.utils.math import gqa_sdpa
    from transactional_emulator.testbench.siglip.utils.vram import (
        load_vram_chunk_major_to_seq,
        load_vram_head_major_q_to_seq,
        load_vram_seq_major_to_seq,
    )

    stage_metrics: dict[str, dict] = {}

    def _record_metric(name: str, metric: dict, *, aliases: tuple[str, ...] = ()) -> None:
        """Record canonical stage metric name and optional backward-compatible aliases."""
        stage_metrics[name] = metric
        for alias in aliases:
            stage_metrics[alias] = metric

    if x_ln1_base is not None and x_ln1_golden is not None:
        x_ln1_sim = load_vram_chunk_major_to_seq(
            vram_bin_file,
            start_elem=x_ln1_base,
            seq_len=s_q_actual,
            hidden_dim=hidden_size_padded,
            mlen=mlen,
        )
        _record_metric(
            "stage1_ln1_out",
            tensor_metrics(x_ln1_sim, x_ln1_golden, atol=atol, rtol=rtol),
            aliases=("x_ln1",),
        )

    if q_seq_base is not None:
        q_seq_sim = load_vram_chunk_major_to_seq(
            vram_bin_file,
            start_elem=q_seq_base,
            seq_len=s_q_actual,
            hidden_dim=hidden_size_padded,
            mlen=mlen,
        )
        _record_metric(
            "stage2_q_proj_chunk_major",
            tensor_metrics(q_seq_sim, q_seq_proj_golden, atol=atol, rtol=rtol),
            aliases=("q_proj_seq",),
        )

    q_sim = None
    if q_base is not None:
        q_sim = load_vram_head_major_q_to_seq(
            vram_bin_file,
            start_elem=q_base,
            s_q=s_q_actual,
            hq=hq,
            d_padded=d_padded,
        )
        _record_metric(
            "stage2_q_proj_head_major",
            tensor_metrics(q_sim, q_seq_proj_golden, atol=atol, rtol=rtol),
            aliases=("q_proj",),
        )

    if debug_attn_snapshot_base is not None:
        attn_snapshot_sim = load_vram_seq_major_to_seq(
            vram_bin_file,
            start_elem=debug_attn_snapshot_base,
            seq_len=s_q_actual,
            hidden_dim=hidden_size_padded,
        )
        attn_golden = x_res1_golden.float() - x_in_padded[:s_q_actual].float()
        _record_metric(
            "stage3_attn_out_snapshot",
            tensor_metrics(attn_snapshot_sim, attn_golden, atol=atol, rtol=rtol),
            aliases=("attn_out_snapshot",),
        )
        if q_sim is not None:
            q_sim_heads = q_sim.reshape(s_q_actual, hq, d_padded).unsqueeze(0)
        else:
            q_sim_heads = q_seq_proj_golden.reshape(s_q_actual, hq, d_padded).unsqueeze(0)
        attn_from_sim_q = gqa_sdpa(
            q_sim_heads,
            k_tile_gqa,
            v_tile_gqa,
            scale,
            hq,
            k_tile_gqa.shape[2],
            kv_valid_len=s_kv_valid,
        ).reshape(s_q_actual, hidden_size_padded)
        _record_metric(
            "stage3_attn_from_q_replay",
            tensor_metrics(
                attn_snapshot_sim,
                attn_from_sim_q,
                atol=atol,
                rtol=rtol,
            ),
            aliases=("attn_from_sim_q_vs_hw",),
        )

    if debug_stage0_snapshot_base is not None:
        residual_saved_sim = load_vram_chunk_major_to_seq(
            vram_bin_file,
            start_elem=debug_stage0_snapshot_base,
            seq_len=s_q_actual,
            hidden_dim=hidden_size_padded,
            mlen=mlen,
        )
        _record_metric(
            "stage0_residual_saved",
            tensor_metrics(residual_saved_sim, x_in_padded[:s_q_actual], atol=atol, rtol=rtol),
            aliases=("residual_saved",),
        )

    x_res1_sim = load_vram_chunk_major_to_seq(
        vram_bin_file,
        start_elem=residual1_base,
        seq_len=s_q_actual,
        hidden_dim=hidden_size_padded,
        mlen=mlen,
    )
    _record_metric(
        "stage5_residual1_out",
        tensor_metrics(x_res1_sim, x_res1_golden, atol=atol, rtol=rtol),
        aliases=("x_res1",),
    )
    attn_sim = x_res1_sim - x_in_padded[:s_q_actual].float()
    attn_golden = x_res1_golden.float() - x_in_padded[:s_q_actual].float()
    _record_metric(
        "stage3_attn_out_derived",
        tensor_metrics(attn_sim, attn_golden, atol=atol, rtol=rtol),
        aliases=("attn_out_derived",),
    )

    x_ln2_sim = load_vram_chunk_major_to_seq(
        vram_bin_file,
        start_elem=ln2_base,
        seq_len=s_q_actual,
        hidden_dim=hidden_size_padded,
        mlen=mlen,
    )
    _record_metric(
        "stage6_ln2_out",
        tensor_metrics(x_ln2_sim, x_ln2_golden, atol=atol, rtol=rtol),
        aliases=("x_ln2",),
    )

    mlp_mid_sim = load_vram_chunk_major_to_seq(
        vram_bin_file,
        start_elem=mlp_inter_base,
        seq_len=s_q_actual,
        hidden_dim=aligned_inter_dim,
        mlen=mlen,
    )
    _record_metric(
        "stage7_mlp_mid",
        tensor_metrics(mlp_mid_sim, mlp_mid_golden, atol=atol, rtol=rtol),
        aliases=("mlp_mid",),
    )

    final_sim = load_vram_chunk_major_to_seq(
        vram_bin_file,
        start_elem=mlp_out_base,
        seq_len=s_q_actual,
        hidden_dim=hidden_size_padded,
        mlen=mlen,
    )
    _record_metric(
        "stage8_final_out",
        tensor_metrics(final_sim, final_golden, atol=atol, rtol=rtol),
        aliases=("final_out",),
    )

    residual_sim = load_vram_chunk_major_to_seq(
        vram_bin_file,
        start_elem=residual1_base,
        seq_len=s_q_actual,
        hidden_dim=hidden_size_padded,
        mlen=mlen,
    )
    mlp_out_sim = final_sim - residual_sim
    _record_metric(
        "stage7_mlp_out",
        tensor_metrics(mlp_out_sim, mlp_out_golden, atol=atol, rtol=rtol),
        aliases=("mlp_out",),
    )

    return stage_metrics


def write_summary_report(
    build_dir: Path,
    summary: dict,
    *,
    file_name: str = "summary.json",
    json_default: Callable | None = None,
) -> None:
    """Write harness summary JSON with optional custom serializer."""
    with open(build_dir / file_name, "w") as f:
        json.dump(summary, f, indent=2, default=json_default)


def write_comparison_params(
    build_dir: Path,
    *,
    start_row_idx: int,
    num_rows: int,
    num_batches: int,
    elements_per_batch: int,
    use_stride_mode: bool | None = None,
    use_slice_mode: bool | None = None,
    slice_per_row: int | None = None,
    extra_params: dict | None = None,
    file_name: str = "comparison_params.json",
) -> dict:
    """Build and write `comparison_params.json` for emulator output checks."""
    params = {
        "start_row_idx": int(start_row_idx),
        "num_rows": int(num_rows),
        "num_batches": int(num_batches),
        "elements_per_batch": int(elements_per_batch),
    }
    if use_stride_mode is not None:
        params["use_stride_mode"] = bool(use_stride_mode)
    if use_slice_mode is not None:
        params["use_slice_mode"] = bool(use_slice_mode)
    if slice_per_row is not None:
        params["slice_per_row"] = int(slice_per_row)
    if extra_params is not None:
        params.update(extra_params)

    with open(build_dir / file_name, "w") as f:
        json.dump(params, f, indent=2)
    return params


def prepare_case_artifacts(
    *,
    case_build_dir: Path,
    input_tensor: dict,
    asm_code: str,
    golden_result: dict,
    fp_preload: list[float],
    vram_preload: np.ndarray | None,
    hbm_mb: int,
    data_order: list[str],
    write_golden_txt: bool = True,
) -> None:
    """Create sim artifacts and HBM preload files for one test case."""
    case_build_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if vram_preload is not None:
        kwargs["vram_preload"] = vram_preload

    create_sim_env(
        input_tensor,
        asm_code,
        golden_result,
        fp_preload,
        build_dir=str(case_build_dir),
        write_golden_txt=write_golden_txt,
        **kwargs,
    )
    create_mem_for_sim(
        data_size=hbm_mb,
        mode="behave_sim",
        asm=None,
        data=None,
        specified_data_order=data_order,
        build_path=str(case_build_dir),
    )
    with open(case_build_dir / "generated_asm_code.asm", "w") as f:
        f.write(asm_code)


def prepare_case_and_run_emulator(
    *,
    case_build_dir: Path,
    input_tensor: dict,
    asm_code: str,
    golden_result: dict,
    fp_preload: list[float],
    vram_preload: np.ndarray | None,
    hbm_mb: int,
    data_order: list[str],
    write_hbm_size: bool = True,
    copy_vram_dump: bool = False,
    vram_dump_source: Path | None = None,
    emulator_hbm_size: int | None = None,
    emulator_log_path: Path | None = None,
) -> None:
    """Create sim artifacts, build HBM preload, run emulator, and optionally copy VRAM dump."""
    prepare_case_artifacts(
        case_build_dir=case_build_dir,
        input_tensor=input_tensor,
        asm_code=asm_code,
        golden_result=golden_result,
        fp_preload=fp_preload,
        vram_preload=vram_preload,
        hbm_mb=hbm_mb,
        data_order=data_order,
        write_golden_txt=True,
    )

    hbm_file = case_build_dir / "hbm_for_behave_sim.bin"
    if write_hbm_size and hbm_file.exists():
        preload_bytes = hbm_file.stat().st_size
        hbm_size_bytes = (((4 * preload_bytes) + 63) // 64) * 64
        (case_build_dir / "hbm_size.txt").write_text(str(int(hbm_size_bytes)))

    run_emulator(
        case_build_dir,
        hbm_size=emulator_hbm_size,
        log_path=emulator_log_path or (case_build_dir / "emulator.log"),
    )

    if copy_vram_dump:
        src = vram_dump_source
        if src is not None and src.exists():
            from shutil import copyfile

            copyfile(src, case_build_dir / "vram_dump.bin")
