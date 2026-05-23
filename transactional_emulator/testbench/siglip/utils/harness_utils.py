"""Shared harness utilities for SigLIP testbench scripts."""

from __future__ import annotations

import json
import os
import shutil
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
    max_chunks: int
    use_tensor_cache: bool
    cache_path: Path | None
    inter_dim_raw: str
    kv_valid_len_raw: str


def load_siglip_harness_run_config(
    *,
    build_dir: Path,
    mlen_default: int,
    vlen_default: int | None = None,
    q_chunk_default: int | None = None,
    max_chunks_default: int = 1,
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

    use_tensor_cache = os.environ.get("SIGLIP_USE_TENSOR_CACHE", "1") != "0"
    cache_path_env = os.environ.get("SIGLIP_TENSOR_CACHE", "").strip()
    cache_path = Path(cache_path_env) if cache_path_env else (build_dir / "siglip_tensors_cache.pt" if use_tensor_cache else None)

    max_chunks = int(os.environ.get("SIGLIP_MAX_CHUNKS", str(max_chunks_default)))
    inter_dim_raw = os.environ.get("SIGLIP_INTER_DIM", str(inter_dim_default))
    kv_valid_len_raw = os.environ.get("SIGLIP_KV_VALID_LEN", "full")

    return SiglipHarnessRunConfig(
        mlen=mlen,
        vlen=vlen,
        max_q_chunk=max_q_chunk,
        max_chunks=max_chunks,
        use_tensor_cache=use_tensor_cache,
        cache_path=cache_path,
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


def clear_chunk_dirs(build_dir: Path) -> None:
    """Delete stale per-chunk directories under a build directory."""
    for old_chunk_dir in build_dir.glob("chunk_*"):
        if old_chunk_dir.is_dir():
            shutil.rmtree(old_chunk_dir, ignore_errors=True)


def format_siglip_run_config(run_cfg: SiglipHarnessRunConfig) -> str:
    """Format the common SigLIP run-config log line."""
    return (
        "Run config: "
        f"SIGLIP_Q_CHUNK={run_cfg.max_q_chunk}, "
        f"SIGLIP_INTER_DIM={run_cfg.inter_dim_raw}, "
        f"SIGLIP_MAX_CHUNKS={run_cfg.max_chunks}"
    )


def format_siglip_extended_run_config(run_cfg: SiglipHarnessRunConfig) -> str:
    """Format SigLIP run-config log line with MLEN/VLEN and KV valid length."""
    return (
        "Run config: "
        f"SIGLIP_MLEN={run_cfg.mlen}, "
        f"SIGLIP_VLEN={run_cfg.vlen}, "
        f"SIGLIP_Q_CHUNK={run_cfg.max_q_chunk}, "
        f"SIGLIP_INTER_DIM={run_cfg.inter_dim_raw}, "
        f"SIGLIP_KV_VALID_LEN={run_cfg.kv_valid_len_raw}, "
        f"SIGLIP_MAX_CHUNKS={run_cfg.max_chunks}"
    )


def resolve_siglip_vram_dump_path() -> Path:
    """Return the canonical repo-level SigLIP VRAM dump path."""
    return Path(__file__).resolve().parents[3] / "vram_dump.bin"


def load_or_prepare_cached_tensors(
    *,
    cache_path: Path | None,
    builder: Callable[[], dict],
    label: str = "tensors",
) -> dict:
    """Load cached tensors when available, otherwise build and persist them."""
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached {label} from {cache_path} ...")
        return torch.load(cache_path, map_location="cpu")

    tensors = builder()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensors, cache_path)
        print(f"Saved {label} cache to {cache_path}")
    return tensors


def build_encoder_stage_metrics(
    *,
    vram_bin_file: Path,
    mlen: int,
    s_q_actual: int,
    hidden_size_padded: int,
    aligned_inter_dim: int,
    attn_base: int,
    debug_attn_snapshot_base: int | None,
    q_base: int,
    x_base: int,
    residual_base: int,
    mlp_inter_base: int,
    mlp_out_base: int,
    x_in_padded: torch.Tensor,
    q_seq_proj_golden: torch.Tensor,
    hq: int,
    d_padded: int,
    k_tile_gqa: torch.Tensor,
    v_tile_gqa: torch.Tensor,
    scale: float,
    s_kv_valid: int | None,
    debug_flash_tile_trace_base: int | None,
    kv_tile_size: int,
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
        load_vram_bf16,
        load_vram_chunk_major_to_seq,
        load_vram_head_major_q_to_seq,
        load_vram_seq_major_to_seq,
    )

    stage_metrics: dict[str, dict] = {}

    q_sim = load_vram_head_major_q_to_seq(
        vram_bin_file,
        start_elem=q_base,
        s_q=s_q_actual,
        hq=hq,
        d_padded=d_padded,
    )
    stage_metrics["q_proj"] = tensor_metrics(q_sim, q_seq_proj_golden, atol=atol, rtol=rtol)

    if debug_attn_snapshot_base is not None:
        attn_snapshot_sim = load_vram_seq_major_to_seq(
            vram_bin_file,
            start_elem=debug_attn_snapshot_base,
            seq_len=s_q_actual,
            hidden_dim=hidden_size_padded,
        )
        attn_golden = x_res1_golden.float() - x_in_padded[:s_q_actual].float()
        stage_metrics["attn_out_snapshot"] = tensor_metrics(attn_snapshot_sim, attn_golden, atol=atol, rtol=rtol)
        q_sim_heads = q_sim.reshape(s_q_actual, hq, d_padded).unsqueeze(0)
        attn_from_sim_q = gqa_sdpa(
            q_sim_heads,
            k_tile_gqa,
            v_tile_gqa,
            scale,
            hq,
            k_tile_gqa.shape[2],
            kv_valid_len=s_kv_valid,
        ).reshape(s_q_actual, hidden_size_padded)
        stage_metrics["attn_from_sim_q_vs_hw"] = tensor_metrics(
            attn_snapshot_sim,
            attn_from_sim_q,
            atol=atol,
            rtol=rtol,
        )

    if debug_flash_tile_trace_base is not None:
        kv_total = k_tile_gqa.shape[1] if s_kv_valid is None else s_kv_valid
        tile_count = (kv_total + kv_tile_size - 1) // kv_tile_size
        tile_stride = 2 * s_q_actual + s_q_actual * d_padded

        q_sim_head0 = q_sim.reshape(s_q_actual, hq, d_padded)[:, 0, :].to(torch.bfloat16)
        k_head0 = k_tile_gqa[0, :, 0, :].to(torch.bfloat16)
        v_head0 = v_tile_gqa[0, :, 0, :].to(torch.bfloat16)

        m_ref = torch.full((s_q_actual,), float("-inf"), dtype=torch.float32)
        m_ref_unscaled = torch.full((s_q_actual,), float("-inf"), dtype=torch.float32)
        l_ref = torch.zeros((s_q_actual,), dtype=torch.float32)
        o_ref = torch.zeros((s_q_actual, d_padded), dtype=torch.float32)

        first_bad_tile = None
        tile_o_metrics = []

        for tile_idx in range(tile_count):
            k_start = tile_idx * kv_tile_size
            k_end = min(k_start + kv_tile_size, kv_total)

            logits_unscaled = (q_sim_head0 @ k_head0[k_start:k_end, :].T).to(torch.float32)
            logits = logits_unscaled * float(scale)
            row_max = torch.max(logits, dim=1).values
            m_new = torch.maximum(m_ref, row_max)
            row_max_unscaled = torch.max(logits_unscaled, dim=1).values
            m_new_unscaled = torch.maximum(m_ref_unscaled, row_max_unscaled)
            alpha = torch.exp(m_ref - m_new)
            p = torch.exp(logits - m_new.unsqueeze(1))
            l_new = l_ref * alpha + torch.sum(p, dim=1)
            o_new = o_ref * alpha.unsqueeze(1) + (p @ v_head0[k_start:k_end, :].to(torch.float32))

            m_visible = m_new.to(torch.bfloat16).float()
            m_unscaled_visible = m_new_unscaled.to(torch.bfloat16).float()
            l_visible = l_new.to(torch.bfloat16).float()
            o_visible = o_new.to(torch.bfloat16).float()

            tile_base = debug_flash_tile_trace_base + tile_idx * tile_stride
            m_sim = load_vram_bf16(vram_bin_file, num_elements=s_q_actual, start_elem=tile_base)
            l_sim = load_vram_bf16(vram_bin_file, num_elements=s_q_actual, start_elem=tile_base + s_q_actual)
            o_sim = load_vram_bf16(
                vram_bin_file,
                num_elements=s_q_actual * d_padded,
                start_elem=tile_base + 2 * s_q_actual,
            ).reshape(s_q_actual, d_padded)

            m_metric = tensor_metrics(m_sim, m_visible, atol=atol, rtol=rtol)
            m_unscaled_metric = tensor_metrics(m_sim, m_unscaled_visible, atol=atol, rtol=rtol)
            l_metric = tensor_metrics(l_sim, l_visible, atol=atol, rtol=rtol)
            o_metric = tensor_metrics(o_sim, o_visible, atol=atol, rtol=rtol)
            tile_o_metrics.append(o_metric["match_rate"])

            stage_metrics[f"tile_{tile_idx:02d}_m"] = m_metric
            stage_metrics[f"tile_{tile_idx:02d}_m_unscaled_ref"] = m_unscaled_metric
            stage_metrics[f"tile_{tile_idx:02d}_l"] = l_metric
            stage_metrics[f"tile_{tile_idx:02d}_o"] = o_metric

            if first_bad_tile is None and (not m_metric["allclose_pass"] or not l_metric["allclose_pass"] or not o_metric["allclose_pass"]):
                first_bad_tile = tile_idx

            m_ref, m_ref_unscaled, l_ref, o_ref = m_visible, m_unscaled_visible, l_visible, o_visible

        stage_metrics["flash_tile_trace_summary"] = {
            "tile_count": tile_count,
            "first_bad_tile": first_bad_tile,
            "mean_o_match_rate": float(np.mean(tile_o_metrics)) if tile_o_metrics else None,
        }

    residual_saved_sim = load_vram_seq_major_to_seq(
        vram_bin_file,
        start_elem=residual_base,
        seq_len=s_q_actual,
        hidden_dim=hidden_size_padded,
    )
    stage_metrics["residual_saved"] = tensor_metrics(residual_saved_sim, x_in_padded[:s_q_actual], atol=atol, rtol=rtol)

    x_res1_sim = load_vram_seq_major_to_seq(
        vram_bin_file,
        start_elem=attn_base,
        seq_len=s_q_actual,
        hidden_dim=hidden_size_padded,
    )
    stage_metrics["x_res1"] = tensor_metrics(x_res1_sim, x_res1_golden, atol=atol, rtol=rtol)
    attn_sim = x_res1_sim - x_in_padded[:s_q_actual].float()
    attn_golden = x_res1_golden.float() - x_in_padded[:s_q_actual].float()
    stage_metrics["attn_out_derived"] = tensor_metrics(attn_sim, attn_golden, atol=atol, rtol=rtol)

    x_ln2_sim = load_vram_chunk_major_to_seq(
        vram_bin_file,
        start_elem=x_base,
        seq_len=s_q_actual,
        hidden_dim=hidden_size_padded,
        mlen=mlen,
    )
    stage_metrics["x_ln2"] = tensor_metrics(x_ln2_sim, x_ln2_golden, atol=atol, rtol=rtol)

    mlp_mid_sim = load_vram_chunk_major_to_seq(
        vram_bin_file,
        start_elem=mlp_inter_base,
        seq_len=s_q_actual,
        hidden_dim=aligned_inter_dim,
        mlen=mlen,
    )
    stage_metrics["mlp_mid"] = tensor_metrics(mlp_mid_sim, mlp_mid_golden, atol=atol, rtol=rtol)

    final_sim = load_vram_chunk_major_to_seq(
        vram_bin_file,
        start_elem=mlp_out_base,
        seq_len=s_q_actual,
        hidden_dim=hidden_size_padded,
        mlen=mlen,
    )
    stage_metrics["final_out"] = tensor_metrics(final_sim, final_golden, atol=atol, rtol=rtol)

    residual_sim = load_vram_chunk_major_to_seq(
        vram_bin_file,
        start_elem=residual_base,
        seq_len=s_q_actual,
        hidden_dim=hidden_size_padded,
        mlen=mlen,
    )
    mlp_out_sim = final_sim - residual_sim
    stage_metrics["mlp_out"] = tensor_metrics(mlp_out_sim, mlp_out_golden, atol=atol, rtol=rtol)

    return stage_metrics


def write_chunk_report(
    chunk_build_dir: Path,
    chunk_idx: int,
    start: int,
    end: int,
    results: dict,
    params: dict,
    *,
    stage_metrics: dict | None = None,
    json_default: Callable | None = None,
) -> None:
    """Write one chunk-level comparison report."""
    report = {
        "chunk_idx": int(chunk_idx),
        "token_start": int(start),
        "token_end": int(end),
        "comparison_params": params,
        "results": results,
    }
    if stage_metrics is not None:
        report["stage_metrics"] = stage_metrics
    with open(chunk_build_dir / "comparison_results.json", "w") as f:
        json.dump(report, f, indent=2, default=json_default)


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


def format_harness_summary_line(*, label: str, summary: dict, token_count: int) -> str:
    """Format the common end-of-run harness summary line."""
    return (
        f"{label} recorded {summary['chunk_count']} chunks for {token_count} tokens; "
        f"aggregate allclose={summary['allclose_pass']}"
    )


def format_chunk_result_line(
    *,
    chunk_idx: int,
    start: int,
    end: int,
    results: dict,
    include_mae: bool = True,
) -> str:
    """Format one chunk-level result line for harness logs."""
    line = (
        f"Chunk {int(chunk_idx)} ({int(start)}-{int(end)}): "
        f"match_rate={float(results['match_rate']):.3f}% "
        f"allclose={bool(results['allclose_pass'])}"
    )
    if include_mae and ("mae" in results) and (results["mae"] is not None):
        line += f" mae={float(results['mae']):.4f}"
    return line


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


def summarize_chunk_reports(build_dir: Path) -> dict:
    """Aggregate chunk-level reports into a compact summary."""
    chunk_reports = sorted(build_dir.glob("chunk_*/comparison_results.json"))
    if not chunk_reports:
        return {"chunk_count": 0, "allclose_pass": False}

    parsed_reports = []
    for report_path in chunk_reports:
        with open(report_path) as f:
            parsed_reports.append(json.load(f))

    return {
        "chunk_count": len(parsed_reports),
        "allclose_pass": all(r["results"]["allclose_pass"] for r in parsed_reports),
        "mean_mse": float(np.mean([r["results"]["mse"] for r in parsed_reports])),
        "mean_mae": float(np.mean([r["results"]["mae"] for r in parsed_reports])),
        "max_abs_error": float(np.max([r["results"]["max_error"] for r in parsed_reports])),
        "min_match_rate": float(np.min([r["results"]["match_rate"] for r in parsed_reports])),
    }


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
