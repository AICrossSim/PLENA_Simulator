"""Layer-stage diagnostics for SigLIP full-model smoke runs."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.utils.math import gqa_sdpa
from transactional_emulator.testbench.siglip.utils.vram import (
    load_vram_chunk_major_to_seq,
    load_vram_seq_major_to_seq,
)

__all__ = ["DIAG_ATOL", "DIAG_RTOL", "DIAG_ABS_BUDGET", "DIAG_REL_BUDGET", "print_layer0_stage_diagnostics"]

DIAG_ATOL = 0.2
DIAG_RTOL = 0.2

# Backward-compatible aliases.
DIAG_ABS_BUDGET = DIAG_ATOL
DIAG_REL_BUDGET = DIAG_RTOL


def _gelu_hardware_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Hardware GELU approximation used by gelu_asm: x * sigmoid(1.702 * x)."""
    return x * torch.sigmoid(1.702 * x)


def print_layer0_stage_diagnostics(
    *,
    build_dir: Path,
    config: dict,
    runtime_config: dict,
    vram_layout: dict,
    runtime_layer_weights_list: list,
    layer_outputs: list[torch.Tensor],
    mlen: int,
    blen: int,
) -> None:
    """Print layer-0 parity metrics from direct emitted probes/buffers only."""
    if not runtime_layer_weights_list or not layer_outputs:
        return

    emulator_root = Path(__file__).resolve().parents[3]
    vram_dump = emulator_root / "vram_dump.bin"
    if not vram_dump.exists():
        print(f"[diag] Skipping stage diagnostics (missing {vram_dump})")
        return

    seq_len = int(vram_layout["seq_len"])
    hidden_visible = int(config["hidden_size"])
    hidden_runtime = int(runtime_config["hidden_size"])
    hq = int(config["num_attention_heads"])
    head_dim = hidden_visible // hq
    eps = float(config.get("layer_norm_eps", 1e-2))
    d_padded = ((head_dim + mlen - 1) // mlen) * mlen

    seq_len_valid_cfg = int(runtime_config.get("seq_len_valid", seq_len))
    seq_valid = max(0, min(seq_len, seq_len_valid_cfg))
    if seq_valid == 0:
        print("[diag] Skipping stage diagnostics (seq_len_valid=0)")
        return
    print(f"[diag] using valid tokens only: {seq_valid}/{seq_len}")

    probe_diagnostics: list[dict[str, float | int | str]] = []

    def _summarize(name: str, sim: torch.Tensor, ref: torch.Tensor) -> None:
        sim_f = sim[:seq_valid].float()
        ref_f = ref[:seq_valid].float()
        diff = (sim_f - ref_f).abs().reshape(-1)
        ref_flat = ref_f.abs().reshape(-1)
        numel = int(diff.numel())
        if numel == 0:
            return

        # Keep metric semantics consistent with check_mem.compare_vram_with_golden.
        relative_errors = torch.where(ref_flat > 1e-10, diff / ref_flat, diff)
        within_relative_tolerance = relative_errors <= DIAG_RTOL
        relative_match_rate = float(within_relative_tolerance.float().mean().item() * 100.0)

        tolerance_threshold = DIAG_ATOL + DIAG_RTOL * ref_flat
        within_tolerance = diff <= tolerance_threshold
        allclose_match_rate = float(within_tolerance.float().mean().item() * 100.0)
        allclose_pass = bool(allclose_match_rate >= 90.0)

        mse = float(torch.mean((sim_f.reshape(-1) - ref_f.reshape(-1)) ** 2).item())
        mae = float(torch.mean(diff).item())
        max_error = float(torch.max(diff).item())
        mean_relative_error = float(torch.mean(relative_errors).item())

        probe_diagnostics.append(
            {
                "probe": name,
                "elements": numel,
                "mse": mse,
                "mae": mae,
                "max_error": max_error,
                "relative_error": mean_relative_error,
                "relative_match_rate": relative_match_rate,
                "allclose_match_rate": allclose_match_rate,
                "match_rate": allclose_match_rate,
                "allclose_pass": allclose_pass,
                "atol": DIAG_ATOL,
                "rtol": DIAG_RTOL,
            }
        )
        allclose_status = "PASS" if allclose_pass else "FAIL"
        print(
            f"[diag] {name}: "
            f"mse={mse:.6e} mae={mae:.6e} max_error={max_error:.6f} "
            f"relative_error={mean_relative_error:.6f} "
            f"relative_match_rate={relative_match_rate:.3f}% "
            f"allclose_match_rate={allclose_match_rate:.3f}% "
            f"allclose={allclose_status} atol={DIAG_ATOL} rtol={DIAG_RTOL}"
        )

    x_ref = torch.zeros(seq_len, hidden_runtime, dtype=torch.float32)
    seq_ref = min(seq_valid, layer_outputs[0].shape[0])
    x_ref[:seq_ref, :hidden_visible] = layer_outputs[0][:seq_ref, :hidden_visible].float()

    layer0_probe_bases = vram_layout.get("layer0_probe_bases", {})
    input_probe_base_chunk = layer0_probe_bases.get("input_chunk_major")
    input_probe_base_token = layer0_probe_bases.get("input_token_major")
    if input_probe_base_chunk is not None:
        x_sim = load_vram_chunk_major_to_seq(
            vram_dump,
            start_elem=int(input_probe_base_chunk),
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
            mlen=mlen,
        )
    elif input_probe_base_token is not None:
        x_sim = load_vram_seq_major_to_seq(
            vram_dump,
            start_elem=int(input_probe_base_token),
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
        )
    else:
        x_sim = load_vram_chunk_major_to_seq(
            vram_dump,
            start_elem=int(vram_layout["embedding_base"]),
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
            mlen=mlen,
        )
    _summarize("layer0_embedding_out", x_sim, x_ref)

    layer0 = runtime_layer_weights_list[0]
    ln1_w = layer0["ln1_weight"].float()
    ln1_b = layer0["ln1_bias"].float() if layer0.get("ln1_bias") is not None else None
    ln1_ref = F.layer_norm(x_ref, (hidden_runtime,), weight=ln1_w, bias=ln1_b, eps=eps)

    q_ref = ln1_ref @ layer0["q_proj_weight"].float()
    q_bias = layer0.get("q_proj_bias")
    if q_bias is not None:
        q_ref = q_ref + q_bias.float()

    try:
        k_tile_flat = torch.load(build_dir / "layer_0_k_proj_weight.pt").reshape(-1).float()
        v_tile_flat = torch.load(build_dir / "layer_0_v_proj_weight.pt").reshape(-1).float()
        kv_elems = int(config["num_key_value_heads"]) * seq_len * d_padded
        k_heads = k_tile_flat[:kv_elems].reshape(int(config["num_key_value_heads"]), seq_len, d_padded)
        v_heads = v_tile_flat[:kv_elems].reshape(int(config["num_key_value_heads"]), seq_len, d_padded)

        q_heads = q_ref.reshape(1, seq_len, hq, d_padded)
        k_heads_b = k_heads.permute(1, 0, 2).reshape(1, seq_len, int(config["num_key_value_heads"]), d_padded)
        v_heads_b = v_heads.permute(1, 0, 2).reshape(1, seq_len, int(config["num_key_value_heads"]), d_padded)
        attn_scale = 1.0 / float(d_padded) ** 0.5

        attn_out = gqa_sdpa(
            q_heads,
            k_heads_b,
            v_heads_b,
            scale=attn_scale,
            hq=hq,
            hkv=int(config["num_key_value_heads"]),
            kv_valid_len=int(runtime_config.get("seq_len_valid", seq_len)),
        )
        attn_out = attn_out.reshape(seq_len, hidden_runtime).to(torch.bfloat16).float()

        attn_probe_base = layer0_probe_bases.get("attn_token_major")
        if attn_probe_base is not None:
            attn_out_sim = load_vram_seq_major_to_seq(
                vram_dump,
                start_elem=int(attn_probe_base),
                seq_len=seq_len,
                hidden_dim=hidden_runtime,
            )
            _summarize("layer0_attn_out", attn_out_sim, attn_out)

        out_w = layer0["out_proj_weight"].float()
        out_b = layer0["out_proj_bias"].float() if layer0.get("out_proj_bias") is not None else None
        attn_proj = F.linear(attn_out, out_w, out_b).to(torch.bfloat16).float()
        x_res1_ref = (x_ref.to(torch.bfloat16) + attn_proj.to(torch.bfloat16)).to(torch.bfloat16).float()

        outproj_probe_base_chunk = layer0_probe_bases.get("outproj_chunk_major")
        outproj_probe_base_token = layer0_probe_bases.get("outproj_token_major")
        if outproj_probe_base_chunk is not None:
            out_proj_sim = load_vram_chunk_major_to_seq(
                vram_dump,
                start_elem=int(outproj_probe_base_chunk),
                seq_len=seq_len,
                hidden_dim=hidden_runtime,
                mlen=mlen,
            )
            _summarize("layer0_out_proj_plus_residual_out", out_proj_sim, x_res1_ref)
        elif outproj_probe_base_token is not None:
            out_proj_sim = load_vram_seq_major_to_seq(
                vram_dump,
                start_elem=int(outproj_probe_base_token),
                seq_len=seq_len,
                hidden_dim=hidden_runtime,
            )
            _summarize("layer0_out_proj_plus_residual_out", out_proj_sim, x_res1_ref)

        ln2_w = layer0["ln2_weight"].float()
        ln2_b = layer0["ln2_bias"].float() if layer0.get("ln2_bias") is not None else None
        ln2_ref = F.layer_norm(x_res1_ref, (hidden_runtime,), weight=ln2_w, bias=ln2_b, eps=eps).to(torch.bfloat16).float()

        fc1_w = layer0["fc1_weight"].float().T
        fc1_b = layer0["fc1_bias"].float() if layer0.get("fc1_bias") is not None else None
        fc1_out = F.linear(ln2_ref, fc1_w, fc1_b).to(torch.bfloat16).float()
        gelu = _gelu_hardware_sigmoid(fc1_out).to(torch.bfloat16).float()

        fc2_w = layer0["fc2_weight"].float().T
        fc2_b = layer0["fc2_bias"].float() if layer0.get("fc2_bias") is not None else None
        fc2_out = F.linear(gelu, fc2_w, fc2_b).to(torch.bfloat16).float()
        x_final_ref = (x_res1_ref.to(torch.bfloat16) + fc2_out.to(torch.bfloat16)).to(torch.bfloat16).float()

        layer0_out_base = int(vram_layout["layer_bases"][0])
        x_final_sim = load_vram_chunk_major_to_seq(
            vram_dump,
            start_elem=layer0_out_base,
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
            mlen=mlen,
        )
        _summarize("layer0_residual2_out", x_final_sim, x_final_ref)
    except Exception as exc:
        print(f"[diag] Skipping downstream diagnostics ({exc})")

    if probe_diagnostics:
        diag_path = build_dir / "layer0_probe_diagnostics.json"
        diag_payload = {
            "seq_valid": int(seq_valid),
            "seq_len": int(seq_len),
            "atol": DIAG_ATOL,
            "rtol": DIAG_RTOL,
            "probes": probe_diagnostics,
        }
        diag_path.write_text(json.dumps(diag_payload, indent=2), encoding="utf-8")
        print(f"[diag] Saved probe diagnostics: {diag_path}")
