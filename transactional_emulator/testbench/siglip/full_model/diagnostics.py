"""Layer-stage diagnostics for SigLIP full-model smoke runs."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.local_asm_templates.layout import compute_vram_layout
from transactional_emulator.testbench.siglip.utils.core import align_up
from transactional_emulator.testbench.siglip.utils.math import gqa_sdpa
from transactional_emulator.testbench.siglip.utils.vram import (
    load_vram_chunk_major_to_seq,
    load_vram_head_major_q_to_seq,
    load_vram_seq_major_to_seq,
)

__all__ = ["DIAG_ABS_BUDGET", "DIAG_REL_BUDGET", "print_layer0_stage_diagnostics"]

DIAG_ABS_BUDGET = 0.0625
DIAG_REL_BUDGET = 0.05


def _compute_persistent_end(
    *,
    vram_layout: dict,
    seq_len: int,
    hidden_size: int,
    inter_size: int,
    num_layers: int,
) -> int:
    """Compute end of persistent VRAM region before scratch/workspace allocations."""
    persistent_end = int(vram_layout["embedding_base"]) + int(vram_layout["embedding_size"])
    layer_bases = vram_layout.get("layer_bases", {})
    layer_sizes = vram_layout.get("layer_sizes", {})
    if layer_bases:
        last_layer_idx = max(layer_bases.keys())
        persistent_end = max(
            persistent_end,
            int(layer_bases[last_layer_idx]) + int(layer_sizes[last_layer_idx]),
        )

    q_bias_bases = vram_layout.get("q_bias_bases", {})
    ln1_weight_bases = vram_layout.get("ln1_weight_bases", {})
    ln1_bias_bases = vram_layout.get("ln1_bias_bases", {})
    ln2_weight_bases = vram_layout.get("ln2_weight_bases", {})
    ln2_bias_bases = vram_layout.get("ln2_bias_bases", {})
    out_bias_bases = vram_layout.get("out_bias_bases", {})
    fc1_bias_bases = vram_layout.get("fc1_bias_bases", {})
    fc2_bias_bases = vram_layout.get("fc2_bias_bases", {})

    for layer_idx in range(num_layers):
        q_bias_base = q_bias_bases.get(layer_idx)
        if q_bias_base is not None:
            persistent_end = max(persistent_end, int(q_bias_base) + seq_len * hidden_size)

        for base_dict in (ln1_weight_bases, ln1_bias_bases, ln2_weight_bases, ln2_bias_bases):
            base = base_dict.get(layer_idx)
            if base is not None:
                persistent_end = max(persistent_end, int(base) + seq_len * hidden_size)

        out_base = out_bias_bases.get(layer_idx)
        if out_base is not None:
            persistent_end = max(persistent_end, int(out_base) + seq_len * hidden_size)

        fc1_base = fc1_bias_bases.get(layer_idx)
        if fc1_base is not None:
            persistent_end = max(persistent_end, int(fc1_base) + seq_len * inter_size)

        fc2_base = fc2_bias_bases.get(layer_idx)
        if fc2_base is not None:
            persistent_end = max(persistent_end, int(fc2_base) + seq_len * hidden_size)

    return max(persistent_end, int(vram_layout.get("total_vram_elements", persistent_end)))


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
    """Print quick layer-0 stage parity metrics (x input and q projection)."""
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
    d_padded = align_up(head_dim, mlen)

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
        diff = (sim_f - ref_f).abs()
        rel = diff / ref_f.abs().clamp_min(1e-6)
        abs_pass = diff <= DIAG_ABS_BUDGET
        rel_pass = rel <= DIAG_REL_BUDGET
        within_budget = abs_pass | rel_pass
        match = within_budget.float().mean().item() * 100.0

        flat_diff = diff.reshape(-1)
        flat_rel = rel.reshape(-1)
        numel = int(flat_diff.numel())
        abs_budget_fail = int((~abs_pass).sum().item())
        rel_budget_fail = int((~rel_pass).sum().item())
        abs_budget_fail_pct = (100.0 * abs_budget_fail / numel) if numel > 0 else 0.0
        rel_budget_fail_pct = (100.0 * rel_budget_fail / numel) if numel > 0 else 0.0
        p95_abs = float(torch.quantile(flat_diff, 0.95).item()) if numel > 0 else 0.0
        p99_abs = float(torch.quantile(flat_diff, 0.99).item()) if numel > 0 else 0.0
        mean_rel = float(flat_rel.mean().item()) if numel > 0 else 0.0

        probe_diagnostics.append(
            {
                "probe": name,
                "elements": numel,
                "match_percent": match,
                "mean_abs": float(flat_diff.mean().item()) if numel > 0 else 0.0,
                "max_abs": float(flat_diff.max().item()) if numel > 0 else 0.0,
                "p95_abs": p95_abs,
                "p99_abs": p99_abs,
                "mean_rel": mean_rel,
                "abs_budget_fail": abs_budget_fail,
                "abs_budget_fail_pct": abs_budget_fail_pct,
                "rel_budget_fail": rel_budget_fail,
                "rel_budget_fail_pct": rel_budget_fail_pct,
            }
        )
        print(
            f"[diag] {name}: match={match:.3f}% "
            f"mean_abs={flat_diff.mean().item():.6f} max_abs={flat_diff.max().item():.6f} "
            f"p95_abs={p95_abs:.6f} p99_abs={p99_abs:.6f} mean_rel={mean_rel:.6f} "
            f"abs_fail={abs_budget_fail}/{numel} ({abs_budget_fail_pct:.3f}%) "
            f"rel_fail={rel_budget_fail}/{numel} ({rel_budget_fail_pct:.3f}%)"
        )

    x_ref = torch.zeros(seq_len, hidden_runtime, dtype=torch.float32)
    seq_ref = min(seq_valid, layer_outputs[0].shape[0])
    x_ref[:seq_ref, :hidden_visible] = layer_outputs[0][:seq_ref, :hidden_visible].float()

    layer0_probe_bases = vram_layout.get("layer0_probe_bases", {})
    input_probe_base = layer0_probe_bases.get("input_token_major")
    if input_probe_base is not None:
        x_sim = load_vram_seq_major_to_seq(
            vram_dump,
            start_elem=int(input_probe_base),
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

    num_layers = len(runtime_layer_weights_list)
    inter_runtime = int(runtime_config.get("intermediate_size", hidden_runtime))
    persistent_end = _compute_persistent_end(
        vram_layout=vram_layout,
        seq_len=seq_len,
        hidden_size=hidden_runtime,
        inter_size=inter_runtime,
        num_layers=num_layers,
    )

    workspace_base = align_up(persistent_end, mlen)
    q_seq_base = workspace_base
    q_vram_base = q_seq_base + seq_len * hidden_runtime

    flash_layout = compute_vram_layout(
        mlen=mlen,
        blen=blen,
        q_len=seq_len,
        hq=hq,
        hkv=int(config["num_key_value_heads"]),
        d=d_padded,
        vector_sram_base=q_vram_base,
    )
    q_base = int(flash_layout["q_base"])

    q_sim = load_vram_head_major_q_to_seq(
        vram_dump,
        start_elem=q_base,
        s_q=seq_len,
        hq=hq,
        d_padded=d_padded,
    )
    _summarize("layer0_q_proj", q_sim, q_ref)

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

        outproj_probe_base = layer0_probe_bases.get("outproj_token_major")
        if outproj_probe_base is not None:
            out_proj_sim = load_vram_seq_major_to_seq(
                vram_dump,
                start_elem=int(outproj_probe_base),
                seq_len=seq_len,
                hidden_dim=hidden_runtime,
            )
            _summarize("layer0_out_proj_out", out_proj_sim, attn_proj)

        x_res1_ref = (x_ref.to(torch.bfloat16) + attn_proj.to(torch.bfloat16)).to(torch.bfloat16).float()

        attn_base = int(flash_layout["o_old_base"])
        x_res1_sim = load_vram_seq_major_to_seq(
            vram_dump,
            start_elem=attn_base,
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
        )
        _summarize("layer0_residual1_out", x_res1_sim, x_res1_ref)

        ln2_w = layer0["ln2_weight"].float()
        ln2_b = layer0["ln2_bias"].float() if layer0.get("ln2_bias") is not None else None
        ln2_ref = F.layer_norm(x_res1_ref, (hidden_runtime,), weight=ln2_w, bias=ln2_b, eps=eps).to(torch.bfloat16).float()

        x_ln2_sim = load_vram_chunk_major_to_seq(
            vram_dump,
            start_elem=int(vram_layout["embedding_base"]),
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
            mlen=mlen,
        )
        _summarize("layer0_ln2_out", x_ln2_sim, ln2_ref)

        fc1_w = layer0["fc1_weight"].float().T
        fc1_b = layer0["fc1_bias"].float() if layer0.get("fc1_bias") is not None else None
        fc1_out = F.linear(ln2_ref, fc1_w, fc1_b).to(torch.bfloat16).float()
        gelu = _gelu_hardware_sigmoid(fc1_out).to(torch.bfloat16).float()

        fc2_w = layer0["fc2_weight"].float().T
        fc2_b = layer0["fc2_bias"].float() if layer0.get("fc2_bias") is not None else None
        fc2_out = F.linear(gelu, fc2_w, fc2_b).to(torch.bfloat16).float()
        x_final_ref = (x_res1_ref.to(torch.bfloat16) + fc2_out.to(torch.bfloat16)).to(torch.bfloat16).float()

        inter_padded = int(runtime_config["intermediate_size"])
        residual_base = attn_base + seq_len * hidden_runtime
        mlp_inter_base = residual_base + seq_len * hidden_runtime
        mlp_inter_sim = load_vram_chunk_major_to_seq(
            vram_dump,
            start_elem=mlp_inter_base,
            seq_len=seq_len,
            hidden_dim=inter_padded,
            mlen=mlen,
        )
        _summarize("layer0_mlp_inter", mlp_inter_sim, gelu)

        layer0_out_base = int(vram_layout["layer_bases"][0])
        x_final_sim = load_vram_chunk_major_to_seq(
            vram_dump,
            start_elem=layer0_out_base,
            seq_len=seq_len,
            hidden_dim=hidden_runtime,
            mlen=mlen,
        )
        fc2_out_sim = x_final_sim - x_res1_sim
        _summarize("layer0_mlp_out", fc2_out_sim, fc2_out)
        _summarize("layer0_residual2_out", x_final_sim, x_final_ref)
    except Exception as exc:
        print(f"[diag] Skipping downstream diagnostics ({exc})")

    if probe_diagnostics:
        diag_path = build_dir / "layer0_probe_diagnostics.json"
        diag_payload = {
            "seq_valid": int(seq_valid),
            "seq_len": int(seq_len),
            "abs_budget": DIAG_ABS_BUDGET,
            "rel_budget": DIAG_REL_BUDGET,
            "probes": probe_diagnostics,
        }
        diag_path.write_text(json.dumps(diag_payload, indent=2), encoding="utf-8")
        print(f"[diag] Saved probe diagnostics: {diag_path}")
