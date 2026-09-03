"""Run a published Mamba-2 checkpoint through the connected L-Compute path.

The published weights drive every layer from the embedding through the final
language-model head.  Projection, convolution, gating, normalization and the
residual are evaluated by an explicit BF16 host model.  The recurrent state
update/output of every layer is compiled to ``L_TILE``, assembled, decoded and
executed by the Rust transactional emulator.  The value Rust writes becomes
the input to the next host stage; layers are not stitched with golden values.

This is deliberately narrower than claiming that every model operation runs in
Rust.  It proves a real-weight, first-layer-to-last-layer connected recurrence
chain while the separate synthetic prefill tests exercise the complete
transactional SSD/KDA chunk programs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILER_ROOT = Path(
    os.environ.get("PLENA_COMPILER_ROOT", REPO_ROOT / "PLENA_Compiler")
).resolve()
for path in (REPO_ROOT, COMPILER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compiler.aten.plena.matrix_recurrence_lowering import (  # noqa: E402
    NEMOTRON_MAMBA,
    MatrixRecurrenceSpec,
    MatrixSramPoint,
    RecurrenceKind,
    RecurrenceLayout,
)
from transactional_emulator.testbench.aten.matrix_lcompute_recurrence_test import (  # noqa: E402
    _bf16,
    _mamba_reference,
    run_prepared_case,
)


REPO_ID = "AntonV/mamba2-130m-hf"
PINNED_SNAPSHOT = "05e8773fc4ac1cd067e8a18a5c45372ce5178405"
DEFAULT_CHECKPOINT = (
    Path("/scratch/shared/mcl123/plena/model_cache/huggingface/hub")
    / "models--AntonV--mamba2-130m-hf"
    / "snapshots"
    / PINNED_SNAPSHOT
)


def _checkpoint_path(explicit: Path | None = None) -> Path:
    candidate = explicit or Path(
        os.environ.get("PLENA_MAMBA2_130M_CHECKPOINT", DEFAULT_CHECKPOINT)
    )
    required = (candidate / "config.json", candidate / "model.safetensors")
    if not all(path.exists() for path in required):
        raise FileNotFoundError(
            f"{REPO_ID} is not available at {candidate}; expected config.json and "
            "model.safetensors (the test never downloads a checkpoint)"
        )
    return candidate.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """BF16 operands, FP32 accumulation, BF16 architectural writeback."""

    return _bf16(_bf16(value) @ _bf16(weight).T)


def _rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    value = _bf16(value)
    scale = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)
    return _bf16(_bf16(value * scale) * _bf16(weight))


def _gated_rms_norm(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    groups: int,
) -> torch.Tensor:
    value = _bf16(value)
    gate = _bf16(torch.nn.functional.silu(_bf16(gate)))
    gated = _bf16(value * gate)
    grouped = gated.reshape(groups, -1)
    scale = torch.rsqrt(grouped.square().mean(dim=-1, keepdim=True) + epsilon)
    normalized = _bf16(grouped * scale).reshape_as(gated)
    return _bf16(normalized * _bf16(weight))


def _prepare_layer(
    layer,
    hidden: torch.Tensor,
    conv_state: torch.Tensor,
    config,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    """Create the exact BF16 fields consumed by one Mamba recurrence."""

    mixer = layer.mixer
    residual = _bf16(hidden)
    normalized = _rms_norm(
        residual,
        layer.norm.weight,
        config.layer_norm_epsilon,
    )
    projected = _linear(normalized, mixer.in_proj.weight)
    inner = config.num_heads * config.head_dim
    group_state = config.n_groups * config.state_size
    z, xbc, dt_raw = projected.split(
        [inner, inner + 2 * group_state, config.num_heads],
        dim=-1,
    )

    window = torch.cat((_bf16(conv_state[:, 1:]), _bf16(xbc)[:, None]), dim=-1)
    convolved = (window * _bf16(mixer.conv1d.weight[:, 0, :])).sum(dim=-1)
    convolved = _bf16(convolved + _bf16(mixer.conv1d.bias))
    convolved = _bf16(torch.nn.functional.silu(convolved))
    x_flat, b_flat, c_flat = convolved.split(
        [inner, group_state, group_state],
        dim=-1,
    )
    x = x_flat.reshape(config.num_heads, config.head_dim)
    b = b_flat.reshape(config.n_groups, config.state_size)
    c = c_flat.reshape(config.n_groups, config.state_size)
    dt = _bf16(torch.nn.functional.softplus(_bf16(dt_raw + _bf16(mixer.dt_bias))))
    dt = _bf16(torch.clamp(dt, min=config.time_step_limit[0], max=config.time_step_limit[1]))
    a = _bf16(-torch.exp(_bf16(mixer.A_log)))
    decay = _bf16(torch.exp(dt * a))
    heads_per_group = config.num_heads // config.n_groups
    b_heads = b.repeat_interleave(heads_per_group, dim=0)
    c_heads = c.repeat_interleave(heads_per_group, dim=0)
    operands = {
        "x": _bf16(x),
        "dt": _bf16(dt),
        "a": _bf16(decay[:, None].expand(-1, config.state_size)),
        "b": _bf16(b_heads),
        "c": _bf16(c_heads),
        "d": _bf16(mixer.D),
    }
    return residual, _bf16(z), operands, _bf16(window)


def _finish_layer(
    layer,
    residual: torch.Tensor,
    gate: torch.Tensor,
    recurrence_output: torch.Tensor,
    config,
) -> torch.Tensor:
    mixed = _gated_rms_norm(
        recurrence_output.reshape(-1),
        gate,
        layer.mixer.norm.weight,
        config.layer_norm_epsilon,
        config.n_groups,
    )
    projected = _linear(mixed, layer.mixer.out_proj.weight)
    return _bf16(residual + projected)


def _topk_equal(left: torch.Tensor, right: torch.Tensor, k: int) -> bool:
    return torch.equal(left.topk(k).indices, right.topk(k).indices)


def run_real_checkpoint(
    *,
    checkpoint: Path,
    output_dir: Path,
    prompt: tuple[int, ...] = (1, 17, 42, 9),
    layers: int | None = None,
    keep_build: bool = False,
) -> dict[str, object]:
    try:
        from transformers import Mamba2ForCausalLM
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("transformers with Mamba2ForCausalLM is required") from error

    torch.set_grad_enabled(False)
    model = Mamba2ForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.float32,
        local_files_only=True,
    ).eval()
    config = model.config
    layer_count = config.num_hidden_layers if layers is None else layers
    if not 1 <= layer_count <= config.num_hidden_layers:
        raise ValueError(f"layers must be in [1, {config.num_hidden_layers}]")
    spec = MatrixRecurrenceSpec(
        name="mamba2_130m_real_checkpoint",
        kind=RecurrenceKind.MAMBA,
        heads=config.num_heads,
        row_elements=config.head_dim,
        recurrence_rows=config.state_size,
        primitives=NEMOTRON_MAMBA.primitives,
    )
    point = MatrixSramPoint()
    prompt_tensor = torch.tensor([prompt], dtype=torch.long)
    with torch.inference_mode():
        prefill = model(prompt_tensor, use_cache=True)
    cache = prefill.cache_params
    token = int(prefill.logits[0, -1].argmax())
    actual_states = [
        _bf16(cache.ssm_states[index][0]).permute(0, 2, 1).contiguous()
        for index in range(layer_count)
    ]
    reference_states = [state.clone() for state in actual_states]
    actual_conv = [
        _bf16(cache.conv_states[index][0]).contiguous()
        for index in range(layer_count)
    ]
    reference_conv = [state.clone() for state in actual_conv]

    # This independent Transformers forward checks the host-side split order,
    # gate/norm/residual chain and final head.  The BF16-vs-BF16 comparison below
    # isolates Rust L_TILE correctness; this FP32 result records the numerical
    # cost of the explicitly selected uniform-BF16 PLENA policy.
    official_logits = None
    if layer_count == config.num_hidden_layers:
        with torch.inference_mode():
            official_logits = model(
                torch.tensor([[token]], dtype=torch.long),
                cache_params=cache,
                use_cache=True,
                cache_position=torch.tensor([len(prompt)]),
            ).logits[0, -1].float()

    embedding = _bf16(model.backbone.embeddings.weight[token])
    actual_hidden = embedding.clone()
    reference_hidden = embedding.clone()
    layer_reports = []
    total_cycles = 0
    total_bank_stalls = 0
    worst_recurrence_output = 0.0
    worst_recurrence_state = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PLENA_USE_NIX_BUILD", "1")

    for layer_index in range(layer_count):
        layer = model.backbone.layers[layer_index]
        actual_residual, actual_gate, actual_operands, actual_conv_next = _prepare_layer(
            layer,
            actual_hidden,
            actual_conv[layer_index],
            config,
        )
        execution = run_prepared_case(
            spec,
            RecurrenceLayout.AFFINE,
            output_dir / "layers",
            initial_state=actual_states[layer_index],
            operands_by_token=(actual_operands,),
            point=point,
            case_name=f"layer_{layer_index:02d}",
            keep_build=keep_build,
        )
        actual_scan = execution.outputs[0]
        actual_states[layer_index] = execution.state
        actual_conv[layer_index] = actual_conv_next
        actual_hidden = _finish_layer(
            layer,
            actual_residual,
            actual_gate,
            actual_scan,
            config,
        )

        reference_residual, reference_gate, reference_operands, reference_conv_next = _prepare_layer(
            layer,
            reference_hidden,
            reference_conv[layer_index],
            config,
        )
        reference_scan, reference_states[layer_index] = _mamba_reference(
            reference_states[layer_index],
            reference_operands,
        )
        reference_conv[layer_index] = reference_conv_next
        reference_hidden = _finish_layer(
            layer,
            reference_residual,
            reference_gate,
            reference_scan,
            config,
        )

        output_error = execution.report["output_error"]["max_abs"]
        state_error = execution.report["state_error"]["max_abs"]
        worst_recurrence_output = max(worst_recurrence_output, float(output_error))
        worst_recurrence_state = max(worst_recurrence_state, float(state_error))
        cycles = int(execution.report["rust_simulation_cycles"])
        counters = execution.report["matrix_view_packet_counters"]
        total_cycles += cycles
        total_bank_stalls += int(counters.get("bank_stall_cycles", 0))
        layer_reports.append(
            {
                "layer": layer_index,
                "cycles": cycles,
                "state_max_abs_error": state_error,
                "output_max_abs_error": output_error,
                "bank_stall_cycles": counters.get("bank_stall_cycles", 0),
                "machine_words": execution.report["compiler_generated_machine_words"],
            }
        )

    actual_final = _rms_norm(
        actual_hidden,
        model.backbone.norm_f.weight,
        config.layer_norm_epsilon,
    )
    reference_final = _rms_norm(
        reference_hidden,
        model.backbone.norm_f.weight,
        config.layer_norm_epsilon,
    )
    actual_logits = _linear(actual_final, model.lm_head.weight)
    reference_logits = _linear(reference_final, model.lm_head.weight)
    hidden_error = float((actual_hidden - reference_hidden).abs().max())
    logit_error = float((actual_logits - reference_logits).abs().max())
    official_logit_error = None
    official_logit_relative_l2 = None
    official_top1_equal = None
    official_top5_equal = None
    if official_logits is not None:
        official_logit_error = float((actual_logits - official_logits).abs().max())
        official_logit_relative_l2 = float(
            torch.linalg.vector_norm(actual_logits - official_logits)
            / torch.linalg.vector_norm(official_logits).clamp_min(1e-12)
        )
        official_top1_equal = bool(actual_logits.argmax() == official_logits.argmax())
        official_top5_equal = _topk_equal(actual_logits, official_logits, 5)
    if not torch.allclose(actual_logits, reference_logits, atol=2e-2, rtol=2e-2):
        raise AssertionError(
            f"24-layer BF16 logits diverged: max_abs={logit_error:.6g}"
        )

    summary: dict[str, object] = {
        "schema_version": 1,
        "model": REPO_ID,
        "snapshot": checkpoint.name,
        "checkpoint_sha256": _sha256(checkpoint / "model.safetensors"),
        "precision": {
            "weights": "bf16",
            "activations": "bf16",
            "recurrent_state": "bf16",
            "matrix_accumulator": "fp32",
        },
        "prompt_tokens": list(prompt),
        "decode_token": token,
        "layers_executed": layer_count,
        "full_checkpoint_layer_chain": layer_count == config.num_hidden_layers,
        "rust_l_tile_layers": layer_count,
        "rust_cycles_sum": total_cycles,
        "matrix_bank_stall_cycles_sum": total_bank_stalls,
        "worst_layer_recurrence_output_max_abs_error": worst_recurrence_output,
        "worst_layer_recurrence_state_max_abs_error": worst_recurrence_state,
        "final_hidden_max_abs_error": hidden_error,
        "final_logit_max_abs_error": logit_error,
        "bf16_reference_top1_equal": bool(actual_logits.argmax() == reference_logits.argmax()),
        "bf16_reference_top5_equal": _topk_equal(actual_logits, reference_logits, 5),
        "official_fp32_logit_max_abs_error": official_logit_error,
        "official_fp32_logit_relative_l2": official_logit_relative_l2,
        "official_fp32_top1_equal": official_top1_equal,
        "official_fp32_top5_equal": official_top5_equal,
        "layer_reports": layer_reports,
        "claim_boundary": (
            "real checkpoint and first-to-last hidden flow; every recurrent core "
            "executes in Rust L_TILE, while projection/conv/norm/residual execute "
            "in an explicit host BF16 model"
        ),
        "not_claimed": (
            "the complete checkpoint does not yet execute every non-recurrent "
            "operation inside the Rust emulator"
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


@pytest.mark.slow
def test_real_checkpoint_first_to_last_layer(tmp_path: Path) -> None:
    try:
        checkpoint = _checkpoint_path()
    except FileNotFoundError as error:  # pragma: no cover - CI has no checkpoint
        pytest.skip(str(error))
    result = run_real_checkpoint(checkpoint=checkpoint, output_dir=tmp_path)
    assert result["layers_executed"] == 24
    assert result["bf16_reference_top1_equal"] is True
    assert result["bf16_reference_top5_equal"] is True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "mamba2_130m_real_checkpoint_lcompute",
    )
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--keep-build", action="store_true")
    args = parser.parse_args()
    summary = run_real_checkpoint(
        checkpoint=_checkpoint_path(args.checkpoint),
        output_dir=args.output_dir.resolve(),
        layers=args.layers,
        keep_build=args.keep_build,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
