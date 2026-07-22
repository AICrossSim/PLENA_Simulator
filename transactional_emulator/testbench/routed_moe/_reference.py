"""Generate (and cache) the GPT-OSS layer reference bundle for the model-backed
routed-MoE tests.

The bundle used by gpt_oss_real_layer0_test / gpt_oss_moe_gather_scatter_test /
gpt_oss_router_gemm_test was previously an out-of-band artifact with no in-repo
generator: the tests only ever ``torch.load`` it. This module reproduces it
from the real ``openai/gpt-oss-20b`` checkpoint (already in the local HF cache),
so the tests become self-contained — the reference is generated on first use
and cached at the same path afterwards.

Bundle schema (all CPU tensors):
    x               : [rows, hidden]      bf16   MoE input hidden states
    topk_indices    : [rows, top_k]       int64  Golden-A routing (== HF routing)
    topk_weights    : [rows, top_k]       bf16   Golden-A softmax route weights
    hf_output       : [rows, hidden]      bf16   real HF GptOssMLP(x) output
    golden_a_output : [rows, hidden]      bf16   independent high-precision Golden-A output

``golden_a_output`` is computed independently (not copied from ``hf_output``);
the consuming test asserts it tracks ``hf_output`` to within a small relative
RMS, which cross-checks both the Golden-A expert math and its routing.
"""

from __future__ import annotations

from pathlib import Path

import torch

_REPO_ID = "openai/gpt-oss-20b"


def build_reference_bundle(*, layer_index: int, rows: int, seed: int) -> dict:
    """Compute a fresh reference bundle from the real gpt-oss-20b checkpoint.

    Requires the checkpoint shard for ``layer_index`` to be present in the local
    HF cache (set ``HF_HOME`` / ``HF_HUB_CACHE``); it is never downloaded here.
    """
    # Imported lazily so this module stays importable (and cheap) even when the
    # compiler package / transformers / the checkpoint are unavailable.
    from aten.models.gpt_oss.moe_reference import gpt_oss_moe_golden_a
    from aten.models.gpt_oss.real_layer_utils import build_hf_mlp, load_layer_tensors
    from transformers import AutoConfig

    tensors = load_layer_tensors(layer_index)
    config = AutoConfig.from_pretrained(_REPO_ID, local_files_only=True)

    torch.manual_seed(seed)
    x = torch.randn(rows, config.hidden_size, dtype=torch.float32).to(torch.bfloat16)

    # Routing: high-precision Golden-A router (topk over BF16 logits + softmax).
    # router_weight is stored HF-style [experts, hidden]; the reference math
    # wants [hidden, experts]. The router is deterministic, so this matches the
    # routing HF applies internally for the same x/weights.
    routing = gpt_oss_moe_golden_a(
        x,
        tensors["router_weight"].T,
        tensors["router_bias"],
        tensors["gate_up_weight"],
        tensors["gate_up_bias"],
        tensors["down_weight"],
        tensors["down_bias"],
        experts_per_token=config.num_experts_per_tok,
    )

    # Real HF GptOssMLP BF16 forward — the ground-truth output. HF expects a
    # [batch, seq, hidden] input.
    mlp = build_hf_mlp(config, tensors)
    with torch.no_grad():
        hf_out = mlp(x.unsqueeze(0))
    hf_output = (hf_out[0] if isinstance(hf_out, tuple) else hf_out).squeeze(0).to(torch.bfloat16)

    # Golden A is the *independent* high-precision MoE reference (BF16 MXFP4-dequant
    # expert math over the Golden-A routing), NOT a copy of hf_output. The consuming
    # test anchors on it agreeing with HF to a small relative RMS, which only holds
    # if the Golden-A routing matches HF's internal routing for this sample — so the
    # anchor genuinely cross-checks the expert math and the routing, rather than a
    # tautological hf == hf.clone().
    golden_a_output = routing.output.to(torch.bfloat16)

    return {
        "x": x,
        "topk_indices": routing.topk_indices.to(torch.int64),
        "topk_weights": routing.topk_weights.to(torch.bfloat16),
        "hf_output": hf_output,
        "golden_a_output": golden_a_output,
        "layer_index": layer_index,
        "rows": rows,
        "seed": seed,
    }


_REQUIRED_KEYS = (
    "x",
    "topk_indices",
    "topk_weights",
    "hf_output",
    "golden_a_output",
    "layer_index",
    "rows",
    "seed",
)


def _validate_bundle(bundle, *, layer_index: int, rows: int, seed: int) -> str:
    """Return "" if ``bundle`` matches the requested (layer_index, rows, seed) and
    carries the full schema, else a human-readable reason it is stale/incompatible."""
    if not isinstance(bundle, dict):
        return f"expected a dict, got {type(bundle).__name__}"
    missing = [k for k in _REQUIRED_KEYS if k not in bundle]
    if missing:
        return f"missing keys {missing} (older schema)"
    for key, want in (("layer_index", layer_index), ("rows", rows), ("seed", seed)):
        got = bundle.get(key)
        if got != want:
            return f"{key}={got!r} but caller requested {want!r}"
    x = bundle["x"]
    if getattr(x, "shape", (None,))[0] != rows:
        return f"x has {getattr(x, 'shape', '?')} rows but rows={rows}"
    return ""


def ensure_reference(path, *, layer_index: int, rows: int, seed: int) -> dict:
    """Return the reference bundle at ``path``, generating and caching it first if
    the file is absent, stale, or incompatible with the requested parameters.

    A cached file is trusted only after ``_validate_bundle`` confirms its stored
    ``layer_index``/``rows``/``seed`` and schema match this call. Anything else
    (a bundle from a different sample, an older schema, or an unreadable file) is
    regenerated from the checkpoint — otherwise a stale artifact at ``path`` would
    be loaded blindly and silently validate the wrong sample.
    """
    path = Path(path).expanduser().resolve()
    if path.exists():
        try:
            bundle = torch.load(path, map_location="cpu")
        except Exception as exc:  # any load failure => regenerate from checkpoint
            print(f"[reference] {path} failed to load ({exc}); regenerating...")
        else:
            reason = _validate_bundle(bundle, layer_index=layer_index, rows=rows, seed=seed)
            if not reason:
                return bundle
            print(f"[reference] {path} is stale/incompatible ({reason}); regenerating...")

    print(f"[reference] generating from {_REPO_ID} (layer={layer_index}, rows={rows}, seed={seed}) -> {path}...")
    bundle = build_reference_bundle(layer_index=layer_index, rows=rows, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)
    print(f"[reference] wrote {path}")
    return bundle
