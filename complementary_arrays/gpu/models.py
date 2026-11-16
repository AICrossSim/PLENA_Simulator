"""Model loading and the emulated expert-weight formats shared by the routing captures and the accuracy runs (one GPU).

Emulated formats reconstruct the expert weights in BF16 after quantizing them; groups and blocks run along the
reduction dimension K of every weight row:
  w8g128   INT8 symmetric round-to-nearest, one FP16 scale per 128 weights
  w4g32    INT4 symmetric round-to-nearest, one FP16 scale per 32 weights
  fp8b128  FP8 E4M3 in 128 x 128 blocks with an FP32 scale per block
  nvfp4    FP4 E2M1 in blocks of 16 with an FP8 E4M3 block scale and an FP32 scale per matrix
"""

import json
import math
import types
from pathlib import Path

RTN_FORMATS = {"w8g128": (8, 128), "w4g32": (4, 32)}
FORMATS = (*RTN_FORMATS, "fp8b128", "nvfp4")
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def read_config(model_dir):
    return json.loads((Path(model_dir) / "config.json").read_text())


def moe_workload(config):
    """Hidden size, expert widths, expert count and top-k of a checkpoint config."""
    c = config.get("text_config", config)
    return dict(
        hidden=c["hidden_size"],
        routed_width=c["moe_intermediate_size"],
        shared_width=c.get(
            "shared_expert_intermediate_size", c.get("n_shared_experts", 0) * c["moe_intermediate_size"]
        ),
        experts=c.get("num_experts", c.get("n_routed_experts")),
        topk=c["num_experts_per_tok"],
    )


def moe_layers(stack):
    """MoE blocks of a decoder stack, by layer index."""
    return {
        i: layer.mlp for i, layer in enumerate(stack) if hasattr(layer.mlp, "experts") and hasattr(layer.mlp, "gate")
    }


def _from_pretrained_kwargs():
    import torch

    return dict(
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    )


def load_bf16_moe(model_dir, kind):
    """Qwen3.5-35B-A3B (kind 'qwen35') or DeepSeek-V2-Lite-Chat ('deepseek') in BF16; an FP8 checkpoint is
    dequantized to BF16 on load. Returns the model and its MoE blocks."""
    from transformers import AutoModelForCausalLM

    kwargs = _from_pretrained_kwargs()
    if kind == "qwen35":
        from transformers import FineGrainedFP8Config, Qwen3_5MoeForConditionalGeneration

        config = read_config(model_dir)
        if "quantization_config" in config:
            quant = dict(config["quantization_config"], dequantize=True)
            kwargs["quantization_config"] = FineGrainedFP8Config.from_dict(quant)
        model = Qwen3_5MoeForConditionalGeneration.from_pretrained(model_dir, **kwargs).eval()
        return model, moe_layers(model.model.language_model.layers)
    model = AutoModelForCausalLM.from_pretrained(model_dir, **kwargs).eval()
    return model, moe_layers(model.model.layers)


def use_vllm_experts(layers, tolerance=0.04):
    """Run every routed-experts module through vLLM's fused_experts after comparing the two on a random input.
    Returns the relative L2 difference per layer."""
    import torch
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

    def forward(self, x, ids, scores):
        return fused_experts(x, self.gate_up_proj, self.down_proj, scores, ids, inplace=False)

    errors = {}
    with torch.inference_mode():
        for li, moe in layers.items():
            experts = moe.experts
            x = torch.randn(2, experts.gate_up_proj.shape[-1], device="cuda", dtype=torch.bfloat16) * 0.1
            _, scores, ids = moe.gate(x)
            reference = experts(x, ids, scores).float()
            fused = fused_experts(x, experts.gate_up_proj, experts.down_proj, scores, ids, inplace=False).float()
            error = float((fused - reference).norm() / reference.norm().clamp_min(1e-8))
            if not (torch.isfinite(fused).all() and error < tolerance):
                raise RuntimeError(f"fused_experts differs from the model on layer {li}: relative L2 {error}")
            errors[li] = error
            experts.forward = types.MethodType(forward, experts)
    return errors


def rtn(w, bits, group):
    """RTN of float32 rows w [R, K]: integer-valued float32 q [R, G, group] and FP16 scales [R, G, 1], with K padded
    with zeros to G * group."""
    import torch
    import torch.nn.functional as F

    R, K = w.shape
    qmax = 2 ** (bits - 1) - 1
    G = -(-K // group)
    wp = F.pad(w, (0, G * group - K)).reshape(R, G, group)
    s = (wp.abs().amax(-1, keepdim=True) / qmax).clamp_min(2**-24).half()
    if not torch.isfinite(s).all():
        raise ValueError("FP16 scale overflow")
    return torch.round(wp / s.float()).clamp(-qmax, qmax), s


class _ErrorStats:
    """Relative L2 reconstruction error, overall and of the worst tensor."""

    def __init__(self):
        self.tensors = self.elements = 0
        self.err = self.norm = self.worst = 0.0

    def add(self, err, norm, elements):
        self.tensors += 1
        self.elements += elements
        self.err += err
        self.norm += norm
        self.worst = max(self.worst, math.sqrt(err / max(norm, 1e-30)))

    def summary(self, **extra):
        rel = math.sqrt(self.err / max(self.norm, 1e-30))
        return dict(tensors=self.tensors, elements=self.elements, rel_l2=rel, max_rel_l2=self.worst, **extra)


def e4m3(v):
    import torch

    return v.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


def grid_round(v, grid):
    """Nearest value of a non-negative grid with v's sign (ties to the smaller magnitude)."""
    import torch

    g = torch.tensor(grid, device=v.device, dtype=torch.float32)
    idx = torch.bucketize(v.abs().clamp(max=float(g[-1])), (g[1:] + g[:-1]) / 2)
    return torch.sign(v) * g[idx]


def reconstruct(w, fmt):
    """BF16-bound float32 reconstruction of float32 rows w [R, K] in an emulated format."""
    import torch.nn.functional as F

    R, K = w.shape
    if fmt in RTN_FORMATS:
        q, s = rtn(w, *RTN_FORMATS[fmt])
        return (q * s.float()).reshape(R, -1)[:, :K]
    if fmt == "fp8b128":
        RR, KK = -(-R // 128) * 128, -(-K // 128) * 128
        wp = F.pad(w, (0, KK - K, 0, RR - R)).reshape(RR // 128, 128, KK // 128, 128)
        s = (wp.abs().amax(dim=(1, 3), keepdim=True) / 448.0).clamp_min(1e-30)
        return (e4m3(wp / s) * s).reshape(RR, KK)[:R, :K]
    if fmt == "nvfp4":
        import torch

        G = -(-K // 16)
        wp = F.pad(w, (0, G * 16 - K)).reshape(R, G, 16)
        gs = (w.abs().amax() / (448.0 * 6.0)).clamp_min(1e-30)
        bs = e4m3(wp.abs().amax(-1, keepdim=True) / 6.0 / gs) * gs
        bs = torch.where(bs > 0, bs, torch.ones_like(bs))
        return (grid_round(wp / bs, E2M1) * bs).reshape(R, -1)[:, :K]
    raise ValueError(fmt)


def fake_quantize_experts(model, fmt):
    """Overwrite the routed experts, the shared experts and the shared-expert gate with their reconstruction in an
    emulated format; router, attention, norms, embeddings and LM head keep the reference weights."""
    import torch

    stats = _ErrorStats()
    with torch.inference_mode():
        for name, p in model.named_parameters():
            parts = (".experts.", ".shared_expert.", ".shared_experts.", ".shared_expert_gate.")
            if p.ndim < 2 or ".mlp." not in name or not any(s in name for s in parts):
                continue
            mats = p.reshape(-1, p.shape[-2], p.shape[-1]) if p.ndim == 3 else p.reshape(1, -1, p.shape[-1])
            err = norm = 0.0
            for i in range(mats.shape[0]):
                w = mats[i].float()
                rec = reconstruct(w, fmt)
                if not torch.isfinite(rec).all():
                    raise ValueError(f"non-finite reconstruction of {name}")
                err += float((rec - w).square().sum())
                norm += float(w.square().sum())
                mats[i].copy_(rec.to(mats.dtype))
            stats.add(err, norm, p.numel())
    if not stats.tensors:
        raise ValueError("no expert weights found")
    return stats.summary(format=fmt)
