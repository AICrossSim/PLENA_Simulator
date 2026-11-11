"""Model loading and weight formats shared by the routing captures and the accuracy runs (one GPU).

RTN formats are symmetric round-to-nearest with one FP16 scale per group of consecutive weights along the reduction
dimension K of every weight row: w8g128 (INT8, groups of 128) and w4g32 (INT4, groups of 32).
"""

import json
import math
import types
from pathlib import Path

RTN_FORMATS = {"w8g128": (8, 128), "w4g32": (4, 32)}
EXPERT_CHUNK_TOKENS = 65536


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


def fake_quantize_experts(model, fmt):
    """Overwrite the routed experts, the shared experts and the shared-expert gate with their RTN reconstruction."""
    import torch

    bits, group = RTN_FORMATS[fmt]
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
                q, s = rtn(w, bits, group)
                rec = (q * s.float()).reshape(w.shape[0], -1)[:, : w.shape[1]]
                err += float((rec - w).square().sum())
                norm += float(w.square().sum())
                mats[i].copy_(rec.to(mats.dtype))
            stats.add(err, norm, p.numel())
    if not stats.tensors:
        raise ValueError("no expert weights found")
    return stats.summary(format=fmt, bits=bits, group=group)


def dequantize_fp8(q, s):
    """BF16 values of FP8 weights [..., R, K] with FP32 block scales [..., R / br, K / bk], computed as transformers
    dequantizes an FP8 checkpoint."""
    import torch

    R, K = q.shape[-2:]
    sr, sc = s.shape[-2:]
    qq = q.to(torch.float32).reshape(-1, sr, R // sr, sc, K // sc)
    ss = s.to(torch.float32).reshape(-1, sr, sc).unsqueeze(-1).unsqueeze(2)
    return (qq * ss).to(torch.bfloat16).reshape(q.shape)


_INT8_KERNEL = {}


def dequantize_int8(q, s, out):
    """out = bf16(float32(q) * float32(s)) for int8 q [..., K] and FP16 scales s [..., K / G], with a Triton kernel."""
    if not _INT8_KERNEL:
        import triton
        import triton.language as tl

        @triton.jit
        def kernel(q_ptr, s_ptr, o_ptr, n, G: tl.constexpr, BLOCK: tl.constexpr):
            offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            q = tl.load(q_ptr + offs, mask=m, other=0).to(tl.float32)
            s = tl.load(s_ptr + offs // G, mask=m, other=0.0).to(tl.float32)
            tl.store(o_ptr + offs, (q * s).to(tl.bfloat16), mask=m)

        _INT8_KERNEL.update(kernel=kernel, cdiv=triton.cdiv)
    n, G, block = q.numel(), q.shape[-1] // s.shape[-1], 4096
    assert q.is_contiguous() and s.is_contiguous() and out.is_contiguous()
    assert q.shape == out.shape and q.shape[-1] == G * s.shape[-1]
    _INT8_KERNEL["kernel"][(_INT8_KERNEL["cdiv"](n, block),)](q, s, out, n, G=G, BLOCK=block)
    return out


class _ExpertWeights:
    """The attributes that transformers' grouped_mm_experts_forward reads from an experts module."""

    has_gate, has_bias, is_transposed = True, False, False

    def __init__(self, gate_up_proj, down_proj, num_experts):
        import torch.nn.functional as F

        self.gate_up_proj, self.down_proj, self.num_experts = gate_up_proj, down_proj, num_experts
        self.act_fn = F.silu

    def _apply_gate(self, gate_up_out):
        gate, up = gate_up_out.chunk(2, dim=-1)
        return self.act_fn(gate) * up


def compact_routed_experts(layers, fmt):
    """Quantize the routed experts of an FP8 checkpoint with RTN, starting from their BF16 dequantization. The integers
    stay on the GPU as int8 with their FP16 scales; before each use a layer's experts are reconstructed into one shared
    BF16 buffer and run with transformers' BF16 grouped_mm forward."""
    import torch
    from transformers.integrations.moe import grouped_mm_experts_forward

    bits, group = RTN_FORMATS[fmt]
    first = layers[min(layers)].experts
    gate_up = torch.empty(tuple(first.gate_up_proj.shape), dtype=torch.bfloat16, device="cuda")
    down = torch.empty(tuple(first.down_proj.shape), dtype=torch.bfloat16, device="cuda")
    weights = _ExpertWeights(gate_up, down, first.num_experts)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        dequantize_int8(self.gate_up_proj_q, self.gate_up_proj_s, gate_up)
        dequantize_int8(self.down_proj_q, self.down_proj_s, down)
        return grouped_mm_experts_forward(weights, hidden_states, top_k_index, top_k_weights)

    stats = _ErrorStats()
    with torch.inference_mode():
        for moe in layers.values():
            experts = moe.experts
            for name in ("gate_up_proj", "down_proj"):
                w8, s8 = getattr(experts, name), getattr(experts, name + "_scale_inv")
                n, R, K = w8.shape
                q_all = torch.empty((n, R, K), dtype=torch.int8, device="cuda")
                s_all = torch.empty((n, R, K // group), dtype=torch.float16, device="cuda")
                err = norm = 0.0
                for e0 in range(0, n, 32):
                    w = dequantize_fp8(w8[e0 : e0 + 32], s8[e0 : e0 + 32]).float()
                    m = w.shape[0]
                    q, s = rtn(w.reshape(m * R, K), bits, group)
                    err += float(((q * s.float()).reshape(m, R, K) - w).square().sum())
                    norm += float(w.square().sum())
                    q_all[e0 : e0 + m] = q.to(torch.int8).reshape(m, R, K)
                    s_all[e0 : e0 + m] = s.reshape(m, R, K // group)
                stats.add(err, norm, n * R * K)
                delattr(experts, name)
                delattr(experts, name + "_scale_inv")
                experts.register_buffer(name + "_q", q_all, persistent=False)
                experts.register_buffer(name + "_s", s_all, persistent=False)
            experts.forward = types.MethodType(forward, experts)
            torch.cuda.empty_cache()
    return stats.summary(format=fmt, bits=bits, group=group)


def split_large_expert_calls(layers, max_tokens=EXPERT_CHUNK_TOKENS):
    """Run the routed experts on chunks of at most max_tokens tokens. The FP8 grouped kernel addresses its operands
    with 32-bit offsets, which overflow above about 105K tokens at top-10 and width 2048 (a 128 x 1024 prefill)."""
    import torch

    for moe in layers.values():

        def forward(hidden_states, top_k_index, top_k_weights, inner=moe.experts.forward):
            T = hidden_states.shape[0]
            if T <= max_tokens:
                return inner(hidden_states, top_k_index, top_k_weights)
            chunks = range(0, T, max_tokens)
            return torch.cat(
                [
                    inner(
                        hidden_states[i : i + max_tokens],
                        top_k_index[i : i + max_tokens],
                        top_k_weights[i : i + max_tokens],
                    )
                    for i in chunks
                ]
            )

        moe.experts.forward = forward


def pin_fp8_kernel(snapshot):
    """Make transformers' FP8 layers use a local snapshot of the kernels-community/finegrained-fp8 hub kernel."""
    import kernels
    from transformers.integrations import hub_kernels

    module = kernels.get_local_kernel(Path(snapshot), "cuda")
    missing = [n for n in ("matmul_2d", "matmul_batched", "matmul_grouped") if not hasattr(module, n)]
    if missing:
        raise RuntimeError(f"{snapshot} lacks {missing}")
    hub_kernels._KERNEL_MODULE_MAPPING["finegrained-fp8"] = module


def load_qwen3next(model_dir, fmt="fp8", fp8_kernel=None):
    """Qwen3-Next-80B-A3B from its official FP8 checkpoint, run natively (FP8 weights and per-token FP8 activations;
    router, norms, embeddings and LM head in BF16). With an RTN format the routed experts are replaced by
    compact_routed_experts. Returns the model, its MoE blocks and the quantization error statistics."""
    from transformers import AutoModelForCausalLM

    if fp8_kernel:
        pin_fp8_kernel(fp8_kernel)
    model = AutoModelForCausalLM.from_pretrained(model_dir, **_from_pretrained_kwargs()).eval()
    layers = moe_layers(model.model.layers)
    stats = compact_routed_experts(layers, fmt) if fmt != "fp8" else None
    split_large_expert_calls(layers)
    return model, layers, stats
