"""Expert-weight formats of the B200 measurements in vLLM and SGLang, and the model patches of the ablation runs.

Only the routed experts and the shared expert change format; router, attention, embeddings and LM head stay BF16. No
quantized checkpoint is written: the BF16 (DeepSeek-V2-Lite-Chat) or FP8-dequantized (Qwen3.5-35B-A3B-FP8) weights
are converted while the engine loads them, by the engine's online quantization or by a patched weight iterator.

vLLM formats:
  bf16           BF16 (the MoE-zeroed baseline runs in this format)
  fp8_w8a8       FP8 E4M3 weights in 128 x 128 blocks, dynamic FP8 activations per token and 128-group (online)
  fp8_w8a16      the same weights with BF16 activations, Marlin
  nvfp4_w4a4     NVFP4 routed experts, NVFP4 activations with per-token scales (online); shared expert BF16
  nvfp4_w4a4_sh  nvfp4_w4a4 with the shared expert in NVFP4 too (dense NVFP4 method below)
  nvfp4_w4a16    weight-only NVFP4 on routed and shared experts (compressed-tensors NVFP4A16 written while loading)
SGLang formats: bf16; fp8_w8a8 (block FP8 path: Qwen3.5 keeps its checkpoint's expert tensors, DeepSeek is quantized
while loading); nvfp4_w4a4 (nvfp4_online, shared expert BF16); nvfp4_w4a16 (nvfp4_online with the FlashInfer CuteDSL
W4A16 MoE runner, routed experts only).
"""

import json
import os
import re
from pathlib import Path

LAYERS = {"qwen35": 40, "deepseek": 26}
EXPERT_WEIGHT = {
    "deepseek": r"^model\.layers\.\d+\.mlp\.(experts\.\d+|shared_experts)\.(gate|up|down)_proj\.weight$",
    "qwen35": r"^model\.language_model\.layers\.\d+\.mlp\.(experts\.\d+|shared_expert)\.(gate|up|down)_proj\.weight$",
}
ROUTED = r"\.mlp\.experts\.\d+\."
ROUTED_PATTERN = r"re:.*mlp\.experts.*"
SHARED_PATTERN = {"deepseek": r"re:.*mlp\.shared_experts\..*", "qwen35": r"re:.*mlp\.shared_expert\..*"}
CT_IGNORE = {  # compressed-tensors: layers that stay BF16
    "deepseek": [
        "lm_head",
        "re:.*self_attn.*",
        r"re:.*mlp\.gate$",
        r"re:.*layers\.0\.mlp\.(gate_up_proj|gate_proj|up_proj|down_proj)$",
    ],
    "qwen35": [
        "lm_head",
        "re:.*visual.*",
        "re:.*self_attn.*",
        "re:.*linear_attn.*",
        r"re:.*mlp\.gate$",
        r"re:.*shared_expert_gate$",
        "re:.*mtp.*",
    ],
}
SGLANG_FP8_IGNORE = {  # SGLang block FP8: dotted-path matches that stay BF16
    "deepseek": ["self_attn", "layers.0.mlp", "mlp.gate", "lm_head"],
    "qwen35": ["self_attn", "linear_attn", "shared_expert_gate", "mlp.gate", "lm_head", "visual", "mtp"],
}
NO_DEEP_GEMM = dict(VLLM_USE_DEEP_GEMM="0", VLLM_MOE_USE_DEEP_GEMM="0")  # DeepGEMM's JIT needs nvcc >= 12.9
VLLM = {
    "bf16": dict(),
    "fp8_w8a8": dict(online="fp8_per_block", env=NO_DEEP_GEMM),
    "fp8_w8a16": dict(online="fp8_per_block", moe_backend="marlin", linear_backend="marlin", env=NO_DEEP_GEMM),
    "nvfp4_w4a4": dict(quantization="nvfp4_per_token"),
    "nvfp4_w4a4_sh": dict(online="nvfp4_all"),
    "nvfp4_w4a16": dict(compressed_tensors="nvfp4"),
}
SGLANG = ("bf16", "fp8_w8a8", "nvfp4_w4a4", "nvfp4_w4a16")


def link_model_dir(src, dst, edit):
    """A directory with the checkpoint's files linked and config.json rewritten by edit(config)."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name != "config.json" and not f.name.startswith(".") and not (dst / f.name).exists():
            (dst / f.name).symlink_to(f.resolve())
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("quantization_config", None)
    if isinstance(cfg.get("text_config"), dict):
        cfg["text_config"].pop("quantization_config", None)
    edit(cfg)
    (dst / "config.json").write_text(json.dumps(cfg, indent=1))
    return str(dst)


def ct_nvfp4_config(kind):
    w = dict(
        num_bits=4,
        type="float",
        symmetric=True,
        strategy="tensor_group",
        group_size=16,
        dynamic=False,
        actorder=None,
        observer="minmax",
        observer_kwargs={},
        block_structure=None,
    )
    fmt = "nvfp4-pack-quantized"
    return dict(
        quant_method="compressed-tensors",
        format=fmt,
        quantization_status="compressed",
        global_compression_ratio=None,
        config_groups=dict(
            group_0=dict(targets=["Linear"], weights=w, input_activations=None, output_activations=None, format=fmt)
        ),
        ignore=CT_IGNORE[kind],
        kv_cache_scheme=None,
        sparsity_config={},
    )


def vllm_model_dir(model, kind, mode, work):
    def edit(cfg):
        if VLLM[mode].get("compressed_tensors"):
            cfg["quantization_config"] = ct_nvfp4_config(kind)

    return link_model_dir(model, Path(work) / "modeldirs" / f"vllm_{kind}_{mode}", edit)


def sglang_model_dir(model, kind, mode, work):
    def edit(cfg):
        if mode == "fp8_w8a8":
            cfg["quantization_config"] = dict(
                quant_method="fp8",
                activation_scheme="dynamic",
                weight_block_size=[128, 128],
                fmt="e4m3",
                ignored_layers=SGLANG_FP8_IGNORE[kind],
            )

    return link_model_dir(model, Path(work) / "modeldirs" / f"sglang_{kind}_{mode}", edit)


def vllm_engine_kwargs(kind, mode, kw):
    """Add the quantization arguments of a vLLM format to engine kwargs."""
    v = VLLM[mode]
    if "quantization" in v:
        kw["quantization"] = v["quantization"]
    if "online" in v:
        kw["quantization"] = "online"
        kw["quantization_config"] = dict(targets={ROUTED_PATTERN: v["online"], SHARED_PATTERN[kind]: v["online"]})
    for k in ("moe_backend", "linear_backend"):
        if k in v:
            kw[k] = v[k]
    return kw


def sglang_engine_kwargs(kind, mode, kw):
    if mode in ("nvfp4_w4a4", "nvfp4_w4a16"):
        kw["quantization"] = "nvfp4_online"
    if mode == "nvfp4_w4a16":
        kw["moe_runner_backend"] = "flashinfer_cutedsl"
    if kind == "deepseek":
        kw["disable_shared_experts_fusion"] = True
    return kw


def set_env(stack, mode):
    if stack == "vllm":
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"  # engine core in this process, so the patches apply
        os.environ.update(VLLM[mode].get("env", {}))
    elif mode == "nvfp4_w4a16":
        os.environ["SGLANG_FLASHINFER_CUTEDSL_NVFP4_W4A16"] = "1"


# ---------------------------------------------------------------- weight conversion while loading
def dequant_fp8_block(w, s, block=128):
    import torch

    n, k = w.shape
    full = s.float().repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return (w.float() * full).to(torch.bfloat16)


def quant_fp8_block(w, block=128):
    import torch

    n, k = w.shape
    N, K = -(-n // block) * block, -(-k // block) * block
    wp = torch.nn.functional.pad(w.float(), (0, K - k, 0, N - n)).view(N // block, block, K // block, block)
    s = (wp.abs().amax(dim=(1, 3), keepdim=True) / 448.0).clamp_min(1e-12)
    q = (wp / s).clamp(-448, 448).to(torch.float8_e4m3fn).view(N, K)[:n, :k].contiguous()
    return q, s.view(N // block, K // block).float()


def pack_fp4(x):
    """compressed-tensors FP4 codes (nearest E2M1 value, ties to the smaller magnitude, sign in bit 3, element 2j in the
    low nibble) without its [rows, cols, 8] temporary."""
    import torch

    mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=x.device, dtype=x.dtype)
    codes = torch.bucketize(x.abs(), mid) | (torch.signbit(x).to(torch.int64) << 3)
    codes = codes.view(x.shape[0], x.shape[1] // 2, 2)
    return (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)


def quant_nvfp4(w, amax):
    """compressed-tensors NVFP4: weight_global_scale = 448 * 6 / amax, FP8 E4M3 scale per 16 weights along K."""
    import torch

    n, k = w.shape
    gs = (448.0 * 6.0) / amax.clamp_min(1e-10)
    blk = w.float().view(n, k // 16, 16)
    s8 = (blk.abs().amax(-1) / 6.0 * gs).clamp(max=448.0).to(torch.float8_e4m3fn)
    eff = (s8.float() / gs).clamp_min(1e-30).unsqueeze(-1)
    x = (blk / eff).clamp(-6.0, 6.0).view(n, k)
    return dict(weight_packed=pack_fp4(x), weight_scale=s8, weight_global_scale=gs.view(1).float())


def pair_fp8(stream, emit, stats):
    """Pair FP8 block weights with their scales (either order) and pass both to emit(name, weight, scale)."""
    import torch

    pend_w, pend_s = {}, {}
    for name, t in stream:
        if name.endswith(".weight_scale_inv"):
            wn = name[: -len("_scale_inv")]
            if wn in pend_w:
                stats["fp8_tensors"] = stats.get("fp8_tensors", 0) + 1
                yield from emit(wn, pend_w.pop(wn), t)
            else:
                pend_s[wn] = t
            continue
        if t.dtype == torch.float8_e4m3fn:
            if name in pend_s:
                stats["fp8_tensors"] = stats.get("fp8_tensors", 0) + 1
                yield from emit(name, t, pend_s.pop(name))
            else:
                pend_w[name] = t
            continue
        yield from emit(name, t, None)
    assert not pend_w and not pend_s, (list(pend_w)[:3], list(pend_s)[:3])


def vllm_transform(stream, kind, mode, stats):
    """FP8 checkpoint weights -> BF16; with nvfp4_w4a16 the expert weights -> compressed-tensors NVFP4 tensors."""
    import torch

    expert_re, routed_re = re.compile(EXPERT_WEIGHT[kind]), re.compile(ROUTED)
    ct = VLLM[mode].get("compressed_tensors")
    pend_pair = {}

    def emit(name, t, scale):
        if scale is not None:
            t = dequant_fp8_block(t.cuda(), scale.cuda())
        if not (ct and expert_re.match(name)):
            stats["passed"] = stats.get("passed", 0) + 1
            yield name, t
            return
        t = t.cuda()
        stats["quantized"] = stats.get("quantized", 0) + 1
        # gate and up of one expert share the global scale (vLLM keeps one per fused w13)
        key = re.sub(r"\.(gate|up)_proj\.weight$", "", name)
        if name.endswith(("gate_proj.weight", "up_proj.weight")):
            pend_pair.setdefault(key, {})[name] = t
            if len(pend_pair[key]) < 2:
                return
            pair = pend_pair.pop(key)
            amax = torch.stack([x.float().abs().max() for x in pair.values()]).max()
        else:
            pair, amax = {name: t}, t.float().abs().max()
        for nm, x in pair.items():
            base = nm[: -len("weight")]
            for suffix, y in quant_nvfp4(x, amax).items():
                yield base + suffix, y
            if routed_re.search(nm):
                yield base + "input_global_scale", torch.ones(1, dtype=torch.float32, device=x.device)

    yield from pair_fp8(stream, emit, stats)
    assert not pend_pair, list(pend_pair)[:3]


def sglang_transform(stream, kind, mode, stats):
    """FP8 checkpoint weights -> BF16, except that fp8_w8a8 keeps (Qwen3.5) or produces (DeepSeek) FP8 expert weights
    with their block scales."""
    expert_re = re.compile(EXPERT_WEIGHT[kind])

    def emit(name, t, scale):
        if mode == "fp8_w8a8" and expert_re.match(name):
            stats["fp8_expert"] = stats.get("fp8_expert", 0) + 1
            if scale is None:
                t, scale = quant_fp8_block(t.cuda())
            yield name, t
            yield name + "_scale_inv", scale
        else:
            yield name, dequant_fp8_block(t.cuda(), scale.cuda()) if scale is not None else t

    yield from pair_fp8(stream, emit, stats)


def patch_vllm_loader(kind, mode, stats):
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    orig = DefaultModelLoader.get_all_weights

    def get_all_weights(self, model_config, model):
        yield from vllm_transform(orig(self, model_config, model), kind, mode, stats)

    DefaultModelLoader.get_all_weights = get_all_weights


def patch_sglang_loader(kind, mode):
    from sglang.srt.model_loader import loader

    orig = loader.DefaultModelLoader._get_weights_iterator
    stats = {}

    def _get_weights_iterator(self, *a, **k):
        yield from sglang_transform(orig(self, *a, **k), kind, mode, stats)
        print("LOAD_STATS", json.dumps(stats), flush=True)

    loader.DefaultModelLoader._get_weights_iterator = _get_weights_iterator


# ---------------------------------------------------------------- model patches
def moe_classes(stack, kind):
    """(MoE block class, shared-expert MLP class, attribute of the shared expert in the block)."""
    if stack == "vllm":
        if kind == "deepseek":
            import vllm.model_executor.models.deepseek_v2 as m

            return m.DeepseekV2MoE, m.DeepseekV2MLP, "shared_experts"
        import vllm.model_executor.models.qwen3_next as m
        from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP

        return m.Qwen3NextSparseMoeBlock, Qwen2MoeMLP, "shared_expert"
    if kind == "deepseek":
        from sglang.srt.models.deepseek_v2 import DeepseekV2MLP, DeepseekV2MoE

        return DeepseekV2MoE, DeepseekV2MLP, "shared_experts"
    from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock

    return Qwen2MoeSparseMoeBlock, Qwen2MoeMLP, "shared_expert"


def zero_moe(stack, kind):
    """Every MoE block (router, shared and routed experts) returns zeros; the step keeps its token count."""
    import torch

    block, _, _ = moe_classes(stack, kind)
    block.forward = lambda self, hidden_states, *a, **k: torch.zeros_like(hidden_states)


def zero_shared(stack, kind):
    """Only the shared expert of every MoE block returns zeros (router and routed experts run)."""
    import torch

    block, mlp, attr = moe_classes(stack, kind)
    init0, forward0 = block.__init__, mlp.forward

    def init(self, *a, **k):
        init0(self, *a, **k)
        if getattr(self, attr, None) is not None:
            getattr(self, attr)._zero_output = True

    block.__init__ = init
    mlp.forward = lambda self, x, *a, **k: (
        torch.zeros_like(x) if getattr(self, "_zero_output", False) else forward0(self, x, *a, **k)
    )


def register_nvfp4_dense():
    """vLLM online shorthand 'nvfp4_all': routed experts with vLLM's online NVFP4 MoE method (per-token activation
    scales) and dense linear layers (the shared experts) with NVFP4 weights (E2M1, FP8 scale per 16, FP32 global scale
    448 * 6 / amax) and NVFP4 activations with a dynamic per-token scale and FP8 scales per 16 elements, through
    FlashInfer's CUTLASS FP4 GEMM; the output is multiplied by each token's decode scale. One scale per call instead of
    per token breaks long generations (a prefill chunk would share the scale of its largest activation). Must run in
    every process that builds the model."""
    import torch
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.config import quantization as qc
    from vllm.model_executor.kernels.linear.nvfp4.flashinfer import FlashInferCutlassNvFp4LinearKernel
    from vllm.model_executor.layers.quantization.online import base as ob
    from vllm.model_executor.layers.quantization.online.fp8 import OnlineLinearBase
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        nvfp4_weight_padding_bytes,
        slice_nvfp4_output,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
    from vllm.model_executor.utils import replace_parameter
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

    if not hasattr(torch.ops, "moe_bench") or not hasattr(torch.ops.moe_bench, "nvfp4_pt_linear"):
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE as BASE,
        )

        @torch.library.custom_op("moe_bench::nvfp4_pt_linear", mutates_args=())
        def nvfp4_pt_linear(
            x: torch.Tensor, w: torch.Tensor, w_sf: torch.Tensor, w_gs: torch.Tensor, n: int
        ) -> torch.Tensor:
            from flashinfer import SfLayout, nvfp4_quantize

            xq, xsf, pts = nvfp4_quantize(
                x.contiguous(), BASE, sfLayout=SfLayout.layout_128x4, per_token_activation=True
            )
            if xsf.dtype != torch.float8_e4m3fn:
                xsf = xsf.view(torch.float8_e4m3fn)
            out = slice_nvfp4_output(flashinfer_scaled_fp4_mm(xq, w, xsf, w_sf, w_gs, x.dtype, backend="cutlass"), n)
            return (out.float() * pts.float().reshape(-1, 1)).to(x.dtype)

        @nvfp4_pt_linear.register_fake
        def _(x, w, w_sf, w_gs, n):
            return x.new_empty((x.shape[0], n))

    class Nvfp4DynamicLinearMethod(OnlineLinearBase):
        def __init__(self):
            super().__init__()
            self.kernel = FlashInferCutlassNvFp4LinearKernel.__new__(FlashInferCutlassNvFp4LinearKernel)

        def process_weights_after_loading(self, layer):
            if getattr(layer, "_already_called_process_weights_after_loading", False):
                return
            w = self.get_weight_for_quantization(layer).contiguous()
            gw = (448.0 * 6.0 / w.abs().amax().float().clamp_min(1e-12)).reshape(())
            w_fp4, w_sf = scaled_fp4_quant(w, gw, is_sf_swizzled_layout=False)
            replace_parameter(layer, "weight", w_fp4.data)
            replace_parameter(layer, "weight_scale", w_sf.view(torch.float8_e4m3fn).data)
            layer.weight_global_scale = torch.nn.Parameter((1.0 / gw).to(torch.float32), requires_grad=False)
            self.kernel.process_weights_after_loading(layer)  # swizzle the block scales, pad K for CUTLASS
            self.release_requantization_source_weights(layer)
            layer._already_called_process_weights_after_loading = True

        def apply(self, layer, x, bias=None):
            n = layer.output_size_per_partition
            assert nvfp4_weight_padding_bytes(layer) == 0, "K padding is not supported by the per-token path"
            out = torch.ops.moe_bench.nvfp4_pt_linear(
                x.reshape(-1, x.shape[-1]), layer.weight, layer.weight_scale, layer.weight_global_scale, n
            )
            if bias is not None:
                out = out + bias
            return out.view(*x.shape[:-1], n)

    ob._ONLINE_LINEAR_METHODS[kNvfp4Static] = Nvfp4DynamicLinearMethod
    qc._ONLINE_SHORTHANDS["nvfp4_all"] = qc.QuantizationConfigArgs(
        linear=qc.QuantSpec(weight=kNvfp4Static), moe=qc.QuantSpec(weight=kNvfp4Static)
    )
    if "nvfp4_all" not in qc.ONLINE_QUANT_SHORTHAND_NAMES:
        qc.ONLINE_QUANT_SHORTHAND_NAMES = (*qc.ONLINE_QUANT_SHORTHAND_NAMES, "nvfp4_all")


def annotate_moe(stack, kind):
    """Profiler ranges: cat::moe around every MoE block, cat::router around the router GEMM and cat::shared around the
    shared-expert MLP (these classes also run outside MoE blocks, where the ranges are ignored)."""
    import torch

    block, mlp, _ = moe_classes(stack, kind)
    if stack == "vllm":
        from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear as Router
    elif kind == "deepseek":
        from sglang.srt.models.deepseek_v2 import MoEGate as Router
    else:
        from sglang.srt.layers.linear import ReplicatedLinear as Router

    def wrap(forward, name):
        def annotated(self, *a, **k):
            with torch.profiler.record_function("cat::" + name):
                return forward(self, *a, **k)

        return annotated

    for cls, name in ((block, "moe"), (Router, "router"), (mlp, "shared")):
        cls.forward = wrap(cls.forward, name)
