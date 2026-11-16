"""Emulated KV-cache formats in vLLM: the keys and values handed to every attention layer (what vLLM writes to its paged
cache and what the current step attends to) are quantized and dequantized in place, so the model sees the cached
precision.

  mxint8, mxint4  blocks of 16 consecutive elements along the head dimension with a shared power-of-two scale
                  2^ceil(log2(amax / qmax)), signed 8- or 4-bit integers
A suffix restricts the format to one part and keeps the other in BF16: '_k' keys or '_v' values (Qwen3.5's attention);
'_c' latent or '_pe' rotary key (DeepSeek-V2-Lite's multi-head latent attention caches a 512-dim latent and a 64-dim
rotary key). Qwen3.5's linear-attention layers keep their recurrent state in BF16. With stats=True (eager mode only)
the relative error ||x - q(x)|| / ||x|| of every part is accumulated and printed at exit.
register() patches vLLM's attention classes and must run in the process that builds the model.
"""

import atexit

FORMATS = ("mxint8", "mxint4")
PARTS = ("k", "v", "c", "pe")
_stats = {}


def mxint(x, bits, block=16):
    import torch

    shape = x.shape
    xf = x.float().reshape(-1, block)
    qmax = 2 ** (bits - 1) - 1
    amax = xf.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    scale = torch.exp2(torch.ceil(torch.log2(amax / qmax)))
    return (torch.round(xf / scale).clamp(-qmax - 1, qmax) * scale).reshape(shape).to(x.dtype)


def fake_quantize(x, fmt, head_dim):
    if fmt == "bf16" or x is None or x.numel() == 0:
        return x
    return mxint(x.reshape(-1, head_dim), {"mxint8": 8, "mxint4": 4}[fmt]).reshape(x.shape)


def _record(name, x, y):
    if x is None or y is None or x.numel() == 0:
        return
    s = _stats.setdefault(name, [0.0, 0.0, 0])
    s[0] += (x.float() - y.float()).pow(2).sum().item()
    s[1] += x.float().pow(2).sum().item()
    s[2] += 1


def _report():
    if _stats:
        parts = (f"{k}: rel_err={(v[0] / max(v[1], 1e-30)) ** 0.5:.4f} calls={v[2]}" for k, v in sorted(_stats.items()))
        print("KV_CACHE_STATS " + " ".join(parts), flush=True)


def parse(spec):
    """'mxint4_pe' -> {'k': fmt, 'v': fmt, 'c': fmt, 'pe': fmt} with BF16 outside the named part."""
    fmt, _, part = spec.partition("_")
    if fmt not in FORMATS or (part and part not in PARTS):
        raise ValueError(f"unknown KV-cache format {spec}")
    return {p: fmt if part in ("", p) else "bf16" for p in PARTS}


def register(spec, stats=False):
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention

    f = parse(spec)
    if stats:
        atexit.register(_report)
    attention, mla = Attention.forward, MLAAttention.forward

    def attention_forward(self, query, key, value, *a, **k):
        k2, v2 = fake_quantize(key, f["k"], self.head_size), fake_quantize(value, f["v"], self.head_size_v)
        if stats:
            _record("k", key, k2)
            _record("v", value, v2)
        return attention(self, query, k2, v2, *a, **k)

    def mla_forward(self, q, kv_c_normed, k_pe, *a, **k):
        c2 = fake_quantize(kv_c_normed, f["c"], kv_c_normed.shape[-1])
        p2 = fake_quantize(k_pe, f["pe"], k_pe.shape[-1])
        if stats:
            _record("latent", kv_c_normed, c2)
            _record("rope_k", k_pe, p2)
        return mla(self, q, c2, p2, *a, **k)

    Attention.forward, MLAAttention.forward = attention_forward, mla_forward
    print("KV cache", spec, flush=True)
