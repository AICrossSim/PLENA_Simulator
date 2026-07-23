"""
Decode-chip analytic model for disaggregated serving on PLENA.

Models the decode chip only. The prefill chip is a separate BF16 device that
hands over the prompt's KV cache, requantised to the decode chip's KV precision.
Each generated token costs
    step_time = max(compute_time, memory_time)
where compute_time comes from the cycle model and memory_time = HBM bytes /
bandwidth. One code path serves every model; only the model JSON differs
(`--model`).

Precision has two widths:
  N — the HBM stream widths (attnW / ffnW / KV). These set memory_time and the
      MLEN bandwidth cap.
  M — the MAC compute width (`--m-bits`, default = widest MAC operand
      max(attnW, ffnW, KV), plus the activation compute width when a CSV point
      carries it). M sets the iso-area compute density: at fixed area an M-bit
      array fits (4/M)^k times the reference multipliers (`--density-exp`).
      Default k = 0 means compute is precision-neutral (the memory-bound
      "upcast back to original compute" assumption); k = 2 turns on the density
      layer once a Synopsys DC sweep calibrates it.
HBM is fixed technology, not a free knob: `--hbm-gen`/`--hbm-channels` set
bandwidth AND capacity together from a real generation x channel count.

Modes:
  default     report one (precision, hardware, batch) point
  --search    right-size the hardware for a fixed precision
  --sweep     accuracy-vs-cost front on the decode chip (from the software DSE)
  --codesign  joint precision x hardware: right-size the array per precision
  --compare   system comparison vs A100/H100/TPU roofline references

The bottleneck is classified by arithmetic intensity vs the roofline ridge.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ANALYTIC = _HERE.parent
for _sub in ("performance", "memory", "utilisation", "roofline"):
    p = str(_ANALYTIC / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from perf_model import PerfModel, load_hardware_config_from_toml           # noqa: E402
from memory_model import MemoryConfig, MemoryModel, MemoryTraffic, load_memory_config_from_toml  # noqa: E402
from llm_memory_model import LLMMemoryModel                                # noqa: E402
from utilisation_model import PLENAUtilization                             # noqa: E402

FREQ_HZ = 1.0e9                      # 1 GHz clock
# Pipeline stages per step (issue + memory). Emulator cross-check: the KV-marginal
# cost agrees within ~2x; absolute per-layer time at the emulator's toy MLEN=64
# geometry is ~4x higher because scalar/control wrappers don't amortise there.
DECODE_PIPELINE_FACTOR = 2
SCALE_BITS = 8                       # one 8-bit shared scale per MX block (E8M0)
ACT_BITS = 16                        # activations stored bf16 on-chip (never HBM); computed low-precision
LM_HEAD_BITS = 16                    # vocab projection left unquantised by the software DSE
EMBED_BITS = 16                      # embedding table stored at bf16
# Mirrors the MXFP formats the software quantiser supports: 4-bit E1M2/E2M1,
# 6-bit E2M3/E3M2, 8-bit E3M4/E4M3/E5M2.
MXFP_FORMATS = {"E1M2": (1, 2), "E2M1": (2, 1), "E2M3": (2, 3), "E3M2": (3, 2),
                "E3M4": (3, 4), "E4M3": (4, 3), "E5M2": (5, 2)}

# Reference array: 4096 multipliers = 0.237 mm^2 (7nm), multiplying 4-bit MX
# operands. MLEN*BLEN counts positions in this 4-bit-MAC unit (the area budget).
REF_MULTIPLIERS = 4096
REF_MM2 = 0.237
MM2_PER_MULTIPLIER = REF_MM2 / REF_MULTIPLIERS

# Iso-area compute density: at fixed area an M-bit array fits (REF_MAC_BITS/M)^k
# times the reference multiplier count (multiplier area grows ~quadratically with
# width, so k ~= 2 once calibrated). Default k = 0 makes compute precision-neutral
# (the memory-bound "upcast back to original compute" assumption). Set
# --density-exp 2.0 to turn on the density layer.
REF_MAC_BITS = 4
DENSITY_EXP = 0.0

# Software-DSE Pareto source for --sweep/--codesign (pass a per-model CSV for other models).
_DEFAULT_CSV = str(_HERE / "software_disagg_decode.csv")

# HBM generations, matching the emulator's Ramulator model (per-channel widths
# HBM=128b, HBM2=64b, HBM3=64b). HBM2e is absent: the Rust code has no preset for
# it. Only HBM2 has a validated timing preset (2 Gbps, 64-bit channel, 8 channels
# per stack), so it is the default; HBM/HBM3 use standard rates and would need a
# Ramulator preset before emulator cross-checking. Per-channel bandwidth =
# ch_bits * gbps / 8; e.g. HBM2 = 16 GB/s/channel, so 512 GB/s ~= 32 channels.
# Bandwidth and capacity scale together with channel count.
HBM_GENS = {
    "HBM":   dict(ch_bits=128, gbps=1.0, stack_ch=8,  ch_gb=1.0),   # 128-bit channel, 16 GB/s/ch
    "HBM2":  dict(ch_bits=64,  gbps=2.0, stack_ch=8,  ch_gb=1.0),   # emulator preset: 16 GB/s/ch, 8 GB/stack
    "HBM3":  dict(ch_bits=64,  gbps=6.4, stack_ch=16, ch_gb=2.0),   # 64-bit channel, standard HBM3 rate
}


def hbm_overrides(gen: str, channels: int) -> dict:
    """HBM_WIDTH (bits per cycle) and HBM_SIZE (bytes) for a generation x channel
    count. channels=0 selects one full stack; the result echoes the count used."""
    s = HBM_GENS[gen]
    ch = channels or s["stack_ch"]
    bw_bytes = ch * s["ch_bits"] * s["gbps"] * 1e9 / 8
    return {"HBM_WIDTH": int(bw_bytes * 8 / FREQ_HZ), "HBM_SIZE": int(ch * s["ch_gb"] * 1e9),
            "channels": ch}


# Roofline references for --compare. GPUs/TPU use published BF16 peak and their
# own HBM; the PLENA entry is filled in from the live config at runtime
# (plena_device) so the compare table and the main report describe the same chip.
# `count` sizes each system (16 small accelerators vs 4 big GPUs, roughly equal
# multiplier budgets). `sq_dim` is the square tile the decode batch must fill
# (PLENA's tile is its small BLEN).
DEVICES = {
    "plena": dict(label="PLENA-sys", kind="plena", count=16),
    "a100":  dict(label="A100",    kind="gpu", count=4,  peak_tflops=312.0,  hbm_gb=80, hbm_tbs=1.99, sq_dim=128),
    "h100":  dict(label="H100",    kind="gpu", count=4,  peak_tflops=989.4,  hbm_gb=80, hbm_tbs=3.35, sq_dim=128),
    "tpu":   dict(label="TPU-v6e", kind="tpu", count=16, peak_tflops=918.0,  hbm_gb=32, hbm_tbs=1.56, sq_dim=256),
}


def plena_device(hw_cfg, base_mem) -> dict:
    """PLENA compare entry built from the live config, so geometry, bandwidth and
    capacity have one source of truth and the compare table matches the model."""
    return {**DEVICES["plena"], "mlen": hw_cfg.MLEN, "blen": hw_cfg.BLEN,
            "hbm_gb": base_mem.HBM_SIZE / 1e9, "hbm_tbs": peak_hbm_bw_bytes(hw_cfg) / 1e12}


def device_peaks(dev: dict, prec: dict | None = None) -> tuple[float, float, float]:
    """System totals (peak compute FLOP/s, peak bandwidth B/s, capacity bytes).
    PLENA peak = 2*MLEN*BLEN*clock x density(M); if `prec` is given, MLEN is also
    capped by the bandwidth bound MLEN*max(W,KV) <= HBM_WIDTH."""
    n = dev["count"]
    if dev["kind"] == "plena":
        mlen, density = dev["mlen"], 1.0
        if prec is not None:
            mlen = min(mlen, int(dev["hbm_tbs"] * 1e12 * 8 / FREQ_HZ // stream_bits(prec)))
            density = compute_density(prec)
        per_compute = 2 * mlen * dev["blen"] * FREQ_HZ * density
    else:
        per_compute = dev["peak_tflops"] * 1e12
    return per_compute * n, dev["hbm_tbs"] * 1e12 * n, dev["hbm_gb"] * 1e9 * n


def fill_util(batch: int, m_tile: int) -> float:
    """Fraction of the array's M-tile filled by the decode batch: B / (ceil(B/m_tile)*m_tile).

    A large square tile (GPU 128, TPU 256) sits mostly idle at small batch, while
    PLENA's small BLEN fills at low batch — the flattened-array advantage.
    """
    return batch / (math.ceil(batch / m_tile) * m_tile)


def effective_bits(fmt: str, width, block: int) -> float:
    """Average stored bits per element, including the per-block shared scale (SCALE_BITS/block)."""
    elem = int(width) if fmt == "mxint" else 1 + int(width[0]) + int(width[1])   # MXFP: sign+exp+frac
    return elem + SCALE_BITS / block


def element_bits(fmt: str, width) -> int:
    """Element width without the block scale -- the bits HBM streams per operand."""
    return int(width) if fmt == "mxint" else 1 + int(width[0]) + int(width[1])


def width_label(fmt: str, width) -> str:
    return f"MXINT{int(width)}" if fmt == "mxint" else f"MXFP_E{width[0]}M{width[1]}"


def precision_from_components(attn_bits, ffn_bits, kv_bits,
                             attn_label="attnW", ffn_label="ffnW", kv_label="KV",
                             attn_elem=None, ffn_elem=None, kv_elem=None,
                             m_bits=0, density_exp=DENSITY_EXP) -> dict:
    """One decode precision point (activations are fixed bf16 in storage).

    `*_elem` are the element widths N that HBM streams (they set the bandwidth cap
    and memory time); they default to the rounded effective bits. `m_bits` is the
    MAC compute width M: HBM streams N-bit operands, the array multiplies at M,
    writeback requantises to N. It drives the iso-area density; 0 means the widest
    streamed operand max(attnW, ffnW, KV). Callers that know the activation compute
    width should fold it in and pass m_bits explicitly (the DSE bridge does). With
    the default density_exp = 0 the density is 1 regardless of M."""
    attn_e = int(round(float(attn_bits))) if attn_elem is None else int(attn_elem)
    ffn_e = int(round(float(ffn_bits))) if ffn_elem is None else int(ffn_elem)
    kv_e = int(round(float(kv_bits))) if kv_elem is None else int(kv_elem)
    return {
        "attn_bits": float(attn_bits), "ffn_bits": float(ffn_bits), "kv_bits": float(kv_bits),
        "a_bits": float(ACT_BITS),
        "attn_label": attn_label, "ffn_label": ffn_label, "kv_label": kv_label,
        "attn_elem": attn_e, "ffn_elem": ffn_e, "kv_elem": kv_e,
        "m_bits": int(m_bits) if m_bits else max(attn_e, ffn_e, kv_e),
        # P = accumulator width (INT accumulate in the MX array). P never reaches
        # HBM: every writeback requantises P -> N first (new KV written at kv_bits,
        # activations stay on-chip), so all HBM byte counts here depend only on N.
        "p_bits": 32,
        "density_exp": float(density_exp),
    }


def compute_density(prec) -> float:
    """Iso-area multiplier gain of M-bit MACs over the 4-bit reference: (4/M)^k.
    Scales peak compute and the cycle model; the N widths scale memory time."""
    return (REF_MAC_BITS / prec["m_bits"]) ** prec["density_exp"]


def _parse_width(fmt: str, tok):
    """Parse one width token: an int for MXINT, an (exp, frac) tuple for MXFP."""
    return int(tok) if fmt == "mxint" else MXFP_FORMATS[tok]


def build_precision(args) -> dict:
    """Build a precision spec from CLI args. attn and ffn share the weight format
    (--w-fmt) but keep independent widths; KV has its own format (--kv-fmt), so
    MXINT weights with MXFP KV is expressible."""
    wf, kf = args.w_fmt, args.kv_fmt
    aw, fw, kv = _parse_width(wf, args.attn_w), _parse_width(wf, args.ffn_w), _parse_width(kf, args.kv)
    return precision_from_components(
        effective_bits(wf, aw, args.block),
        effective_bits(wf, fw, args.block),
        effective_bits(kf, kv, args.block),
        width_label(wf, aw), width_label(wf, fw), width_label(kf, kv),
        attn_elem=element_bits(wf, aw), ffn_elem=element_bits(wf, fw),
        kv_elem=element_bits(kf, kv),
        m_bits=args.m_bits, density_exp=args.density_exp)


def stream_bits(prec) -> int:
    """Widest operand HBM streams in decode: max(attnW, ffnW, KV) bits."""
    return max(prec["attn_elem"], prec["ffn_elem"], prec["kv_elem"])


def mlen_bandwidth_cap(hw_cfg, prec) -> int:
    """Largest MLEN HBM can feed per cycle: HBM_WIDTH / stream_bits. Wider operands => smaller MLEN."""
    return hw_cfg.HBM_WIDTH // stream_bits(prec)


def load_model_dims(path: str) -> dict:
    """Architecture sizes from the model JSON. `num_experts`/`experts_per_token`
    default to (1, 1) for a dense model (>1 experts uses the MoE FFN path).
    `sliding_window` + `layer_types` give the attention split; no sliding_window
    means all full attention (n_sliding = 0)."""
    with open(path) as f:
        p = json.load(f)
    ah = p["num_attention_heads"]
    layers = p["num_hidden_layers"]
    window = p.get("sliding_window") or 0                      # null / 0 -> no windowing
    if p.get("use_sliding_window") is False:                   # window size set but disabled (some Qwen)
        window = 0
    n_sliding = (sum(1 for lt in p.get("layer_types", []) if lt == "sliding_attention")
                 if window > 0 else 0)
    return {"hidden": p["hidden_size"], "heads": ah, "kv_heads": p["num_key_value_heads"],
            "head_dim": p.get("head_dim", p["hidden_size"] // ah), "layers": layers,
            "inter": p["intermediate_size"], "vocab": p["vocab_size"],
            "tie_embeddings": p.get("tie_word_embeddings", False),
            "num_experts": p.get("num_local_experts", 1),
            "experts_per_token": p.get("experts_per_token", p.get("num_experts_per_tok", 1)),
            "sliding_window": window, "n_sliding": n_sliding, "n_full": layers - n_sliding}


def is_moe(d: dict) -> bool:
    return d.get("num_experts", 1) > 1


def _attn_split(d: dict) -> tuple[int, int, int]:
    """Attention layers by KV span: (full count, sliding-window count, window).
    An all-full model is (layers, 0, 0), collapsing every windowed branch below."""
    return d["n_full"], d["n_sliding"], d["sliding_window"]


def peak_hbm_bw_bytes(hw_cfg) -> float:
    """Peak HBM bandwidth = (HBM_WIDTH / 8) * clock  [bytes/s]."""
    return (hw_cfg.HBM_WIDTH / 8.0) * FREQ_HZ


def matrix_overfetch_factor(hw_cfg) -> float:
    """Wasted-read factor on matrix loads. A prefetch reads M_LOAD elements rounded
    up to a multiple of MLEN, so M_LOAD > MLEN reads M_LOAD/MLEN x the needed
    bytes. Best case M_LOAD = MLEN gives factor 1."""
    mlen = hw_cfg.MLEN
    m_load = getattr(hw_cfg, "HBM_M_Prefetch_Amount", mlen)
    return math.ceil(max(m_load, mlen) / mlen)


# Area model: "proxy" (mm^2 per multiplier) or "calibrated" (precision-aware
# MatrixMachine census, DC-fitted). Set once from the CLI so area_mm2() serves
# every caller without passing a flag through each call site.
_AREA_MODEL = "proxy"
_AREA_PREC: dict | None = None


def set_area_model(model: str, prec: dict | None = None) -> None:
    global _AREA_MODEL, _AREA_PREC
    _AREA_MODEL, _AREA_PREC = model, prec


def area_multipliers(hw_cfg) -> int:
    """Matrix-array multiplier count = MLEN * BLEN (sets silicon area)."""
    return hw_cfg.MLEN * hw_cfg.BLEN


def area_mm2(hw_cfg) -> float:
    if _AREA_MODEL == "calibrated" and _AREA_PREC is not None:
        sys.path.insert(0, str(_HERE.parent.parent))
        from analytic_models.disagg_serve.area import area_mm2 as _calibrated
        return _calibrated("calibrated", hw_cfg, _AREA_PREC)
    return area_multipliers(hw_cfg) * MM2_PER_MULTIPLIER


def onchip_activation_bytes(d: dict, batch: int) -> int:
    """Decode activation working set held on-chip in FP16 (Vector SRAM); never written to HBM."""
    return math.ceil(batch * (d["hidden"] + d["inter"]) * ACT_BITS / 8)


# Per-component HBM footprint + decode traffic.
# `MemoryModel` carries a single `weight_bits`, but we need three (attn / FFN / vocab),
# so we reset it per component and sum. This is the only place per-component
# precision enters the byte model. The FFN term picks dense vs MoE by expert count.
def weight_footprint_bytes(mem: MemoryModel, d: dict, prec: dict) -> dict:
    """HBM weight storage: attention @ attn_bits, FFN/experts @ ffn_bits,
    embedding/lm_head/norms @ bf16. MoE keeps every expert (and the router)
    resident, which sets the capacity wall for these models.
    """
    h, ah, kvh, hd = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"]
    inter, vocab, layers = d["inter"], d["vocab"], d["layers"]

    mem.weight_bits = EMBED_BITS
    embedding = mem.embedding_weights(vocab, h)
    norms = mem.layer_norm_weights(h) * 2 * layers
    mem.weight_bits = LM_HEAD_BITS
    lm_head = mem.lm_head_weights(h, vocab, d.get("tie_embeddings", False))

    mem.weight_bits = prec["attn_bits"]
    attention = mem.attention_weights(h, ah, kvh, hd) * layers
    mem.weight_bits = prec["ffn_bits"]
    if is_moe(d):
        router, experts = mem.moe_weights(h, inter, d["num_experts"])   # router + all experts resident
        ffn = (router + experts) * layers
    else:
        ffn = mem.ffn_weights(h, inter) * layers

    total = embedding + norms + lm_head + attention + ffn
    return {"embedding": embedding, "norms": norms, "lm_head": lm_head,
            "attention": attention, "ffn": ffn, "total": total}


def kv_footprint_bytes(mem: MemoryModel, d: dict, prec: dict, ctx: int, batch: int) -> int:
    """HBM KV-cache footprint at kv_bits. Full layers store the whole `ctx`;
    sliding-window layers store only the last `window` tokens -- windowing's
    capacity win."""
    mem.kv_cache_bits = prec["kv_bits"]
    kvh, hd = d["kv_heads"], d["head_dim"]
    n_full, n_slide, window = _attn_split(d)
    total = mem.kv_cache_footprint(kvh, hd, n_full, ctx, batch).total_bytes
    if n_slide:
        total += mem.kv_cache_footprint(kvh, hd, n_slide, min(ctx, window), batch).total_bytes
    return total


def decode_traffic(mem: MemoryModel, d: dict, kv_size: int, batch: int, prec: dict) -> MemoryTraffic:
    """Per-token HBM traffic: read every weight once (attn @ attn_bits,
    FFN/experts @ ffn_bits, lm_head @ bf16), read the whole KV cache @ kv_bits,
    write the new token's KV. MoE reads the router + top-k experts, not all."""
    h, ah, kvh, hd = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"]
    inter, vocab, layers = d["inter"], d["vocab"], d["layers"]

    mem.weight_bits = prec["attn_bits"]
    mem.kv_cache_bits = prec["kv_bits"]
    n_full, n_slide, window = _attn_split(d)
    proj = mem.projection_traffic(h, ah, kvh, hd, 1, batch, "decode")     # QKV weights + one-token KV write
    out_proj = mem.output_projection_traffic(h, ah, hd, 1, batch, "decode")
    # Full layers read the whole KV; windowed layers read only the last `window` keys.
    attn = (proj + out_proj) * layers + mem.flash_attention_traffic(ah, kvh, hd, 1, kv_size, batch, "decode") * n_full
    if n_slide:
        attn += mem.flash_attention_traffic(ah, kvh, hd, 1, min(kv_size, window), batch, "decode") * n_slide

    mem.weight_bits = prec["ffn_bits"]
    if is_moe(d):
        ffn = mem.moe_traffic(h, inter, d["num_experts"], d["experts_per_token"], 1, batch, "decode") * layers
    else:
        ffn = mem.ffn_traffic(h, inter, 1, batch, "decode") * layers

    mem.weight_bits = LM_HEAD_BITS
    head = mem.lm_head_traffic(h, vocab)
    return attn + ffn + head


# Compute cycles + FLOPs per decode token
def _ffn_label_cycles(perf: PerfModel, d: dict, batch: int) -> tuple[str, int]:
    """FFN cycles for the decode step. MoE runs the top-k experts as k FFN passes;
    each expert is a full FFN of width `inter`."""
    ffn = perf.feed_forward(d["hidden"], d["inter"], 1, batch, "decode")
    if is_moe(d):
        return f"MoE {d['experts_per_token']}/{d['num_experts']} experts", ffn * d["experts_per_token"]
    return "FFN (gate/up/down)", ffn


def _flash_cycles(perf: PerfModel, d: dict, kv: int, batch: int) -> int:
    """Per-token flash-attention cycles over the stack: full layers attend to the
    whole KV, sliding-window layers to only the last `window` keys."""
    ah, kvh, hd = d["heads"], d["kv_heads"], d["head_dim"]
    n_full, n_slide, window = _attn_split(d)
    total = perf.flash_attention(ah, kvh, hd, 1, kv, batch, "decode") * n_full
    if n_slide:
        total += perf.flash_attention(ah, kvh, hd, 1, min(kv, window), batch, "decode") * n_slide
    return total


def decode_token_components(perf: PerfModel, d: dict, kv: int, batch: int) -> dict:
    """Cycles to generate one token: per-layer ops x layers + once-per-token head
    ops. Windowed layers are charged only for their `window` keys."""
    h, ah, kvh, hd, layers = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"], d["layers"]
    ffn_label, ffn_cyc = _ffn_label_cycles(perf, d, batch)
    comp = {
        f"RMSNorm (x2) x{layers} layers":            perf.rms_layer(h, 1, batch, "decode") * 2 * layers,
        f"Q/K/V proj + RoPE x{layers} layers":       perf.projection(h, ah, kvh, hd, 1, batch, "decode") * layers,
        f"Flash attention x{layers} layers":         _flash_cycles(perf, d, kv, batch),
        f"Output projection (W_O) x{layers} layers": perf.output_projection(h, ah, hd, 1, batch, "decode") * layers,
        f"Residual adds (x2) x{layers} layers":      perf.residual(h, 1, batch, "decode") * 2 * layers,
        f"{ffn_label} x{layers} layers":             ffn_cyc * layers,
    }
    comp["Embedding lookup"] = perf.embeddings(h, 1, batch, "decode")
    comp["LM head"] = perf.lm_head(h, d["vocab"], batch)
    comp["Vocab softmax"] = perf.softmax_full_seq(d["vocab"], 1, batch)
    return comp


def decode_token_cycles(perf: PerfModel, d: dict, kv: int, batch: int) -> int:
    return sum(decode_token_components(perf, d, kv, batch).values())


def decode_step_flops(d: dict, kv: int, batch: int) -> int:
    """FLOPs for one decode step over the batch (2 per MAC), for arithmetic
    intensity. MoE counts only top-k experts; sliding layers only `window` keys."""
    h, ah, kvh, hd = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"]
    qkvo = h * ah * hd + 2 * (h * kvh * hd) + (ah * hd) * h    # Q, K, V, O projections
    ffn = 2 * h * d["inter"] + d["inter"] * h                  # gate, up, down (one expert)
    if is_moe(d):
        ffn *= d["experts_per_token"]
    n_full, n_slide, window = _attn_split(d)
    attn = 2 * ah * hd * kv * n_full                           # QK^T + attention @ V (full layers)
    if n_slide:
        attn += 2 * ah * hd * min(kv, window) * n_slide        # windowed layers
    body = (qkvo + ffn) * d["layers"] + attn
    return 2 * batch * (body + h * d["vocab"])   # + LM head (once per token)


def run_decode_loop(perf, mem, d, prec, input_seq, output_seq, batch, peak_bw, stride, overfetch,
                    n_chips=1, bw_model=None, hbm_gen="HBM2", hbm_channels=8):
    """Walk the growing-context decode; each token costs max(compute, memory).
    `stride` subsamples the loop for speed. `n_chips` tensor-parallel chips each do
    1/n of the work, so step time is /n and bandwidth is xn.

    memory_time: with a `bw_model` (disagg_serve.memory.CalibratedBandwidth) bytes
    are priced at the measured effective bandwidth per class for (hbm_gen,
    hbm_channels); otherwise at aggregate peak. A token is memory-bound when
    memory_time actually paced it (>= compute_time) — the achieved-time rule, not
    the roofline AI-vs-ridge rule, which can disagree below the compute ceiling."""
    # M-bit density scales the compute side: the cycle model is calibrated on the
    # 4-bit array, and density x more (or fewer) MACs finish density x faster.
    density = compute_density(prec)
    total_time, total_bytes, first_step, mem_bound = 0.0, 0, None, 0
    t = 0
    while t < output_seq:
        kv = input_seq + t                                    # KV cache grows by one each token
        compute_time = decode_token_cycles(perf, d, kv, batch) * DECODE_PIPELINE_FACTOR / FREQ_HZ / density / n_chips
        tr = decode_traffic(mem, d, kv, batch, prec)
        bytes_tok = tr.read_bytes * overfetch + tr.write_bytes
        if bw_model is not None:
            # One H_PREFETCH_M moves an MLEN x MLEN weight tile at the widest
            # streamed width; that per-DMA size keys the size-aware bandwidth
            # curve (large arrays issue larger, faster transfers).
            wt_transfer = perf.mlen * perf.mlen * max(prec["attn_elem"], prec["ffn_elem"]) / 8
            memory_time = bw_model.memory_time(
                {"weights_kv": tr.read_bytes * overfetch, "writeback": tr.write_bytes},
                hbm_gen, hbm_channels, transfer_bytes=wt_transfer) / n_chips
        else:
            memory_time = bytes_tok / (peak_bw * n_chips)

        step_time = max(compute_time, memory_time)
        span = min(stride, output_seq - t)                    # tokens this sample stands for
        total_time += step_time * span
        total_bytes += bytes_tok * span
        if memory_time >= compute_time:                       # memory paced this token
            mem_bound += span
        if first_step is None:
            first_step = step_time
        t += stride
    return {"total_time": total_time, "tpot": total_time / output_seq,
            "tps": (batch * output_seq) / total_time, "first_step": first_step,
            "avg_bytes_per_token": total_bytes / output_seq, "frac_mem_bound": mem_bound / output_seq}


def evaluate(model_path, dims, hw_cfg, isa_path, base_mem, prec, batch,
             input_seq, output_seq, hw_over=None, stride=1, n_chips=0,
             bw_model=None, hbm_gen="HBM2", hbm_channels=8):
    """Metrics for one (hardware, precision, batch) point. `n_chips`: 0 = auto
    (fewest HBM stacks that hold the model), else a fixed count; a model that fits
    one stack resolves to 1 chip."""
    perf = PerfModel(hw_cfg, isa_path)
    # Activations bf16; KV at kv_bits. weight_bits is set per-component in decode_traffic.
    mem_cfg = base_mem.model_copy(update={"weight_bits": prec["ffn_bits"], "activation_bits": ACT_BITS,
                                          "kv_cache_bits": prec["kv_bits"], **(hw_over or {})})
    mem_model = LLMMemoryModel(model_path, mem_cfg, batch_size=batch,
                               input_seq_len=input_seq, output_seq_len=output_seq)
    mem = mem_model.mem
    peak_bw = peak_hbm_bw_bytes(hw_cfg)

    # HBM holds only weights + KV cache; activations stay on-chip. The footprint
    # is chip-count independent, so resolve the chip count from it.
    ctx = input_seq + output_seq
    wf = weight_footprint_bytes(mem, dims, prec)
    kv_bytes = kv_footprint_bytes(mem, dims, prec, ctx, batch)
    hbm_required = wf["total"] + kv_bytes
    # Per-chip capacity honours hw_over: a searched HBM_SIZE (channel count) must
    # gate the fit check, not the TOML default.
    hbm_per_chip = mem_cfg.HBM_SIZE
    chips = n_chips if n_chips else max(1, math.ceil(hbm_required / hbm_per_chip))
    hbm_capacity = hbm_per_chip * chips

    loop = run_decode_loop(perf, mem, dims, prec, input_seq, output_seq, batch, peak_bw,
                           stride, matrix_overfetch_factor(hw_cfg), n_chips=chips,
                           bw_model=bw_model, hbm_gen=hbm_gen, hbm_channels=hbm_channels)
    loop.update(hbm_required=hbm_required, fits_in_hbm=hbm_required <= hbm_capacity,
                hbm_capacity=hbm_capacity, hbm_per_chip=hbm_per_chip, n_chips=chips,
                weight_footprint=wf, kv_footprint=kv_bytes,
                mem=mem, perf=perf, peak_bw=peak_bw, dims=dims,
                bw_model=bw_model, hbm_gen=hbm_gen, hbm_channels=hbm_channels)
    return loop


def max_batch_capacity(result, batch: int) -> int:
    """Largest batch HBM holds: weights are fixed, KV grows with batch x context."""
    kv_per_batch = result["kv_footprint"] / max(batch, 1)
    return int((result["hbm_capacity"] - result["weight_footprint"]["total"]) // max(kv_per_batch, 1))


def _fmt_bytes(n):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.3f} {unit}"
    return f"{n:.0f} B"


def _decode_utilization(hw_cfg, d: dict, kv: int, batch: int) -> tuple[float, float]:
    """(attention, FFN) matrix-array utilisation for the decode step, a display
    metric (attainable / theoretical ops). Attention blends full and
    sliding-window layers (windowed layers share the projection but shrink the
    flash part). FFN is the per-GEMM fill, same for a dense FFN or one MoE expert."""
    plena = PLENAUtilization({"MLEN": hw_cfg.MLEN, "BLEN": hw_cfg.BLEN, "VLEN": hw_cfg.VLEN})
    h, ah, kvh, hd = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"]
    n_full, n_slide, window = _attn_split(d)
    proj_a, proj_t = plena.projection_utilization(h, ah, kvh, hd, 1, batch, "decode")   # kv-independent
    fa_a, fa_t = plena.flash_attention_utilization(ah, kvh, hd, 1, kv, batch, "decode")
    att_a, att_t = (proj_a + fa_a) * n_full, (proj_t + fa_t) * n_full
    if n_slide:
        s_a, s_t = plena.flash_attention_utilization(ah, kvh, hd, 1, min(kv, window), batch, "decode")
        att_a += (proj_a + s_a) * n_slide
        att_t += (proj_t + s_t) * n_slide
    ffn_a, ffn_t = plena.ffn_utilization(h, d["inter"], 1, batch, "decode")
    return (att_a / att_t if att_t else 0.0), (ffn_a / ffn_t if ffn_t else 0.0)


def print_report(args, dims, hw_cfg, prec, result):
    bar = "=" * 78
    n_chips = result["n_chips"]
    density = compute_density(prec)
    peak_compute = 2 * hw_cfg.MLEN * hw_cfg.BLEN * FREQ_HZ * density   # per chip, at M-bit MACs
    peak_bw = result["peak_bw"]                                  # per chip
    peak_compute_sys, peak_bw_sys = peak_compute * n_chips, peak_bw * n_chips   # aggregate over the group
    moe_str = f", MoE {dims['experts_per_token']}/{dims['num_experts']} experts" if is_moe(dims) else ""
    sw_str = (f", sliding {dims['n_sliding']}/{dims['layers']} layers @ w{dims['sliding_window']}"
              if dims["n_sliding"] else "")
    print(bar)
    print("  DECODE-CHIP REPORT — Disaggregated Serving on PLENA")
    print(bar)
    print(f"  Model:     {args.model}  (hidden={dims['hidden']}, layers={dims['layers']}, "
          f"heads={dims['heads']}/{dims['kv_heads']}KV, head_dim={dims['head_dim']}, inter={dims['inter']}{moe_str}{sw_str})")
    print(f"  Workload:  batch={args.batch}  input_seq={args.input_seq} (handed-off KV)  "
          f"output_seq={args.output_seq}")
    print(f"  Precision: attnW:{prec['attn_label']} ffnW:{prec['ffn_label']} KV:{prec['kv_label']} "
          f"(block {args.block})  ->  {prec['attn_bits']:.3f}/{prec['ffn_bits']:.3f}/{prec['kv_bits']:.3f} eff bits")
    print(f"             activations: bf16 in on-chip SRAM, never in HBM (computed low-precision)   [prefill: separate FP16]")
    # Two-level scheme: HBM streams the N-bit operands above; the array multiplies at M-bit.
    print(f"             compute MACs: M={prec['m_bits']}-bit -> iso-area density x{density:.2f} "
          f"vs the {REF_MAC_BITS}-bit reference array (k={prec['density_exp']:.1f})")
    hbm_src = (f"  [{args.hbm_gen} x {args.hbm_channels} ch]" if args.hbm_gen else "")
    print(f"  Hardware:  MLEN={hw_cfg.MLEN} BLEN={hw_cfg.BLEN} VLEN={hw_cfg.VLEN} HLEN={hw_cfg.HLEN}  "
          f"clock={FREQ_HZ/1e9:.0f} GHz")
    print(f"             peak compute = 2*MLEN*BLEN*clock x density = {peak_compute/1e12:.2f} TFLOP/s   "
          f"peak HBM BW = {peak_bw/1e9:.0f} GB/s{hbm_src}")
    mac_note = (f" -> {int(area_multipliers(hw_cfg) * density):,} M-bit MACs at iso-area"
                if density != 1.0 else "")
    print(f"             matrix array = {area_multipliers(hw_cfg):,} multipliers (~{area_mm2(hw_cfg):.3f} mm^2){mac_note}")
    if n_chips > 1:   # model exceeds one HBM stack -> tensor-parallel over UALink
        print(f"             system: {n_chips} chips (ideal tensor-parallel, UALink) -> aggregate "
              f"{peak_compute_sys/1e12:.2f} TFLOP/s, {peak_bw_sys/1e9:.0f} GB/s, {result['hbm_capacity']/1e9:.0f} GB")
    # HBM feeds MLEN operands/cycle, so the widest of (attnW, ffnW, KV) caps how wide MLEN can be.
    print(f"             bandwidth bound: MLEN <= HBM_WIDTH / max(attnW,ffnW,KV) = "
          f"{hw_cfg.HBM_WIDTH}/{stream_bits(prec)} = {mlen_bandwidth_cap(hw_cfg, prec)}")

    # TTFT is the prefill chip's job; the decode chip's first step makes token #2.
    print("\n[1] LATENCY  (TTFT from prefill)")
    print(f"      First decode step (kv={args.input_seq}):  {result['first_step']*1e3:.3f} ms")
    print(f"      TPOT (avg time / output token):  {result['tpot']*1e3:.3f} ms")
    print(f"      Total generation ({args.output_seq} tok):    {result['total_time']*1e3:.2f} ms")

    avg_kv = args.input_seq + args.output_seq // 2
    print("\n[2] PERFORMANCE")
    print(f"      TPS (batch*output / total):      {result['tps']:.1f} tokens/s")
    print(f"      Per-stream rate (1 / TPOT):      {1.0/result['tpot']:.1f} tokens/s")
    print(f"      Achieved compute:                "
          f"{decode_step_flops(dims, avg_kv, args.batch) / result['tpot'] / 1e12:.2f} TFLOP/s")

    wf, kv_bytes = result["weight_footprint"], result["kv_footprint"]
    max_batch = max_batch_capacity(result, args.batch)
    ffn_name = "experts" if is_moe(dims) else "ffn"
    print("\n[3] MEMORY  (HBM = weights + KV; activations stay on-chip)")
    print(f"      Weights (HBM):        {_fmt_bytes(wf['total'])}  "
          f"(attn {_fmt_bytes(wf['attention'])} @ {prec['attn_bits']:.2f}b, "
          f"{ffn_name} {_fmt_bytes(wf['ffn'])} @ {prec['ffn_bits']:.2f}b, lm_head/emb @ bf16)")
    kv_note = (f", {dims['n_sliding']} sliding layers capped @ {dims['sliding_window']}"
               if dims["n_sliding"] else "")
    print(f"      KV cache (HBM):       {_fmt_bytes(kv_bytes)}  "
          f"(context={args.input_seq+args.output_seq}, batch={args.batch}{kv_note})")
    cap_note = f" ({n_chips} chips)" if n_chips > 1 else ""
    print(f"      HBM used / capacity:  {_fmt_bytes(result['hbm_required'])} / "
          f"{_fmt_bytes(result['hbm_capacity'])}{cap_note}  ->  {'FITS' if result['fits_in_hbm'] else 'EXCEEDS'} "
          f"({result['hbm_required']/result['hbm_capacity']*100:.1f}%)")
    print(f"      Activations on-chip:  {_fmt_bytes(onchip_activation_bytes(dims, args.batch))}")
    print(f"      Max batch (Capacity bound): {max_batch}  (KV grows with batch x context)")
    print(f"      HBM bytes / decode step: {_fmt_bytes(result['avg_bytes_per_token'])}  "
          f"({_fmt_bytes(result['avg_bytes_per_token']/max(args.batch,1))}/token x {args.batch} batch)")

    attn_util, ffn_util = _decode_utilization(hw_cfg, dims, avg_kv, args.batch)
    perf = result["perf"]
    compute_time = decode_token_cycles(perf, dims, avg_kv, args.batch) * DECODE_PIPELINE_FACTOR / FREQ_HZ / density / n_chips
    tr = decode_traffic(result["mem"], dims, avg_kv, args.batch, prec)
    bytes_tok = tr.read_bytes * matrix_overfetch_factor(hw_cfg) + tr.write_bytes
    achieved_bw = bytes_tok / max(compute_time, bytes_tok / peak_bw_sys)
    print("\n[4] UTILISATION  (@ avg context)")
    print(f"      Matrix array, attention:  {attn_util*100:.1f}% of peak")
    print(f"      Matrix array, FFN:        {ffn_util*100:.1f}% of peak")
    print(f"      HBM bandwidth:            {achieved_bw/1e9:.1f} / {peak_bw_sys/1e9:.0f} GB/s "
          f"({achieved_bw/peak_bw_sys*100:.1f}% of peak)")

    ridge = peak_compute_sys / peak_bw_sys                    # == per-chip ridge (n cancels)
    flops = decode_step_flops(dims, avg_kv, args.batch)
    ai = flops / bytes_tok
    ideal_compute_time = flops / peak_compute_sys            # roofline compute ceiling (aggregate)
    # Price memory the same way the decode loop did: calibrated effective
    # bandwidth when --bw-model calibrated, else aggregate peak.
    if result.get("bw_model") is not None:
        wt_transfer = hw_cfg.MLEN * hw_cfg.MLEN * max(prec["attn_elem"], prec["ffn_elem"]) / 8
        memory_time = result["bw_model"].memory_time(
            {"weights_kv": tr.read_bytes * matrix_overfetch_factor(hw_cfg),
             "writeback": tr.write_bytes},
            result["hbm_gen"], result["hbm_channels"],
            transfer_bytes=wt_transfer) / result["n_chips"]
        mem_label = "bytes / calibrated effective BW (size-aware)"
    else:
        memory_time = bytes_tok / peak_bw_sys
        mem_label = "bytes / peak HBM BW"
    mem_bound = memory_time >= compute_time                  # which time actually paces the token
    compute_util_pct = ideal_compute_time / compute_time * 100   # achieved vs roofline-ideal compute
    print(f"\n[5] ROOFLINE  (decode step @ avg context kv={avg_kv})   [bound rule: achieved times; "
          f"AI-vs-ridge shown for reference]")
    print(f"      Arithmetic intensity:  {ai:.2f} FLOP/byte   (ridge {ridge:.1f} FLOP/byte)")
    print(f"      memory  time / token:  {memory_time*1e3:.3f} ms   ({mem_label})")
    print(f"      compute time / token:  {ideal_compute_time*1e3:.3f} ms ideal (FLOPs / peak compute)")
    print(f"                             {compute_time*1e3:.3f} ms achieved (cycle model x{DECODE_PIPELINE_FACTOR} "
          f"pipeline -> {compute_util_pct:.0f}% of the compute ceiling realised; the gap is the dot below the roofline)")
    print(f"      -> Bottleneck:         {'MEMORY-bound — HBM bandwidth sets the pace' if mem_bound else 'COMPUTE-bound — the matrix array sets the pace'}"
          f"  ({result['frac_mem_bound']*100:.0f}% of tokens memory-bound)")
    comps = decode_token_components(perf, dims, avg_kv, args.batch)
    tot = sum(comps.values())
    print("      Decode-step cycles per operation:")
    for name, c in comps.items():
        print(f"        {name:<34} {c*DECODE_PIPELINE_FACTOR:>14,d} cyc  ({c/tot*100:5.1f}%)")

    # Prefill -> decode KV hand-off. Wire bytes are at the decode KV precision
    # (the prefill side quantizes on write).
    sys.path.insert(0, str(_HERE.parent.parent))
    from analytic_models.disagg_serve import handoff as _handoff
    print(f"\n[6] KV HAND-OFF  (prefill chip -> decode chip, quantize-on-write)")
    print(_handoff.report(dims, prec, args.input_seq, args.batch,
                          link_gen=args.link_gen,
                          link_bw=args.link_bw * 1e9 if args.link_bw else None))
    print(bar)


# Hardware + batch search (precision fixed). VLEN is not independent: the compiler
# requires VLEN == MLEN, so the vector unit follows the matrix unit everywhere.
SEARCH_SPACE = {
    "MLEN": [64, 128, 256, 512, 1024, 2048],   # matrix reduction tile (capped by bandwidth); VLEN follows
    "BLEN": [4, 8, 16, 32, 64, 128, 256],      # sub-array width; area = MLEN*BLEN; BLEN=batch fills the FFN tile
    "HLEN": [16, 32, 64, 128],       # head lane; MLEN//HLEN heads run in parallel
    "BATCH": [1, 4, 8, 16, 64, 256],    # serving knob (throughput vs latency); KV capacity caps it
}
RIGHTSIZE_TPS_TOL = 0.01             # within 1% of the best TPS still counts as "peak"


def _valid(mlen, blen, vlen, hlen, hidden=0) -> bool:
    """Legal array geometry: MLEN divisible by BLEN and HLEN, MLEN >= HLEN >= BLEN,
    plus the compiler constraints VLEN == MLEN and hidden % VLEN == 0 (hidden=0
    skips the divisibility check when model dims aren't available)."""
    return (mlen % blen == 0 and mlen % hlen == 0 and blen <= hlen <= mlen
            and vlen == mlen and (hidden == 0 or hidden % mlen == 0))


def _bandwidth_ok(mlen, hw_cfg, prec) -> bool:
    """Bandwidth check: the array needs MLEN operands per cycle, so HBM must supply
    MLEN * stream_bits <= HBM_WIDTH, else the array starves."""
    return mlen <= mlen_bandwidth_cap(hw_cfg, prec)


def _candidate(hw_cfg, dim, value):
    """One swept value with the other axes at baseline. VLEN follows MLEN (compiler
    constraint) and M_LOAD follows MLEN (no wasted reads)."""
    mlen, blen, hlen = hw_cfg.MLEN, hw_cfg.BLEN, hw_cfg.HLEN
    batch = None
    if dim == "BATCH":   batch = value
    elif dim == "MLEN":  mlen = value
    elif dim == "BLEN":  blen = value
    elif dim == "HLEN":  hlen = value
    vlen = mlen
    hw2 = hw_cfg.model_copy(update={"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen,
                                    "HBM_M_Prefetch_Amount": mlen})
    return hw2, {"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen}, batch, (mlen, blen, vlen, hlen)


def run_search(args, model_path, dims, base_hw, isa, base_mem, prec):
    """Two phases: (1) right-size each hardware axis at the user's batch -- the
    smallest array that still reaches ~peak TPS; (2) sweep batch on the right-sized
    chip to show its throughput/latency trade-off."""
    stride = max(1, args.output_seq // 24)
    # Clamp the starting MLEN to this precision's bandwidth cap so every sweep is
    # feasible, and tie VLEN to MLEN (compiler constraint).
    cap = mlen_bandwidth_cap(base_hw, prec)
    start_mlen = min(base_hw.MLEN, cap)
    base_hw = base_hw.model_copy(update={"MLEN": start_mlen, "VLEN": start_mlen,
                                         "HBM_M_Prefetch_Amount": start_mlen})

    def sweep_axis(dim, values, hw, batch_fixed):
        """Sweep one axis with the others fixed; print the table, return (rows, eligible)."""
        print(f"\n  -- {dim} sweep --")
        print(f"     {'value':>6} | {'TPOT(ms)':>9} | {'TPS':>9} | {'area(mm^2)':>10} | {'bound':>7} | fits")
        rows, eligible = [], []
        for v in values:
            hw2, mem_over, batch_v, geo = _candidate(hw, dim, v)
            batch = batch_v if batch_v is not None else batch_fixed
            if not (_valid(*geo, hidden=dims["hidden"]) and _bandwidth_ok(geo[0], hw2, prec)):
                continue
            try:
                r = evaluate(model_path, dims, hw2, isa, base_mem, prec, batch,
                             args.input_seq, args.output_seq, mem_over, stride=stride, n_chips=args.chips)
            except Exception as e:                            # keep the sweep alive
                print(f"     {v:>6} |  (skipped: {type(e).__name__})")
                continue
            area = area_multipliers(hw2)
            bound = "memory" if r["frac_mem_bound"] >= 0.5 else "compute"
            print(f"     {v:>6} | {r['tpot']*1e3:>9.3f} | {r['tps']:>9.1f} | "
                  f"{area*MM2_PER_MULTIPLIER:>10.3f} | {bound:>7} | {'yes' if r['fits_in_hbm'] else 'NO'}")
            rows.append({"value": v, "tps": r["tps"], "tpot": r["tpot"], "area": area, "fits": r["fits_in_hbm"]})
            if r["fits_in_hbm"]:
                eligible.append({"value": v, "tps": r["tps"], "area": area, "tpot": r["tpot"]})
        return rows, eligible

    print("\n" + "#" * 78)
    print("[6] HARDWARE + BATCH SEARCH  (precision fixed; one axis at a time)")
    print(f"    Baseline: MLEN={base_hw.MLEN} BLEN={base_hw.BLEN} VLEN={base_hw.VLEN} "
          f"HLEN={base_hw.HLEN}  ({area_multipliers(base_hw):,} mult, {area_mm2(base_hw):.3f} mm^2)  "
          f"batch={args.batch}")
    print("    Right-size each axis = the smallest array still reaching ~peak TPS at your batch.")
    print("#" * 78)

    # Phase 1: right-size each hardware axis at the user's batch (VLEN follows MLEN).
    sweeps, best = {}, {}
    for dim in ("MLEN", "BLEN", "HLEN"):
        rows, eligible = sweep_axis(dim, SEARCH_SPACE[dim], base_hw, args.batch)
        sweeps[dim] = rows
        if not eligible:
            continue
        peak = max(e["tps"] for e in eligible)
        knee = min((e for e in eligible if e["tps"] >= (1 - RIGHTSIZE_TPS_TOL) * peak), key=lambda e: e["area"])
        best[dim] = knee["value"]
        print(f"     -> right-size {dim}={knee['value']}  (TPS={knee['tps']:.1f}, "
              f"{knee['area']*MM2_PER_MULTIPLIER:.3f} mm^2)")

    # Assemble the right-sized chip (VLEN == MLEN by the compiler constraint).
    mlen, blen = best.get("MLEN", base_hw.MLEN), best.get("BLEN", base_hw.BLEN)
    vlen, hlen = mlen, best.get("HLEN", base_hw.HLEN)
    print(f"\n  -- Right-sized decode chip @ batch={args.batch} --")
    best_hw, best_result = None, None
    if _valid(mlen, blen, vlen, hlen, hidden=dims["hidden"]) and _bandwidth_ok(mlen, base_hw, prec):
        best_hw = base_hw.model_copy(update={"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen,
                                             "HBM_M_Prefetch_Amount": mlen})
        best_result = evaluate(model_path, dims, best_hw, isa, base_mem, prec, args.batch, args.input_seq,
                               args.output_seq, {"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen},
                               stride=stride, n_chips=args.chips)
        area = area_multipliers(best_hw)
        print(f"     MLEN={mlen} BLEN={blen} VLEN={vlen} HLEN={hlen} batch={args.batch}")
        print(f"     -> TPOT={best_result['tpot']*1e3:.3f} ms   TPS={best_result['tps']:.1f}   "
              f"{'memory-bound' if best_result['frac_mem_bound'] >= 0.5 else 'compute-bound'}")
        print(f"     -> {area:,} mult (~{area_mm2(best_hw):.3f} mm^2 = {area/REF_MULTIPLIERS:.2f}x the "
              f"{REF_MM2:.3f} mm^2 baseline)   fits={'yes' if best_result['fits_in_hbm'] else 'NO'}")
    else:
        print("     (best-per-axis combination breaks the geometry/bandwidth rules)")

    # Phase 2: batch trade-off on the right-sized chip. BLEN=batch fills only the
    # FFN GEMM tile, and HBM capacity (KV cache) caps the batch regardless.
    rows, eligible = sweep_axis("BATCH", SEARCH_SPACE["BATCH"], best_hw or base_hw, args.batch)
    sweeps["BATCH"] = rows
    if eligible:
        peak = max(e["tps"] for e in eligible)
        eff = min((e for e in eligible if e["tps"] >= (1 - RIGHTSIZE_TPS_TOL) * peak), key=lambda e: e["value"])
        cap = f"; KV capacity caps batch at {max_batch_capacity(best_result, args.batch):,}" if best_result else ""
        print(f"     -> efficient batch={eff['value']}  (TPS={eff['tps']:.1f}, TPOT={eff['tpot']*1e3:.1f} ms; "
              f"fills the FFN BLEN tile -- attention gains nothing past it{cap})")
    print("#" * 78)
    return sweeps, best_hw, best_result


# Software DSE bridge: accuracy (continuation PPL) <-> decode cost
def load_precision_points(path: str) -> list[dict]:
    """Read (tag, perplexity, per-component eff bits) from the software DSE CSV.
    Accepts the per-component schema (attn_w_bits/ffn_w_bits/kv_bits) or the
    single-weight one (w_eff_bits). `cost_mb_per_token` is optional; when absent
    the front is ordered by total stored bits instead of a byte proxy."""
    pts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("error"):           # crashed / penalty trials carry an error string
                continue
            try:
                ppl = float(r["cont_ppl"])
                if "attn_w_bits" in r and r.get("attn_w_bits") not in (None, ""):
                    attn, ffn, kv = float(r["attn_w_bits"]), float(r["ffn_w_bits"]), float(r["kv_bits"])
                else:   # single-weight CSV: reuse W for attention and FFN
                    w = float(r["w_eff_bits"])
                    attn = ffn = w
                    kv = float(r["kv_eff_bits"])
            except (KeyError, ValueError):
                continue
            act = r.get("act_bits")
            block = r.get("block")
            cost = r.get("cost_mb_per_token")
            sw_mb = float(cost) if cost not in (None, "") else (attn + ffn + kv)  # bits used only to order
            calibrated = any(m in r.get("tag", "").lower() for m in ("gptq", "rot"))
            pts.append({"tag": r.get("tag", ""), "ppl": ppl, "attn_bits": attn, "ffn_bits": ffn,
                        "kv_bits": kv, "sw_mb": sw_mb, "gptq": calibrated,
                        "act_bits": float(act) if act not in (None, "") else None,
                        "block": int(float(block)) if block not in (None, "") else 32})
    return pts


def pareto_front(points: list[dict]) -> list[dict]:
    """Keep only the best precisions: as memory cost rises, keep each point that lowers perplexity."""
    front, best_ppl = [], float("inf")
    for p in sorted(points, key=lambda p: p["sw_mb"]):
        if p["ppl"] < best_ppl - 1e-9:
            front.append(p)
            best_ppl = p["ppl"]
    return front


def _prec_from_point(p: dict, args) -> dict:
    """CSV point -> precision spec. Element widths subtract the per-block scale
    share before rounding (eff = elem + SCALE_BITS/block); rounding eff bits
    directly mis-sizes odd widths (e.g. MXINT3 @ block 16 = 3.5 eff bits would
    round to 4). The activation compute width joins the M-bit default when the CSV
    carries it."""
    share = SCALE_BITS / p.get("block", 32)

    def _elem(bits):
        return max(1, round(float(bits) - share))

    elems = [_elem(p["attn_bits"]), _elem(p["ffn_bits"]), _elem(p["kv_bits"])]
    if p.get("act_bits") is not None:
        elems.append(_elem(p["act_bits"]))
    return precision_from_components(p["attn_bits"], p["ffn_bits"], p["kv_bits"],
                                     attn_elem=elems[0], ffn_elem=elems[1], kv_elem=elems[2],
                                     m_bits=args.m_bits or max(elems),
                                     density_exp=args.density_exp)


def run_precision_sweep(args, model_path, dims, hw_cfg, isa, base_mem):
    """Run each point on the software PPL-vs-memory front on the decode chip ->
    MB/token, TPS, HBM-fit. When compute-bound, TPS barely moves with precision,
    so the trade-off is PPL vs MB."""
    points = load_precision_points(args.sweep)
    front = pareto_front(points)
    stride = max(1, args.output_seq // 24)
    print("=" * 96)
    print(f"  PRECISION SWEEP (accuracy vs decode cost) — {args.model}  batch={args.batch}  "
          f"in={args.input_seq} out={args.output_seq}")
    print(f"  {len(points)} precisions in {Path(args.sweep).name}; "
          f"{len(front)} best ones (perplexity-vs-memory front)")
    print("=" * 96)
    print(f"  {'precision':<40} {'perplexity':>10} {'attn/ffn/kv':>13} {'MB/tok':>8} {'TPS':>8} fits")
    rows = []
    for p in front:
        prec = _prec_from_point(p, args)
        # Clamp MLEN to this precision's bandwidth cap so the TPS is feasible;
        # VLEN follows MLEN. Memory cost (MB/token) does not depend on MLEN.
        mlen = min(hw_cfg.MLEN, mlen_bandwidth_cap(hw_cfg, prec))
        hw = hw_cfg.model_copy(update={"MLEN": mlen, "VLEN": mlen, "HBM_M_Prefetch_Amount": mlen})
        r = evaluate(model_path, dims, hw, isa, base_mem, prec, args.batch,
                     args.input_seq, args.output_seq, {"MLEN": mlen, "VLEN": mlen},
                     stride=stride, n_chips=args.chips)
        mb = r["avg_bytes_per_token"] / 1e6
        label = f"{prec['attn_elem']}/{prec['ffn_elem']}/{prec['kv_elem']}"
        print(f"  {p['tag'][:40]:<40} {p['ppl']:>10.3f} {label:>13} {mb:>8.1f} {r['tps']:>8.1f}  "
              f"{'yes' if r['fits_in_hbm'] else 'NO'}")
        rows.append({"tag": p["tag"], "ppl": p["ppl"], "mb": mb, "fits": r["fits_in_hbm"],
                     "gptq": p["gptq"], "label": label})
    print("=" * 96)
    return rows


def right_size(args, model_path, dims, base_hw, isa, base_mem, prec, stride):
    """Smallest area at peak throughput (TPS). VLEN follows MLEN (compiler constraint)."""
    cap = mlen_bandwidth_cap(base_hw, prec)
    start_mlen = min(base_hw.MLEN, cap)
    start = base_hw.model_copy(update={"MLEN": start_mlen, "VLEN": start_mlen,
                                       "HBM_M_Prefetch_Amount": start_mlen})
    best = {}
    for dim in ("MLEN", "BLEN", "HLEN"):
        eligible = []
        for v in SEARCH_SPACE[dim]:
            hw2, mem_over, _, geo = _candidate(start, dim, v)
            if not (_valid(*geo, hidden=dims["hidden"]) and _bandwidth_ok(geo[0], hw2, prec)):
                continue
            try:
                r = evaluate(model_path, dims, hw2, isa, base_mem, prec, args.batch,
                             args.input_seq, args.output_seq, mem_over, stride=stride, n_chips=args.chips)
            except Exception:
                continue
            if r["fits_in_hbm"]:
                eligible.append({"value": v, "tps": r["tps"], "area": area_multipliers(hw2)})
        if eligible:
            peak = max(e["tps"] for e in eligible)
            best[dim] = min((e for e in eligible if e["tps"] >= (1 - RIGHTSIZE_TPS_TOL) * peak),
                            key=lambda e: e["area"])["value"]
    mlen, blen = best.get("MLEN", start.MLEN), best.get("BLEN", start.BLEN)
    vlen, hlen = mlen, best.get("HLEN", start.HLEN)
    if _valid(mlen, blen, vlen, hlen, hidden=dims["hidden"]) and _bandwidth_ok(mlen, start, prec):
        return start.model_copy(update={"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen,
                                        "HBM_M_Prefetch_Amount": mlen})
    return start


def run_codesign(args, model_path, dims, base_hw, isa, base_mem):
    points = pareto_front(load_precision_points(args.codesign))
    stride = max(1, args.output_seq // 24)
    print("\n" + "#" * 104)
    print(f"[7] PRECISION x HARDWARE CO-DESIGN — {args.model}  batch={args.batch}  "
          f"in={args.input_seq} out={args.output_seq}")
    print("    Each precision: cap MLEN by bandwidth, then right-size the whole array")
    print("    ('bound' shows where each precision lands, no assumption made).")
    print("#" * 104)
    print(f"  {'precision':<34} {'PPL':>9} {'maxW|KV':>7} {'MLEN':>5} {'BLEN':>5} {'area(mm^2)':>10} "
          f"{'TPS':>8} {'max-batch':>10} {'chips':>5} {'bound':>5} fits")
    rows = []
    for p in points:
        prec = _prec_from_point(p, args)
        hw = right_size(args, model_path, dims, base_hw, isa, base_mem, prec, stride)
        r = evaluate(model_path, dims, hw, isa, base_mem, prec, args.batch, args.input_seq, args.output_seq,
                     {"MLEN": hw.MLEN, "BLEN": hw.BLEN, "VLEN": hw.VLEN, "HLEN": hw.HLEN},
                     stride=stride, n_chips=args.chips)
        max_batch = max_batch_capacity(r, args.batch)
        bound = "mem" if r["frac_mem_bound"] >= 0.5 else "cmp"
        label = f"{prec['attn_elem']}/{prec['ffn_elem']}/{prec['kv_elem']}"
        print(f"  {p['tag'][:34]:<34} {p['ppl']:>9.3f} {stream_bits(prec):>7} {hw.MLEN:>5} {hw.BLEN:>5} "
              f"{area_mm2(hw):>10.3f} {r['tps']:>8.1f} {max_batch:>10,} {r['n_chips']:>5} {bound:>5} {'yes' if r['fits_in_hbm'] else 'NO'}")
        rows.append({"ppl": p["ppl"], "tps": r["tps"], "max_batch": max_batch, "fits": r["fits_in_hbm"],
                     "area": area_mm2(hw), "bound": bound, "label": label, "n_chips": r["n_chips"]})
    print("#" * 104)
    return rows


# =============================================================================
# Tier-2: system comparison vs A100 / H100 / TPU roofline references
# =============================================================================
def run_compare(args, model_path, dims, hw_cfg, base_mem, prec, device_names):
    """Throughput vs A100/H100 at each device's max-fitting batch (full HBM use).
    Workload bytes/FLOPs are recomputed per device because they scale with its
    batch; util = how well the decode batch fills the M-tile (PLENA's small BLEN
    fills earlier than a GPU)."""
    kv, ctx = args.input_seq + args.output_seq // 2, args.input_seq + args.output_seq
    mem = LLMMemoryModel(model_path, base_mem.model_copy(update={
        "weight_bits": prec["ffn_bits"], "activation_bits": ACT_BITS, "kv_cache_bits": prec["kv_bits"]}),
        batch_size=1, input_seq_len=args.input_seq, output_seq_len=args.output_seq).mem
    wf = weight_footprint_bytes(mem, dims, prec)["total"]       # sharded once across the system
    kv_per_batch = kv_footprint_bytes(mem, dims, prec, ctx, 1)  # KV bytes for one stream at full context

    print("\n" + "=" * 112)
    print(f"[8] SYSTEM COMPARISON — {args.model} decode  in={args.input_seq} out={args.output_seq}")
    print(f"    precision attnW/ffnW/KV = {prec['attn_bits']:.2f}/{prec['ffn_bits']:.2f}/{prec['kv_bits']:.2f}"
          f"   BS = largest fitting each system's aggregate HBM (full-capacity utilisation)")
    print("=" * 112)
    print(f"  {'device':<11} {'peakTF':>8} {'BW(TB/s)':>9} {'cap(GB)':>8} {'BS':>8} {'util%':>6} "
          f"{'bound':>6} {'TPS':>11} {'xA100':>7}")
    rows, a100 = [], None
    for name in device_names:
        dev = plena_device(hw_cfg, base_mem) if name == "plena" else DEVICES[name]
        peak_c, peak_bw, cap = device_peaks(dev, prec)         # PLENA peak is bandwidth-capped per precision
        bs = max(1, int((cap - wf) // max(kv_per_batch, 1)))   # largest batch whose KV still fits
        tr = decode_traffic(mem, dims, kv, bs, prec)
        bytes_step = tr.read_bytes + tr.write_bytes
        flops_step = decode_step_flops(dims, kv, bs)
        m_tile = dev["blen"] if dev["kind"] == "plena" else dev["sq_dim"]
        util = fill_util(bs, m_tile)
        comp_t, mem_t = flops_step / (peak_c * util), bytes_step / peak_bw
        bound = "mem" if mem_t >= comp_t else "cmp"            # wall this device hits
        tps = bs / max(comp_t, mem_t)                          # tokens/s at the max-fitting batch
        r = dict(label=dev["label"], peak_c=peak_c, peak_bw=peak_bw, cap=cap, bs=bs, util=util,
                 bound=bound, tps=tps)
        rows.append(r)
        if name == "a100":
            a100 = r
    for r in rows:
        xt = r["tps"] / a100["tps"] if a100 else float("nan")
        print(f"  {r['label']:<11} {r['peak_c']/1e12:>8.0f} {r['peak_bw']/1e12:>9.2f} {r['cap']/1e9:>8.0f} "
              f"{r['bs']:>8,} {r['util']*100:>5.0f}% {r['bound']:>6} {r['tps']:>11.1f} {xt:>6.2f}x")
    print("=" * 112)
    print("    BS = largest batch fitting each system's aggregate HBM; util = decode-batch fill of the M-tile.")
    print("    PLENA row = the configured chip (geometry/HBM from the live config, compute at M-bit density).")
    return rows


def resolve_model_path(model_name, model_lib):
    p = Path(model_lib) / f"{model_name}.json"
    if not p.exists():
        avail = ", ".join(sorted(f.stem for f in Path(model_lib).glob("*.json")))
        raise FileNotFoundError(f"Model '{model_name}' not found in {model_lib}. Available: {avail}")
    return str(p)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llama-3.2-1b")
    ap.add_argument("--model-lib", default=str(_ANALYTIC.parent / "compiler" / "doc" / "Model_Lib"))
    ap.add_argument("--config", default=str(_ANALYTIC.parent / "plena_settings.toml"))
    ap.add_argument("--isa-lib", default=str(_HERE / "customISA_lib.json"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--input-seq", type=int, default=256)
    ap.add_argument("--output-seq", type=int, default=16384)
    ap.add_argument("--w-fmt", choices=["mxint", "mxfp"], default="mxint",
                    help="weight format, shared by attention + FFN weights")
    ap.add_argument("--kv-fmt", choices=["mxint", "mxfp"], default="mxint",
                    help="KV-cache format, independent of the weights (mixed precision)")
    ap.add_argument("--attn-w", default="4", help="attention-projection weight width (q/k/v/o)")
    ap.add_argument("--ffn-w", default="4", help="FFN/expert-projection weight width (gate/up/down)")
    ap.add_argument("--kv", default="4", help="KV-cache width")
    ap.add_argument("--block", type=int, default=32,
                    help="MX block size for effective-bits accounting. 32 = the software-DSE setting; "
                         "the emulator packs block=8, so its scale share is slightly larger")
    ap.add_argument("--m-bits", type=int, default=0,
                    help="MAC compute width M: HBM streams N-bit operands, the array multiplies at M, "
                         "writeback requantises to N. 0 = widest operand max(attnW, ffnW, KV) "
                         "(+ the activation compute width for CSV points)")
    ap.add_argument("--density-exp", type=float, default=DENSITY_EXP,
                    help="iso-area density exponent k: an M-bit array fits (4/M)^k x the reference "
                         "multipliers. Default 0 = precision-neutral compute; set 2.0 to enable the "
                         "density layer (until a Synopsys DC sweep calibrates it)")
    ap.add_argument("--hbm-gen", choices=sorted(HBM_GENS), default=None,
                    help="set HBM bandwidth and capacity from this generation x --hbm-channels "
                         "instead of the TOML HBM_WIDTH/HBM_SIZE")
    ap.add_argument("--hbm-channels", type=int, default=0,
                    help="HBM channel count (the searchable HBM knob); 0 = one full stack of --hbm-gen")
    ap.add_argument("--chips", type=int, default=0,
                    help="tensor-parallel decode chips over UALink (0 = auto: fewest HBM stacks that hold "
                         "the model). Large models (405B / 235B) need several stacks.")
    ap.add_argument("--area-model", choices=("proxy", "calibrated"), default="proxy",
                    help="chip-area model: 'proxy' = mm^2/multiplier; 'calibrated' = precision-aware "
                         "MatrixMachine structural census (DC-fitted, validated at large MLEN; "
                         "matches the known 0.237 mm^2 at 4x1024 MXINT4)")
    ap.add_argument("--link-gen", choices=("nvlink3", "nvlink4", "ualink", "pcie5"), default="nvlink4",
                    help="prefill->decode interconnect for the KV hand-off timing")
    ap.add_argument("--link-bw", type=float, default=0,
                    help="override the interconnect bandwidth, GB/s per direction (0 = --link-gen preset)")
    ap.add_argument("--mlen", type=int, default=0,
                    help="override the TOML MLEN (matrix tile length), e.g. 2048 for the baseline array")
    ap.add_argument("--blen", type=int, default=0,
                    help="override the TOML BLEN (block/batch tile), e.g. 32 for the baseline array")
    ap.add_argument("--bw-model", choices=("peak", "calibrated"), default="peak",
                    help="memory-time pricing: 'peak' = bytes / aggregate peak bandwidth; 'calibrated' = "
                         "per-class effective bandwidth measured on the emulator "
                         "(disagg_serve/calibration_bw.csv)")
    ap.add_argument("--search", action="store_true", help="right-size the decode hardware for this precision")
    ap.add_argument("--sweep", nargs="?", const=_DEFAULT_CSV, default=None,
                    help=f"precision sweep over the software CSV (default: {Path(_DEFAULT_CSV).name})")
    ap.add_argument("--codesign", nargs="?", const=_DEFAULT_CSV, default=None,
                    help="joint precision x hardware co-design over the software CSV (right-size per precision)")
    ap.add_argument("--compare", action="store_true",
                    help="Tier-2 system comparison vs A100/H100/TPU roofline references")
    ap.add_argument("--compare-devices", default="plena,a100,h100",
                    help="comma list from {plena,a100,h100,tpu}")
    args = ap.parse_args()

    model_path = resolve_model_path(args.model, args.model_lib)
    dims = load_model_dims(model_path)
    hw_cfg = load_hardware_config_from_toml(args.config)
    base_mem = load_memory_config_from_toml(args.config)
    if args.hbm_gen:                       # real HBM spec replaces the TOML bandwidth + capacity
        over = hbm_overrides(args.hbm_gen, args.hbm_channels)
        args.hbm_channels = over.pop("channels")
        hw_cfg, base_mem = hw_cfg.model_copy(update=over), base_mem.model_copy(update=over)
    tile_over = {k: v for k, v in (("MLEN", args.mlen), ("BLEN", args.blen)) if v}
    if tile_over:                          # array-shape overrides, e.g. the 2048x32 baseline
        hw_cfg = hw_cfg.model_copy(update=tile_over)

    # Software-DSE CSVs come from the separate software search; skip until they exist.
    for flag in ("sweep", "codesign"):
        path = getattr(args, flag)
        if path and not Path(path).exists():
            print(f"[--{flag} skipped: {path} not found -- run the software DSE to generate it]")
            setattr(args, flag, None)

    prec = build_precision(args)
    if args.area_model == "calibrated":
        set_area_model("calibrated", prec)
    bw_model = None
    if args.bw_model == "calibrated":
        # Repo root on sys.path so the disagg_serve package resolves when this
        # file is run as a script from anywhere.
        sys.path.insert(0, str(_HERE.parent.parent))
        from analytic_models.disagg_serve.memory import CalibratedBandwidth
        bw_model = CalibratedBandwidth.load()
    stride = max(1, args.output_seq // 256)            # subsample the context loop for speed
    result = evaluate(model_path, dims, hw_cfg, args.isa_lib, base_mem, prec,
                      args.batch, args.input_seq, args.output_seq, stride=stride, n_chips=args.chips,
                      bw_model=bw_model, hbm_gen=args.hbm_gen or "HBM2",
                      hbm_channels=args.hbm_channels or 8)
    print_report(args, dims, hw_cfg, prec, result)

    if args.search:
        run_search(args, model_path, dims, hw_cfg, args.isa_lib, base_mem, prec)
    if args.sweep:
        run_precision_sweep(args, model_path, dims, hw_cfg, args.isa_lib, base_mem)
    if args.codesign:
        run_codesign(args, model_path, dims, hw_cfg, args.isa_lib, base_mem)
    if args.compare:
        names = [d.strip() for d in args.compare_devices.split(",") if d.strip() in DEVICES]
        if not names:
            raise SystemExit(f"--compare-devices: no valid devices in '{args.compare_devices}' "
                             f"(choose from {', '.join(DEVICES)})")
        run_compare(args, model_path, dims, hw_cfg, base_mem, prec, names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
