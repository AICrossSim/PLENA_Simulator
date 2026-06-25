"""
Decode-Chip Analytic Model — Disaggregated Serving on PLENA.

Models only the decode chip. The prefill chip is separate (FP16) and just hands off the
KV cache for `input_seq` tokens, requantised to the decode chip's KV precision.

Each generated token costs  step_time = max(compute_time, memory_time):
  compute_time = decode-step cycles / clock        does NOT change with precision (the matrix
                                                   array has a fixed MLEN*BLEN multiplier count)
  memory_time  = decode-step HBM bytes / bandwidth  drops with precision (fewer bits => fewer
                                                   bytes/token, and more batch/context fits in HBM)
So if compute is the bottleneck, lower precision only buys HBM capacity; if memory is the
bottleneck, lower precision also raises throughput.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402

_HERE = Path(__file__).resolve().parent
_ANALYTIC = _HERE.parent
for _sub in ("performance", "memory", "utilisation", "roofline"):
    p = str(_ANALYTIC / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from perf_model import PerfModel, load_hardware_config_from_toml          # noqa: E402
from memory_model import MemoryConfig, load_memory_config_from_toml        # noqa: E402
from llm_memory_model import LLMMemoryModel                                # noqa: E402
from utilisation_model import LLaMAUtilizationModel                        # noqa: E402

FREQ_HZ = 1.0e9                      # 1 GHz clock
DECODE_PIPELINE_FACTOR = 2           # covers instruction-issue + memory pipeline stages per step
SCALE_BITS = 8                       # each MX block carries one 8-bit shared scale (E8M0)
MXFP_FORMATS = {"E1M2": (1, 2), "E2M1": (2, 1), "E4M3": (4, 3), "E5M2": (5, 2)}

# The baseline chip's matrix array is 4096 multipliers and measures 0.237 mm^2 (7nm). Silicon
# area scales with the multiplier count, so every searched config is reported relative to this.
REF_MULTIPLIERS = 4096
REF_MM2 = 0.237
MM2_PER_MULTIPLIER = REF_MM2 / REF_MULTIPLIERS

_DEFAULT_CSV = str(_HERE / "software_disagg_decode.csv")


def effective_bits(fmt: str, width, block: int) -> float:
    """Average stored bits per element, including the per-block shared scale (SCALE_BITS/block)."""
    elem = int(width) if fmt == "mxint" else 1 + int(width[0]) + int(width[1])   # MXFP: 1+exp+frac
    return elem + SCALE_BITS / block


def width_label(fmt: str, width) -> str:
    return f"MXINT{int(width)}" if fmt == "mxint" else f"MXFP_E{width[0]}M{width[1]}"


def precision_from_bits(w_bits, a_bits, kv_bits, w_label="W", a_label="A", kv_label="KV",
                        w_elem_bits=None, kv_elem_bits=None) -> dict:
    """A precision point as per-axis effective bits.

    `w_elem_bits`/`kv_elem_bits` are the plain weight/KV element widths (no block scale). Decode
    streams weights AND KV from HBM, so the wider of the two bounds MLEN via the bandwidth check
    MLEN * max(W,KV) <= HBM_WIDTH; they default to the rounded effective bits.
    """
    return {"w_bits": float(w_bits), "a_bits": float(a_bits), "kv_bits": float(kv_bits),
            "w_label": w_label, "a_label": a_label, "kv_label": kv_label,
            "w_elem_bits": int(round(float(w_bits))) if w_elem_bits is None else int(w_elem_bits),
            "kv_elem_bits": int(round(float(kv_bits))) if kv_elem_bits is None else int(kv_elem_bits)}


def build_precision(args) -> dict:
    """Decode precision from --fmt/--w/--a/--kv/--block."""
    if args.fmt == "mxint":
        w, a, kv = int(args.w), int(args.a), int(args.kv)
        w_elem, kv_elem = int(args.w), int(args.kv)
    else:  # mxfp: --w/--a/--kv are tokens like E2M1
        w, a, kv = MXFP_FORMATS[args.w], MXFP_FORMATS[args.a], MXFP_FORMATS[args.kv]
        w_elem, kv_elem = 1 + w[0] + w[1], 1 + kv[0] + kv[1]
    return precision_from_bits(
        effective_bits(args.fmt, w, args.block),
        effective_bits(args.fmt, a, args.block),
        effective_bits(args.fmt, kv, args.block),
        width_label(args.fmt, w), width_label(args.fmt, a), width_label(args.fmt, kv),
        w_elem_bits=w_elem, kv_elem_bits=kv_elem)


def stream_bits(prec) -> int:
    """Widest operand streamed from HBM in decode = max(W, KV) bits (activations stay on-chip)."""
    return max(prec["w_elem_bits"], prec["kv_elem_bits"])


def mlen_bandwidth_cap(hw_cfg, prec) -> int:
    """Largest MLEN the HBM can feed per cycle: HBM_WIDTH / max(W,KV). Higher precision => smaller MLEN."""
    return hw_cfg.HBM_WIDTH // stream_bits(prec)


def load_model_dims(path: str) -> dict:
    with open(path) as f:
        p = json.load(f)
    ah = p["num_attention_heads"]
    return {"hidden": p["hidden_size"], "heads": ah, "kv_heads": p["num_key_value_heads"],
            "head_dim": p.get("head_dim", p["hidden_size"] // ah), "layers": p["num_hidden_layers"],
            "inter": p["intermediate_size"], "vocab": p["vocab_size"]}


def build_decode_memory_config(base: MemoryConfig, prec: dict, hw_over: dict) -> MemoryConfig:
    """Base TOML memory config + this precision (effective bits) + any per-search HW overrides."""
    update = {"weight_bits": prec["w_bits"], "activation_bits": prec["a_bits"],
              "kv_cache_bits": prec["kv_bits"], "weight_format": prec["w_label"],
              "activation_format": prec["a_label"], "kv_cache_format": prec["kv_label"]}
    update.update(hw_over)
    return base.model_copy(update=update)


def peak_hbm_bw_bytes(hw_cfg) -> float:
    """Peak HBM bandwidth = (HBM_WIDTH / 8) * clock  [bytes/s]."""
    return (hw_cfg.HBM_WIDTH / 8.0) * FREQ_HZ


def matrix_overfetch_factor(hw_cfg) -> float:
    """Wasted-read multiplier on matrix loads. The prefetch reads M_LOAD elements, rounded up to
    a multiple of MLEN, so M_LOAD > MLEN reads M_LOAD/MLEN x the needed bytes. Optimal: M_LOAD=MLEN."""
    mlen = hw_cfg.MLEN
    m_load = getattr(hw_cfg, "HBM_M_Prefetch_Amount", mlen)
    return math.ceil(max(m_load, mlen) / mlen)


def area_multipliers(hw_cfg) -> int:
    """Matrix-array multiplier count = MLEN * BLEN (sets silicon area)."""
    return hw_cfg.MLEN * hw_cfg.BLEN


def area_mm2(hw_cfg) -> float:
    return area_multipliers(hw_cfg) * MM2_PER_MULTIPLIER


def onchip_activation_bytes(d: dict, batch: int) -> int:
    """Decode activation working set held on-chip in FP16 (Vector SRAM); never written to HBM."""
    return math.ceil(batch * (d["hidden"] + d["inter"]) * 16 / 8)


def decode_token_components(perf: PerfModel, d: dict, kv: int, batch: int) -> dict:
    """Cycles to generate one token: every per-layer op x layers, then the once-per-token head ops."""
    h, ah, kvh, hd, inter, layers = (d["hidden"], d["heads"], d["kv_heads"],
                                     d["head_dim"], d["inter"], d["layers"])
    layer = {
        "RMSNorm (x2)":            perf.rms_layer(h, 1, batch, "decode") * 2,
        "Q/K/V proj + RoPE":       perf.projection(h, ah, kvh, hd, 1, batch, "decode"),
        "Flash attention":         perf.flash_attention(ah, kvh, hd, 1, kv, batch, "decode"),
        "Output projection (W_O)": perf.output_projection(h, ah, hd, 1, batch, "decode"),
        "Residual adds (x2)":      perf.residual(h, 1, batch, "decode") * 2,
        "FFN (gate/up/down)":      perf.feed_forward(h, inter, 1, batch, "decode"),
    }
    comp = {f"{k} x{layers} layers": v * layers for k, v in layer.items()}
    comp["Embedding lookup"] = perf.embeddings(h, 1, batch, "decode")
    comp["LM head"] = perf.lm_head(h, d["vocab"], batch)
    comp["Vocab softmax"] = perf.softmax_full_seq(d["vocab"], 1, batch)
    return comp


def decode_token_cycles(perf: PerfModel, d: dict, kv: int, batch: int) -> int:
    return sum(decode_token_components(perf, d, kv, batch).values())


def decode_step_flops(d: dict, kv: int, batch: int) -> int:
    """Real FLOPs for one decode step over the batch (2 per MAC). Used for arithmetic intensity."""
    h, ah, kvh, hd = d["hidden"], d["heads"], d["kv_heads"], d["head_dim"]
    qkvo = h * ah * hd + 2 * (h * kvh * hd) + (ah * hd) * h    # Q, K, V, O projections
    ffn = 2 * h * d["inter"] + d["inter"] * h                  # gate, up, down
    attn = 2 * ah * hd * kv                                    # QK^T + attention @ V
    per_layer = qkvo + ffn + attn
    return 2 * batch * (per_layer * d["layers"] + h * d["vocab"])   # + LM head (once per token)


def run_decode_loop(perf, mem_model, d, input_seq, output_seq, batch, peak_bw, stride, overfetch):
    """Generate `output_seq` tokens; per token step_time = max(compute, memory). `stride`
    subsamples the growing-context loop for speed (each sample stands in for the skipped tokens)."""
    total_time, total_bytes, first_step, mem_bound = 0.0, 0, None, 0
    t = 0
    while t < output_seq:
        kv = input_seq + t                                    # KV cache grows by one each token
        compute_time = decode_token_cycles(perf, d, kv, batch) * DECODE_PIPELINE_FACTOR / FREQ_HZ
        tr = mem_model.compute_decode_traffic(num_output_tokens=t).total_traffic
        bytes_tok = tr.read_bytes * overfetch + tr.write_bytes
        memory_time = bytes_tok / peak_bw

        step_time = max(compute_time, memory_time)
        span = min(stride, output_seq - t)                    # tokens this sample represents
        total_time += step_time * span
        total_bytes += bytes_tok * span
        if memory_time >= compute_time:
            mem_bound += span
        if first_step is None:
            first_step = step_time
        t += stride
    return {"total_time": total_time, "tpot": total_time / output_seq,
            "tps": (batch * output_seq) / total_time, "first_step": first_step,
            "avg_bytes_per_token": total_bytes / output_seq, "frac_mem_bound": mem_bound / output_seq}


def evaluate(model_path, dims, hw_cfg, isa_path, base_mem, prec, batch,
             input_seq, output_seq, hw_over=None, stride=1):
    """Build the decode chip for one (hardware, precision, batch) point and return its metrics."""
    perf = PerfModel(hw_cfg, isa_path)
    mem_cfg = build_decode_memory_config(base_mem, prec, hw_over or {})
    mem_model = LLMMemoryModel(model_path, mem_cfg, batch_size=batch,
                               input_seq_len=input_seq, output_seq_len=output_seq)
    peak_bw = peak_hbm_bw_bytes(hw_cfg)
    loop = run_decode_loop(perf, mem_model, dims, input_seq, output_seq, batch, peak_bw,
                           stride, matrix_overfetch_factor(hw_cfg))

    an = mem_model.analyze()
    # HBM holds only weights + KV cache; activations live on-chip (Vector SRAM), never spilled.
    hbm_required = an.weight_footprint.total_bytes + an.kv_cache_footprint.total_bytes
    loop.update(hbm_required=hbm_required, fits_in_hbm=hbm_required <= an.hbm_capacity_bytes,
                analysis=an, mem_model=mem_model, perf=perf, peak_bw=peak_bw)
    return loop


def _fmt_bytes(n):
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if n >= div:
            return f"{n / div:.3f} {unit}"
    return f"{n:.0f} B"


def print_report(args, dims, hw_cfg, prec, result, model_path):
    bar = "=" * 78
    peak_compute = 2 * hw_cfg.MLEN * hw_cfg.BLEN * FREQ_HZ
    peak_bw = result["peak_bw"]
    print(bar)
    print("  DECODE-CHIP REPORT — Disaggregated Serving on PLENA")
    print(bar)
    print(f"  Model:     {args.model}  (hidden={dims['hidden']}, layers={dims['layers']}, "
          f"heads={dims['heads']}/{dims['kv_heads']}KV, head_dim={dims['head_dim']})")
    print(f"  Workload:  batch={args.batch}  input_seq={args.input_seq} (handed-off KV)  "
          f"output_seq={args.output_seq}")
    print(f"  Precision: W:{prec['w_label']} A:{prec['a_label']} KV:{prec['kv_label']} (block {args.block})"
          f"  ->  {prec['w_bits']:.3f}/{prec['a_bits']:.3f}/{prec['kv_bits']:.3f} eff bits"
          f"   [prefill chip: separate FP16]")
    print(f"  Hardware:  MLEN={hw_cfg.MLEN} BLEN={hw_cfg.BLEN} VLEN={hw_cfg.VLEN} HLEN={hw_cfg.HLEN}  "
          f"clock={FREQ_HZ/1e9:.0f} GHz")
    print(f"             peak compute = 2*MLEN*BLEN*clock = {peak_compute/1e12:.2f} TFLOP/s   "
          f"peak HBM BW = {peak_bw/1e9:.0f} GB/s")
    print(f"             matrix array = {area_multipliers(hw_cfg):,} multipliers (~{area_mm2(hw_cfg):.3f} mm^2)")
    # Bandwidth - HBM feeds MLEN operands/cycle, so precision caps how wide MLEN can be
    # Higher precision (more bits) => smaller MLEN ceiling => less peak compute
    print(f"             bandwidth bound: MLEN <= HBM_WIDTH / max(W,KV) = {hw_cfg.HBM_WIDTH}/{stream_bits(prec)} "
          f"= {mlen_bandwidth_cap(hw_cfg, prec)}")

    # The first token (TTFT) is the prefill chip's job; the decode chip's first step makes token #2.
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

    an = result["analysis"]
    # Capacity - weights are fixed and KV grows with batch x context
    # The largest batch that fits is (HBM - weights) / (KV bytes per batch)
    # Higher precision or longer context like agentic workload => bigger KV => smaller max batch.
    kv_per_batch = an.kv_cache_footprint.total_bytes / max(args.batch, 1)
    max_batch = int((an.hbm_capacity_bytes - an.weight_footprint.total_bytes) // max(kv_per_batch, 1))
    print("\n[3] MEMORY  (HBM = weights + KV; activations stay on-chip)")
    print(f"      Weights (HBM):        {_fmt_bytes(an.weight_footprint.total_bytes)}")
    print(f"      KV cache (HBM):       {_fmt_bytes(an.kv_cache_footprint.total_bytes)}  "
          f"(context={args.input_seq+args.output_seq}, batch={args.batch})")
    print(f"      HBM used / capacity:  {_fmt_bytes(result['hbm_required'])} / "
          f"{_fmt_bytes(an.hbm_capacity_bytes)}  ->  {'FITS' if result['fits_in_hbm'] else 'EXCEEDS'} "
          f"({result['hbm_required']/an.hbm_capacity_bytes*100:.1f}%)")
    print(f"      Activations on-chip:  {_fmt_bytes(onchip_activation_bytes(dims, args.batch))}")
    print(f"      Max batch (Capacity bound): {max_batch}  (KV grows with batch x context)")
    print(f"      Bytes / output token: {_fmt_bytes(result['avg_bytes_per_token'])}")

    util = LLaMAUtilizationModel(
        model_config_path=model_path,
        hardware_config={"MLEN": hw_cfg.MLEN, "BLEN": hw_cfg.BLEN, "VLEN": hw_cfg.VLEN},
        batch_size=args.batch, input_seq_len=args.input_seq, output_seq_len=args.output_seq,
    ).compute_decode_utilization(kv_size=avg_kv, verbose=False)
    perf = result["perf"]
    compute_time = decode_token_cycles(perf, dims, avg_kv, args.batch) * DECODE_PIPELINE_FACTOR / FREQ_HZ
    tr = result["mem_model"].compute_decode_traffic(num_output_tokens=args.output_seq // 2).total_traffic
    bytes_tok = tr.read_bytes * matrix_overfetch_factor(hw_cfg) + tr.write_bytes
    achieved_bw = bytes_tok / max(compute_time, bytes_tok / peak_bw)
    # FFN array use peaks when BLEN ~ batch; attention use is set by how many heads run at once
    # (MLEN//HLEN). These raise TPS only when compute-bound; when memory-bound, HBM sets the pace.
    print("\n[4] UTILISATION  (@ avg context)")
    print(f"      Matrix array, attention:  {util['attention']['utilization']*100:.1f}% of peak")
    print(f"      Matrix array, FFN:        {util['ffn']['utilization']*100:.1f}% of peak")
    print(f"      HBM bandwidth:            {achieved_bw/1e9:.1f} / {peak_bw/1e9:.0f} GB/s "
          f"({achieved_bw/peak_bw*100:.1f}% of peak)")

    ai = decode_step_flops(dims, avg_kv, args.batch) / bytes_tok
    mem_bound = bytes_tok / peak_bw >= compute_time         # each token waits on the larger time
    print(f"\n[5] ROOFLINE  (decode step @ avg context kv={avg_kv})")
    print(f"      Arithmetic intensity:  {ai:.2f} FLOP/byte   (ridge {peak_compute/peak_bw:.1f})")
    print(f"      compute time / token:  {compute_time*1e3:.3f} ms")
    print(f"      memory  time / token:  {bytes_tok/peak_bw*1e3:.3f} ms")
    print(f"      -> Bottleneck:         {'MEMORY-bound — HBM bandwidth' if mem_bound else 'COMPUTE-bound — the matrix array'}"
          f" sets the pace  ({result['frac_mem_bound']*100:.0f}% of tokens memory-bound)")
    comps = decode_token_components(perf, dims, avg_kv, args.batch)
    tot = sum(comps.values())
    print("      Decode-step cycles per operation:")
    for name, c in comps.items():
        print(f"        {name:<34} {c*DECODE_PIPELINE_FACTOR:>14,d} cyc  ({c/tot*100:5.1f}%)")
    print(bar)


# Excluded:
#   M_LOAD is pinned to MLEN (no wasted reads); the vector-load/-write amounts are tiny because
#   activations stay on-chip; the SRAM sizes are fit-or-fail scratchpads, not throughput knobs.
SEARCH_SPACE = {
    "MLEN": [64, 128, 256, 512, 1024],   # matrix reduction tile (<=1024 here, capped by the bandwidth check)
    "BLEN": [4, 8, 16, 32, 64, 128],      # sub-array width; area = MLEN*BLEN; FFN use peaks at BLEN~batch
    "VLEN": [512, 1024, 2048],       # vector-unit width
    "HLEN": [16, 32, 64, 128],       # head lane; MLEN//HLEN heads run in parallel
    "BATCH": [1, 4, 8, 16, 64, 256],    # serving knob (throughput vs latency); flips compute/memory bound
}
RIGHTSIZE_TPS_TOL = 0.01             # a config within 1% of the best TPS still counts as "peak"


def _valid(mlen, blen, vlen, hlen) -> bool:
    """Geometry the array requires: MLEN divisible by BLEN and HLEN, MLEN >= HLEN >= BLEN, VLEN >= BLEN."""
    return mlen % blen == 0 and mlen % hlen == 0 and blen <= hlen <= mlen and vlen >= blen


def _bandwidth_ok(mlen, hw_cfg, prec) -> bool:
    """Precision constraint: the matrix unit consumes MLEN operands per cycle, so HBM must deliver
    that many bits per cycle -- MLEN * max(W,KV) <= HBM_WIDTH, else the systolic array starves"""
    return mlen <= mlen_bandwidth_cap(hw_cfg, prec)


def _candidate(hw_cfg, dim, value):
    """One swept value with the other axes left at baseline. M_LOAD follows MLEN (no wasted reads)."""
    mlen, blen, vlen, hlen = hw_cfg.MLEN, hw_cfg.BLEN, hw_cfg.VLEN, hw_cfg.HLEN
    batch = None
    if dim == "BATCH":   batch = value
    elif dim == "MLEN":  mlen = value
    elif dim == "BLEN":  blen = value
    elif dim == "VLEN":  vlen = value
    elif dim == "HLEN":  hlen = value
    hw2 = hw_cfg.model_copy(update={"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen,
                                    "HBM_M_Prefetch_Amount": mlen})
    return hw2, {"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen}, batch, (mlen, blen, vlen, hlen)


def run_search(args, model_path, dims, base_hw, isa, base_mem, prec):
    """Two phases: (1) right-size each hardware axis at the user's batch -- the smallest array that
    still reaches ~peak TPS; (2) sweep batch ON the right-sized chip to show its throughput/latency
    trade-off (Matches the batch the chip was built for)."""
    stride = max(1, args.output_seq // 24)
    # Clamp the starting MLEN to this precision's bandwidth cap so every per-axis sweep is feasible
    cap = mlen_bandwidth_cap(base_hw, prec)
    base_hw = base_hw.model_copy(update={"MLEN": min(base_hw.MLEN, cap),
                                         "HBM_M_Prefetch_Amount": min(base_hw.MLEN, cap)})

    def sweep_axis(dim, values, hw, batch_fixed):
        """Sweep one axis with the others fixed at `hw`; print the table, return (rows, eligible)."""
        print(f"\n  -- {dim} sweep --")
        print(f"     {'value':>6} | {'TPOT(ms)':>9} | {'TPS':>9} | {'area(mm^2)':>10} | {'bound':>7} | fits")
        rows, eligible = [], []
        for v in values:
            hw2, mem_over, batch_v, geo = _candidate(hw, dim, v)
            batch = batch_v if batch_v is not None else batch_fixed
            if not (_valid(*geo) and _bandwidth_ok(geo[0], hw2, prec)):
                continue
            try:
                r = evaluate(model_path, dims, hw2, isa, base_mem, prec, batch,
                             args.input_seq, args.output_seq, mem_over, stride=stride)
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

    # Phase 1: right-size each hardware axis at the user's batch.
    sweeps, best = {}, {}
    for dim in ("MLEN", "BLEN", "VLEN", "HLEN"):
        rows, eligible = sweep_axis(dim, SEARCH_SPACE[dim], base_hw, args.batch)
        sweeps[dim] = rows
        if not eligible:
            continue
        peak = max(e["tps"] for e in eligible)
        knee = min((e for e in eligible if e["tps"] >= (1 - RIGHTSIZE_TPS_TOL) * peak), key=lambda e: e["area"])
        best[dim] = knee["value"]
        print(f"     -> right-size {dim}={knee['value']}  (TPS={knee['tps']:.1f}, "
              f"{knee['area']*MM2_PER_MULTIPLIER:.3f} mm^2)")

    # Assemble the right-sized chip.
    mlen, blen = best.get("MLEN", base_hw.MLEN), best.get("BLEN", base_hw.BLEN)
    vlen, hlen = best.get("VLEN", base_hw.VLEN), best.get("HLEN", base_hw.HLEN)
    print(f"\n  -- Right-sized decode chip @ batch={args.batch} --")
    best_hw, best_result = None, None
    if _valid(mlen, blen, vlen, hlen) and _bandwidth_ok(mlen, base_hw, prec):
        best_hw = base_hw.model_copy(update={"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen,
                                             "HBM_M_Prefetch_Amount": mlen})
        best_result = evaluate(model_path, dims, best_hw, isa, base_mem, prec, args.batch, args.input_seq,
                               args.output_seq, {"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen}, stride=stride)
        area = area_multipliers(best_hw)
        print(f"     MLEN={mlen} BLEN={blen} VLEN={vlen} HLEN={hlen} batch={args.batch}")
        print(f"     -> TPOT={best_result['tpot']*1e3:.3f} ms   TPS={best_result['tps']:.1f}   "
              f"{'memory-bound' if best_result['frac_mem_bound'] >= 0.5 else 'compute-bound'}")
        print(f"     -> {area:,} mult (~{area_mm2(best_hw):.3f} mm^2 = {area/REF_MULTIPLIERS:.2f}x the "
              f"{REF_MM2:.3f} mm^2 baseline)   fits={'yes' if best_result['fits_in_hbm'] else 'NO'}")
    else:
        print("     (best-per-axis combination breaks the geometry/bandwidth rules)")

    # Phase 2: batch trade-off on the right-sized chip (efficient batch ~ BLEN -> matches your batch).
    rows, eligible = sweep_axis("BATCH", SEARCH_SPACE["BATCH"], best_hw or base_hw, args.batch)
    sweeps["BATCH"] = rows
    if eligible:
        peak = max(e["tps"] for e in eligible)
        eff = min((e for e in eligible if e["tps"] >= (1 - RIGHTSIZE_TPS_TOL) * peak), key=lambda e: e["value"])
        print(f"     -> efficient batch={eff['value']}  (TPS={eff['tps']:.1f}, TPOT={eff['tpot']*1e3:.1f} ms; "
              f"bigger batch only adds latency)")
    print("#" * 78)
    return sweeps, best_hw, best_result


def load_precision_points(path: str) -> list[dict]:
    """Read (tag, perplexity, per-axis effective bits, software MB/token) per precision from the CSV."""
    pts = []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                pts.append({"tag": r["tag"], "ppl": float(r["cont_ppl"]),
                            "w_bits": float(r["w_eff_bits"]), "a_bits": float(r["a_eff_bits"]),
                            "kv_bits": float(r["kv_eff_bits"]), "sw_mb": float(r["cost_mb_per_token"])})
            except (KeyError, ValueError):
                continue
    return pts


def pareto_front(points: list[dict]) -> list[dict]:
    """Keep only the best precisions: as memory cost rises, keep each point that lowers perplexity."""
    front, best_ppl = [], float("inf")
    for p in sorted(points, key=lambda p: p["sw_mb"]):
        if p["ppl"] < best_ppl - 1e-9:
            front.append(p)
            best_ppl = p["ppl"]
    return front


def run_precision_sweep(args, model_path, dims, hw_cfg, isa, base_mem):
    """Bridge accuracy <-> decode cost: take the perplexity-vs-memory Pareto front from the software
    CSV and run each point on the (configured) decode chip, reporting MB/token, TPS and HBM-fit.
    If compute bound, TPS barely moves with precision -- the trade-off is PPL vs MB."""
    points = load_precision_points(args.sweep)
    front = pareto_front(points)
    stride = max(1, args.output_seq // 24)
    print("=" * 92)
    print(f"  PRECISION SWEEP (accuracy vs decode cost) — {args.model}  batch={args.batch}  "
          f"in={args.input_seq} out={args.output_seq}")
    print(f"  {len(points)} precisions in {Path(args.sweep).name}; "
          f"{len(front)} best ones (perplexity-vs-memory front)")
    print("=" * 92)
    print(f"  {'precision':<36} {'perplexity':>10} {'W/A/KV bits':>13} {'MB/tok':>8} {'TPS':>8} fits")
    rows = []
    for p in front:
        prec = precision_from_bits(p["w_bits"], p["a_bits"], p["kv_bits"],
                                   w_elem_bits=round(p["w_bits"]), kv_elem_bits=round(p["kv_bits"]))
        # Clamp MLEN to this precision's bandwidth cap so the TPS is feasible (a wider MLEN would
        # starve the array). Memory cost (MB/token) is unaffected -- it does not depend on MLEN.
        mlen = min(hw_cfg.MLEN, mlen_bandwidth_cap(hw_cfg, prec))
        hw = hw_cfg.model_copy(update={"MLEN": mlen, "HBM_M_Prefetch_Amount": mlen})
        r = evaluate(model_path, dims, hw, isa, base_mem, prec, args.batch,
                     args.input_seq, args.output_seq, {"MLEN": mlen}, stride=stride)
        mb = r["avg_bytes_per_token"] / 1e6
        print(f"  {p['tag']:<36} {p['ppl']:>10.3f} "
              f"{p['w_bits']:>4.2f}/{p['a_bits']:.2f}/{p['kv_bits']:.2f} {mb:>8.1f} {r['tps']:>8.1f}  "
              f"{'yes' if r['fits_in_hbm'] else 'NO'}")
        rows.append({"tag": p["tag"], "ppl": p["ppl"], "mb": mb, "fits": r["fits_in_hbm"],
                     "gptq": "gptq" in p["tag"].lower(),
                     "label": f"{p['w_bits']:.0f}/{p['a_bits']:.0f}/{p['kv_bits']:.0f}"})
    print("=" * 92)
    return rows


def right_size(args, model_path, dims, base_hw, isa, base_mem, prec, stride):
    """Smallest area at peak trhoughput (TPS)"""
    cap = mlen_bandwidth_cap(base_hw, prec)
    start = base_hw.model_copy(update={"MLEN": min(base_hw.MLEN, cap),
                                       "HBM_M_Prefetch_Amount": min(base_hw.MLEN, cap)})
    best = {}
    for dim in ("MLEN", "BLEN", "VLEN", "HLEN"):
        eligible = []
        for v in SEARCH_SPACE[dim]:
            hw2, mem_over, _, geo = _candidate(start, dim, v)
            if not (_valid(*geo) and _bandwidth_ok(geo[0], hw2, prec)):
                continue
            try:
                r = evaluate(model_path, dims, hw2, isa, base_mem, prec, args.batch,
                             args.input_seq, args.output_seq, mem_over, stride=stride)
            except Exception:
                continue
            if r["fits_in_hbm"]:
                eligible.append({"value": v, "tps": r["tps"], "area": area_multipliers(hw2)})
        if eligible:
            peak = max(e["tps"] for e in eligible)
            best[dim] = min((e for e in eligible if e["tps"] >= (1 - RIGHTSIZE_TPS_TOL) * peak),
                            key=lambda e: e["area"])["value"]
    mlen, blen = best.get("MLEN", start.MLEN), best.get("BLEN", start.BLEN)
    vlen, hlen = best.get("VLEN", start.VLEN), best.get("HLEN", start.HLEN)
    if _valid(mlen, blen, vlen, hlen) and _bandwidth_ok(mlen, start, prec):
        return start.model_copy(update={"MLEN": mlen, "BLEN": blen, "VLEN": vlen, "HLEN": hlen,
                                        "HBM_M_Prefetch_Amount": mlen})
    return start


def run_codesign(args, model_path, dims, base_hw, isa, base_mem):
    points = pareto_front(load_precision_points(args.codesign))
    stride = max(1, args.output_seq // 24)
    print("\n" + "#" * 100)
    print(f"[7] PRECISION x HARDWARE CO-DESIGN — {args.model}  batch={args.batch}  "
          f"in={args.input_seq} out={args.output_seq}")
    print("    Each precision: MLEN capped by bandwidth max(W,KV), then the WHOLE array is right-sized")
    print("    (no compute/memory-bound assumption -- 'bound' shows where each precision lands).")
    print("#" * 100)
    print(f"  {'precision':<34} {'PPL':>9} {'W|KV':>5} {'MLEN':>5} {'BLEN':>5} {'area(mm^2)':>10} "
          f"{'TPS':>8} {'max-batch':>10} {'bound':>5} fits")
    rows = []
    for p in points:
        prec = precision_from_bits(p["w_bits"], p["a_bits"], p["kv_bits"],
                                   w_elem_bits=round(p["w_bits"]), kv_elem_bits=round(p["kv_bits"]))
        hw = right_size(args, model_path, dims, base_hw, isa, base_mem, prec, stride)
        r = evaluate(model_path, dims, hw, isa, base_mem, prec, args.batch, args.input_seq, args.output_seq,
                     {"MLEN": hw.MLEN, "BLEN": hw.BLEN, "VLEN": hw.VLEN, "HLEN": hw.HLEN}, stride=stride)
        an = r["analysis"]
        kv_per_batch = an.kv_cache_footprint.total_bytes / max(args.batch, 1)
        max_batch = int((an.hbm_capacity_bytes - an.weight_footprint.total_bytes) // max(kv_per_batch, 1))
        bound = "mem" if r["frac_mem_bound"] >= 0.5 else "cmp"
        print(f"  {p['tag']:<34} {p['ppl']:>9.3f} {stream_bits(prec):>5} {hw.MLEN:>5} {hw.BLEN:>5} "
              f"{area_mm2(hw):>10.3f} {r['tps']:>8.1f} {max_batch:>10,} {bound:>5} {'yes' if r['fits_in_hbm'] else 'NO'}")
        rows.append({"ppl": p["ppl"], "tps": r["tps"], "max_batch": max_batch, "fits": r["fits_in_hbm"],
                     "area": area_mm2(hw), "bound": bound,
                     "label": f"{p['w_bits']:.0f}/{p['a_bits']:.0f}/{p['kv_bits']:.0f}"})
    print("#" * 100)
    return rows


def plot_roofline(args, dims, configs, out):
    """Roofline(s) sharing the HBM slant; each chip's slant rises to ITS peak-compute ceiling, meeting
    at that chip's ridge. The workload's arithmetic intensity is FIXED (navy line) -- the DSE only
    moves the ridge: adding multipliers lifts the ceiling and slides the ridge toward the workload,
    and the dot (achieved FLOP/s) climbs. configs = [(label, hw_cfg, result), ...], first = baseline."""
    peak_bw = configs[0][2]["peak_bw"]
    avg_kv = args.input_seq + args.output_seq // 2
    flops = decode_step_flops(dims, avg_kv, args.batch)
    tr = configs[0][2]["mem_model"].compute_decode_traffic(num_output_tokens=args.output_seq // 2).total_traffic
    bytes_tok = tr.read_bytes * matrix_overfetch_factor(configs[0][1]) + tr.write_bytes
    ai = flops / bytes_tok
    top_peak = max(2 * hw.MLEN * hw.BLEN * FREQ_HZ for _, hw, _ in configs)

    xs = [10 ** (i / 10) for i in range(-10, int(10 * math.log10(max(top_peak / peak_bw, ai) * 5)) + 1)]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.loglog(xs, [x * peak_bw for x in xs], "-", color="0.75", lw=1.0)        # shared HBM slant
    for (label, hw, res), color in zip(configs, ("0.55", "crimson")): # Gray - Baseline and Red - Right size with their ceiling TFLOPS/s
        peak = 2 * hw.MLEN * hw.BLEN * FREQ_HZ
        ridge = peak / peak_bw
        ax.loglog(xs, [min(peak, x * peak_bw) for x in xs], "-", color=color, lw=1.7,
                  label=f"{label} {hw.MLEN}x{hw.BLEN}  (ridge {ridge:.0f}, peak {peak/1e12:.1f} TFLOP/s)")
        ax.axvline(ridge, ls=":", color=color, lw=1.0, alpha=0.7)
        ct = decode_token_cycles(res["perf"], dims, avg_kv, args.batch) * DECODE_PIPELINE_FACTOR / FREQ_HZ
        ax.scatter([ai], [flops / max(ct, bytes_tok / peak_bw)], s=70, color=color, zorder=5) # Two dots are the achieved FLOPS/sec
    ax.axvline(ai, ls="--", color="navy", lw=1.3, label=f"workload AI = {ai:.0f} FLOP/byte (Fixed)") # Workload AI
    ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
    ax.set_ylabel("Performance (FLOP / s)")
    ax.set_title(f"{args.model} · W{args.w}/A{args.a}/KV{args.kv} · Batch {args.batch}", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [plot] {out}")


def plot_search(args, sweeps, out):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # Keep the best throughput at each area.
    area_tps = {}
    for dim in ("MLEN", "BLEN"):
        for p in sweeps.get(dim, []):
            if p["fits"]:
                a = round(p["area"] * MM2_PER_MULTIPLIER, 4)
                area_tps[a] = max(area_tps.get(a, 0.0), p["tps"])
    if area_tps:
        xs, ys = zip(*sorted(area_tps.items()))
        axL.plot(xs, ys, "-o", color="seagreen", ms=5)
        knee = max(sorted(area_tps.items()), key=lambda t: t[1])
        axL.scatter([knee[0]], [knee[1]], s=120, marker="*", color="crimson", zorder=6,
                    label=f"right size ({knee[0]:.2f} mm²)")
        axL.axvline(REF_MM2, ls=":", color="0.5", lw=1, label=f"baseline ({REF_MM2:.2f} mm²)")
        axL.set_xscale("log", base=2)
        axL.set_xlabel("Matrix-array area (mm²)")
        axL.set_ylabel("Throughput (tokens/s)")
        axL.grid(True, which="both", alpha=0.25)
        axL.legend(fontsize=8)

    # Batch: throughput (saturates) and latency (climbs); mark the efficient batch.
    bt = sweeps.get("BATCH", [])
    if bt:
        xb = [p["value"] for p in bt]
        axR.plot(xb, [p["tps"] for p in bt], "o-", color="navy", ms=5, label="throughput")
        peak = max(p["tps"] for p in bt)
        eff = min((p for p in bt if p["tps"] >= (1 - RIGHTSIZE_TPS_TOL) * peak), key=lambda p: p["value"])
        axR.scatter([eff["value"]], [eff["tps"]], s=120, marker="*", color="crimson", zorder=6,
                    label=f"efficient batch ({eff['value']})")
        axR.set_xscale("log", base=2)
        axR.set_xlabel("Batch size")
        axR.set_ylabel("Throughput (tokens/s)", color="navy")
        axR.tick_params(axis="y", labelcolor="navy")
        ax2 = axR.twinx()
        ax2.plot(xb, [p["tpot"] * 1e3 for p in bt], "s--", color="darkorange", ms=4, label="latency")
        ax2.set_ylabel("Latency (ms/token)", color="darkorange")
        ax2.tick_params(axis="y", labelcolor="darkorange")
        axR.grid(True, alpha=0.25)
        h1, l1 = axR.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        axR.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    fig.suptitle(f"{args.model} - Decode hardware search", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [plot] {out}")


def plot_precision_pareto(args, rows, out):
    """Accuracy vs decode memory (best = bottom-left). Green fits HBM, red exceeds; ★ = GPTQ."""
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["mb"])
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.plot([r["mb"] for r in rows], [r["ppl"] for r in rows], "-", color="0.7", lw=1, zorder=1)
    for r in rows:
        ax.scatter([r["mb"]], [r["ppl"]], s=150 if r["gptq"] else 55, zorder=3,
                   marker="*" if r["gptq"] else "o", color="seagreen" if r["fits"] else "crimson")
        ax.annotate(r["label"], (r["mb"], r["ppl"]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.scatter([], [], color="seagreen", marker="o", label="fits HBM")
    ax.scatter([], [], color="crimson", marker="o", label="exceeds HBM")
    ax.scatter([], [], color="0.4", marker="*", label="GPTQ")
    ax.set_yscale("log")
    ax.set_xlabel("Memory (MB / token)")
    ax.set_ylabel("Continuation perplexity")
    ax.set_title(f"{args.model} · Batch {args.batch} · Accuracy vs Memory", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [plot] {out}")


def plot_codesign(args, rows, out):
    """Joint precision x hardware. Left: better accuracy (lower PPL) forces 
    higher precision -> the bandwidth bound shrinks MLEN -> lower throughput
    Right: higher precision = bigger weights+KV -> smaller max batch (Capacity wall)"""
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["ppl"])
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Points coloured by the regime each precision lands in (so a memory-bound flip is visible).
    bcolor = {"cmp": "navy", "mem": "darkorange"}
    axL.plot([r["ppl"] for r in rows], [r["tps"] for r in rows], "-", color="0.7", lw=1, zorder=1)
    for r in rows:
        axL.scatter([r["ppl"]], [r["tps"]], s=45, color=bcolor.get(r["bound"], "navy"), zorder=3)
        axL.annotate(r["label"], (r["ppl"], r["tps"]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    axL.scatter([], [], color="navy", label="compute-bound")
    axL.scatter([], [], color="darkorange", label="memory-bound")
    axL.set_xscale("log")
    axL.set_xlabel("Continuation Perplexity")
    axL.set_ylabel("Throughput (tokens/s)")
    axL.set_title("Accuracy vs Throughput", fontsize=10)
    axL.grid(True, which="both", alpha=0.25)
    axL.legend(fontsize=8)

    axR.plot([r["ppl"] for r in rows], [r["max_batch"] for r in rows], "-", color="0.7", lw=1, zorder=1)
    for r in rows:
        axR.scatter([r["ppl"]], [r["max_batch"]], s=45, marker="s",
                    color=bcolor.get(r["bound"], "seagreen"), zorder=3)
        axR.annotate(r["label"], (r["ppl"], r["max_batch"]), textcoords="offset points", xytext=(5, 3), fontsize=7)
    axR.set_xscale("log")
    axR.set_yscale("log")
    axR.set_xlabel("Continuation Perplexity")
    axR.set_ylabel("Max batch (Capacity)")
    axR.set_title("Accuracy vs Capacity", fontsize=10)
    axR.grid(True, which="both", alpha=0.25)

    fig.suptitle(f"{args.model} - Precision x Hardware co-design  (W/A/KV bits)", fontsize=11)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [plot] {out}")


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
    ap.add_argument("--fmt", choices=["mxint", "mxfp"], default="mxint")
    ap.add_argument("--w", default="4")
    ap.add_argument("--a", default="8")
    ap.add_argument("--kv", default="4")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--search", action="store_true", help="right-size the decode hardware for this precision")
    ap.add_argument("--sweep", nargs="?", const=_DEFAULT_CSV, default=None,
                    help=f"precision sweep over the software CSV (default: {Path(_DEFAULT_CSV).name})")
    ap.add_argument("--codesign", nargs="?", const=_DEFAULT_CSV, default=None,
                    help="joint precision x hardware co-design over the software CSV (right-size per precision)")
    ap.add_argument("--plot", action="store_true", help="save figures to --plot-dir")
    ap.add_argument("--plot-dir", default=str(_HERE / "results"))
    args = ap.parse_args()

    model_path = resolve_model_path(args.model, args.model_lib)
    dims = load_model_dims(model_path)
    hw_cfg = load_hardware_config_from_toml(args.config)
    base_mem = load_memory_config_from_toml(args.config)
    plot_dir = Path(args.plot_dir)
    if args.plot:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # One run can do all of: report (always), --search (right-size), --sweep (accuracy<->cost).
    prec = build_precision(args)
    stride = max(1, args.output_seq // 256)            # sample the context loop -- fast and accurate
    result = evaluate(model_path, dims, hw_cfg, args.isa_lib, base_mem, prec,
                      args.batch, args.input_seq, args.output_seq, stride=stride)
    print_report(args, dims, hw_cfg, prec, result, model_path)

    sweeps = best_hw = best_result = None
    if args.search:
        sweeps, best_hw, best_result = run_search(args, model_path, dims, hw_cfg, args.isa_lib, base_mem, prec)

    if args.plot:
        # Roofline overlays the baseline and (if searched) the right-sized chip, so you can see the
        # ridge slide toward the fixed workload AI as the DSE adds multipliers.
        configs = [("baseline", hw_cfg, result)]
        if best_hw is not None:
            configs.append(("right-sized", best_hw, best_result))
        plot_roofline(args, dims, configs, plot_dir / "roofline.png")
        if sweeps is not None:
            plot_search(args, sweeps, plot_dir / "search.png")

    if args.sweep:
        rows = run_precision_sweep(args, model_path, dims, hw_cfg, args.isa_lib, base_mem)
        if args.plot:
            plot_precision_pareto(args, rows, plot_dir / "precision_pareto.png")

    if args.codesign:
        rows = run_codesign(args, model_path, dims, hw_cfg, args.isa_lib, base_mem)
        if args.plot:
            plot_codesign(args, rows, plot_dir / "codesign.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
