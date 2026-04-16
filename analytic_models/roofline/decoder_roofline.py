"""
PLENA Decoder Layer Roofline Analysis

Parses emulator build output (latency + HBM stats) and computes roofline metrics
for the single-layer decoder pipeline test.

Usage:
    # Capture sim output, then analyze:
    just build-emulator smollm2_135m_decoder 2>&1 | python3 analytic_models/roofline/decoder_roofline.py

    # Or pass latency/bytes directly:
    python3 analytic_models/roofline/decoder_roofline.py --latency-ns 156099 --hbm-bytes 106496

    # Analyze full-scale PLENA (ANALYTIC config) projected from behavioral sim:
    python3 analytic_models/roofline/decoder_roofline.py --latency-ns 156099 --hbm-bytes 106496 --config analytic
"""

import sys
import re
import argparse
from pathlib import Path

# ── Hardware config from plena_settings.toml ────────────────────────────────

SETTINGS_PATH = Path(__file__).parent.parent.parent / "plena_settings.toml"


def load_hw_config(mode: str = "behavior") -> dict:
    """Load hardware parameters from plena_settings.toml using regex."""
    text = SETTINGS_PATH.read_text()
    section = mode.upper()

    def get_val(key: str) -> int:
        # Match [SECTION.CONFIG.KEY] followed by value = N
        pattern = rf"\[{re.escape(section)}\.CONFIG\.{re.escape(key)}\]\s*\nvalue\s*=\s*(\d+)"
        m = re.search(pattern, text)
        if not m:
            raise KeyError(f"Key {section}.CONFIG.{key} not found in plena_settings.toml")
        return int(m.group(1))

    return {
        "mlen": get_val("MLEN"),
        "blen": get_val("BLEN"),
        "vlen": get_val("VLEN"),
        "hbm_width_bits": get_val("HBM_WIDTH"),
        "clock_ghz": 1.0,  # PERIOD = 1ns in main.rs
    }


def peak_hbm_bw(cfg: dict) -> float:
    """Peak HBM bandwidth in GB/s."""
    bytes_per_cycle = cfg["hbm_width_bits"] / 8
    return bytes_per_cycle * cfg["clock_ghz"]  # GB/s


def peak_systolic_gflops(cfg: dict) -> float:
    """
    Peak systolic array throughput in GFLOPS.
    Each MM instruction: mlen x mlen MACs = 2 * mlen^2 FLOPs.
    Latency: mlen cycles (pipeline fills in mlen cycles after overhead).
    Peak = 2 * mlen^2 / mlen * clock = 2 * mlen * clock_ghz GFLOPS.
    """
    return 2.0 * cfg["mlen"] * cfg["clock_ghz"]  # GFLOPS


# ── FLOPs calculation ────────────────────────────────────────────────────────


def decoder_layer_flops(seq_len: int, hidden: int, inter: int, num_heads: int = 1) -> int:
    """
    Compute FLOPs for a single decoder layer (prefill with seq_len tokens).

    Components:
        - Q, K, V projections:  3 × 2 × seq_len × hidden × (hidden/num_heads)
        - O projection:         2 × seq_len × hidden × hidden
        - Flash attn (QKᵀ+AV): 2 × 2 × seq_len² × (hidden/num_heads) × num_heads
        - FFN gate + up:        2 × 2 × seq_len × hidden × inter
        - FFN down:             2 × seq_len × inter × hidden
        - RMSNorm ×2:           ~2 × seq_len × hidden (negligible)
    """
    head_dim = hidden // num_heads

    attn_proj = 4 * 2 * seq_len * hidden * head_dim * num_heads  # Q,K,V,O
    flash_attn = 2 * 2 * seq_len * seq_len * head_dim * num_heads  # QKᵀ + AV
    ffn = (
        2 * seq_len * hidden * inter  # gate
        + 2 * seq_len * hidden * inter  # up
        + 2 * seq_len * inter * hidden
    )  # down
    rms_norm = 2 * 2 * seq_len * hidden  # ×2 norms

    return attn_proj + flash_attn + ffn + rms_norm


# ── Parse sim output from stdin ──────────────────────────────────────────────


def parse_sim_output(text: str):
    lat_match = re.search(r"Simulation completed\. Latency ([0-9.]+)ns", text)
    hbm_match = re.search(r"Bytes read: (\d+)", text)
    if not lat_match or not hbm_match:
        return None
    return {
        "latency_ns": float(lat_match.group(1)),
        "hbm_bytes": int(hbm_match.group(1)),
    }


# ── Roofline analysis ────────────────────────────────────────────────────────


def roofline_analysis(
    latency_ns: float,
    hbm_bytes: int,
    seq_len: int,
    hidden: int,
    inter: int,
    config_mode: str = "behavior",
):
    cfg = load_hw_config(config_mode)
    peak_bw = peak_hbm_bw(cfg)  # GB/s
    peak_compute = peak_systolic_gflops(cfg)  # GFLOPS

    ridge_point = peak_compute / peak_bw  # FLOPs/byte

    total_flops = decoder_layer_flops(seq_len, hidden, inter)
    latency_s = latency_ns * 1e-9

    achieved_bw_gbs = hbm_bytes / latency_s / 1e9
    achieved_gflops = total_flops / latency_s / 1e9
    arithmetic_intensity = total_flops / hbm_bytes  # FLOPs/byte

    bw_bound = arithmetic_intensity < ridge_point
    bottleneck = "Memory-bandwidth bound" if bw_bound else "Compute bound"

    # Roofline ceiling
    roofline_gflops = min(peak_compute, arithmetic_intensity * peak_bw)
    efficiency = achieved_gflops / roofline_gflops * 100

    print(f"\n{'=' * 60}")
    print("  PLENA Decoder Layer Roofline Analysis")
    print(f"  Config: {config_mode.upper()}  (MLEN={cfg['mlen']}, BLEN={cfg['blen']})")
    print(f"  Layer params: seq_len={seq_len}, hidden={hidden}, inter={inter}")
    print(f"{'=' * 60}")
    print("\n  Hardware peaks:")
    print(f"    Peak HBM bandwidth : {peak_bw:.1f} GB/s")
    print(f"    Peak compute       : {peak_compute:.1f} GFLOPS")
    print(f"    Ridge point        : {ridge_point:.1f} FLOPs/byte")
    print("\n  Workload:")
    print(f"    Total FLOPs        : {total_flops / 1e6:.3f} MFLOPs")
    print(f"    HBM bytes read     : {hbm_bytes / 1024:.1f} KB")
    print(f"    Arith. intensity   : {arithmetic_intensity:.1f} FLOPs/byte")
    print("\n  Simulation results:")
    print(f"    Latency            : {latency_ns:.0f} ns  ({latency_ns / 1e3:.2f} µs)")
    print(
        f"    Achieved BW        : {achieved_bw_gbs * 1e3:.1f} MB/s  ({achieved_bw_gbs / peak_bw * 100:.2f}% of peak)"
    )
    print(
        f"    Achieved compute   : {achieved_gflops * 1e3:.2f} MFLOPS  ({achieved_gflops / peak_compute * 100:.4f}% of peak)"
    )
    print("\n  Roofline:")
    print(f"    Bottleneck         : {bottleneck}")
    print(f"    Roofline ceiling   : {roofline_gflops:.1f} GFLOPS")
    print(f"    Efficiency         : {efficiency:.2f}% of roofline ceiling")
    print(f"{'=' * 60}\n")

    return {
        "latency_ns": latency_ns,
        "hbm_bytes": hbm_bytes,
        "total_flops": total_flops,
        "arithmetic_intensity": arithmetic_intensity,
        "achieved_gflops": achieved_gflops,
        "achieved_bw_gbs": achieved_bw_gbs,
        "peak_bw_gbs": peak_bw,
        "peak_gflops": peak_compute,
        "ridge_point": ridge_point,
        "bottleneck": bottleneck,
        "efficiency_pct": efficiency,
    }


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="PLENA decoder roofline analysis")
    parser.add_argument("--latency-ns", type=float, help="Simulated latency in ns")
    parser.add_argument("--hbm-bytes", type=int, help="HBM bytes read")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--inter", type=int, default=128)
    parser.add_argument(
        "--config",
        choices=["behavior", "analytic"],
        default="behavior",
        help="Hardware config section from plena_settings.toml",
    )
    args = parser.parse_args()

    if args.latency_ns and args.hbm_bytes:
        roofline_analysis(args.latency_ns, args.hbm_bytes, args.seq_len, args.hidden, args.inter, args.config)
    else:
        # Read from stdin (piped from emulator build output)
        text = sys.stdin.read()
        parsed = parse_sim_output(text)
        if parsed is None:
            print("ERROR: Could not parse 'Simulation completed. Latency Xns' and 'Bytes read: X' from input.")
            print(
                "Usage: just build-emulator smollm2_135m_decoder 2>&1 | python3 analytic_models/roofline/decoder_roofline.py"
            )
            sys.exit(1)
        roofline_analysis(parsed["latency_ns"], parsed["hbm_bytes"], args.seq_len, args.hidden, args.inter, args.config)


if __name__ == "__main__":
    main()
