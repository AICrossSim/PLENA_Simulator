"""
Sweep over (MLEN, BLEN, VLEN, VECTOR_SRAM_SIZE, MATRIX_SRAM_SIZE, HBM_V_Prefetch_Amount).
For each config, call compute_decode_time() — models the disaggregated decode chip where
KV caches arrive from a prefill chip and decode is fully HBM-bandwidth-bound at batch=1.
"""
import csv
import itertools
import math
import sys
import time
from pathlib import Path
import toml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent  # analytic_models/performance/
ROOT       = SCRIPT_DIR.parent.parent         # PLENA_Simulator/
sys.path.insert(0, str(SCRIPT_DIR))

from perf_model  import HardwareConfig
from llama_model import LLaMAModel

SETTINGS_PATH = ROOT / "plena_settings.toml"
ISA_PATH      = SCRIPT_DIR / "customISA_lib.json"
MODEL_LIB     = ROOT / "compiler" / "doc" / "Model_Lib"
OUTPUT_DIR    = ROOT / "disagg_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# For cycles to seconds
CLOCK_GHZ  = 1.0 # Similar to that of the paper
MODEL_NAME = "llama-3.1-8b"

# Agentic decode-heavy workload
WORKLOADS = {
    "decode_heavy": {
        "input_seq":  5600,
        "output_seq": 85000,
        "desc":       "Agentic decode-heavy (in=5600, out=85000)",
    },
}

SEARCH_SPACE = {
    # Systolic array dimension - drives weight-matrix parallelism and KV tile size
    "MLEN": [1024, 2048],
    # Block size - controls M_BTMV latency (1+BLEN cycles) in flash-attention decode
    "BLEN": [8, 16, 32, 64],
    # Vector unit width - scales vector-op throughput
    "VLEN": [1024, 2048],
    # Activation SRAM - batch=1 decode activations are tiny
    "VECTOR_SRAM_SIZE": [8192, 16384, 32768, 65536],
    # Weight SRAM depth (rows of MLEN elements) - larger = fewer HBM weight fetches per step
    "MATRIX_SRAM_SIZE": [2048, 4096, 8192],
    # Vector HBM prefetch depth - affects KV/activation prefetch pipeline fill
    "HBM_V_Prefetch_Amount": [4, 8, 16],
}

# LLaMA-3.1-8B: hidden_size=4096, 32 heads and head_dim = 128
FIXED_HLEN = 128

# Area budget not exceeding the baseline PLENA chip
# area_proxy = MLEN^2 + VLEN×10 + VECTOR_SRAM_SIZE
# 2048^2 + 2048×10 + 65536 = 4,280,320
_BASELINE_HW = {"MLEN": 2048, "VLEN": 2048, "VECTOR_SRAM_SIZE": 65536}

def area_proxy(mlen: int, vlen: int, vector_sram_size: int) -> int:
    return mlen ** 2 + vlen * 10 + vector_sram_size

AREA_BUDGET = area_proxy(
    _BASELINE_HW["MLEN"], _BASELINE_HW["VLEN"], _BASELINE_HW["VECTOR_SRAM_SIZE"]
)

# Baseline config - TOML hardware dims used as the reference point for speedup
BASELINE_PARAMS = {
    "MLEN":                  2048,
    "BLEN":                  128,
    "VLEN":                  2048,
    "MATRIX_SRAM_SIZE":      256,
    "HBM_V_Prefetch_Amount": 16,
}

def _is_baseline(r: dict) -> bool:
    return all(r[k] == v for k, v in BASELINE_PARAMS.items())


def load_raw_config_dict(toml_path: Path) -> dict:
    """Load ANALYTIC CONFIG + LATENCY sections from TOML into a flat dict."""
    with open(toml_path) as f:
        data = toml.load(f)

    cfg = {}
    analytic = data.get("ANALYTIC", {})

    for name, val in analytic.get("CONFIG", {}).items():
        if isinstance(val, dict) and "value" in val:
            cfg[name] = val["value"]

    for name, val in analytic.get("LATENCY", {}).items():
        if isinstance(val, dict):
            if "dc_lib_en" in val:
                cfg[name] = val["dc_lib_en"]
            elif "value" in val:
                cfg[name] = val["value"]

    return cfg


def make_hardware_config(
    base_dict:        dict,
    mlen:             int,
    blen:             int,
    vlen:             int,
    vector_sram_size: int,
    matrix_sram_size: int,
    hbm_v_prefetch:   int,
    hbm_m_prefetch:   int,
) -> HardwareConfig | None:
    """Return a HardwareConfig with overridden dimensions, or None if invalid."""
    cfg = base_dict.copy()
    cfg["MLEN"]                  = mlen
    cfg["BLEN"]                  = blen
    cfg["VLEN"]                  = vlen
    cfg["HLEN"]                  = FIXED_HLEN
    cfg["VECTOR_SRAM_SIZE"]      = vector_sram_size
    cfg["MATRIX_SRAM_SIZE"]      = matrix_sram_size
    cfg["HBM_M_Prefetch_Amount"] = hbm_m_prefetch
    cfg["HBM_V_Prefetch_Amount"] = hbm_v_prefetch

    try:
        return HardwareConfig(**cfg)
    except Exception:
        return None


def run_search() -> list[dict]:
    model_path = MODEL_LIB / f"{MODEL_NAME}.json"
    base_dict  = load_raw_config_dict(SETTINGS_PATH)

    # HBM bandwidth transfer time modeled in _weight_stream_cycles / flash_attention
    hbm_m_fixed = base_dict.get("HBM_M_Prefetch_Amount", 2048)
    hbm_bw_GBs  = base_dict.get("HBM_WIDTH", 512) * CLOCK_GHZ  # bytes/cycle × GHz = GB/s

    total_combos   = math.prod(len(v) for v in SEARCH_SPACE.values())
    progress_every = max(10, total_combos // 30)

    print("=" * 65)
    print("PLENA Disaggregated Decode Search")
    print("=" * 65)
    print(f"  Model:           {MODEL_NAME}")
    print(f"  Clock:           {CLOCK_GHZ} GHz")
    print(f"  HBM bandwidth:   {hbm_bw_GBs:.0f} GB/s")
    print(f"  Area budget:     {AREA_BUDGET:,}  "
          f"(baseline: MLEN=2048, VLEN=2048, VSRAM=65536)")
    print(f"  Search combos:   {total_combos:,}")
    print()
    for wl_name, wl in WORKLOADS.items():
        print(f"  Workload [{wl_name}]:")
        print(f"    {wl['desc']}")
    print()

    results         = []
    skipped_budget  = 0
    skipped_invalid = 0
    valid_count     = 0

    keys   = list(SEARCH_SPACE.keys())
    combos = list(itertools.product(*SEARCH_SPACE.values()))

    for combo in combos:
        params = dict(zip(keys, combo))
        mlen  = params["MLEN"]
        blen  = params["BLEN"]
        vlen  = params["VLEN"]
        vsram = params["VECTOR_SRAM_SIZE"]
        msram = params["MATRIX_SRAM_SIZE"]
        hbm_v = params["HBM_V_Prefetch_Amount"]

        area = area_proxy(mlen, vlen, vsram)
        if area > AREA_BUDGET:
            skipped_budget += 1
            continue

        # flash_attention computes mlen // hlen — skip degenerate configs
        if mlen < FIXED_HLEN:
            skipped_invalid += 1
            continue

        hw_config = make_hardware_config(
            base_dict, mlen, blen, vlen, vsram, msram, hbm_v, hbm_m_fixed
        )
        if hw_config is None:
            skipped_invalid += 1
            continue

        valid_count += 1
        row = {
            "MLEN":                  mlen,
            "BLEN":                  blen,
            "VLEN":                  vlen,
            "VECTOR_SRAM_SIZE":      vsram,
            "MATRIX_SRAM_SIZE":      msram,
            "HBM_V_Prefetch_Amount": hbm_v,
            "area_proxy":            area,
        }

        for wl_name, wl in WORKLOADS.items():
            model = LLaMAModel(
                model_config_path = str(model_path),
                hardware_config   = hw_config,
                custom_isa_path   = str(ISA_PATH),
                batch_size        = 1,
                input_seq_len     = wl["input_seq"],
                output_seq_len    = wl["output_seq"],
            )
            # compute_decode_time returns total wall time in seconds
            decode_s = model.compute_decode_time(wl["output_seq"], verbose=False)
            tps      = wl["output_seq"] / decode_s if decode_s > 0 else 0.0
            row[f"decode_s_{wl_name}"] = round(decode_s, 3)
            row[f"tps_{wl_name}"]      = round(tps,      4)

        results.append(row)

        if valid_count % progress_every == 0:
            print(
                f"  {valid_count:4d} valid configs "
                f"| MLEN={mlen:4d}  BLEN={blen:3d}  VLEN={vlen:4d}  MSRAM={msram:5d}",
                flush=True,
            )

    print()
    print("Search complete:")
    print(f"  Total combinations:      {total_combos:,}")
    print(f"  Valid + within budget:   {valid_count:,}")
    print(f"  Skipped (over budget):   {skipped_budget:,}")
    print(f"  Skipped (invalid combo): {skipped_invalid:,}")

    return results


def write_csv(results: list[dict], path: Path) -> None:
    if not results:
        print("No results — nothing to write.")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV written → {path}")
    print(f"  ({len(results)} rows, {len(results[0])} columns)")


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s:.1f}s"
    if m > 0:
        return f"{m}m {s:.1f}s"
    return f"{s:.2f}s"


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 65)
    print("RESULTS SUMMARY")
    print("=" * 65)

    for wl_name, wl in WORKLOADS.items():
        tps_key = f"tps_{wl_name}"
        d_key   = f"decode_s_{wl_name}"

        best     = max(results, key=lambda r: r[tps_key])
        baseline = next((r for r in results if _is_baseline(r)), None)

        print(f"\nWorkload: {wl_name}  —  {wl['desc']}")
        print()

        def _print_row(r: dict, label: str) -> None:
            print(f"  [{label:12s}]")
            print(f"    MLEN={r['MLEN']:4d}  BLEN={r['BLEN']:3d}  "
                  f"VLEN={r['VLEN']:4d}  MSRAM={r['MATRIX_SRAM_SIZE']:5d}  "
                  f"VSRAM={r['VECTOR_SRAM_SIZE']:6d}  HBM_V={r['HBM_V_Prefetch_Amount']:2d}")
            print(f"    area={r['area_proxy']:>10,}  "
                  f"decode={_fmt_duration(r[d_key])}  "
                  f"TPS={r[tps_key]:.2f} tok/s")

        _print_row(best, "BEST DECODE")

        if baseline:
            print()
            _print_row(baseline, "BASELINE")
            speedup_x   = best[tps_key] / baseline[tps_key]
            speedup_pct = (speedup_x - 1.0) * 100.0
            print(f"\n  Decode speedup over baseline: {speedup_x:.2f}×  ({speedup_pct:+.1f}%)")
        else:
            print("\n  [BASELINE not found — verify BASELINE_PARAMS match SEARCH_SPACE]")


def compute_pareto_front_tps(
    results: list[dict], area_key: str, tps_key: str
) -> list[dict]:
    """Pareto front in (area lower=better, TPS higher=better) space."""
    pareto = []
    for i, r in enumerate(results):
        dominated = any(
            results[j][area_key] <= r[area_key] and
            results[j][tps_key]  >= r[tps_key]  and
            (results[j][area_key] < r[area_key] or results[j][tps_key] > r[tps_key])
            for j in range(len(results)) if j != i
        )
        if not dominated:
            pareto.append(r)
    return pareto


def plot_tps_vs_area(results: list[dict], workload_name: str, output_path: Path) -> None:
    tps_key = f"tps_{workload_name}"

    # X = area, Y = throughput
    all_x = [r["area_proxy"] for r in results]
    all_y = [r[tps_key]      for r in results]

    pareto        = compute_pareto_front_tps(results, "area_proxy", tps_key)
    pareto_sorted = sorted(pareto, key=lambda r: r["area_proxy"])
    px = [r["area_proxy"] for r in pareto_sorted]
    py = [r[tps_key]      for r in pareto_sorted]

    baseline = next((r for r in results if _is_baseline(r)), None)
    best     = max(results, key=lambda r: r[tps_key])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Blue — Pareto front
    ax.scatter(px, py, color="royalblue", s=30, zorder=4,
               label="Pareto front")
    if len(px) > 1:
        ax.plot(px, py, color="royalblue", linewidth=1.0,
                linestyle="--", alpha=0.5)

    # Red star — baseline
    if baseline:
        ax.scatter(
            baseline["area_proxy"], baseline[tps_key],
            color="crimson", s=80, marker="*", zorder=6,
            label="Baseline",
        )

    # Orange triangle — best TPS config
    ax.scatter(
        best["area_proxy"], best[tps_key],
        color="darkorange", s=80, marker="^", zorder=7,
        label="Best TPS",
    )

    ax.set_xlabel("Area proxy", fontsize=11)
    ax.set_ylabel("Tokens / s", fontsize=11)

    wl = WORKLOADS[workload_name]
    ax.set_title(f"Decode: area vs TPS ({workload_name})", fontsize=12)

    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved  -> {output_path}")


if __name__ == "__main__":
    t_start = time.time()

    results = run_search()

    if not results:
        print("\nNo valid configs found — check SEARCH_SPACE and AREA_BUDGET.")
        sys.exit(1)

    csv_path = OUTPUT_DIR / "disagg_search_results.csv"
    write_csv(results, csv_path)
    print_summary(results)

    print()
    for wl_name in WORKLOADS:
        plot_tps_vs_area(
            results,
            wl_name,
            OUTPUT_DIR / f"tps_area_{wl_name}.png",
        )

    elapsed = time.time() - t_start
    print()
    print(f"Total wall-clock time: {_fmt_duration(elapsed)}")
    print(f"Results directory:     {OUTPUT_DIR}")
