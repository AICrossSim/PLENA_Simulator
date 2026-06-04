#!/usr/bin/env python3
"""Build a kind-tagged weight manifest from a real PLENA compile.

The capacity-aware memory model in the emulator (``lib/memory/src/streaming.rs``)
selects a DDR-residency regime (resident / kv-swap / weight-stream) from a
``WeightManifest`` that tags every persistent HBM region as ``weight``, ``kv`` or
``activation`` and carries the board's DDR capacity.  This script turns the
artifacts a real compile drops into ``<build-dir>`` into exactly that manifest, so
the regimes fire on a real SmolVLM2 / SmolLM2 / clm workload instead of a
synthetic stand-in.

Inputs (written by ``run_model.py`` right after the ISA, for any --case):
  * ``hbm_addrs.json``      name -> HBM base byte offset                 (REQUIRED)
  * ``hbm_sizes.json``      name -> region bytes (authoritative; incl. KV) (preferred)
  * ``tensor_layouts.json`` name -> {storage_shape, ...} for INPUT tensors (fallback)

Sizing precedence per region:
  1. ``hbm_sizes[name]``                         - authoritative; covers KV stores.
  2. ``int(prod(storage_shape) * 1.125)``        - exact for inputs/weights; misses KV.
     (MXFP8 E4M3, block=8, scale_exponent=8 -> real_data_ratio = 9/8 = 1.125,
      matching the compiler: program_tensors.py ``hbm_size = int(size*1.125)``.)
  3. sorted-offset address delta                 - last resort, contiguous prefix only.

Classification by region name (see plena_frontend.py emission conventions):
  * KV         : name contains a ``_stored_`` token preceded by K or V
                 (K_stored_*, V_stored_*, V_K_stored_*, V_V_stored_*).
  * weight     : decoder matmul weights W_*; vision attention/FFN weights V_W_*;
                 vision biases V_B_* (uppercase B: V_B_q/k/v/o/fc1/fc2); vision
                 layernorm params V_LN1_*/V_LN2_*/V_POST_LN_* (UPPERCASE); and the
                 patch-embed/connector weights V_PATCH_W, V_PATCH_BIAS_POS,
                 V_CONNECTOR_W, V_CONNECTOR_B.
  * activation : everything else (X, POS, COS, SIN, R_rope, causal_mask, V_PIXELS,
                 the post-LN output vision_post_ln, and per-layer intermediates like
                 K_{l}_h{kv}, O_proj_{l}, V_FC1_{l}, V_PATCH_OUT, V_CONNECTOR_OUT).
  NB classify KV by the ``_stored_`` token, NOT a leading K/V: bare K_{l}_h{kv} is a
  pre-store projection activation, and the V_ prefix is the *vision* namespace.

Usage:
  python testbench/make_weight_manifest.py \
      --build-dir testbench/build/clm60m_native_64x64x16_b1_decoder \
      --board testbench/board_configs/custom_a7.yaml
  # optional: --total-layers 30   (extrapolate full-model footprint in the report)
  #           --ddr-capacity 256M  (override board capacity, e.g. to force a regime)
  #           -o path/to/weight_manifest.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import prod
from pathlib import Path

# MXFP8 (E4M3, block=8, scale_exponent=8): 1 data byte/elem + 1 scale byte / 8
# elems = 9/8 bytes per element. Hardcoded in the compiler as real_data_ratio.
REAL_DATA_RATIO = 1.125

_KV_RE = re.compile(r"(^|_)[KV]_stored_")
# Weight (persistent parameter) name prefixes. Decoder matmul weights are W_*;
# vision attention/FFN weights V_W_*; vision biases V_B_* (uppercase B, incl.
# V_B_q/k/v/o/fc1/fc2); vision layernorm params are UPPERCASE V_LN1_*/V_LN2_*.
_WEIGHT_PREFIXES = ("W_", "V_W_", "V_B_", "V_LN1_", "V_LN2_", "V_POST_LN_")
# Vision patch-embed + connector weights/biases (exact names: their _OUT /
# _SHUFFLED siblings are VRAM alloc intermediates, never HBM regions). NB the
# bare "vision_post_ln" is the post-LN *output* activation — the params are
# V_POST_LN_weight/bias (covered by the V_POST_LN_ prefix above).
_WEIGHT_EXACT = frozenset({"V_PATCH_W", "V_PATCH_BIAS_POS", "V_CONNECTOR_W", "V_CONNECTOR_B"})
# Per-head suffix and digit-bearing tokens (vision fc1/fc2/ln1/ln2) that are NOT
# the layer index; stripped/neutralised before extracting the layer id.
_HEAD_SUFFIX_RE = re.compile(r"_h\d+$")
_LAYER_TOKEN_RE = re.compile(r"(?i)(fc|ln)[12]")
_FIRST_INT_RE = re.compile(r"\d+")

# RegionKind strings the Rust deserializer accepts (#[serde(rename_all="lowercase")]).
KIND_WEIGHT = "weight"
KIND_KV = "kv"
KIND_ACTIVATION = "activation"


def classify(name: str) -> str:
    """Map a region name to a RegionKind string."""
    if _KV_RE.search(name):  # classify KV by the _stored_ token, not a leading K/V
        return KIND_KV
    if name in _WEIGHT_EXACT or name.startswith(_WEIGHT_PREFIXES):
        return KIND_WEIGHT
    return KIND_ACTIVATION


def layer_id(name: str) -> int:
    """Layer index from the name; 0 for global/shared tensors.

    Strips the per-head suffix (_h<n>) and neutralises digit-bearing tokens that
    are not the layer (vision fc1/fc2/ln1/ln2 -> fc/ln) so the first remaining
    integer is the layer index (e.g. V_W_fc1_4 -> 4, vision_l4_ln1 -> 4).
    """
    base = _HEAD_SUFFIX_RE.sub("", name)
    base = _LAYER_TOKEN_RE.sub(lambda m: m.group(0)[:-1], base)
    m = _FIRST_INT_RE.search(base)
    return int(m.group()) if m else 0


def parse_size(text: str) -> int:
    """Parse '512M' / '512MiB' / '256Mi' / '1G' / plain bytes -> integer bytes.

    Binary units (K/M/G = 2^10/2^20/2^30), matching the Rust --ddr3-capacity
    parser and the board YAML capacity_bytes values. Rejects malformed or
    non-positive sizes with a clear error rather than a deep ValueError.
    """
    t = str(text).strip().upper()
    if t.endswith("B"):
        t = t[:-1]
    if t.endswith("I"):  # MiB/Mi -> M
        t = t[:-1]
    mult = 1
    if t.endswith("K"):
        mult, t = 1 << 10, t[:-1]
    elif t.endswith("M"):
        mult, t = 1 << 20, t[:-1]
    elif t.endswith("G"):
        mult, t = 1 << 30, t[:-1]
    try:
        val = int(float(t) * mult)
    except ValueError:
        raise SystemExit(f"--ddr-capacity: cannot parse size {text!r}")
    if val <= 0:
        raise SystemExit(f"--ddr-capacity must be a positive size, got {text!r}")
    return val


def load_board_capacity(board_path: Path) -> tuple[int, str]:
    """Return (capacity_bytes, board_name) from a board YAML's memory section."""
    import yaml

    cfg = yaml.safe_load(board_path.read_text())
    mem = cfg.get("memory", {})
    cap = mem.get("capacity_bytes") or mem.get("physical_size_bytes")
    if cap is None:
        raise SystemExit(f"board {board_path} has no memory.capacity_bytes / physical_size_bytes")
    return int(cap), cfg.get("name", board_path.stem)


def tl_size(entry: dict) -> int | None:
    """Region bytes from a tensor_layouts entry: int(prod(storage_shape)*1.125)."""
    shape = entry.get("storage_shape")
    if not shape:
        return None
    return int(prod(int(d) for d in shape) * REAL_DATA_RATIO)


def build_regions(build_dir: Path) -> tuple[list[dict], dict]:
    """Build sorted, kind-tagged regions from a build dir's HBM artifacts.

    Returns (regions, diag) where diag records which size source was used.
    """
    addrs_path = build_dir / "hbm_addrs.json"
    if not addrs_path.exists():
        raise SystemExit(
            f"missing {addrs_path}\n"
            "Recompile with run_model.py (any case) to emit hbm_addrs.json; e.g.\n"
            "  python testbench/run_model.py clm60m --config native_64x64x16_b1 "
            "--case decoder --layers 1 --compile-only"
        )
    addrs: dict[str, int] = json.loads(addrs_path.read_text())

    sizes_path = build_dir / "hbm_sizes.json"
    sizes: dict[str, int | None] = json.loads(sizes_path.read_text()) if sizes_path.exists() else {}
    layouts_path = build_dir / "tensor_layouts.json"
    layouts: dict[str, dict] = json.loads(layouts_path.read_text()) if layouts_path.exists() else {}

    # Sorted by offset so the delta fallback (and human-readable output) is
    # stable. Skip entries without a usable integer offset (e.g. an unmapped
    # input with hbm_addr None) rather than crashing the sort.
    ordered = sorted(
        ((n, o) for n, o in addrs.items() if isinstance(o, int)),
        key=lambda kv: kv[1],
    )
    dropped_offset = [n for n, o in addrs.items() if not isinstance(o, int)]
    src_count = {"hbm_sizes": 0, "tensor_layouts": 0, "delta": 0}
    unsized: list[str] = []
    regions: list[dict] = []

    for idx, (name, offset) in enumerate(ordered):
        # Resolve size by precedence; accept a source only if it yields a
        # positive int. A None/0 from hbm_sizes must NOT shadow a usable
        # tensor_layouts/delta size, so fall through on each miss.
        size = None
        source = None
        cand = sizes.get(name)
        if isinstance(cand, int) and cand > 0:
            size, source = cand, "hbm_sizes"
        if size is None:
            entry = layouts.get(name)
            cand = tl_size(entry) if entry is not None else None
            if isinstance(cand, int) and cand > 0:
                size, source = cand, "tensor_layouts"
        if size is None and idx + 1 < len(ordered):
            # contiguous-prefix address delta: an UPPER bound (stride incl. any
            # padding/gap), reliable only for the never-freed prefix.
            cand = ordered[idx + 1][1] - offset
            if isinstance(cand, int) and cand > 0:
                size, source = cand, "delta"
        if size is None:
            unsized.append(name)
            continue
        src_count[source] += 1
        regions.append(
            {
                "layer_id": layer_id(name),
                "offset": int(offset),
                "size": int(size),
                "kind": classify(name),
                "_name": name,  # diagnostic only; stripped before writing
            }
        )

    # Guard overlaps: Rust region_kind()/region_bounds() are first-match-wins on
    # offset-sorted regions, so an oversized region would shadow (misclassify)
    # the next. Clamp to abut the next region and record the correction.
    overlaps: list[tuple] = []
    for i in range(len(regions) - 1):
        end = regions[i]["offset"] + regions[i]["size"]
        nxt = regions[i + 1]["offset"]
        if end > nxt:
            overlaps.append((regions[i]["_name"], regions[i]["size"], nxt - regions[i]["offset"]))
            regions[i]["size"] = nxt - regions[i]["offset"]

    return regions, {
        "src_count": src_count,
        "unsized": unsized,
        "dropped_offset": dropped_offset,
        "overlaps": overlaps,
        "n_addrs": len(addrs),
    }


def footprint(regions: list[dict]) -> dict[str, int]:
    f = {KIND_WEIGHT: 0, KIND_KV: 0, KIND_ACTIVATION: 0}
    for r in regions:
        f[r["kind"]] += r["size"]
    return f


def choose_regime(weight: int, total: int, capacity: int) -> str:
    """Mirror of streaming.rs::choose_regime."""
    if total <= capacity:
        return "resident"
    if weight <= capacity:
        return "kv-swap"
    return "weight-stream"


def fmt_mb(n: int) -> str:
    return f"{n / (1 << 20):.2f} MiB" if n else "0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-dir", required=True, type=Path, help="compile output dir with hbm_addrs.json")
    ap.add_argument("--board", required=True, type=Path, help="board YAML (supplies memory.capacity_bytes)")
    ap.add_argument("--ddr-capacity", type=str, default=None, help="override board capacity, e.g. 256M")
    ap.add_argument(
        "--total-layers", type=int, default=None, help="extrapolate per-layer footprint to N layers in the report"
    )
    ap.add_argument(
        "-o", "--output", type=Path, default=None, help="manifest path (default: <build-dir>/weight_manifest.json)"
    )
    args = ap.parse_args()

    regions, diag = build_regions(args.build_dir)
    if not regions:
        raise SystemExit("no sizeable HBM regions found")

    capacity, board_name = load_board_capacity(args.board)
    if args.ddr_capacity:
        capacity = parse_size(args.ddr_capacity)

    fp = footprint(regions)
    weight, kv, act = fp[KIND_WEIGHT], fp[KIND_KV], fp[KIND_ACTIVATION]
    total = weight + kv + act

    # activation_ceiling: legacy-only field (consumed by is_weight_addr in the
    # host-stream/layer-swap models, NOT by the capacity model). Set it to the
    # lowest weight offset so the legacy view still separates weights from the
    # activations/embeddings laid out before them.
    weight_offsets = [r["offset"] for r in regions if r["kind"] == KIND_WEIGHT]
    activation_ceiling = min(weight_offsets) if weight_offsets else 0

    out_regions = [{k: r[k] for k in ("layer_id", "offset", "size", "kind")} for r in regions]
    manifest = {
        "activation_ceiling": activation_ceiling,
        "regions": out_regions,
        "ddr_capacity_bytes": capacity,
    }

    out_path = args.output or (args.build_dir / "weight_manifest.json")
    out_path.write_text(json.dumps(manifest, indent=2))

    # ---- report -----------------------------------------------------------
    by_kind = {KIND_WEIGHT: 0, KIND_KV: 0, KIND_ACTIVATION: 0}
    for r in regions:
        by_kind[r["kind"]] += 1
    print(f"Wrote {out_path}")
    print(f"  board:    {board_name}  (DDR capacity {fmt_mb(capacity)})")
    print(
        f"  regions:  {len(regions)}  ({by_kind[KIND_WEIGHT]} weight, {by_kind[KIND_KV]} kv, {by_kind[KIND_ACTIVATION]} activation)"
    )
    print(
        f"  sizes:    {diag['src_count']['hbm_sizes']} from hbm_sizes, "
        f"{diag['src_count']['tensor_layouts']} from tensor_layouts, "
        f"{diag['src_count']['delta']} from address-delta"
    )
    if diag["unsized"]:
        print(
            f"  WARNING: {len(diag['unsized'])} region(s) had no size and were DROPPED (coverage hole): {diag['unsized']}"
        )
    if diag["src_count"]["delta"]:
        print(
            f"  WARNING: {diag['src_count']['delta']} region(s) sized by address-delta (upper bound; footprint may be inflated). "
            "Recompile with an hbm_sizes-emitting compiler for exact sizes."
        )
    if diag.get("overlaps"):
        print(f"  WARNING: clamped {len(diag['overlaps'])} overlapping region(s) to abut the next: {diag['overlaps']}")
    if diag.get("dropped_offset"):
        print(
            f"  WARNING: {len(diag['dropped_offset'])} region(s) had no integer offset and were skipped: {diag['dropped_offset']}"
        )
    print(
        f"  footprint (this build): weight {fmt_mb(weight)} | kv {fmt_mb(kv)} | activation {fmt_mb(act)} | total {fmt_mb(total)}"
    )
    print(f"  regime @ {fmt_mb(capacity)} (this build): {choose_regime(weight, total, capacity)}")

    if args.total_layers:
        n = args.total_layers
        # Uniform-layer extrapolation for the REPORT only (the manifest keeps the
        # real regions). Per-layer weight/kv = this build's weight/kv divided by
        # the number of distinct layers actually present, so a multi-layer
        # (--layers N) build extrapolates correctly. Shared globals (embeddings,
        # rope tables) and per-layer activation intermediates are NOT modelled, so
        # the activation term is a floor, not the full-model activation footprint.
        layers_in_build = len({r["layer_id"] for r in regions if r["kind"] in (KIND_WEIGHT, KIND_KV)}) or 1
        per_layer_w = weight / layers_in_build
        per_layer_kv = kv / layers_in_build
        est_w = int(per_layer_w * n)
        est_kv = int(per_layer_kv * n)
        est_total = est_w + est_kv + act
        print(
            f"  --- estimated full model ({n} layers; build holds {layers_in_build}; embeddings/lm_head + per-layer activations excluded) ---"
        )
        print(
            f"  footprint (est): weight {fmt_mb(est_w)} | kv {fmt_mb(est_kv)} | activation>={fmt_mb(act)} | total>={fmt_mb(est_total)}"
        )
        regime = choose_regime(est_w, est_total, capacity)
        print(f"  regime @ {fmt_mb(capacity)} (est full model): {regime}")
        print(
            f"  (manifest contains the REAL build regions; to reproduce the full-model regime\n"
            f"   on this build, run the emulator with --memory-model {regime})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
