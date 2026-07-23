"""Single import surface for the decode-chip analytic stack.

The evaluator still lives in analytic_models/performance/disagg_decode.py, which
the DSE bridge and justfile recipes import directly. This module re-exports it
alongside the package-native models so callers need only one namespace:

  from disagg_serve import serve, memory, handoff, area

  serve.evaluate(...)           # decode-step metrics for one design point
  memory.CalibratedBandwidth    # emulator-measured effective bandwidth
  handoff.handoff_time(...)     # prefill -> decode KV transfer timing
  area.area_mm2(...)            # proxy / DC-calibrated chip area

Re-exporting keeps existing callers working while new code targets the package
path, so moving the evaluator here later is a file move, not an API change.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "performance"))

from disagg_decode import (  # noqa: E402,F401
    area_mm2,
    area_multipliers,
    build_precision,
    compute_density,
    decode_traffic,
    evaluate,
    hbm_overrides,
    kv_footprint_bytes,
    load_hardware_config_from_toml,
    load_memory_config_from_toml,
    load_model_dims,
    max_batch_capacity,
    mlen_bandwidth_cap,
    precision_from_components,
    resolve_model_path,
    run_decode_loop,
    set_area_model,
    stream_bits,
)
