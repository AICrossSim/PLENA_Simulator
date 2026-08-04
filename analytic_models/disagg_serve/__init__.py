"""Analytic models for the decode chip in disaggregated serving.

  memory.py   — HBM model fitted to retained aggregate emulator DMA tables.
  calibration_provenance.py — hashes, grids, history, and missing run receipts.
  hbm_technology.py — explicit HBM rate/capacity operating points.
  handoff.py  — prefill -> decode KV-cache transfer timing.
  area.py     — chip area, multiplier proxy or DC-calibrated model.
  serve.py    — single import surface over the decode-step evaluator.
"""
