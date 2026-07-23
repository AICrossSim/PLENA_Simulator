"""Analytic models for the decode chip in disaggregated serving.

  memory.py   — HBM effective-bandwidth model, calibrated on emulator DMA runs.
  handoff.py  — prefill -> decode KV-cache transfer timing.
  area.py     — chip area, multiplier proxy or DC-calibrated model.
  serve.py    — single import surface over the decode-step evaluator.
"""
