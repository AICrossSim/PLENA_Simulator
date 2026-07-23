"""MoE timing-replay harnesses for the PLENA transactional emulator.

Capture real MoE routing, replay it through the emulator, and measure cycles /
HBM bytes. Layers:

- ``replay``   — generic route-trace schema, validator, replay runner, and timing
                 validation gates.
- ``qwen``     — Qwen3 true-route generation from tokenized inputs + local weights,
                 conversion to the replay schema, and batch replay.
- ``campaign`` — representative-subset selection and checkpointed parallel replay.

SCOPE: these measure the timing of MoE layers *in isolation*. End-to-end model
timing (composing MoE with attention and other layers) is out of scope here and
belongs in ``analytic_models/`` as a composition layer over the per-component
numbers. Numerical correctness of the MoE ops is validated by the routed-MoE op
tests (``routed_moe/gpt_oss_moe_*_test.py``, ``real_layer0``), not by these
timing harnesses. Overlap-adjusted cycle counts (opt-in) are an upper-bound
estimate, not a measured value — see ``timing_overlay.rs``.
"""
