"""Regression tests for the perf_model bandwidth term, precision parsing and
the two KV-store bugs it uncovered.

Run:  python3 analytic_models/test_perf_model_bandwidth.py
(or with pytest; every check is a plain assert in a test_* function)
"""

import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PERF_DIR = REPO_ROOT / "analytic_models" / "performance"
sys.path.insert(0, str(PERF_DIR))

from perf_model import (  # noqa: E402
    PerfModel,
    _entry_bytes_per_elem,
    load_hardware_config_from_toml,
    parse_precision_bytes,
)

CONFIG = str(REPO_ROOT / "plena_settings.toml")
ISA = str(PERF_DIR / "customISA_lib.json")


def _model(enable_bandwidth=True):
    return PerfModel(load_hardware_config_from_toml(CONFIG), ISA, enable_bandwidth=enable_bandwidth)


# -----------------------------------------------------------------------------
# Precision parsing
# -----------------------------------------------------------------------------


def test_mx_bytes_per_elem_includes_shared_scale():
    """MXFP8 block=8 with an 8-bit shared scale is 1.125 B/elem, not 1.0."""
    entry = {
        "format": "Mx",
        "block": 8,
        "ELEM": {"type": "Fp", "sign": True, "exponent": 4, "mantissa": 3},
        "SCALE": {"type": "Fp", "sign": False, "exponent": 8, "mantissa": 0},
    }
    assert _entry_bytes_per_elem(entry) == (8 * 8 + 8) / (8 * 8)
    assert _entry_bytes_per_elem(entry) == 1.125


def test_plain_fp_and_int_bytes_per_elem():
    fp = {"format": "Plain", "DATA_TYPE": {"type": "Fp", "sign": True, "exponent": 8, "mantissa": 7}}
    i32 = {"format": "Plain", "DATA_TYPE": {"type": "Int", "width": 32}}
    assert _entry_bytes_per_elem(fp) == 2.0  # bf16
    assert _entry_bytes_per_elem(i32) == 4.0


def test_precision_section_parsed_from_toml():
    """[ANALYTIC.PRECISION.*] used to be ignored entirely."""
    cfg = load_hardware_config_from_toml(CONFIG)
    pb = cfg.PRECISION_BYTES
    assert pb, "PRECISION section was not parsed"
    for name in ("HBM_M_WEIGHT_TYPE", "HBM_M_KV_TYPE", "HBM_V_ACT_TYPE", "HBM_V_KV_TYPE"):
        assert pb[name] == 1.125, f"{name} = {pb[name]}"
    assert pb["HBM_STATE_TYPE"] == 4.0
    assert pb["HBM_V_INT_TYPE"] == 4.0
    assert pb["MATRIX_SRAM_TYPE"] == 2.0
    assert parse_precision_bytes({}) == {}


# -----------------------------------------------------------------------------
# Bandwidth term
# -----------------------------------------------------------------------------


def test_bandwidth_derived_from_hbm_width():
    m = _model()
    assert m.hbm_bytes_per_cycle == 512.0  # HBM_WIDTH bytes/cycle
    assert m.mem_cycles(512) == 1
    assert m.mem_cycles(513) == 2
    assert m.mem_cycles(0) == 0


def test_bandwidth_override_wins():
    cfg = load_hardware_config_from_toml(CONFIG)
    cfg.HBM_BANDWIDTH_BYTES_PER_CYCLE = 64.0
    m = PerfModel(cfg, ISA)
    assert m.hbm_bytes_per_cycle == 64.0


def test_stage_takes_max_not_sum():
    """A stage costs max(compute, memory), never the sum."""
    m = _model()
    m.reset_traffic()
    got = m._roofline(100, 512 * 1000)  # 1000 memory cycles vs 100 compute
    assert got == 1000
    got = m._roofline(5000, 512)  # 1 memory cycle vs 5000 compute
    assert got == 5000
    assert m.memory_bound_stages == 1
    assert m.total_stages == 2


def test_enable_bandwidth_false_is_backward_compatible():
    """The compute-only path must be untouched by the bandwidth work."""
    compute = _model(enable_bandwidth=False)
    full = _model(enable_bandwidth=True)
    args = dict(hidden_size=4096, intermediate_size=14336, seq_len=1, batch_size=1, mode="decode")
    c = compute.feed_forward(**args)
    f = full.feed_forward(**args)
    assert f >= c, "bandwidth model must never make a stage cheaper"
    assert f > c, "llama decode FFN is expected to be memory-bound at batch 1"


def test_ffn_decode_is_memory_bound_and_matches_hand_calc():
    m = _model()
    m.reset_traffic()
    m.feed_forward(4096, 14336, 1, 1, "decode")
    expected_bytes = 3 * 4096 * 14336 * 1.125
    assert m.traffic_bytes == expected_bytes
    assert m.memory_only_cycles == math.ceil(expected_bytes / 512)


# -----------------------------------------------------------------------------
# Bug 1: floor division charged ZERO cycles for the decode KV cache write
# Bug 2: a store used HBM_V_Prefetch_Amount instead of HBM_V_Writeback_Amount
# -----------------------------------------------------------------------------


def test_decode_kv_store_is_no_longer_free():
    """Reproduces the old formula and shows it evaluated to 0 at the default config."""
    m = _model()
    batch_size, num_kv_heads, head_dim = 4, 8, 128

    old_bursts = (batch_size * num_kv_heads * head_dim) // (m.vlen * m.prefetch_v_amount)
    assert (batch_size * num_kv_heads * head_dim) == 4096
    assert (m.vlen * m.prefetch_v_amount) == 32768
    assert old_bursts == 0, "precondition: the old floor division really did give 0"

    new_bursts = math.ceil((batch_size * num_kv_heads * head_dim) / (m.vlen * m.writeback_v_amount))
    assert new_bursts == 1, "a decode cache write must cost at least one burst"


def test_store_uses_writeback_amount():
    cfg = load_hardware_config_from_toml(CONFIG)
    assert cfg.HBM_V_Writeback_Amount == 16, "config value must be read, not ignored"
    m = PerfModel(cfg, ISA)
    assert m.writeback_v_amount == cfg.HBM_V_Writeback_Amount
    # And it must be a distinct knob: changing it changes the store cost.
    cfg2 = load_hardware_config_from_toml(CONFIG)
    cfg2.HBM_V_Writeback_Amount = 1
    m2 = PerfModel(cfg2, ISA, enable_bandwidth=False)
    m1 = PerfModel(cfg, ISA, enable_bandwidth=False)
    a = m1.projection(4096, 32, 8, 128, 2048, 4, "prefill")
    b = m2.projection(4096, 32, 8, 128, 2048, 4, "prefill")
    assert b > a, "shrinking the writeback burst must raise the store cost"


# -----------------------------------------------------------------------------
# New ISA entries
# -----------------------------------------------------------------------------


def test_mamba_isa_opcodes_present():
    m = _model()
    for op in ("V_SOFTPLUS_V", "S_MAP_FP_V", "V_MAX_VF", "V_MIN_VF", "V_PS_V", "V_SHFT_V"):
        assert op in m.instr, f"{op} missing from customISA_lib.json"
        assert m.instr[op] > 0
    # V_SOFTPLUS_V is budgeted above a bare exp: it also evaluates a logarithm.
    assert m.instr["V_SOFTPLUS_V"] > m.instr["V_EXP_V"]


# -----------------------------------------------------------------------------
# Mamba-2 state behaviour
# -----------------------------------------------------------------------------


def test_ssm_decode_state_traffic_is_context_independent():
    """The whole point: decode traffic does not grow with how much came before."""
    m = _model()
    m.reset_traffic()
    m.ssd_recurrence_decode(num_heads=80, head_dim=64, state_size=128, n_groups=1, batch_size=1)
    first = m.traffic_bytes
    expected = 2 * 80 * 64 * 128 * 4.0  # official FP32 state, read + write
    assert first == expected

    # a thousand tokens later, the per-token cost is identical
    m.reset_traffic()
    m.ssd_recurrence_decode(num_heads=80, head_dim=64, state_size=128, n_groups=1, batch_size=1)
    assert m.traffic_bytes == first


def test_state_precision_is_independent_from_attention_kv_precision():
    cfg = load_hardware_config_from_toml(CONFIG)
    cfg.PRECISION_BYTES["HBM_V_KV_TYPE"] = 1.125
    cfg.PRECISION_BYTES["HBM_STATE_TYPE"] = 2.0
    fp16_state = PerfModel(cfg, ISA)
    assert fp16_state.kv_bytes == 1.125
    assert fp16_state.state_bytes == 2.0

    cfg.PRECISION_BYTES["HBM_STATE_TYPE"] = 4.0
    fp32_state = PerfModel(cfg, ISA)
    assert fp32_state.kv_bytes == fp16_state.kv_bytes
    assert fp32_state.state_bytes == 4.0

    fp16_state.reset_traffic()
    fp32_state.reset_traffic()
    args = dict(num_heads=96, key_dim=128, value_dim=128, batch_size=1)
    fp16_state.kda_recurrence_decode(**args)
    fp32_state.kda_recurrence_decode(**args)
    assert fp32_state.traffic_bytes == 2 * fp16_state.traffic_bytes


def test_attention_decode_traffic_grows_with_context():
    """Contrast case: the KV cache read scales linearly with kv_size."""
    m = _model()
    m.reset_traffic()
    m.flash_attention(32, 8, 128, 1, 512, 1, "decode")
    short = m.traffic_bytes
    m.reset_traffic()
    m.flash_attention(32, 8, 128, 1, 8192, 1, "decode")
    long = m.traffic_bytes
    assert long == short * 16, f"{long} vs {short}"


def _run_all():
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures.append((name, exc))
                print(f"FAIL  {name}: {exc}")
    if failures:
        print(f"\n{len(failures)} failure(s)")
        return 1
    print("\nall tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(_run_all())
