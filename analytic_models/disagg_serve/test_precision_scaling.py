"""Precision invariants of the decode analytic model.

Three widths matter: N = HBM stream width (weight / KV element bits) sets every
memory quantity; M = MAC operand width (compute only); P = accumulator width,
requantized to N on writeback so it never appears in byte counts. These tests
check that contract:

  1. Halving KV bits halves KV read traffic, KV footprint, and hand-off bytes.
  2. Write traffic depends on N (kv_bits), not P.
  3. Halving the widest stream width doubles the MLEN bandwidth cap.
  4. M (m_bits) changes no memory quantity.

Run:  python analytic_models/disagg_serve/test_precision_scaling.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "performance"))
sys.path.insert(0, str(_HERE.parent.parent))

from disagg_decode import (  # noqa: E402
    decode_traffic,
    kv_footprint_bytes,
    load_hardware_config_from_toml,
    load_memory_config_from_toml,
    load_model_dims,
    mlen_bandwidth_cap,
    precision_from_components,
    resolve_model_path,
)
from analytic_models.disagg_serve.handoff import kv_wire_bytes  # noqa: E402


def _prec(attn=4, ffn=4, kv=4, m_bits=None):
    return precision_from_components(
        float(attn), float(ffn), float(kv), f"i{attn}", f"i{ffn}", f"i{kv}",
        attn_elem=attn, ffn_elem=ffn, kv_elem=kv, m_bits=m_bits,
    )


def main() -> None:
    repo = _HERE.parent.parent
    model_path = resolve_model_path("llama-3.1-8b", str(repo / "compiler" / "doc" / "Model_Lib"))
    dims = load_model_dims(model_path)
    hw = load_hardware_config_from_toml(str(repo / "plena_settings.toml"))
    base_mem = load_memory_config_from_toml(str(repo / "plena_settings.toml"))

    checks = []

    def check(name, cond):
        checks.append((name, cond))
        print(("PASS " if cond else "FAIL ") + name)

    p4, p2 = _prec(kv=4), _prec(kv=2)
    mem = base_mem

    from disagg_decode import LLMMemoryModel  # noqa: E402
    mm4 = LLMMemoryModel(model_path, mem.model_copy(update={"kv_cache_bits": 4}),
                         batch_size=8, input_seq_len=1024, output_seq_len=1024).mem
    mm2 = LLMMemoryModel(model_path, mem.model_copy(update={"kv_cache_bits": 2}),
                         batch_size=8, input_seq_len=1024, output_seq_len=1024).mem

    # 1a. Isolate the KV read term by taking the traffic difference between two
    # context lengths (weights cancel), and check it halves with kv_bits.
    kv_lo, kv_hi = 1024, 2048
    d4 = decode_traffic(mm4, dims, kv_hi, 8, p4).read_bytes - decode_traffic(mm4, dims, kv_lo, 8, p4).read_bytes
    d2 = decode_traffic(mm2, dims, kv_hi, 8, p2).read_bytes - decode_traffic(mm2, dims, kv_lo, 8, p2).read_bytes
    check("KV read traffic halves with kv_bits 4->2", abs(d2 / d4 - 0.5) < 0.05)

    # 1b. KV footprint halves.
    f4 = kv_footprint_bytes(mm4, dims, p4, 2048, 8)
    f2 = kv_footprint_bytes(mm2, dims, p2, 2048, 8)
    check("KV footprint halves with kv_bits 4->2", abs(f2 / f4 - 0.5) < 0.05)

    # 1c. Hand-off wire bytes halve.
    w4 = kv_wire_bytes(dims, p4, 1024, 8)
    w2 = kv_wire_bytes(dims, p2, 1024, 8)
    check("hand-off wire bytes halve with kv_bits 4->2", abs(w2 / w4 - 0.5) < 0.05)

    # 2. Writes depend on N, not P: same N with different m_bits must give the
    # same write bytes.
    pm4, pm8 = _prec(kv=4, m_bits=4), _prec(kv=4, m_bits=8)
    wr_a = decode_traffic(mm4, dims, 2048, 8, pm4).write_bytes
    wr_b = decode_traffic(mm4, dims, 2048, 8, pm8).write_bytes
    check("write bytes independent of M/P", wr_a == wr_b)

    # 3. MLEN bandwidth cap doubles when the widest stream halves.
    cap8 = mlen_bandwidth_cap(hw, _prec(attn=8, ffn=8, kv=8))
    cap4 = mlen_bandwidth_cap(hw, _prec(attn=4, ffn=4, kv=4))
    check("MLEN bandwidth cap doubles with stream bits 8->4", cap4 == 2 * cap8)

    # 4. M moves no memory-facing quantity.
    rd_a = decode_traffic(mm4, dims, 2048, 8, pm4).read_bytes
    rd_b = decode_traffic(mm4, dims, 2048, 8, pm8).read_bytes
    check("read bytes independent of M", rd_a == rd_b)

    failed = [n for n, ok in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
