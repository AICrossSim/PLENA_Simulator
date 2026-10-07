"""KDA stage-cost regression against Compiler instruction counts.

Unit opcode latencies isolate the structural count from timing assumptions.
The constants below are Compiler counts at MLEN/VLEN=64 and BLEN=4, also
covered by Compiler's test_kda_prefill_structure.py. This approximate regression
checks count magnitude, axis scaling and state traffic; it does not validate
matrix latency or replace the actual Compiler-to-Rust stage tests.
"""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analytic_models" / "performance"))

from perf_model import PerfModel, load_hardware_config_from_toml  # noqa: E402


COMPILER_PREFILL_STATIC = {
    (4, 64, 64): 818,
    (4, 128, 64): 1237,
    (4, 128, 128): 1470,
    (8, 64, 64): 1039,
    (8, 128, 64): 1547,
    (8, 128, 128): 1792,
    (16, 64, 64): 1476,
    (16, 128, 64): 2157,
    (16, 128, 128): 2426,
}
COMPILER_DECODE_DYNAMIC = 2616
ISA_PATH = str(ROOT / "analytic_models" / "performance" / "customISA_lib.json")


class _UnitLatency:
    def __getitem__(self, key: str) -> int:
        return 1

    def __contains__(self, key: str) -> bool:
        return True


def _compiler_shaped_model() -> PerfModel:
    hw = load_hardware_config_from_toml(str(ROOT / "plena_settings.toml"))
    hw.MLEN, hw.BLEN, hw.VLEN = 64, 4, 64
    perf = PerfModel(hw, ISA_PATH, enable_bandwidth=False)
    perf.instr = _UnitLatency()
    return perf


@pytest.mark.parametrize("shape", sorted(COMPILER_PREFILL_STATIC))
def test_prefill_instruction_count_tracks_the_compiler(shape) -> None:
    chunk, key_dim, value_dim = shape
    modelled = _compiler_shaped_model().kda_chunk_prefill(
        num_heads=1,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=chunk,
        seq_len=chunk,
        batch_size=1,
    )
    assert 0.80 <= modelled / COMPILER_PREFILL_STATIC[shape] <= 1.20


def test_decode_instruction_count_tracks_the_compiler() -> None:
    modelled = _compiler_shaped_model().kda_recurrence_decode(
        num_heads=1,
        key_dim=128,
        value_dim=128,
        batch_size=1,
    )
    assert 0.90 <= modelled / COMPILER_DECODE_DYNAMIC <= 1.10


def test_prefill_scales_with_both_axes_and_the_chunk() -> None:
    perf = _compiler_shaped_model()

    def cost(chunk: int, key_dim: int, value_dim: int) -> int:
        return perf.kda_chunk_prefill(
            num_heads=1,
            key_dim=key_dim,
            value_dim=value_dim,
            chunk_size=chunk,
            seq_len=chunk,
            batch_size=1,
        )

    base = cost(8, 64, 64)
    assert cost(8, 128, 64) > base
    assert cost(8, 64, 128) > base
    assert cost(16, 64, 64) > base


def test_decode_state_traffic_is_the_whole_state_read_and_written() -> None:
    hw = load_hardware_config_from_toml(str(ROOT / "plena_settings.toml"))
    perf = PerfModel(hw, ISA_PATH)
    heads, key_dim, value_dim = 32, 128, 128
    perf.kda_recurrence_decode(
        num_heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        batch_size=1,
    )
    assert perf.traffic_bytes == pytest.approx(2 * heads * key_dim * value_dim * perf.state_bytes)
