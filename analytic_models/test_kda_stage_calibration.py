"""The KDA cost model, checked against the compiler's real instruction counts.

Every other stage in `perf_model.py` is a hand-derived formula with nothing to
check it against: the lowering it describes does not exist, so the formula can be
wrong in any direction and no test will say so. The KDA stages are the exception.
`PLENA_Compiler` emits this kernel, `test_kda_prefill_structure.py` counts the
instructions it emits, and those counts are reproduced below as the oracle.

What is compared, and why it is instructions and not cycles
-----------------------------------------------------------
`PerfModel` returns cycles, which are latencies times counts, and the latencies
are themselves unvalidated. Setting every opcode latency to 1 turns the same
formula into a count of instructions issued, which the compiler can be asked for
directly. So this pins the part that can be checked -- how many of each thing the
kernel does, and how that scales -- and leaves the part that cannot.

The hardware shape is forced to `MLEN 64 / BLEN 4 / VLEN 64`, the TRANSACTIONAL
config the measurements were taken at, not the ANALYTIC `MLEN 2048` the driver
runs with. Those two disagree (see `hybrid_model.py`'s module docstring); this
file deliberately uses the one the oracle came from.

Where it lands
--------------
Prefill ratios run **0.87 to 1.14** across the grid, centred on 1.0; decode is
0.98. The residual is bounded rather than tuned to zero -- the terms that would
close it are bookkeeping the formula does not model in detail (register spills,
immediate legalisation, loop prologues), and a fudge factor would make the
number agree without making the model more correct.

One structural miss was found this way and fixed rather than absorbed. The model
scaled with `key_dim` at half the compiler's rate, worst at `key_dim 128,
value_dim 64` (0.75). The cause was real: a VRAM matrix wider than `mlen` is
column-block-major, so a spill's zero-fill and store walk every block, and
`per_spill` was billing one. Fixing that took the range from 0.75-0.96 to
0.87-1.14. **That is the whole value of having an oracle** -- the formula was
wrong in a specific, findable way, and nothing else here could have said so.

What this covers, and what it does not
--------------------------------------
Zeroing a term and re-running gives what each is worth. Against the compiler's
2,426 (prefill, chunk 16, 128 x 128) and 2,616 (decode):

    prefill, whole spill term removed        0.70   caught
    prefill, UT forward substitution removed 0.91   NOT caught
    prefill, all seven matmuls removed       0.91   NOT caught
    decode,  scalar sweep overhead removed   0.39   caught
    decode,  the FMA itself removed          0.69   caught

**The instruction-count oracle is blind to the matmuls and the UT transform.**
Each matmul is a single `M_TMM` issue and the substitution is ~120 sweeps out of
2,450, so removing either moves the count by 9% and stays inside any bound loose
enough for the real residual -- while both are large in *cycles*, which is what
the driver actually reports. Those terms have the same unvalidated status as
every other stage in `perf_model.py`. What is pinned here is the overall
magnitude, the scaling in all three axes, and the two terms that dominate the
count: the spills and the scalar sweep overhead.

The decode form is the tighter of the two, and only because that overhead is
billed. Half the compiled kernel's dynamic stream is `S_ADDI_INT` pointer
arithmetic and a quarter is `S_LD_FP`, against 18% for the arithmetic -- a model
counting only the vector ops lands at 0.39.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analytic_models" / "performance"))

from perf_model import PerfModel, load_hardware_config_from_toml  # noqa: E402

#: Static instruction counts emitted by `PLENA_Compiler` for one head, measured
#: 2026-08-26 and pinned by `aten/tests/test_kda_prefill_structure.py`'s
#: `PREFILL_STATIC_MAX` at the same shapes.
MEASURED_PREFILL_STATIC = {
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

#: Dynamic instruction count for one KDA decode step at Kimi's per-head shape,
#: from `doc/static_path_measurements.md`.
MEASURED_DECODE_DYNAMIC = 2616

#: The model must stay inside these of the compiler. Measured range is 0.87-1.14,
#: so this leaves under 10% of margin on each side -- tight enough that removing
#: the spill term (0.70) fails, loose enough not to be brittle against the known
#: residual. See the module docstring for what these bounds cannot see.
RATIO_MIN, RATIO_MAX = 0.80, 1.20


class _UnitLatency:
    """Every opcode costs 1, so `cycles` reads as `instructions issued`."""

    def __getitem__(self, key: str) -> int:
        return 1

    def __contains__(self, key: str) -> bool:
        return True


def _compiler_shaped_model() -> PerfModel:
    hw = load_hardware_config_from_toml(str(ROOT / "plena_settings.toml"))
    # The TRANSACTIONAL shape, which is what the oracle was measured at.
    hw.MLEN, hw.BLEN, hw.VLEN = 64, 4, 64
    perf = PerfModel(hw, str(ROOT / "analytic_models" / "performance" / "customISA_lib.json"), enable_bandwidth=False)
    perf.instr = _UnitLatency()
    return perf


@pytest.mark.parametrize("shape", sorted(MEASURED_PREFILL_STATIC))
def test_prefill_instruction_count_tracks_the_compiler(shape) -> None:
    chunk, key_dim, value_dim = shape
    perf = _compiler_shaped_model()
    modelled = perf.kda_chunk_prefill(
        num_heads=1,
        key_dim=key_dim,
        value_dim=value_dim,
        chunk_size=chunk,
        seq_len=chunk,
        batch_size=1,
    )
    measured = MEASURED_PREFILL_STATIC[shape]
    ratio = modelled / measured
    assert RATIO_MIN <= ratio <= RATIO_MAX, (
        f"chunk {chunk}, key {key_dim}, value {value_dim}: modelled {modelled} "
        f"against the compiler's {measured} is a ratio of {ratio:.2f}, outside "
        f"[{RATIO_MIN}, {RATIO_MAX}]"
    )


def test_decode_instruction_count_tracks_the_compiler() -> None:
    """The decode form, which lands at 0.98.

    It does so only because the scalar sweep overhead is billed: half the
    compiled kernel's dynamic stream is `S_ADDI_INT` pointer arithmetic and a
    quarter is `S_LD_FP`, against 18% for the arithmetic. Counting only the
    vector ops -- the natural thing to write -- lands at 0.39.
    """
    perf = _compiler_shaped_model()
    modelled = perf.kda_recurrence_decode(num_heads=1, key_dim=128, value_dim=128, batch_size=1)
    ratio = modelled / MEASURED_DECODE_DYNAMIC
    assert 0.90 <= ratio <= 1.10, (
        f"modelled {modelled} against the compiler's {MEASURED_DECODE_DYNAMIC} is a ratio of {ratio:.2f}"
    )


def test_prefill_scales_with_both_axes_and_the_chunk() -> None:
    """The magnitude could be right by luck; the scaling could not.

    Doubling the key axis, the value axis or the chunk must each raise the count,
    and none of them may leave it unchanged -- an axis a formula ignores is the
    failure mode a single-shape check cannot see.
    """
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
    assert cost(8, 128, 64) > base, "the key axis must cost something"
    assert cost(8, 64, 128) > base, "the value axis must cost something"
    assert cost(16, 64, 64) > base, "the chunk must cost something"


def test_decode_state_traffic_is_the_whole_state_read_and_written() -> None:
    """The O(1)-in-context term, which is the point of a recurrent mixer.

    With no memory term the state update looks free, and the comparison against
    a growing KV cache becomes meaningless. Checked as bytes rather than cycles
    because `_roofline` takes the max of the two and the compute side would hide
    a wrong byte count at this shape.
    """
    hw = load_hardware_config_from_toml(str(ROOT / "plena_settings.toml"))
    perf = PerfModel(hw, str(ROOT / "analytic_models" / "performance" / "customISA_lib.json"))
    heads, key_dim, value_dim = 32, 128, 128

    perf.reset_traffic()
    perf.kda_recurrence_decode(num_heads=heads, key_dim=key_dim, value_dim=value_dim, batch_size=1)
    expected = 2 * heads * key_dim * value_dim * perf.state_bytes
    assert perf.traffic_bytes == pytest.approx(expected), (
        "decode must read and write the whole recurrent state exactly once"
    )


def test_row_granular_prefetch_is_off_by_default_and_changes_nothing() -> None:
    """The hypothetical must not move the calibrated number.

    `row_granular_prefetch` prices an instruction that does not exist. If it
    leaked into the default it would move this file's oracle comparison against
    a compiler that emits the instruction set that does, and the calibration
    would be measuring the wrong machine.
    """
    perf = _compiler_shaped_model()
    shape = dict(num_heads=1, key_dim=128, value_dim=128, chunk_size=16, seq_len=16, batch_size=1)
    assert perf.kda_chunk_prefill(**shape) == perf.kda_chunk_prefill(**shape, row_granular_prefetch=False)


def test_row_granular_prefetch_pays_in_proportion_to_machine_width() -> None:
    """What the instruction is worth, and the reason it is worth anything.

    The fill it removes scales with `mlen` while the data it protects scales
    with `chunk`, so the saving has to grow with width -- a factor that did not
    would mean the term was being modelled as something other than a per-row
    fill over an `mlen`-tall tile.

    The bounds are loose because the point is the shape, not the digits: nearly
    nothing at `mlen` 64, an order of magnitude at 2048. Anyone quoting the wide
    figure as "the value of this instruction" should be made to say which
    machine they mean.
    """
    shape = dict(num_heads=1, key_dim=128, value_dim=128, chunk_size=16, seq_len=16, batch_size=1)
    factors = {}
    for mlen, blen, hlen, vlen in ((64, 4, 16, 64), (512, 32, 64, 512), (2048, 128, 128, 2048)):
        hw = load_hardware_config_from_toml(str(ROOT / "plena_settings.toml"))
        hw.MLEN, hw.BLEN, hw.HLEN, hw.VLEN = mlen, blen, hlen, vlen
        perf = PerfModel(
            hw,
            str(ROOT / "analytic_models" / "performance" / "customISA_lib.json"),
            enable_bandwidth=False,
        )
        perf.instr = _UnitLatency()
        whole = perf.kda_chunk_prefill(**shape)
        rows = perf.kda_chunk_prefill(**shape, row_granular_prefetch=True)
        factors[mlen] = whole / rows

    assert factors[64] < 2.0, f"at mlen 64 it should barely pay; got {factors[64]:.1f}x"
    assert factors[2048] > 8.0, f"at mlen 2048 it should pay heavily; got {factors[2048]:.1f}x"
    assert factors[64] < factors[512] < factors[2048], f"the saving must grow with machine width, got {factors}"
