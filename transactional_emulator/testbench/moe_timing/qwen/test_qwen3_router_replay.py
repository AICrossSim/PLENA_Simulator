"""Guards that the campaign's replay program routes on device.

The trace replay used to read ``topk_indices`` out of the trace and preload them
into INT SRAM, so ``V_TOPK`` -- the instruction the routed-MoE ISA work exists
for -- never appeared in a program the timing campaign measured. Every cycle
count it produced excluded routing, and ``router_topk`` was a stage name nothing
emitted.

These build the program and read the assembly. Running it is the separate,
slower gate in ``test_router_replay_end_to_end``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay import (
    _router_logits_layout,
    _vram_extent,
    build_artifacts,
    build_parser,
)
from transactional_emulator.testbench.moe_timing.qwen.synthetic_trace import synthetic_trace

TOKENS = 2
MLEN = 128


def _instruction_lines(asm: str) -> list[str]:
    """Emitted instructions only.

    The router's stage marker quotes the opcode it is about ("[qwen3_moe] V_TOPK
    ..."), so counting substrings over the whole listing returns two per token
    and a program that emitted no instruction at all could still look right.
    """
    return [line.strip() for line in asm.splitlines() if line.strip() and not line.lstrip().startswith(";")]


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build once: the program is ~86k instructions and takes a couple of seconds."""
    root = tmp_path_factory.mktemp("router_replay")
    trace_path = root / "trace.json"
    trace_path.write_text(json.dumps(synthetic_trace(tokens=TOKENS, seed=7, mlen=MLEN)) + "\n")
    args = build_parser().parse_args(
        [str(trace_path), "--mlen", str(MLEN), "--build-dir", str(root / "build"), "--stage-profile", "--no-run"]
    )
    result = build_artifacts(args)
    asm = (Path(result["build_dir"]) / "generated_asm_code.asm").read_text()
    return {
        "asm": asm,
        "manifest": result["manifest"],
        "trace": result["trace"],
        "staged_router_logits": result["staged_router_logits"],
    }


def test_the_program_issues_one_v_topk_per_token(built: dict) -> None:
    """One selection per token, not one per (token, expert) pair.

    V_TOPK returns the whole top-k for a row in one instruction. Emitting it per
    pair would still produce correct indices and would still make `router_topk`
    appear -- while inflating the router's measured cost by top_k.
    """
    issued = [line for line in _instruction_lines(built["asm"]) if line.startswith("V_TOPK")]

    assert len(issued) == TOKENS


def test_the_router_stage_reaches_the_emitted_program(built: dict) -> None:
    """`router_topk` was in MOE_STAGES with no emitter reaching a measured program.

    A stage in the vocabulary that never appears in a profile is indistinguishable
    from a stage whose cost is zero.
    """
    assert "@stage=router_topk" in built["asm"]


def test_selection_precedes_every_expert_weight_address(built: dict) -> None:
    """The indices must be computed before anything reads them.

    `_emit_expert_id_to_weight_base_v0` loads the expert id from INT SRAM with
    S_LD_INT. Emitting V_TOPK after that point leaves the load reading whatever
    the preload left behind -- which, now that the preload is gone, is zeros:
    every pair would silently route to expert 0 and the program would still run.
    """
    lines = built["asm"].splitlines()
    last_v_topk = max(i for i, line in enumerate(lines) if line.strip().startswith("V_TOPK"))
    first_expert_load = min(i for i, line in enumerate(lines) if "@stage=expert_weight_address" in line)

    assert last_v_topk < first_expert_load


def test_the_manifest_records_that_routing_ran_on_device(built: dict) -> None:
    """The artifact has to say so: results are read long after the run.

    `export_pilot_results` already carries a note saying trace replay excludes
    the router. Without a machine-readable flag the two can drift apart with
    nothing to catch it.
    """
    router = built["manifest"]["router"]

    assert router["on_device"] is True
    assert router["v_topk_count"] == TOKENS


def test_the_manifest_names_what_the_reconstruction_does_not_cover(built: dict) -> None:
    """The router GEMM is still absent, and the unselected logits are synthetic.

    Stating it in the artifact rather than only in a docstring: this is the
    number someone will quote, and "the router is now measured" is false in a way
    that matters -- the projection from hidden to num_experts is not in it.
    """
    router = built["manifest"]["router"]

    assert router["router_gemm_included"] is False
    assert "logits" in router["reconstruction"]


def test_the_reconstructed_logits_are_wide_enough_for_v_topk(built: dict) -> None:
    """V_TOPK reads exactly num_experts values from the row base.

    A row narrower than num_experts would make it scan into the next token's
    logits, picking experts for the wrong token -- with indices that are all in
    range and a program that runs clean.
    """
    router = built["manifest"]["router"]

    assert router["logits_cols"] >= built["trace"]["model"]["num_experts"]


# The campaign runs at MLEN 128, where 128 experts fit one row and the fold below
# is the identity. These cover MLEN 64, where each token spans two rows -- a path
# the end-to-end run never reaches.


def test_a_token_wider_than_mlen_folds_into_consecutive_rows() -> None:
    """V_TOPK addresses only the first row and reads num_experts values onward.

    So the second half of a token's logits has to be the row immediately after
    the first. Keeping one row per token and padding to num_experts would make
    V_TOPK read 64 real logits and then 64 belonging to the next token.
    """
    logits = torch.arange(2 * 128, dtype=torch.bfloat16).reshape(2, 128)

    folded, physical = _router_logits_layout(logits, mlen=64, blen=4)

    assert folded.shape == (4, 64)
    assert physical == (4, 64)
    assert torch.equal(folded[0], logits[0, :64])
    assert torch.equal(folded[1], logits[0, 64:])
    assert torch.equal(folded[2], logits[1, :64])


def test_a_token_that_fits_one_row_is_left_alone() -> None:
    """At MLEN 128 the fold must be the identity, or the campaign's own shape moves.

    Fails if the reshape is applied unconditionally: (2, 128) would become
    (2, 128) by luck here but (2, 96) at 96 experts would become garbage.
    """
    logits = torch.arange(2 * 128, dtype=torch.bfloat16).reshape(2, 128)

    kept, physical = _router_logits_layout(logits, mlen=128, blen=4)

    assert torch.equal(kept, logits)
    assert physical == (4, 128)


def test_rows_are_padded_to_a_whole_number_of_blen_tiles() -> None:
    """Two tokens at MLEN 64 is four logit rows, which is already BLEN-aligned.

    One token is two rows and must still be padded to four, matching how the
    hidden state is staged. An unpadded matrix would leave the accumulator and
    the logits disagreeing about where the next allocation starts.
    """
    _, physical = _router_logits_layout(torch.zeros(1, 128, dtype=torch.bfloat16), mlen=64, blen=4)

    assert physical == (4, 64)


def test_the_vram_extent_counts_whole_column_blocks() -> None:
    """A matrix narrower than MLEN still costs a full block.

    This is what places the logits clear of the hidden state. Computing it as
    rows*cols would overlap the two whenever cols is not a multiple of MLEN, and
    the hidden state would be overwritten by logits before the program ran.
    """
    assert _vram_extent(rows=4, cols=128, mlen=64) == 2 * 4 * 64
    assert _vram_extent(rows=4, cols=96, mlen=64) == 2 * 4 * 64
    assert _vram_extent(rows=4, cols=64, mlen=64) == 1 * 4 * 64


# `_router_gate` reads two binary dumps at manifest-supplied offsets and reshapes
# them. Swapped bases, a dropped per-token stride, or a transposed reshape all
# leave every guard above passing -- they only surface in the end-to-end recipe,
# a build-and-run of tens of thousands of instructions. These drive the gate
# directly from dumps written here.


def _write_dumps(build_dir: Path, *, indices, weights, indices_base: int, weights_base: int) -> None:
    """Write intsram/fpsram dumps in the encoding the emulator produces."""
    import numpy as np

    ints = np.zeros(1024, dtype="<u4")
    flat = [expert for row in indices for expert in row]
    ints[indices_base : indices_base + len(flat)] = np.array(flat, dtype="<u4")
    (build_dir / "intsram_dump.bin").write_bytes(ints.tobytes())

    fps = torch.zeros(1024, dtype=torch.bfloat16)
    flat_w = torch.tensor([w for row in weights for w in row], dtype=torch.bfloat16)
    fps[weights_base : weights_base + flat_w.numel()] = flat_w
    (build_dir / "fpsram_dump.bin").write_bytes(fps.view(torch.uint16).numpy().astype("<u2").tobytes())


def _gate_inputs(built: dict, tmp_path: Path, *, mutate=None):
    """Dumps written against the manifest a real build produced.

    The bases come from `built`, not from literals. An earlier version wrote both
    the dumps and the manifest at a hand-typed 145: any change to the FP layout
    would move the real base while these kept agreeing with themselves, which is
    the failure the campaign guards were rewritten to avoid.
    """
    manifest, trace, staged = built["manifest"], built["trace"], built["staged_router_logits"]
    indices = trace["routing"]["topk_indices"]
    top_k = trace["model"]["top_k"]
    weights = torch.softmax(torch.topk(staged.float(), k=top_k, dim=-1).values, dim=-1).tolist()

    device_indices = [list(row) for row in indices]
    if mutate is not None:
        mutate(device_indices)

    build = tmp_path / "b"
    build.mkdir()
    _write_dumps(
        build,
        indices=device_indices,
        weights=weights,
        indices_base=manifest["topk_indices_int_base"],
        weights_base=manifest["topk_weights_fp_base"],
    )
    return build, manifest, trace, staged


def test_the_gate_passes_when_the_dumps_carry_the_traces_experts(built: dict, tmp_path: Path) -> None:
    from transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay import _router_gate

    build, manifest, trace, staged = _gate_inputs(built, tmp_path)

    gate = _router_gate(build, manifest, trace, staged)

    assert gate["passed"] is True
    assert gate["expert_ids_match"] is True
    assert gate["pairs_checked"] == TOKENS * trace["model"]["top_k"]


def test_the_gate_reports_one_wrong_expert_with_its_coordinate(built: dict, tmp_path: Path) -> None:
    """One id off, in the last token, so a gate reading only the first row misses it."""
    from transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay import _router_gate

    def flip(rows):
        rows[-1][5] = (rows[-1][5] + 1) % 128

    build, manifest, trace, staged = _gate_inputs(built, tmp_path, mutate=flip)

    gate = _router_gate(build, manifest, trace, staged)

    assert gate["passed"] is False
    assert gate["expert_ids_match"] is False
    assert gate["index_mismatch_count"] == 1
    assert gate["index_mismatch_coordinates"] == [[TOKENS - 1, 5]]


def test_the_gate_reads_each_token_at_its_own_offset(built: dict, tmp_path: Path) -> None:
    """Rotating the rows keeps every id present but moves each to another token.

    A gate that read the whole block without honouring the per-token stride --
    or that reshaped as (top_k, rows) -- would still see the same multiset and
    pass.
    """
    from transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay import _router_gate

    def rotate(rows):
        rows.append(rows.pop(0))

    build, manifest, trace, staged = _gate_inputs(built, tmp_path, mutate=rotate)

    gate = _router_gate(build, manifest, trace, staged)

    assert gate["passed"] is False


def test_the_gate_fails_when_the_router_wrote_nothing(built: dict, tmp_path: Path) -> None:
    """Zeroed INT SRAM is what a program that never issued V_TOPK leaves behind."""
    from transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay import _router_gate

    def zero(rows):
        for row in rows:
            row[:] = [0] * len(row)

    build, manifest, trace, staged = _gate_inputs(built, tmp_path, mutate=zero)

    gate = _router_gate(build, manifest, trace, staged)

    assert gate["passed"] is False
    assert gate["index_mismatch_count"] > 0


def test_a_missing_dump_is_reported_as_an_unreadable_gate(built: dict, tmp_path: Path) -> None:
    """Not a numpy traceback, and not a pass.

    The run happened and cannot be verified, which is neither "passed" nor the
    "absent means unknown" case that applies to artifacts predating the gate.
    Letting FileNotFoundError escape scores it as a crash and names numpy as the
    culprit; reporting it as passing puts an unchecked measurement in the medians.
    """
    from transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay import _router_gate

    build, manifest, trace, staged = _gate_inputs(built, tmp_path)
    (build / "intsram_dump.bin").unlink()

    gate = _router_gate(build, manifest, trace, staged)

    assert gate["passed"] is False
    assert "intsram_dump.bin is missing" in gate["unavailable"]


def test_a_truncated_dump_names_the_shortfall(built: dict, tmp_path: Path) -> None:
    """A short dump slices short and dies in reshape, blaming tensor sizes."""
    from transactional_emulator.testbench.moe_timing.qwen.qwen3_trace_replay import _router_gate

    build, manifest, trace, staged = _gate_inputs(built, tmp_path)
    (build / "fpsram_dump.bin").write_bytes(b"\x00" * 16)

    gate = _router_gate(build, manifest, trace, staged)

    assert gate["passed"] is False
    assert "fpsram_dump.bin holds 8 entries" in gate["unavailable"]


def test_the_manifest_records_whether_the_run_matched_the_traces_geometry(built: dict) -> None:
    """mlen now decides the logit fold, not just tiling.

    `run_trace_batch` passes its own --mlen to every replay regardless of what the
    trace was built for, so the artifact has to say whether the two agreed.
    """
    router = built["manifest"]["router"]

    assert router["mlen_matches_trace_metadata"] is True
    assert router["trace_metadata_mlen"] == MLEN
