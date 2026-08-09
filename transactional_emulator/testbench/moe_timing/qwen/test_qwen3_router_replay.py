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
    return {"asm": asm, "manifest": result["manifest"], "trace": result["trace"]}


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
