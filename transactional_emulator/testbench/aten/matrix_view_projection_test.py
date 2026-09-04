"""Compiler-to-Rust Matrix-view projection and direct-consumer test.

The Matrix accumulator produces sixteen BLEN-wide fragments. The configured
consumer view groups two fragments into each logical head tile, so this test
would fail if the producer used the old producer-only 16x4 descriptor. Existing
V_ADD_VV arithmetic reads the restored Matrix packet directly through its `.MV`
addressing mode; no layout-copy opcode is emitted.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
_compiler_override = os.environ.get("PLENA_COMPILER_ROOT")
_compiler_candidates = (
    [Path(_compiler_override)]
    if _compiler_override
    else [
        _REPO_ROOT.parent / "compiler-static-kda-latest",
        _REPO_ROOT / "PLENA_Compiler",
        _REPO_ROOT.parent / "PLENA_Compiler",
    ]
)
_ACTIVE_COMPILER_ROOT = None
for _compiler_root in _compiler_candidates:
    if (_compiler_root / "aten" / "plena" / "mview.py").exists():
        sys.path.insert(0, str(_compiler_root))
        _ACTIVE_COMPILER_ROOT = _compiler_root
        break
if _ACTIVE_COMPILER_ROOT is None:
    raise RuntimeError("A Compiler checkout with Matrix-view support was not found; set PLENA_COMPILER_ROOT")

from compiler.asm_templates._imm import load_large_int  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.mview import (  # noqa: E402
    MatrixViewDescriptor,
    MatrixViewMap,
    MatrixViewShape,
    validate_matrix_view_dominance,
)
from transactional_emulator.testbench.aten.golden import golden_linear  # noqa: E402
from transactional_emulator.testbench.emulator_runner import run_and_assert  # noqa: E402
from transactional_emulator.testbench.sim_env_utils import (  # noqa: E402
    create_mem_for_sim,
)
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402


MLEN = 64
BLEN = 4
ROWS = 1
K = 64
N = 64
CONSUMER_WIDTH = 8
COMPILER_ROOT = _ACTIVE_COMPILER_ROOT


def main() -> None:
    build_dir = Path(__file__).parent / "build" / "matrix_view_projection"
    torch.manual_seed(20260901)
    x = torch.randn(ROWS, K)
    x_storage = torch.zeros(BLEN, K)
    x_storage[:ROWS].copy_(x)
    weight = torch.randn(K, N)
    golden = golden_linear(x, weight)

    program = PlenaCompiler(mlen=MLEN, blen=BLEN, mram_tile_capacity=64)
    x_input = program.input(
        "X",
        shape=(ROWS, K),
        physical_shape=(BLEN, K),
        real_data_ratio=1.0,
    )
    w_input = program.input(
        "W",
        shape=(K, N),
        physical_shape=(K, N),
    )
    zero_input = program.input(
        "zero",
        shape=(ROWS, N),
        physical_shape=(BLEN, N),
        real_data_ratio=1.0,
    )
    x_vram = program.load_batch(x_input, name="X_vram")
    zero = program.load_batch(zero_input, name="zero_vram")
    output_placeholder = program.alloc(
        "matrix_output_placeholder",
        ROWS,
        N,
        strict=False,
        physical_shape=(BLEN, N),
    )
    restored = program.alloc(
        "restored",
        ROWS,
        N,
        strict=False,
        physical_shape=(BLEN, N),
    )
    descriptor = MatrixViewDescriptor(
        shape=MatrixViewShape(
            rows=ROWS,
            cols=CONSUMER_WIDTH,
            tile_count=MLEN // CONSUMER_WIDTH,
        ),
        mapping=MatrixViewMap(
            tile_pitch_rows=CONSUMER_WIDTH // BLEN,
        ),
    )
    matrix_base = program.reserve_matrix_view_scratch_v0("matrix_view_projection")
    program.vram_sub_projection_stream_k_accum_to(
        x_vram,
        0,
        w_input,
        0,
        output_placeholder,
        0,
        0,
        max_k_tiles=1,
        matrix_precision="weights",
        set_scale=True,
        hbm_element_bytes=1,
        matrix_view_descriptor=descriptor,
        matrix_view_base=matrix_base,
        matrix_view_slot=1,
    )

    gp_dst, gp_matrix, gp_zero = program.register_allocator.allocate_gp(3)
    try:
        restored_addr = program._compiler.get_vram_addr(restored.name)
        zero_addr = program._compiler.get_vram_addr(zero.name)
        program._emit(
            "\n".join(
                [
                    *load_large_int(gp_dst, restored_addr),
                    *load_large_int(gp_matrix, matrix_base),
                    *load_large_int(gp_zero, zero_addr),
                    f"V_ADD_VV.MV gp{gp_dst}, gp{gp_matrix}, gp{gp_zero}, 0, 2",
                ]
            )
            + "\n"
        )
    finally:
        program.register_allocator.free_gp([gp_dst, gp_matrix, gp_zero])

    isa = program.compile()
    validate_matrix_view_dominance(isa)
    assert "L_MVIEW_LOAD" not in isa
    assert "L_MVIEW_STORE" not in isa
    assert "V_ADD_VV.MV" in isa

    inputs = {
        "X": x_storage,
        "W": weight,
        "zero": torch.zeros(BLEN, N),
    }
    create_sim_env(
        inputs,
        isa,
        {"original_output": golden},
        [0.0] * 10,
        build_dir=str(build_dir),
    )
    hbm_addrs = {name: program._compiler.get_hbm_layout(name).hbm_base_addr for name in inputs}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="matrix_view_projection",
        data=None,
        specified_data_order=["X", "W"],
        build_path=build_dir,
        input_tensors=inputs,
        hbm_addrs=hbm_addrs,
        compiler_root=COMPILER_ROOT,
    )

    output_addr = program._compiler.get_vram_addr(restored.name)
    (build_dir / "comparison_params.json").write_text(
        json.dumps(
            {
                "start_row_idx": output_addr // MLEN,
                "num_rows": ROWS,
                "num_batches": ROWS,
                "elements_per_batch": N,
                "row_dim": MLEN,
            },
            indent=2,
        )
    )
    (build_dir / "generated_asm_code.asm").write_text(isa)

    dump_names = (
        "mram_dump.bin",
        "vram_dump.bin",
        "fpsram_dump.bin",
        "intsram_dump.bin",
    )
    dump_paths = [build_dir / name for name in dump_names]
    try:
        metrics = run_and_assert(
            build_dir,
            "matrix-view projection",
            mlen=MLEN,
            blen=BLEN,
            vlen=MLEN,
            stage_profile=True,
        )
        counters = metrics["matrix_view_packet_counters"]
        projection_fragments = N // BLEN
        weight_read_packets = projection_fragments * K
        producer_write_packets = projection_fragments
        consumer_read_packets = 1
        expected_packets = weight_read_packets + producer_write_packets + consumer_read_packets
        expected_values = weight_read_packets * MLEN + 2 * N
        expected_bank_words = weight_read_packets * (MLEN // BLEN) + producer_write_packets + N // BLEN
        assert counters == {
            # The physical counter includes ordinary M_MM weight-row reads,
            # direct affine accumulator writes, and the restored consumer read.
            "packets": expected_packets,
            "values": expected_values,
            "bank_words": expected_bank_words,
            "service_cycles": expected_packets,
            "ideal_cycles": expected_packets,
            "bank_stall_cycles": 0,
        }
        (build_dir / "connected_result.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    finally:
        for dump_path in dump_paths:
            dump_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
