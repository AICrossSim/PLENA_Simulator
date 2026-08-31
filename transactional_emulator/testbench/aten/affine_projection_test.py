"""Compiler-to-Rust affine Matrix writeback numerical test.

The projection uses a K dimension larger than the configured Matrix-SRAM tile
capacity, so the compiler must retain partial sums and apply the affine layout
only at the final writeback.  A normal Vector opcode then reads the affine
tensor through the same descriptor and materialises a conventional row-major
result for comparison.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
import compiler as compiler_package

from compiler.asm_templates._imm import load_large_int
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.affine_layout import AffineLayout, LayoutKind
from compiler.aten.plena.lstream import (
    StreamBinding,
    StreamConfigField,
    emit_stream_configuration,
    stream_view_mask,
)
from transactional_emulator.testbench.aten.golden import golden_linear
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


MLEN = 64
BLEN = 4
ROWS = 4
K = 128
N = 64
COMPILER_ROOT = Path(compiler_package.__file__).resolve().parents[1]


def _emit_affine_restore(
    program: PlenaCompiler,
    source,
    target,
    layout: AffineLayout,
    *,
    fpram_one_addr: int,
) -> None:
    source_base = program._compiler.get_vram_addr(source.name)
    target_base = program._compiler.get_vram_addr(target.name)
    bound_layout = replace(layout, bank_row_base=source_base // MLEN + layout.bank_row_base)

    gp_src, gp_dst, gp_fp, gp_value = program.register_allocator.allocate_gp(4)
    fp_one = program.allocate_fp_reg(1)[0]
    binding = StreamBinding(
        slot=0,
        target_register=gp_src,
        target_is_fp=False,
        base=source_base,
        advance=MLEN,
        packet_elements=MLEN,
        storage_atom=BLEN,
    )
    try:
        program._emit(
            emit_stream_configuration(
                value_gp=gp_value,
                binding=binding,
                layout=bound_layout,
            ).render()
        )
        lines = [
            *load_large_int(gp_src, source_base),
            *load_large_int(gp_fp, fpram_one_addr),
            f"S_LD_FP f{fp_one}, gp{gp_fp}, 0",
        ]
        physical_rows, physical_cols = source.physical_shape
        for col_block in range(physical_cols // MLEN):
            for row in range(physical_rows):
                logical_offset = col_block * physical_rows * MLEN + row * MLEN
                lines.extend(load_large_int(gp_dst, target_base + logical_offset))
                lines.append(
                    f"V_MUL_VF gp{gp_dst}, gp{gp_src}, f{fp_one}, 0, "
                    f"{stream_view_mask(0)}"
                )
        lines.append(f"L_CFG gp0, gp{gp_src}, 0, {int(StreamConfigField.RESET)}")
        program._emit("\n".join(lines) + "\n")
    finally:
        program.free_fp_reg([fp_one])
        program.register_allocator.free_gp([gp_src, gp_dst, gp_fp, gp_value])


def main() -> None:
    build_dir = Path(__file__).parent / "build" / "affine_projection"
    torch.manual_seed(20260830)
    x = torch.randn(ROWS, K)
    weight = torch.randn(K, N)
    golden = golden_linear(x, weight)

    program = PlenaCompiler(mlen=MLEN, blen=BLEN, mram_tile_capacity=1)
    x_input = program.input("X", shape=(ROWS, K), physical_shape=(ROWS, K))
    w_input = program.input("W", shape=(K, N), physical_shape=(K, N))
    x_vram = program.load_batch(x_input, name="X_vram")
    layout = AffineLayout(LayoutKind.AFFINE_SKEW, 1, 1, ROWS, N, alpha=1)
    y_affine = program.linear_projection(
        x_vram,
        w_input,
        name="Y_affine",
        output_layout=layout,
    )
    y_restored = program.alloc(
        "Y_restored",
        ROWS,
        N,
        strict=False,
        physical_shape=(ROWS, N),
    )
    _emit_affine_restore(program, y_affine, y_restored, layout, fpram_one_addr=2)
    isa = program.compile()

    inputs = {"X": x, "W": weight}
    create_sim_env(
        inputs,
        isa,
        {"original_output": golden},
        [0.0, 1e-6, 1.0] + [0.0] * 7,
        build_dir=str(build_dir),
    )
    hbm_addrs = {name: program._compiler.get_hbm_layout(name).hbm_base_addr for name in inputs}
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="affine_projection",
        data=None,
        specified_data_order=["X", "W"],
        build_path=build_dir,
        input_tensors=inputs,
        hbm_addrs=hbm_addrs,
        compiler_root=COMPILER_ROOT,
    )

    output_addr = program._compiler.get_vram_addr(y_restored.name)
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

    metrics = run_and_assert(
        build_dir,
        "affine projection",
        mlen=MLEN,
        blen=BLEN,
        vlen=MLEN,
        stage_profile=True,
    )
    (build_dir / "connected_result.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
