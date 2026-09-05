"""Compiler-to-Rust Matrix-SRAM projection and four-token recurrence checks.

The projection must write the consumer view directly. Prepared Mamba/KDA
programs use deterministic BF16 inputs at the official state sizes, execute
four tokens in Rust, and read every output plus the final state from HBM.
Phased layout requires exact BF16 results; fixed layout has a 1% relative-L2
budget. These are mechanism tests, with no checkpoint or full-model execution.

The CLI first runs the small unittest guards, then the integration checks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import unittest

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILER_ROOT = REPO_ROOT / "PLENA_Compiler"
for path in (REPO_ROOT / "PLENA_Tools", COMPILER_ROOT, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from compiler.asm_templates._imm import load_large_int  # noqa: E402
from compiler.assembler.assembly_to_binary import AssemblyToBinary  # noqa: E402
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena.isa_matrix_view import (  # noqa: E402
    MatrixViewDescriptor,
    MatrixViewMap,
    MatrixViewShape,
    validate_matrix_view_dominance,
)
from compiler.aten.plena.program_lcompute import (  # noqa: E402
    KIMI_KDA,
    NEMOTRON_MAMBA,
    MatrixRecurrenceSpec,
    MatrixSramPoint,
    RecurrenceKind,
    RecurrenceLayout,
    build_recurrence_field_manifest,
    build_recurrence_working_set,
    lower_matrix_recurrence,
    validate_recurrence_output_stores,
)
from transactional_emulator.testbench.aten import _matrix_lcompute as kit  # noqa: E402
from transactional_emulator.testbench.aten.golden import golden_linear  # noqa: E402
from transactional_emulator.testbench.emulator_runner import (  # noqa: E402
    _parse_matrix_view_packet_counters,
    run_and_assert,
    run_emulator,
)
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim  # noqa: E402
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402

MLEN, BLEN = 64, 4
ROWS, K, N = 1, 64, 64
CONSUMER_WIDTH = 8
TOKENS = 4


def run_projection(build_dir: Path) -> dict:
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
        return {"case": "projection", "matrix_view_packet_counters": counters}
    finally:
        for dump_path in dump_paths:
            dump_path.unlink(missing_ok=True)


def run_recurrence(spec: MatrixRecurrenceSpec, layout: RecurrenceLayout, build_dir: Path) -> dict:
    point = MatrixSramPoint()
    working_set = build_recurrence_working_set(spec, layout=layout, point=point)
    exact = layout is RecurrenceLayout.AFFINE
    initial_state = kit._state_seed(spec)
    expected_state = initial_state.clone()
    make_inputs, reference, packet_values = (
        (kit._mamba_inputs, kit._mamba_reference, kit._mamba_packet_values)
        if spec.kind is RecurrenceKind.MAMBA
        else (kit._kda_inputs, kit._kda_reference, kit._kda_packet_values)
    )
    operands_by_token = tuple(make_inputs(token) for token in range(TOKENS))
    expected_outputs, manifests, assemblies = [], [], []
    field_base = kit._round_up(spec.state_bytes_per_layer, 64)
    for operands in operands_by_token:
        manifest = build_recurrence_field_manifest(working_set, field_hbm_base=field_base)
        assembly = lower_matrix_recurrence(
            spec,
            layout=layout,
            point=point,
            state_hbm_base=0,
            field_hbm_base=field_base,
        )
        validate_recurrence_output_stores(assembly, expected_groups=working_set.groups)
        expected_output, expected_state = reference(expected_state, operands)
        expected_outputs.append(expected_output)
        manifests.append(manifest)
        assemblies.append(assembly)
        field_base = kit._round_up(manifest.end, 64)

    program = "\n".join(assemblies)
    validate_matrix_view_dominance(program)
    build_dir.mkdir(parents=True, exist_ok=True)
    asm_path = build_dir / "generated_asm_code.asm"
    asm_path.write_text(program)
    assembler = AssemblyToBinary(
        str(COMPILER_ROOT / "doc/operation.svh"),
        str(COMPILER_ROOT / "doc/configuration.svh"),
    )
    assembler.generate_binary(str(asm_path), str(build_dir / "generated_machine_code.mem"))

    image = bytearray(field_base)
    image[: spec.state_bytes_per_layer] = kit._pack_state_hbm(initial_state, working_set)
    for operands, manifest in zip(operands_by_token, manifests, strict=True):
        for packet in manifest.packets:
            kit._write_packet(image, packet, packet_values(packet, operands, working_set))
    (build_dir / "hbm_for_behave_sim.bin").write_bytes(image)
    (build_dir / "fp_sram.bin").write_bytes(bytes(64))
    (build_dir / "int_sram.bin").write_bytes(bytes(64))
    settings = kit._write_settings(build_dir, point)
    with kit._setting_override(settings):
        metrics = run_emulator(
            build_dir,
            hbm_size=kit._round_up(len(image), 64),
            threads=1,
            dump_cwd=build_dir,
            dump_hbm=True,
        )

    post = (build_dir / "hbm_dump.bin").read_bytes()
    actual_state = kit._unpack_state_hbm(post, working_set)
    state_error = kit._assert_close("final state", actual_state, expected_state, exact=exact)
    output_errors = []
    for token, (manifest, expected) in enumerate(zip(manifests, expected_outputs, strict=True)):
        actual_groups = []
        for group in range(working_set.groups):
            packet = manifest.packet("output_result", group=group)
            values = kit._read_bf16(post, packet.hbm_byte_offset, packet.logical_values)
            actual_groups.append(values.reshape(working_set.group_heads, spec.row_elements))
        actual = torch.cat(actual_groups, dim=0)
        output_errors.append(kit._assert_close(f"token {token} output", actual, expected, exact=exact))
        for packet in manifest.packets:
            if packet.field != "output_result":
                start = packet.hbm_byte_offset
                end = start + packet.transfer_values * 2
                assert post[start:end] == image[start:end], f"input field overwritten: {packet.key}"

    counters = metrics["matrix_view_packet_counters"]
    assert counters["packets"] > 0
    assert counters["service_cycles"] == counters["ideal_cycles"] + counters["bank_stall_cycles"]
    if exact:
        assert counters["bank_stall_cycles"] == 0
    return {
        "case": spec.name,
        "layout": layout.value,
        "tokens": TOKENS,
        "state_error": state_error,
        "output_errors": output_errors,
        "matrix_view_packet_counters": counters,
    }


class MatrixLComputeGuards(unittest.TestCase):
    def test_head_or_lane_permutations(self):
        expected = torch.arange(256, dtype=torch.float32).reshape(4, 64)
        for axis in (0, 1):
            with self.subTest(axis=axis), self.assertRaisesRegex(AssertionError, "values mismatch"):
                kit._assert_close("permutation", expected.roll(1, dims=axis), expected)

    def test_non_finite_values(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            for actual, expected in ((torch.tensor([bad]), torch.zeros(1)), (torch.zeros(1), torch.tensor([bad]))):
                with self.subTest(bad=bad), self.assertRaisesRegex(AssertionError, "non-finite"):
                    kit._assert_close("non-finite", actual, expected)

    def test_fixed_budget_and_phased_exactness(self):
        expected = torch.linspace(-0.03, 0.03, 256)
        kit._assert_close("within budget", expected * 0.992, expected)
        with self.assertRaisesRegex(AssertionError, "l2_budget"):
            kit._assert_close("outside budget", expected * 0.988, expected)
        with self.assertRaisesRegex(AssertionError, "l2_budget"):
            kit._assert_close("near zero corruption", torch.full((256,), 2e-7), torch.zeros(256))
        kit._assert_close("identity", torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]), exact=True)
        for changed in (torch.tensor([1.0078125, 0.0]), torch.tensor([1.0, -0.0])):
            with self.subTest(changed=changed), self.assertRaisesRegex(AssertionError, "exact BF16"):
                kit._assert_close("exactness", changed, torch.tensor([1.0, 0.0]), exact=True)

    def test_ten_percent_loss(self):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            for spec, inputs, reference in (
                (NEMOTRON_MAMBA, kit._mamba_inputs, kit._mamba_reference),
                (KIMI_KDA, kit._kda_inputs, kit._kda_reference),
            ):
                state = kit._state_seed(spec)
                for token in range(TOKENS):
                    output, state = reference(state, inputs(token))
                    with (
                        self.subTest(model=spec.name, token=token),
                        self.assertRaisesRegex(AssertionError, "l2_budget"),
                    ):
                        kit._assert_close("gain loss", kit._bf16(output * 0.9), output)
        finally:
            torch.set_num_threads(previous_threads)

    def test_packet_counters(self):
        line = (
            "\x1b[32mINFO\x1b[0m Matrix-view packet counters "
            "packets=17 values=128 bank_words=32 service_cycles=19 ideal_cycles=17 bank_stall_cycles=2"
        )
        self.assertEqual(
            _parse_matrix_view_packet_counters(line),
            {
                "packets": 17,
                "values": 128,
                "bank_words": 32,
                "service_cycles": 19,
                "ideal_cycles": 17,
                "bank_stall_cycles": 2,
            },
        )
        self.assertIsNone(_parse_matrix_view_packet_counters("Matrix-view packet counters packets=17"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("projection", "mamba", "kda", "all"), default="all")
    parser.add_argument("--build-dir", type=Path, default=Path(__file__).parent / "build/matrix_lcompute")
    parser.add_argument("--keep-build", action="store_true")
    args = parser.parse_args()
    build_root = args.build_dir.resolve()
    torch.set_num_threads(1)
    guards = unittest.defaultTestLoader.loadTestsFromTestCase(MatrixLComputeGuards)
    if not unittest.TextTestRunner().run(guards).wasSuccessful():
        raise SystemExit(1)
    results = []
    if args.case in {"projection", "all"}:
        build_dir = build_root / "projection"
        with kit._setting_override(REPO_ROOT / "plena_settings.toml"):
            results.append(run_projection(build_dir))
        if not args.keep_build:
            shutil.rmtree(build_dir)
    for case, spec in (("mamba", NEMOTRON_MAMBA), ("kda", KIMI_KDA)):
        if args.case not in {case, "all"}:
            continue
        for layout in (RecurrenceLayout.FIXED, RecurrenceLayout.AFFINE):
            build_dir = build_root / case / layout.value
            result = run_recurrence(spec, layout, build_dir)
            results.append(result)
            print(json.dumps(result), flush=True)
            if not args.keep_build:
                shutil.rmtree(build_dir)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
