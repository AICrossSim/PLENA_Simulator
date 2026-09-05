"""Execute controlled A/B/D kernels with private request arenas in one Rust run.

A/B are new packed ordinary-VV controls (explicit/static-reused addresses), not
replays of the historical analytic Arlo census. BF16 rounding and coefficient
expansion are reported. A timing ratio is qualified only if the common numeric
budget also passes. No full-model or weight execution is claimed.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil

import torch

from transactional_emulator.testbench.aten.matrix_lcompute_recurrence_test import (
    AssemblyToBinary,
    COMPILER_ROOT,
    KIMI_KDA,
    NEMOTRON_MAMBA,
    SEED,
    MatrixSramPoint,
    RecurrenceKind,
    RecurrenceLayout,
    _assert_close,
    _bf16,
    _bf16_bytes,
    _kda_inputs,
    _kda_packet_values,
    _kda_reference,
    _mamba_inputs,
    _mamba_packet_values,
    _mamba_reference,
    _pack_state_hbm,
    _read_bf16,
    _round_up,
    _setting_override,
    _state_seed,
    _unpack_state_hbm,
    _write_packet,
    _write_settings,
    build_recurrence_field_manifest,
    build_recurrence_working_set,
    lower_matrix_recurrence,
    run_emulator,
)
from compiler.aten.plena.prepared_vector_recurrence import (
    PreparedVectorGroup,
    lower_prepared_vector_recurrence,
)


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class Arena:
    def __init__(self):
        self.image = bytearray()
        self.writable: list[tuple[int, int]] = []

    def reserve(self, size: int, *, writable: bool = False) -> int:
        # Guard every allocation, including request boundaries and field tails.
        self.image.extend(bytes([0xA5]) * 64)
        base = len(self.image)
        self.image.extend(bytes(_round_up(size, 64)))
        if writable:
            self.writable.append((base, base + size))
        return base

    def tensor(self, value: torch.Tensor, *, writable: bool = False) -> int:
        data = _bf16_bytes(value)
        base = self.reserve(len(data), writable=writable)
        self.image[base : base + len(data)] = data
        return base

    def check_immutable(self, post: bytes):
        normalized = bytearray(post[: len(self.image)])
        if len(normalized) != len(self.image):
            raise AssertionError("truncated HBM dump")
        for start, end in self.writable:
            normalized[start:end] = self.image[start:end]
        if normalized != self.image:
            first = next(i for i, (a, b) in enumerate(zip(normalized, self.image)) if a != b)
            raise AssertionError(f"write escaped state/output arenas at byte {first}")


def ordinary_reference(state, operands, kind):
    """Independent reference for the BF16 boundary after each ordinary VV op."""
    if kind is RecurrenceKind.MAMBA:
        scratch = _bf16(operands["dt"][:, None] * operands["x"])
        state = _bf16(_bf16(operands["a"][:, :, None] * state) + _bf16(operands["b"][:, :, None] * scratch[:, None, :]))
        output = torch.zeros_like(operands["x"])
        for row in range(state.shape[1]):
            output = _bf16(output + _bf16(state[:, row] * operands["c"][:, row, None]))
        return _bf16(output + _bf16(operands["d"][:, None] * operands["x"])), state
    state = _bf16(operands["decay"][:, :, None] * state)
    prediction = torch.zeros_like(operands["value"])
    for row in range(state.shape[1]):
        prediction = _bf16(prediction + _bf16(state[:, row] * operands["key"][:, row, None]))
    error = _bf16(operands["beta"][:, None] * _bf16(operands["value"] - prediction))
    state = _bf16(state + _bf16(operands["key"][:, :, None] * error[:, None, :]))
    output = torch.zeros_like(error)
    for row in range(state.shape[1]):
        output = _bf16(output + _bf16(state[:, row] * operands["query"][:, row, None]))
    return output, state


def pack_vector_state(state, group_heads):
    heads, rows, width = state.shape
    return state.reshape(heads // group_heads, group_heads, rows, width).permute(0, 2, 1, 3).contiguous()


def unpack_vector_state(post, base, spec, group_heads):
    return (
        _read_bf16(post, base, spec.heads * spec.recurrence_rows * spec.row_elements)
        .reshape(spec.heads // group_heads, spec.recurrence_rows, group_heads, spec.row_elements)
        .permute(0, 2, 1, 3)
        .reshape(spec.heads, spec.recurrence_rows, spec.row_elements)
        .contiguous()
    )


def vector_fields(arena, spec, operands, state_base, mlen):
    group_heads = mlen // spec.row_elements
    groups = []
    rows = spec.recurrence_rows
    for group in range(spec.heads // group_heads):
        first, last = group * group_heads, (group + 1) * group_heads
        fields = {}
        for name, value in operands.items():
            selected = value[first:last]
            if name in ("a", "b", "c", "decay", "key", "query"):
                selected = selected.T[:, :, None].expand(rows, group_heads, spec.row_elements)
            elif name not in ("x", "value"):
                selected = selected[:, None].expand(group_heads, spec.row_elements)
            fields[name] = arena.tensor(selected.contiguous())
        fields["zero"] = arena.tensor(torch.zeros(mlen))
        fields["output"] = arena.tensor(torch.zeros(mlen), writable=True)
        groups.append(PreparedVectorGroup(state_base + group * rows * mlen * 2, fields))
    return tuple(groups)


@contextmanager
def unified_timing():
    previous = os.environ.get("PLENA_UNIFIED_SERIAL_TIMING")
    os.environ["PLENA_UNIFIED_SERIAL_TIMING"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("PLENA_UNIFIED_SERIAL_TIMING", None)
        else:
            os.environ["PLENA_UNIFIED_SERIAL_TIMING"] = previous


def run_variant(spec, variant, batch, tokens, output_dir, *, keep_build=False):
    point = MatrixSramPoint()
    working = build_recurrence_working_set(spec, layout=RecurrenceLayout.AFFINE, point=point)
    group_heads = point.mlen // spec.row_elements
    arena = Arena()
    states = [_state_seed(spec, SEED + 7919 * request) for request in range(batch)]
    input_state_hashes = [digest(_bf16_bytes(state)) for state in states]
    common_states = [state.clone() for state in states]
    bases = []
    for state in states:
        payload = (
            _pack_state_hbm(state, working) if variant == "D" else _bf16_bytes(pack_vector_state(state, group_heads))
        )
        base = arena.reserve(len(payload), writable=True)
        arena.image[base : base + len(payload)] = payload
        bases.append(base)
    assemblies, records, operand_hashes = [], [], []
    input_fn = _mamba_inputs if spec.kind is RecurrenceKind.MAMBA else _kda_inputs
    reference = _mamba_reference if spec.kind is RecurrenceKind.MAMBA else _kda_reference
    for token in range(tokens):
        for request in range(batch):
            operands = input_fn(token, SEED + 7919 * request)
            operand_hashes.append({name: digest(_bf16_bytes(value)) for name, value in operands.items()})
            common_output, common_states[request] = reference(common_states[request], operands)
            if variant == "D":
                # Reserve a complete ABI packet range, including its declared padding.
                template = build_recurrence_field_manifest(working, field_hbm_base=0)
                field_base = arena.reserve(template.end)
                manifest = build_recurrence_field_manifest(working, field_hbm_base=field_base)
                packet_values = _mamba_packet_values if spec.kind is RecurrenceKind.MAMBA else _kda_packet_values
                for packet in manifest.packets:
                    _write_packet(arena.image, packet, packet_values(packet, operands, working))
                    if packet.field == "output_result":
                        arena.writable.append(
                            (packet.hbm_byte_offset, packet.hbm_byte_offset + packet.transfer_values * 2)
                        )
                assemblies.append(
                    lower_matrix_recurrence(
                        spec,
                        layout=RecurrenceLayout.AFFINE,
                        point=point,
                        state_hbm_base=bases[request],
                        field_hbm_base=field_base,
                    )
                )
                output_offsets = [
                    (
                        manifest.packet("output_result", group=g).hbm_byte_offset,
                        manifest.packet("output_result", group=g).logical_values,
                    )
                    for g in range(working.groups)
                ]
                expected, states[request] = reference(states[request], operands)
            else:
                groups = vector_fields(arena, spec, operands, bases[request], point.mlen)
                assemblies.append(
                    lower_prepared_vector_recurrence(
                        spec,
                        groups,
                        mlen=point.mlen,
                        static_address_reuse=variant == "B",
                    )
                )
                output_offsets = [(g.fields["output"], point.mlen) for g in groups]
                expected, states[request] = ordinary_reference(states[request], operands, spec.kind)
            records.append((token, request, output_offsets, expected, common_output))
    program = "\n".join(assemblies)
    case = f"{spec.name}_{variant}_b{batch}_t{tokens}"
    build = output_dir / case
    build.mkdir(parents=True, exist_ok=True)
    asm = build / "generated_asm_code.asm"
    asm.write_text(program)
    machine = build / "generated_machine_code.mem"
    AssemblyToBinary(
        str(COMPILER_ROOT / "doc/operation.svh"), str(COMPILER_ROOT / "doc/configuration.svh")
    ).generate_binary(str(asm), str(machine))
    (build / "hbm_for_behave_sim.bin").write_bytes(arena.image)
    (build / "fp_sram.bin").write_bytes(bytes(64))
    (build / "int_sram.bin").write_bytes(bytes(64))
    settings = _write_settings(build, point)
    with _setting_override(settings), unified_timing():
        metrics = run_emulator(
            build,
            hbm_size=_round_up(len(arena.image), 64),
            threads=1,
            timing_model="serial",
            dump_cwd=build,
            dump_hbm=True,
        )
    post = (build / "hbm_dump.bin").read_bytes()
    arena.check_immutable(post)
    output_errors, state_errors, actual_hashes, common_failures = [], [], [], []
    for token, request, offsets, expected, common in records:
        actual = torch.cat([_read_bf16(post, offset, count) for offset, count in offsets]).reshape(
            spec.heads, spec.row_elements
        )
        _assert_close(f"{case} token{token} request{request} own reference", actual, expected, exact=True)
        actual_hashes.append(digest(_bf16_bytes(actual)))
        try:
            error = _assert_close("common recurrence output", actual, common)
        except AssertionError as failure:
            common_failures.append(str(failure))
            error = {
                "relative_l2": float(
                    torch.linalg.vector_norm(actual - common) / torch.linalg.vector_norm(common).clamp_min(1e-12)
                ),
                "max_abs": float((actual - common).abs().max()),
            }
        output_errors.append(error)
    final_hashes = []
    for request, base in enumerate(bases):
        actual = (
            _unpack_state_hbm(post[base : base + spec.state_bytes_per_layer], working)
            if variant == "D"
            else unpack_vector_state(post, base, spec, group_heads)
        )
        _assert_close(f"{case} request{request} final state", actual, states[request], exact=True)
        final_hashes.append(digest(_bf16_bytes(actual)))
        try:
            error = _assert_close("common recurrence state", actual, common_states[request])
        except AssertionError as failure:
            common_failures.append(str(failure))
            error = {
                "relative_l2": float(
                    torch.linalg.vector_norm(actual - common_states[request])
                    / torch.linalg.vector_norm(common_states[request]).clamp_min(1e-12)
                ),
                "max_abs": float((actual - common_states[request]).abs().max()),
            }
        state_errors.append(error)
    timing = json.loads((build / "execution_timing.json").read_text())
    result = {
        "schema_version": 1,
        "model": spec.name,
        "variant": variant,
        "batch": batch,
        "tokens": tokens,
        "baseline_scope": "new controlled packed VV A/B; not historical analytic original/Arlo replay",
        "precision": "BF16 state and prepared coefficients; no weights in this recurrence kernel",
        "rounding": "local FP32 reduction then BF16" if variant == "D" else "BF16 after every VV operation",
        "coefficient_storage": "compact descriptor fields" if variant == "D" else "explicit lane expansion in HBM",
        "initial_state_sha256_by_request": input_state_hashes,
        "operand_sha256": operand_hashes,
        "assembly_sha256": digest(program.encode()),
        "machine_code_sha256": digest(machine.read_bytes()),
        "input_hbm_sha256": digest(arena.image),
        "input_hbm_bytes": len(arena.image),
        "output_sha256": actual_hashes,
        "state_sha256": final_hashes,
        "state_arenas": [
            {"request": r, "begin": base, "end": base + spec.state_bytes_per_layer} for r, base in enumerate(bases)
        ],
        "private_state_and_immutable_guards_passed": True,
        "own_rounding_reference_exact": True,
        "common_relative_l2_limit": 1e-2,
        "common_budget_passed": not common_failures,
        "common_output_errors": output_errors,
        "common_state_errors": state_errors,
        "common_budget_failures": common_failures,
        "timing": timing,
        "physical_hbm_bytes_read": metrics["hbm_bytes_read"],
        "physical_hbm_bytes_written": metrics["hbm_bytes_written"],
        "matrix_view_packet_counters": metrics.get("matrix_view_packet_counters", {}),
        "l_tile_exec_count": program.count("L_TILE_EXEC"),
    }
    (output_dir / f"{case}.json").write_text(json.dumps(result, indent=2) + "\n")
    if not keep_build:
        shutil.rmtree(build)
    return result


def write_execution_tables(results, output_dir):
    """Publish only qualified speedup cells; preserve failed controls as diagnostics."""
    components = []
    for result in results:
        timing = result["timing"]
        period = timing["period_picos"]
        counters = timing["counters"]
        row = {
            "model": result["model"],
            "variant": result["variant"],
            "batch": result["batch"],
            "tokens": result["tokens"],
            "issued_instructions": counters["issued_instructions"],
            "issue_cycles": counters["issue_cycles"],
            "bank_service_cycles": counters["bank_service_cycles"],
            "arithmetic_cycles": counters["arithmetic_cycles"],
            "scalar_control_cycles": timing["scalar_and_control_cycles"],
            "dma_memory_wait_cycles": timing["dma_and_memory_wait_picos"] / period,
            "total_cycles": timing["total_picos"] / period,
            "physical_hbm_bytes_read": result.get("physical_hbm_bytes_read"),
            "physical_hbm_bytes_written": result.get("physical_hbm_bytes_written"),
            "common_budget_passed": result["common_budget_passed"],
            "max_output_relative_l2": max(e["relative_l2"] for e in result["common_output_errors"]),
        }
        assert row["total_cycles"] == sum(
            row[key]
            for key in (
                "issue_cycles",
                "bank_service_cycles",
                "arithmetic_cycles",
                "scalar_control_cycles",
                "dma_memory_wait_cycles",
            )
        )
        components.append(row)
    comparisons = []
    for batch in sorted({r["batch"] for r in results}):
        variants = {r["variant"]: r for r in results if r["batch"] == batch}
        if set(variants) != {"A", "B", "D"}:
            continue
        qualified = all(r["common_budget_passed"] for r in variants.values())
        comparisons.append(
            {
                "model": variants["D"]["model"],
                "batch": batch,
                "tokens": variants["D"]["tokens"],
                "common_numeric_budget_passed": qualified,
                "D_speedup_vs_B_qualified": (
                    variants["B"]["timing"]["total_picos"] / variants["D"]["timing"]["total_picos"]
                    if qualified
                    else None
                ),
                "scope": "controlled packed VV explicit/static address reuse vs L_TILE; not historical Arlo replay",
            }
        )
    for name, rows in (("timing_components.csv", components), ("qualified_comparison.csv", comparisons)):
        if rows:
            with (output_dir / name).open("w", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    (output_dir / "README.md").write_text("""# Controlled BF16 recurrence execution

Every A/B/D row executes Compiler machine words in Rust. A and B are newly
materialized packed ordinary-VV controls, with explicit addresses and static
address reuse respectively; they are **not** the historical analytic
original/Arlo instruction census. The tests use the same logical BF16 initial
state and prepared scalar values, and the same SRAM topology for every variant.
No checkpoint weights, projection, routing or full model are executed here.

A/B explicitly expand coefficients into HBM lanes; D transfers compact fields.
The measured DMA difference is part of this dataflow comparison, not an isolated
bank-only gain. A/B round after each VV instruction. D performs local FP32
reductions before writing BF16 state/output. Each sequence must exactly match
its independent rounding reference; a separate common recurrence budget is
relative L2 <= 1%, plus the declared element bound. A blank speedup cell means
that common budget failed and no speedup claim is qualified.

Each request has a private persistent HBM state arena. The whole batch runs
in a single Rust invocation, token-major, without host state updates. All
non-output/non-state bytes, including guard regions and input fields, must be
unchanged. Distinct request inputs and per-request reference results also catch
writes to the wrong live state arena.

`timing_components.csv` is a dependency-safe serial model: one issue cycle per
instruction; ordinary VV rows pay two single-port bank reads and one write;
viewed operations pay actual packet bank service; both execute their arithmetic
latency code. DMA/memory wait is residual dispatch waiting, **not** an independent
HBM-engine busy counter. Components add to total elapsed virtual time. No ideal
overlap credit, RTL frequency or PPA is claimed. The opt-in timing contract
rejects opcodes outside the implemented recurrence subset.
""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--tokens", type=int, default=2)
    parser.add_argument("--model", choices=["mamba", "kda"], default="mamba")
    parser.add_argument("--variants", nargs="+", choices=["A", "B", "D"], default=["A", "B", "D"])
    parser.add_argument("--keep-build", action="store_true")
    args = parser.parse_args()
    if args.tokens < 1 or any(batch not in (1, 2, 4, 8, 16) for batch in args.batches):
        parser.error("tokens must be positive; supported batches: 1,2,4,8,16")
    torch.set_num_threads(1)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    spec = NEMOTRON_MAMBA if args.model == "mamba" else KIMI_KDA
    for batch in args.batches:
        for variant in args.variants:
            result = run_variant(spec, variant, batch, args.tokens, args.output_dir, keep_build=args.keep_build)
            peers = [r for r in results if r["batch"] == batch]
            for peer in peers:
                assert peer["initial_state_sha256_by_request"] == result["initial_state_sha256_by_request"]
                assert peer["operand_sha256"] == result["operand_sha256"]
                if peer["variant"] in ("A", "B") and variant in ("A", "B"):
                    assert peer["output_sha256"] == result["output_sha256"]
                    assert peer["state_sha256"] == result["state_sha256"]
            results.append(result)
            (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(
                f"{spec.name} {variant} B{batch}: exact own-reference; common budget={result['common_budget_passed']}",
                flush=True,
            )

    write_execution_tables(results, args.output_dir)


if __name__ == "__main__":
    main()
