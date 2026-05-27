"""Compare ATen linear GEMM: hardware-looped vs Python-unrolled."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any

from compiler.aten.ops.registry import Backend, OpRegistry
import compiler.aten.ops as ops
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.compare.isa_analysis import analyze_asm, load_behavior_cycle_model


MLEN = 64
BLEN = 4
REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)

DEFAULT_SHAPES = (
    (4, 256, 256),
    (64, 256, 256),
    (256, 256, 256),
    (64, 512, 128),
)


@contextlib.contextmanager
def forced_aten_unroll(enabled: bool):
    """Toggle ATEN_UNROLL env var for emission."""
    old_value = os.environ.get("ATEN_UNROLL")
    os.environ["ATEN_UNROLL"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("ATEN_UNROLL", None)
        else:
            os.environ["ATEN_UNROLL"] = old_value


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.split(r"[x, ]+", text.strip()) if part]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,K,N or MxKxN")
    return tuple(parts)  # type: ignore[return-value]


def emit_aten_linear(m: int, k: int, n: int, *, unroll_loops: bool) -> tuple[str, int]:
    """Emit ATen compiler linear ASM and return (asm, result_vram_addr)."""
    with forced_aten_unroll(unroll_loops):
        registry = OpRegistry.load()
        registry.set_backend(Backend.PLENA)

        prog = PlenaCompiler(mlen=MLEN, blen=BLEN, real_data_ratio=REAL_DATA_RATIO, unroll_loops=unroll_loops)
        x_input = prog.input("X", shape=(m, k))
        w_input = prog.input("W", shape=(k, n))
        x_batch = prog.load_batch(x_input, name="X")
        y = ops.linear(prog, x_batch, w_input)
        asm = prog.compile()
        return asm, prog.get_vram_addr(y.name)


def _write_case(
    out_dir: Path,
    *,
    shape: tuple[int, int, int],
    mode: str,
    asm: str,
    result_vram_addr: int,
    cycle_model,
) -> dict[str, Any]:
    m, k, n = shape
    case = f"aten_linear_{mode}_m{m}_k{k}_n{n}"
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    asm_path = case_dir / "generated_asm_code.asm"
    asm_path.write_text(asm)

    row = {
        "case": case,
        "mode": mode,
        "shape": f"M={m} K={k} N={n}",
        "m": m,
        "k": k,
        "n": n,
        "result_vram_addr": result_vram_addr,
        "asm_path": str(asm_path),
    }
    row.update(
        analyze_asm(
            asm,
            cycle_model=cycle_model,
            selected_opcodes=(
                "C_LOOP_START",
                "C_LOOP_END",
                "H_PREFETCH_M",
                "H_PREFETCH_V",
                "M_MM",
                "M_MM_WO",
                "V_ADD_VV",
                "S_ADDI_INT",
                "S_LUI_INT",
            ),
        )
    )
    (case_dir / "stats.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def rows_by_shape_mode(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, dict[str, Any]]]:
    by_shape: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (row["m"], row["k"], row["n"])
        by_shape.setdefault(key, {})[row["mode"]] = row
    return by_shape


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}x"


def assert_step2_invariants(rows: list[dict[str, Any]]) -> None:
    """Assert looped path reduces static code and emits hardware loops."""
    failures: list[str] = []
    for shape, pair in rows_by_shape_mode(rows).items():
        looped = pair["looped"]
        unrolled = pair["unrolled"]
        label = f"M={shape[0]} K={shape[1]} N={shape[2]}"
        if looped["loop_start_lines"] <= 0:
            failures.append(f"{label}: looped path emitted no C_LOOP_START")
        if looped["loop_start_lines"] <= unrolled["loop_start_lines"]:
            failures.append(
                f"{label}: looped path did not add GEMM loops "
                f"(looped={looped['loop_start_lines']}, unrolled={unrolled['loop_start_lines']})"
            )
        if looped["source_lines"] >= unrolled["source_lines"]:
            failures.append(
                f"{label}: looped source lines {looped['source_lines']} >= unrolled {unrolled['source_lines']}"
            )
        for opcode in ("M_MM", "M_MM_WO"):
            looped_count = looped["selected_opcodes_dynamic"].get(opcode, 0)
            unrolled_count = unrolled["selected_opcodes_dynamic"].get(opcode, 0)
            if looped_count != unrolled_count:
                failures.append(
                    f"{label}: dynamic {opcode} mismatch (looped={looped_count}, unrolled={unrolled_count})"
                )
    if failures:
        raise AssertionError("\n".join(failures))


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_shape = rows_by_shape_mode(rows)
    comparisons: list[dict[str, Any]] = []

    table = [
        "| Shape | Looped source | Unrolled source | Source reduction | Looped dynamic | Unrolled dynamic | Dynamic ratio | Looped cycles | Unrolled cycles | Cycle ratio | C_LOOP_START looped/unrolled | Dynamic M_MM |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for shape in sorted(by_shape):
        looped = by_shape[shape]["looped"]
        unrolled = by_shape[shape]["unrolled"]
        source_reduction = _ratio(unrolled["source_lines"], looped["source_lines"])
        dynamic_ratio = _ratio(looped["dynamic_instruction_count"], unrolled["dynamic_instruction_count"])
        cycle_ratio = _ratio(looped["estimated_cycles"], unrolled["estimated_cycles"])
        comparison = {
            "shape": {"m": shape[0], "k": shape[1], "n": shape[2]},
            "source_reduction_unrolled_over_looped": source_reduction,
            "dynamic_ratio_looped_over_unrolled": dynamic_ratio,
            "cycle_ratio_looped_over_unrolled": cycle_ratio,
            "looped_case": looped["case"],
            "unrolled_case": unrolled["case"],
        }
        comparisons.append(comparison)
        table.append(
            f"| `M={shape[0]} K={shape[1]} N={shape[2]}` | "
            f"{looped['source_lines']} | {unrolled['source_lines']} | {_fmt_ratio(source_reduction)} | "
            f"{looped['dynamic_instruction_count']} | {unrolled['dynamic_instruction_count']} | "
            f"{_fmt_ratio(dynamic_ratio)} | {looped['estimated_cycles']} | {unrolled['estimated_cycles']} | "
            f"{_fmt_ratio(cycle_ratio)} | {looped['loop_start_lines']}/{unrolled['loop_start_lines']} | "
            f"{looped['selected_opcodes_dynamic'].get('M_MM', 0)} |"
        )

    table.extend(
        [
            "",
            "Notes:",
            "",
            f"- Cycle model: {rows[0]['cycle_model']}.",
            "- `source_reduction` is `unrolled source lines / looped source lines`; larger is better for instruction memory.",
            "- `dynamic_ratio` and `cycle_ratio` are `looped / unrolled`; values near 1 mean the looped path preserves runtime work.",
            "- The unrolled full program can still contain non-GEMM loops from preload or helper code; compare the looped/unrolled C_LOOP_START pair, not zero versus nonzero.",
            "- `Dynamic M_MM` is equal between modes for each shape, which checks that the matrix compute work is preserved.",
            "- The harness forces `ATEN_UNROLL=0` for looped rows and `ATEN_UNROLL=1` for unrolled rows while emitting, then restores the caller environment.",
            "- This is a static ISA estimate. It does not model asynchronous HBM transfer wait time beyond the explicit simulator instruction costs.",
        ]
    )

    (out_dir / "summary.md").write_text("\n".join(table) + "\n")
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows, "comparisons": comparisons}, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        dest="shapes",
        type=parse_shape,
        action="append",
        help="Shape as M,K,N or MxKxN. Can be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "build" / "gemm_looping_compare",
        help="Directory for emitted ASM and summary files.",
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=None,
        help="Path to plena_settings.toml. Defaults to the repo-root settings file.",
    )
    parser.add_argument(
        "--dc-en",
        type=int,
        choices=(0, 1),
        default=1,
        help="Use dc_lib_en or dc_lib_dis latency values from BEHAVIOR.LATENCY. Defaults to 1.",
    )
    parser.add_argument(
        "--latency-profile",
        default=None,
        help="Optional named BEHAVIOR.LATENCY profile to use when --dc-en=0.",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Write artifacts even if looped/unrolled static-code invariants fail.",
    )
    args = parser.parse_args()

    shapes = args.shapes if args.shapes else list(DEFAULT_SHAPES)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cycle_model = load_behavior_cycle_model(
        settings_path=args.settings_path,
        dc_en=args.dc_en,
        latency_profile=args.latency_profile,
    )

    rows: list[dict[str, Any]] = []
    for shape in shapes:
        for mode, unroll_loops in (("looped", False), ("unrolled", True)):
            asm, result_vram_addr = emit_aten_linear(*shape, unroll_loops=unroll_loops)
            rows.append(
                _write_case(
                    args.out_dir,
                    shape=shape,
                    mode=mode,
                    asm=asm,
                    result_vram_addr=result_vram_addr,
                    cycle_model=cycle_model,
                )
            )

    write_summary(args.out_dir, rows)
    if not args.no_assert:
        assert_step2_invariants(rows)

    print(f"Wrote GEMM looping comparison artifacts to: {args.out_dir}")
    print()
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
