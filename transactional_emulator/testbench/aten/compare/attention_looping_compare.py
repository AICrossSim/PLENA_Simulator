"""Compare ATen MHA attention: looped vs unrolled helpers."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
from pathlib import Path
from typing import Any

from compiler.aten.ops.registry import Backend, OpRegistry
import compiler.aten.ops as ops
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.compare.isa_analysis import analyze_asm, load_behavior_cycle_model


MLEN = 64
BLEN = 4
REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)

SELECTED_OPCODES = (
    "C_LOOP_START",
    "C_LOOP_END",
    "H_PREFETCH_M",
    "H_PREFETCH_V",
    "M_TMM",
    "M_MM",
    "M_MM_WO",
    "V_ADD_VV",
    "V_SUB_VF",
    "V_MUL_VF",
    "V_EXP_V",
    "V_RED_MAX",
    "V_RED_SUM",
    "S_ADDI_INT",
    "S_EXP_FP",
    "S_RECI_FP",
    "S_LD_FP",
    "S_ST_FP",
)

PRESERVED_DYNAMIC_OPS = (
    "M_TMM",
    "M_MM",
    "M_MM_WO",
    "V_ADD_VV",
    "V_SUB_VF",
    "V_MUL_VF",
    "V_EXP_V",
    "V_RED_MAX",
    "V_RED_SUM",
    "S_EXP_FP",
    "S_RECI_FP",
    "S_LD_FP",
    "S_ST_FP",
    "H_PREFETCH_M",
    "H_PREFETCH_V",
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


def emit_aten_mha_one_head(
    *,
    seq_len: int,
    head_dim: int,
    causal_mask: bool,
    unroll_attention: bool,
) -> str:
    """Emit ATen per-head MHA path."""
    with forced_aten_unroll(False):
        registry = OpRegistry.load()
        registry.set_backend(Backend.PLENA)

        prog = PlenaCompiler(mlen=MLEN, blen=BLEN, real_data_ratio=REAL_DATA_RATIO, unroll_loops=False)
        prog.unroll_attention = unroll_attention

        q_input = prog.input("Q", shape=(seq_len, head_dim), prestaged_vram_addr=0)
        Q = prog.load_batch(q_input, name="Q")

        if causal_mask:
            mask_input = prog.input("CAUSAL_MASK", shape=(MLEN, MLEN), prestaged_vram_addr=seq_len * head_dim)
            mask = prog.load_batch(mask_input, name="CAUSAL_MASK")
        else:
            mask = None

        k_input = prog.input("K", shape=(seq_len, head_dim))
        v_input = prog.input("V", shape=(seq_len, head_dim))

        ops.flash_attention(
            prog,
            Q,
            k_input,
            v_input,
            1.0 / math.sqrt(head_dim),
            causal_mask=mask,
        )
        return prog.compile()


def _write_case(
    out_dir: Path,
    *,
    mode: str,
    asm: str,
    seq_len: int,
    head_dim: int,
    causal_mask: bool,
    cycle_model,
) -> dict[str, Any]:
    case = f"aten_mha_{mode}_seq{seq_len}_d{head_dim}"
    case_dir = out_dir / case
    case_dir.mkdir(parents=True, exist_ok=True)
    asm_path = case_dir / "generated_asm_code.asm"
    asm_path.write_text(asm)

    row = {
        "case": case,
        "mode": mode,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "causal_mask": causal_mask,
        "asm_path": str(asm_path),
    }
    row.update(analyze_asm(asm, cycle_model=cycle_model, selected_opcodes=SELECTED_OPCODES))
    (case_dir / "stats.json").write_text(json.dumps(row, indent=2) + "\n")
    return row


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _fmt_ratio(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}x"


def assert_attention_looping_invariants(rows: list[dict[str, Any]]) -> None:
    by_mode = {row["mode"]: row for row in rows}
    looped = by_mode["looped"]
    unrolled = by_mode["unrolled"]

    failures: list[str] = []
    if looped["loop_start_lines"] <= unrolled["loop_start_lines"]:
        failures.append(
            "looped attention did not add hardware loops "
            f"(looped={looped['loop_start_lines']}, unrolled={unrolled['loop_start_lines']})"
        )
    if looped["source_lines"] >= unrolled["source_lines"]:
        failures.append(f"looped source lines {looped['source_lines']} >= unrolled {unrolled['source_lines']}")
    if looped["static_instruction_lines"] >= unrolled["static_instruction_lines"]:
        failures.append(
            f"looped static instr {looped['static_instruction_lines']} >= "
            f"unrolled {unrolled['static_instruction_lines']}"
        )

    looped_ops = looped["selected_opcodes_dynamic"]
    unrolled_ops = unrolled["selected_opcodes_dynamic"]
    for opcode in PRESERVED_DYNAMIC_OPS:
        if looped_ops.get(opcode, 0) != unrolled_ops.get(opcode, 0):
            failures.append(
                f"dynamic {opcode} mismatch "
                f"(looped={looped_ops.get(opcode, 0)}, unrolled={unrolled_ops.get(opcode, 0)})"
            )

    if failures:
        raise AssertionError("\n".join(failures))


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    by_mode = {row["mode"]: row for row in rows}
    looped = by_mode["looped"]
    unrolled = by_mode["unrolled"]

    source_reduction = _ratio(unrolled["source_lines"], looped["source_lines"])
    static_reduction = _ratio(unrolled["static_instruction_lines"], looped["static_instruction_lines"])
    dynamic_ratio = _ratio(looped["dynamic_instruction_count"], unrolled["dynamic_instruction_count"])
    cycle_ratio = _ratio(looped["estimated_cycles"], unrolled["estimated_cycles"])

    summary = {
        "rows": rows,
        "comparison": {
            "source_reduction_unrolled_over_looped": source_reduction,
            "static_reduction_unrolled_over_looped": static_reduction,
            "dynamic_ratio_looped_over_unrolled": dynamic_ratio,
            "cycle_ratio_looped_over_unrolled": cycle_ratio,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    table = [
        "| Mode | Source lines | Static instr | Dynamic instr | Est cycles | Est ms @1GHz | C_LOOP_START | Dynamic M_TMM | Dynamic M_MM | Dynamic V_EXP_V | Dynamic V_MUL_VF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        ops_dynamic = row["selected_opcodes_dynamic"]
        table.append(
            f"| `{row['mode']}` | {row['source_lines']} | {row['static_instruction_lines']} | "
            f"{row['dynamic_instruction_count']} | {row['estimated_cycles']} | "
            f"{row['estimated_ms_at_1ghz']:.6f} | {row['loop_start_lines']} | "
            f"{ops_dynamic.get('M_TMM', 0)} | {ops_dynamic.get('M_MM', 0)} | "
            f"{ops_dynamic.get('V_EXP_V', 0)} | {ops_dynamic.get('V_MUL_VF', 0)} |"
        )

    table.extend(
        [
            "",
            "Ratios:",
            "",
            f"- Source reduction, unrolled / looped: {_fmt_ratio(source_reduction)}.",
            f"- Static instruction reduction, unrolled / looped: {_fmt_ratio(static_reduction)}.",
            f"- Dynamic instruction ratio, looped / unrolled: {_fmt_ratio(dynamic_ratio)}.",
            f"- Estimated cycle ratio, looped / unrolled: {_fmt_ratio(cycle_ratio)}.",
            "",
            "Notes:",
            "",
            f"- Cycle model: {rows[0]['cycle_model']}.",
            "- Both rows force generic ATen GEMM lowering to `ATEN_UNROLL=0`; only attention helper unrolling changes.",
            "- The looped row is expected to carry more dynamic scalar/control overhead. The goal is lower instruction memory while preserving dynamic matrix/vector work.",
            "- The harness asserts that the dynamic counts for matrix ops, vector softmax/scaling ops, and scalar FP loads/stores match between modes.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(table) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--no-causal-mask", action="store_true")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "build" / "attention_looping_compare",
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
        help="Write artifacts even if looped/unrolled invariants fail.",
    )
    args = parser.parse_args()

    causal_mask = not args.no_causal_mask
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cycle_model = load_behavior_cycle_model(
        settings_path=args.settings_path,
        dc_en=args.dc_en,
        latency_profile=args.latency_profile,
    )

    rows: list[dict[str, Any]] = []
    for mode, unroll_attention in (("looped", False), ("unrolled", True)):
        asm = emit_aten_mha_one_head(
            seq_len=args.seq_len,
            head_dim=args.head_dim,
            causal_mask=causal_mask,
            unroll_attention=unroll_attention,
        )
        rows.append(
            _write_case(
                args.out_dir,
                mode=mode,
                asm=asm,
                seq_len=args.seq_len,
                head_dim=args.head_dim,
                causal_mask=causal_mask,
                cycle_model=cycle_model,
            )
        )

    write_summary(args.out_dir, rows)
    if not args.no_assert:
        assert_attention_looping_invariants(rows)

    print(f"Wrote attention looping comparison artifacts to: {args.out_dir}")
    print()
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
