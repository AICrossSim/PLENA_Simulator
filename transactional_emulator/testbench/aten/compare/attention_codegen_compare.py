"""Compare ATen attention codegen against direct asm_template emission."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from pathlib import Path
from typing import Any

from compiler.asm_templates import preload_act_asm, preload_addr_reg_asm, reset_reg_asm
from compiler.asm_templates.flashattn import flash_attn_asm
from compiler.aten.ops.registry import Backend, OpRegistry
import compiler.aten.ops as ops
from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.compare.isa_analysis import analyze_asm, load_behavior_cycle_model


MLEN = 64
BLEN = 4
REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)
INSTR_PREFIXES = ("S_", "C_", "H_", "V_", "M_")


def _quiet_flash_attn_asm(**kwargs: Any) -> str:
    """Suppress flash_attn_asm stdout."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        return flash_attn_asm(**kwargs)


def emit_direct_flashattn_teststyle(*, hq: int, hkv: int, head_dim: int, include_q_preload: bool = True) -> str:
    """Emit direct_emit-style flash attention ASM."""
    seq_len = 64
    batch = 1

    q_hbm_size = int(seq_len * hq * head_dim * batch * REAL_DATA_RATIO)
    padded_kv_heads = max(hkv, MLEN // head_dim)
    k_hbm_size = int(seq_len * padded_kv_heads * head_dim * batch * REAL_DATA_RATIO)
    k_hbm_offset = q_hbm_size
    v_hbm_offset = q_hbm_size + k_hbm_size

    code = "; direct_emit-style flash attention codegen compare\n"
    code += f"; Config: batch={batch}, seq={seq_len}, hq={hq}, hkv={hkv}, d={head_dim}\n"
    code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[k_hbm_offset, v_hbm_offset],
    )

    if include_q_preload:
        code += preload_act_asm(
            vlen=MLEN,
            preload_len=4,
            batch=batch,
            hidden_size=head_dim * hq * seq_len,
            alive_registers=[1, 2, 3, 4, 5],
            act_vram_offset=0,
            activation_offset_reg=0,
        )
        code += reset_reg_asm(alive_registers=[1, 2, 3, 4, 5])

    code += _quiet_flash_attn_asm(
        mlen=MLEN,
        blen=BLEN,
        vlen=MLEN,
        batch=batch,
        hq=hq,
        hkv=hkv,
        d=head_dim,
        q_len=seq_len,
        kv_len=seq_len,
        alive_registers_int=list(range(1, 16)),
        alive_registers_fp=list(range(1, 8)),
        vector_sram_base_address=0,
        fp_sram_start_address=3,
        k_base_hbm_offset_reg=1,
        v_base_hbm_offset_reg=2,
    )
    return code


def emit_aten_mha_one_head(*, causal_mask: bool = True) -> str:
    """Emit ATen MHA primitive path (single head)."""
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    seq_len = 64
    head_dim = 64
    prog = PlenaCompiler(mlen=MLEN, blen=BLEN, real_data_ratio=REAL_DATA_RATIO)

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


def emit_aten_gqa_fused(*, hq: int, hkv: int, head_dim: int) -> str:
    """Emit ATen GQA fused path."""
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    seq_len = 64
    hidden_size = hq * head_dim
    prog = PlenaCompiler(mlen=MLEN, blen=BLEN, real_data_ratio=REAL_DATA_RATIO)

    q_input = prog.input("Q", shape=(seq_len, hidden_size), prestaged_vram_addr=0)
    k_input = prog.input("K", shape=(seq_len, MLEN))
    v_input = prog.input("V", shape=(seq_len, MLEN))
    Q = prog.load_batch(q_input, name="Q")

    with contextlib.redirect_stdout(io.StringIO()):
        ops.flash_attention(
            prog,
            Q,
            k_input,
            v_input,
            1.0 / math.sqrt(head_dim),
            hq=hq,
            hkv=hkv,
            h_qkv=head_dim,
        )
    return prog.compile()


def _write_case(
    out_dir: Path,
    name: str,
    asm: str,
    *,
    cycle_model,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case_dir = out_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "generated_asm_code.asm").write_text(asm)

    result = {"case": name, "asm_path": str(case_dir / "generated_asm_code.asm")}
    result.update(analyze_asm(asm, cycle_model=cycle_model))
    if extra:
        result.update(extra)
    (case_dir / "stats.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _ratio(a: int, b: int) -> float | None:
    if b == 0:
        return None
    return a / b


def _format_ratio(a: int, b: int) -> str:
    value = _ratio(a, b)
    if value is None:
        return "n/a"
    return f"{value:.2f}x"


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    summary_json = {
        "rows": rows,
        "comparisons": {
            "aten_mha_vs_direct_mha_source_lines": _ratio(
                rows_by_case(rows)["aten_mha_1h_d64"]["source_lines"],
                rows_by_case(rows)["direct_emit_mha_1h_d64_teststyle"]["source_lines"],
            ),
            "aten_gqa_vs_direct_gqa_source_lines": _ratio(
                rows_by_case(rows)["aten_gqa_fused_4h_d16"]["source_lines"],
                rows_by_case(rows)["direct_emit_gqa_4h_d16_teststyle"]["source_lines"],
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary_json, indent=2) + "\n")

    table_lines = [
        "| Case | Source lines | Static instr | Dynamic instr | Est cycles | Est ms @1GHz | C_LOOP_START | ASM |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        rel_path = Path(row["asm_path"]).relative_to(out_dir)
        table_lines.append(
            f"| `{row['case']}` | {row['source_lines']} | {row['static_instruction_lines']} | "
            f"{row['dynamic_instruction_count']} | {row['estimated_cycles']} | "
            f"{row['estimated_ms_at_1ghz']:.6f} | {row['loop_start_lines']} | `{rel_path}` |"
        )

    by_case = rows_by_case(rows)
    table_lines.extend(
        [
            "",
            "Ratios:",
            "",
            "- ATen MHA primitive vs direct_emit MHA teststyle source lines: "
            + _format_ratio(
                by_case["aten_mha_1h_d64"]["source_lines"],
                by_case["direct_emit_mha_1h_d64_teststyle"]["source_lines"],
            ),
            "- ATen GQA fused vs direct_emit GQA teststyle source lines: "
            + _format_ratio(
                by_case["aten_gqa_fused_4h_d16"]["source_lines"],
                by_case["direct_emit_gqa_4h_d16_teststyle"]["source_lines"],
            ),
            "- Native CLM has 6 Q heads, so a rough attention-core-only static "
            "source estimate is 6x the ATen MHA one-head path, before Q RoPE and O-head copy overhead.",
            "",
            "Notes:",
            "",
            f"- Cycle model: {rows[0]['cycle_model']}.",
            "- The MHA rows are not equivalent micro-kernels. The direct per-head template emits "
            "`d / blen` inner-loop `M_TMM`s for QK; ATen's `vram_sub_projection_T` emits "
            "`d / mlen` hidden-block `M_TMM`s. For `d=64, mlen=64, blen=4`, that is "
            "4096 dynamic `M_TMM`s in direct MHA versus 256 in ATen MHA.",
            "- Treat the lower ATen MHA dynamic/cycle estimate as a codegen-path discrepancy to inspect, "
            "not proof that one semantically identical attention implementation is 3x faster.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(table_lines))


def rows_by_case(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["case"]: row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "build" / "attention_codegen_compare",
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
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cycle_model = load_behavior_cycle_model(
        settings_path=args.settings_path,
        dc_en=args.dc_en,
        latency_profile=args.latency_profile,
    )

    cases = [
        (
            "direct_emit_mha_1h_d64_teststyle",
            emit_direct_flashattn_teststyle(hq=1, hkv=1, head_dim=64),
            {"path_kind": "direct_emit", "shape": "hq=1,hkv=1,d=64"},
        ),
        (
            "aten_mha_1h_d64",
            emit_aten_mha_one_head(causal_mask=True),
            {"path_kind": "aten_mha_primitive", "shape": "hq=1,hkv=1,d=64,causal_mask=True"},
        ),
        (
            "direct_emit_gqa_4h_d16_teststyle",
            emit_direct_flashattn_teststyle(hq=4, hkv=1, head_dim=16),
            {"path_kind": "direct_emit", "shape": "hq=4,hkv=1,d=16"},
        ),
        (
            "aten_gqa_fused_4h_d16",
            emit_aten_gqa_fused(hq=4, hkv=1, head_dim=16),
            {"path_kind": "aten_gqa_fused", "shape": "hq=4,hkv=1,d=16"},
        ),
    ]

    rows = [_write_case(args.out_dir, name, asm, cycle_model=cycle_model, extra=extra) for name, asm, extra in cases]
    write_summary(args.out_dir, rows)

    print(f"Wrote comparison artifacts to: {args.out_dir}")
    print()
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
