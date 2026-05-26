"""Compare direct_emit linear ASM against ATen compiler linear ASM."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from plena_quant.mxfp import _mx_fp_quantize_hardware

from compiler.asm_templates import preload_act_asm, preload_addr_reg_asm, projection_asm, reset_reg_asm
from transactional_emulator.testbench.emulator_runner import compare_emulator_output, run_emulator
from transactional_emulator.testbench.aten.compare.isa_analysis import analyze_asm, load_behavior_cycle_model
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env


MLEN = 64
BLEN = 4
REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)
DEFAULT_SHAPES = (
    (4, 256, 256),
    (4, 512, 128),
    (4, 2048, 2048),
    (4, 2048, 6144),
    (4, 6144, 2048),
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


def quantize_to_mxfp(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to MXFP and return dequantized result."""
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor,
        width=8,
        exponent_width=4,
        exponent_bias_width=8,
        block_size=[8],
    )
    return bm_x.reshape(orig_shape)


def make_inputs(m: int, k: int, n: int, seed: int) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn(m, k, dtype=torch.bfloat16)
    w = torch.randn(k, n, dtype=torch.bfloat16)

    x_mxfp = quantize_to_mxfp(x).to(torch.bfloat16)
    w_mxfp = quantize_to_mxfp(w).to(torch.bfloat16)
    golden = torch.mm(x_mxfp, w_mxfp)
    return {"X": x, "W": w}, golden


def emit_direct_linear(
    m: int,
    k: int,
    n: int,
    *,
    matrix_sram_tiles: int | None,
) -> tuple[str, int]:
    """Emit direct_emit-style linear ASM. Returns (asm, result_vram_addr)."""
    if m % BLEN != 0:
        raise ValueError(f"direct_emit projection_asm requires M multiple of BLEN={BLEN}, got M={m}")
    if k % MLEN != 0 or n % BLEN != 0:
        raise ValueError(f"expected K multiple of {MLEN} and N multiple of {BLEN}, got K={k}, N={n}")

    act_hbm_size = int(m * k * REAL_DATA_RATIO)
    weight_hbm_offset = act_hbm_size
    weight_hbm_end = int((m * k + k * n) * REAL_DATA_RATIO)
    result_vram_offset = m * k
    scratch_vram_offset = result_vram_offset + m * n

    code = "; direct_emit handwritten linear codegen compare\n"
    code += f"; Shape: ({m}, {k}) @ ({k}, {n}) -> ({m}, {n})\n"
    code += "; Direct path: raw asm_templates only; no PlenaCompiler/OpRegistry/ops.linear\n"
    code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[weight_hbm_offset, weight_hbm_end],
    )
    code += reset_reg_asm(alive_registers=[1, 2, 3])
    code += preload_act_asm(
        vlen=MLEN,
        preload_len=4,
        batch=m,
        hidden_size=k,
        alive_registers=[1, 2, 3, 4, 5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=k,
    )
    code += reset_reg_asm(alive_registers=[1, 2, 3, 4])

    projection_kwargs: dict[str, Any] = {}
    if matrix_sram_tiles is not None:
        projection_kwargs["matrix_sram_size"] = matrix_sram_tiles * MLEN
        projection_kwargs["scratch_base_address"] = scratch_vram_offset

    code += projection_asm(
        mlen=MLEN,
        blen=BLEN,
        batch=m,
        hidden_size=k,
        out_features=n,
        alive_registers=[1, 2, 3, 4, 5, 6],
        w_base_hbm_offset_reg=1,
        activation_base_address=0,
        result_base_address=result_vram_offset,
        rope_enabled=False,
        **projection_kwargs,
    )
    return code, result_vram_offset


def emit_aten_linear(m: int, k: int, n: int, *, unroll_loops: bool = False) -> tuple[str, int]:
    """Emit ATen compiler linear ASM. Returns (asm, result_vram_addr)."""
    from compiler.aten.ops.registry import Backend, OpRegistry
    import compiler.aten.ops as ops
    from compiler.aten.plena import PlenaCompiler

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


def write_sim_artifacts(
    *,
    build_dir: Path,
    asm: str,
    tensors: dict[str, torch.Tensor],
    golden: torch.Tensor,
    result_vram_addr: int,
    m: int,
    k: int,
    n: int,
) -> None:
    fp_preload = [0.0, 1e-6, 1.0 / k] + [0.0] * 7
    create_sim_env(tensors, asm, {"original_output": golden}, fp_preload, build_dir=str(build_dir))
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="linear_codegen_compare",
        data=None,
        specified_data_order=["X", "W"],
        build_path=build_dir,
    )

    params = {
        "start_row_idx": result_vram_addr // MLEN,
        "num_rows": (m * n + MLEN - 1) // MLEN,
        "num_batches": m,
        "elements_per_batch": n,
        "row_dim": MLEN,
    }
    (build_dir / "comparison_params.json").write_text(json.dumps(params, indent=2) + "\n")
    (build_dir / "generated_asm_code.asm").write_text(asm)


def verify_build(build_dir: Path) -> dict[str, Any]:
    run_emulator(build_dir)
    results, _params = compare_emulator_output(build_dir)
    return {
        "allclose_pass": bool(results["allclose_pass"]),
        "allclose_match_rate": float(results["allclose_match_rate"]),
        "max_error": float(results["max_error"]),
        "mae": float(results["mae"]),
    }


def parse_shape(text: str) -> tuple[int, int, int]:
    parts = [int(part) for part in re.split(r"[x, ]+", text.strip()) if part]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,K,N or MxKxN")
    return tuple(parts)  # type: ignore[return-value]


def case_name(path_kind: str, m: int, k: int, n: int) -> str:
    return f"{path_kind}_m{m}_k{k}_n{n}"


def write_summary(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    (out_dir / "summary.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n")

    table = [
        "| Path | Shape | Source lines | Static instr | Dynamic instr | Est cycles | Est ms @1GHz | C_LOOP_START | Verify | ASM |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        verify = row.get("verify")
        if verify is None:
            verify_text = "not run"
        else:
            status = "PASS" if verify["allclose_pass"] else "FAIL"
            verify_text = f"{status}, match={verify['allclose_match_rate']:.2f}%"
        asm_path = Path(row["asm_path"]).relative_to(out_dir)
        table.append(
            f"| `{row['path_kind']}` | `{row['shape']}` | {row['source_lines']} | "
            f"{row['static_instruction_lines']} | {row['dynamic_instruction_count']} | "
            f"{row['estimated_cycles']} | {row['estimated_ms_at_1ghz']:.6f} | "
            f"{row['loop_start_lines']} | "
            f"{verify_text} | `{asm_path}` |"
        )

    table.extend(
        [
            "",
            "Notes:",
            "",
            "- `direct_emit` uses only low-level `compiler.asm_templates` emitters.",
            "- `aten_linear_plena` uses `PlenaCompiler` + `OpRegistry` + `ops.linear`.",
            "- `aten_linear_plena` defaults to looped GEMM emission; use `--aten-unroll unrolled` to force Python-unrolled GEMM emission.",
            "- `estimated_cycles` uses behavior simulator constants from `plena_settings.toml` by default.",
            "- Use `--verify` for actual simulator execution and golden comparison.",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(table) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        dest="shapes",
        type=parse_shape,
        action="append",
        help="Shape as M,K,N or MxKxN. Can be passed multiple times. Defaults to the disputed M=4 shapes.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "build" / "linear_codegen_compare",
        help="Directory for emitted ASM, stats, and optional sim artifacts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--matrix-sram-tiles",
        type=int,
        default=None,
        help=(
            "Direct_emit projection_asm K-tile budget. Omit to use the template default; "
            "pass 4 to match the local ATen linear_plena K-split budget."
        ),
    )
    parser.add_argument(
        "--paths",
        choices=("direct", "aten", "both"),
        default="both",
        help="Which codegen paths to emit.",
    )
    parser.add_argument(
        "--aten-unroll",
        choices=("looped", "unrolled"),
        default="looped",
        help=(
            "ATen linear GEMM emission mode. Defaults to looped and forces "
            "ATEN_UNROLL=0 while emitting so external env does not change results."
        ),
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run transactional emulator and compare against the PyTorch MXFP golden.",
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

    shapes = args.shapes if args.shapes else list(DEFAULT_SHAPES)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cycle_model = load_behavior_cycle_model(
        settings_path=args.settings_path,
        dc_en=args.dc_en,
        latency_profile=args.latency_profile,
    )

    rows: list[dict[str, Any]] = []
    for m, k, n in shapes:
        tensors: dict[str, torch.Tensor] | None = None
        golden: torch.Tensor | None = None
        if args.verify:
            tensors, golden = make_inputs(m, k, n, args.seed)

        generated: list[tuple[str, str, int]] = []
        shape = f"M={m} K={k} N={n}"

        if args.paths in {"direct", "both"}:
            asm, result_addr = emit_direct_linear(m, k, n, matrix_sram_tiles=args.matrix_sram_tiles)
            generated.append(("direct_emit", asm, result_addr))

        if args.paths in {"aten", "both"}:
            asm, result_addr = emit_aten_linear(m, k, n, unroll_loops=args.aten_unroll == "unrolled")
            generated.append(("aten_linear_plena", asm, result_addr))

        for path_kind, asm, result_addr in generated:
            name = case_name(path_kind, m, k, n)
            case_dir = args.out_dir / name
            case_dir.mkdir(parents=True, exist_ok=True)
            (case_dir / "generated_asm_code.asm").write_text(asm)

            row = {
                "case": name,
                "path_kind": path_kind,
                "shape": shape,
                "asm_path": str(case_dir / "generated_asm_code.asm"),
                "matrix_sram_tiles": args.matrix_sram_tiles if path_kind == "direct_emit" else 4,
                "aten_unroll_mode": args.aten_unroll if path_kind == "aten_linear_plena" else None,
                "result_vram_addr": result_addr,
            }
            row.update(analyze_asm(asm, cycle_model=cycle_model))

            if args.verify:
                assert tensors is not None and golden is not None
                write_sim_artifacts(
                    build_dir=case_dir,
                    asm=asm,
                    tensors=tensors,
                    golden=golden,
                    result_vram_addr=result_addr,
                    m=m,
                    k=k,
                    n=n,
                )
                row["verify"] = verify_build(case_dir)

            (case_dir / "stats.json").write_text(json.dumps(row, indent=2) + "\n")
            rows.append(row)

    write_summary(args.out_dir, rows)
    print(f"Wrote linear comparison artifacts to: {args.out_dir}")
    print()
    print((args.out_dir / "summary.md").read_text())


if __name__ == "__main__":
    main()
