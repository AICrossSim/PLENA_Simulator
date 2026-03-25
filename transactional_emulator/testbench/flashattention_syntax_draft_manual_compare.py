import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
TESTBENCH_DIR = THIS_FILE.parent
TOOLS_DIR = TESTBENCH_DIR.parent / "tools"
if str(TESTBENCH_DIR) not in sys.path:
    sys.path.insert(0, str(TESTBENCH_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from check_mem import parse_golden_output, read_bin_file_as_array, reorder_stride_mode, slice_rows
from mh_flashattention_bshd_wf_test import build_golden


def _summarize(
    title: str,
    golden: np.ndarray,
    simulated: np.ndarray,
    *,
    atol: float,
    rtol: float,
    rel_threshold: float,
    rel_zero_threshold: float,
    topk: int,
) -> list[str]:
    golden = golden.reshape(-1).astype(np.float32)
    simulated = simulated.reshape(-1).astype(np.float32)
    abs_err = np.abs(golden - simulated)
    tol = atol + rtol * np.abs(golden)
    abs_golden = np.abs(golden)
    rel_err = np.where(abs_golden > rel_zero_threshold, abs_err / abs_golden, abs_err)
    allclose_mask = abs_err <= tol
    rel_mask = rel_err <= rel_threshold

    lines: list[str] = []
    lines.append(f"[{title}]")
    lines.append(f"shape={tuple(golden.shape)}")
    lines.append(f"MAE={float(abs_err.mean()):.8e}")
    lines.append(f"MSE={float(np.mean((golden - simulated) ** 2)):.8e}")
    lines.append(f"MAX_ERR={float(abs_err.max()):.8e}")
    lines.append(
        f"REL_MATCH(|err|/|golden|<={rel_threshold}, zero->{rel_zero_threshold})="
        f"{float(rel_mask.mean()) * 100:.2f}%"
    )
    lines.append(f"ALLCLOSE_MATCH(atol={atol},rtol={rtol})={float(allclose_mask.mean()) * 100:.2f}%")
    lines.append(f"ALLCLOSE_PASS={'PASS' if bool(allclose_mask.all()) else 'FAIL'}")
    lines.append(f"GOLDEN_RANGE=[{float(golden.min()):.8e}, {float(golden.max()):.8e}]")
    lines.append(f"SIM_RANGE=[{float(simulated.min()):.8e}, {float(simulated.max()):.8e}]")
    lines.append(f"TOP_{topk}_ERRORS:")
    top_idx = np.argsort(abs_err)[-max(1, int(topk)) :][::-1]
    for idx in top_idx:
        lines.append(
            f"idx={int(idx)} golden={golden[idx]:.8e} sim={simulated[idx]:.8e} "
            f"abs_err={abs_err[idx]:.8e}"
        )
    lines.append("")
    return lines


def _load_recomputed_golden(build_dir: Path) -> np.ndarray:
    table_path = build_dir / "flashattention_syntax_draft_table.json"
    with open(table_path, "r", encoding="utf-8") as f:
        table = json.load(f)

    def load_tensor(name: str) -> torch.Tensor:
        logical_shape = tuple(int(dim) for dim in table[name]["shape"])
        raw = torch.load(build_dir / f"{name}.hbm.pt")
        flat = torch.as_tensor(raw, dtype=torch.float32).contiguous().reshape(-1)
        need = 1
        for dim in logical_shape:
            need *= dim
        if flat.numel() < need:
            raise RuntimeError(f"{name}.hbm.pt too small: got={flat.numel()} need={need}")
        return flat[:need].reshape(logical_shape)

    q = load_tensor("Q_IN")
    k = load_tensor("K_IN")
    v = load_tensor("V_IN")
    golden = build_golden(q=q, k=k, v=v)
    return golden.reshape(-1).cpu().to(torch.float32).numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Compare flashattention_syntax_draft output using the same flow as tile_tensor_flash_attention_compare.py."
    )
    parser.add_argument("--build-dir", default=str(TESTBENCH_DIR / "build"))
    parser.add_argument(
        "--vram-dump",
        default=str(TESTBENCH_DIR.parent / "vram_dump.bin"),
        help="Path to simulator VRAM dump binary.",
    )
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--rel-threshold", type=float, default=0.2)
    parser.add_argument("--rel-zero-threshold", type=float, default=1e-10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--report-file",
        default=None,
        help="Default: <build-dir>/flashattention_syntax_draft_manual_compare_report.txt",
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    vram_dump = Path(args.vram_dump)
    report_path = (
        Path(args.report_file)
        if args.report_file
        else build_dir / "flashattention_syntax_draft_manual_compare_report.txt"
    )

    with open(build_dir / "comparison_params.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    golden_from_txt = parse_golden_output(str(build_dir / "golden_result.txt")).astype(np.float32)
    simulated = read_bin_file_as_array(
        str(vram_dump),
        exp_width=8,
        man_width=7,
        row_dim=int(params["row_dim"]),
        num_bytes_per_val=2,
        start_row_idx=int(params["start_row_idx"]),
        num_rows=int(params["num_rows"]),
    ).astype(np.float32)

    if params.get("use_slice_mode", False):
        slice_per_row = int(params["slice_per_row"])
        row_dim = int(params["row_dim"])
        num_rows = int(params["num_rows"])
        golden_from_txt = slice_rows(
            golden_from_txt,
            row_dim=row_dim,
            slice_per_row=slice_per_row,
            num_rows=num_rows,
        ).astype(np.float32)
        simulated = slice_rows(
            simulated,
            row_dim=row_dim,
            slice_per_row=slice_per_row,
            num_rows=num_rows,
        ).astype(np.float32)

    if params.get("use_stride_mode", False):
        simulated = reorder_stride_mode(
            simulated,
            num_batches=int(params["num_batches"]),
            elements_per_batch=int(params["elements_per_batch"]),
            stride=int(params["row_dim"]),
        ).astype(np.float32)

    if len(simulated) < len(golden_from_txt):
        raise RuntimeError(f"Simulated data too short: got {len(simulated)}, need {len(golden_from_txt)}")
    simulated = simulated[: len(golden_from_txt)]

    golden_fp32_path = build_dir / "flashattention_syntax_draft_golden_fp32.npy"
    golden_bf16_path = build_dir / "flashattention_syntax_draft_golden_bf16.npy"
    if golden_fp32_path.exists():
        golden_recomputed = np.load(golden_fp32_path).astype(np.float32).reshape(-1)
    else:
        golden_recomputed = _load_recomputed_golden(build_dir)
    if len(golden_recomputed) != len(simulated):
        raise RuntimeError(
            f"Recomputed golden length mismatch: recomputed={len(golden_recomputed)} simulated={len(simulated)}"
        )
    if golden_bf16_path.exists():
        golden_bf16 = np.load(golden_bf16_path).astype(np.float32).reshape(-1)
    else:
        golden_bf16 = torch.from_numpy(golden_recomputed).to(torch.bfloat16).to(torch.float32).numpy()

    lines: list[str] = []
    lines.append(f"build_dir={build_dir}")
    lines.append(f"vram_dump={vram_dump}")
    lines.append(f"comparison_params={json.dumps(params, ensure_ascii=True)}")
    lines.append("")
    lines.extend(
        _summarize(
            "golden_result_txt_fp32",
            golden_from_txt,
            simulated,
            atol=float(args.atol),
            rtol=float(args.rtol),
            rel_threshold=float(args.rel_threshold),
            rel_zero_threshold=float(args.rel_zero_threshold),
            topk=int(args.topk),
        )
    )
    lines.extend(
        _summarize(
            "recomputed_golden_fp32",
            golden_recomputed,
            simulated,
            atol=float(args.atol),
            rtol=float(args.rtol),
            rel_threshold=float(args.rel_threshold),
            rel_zero_threshold=float(args.rel_zero_threshold),
            topk=int(args.topk),
        )
    )
    lines.extend(
        _summarize(
            "recomputed_golden_bf16",
            golden_bf16,
            simulated,
            atol=float(args.atol),
            rtol=float(args.rtol),
            rel_threshold=float(args.rel_threshold),
            rel_zero_threshold=float(args.rel_zero_threshold),
            topk=int(args.topk),
        )
    )

    np.save(build_dir / "flashattention_syntax_draft_manual_sampled_fp32.npy", simulated.astype(np.float32))
    np.save(build_dir / "flashattention_syntax_draft_manual_golden_fp32.npy", golden_recomputed.astype(np.float32))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report_written={report_path}")


if __name__ == "__main__":
    main()
