#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from transactional_emulator.testbench.siglip.utils.math import (
    gelu_with_bf16_intermediates,
    quantize_to_mxfp,
)


SEED = 42
BATCH = 768
HIDDEN = 4352
ATOL = 0.2
RTOL = 0.2
PLOT_X_MIN = -1.0
PLOT_X_MAX = 1.0
PLOT_NUM_POINTS = 5000


def gelu_tanh_with_bf16_intermediates(x: torch.Tensor) -> torch.Tensor:
    """GELU tanh approximation with BF16 truncation at hardware-visible steps."""
    x_f32 = x.float()

    x2 = (x_f32 * x_f32).to(torch.bfloat16)
    x3 = (x2.float() * x_f32).to(torch.bfloat16)
    cubic_term = (0.044715 * x3.float()).to(torch.bfloat16)
    poly = (x_f32 + cubic_term.float()).to(torch.bfloat16)

    z = (0.7978845608028654 * poly.float()).to(torch.bfloat16)  # sqrt(2/pi)
    two_z = (z.float() + z.float()).to(torch.bfloat16)
    exp_2z = torch.exp(two_z.float()).to(torch.bfloat16)

    num = (exp_2z.float() - 1.0).to(torch.bfloat16)
    den = (num.float() + 2.0).to(torch.bfloat16)
    tanh_z = (num.float() * (1.0 / den.float())).to(torch.bfloat16)

    one_plus_tanh = (1.0 + tanh_z.float()).to(torch.bfloat16)
    half_x = (0.5 * x_f32).to(torch.bfloat16)
    return (half_x.float() * one_plus_tanh.float()).to(torch.bfloat16)


def pairwise_metrics(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float) -> dict[str, float]:
    """Compute simple numeric agreement metrics between two tensors."""
    a_f = a.float()
    b_f = b.float()
    diff = torch.abs(a_f - b_f)
    mask = torch.isclose(a_f, b_f, atol=atol, rtol=rtol)
    return {
        "mse": float(torch.mean((a_f - b_f) ** 2).item()),
        "mae": float(diff.mean().item()),
        "max_error": float(diff.max().item()),
        "allclose_match_rate": float((mask.float().mean() * 100).item()),
        "allclose_pass": bool(mask.all().item()),
    }


def print_metrics(title: str, m: dict[str, float], atol: float, rtol: float) -> None:
    print(title)
    print(f"  MSE: {m['mse']:.6e}")
    print(f"  MAE: {m['mae']:.6f}")
    print(f"  Max Error: {m['max_error']:.6f}")
    print(
        f"  Allclose @ atol={atol}, rtol={rtol}: "
        f"{m['allclose_match_rate']:.2f}% ({'PASS' if m['allclose_pass'] else 'FAIL'})"
    )


def save_difference_plots(
    x_in: torch.Tensor,
    tanh_hw: torch.Tensor,
    sigmoid_hw: torch.Tensor,
    pytorch_tanh: torch.Tensor,
    output_dir: Path,
) -> None:
    """Save overlay and absolute-difference plots for GELU approximations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot over a fixed input sweep for consistent visual comparison.
    x_plot = torch.linspace(PLOT_X_MIN, PLOT_X_MAX, PLOT_NUM_POINTS, dtype=torch.float32)
    x_plot_bf16 = x_plot.to(torch.bfloat16)
    y_sigmoid_sorted = gelu_with_bf16_intermediates(x_plot_bf16).float()
    y_tanh_sorted = gelu_tanh_with_bf16_intermediates(x_plot_bf16).float()
    y_ref_sorted = F.gelu(x_plot.float(), approximate="tanh")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_plot.numpy(), y_sigmoid_sorted.numpy(), label="sigmoid_hw", linewidth=1.5)
    ax.plot(x_plot.numpy(), y_tanh_sorted.numpy(), label="tanh_hw", linewidth=1.5)
    ax.plot(x_plot.numpy(), y_ref_sorted.numpy(), label="pytorch_tanh", linewidth=1.5)
    ax.set_title("GELU Approximation Overlay")
    ax.set_xlabel("Input x")
    ax.set_ylabel("Output")
    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gelu_overlay.png", dpi=180)
    plt.close(fig)

    abs_err_sigmoid = torch.abs(y_sigmoid_sorted - y_ref_sorted)
    abs_err_tanh = torch.abs(y_tanh_sorted - y_ref_sorted)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x_plot.numpy(), abs_err_sigmoid.numpy(), label="|sigmoid_hw - pytorch_tanh|", linewidth=1.5)
    ax.plot(x_plot.numpy(), abs_err_tanh.numpy(), label="|tanh_hw - pytorch_tanh|", linewidth=1.5)
    ax.set_title("GELU Approximation Absolute Difference")
    ax.set_xlabel("Input x")
    ax.set_ylabel("Absolute Error")
    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "gelu_difference.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare GELU approximation variants.")
    parser.add_argument("--plot", action="store_true", help="Save approximation difference plots.")
    parser.add_argument(
        "--plot-dir",
        default=str(Path(__file__).parent / "build" / "gelu_plots"),
        help="Directory where plot PNG files are saved.",
    )
    args = parser.parse_args()

    torch.manual_seed(SEED)
    x = torch.randn(BATCH, HIDDEN, dtype=torch.bfloat16)
    x_in = quantize_to_mxfp(x).to(torch.bfloat16)

    gelu_tanh_hw = gelu_tanh_with_bf16_intermediates(x_in)
    gelu_sigmoid_hw = gelu_with_bf16_intermediates(x_in)
    gelu_pytorch_tanh = F.gelu(x_in.float(), approximate="tanh").to(torch.bfloat16)

    print("================================================")
    print("GELU Variant Comparison")
    print("================================================")
    print(f"Input shape: {tuple(x_in.shape)}")
    print("Input mode: MXFP-quantized BF16")
    print(f"Allclose thresholds: atol={ATOL}, rtol={RTOL}")
    print("")

    print_metrics(
        "[tanh_hw] vs [sigmoid_hw]",
        pairwise_metrics(gelu_tanh_hw, gelu_sigmoid_hw, atol=ATOL, rtol=RTOL),
        atol=ATOL,
        rtol=RTOL,
    )
    print("")
    print_metrics(
        "[tanh_hw] vs [pytorch_tanh]",
        pairwise_metrics(gelu_tanh_hw, gelu_pytorch_tanh, atol=ATOL, rtol=RTOL),
        atol=ATOL,
        rtol=RTOL,
    )
    print("")
    print_metrics(
        "[sigmoid_hw] vs [pytorch_tanh]",
        pairwise_metrics(gelu_sigmoid_hw, gelu_pytorch_tanh, atol=ATOL, rtol=RTOL),
        atol=ATOL,
        rtol=RTOL,
    )

    if args.plot:
        out_dir = Path(args.plot_dir).resolve()
        save_difference_plots(
            x_in=x_in,
            tanh_hw=gelu_tanh_hw,
            sigmoid_hw=gelu_sigmoid_hw,
            pytorch_tanh=gelu_pytorch_tanh,
            output_dir=out_dir,
        )
        print("")
        print(f"Saved plots to: {out_dir}")
