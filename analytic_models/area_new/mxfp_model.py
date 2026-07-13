"""Calibrated MXFP area equations for the PLENA MatrixMachine hierarchy.

MXFP PEs have asymmetric exponent/mantissa widths on the T and L sides. The
model separates total element width from exponent-specific overhead, then
builds a mini-array and MatrixMachine hierarchy. Areas are 7 nm DC cell area
in um^2; they do not include SRAM macros.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MXFP_COEFFS: dict[str, float] = {
    "pe_c0": 1.0,
    "pe_c_tl": 1.0,
    "pe_c_sum": 1.0,
    "pe_c_exp": 1.0,
    "mini_pe_scale": 1.0,
    "mini_a_scale": 1.0,
    "mini_a_grid": 1.0,
    "mini_a0": 1.0,
    "mm_alpha_base": 1.0,
    "mm_beta_width_blen": 0.0,
    "mm_beta_width": 1.0,
    "mm_gamma_mlen": 1.0,
    "mm_const": 0.0,
}


def pe_area(
    t_exp: int,
    t_mant: int,
    l_exp: int,
    l_mant: int,
    coeffs: dict[str, float] | None = None,
) -> float:
    """Estimate one asymmetric MXFP PE in um^2.

    Total-width product approximates significand datapath complexity, while
    the explicit exponent term captures alignment and exponent handling that
    equal-width integer PEs do not require.
    """
    c = coeffs or DEFAULT_MXFP_COEFFS
    t_width = 1 + t_exp + t_mant
    l_width = 1 + l_exp + l_mant
    return (
        c["pe_c0"]
        + c["pe_c_tl"] * t_width * l_width
        + c["pe_c_sum"] * (t_width + l_width)
        + c["pe_c_exp"] * (t_exp + l_exp)
    )


def mini_array_area(
    block_dim: int,
    t_exp: int,
    t_mant: int,
    l_exp: int,
    l_mant: int,
    *,
    scale_width: int = 8,
    coeffs: dict[str, float] | None = None,
) -> float:
    """Estimate a square MXFP mini systolic array in um^2."""
    c = coeffs or DEFAULT_MXFP_COEFFS
    pe = pe_area(t_exp, t_mant, l_exp, l_mant, c)
    return (
        c.get("mini_pe_scale", 1.0) * block_dim * block_dim * pe
        + c["mini_a_scale"] * block_dim * scale_width
        + c["mini_a_grid"] * block_dim * block_dim
        + c["mini_a0"]
    )


def matrix_machine_area(
    mlen: int,
    blen: int,
    t_exp: int,
    t_mant: int,
    l_exp: int,
    l_mant: int,
    *,
    scale_width: int = 8,
    acc_fp_width: int = 16,
    coeffs: dict[str, float] | None = None,
) -> float:
    """Estimate a complete MXFP MatrixMachine in um^2.

    Hierarchy-residual v2 uses a calibrated mini-array stack plus reduction,
    accumulator, and top-glue terms. ``acc_fp_width`` controls the reduction
    datapath width and defaults to the 16-bit RTL calibration assumption.
    """
    c = coeffs or DEFAULT_MXFP_COEFFS
    t_width = 1 + t_exp + t_mant
    l_width = 1 + l_exp + l_mant
    mini = mini_array_area(blen, t_exp, t_mant, l_exp, l_mant, scale_width=scale_width, coeffs=c)
    k_splits = mlen / blen
    width = t_width + l_width + scale_width
    reduce_width = c.get("mm_acc_fp_width", acc_fp_width)
    if "mm_reduce_c" in c:
        return (
            c.get("mm_stack_c", 1.0) * k_splits * mini
            + c["mm_reduce_c"] * mlen * blen * reduce_width
            + c.get("mm_accum_c", 0.0) * blen * blen
            + c.get("mm_top_c", 0.0)
        )
    if "mm_alpha_base" in c or "mm_beta_width_blen" in c or "mm_const" in c:
        return (
            c.get("mm_alpha_base", 1.0) * k_splits * mini
            + c.get("mm_beta_width_blen", 0.0) * mlen * blen * width
            + c.get("mm_gamma_mlen", 0.0) * mlen
            + c.get("mm_const", 0.0)
        )
    return (
        k_splits * mini
        + c.get("mm_beta_width", c.get("mm_beta", 0.0)) * mlen * width
        + c.get("mm_gamma_mlen", c.get("mm_gamma", 0.0)) * mlen
    )


def estimate(inputs: dict[str, Any], coeffs: dict[str, float] | None = None) -> dict[str, Any]:
    """Return MXFP MatrixMachine area, model metadata, and hierarchy terms."""
    t_exp = int(inputs["t_exp"])
    t_mant = int(inputs["t_mant"])
    l_exp = int(inputs["l_exp"])
    l_mant = int(inputs["l_mant"])
    mlen = int(inputs["MLEN"])
    blen = int(inputs["BLEN"])
    scale_width = int(inputs.get("scale_width", 8))
    acc_fp_width = int(inputs.get("ACC_FP_WIDTH", inputs.get("acc_fp_width", 16)))
    pe = pe_area(t_exp, t_mant, l_exp, l_mant, coeffs)
    mini = mini_array_area(blen, t_exp, t_mant, l_exp, l_mant, scale_width=scale_width, coeffs=coeffs)
    total = matrix_machine_area(
        mlen,
        blen,
        t_exp,
        t_mant,
        l_exp,
        l_mant,
        scale_width=scale_width,
        acc_fp_width=acc_fp_width,
        coeffs=coeffs,
    )
    c = coeffs or DEFAULT_MXFP_COEFFS
    stack = (mlen / blen) * mini
    if "mm_reduce_c" in c:
        stack *= c.get("mm_stack_c", 1.0)
        reduce_width = c.get("mm_acc_fp_width", acc_fp_width)
        reduce_tree = c["mm_reduce_c"] * mlen * blen * reduce_width
        accumulator_grid = c.get("mm_accum_c", 0.0) * blen * blen
        top_glue = c.get("mm_top_c", 0.0)
    else:
        reduce_tree = max(0.0, total - stack)
        accumulator_grid = 0.0
        top_glue = 0.0
    return {
        "area": total,
        "area_proxy": total,
        "area_model": "matrix_machine_mxfp_hierarchy_residual_v2" if "mm_reduce_c" in c else "matrix_machine_mxfp_precision_proxy_v1",
        "breakdown": {
            "pe_area": pe,
            "mini_array_area": mini,
            "mini_array_stack_area": stack,
            "reduce_tree_area": reduce_tree,
            "accumulator_grid_area": accumulator_grid,
            "top_glue_area": top_glue,
            "matrix_machine_area": total,
        },
        "inputs": dict(inputs),
        "coefficients": dict(coeffs or DEFAULT_MXFP_COEFFS),
    }
