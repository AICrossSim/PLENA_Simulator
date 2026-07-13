"""Calibrated MXINT area equations for the PLENA MatrixMachine hierarchy.

The hierarchy is modeled at three levels: one asymmetric PE, one BLEN x BLEN
mini systolic array, and the complete MatrixMachine. Coefficients are fitted
against 7 nm Synopsys DC cell area and all returned areas are in um^2.

The runtime supports older coefficient schemas so historical experiments stay
readable. New calibration artifacts use the direct-width v3 equation, selected
by the presence of ``mm_direct_tl``.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MXINT_COEFFS: dict[str, float] = {
    "pe_c0": 1.0,
    "pe_c_tl": 1.0,
    "pe_c_sum": 1.0,
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


def pe_area(t_bits: int, l_bits: int, coeffs: dict[str, float] | None = None) -> float:
    """Estimate one asymmetric MXINT PE.

    ``T*L`` represents multiplier bit complexity; ``T+L`` captures operand
    registers, forwarding, sign handling, and muxing around the multiplier.
    """
    c = coeffs or DEFAULT_MXINT_COEFFS
    return c["pe_c0"] + c["pe_c_tl"] * t_bits * l_bits + c["pe_c_sum"] * (t_bits + l_bits)


def mini_array_area(
    block_dim: int,
    t_bits: int,
    l_bits: int,
    *,
    scale_width: int = 8,
    coeffs: dict[str, float] | None = None,
) -> float:
    """Estimate a square ``block_dim`` mini systolic array in um^2.

    The leading term replicates ``block_dim^2`` PEs. Remaining terms model
    scale distribution, grid wiring/registers, and fixed module control.
    """
    c = coeffs or DEFAULT_MXINT_COEFFS
    pe = pe_area(t_bits, l_bits, c)
    return (
        c.get("mini_pe_scale", 1.0) * block_dim * block_dim * pe
        + c["mini_a_scale"] * block_dim * scale_width
        + c["mini_a_grid"] * block_dim * block_dim
        + c["mini_a0"]
    )


def matrix_machine_area(
    mlen: int,
    blen: int,
    t_bits: int,
    l_bits: int,
    *,
    scale_width: int = 8,
    coeffs: dict[str, float] | None = None,
) -> float:
    """Estimate the complete MXINT MatrixMachine in um^2.

    ``MLEN`` controls the number of matrix lanes/K slices and ``BLEN`` controls
    the physical mini-array dimension. The selected equation depends on the
    coefficient schema:

    * direct-width v3 fits total area from precision-aware grid features;
    * hierarchy-residual v2 composes a mini-array stack, reduction tree,
      accumulator grid, and top glue;
    * v1 is retained only for backward compatibility.
    """
    c = coeffs or DEFAULT_MXINT_COEFFS
    width = t_bits + l_bits + scale_width
    if "mm_direct_tl" in c:
        return (
            c.get("mm_direct_tl", 0.0) * mlen * blen * t_bits * l_bits
            + c.get("mm_direct_sum", 0.0) * mlen * blen * (t_bits + l_bits)
            + c.get("mm_direct_scale", 0.0) * mlen * blen * scale_width
            + c.get("mm_direct_tile", 0.0) * mlen * blen
            + c.get("mm_direct_b2w", 0.0) * blen * blen * width
            + c.get("mm_direct_b2", 0.0) * blen * blen
            + c.get("mm_direct_mw", 0.0) * mlen * width
            + c.get("mm_direct_m", 0.0) * mlen
            + c.get("mm_direct_const", 0.0)
        )
    mini = mini_array_area(blen, t_bits, l_bits, scale_width=scale_width, coeffs=c)
    k_splits = mlen / blen
    if "mm_reduce_c" in c:
        return (
            c.get("mm_stack_c", 1.0) * k_splits * mini
            + c["mm_reduce_c"] * mlen * blen * width
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
    """Return total MXINT MatrixMachine area and an explanatory breakdown.

    Args:
        inputs: Normalized precision-side fields plus ``MLEN`` and ``BLEN``.
        coeffs: Optional fitted coefficients; bootstrap values are used when
            omitted.

    Returns:
        A serializable dictionary containing area in um^2, model version,
        feature-level breakdown, normalized inputs, and coefficients.
    """
    t_bits = int(inputs["t_bits"])
    l_bits = int(inputs["l_bits"])
    mlen = int(inputs["MLEN"])
    blen = int(inputs["BLEN"])
    scale_width = int(inputs.get("scale_width", 8))
    pe = pe_area(t_bits, l_bits, coeffs)
    mini = mini_array_area(blen, t_bits, l_bits, scale_width=scale_width, coeffs=coeffs)
    total = matrix_machine_area(mlen, blen, t_bits, l_bits, scale_width=scale_width, coeffs=coeffs)
    c = coeffs or DEFAULT_MXINT_COEFFS
    width = t_bits + l_bits + scale_width
    stack = (mlen / blen) * mini
    if "mm_direct_tl" in c:
        direct_terms = {
            "precision_grid_area": c.get("mm_direct_tl", 0.0) * mlen * blen * t_bits * l_bits
            + c.get("mm_direct_sum", 0.0) * mlen * blen * (t_bits + l_bits),
            "scale_path_area": c.get("mm_direct_scale", 0.0) * mlen * blen * scale_width,
            "tile_control_area": c.get("mm_direct_tile", 0.0) * mlen * blen
            + c.get("mm_direct_b2w", 0.0) * blen * blen * width
            + c.get("mm_direct_b2", 0.0) * blen * blen,
            "mlen_control_area": c.get("mm_direct_mw", 0.0) * mlen * width
            + c.get("mm_direct_m", 0.0) * mlen,
            "top_glue_area": c.get("mm_direct_const", 0.0),
        }
        stack = direct_terms["precision_grid_area"]
        reduce_tree = direct_terms["scale_path_area"]
        accumulator_grid = direct_terms["tile_control_area"] + direct_terms["mlen_control_area"]
        top_glue = direct_terms["top_glue_area"]
    elif "mm_reduce_c" in c:
        stack *= c.get("mm_stack_c", 1.0)
        reduce_tree = c["mm_reduce_c"] * mlen * blen * width
        accumulator_grid = c.get("mm_accum_c", 0.0) * blen * blen
        top_glue = c.get("mm_top_c", 0.0)
    else:
        reduce_tree = max(0.0, total - stack)
        accumulator_grid = 0.0
        top_glue = 0.0
    return {
        "area": total,
        "area_proxy": total,
        "area_model": (
            "matrix_machine_mxint_direct_width_proxy_v3"
            if "mm_direct_tl" in c
            else "matrix_machine_mxint_hierarchy_residual_v2"
            if "mm_reduce_c" in c
            else "matrix_machine_mxint_precision_proxy_v1"
        ),
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
        "coefficients": dict(coeffs or DEFAULT_MXINT_COEFFS),
    }
