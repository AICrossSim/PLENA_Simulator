"""Small arithmetic audit for prospective BF16/FP32 physical-PK geometries.

This is a functional reference, not timed SRAM payload execution or trained
model quality validation.  All PK choices use the same operand precision and
FP32 operations.  Their reduction order may produce different result bits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .compute import Core, PK_VALUES


def bf16_bits(values: Any) -> np.ndarray:
    """FP32 to BF16, round nearest with ties to even; retain NaNs as NaNs."""
    a = np.asarray(values, dtype=np.float32)
    u = a.view(np.uint32)
    rounded = (u + np.uint32(0x7FFF) + ((u >> 16) & 1)) >> 16
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    return np.where(nan, (u >> 16) | 0x0040, rounded).astype(np.uint16)


def from_bf16(bits: Any) -> np.ndarray:
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def bf16(values: Any) -> np.ndarray:
    return from_bf16(bf16_bits(values))


def dot_tree(products: np.ndarray, pk: int) -> np.ndarray:
    """Adjacent balanced FP32 sums over exactly PK leaves, including zeros."""
    if pk not in PK_VALUES:
        raise ValueError("unsupported physical PK")
    products = np.asarray(products, dtype=np.float32)
    if not products.ndim or products.shape[-1] > pk:
        raise ValueError("a product vector must fit one physical PK tile")
    values = np.zeros(products.shape[:-1] + (pk,), dtype=np.float32)
    values[..., :products.shape[-1]] = products
    while values.shape[-1] > 1:
        values = np.add(values[..., 0::2], values[..., 1::2], dtype=np.float32)
    return values[..., 0]


def _operands(x: Any, weights_nk: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    weights = np.asarray(weights_nk, dtype=np.float32)
    if (x.ndim != 2 or weights.ndim != 2 or min(*x.shape, *weights.shape) < 1
            or x.shape[1] != weights.shape[1]):
        raise ValueError("positive X[M,K] and W[N,K] with matching K required")
    if not np.isfinite(x).all() or not np.isfinite(weights).all():
        raise ValueError("finite operands required for this numerical audit")
    return bf16(x), bf16(weights)


def projection_bf16_fp32(x: Any, weights_nk: Any, core: Core,
                         round_output_to_bf16: bool = False) -> np.ndarray:
    """Physical M/N/K tiles, separate FP32 multiplies, ascending segment adds.

    Inputs and weights first undergo BF16 RNE.  Absent M/N/K lanes are zero.
    Each tile uses its installed full-PK tree even for a short last segment.
    The default returns FP32 partial-sum results; optional final BF16 RNE is
    distinct from the accumulation contract.
    """
    xq, wq = _operands(x, weights_nk)
    m, k = xq.shape
    n = wq.shape[0]
    out = np.zeros((m, n), dtype=np.float32)
    for mi in range(0, m, core.pm):
        mr = min(core.pm, m - mi)
        for ni in range(0, n, core.pn):
            nr = min(core.pn, n - ni)
            acc = np.zeros((core.pm, core.pn), dtype=np.float32)
            for ki in range(0, k, core.pk):
                kr = min(core.pk, k - ki)
                xs = np.zeros((core.pm, core.pk), dtype=np.float32)
                ws = np.zeros((core.pn, core.pk), dtype=np.float32)
                xs[:mr, :kr] = xq[mi:mi + mr, ki:ki + kr]
                ws[:nr, :kr] = wq[ni:ni + nr, ki:ki + kr]
                products = np.multiply(xs[:, None, :], ws[None, :, :], dtype=np.float32)
                acc = np.add(acc, dot_tree(products, core.pk), dtype=np.float32)
            out[mi:mi + mr, ni:ni + nr] = acc[:mr, :nr]
    return bf16(out) if round_output_to_bf16 else out


def reference_fp64(x: Any, weights_nk: Any) -> np.ndarray:
    """FP64 dot of the same BF16-rounded operands, not original FP32 values."""
    xq, wq = _operands(x, weights_nk)
    return xq.astype(np.float64) @ wq.astype(np.float64).T


def error_statistics(observed: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if observed.shape != reference.shape or not observed.size:
        raise ValueError("matching nonempty outputs required")
    if not np.isfinite(observed).all() or not np.isfinite(reference).all():
        raise ValueError("finite outputs required for error statistics")
    error = observed - reference
    absolute = np.abs(error)
    norm_reference = float(np.linalg.norm(reference))
    norm_error = float(np.linalg.norm(error))
    return {
        "elements": int(observed.size),
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "relative_l2_error": norm_error / max(norm_reference, np.finfo(np.float64).tiny),
        "max_error_scaled_by_max1_abs_reference": float((absolute / np.maximum(1.0, np.abs(reference))).max()),
    }


def cancellation_counterexample() -> tuple[np.ndarray, np.ndarray]:
    """BF16-exact 2**24 terms expose PK32 segment-order loss at K=97."""
    x = np.ones((1, 97), dtype=np.float32)
    w = np.zeros((1, 97), dtype=np.float32)
    w[0, [0, 32, 64, 96]] = [2.0 ** 24, 1.0, -(2.0 ** 24), -1.0]
    return x, w


def validation_report(seed: int = 20261004) -> dict[str, Any]:
    """Small seeded audit consumable by the DSE runner; no route/perf reads."""
    rng = np.random.default_rng(seed)
    cases = []
    for m, n, k in ((3, 5, 79), (5, 7, 257), (2, 4, 1031)):
        # Different scales exercise cancellation and nontrivial tail segments.
        x = (rng.normal(size=(m, k)) * rng.lognormal(0, 0.6, size=(1, k))).astype(np.float32)
        w = (rng.normal(size=(n, k)) * rng.lognormal(0, 0.6, size=(1, k))).astype(np.float32)
        reference = reference_fp64(x, w)
        baseline = projection_bf16_fp32(x, w, Core(2, 3, 512))
        records = []
        for pk in PK_VALUES:
            out = projection_bf16_fp32(x, w, Core(2, 3, pk))
            records.append({
                "pk": pk,
                "fp32_error_vs_fp64_same_bf16_operands": error_statistics(out, reference),
                "bf16_output_error_vs_fp64_same_bf16_operands": error_statistics(bf16(out), reference),
                "fp32_different_bits_vs_pk512": int(np.count_nonzero(out.view(np.uint32) != baseline.view(np.uint32))),
                "bf16_different_bits_vs_pk512": int(np.count_nonzero(bf16_bits(out) != bf16_bits(baseline))),
            })
        cases.append({"shape_mnk": [m, n, k], "physical_pm_pn": [2, 3],
                      "useful_macs": m * n * k, "results_by_pk": records})
    x, w = cancellation_counterexample()
    counterexample = {str(pk): float(projection_bf16_fp32(x, w, Core(1, 1, pk))[0, 0])
                      for pk in PK_VALUES}
    return {
        "schema": "plena_geometry3d_numerical_audit_v1", "seed": seed,
        "scope": "small seeded functional arithmetic audit; no route timing or trained-model quality",
        "precision": "BF16 RNE inputs/weights; separate FP32 multiply and addition",
        "tree_contract": "adjacent balanced full physical-PK tree with zero tails; ascending FP32 K-segment sums",
        "reference": "FP64 dot of the same BF16-rounded operands",
        "same_precision_does_not_require_bitwise_equivalence": counterexample["32"] != counterexample["64"],
        "counterexample": {
            "shape_mnk": [1, 1, 97], "x": "all ones",
            "nonzero_w_indices": [0, 32, 64, 96],
            "nonzero_w_values": [2 ** 24, 1, -(2 ** 24), -1],
            "fp64_reference": float(reference_fp64(x, w)[0, 0]),
            "fp32_outputs_by_pk": counterexample,
        },
        "random_matrix_cases": cases,
        "limitations": ["No numerical claim about trained-model task quality",
                        "No functional payload replay of the supply timeline",
                        "No guarantee about fused MAC or physical RTL rounding"],
    }


def write_report(path: Path, seed: int = 20261004) -> dict[str, Any]:
    report = validation_report(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261004)
    args = parser.parse_args()
    report = write_report(args.output, args.seed)
    print(json.dumps({"output": str(args.output.resolve()),
                      "matrix_cases": len(report["random_matrix_cases"]),
                      "counterexample_outputs": report["counterexample"]["fp32_outputs_by_pk"]}, sort_keys=True))


if __name__ == "__main__":
    main()
