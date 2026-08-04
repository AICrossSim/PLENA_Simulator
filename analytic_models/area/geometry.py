"""Discrete legal-geometry search under a full-chip silicon-area budget."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


def _positive_candidates(name: str, values: Iterable[int]) -> tuple[int, ...]:
    candidates = tuple(sorted({int(value) for value in values}))
    if not candidates or candidates[0] <= 0:
        raise ValueError(f"{name} candidates must be non-empty and positive")
    return candidates


def solve_geometry_for_area(
    config: Mapping[str, Any],
    area_budget_um2: float,
    *,
    mlen_candidates: Iterable[int],
    blen_candidates: Iterable[int],
    vlen_candidates: Iterable[int] | None = None,
    require_vlen_equal_mlen: bool = True,
    hidden_size: int | None = None,
) -> dict[str, Any]:
    """Choose the maximum-compute legal geometry whose full chip fits.

    Candidates are ranked by ``MLEN * BLEN`` (matrix MAC positions), then VLEN,
    MLEN, BLEN, and finally lower area. The default enforces the current compiler
    contract ``VLEN == MLEN``. Matrix SRAM depth is raised to the RTL/software
    floor of four MLEN rows when a candidate requires it.
    """

    from . import estimate_area

    budget = float(area_budget_um2)
    if budget <= 0.0:
        raise ValueError("area_budget_um2 must be positive")
    mlens = _positive_candidates("MLEN", mlen_candidates)
    blens = _positive_candidates("BLEN", blen_candidates)
    vlens = (
        _positive_candidates("VLEN", vlen_candidates)
        if vlen_candidates is not None
        else mlens
    )
    hlen = int(config.get("HLEN", 0))
    feasible: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    evaluated = 0
    legal = 0

    for mlen in mlens:
        candidate_vlens = (mlen,) if require_vlen_equal_mlen else vlens
        for blen in blens:
            for vlen in candidate_vlens:
                if mlen % blen or vlen < mlen:
                    continue
                if hlen and (mlen % hlen or vlen % hlen or not (blen <= hlen <= mlen)):
                    continue
                if hidden_size is not None and int(hidden_size) % vlen:
                    continue
                legal += 1
                candidate = dict(config)
                candidate.update({"MLEN": mlen, "BLEN": blen, "VLEN": vlen})
                if "BLOCK_DIM" in candidate:
                    candidate["BLOCK_DIM"] = blen
                matrix_depth = int(
                    candidate.get(
                        "MATRIX_SRAM_DEPTH",
                        candidate.get("MATRIX_SRAM_SIZE", 4 * mlen),
                    )
                )
                candidate["MATRIX_SRAM_DEPTH"] = max(matrix_depth, 4 * mlen)
                candidate.pop("MATRIX_SRAM_SIZE", None)
                if "HBM_M_Prefetch_Amount" in candidate:
                    candidate["HBM_M_Prefetch_Amount"] = mlen
                estimated = estimate_area(candidate)
                evaluated += 1
                area = float(estimated["area"])
                if area > budget:
                    continue
                record = {
                    "MLEN": mlen,
                    "BLEN": blen,
                    "VLEN": vlen,
                    "area_um2": area,
                    "area_budget_um2": budget,
                    "area_utilization": area / budget,
                    "matrix_multipliers": mlen * blen,
                    "breakdown": dict(estimated["breakdown"]),
                    "block_evidence": dict(estimated["block_evidence"]),
                    "evidence_tier": estimated["evidence_tier"],
                    "resolved_config": candidate,
                }
                score = (
                    float(mlen * blen),
                    float(vlen),
                    float(mlen),
                    float(blen),
                    -area,
                )
                feasible.append((score, record))

    if not feasible:
        raise ValueError(
            f"no legal geometry fits {budget / 1e6:.6f} mm^2 "
            f"({legal} legal candidates evaluated)"
        )
    _, best = max(feasible, key=lambda item: item[0])
    best.update(
        {
            "objective": "maximize_MLEN_times_BLEN_then_VLEN",
            "legal_candidates": legal,
            "evaluated_candidates": evaluated,
            "feasible_candidates": len(feasible),
            "require_vlen_equal_mlen": bool(require_vlen_equal_mlen),
        }
    )
    return best
