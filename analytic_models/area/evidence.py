"""Evidence labels shared by the chip-area estimators."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

AGGREGATE_DC_FIT = "dc_synthesized_aggregate_fit"
AGGREGATE_DC_EXTRAPOLATION = "dc_synthesized_aggregate_structural_extrapolation"
SRAM_MACRO_TABLE = "published_sram_macro_geometry"
PUBLISHED_DENSITY_SCALING = "published_density_node_scaled"
STRUCTURAL_ESTIMATE = "declared_structural_estimate"

PROVENANCE_GRADE = "aggregate_area_tables_without_raw_dc_reports"


def aggregate_dc_evidence(
    source: str,
    *,
    extrapolated: bool,
    calibration_domain: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe a fitted aggregate DC model without implying raw-report custody."""

    return {
        "tier": (AGGREGATE_DC_EXTRAPOLATION if extrapolated else AGGREGATE_DC_FIT),
        "source": source,
        "provenance_grade": PROVENANCE_GRADE,
        "raw_dc_reports_available": False,
        "structural_extrapolation": bool(extrapolated),
        "calibration_domain": dict(calibration_domain),
    }


def weakest_tier(records: Iterable[Mapping[str, Any]]) -> str:
    """Return a conservative combined label for a mixed-evidence estimate."""

    tiers = {str(record["tier"]) for record in records}
    if STRUCTURAL_ESTIMATE in tiers:
        return STRUCTURAL_ESTIMATE
    if AGGREGATE_DC_EXTRAPOLATION in tiers:
        return AGGREGATE_DC_EXTRAPOLATION
    if len(tiers) == 1:
        return tiers.pop()
    return "mixed_analytic_evidence"
