"""Gates on the independent gate-level cross-validation of the area census.

The campaign these tests guard is evidence, not a fit input, and the honest
handling of it is easy to erode: someone could quietly refit the census onto it,
drop the scope boundary that keeps µm² block figures away from mm² full-chip
figures, or let the stored artifact drift away from the measurements it was
derived from. Each of those is pinned here.
"""

from __future__ import annotations

import json

import pytest

from .calibration_provenance import _CALIBRATION_FILES
from .gate_level_validation import (
    AREA_TABLE,
    ARTIFACT,
    ENERGY_TABLE,
    SCHEMA,
    build_record,
    load_area_points,
    load_energy_points,
)
from .matrix import load_coefficients


@pytest.fixture(scope="module")
def stored() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def derived() -> dict:
    return build_record()


def test_measured_tables_are_present_and_complete() -> None:
    """The campaign is vendored, so the record can always be re-derived."""

    points = load_area_points()
    assert len(points) == 8
    assert {point["mlen"] for point in points} == {16, 32, 64}
    assert {point["blen"] for point in points} == {4, 8}
    assert all(point["slack_ps"] >= 0.0 for point in points), "all points closed timing"
    assert len(load_energy_points()) == 22


def test_stored_artifact_matches_its_derivation(stored: dict, derived: dict) -> None:
    """A hand-edited artifact must not survive; the CSVs are the authority."""

    assert stored == derived
    assert stored["schema"] == SCHEMA


def test_precision_law_reproduces_the_measured_ladder(stored: dict) -> None:
    law = stored["precision_law"]
    assert law["constant_um2"] == pytest.approx(18448.0, abs=1.0)
    assert law["um2_per_exponent_bit"] == pytest.approx(557.0, abs=1.0)
    assert law["um2_per_mantissa_bit"] == pytest.approx(978.0, abs=1.0)
    assert law["mantissa_to_exponent_cost_ratio"] == pytest.approx(1.757, abs=0.01)
    assert law["worst_abs_residual_pct"] < 0.5
    # Both ends of the ladder are reported: the precision-independent share is
    # meaningless without saying which format it is quoted against.
    fraction = law["precision_independent_fraction_pct"]
    assert fraction["vs_widest_format"] == pytest.approx(78.34, abs=0.1)
    assert fraction["vs_narrowest_format"] == pytest.approx(90.09, abs=0.1)


def test_geometry_law_holds_across_an_eightfold_extrapolation(stored: dict) -> None:
    check = stored["geometry_law"]["holdout_extrapolation"]
    assert check["pe_ratio"] == 8.0
    assert abs(check["error_pct"]) < 0.25


def test_leakage_density_is_recorded_with_its_corner_and_scope(stored: dict) -> None:
    leakage = stored["leakage_density"]
    assert leakage["mw_per_um2"] == pytest.approx(9.209669e-07, rel=1e-6)
    assert leakage["w_per_mm2"] == pytest.approx(9.209669e-04, rel=1e-6)
    assert leakage["n_points"] == 8
    assert leakage["spread_pct"] < 2.0
    assert leakage["independent_fit_agreement_pct"] < 1.0
    assert leakage["temperature_c"] == 25
    assert "25C" in leakage["corner"].replace("_", "").replace(" ", "")
    assert "matrix_machine" in leakage["scope"]
    assert "not full-chip" in leakage["scope"]


def test_census_disagreement_is_a_uniform_offset_not_a_shape_error(
    stored: dict,
) -> None:
    """The headline finding: level differs, every trade-off agrees.

    If this ever fails the campaign has stopped confirming the census and the
    coefficients need revisiting rather than the assertion relaxing.
    """

    census = stored["census_cross_validation"]
    assert len(census["points"]) == 8
    assert census["uniform_offset_census_over_campaign"] == pytest.approx(
        1.1242, abs=0.005
    )
    residual = census["shape_and_precision_error_after_offset_pct"]
    assert residual["median"] < 1.0
    assert residual["max"] < 3.5


def test_equal_width_mxfp_formats_expose_a_declared_census_limit(
    stored: dict,
) -> None:
    """E1M2 and E2M1 are both 4 bits wide, so the census cannot separate them."""

    by_key = {
        (entry["precision"], entry["geometry"]): entry
        for entry in stored["census_cross_validation"]["points"]
    }
    e1m2 = by_key[("MXFP_E1M2", "16x4")]
    e2m1 = by_key[("MXFP_E2M1", "16x4")]
    assert e1m2["census_dc_corner_um2"] == pytest.approx(
        e2m1["census_dc_corner_um2"], rel=1e-9
    )
    assert e1m2["measured_um2"] != pytest.approx(e2m1["measured_um2"], rel=1e-3)
    assert "equal width" in stored["census_cross_validation"]["known_model_limit"]


def test_compute_energy_envelope_brackets_the_analytic_anchor(stored: dict) -> None:
    envelope = stored["compute_energy_envelope"]
    assert envelope["anchor_pj_per_mac"] == 0.203
    assert envelope["envelope_pj_per_mac"]["min"] < 0.203
    assert envelope["envelope_pj_per_mac"]["max"] > 0.203
    implied = envelope["implied_toggle_rate_by_geometry"]
    assert implied["MXFP_E1M2_32x4"] == pytest.approx(0.0797, abs=0.001)
    assert implied["MXFP_E1M2_16x4"] == pytest.approx(0.0835, abs=0.001)
    # The estimate is declared activity, not annotated decode switching, and
    # saying so is the whole basis on which it may be reported.
    assert "declared-activity" in envelope["evidence_scope"]
    assert "not measured from decode" in envelope["evidence_scope"]


def test_the_record_declares_its_scope_boundary(stored: dict) -> None:
    """Block µm² at small MLEN must never be offered as full-chip mm²."""

    scope = stored["scope"]
    assert scope["block"] == "matrix_machine"
    assert scope["unit"] == "um^2"
    assert scope["measured_mlen"] == [16, 32, 64]
    assert scope["families"] == ["mxfp"]
    assert "full-chip mm^2" in scope["not_comparable_to"]
    assert "MLEN 128-1024" in scope["not_comparable_to"]


def test_the_campaign_did_not_move_the_shipped_coefficients(stored: dict) -> None:
    """The census is validated by this campaign, not refitted onto it."""

    assert stored["coefficients_changed"] is False
    assert stored["independent_of_the_fit"] is True
    # The fit sources named in the provenance audit are unchanged, so the
    # campaign tables cannot have become fit inputs by the back door.
    assert stored["sources"]["areas"] not in _CALIBRATION_FILES
    assert stored["sources"]["activity_envelope"] not in _CALIBRATION_FILES
    assert AREA_TABLE.name not in _CALIBRATION_FILES
    assert ENERGY_TABLE.name not in _CALIBRATION_FILES
    # And the shipped MXFP coefficients still resolve, unchanged in structure.
    coefficients = load_coefficients("mxfp")
    assert coefficients is not None
    assert coefficients["pe_0"] == pytest.approx(256.96604671603546, rel=1e-12)
