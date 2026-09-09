"""Routed-MoE coverage contracts for heterogeneous calibrated events."""

from __future__ import annotations

from types import SimpleNamespace

from .model import EventCount, estimate_power


class _StructuralArea:
    payload = {"pdk_scale_reference": 1.0}
    evidence_id = "structural-test"

    @staticmethod
    def matrix_area_mm2(signature: str, **_kwargs) -> float:
        return {
            "LINEAR:MXINT4xMXINT4": 4.0,
            "LINEAR:MXINT8xMXINT8": 8.0,
            "QK:MXINT4xMXINT4": 4.0,
            "PV:MXINT4xMXINT4": 4.0,
        }[signature]

    @staticmethod
    def sram_area_mm2(*_args, **_kwargs) -> float:
        return 1.0


class _Calibration:
    def __init__(self, *, omit: str | None = None) -> None:
        signatures = {
            "LINEAR:MXINT4xMXINT4",
            "LINEAR:MXINT8xMXINT8",
            "QK:MXINT4xMXINT4",
            "PV:MXINT4xMXINT4",
            "VECTOR:FP_E3M2",
            "SELECTOR:PACKED_KV",
        }
        if omit is not None:
            signatures.remove(omit)
        self.signatures = signatures
        self.structural_area_model = _StructuralArea()
        self.fixed_area_mm2 = 1.0
        self.hbm_energy_j_per_byte = 0.5
        self.synthesis_context = {"hardware_fp_binding": "FP_E3M2"}
        self.validation = SimpleNamespace(
            passed=True,
            meets_publication_gate=True,
        )
        self.source_sha256 = "1" * 64
        self.provenance_hash = "2" * 64
        self.activity_provenance_hash = "3" * 64

    def event_energy_j(self, event: EventCount) -> float:
        if event.signature not in self.signatures:
            raise KeyError(event.signature)
        return 1.0

    @staticmethod
    def complete_chip_leakage_w(**_kwargs) -> float:
        return 1.0

    @staticmethod
    def vector_area_mm2(*_args, **_kwargs) -> float:
        return 1.0

    @staticmethod
    def selector_area_mm2(**_kwargs) -> float:
        return 1.0


def _events() -> tuple[EventCount, ...]:
    return tuple(
        EventCount(signature, 1, 16, 4)
        for signature in (
            "LINEAR:MXINT4xMXINT4",
            "LINEAR:MXINT8xMXINT8",
            "QK:MXINT4xMXINT4",
            "PV:MXINT4xMXINT4",
            "VECTOR:FP_E3M2",
            "SELECTOR:PACKED_KV",
        )
    )


def test_multiple_linear_signatures_keep_max_area_and_rankability() -> None:
    estimate = estimate_power(
        _Calibration(),
        _events(),
        elapsed_s=1.0,
        hbm_bytes=1.0,
        vector_fp="FP_E3M2",
        area_config={"MX_SCALE_WIDTH": 8},
    )
    assert estimate.rankable
    assert estimate.missing_signatures == ()
    assert estimate.matrix_area_mm2 == 8.0
    assert estimate.array_signatures == (
        "LINEAR:MXINT4xMXINT4",
        "LINEAR:MXINT8xMXINT8",
        "PV:MXINT4xMXINT4",
        "QK:MXINT4xMXINT4",
    )


def test_one_missing_linear_signature_remains_fail_closed() -> None:
    estimate = estimate_power(
        _Calibration(omit="LINEAR:MXINT8xMXINT8"),
        _events(),
        elapsed_s=1.0,
        hbm_bytes=1.0,
        vector_fp="FP_E3M2",
        area_config={"MX_SCALE_WIDTH": 8},
    )
    assert not estimate.rankable
    assert "LINEAR:MXINT8xMXINT8" in estimate.missing_signatures

