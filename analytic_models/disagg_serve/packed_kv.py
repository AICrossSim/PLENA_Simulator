"""Physical KV-cache traffic for the four PackedKV ablation modes."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass

PADDED_PER_HEAD = "padded_per_head"
DENSE_COMPILER = "dense_compiler"
DENSE_SELECTOR = "dense_selector"
IDEAL_TRAFFIC = "ideal_traffic"
PACKED_KV_MODES = (
    PADDED_PER_HEAD,
    DENSE_COMPILER,
    DENSE_SELECTOR,
    IDEAL_TRAFFIC,
)

DRAIN_ACCUMULATOR_BYTES_PER_CHIP = 576
SOFTMAX_CONSTANT_SLOTS = 6
SOFTMAX_STATE_VALUES_PER_ROW = 3
CURRENT_FP_SRAM_DEPTH = 512
KV_HEAD_REUSE_MEASURED_LATENCY_DELTA = {
    2: -0.0061,
    4: -0.0116,
}


def _aligned_bytes(bits: int, alignment_bytes: int) -> int:
    raw_bytes = math.ceil(bits / 8)
    return math.ceil(raw_bytes / alignment_bytes) * alignment_bytes


@dataclass(frozen=True)
class PackedKVTraffic:
    """Element/scale-plane layout for one token and one K or V tensor."""

    kv_heads: int
    head_dim: int
    mlen: int
    element_bits: int
    block_size: int = 8
    scale_bits: int = 8
    alignment_bytes: int = 64

    def __post_init__(self) -> None:
        for name in (
            "kv_heads",
            "head_dim",
            "mlen",
            "element_bits",
            "block_size",
            "alignment_bytes",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.scale_bits < 0:
            raise ValueError("scale_bits must be non-negative")
        if self.head_dim > self.mlen or self.mlen % self.head_dim:
            raise ValueError("MLEN must contain integral head slots")
        if self.head_dim % self.block_size:
            raise ValueError("head_dim must contain complete MX blocks")

    @property
    def active_elements(self) -> int:
        return self.kv_heads * self.head_dim

    @property
    def dense_rows(self) -> int:
        return math.ceil(self.active_elements / self.mlen)

    @property
    def effective_bits(self) -> float:
        return self.element_bits + self.scale_bits / self.block_size

    @property
    def layout_id(self) -> str:
        payload = {
            "schema": "plena-packed-kv-layout",
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "mlen": self.mlen,
            "element_bits": self.element_bits,
            "block_size": self.block_size,
            "scale_bits": self.scale_bits,
            "alignment_bytes": self.alignment_bytes,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return "packed-kv-" + digest

    @property
    def element_row_bytes(self) -> int:
        return _aligned_bytes(
            self.mlen * self.element_bits,
            self.alignment_bytes,
        )

    @property
    def scale_row_bytes(self) -> int:
        return (
            _aligned_bytes(
                self.mlen // self.block_size * self.scale_bits,
                self.alignment_bytes,
            )
            if self.scale_bits
            else 0
        )

    @property
    def dense_row_bytes(self) -> int:
        return self.element_row_bytes + self.scale_row_bytes

    @property
    def ideal_element_bytes(self) -> int:
        return math.ceil(self.active_elements * self.element_bits / 8)

    @property
    def ideal_scale_bytes(self) -> int:
        blocks = math.ceil(self.active_elements / self.block_size)
        return math.ceil(blocks * self.scale_bits / 8)

    @property
    def ideal_bytes(self) -> int:
        return math.ceil(self.active_elements * self.effective_bits / 8)

    def storage_bytes(self, mode: str) -> int:
        return self.storage_element_bytes(mode) + self.storage_scale_bytes(mode)

    def storage_element_bytes(self, mode: str) -> int:
        self._check_mode(mode)
        if mode == PADDED_PER_HEAD:
            return self.kv_heads * self.element_row_bytes
        if mode in {DENSE_COMPILER, DENSE_SELECTOR}:
            return self.dense_rows * self.element_row_bytes
        return self.ideal_element_bytes

    def storage_scale_bytes(self, mode: str) -> int:
        self._check_mode(mode)
        if mode == PADDED_PER_HEAD:
            return self.kv_heads * self.scale_row_bytes
        if mode in {DENSE_COMPILER, DENSE_SELECTOR}:
            return self.dense_rows * self.scale_row_bytes
        return self.ideal_scale_bytes

    def read_bytes(self, mode: str) -> int:
        return self.read_element_bytes(mode) + self.read_scale_bytes(mode)

    def read_element_bytes(self, mode: str) -> int:
        self._check_mode(mode)
        if mode == DENSE_COMPILER:
            return self.kv_heads * self.element_row_bytes
        return self.storage_element_bytes(mode)

    def read_scale_bytes(self, mode: str) -> int:
        self._check_mode(mode)
        if mode == DENSE_COMPILER:
            return self.kv_heads * self.scale_row_bytes
        return self.storage_scale_bytes(mode)

    def storage_ratio(self, mode: str) -> float:
        return self.storage_bytes(mode) / self.ideal_bytes

    def read_ratio(self, mode: str) -> float:
        return self.read_bytes(mode) / self.ideal_bytes

    def _check_mode(self, mode: str) -> None:
        if mode not in PACKED_KV_MODES:
            raise ValueError(f"unknown PackedKV mode {mode!r}")


def traffic_from_precision(
    *,
    kv_heads: int,
    head_dim: int,
    mlen: int,
    element_bits: int,
    effective_bits: float,
    block_size: int = 8,
) -> PackedKVTraffic:
    scale_bits = round((effective_bits - element_bits) * block_size)
    if scale_bits < 0:
        raise ValueError("effective bits cannot be narrower than the element")
    return PackedKVTraffic(
        kv_heads=kv_heads,
        head_dim=head_dim,
        mlen=mlen,
        element_bits=element_bits,
        block_size=block_size,
        scale_bits=scale_bits,
    )


def kv_head_reuse_status(
    *,
    enabled: bool,
    mlen: int,
    hlen: int,
    blen: int,
    kv_heads: int,
    fp_sram_depth: int = CURRENT_FP_SRAM_DEPTH,
) -> dict[str, object]:
    """Return schedule legality, traffic amplification, and evidence scope."""

    if not isinstance(enabled, bool):
        raise TypeError("enabled must be boolean")
    for name, value in (
        ("mlen", mlen),
        ("hlen", hlen),
        ("blen", blen),
        ("kv_heads", kv_heads),
        ("fp_sram_depth", fp_sram_depth),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if mlen % hlen:
        raise ValueError("HLEN must divide MLEN")
    broadcast_heads = mlen // hlen
    required_slots = (
        SOFTMAX_CONSTANT_SLOTS
        + SOFTMAX_STATE_VALUES_PER_ROW
        * blen
        * broadcast_heads
        * kv_heads
    )
    supported = (
        not enabled
        or (
            kv_heads <= broadcast_heads
            and required_slots <= fp_sram_depth
        )
    )
    measured_delta = (
        KV_HEAD_REUSE_MEASURED_LATENCY_DELTA.get(kv_heads)
        if enabled
        else None
    )
    if not enabled:
        evidence_tier = "source_derived_per_head_schedule"
    elif measured_delta is not None:
        evidence_tier = "transactional_emulator_measured"
    else:
        evidence_tier = "analytic_extrapolation_from_hkv2_hkv4"
    return {
        "enabled": enabled,
        "supported": supported,
        "broadcast_heads": broadcast_heads,
        "kv_heads": kv_heads,
        "kv_read_factor": 1 if enabled else kv_heads,
        "traffic_reduction_vs_per_head": kv_heads if enabled else 1,
        "required_fp_sram_slots": required_slots if enabled else 0,
        "available_fp_sram_slots": fp_sram_depth,
        "measured_latency_delta_fraction": measured_delta,
        "evidence_tier": evidence_tier,
        "evidence_scope": (
            "exact K/V traffic reconciliation and latency at hkv=2/4; "
            "other head counts are explicitly extrapolated"
        ),
        "source": (
            "transactional_emulator/testbench/misc/"
            "flash_attention_gqa_fused_test.py"
        ),
    }


def architecture_option_area_mm2(
    *,
    mlen: int,
    hlen: int,
    kv_heads: int,
    kv_head_reuse: bool,
    drain_overlapped: bool,
) -> dict[str, object]:
    """Price E2 control and storage with the retained A1 area machinery."""

    if not isinstance(kv_head_reuse, bool) or not isinstance(
        drain_overlapped,
        bool,
    ):
        raise TypeError("architecture options must be boolean")
    for name, value in (
        ("mlen", mlen),
        ("hlen", hlen),
        ("kv_heads", kv_heads),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if mlen % hlen:
        raise ValueError("architecture-option geometry is invalid")

    try:
        from analytic_models.area.evidence import STRUCTURAL_ESTIMATE, weakest_tier
        from analytic_models.area.matrix import (
            load_calibration_artifact,
            load_pdk_scale,
        )
        from analytic_models.area.sram import estimate_buffer_area
    except ImportError:
        from area.evidence import STRUCTURAL_ESTIMATE, weakest_tier
        from area.matrix import load_calibration_artifact, load_pdk_scale
        from area.sram import estimate_buffer_area

    breakdown: dict[str, float] = {}
    evidence: dict[str, dict[str, object]] = {}
    if kv_head_reuse:
        artifact = load_calibration_artifact()
        coefficient = float(
            artifact["full_chip"]["top"]["slice_controls"]
        )
        reference_scale = float(load_pdk_scale())
        selector_bits = max(1, math.ceil(math.log2(kv_heads)))
        control_units = (mlen // hlen) * selector_bits
        breakdown["KVHeadReuseControl"] = (
            coefficient * reference_scale * control_units / 1e6
        )
        evidence["KVHeadReuseControl"] = {
            "tier": STRUCTURAL_ESTIMATE,
            "source": (
                "area/calibration/matrix_structural_coefficients.json:"
                "full_chip.top.slice_controls"
            ),
            "model_scope": (
                "selector state and loop-hoist control units priced with the "
                "fitted top-level slice-control unit area"
            ),
            "control_units": control_units,
            "selector_bits": selector_bits,
            "raw_dc_reports_available": False,
        }
    if drain_overlapped:
        buffer = estimate_buffer_area(
            DRAIN_ACCUMULATOR_BYTES_PER_CHIP,
            word_bits=32,
            ports=1,
        )
        breakdown["DrainOverlapAccumulatorBank"] = float(buffer["area"]) / 1e6
        evidence["DrainOverlapAccumulatorBank"] = {
            **dict(buffer["evidence"]),
            "capacity_bytes": DRAIN_ACCUMULATOR_BYTES_PER_CHIP,
            "word_bits": 32,
            "ports": 1,
        }
    return {
        "area_mm2_per_chip": sum(breakdown.values()),
        "breakdown_mm2_per_chip": breakdown,
        "evidence": evidence,
        "evidence_tier": (
            weakest_tier(evidence.values()) if evidence else "not_applicable"
        ),
    }


__all__ = [
    "DENSE_COMPILER",
    "DENSE_SELECTOR",
    "DRAIN_ACCUMULATOR_BYTES_PER_CHIP",
    "IDEAL_TRAFFIC",
    "PACKED_KV_MODES",
    "PADDED_PER_HEAD",
    "PackedKVTraffic",
    "architecture_option_area_mm2",
    "kv_head_reuse_status",
    "traffic_from_precision",
]
