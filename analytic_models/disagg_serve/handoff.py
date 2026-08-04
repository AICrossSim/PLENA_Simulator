"""BF16 prefill-to-decode KV transfer and decode-cache admission costs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

LINK_GENS = {
    "nvlink3": 300e9,
    "nvlink4": 450e9,
    "ualink": 400e9,
    "pcie5": 64e9,
}
LINK_ENERGY_PJ_PER_BIT = {
    "nvlink3": 2.0,
    "nvlink4": 1.5,
    "ualink": 2.5,
    "pcie5": 10.0,
}
LINK_ENERGY_SOURCE = {
    "evidence_scope": "declared inter-package/on-board link sensitivity",
    "values_pj_per_bit": LINK_ENERGY_PJ_PER_BIT,
}

BF16_BITS = 16
HANDOFF_SCOPE = "prefill_to_decode_admission_ttft"
HANDOFF_REGIMES = ("fully_pipelined", "back_pressure", "host_buffered")
HANDOFF_SCHEDULE_SCOPE = "request_level_prefill_decode_schedule"
_CONTENT_ADDRESSED_ID = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*-[0-9a-f]{64}$"
)


@dataclass(frozen=True)
class AdmissionModel:
    """Decode-ingress quantizer assumptions.

    Bandwidth is aggregate BF16-read plus packed-write throughput. Energy
    coefficients are intentionally explicit so uncalibrated defaults remain
    distinguishable from measured values.
    """

    bandwidth_bytes_per_s: float = 900e9
    quantize_energy_j_per_element: float = 0.0
    memory_energy_j_per_byte: float = 0.0
    calibrated: bool = False
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "bandwidth_bytes_per_s",
            "quantize_energy_j_per_element",
            "memory_energy_j_per_byte",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.bandwidth_bytes_per_s <= 0:
            raise ValueError("admission bandwidth must be positive")
        if not isinstance(self.calibrated, bool):
            raise TypeError("calibrated must be boolean")
        if self.calibrated != bool(self.calibration_id):
            raise ValueError(
                "admission calibration state requires one calibration identity"
            )
        if (
            self.calibration_id is not None
            and (
                not _CONTENT_ADDRESSED_ID.fullmatch(self.calibration_id)
                or not self.calibration_id.startswith("admission-")
            )
        ):
            raise ValueError(
                "admission calibration must be a content-addressed identity"
            )


@dataclass(frozen=True)
class HandoffTime:
    wire_bytes: float
    decode_cache_bytes: float
    link_bw: float
    transfer_bulk_s: float
    transfer_streamed_s: float
    admission_s: float
    admission_energy_j: float
    admission_calibrated: bool
    admission_calibration_id: str | None
    scope: str = HANDOFF_SCOPE

    def __post_init__(self) -> None:
        if self.scope != HANDOFF_SCOPE:
            raise ValueError("handoff scope is unsupported")
        for name in (
            "wire_bytes",
            "decode_cache_bytes",
            "link_bw",
            "transfer_bulk_s",
            "transfer_streamed_s",
            "admission_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.admission_energy_j)
            or self.admission_energy_j < 0
        ):
            raise ValueError(
                "admission_energy_j must be finite and non-negative"
            )
        if self.admission_calibrated != bool(
            self.admission_calibration_id
        ):
            raise ValueError("admission calibration identity is inconsistent")
        if (
            self.admission_calibration_id is not None
            and (
                not _CONTENT_ADDRESSED_ID.fullmatch(
                    self.admission_calibration_id
                )
                or not self.admission_calibration_id.startswith("admission-")
            )
        ):
            raise ValueError(
                "admission calibration must be a content-addressed identity"
            )

    @property
    def kv_bytes(self) -> float:
        """Backward-compatible alias for BF16 bytes transmitted on the link."""
        return self.wire_bytes

    @property
    def bulk_s(self) -> float:
        return self.transfer_bulk_s + self.admission_s

    @property
    def streamed_s(self) -> float:
        return self.transfer_streamed_s + self.admission_s

    @property
    def steady_state_tpot_s(self) -> float:
        return 0.0

    @property
    def publication_rankable(self) -> bool:
        return False

    def ttft_add(self, mode: str) -> float:
        if mode not in {"bulk", "streamed"}:
            raise ValueError(f"unknown handoff mode {mode!r}")
        return self.bulk_s if mode == "bulk" else self.streamed_s


@dataclass(frozen=True)
class HandoffScheduleResult:
    """Critical-path and resource balance for one handoff schedule."""

    regime: str
    prompt_tokens: int
    generation_tokens: int
    precision: str
    prefill_s: float
    transfer_s: float
    admission_s: float
    wait_s: float
    host_spill_s: float
    ttft_s: float
    energy_j: float
    prefill_utilization: float
    prefill_decode_ratio: float
    energy_tier: str
    scope: str = HANDOFF_SCHEDULE_SCOPE

    def __post_init__(self) -> None:
        if self.regime not in HANDOFF_REGIMES:
            raise ValueError(f"unsupported handoff regime {self.regime!r}")
        if self.scope != HANDOFF_SCHEDULE_SCOPE:
            raise ValueError("handoff schedule scope is unsupported")
        if self.prompt_tokens <= 0 or self.generation_tokens <= 0:
            raise ValueError("prompt and generation lengths must be positive")
        if not self.precision or not self.energy_tier:
            raise ValueError("precision and energy tier must be explicit")
        for name in (
            "prefill_s",
            "transfer_s",
            "admission_s",
            "wait_s",
            "host_spill_s",
            "ttft_s",
            "energy_j",
            "prefill_decode_ratio",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.prefill_s <= 0 or self.ttft_s <= 0 or self.prefill_decode_ratio <= 0:
            raise ValueError("schedule latency and balance ratio must be positive")
        if (
            not math.isfinite(self.prefill_utilization)
            or not 0 < self.prefill_utilization <= 1
        ):
            raise ValueError("prefill utilisation must lie in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "regime": self.regime,
            "prompt_tokens": self.prompt_tokens,
            "generation_tokens": self.generation_tokens,
            "precision": self.precision,
            "prefill_s": self.prefill_s,
            "transfer_s": self.transfer_s,
            "admission_s": self.admission_s,
            "wait_s": self.wait_s,
            "host_spill_s": self.host_spill_s,
            "ttft_s": self.ttft_s,
            "energy_j": self.energy_j,
            "prefill_utilization": self.prefill_utilization,
            "prefill_decode_ratio": self.prefill_decode_ratio,
            "energy_tier": self.energy_tier,
        }


def kv_elements(dims: dict, input_seq: int, batch: int) -> int:
    """K and V elements across every layer in one immutable prefill artifact."""
    return (
        2
        * int(dims["kv_heads"])
        * int(dims["head_dim"])
        * int(input_seq)
        * int(batch)
        * int(dims["layers"])
    )


def kv_wire_bytes(
    dims: dict,
    prec: dict,
    input_seq: int,
    batch: int,
    *,
    wire_bits: int = BF16_BITS,
) -> float:
    """Transferred prompt KV bytes.

    ``prec`` is accepted for API compatibility; transfer precision is fixed by
    ``wire_bits`` and is independent of the decode-cache format.
    """
    del prec
    if wire_bits <= 0:
        raise ValueError("wire_bits must be positive")
    return kv_elements(dims, input_seq, batch) * wire_bits / 8


def decode_cache_bytes(dims: dict, prec: dict, input_seq: int, batch: int) -> float:
    """Packed decode-cache bytes after admission, including effective MX scale bits."""
    return kv_elements(dims, input_seq, batch) * float(prec["kv_bits"]) / 8


def handoff_time(
    dims: dict,
    prec: dict,
    input_seq: int,
    batch: int,
    link_gen: str = "nvlink4",
    link_bw: float | None = None,
    *,
    admission: AdmissionModel | None = None,
) -> HandoffTime:
    admission = admission or AdmissionModel()
    bw = float(link_bw) if link_bw else LINK_GENS[link_gen]
    if bw <= 0 or admission.bandwidth_bytes_per_s <= 0:
        raise ValueError("link and admission bandwidths must be positive")

    elements = kv_elements(dims, input_seq, batch)
    wire = kv_wire_bytes(dims, prec, input_seq, batch)
    packed = decode_cache_bytes(dims, prec, input_seq, batch)
    admission_bytes = wire + packed
    return HandoffTime(
        wire_bytes=wire,
        decode_cache_bytes=packed,
        link_bw=bw,
        transfer_bulk_s=wire / bw,
        transfer_streamed_s=(wire / int(dims["layers"])) / bw,
        admission_s=admission_bytes / admission.bandwidth_bytes_per_s,
        admission_energy_j=(
            elements * admission.quantize_energy_j_per_element
            + admission_bytes * admission.memory_energy_j_per_byte
        ),
        admission_calibrated=admission.calibrated,
        admission_calibration_id=admission.calibration_id,
    )


def evaluate_handoff_regimes(
    handoff: HandoffTime,
    *,
    prompt_tokens: int,
    generation_tokens: int,
    precision: str,
    prefill_latency_s: float,
    decode_tpot_s: float,
    decode_ready_delay_s: float,
    prefill_energy_j: float,
    decode_energy_per_token_j: float,
    prefill_stall_power_w: float = 0.0,
    decode_idle_power_w: float = 0.0,
    direct_link_generation: str = "nvlink4",
    host_link_generation: str = "pcie5",
    direct_link_energy_pj_per_bit: float | None = None,
    host_link_energy_pj_per_bit: float | None = None,
    energy_tier: str = "analytic_anchored",
) -> tuple[HandoffScheduleResult, ...]:
    """Compare static, stalled, and host-buffered handoff schedules.

    The host-buffered path counts a physical PCIe write followed by a read.
    The initial write overlaps decode unavailability; only the residual wait is
    placed on the TTFT critical path.  The balancing ratio equates prefill-chip
    and decode-chip service rates for one request:

        N_prefill / N_decode = T_prefill / (T_admission + G * TPOT_decode)
    """

    for name, value in (
        ("prefill_latency_s", prefill_latency_s),
        ("decode_tpot_s", decode_tpot_s),
        ("decode_ready_delay_s", decode_ready_delay_s),
        ("prefill_energy_j", prefill_energy_j),
        ("decode_energy_per_token_j", decode_energy_per_token_j),
        ("prefill_stall_power_w", prefill_stall_power_w),
        ("decode_idle_power_w", decode_idle_power_w),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if prefill_latency_s <= 0 or decode_tpot_s <= 0:
        raise ValueError("prefill latency and decode TPOT must be positive")
    if prompt_tokens <= 0 or generation_tokens <= 0:
        raise ValueError("prompt and generation lengths must be positive")
    if not precision or not energy_tier:
        raise ValueError("precision and energy tier must be explicit")
    if direct_link_generation not in LINK_GENS or host_link_generation not in LINK_GENS:
        raise ValueError("handoff schedule uses an unsupported link generation")

    direct_pj = (
        LINK_ENERGY_PJ_PER_BIT[direct_link_generation]
        if direct_link_energy_pj_per_bit is None
        else float(direct_link_energy_pj_per_bit)
    )
    host_pj = (
        LINK_ENERGY_PJ_PER_BIT[host_link_generation]
        if host_link_energy_pj_per_bit is None
        else float(host_link_energy_pj_per_bit)
    )
    for name, value in (
        ("direct_link_energy_pj_per_bit", direct_pj),
        ("host_link_energy_pj_per_bit", host_pj),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")

    decode_service_s = handoff.admission_s + generation_tokens * decode_tpot_s
    balanced_ratio = prefill_latency_s / decode_service_s
    base_energy_j = (
        prefill_energy_j
        + generation_tokens * decode_energy_per_token_j
        + handoff.admission_energy_j
    )
    direct_link_energy_j = handoff.wire_bytes * 8.0 * direct_pj * 1e-12
    host_link_energy_j = 2.0 * handoff.wire_bytes * 8.0 * host_pj * 1e-12

    fully_pipelined = HandoffScheduleResult(
        regime="fully_pipelined",
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        precision=precision,
        prefill_s=prefill_latency_s,
        transfer_s=handoff.transfer_streamed_s,
        admission_s=handoff.admission_s,
        wait_s=0.0,
        host_spill_s=0.0,
        ttft_s=(
            prefill_latency_s
            + handoff.transfer_streamed_s
            + handoff.admission_s
            + decode_tpot_s
        ),
        energy_j=base_energy_j + direct_link_energy_j,
        prefill_utilization=1.0,
        prefill_decode_ratio=balanced_ratio,
        energy_tier=energy_tier,
    )

    back_pressure = HandoffScheduleResult(
        regime="back_pressure",
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        precision=precision,
        prefill_s=prefill_latency_s,
        transfer_s=handoff.transfer_bulk_s,
        admission_s=handoff.admission_s,
        wait_s=decode_ready_delay_s,
        host_spill_s=0.0,
        ttft_s=(
            prefill_latency_s
            + decode_ready_delay_s
            + handoff.transfer_bulk_s
            + handoff.admission_s
            + decode_tpot_s
        ),
        energy_j=(
            base_energy_j
            + direct_link_energy_j
            + decode_ready_delay_s
            * (prefill_stall_power_w + decode_idle_power_w)
        ),
        prefill_utilization=(
            prefill_latency_s / (prefill_latency_s + decode_ready_delay_s)
        ),
        prefill_decode_ratio=balanced_ratio,
        energy_tier=energy_tier,
    )

    host_write_s = handoff.wire_bytes / LINK_GENS[host_link_generation]
    host_read_s = host_write_s
    residual_wait_s = max(0.0, decode_ready_delay_s - host_write_s)
    host_buffered = HandoffScheduleResult(
        regime="host_buffered",
        prompt_tokens=prompt_tokens,
        generation_tokens=generation_tokens,
        precision=precision,
        prefill_s=prefill_latency_s,
        transfer_s=0.0,
        admission_s=handoff.admission_s,
        wait_s=residual_wait_s,
        host_spill_s=host_write_s + host_read_s,
        ttft_s=(
            prefill_latency_s
            + max(decode_ready_delay_s, host_write_s)
            + host_read_s
            + handoff.admission_s
            + decode_tpot_s
        ),
        energy_j=(
            base_energy_j
            + host_link_energy_j
            + max(decode_ready_delay_s, host_write_s) * decode_idle_power_w
        ),
        prefill_utilization=1.0,
        prefill_decode_ratio=balanced_ratio,
        energy_tier=energy_tier,
    )
    return fully_pipelined, back_pressure, host_buffered


def report(
    dims: dict,
    prec: dict,
    input_seq: int,
    batch: int,
    link_gen: str = "nvlink4",
    link_bw: float | None = None,
    *,
    admission: AdmissionModel | None = None,
) -> str:
    h = handoff_time(
        dims,
        prec,
        input_seq,
        batch,
        link_gen,
        link_bw,
        admission=admission,
    )
    calibration = (
        h.admission_calibration_id
        if h.admission_calibrated
        else "uncalibrated sensitivity"
    )
    return (
        f"      BF16 KV wire bytes:  {h.wire_bytes/1e9:.3f} GB\n"
        f"      Decode-cache bytes:  {h.decode_cache_bytes/1e9:.3f} GB "
        f"(@ {prec['kv_bits']:.2f} effective bits)\n"
        f"      Link sensitivity:    {link_gen} = {h.link_bw/1e9:.0f} GB/s per direction\n"
        f"      Admission convert:   {h.admission_s*1e3:.3f} ms, "
        f"{h.admission_energy_j*1e3:.3f} mJ ({calibration})\n"
        f"      TTFT add, bulk:      {h.bulk_s*1e3:.3f} ms\n"
        f"      TTFT add, streamed:  {h.streamed_s*1e3:.3f} ms\n"
        "      Steady-state TPOT:   +0.000 ms (admission excluded)"
    )
