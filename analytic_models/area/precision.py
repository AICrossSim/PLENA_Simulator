"""Normalize software precision knobs into RTL operand-side capabilities.

PLENA uses asymmetric processing elements. Activations drive the L operand
side, while weights and cached K/V values share the T-side hardware over
different operations. A single global datapath must therefore support the
widest K/V or weight format selected by the software profile.

This module is deliberately strict: unsupported widths and mixed MXINT/MXFP
profiles fail early instead of silently selecting an uncalibrated model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


class PrecisionError(ValueError):
    """Raised when a precision profile is outside the calibrated area model."""


@dataclass(frozen=True)
class MXPrecision:
    """Canonical representation of one block-scaled numeric format.

    Exactly one of ``bits`` (MXINT) or ``exp``/``mant`` (MXFP) is meaningful.
    ``scale_width`` describes the shared block-scale storage/datapath width;
    it is not included in the per-element width.
    """

    kind: str
    bits: int | None = None
    exp: int | None = None
    mant: int | None = None
    scale_width: int = 8

    @property
    def element_width(self) -> int:
        """Return bits stored per numeric element, excluding block scales."""
        if self.kind == "MXINT":
            if self.bits is None:
                raise PrecisionError("MXINT precision missing bits")
            return self.bits
        if self.kind == "MXFP":
            if self.exp is None or self.mant is None:
                raise PrecisionError("MXFP precision missing exp/mant")
            return 1 + self.exp + self.mant
        raise PrecisionError(f"unsupported precision kind: {self.kind}")

    @property
    def name(self) -> str:
        """Return the normalized software-facing precision token."""
        if self.kind == "MXINT":
            return f"MXINT{self.bits}"
        return f"MXFP_E{self.exp}M{self.mant}"


_MXINT_RE = re.compile(r"^MXINT_?(\d+)$", re.IGNORECASE)
_MXFP_RE = re.compile(r"^MXFP_?E(\d+)M(\d+)$", re.IGNORECASE)

SUPPORTED_MXINT_ACT_KV = {2, 4, 8}
SUPPORTED_MXINT_WEIGHT = {4, 8}
SUPPORTED_MXFP = {(1, 2), (2, 1), (4, 3), (5, 2)}


def parse_precision(value: Any, *, default_scale_width: int = 8) -> MXPrecision:
    """Parse a software precision knob into a normalized precision object.

    Args:
        value: An existing :class:`MXPrecision`, a token such as ``MXINT4`` or
            ``MXFP_E4M3``, or a mapping with explicit numeric fields.
        default_scale_width: Shared-scale width used when ``value`` does not
            specify one.

    Returns:
        A canonical immutable precision description.

    Raises:
        PrecisionError: If the input syntax or required fields are invalid.
    """
    if isinstance(value, MXPrecision):
        return value
    if isinstance(value, str):
        token = value.strip().upper()
        if match := _MXINT_RE.match(token):
            return MXPrecision(kind="MXINT", bits=int(match.group(1)), scale_width=default_scale_width)
        if match := _MXFP_RE.match(token):
            return MXPrecision(
                kind="MXFP",
                exp=int(match.group(1)),
                mant=int(match.group(2)),
                scale_width=default_scale_width,
            )
    if isinstance(value, dict):
        kind = str(value.get("kind", "")).upper()
        scale_width = int(value.get("scale_width", default_scale_width))
        if kind == "MXINT":
            bits = value.get("bits", value.get("width"))
            if bits is None:
                raise PrecisionError(f"MXINT dict missing bits/width: {value!r}")
            return MXPrecision(kind="MXINT", bits=int(bits), scale_width=scale_width)
        if kind == "MXFP":
            if "exp" not in value or "mant" not in value:
                raise PrecisionError(f"MXFP dict missing exp/mant: {value!r}")
            return MXPrecision(kind="MXFP", exp=int(value["exp"]), mant=int(value["mant"]), scale_width=scale_width)
    raise PrecisionError(f"unsupported precision spec: {value!r}")


def _validate_role(precision: MXPrecision, role: str) -> None:
    if precision.kind == "MXINT":
        if role == "weight":
            allowed = SUPPORTED_MXINT_WEIGHT
        else:
            allowed = SUPPORTED_MXINT_ACT_KV
        if precision.bits not in allowed:
            raise PrecisionError(f"{role} MXINT{precision.bits} unsupported; allowed={sorted(allowed)}")
        return
    if precision.kind == "MXFP":
        key = (precision.exp, precision.mant)
        if key not in SUPPORTED_MXFP:
            raise PrecisionError(f"{role} {precision.name} unsupported")
        return
    raise PrecisionError(f"{role} unsupported precision kind: {precision.kind}")


def derive_compute_sides(
    act: Any,
    kv: Any,
    weight: Any = "MXINT4",
    *,
    default_scale_width: int = 8,
) -> dict[str, Any]:
    """Map software precision knobs to asymmetric MatrixMachine PE sides.

    The v1 capability model uses ACT on the L side and max(KV, Weight) on the
    T side. For MXFP, exponent and mantissa maxima are taken independently so
    the synthesized T-side format can represent both source formats. Mixed
    MXINT/MXFP profiles are intentionally rejected because they require a
    separate dual-family datapath model.

    Returns:
        A dictionary containing the normalized profile, operand widths,
        shared-scale width, and model family consumed by MatrixMachine and
        SRAM estimators.

    Raises:
        PrecisionError: If a role uses an unsupported format or the profile
            mixes MXINT and MXFP families.
    """
    act_p = parse_precision(act, default_scale_width=default_scale_width)
    kv_p = parse_precision(kv, default_scale_width=default_scale_width)
    wt_p = parse_precision(weight, default_scale_width=default_scale_width)
    _validate_role(act_p, "act")
    _validate_role(kv_p, "kv")
    _validate_role(wt_p, "weight")

    kinds = {act_p.kind, kv_p.kind, wt_p.kind}
    if len(kinds) != 1:
        raise PrecisionError(f"mixed MXINT/MXFP profiles are unsupported: {act_p.name}, {kv_p.name}, {wt_p.name}")

    if act_p.kind == "MXINT":
        t_bits = max(int(kv_p.bits), int(wt_p.bits))
        l_bits = int(act_p.bits)
        return {
            "mode": "mxint",
            "t_bits": t_bits,
            "l_bits": l_bits,
            "t_width": t_bits,
            "l_width": l_bits,
            "scale_width": max(act_p.scale_width, kv_p.scale_width, wt_p.scale_width),
            "act": act_p.name,
            "kv": kv_p.name,
            "weight": wt_p.name,
        }

    t_exp = max(int(kv_p.exp), int(wt_p.exp))
    t_mant = max(int(kv_p.mant), int(wt_p.mant))
    l_exp = int(act_p.exp)
    l_mant = int(act_p.mant)
    return {
        "mode": "mxfp",
        "t_exp": t_exp,
        "t_mant": t_mant,
        "l_exp": l_exp,
        "l_mant": l_mant,
        "t_width": 1 + t_exp + t_mant,
        "l_width": 1 + l_exp + l_mant,
        "scale_width": max(act_p.scale_width, kv_p.scale_width, wt_p.scale_width),
        "act": act_p.name,
        "kv": kv_p.name,
        "weight": wt_p.name,
    }


def mxfp_format_name(exp: int, mant: int) -> str:
    """Build the canonical MXFP token for exponent and mantissa widths."""
    return f"MXFP_E{exp}M{mant}"
