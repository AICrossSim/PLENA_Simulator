"""Precision-profile grouping for hardware-aware DSE sampling."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PRECISION_SEARCH_ENCODINGS = (
    "hardware-signature-v1",
    "profile-categorical-v1",
)
PRECISION_SIGNATURE_PARAM = "MATRIX_DATAPATH_SIGNATURE"
PRECISION_SIGNATURE_SCHEMA = "matrix_weight_activation_internal_fp_v1"


def _parse_mx(spec: Any, default_scale_width: int = 8) -> dict[str, int | str]:
    scale_width = default_scale_width
    if isinstance(spec, str):
        text = spec.upper().replace("_", "")
        if text.startswith("MXINT"):
            return {
                "family": "mxint",
                "width": int(text.removeprefix("MXINT")),
                "scale_width": scale_width,
            }
        text = text.removeprefix("MXFP")
        if text.startswith("E") and "M" in text:
            exp_text, mant_text = text[1:].split("M", 1)
            exp = int(exp_text)
            mant = int(mant_text)
            return {
                "family": "mxfp",
                "exp": exp,
                "mant": mant,
                "width": 1 + exp + mant,
                "scale_width": scale_width,
            }
    if isinstance(spec, Mapping):
        kind = str(spec.get("kind", spec.get("type", ""))).upper().replace(
            "_", ""
        )
        scale_width = int(
            spec.get("scale_width", spec.get("scale", default_scale_width))
        )
        if "MXINT" in kind or "width" in spec or "bits" in spec:
            width = (
                int(kind.removeprefix("MXINT"))
                if kind.startswith("MXINT") and kind != "MXINT"
                else int(spec.get("width", spec.get("bits")))
            )
            return {
                "family": "mxint",
                "width": width,
                "scale_width": scale_width,
            }
        if "MXFP" in kind or {"exp", "mant"} <= set(spec):
            if kind.startswith("MXFP") and kind != "MXFP":
                exp_text, mant_text = kind.removeprefix("MXFP")[1:].split(
                    "M", 1
                )
                exp = int(exp_text)
                mant = int(mant_text)
            else:
                exp = int(spec["exp"])
                mant = int(spec["mant"])
            return {
                "family": "mxfp",
                "exp": exp,
                "mant": mant,
                "width": 1 + exp + mant,
                "scale_width": scale_width,
            }
    raise ValueError(f"unsupported MX precision spec: {spec!r}")


def _component_key(spec: Any, default_scale_width: int) -> dict[str, Any]:
    parsed = _parse_mx(spec, default_scale_width)
    result: dict[str, Any] = {
        "family": parsed["family"],
        "width": int(parsed["width"]),
        "scale_width": int(parsed["scale_width"]),
    }
    if parsed["family"] == "mxfp":
        result.update(
            {
                "exp": int(parsed["exp"]),
                "mant": int(parsed["mant"]),
            }
        )
    if isinstance(spec, Mapping):
        result["block_size"] = int(spec.get("block_size", spec.get("block", 64)))
    else:
        result["block_size"] = 64
    return result


def _component_label(component: Mapping[str, Any]) -> str:
    if component["family"] == "mxint":
        base = f"i{component['width']}"
    else:
        base = f"f{component['exp']}m{component['mant']}"
    return (
        f"{base}s{component['scale_width']}"
        f"b{component['block_size']}"
    )


@dataclass(frozen=True)
class MatrixDatapathSignature:
    """Physical MatrixMachine operand and output-conversion configuration."""

    signature_id: str
    weight: Mapping[str, Any]
    activation: Mapping[str, Any]
    internal_fp_exp: int
    internal_fp_mant: int
    profile_names: tuple[str, ...]

    @property
    def weight_port_bits(self) -> int:
        return int(self.weight["width"])

    @property
    def activation_port_bits(self) -> int:
        return int(self.activation["width"])

    @property
    def pe_bit_product(self) -> int:
        return self.weight_port_bits * self.activation_port_bits

    @property
    def output_fp_bits(self) -> int:
        return 1 + self.internal_fp_exp + self.internal_fp_mant

    def metadata(self) -> dict[str, Any]:
        return {
            "matrix_datapath_signature": self.signature_id,
            "matrix_weight_operand_family": self.weight["family"],
            "matrix_activation_operand_family": self.activation["family"],
            "matrix_weight_port_bits": self.weight_port_bits,
            "matrix_activation_port_bits": self.activation_port_bits,
            "matrix_pe_bit_product": self.pe_bit_product,
            "matrix_output_fp_bits": self.output_fp_bits,
            "matrix_weight_scale_width": int(self.weight["scale_width"]),
            "matrix_activation_scale_width": int(
                self.activation["scale_width"]
            ),
            "matrix_weight_block_size": int(self.weight["block_size"]),
            "matrix_activation_block_size": int(
                self.activation["block_size"]
            ),
            "precision_variant_count": len(self.profile_names),
        }


def build_matrix_datapath_signatures(
    profiles: Sequence[Mapping[str, Any]],
    *,
    default_scale_width: int = 8,
) -> tuple[
    tuple[MatrixDatapathSignature, ...],
    dict[str, str],
]:
    """Group validated profiles by the fields that configure Matrix hardware.

    KV precision is intentionally excluded: it changes memory traffic and
    accuracy but not the MatrixMachine PE operand ports or output converter.
    """

    grouped: dict[str, dict[str, Any]] = {}
    profile_to_key: dict[str, str] = {}
    for profile in profiles:
        weight = _component_key(
            profile["WEIGHT_WIDTH"], default_scale_width
        )
        activation = _component_key(
            profile["ACT_WIDTH"], default_scale_width
        )
        fp = profile["FP_SETTING"]
        semantic = {
            "weight": weight,
            "activation": activation,
            "internal_fp": {
                "exp": int(fp["exp"]),
                "mant": int(fp["mant"]),
            },
        }
        key = json.dumps(semantic, sort_keys=True, separators=(",", ":"))
        name = str(profile["name"])
        profile_to_key[name] = key
        grouped.setdefault(
            key,
            {
                "weight": weight,
                "activation": activation,
                "fp_exp": int(fp["exp"]),
                "fp_mant": int(fp["mant"]),
                "profiles": [],
            },
        )["profiles"].append(name)

    signatures: list[MatrixDatapathSignature] = []
    key_to_id: dict[str, str] = {}
    for key, item in grouped.items():
        readable = (
            f"w{_component_label(item['weight'])}"
            f"_a{_component_label(item['activation'])}"
            f"_fp{item['fp_exp']}m{item['fp_mant']}"
        )
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        signature_id = f"{readable}_{digest}"
        key_to_id[key] = signature_id
        signatures.append(
            MatrixDatapathSignature(
                signature_id=signature_id,
                weight=item["weight"],
                activation=item["activation"],
                internal_fp_exp=item["fp_exp"],
                internal_fp_mant=item["fp_mant"],
                profile_names=tuple(sorted(item["profiles"])),
            )
        )
    signatures.sort(key=lambda item: item.signature_id)
    return (
        tuple(signatures),
        {
            profile_name: key_to_id[key]
            for profile_name, key in profile_to_key.items()
        },
    )


def conditional_precision_variant_param_name(signature_id: str) -> str:
    """Return a stable Optuna parameter name for one signature's variants."""

    safe = re.sub(r"[^A-Za-z0-9]+", "_", signature_id).strip("_").upper()
    return f"PRECISION_VARIANT_{safe}"


def matrix_datapath_signature_distance(
    left: str,
    right: str,
    signatures: Mapping[str, MatrixDatapathSignature],
) -> float:
    """Distance based on PE ports and output conversion, not profile names."""

    if left == right:
        return 0.0
    lhs = signatures[str(left)]
    rhs = signatures[str(right)]
    family_penalty = (
        2.0 if lhs.weight["family"] != rhs.weight["family"] else 0.0
    ) + (
        2.0
        if lhs.activation["family"] != rhs.activation["family"]
        else 0.0
    )
    port_distance = (
        abs(lhs.weight_port_bits - rhs.weight_port_bits)
        + abs(lhs.activation_port_bits - rhs.activation_port_bits)
    ) / 8.0
    pe_distance = abs(lhs.pe_bit_product - rhs.pe_bit_product) / 64.0
    output_distance = abs(lhs.output_fp_bits - rhs.output_fp_bits) / 16.0
    format_distance = (
        abs(
            int(lhs.weight.get("exp", 0))
            - int(rhs.weight.get("exp", 0))
        )
        + abs(
            int(lhs.activation.get("exp", 0))
            - int(rhs.activation.get("exp", 0))
        )
    ) / 16.0
    return (
        family_penalty
        + port_distance
        + pe_distance
        + output_distance
        + format_distance
    )


def precision_variant_distance(
    left: str,
    right: str,
    profiles: Mapping[str, Mapping[str, Any]],
) -> float:
    """Distance within a Matrix signature: KV traffic plus accuracy."""

    if left == right:
        return 0.0
    lhs = profiles[str(left)]
    rhs = profiles[str(right)]
    lhs_kv = _parse_mx(lhs["KV_WIDTH"])
    rhs_kv = _parse_mx(rhs["KV_WIDTH"])
    distance = (
        2.0 if lhs_kv["family"] != rhs_kv["family"] else 0.0
    )
    distance += abs(int(lhs_kv["width"]) - int(rhs_kv["width"])) / 8.0
    distance += (
        abs(float(lhs["accuracy_score"]) - float(rhs["accuracy_score"]))
        * 10.0
    )
    return distance
