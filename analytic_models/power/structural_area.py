"""Content-addressed adapter for the validated structural area model."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

STRUCTURAL_AREA_SCHEMA = "plena-structural-area-evidence"
STRUCTURAL_AREA_MODEL_VERSION = "matrix_structural_census"
REFERENCE_ANCHOR_UM2 = 237_000.0
HOLDOUT_LIMIT_PCT = 10.0
ANCHOR_LIMIT_PCT = 1.0
FEATURE_NAMES = (
    "pe_tl",
    "pe_sum",
    "pe_0",
    "reduce",
    "scale",
    "out",
    "fixed",
    "const",
)
REQUIRED_AREA_SOURCES = frozenset(
    {
        "matrix_structural_coefficients.json",
        "asap7_sram_macro_table.csv",
        "matrix_machine_mxint.csv",
        "matrix_machine_mxfp.csv",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: object) -> bool:
    token = str(value)
    return len(token) == 64 and all(
        character in "0123456789abcdef" for character in token
    )


def _features(
    mlen: int,
    blen: int,
    t_width: int,
    l_width: int,
    scale_width: int,
) -> dict[str, float]:
    if mlen <= 0 or blen <= 0 or mlen % blen:
        raise ValueError("area geometry must satisfy MLEN>0, BLEN>0, MLEN%BLEN==0")
    return {
        "pe_tl": float(mlen * blen * t_width * l_width),
        "pe_sum": float(mlen * blen * (t_width + l_width)),
        "pe_0": float(mlen * blen),
        "reduce": float(blen * (mlen - blen)),
        "scale": float(mlen * scale_width),
        "out": float(blen * blen),
        "fixed": float(mlen // blen),
        "const": 1.0,
    }


def _format(token: str) -> tuple[str, int, int | None, int | None]:
    value = token.strip().upper()
    if value.startswith("MXINT"):
        bits = int(value.removeprefix("MXINT").lstrip("_"))
        if bits not in {2, 4, 8}:
            raise ValueError(f"unsupported structural MXINT format {token!r}")
        return "mxint", bits, None, None
    if value.startswith("MXFP_E") and "M" in value:
        exp_raw, mant_raw = value.removeprefix("MXFP_E").split("M", 1)
        exp, mant = int(exp_raw), int(mant_raw)
        if (exp, mant) not in {(1, 2), (2, 1), (4, 3), (5, 2)}:
            raise ValueError(f"unsupported structural MXFP format {token!r}")
        return "mxfp", 1 + exp + mant, exp, mant
    raise ValueError(f"unsupported structural format {token!r}")


def _signature_operands(signature: str) -> tuple[str, str, str]:
    try:
        operation, operands = signature.split(":", 1)
        left, right = operands.split("x", 1)
    except ValueError as exc:
        raise ValueError(f"invalid matrix signature {signature!r}") from exc
    if operation not in {"LINEAR", "QK", "PV"}:
        raise ValueError(f"invalid matrix operation {operation!r}")
    return operation, left, right


def _matrix_um2(
    coefficients: Mapping[str, Mapping[str, float]],
    *,
    family: str,
    mlen: int,
    blen: int,
    t_width: int,
    l_width: int,
    scale_width: int,
) -> float:
    try:
        model = coefficients[family]
    except KeyError as exc:
        raise ValueError(f"missing structural coefficients for {family!r}") from exc
    features = _features(mlen, blen, t_width, l_width, scale_width)
    area = sum(float(model[name]) * features[name] for name in FEATURE_NAMES)
    if not math.isfinite(area) or area <= 0:
        raise ValueError("structural matrix area must be positive and finite")
    return area


def _fp_width(vector_fp: str) -> int:
    token = vector_fp.strip().upper().removeprefix("FP_")
    if token == "BF16":
        return 16
    if not token.startswith("E") or "M" not in token:
        raise ValueError(f"unsupported vector format {vector_fp!r}")
    exp, mant = token[1:].split("M", 1)
    width = 1 + int(exp) + int(mant)
    if width <= 0:
        raise ValueError("vector width must be positive")
    return width


def _tile_area_um2(
    *,
    depth: int,
    width: int,
    ports: int,
    macros: Sequence[Mapping[str, Any]],
) -> float:
    if depth <= 0 or width <= 0 or ports <= 0 or not macros:
        raise ValueError("SRAM dimensions, ports, and macro table must be positive")
    return min(
        math.ceil(depth / int(macro["depth"]))
        * math.ceil(width / int(macro["width"]))
        * ports
        * float(macro["area_um2"])
        for macro in macros
    )


def _role_formats(
    signatures: Sequence[str],
) -> tuple[str, str, str, str]:
    roles: dict[str, str] = {}
    activations: set[str] = set()
    for signature in signatures:
        operation, left, right = _signature_operands(signature)
        activations.add(right)
        if operation == "LINEAR":
            roles["weight"] = left
        elif operation == "QK":
            roles["key"] = left
        else:
            roles["value"] = left
    if set(roles) != {"weight", "key", "value"} or len(activations) != 1:
        raise ValueError("area signatures must identify one W, A, K, and V format")
    return (
        roles["weight"],
        next(iter(activations)),
        roles["key"],
        roles["value"],
    )


def _normalise_area_config(
    config: Mapping[str, Any],
    *,
    mlen: int,
    blen: int,
) -> dict[str, int]:
    required = (
        "MLEN",
        "BLEN",
        "VLEN",
        "MATRIX_SRAM_DEPTH",
        "VECTOR_SRAM_DEPTH",
        "INT_SRAM_DEPTH",
        "FP_SRAM_DEPTH",
        "INT_DATA_WIDTH",
        "MX_SCALE_WIDTH",
        "BLOCK_DIM",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"area config is missing {', '.join(missing)}")
    out = {name: int(config[name]) for name in required}
    if any(value <= 0 for value in out.values()):
        raise ValueError("area config values must be positive")
    if out["MLEN"] != mlen or out["BLEN"] != blen:
        raise ValueError("area config and event geometry must match")
    if out["MLEN"] % out["BLEN"] or out["VLEN"] != out["MLEN"]:
        raise ValueError("area config has incompatible matrix/vector geometry")
    if out["BLOCK_DIM"] != 8:
        raise ValueError("area config requires native MX block size 8")
    return out


def _validate_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    failures: list[str] = []
    if payload.get("schema_version") != STRUCTURAL_AREA_SCHEMA:
        failures.append("schema_version")
    if payload.get("model_version") != STRUCTURAL_AREA_MODEL_VERSION:
        failures.append("model_version")
    source_hashes = dict(payload.get("source_sha256", {}))
    if set(source_hashes) != REQUIRED_AREA_SOURCES:
        failures.append("source_sha256")
    for name, digest in source_hashes.items():
        if not name or not _is_sha256(digest):
            failures.append(f"source_sha256:{name}")
    coefficients = payload.get("coefficients", {})
    if not isinstance(coefficients, Mapping):
        failures.append("coefficients")
        coefficients = {}
    for family in ("mxint", "mxfp"):
        model = coefficients.get(family, {})
        if not isinstance(model, Mapping) or set(model) != set(FEATURE_NAMES):
            failures.append(f"coefficients:{family}")
            continue
        try:
            values = tuple(float(model[name]) for name in FEATURE_NAMES)
        except (TypeError, ValueError):
            failures.append(f"coefficients:{family}")
            continue
        if any(not math.isfinite(value) or value < 0 for value in values):
            failures.append(f"coefficients:{family}")
        if not any(value > 0 for value in values):
            failures.append(f"coefficients:{family}")
    try:
        pdk_scale = float(payload["pdk_scale_reference"])
        reference_anchor = float(payload["reference_anchor_um2"])
    except (KeyError, TypeError, ValueError):
        failures.append("reference_corner")
        pdk_scale, reference_anchor = 0.0, 0.0
    if (
        not math.isfinite(pdk_scale)
        or pdk_scale <= 0
        or not math.isfinite(reference_anchor)
        or reference_anchor <= 0
    ):
        failures.append("reference_corner")
    holdouts = payload.get("holdout_mape_pct", {})
    if not isinstance(holdouts, Mapping):
        failures.append("holdout")
    else:
        for family in ("mxint", "mxfp"):
            try:
                value = float(holdouts[family])
            except (KeyError, TypeError, ValueError):
                failures.append(f"holdout:{family}")
                continue
            if not math.isfinite(value) or value < 0 or value > HOLDOUT_LIMIT_PCT:
                failures.append(f"holdout:{family}")
    macros = payload.get("sram_macros", ())
    if not isinstance(macros, Sequence) or isinstance(macros, (str, bytes)):
        failures.append("sram_macros")
        macros = ()
    for index, macro in enumerate(macros):
        if not isinstance(macro, Mapping):
            failures.append(f"sram_macro:{index}")
            continue
        try:
            depth = int(macro["depth"])
            width = int(macro["width"])
            area = float(macro["area_um2"])
        except (KeyError, TypeError, ValueError):
            failures.append(f"sram_macro:{index}")
            continue
        if (
            not str(macro.get("macro", "")).strip()
            or depth <= 0
            or width <= 0
            or not math.isfinite(area)
            or area <= 0
        ):
            failures.append(f"sram_macro:{index}")
    if not macros:
        failures.append("sram_macros")
    if not failures:
        try:
            anchor = _matrix_um2(
                coefficients,
                family="mxint",
                mlen=1024,
                blen=4,
                t_width=4,
                l_width=4,
                scale_width=8,
            ) * pdk_scale
            anchor_error = abs(anchor - reference_anchor) / reference_anchor * 100.0
            if anchor_error > ANCHOR_LIMIT_PCT:
                failures.append("anchor")
            shapes = ((16, 4), (32, 4), (64, 8), (256, 8), (1024, 4))
            areas = [
                _matrix_um2(
                    coefficients,
                    family="mxint",
                    mlen=shape[0],
                    blen=shape[1],
                    t_width=4,
                    l_width=4,
                    scale_width=8,
                )
                for shape in shapes
            ]
            if any(right <= left for left, right in zip(areas, areas[1:])):
                failures.append("monotonicity")
            low = _matrix_um2(
                coefficients,
                family="mxint",
                mlen=1024,
                blen=4,
                t_width=4,
                l_width=2,
                scale_width=8,
            )
            middle = _matrix_um2(
                coefficients,
                family="mxint",
                mlen=1024,
                blen=4,
                t_width=4,
                l_width=4,
                scale_width=8,
            )
            high = _matrix_um2(
                coefficients,
                family="mxint",
                mlen=1024,
                blen=4,
                t_width=8,
                l_width=8,
                scale_width=8,
            )
            if not low < middle < high:
                failures.append("precision_order")
        except (KeyError, TypeError, ValueError):
            failures.append("structural_evaluation")
    return tuple(sorted(set(failures)))


@dataclass(frozen=True)
class StructuralAreaEvidence:
    """Embedded coefficients, SRAM macros, validation, and source identities."""

    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        body = json.loads(_canonical_bytes(dict(self.payload)))
        object.__setattr__(self, "payload", body)

    @property
    def failures(self) -> tuple[str, ...]:
        return _validate_payload(self.payload)

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def evidence_id(self) -> str:
        return f"structural-area-{hashlib.sha256(_canonical_bytes(self.payload)).hexdigest()}"

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def matrix_area_mm2(
        self,
        signature: str,
        *,
        mlen: int,
        blen: int,
        reference_corner: bool,
        scale_width: int = 8,
    ) -> float:
        _, left, right = _signature_operands(signature)
        left_family, left_width, _, _ = _format(left)
        right_family, right_width, _, _ = _format(right)
        if left_family != right_family:
            raise ValueError("structural area does not support mixed-family operands")
        area = _matrix_um2(
            self.payload["coefficients"],
            family=left_family,
            mlen=mlen,
            blen=blen,
            t_width=left_width,
            l_width=right_width,
            scale_width=scale_width,
        )
        if reference_corner:
            area *= float(self.payload["pdk_scale_reference"])
        return area / 1e6

    def sram_area_mm2(
        self,
        signatures: Sequence[str],
        *,
        vector_fp: str,
        area_config: Mapping[str, Any],
        mlen: int,
        blen: int,
    ) -> float:
        config = _normalise_area_config(area_config, mlen=mlen, blen=blen)
        weight, activation, key, value = _role_formats(signatures)
        formats = [_format(token) for token in (weight, activation, key, value)]
        families = {item[0] for item in formats}
        if len(families) != 1:
            raise ValueError("structural SRAM area does not support mixed families")
        t_width = max(formats[index][1] for index in (0, 2, 3))
        act_width = formats[1][1]
        kv_width = max(formats[2][1], formats[3][1])
        scale_width = config["MX_SCALE_WIDTH"]
        matrix_width = mlen * (t_width + scale_width)
        vector_width = (
            config["VLEN"] * (_fp_width(vector_fp) + act_width + kv_width)
            + 2
            * scale_width
            * max(1, config["VLEN"] // config["BLOCK_DIM"])
        )
        macros = self.payload["sram_macros"]
        area = sum(
            (
                _tile_area_um2(
                    depth=config["MATRIX_SRAM_DEPTH"],
                    width=matrix_width,
                    ports=2,
                    macros=macros,
                ),
                _tile_area_um2(
                    depth=config["VECTOR_SRAM_DEPTH"],
                    width=vector_width,
                    ports=2,
                    macros=macros,
                ),
                _tile_area_um2(
                    depth=config["INT_SRAM_DEPTH"],
                    width=config["INT_DATA_WIDTH"],
                    ports=1,
                    macros=macros,
                ),
                _tile_area_um2(
                    depth=config["FP_SRAM_DEPTH"],
                    width=_fp_width(vector_fp),
                    ports=1,
                    macros=macros,
                ),
            )
        )
        return area / 1e6


def build_structural_area_evidence(
    coefficient_path: str | Path,
    macro_table_path: str | Path,
    *,
    calibration_inputs: Sequence[str | Path] = (),
) -> StructuralAreaEvidence:
    """Load and bind the existing structural-area calibration without refitting."""

    coefficient_source = Path(coefficient_path)
    macro_source = Path(macro_table_path)
    raw = json.loads(coefficient_source.read_text())
    with macro_source.open(newline="") as handle:
        macros = [
            {
                "macro": str(row["macro"]),
                "depth": int(row["depth"]),
                "width": int(row["width"]),
                "area_um2": float(row["area_um2"]),
            }
            for row in csv.DictReader(handle)
        ]
    source_hashes = {
        coefficient_source.name: _sha256(coefficient_source),
        macro_source.name: _sha256(macro_source),
    }
    for source in calibration_inputs:
        path = Path(source)
        source_hashes[path.name] = _sha256(path)
    reports = raw.get("report", {})
    payload = {
        "schema_version": STRUCTURAL_AREA_SCHEMA,
        "model_version": raw.get("model_version"),
        "reference_anchor_um2": raw.get("reference_anchor_um2"),
        "pdk_scale_reference": raw.get("pdk_scale_reference"),
        "coefficients": {
            family: {
                name: float(raw[family][name])
                for name in FEATURE_NAMES
            }
            for family in ("mxint", "mxfp")
        },
        "holdout_mape_pct": {
            family: float(reports[family]["holdout_mape_pct"])
            for family in ("mxint", "mxfp")
        },
        "sram_macros": macros,
        "source_sha256": dict(sorted(source_hashes.items())),
    }
    return StructuralAreaEvidence(payload)


__all__ = [
    "ANCHOR_LIMIT_PCT",
    "FEATURE_NAMES",
    "HOLDOUT_LIMIT_PCT",
    "REFERENCE_ANCHOR_UM2",
    "STRUCTURAL_AREA_MODEL_VERSION",
    "STRUCTURAL_AREA_SCHEMA",
    "StructuralAreaEvidence",
    "build_structural_area_evidence",
]
