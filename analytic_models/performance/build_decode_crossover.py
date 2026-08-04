"""Build a provenance-bound batch-by-context decode crossover artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from .decode_crossover import DecodeCrossoverStudy
    from .disagg_decode import (
        EXTERNAL_BF16_HEAD,
        OUTPUT_HEAD_LOCATIONS,
        TimingEvidence,
        decode_crossover_point,
        effective_bits,
        element_bits,
        evaluate,
        hbm_overrides,
        load_hardware_config_from_toml,
        load_memory_config_from_toml,
        load_model_dims,
        precision_from_components,
        width_label,
        _parse_width,
    )
except ImportError:
    from decode_crossover import DecodeCrossoverStudy
    from disagg_decode import (
        EXTERNAL_BF16_HEAD,
        OUTPUT_HEAD_LOCATIONS,
        TimingEvidence,
        decode_crossover_point,
        effective_bits,
        element_bits,
        evaluate,
        hbm_overrides,
        load_hardware_config_from_toml,
        load_memory_config_from_toml,
        load_model_dims,
        precision_from_components,
        width_label,
        _parse_width,
    )


def _positive_list(value: str, name: str) -> tuple[int, ...]:
    values = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _precision(args: argparse.Namespace) -> dict:
    weight = _parse_width(args.w_fmt, args.weight)
    activation = _parse_width(args.a_fmt, args.activation)
    kv = _parse_width(args.kv_fmt, args.kv)
    weight_element_bits = element_bits(args.w_fmt, weight)
    activation_element_bits = element_bits(args.a_fmt, activation)
    kv_element_bits = element_bits(args.kv_fmt, kv)
    return precision_from_components(
        effective_bits(args.w_fmt, weight, args.block),
        effective_bits(args.w_fmt, weight, args.block),
        effective_bits(args.kv_fmt, kv, args.block),
        width_label(args.w_fmt, weight),
        width_label(args.w_fmt, weight),
        width_label(args.kv_fmt, kv),
        attn_elem=weight_element_bits,
        ffn_elem=weight_element_bits,
        kv_elem=kv_element_bits,
        m_bits=max(
            weight_element_bits,
            activation_element_bits,
            kv_element_bits,
        ),
        block_size=args.block,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-json", required=True)
    parser.add_argument("--settings", required=True)
    parser.add_argument("--isa", required=True)
    parser.add_argument("--timing-evidence", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--contexts", default="512,4096,8192,16384,32768")
    parser.add_argument("--batches", default="1,4,8,16,32,64,128,256")
    parser.add_argument("--w-fmt", choices=("mxint", "mxfp"), default="mxint")
    parser.add_argument("--a-fmt", choices=("mxint", "mxfp"), default="mxint")
    parser.add_argument("--kv-fmt", choices=("mxint", "mxfp"), default="mxint")
    parser.add_argument("--weight", default="4")
    parser.add_argument("--activation", default="4")
    parser.add_argument("--kv", default="4")
    parser.add_argument("--block", type=int, choices=(8,), default=8)
    parser.add_argument(
        "--hbm-gen",
        choices=("HBM2", "HBM2E", "HBM3", "HBM3E", "HBM4"),
        default="HBM2",
    )
    parser.add_argument(
        "--hbm-channels",
        type=int,
        default=16,
        help="64-bit interface units; 16 units form a 1024-bit stack",
    )
    parser.add_argument("--chips", type=int, default=1)
    parser.add_argument("--runtime-hbm-reserve-bytes", type=int, default=0)
    parser.add_argument(
        "--bandwidth-mode",
        choices=("calibrated", "peak"),
        default="calibrated",
    )
    parser.add_argument(
        "--output-head-location",
        choices=sorted(OUTPUT_HEAD_LOCATIONS),
        default=EXTERNAL_BF16_HEAD,
    )
    parser.add_argument("--require-rankable", action="store_true")
    args = parser.parse_args()

    for name in ("hbm_channels", "chips"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.runtime_hbm_reserve_bytes < 0:
        raise ValueError("runtime HBM reserve must be non-negative")

    model_path = Path(args.model_json).resolve()
    settings_path = Path(args.settings).resolve()
    isa_path = Path(args.isa).resolve()
    timing_path = Path(args.timing_evidence).resolve()
    for path in (model_path, settings_path, isa_path, timing_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    dimensions = load_model_dims(str(model_path))
    hardware = load_hardware_config_from_toml(str(settings_path))
    memory = load_memory_config_from_toml(str(settings_path))
    hbm = hbm_overrides(args.hbm_gen, args.hbm_channels)
    resolved_channels = int(hbm.pop("channels"))
    hardware = hardware.model_copy(update=hbm)
    memory = memory.model_copy(update=hbm)
    evidence = TimingEvidence.load(timing_path)
    precision = _precision(args)

    bandwidth = None
    if args.bandwidth_mode == "calibrated":
        try:
            from ..disagg_serve.memory import CalibratedBandwidth
        except ImportError:
            from analytic_models.disagg_serve.memory import CalibratedBandwidth
        bandwidth = CalibratedBandwidth.load()

    points = []
    for context in _positive_list(args.contexts, "contexts"):
        for batch in _positive_list(args.batches, "batches"):
            result = evaluate(
                str(model_path),
                dimensions,
                hardware,
                str(isa_path),
                memory,
                precision,
                batch,
                context,
                1,
                stride=1,
                n_chips=args.chips,
                bw_model=bandwidth,
                hbm_gen=args.hbm_gen,
                hbm_channels=resolved_channels,
                timing_mode=evidence.mode,
                timing_evidence=evidence,
                runtime_hbm_reserve_bytes=args.runtime_hbm_reserve_bytes,
                output_head_location=args.output_head_location,
            )
            points.append(
                decode_crossover_point(
                    result,
                    context=context,
                    batch=batch,
                )
            )

    study = DecodeCrossoverStudy.from_points(points)
    if args.require_rankable and not study.rankable:
        raise RuntimeError("crossover artifact failed timing, bandwidth, or capacity gates")

    source_dir = Path(__file__).resolve().parent
    payload = {
        **study.to_dict(),
        "metric_scope": (
            "decode_body_only"
            if args.output_head_location == EXTERNAL_BF16_HEAD
            else "decode_with_local_bf16_head_sensitivity"
        ),
        "configuration": {
            "precision": precision,
            "hbm_generation": args.hbm_gen,
            "hbm_interface_units": resolved_channels,
            "chip_count": args.chips,
            "bandwidth_mode": args.bandwidth_mode,
            "output_head_location": args.output_head_location,
            "runtime_hbm_reserve_bytes": args.runtime_hbm_reserve_bytes,
        },
        "provenance": {
            "model_json_sha256": _sha256(model_path),
            "settings_sha256": _sha256(settings_path),
            "isa_sha256": _sha256(isa_path),
            "timing_evidence_sha256": _sha256(timing_path),
            "disagg_decode_sha256": _sha256(source_dir / "disagg_decode.py"),
            "decode_crossover_sha256": _sha256(
                source_dir / "decode_crossover.py"
            ),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["artifact_id"] = (
        "decode-crossover-"
        + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    )

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(output)
    print(payload["artifact_id"])
    print(f"rankable={study.rankable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
