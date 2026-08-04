"""Collect matched cross-stack cycle anchors into timing evidence.

Each anchor names a workload build directory. This tool runs the
transactional emulator on that build directory's
``generated_machine_code.mem`` instruction stream, prices the same program
with the production analytic timing path, and assembles the anchor CSV,
shared geometry and precision manifests, raw evidence receipts, and finally
the immutable timing-evidence artifact via ``build_timing_evidence``.

Two evidence tiers are supported. RTL-tier collection
(``--timing-mode rtl_serialized``) additionally requires each anchor's
``rtl_log`` from an RTL simulation run of the identical instruction stream
(its ``sim.log`` carries the C_BREAK cycle count). Emulator-tier collection
(``--timing-mode emulator_serialized``) assembles the anchor set from the
analytic and emulator sides only: anchors must not name an ``rtl_log``, the
CSV leaves the RTL columns empty, and no RTL provenance is recorded. Both
tiers price programs under the serialized issue contract.

Anchor kinds follow the calibration contract: ``linear``, ``qk``, ``pv`` and
``vector`` are operation anchors (single-operation or single-kernel
programs); ``layer`` anchors are whole decode-layer programs at consecutive
batched cache-append ordinals. A batched decode step appends one token per
sequence, so successive testbench programs at growing KV lengths are the
consecutive cache appends of the modelled serving loop; the declared
``cache_position`` is that append ordinal.

Example (emulator tier; RTL tier adds ``rtl_log=...`` per anchor and uses
``--timing-mode rtl_serialized``):

    python -m analytic_models.performance.collect_decode_anchors \
        --timing-mode emulator_serialized \
        --anchor kind=linear,build_dir=... \
        --anchor kind=qk,build_dir=... \
        --anchor kind=pv,build_dir=... \
        --anchor kind=vector,build_dir=... \
        --anchor kind=layer,cache_position=2,build_dir=... \
        --anchor kind=layer,cache_position=3,build_dir=... \
        --output-dir evidence/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from analytic_models.disagg_serve.memory import CALIBRATED_PIN_RATES_GBPS
from compiler.aten.execution_trace import (
    NO_DMA,
    CompilationArtifact,
    build_execution_trace,
)

try:
    from .decode_cost_model import DecodeCostModel
    from .decode_stage_validation import (
        DEFAULT_REQUEST_MEMORY_CALIBRATION,
        PICOSECONDS_PER_CYCLE,
        _fit_request_latency_model,
        sha256_file,
    )
    from .decode_timing import EMULATOR_SERIALIZED, RTL_SERIALIZED, TIMING_EVIDENCE_MODES
    from .perf_model import PerfModel, load_hardware_config_from_toml
    from .compiler_trace_timing import (
        HBMOperatingPoint,
        MATRIX_ENGINE,
        VECTOR_ENGINE,
        RequestModelStageMemoryPricer,
        request_memory_sidecar_from_compiler,
    )
except ImportError:
    from decode_cost_model import DecodeCostModel
    from decode_stage_validation import (
        DEFAULT_REQUEST_MEMORY_CALIBRATION,
        PICOSECONDS_PER_CYCLE,
        _fit_request_latency_model,
        sha256_file,
    )
    from decode_timing import EMULATOR_SERIALIZED, RTL_SERIALIZED, TIMING_EVIDENCE_MODES
    from perf_model import PerfModel, load_hardware_config_from_toml
    from compiler_trace_timing import (
        HBMOperatingPoint,
        MATRIX_ENGINE,
        VECTOR_ENGINE,
        RequestModelStageMemoryPricer,
        request_memory_sidecar_from_compiler,
    )

from transactional_emulator.testbench.emulator_runner import run_emulator

_REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_KINDS = ("linear", "qk", "pv", "vector", "layer")
ASM_NAME = "generated_asm_code.asm"
MACHINE_CODE_NAME = "generated_machine_code.mem"
COMPILATION_ARTIFACT_NAME = "compilation_artifact.json"
_RTL_CYCLES = re.compile(r"=== EXECUTION CLOCKS: (\d+) ===")


@dataclass(frozen=True)
class AnchorRequest:
    kind: str
    build_dir: Path
    rtl_log: Path | None
    cache_position: int | None
    batch: int | None


@dataclass(frozen=True)
class PricedProgram:
    compute_cycles: int
    matrix_memory_cycles: int
    vector_memory_cycles: int
    overlapped_cycles: int

    @property
    def memory_cycles(self) -> int:
        return self.matrix_memory_cycles + self.vector_memory_cycles


def _parse_anchor(value: str) -> AnchorRequest:
    fields = dict(part.split("=", 1) for part in value.split(","))
    kind = fields.pop("kind")
    if kind not in ANCHOR_KINDS:
        raise ValueError(f"unknown anchor kind {kind!r}")
    build_dir = Path(fields.pop("build_dir")).resolve()
    rtl_log = fields.pop("rtl_log", None)
    cache_position = fields.pop("cache_position", None)
    batch = fields.pop("batch", None)
    if fields:
        raise ValueError(f"unknown anchor fields: {sorted(fields)}")
    if kind == "layer" and cache_position is None:
        raise ValueError("layer anchors require cache_position")
    return AnchorRequest(
        kind=kind,
        build_dir=build_dir,
        rtl_log=Path(rtl_log).resolve() if rtl_log else None,
        cache_position=int(cache_position) if cache_position else None,
        batch=int(batch) if batch else None,
    )


def parse_rtl_cycles(log_path: Path) -> int:
    matches = _RTL_CYCLES.findall(log_path.read_text(errors="replace"))
    if not matches:
        raise ValueError(f"{log_path} carries no C_BREAK cycle count")
    return int(matches[-1])


def price_program(
    build_dir: Path,
    *,
    settings_path: Path,
    isa_path: Path,
    request_memory_calibration: Path,
    hbm_generation: str,
    hbm_channels: int,
    mlen: int,
    blen: int,
    hlen: int,
    timing_mode: str,
) -> PricedProgram:
    """Price one program with the production compiler-trace timing path.

    Compute comes from the decode cost model over the structured execution
    trace; DMA time comes from the request-level memory model over the
    address-resolved compiler sidecar. DMA-free kernels price as pure
    compute. The overlapped total composes per trace stage as
    ``max(compute, matrix DMA) + vector DMA``.
    """

    hardware = load_hardware_config_from_toml(str(settings_path)).model_copy(
        update={
            "MLEN": mlen,
            "VLEN": mlen,
            "BLEN": blen,
            "HLEN": hlen,
            "BROADCAST_AMOUNT": mlen // hlen,
        }
    )
    perf = PerfModel(hardware, str(isa_path), timing_mode=timing_mode)
    assembly = (build_dir / ASM_NAME).read_text(encoding="utf-8")
    artifact_path = build_dir / COMPILATION_ARTIFACT_NAME
    artifact: CompilationArtifact | None = None
    if artifact_path.is_file():
        artifact = CompilationArtifact.from_dict(
            json.loads(artifact_path.read_text(encoding="utf-8"))
        )
        if artifact.assembly != assembly:
            raise ValueError(
                f"{artifact_path} does not bind {build_dir / ASM_NAME}"
            )
        trace = artifact.execution_trace
    else:
        vector_store_amount = int(
            getattr(
                hardware,
                "HBM_V_Writeback_Amount",
                hardware.HBM_V_Prefetch_Amount,
            )
        )
        trace = build_execution_trace(
            assembly,
            mlen=mlen,
            blen=blen,
            vlen=mlen,
            hlen=hlen,
            vector_prefetch_amount=hardware.HBM_V_Prefetch_Amount,
            vector_store_amount=vector_store_amount,
        )

    has_dma = any(entry.dma_direction != NO_DMA for entry in trace.entries)
    if has_dma and (artifact is None or artifact.request_memory is None):
        raise RuntimeError(
            f"{build_dir} program issues DMA but has no address-resolved "
            f"{COMPILATION_ARTIFACT_NAME}"
        )

    compute_cost = DecodeCostModel.from_perf_model(perf).evaluate(trace)
    compute_by_stage = {
        stage.stage: stage.compute_cycles for stage in compute_cost.stages
    }
    matrix_by_stage: dict[str, int] = {}
    vector_by_stage: dict[str, int] = {}
    if has_dma:
        generation = hbm_generation.upper()
        hbm = HBMOperatingPoint(
            generation=generation,
            channels=hbm_channels,
            pin_rate_gbps=CALIBRATED_PIN_RATES_GBPS[generation],
        )
        assert artifact is not None and artifact.request_memory is not None
        sidecar = request_memory_sidecar_from_compiler(
            artifact.request_memory,
            hbm,
        )
        sidecar.validate(trace, hbm)
        pricer = RequestModelStageMemoryPricer(
            _fit_request_latency_model(
                str(request_memory_calibration.resolve()),
                sha256_file(request_memory_calibration),
            )
        )
        frequency_hz = 1e12 / PICOSECONDS_PER_CYCLE
        for stage, engines in pricer.price_trace_by_engine(
            trace,
            sidecar,
            hbm,
        ).items():
            matrix_by_stage[stage] = math.ceil(
                engines.get(MATRIX_ENGINE, 0.0) * frequency_hz
            )
            vector_by_stage[stage] = math.ceil(
                engines.get(VECTOR_ENGINE, 0.0) * frequency_hz
            )

    overlapped = sum(
        max(compute_by_stage.get(stage, 0), matrix_by_stage.get(stage, 0))
        + vector_by_stage.get(stage, 0)
        for stage in trace.stage_order
    )
    return PricedProgram(
        compute_cycles=sum(compute_by_stage.values()),
        matrix_memory_cycles=sum(matrix_by_stage.values()),
        vector_memory_cycles=sum(vector_by_stage.values()),
        overlapped_cycles=overlapped,
    )


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def _compiler_identity(output_dir: Path) -> Path:
    """One content-derived compiler identity file shared by every anchor."""

    compiler_root = _REPO_ROOT / "compiler" / "aten"
    entries = {}
    for source in sorted(compiler_root.rglob("*.py")):
        if "__pycache__" in source.parts or "tests" in source.parts:
            continue
        entries[source.relative_to(compiler_root).as_posix()] = (
            hashlib.sha256(source.read_bytes()).hexdigest()
        )
    return _write_json(
        output_dir / "compiler_identity.json",
        {"schema": "plena-compiler-identity", "sources": entries},
    )


def _precision_manifest(settings_path: Path, output_dir: Path) -> Path:
    with settings_path.open("rb") as handle:
        settings = tomllib.load(handle)
    transactional = settings["TRANSACTIONAL"]
    precision = {
        name: value
        for name, value in transactional.items()
        if "TYPE" in name or "WIDTH" in name or "PRECISION" in name
    }
    return _write_json(
        output_dir / "precision_manifest.json",
        {"schema": "plena-transactional-precision", "precision": precision},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--settings", type=Path, default=_REPO_ROOT / "plena_settings.toml"
    )
    parser.add_argument(
        "--isa-lib",
        type=Path,
        default=Path(__file__).resolve().parent / "customISA_lib.json",
    )
    parser.add_argument(
        "--request-memory-calibration",
        type=Path,
        default=DEFAULT_REQUEST_MEMORY_CALIBRATION,
    )
    parser.add_argument("--hbm-gen", default="HBM2")
    parser.add_argument("--hbm-channels", type=int, default=8)
    parser.add_argument("--mlen", type=int, default=64)
    parser.add_argument("--blen", type=int, default=4)
    parser.add_argument("--hlen", type=int, default=16)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument(
        "--timing-mode",
        default=RTL_SERIALIZED,
        choices=TIMING_EVIDENCE_MODES,
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="timing-evidence output path"
    )
    args = parser.parse_args()

    emulator_tier = args.timing_mode == EMULATOR_SERIALIZED
    # Both tiers price under the serialized issue contract; the evidence
    # mode records which measurement reference backs the anchors.
    pricing_mode = RTL_SERIALIZED if emulator_tier else args.timing_mode
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    requests = [_parse_anchor(value) for value in args.anchor]
    observed_kinds = {request.kind for request in requests}
    missing = sorted(set(ANCHOR_KINDS) - observed_kinds)
    if missing:
        raise SystemExit(f"anchor kinds missing: {missing}")
    for request in requests:
        if emulator_tier and request.rtl_log is not None:
            raise SystemExit(
                f"{request.kind}: emulator-tier anchors must not name an rtl_log"
            )
        if not emulator_tier and request.rtl_log is None:
            raise SystemExit(f"{request.kind}: RTL-tier anchors require rtl_log")

    geometry_path = _write_json(
        output_dir / "geometry_manifest.json",
        {
            "schema": "plena-anchor-geometry",
            "mlen": args.mlen,
            "blen": args.blen,
            "hlen": args.hlen,
            "vlen": args.mlen,
        },
    )
    precision_path = _precision_manifest(args.settings.resolve(), output_dir)
    compiler_identity = _compiler_identity(output_dir)

    rows = []
    analytic_receipt = {}
    emulator_receipt = {}
    rtl_receipt = {}
    for index, request in enumerate(requests):
        anchor_id = (
            f"{request.kind}-{index}"
            if request.kind != "layer"
            else f"layer-p{request.cache_position}"
        )
        anchor_dir = output_dir / anchor_id
        anchor_dir.mkdir(exist_ok=True)
        asm_path = shutil.copy2(
            request.build_dir / ASM_NAME, anchor_dir / ASM_NAME
        )
        trace_path = shutil.copy2(
            request.build_dir / MACHINE_CODE_NAME,
            anchor_dir / MACHINE_CODE_NAME,
        )

        rtl_cycles: int | None = None
        if not emulator_tier:
            rtl_cycles = parse_rtl_cycles(request.rtl_log)
            rtl_receipt[anchor_id] = {
                "log": request.rtl_log.name,
                "log_sha256": sha256_file(request.rtl_log),
                "cycles": rtl_cycles,
            }

        metrics = run_emulator(request.build_dir)
        emulator_cycles = int(round(float(metrics["sim_latency_ns"])))
        emulator_receipt[anchor_id] = {
            "sim_latency_ns": metrics["sim_latency_ns"],
            "hbm_bytes_read": metrics.get("hbm_bytes_read"),
            "hbm_bytes_written": metrics.get("hbm_bytes_written"),
            "machine_code_sha256": sha256_file(Path(trace_path)),
        }

        priced = price_program(
            request.build_dir,
            settings_path=args.settings.resolve(),
            isa_path=args.isa_lib.resolve(),
            request_memory_calibration=args.request_memory_calibration,
            hbm_generation=args.hbm_gen,
            hbm_channels=args.hbm_channels,
            mlen=args.mlen,
            blen=args.blen,
            hlen=args.hlen,
            timing_mode=pricing_mode,
        )
        analytic_receipt[anchor_id] = {
            "compute_cycles": priced.compute_cycles,
            "matrix_memory_cycles": priced.matrix_memory_cycles,
            "vector_memory_cycles": priced.vector_memory_cycles,
            "overlapped_cycles": priced.overlapped_cycles,
        }

        row = {
            "anchor_id": anchor_id,
            "anchor_kind": request.kind,
            "emulator_cycles": emulator_cycles,
            "rtl_cycles": rtl_cycles if rtl_cycles is not None else "",
            "mlen": args.mlen,
            "blen": args.blen,
            "hlen": args.hlen,
            "vlen": args.mlen,
            "geometry_path": str(
                Path(geometry_path).relative_to(output_dir)
            ),
            "precision_path": str(
                Path(precision_path).relative_to(output_dir)
            ),
            "asm_path": str(Path(asm_path).relative_to(output_dir)),
            "analytical_trace_path": str(
                Path(trace_path).relative_to(output_dir)
            ),
            "emulator_trace_path": str(
                Path(trace_path).relative_to(output_dir)
            ),
            "rtl_trace_path": (
                ""
                if emulator_tier
                else str(Path(trace_path).relative_to(output_dir))
            ),
        }
        if request.kind == "layer":
            hbm_bytes = int(metrics.get("hbm_bytes_read", 0)) + int(
                metrics.get("hbm_bytes_written", 0)
            )
            if hbm_bytes <= 0:
                raise SystemExit(
                    f"{anchor_id}: layer anchors require measured HBM bytes"
                )
            row |= {
                "analytical_cycles": max(
                    priced.compute_cycles, priced.memory_cycles
                ),
                "analytical_compute_cycles": priced.compute_cycles,
                "analytical_memory_cycles": priced.memory_cycles,
                "cache_position": request.cache_position,
                "batch": request.batch or args.batch,
                "physical_hbm_bytes": hbm_bytes,
            }
        else:
            row |= {
                "analytical_cycles": priced.overlapped_cycles,
                "analytical_compute_cycles": "",
                "analytical_memory_cycles": "",
                "cache_position": "",
                "batch": "",
                "physical_hbm_bytes": "",
            }
        rows.append(row)

    analytic_path = _write_json(
        output_dir / "analytic_cycles.json",
        {
            "schema": "plena-anchor-analytic-cycles",
            "timing_mode": args.timing_mode,
            "anchors": analytic_receipt,
        },
    )
    emulator_path = _write_json(
        output_dir / "emulator_runs.json",
        {"schema": "plena-anchor-emulator-runs", "anchors": emulator_receipt},
    )
    rtl_path = None
    if not emulator_tier:
        rtl_path = _write_json(
            output_dir / "rtl_runs.json",
            {"schema": "plena-anchor-rtl-runs", "anchors": rtl_receipt},
        )

    field_order = (
        "anchor_id",
        "anchor_kind",
        "analytical_cycles",
        "analytical_compute_cycles",
        "analytical_memory_cycles",
        "cache_position",
        "batch",
        "physical_hbm_bytes",
        "emulator_cycles",
        "rtl_cycles",
        "mlen",
        "blen",
        "hlen",
        "vlen",
        "geometry_path",
        "precision_path",
        "asm_path",
        "analytical_trace_path",
        "emulator_trace_path",
        "rtl_trace_path",
    )
    anchors_csv = output_dir / "decode_cycle_anchors.csv"
    with anchors_csv.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(field_order) + "\n")
        for row in rows:
            handle.write(
                ",".join(str(row[field]) for field in field_order) + "\n"
            )

    out_path = (
        args.out.resolve()
        if args.out is not None
        else output_dir / "decode_timing_evidence.json"
    )
    command = [
        sys.executable,
        "-m",
        "analytic_models.performance.build_timing_evidence",
        "--mode",
        args.timing_mode,
        "--anchors",
        str(anchors_csv),
        "--provenance",
        f"compiler={compiler_identity}",
        "--provenance",
        f"analytic={analytic_path}",
        "--provenance",
        f"emulator={emulator_path}",
    ]
    if rtl_path is not None:
        command += ["--provenance", f"rtl={rtl_path}"]
    command += ["--out", str(out_path)]
    completed = subprocess.run(command, cwd=_REPO_ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
