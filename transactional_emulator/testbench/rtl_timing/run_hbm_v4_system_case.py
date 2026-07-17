#!/usr/bin/env python3
"""Compile and validate one production-DMA HBM V4 system holdout.

Each case loads only decoder layer 0 from the local Hugging Face cache, emits
the native PLENA ISA, runs the transactional emulator with a production DMA
event trace for numerical regression, then validates V4 with an arrival-aware
production-DMA fixed-point replay.  The existing numerical comparison is
recorded but its thresholds are never changed or used to accept the latency
model.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any

import tomlkit
import torch


ROOT = Path(__file__).resolve().parents[3]
for dependency in (ROOT, ROOT / "PLENA_Compiler", ROOT / "PLENA_Tools"):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from transactional_emulator.testbench.emulator_runner import (  # noqa: E402
    _write_comparison_summary,
    compare_emulator_output,
    run_emulator,
)
from transactional_emulator.testbench.sim_env_utils import (  # noqa: E402
    create_mem_for_sim,
)
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402
from Workspace.qwen3_32b_transactional_prefetch_sweep.run_prefetch_dse import (  # noqa: E402
    BaseHardware,
    PrefetchPoint,
    compile_trial,
    load_layer_tensors,
    make_lightweight_decoder_model,
    write_trial_toml,
)


DEFAULT_CALIBRATION = (
    ROOT / "analytic_models/performance/calibration/hbm_dma_service_v4.json"
)
ARRIVAL_REPLAY_RUNNER = (
    ROOT
    / "transactional_emulator/testbench/rtl_timing/run_hbm_v4_arrival_replay.py"
)

RESUME_ARTIFACTS = (
    "plena_settings.toml",
    "model_config.json",
    "generated_machine_code.mem",
    "hbm_for_behave_sim.bin",
    "fp_sram.bin",
    "int_sram.bin",
    "golden_result.txt",
    "comparison_params.json",
)


@dataclass(frozen=True)
class SystemCase:
    name: str
    model_id: str
    seq_len: int
    mlen: int
    blen: int
    hbm_channels: int
    precision: str
    block: int


CASES = {
    case.name: case
    for case in (
        SystemCase(
            "qwen3_8b_seq64_m128_b16_mxint4_c128",
            "Qwen/Qwen3-8B",
            64,
            128,
            16,
            128,
            "MXINT4",
            64,
        ),
        SystemCase(
            "qwen3_8b_seq128_m256_b32_mxint8_c32",
            "Qwen/Qwen3-8B",
            128,
            256,
            32,
            32,
            "MXINT8",
            64,
        ),
        SystemCase(
            "qwen3_32b_seq128_m256_b64_mxfp_e1m2_c8",
            "Qwen/Qwen3-32B",
            128,
            256,
            64,
            8,
            "MXFP_E1M2",
            64,
        ),
    )
}


def _mx_section(name: str, block: int) -> dict[str, Any]:
    if name.startswith("MXINT"):
        element = {"type": "Int", "width": int(name.removeprefix("MXINT"))}
    elif name.startswith("MXFP_E") and "M" in name:
        exp_text, mant_text = name.removeprefix("MXFP_E").split("M", 1)
        element = {
            "type": "Fp",
            "sign": True,
            "exponent": int(exp_text),
            "mantissa": int(mant_text),
        }
    else:
        raise ValueError(f"unsupported system-validation precision {name!r}")
    return {
        "format": "Mx",
        "block": block,
        "ELEM": element,
        "SCALE": {
            "type": "Fp",
            "sign": False,
            "exponent": 8,
            "mantissa": 0,
        },
    }


def _precision_config(case: SystemCase) -> dict[str, Any]:
    return {
        "weight": case.precision,
        "activation": case.precision,
        "kv": case.precision,
        "block": case.block,
        "scale_bits": 8,
        "integer_bits": 32,
        "matrix_internal_fp": {"exp": 8, "mant": 7},
        "vector_internal_fp": {"exp": 8, "mant": 7},
        "scalar_fp": {"exp": 8, "mant": 7},
    }


def patch_precision(path: Path, case: SystemCase) -> None:
    with path.open(encoding="utf-8") as handle:
        data = tomlkit.load(handle)
    section = data["TRANSACTIONAL"].setdefault("PRECISION", {})
    mx = _mx_section(case.precision, case.block)
    for name in (
        "HBM_M_WEIGHT_TYPE",
        "HBM_M_KV_TYPE",
        "HBM_V_ACT_TYPE",
        "HBM_V_KV_TYPE",
    ):
        section[name] = copy.deepcopy(mx)
    fp = {
        "format": "Plain",
        "DATA_TYPE": {
            "type": "Fp",
            "sign": True,
            "exponent": 8,
            "mantissa": 7,
        },
    }
    section["MATRIX_SRAM_TYPE"] = copy.deepcopy(fp)
    section["VECTOR_SRAM_TYPE"] = copy.deepcopy(fp)
    section["SCALAR_FP"] = copy.deepcopy(fp["DATA_TYPE"])
    section["HBM_V_INT_TYPE"] = {
        "format": "Plain",
        "DATA_TYPE": {"type": "Int", "width": 32},
    }
    with path.open("w", encoding="utf-8") as handle:
        tomlkit.dump(data, handle)


def write_model_config(path: Path, config: Any, case: SystemCase) -> None:
    num_heads = int(config.num_attention_heads)
    hidden_size = int(config.hidden_size)
    payload = {
        "source_model_id": case.model_id,
        "model_name": case.model_id.rsplit("/", 1)[-1].lower(),
        "model_type": str(getattr(config, "model_type", "qwen3")),
        "architecture_family": "qwen3_dense_llama_style",
        "hidden_size": hidden_size,
        "intermediate_size": int(config.intermediate_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "num_attention_heads": num_heads,
        "num_key_value_heads": int(config.num_key_value_heads),
        "head_dim": int(getattr(config, "head_dim", hidden_size // num_heads)),
        "vocab_size": int(config.vocab_size),
        "rope_theta": float(getattr(config, "rope_theta", 1_000_000.0)),
        "rms_norm_eps": float(getattr(config, "rms_norm_eps", 1e-6)),
        "tie_word_embeddings": bool(getattr(config, "tie_word_embeddings", False)),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.int16).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def create_emulator_artifacts(result: dict[str, Any], output: Path) -> None:
    (output / "compile_memory_layout.json").write_text(
        json.dumps(
            {
                "data_order": list(result["data_order"]),
                "hbm_addrs": result.get("hbm_addrs", {}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    create_sim_env(
        result.get("input_tensors", {}),
        result["isa"],
        result.get(
            "golden_result", {"original_output": result.get("golden_output")}
        ),
        result["fp_preload"],
        build_dir=str(output),
        tensor_layouts=result.get("tensor_layouts"),
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=output.name,
        data=None,
        specified_data_order=result["data_order"],
        build_path=output,
        input_tensors=result.get("input_tensors"),
        tensor_layouts=result.get("tensor_layouts"),
        hbm_addrs=result.get("hbm_addrs"),
    )
    (output / "comparison_params.json").write_text(
        json.dumps(result["comparison_params"], indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "generated_asm_code.asm").write_text(
        result["isa"], encoding="utf-8"
    )


def cleanup_heavy_artifacts(output: Path) -> list[str]:
    removed = []
    explicit = {
        "generated_asm_code.asm",
        "generated_machine_code.mem",
        "hbm_for_behave_sim.bin",
        "vram_preload.bin",
        "golden_output.pt",
    }
    for path in output.iterdir():
        if path.name in explicit or path.suffix == ".pt":
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)
                removed.append(path.name)
    for name in ("vram_dump.bin", "mram_dump.bin", "fpsram_dump.bin"):
        (ROOT / "transactional_emulator" / name).unlink(missing_ok=True)
        (output / name).unlink(missing_ok=True)
    return sorted(removed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(CASES), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--arrival-replay-iterations", type=int, default=3)
    parser.add_argument("--arrival-replay-cycle-tolerance", type=int, default=0)
    parser.add_argument("--keep-heavy-artifacts", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a complete compile/emulator-input artifact set in --out-dir.",
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case = CASES[args.case]
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "system_validation.json").exists():
        raise FileExistsError(f"completed output already exists at {output}")

    hardware = BaseHardware(
        mlen=case.mlen,
        vlen=case.mlen,
        blen=case.blen,
        hlen=128,
        broadcast=8,
        mram_tile_capacity=16,
    )
    prefetch = PrefetchPoint(case.mlen, case.blen, case.blen)
    settings_path = output / "plena_settings.toml"
    model_config_path = output / "model_config.json"
    if args.resume:
        missing = [name for name in RESUME_ARTIFACTS if not (output / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "cannot resume because required artifacts are missing: "
                + ", ".join(missing)
            )
        print(f"Reusing compile and emulator input artifacts from {output}")
    else:
        settings_path = write_trial_toml(
            output,
            hardware,
            prefetch,
            case.hbm_channels,
            1_000_000,
            4_194_304,
        )
        patch_precision(settings_path, case)

        config, tensors = load_layer_tensors(
            case.model_id, 0, local_files_only=args.local_files_only
        )
        write_model_config(model_config_path, config, case)
        model = make_lightweight_decoder_model(config, tensors)
        compile_result = compile_trial(
            model,
            output,
            hardware,
            seq_len=case.seq_len,
            batch_size=1,
            seed=42,
        )
        del tensors, model
        gc.collect()
        create_emulator_artifacts(compile_result, output)
        del compile_result
        gc.collect()

    os.environ["PLENA_SETTINGS_TOML"] = str(settings_path)
    precision_config_path = output / "precision_config.json"
    precision_config_path.write_text(
        json.dumps(_precision_config(case), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    dma_trace = output / "dma_event_trace.json"
    run_metrics = run_emulator(
        output,
        hbm_channels=case.hbm_channels,
        threads=args.threads,
        profile_memory=True,
        profile_memory_level="opcode",
        timing_mode="rtl-v1",
        dma_event_trace=dma_trace,
        require_rtl_validated=False,
    )
    comparison, params = compare_emulator_output(output)
    _write_comparison_summary(output, comparison, params)
    numerical = {
        "correctness_gate_modified": False,
        "test_pass": bool(
            comparison.get("test_pass", comparison.get("allclose_pass", False))
        ),
        "match_rate": comparison.get("match_rate"),
        "allclose_match_rate": comparison.get("allclose_match_rate"),
        "simulated_output_sha256_bf16": tensor_sha256(
            comparison["simulated_values"]
        ),
    }

    # The functional executor advances Ramulator on a serial numerical time
    # axis, so its DMA event trace is useful for debugging request semantics
    # but is not valid rtl-v1 arrival evidence.  Formal latency acceptance is
    # delegated to the fixed-point replay that drives the same production DMA
    # requests at scheduler-derived absolute arrival cycles.
    arrival_output = output / "arrival_replay"
    arrival_command = [
        sys.executable,
        str(ARRIVAL_REPLAY_RUNNER),
        "--model-config",
        str(model_config_path),
        "--settings",
        str(settings_path),
        "--calibration",
        str(args.calibration),
        "--precision-config",
        str(precision_config_path),
        "--seq-len",
        str(case.seq_len),
        "--batch-size",
        "1",
        "--num-layers",
        "1",
        "--out-dir",
        str(arrival_output),
        "--max-iterations",
        str(args.arrival_replay_iterations),
        "--convergence-cycle-tolerance",
        str(args.arrival_replay_cycle_tolerance),
        "--resume",
    ]
    arrival_result = subprocess.run(arrival_command, check=False)
    arrival_validation = arrival_output / "system_validation.json"
    if not arrival_validation.is_file():
        raise RuntimeError(
            "arrival-aware HBM validation did not produce system_validation.json "
            f"(exit={arrival_result.returncode})"
        )
    validation = json.loads(arrival_validation.read_text())
    validation["case"] = asdict(case)
    validation["numerical_evidence"] = numerical
    validation["emulator_latency_ns"] = run_metrics.get("sim_latency_ns")
    validation["functional_dma_trace"] = {
        "path": str(dma_trace),
        "timing_semantics": (
            "functional-executor-service-interval-replayed-on-rtl-v1"
        ),
        "used_for_latency_acceptance": False,
    }
    (output / "system_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    removed = []
    if validation["accepted"] and not args.keep_heavy_artifacts:
        removed = cleanup_heavy_artifacts(output)
    summary = {
        "case": asdict(case),
        "accepted": validation["accepted"],
        "numerical_evidence": numerical,
        "removed_heavy_artifacts": removed,
        "system_validation": str(output / "system_validation.json"),
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if validation["accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
