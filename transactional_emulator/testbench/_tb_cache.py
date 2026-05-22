"""Shared testbench helpers — MX-E4M3 round-trip + fingerprint cache.

Factored out of ``tvm_flash_attention_min_test.py`` so the vector-kernel
testbenches (layernorm / modulate / rmsnorm / rope / gelu / residual_gate
/ silu) can all reuse the SAME two slow-path optimisations the big
matmul testbenches use:

  1. MX-E4M3 quantize/dequantize round-trip (:func:`mx_roundtrip`) — every
     HBM tensor the behave_sim consumes is MX-packed (1 byte E4M3 elem +
     1/8-byte E8M0 scale, block=8), NOT the kernel's declared fp16. The
     golden must be MX-round-tripped on BOTH sides (inputs before the
     math, output after) or the HBM-direct comparator diffs an fp32 ideal
     against dequantized MX and reports spurious cosine~0 error. See
     reference memory ``hbm-testbench-mx-roundtrip``.

  2. Fingerprint cache (:func:`run_cached`) — ``build_inputs_and_golden``
     and the HBM pack inside ``create_mem_for_sim`` are the slow,
     deterministic-given-config steps. We hash the config knobs into a
     fingerprint and snapshot build/ artifacts under
     ``build/.tb_cache/<fingerprint>/``; a repeat run with the same
     fingerprint restores them and skips both the golden derivation AND
     the HBM write. The ISA is always recompiled (cheap, the thing most
     likely to change while iterating). Disable with ``TB_CACHE=0``.

The MX config is read from ``plena_settings.toml`` PRECISION (same source
``create_mem_for_sim`` feeds ``_mx_fp_quantize_hardware``), so the
testbench geometry never drifts from what the sim actually packs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Callable

import torch

from tilelang_tvm_compiler.test_helper import (
    TvmTestbenchSpec,
    run,
    REPO_ROOT,
    TESTBENCH_DIR,
)


# ---------------------------------------------------------------------------
# MX-E4M3 round-trip — copied verbatim from the linear / flash testbenches.
# ---------------------------------------------------------------------------
def mx_quant_config() -> dict:
    """MX-E4M3 quant params, read from plena_settings.toml PRECISION — the
    SAME config create_mem_for_sim feeds _mx_fp_quantize_hardware."""
    from utils.load_config import load_toml_config

    prec = load_toml_config(str(REPO_ROOT / "plena_settings.toml"), "PRECISION")
    return {
        "exp_w": prec["HBM_V_ACT_TYPE"]["ELEM"]["exponent"],
        "man_w": prec["HBM_V_ACT_TYPE"]["ELEM"]["mantissa"],
        "exp_bias_w": prec["HBM_V_ACT_TYPE"]["SCALE"]["exponent"],
        "block": prec["HBM_M_WEIGHT_TYPE"]["block"],
    }


def mx_quant(x: torch.Tensor, cfg: dict | None = None) -> torch.Tensor:
    """One MX-E4M3 quantize-dequantize pass over ``x`` (any shape).

    Returns a tensor of the same shape/dtype as ``x``, with values rounded
    to MX-E4M3 (block=8 along the last axis).
    """
    from quant.quantizer.hardware_quantizer import _mx_fp_quantize_hardware

    if cfg is None:
        cfg = mx_quant_config()
    x2d = x if x.ndim >= 2 else x.unsqueeze(0)
    bm_x, *_ = _mx_fp_quantize_hardware(
        x2d,
        width=cfg["exp_w"] + cfg["man_w"] + 1,
        exponent_width=cfg["exp_w"],
        exponent_bias_width=cfg["exp_bias_w"],
        block_size=[1, cfg["block"]],
        skip_first_dim=False,
    )
    return bm_x.reshape(x.shape).to(x.dtype)


def mx_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Whole-tensor MX-E4M3 round-trip — matches create_mem_for_sim, which
    quantizes the entire staged .pt at once."""
    return mx_quant(x, mx_quant_config())


# ---------------------------------------------------------------------------
# Fingerprint cache around the slow, config-determined steps.
# ---------------------------------------------------------------------------
# Artifacts build_inputs_and_golden + create_mem_for_sim drop into build/
# that fully capture "the inputs and golden + packed HBM for this config".
# Restoring exactly these lets the downstream sim run without re-deriving
# anything. The .pt inputs feed the HBM pack; the hbm_* bins ARE the packed
# HBM; the golden_* files are what check_mem diffs against.
#
# Unlike flash_attention's hard-coded Q/K/V list, the input .pt set varies
# per kernel, so we discover the *_hbm.pt files dynamically at save time
# and additionally cache the fixed golden/HBM products.
_FIXED_CACHED_ARTIFACTS = (
    "input_tensor.pt",
    "golden_output.pt",
    "golden_result.txt",
    "hbm_for_behave_sim.bin",
    "hbm_for_behave_sim.mem",
    "hbm_ele.mem",
    "hbm_scale.mem",
    # The emulator loads these directly (main.rs reads --fpsram / --intsram
    # unconditionally). create_sim_env writes them, but a cache HIT skips
    # create_sim_env — so they MUST be cached too, or the sim panics with
    # "No such file or directory". int_sram.bin may be absent (kernels with
    # no int preload); _save_cache only records files that exist, so a
    # missing one simply isn't restored and the sim falls back / errors
    # only if it truly needs it.
    "fp_sram.bin",
    "int_sram.bin",
)

CACHE_ENABLED = os.environ.get("TB_CACHE", "1") == "1"


def _cache_fingerprint(fingerprint_payload: dict) -> str:
    """Hash every knob build_inputs_and_golden + the HBM pack depend on.

    Same fingerprint <=> byte-identical inputs/golden/HBM, so the cache is
    safe to reuse. The caller passes the config geometry + seed; we fold in
    the full MX quant config (from plena_settings.toml) so changing
    precision also busts the cache.
    """
    payload = dict(fingerprint_payload)
    payload["mx_quant"] = mx_quant_config()
    # Cache schema version. Bump to invalidate ALL existing caches when the
    # cached-artifact set changes. schema 2: fp_sram.bin / int_sram.bin are
    # now cached (a hit skips create_sim_env, which used to be the only
    # writer of those — older caches lack them and would crash the sim).
    payload["cache_schema"] = 2
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_dir(asm_name: str, fingerprint: str) -> Path:
    # NB: caches live in testbench/.tb_cache, NOT build/.tb_cache —
    # ``just build-emulator-debug`` does ``rm -rf build`` at the top of
    # every run, so anything under build/ is wiped before it can be
    # reused. testbench/.tb_cache survives across runs.
    return TESTBENCH_DIR / ".tb_cache" / asm_name / fingerprint


def _cached_artifact_names(build_dir: Path) -> tuple[str, ...]:
    """Fixed golden/HBM products + every staged *_hbm.pt input."""
    pts = sorted(p.name for p in build_dir.glob("*_hbm.pt"))
    return (*_FIXED_CACHED_ARTIFACTS, *pts)


def _cache_is_complete(d: Path) -> bool:
    # A complete cache must at least carry the fixed products; the input
    # .pt set is recorded in a manifest written at save time.
    if not d.is_dir():
        return False
    manifest = d / ".manifest.json"
    if not manifest.exists():
        return False
    names = json.loads(manifest.read_text())
    return all((d / f).exists() for f in names)


def _restore_cache(d: Path, build_dir: Path) -> None:
    names = json.loads((d / ".manifest.json").read_text())
    for f in names:
        shutil.copy2(d / f, build_dir / f)


def _save_cache(d: Path, build_dir: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    names = _cached_artifact_names(build_dir)
    saved = []
    for f in names:
        src = build_dir / f
        if src.exists():
            shutil.copy2(src, d / f)
            saved.append(f)
    (d / ".manifest.json").write_text(json.dumps(saved))


def run_cached(spec: TvmTestbenchSpec, fingerprint_payload: dict) -> int:
    """``test_helper.run`` with a fingerprint cache around the slow,
    config-determined steps (inputs+golden derivation and the HBM pack).

    On a cache MISS we fall straight through to ``run(spec)`` (a full
    rebuild) and snapshot the resulting artifacts. On a HIT we recompile
    the ISA + reassemble fresh (so compiler changes always take effect)
    but restore the cached inputs/golden/HBM bins instead of recomputing
    them, then re-emit comparison_params.json. ``TB_CACHE=0`` forces the
    plain full rebuild every time.

    ``fingerprint_payload`` — the config knobs build_inputs_and_golden and
    the HBM pack depend on (geometry, seed). The MX quant config is folded
    in automatically.
    """
    if not CACHE_ENABLED:
        return run(spec)

    from tilelang_tvm_compiler.test_helper import _compile_via_subprocess
    from tilelang_tvm_compiler.isa_pass import _normalize_large_addi_immediates

    build_dir = TESTBENCH_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _cache_fingerprint(fingerprint_payload)
    cache_dir = _cache_dir(spec.asm_name, fingerprint)

    if not _cache_is_complete(cache_dir):
        # MISS — full rebuild, then snapshot for next time.
        print(f"[cache] miss ({spec.asm_name}/{fingerprint}) — full rebuild")
        rc = run(spec)
        if rc == 0:
            _save_cache(cache_dir, build_dir)
            print(f"[cache] saved -> {cache_dir}")
        return rc

    # HIT — recompile ISA only; restore inputs/golden/HBM from cache.
    print(f"[cache] hit ({spec.asm_name}/{fingerprint}) — reusing "
          f"inputs/golden/HBM, recompiling ISA")
    from compiler.sim_env_utils.build_sys_tools import env_setup
    from compiler.sim_env_utils.build_env import MemoryDataManager
    from utils.load_config import load_toml_config

    artifact_prefix = spec.artifact_prefix or spec.asm_name
    hlir_path = build_dir / f"{spec.asm_name}.hlir.txt"
    addrs_path = (
        build_dir / f"{spec.asm_name}.buffer_addrs.json"
        if spec.parse_buffer_addrs is not None else None
    )
    print(f"[1/3] Compiling TVM {spec.asm_name} kernel ...")
    kernel_isa = _compile_via_subprocess(
        spec, hlir_path=hlir_path, addrs_path=addrs_path,
    )
    addrs: dict = {}
    raw_addrs: dict = {}
    if addrs_path is not None:
        raw_addrs = json.loads(addrs_path.read_text())
        addrs = spec.parse_buffer_addrs(raw_addrs)  # type: ignore[misc]

    # Optional pre-kernel stub + patch_isa (matches run()'s assembly path).
    stub_isa = ""
    if spec.build_pre_kernel_stub is not None:
        stub_isa = spec.build_pre_kernel_stub(addrs)
    isa_text = stub_isa + kernel_isa
    if spec.patch_isa is not None:
        isa_text = spec.patch_isa(isa_text)
    isa_text = _normalize_large_addi_immediates(isa_text)
    (build_dir / "generated_asm_code.asm").write_text(isa_text)
    print(f"      OK  ({isa_text.count(chr(10))} lines)")

    print("[2/3] Restoring cached inputs + golden + HBM bins ...")
    _restore_cache(cache_dir, build_dir)

    # Reassemble the freshly compiled ISA into machine code WITHOUT
    # touching the restored HBM bins. env_setup with an empty memory data
    # manager only runs the assembler step; it does NOT call init_mem, so
    # hbm_for_behave_sim.* stays as restored.
    config = load_toml_config(str(REPO_ROOT / "plena_settings.toml"), "CONFIG")
    precision = load_toml_config(str(REPO_ROOT / "plena_settings.toml"), "PRECISION")
    data_config = {
        "tensor_size": [1, 1],
        "block_size": [1, precision["HBM_M_WEIGHT_TYPE"]["block"]],
    }
    quant_config = {
        "exp_width": precision["HBM_V_ACT_TYPE"]["ELEM"]["exponent"],
        "man_width": precision["HBM_V_ACT_TYPE"]["ELEM"]["mantissa"],
        "exp_bias_width": precision["HBM_V_ACT_TYPE"]["SCALE"]["exponent"],
        "int_width": precision["HBM_V_INT_TYPE"]["DATA_TYPE"]["width"],
    }
    env_setup(
        MemoryDataManager(), build_dir, data_config, quant_config,
        hbm_row_width=config["HBM_WIDTH"]["value"],
    )

    print("[3/3] Writing comparison_params.json ...")
    comparison_params = spec.build_comparison_params({}, addrs)
    (build_dir / "comparison_params.json").write_text(
        json.dumps(comparison_params, indent=2)
    )
    (build_dir / f"{artifact_prefix}_generated_asm_code.asm").write_text(isa_text)
    print(f"      OK  -> {build_dir}")
    print("=" * 60)
    print(f"build/ ready (cached) for: just build-emulator-debug "
          f"{artifact_prefix}")
    print("=" * 60)
    return 0
