"""TVM minimal FlashAttention testbench (multi-block flow).

Drives the ``tilelang_tvm_compiler.kernels.flash_attention_min`` kernel
through the test_helper. Builds Q/K/V, FP-preload (scale / m_init /
l_init from the compiler's --dump-buffer-addrs JSON), and per-head
softmax-attention golden.

What the kernel computes (all heads):
    score = Q @ K^T                       (BTMM #1, all lanes get raw score)
    P     = softmax(scale * score)        (online softmax over kv blocks)
    O     = P @ V                         (BTMM #2)
    -> O_hbm

Hardware-aligned golden (env ``HW_GOLDEN=1``): the kernel only DMAs
Q/K/V from HBM and writes O back — score / softmax / P@V all stay
on-chip (VRAM/MRAM/FPRAM, no HBM spill), so quantization error enters
ONLY through the Q/K/V HBM load. ``create_mem_for_sim`` runs each
staged .pt through ``_mx_fp_quantize_hardware`` (MX-E4M3, block=8,
E8M0 scale), so HW_GOLDEN quantizes Q/K/V with that exact same
function/config. On-chip math stays fp32 (the Rust ops are fp32).

HBM-direct comparison: instead of re-staging O back into VRAM and
comparing BF16, this testbench points the comparator straight at the
kernel's own ``O_hbm`` store in ``hbm_dump.bin`` (``check_hbm=True``).
The kernel writes O as MX-E4M3 (1 byte/elem + 1/8-byte E8M0 scale,
block=8), so the compared value is ALWAYS MX-quantized — the golden's
O is therefore MX-round-tripped unconditionally, not gated on
HW_GOLDEN. ``result_hbm_start_byte`` is read from the compiler's
``--dump-buffer-addrs`` JSON: ``AddressAllocationPass`` gives each HBM
buffer its MXFP-packed byte base, so ``O_hbm["address"]`` IS that
offset directly (no hand-derived layout).

Input + golden cache (``FA_CACHE=1``, default on): the random Q/K/V,
the softmax golden, and the packed ``hbm_for_behave_sim`` bins are
expensive to regenerate but depend only on a small set of config
knobs. We hash those knobs into a fingerprint and snapshot the
build/ artifacts under ``build/.fa_cache/<fingerprint>/``; a later
run with the same fingerprint restores them and skips both
``build_inputs_and_golden`` and the HBM pack. The ISA is always
recompiled (cheap, and the thing most likely to change). Set
``FA_CACHE=0`` to force a full rebuild.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path

# venv probing + sys.path setup (must run before torch / helper imports).
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _THIS_FILE.parents if (p / ".venv").is_dir() and (p / "compiler").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError(f"could not locate repo root above {_THIS_FILE}")
_PY_VERSION_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    _venv_lib = _parent / ".venv" / "lib"
    if not _venv_lib.is_dir():
        continue
    for _site_pkg in _venv_lib.glob(f"{_PY_VERSION_TAG}/site-packages"):
        sys.path.append(str(_site_pkg))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "compiler"))

import torch  # noqa: E402

from tilelang_tvm_compiler.test_helper import (  # noqa: E402
    TvmTestbenchSpec, run, resolve_output_layout, REPO_ROOT, TESTBENCH_DIR,
)
from tilelang_tvm_compiler.plena_settings import load_sizes as _load_sizes  # noqa: E402


# Hardware sizes come from plena_settings.toml (active mode) — the same
# single source of truth the compiler reads, so the testbench never
# drifts from the target geometry.
_HW = _load_sizes()

BATCH = 1
MLEN = _HW.mlen      # hardware geometry — from plena_settings.toml
HLEN = _HW.hlen
ROWS = MLEN          # q-tile length == mlen
HEAD_COUNT = (MLEN // HLEN)*2
HARDWARE_LANE_COUNT = MLEN // HLEN
ACTIVE_LANE = 2
NUM_KV_BLOCKS = 2   # full multi-block online softmax
NUM_Q_BLOCKS = 2    # multi-Q outer loop
KV_SEQ = NUM_KV_BLOCKS * ROWS
Q_SEQ = NUM_Q_BLOCKS * ROWS

# Finite "negative-infinity" surrogate compatible with float16 / FP-scalar
# arithmetic. Mirrors attention.py's choice.
NEG_INF = -1.0e4

HW_GOLDEN = os.environ.get("HW_GOLDEN", "0") == "1"
SEED = int(os.environ.get("SEED", "0"))

# Canonical output layout — the single source of truth for how the
# logical (B, Q_SEQ, HEAD_COUNT, HLEN) output maps to VRAM staging, and
# therefore how golden is flattened AND how the comparator reorders the
# staged dump. golden flatten + comparison_params both derive from this,
# so they cannot disagree.
_OUT_LAYOUT = resolve_output_layout(
    b=BATCH, s=Q_SEQ, h=HEAD_COUNT, d=HLEN, mlen=MLEN, hlen=HLEN,
)


def _mx_quant_config() -> dict:
    """MX-E4M3 quant params, read once from plena_settings.toml — the SAME
    config create_mem_for_sim feeds to _mx_fp_quantize_hardware."""
    from utils.load_config import load_toml_config

    prec = load_toml_config(str(_REPO_ROOT / "plena_settings.toml"), "PRECISION")
    return {
        "exp_w": prec["HBM_V_ACT_TYPE"]["ELEM"]["exponent"],
        "man_w": prec["HBM_V_ACT_TYPE"]["ELEM"]["mantissa"],
        "exp_bias_w": prec["HBM_V_ACT_TYPE"]["SCALE"]["exponent"],
        "block": prec["HBM_M_WEIGHT_TYPE"]["block"],
    }


def _mx_quant(x: torch.Tensor, cfg: dict) -> torch.Tensor:
    """One MX-E4M3 quantize-dequantize pass over tensor ``x`` (any shape)."""
    from quant.quantizer.hardware_quantizer import _mx_fp_quantize_hardware

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


def _mx_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Whole-tensor MX-E4M3 round-trip — matches create_mem_for_sim, which
    quantizes the entire staged .pt at once."""
    return _mx_quant(x, _mx_quant_config())


def _mx_roundtrip_per_slice(
    x: torch.Tensor, seq_block: int, head_block: int
) -> torch.Tensor:
    """MX-E4M3 round-trip, but quantized one DMA slice at a time.

    The kernel DMAs Q/K/V in (1, seq_block, head_block, hlen) chunks
    (HLIR ``dma_h2{v,m}_slice``). This quantizes each such chunk on its
    OWN — if the result differs from :func:`_mx_roundtrip` (whole-tensor),
    the MX block grouping straddles DMA-slice boundaries and the per-slice
    layout, not the whole tensor, is what the sim actually consumes.

    x: (1, seq, head, hlen)
    """
    cfg = _mx_quant_config()
    _, seq, head, hlen = x.shape
    out = torch.empty_like(x)
    for s0 in range(0, seq, seq_block):
        for h0 in range(0, head, head_block):
            sl = x[:, s0:s0 + seq_block, h0:h0 + head_block, :]
            out[:, s0:s0 + seq_block, h0:h0 + head_block, :] = _mx_quant(sl, cfg)
    return out


def parse_buffer_addrs(raw: dict) -> dict:
    """Pull O_hbm's MXFP-packed byte offset for the HBM-direct compare.

    SCALE / M_INIT / L_INIT are anonymous ``__const_f16_*`` slots the
    compiler auto-hoists and ``test_helper`` auto-preloads from the
    dump's ``value`` field, so nothing to introspect there. The one
    thing we DO need is where the kernel's output O lands in HBM:
    ``AddressAllocationPass`` sets ``O_hbm["address"]`` to the byte base
    O occupies in ``hbm_for_behave_sim.bin`` (advanced by the packed MX
    size, not the raw fp16 size), so it IS ``result_hbm_start_byte``.
    """
    o = raw["O_hbm"]
    return {"result_hbm_start_byte": int(o["address"])}


def _f16(t: torch.Tensor) -> torch.Tensor:
    """Truncate to f16 precision, keep fp32 storage — mirrors the sim's
    FPRAM scalars (``fp_reg: [f16; 8]`` / ``fpsram: Vec<f16>``), where every
    S_ADD_FP / S_MUL_FP / S_EXP_FP / S_RECI_FP / V_RED_* result is run
    through ``f16::from_f32``."""
    return t.to(torch.float16).to(torch.float32)


def build_inputs_and_golden(seed: int = 0) -> dict:
    """Plain scaled-dot-product softmax-attention golden.

    The golden is a one-shot softmax(scale * Q@K^T) @ V per head — NOT
    the online/flash formulation. (The kernel may still compute it the
    online way; mathematically the result is the same.)

    The OUTPUT O is always MX-E4M3 round-tripped: the kernel's
    dma_v2h_slice stores O_loc to O_hbm (H_STORE_V -> into_bytes) and the
    HBM-direct comparator reads those exact bytes back, so the compared
    value is O after a full MX-E4M3 HBM round-trip regardless of
    HW_GOLDEN. HW_GOLDEN=1 additionally adds the INPUT-side precision:
      1. MX-E4M3 on the Q/K/V HBM load (create_mem_for_sim quantizes them).
      2. f16 on the FPRAM softmax scalars (row-max / row-sum / 1/sum).
    """
    torch.manual_seed(seed)
    # hbm_inputs must stay fp32 — they are the raw HBM tensors the sim
    # quantizes itself (create_mem_for_sim -> MX-E4M3).
    q = torch.randn(BATCH, Q_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    k = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    v = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5

    # HW_GOLDEN: the kernel reads Q/K/V back from HBM already quantized to
    # MX-E4M3. Quantize each tensor one DMA slice at a time — the kernel
    # DMAs Q/K/V in (1, ROWS, HARDWARE_LANE_COUNT, HLEN) chunks, so the MX
    # block grouping the sim actually consumes is per-slice. The diagnostic
    # prints how far per-slice diverges from whole-tensor quantization: if
    # ~0, DMA slicing is NOT an error source; if large, it is.
    if HW_GOLDEN:
        q_eff = _mx_roundtrip_per_slice(q, ROWS, HARDWARE_LANE_COUNT)
        k_eff = _mx_roundtrip_per_slice(k, ROWS, HARDWARE_LANE_COUNT)
        v_eff = _mx_roundtrip_per_slice(v, ROWS, HARDWARE_LANE_COUNT)

        for name, t, te in (("Q", q, q_eff), ("K", k, k_eff), ("V", v, v_eff)):
            whole = _mx_roundtrip(t)
            d = (te - whole).abs()
            print(f"[HW_GOLDEN] {name}: per-slice vs whole-tensor MX  "
                  f"max={d.max():.3e} mean={d.mean():.3e}")
    else:
        q_eff, k_eff, v_eff = q, k, v

    scale = 1.0 / math.sqrt(HLEN)
    out = torch.empty(BATCH, Q_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32)

    for h in range(HEAD_COUNT):
        q_h = q_eff[0, :, h, :]   # (Q_SEQ, HLEN)
        k_h = k_eff[0, :, h, :]   # (KV_SEQ, HLEN)
        v_h = v_eff[0, :, h, :]
        # Plain one-shot scaled-dot-product softmax attention — NOT the
        # online/flash formulation. score / P / P@V live in VRAM (fp32);
        # the softmax row-max / row-sum / reciprocal are FPRAM scalars,
        # so HW_GOLDEN f16-truncates those.
        s = (q_h @ k_h.T) * scale                         # (Q_SEQ, KV_SEQ)
        if HW_GOLDEN:
            row_max = _f16(s.max(dim=-1, keepdim=True).values)
            e = torch.exp(s - row_max)
            row_sum = _f16(e.sum(dim=-1, keepdim=True))
            p = e * _f16(1.0 / row_sum)
        else:
            p = torch.softmax(s, dim=-1)
        out[0, :, h, :] = p @ v_h

    # HBM-direct compare: the comparator reads O straight from
    # hbm_dump.bin as MX-E4M3, so the golden's O must be MX-round-tripped
    # ALWAYS (not gated on HW_GOLDEN) — otherwise we'd diff an fp32 ideal
    # against dequantized MX and see spurious error. (HW_GOLDEN still
    # controls the *input*-side and FPRAM-scalar precision above.)
    out = _mx_roundtrip(out)

    # Canonical golden flatten — batch-major [s][h][d]. The comparator
    # reorders the VRAM side to match this; never the reverse.
    golden_flat = _OUT_LAYOUT.flatten_golden(out)
    return {
        "hbm_inputs": {
            "Q_hbm": q,
            "K_hbm": k,
            "V_hbm": v,
            "O_hbm": torch.zeros_like(q),
        },
        "golden_flat": golden_flat,
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    # Geometry (num_rows / num_batches / elements_per_batch / row_dim /
    # use_stride_mode) all come from the canonical _OUT_LAYOUT so they
    # agree with flatten_golden by construction. Only kernel-specific
    # keys are set here.
    del io
    params = _OUT_LAYOUT.comparison_params()
    # HBM-direct compare: read O straight from hbm_dump.bin as MX-E4M3.
    #   result_hbm_start_byte — O's MXFP-packed byte base (from the
    #     buffer-addrs dump, via parse_buffer_addrs).
    #   scale_offset — byte distance from O's element region to its scale
    #     region. With 1 byte/elem this is num_batches * elements_per_batch
    #     (matches view_mem.py's default derivation).
    return {
        "check_hbm": True,
        "start_row_idx": 0,
        "compare_fpsram": False,
        "result_hbm_start_byte": int(addrs["result_hbm_start_byte"]),
        "scale_offset": params["num_batches"] * params["elements_per_batch"],
        **params,
    }


# ---------------------------------------------------------------------------
# Input + golden cache.
#
# build_inputs_and_golden (per-head softmax + MX round-trips) and the HBM
# pack inside create_mem_for_sim are the slow, deterministic-given-config
# parts of a run. We fingerprint the config knobs they depend on and
# snapshot their products under build/.fa_cache/<fingerprint>/. A repeat
# run with the same fingerprint restores them and skips both. The ISA is
# always recompiled — it is cheap and is the thing that actually changes
# while iterating on the compiler.
# ---------------------------------------------------------------------------

CACHE_ENABLED = os.environ.get("FA_CACHE", "1") == "1"

# Files build_inputs_and_golden + create_mem_for_sim drop into build/ that
# fully capture "the inputs and golden for this config". Restoring exactly
# these lets the downstream sim run without re-deriving anything. The .pt
# inputs feed the HBM pack; the hbm_* bins ARE the packed HBM; the golden_*
# files are what check_mem diffs against.
_CACHED_ARTIFACTS = (
    "Q_hbm.pt", "K_hbm.pt", "V_hbm.pt", "O_hbm.pt",
    "input_tensor.pt",
    "golden_output.pt", "golden_result.txt",
    "hbm_for_behave_sim.bin", "hbm_for_behave_sim.mem",
    "hbm_ele.mem", "hbm_scale.mem",
)


def _cache_fingerprint() -> str:
    """Hash every knob build_inputs_and_golden + the HBM pack depend on.

    Same fingerprint <=> byte-identical inputs/golden/HBM, so the cache is
    safe to reuse. Config geometry, seed, HW_GOLDEN, and the full MX quant
    config (from plena_settings.toml) all go in — change any and the
    fingerprint changes, forcing a rebuild.
    """
    payload = {
        "batch": BATCH, "mlen": MLEN, "hlen": HLEN, "rows": ROWS,
        "head_count": HEAD_COUNT, "active_lane": ACTIVE_LANE,
        "num_kv_blocks": NUM_KV_BLOCKS, "num_q_blocks": NUM_Q_BLOCKS,
        "q_seq": Q_SEQ, "kv_seq": KV_SEQ,
        "hw_golden": HW_GOLDEN, "seed": SEED,
        "mx_quant": _mx_quant_config(),
        # Bump when the golden math or cached-artifact set changes so old
        # caches from a previous testbench version can't be reused.
        "schema": 2,
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _cache_dir() -> Path:
    return TESTBENCH_DIR / "build" / ".fa_cache" / _cache_fingerprint()


def _cache_is_complete(d: Path) -> bool:
    return d.is_dir() and all((d / f).exists() for f in _CACHED_ARTIFACTS)


def _restore_cache(d: Path, build_dir: Path) -> None:
    for f in _CACHED_ARTIFACTS:
        shutil.copy2(d / f, build_dir / f)


def _save_cache(d: Path, build_dir: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for f in _CACHED_ARTIFACTS:
        src = build_dir / f
        if src.exists():
            shutil.copy2(src, d / f)


SPEC = TvmTestbenchSpec(
    asm_name="flash_attention_min",
    kernel="tilelang_tvm_compiler.kernels.flash_attention_min:make_flash_attention_min",
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "active_lane": ACTIVE_LANE,
        "num_kv_blocks": NUM_KV_BLOCKS, "num_q_blocks": NUM_Q_BLOCKS,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    # No stage_output: HBM-direct compare reads the kernel's own O_hbm
    # store from hbm_dump.bin, so there is no re-stage HBM->VRAM step.
    # Route compilation through the PreIsaPassV2 → MIR → ISA path
    # (LICM / reassoc / IntRAM spill). Set USE_V2=0 in the env to
    # fall back to the legacy single-pass emitter for comparison.
    use_v2=os.environ.get("USE_V2", "1") == "1",
    seed=SEED,
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


def run_cached(spec: TvmTestbenchSpec) -> int:
    """``test_helper.run`` with a fingerprint cache around the slow,
    config-determined steps (inputs+golden derivation and the HBM pack).

    On a cache MISS we fall straight through to ``run(spec)`` (a full
    rebuild) and snapshot the resulting artifacts. On a HIT we recompile
    the ISA + reassemble fresh (so compiler changes always take effect)
    but restore the cached inputs/golden/HBM bins instead of recomputing
    them, then re-emit comparison_params.json. ``FA_CACHE=0`` forces the
    plain full rebuild every time.
    """
    if not CACHE_ENABLED:
        return run(spec)

    from tilelang_tvm_compiler.test_helper import (
        _compile_via_subprocess, _validate_io,
    )
    from tilelang_tvm_compiler.isa_pass import _normalize_large_addi_immediates

    build_dir = TESTBENCH_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _cache_dir()

    if not _cache_is_complete(cache_dir):
        # MISS — full rebuild, then snapshot for next time.
        print(f"[cache] miss ({cache_dir.name}) — full rebuild")
        rc = run(spec)
        if rc == 0:
            _save_cache(cache_dir, build_dir)
            print(f"[cache] saved -> {cache_dir}")
        return rc

    # HIT — recompile ISA only; restore inputs/golden/HBM from cache.
    print(f"[cache] hit ({cache_dir.name}) — reusing inputs/golden/HBM, "
          f"recompiling ISA")
    from compiler.sim_env_utils.build_sys_tools import env_setup
    from compiler.sim_env_utils.build_env import MemoryDataManager
    from utils.load_config import load_toml_config

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
    isa_text = _normalize_large_addi_immediates(kernel_isa)
    (build_dir / "generated_asm_code.asm").write_text(isa_text)
    print(f"      OK  ({isa_text.count(chr(10))} lines)")

    print("[2/3] Restoring cached inputs + golden + HBM bins ...")
    _restore_cache(cache_dir, build_dir)

    # Reassemble the freshly compiled ISA into machine code WITHOUT
    # touching the restored HBM bins. env_setup with an empty memory data
    # manager only runs the assembler step (the per-tensor HBM packing
    # loop has nothing to do); it does NOT call init_mem (that lives in
    # create_mem_for_sim), so hbm_for_behave_sim.* stays as restored.
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
    # Rebuild io shells only enough for build_comparison_params (it uses
    # addrs; io is unused here). golden_flat already on disk from cache.
    comparison_params = spec.build_comparison_params({}, addrs)
    (build_dir / "comparison_params.json").write_text(
        json.dumps(comparison_params, indent=2)
    )
    artifact_prefix = spec.artifact_prefix or spec.asm_name
    (build_dir / f"{artifact_prefix}_generated_asm_code.asm").write_text(isa_text)
    print(f"      OK  -> {build_dir}")
    print("=" * 60)
    print(f"build/ ready (cached) for: just build-emulator-debug "
          f"{artifact_prefix}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(run_cached(SPEC))
