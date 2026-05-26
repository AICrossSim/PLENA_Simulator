"""TVM minimal FlashDecode (single-token) testbench.

Drives ``tilelang_tvm_compiler.kernels.flash_decode_min``:

    score = q @ K^T,  P = softmax(scale * score),  o = P @ V

Single-token decode: q lives in FPRAM (preloaded + staged into VRAM by a
pre-kernel stub), K/V live in HBM. Output is written to VRAM ``O_cache``;
view_mem reads it directly from VRAM (start_row_idx = O_CACHE / MLEN).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# venv probing + sys.path setup.
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
from tilelang_tvm_compiler.plena_settings import load_sizes as _load_sizes  # noqa: E402

from tilelang_tvm_compiler.test_helper import (  # noqa: E402
    TvmTestbenchSpec, run, resolve_output_layout,
)


BATCH = 1
_HW = _load_sizes()  # hardware geometry — single source of truth, plena_settings.toml

HLEN = _HW.hlen  # from plena_settings.toml
MLEN = _HW.mlen  # from plena_settings.toml
ROWS = MLEN  # rows per tile == mlen
HEAD_COUNT = 8
HARDWARE_LANE_COUNT = MLEN // HLEN
NUM_KV_BLOCKS = 2
KV_SEQ = NUM_KV_BLOCKS * ROWS

CACHE_MLEN_ROWS = (HEAD_COUNT * HLEN) // MLEN  # = HEAD_COUNT // HARDWARE_LANE_COUNT
NEG_INF = -1.0e4


def parse_buffer_addrs(raw: dict) -> dict:
    """Single source of truth for buffer addresses — read from the
    compiler's --dump-buffer-addrs JSON.

    ``Q_FP_STAGE`` is a testbench-only convention (no kernel buffer of
    that name): we place it AFTER the highest FPRAM slot the compiler
    used, so the pre-kernel S_MAP_V_FP stub doesn't clobber any
    kernel-allocated FPRAM. We can't anchor it on a specific named
    buffer anymore (SCALE / M_INIT / L_INIT are now anonymous
    auto-hoisted ``__const_f16_*`` slots whose names embed the value),
    so compute the high-water mark from every fpram entry in the dump.
    """
    def addr_of(name: str) -> int:
        if name not in raw:
            raise KeyError(f"buffer {name!r} not in HLIR; known: {sorted(raw)}")
        return int(raw[name]["address"])
    fpram_max = max(
        int(entry["address"])
        for entry in raw.values()
        if isinstance(entry, dict) and entry.get("scope") == "fpram"
    )
    # +1 slot of headroom past the highest known FPRAM use, then add
    # lane_count so the multi-lane S_MAP_V_FP stub has space.
    q_fp_stage = fpram_max + 1 + HARDWARE_LANE_COUNT
    return {
        "M_OLD":      addr_of("M_OLD"),
        "M_CURR":     addr_of("M_CURR"),
        "M_RES":      addr_of("M_RES"),
        "L_OLD":      addr_of("L_OLD"),
        "L_NEW":      addr_of("L_NEW"),
        "P_SUM":      addr_of("P_SUM"),
        # SCALE / M_INIT / L_INIT are no longer named buffers — kernel
        # embeds literals directly, compiler auto-hoists them into
        # ``__const_f16_*`` global.fpram slots, test_helper auto-loads
        # their values from the dump's "value" field.
        "L_INV":      addr_of("L_INV"),
        "Q_FP_STAGE": q_fp_stage,
        "Q_CACHE":    addr_of("Q_cache"),
        "O_CACHE":    addr_of("O_cache"),
    }


def build_inputs_and_golden(seed: int = 0) -> dict:
    """Single-token Q + KV cache, per-head softmax-attention golden.

    Q is NOT a HBM input — it goes into FPRAM via fp_preload. We stash
    it under ``q_token`` in the returned dict so build_fp_preload can
    pick it up.
    """
    torch.manual_seed(seed)
    q = torch.randn(BATCH, 1,      HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    k = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    v = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5

    scale = 1.0 / math.sqrt(HLEN)
    score = torch.einsum("bihd,bjhd->bihj", q, k)  # (B, 1, H, KV_SEQ)

    out = torch.empty(BATCH, 1, HEAD_COUNT, HLEN, dtype=torch.float32)
    for h in range(HEAD_COUNT):
        score_h = score[:, :, h, :] * scale
        p       = torch.softmax(score_h, dim=-1)
        out[:, :, h, :] = torch.einsum("bij,bjd->bid", p, v[:, :, h, :])

    golden_flat = out.reshape(BATCH * 1, HEAD_COUNT * HLEN)

    return {
        "hbm_inputs":  {"K_hbm": k, "V_hbm": v},
        "golden_flat": golden_flat,
        "q_token":     q,
    }


def build_pre_kernel_stub(addrs: dict) -> str:
    """ASM stub: FPRAM[Q_FP_STAGE] -> VRAM[Q_CACHE], one MLEN-wide
    S_MAP_V_FP per CACHE_MLEN_ROWS row."""
    lines: list[str] = [
        "; ============================================================",
        "; pre-kernel cache init: FPRAM[Q_FP_STAGE] -> VRAM[Q_CACHE]",
        f"; rows={CACHE_MLEN_ROWS}  MLEN={MLEN}",
        "; ============================================================",
        f"S_ADDI_INT gp1, gp0, {addrs['Q_CACHE']}",
        f"S_ADDI_INT gp2, gp0, {addrs['Q_FP_STAGE']}",
    ]
    for i in range(CACHE_MLEN_ROWS):
        lines.append("S_MAP_V_FP gp1, gp2, 0")
        if i < CACHE_MLEN_ROWS - 1:
            lines.append(f"S_ADDI_INT gp1, gp1, {MLEN}")
            lines.append(f"S_ADDI_INT gp2, gp2, {MLEN}")
    lines += [
        "; ============================================================",
        "; kernel proper",
        "; ============================================================",
    ]
    return "\n".join(lines) + "\n"


def build_fp_preload(io: dict, addrs: dict):
    """FP preload: Q values only.

    ``q_token`` is shape (BATCH, 1, HEAD_COUNT, HLEN). Stored head-major
    at ``addrs['Q_FP_STAGE']``; pre-kernel stub copies to VRAM Q_cache.

    Softmax constants (SCALE / -inf surrogate / 0) are auto-hoisted by
    the compiler from inline ``T.float16(...)`` literals and
    auto-preloaded by ``test_helper`` from the dump's ``value`` field —
    no manual writes needed here.
    """
    q_token = io["q_token"]
    total = addrs["Q_FP_STAGE"] + HEAD_COUNT * HLEN
    fp = torch.zeros(total, dtype=torch.float16)

    # Q staging: head-major (head_count, hlen) at Q_FP_STAGE.
    q_flat = q_token[0, 0].reshape(HEAD_COUNT * HLEN).to(torch.float16)
    fp[addrs["Q_FP_STAGE"] : addrs["Q_FP_STAGE"] + HEAD_COUNT * HLEN] = q_flat

    return fp


def build_comparison_params(io: dict, addrs: dict) -> dict:
    """Compare against VRAM-resident O_cache (kernel writes O_loc -> O_cache
    via vram→vram V_ADD_VF). All HEAD_COUNT heads end up at
    ``VRAM[O_CACHE .. +HEAD_COUNT*HLEN)``; view_mem reads the
    OutputLayout's mlen-wide rows starting at O_CACHE / MLEN."""
    del io
    # Geometry from the canonical OutputLayout. num_batches=1 (single
    # decode token), so num_rows == chunks_per_batch == CACHE_MLEN_ROWS.
    # start_row_idx is kernel-specific (O_CACHE is not at VRAM 0).
    layout = resolve_output_layout(
        num_batches=BATCH * 1,
        elements_per_batch=HEAD_COUNT * HLEN,
        mlen=MLEN,
    )
    return {
        "check_hbm": False,
        "start_row_idx": addrs["O_CACHE"] // MLEN,
        "compare_fpsram": False,
        **layout.comparison_params(),
    }


SPEC = TvmTestbenchSpec(
    asm_name="flash_decode_min",
    kernel="tilelang_tvm_compiler.kernels.flash_decode_min:make_flash_decode_min",
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "num_kv_blocks": NUM_KV_BLOCKS,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    # No --stage-output here: kernel writes O_cache directly into VRAM,
    # comparator reads it via start_row_idx in build_comparison_params.
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_pre_kernel_stub=build_pre_kernel_stub,
    build_fp_preload=build_fp_preload,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
