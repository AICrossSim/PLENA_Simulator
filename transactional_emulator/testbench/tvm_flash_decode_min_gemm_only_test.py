"""TVM flash-decode gemm-only debug testbench.

Drives ``tilelang_tvm_compiler.kernels.flash_decode_min_gemm_only``:

    score = Q @ K^T                 (BTMV, packed-head)
    out   = score @ V               (MV,   per-head)

No scale, no softmax, no online state. Same Q_cache / O_cache cache
layout as flash_decode_min (head-major (HEAD_COUNT, HLEN)). Bisects the
new region+dim_roles gemm schema across multi by_number.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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

CACHE_MLEN_ROWS = (HEAD_COUNT * HLEN) // MLEN

# The simulator ALWAYS runs with quantization (K/V via MX-E4M3, Q via
# f16). HW_GOLDEN gates whether the golden models that hardware
# precision; defaults OFF (ideal fp32 golden) for consistency with the
# other testbenches. Set HW_GOLDEN=1 to model hardware precision.
HW_GOLDEN = os.environ.get("HW_GOLDEN", "0") == "1"
# fd_steps dial (0..8): adds the stripped softmax ops back one at a time.
# The kernel reads the same value via kernel_kwargs; the golden below
# mirrors each level exactly. Run FD_STEPS=0,1,...,8 and watch which
# level the correctness drops at.
FD_STEPS = int(os.environ.get("FD_STEPS", "0"))

import math  # noqa: E402

SCALE = 1.0 / math.sqrt(HLEN)
NEG_INF = -1.0e4


# ---------------------------------------------------------------------------
# Hardware precision model — gemm-only, so the ONLY effects are:
#   * Q  : fp16-preloaded into FPRAM, then S_MAP_V_FP'd to VRAM -> real f16.
#   * K/V: hbm_inputs -> create_mem_for_sim quantizes to MX-E4M3 -> HBM ->
#          DMA into MRAM. So K/V are MX-E4M3 (block=8, E8M0 scale).
#   * BTMV / MV themselves are pure f32 in the Rust sim (no truncation).
#   * O   : written to O_cache in VRAM and compared from VRAM — VRAM
#          QuantTensor::quantize is a no-op, so O is NOT quantized.
# If HW_GOLDEN here still mismatches, the error is in the matmul / MX
# quantization path, NOT in softmax (this kernel has no softmax).
# ---------------------------------------------------------------------------
def _f16(t: torch.Tensor) -> torch.Tensor:
    """Truncate to f16 precision (fp32 storage) — mirrors FPRAM f16."""
    return t.to(torch.float16).to(torch.float32)


def _mx_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """MX-E4M3 quantize-dequantize, same _mx_fp_quantize_hardware + config
    create_mem_for_sim uses to stage K/V into HBM."""
    from utils.load_config import load_toml_config
    from quant.quantizer.hardware_quantizer import _mx_fp_quantize_hardware

    prec = load_toml_config(str(_REPO_ROOT / "plena_settings.toml"), "PRECISION")
    exp_w = prec["HBM_V_ACT_TYPE"]["ELEM"]["exponent"]
    man_w = prec["HBM_V_ACT_TYPE"]["ELEM"]["mantissa"]
    exp_bias_w = prec["HBM_V_ACT_TYPE"]["SCALE"]["exponent"]
    block = prec["HBM_M_WEIGHT_TYPE"]["block"]

    x2d = x if x.ndim >= 2 else x.unsqueeze(0)
    bm_x, *_ = _mx_fp_quantize_hardware(
        x2d,
        width=exp_w + man_w + 1,
        exponent_width=exp_w,
        exponent_bias_width=exp_bias_w,
        block_size=[1, block],
        skip_first_dim=False,
    )
    return bm_x.reshape(x.shape).to(x.dtype)


def parse_buffer_addrs(raw: dict) -> dict:
    def addr_of(name: str) -> int:
        if name not in raw:
            raise KeyError(f"buffer {name!r} not in HLIR; known: {sorted(raw)}")
        return int(raw[name]["address"])

    # Q_FP_STAGE must NOT collide with FPRAM buffers the compiler
    # allocates itself. From fd_steps>=1 the compiler hoists the softmax
    # scale into a `__const_f16_*` FPRAM slot (e.g. addr 32). If
    # Q_FP_STAGE stayed at 0, build_fp_preload's 128-element Q write
    # ([0,128)) would overlap that const slot — the scale read by
    # V_MUL_VF would be garbage. Place Q_FP_STAGE strictly above every
    # compiler-allocated fpram buffer.  (See feedback_testbench_parse_buffer_addrs.)
    fpram_end = 0
    for name, info in raw.items():
        if info.get("scope") != "fpram":
            continue
        shape = info.get("shape") or []
        n = 1
        for d in shape:
            n *= int(d)
        n = max(n, 1)  # scalar consts have shape [] -> 1 slot
        fpram_end = max(fpram_end, int(info["address"]) + n)

    q_fp_stage = fpram_end  # first free FPRAM slot after compiler buffers
    print(f"[parse_buffer_addrs] compiler fpram end={fpram_end}, "
          f"Q_FP_STAGE -> {q_fp_stage}")
    return {
        "Q_CACHE":    addr_of("Q_cache"),
        "O_CACHE":    addr_of("O_cache"),
        "Q_FP_STAGE": q_fp_stage,
    }


def _golden_one_head(q_h, k_h, v_h):
    """Per-head golden mirroring flash_decode_min_gemm_only.py at FD_STEPS.

    q_h: (HLEN,)   k_h/v_h: (KV_SEQ, HLEN)
    Reproduces the kv-block loop and every fd_steps level. Scalars that
    live in FPRAM (M_*/L_*/P_SUM) are f16-truncated when HW_GOLDEN.

    Levels 2/3/5/6 issue an op that does NOT change O — the golden output
    at those levels is identical to 1 (for 2/3) or 4 (for 5/6).
    """
    s = _f16 if HW_GOLDEN else (lambda t: t)

    o_loc = torch.zeros(HLEN)
    m_old = torch.full((1,), NEG_INF)
    l_old = torch.zeros(1)
    l_new = l_old

    for kvb in range(NUM_KV_BLOCKS):
        k_blk = k_h[kvb * ROWS:(kvb + 1) * ROWS]    # (ROWS, HLEN)
        v_blk = v_h[kvb * ROWS:(kvb + 1) * ROWS]

        # BTMV Q@K^T -> S_loc : pure f32 matmul.
        sc = q_h @ k_blk.T                          # (ROWS,)

        # STEP 1: S *= scale
        if FD_STEPS >= 1:
            sc = sc * SCALE

        # STEP 2: M_CURR = max(M_OLD, rowmax(S)) — f16 FPRAM scalar.
        if FD_STEPS >= 2:
            m_curr = s(torch.maximum(m_old, sc.max().reshape(1)))

        # STEP 3: M_RES = exp(M_OLD - M_CURR) — f16 scalar (2 ops).
        if FD_STEPS >= 3:
            m_res = s(torch.exp(s(m_old - m_curr)))

        # STEP 4: S = exp(S - M_CURR) — vector op, fp32.
        if FD_STEPS >= 4:
            sc = torch.exp(sc - m_curr)

        # STEP 5: P_SUM = rowsum(S) — f16 FPRAM scalar.
        if FD_STEPS >= 5:
            p_sum = s(sc.sum().reshape(1))

        # STEP 6: L_NEW = L_OLD*M_RES + P_SUM — f16 scalar (2 ops).
        if FD_STEPS >= 6:
            l_new = s(s(l_old * m_res) + p_sum)

        # STEP 7: O_loc *= M_RES.
        if FD_STEPS >= 7:
            o_loc = o_loc * m_res

        # Advance online state once it is actually consumed (step >= 6).
        if FD_STEPS >= 6:
            m_old, l_old = m_curr, l_new

        # MV S@V -> PV_loc ; O += PV : pure f32 matmul + add.
        o_loc = o_loc + sc @ v_blk

    # STEP 8: O = O / L_NEW.
    if FD_STEPS >= 8:
        l_inv = s(1.0 / l_new)
        o_loc = o_loc * l_inv

    return o_loc


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    q = torch.randn(BATCH, 1,      HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    k = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    v = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5

    # HW_GOLDEN: feed each tensor at its real hardware precision.
    #   Q -> f16 (FPRAM preload), K/V -> MX-E4M3 (HBM stage).
    if HW_GOLDEN:
        q_eff = _f16(q)
        k_eff = _mx_roundtrip(k)
        v_eff = _mx_roundtrip(v)
        for name, raw, eff in (("Q", q, q_eff), ("K", k, k_eff), ("V", v, v_eff)):
            d = (eff - raw).abs()
            print(f"[HW_GOLDEN] {name}: input quant err  "
                  f"max={d.max():.3e} mean={d.mean():.3e}")
    else:
        q_eff, k_eff, v_eff = q, k, v

    print(f"[FD_STEPS={FD_STEPS}] HW_GOLDEN={HW_GOLDEN}")

    out = torch.empty(BATCH, 1, HEAD_COUNT, HLEN, dtype=torch.float32)
    for h in range(HEAD_COUNT):
        out[0, 0, h, :] = _golden_one_head(
            q_eff[0, 0, h, :], k_eff[0, :, h, :], v_eff[0, :, h, :]
        )

    golden_flat = out.reshape(BATCH * 1, HEAD_COUNT * HLEN)

    return {
        "hbm_inputs":  {"K_hbm": k, "V_hbm": v},
        "golden_flat": golden_flat,
        "q_token":     q,
    }


def build_pre_kernel_stub(addrs: dict) -> str:
    lines: list[str] = [
        "; pre-kernel cache init: FPRAM[Q_FP_STAGE] -> VRAM[Q_CACHE]",
        f"S_ADDI_INT gp1, gp0, {addrs['Q_CACHE']}",
        f"S_ADDI_INT gp2, gp0, {addrs['Q_FP_STAGE']}",
    ]
    for i in range(CACHE_MLEN_ROWS):
        lines.append("S_MAP_V_FP gp1, gp2, 0")
        if i < CACHE_MLEN_ROWS - 1:
            lines.append(f"S_ADDI_INT gp1, gp1, {MLEN}")
            lines.append(f"S_ADDI_INT gp2, gp2, {MLEN}")
    return "\n".join(lines) + "\n"


def build_fp_preload(io: dict, addrs: dict):
    q_token = io["q_token"]
    total = addrs["Q_FP_STAGE"] + HEAD_COUNT * HLEN
    fp = torch.zeros(total, dtype=torch.float16)
    q_flat = q_token[0, 0].reshape(HEAD_COUNT * HLEN).to(torch.float16)
    fp[addrs["Q_FP_STAGE"] : addrs["Q_FP_STAGE"] + HEAD_COUNT * HLEN] = q_flat
    return fp


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io
    # Geometry from the canonical OutputLayout; start_row_idx is
    # kernel-specific (O_CACHE is not at VRAM 0).
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
    asm_name="flash_decode_min_gemm_only",
    kernel=(
        "tilelang_tvm_compiler.kernels.flash_decode_min_gemm_only:"
        "make_flash_decode_min_gemm_only"
    ),
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "num_kv_blocks": NUM_KV_BLOCKS,
        "fd_steps": FD_STEPS,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_pre_kernel_stub=build_pre_kernel_stub,
    build_fp_preload=build_fp_preload,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
