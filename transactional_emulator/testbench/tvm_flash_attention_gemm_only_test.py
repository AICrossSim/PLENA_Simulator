"""TVM minimal Q K V gemm-only testbench (no softmax).

Drives ``tilelang_tvm_compiler.kernels.flash_attention_gemm_only``.
Golden: ``out = (Q @ K^T) @ V`` per head (no softmax, no scale).

Hardware-aligned golden: the simulator ALWAYS runs quantized, so the
golden always models hardware precision (no toggle):
  * Q/K/V are staged into HBM via create_mem_for_sim ->
    _mx_fp_quantize_hardware (MX-E4M3, block=8, E8M0 scale).
  * The two matmuls (BTMM Q@K^T, matmul S@V) are pure f32 in the Rust
    sim — no truncation.
  * O is stored to O_hbm then re-staged to VRAM (--stage-output) for the
    view_mem comparison, so O goes through one more MX-E4M3 round-trip.
"""

from __future__ import annotations

import math
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

from tilelang_tvm_compiler.test_helper import (  # noqa: E402
    TvmTestbenchSpec,
    run,
    resolve_output_layout,
)
from tilelang_tvm_compiler.plena_settings import load_sizes as _load_sizes  # noqa: E402


# Hardware sizes come from plena_settings.toml (active mode) — the same
# single source of truth the compiler reads, so the testbench never
# drifts from the target geometry.
_HW = _load_sizes()

BATCH = 1
MLEN = _HW.mlen
HLEN = _HW.hlen
ROWS = MLEN  # q-tile length == mlen
HEAD_COUNT = MLEN // HLEN  # one packed-head group (= MLEN/HLEN)
NUM_KV_BLOCKS = 1
NUM_Q_BLOCKS = 1
KV_SEQ = NUM_KV_BLOCKS * ROWS
Q_SEQ = NUM_Q_BLOCKS * ROWS

# fd_steps dial (0..8): adds the stripped softmax ops back one at a time.
# The kernel reads the same value via kernel_kwargs; the golden mirrors
# each level exactly. Run FD_STEPS=0,1,...,8 and watch which level drops.
FD_STEPS = int(os.environ.get("FD_STEPS", "0"))
SCALE = 1.0 / math.sqrt(HLEN)
NEG_INF = -1.0e4


def _f16(t: torch.Tensor) -> torch.Tensor:
    """Truncate to f16 precision (fp32 storage) — mirrors the sim's FPRAM
    scalars (fp_reg: [f16; 8] / fpsram: Vec<f16>), where every S_ADD_FP /
    S_MUL_FP / S_EXP_FP / S_RECI_FP / V_RED_* result is f16-truncated."""
    return t.to(torch.float16).to(torch.float32)


def _mx_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """MX-E4M3 quantize-dequantize, using the SAME _mx_fp_quantize_hardware
    + plena_settings.toml config that create_mem_for_sim uses to stage HBM
    tensors. Returns the fp32 values the simulator actually reads back."""
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


def _golden_one_head(q_h, k_h, v_h):
    """Per-head golden mirroring flash_attention_gemm_only.py at FD_STEPS.

    fd_steps 0..6 — each level maps to a verbatim code block of
    flash_attention_min.py:

      0  O = (Q@K^T) @ V
      1  block A : S *= scale ; M_CURR = M_OLD
      2  reduce_max -> M_CURR
      3  block B : M_RES = exp(M_OLD-M_CURR) ; S = exp(S-M_CURR) ; P_SUM=0
      4  reduce_sum -> P_SUM
      5  block C : L_NEW = L_OLD*M_RES+P_SUM ; O *= M_RES ; advance state
      6  block D : L_INV = 1/L_NEW ; O *= L_INV

    FPRAM scalars (M_*/L_*/P_SUM) are f16-truncated (real f16 hardware).
    S / O / matmuls stay f32 (VRAM QuantTensor::quantize is a no-op,
    Rust matmul is f32).

    q_h: (Q_SEQ, HLEN)   k_h/v_h: (KV_SEQ, HLEN)
    """
    o_loc = torch.zeros(ROWS, HLEN)
    m_old = torch.full((ROWS,), NEG_INF)
    l_old = torch.zeros(ROWS)
    l_new = l_old
    m_curr = m_old
    m_res = torch.ones(ROWS)
    p_sum = torch.zeros(ROWS)

    for kvb in range(NUM_KV_BLOCKS):
        k_blk = k_h[kvb * ROWS : (kvb + 1) * ROWS]  # (ROWS, HLEN)
        v_blk = v_h[kvb * ROWS : (kvb + 1) * ROWS]

        # BTMM Q@K^T -> S_loc : pure f32 matmul.
        sc = q_h @ k_blk.T  # (ROWS, ROWS)

        # --- block A : S *= scale ; M_CURR = M_OLD --------------------
        if FD_STEPS >= 1:
            sc = sc * SCALE
            m_curr = m_old

        # --- reduce_max -> M_CURR  (f16 FPRAM scalar) -----------------
        if FD_STEPS >= 2:
            m_curr = _f16(torch.maximum(m_curr, sc.max(dim=1).values))

        # --- block B : M_RES, S=exp(S-M_CURR), P_SUM=0 ----------------
        if FD_STEPS >= 3:
            m_res = _f16(torch.exp(_f16(m_old - m_curr)))  # 2 f16 ops
            sc = torch.exp(sc - m_curr[:, None])  # VRAM, f32
            p_sum = torch.zeros(ROWS)

        # --- reduce_sum -> P_SUM  (f16 FPRAM scalar) ------------------
        if FD_STEPS >= 4:
            p_sum = _f16(sc.sum(dim=1))

        # --- block C : L_NEW, O *= M_RES, advance state ---------------
        if FD_STEPS >= 5:
            l_new = _f16(_f16(l_old * m_res) + p_sum)  # 2 f16 ops
            o_loc = o_loc * m_res[:, None]
            m_old, l_old = m_curr, l_new

        # P @ V -> PV_loc ; O += PV : pure f32 matmul + add.
        o_loc = o_loc + sc @ v_blk

    # --- block D : O = O / L_NEW ---------------------------------------
    if FD_STEPS >= 6:
        l_inv = _f16(1.0 / l_new)
        o_loc = o_loc * l_inv[:, None]

    return o_loc


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    q = torch.randn(BATCH, Q_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.25
    k = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.25
    v = torch.randn(BATCH, KV_SEQ, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.25

    # Q/K/V are read back from HBM already MX-E4M3 quantized — compute the
    # golden on the SAME quantized values the kernel actually consumes.
    q_eff = _mx_roundtrip(q)
    k_eff = _mx_roundtrip(k)
    v_eff = _mx_roundtrip(v)
    for name, raw, eff in (("Q", q, q_eff), ("K", k, k_eff), ("V", v, v_eff)):
        d = (eff - raw).abs()
        print(f"[HW_GOLDEN] {name}: MX-E4M3 input quant err  max={d.max():.3e} mean={d.mean():.3e}")
    print(f"[FD_STEPS={FD_STEPS}]")

    # DMA-in-only: the kernel only copies Q_hbm -> Q_sh (VRAM) and we
    # compare Q_sh directly. Q_sh therefore equals Q after a single
    # HBM->VRAM load, i.e. q_eff (Q already MX-E4M3 quantized once on
    # the way into HBM). No second round-trip — no O_hbm writeback.
    out = q_eff

    golden_flat = out.reshape(BATCH * Q_SEQ, HEAD_COUNT * HLEN)
    return {
        "hbm_inputs": {
            "Q_hbm": q,
            "K_hbm": k,
            "V_hbm": v,
            "O_hbm": torch.zeros_like(q),
        },
        "golden_flat": golden_flat,
    }


def parse_buffer_addrs(raw: dict) -> dict:
    """Trivial passthrough. Its only job is to be non-None so the SPEC
    passes --dump-buffer-addrs to the compiler — that dump is what lets
    test_helper auto-preload hoisted float constants (the softmax scale
    becomes a __const_f16_* FPRAM slot from fd_steps>=1; without the
    dump, its FPRAM slot stays 0 and S*=scale multiplies S by 0 -> all
    output zero).  See feedback_testbench_parse_buffer_addrs.
    """
    del raw
    return {}


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io, addrs
    # Geometry (num_rows / num_batches / elements_per_batch /
    # row_dim / use_stride_mode) from the canonical OutputLayout
    # so it agrees with golden_flat by construction.
    layout = resolve_output_layout(
        num_batches=BATCH * Q_SEQ,
        elements_per_batch=HEAD_COUNT * HLEN,
        mlen=MLEN,
    )
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        "compare_fpsram": False,
        **layout.comparison_params(),
    }


SPEC = TvmTestbenchSpec(
    asm_name="flash_attention_gemm_only",
    kernel=("tilelang_tvm_compiler.kernels.flash_attention_gemm_only:make_flash_attention_gemm_only"),
    kernel_kwargs={
        "rows": ROWS,
        "hlen": HLEN,
        "head_count": HEAD_COUNT,
        "num_kv_blocks": NUM_KV_BLOCKS,
        "num_q_blocks": NUM_Q_BLOCKS,
        "fd_steps": FD_STEPS,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    # DMA-in-only probe: kernel leaves the result in Q_sh (VRAM) and
    # does not touch O_hbm. No re-stage needed — compare VRAM directly.
    stage_output=None,
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
