"""TVM minimal Linear testbench.

Drives ``tilelang_tvm_compiler.kernels.linear_min`` — multi-tile GEMM
with M/N/K all multiples of MLEN. Tile counts (M_BLOCKS / N_BLOCKS /
K_BLOCKS) and bias mode are env-controlled so the same testbench
exercises (1, 1, 1) up through bigger tile grids.

Golden: ``torch.nn.functional.linear(a, b, bias) == a @ b.T + bias``.
"""

from __future__ import annotations

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
    TvmTestbenchSpec, run, resolve_output_layout, REPO_ROOT,
)
from tilelang_tvm_compiler.plena_settings import load_sizes as _load_sizes  # noqa: E402


# --- MX-E4M3 round-trip (matches create_mem_for_sim + the HBM-direct
#     comparator), copied from tvm_flash_attention_min_test.py. ALL HBM
#     tensors here are MX-packed: create_mem_for_sim quantizes every
#     input .pt to MX-E4M3, and the kernel writes C to HBM as MX-E4M3
#     too (same H_STORE_V path as flash attn's O). So the golden must be
#     MX-round-tripped on BOTH sides — inputs before the matmul, output
#     after — or we'd diff an fp32 ideal against dequantized MX and see
#     spurious (cosine~0) error.
def _mx_quant_config() -> dict:
    from utils.load_config import load_toml_config
    prec = load_toml_config(str(REPO_ROOT / "plena_settings.toml"), "PRECISION")
    return {
        "exp_w": prec["HBM_V_ACT_TYPE"]["ELEM"]["exponent"],
        "man_w": prec["HBM_V_ACT_TYPE"]["ELEM"]["mantissa"],
        "exp_bias_w": prec["HBM_V_ACT_TYPE"]["SCALE"]["exponent"],
        "block": prec["HBM_M_WEIGHT_TYPE"]["block"],
    }


def _mx_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Whole-tensor MX-E4M3 quantize-dequantize — matches
    create_mem_for_sim, which quantizes the entire staged .pt at once."""
    from quant.quantizer.hardware_quantizer import _mx_fp_quantize_hardware
    cfg = _mx_quant_config()
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


# MLEN comes from plena_settings (BEHAVIOR mode) — the SAME geometry the
# compiler + emulator use, so the testbench never drifts (a hard-coded
# MLEN=1024 against a settings MLEN=512 makes LANE_COUNT mismatch and the
# tile_layout pass throws "H must be a multiple of LANE_COUNT"). With the
# current MLEN=512, M=N=K=2*512=1024 — exactly the single_stream_block
# chain's linear size, so this doubles as the chain-config linear latency.
# M/K multi-tile stresses the v2 stable register allocator; N stays one
# column block so C_hbm is clean row-major (M,N) == golden.reshape(-1),
# and the HBM-direct compare needs NO reordering.
MLEN = _load_sizes().mlen
M_BLOCKS = 2
N_BLOCKS = 2
K_BLOCKS = 2
WITH_BIAS = True

M = M_BLOCKS * MLEN
N = N_BLOCKS * MLEN
K = K_BLOCKS * MLEN


# Canonical output layout — single source of truth for golden flatten.
# C_hbm is logical (1, M, 1, N): rows = M, cols = N. With N == MLEN there
# is one mlen-wide chunk per row (chunks_per_batch == 1), so the physical
# HBM order is plain row-major (M, N) == golden.reshape(-1).
_OUT_LAYOUT = resolve_output_layout(
    mlen=MLEN, num_batches=M, elements_per_batch=N,
)


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    # Small magnitudes so the fp16 K-reduction stays in range. Same idea
    # as the GPU tilelang gemm test that runs at 0.5 scale.
    a_2d = torch.randn(M, K, dtype=torch.float32) * 0.25
    # Weight is (N, K) — nn.Linear convention. The kernel issues
    # T.gemm(..., transpose_B=True), which lowers to M_TMM so the host
    # does NOT need to transpose.
    b_2d = torch.randn(N, K, dtype=torch.float32) * 0.25

    # PLENA HBM tensors are 4D BSHD; linear has no head axis, so lay
    # (rows, cols) along (seq, hlen) with batch=head=1. hbm_inputs stay
    # fp32 — create_mem_for_sim quantizes them to MX-E4M3 itself.
    a = a_2d.view(1, M, 1, K).contiguous()
    b = b_2d.view(1, N, 1, K).contiguous()

    # The kernel reads A/B/BIAS back from HBM already MX-E4M3 quantized,
    # so the golden must compute on the MX-round-tripped inputs (matches
    # what the sim actually multiplies — see flash_attention's q_eff).
    a_eff = _mx_roundtrip(a_2d)
    b_eff = _mx_roundtrip(b_2d)

    if WITH_BIAS:
        bias_1d = torch.randn(N, dtype=torch.float32) * 0.1
        bias_tile = bias_1d.view(1, 1, 1, N).expand(1, M, 1, N).contiguous()
        bias_eff = _mx_roundtrip(bias_tile.reshape(M, N))
        c_2d_golden = a_eff @ b_eff.T + bias_eff
        hbm_inputs = {
            "A_hbm":    a,
            "B_hbm":    b,
            "BIAS_hbm": bias_tile,
            "C_hbm":    torch.zeros(1, M, 1, N, dtype=torch.float32),
        }
    else:
        c_2d_golden = a_eff @ b_eff.T
        hbm_inputs = {
            "A_hbm": a,
            "B_hbm": b,
            "C_hbm": torch.zeros(1, M, 1, N, dtype=torch.float32),
        }

    # The kernel writes C to HBM as MX-E4M3 (same H_STORE_V path as flash
    # attn's O), and the HBM-direct comparator reads those exact bytes
    # back — so the golden's C must be MX-round-tripped too, else we'd
    # diff an fp32 ideal against dequantized MX (spurious cosine~0).
    c_2d_golden = _mx_roundtrip(c_2d_golden)

    return {
        "hbm_inputs": hbm_inputs,
        # Flatten through the canonical layout so golden's element order
        # matches the HBM-direct comparator's flat read by construction.
        "golden_flat": _OUT_LAYOUT.flatten_golden(c_2d_golden),
    }


def parse_buffer_addrs(raw: dict) -> dict:
    """Pull C_hbm's byte base for the HBM-direct compare.

    AddressAllocationPass sets ``C_hbm["address"]`` to the byte offset C
    occupies in ``hbm_for_behave_sim.bin`` (advanced by the MXFP-packed
    size, since C is MX-E4M3 like every HBM tensor here), which IS the
    comparator's ``result_hbm_start_byte``. Same mechanism flash_attention
    uses for O_hbm.
    """
    c = raw["C_hbm"]
    return {"result_hbm_start_byte": int(c["address"])}


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io
    # HBM-direct compare, read straight off hbm_dump.bin — NO VRAM
    # staging, NO stride/reorder. C is MX-E4M3 (1 byte/elem + 1/8-byte
    # E8M0 scale, block=8), same as flash attn's O — so we use the
    # comparator's default MX path (no hbm_dtype override) plus the scale
    # region offset. Geometry comes from the canonical layout.
    params = _OUT_LAYOUT.comparison_params()
    return {
        "check_hbm": True,
        "start_row_idx": 0,
        "compare_fpsram": False,
        "result_hbm_start_byte": int(addrs["result_hbm_start_byte"]),
        "scale_offset": params["num_batches"] * params["elements_per_batch"],
        **params,
    }


SPEC = TvmTestbenchSpec(
    asm_name="linear_min",
    kernel="tilelang_tvm_compiler.kernels.linear_min:make_linear_min",
    kernel_kwargs={
        "m_blocks": M_BLOCKS,
        "n_blocks": N_BLOCKS,
        "k_blocks": K_BLOCKS,
        "with_bias": WITH_BIAS,
    },
    mlen=MLEN,
    stage_output="C_hbm",
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
    parse_buffer_addrs=parse_buffer_addrs,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
