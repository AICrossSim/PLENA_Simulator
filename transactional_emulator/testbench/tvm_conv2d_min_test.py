"""TVM Conv2D testbench — NCHW conv with optional multi-channel support.

Drives the ``tilelang_tvm_compiler.kernels.conv2d_min`` kernel:

    Input  (1, C_IN,  H_PAD, W_PAD)   pre-padded right/bottom
    Weight (C_OUT, C_IN, KH, KW)      flattened to FPRAM
    Output (1, C_OUT, H, W)

For the simplest case (C_IN=C_OUT=1) this matches the original
single-channel testbench.

C_OUT > 1 needs ``__main__._emit_output_staging`` to learn the per-
channel HBM stride; until that lands we stick with C_OUT == 1 and
defer multi-output-channel correctness checking.
"""

from __future__ import annotations

import sys
from pathlib import Path

# venv probing + sys.path setup. Has to run BEFORE we import torch or
# anything from the compiler tree.
_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _THIS_FILE.parents if (p / ".venv").is_dir() and (p / "compiler").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError(f"could not locate repo root (with .venv and compiler/) above {_THIS_FILE}")
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
import torch.nn.functional as F  # noqa: E402

from tilelang_tvm_compiler.address_alloc import FPRAM_USER_BASE  # noqa: E402
from tilelang_tvm_compiler.test_helper import (  # noqa: E402
    TvmTestbenchSpec,
    run,
    resolve_output_layout,
)


# ---------------------------------------------------------------------------
# Shape constants — multi-channel conv smoke test.
# ---------------------------------------------------------------------------
H = 64  # = MLEN
W = 64  # = MLEN
KH = 4  # KH * KW = HLEN
KW = 4
_HW = _load_sizes()  # hardware geometry — single source of truth, plena_settings.toml

MLEN = _HW.mlen  # from plena_settings.toml
HLEN = _HW.hlen  # from plena_settings.toml

# Multi-channel knobs. Defaults to single-channel for backward compat
# with the original conv2d_min smoke test. Stage_output staging only
# handles C_OUT == 1 today, so don't bump C_OUT until that's extended.
C_IN = 2
C_OUT = 2


def _round_up_to_mlen(x: int) -> int:
    return (x + MLEN - 1) // MLEN * MLEN


H_PAD = _round_up_to_mlen(H + KH - 1)  # 67 -> 128
W_PAD = _round_up_to_mlen(W + KW - 1)  # 67 -> 128
K_FLAT = KH * KW  # = HLEN = 16
OC_IC = C_OUT * C_IN

# Right-only / bottom-only padding (matches the kernel's zero-tail
# receptive-field assumption).
PAD_TOP = 0
PAD_BOTTOM_VALID = KH - 1
PAD_LEFT = 0
PAD_RIGHT_VALID = KW - 1
PAD_BOTTOM_TAIL = H_PAD - (H + KH - 1)
PAD_RIGHT_TAIL = W_PAD - (W + KW - 1)
PAD_BOTTOM = PAD_BOTTOM_VALID + PAD_BOTTOM_TAIL
PAD_RIGHT = PAD_RIGHT_VALID + PAD_RIGHT_TAIL
assert PAD_TOP + H + PAD_BOTTOM == H_PAD
assert PAD_LEFT + W + PAD_RIGHT == W_PAD


# ---------------------------------------------------------------------------
# Kernel-specific hooks
# ---------------------------------------------------------------------------


def _shape_elements(shape) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def parse_buffer_addrs(raw: dict) -> dict:
    """Pull addresses we need out of the compiler's
    ``--dump-buffer-addrs`` JSON.

    Just B_FP's FPRAM address — testbench preloads the weight tensor
    directly there (no VRAM staging cache, no pre-kernel S_MAP stub).
    """
    if "B_FP" not in raw:
        raise KeyError(f"buffer 'B_FP' not in HLIR dump; known: {sorted(raw)}")
    fpram_max_end = max(
        (int(b["address"]) + _shape_elements(b["shape"]) for b in raw.values() if b.get("scope") == "fpram"),
        default=FPRAM_USER_BASE,
    )
    return {
        "B_FP": int(raw["B_FP"]["address"]),
        "FPRAM_MAX_END": fpram_max_end,
    }


def build_inputs_and_golden(seed: int = 0) -> dict:
    """Build padded multi-channel Input + Weight + golden Output.

    Returned dict:
        - hbm_inputs: {"Input": (1, C_IN, H_PAD, W_PAD)} NCHW
        - golden_flat: (H, MLEN) — single-channel output flattened
                       (only one channel today; C_OUT > 1 needs
                       stage_output extension before its golden can be
                       checked end-to-end)
        - weight_flat: (OC_IC, K_FLAT) — taps in (oc, ic, kh*KW + kw)
                       order. ``build_fp_preload`` reads this dict.
    """
    torch.manual_seed(seed)
    input_logical = torch.randn(C_IN, H, W, dtype=torch.float32) * 0.5
    weight_logical = torch.randn(C_OUT, C_IN, KH, KW, dtype=torch.float32) * 0.5

    # F.pad with last dim varying fastest: (W_left, W_right, H_top, H_bottom).
    # Apply per-channel by padding each (H, W) slice.
    input_padded = F.pad(
        input_logical,
        (PAD_LEFT, PAD_RIGHT, PAD_TOP, PAD_BOTTOM),
        value=0.0,
    )
    assert tuple(input_padded.shape) == (C_IN, H_PAD, W_PAD), input_padded.shape

    in_nchw = input_padded.unsqueeze(0).contiguous()  # (1, C_IN, H_PAD, W_PAD)
    w_oihw = weight_logical.contiguous()  # (C_OUT, C_IN, KH, KW)
    golden_nchw = F.conv2d(in_nchw, w_oihw, padding=0)  # (1, C_OUT, H_PAD-KH+1, W_PAD-KW+1)
    golden = golden_nchw[0, :, :H, :W].contiguous()  # (C_OUT, H, W)

    # Stage_output staging walks the Output tile_layout in
    # (D_TILES, S_TILES, H_GROUPS, B) order. For NCHW
    # (1, C_OUT, MLEN, MLEN) with d_inner=MLEN, lane_count=1, we get
    # d_tiles=1, s_tiles=1, h_groups=C_OUT — i.e. one MLEN×MLEN tile
    # per output channel, dropped into VRAM[0..] consecutively in
    # channel order.
    #
    # So the comparator-side flat view is "channel-major" — channel 0's
    # H rows first, then channel 1's H rows, ... For W == MLEN each
    # logical row is one MLEN-wide chunk, so num_rows == num_batches
    # == C_OUT * H.
    golden_flat = golden.reshape(C_OUT * H, W)

    # Pack weight as (OC_IC, K_FLAT) in row-major (oc, ic) outer order
    # — same order ``B_cache[oc * C_IN + ic, k_tap]`` expects.
    weight_flat = weight_logical.reshape(OC_IC, K_FLAT).contiguous()

    return {
        "hbm_inputs": {"Input": in_nchw.to(torch.float32)},
        "golden_flat": golden_flat,
        "weight_flat": weight_flat,
    }


def build_fp_preload(io: dict, addrs: dict):
    """Preload weight tensor directly into ``FPRAM[B_FP]``.

    No pre-kernel stub: B_FP's FPRAM address comes from the
    address-allocation pass (read via ``--dump-buffer-addrs``), and
    the kernel reads ``B_FP[(oc * C_IN + ic) * MLEN + k_tap]``
    expecting the entry to already be there. We just write the right
    bytes at the right addresses before the kernel starts.

    Layout:
        FPRAM[B_FP + r * MLEN + k] = weight[oc, ic, kh, kw]
        where r = oc * C_IN + ic, k = kh * KW + kw, k < K_FLAT.
    Slots [K_FLAT, MLEN) of every row stay zero (kernel never reads
    past k_tap = HLEN - 1 anyway).
    """
    weight_flat = io["weight_flat"]
    assert weight_flat.shape == (OC_IC, K_FLAT), weight_flat.shape

    base = addrs["B_FP"]
    total = max(addrs["FPRAM_MAX_END"], base + OC_IC * MLEN)
    fp = torch.zeros(total, dtype=torch.float16)
    for r in range(OC_IC):
        row_start = base + r * MLEN
        fp[row_start : row_start + K_FLAT] = weight_flat[r].to(torch.float16)
    return fp


def build_comparison_params(io: dict, addrs: dict) -> dict:
    """``stage_output=Output`` reloads each output channel as one
    MLEN×MLEN tile into VRAM[0..] (channel-major). The output is the
    2D grid (C_OUT*H rows, W cols) — fed to the canonical OutputLayout
    in its explicit-2D form so golden flatten and the comparator agree.
    """
    del io, addrs
    layout = resolve_output_layout(
        num_batches=C_OUT * H,
        elements_per_batch=W,
        mlen=MLEN,
    )
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        "compare_fpsram": False,
        **layout.comparison_params(),
    }


SPEC = TvmTestbenchSpec(
    asm_name="conv2d_min",
    kernel="tilelang_tvm_compiler.kernels.conv2d_min:make_conv2d_min",
    kernel_kwargs={
        "h_in": H,
        "w_in": W,
        "kh": KH,
        "kw": KW,
        "c_in": C_IN,
        "c_out": C_OUT,
    },
    mlen=MLEN,
    stage_output="Output",
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_fp_preload=build_fp_preload,
    build_comparison_params=build_comparison_params,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
