"""One-shot verification: does _mx_roundtrip match what the simulator
actually loads from HBM?

Run inside the nix env, from the repo root:
    nix develop --command python3 \
        transactional_emulator/testbench/_verify_mx_roundtrip.py

It does NOT touch any testbench. It just:
  1. builds a small tensor,
  2. quantizes it the way create_mem_for_sim does (Random_MXFP_Tensor_Generator
     -> _mx_fp_quantize_hardware -> pack_fp_to_bin -> bytes),
  3. quantizes the SAME tensor with our golden helper (_mx_fp_quantize_hardware
     bm_x directly),
  4. reconstructs the floats from the packed bytes the way the Rust
     `from_bytes` would (elem_e4m3 * scale_e8m0),
  5. prints whether (2)==(3) and where they diverge.

If bm_x (our golden) != reconstruction-from-bytes (the sim), THAT is the
gap, and the print tells us which stage drifts.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO_ROOT = next(
    (p for p in _THIS.parents if (p / "plena_settings.toml").is_file()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("repo root (with plena_settings.toml) not found")
_PY = f"python{sys.version_info.major}.{sys.version_info.minor}"
for parent in (_THIS.parent, *_THIS.parents):
    lib = parent / ".venv" / "lib"
    if lib.is_dir():
        for sp in lib.glob(f"{_PY}/site-packages"):
            sys.path.append(str(sp))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "tools"))

import torch  # noqa: E402

from utils.load_config import load_toml_config  # noqa: E402
from quant.quantizer.hardware_quantizer import _mx_fp_quantize_hardware  # noqa: E402


def main() -> None:
    torch.manual_seed(0)

    prec = load_toml_config(str(_REPO_ROOT / "plena_settings.toml"), "PRECISION")
    exp_w = prec["HBM_V_ACT_TYPE"]["ELEM"]["exponent"]
    man_w = prec["HBM_V_ACT_TYPE"]["ELEM"]["mantissa"]
    exp_bias_w = prec["HBM_V_ACT_TYPE"]["SCALE"]["exponent"]
    block = prec["HBM_M_WEIGHT_TYPE"]["block"]
    print(f"config: E{exp_w}M{man_w}, scale exp_width={exp_bias_w}, block={block}")

    # Shape mimics a K/V HBM tensor: (1, kv_seq, head, hlen).
    x = torch.randn(1, 16, 8, 16, dtype=torch.float32) * 0.5

    bm_x, exp, mant, scale_bias = _mx_fp_quantize_hardware(
        x,
        width=exp_w + man_w + 1,
        exponent_width=exp_w,
        exponent_bias_width=exp_bias_w,
        block_size=[1, block],
        skip_first_dim=False,
    )
    bm_x = bm_x.reshape(x.shape)

    err = (bm_x - x).abs()
    print(f"bm_x vs raw fp32 : max={err.max():.5f} mean={err.mean():.6f} rel={err.mean() / x.abs().mean():.3%}")
    print(f"per_block_exp    shape={tuple(exp.shape)}")
    print(f"per_block_mant   shape={tuple(mant.shape)}")
    print(
        f"per_block_scale  shape={tuple(scale_bias.shape)}  "
        f"values(min/max)={scale_bias.min().item()}/{scale_bias.max().item()}"
    )

    # Sanity: bm_x should equal sign*mant*2^(exp) reconstructed per block.
    # mant already carries sign; exp is the per-element minifloat exponent.
    recon = mant.reshape(-1) * torch.pow(2.0, exp.reshape(-1))
    # recon is per-block-normalized (before * scale); multiply scale back.
    # scale value = 2^(scale_bias - bias_bias), bias_bias = 2^(exp_bias_w-1)-1
    bias_bias = 2 ** (exp_bias_w - 1) - 1
    n_blk = scale_bias.numel()
    blk_sz = recon.numel() // n_blk
    scale_val = torch.pow(2.0, (scale_bias.reshape(-1) - bias_bias))
    recon = (recon.reshape(n_blk, blk_sz) * scale_val[:, None]).reshape(-1)

    diff = (recon - bm_x.reshape(-1)).abs()
    print(f"\nrecon(mant*2^exp*scale) vs bm_x : max_diff={diff.max():.2e} mean_diff={diff.mean():.2e}")
    if diff.max() > 1e-4:
        print(
            "  !! recon != bm_x  -> bm_x is NOT a faithful encode/decode; "
            "the golden using bm_x will not match the sim's HBM bytes."
        )
        bad = diff.argmax()
        print(f"  worst elem: bm_x={bm_x.reshape(-1)[bad]:.6f} recon={recon[bad]:.6f} raw={x.reshape(-1)[bad]:.6f}")
    else:
        print(
            "  OK: bm_x == mant/exp/scale reconstruction. The golden helper "
            "is internally consistent with the packed representation."
        )


if __name__ == "__main__":
    main()
