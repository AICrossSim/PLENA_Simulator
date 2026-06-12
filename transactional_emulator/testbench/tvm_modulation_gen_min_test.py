"""TVM minimal modulation-generation testbench.

Verifies the adaLN-Zero modulation generator:

    shift = silu(vec) @ W_shift.T
    scale = silu(vec) @ W_scale.T
    gate  = silu(vec) @ W_gate.T

for ONE selected vec (``pipe_idx``) out of a (1, pipelined, 1, HD) input.
Each Linear is a square HD->HD gemm; rows=1 LHS routes to M_MV / M_TMV
(transpose_B). K is accumulated by multiple single-block gemms + manual add.

Verification: MX-E4M3 round-trip on BOTH golden sides + HBM-direct compare
on SHIFT_hbm (the other two outputs share the identical path).
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
from tilelang_tvm_compiler.plena_settings import load_sizes as _load_sizes  # noqa: E402
from tilelang_tvm_compiler.test_helper import (  # noqa: E402
    TvmTestbenchSpec, resolve_output_layout,
)

sys.path.insert(0, str(_THIS_FILE.parent))
from _tb_cache import mx_roundtrip, run_cached  # noqa: E402


_HW = _load_sizes()
MLEN = _HW.mlen
HLEN = _HW.hlen
# HD follows geometry: HEAD*HLEN = (mlen//hlen*2)*hlen — SAME as the manager
# _validate_modulation run (HD=128 at mlen=64/hlen=16), so this matches the
# already-passing config. (HD=MLEN single-tile was triggering a different
# packing path.)
HEAD = (MLEN // HLEN) * 2
HD = HEAD * HLEN
PIPELINED = 4
PIPE_IDX = 2     # process the 3rd vec
SEQ = 2 * MLEN   # output broadcast over SEQ rows (2 s_blocks)


# Output SHIFT_hbm is (1, SEQ, 1, HD), written row-major as plain (SEQ, HD)
# tiles. HD > MLEN, but the kernel writes columns contiguously (no
# head-group-chunk reorder), so view it as (SEQ*HD/MLEN, MLEN): that keeps
# elements_per_batch == MLEN -> chunks_per_batch == 1 -> NO stride-mode
# reorder. (elements_per_batch=HD would set chunks_per_batch=2 and make the
# comparator reorder a side that isn't chunk-laid-out, corrupting the match.)
_OUT_LAYOUT = resolve_output_layout(
    num_batches=SEQ * (HD // MLEN),
    elements_per_batch=MLEN,
    mlen=MLEN,
)


def parse_buffer_addrs(raw: dict) -> dict:
    return {"result_hbm_start_byte": int(raw["SHIFT_hbm"]["address"])}


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)

    # VEC_hbm is (1, MLEN, 1, HD): pipelined vecs in the first PIPELINED rows,
    # rest is zero pad (DMA stages a full MLEN-row tile, like linear's A).
    vec2d = torch.randn(PIPELINED, HD, dtype=torch.float32) * 0.5
    vec_padded = torch.zeros(MLEN, HD, dtype=torch.float32)
    vec_padded[:PIPELINED] = vec2d
    vec = vec_padded.view(1, MLEN, 1, HD).contiguous()

    w_shift = torch.randn(HD, HD, dtype=torch.float32) * 0.1
    w_scale = torch.randn(HD, HD, dtype=torch.float32) * 0.1
    w_gate = torch.randn(HD, HD, dtype=torch.float32) * 0.1

    # MX-E4M3 round-trip each hop (MX blocks along the LAST dim).
    vec_eff = mx_roundtrip(vec_padded)                  # (MLEN, HD)
    ws_eff = mx_roundtrip(w_shift)                      # (HD, HD)

    # selected vec row -> (HD,). SiLU DISABLED (kernel's silu is commented
    # out): golden is plain vec @ W.T, broadcast across all SEQ rows.
    # Weights (N,K) row-major; kernel transpose_B=True.
    v_sel = vec_eff[PIPE_IDX]                            # (HD,)
    shift_row = v_sel @ ws_eff.T                        # (HD,)  [no silu]

    shift_golden = shift_row.view(1, HD).expand(SEQ, HD).contiguous()
    shift_golden = mx_roundtrip(shift_golden)           # (SEQ, HD)
    shift_golden = shift_golden.view(1, SEQ, 1, HD)

    return {
        "hbm_inputs": {
            "VEC_hbm": vec,
            "W_SHIFT": w_shift.view(1, HD, 1, HD),
            "W_SCALE": w_scale.view(1, HD, 1, HD),
            "W_GATE": w_gate.view(1, HD, 1, HD),
            "SHIFT_hbm": torch.zeros(1, SEQ, 1, HD),
            "SCALE_hbm": torch.zeros(1, SEQ, 1, HD),
            "GATE_hbm": torch.zeros(1, SEQ, 1, HD),
        },
        "golden_flat": _OUT_LAYOUT.flatten_golden(
            shift_golden.reshape(SEQ, HD)
        ),
    }


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io
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
    asm_name="modulation_gen_min",
    kernel="tilelang_tvm_compiler.kernels.modulation_gen_min:make_modulation_gen_min",
    kernel_kwargs={"hd": HD, "pipelined": PIPELINED, "pipe_idx": PIPE_IDX, "seq": SEQ},
    mlen=MLEN,
    btmm_hlen=HLEN,
    parse_buffer_addrs=parse_buffer_addrs,
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
)


def _fingerprint() -> dict:
    return {
        "kernel": "modulation_gen_min",
        "mlen": MLEN, "hlen": HLEN, "hd": HD, "seq": SEQ,
        "pipelined": PIPELINED, "pipe_idx": PIPE_IDX, "seed": 0, "schema": 1,
    }


if __name__ == "__main__":
    sys.exit(run_cached(SPEC, _fingerprint()))
