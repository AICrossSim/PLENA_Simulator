"""TVM GELU testbench — head-offset output variant.

Exercises ``make_gelu_min``'s ``o_head_count`` / ``o_head_offset``:
GELU reads its own compact ``head_count``-head input but writes the
result into a head-slice of a WIDER ``o_head_count``-head HBM tensor.

The single-stream-block chain uses this to drop GELU(mlp) into the
RIGHT half of ``concat([attn_out, mlp_out])``. This testbench sets
o_head_count = 2*head_count, o_head_offset = head_count (the "GELU
writes right half" case); the left half stays zero (staged scratch),
so a correct kernel writes exactly its slice and nothing else.
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

from tilelang_tvm_compiler.test_helper import TvmTestbenchSpec, run  # noqa: E402


BATCH = 1
ROWS = 64
HLEN = 16
MLEN = 64
HEAD_COUNT = 8
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS

# Output tensor is 2x wide; GELU writes the RIGHT half.
O_HEAD_COUNT = 2 * HEAD_COUNT
O_HEAD_OFFSET = HEAD_COUNT


def build_inputs_and_golden(seed: int = 0) -> dict:
    torch.manual_seed(seed)
    x = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    y = torch.nn.functional.gelu(x, approximate="tanh")

    # Place GELU output into the right-half head-slice of a 2x-wide
    # tensor; the left half stays zero (staged scratch).
    y_wide = torch.zeros(BATCH, SEQ_LEN, O_HEAD_COUNT, HLEN, dtype=torch.float32)
    y_wide[:, :, O_HEAD_OFFSET : O_HEAD_OFFSET + HEAD_COUNT, :] = y

    golden_flat = y_wide.reshape(BATCH * SEQ_LEN, O_HEAD_COUNT * HLEN)
    return {
        "hbm_inputs": {
            "X_hbm": x,
            "Y_hbm": torch.zeros(BATCH, SEQ_LEN, O_HEAD_COUNT, HLEN, dtype=torch.float32),
        },
        "golden_flat": golden_flat,
    }


def parse_buffer_addrs(raw: dict) -> dict:
    """Trivial passthrough. Its mere PRESENCE (non-None) is what matters:
    it makes test_helper pass --dump-buffer-addrs to the compiler and
    run the hoisted-const auto-preload. The GELU kernel embeds float
    coefficients (e.g. T.float16(0.5)), hoisted into __const_f16_*
    global.fpram slots — without auto-preload those slots stay 0.
    """
    del raw
    return {}


def build_comparison_params(io: dict, addrs: dict) -> dict:
    del io, addrs
    chunks_per_batch = (O_HEAD_COUNT * HLEN + MLEN - 1) // MLEN
    return {
        "check_hbm": False,
        "start_row_idx": 0,
        "num_rows": BATCH * SEQ_LEN * chunks_per_batch,
        "num_batches": BATCH * SEQ_LEN,
        "elements_per_batch": O_HEAD_COUNT * HLEN,
        "row_dim": MLEN,
        "compare_fpsram": False,
    }


SPEC = TvmTestbenchSpec(
    asm_name="gelu_offset",
    kernel="tilelang_tvm_compiler.kernels.gelu_min:make_gelu_min",
    kernel_kwargs={
        "rows": ROWS, "hlen": HLEN, "head_count": HEAD_COUNT,
        "num_s_blocks": NUM_S_BLOCKS, "batch": BATCH,
        "o_head_count": O_HEAD_COUNT, "o_head_offset": O_HEAD_OFFSET,
    },
    mlen=MLEN,
    btmm_hlen=HLEN,
    stage_output="Y_hbm",
    build_inputs_and_golden=build_inputs_and_golden,
    build_comparison_params=build_comparison_params,
    parse_buffer_addrs=parse_buffer_addrs,
)


if __name__ == "__main__":
    sys.exit(run(SPEC))
