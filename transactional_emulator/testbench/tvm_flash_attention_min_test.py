"""TVM minimal FlashAttention testbench (single q-block, single kv-block).

Mirrors `tvm_online_softmax_min.py`. Drives the
`tilelang_tvm_compiler.kernels.flash_attention_min` kernel through the
TVM venv subprocess, builds matching inputs / golden / FP preload, and
emits the same artefact set the emulator runner expects.

What the kernel computes (all heads):
    score = Q @ K^T                  (BTMM #1, all lanes get raw score)
    For each head lane:
        P  = exp(score - max(score, dim=-1, keepdim=True))   (un-normalised)
    O = P @ V                        (BTMM #2)
    O_v output is written to O_hbm
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
# Walk up from this file to find the repo root (the dir that contains
# `.venv-tvm` and `compiler/`). The build recipe sometimes places this
# script at testbench/<name>_test.py and sometimes at
# testbench/tile_tensor_kernel_programs/<name>.py, so we can't hard-code
# the depth.
_REPO_ROOT = next(
    (p for p in _THIS_FILE.parents if (p / ".venv-tvm").is_dir() and (p / "compiler").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError(
        f"could not locate repo root (with .venv-tvm and compiler/) above {_THIS_FILE}"
    )
_TESTBENCH_DIR = _REPO_ROOT / "transactional_emulator" / "testbench"
_PY_VERSION_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    _venv_lib = _parent / ".venv" / "lib"
    if not _venv_lib.is_dir():
        continue
    for _site_pkg in _venv_lib.glob(f"{_PY_VERSION_TAG}/site-packages"):
        sys.path.append(str(_site_pkg))

import torch  # noqa: E402

sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "compiler"))

from compiler.sim_env_utils import create_mem_for_sim  # noqa: E402
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402


BATCH = 1
ROWS = 64           # q-tile length == mlen
HLEN = 16
LANE_COUNT = 4      # group_heads
ACTIVE_LANE = 2
MLEN = 64
NUM_KV_BLOCKS = 2   # exercise the multi-block FlashAttention path
NUM_Q_BLOCKS = 2    # multi-Q outer loop
KV_SEQ = NUM_KV_BLOCKS * ROWS
Q_SEQ = NUM_Q_BLOCKS * ROWS
ASM_NAME = "flash_attention_min"
ARTIFACT_PREFIX = "flash_attention_min"
VENV_TVM_PYTHON = str(_REPO_ROOT / ".venv-tvm" / "bin" / "python")
# Kept in sync with tilelang_tvm_compiler.address_alloc:FPRAM_USER_BASE.
FPRAM_USER_BASE = 32
FP_STATE_ELEMS = LANE_COUNT * ROWS
ACTIVE_LANE_ROW_BASE = ACTIVE_LANE * ROWS
# FP buffer addresses follow the kernel's T.alloc_buffer declaration order:
# M_old, M_curr, M_res, L_old, L_new, P_sum, Scale, L_inv, M_init, L_init.
SCALE_ADDR  = FPRAM_USER_BASE + 6 * FP_STATE_ELEMS
M_INIT_ADDR = FPRAM_USER_BASE + 8 * FP_STATE_ELEMS
L_INIT_ADDR = FPRAM_USER_BASE + 9 * FP_STATE_ELEMS

# A finite "negative-infinity" surrogate compatible with float16 and the
# kernel's own FP-scalar arithmetic. Mirrors attention.py's choice.
NEG_INF = -1.0e4


def compile_isa_via_tvm(
    asm_name: str,
    *,
    stage_output: str | None = None,
    dump_hlir: Path | None = None,
) -> str:
    kernel_kwargs = (
        f"rows={ROWS},hlen={HLEN},lane_count={LANE_COUNT},"
        f"active_lane={ACTIVE_LANE},"
        f"num_kv_blocks={NUM_KV_BLOCKS},num_q_blocks={NUM_Q_BLOCKS}"
    )
    cmd = [
        VENV_TVM_PYTHON, "-m", "tilelang_tvm_compiler", "compile",
        "--kernel", "tilelang_tvm_compiler.kernels.flash_attention_min:make_flash_attention_min",
        "--kernel-kwargs", kernel_kwargs,
        "--asm-name", asm_name,
        "--mlen", str(MLEN),
        "--btmm-hlen", str(HLEN),
    ]
    if stage_output is not None:
        cmd += ["--stage-output", stage_output]
    if dump_hlir is not None:
        cmd += ["--dump-hlir", str(dump_hlir)]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ""
    env["PYTHONPATH"] = str(_REPO_ROOT / "compiler")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"TVM subprocess failed (returncode={res.returncode}). See stderr above."
        )
    return res.stdout


def build_inputs_and_golden(seed: int = 0):
    """Build Q/K/V tensors and per-head golden output."""
    torch.manual_seed(seed)
    q = torch.randn(BATCH, Q_SEQ, LANE_COUNT, HLEN, dtype=torch.float32) * 0.5
    k = torch.randn(BATCH, KV_SEQ, LANE_COUNT, HLEN, dtype=torch.float32) * 0.5
    v = torch.randn(BATCH, KV_SEQ, LANE_COUNT, HLEN, dtype=torch.float32) * 0.5

    scale = 1.0 / math.sqrt(HLEN)

    # Per-head: score[b, i, h, j] = sum_d Q[b,i,h,d] * K[b,j,h,d]
    # i ranges over the full Q_SEQ, j over the full KV_SEQ.
    score = torch.einsum("bihd,bjhd->bihj", q, k)  # (B, Q_SEQ, H, KV_SEQ)

    out = torch.empty(BATCH, Q_SEQ, LANE_COUNT, HLEN, dtype=torch.float32)
    for h in range(LANE_COUNT):
        score_h = score[:, :, h, :]  # (B, Q_SEQ, KV_SEQ)
        v_h = v[:, :, h, :]          # (B, KV_SEQ, hlen)
        # Full softmax(scale * score) @ V over the entire kv sequence,
        # for every Q row. The kernel computes this in NUM_Q_BLOCKS
        # outer iterations, each with NUM_KV_BLOCKS online softmax
        # steps -- mathematically invariant to the block split.
        scaled = score_h * scale
        p = torch.softmax(scaled, dim=-1)
        out_h = torch.einsum("bij,bjd->bid", p, v_h)
        out[:, :, h, :] = out_h

    golden_flat = out.reshape(BATCH * Q_SEQ, LANE_COUNT * HLEN)

    return (
        {
            "Q_hbm": q,
            "K_hbm": k,
            "V_hbm": v,
            "O_hbm": torch.zeros_like(q),
        },
        golden_flat,
    )


def build_fp_preload() -> torch.Tensor:
    """FP preload (read-only constants the kernel copies from per Q tile):
      * Scale[h,  :] = 1 / sqrt(d_k)
      * M_init[h, :] = -inf surrogate
      * L_init[h, :] = 0
    M_old / L_old themselves don't need preloading anymore -- the kernel
    resets them from M_init / L_init at the start of every q_block.
    """
    # 10 fp buffers in declaration order:
    #   M_old, M_curr, M_res, L_old, L_new, P_sum, Scale, L_inv, M_init, L_init
    total = FPRAM_USER_BASE + 10 * FP_STATE_ELEMS
    fp = torch.zeros(total, dtype=torch.float16)

    scale_val = 1.0 / math.sqrt(HLEN)
    for h in range(LANE_COUNT):
        row_base = h * ROWS
        scale_start = SCALE_ADDR + row_base
        fp[scale_start:scale_start + ROWS] = scale_val

        m_init_start = M_INIT_ADDR + row_base
        fp[m_init_start:m_init_start + ROWS] = float(NEG_INF)

        l_init_start = L_INIT_ADDR + row_base
        fp[l_init_start:l_init_start + ROWS] = 0.0
    return fp


def main() -> int:
    build_dir = _TESTBENCH_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Compiling TVM flash-attention-min kernel ...")
    hlir_path = build_dir / f"{ASM_NAME}.hlir.txt"
    isa_text = compile_isa_via_tvm(
        ASM_NAME,
        stage_output="O_hbm",
        dump_hlir=hlir_path,
    )
    print(f"      OK  ({isa_text.count(chr(10))} ISA lines, HLIR -> {hlir_path.name})")

    print("[2/4] Generating inputs, FPSRAM preload, and golden ...")
    inputs, golden_flat = build_inputs_and_golden(seed=0)
    fp_preload = build_fp_preload()
    input_feed = {name: t.contiguous().reshape(1, -1) for name, t in inputs.items()}
    input_order = list(input_feed)
    print(
        f"      OK  Q shape={tuple(inputs['Q_hbm'].shape)}  "
        f"golden flat={tuple(golden_flat.shape)}  fp_preload={tuple(fp_preload.shape)}"
    )

    print("[3/4] create_sim_env -> .pt + .asm + fp/int sram bins ...")
    create_sim_env(
        input_tensor=input_feed,
        generated_code=isa_text,
        golden_result={"original_output": golden_flat},
        fp_preload=fp_preload,
        int_preload=None,
        build_dir=str(build_dir),
    )
    print(f"      OK  -> {build_dir}")

    print("[4/4] create_mem_for_sim -> assemble .asm + pack HBM bin ...")
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=ASM_NAME,
        data=None,
        specified_data_order=input_order,
        build_path=build_dir,
    )
    print("      OK  -> generated_machine_code.mem + hbm_for_behave_sim.bin")

    comparison_params = {
        "check_hbm": False,
        "start_row_idx": 0,
        "num_rows": BATCH * Q_SEQ,
        "num_batches": BATCH * Q_SEQ,
        "elements_per_batch": LANE_COUNT * HLEN,
        "row_dim": MLEN,
        "compare_fpsram": False,
    }
    cmp_path = build_dir / "comparison_params.json"
    cmp_path.write_text(json.dumps(comparison_params, indent=2))

    (build_dir / f"{ARTIFACT_PREFIX}_generated_asm_code.asm").write_text(isa_text)

    print()
    print("=" * 60)
    print("build/ ready for emulator run")
    print(f"script: {_THIS_FILE.name}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
