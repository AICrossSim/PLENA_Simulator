"""End-to-end test for the tiled regular MM TVM kernel.

What it does:
    Per (q_block, head, d_block) output tile:
        - zero an accumulator C_v in VRAM
        - loop over kv_block (contracts T in MLEN chunks):
              load mlen*mlen single-head slice of A (BSHT) -> A_v (vram)
              load mlen*mlen single-head slice of B (BTHD) -> B_m (mram)
              M_MM single-tile MM into C_partial (vram)
              V_ADD C_v += C_partial
        - dma_v2h C_v -> mlen*mlen single-head slice of C (BSHD)

Pipeline mirrors tvm_tiled_btmm_test.py (subprocess into .venv-tvm to
compile, then create_sim_env / create_mem_for_sim to lay down inputs).

Constraints:
    * SEQ_Q, SEQ_K all multiples of MLEN (=64)
    * HEAD_COUNT * D must be a multiple of MLEN so the staged output can be
      decomposed into whole 64x64 tiles for comparison.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# venv probing -- main project venv (3.12, has torch) for sim env / golden
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PY_VERSION_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    _venv_lib = _parent / ".venv" / "lib"
    if not _venv_lib.is_dir():
        continue
    for _site_pkg in _venv_lib.glob(f"{_PY_VERSION_TAG}/site-packages"):
        sys.path.append(str(_site_pkg))

import numpy as np  # noqa: E402
import torch  # noqa: E402

_TESTBENCH_DIR = _THIS_FILE.parent
_REPO_ROOT = _TESTBENCH_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "compiler"))

from compiler.sim_env_utils import create_mem_for_sim  # noqa: E402
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402


# ---------------------------------------------------------------------------
# Shape constants. Constraint reminders:
#   * SEQ_Q / SEQ_K MLEN-aligned for loop divisibility
#   * HEAD_COUNT * D MLEN-aligned so staged output becomes an integral
#     number of 64-wide chunks per logical row. The stride-mode comparator
#     reassembles arbitrary col_block counts correctly from that layout.
# ---------------------------------------------------------------------------
BATCH = 1
SEQ_Q = 128           # S -> 2 row blocks
SEQ_K = 128          # T (contracted) -> NUM_K = 2  (exercises V_ADD streaming)
HEAD_COUNT = 8       # exercises the explicit (unrolled) head loop
D = 16               # output last dim -> NUM_D = 1
# NOTE: q_block / h / d_block are unrolled at compile time in the kernel
# (T.unroll), so per-iter dynamic instruction count stays bounded by one
# emit_matmul expansion + DMAs (~3k). Only kv_block remains a hardware
# loop. Scale H / NUM_D / NUM_Q freely without hitting MAX_LOOP_INSTRUCTIONS.
MLEN = 64
NUM_Q = SEQ_Q // MLEN
NUM_K = SEQ_K // MLEN
NUM_D = D // MLEN
ASM_NAME = "tvm_tiled_mm_kernel"

VENV_TVM_PYTHON = str(_REPO_ROOT / ".venv-tvm" / "bin" / "python")


def compile_isa_via_tvm(
    asm_name: str,
    *,
    stage_output: str | None = None,
    dump_hlir: Path | None = None,
) -> str:
    kernel_kwargs = (
        f"batch={BATCH},seq_q={SEQ_Q},seq_k={SEQ_K},"
        f"head_count={HEAD_COUNT},d_dim={D}"
    )
    cmd = [
        VENV_TVM_PYTHON, "-m", "tilelang_tvm_compiler", "compile",
        "--kernel", "tilelang_tvm_compiler.kernels.tiled_mm:make_tiled_mm",
        "--kernel-kwargs", kernel_kwargs,
        "--asm-name", asm_name,
        "--mlen", str(MLEN),
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
    """Generate A / B / C inputs + golden.

        A: (B, S, H, T)       BSHT
        B: (B, T, H, D)       BTHD
        C: (B, S, H, D)       BSHD  (zeros initially)
        golden[b, s, h, d] = sum_t A[b, s, h, t] * B[b, t, h, d]
    """
    torch.manual_seed(seed)
    A = torch.randn(BATCH, SEQ_Q, HEAD_COUNT, SEQ_K, dtype=torch.float32) * 0.5
    B = torch.randn(BATCH, SEQ_K, HEAD_COUNT, D, dtype=torch.float32) * 0.5
    C = torch.zeros(BATCH, SEQ_Q, HEAD_COUNT, D, dtype=torch.float32)
    golden = torch.einsum("bsht,bthd->bshd", A, B)
    return {"A_hbm": A, "B_hbm": B, "C_hbm": C}, golden


def main() -> int:
    build_dir = _TESTBENCH_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Compiling tiled-MM via .venv-tvm (with output staging + HLIR dump) ...")
    hlir_path = build_dir / f"{ASM_NAME}.hlir.txt"
    isa_text = compile_isa_via_tvm(
        ASM_NAME,
        stage_output="C_hbm",
        dump_hlir=hlir_path,
    )
    print(f"      OK  ({isa_text.count(chr(10))} ISA lines, HLIR -> {hlir_path.name})")

    print(f"[2/4] Generating inputs + golden ...")
    inputs, golden = build_inputs_and_golden(seed=0)
    # Output BSHD -> 2D collapse: rows = B*S, cols = H*D.
    golden_flat = golden.reshape(BATCH * SEQ_Q, HEAD_COUNT * D)
    print(
        f"      OK  inputs: {list(inputs)}  "
        f"golden: 4D{tuple(golden.shape)} -> flat{tuple(golden_flat.shape)}"
    )

    input_feed = {name: t.contiguous().reshape(1, -1) for name, t in inputs.items()}
    input_order = list(input_feed)

    print(f"[3/4] create_sim_env -> .pt + .asm + fp/int sram bins ...")
    create_sim_env(
        input_tensor=input_feed,
        generated_code=isa_text,
        golden_result={"original_output": golden_flat},
        fp_preload=None,
        int_preload=None,
        build_dir=str(build_dir),
    )
    print(f"      OK  -> {build_dir}")

    print(f"[4/4] create_mem_for_sim -> assemble .asm + pack HBM bin ...")
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=ASM_NAME,
        data=None,
        specified_data_order=input_order,
        build_path=build_dir,
    )
    print(f"      OK  -> generated_machine_code.mem + hbm_for_behave_sim.bin")

    # ----- comparison_params.json -----
    # Output C: (B, S, H, D). Logical 2D collapse: rows = B*S, cols = H*D.
    # Staging dumps tiles in col-block-major order; the stride-mode
    # comparator reorders those 64-wide chunks back into per-row layout
    # using `num_batches = rows` and `elements_per_batch = cols`.
    out_logical_rows = BATCH * SEQ_Q
    out_logical_cols = HEAD_COUNT * D
    if out_logical_cols % MLEN:
        raise RuntimeError(
            "tvm_tiled_mm_test comparator requires HEAD_COUNT * D to be MLEN-aligned "
            f"(got HEAD_COUNT * D = {out_logical_cols}, MLEN = {MLEN})"
        )
    col_blocks = out_logical_cols // MLEN
    comparison_params = {
        "check_hbm": False,
        "start_row_idx": 0,
        "num_rows": out_logical_rows * col_blocks,
        "num_batches": out_logical_rows,
        "elements_per_batch": out_logical_cols,
        "row_dim": MLEN,
    }
    cmp_path = build_dir / "comparison_params.json"
    cmp_path.write_text(json.dumps(comparison_params, indent=2))
    print(f"      wrote comparison_params.json -> {cmp_path}")

    print()
    print("=" * 60)
    print(f"build/ ready for: just build-emulator-debug tvm_tiled_mm")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
