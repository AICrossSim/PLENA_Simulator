"""End-to-end test for the tiled-BTMM TVM kernel.

What it does:
    Outer loop over (q_block, kv_block); each iteration loads an mlen-row
    slice of A (BSHD) and an mlen-row slice of B (BSHD), runs BTMM,
    writes the resulting per-head tiles into the corresponding region of
    C_hbm (shape BSHS, multi-tile slice writeback).

Pipeline (mirrors tvm_btmm_test.py):
    1. Generate A / B / C inputs (torch, deterministic seed).
    2. Compute golden via einsum("bshd,bthd->bsht", A, B).
    3. Subprocess into .venv-tvm to compile the kernel + per-output-tile
       compare staging.
    4. Hand the ISA + .pt inputs + golden to the runtime helpers
       (create_sim_env, create_mem_for_sim) so the rest of
       `just build-emulator-debug` can drive the emulator.

Shape constants must agree with TILED_BTMM_DEFAULT_PARAMS in
compiler/tilelang_tvm_compiler/kernels/tiled_btmm.py. The kernel module
exports `tiled_btmm_default` baked with these shapes.
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
# Shape constants. These are passed to the kernel factory via
# `--kernel-kwargs` so the compiled kernel and the runtime data layout
# stay in lock-step.
#
# Constraint reminders:
#   * LANE_COUNT * HLEN  must equal MLEN (=64)  -- BTMM hardware shape
#   * HEAD_COUNT % LANE_COUNT == 0              -- clean head grouping
#   * SEQ_Q and SEQ_K MLEN-aligned              -- loop bound divisibility
#   * SEQ_Q == MLEN keeps the output to a single row block, which is the
#     only multi-tile slice geometry the stride-mode comparator handles
#     today. SEQ_Q > MLEN compiles fine but the comparison stage will
#     interleave row blocks vs. col blocks in a way `view_mem` can't
#     reassemble; the kernel's correctness still holds, just isn't
#     auto-checked.
# ---------------------------------------------------------------------------
BATCH = 1
SEQ_Q = 128         
SEQ_K = 128          # 2 kv_block iterations
HEAD_COUNT = 8       # total heads in the tensors -- exercises hg loop (NUM_HG = 8/4 = 2)
LANE_COUNT = 4       # hardware BTMM lane count (hardwired in the kernel)
HLEN = 16
MLEN = 64
NUM_Q = SEQ_Q // MLEN
NUM_K = SEQ_K // MLEN
NUM_HG = HEAD_COUNT // LANE_COUNT
ASM_NAME = "tvm_tiled_btmm_kernel"

VENV_TVM_PYTHON = str(_REPO_ROOT / ".venv-tvm" / "bin" / "python")


def compile_isa_via_tvm(
    asm_name: str,
    *,
    stage_output: str | None = None,
    dump_hlir: Path | None = None,
) -> str:
    """Subprocess into .venv-tvm to compile the tiled-BTMM kernel with
    shape parameters drawn from this test driver's constants -- so the
    compiled HBM addresses match what `create_mem_for_sim` will lay
    down on disk."""
    kernel_kwargs = (
        f"batch={BATCH},seq_q={SEQ_Q},seq_k={SEQ_K},"
        f"head_count={HEAD_COUNT},hlen={HLEN}"
    )
    cmd = [
        VENV_TVM_PYTHON, "-m", "tilelang_tvm_compiler", "compile",
        "--kernel", "tilelang_tvm_compiler.kernels.tiled_btmm:make_tiled_btmm",
        "--kernel-kwargs", kernel_kwargs,
        "--asm-name", asm_name,
        "--mlen", str(MLEN),
        "--btmm-lane-count", str(LANE_COUNT),
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
    """Generate A / B / C inputs + golden.

        A: (B, SEQ_Q,  H, D)
        B: (B, SEQ_K,  H, D)
        C: (B, SEQ_Q,  H, SEQ_K)   (zeros initially)
        golden[b, s, h, t] = sum_d A[b, s, h, d] * B[b, t, h, d]
    """
    torch.manual_seed(seed)
    A = torch.randn(BATCH, SEQ_Q, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    B = torch.randn(BATCH, SEQ_K, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.5
    C = torch.zeros(BATCH, SEQ_Q, HEAD_COUNT, SEQ_K, dtype=torch.float32)
    golden = torch.einsum("bshd,bthd->bsht", A, B)
    return {"A_hbm": A, "B_hbm": B, "C_hbm": C}, golden


def main() -> int:
    build_dir = _TESTBENCH_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Compiling tiled-BTMM via .venv-tvm (with output staging + HLIR dump) ...")
    hlir_path = build_dir / f"{ASM_NAME}.hlir.txt"
    isa_text = compile_isa_via_tvm(
        ASM_NAME,
        stage_output="C_hbm",
        dump_hlir=hlir_path,
    )
    print(f"      OK  ({isa_text.count(chr(10))} ISA lines, HLIR -> {hlir_path.name})")

    print(f"[2/4] Generating inputs + golden ...")
    inputs, golden = build_inputs_and_golden(seed=0)
    # Flatten golden BSHD -> (B*S, H*D) -- the layout the staged VRAM
    # gets reassembled into by the comparator.
    golden_flat = golden.reshape(BATCH * SEQ_Q, HEAD_COUNT * SEQ_K)
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

    # ----- comparison_params.json: tells view_mem.py how to read VRAM result.
    # Output C shape: (B, SEQ_Q, H, SEQ_K).  Logical 2D collapse: rows = B*SEQ_Q,
    # cols = H*SEQ_K.  With SEQ_Q == MLEN, row_blocks == 1, so the staged
    # VRAM is `col_blocks` consecutive 64-row chunks  -- exactly the shape
    # the stride-mode comparator expects.
    out_logical_rows = BATCH * SEQ_Q                  # 64 (= MLEN, single row block)
    out_logical_cols = HEAD_COUNT * SEQ_K             # 8 * 128 = 1024
    col_blocks = out_logical_cols // MLEN             # 16
    comparison_params = {
        "check_hbm": False,
        "start_row_idx": 0,
        "num_rows": out_logical_rows * col_blocks,    # 64 * 16 = 1024 VRAM rows
        "num_batches": out_logical_rows,              # 64 logical rows of golden
        "elements_per_batch": out_logical_cols,       # 1024 elements per row
        "row_dim": MLEN,                              # 64
    }
    cmp_path = build_dir / "comparison_params.json"
    cmp_path.write_text(json.dumps(comparison_params, indent=2))
    print(f"      wrote comparison_params.json -> {cmp_path}")

    print()
    print("=" * 60)
    print(f"build/ ready for: just build-emulator-debug tvm_tiled_btmm")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
