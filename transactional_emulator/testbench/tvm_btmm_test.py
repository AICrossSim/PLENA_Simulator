"""End-to-end driver for the TVM-compiled BTMM kernel.

Modeled after tile_tensor_kernel_programs/linear.py: this script is the
single thing the user runs. It does everything the runtime kernel
programs do, just with the ISA coming from the TVM compiler instead of
TileTensorProgram.

Pipeline:
    1. Generate inputs (torch, deterministic seed)
    2. Compute golden (torch.einsum)
    3. Subprocess into .venv-tvm to compile TIR -> PLENA ISA text
       (we cannot import tvm in this 3.12 venv -- the wheel is 3.11 only)
    4. Call runtime's create_sim_env(input_tensor, generated_code, ...)
       to lay down .pt input files, generated_asm_code.asm, fp_sram.bin,
       int_sram.bin, golden_result.txt
    5. Call runtime's create_mem_for_sim(specified_data_order, ...)
       to assemble the .asm into generated_machine_code.mem and pack
       the .pt files into hbm_for_behave_sim.bin

After this script runs, build/ has all the artifacts that
`just build-emulator-debug` expects, so the rest of the just recipe
(cargo run + view_mem.py) can do its thing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Locate the main project venv. `python3` invoked from `just` does not always
# point at the venv interpreter, so its site-packages may be missing -- exactly
# the trick used in tile_tensor_kernel_programs/linear.py. Walk up from this
# file looking for `.venv/lib/python<X.Y>/site-packages` and add to sys.path.
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_PY_VERSION_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    _venv_lib = _parent / ".venv" / "lib"
    if not _venv_lib.is_dir():
        continue
    for _site_pkg in _venv_lib.glob(f"{_PY_VERSION_TAG}/site-packages"):
        sys.path.append(str(_site_pkg))

import numpy as np  # noqa: E402  -- after sys.path patch above
import torch  # noqa: E402

_TESTBENCH_DIR = _THIS_FILE.parent
_REPO_ROOT = _TESTBENCH_DIR.parent.parent

# Make compiler.* (sim_env_utils etc.) importable in the main venv.
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "compiler"))

from compiler.sim_env_utils import create_mem_for_sim  # noqa: E402
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402


# ---------------------------------------------------------------------------
# Constants -- mirror tilelang_tvm_compiler.kernels.minimal_btmm
# (BSHD layout: GROUP_HEADS * HLEN == MLEN so the "head merge" in the
#  address_alloc pass collapses to one mlen-wide tile per HBM row.)
# ---------------------------------------------------------------------------
BATCH = 1
SEQ = 64
MLEN = 64
GROUP_HEADS = 4
HLEN = 16
ASM_NAME = "tvm_btmm_kernel"

# Where the .venv-tvm Python lives. Kept as a constant so the entire
# venv-bridging concern is one line to update.
VENV_TVM_PYTHON = str(_REPO_ROOT / ".venv-tvm" / "bin" / "python")


def compile_isa_via_tvm(
    asm_name: str,
    *,
    stage_output: str | None = None,
    dump_hlir: Path | None = None,
) -> str:
    """Subprocess into .venv-tvm to compile minimal_btmm; return ISA text.

    If stage_output is set, the CLI also appends per-tile DMAs that re-load
    the named output buffer from HBM into VRAM[0..]. This is the equivalent
    of `stage_input_tensor_for_stride_compare()` in the runtime testbench:
    view_mem.py reads VRAM[0..] and compares against golden_result.txt.

    If dump_hlir is given, the post-address-alloc HLIR is written there too.
    """
    cmd = [
        VENV_TVM_PYTHON, "-m", "tilelang_tvm_compiler", "compile",
        "--kernel", "tilelang_tvm_compiler.kernels.minimal_btmm:minimal_btmm",
        "--asm-name", asm_name,
        "--mlen", str(MLEN),
        "--btmm-lane-count", str(GROUP_HEADS),
        "--btmm-hlen", str(HLEN),
    ]
    if stage_output is not None:
        cmd += ["--stage-output", stage_output]
    if dump_hlir is not None:
        cmd += ["--dump-hlir", str(dump_hlir)]
    env = os.environ.copy()
    # The Nix-provided libstdc++ in LD_LIBRARY_PATH conflicts with TVM's
    # manylinux wheel; clear it just for the subprocess.
    env["LD_LIBRARY_PATH"] = ""
    env["PYTHONPATH"] = str(_REPO_ROOT / "compiler")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stderr)
        raise RuntimeError(
            f"TVM subprocess failed (returncode={res.returncode}). See stderr above."
        )
    return res.stdout


def build_btmm_inputs_and_golden(
    seed: int = 0,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Generate torch inputs + golden for the BTMM kernel.

    BSHD layout. Per-head Q @ K.T:
        A: (B, S, H, D)             -- Q
        B: (B, S, H, D)             -- K (we matmul A @ B.T over D, per head)
        C: (B, S, H, S)             -- per-head attention scores
        C[b, s, h, t] = sum_d A[b, s, h, d] * B[b, t, h, d]
    """
    torch.manual_seed(seed)
    A = torch.randn(BATCH, SEQ, GROUP_HEADS, HLEN, dtype=torch.float32) * 0.5
    B = torch.randn(BATCH, SEQ, GROUP_HEADS, HLEN, dtype=torch.float32) * 0.5
    C = torch.zeros(BATCH, SEQ, GROUP_HEADS, MLEN, dtype=torch.float32)
    golden = torch.einsum("bshd,bthd->bsht", A, B)
    return {"A_hbm": A, "B_hbm": B, "C_hbm": C}, golden


def main() -> int:
    build_dir = _TESTBENCH_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Compiling TIR -> ISA via .venv-tvm (with output staging + HLIR dump) ...")
    hlir_path = build_dir / f"{ASM_NAME}.hlir.txt"
    isa_text = compile_isa_via_tvm(
        ASM_NAME,
        stage_output="C_hbm",
        dump_hlir=hlir_path,
    )
    print(f"      OK  ({isa_text.count(chr(10))} ISA lines, HLIR -> {hlir_path.name})")

    print(f"[2/4] Generating inputs + golden ...")
    inputs, golden = build_btmm_inputs_and_golden(seed=0)
    # Flatten golden to (B*S, H*D) -- the same flat layout the staged VRAM
    # ends up in after our --stage-output tiles get pulled back from HBM
    # in col-block-major order. compare_vram_with_golden uses "stride mode"
    # to reassemble; without this reshape the comparator interprets the
    # 4D tensor's first dim as rows and the layout breaks.
    golden_flat = golden.reshape(BATCH * SEQ, GROUP_HEADS * MLEN)
    print(
        f"      OK  inputs: {list(inputs)}  "
        f"golden 4D: {tuple(golden.shape)} -> flat: {tuple(golden_flat.shape)}"
    )

    # build_input_feed-equivalent: keys become .pt filenames; flatten to (1, -1).
    # Using bare names (no ".hbm" suffix) keeps the artifact list short and
    # mirrors the kernel's PrimFunc parameter names.
    input_feed = {name: t.contiguous().reshape(1, -1) for name, t in inputs.items()}
    input_order = list(input_feed)

    print(f"[3/4] create_sim_env -> .pt + .asm + fp/int sram bins ...")
    create_sim_env(
        input_tensor=input_feed,
        generated_code=isa_text,
        golden_result={"original_output": golden_flat},
        fp_preload=None,        # BTMM kernel uses no FP constants
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
    # Output C shape: (BATCH, SEQ, GROUP_HEADS, MLEN) = (1, 64, 4, 64).
    # BSHD logical-2D collapse (H*D merge): rows = B*S = 64, cols = H*D = 256.
    # With MLEN=64 tiles: 1 row-block x 4 col-blocks, staged contiguously
    # into VRAM[0..256-rows] (4 tiles * 64 rows = 256 VRAM rows).
    out_logical_rows = BATCH * SEQ                   # 64
    out_logical_cols = GROUP_HEADS * MLEN            # 256
    row_blocks = out_logical_rows // MLEN
    col_blocks = out_logical_cols // MLEN
    comparison_params = {
        "check_hbm": False,                      # comparing VRAM staged region
        "start_row_idx": 0,
        "num_rows": out_logical_rows * col_blocks,   # VRAM rows to read
        "num_batches": out_logical_rows,             # logical rows of golden
        "elements_per_batch": out_logical_cols,      # logical cols of golden
        "row_dim": MLEN,                             # bytes-per-row width
    }
    cmp_path = build_dir / "comparison_params.json"
    cmp_path.write_text(json.dumps(comparison_params, indent=2))
    print(f"      wrote comparison_params.json -> {cmp_path}")

    print()
    print("=" * 60)
    print(f"build/ ready for: just build-emulator-debug tvm_btmm")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
