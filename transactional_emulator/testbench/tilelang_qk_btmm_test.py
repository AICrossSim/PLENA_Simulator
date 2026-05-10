"""End-to-end driver for the tilelang Q@K^T BTMM kernel.

Mirrors `tvm_btmm_test.py`'s flow but compiles the kernel through the
new tilelang frontend (in-process, single venv -- no subprocess hop).
The kernel is `qk_btmm_tilelang.make_qk_btmm`.

Pipeline:
    1. Generate inputs (torch, deterministic seed)
    2. Compute golden (per-head Q @ K.T via einsum)
    3. Compile tilelang kernel -> ISA
    4. Lay down sim env (.pt inputs, generated_asm_code.asm, fp/int sram bins)
    5. Pack .asm into .mem and HBM tensors into hbm_for_behave_sim.bin

After this script: `just build-emulator-debug tilelang_qk_btmm` runs
the rest (cargo run + view_mem.py compare).
"""

import json
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PY_VERSION_TAG = f"python{sys.version_info.major}.{sys.version_info.minor}"
for _parent in (_THIS_FILE.parent, *_THIS_FILE.parents):
    _venv_lib = _parent / ".venv" / "lib"
    if not _venv_lib.is_dir():
        continue
    for _site_pkg in _venv_lib.glob(f"{_PY_VERSION_TAG}/site-packages"):
        sys.path.append(str(_site_pkg))

import torch  # noqa: E402

_TESTBENCH_DIR = _THIS_FILE.parent
_REPO_ROOT = _TESTBENCH_DIR.parent.parent

sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "compiler"))

from compiler.sim_env_utils import create_mem_for_sim  # noqa: E402
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402

import tilelang_tvm_compiler  # noqa: E402  -- bootstraps tilelang's TVM 0.23
from tilelang_tvm_compiler.kernels.qk_btmm_tilelang import make_qk_btmm  # noqa: E402
from tilelang_tvm_compiler.frontend import compile_func, compile_to_tir_text  # noqa: E402
from tilelang_tvm_compiler.pipeline import compile_kernel, PlenaTarget  # noqa: E402
from tilelang_tvm_compiler.hlir import format_hlir  # noqa: E402


BATCH = 1
SEQ = 64
MLEN = 64
LANE_COUNT = 4
HLEN = 16
ASM_NAME = "tilelang_qk_btmm"


def build_inputs_and_golden(seed: int = 0):
    """Q, K in BSHD layout. Per-head output S = Q @ K^T, shape (B, S_q, H, S_kv)."""
    torch.manual_seed(seed)
    Q = torch.randn(BATCH, SEQ, LANE_COUNT, HLEN, dtype=torch.float32) * 0.5
    K = torch.randn(BATCH, SEQ, LANE_COUNT, HLEN, dtype=torch.float32) * 0.5
    golden = torch.einsum("bshd,bthd->bsht", Q, K)
    return {"Q_hbm": Q, "K_hbm": K}, golden


def main() -> int:
    build_dir = _TESTBENCH_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Compiling tilelang Q@K^T kernel (in-process) ...")
    src = make_qk_btmm()
    # Dump the lowered TIR (post-frontend, pre-codegen) for inspection.
    tir_text = compile_to_tir_text(src, name=ASM_NAME)
    func = compile_func(src)
    ck = compile_kernel(func, target=PlenaTarget(), name=ASM_NAME)
    isa_text = ck.isa_text
    # Dump intermediate artifacts alongside the ISA so build/ has the
    # full picture for debugging.
    (build_dir / f"{ASM_NAME}.tir.txt").write_text(tir_text)
    (build_dir / f"{ASM_NAME}.hlir.txt").write_text(format_hlir(ck.hlir))
    print(f"      OK  {isa_text.count(chr(10))} ISA lines  "
          f"(TIR + HLIR dumped to build/{ASM_NAME}.{{tir,hlir}}.txt)")

    print("[2/4] Generating inputs + golden ...")
    inputs, golden = build_inputs_and_golden(seed=0)
    # BTMM lays out S in BHSD: 4 lanes, each (S_q=64, S_kv=64). The
    # golden's einsum is BSHD; permute to (B, H, S_q, S_kv) then flatten.
    golden_bhsd = golden.permute(0, 2, 1, 3).contiguous()
    golden_flat = golden_bhsd.reshape(LANE_COUNT * SEQ, MLEN)
    print(
        f"      OK  Q shape={tuple(inputs['Q_hbm'].shape)}  "
        f"K shape={tuple(inputs['K_hbm'].shape)}  "
        f"golden flat={tuple(golden_flat.shape)}"
    )

    input_feed = {name: t.contiguous().reshape(1, -1) for name, t in inputs.items()}
    input_order = list(input_feed)

    print("[3/4] create_sim_env -> .pt + .asm + fp/int sram bins ...")
    create_sim_env(
        input_tensor=input_feed,
        generated_code=isa_text,
        golden_result={"original_output": golden_flat},
        fp_preload=None,
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

    # S_loc lives at vram addr 4096 (after Q_sh occupies vram[0..4095]).
    # 4 BHSD tiles of 64x64 -> 256 vram rows.
    comparison_params = {
        "check_hbm": False,
        "start_row_idx": 64,            # skip Q_sh
        "num_rows": LANE_COUNT * SEQ,
        "num_batches": LANE_COUNT * SEQ,
        "elements_per_batch": MLEN,
        "row_dim": MLEN,
    }
    cmp_path = build_dir / "comparison_params.json"
    cmp_path.write_text(json.dumps(comparison_params, indent=2))
    print(f"      wrote comparison_params.json -> {cmp_path}")

    (build_dir / f"{ASM_NAME}_generated_asm_code.asm").write_text(isa_text)

    print()
    print("=" * 60)
    print(f"build/ ready for: just build-emulator-debug {ASM_NAME}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
