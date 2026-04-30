"""TVM online-softmax minimal testbench for the transactional emulator.

Default configuration intentionally exercises the packed HLEN path:
  * shape: (1, 64, 4, 16)
  * active lane selected by V_MASK
  * m_old / l_old initial state preloaded into FPSRAM
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_TESTBENCH_DIR = _THIS_FILE.parent.parent
_REPO_ROOT = _TESTBENCH_DIR.parent.parent
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
ROWS = 64
HLEN = 16
LANE_COUNT = 4
ACTIVE_LANE = 2
MLEN = 64
ASM_NAME = "tvm_online_softmax_min"
ARTIFACT_PREFIX = "tvm_online_softmax_min"
VENV_TVM_PYTHON = str(_REPO_ROOT / ".venv-tvm" / "bin" / "python")
# Keep this in sync with tilelang_tvm_compiler.address_alloc:FPRAM_USER_BASE.
# The testbench runs outside the TVM env, so importing the compiler package
# here would pull in `tvm` too early and fail on hosts without that module.
FPRAM_USER_BASE = 32
FP_STATE_ELEMS = LANE_COUNT * ROWS
M_OLD_ADDR = FPRAM_USER_BASE + 0 * FP_STATE_ELEMS
M_CURR_ADDR = FPRAM_USER_BASE + 1 * FP_STATE_ELEMS
M_RES_ADDR = FPRAM_USER_BASE + 2 * FP_STATE_ELEMS
L_OLD_ADDR = FPRAM_USER_BASE + 3 * FP_STATE_ELEMS
L_NEW_ADDR = FPRAM_USER_BASE + 4 * FP_STATE_ELEMS
P_SUM_ADDR = FPRAM_USER_BASE + 5 * FP_STATE_ELEMS


def compile_isa_via_tvm(
    asm_name: str,
    *,
    stage_output: str | None = None,
    dump_hlir: Path | None = None,
) -> str:
    kernel_kwargs = (
        f"rows={ROWS},hlen={HLEN},lane_count={LANE_COUNT},active_lane={ACTIVE_LANE}"
    )
    cmd = [
        VENV_TVM_PYTHON, "-m", "tilelang_tvm_compiler", "compile",
        "--kernel", "tilelang_tvm_compiler.kernels.online_softmax_min:make_online_softmax_hbm",
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
    torch.manual_seed(seed)
    score = torch.randn(BATCH, ROWS, LANE_COUNT, HLEN, dtype=torch.float32) * 0.5
    m_old = torch.randn(LANE_COUNT, ROWS, dtype=torch.float32) * 0.2
    l_old = torch.rand(LANE_COUNT, ROWS, dtype=torch.float32) * 0.5 + 0.5

    score_out = score.clone()
    exp_m_res = torch.empty(LANE_COUNT, ROWS, dtype=torch.float32)
    l_new = torch.empty(LANE_COUNT, ROWS, dtype=torch.float32)
    for lane in range(LANE_COUNT):
        lane_slice = slice(lane, lane + 1)
        active = score[:, :, lane_slice, :].clone()
        active_flat = active.reshape(ROWS, HLEN)
        m_curr = torch.max(active_flat.max(dim=-1).values, m_old[lane])
        exp_m_res[lane] = torch.exp(m_old[lane] - m_curr)
        shifted = torch.exp(active_flat - m_curr[:, None])
        l_new[lane] = l_old[lane] * exp_m_res[lane] + shifted.sum(dim=-1)
        score_out[:, :, lane_slice, :] = shifted.reshape(BATCH, ROWS, 1, HLEN)
    golden_flat = score_out.reshape(BATCH * ROWS, LANE_COUNT * HLEN)

    return (
        {
            "Score_hbm": score,
            "Score_out_hbm": torch.zeros_like(score),
        },
        golden_flat,
        {
            "m_old_init": m_old,
            "l_old_init": l_old,
            "golden_exp_m_res": exp_m_res,
            "golden_l_new": l_new,
        },
    )


def build_fp_preload(state: dict[str, torch.Tensor]) -> torch.Tensor:
    total = FPRAM_USER_BASE + 6 * FP_STATE_ELEMS
    fp = torch.zeros(total, dtype=torch.float16)
    fp[M_OLD_ADDR:M_OLD_ADDR + FP_STATE_ELEMS] = state["m_old_init"].reshape(-1).to(torch.float16)
    fp[L_OLD_ADDR:L_OLD_ADDR + FP_STATE_ELEMS] = state["l_old_init"].reshape(-1).to(torch.float16)
    return fp


def main() -> int:
    build_dir = _TESTBENCH_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Compiling TVM online-softmax-min kernel ...")
    hlir_path = build_dir / f"{ASM_NAME}.hlir.txt"
    isa_text = compile_isa_via_tvm(
        ASM_NAME,
        stage_output="Score_out_hbm",
        dump_hlir=hlir_path,
    )
    print(f"      OK  ({isa_text.count(chr(10))} ISA lines, HLIR -> {hlir_path.name})")

    print("[2/4] Generating inputs, FPSRAM preload, and golden ...")
    inputs, golden_flat, state = build_inputs_and_golden(seed=0)
    fp_preload = build_fp_preload(state)
    input_feed = {name: t.contiguous().reshape(1, -1) for name, t in inputs.items()}
    input_order = list(input_feed)
    print(
        f"      OK  score shape={tuple(inputs['Score_hbm'].shape)}  "
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
        "num_rows": BATCH * ROWS,
        "num_batches": BATCH * ROWS,
        "elements_per_batch": LANE_COUNT * HLEN,
        "row_dim": MLEN,
        "compare_fpsram": True,
        "fpsram_num_elements": FP_STATE_ELEMS,
        "fpsram_m_res_start": M_RES_ADDR,
        "fpsram_l_start": L_OLD_ADDR,
    }
    cmp_path = build_dir / "comparison_params.json"
    cmp_path.write_text(json.dumps(comparison_params, indent=2))

    torch.save(
        {
            "golden_exp_m_res": state["golden_exp_m_res"].reshape(-1).to(torch.float32),
            "golden_l_new": state["golden_l_new"].reshape(-1).to(torch.float32),
            "fpsram_m_res_start": M_RES_ADDR,
            "fpsram_l_start": L_OLD_ADDR,
        },
        build_dir / "golden_fpsram.pt",
    )

    (build_dir / f"{ARTIFACT_PREFIX}_generated_asm_code.asm").write_text(isa_text)

    print()
    print("=" * 60)
    print("build/ ready for emulator run")
    print(f"script: {_THIS_FILE.name}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
