"""SSB stepwise driver — run ONE kernel of the SingleStreamBlock per
emulator invocation, double-compared against TWO goldens.

Motivation
----------
Running the whole chained block through the emulator once is expensive.
This driver runs each kernel of the block as its OWN emulator run, so a
single simulation only covers one kernel. Between steps the HBM image is
carried forward: the previous kernel's real (quantized, error-carrying)
HBM output becomes the next kernel's input — exactly what the hardware
sees.

Two goldens per step (same HBM output, compared twice):

  * LOCAL golden — recompute THIS kernel on the host from the PREVIOUS
    kernel's REAL HBM output (read back from hbm_dump.bin, MX-E4M3
    dequantized). Isolates whether THIS kernel is itself correct,
    independent of upstream error.

  * GLOBAL golden — the ideal end-to-end PyTorch chain computed from the
    original block inputs up to this step. Measures the ACCUMULATED
    correctness of the block up to here.

If a step's LOCAL cosine drops below ``LOCAL_COSINE_STOP`` (= 0.85) the
driver flags it — that step's own math is wrong, not just upstream
drift.

Orchestration split
-------------------
This module is the per-step Python worker. It does NOT run the emulator
(that needs cargo + LD_LIBRARY_PATH and is the slow part). The driver
bash ``run_ssb_stepwise.sh`` loops over steps:

    for each step:
        python tvm_ssb_stepwise_test.py prepare <step>   # compile + build/
        just/cargo run the emulator                      # writes hbm_dump.bin
        python tvm_ssb_stepwise_test.py compare <step>   # double-compare
        (snapshot hbm_dump.bin -> .ssb_steps/<step>.hbm.bin)

The per-step HBM snapshots live under ``testbench/.ssb_steps/`` (NOT
build/, which ``just`` wipes every run).

Usage (direct):
    python tvm_ssb_stepwise_test.py prepare layernorm
    python tvm_ssb_stepwise_test.py compare layernorm
"""

from __future__ import annotations

import json
import shutil
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
sys.path.insert(0, str(_THIS_FILE.parent))

import torch  # noqa: E402

# Reuse the chained block's address plan + per-kernel compile_*_step +
# the ideal end-to-end golden (intermediates) — the single source of
# truth for geometry and global golden.
import tvm_single_stream_block_test as ssb  # noqa: E402
from _tb_cache import mx_roundtrip  # noqa: E402
from tilelang_tvm_compiler.pipeline import PlenaTarget  # noqa: E402

# MX dequant for reading a previous kernel's real HBM output back to host.
from transactional_emulator.tools.check_mem import (  # noqa: E402
    read_hbm_bin_file_as_array,
    compare_hbm_with_golden,
)

BUILD_DIR = _THIS_FILE.parent / "build"
STEP_DIR = _THIS_FILE.parent / ".ssb_steps"   # survives `rm -rf build`
HBM_DUMP = _REPO_ROOT / "transactional_emulator" / "hbm_dump.bin"

LOCAL_COSINE_STOP = 0.85

MLEN = ssb.MLEN
HLEN = ssb.HLEN
HEAD_COUNT = ssb.HEAD_COUNT
HIDDEN_SIZE = ssb.HIDDEN_SIZE
SEQ_LEN = ssb.SEQ_LEN
BATCH = ssb.BATCH
EPS = ssb.EPS


# ---------------------------------------------------------------------------
# Step table — name, compile fn closure, output buffer name, output HBM
# logical name (for the address plan), and per-step geometry needed to
# read the output back as a tensor.
#
# Built lazily inside _plan() because the compile closures need addr_plan.
# ---------------------------------------------------------------------------
def _plan():
    """Compute the shared HBM address plan + the ordered step list, exactly
    as the chained driver does. Returns (addr_plan, steps, io)."""
    addr_cfg_proto = ssb.AddressAllocConfig(mlen=MLEN, blen=4, hlen=HLEN)
    layout = ssb._block_hbm_layout()
    addr_plan = ssb._compute_address_plan(layout, addr_cfg_proto)
    io = ssb.build_inputs_and_golden(layout, seed=0)
    return addr_plan, layout, io


def _read_hbm_output(start_byte: int, num_elements: int) -> torch.Tensor:
    """Read a kernel's real HBM output back as a flat fp32 tensor, MX-E4M3
    dequantized — the SAME read the comparator uses.

    ``scale_offset`` mirrors the testbench convention: 1 byte/elem so the
    scale region starts num_elements bytes after the element region.
    """
    arr = read_hbm_bin_file_as_array(
        str(HBM_DUMP),
        exp_width=4, man_width=3,
        start_byte_offset=start_byte,
        num_elements=num_elements,
        element_bytes=1,
        scale_width=8, block_size=8,
        scale_offset=num_elements,
    )
    return torch.tensor(arr, dtype=torch.float32)


# ===========================================================================
# Per-step single-kernel host functions (LOCAL golden).
#
# Each takes the PREVIOUS kernel's real HBM output (already MX-dequantized
# fp32) plus the step's other HBM inputs, and recomputes ONLY this kernel
# in fp32, MX-round-tripping the inputs (matching what the sim consumes)
# and the output. Math mirrors build_inputs_and_golden's per-step block
# but with the upstream value sourced from REAL hardware output.
# ===========================================================================
def _local_layernorm(x_flat: torch.Tensor, io: dict) -> torch.Tensor:
    # layernorm is the FIRST step; "previous HBM output" is the block's
    # original X input, which equals io's staged X_hbm. We still read it
    # back through the same MX path for consistency.
    x = x_flat.reshape(BATCH, SEQ_LEN, 1, HIDDEN_SIZE)
    scale = io["hbm_inputs"]["LN_SCALE_hbm"]
    bias = io["hbm_inputs"]["LN_BIAS_hbm"]
    x_eff = mx_roundtrip(x)
    s_eff = mx_roundtrip(scale)
    b_eff = mx_roundtrip(bias)
    mu = x_eff.mean(dim=-1, keepdim=True)
    xc = x_eff - mu
    var = (xc * xc).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + EPS)
    y = xc * inv * s_eff + b_eff
    return mx_roundtrip(y).reshape(SEQ_LEN, HIDDEN_SIZE)


def _local_modulate(prev_flat: torch.Tensor, io: dict) -> torch.Tensor:
    # prev = layernorm output (real HBM), as BSHD.
    x = prev_flat.reshape(BATCH, SEQ_LEN, HEAD_COUNT, HLEN)
    s1p = io["hbm_inputs"]["MOD_SCALE1P_hbm"]
    shift = io["hbm_inputs"]["MOD_SHIFT_hbm"]
    x_eff = mx_roundtrip(x)
    s_eff = mx_roundtrip(s1p)
    sh_eff = mx_roundtrip(shift)
    y = s_eff * x_eff + sh_eff
    return mx_roundtrip(y).reshape(SEQ_LEN, HIDDEN_SIZE)


def _local_linear_q(prev_flat: torch.Tensor, io: dict) -> torch.Tensor:
    # prev = modulate output (real HBM), reshaped (M, K).
    a = prev_flat.reshape(SEQ_LEN, HIDDEN_SIZE)
    w = io["hbm_inputs"]["LINQ_W_hbm"].reshape(HIDDEN_SIZE, HIDDEN_SIZE)
    b = io["hbm_inputs"]["LINQ_BIAS_hbm"].reshape(SEQ_LEN, HIDDEN_SIZE)
    a_eff = mx_roundtrip(a)
    w_eff = mx_roundtrip(w)
    b_eff = mx_roundtrip(b)
    y = a_eff @ w_eff.T + b_eff
    return mx_roundtrip(y).reshape(SEQ_LEN, HIDDEN_SIZE)


# Step registry: name -> dict(local_fn, out_buf, out_hbm_name, prev_step,
# cols). Only the first 3 steps wired for the framework-verification pass;
# the rest follow the same shape and get added once these check out.
STEP_TABLE = {
    "layernorm": dict(
        local_fn=_local_layernorm, out_buf="Y_hbm", out_hbm="LN_Y_hbm",
        prev_step=None, prev_hbm="X_hbm", cols=HIDDEN_SIZE,
    ),
    "modulate": dict(
        local_fn=_local_modulate, out_buf="Y_hbm", out_hbm="MOD_Y_hbm",
        prev_step="layernorm", prev_hbm="LN_Y_hbm", cols=HEAD_COUNT * HLEN,
    ),
    "linear_q": dict(
        local_fn=_local_linear_q, out_buf="C_hbm", out_hbm="Q_hbm",
        prev_step="modulate", prev_hbm="MOD_Y_hbm", cols=HIDDEN_SIZE,
    ),
}

# Ordered list driving the bash loop.
STEP_ORDER = ["layernorm", "modulate", "linear_q"]


def _compile_step(name: str, addr_plan: dict):
    """Compile just this one kernel with its pinned HBM addresses, using
    the chained driver's compile_*_step factories (FPRAM const base 0 —
    each step runs standalone so there is no inter-kernel const overlap)."""
    fb = ssb.FPRAM_USER_BASE
    if name == "layernorm":
        return ssb.compile_layernorm_step(addr_plan=addr_plan, fpram_const_base=fb)
    if name == "modulate":
        return ssb.compile_modulate_step(addr_plan=addr_plan, fpram_const_base=fb)
    if name == "linear_q":
        return ssb.compile_linear_step(
            name="linear_q", n_blocks=ssb.QKV_N_BLOCKS,
            a_addr=addr_plan["LINQ_A_hbm"], w_addr=addr_plan["LINQ_W_hbm"],
            bias_addr=addr_plan["LINQ_BIAS_hbm"], y_addr=addr_plan["Q_hbm"],
            fpram_const_base=fb,
        )
    raise ValueError(f"unknown step {name!r}")


# ===========================================================================
# prepare <step>: compile this kernel, build the per-step sim env (ASM +
# fp_sram + the input HBM image carried forward from the previous step).
# ===========================================================================
def _assemble_only() -> None:
    """Assemble build/generated_asm_code.asm into generated_machine_code.mem
    WITHOUT repacking HBM.

    ``env_setup`` with an empty ``MemoryDataManager`` runs only the
    assembler step (the per-tensor HBM packing loop has nothing to do)
    and does NOT call init_mem, so the input HBM image on disk is left
    untouched — exactly the trick flash_attention's run_cached uses.
    """
    from compiler.sim_env_utils.build_sys_tools import env_setup
    from compiler.sim_env_utils.build_env import MemoryDataManager
    from utils.load_config import load_toml_config

    config = load_toml_config(str(_REPO_ROOT / "plena_settings.toml"), "CONFIG")
    precision = load_toml_config(str(_REPO_ROOT / "plena_settings.toml"), "PRECISION")
    data_config = {
        "tensor_size": [1, 1],
        "block_size": [1, precision["HBM_M_WEIGHT_TYPE"]["block"]],
    }
    quant_config = {
        "exp_width": precision["HBM_V_ACT_TYPE"]["ELEM"]["exponent"],
        "man_width": precision["HBM_V_ACT_TYPE"]["ELEM"]["mantissa"],
        "exp_bias_width": precision["HBM_V_ACT_TYPE"]["SCALE"]["exponent"],
        "int_width": precision["HBM_V_INT_TYPE"]["DATA_TYPE"]["width"],
    }
    env_setup(
        MemoryDataManager(), BUILD_DIR, data_config, quant_config,
        hbm_row_width=config["HBM_WIDTH"]["value"],
    )


def prepare(name: str) -> int:
    from compiler.sim_env_utils import create_mem_for_sim
    from transactional_emulator.tools.create_sim_env import create_sim_env

    spec = STEP_TABLE[name]
    addr_plan, layout, io = _plan()
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[prepare {name}] compiling single kernel ...")
    step = _compile_step(name, addr_plan)
    isa_text = step.compiled.isa_text

    # Append the HBM->VRAM->HBM staging the chained driver uses so the
    # output lands at the pinned HBM address (the comparator reads it
    # straight off hbm_dump.bin).
    from tilelang_tvm_compiler.__main__ import _emit_output_staging
    staging = _emit_output_staging(
        step.compiled, PlenaTarget(mlen=MLEN, btmm_hlen=HLEN), spec["out_buf"],
    )
    isa_text = isa_text.rstrip() + staging

    # Normalise large ADDI immediates over the full assembled text — the
    # staging stub is emitted outside isa_pass, so its MLEN-512 addresses
    # would otherwise overflow the 18-bit immediate slot (same fix run()
    # applies). create_mem_for_sim re-reads this from build/, so normalise
    # before it is written.
    from tilelang_tvm_compiler.isa_pass import _normalize_large_addi_immediates
    isa_text = _normalize_large_addi_immediates(isa_text)

    # Per-step fp_preload — just THIS kernel's hoisted consts.
    fp_preload = ssb.merge_fp_preload([step])
    g_cols = spec["cols"]
    golden_placeholder = torch.zeros(SEQ_LEN, g_cols, dtype=torch.float32)
    prev = spec["prev_step"]

    if prev is None:
        # ---- FIRST STEP: build the full input HBM image once. ----
        # io["hbm_inputs"] is the WHOLE chain's inputs (every weight /
        # scale / bias the downstream kernels will need), so the single
        # packed HBM seeds them all in place. This MX pack is the slow
        # part — paid exactly once for the whole stepwise run.
        input_feed = {
            n: t.contiguous().reshape(1, -1) for n, t in io["hbm_inputs"].items()
        }
        create_sim_env(
            input_tensor=input_feed,
            generated_code=isa_text,
            golden_result={"original_output": golden_placeholder},
            fp_preload=fp_preload,
            int_preload=None,
            build_dir=str(BUILD_DIR),
        )
        create_mem_for_sim(
            data_size=256, mode="behave_sim", asm=step.name,
            data=None, specified_data_order=list(input_feed),
            build_path=BUILD_DIR,
        )
        print(f"[prepare {name}] first step — packed full input HBM "
              f"({len(input_feed)} tensors)")
    else:
        # ---- LATER STEPS: skip the .pt dump + HBM repack entirely. ----
        # The input HBM IS the previous step's hbm_dump snapshot (carried
        # forward), so there is nothing to pack. We only need to (a) write
        # the ASM + this kernel's fp_sram, (b) assemble ASM -> .mem, and
        # (c) copy the previous snapshot in as the input image.
        snap = STEP_DIR / f"{prev}.hbm.bin"
        if not snap.exists():
            raise RuntimeError(
                f"[prepare {name}] missing previous snapshot {snap}; "
                f"run step {prev!r} first"
            )
        # (a) ASM + fp_sram only — input_tensor={} writes no .pt files.
        create_sim_env(
            input_tensor={},
            generated_code=isa_text,
            golden_result={"original_output": golden_placeholder},
            fp_preload=fp_preload,
            int_preload=None,
            build_dir=str(BUILD_DIR),
        )
        # (b) assemble the freshly written ASM into machine code WITHOUT
        # repacking HBM (env_setup with an empty memory manager runs only
        # the assembler step — same trick flash_attention's run_cached
        # uses). Then (c) seed the input image from the prev snapshot.
        _assemble_only()
        shutil.copy2(snap, BUILD_DIR / "hbm_for_behave_sim.bin")
        print(f"[prepare {name}] reused HBM from {prev} snapshot "
              f"(no .pt dump, no MX repack)")

    # Stash the plan info compare() needs (addresses computed here).
    meta = {
        "name": name,
        "out_start_byte": int(addr_plan[spec["out_hbm"]]),
        "cols": g_cols,
        "num_elements": SEQ_LEN * g_cols,
    }
    (BUILD_DIR / "ssb_step_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[prepare {name}] OK — out @ byte {meta['out_start_byte']}, "
          f"{meta['num_elements']} elems")
    return 0


# ===========================================================================
# compare <step>: after the emulator wrote hbm_dump.bin, run the TWO
# compares (local + global) on the same output bytes, snapshot the HBM
# image forward, and flag if local cosine < threshold.
# ===========================================================================
def _write_golden_pt(g_flat: torch.Tensor, subdir: Path) -> Path:
    """Write a golden as ``golden_output.pt`` inside ``subdir``.

    ``check_mem.parse_golden_output`` prefers the lossless
    ``golden_output.pt`` sitting next to whatever golden path it is given
    (it ignores the path's filename and looks for ``golden_output.pt`` in
    the same dir). So local / global goldens MUST live in SEPARATE dirs,
    each with its own golden_output.pt, or both compares read the same
    tensor. Returns the dummy golden_result.txt path to hand the
    comparator (its dirname is what matters).
    """
    subdir.mkdir(parents=True, exist_ok=True)
    torch.save(g_flat.reshape(-1).float(), subdir / "golden_output.pt")
    txt = subdir / "golden_result.txt"
    txt.write_text("")  # presence-only; .pt takes precedence
    return txt


def compare(name: str) -> int:
    spec = STEP_TABLE[name]
    meta = json.loads((BUILD_DIR / "ssb_step_meta.json").read_text())
    start_byte = meta["out_start_byte"]
    num_elems = meta["num_elements"]
    cols = meta["cols"]

    addr_plan, layout, io = _plan()

    # ---- LOCAL golden: recompute this kernel from previous REAL output ----
    prev = spec["prev_step"]
    if prev is None:
        # First step input == block X; read it back via the same MX path
        # from the freshly built input image (or just use io's X).
        prev_out = io["hbm_inputs"]["X_hbm"].reshape(-1)
    else:
        prev_spec = STEP_TABLE[prev]
        prev_start = int(addr_plan[prev_spec["out_hbm"]])
        prev_elems = SEQ_LEN * prev_spec["cols"]
        prev_out = _read_hbm_output(prev_start, prev_elems)
    local_golden = spec["local_fn"](prev_out, io)

    # ---- GLOBAL golden: the ideal end-to-end value at this step ----
    global_golden, _buf = io["intermediates"][name]

    # Each golden in its OWN dir (parse_golden_output keys off the dir,
    # not the filename — see _write_golden_pt).
    local_txt = _write_golden_pt(local_golden, BUILD_DIR / "golden_local")
    global_txt = _write_golden_pt(global_golden, BUILD_DIR / "golden_global")

    def _cmp(golden_path: Path) -> dict:
        return compare_hbm_with_golden(
            str(HBM_DUMP), str(golden_path),
            exp_width=4, man_width=3, element_bytes=1,
            start_byte_offset=start_byte, num_elements=num_elems,
            num_batches=SEQ_LEN, elements_per_batch=cols,
            scale_width=8, block_size=8, scale_offset=num_elems,
        )

    res_local = _cmp(local_txt)
    res_global = _cmp(global_txt)

    lc = res_local["global_cosine"]
    gc = res_global["global_cosine"]
    print("=" * 64)
    print(f"STEP {name}")
    print(f"  LOCAL  cosine={lc:.6f}  NRMSE={res_local['nrmse']*100:.3f}%  "
          f"(this kernel alone)")
    print(f"  GLOBAL cosine={gc:.6f}  NRMSE={res_global['nrmse']*100:.3f}%  "
          f"(accumulated to here)")
    print("=" * 64)

    # Snapshot the full HBM image forward for the next step.
    STEP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HBM_DUMP, STEP_DIR / f"{name}.hbm.bin")

    # Append to the running report.
    report = STEP_DIR / "stepwise_report.txt"
    with open(report, "a") as f:
        f.write(f"{name}\tlocal={lc:.6f}\tglobal={gc:.6f}\t"
                f"local_nrmse={res_local['nrmse']:.6f}\t"
                f"global_nrmse={res_global['nrmse']:.6f}\n")

    if lc < LOCAL_COSINE_STOP:
        print(f"!! LOCAL cosine {lc:.4f} < {LOCAL_COSINE_STOP} — this "
              f"kernel's own math is WRONG. Stopping.")
        return 2
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[1] not in ("prepare", "compare"):
        print("usage: tvm_ssb_stepwise_test.py {prepare|compare} <step>")
        print("steps:", ", ".join(STEP_ORDER))
        return 1
    mode, step = argv[1], argv[2]
    if step not in STEP_TABLE:
        print(f"unknown step {step!r}; known: {', '.join(STEP_ORDER)}")
        return 1
    return prepare(step) if mode == "prepare" else compare(step)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
