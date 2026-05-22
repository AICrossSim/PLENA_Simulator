"""TVM SingleStreamBlock — STAGED diagnostic testbench.

Companion to ``tvm_single_stream_block_test.py`` (the full-block
reference). That driver always compiles the WHOLE chain and verifies
the final output; if the block is off, it cannot tell which kernel is
to blame.

This driver instead TRUNCATES the chain at ``CHAIN_UPTO``: it compiles
and concatenates ONLY the steps up to (and including) that step, then
stages THAT step's output buffer back to VRAM[0..] for view_mem to
compare. Because nothing after the truncation point runs, the staged
buffer is the clean, just-computed result — no later kernel can
overwrite the VRAM region or reuse the scratch.

Workflow: bump ``CHAIN_UPTO`` one step at a time, rebuild, run, record
the match rate. The first step whose rate collapses is the culprit.

The data flow, addresses, kernels, and golden are all inherited
verbatim from ``tvm_single_stream_block_test.py`` — this file only
adds the truncation logic, so the two stay in lock-step.

Run:
    just build-emulator-debug tvm_ssb_staged
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path bootstrap (mirrors tvm_single_stream_block_test.py).
# ---------------------------------------------------------------------------
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
sys.path.insert(0, str(_THIS_FILE.parent))  # so the reference test imports

import math  # noqa: E402
import torch  # noqa: E402,F401

import tilelang_tvm_compiler  # bootstrap TVM 0.23  # noqa: E402,F401
from tilelang_tvm_compiler.address_alloc import (  # noqa: E402
    AddressAllocConfig,
    _hbm_packed_byte_size,
)
from tilelang_tvm_compiler.hlir import format_hlir  # noqa: E402
from tilelang_tvm_compiler.pipeline import PlenaTarget  # noqa: E402
from tilelang_tvm_compiler.test_helper import resolve_output_layout  # noqa: E402

# Reuse everything from the full-block reference verbatim — addresses,
# kernels, golden all stay in lock-step with it.
from tvm_single_stream_block_test import (  # noqa: E402
    FPRAM_USER_BASE,
    HLEN,
    MLEN,
    ROWS,
    HEAD_COUNT,
    HIDDEN_SIZE,
    ATTN_ACTIVE_LANE,
    ATTN_NUM_KV_BLOCKS,
    ATTN_NUM_Q_BLOCKS,
    QKV_N_BLOCKS,
    MLP_N_BLOCKS,
    LINEAR2_K_BLOCKS,
    LINEAR2_N_BLOCKS,
    HbmTensor,
    _block_hbm_layout,
    _compute_address_plan,
    build_inputs_and_golden,
    compile_concat_step,
    compile_flash_attention_step,
    compile_gelu_step,
    compile_layernorm_step,
    compile_linear_step,
    compile_modulate_step,
    compile_residual_gate_step,
    compile_rmsnorm_step,
    compile_rope_step,
    merge_fp_preload,
)


# ===========================================================================
# >>> CHAIN_UPTO — the only knob. Bump this one step at a time. <<<
#
# Build + verify the chain truncated at this step. Valid values, in
# chain order:
#   layernorm, modulate,
#   linear_q, linear_k, linear_v, linear_mlp,
#   qknorm_q, qknorm_k,
#   rope_q, rope_k,
#   flash_attention, gelu, concat,
#   linear2, residual_gate
# ===========================================================================
# Env var SSB_UPTO overrides this (used by the sweep script
# run_ssb_sweep.sh); the literal below is the default for a plain run.
CHAIN_UPTO = os.environ.get("SSB_UPTO", "linear_k")


# ===========================================================================
# >>> THREE diagnostic modes — precedence + how to switch. <<<
#
# `just` cannot pass env vars, so each mode env var defaults to "1" and
# precedence (highest first) picks the mode. To switch modes for a
# `just build-emulator-debug tvm_ssb_staged` run, set the higher-priority
# env vars' DEFAULTS below to "0" (or pass them as "0" on the CLI).
#
#   Precedence (highest wins):
#     1. SSB_CHAIN_CLEAN_ATTN -> _main_chain_clean_attn()
#          Runs the FULL chain up to AND including flash_attention
#          (layernorm -> modulate -> linear_* -> qknorm_* -> rope_* ->
#          flash_attention), so the real chain environment is present
#          (real HBM layout, shared FPRAM, every upstream kernel runs
#          and writes its scratch). BUT flash_attention's Q/K/V are NOT
#          the chain's rope_q/rope_k/linear_v outputs — they are SEPARATE
#          clean `randn*0.5` input tensors. Isolates "does the full-chain
#          environment corrupt attention" from "attention just gets
#          degraded inputs": ~82% here means the chain env is clean.
#     2. SSB_CLEAN_ATTN -> _main_clean_attn()
#          Compiles ONLY the flash_attention kernel, standalone, fed
#          clean `randn*0.5` Q/K/V. Isolates the driver wiring from the
#          kernel itself.
#     3. (neither set) -> normal staged main()
#          Compiles the chain truncated at CHAIN_UPTO and verifies that
#          step against the chain's real golden.
#
# To revert all three to "off" defaults later, set the three literals
# below back to "".
# ===========================================================================
# Defaults are "" (off) — a plain run is the NORMAL staged chain.
# Attention was diagnosed (see SSB_DIAGNOSIS_REPORT.md) and the clean
# modes are no longer the default. To re-enable a diagnostic mode,
# either pass the env var ("SSB_CHAIN_CLEAN_ATTN=1 python3 ...") or
# temporarily flip the literal default below.
SSB_CHAIN_CLEAN_ATTN = os.environ.get(
    "SSB_CHAIN_CLEAN_ATTN", "").lower() in ("1", "true")
SSB_CLEAN_ATTN = os.environ.get("SSB_CLEAN_ATTN", "").lower() in ("1", "true")


# Chain order — the canonical sequence the steps execute in.
STEP_ORDER = [
    "layernorm",
    "modulate",
    "linear_q", "linear_k", "linear_v", "linear_mlp",
    "qknorm_q", "qknorm_k",
    "rope_q", "rope_k",
    "flash_attention", "gelu", "concat",
    "linear2", "residual_gate",
]


def _compile_all_steps(
    addr_plan: dict[str, int],
    attn_qkv_override: tuple[str, str, str] | None = None,
) -> dict:
    """Compile every chain step with pinned addresses, FPRAM-cursored in
    chain order. Returns an ordered {name: Step} dict.

    This mirrors ``tvm_single_stream_block_test.main()``'s compile block
    exactly — same kernels, same addresses, same FPRAM layout — so a
    truncated build is bit-identical to the full build's prefix.

    ``attn_qkv_override``: optional (q_key, k_key, v_key) addr_plan keys
    to pin flash_attention's Q/K/V to instead of the default
    ATTN_Q_hbm / ATTN_K_hbm / ATTN_V_hbm. Used by the
    SSB_CHAIN_CLEAN_ATTN mode to feed attention clean input tensors
    while the rest of the chain runs unchanged.
    """
    steps: dict = {}
    fpram_cursor = FPRAM_USER_BASE

    def _emit(chain_name, step):
        # ``chain_name`` is the STEP_ORDER key (e.g. "layernorm"); the
        # compiled Step's own .name may differ (layernorm_min /
        # modulate_min keep the kernel's default name). Key by
        # chain_name so CHAIN_UPTO lookups match.
        nonlocal fpram_cursor
        print(f"      {chain_name:<16s} hoisted_consts={step.fpram_const_count}, "
              f"FPRAM slots {fpram_cursor}..{fpram_cursor + step.fpram_const_count}")
        fpram_cursor += step.fpram_const_count
        steps[chain_name] = step

    _emit("layernorm", compile_layernorm_step(
        addr_plan=addr_plan, fpram_const_base=fpram_cursor))
    _emit("modulate", compile_modulate_step(
        addr_plan=addr_plan, fpram_const_base=fpram_cursor))

    # q / k / v / mlp projections.
    linear_specs = [
        ("linear_q",   QKV_N_BLOCKS, "LINQ_A_hbm", "LINQ_W_hbm", "LINQ_BIAS_hbm", "Q_hbm"),
        ("linear_k",   QKV_N_BLOCKS, "LINK_A_hbm", "LINK_W_hbm", "LINK_BIAS_hbm", "K_hbm"),
        ("linear_v",   QKV_N_BLOCKS, "LINV_A_hbm", "LINV_W_hbm", "LINV_BIAS_hbm", "V_hbm"),
        ("linear_mlp", MLP_N_BLOCKS, "LINM_A_hbm", "LINM_W_hbm", "LINM_BIAS_hbm", "MLP_hbm"),
    ]
    for name, n_blocks, a_key, w_key, bias_key, y_key in linear_specs:
        _emit(name, compile_linear_step(
            name=name, n_blocks=n_blocks,
            a_addr=addr_plan[a_key], w_addr=addr_plan[w_key],
            bias_addr=addr_plan[bias_key], y_addr=addr_plan[y_key],
            fpram_const_base=fpram_cursor))

    # QKNorm q / k.
    qknorm_specs = [
        ("qknorm_q", "QKN_Q_X_hbm", "QKN_Q_SCALE_hbm", "QKN_Q_Y_hbm"),
        ("qknorm_k", "QKN_K_X_hbm", "QKN_K_SCALE_hbm", "QKN_K_Y_hbm"),
    ]
    for name, x_key, scale_key, y_key in qknorm_specs:
        _emit(name, compile_rmsnorm_step(
            name=name,
            x_addr=addr_plan[x_key], scale_addr=addr_plan[scale_key],
            y_addr=addr_plan[y_key], fpram_const_base=fpram_cursor))

    # RoPE q / k (shared rotary frequency).
    for name, x_key, y_key in [
        ("rope_q", "ROPE_Q_X_hbm", "ROPE_Q_Y_hbm"),
        ("rope_k", "ROPE_K_X_hbm", "ROPE_K_Y_hbm"),
    ]:
        _emit(name, compile_rope_step(
            name=name,
            xq_addr=addr_plan[x_key],
            cos_addr=addr_plan["ROPE_COS_hbm"],
            sin_addr=addr_plan["ROPE_SIN_hbm"],
            neg_sin_addr=addr_plan["ROPE_NEG_SIN_hbm"],
            y_addr=addr_plan[y_key], fpram_const_base=fpram_cursor))

    # flash_attention — writes a compact [S, HIDDEN_SIZE] output.
    if attn_qkv_override is not None:
        _attn_q_key, _attn_k_key, _attn_v_key = attn_qkv_override
    else:
        _attn_q_key, _attn_k_key, _attn_v_key = (
            "ATTN_Q_hbm", "ATTN_K_hbm", "ATTN_V_hbm")
    _emit("flash_attention", compile_flash_attention_step(
        name="flash_attention",
        q_addr=addr_plan[_attn_q_key], k_addr=addr_plan[_attn_k_key],
        v_addr=addr_plan[_attn_v_key], o_addr=addr_plan["ATTN_O_hbm"],
        fpram_const_base=fpram_cursor))

    # gelu — writes a compact [S, MLP_HIDDEN_DIM] output.
    _emit("gelu", compile_gelu_step(
        name="gelu",
        x_addr=addr_plan["GELU_X_hbm"], y_addr=addr_plan["GELU_OUT_hbm"],
        fpram_const_base=fpram_cursor))

    # concat — join attention's + gelu's compact outputs into CONCAT_hbm.
    _emit("concat", compile_concat_step(
        name="concat",
        a_addr=addr_plan["CONCAT_A_hbm"], b_addr=addr_plan["CONCAT_B_hbm"],
        y_addr=addr_plan["CONCAT_hbm"],
        fpram_const_base=fpram_cursor))

    # linear2 — project concat([attn, gelu]) back to [M, H*D].
    _emit("linear2", compile_linear_step(
        name="linear2", n_blocks=LINEAR2_N_BLOCKS,
        a_addr=addr_plan["LIN2_A_hbm"], w_addr=addr_plan["LIN2_W_hbm"],
        bias_addr=addr_plan["LIN2_BIAS_hbm"], y_addr=addr_plan["LIN2_OUT_hbm"],
        fpram_const_base=fpram_cursor, k_blocks=LINEAR2_K_BLOCKS))

    # residual_gate — x + gate * linear2_out.
    _emit("residual_gate", compile_residual_gate_step(
        name="residual_gate",
        x_addr=addr_plan["GATE_X_hbm"], gate_addr=addr_plan["GATE_G_hbm"],
        y_addr=addr_plan["GATE_Y_hbm"], out_addr=addr_plan["BLOCK_OUT_hbm"],
        fpram_const_base=fpram_cursor))

    return steps


def _dump_step_hlir(steps, build_dir: Path) -> None:
    """Feature 1: dump each kept step's HLIR to build/<name>.hlir.txt.

    Lets us diff the chain's per-kernel HLIR against a known-good
    standalone HLIR.
    """
    for step in steps:
        out_path = build_dir / f"{step.name}.hlir.txt"
        out_path.write_text(format_hlir(step.compiled.hlir))
        print(f"      HLIR dump: {out_path}")


def _main_clean_attn() -> int:
    """SSB_CLEAN_ATTN mode: run the standalone attention test through the
    staged driver's compile / address / build path.

    Only the flash_attention kernel is compiled. Q/K/V are CLEAN random
    inputs (torch.randn * 0.5) — NOT the chain's rope/linear_v outputs.
    The golden is per-head scaled-dot-product softmax attention computed
    from those same clean Q/K/V.
    """
    SEQ_LEN = ATTN_NUM_Q_BLOCKS * ROWS
    KV_SEQ = ATTN_NUM_KV_BLOCKS * ROWS

    print(f"[ssb_staged] SSB_CLEAN_ATTN -> compiling ONLY flash_attention "
          f"(clean Q/K/V, standalone golden)")
    print()

    # ----- 1. standalone HBM layout: Q/K/V inputs + O scratch -----
    # All four are (1, seq, head_count, hlen) compact BSHD tensors.
    q_elems = BATCH_CLEAN * SEQ_LEN * HEAD_COUNT * HLEN
    kv_elems = BATCH_CLEAN * KV_SEQ * HEAD_COUNT * HLEN
    layout = [
        HbmTensor(name="Q_hbm", num_elements=q_elems, role="input"),
        HbmTensor(name="K_hbm", num_elements=kv_elems, role="input"),
        HbmTensor(name="V_hbm", num_elements=kv_elems, role="input"),
        HbmTensor(name="O_hbm", num_elements=q_elems, role="scratch"),
    ]
    addr_cfg_proto = AddressAllocConfig(mlen=MLEN, blen=4, hlen=HLEN)
    addr_plan = _compute_address_plan(layout, addr_cfg_proto)
    print(f"[1/5] HBM address plan: {len(layout)} tensors "
          f"(Q={addr_plan['Q_hbm']}, K={addr_plan['K_hbm']}, "
          f"V={addr_plan['V_hbm']}, O={addr_plan['O_hbm']})")

    # ----- 2. compile ONLY the flash_attention kernel -----
    print("[2/5] Compiling flash_attention ...")
    step = compile_flash_attention_step(
        name="flash_attention",
        q_addr=addr_plan["Q_hbm"], k_addr=addr_plan["K_hbm"],
        v_addr=addr_plan["V_hbm"], o_addr=addr_plan["O_hbm"],
        fpram_const_base=FPRAM_USER_BASE,
    )
    steps = [step]

    # ----- 3. clean inputs + per-head softmax golden -----
    # Mirrors tvm_flash_attention_min_test.build_inputs_and_golden.
    torch.manual_seed(0)
    q = torch.randn(BATCH_CLEAN, SEQ_LEN, HEAD_COUNT, HLEN,
                    dtype=torch.float32) * 0.5
    k = torch.randn(BATCH_CLEAN, KV_SEQ, HEAD_COUNT, HLEN,
                    dtype=torch.float32) * 0.5
    v = torch.randn(BATCH_CLEAN, KV_SEQ, HEAD_COUNT, HLEN,
                    dtype=torch.float32) * 0.5

    scale = 1.0 / math.sqrt(HLEN)
    score = torch.einsum("bihd,bjhd->bihj", q, k)   # (B, SEQ, H, KV_SEQ)
    out = torch.empty(BATCH_CLEAN, SEQ_LEN, HEAD_COUNT, HLEN,
                      dtype=torch.float32)
    for h in range(HEAD_COUNT):
        score_h = score[:, :, h, :]
        v_h = v[:, :, h, :]
        p = torch.softmax(score_h * scale, dim=-1)
        out[:, :, h, :] = torch.einsum("bij,bjd->bid", p, v_h)
    golden_flat = out.reshape(BATCH_CLEAN * SEQ_LEN, HEAD_COUNT * HLEN)

    hbm_inputs = {"Q_hbm": q, "K_hbm": k, "V_hbm": v}
    print(f"[3/5] Clean Q/K/V + per-head softmax golden "
          f"{tuple(golden_flat.shape)}")

    # ----- concatenate ASM + stage O_hbm to VRAM -----
    from tilelang_tvm_compiler.__main__ import _emit_output_staging
    isa_text = (
        f"\n; ============================================================\n"
        f"; >>> STEP: {step.name}  (SSB_CLEAN_ATTN)\n"
        f"; ============================================================\n"
    )
    isa_text += step.compiled.isa_text
    staging_isa = _emit_output_staging(
        step.compiled,
        PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        step.output_buffer_name,
    )
    isa_text = isa_text.rstrip() + staging_isa
    print(f"      total ISA: {isa_text.count(chr(10))} lines")

    # ----- 4. merge FP preload -----
    print("[4/5] Merging FP preload + inputs ...")
    fp_preload = merge_fp_preload(steps)
    print(f"      fp_preload: shape={tuple(fp_preload.shape)}, "
          f"nonzero={int((fp_preload != 0).sum())}")

    # ----- 5. write build artifacts -----
    print("[5/5] Writing build artifacts ...")
    from compiler.sim_env_utils import create_mem_for_sim
    from transactional_emulator.tools.create_sim_env import create_sim_env

    build_dir = _THIS_FILE.parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_feed = {
        name: t.contiguous().reshape(1, -1) for name, t in hbm_inputs.items()
    }
    create_sim_env(
        input_tensor=input_feed,
        generated_code=isa_text,
        golden_result={"original_output": golden_flat},
        fp_preload=fp_preload,
        int_preload=None,
        build_dir=str(build_dir),
    )
    create_mem_for_sim(
        data_size=256, mode="behave_sim", asm="tvm_ssb_staged",
        data=None, specified_data_order=list(input_feed),
        build_path=build_dir,
    )

    # comparison_params — VRAM compare; output is [SEQ_LEN, HIDDEN_SIZE].
    # Mirrors the standalone tvm_flash_attention_min_test.
    # Geometry from the canonical OutputLayout so num_rows /
    # use_stride_mode agree with golden_flat by construction.
    _layout = resolve_output_layout(
        num_batches=BATCH_CLEAN * SEQ_LEN,
        elements_per_batch=HIDDEN_SIZE,
        mlen=MLEN,
    )
    comparison_params = {
        "check_hbm": False,
        "start_row_idx": 0,
        "compare_fpsram": False,
        **_layout.comparison_params(),
    }
    (build_dir / "comparison_params.json").write_text(
        json.dumps(comparison_params, indent=2)
    )
    (build_dir / "tvm_ssb_staged_generated_asm_code.asm").write_text(isa_text)

    # ----- Feature 1: dump the attention HLIR -----
    _dump_step_hlir(steps, build_dir)

    print()
    print("=" * 60)
    print(f"SSB_CLEAN_ATTN: staged {step.name}.{step.output_buffer_name} "
          f"-> VRAM[0..], shape ({BATCH_CLEAN * SEQ_LEN}, {HIDDEN_SIZE})")
    print("build/ ready. Next:")
    print("  just build-emulator-debug tvm_ssb_staged")
    print("=" * 60)
    return 0


# BATCH for clean attention mode (matches the standalone testbench).
BATCH_CLEAN = 1


def _main_chain_clean_attn() -> int:
    """SSB_CHAIN_CLEAN_ATTN mode: run the FULL chain up to and including
    flash_attention, but feed attention SEPARATE clean Q/K/V.

    Unlike _main_clean_attn (which compiles ONLY attention), this builds
    the whole prefix layernorm -> modulate -> linear_* -> qknorm_* ->
    rope_* -> flash_attention. Every upstream kernel runs and writes its
    scratch, so the real chain environment (HBM layout, shared FPRAM,
    serial kernel execution) is present. But flash_attention's Q/K/V are
    pinned to three EXTRA clean input tensors (CLEAN_Q/K/V_hbm,
    randn*0.5), NOT the chain's rope_q/rope_k/linear_v outputs.

    ~82% here (== standalone) => the chain environment is clean and the
    chain's degraded score is purely input degradation. Notably less =>
    some upstream kernel's scratch/FPRAM pollutes attention.
    """
    SEQ_LEN = ATTN_NUM_Q_BLOCKS * ROWS

    print(f"[ssb_staged] SSB_CHAIN_CLEAN_ATTN -> full chain to "
          f"flash_attention, attention fed CLEAN Q/K/V")
    print()

    # ----- 1. extended HBM layout: full chain + 3 clean input tensors -----
    addr_cfg_proto = AddressAllocConfig(mlen=MLEN, blen=4, hlen=HLEN)
    base_layout = _block_hbm_layout()
    clean_elems = BATCH_CLEAN * SEQ_LEN * HEAD_COUNT * HLEN
    layout = list(base_layout) + [
        HbmTensor(name="CLEAN_Q_hbm", num_elements=clean_elems, role="input"),
        HbmTensor(name="CLEAN_K_hbm", num_elements=clean_elems, role="input"),
        HbmTensor(name="CLEAN_V_hbm", num_elements=clean_elems, role="input"),
    ]
    addr_plan = _compute_address_plan(layout, addr_cfg_proto)
    print(f"[1/5] HBM address plan: {len(layout)} tensors "
          f"({len(base_layout)} chain + 3 clean) "
          f"(CLEAN_Q={addr_plan['CLEAN_Q_hbm']}, "
          f"CLEAN_K={addr_plan['CLEAN_K_hbm']}, "
          f"CLEAN_V={addr_plan['CLEAN_V_hbm']})")

    # ----- 2. compile the chain up to + including flash_attention.
    # flash_attention's Q/K/V are pinned to the CLEAN_ tensors. -----
    print("[2/5] Compiling chain (layernorm .. flash_attention) ...")
    all_steps = _compile_all_steps(
        addr_plan,
        attn_qkv_override=("CLEAN_Q_hbm", "CLEAN_K_hbm", "CLEAN_V_hbm"),
    )
    kept_names = STEP_ORDER[: STEP_ORDER.index("flash_attention") + 1]
    steps = [all_steps[n] for n in kept_names]
    last_step = steps[-1]
    print(f"      kept {len(kept_names)} step(s): {' -> '.join(kept_names)}")

    # ----- 3. clean Q/K/V + per-head softmax golden -----
    # Mirrors tvm_flash_attention_min_test.build_inputs_and_golden /
    # _main_clean_attn — self-attention so kv_seq == q_seq == SEQ_LEN.
    torch.manual_seed(0)
    q = torch.randn(BATCH_CLEAN, SEQ_LEN, HEAD_COUNT, HLEN,
                    dtype=torch.float32) * 0.5
    k = torch.randn(BATCH_CLEAN, SEQ_LEN, HEAD_COUNT, HLEN,
                    dtype=torch.float32) * 0.5
    v = torch.randn(BATCH_CLEAN, SEQ_LEN, HEAD_COUNT, HLEN,
                    dtype=torch.float32) * 0.5

    scale = 1.0 / math.sqrt(HLEN)
    score = torch.einsum("bihd,bjhd->bihj", q, k)   # (B, SEQ, H, KV_SEQ)
    out = torch.empty(BATCH_CLEAN, SEQ_LEN, HEAD_COUNT, HLEN,
                      dtype=torch.float32)
    for h in range(HEAD_COUNT):
        p = torch.softmax(score[:, :, h, :] * scale, dim=-1)
        out[:, :, h, :] = torch.einsum("bij,bjd->bid", p, v[:, :, h, :])
    golden_flat = out.reshape(BATCH_CLEAN * SEQ_LEN, HEAD_COUNT * HLEN)

    # All the chain's real inputs — staged so upstream kernels run on
    # real data — PLUS the three clean attention input tensors.
    io = build_inputs_and_golden(base_layout, seed=0)
    hbm_inputs = dict(io["hbm_inputs"])
    hbm_inputs["CLEAN_Q_hbm"] = q
    hbm_inputs["CLEAN_K_hbm"] = k
    hbm_inputs["CLEAN_V_hbm"] = v
    print(f"[3/5] Chain inputs ({len(io['hbm_inputs'])}) + clean Q/K/V; "
          f"per-head softmax golden {tuple(golden_flat.shape)}")

    # ----- concatenate ASM for all kept steps + stage attention's O -----
    from tilelang_tvm_compiler.__main__ import _emit_output_staging
    isa_text = ""
    for step in steps:
        isa_text += (
            f"\n; ============================================================\n"
            f"; >>> STEP: {step.name}  (SSB_CHAIN_CLEAN_ATTN)\n"
            f"; ============================================================\n"
        )
        isa_text += step.compiled.isa_text
    staging_isa = _emit_output_staging(
        last_step.compiled,
        PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        last_step.output_buffer_name,
    )
    isa_text = isa_text.rstrip() + staging_isa
    print(f"      total ISA: {isa_text.count(chr(10))} lines")

    # ----- 4. merge FP preload (all kept steps' constants) -----
    print("[4/5] Merging FP preload + inputs ...")
    fp_preload = merge_fp_preload(steps)
    print(f"      fp_preload: shape={tuple(fp_preload.shape)}, "
          f"nonzero={int((fp_preload != 0).sum())}")

    # ----- 5. write build artifacts -----
    print("[5/5] Writing build artifacts ...")
    from compiler.sim_env_utils import create_mem_for_sim
    from transactional_emulator.tools.create_sim_env import create_sim_env

    build_dir = _THIS_FILE.parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_feed = {
        name: t.contiguous().reshape(1, -1) for name, t in hbm_inputs.items()
    }
    create_sim_env(
        input_tensor=input_feed,
        generated_code=isa_text,
        golden_result={"original_output": golden_flat},
        fp_preload=fp_preload,
        int_preload=None,
        build_dir=str(build_dir),
    )
    create_mem_for_sim(
        data_size=256, mode="behave_sim", asm="tvm_ssb_staged",
        data=None, specified_data_order=list(input_feed),
        build_path=build_dir,
    )

    # comparison_params — VRAM compare; output is [SEQ_LEN, HIDDEN_SIZE].
    # Mirrors the clean-attn mode.
    # Geometry from the canonical OutputLayout so num_rows /
    # use_stride_mode agree with golden_flat by construction.
    _layout = resolve_output_layout(
        num_batches=BATCH_CLEAN * SEQ_LEN,
        elements_per_batch=HIDDEN_SIZE,
        mlen=MLEN,
    )
    comparison_params = {
        "check_hbm": False,
        "start_row_idx": 0,
        "compare_fpsram": False,
        **_layout.comparison_params(),
    }
    (build_dir / "comparison_params.json").write_text(
        json.dumps(comparison_params, indent=2)
    )
    (build_dir / "tvm_ssb_staged_generated_asm_code.asm").write_text(isa_text)

    # ----- dump each kept step's HLIR -----
    _dump_step_hlir(steps, build_dir)

    print()
    print("=" * 60)
    print(f"SSB_CHAIN_CLEAN_ATTN: staged {last_step.name}."
          f"{last_step.output_buffer_name} -> VRAM[0..], "
          f"shape ({BATCH_CLEAN * SEQ_LEN}, {HIDDEN_SIZE})")
    print("build/ ready. Next:")
    print("  just build-emulator-debug tvm_ssb_staged")
    print("=" * 60)
    return 0


def main() -> int:
    if SSB_CHAIN_CLEAN_ATTN:
        return _main_chain_clean_attn()
    if SSB_CLEAN_ATTN:
        return _main_clean_attn()
    if CHAIN_UPTO not in STEP_ORDER:
        raise ValueError(
            f"CHAIN_UPTO={CHAIN_UPTO!r} not a valid step. "
            f"Valid: {STEP_ORDER}"
        )
    trunc_idx = STEP_ORDER.index(CHAIN_UPTO)
    kept_names = STEP_ORDER[: trunc_idx + 1]

    print(f"[ssb_staged] CHAIN_UPTO = {CHAIN_UPTO!r}  "
          f"-> compiling {len(kept_names)} step(s): {' -> '.join(kept_names)}")
    print()

    # ----- 1. plan HBM addresses (full layout — addresses are stable
    # whether or not a tensor's producer is in the truncated chain) -----
    addr_cfg_proto = AddressAllocConfig(mlen=MLEN, blen=4, hlen=HLEN)
    layout = _block_hbm_layout()
    addr_plan = _compute_address_plan(layout, addr_cfg_proto)
    print(f"[1/5] HBM address plan: {len(layout)} tensors")

    # ----- 2. compile EVERY step, then keep only the truncated prefix.
    # Compiling all of them keeps the FPRAM cursor identical to the full
    # build, so a truncated chain is bit-identical to the full build's
    # prefix. -----
    print("[2/5] Compiling chain ...")
    all_steps = _compile_all_steps(addr_plan)
    steps = [all_steps[n] for n in kept_names]
    last_step = steps[-1]

    # ----- 3. build golden + pick the truncated step's output buffer.
    io = build_inputs_and_golden(layout, seed=0)
    if CHAIN_UPTO not in io["intermediates"]:
        raise ValueError(
            f"no golden intermediate for {CHAIN_UPTO!r}; "
            f"have: {sorted(io['intermediates'])}"
        )
    golden_flat, stage_buf = io["intermediates"][CHAIN_UPTO]
    io["golden_flat"] = golden_flat

    print(f"[3/5] Truncated chain ends at {last_step.name}; "
          f"staging {last_step.name}.{stage_buf}  golden {tuple(golden_flat.shape)}")

    # ----- concatenate ASM for the kept steps + stage the last one -----
    from tilelang_tvm_compiler.__main__ import _emit_output_staging
    isa_text = ""
    for step in steps:
        isa_text += (
            f"\n; ============================================================\n"
            f"; >>> STEP: {step.name}\n"
            f"; ============================================================\n"
        )
        isa_text += step.compiled.isa_text
    staging_isa = _emit_output_staging(
        last_step.compiled,
        PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        stage_buf,
    )
    isa_text = isa_text.rstrip() + staging_isa
    print(f"      total ISA: {isa_text.count(chr(10))} lines")

    # ----- 4. merge FP preload (only the kept steps' constants) -----
    print("[4/5] Merging FP preload + inputs ...")
    fp_preload = merge_fp_preload(steps)
    print(f"      fp_preload: shape={tuple(fp_preload.shape)}, "
          f"nonzero={int((fp_preload != 0).sum())}")

    # ----- 5. write build artifacts -----
    print("[5/5] Writing build artifacts ...")
    from compiler.sim_env_utils import create_mem_for_sim
    from transactional_emulator.tools.create_sim_env import create_sim_env

    build_dir = _THIS_FILE.parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_feed = {
        name: t.contiguous().reshape(1, -1) for name, t in io["hbm_inputs"].items()
    }
    create_sim_env(
        input_tensor=input_feed,
        generated_code=isa_text,
        golden_result={"original_output": io["golden_flat"]},
        fp_preload=fp_preload,
        int_preload=None,
        build_dir=str(build_dir),
    )
    create_mem_for_sim(
        data_size=256, mode="behave_sim", asm="tvm_ssb_staged",
        data=None, specified_data_order=list(input_feed),
        build_path=build_dir,
    )

    # comparison_params — VRAM compare; shape follows the staged buffer.
    # Geometry from the canonical OutputLayout so num_rows /
    # use_stride_mode agree with golden_flat by construction.
    final_rows, final_cols = io["golden_flat"].shape
    _layout = resolve_output_layout(
        num_batches=final_rows,
        elements_per_batch=final_cols,
        mlen=MLEN,
    )
    comparison_params = {
        "check_hbm": False,
        "start_row_idx": 0,
        "compare_fpsram": False,
        **_layout.comparison_params(),
    }
    (build_dir / "comparison_params.json").write_text(
        json.dumps(comparison_params, indent=2)
    )
    (build_dir / "tvm_ssb_staged_generated_asm_code.asm").write_text(isa_text)

    # ----- Feature 1: dump each kept step's HLIR -----
    _dump_step_hlir(steps, build_dir)

    print()
    print("=" * 60)
    print(f"CHAIN_UPTO={CHAIN_UPTO!r}: staged {last_step.name}.{stage_buf} "
          f"-> VRAM[0..], shape ({final_rows}, {final_cols})")
    print("build/ ready. Next:")
    print("  just build-emulator-debug tvm_ssb_staged")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
