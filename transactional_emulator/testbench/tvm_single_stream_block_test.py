"""TVM single-stream-block testbench — chained _min kernels in one cargo run.

Stitches several ``kernels/*_min.py`` factories together into ONE
continuous ASM stream that runs in a single ``cargo run`` of the
transactional emulator. The kernels share HBM (which is naturally
global) and share a single FPRAM preload.

The driver pre-plans a GLOBAL HBM layout (which fp16 tensor lives at
which byte address), then compiles each kernel with per-buffer address
pins (``AddressAllocConfig.hbm_address_overrides``) so every kernel's
HBM references land on the planned bytes. Producer.output_addr ==
consumer.input_addr is just two pins on the same address. Layout
shape changes (BSHD <-> B,S,1,H*D) are pure relabels — fp16 row-major
bytes are identical, so the next kernel's signature just declares a
different shape over the same byte range.

FPRAM convention:
  - kernels reuse the FPRAM scratch area freely — kernel-local fragments
    are short-lived and inter-kernel ordering means no overlap.
  - hoisted ``T.float16(c)`` constants get unique slots across kernels:
    driver runs a fpram_const_cursor starting at FPRAM_USER_BASE and
    pins each kernel's hoisted-constant buffers via
    ``fpram_address_overrides``. The merged fp_sram preload writes
    every kernel's constants at their pinned address.

Mirrors the Open-Sora SingleStreamBlock (see tilelang_kernels/
single_stream_block.py — the GPU reference this chain reproduces on
PLENA):

  layernorm -> modulate -> linear_{q,k,v,mlp} -> qknorm_{q,k}
  -> rope_{q,k} -> flash_attention --+
                      gelu(mlp) -----+--> concat
                                     -> linear2 -> residual_gate

attention and GELU(mlp) each write their OWN compact output tensor; a
dedicated ``concat_min`` kernel joins the two along the feature axis;
ONE linear2 then projects the [H*D + mlp_dim] concat back to [H*D]; a
single gated residual ``x + gate * linear2_out`` closes the block.

Run via:
    .venv/bin/python transactional_emulator/testbench/tvm_single_stream_block_test.py
then:
    cd transactional_emulator && cargo run --release -- ...
or, after wiring justfile:
    just build-emulator-debug tvm_single_stream_block
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# sys.path bootstrap (mirrors tvm_layernorm_min_test.py).
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

import torch  # noqa: E402
from tilelang_tvm_compiler.plena_settings import load_sizes as _load_sizes  # noqa: E402

import tilelang_tvm_compiler  # bootstrap TVM 0.23  # noqa: E402,F401
from tilelang_tvm_compiler.address_alloc import (  # noqa: E402
    AddressAllocConfig,
    FPRAM_USER_BASE,
    _hbm_packed_byte_size,
)
from tilelang_tvm_compiler.kernels.concat_min import make_concat_min  # noqa: E402
from tilelang_tvm_compiler.kernels.flash_attention_min import make_flash_attention_min  # noqa: E402
from tilelang_tvm_compiler.kernels.gelu_min import make_gelu_min  # noqa: E402
from tilelang_tvm_compiler.kernels.layernorm_min import make_layernorm_min  # noqa: E402
from tilelang_tvm_compiler.kernels.linear_min import make_linear_min  # noqa: E402
from tilelang_tvm_compiler.kernels.modulate_min import make_modulate_min  # noqa: E402
from tilelang_tvm_compiler.kernels.residual_gate_min import make_residual_gate_min  # noqa: E402
from tilelang_tvm_compiler.kernels.rmsnorm_min import make_rmsnorm_min  # noqa: E402
from tilelang_tvm_compiler.kernels.rope_min import make_rope_min  # noqa: E402
from tilelang_tvm_compiler.pipeline import (  # noqa: E402
    CompiledKernel,
    PlenaTarget,
    compile_kernel,
)
from tilelang_tvm_compiler.test_helper import (  # noqa: E402
    resolve_output_layout,
)


# ---------------------------------------------------------------------------
# Block config (Stage A) — small enough to land in <1MB HBM.
# ---------------------------------------------------------------------------
BATCH = 1
_HW = _load_sizes()  # hardware geometry — single source of truth, plena_settings.toml

MLEN = _HW.mlen  # from plena_settings.toml
ROWS = MLEN  # rows per tile == mlen
HLEN = _HW.hlen  # from plena_settings.toml
HEAD_COUNT = 8
HIDDEN_SIZE = HEAD_COUNT * HLEN          # = 128
NUM_S_BLOCKS = 2
SEQ_LEN = NUM_S_BLOCKS * ROWS            # = 128
EPS = 1e-6

# ---- linear1 (QKV + MLP-in projection) ----
# Conceptually one fused linear x_mod [M,K] @ W1 [3*HD+mlp, K].T, but
# split into FOUR independent GEMMs (q / k / v / mlp_in). Each writes a
# COMPACT [M, out_dim] HBM region, so the head split (H*D -> H, D) that
# QKNorm / RoPE / attention need is a pure compiler reshape with no
# strided DMA. (A single fused output would leave q/k/v interleaved
# column-wise inside one [M, N1] row — strided, painful for the
# head-packed DMA path.)
#
# x_mod: [M=SEQ_LEN, K=HIDDEN_SIZE]
#   q     = x_mod @ Wq[HIDDEN_SIZE, K].T + bq   -> Q_hbm  [M, HIDDEN_SIZE]
#   k     = x_mod @ Wk[HIDDEN_SIZE, K].T + bk   -> K_hbm  [M, HIDDEN_SIZE]
#   v     = x_mod @ Wv[HIDDEN_SIZE, K].T + bv   -> V_hbm  [M, HIDDEN_SIZE]
#   mlp   = x_mod @ Wm[MLP_HIDDEN_DIM, K].T + bm-> MLP_hbm[M, MLP_HIDDEN_DIM]
MLP_HIDDEN_DIM = 128
LINEAR_M_BLOCKS = SEQ_LEN // MLEN              # = 2

# FPRAM split: hoisted constants live in a low, PERSISTENT segment
# [FPRAM_USER_BASE, FPRAM_SCRATCH_BASE); kernel-local scratch fragments
# (SS / SS_N / ... — lane-expanded, 256 slots each) live ABOVE that and
# are freely reused across kernels (inter-kernel ordering means no live
# overlap). The two segments MUST NOT overlap: a kernel's scratch must
# not sit on the next kernel's pinned const slot, or running this kernel
# silently corrupts the next kernel's preloaded constant. 64 slots is
# ample headroom for every kernel's hoisted consts combined (the chain
# currently uses 13: layernorm 2 + qknorm_q 2 + qknorm_k 2 +
# flash_attention 2 + gelu 5).
FPRAM_CONST_HEADROOM = 64
FPRAM_SCRATCH_BASE = FPRAM_USER_BASE + FPRAM_CONST_HEADROOM
LINEAR_K_BLOCKS = HIDDEN_SIZE // MLEN          # = 2
QKV_N_BLOCKS = HIDDEN_SIZE // MLEN             # = 2  (q/k/v each)
MLP_N_BLOCKS = MLP_HIDDEN_DIM // MLEN          # = 2

# ---- flash_attention (self-attention over the projected q/k/v) ----
# q comes from rope_q (rotated), k from qknorm_k (normed), v straight
# from linear_v. All three are [SEQ_LEN, HIDDEN_SIZE] compact, which is
# exactly [1, SEQ_LEN, HEAD_COUNT, HLEN] BSHD bytes. Self-attention so
# kv_seq == q_seq == SEQ_LEN.
ATTN_ACTIVE_LANE = 2
ATTN_NUM_KV_BLOCKS = SEQ_LEN // ROWS           # = 2
ATTN_NUM_Q_BLOCKS = SEQ_LEN // ROWS            # = 2

# ---- FFN back half: gelu(mlp_in) ----
# gelu activates the mlp-in projection [S, MLP_HIDDEN_DIM] elementwise
# (MLP_HIDDEN_DIM == MLP_HEAD_COUNT*HLEN so it tiles as BSHD).
GELU_NUM_S_BLOCKS = NUM_S_BLOCKS               # = 2
MLP_HEAD_COUNT = MLP_HIDDEN_DIM // HLEN        # = 8  (mlp tiled as heads)

# ---- concat([attn_out, mlp_out]) -> linear2 ----
# Mirrors Open-Sora SingleStreamBlock: attention and GELU(mlp) each
# write their OWN compact output tensor; a dedicated concat_min kernel
# joins them along the feature axis, then ONE linear (linear2) projects
# the [H*D + mlp_dim] concat back to [H*D]. No o_head_offset writeback.
CONCAT_DIM = HIDDEN_SIZE + MLP_HIDDEN_DIM           # = 256 = H*D + mlp_dim
CONCAT_NUM_S_BLOCKS = NUM_S_BLOCKS                  # = 2

# linear2: combined [M, CONCAT_DIM] @ W2[H*D, CONCAT_DIM].T + b2
# -> [M, H*D].
LINEAR2_K_BLOCKS = CONCAT_DIM // MLEN               # = 4
LINEAR2_N_BLOCKS = HIDDEN_SIZE // MLEN              # = 2


# ---------------------------------------------------------------------------
# Global HBM layout — every fp16 tensor that lives in HBM across the
# whole block, in the order create_sim_env will stage it into
# hbm_for_behave_sim.bin. Aliasing is encoded by giving two logical
# tensors the same offset (we DROP the alias entry from the staged
# input_feed so it doesn't double-book bytes).
#
# Each entry: (logical_name, num_elements, role)
#   role == "input"      -> seeded with real data by build_inputs_and_golden
#   role == "scratch"    -> seeded with zeros (kernels overwrite)
#   role == "alias_of:X" -> NO staging slot; X's bytes are reused
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HbmTensor:
    name: str
    num_elements: int
    role: str

    @property
    def is_alias(self) -> bool:
        return self.role.startswith("alias_of:")

    @property
    def alias_target(self) -> str:
        if not self.is_alias:
            raise ValueError(f"{self.name} is not an alias")
        return self.role.split(":", 1)[1]


def _block_hbm_layout() -> list[HbmTensor]:
    """HBM tensors across the chained block, in staging order.

    LN -> modulate -> {linear_q, linear_k, linear_v, linear_mlp}.
    """
    bs_packed = BATCH * SEQ_LEN * 1 * HIDDEN_SIZE       # B,S,1,H*D for LN
    bs_bshd = BATCH * SEQ_LEN * HEAD_COUNT * HLEN       # B,S,H,D for modulate
    assert bs_packed == bs_bshd, (
        "B,S,1,H*D and B,S,H,D must have the same element count so they "
        "alias byte-for-byte"
    )
    # Each linear: A (M,K), W (out,K), bias (M,out), Y (M,out) — all
    # laid flat as (1, dim0, 1, dim1).
    qkv_w = HIDDEN_SIZE * HIDDEN_SIZE         # Wq/Wk/Wv (HIDDEN_SIZE, K)
    qkv_b = SEQ_LEN * HIDDEN_SIZE             # bias broadcast (M, HIDDEN_SIZE)
    qkv_y = SEQ_LEN * HIDDEN_SIZE             # q/k/v out (M, HIDDEN_SIZE)
    mlp_w = MLP_HIDDEN_DIM * HIDDEN_SIZE
    mlp_b = SEQ_LEN * MLP_HIDDEN_DIM
    mlp_y = SEQ_LEN * MLP_HIDDEN_DIM
    # concat([attn_out, mlp_out]) along the feature axis — the linear2
    # input. CONCAT_DIM = H*D + mlp_dim. attention and gelu each write
    # their own compact tensor; the concat_min kernel joins them here.
    bs_concat = BATCH * SEQ_LEN * 1 * CONCAT_DIM
    # linear2: W (H*D, CONCAT_DIM), bias (M, H*D), Y (M, H*D).
    lin2_w = HIDDEN_SIZE * CONCAT_DIM
    return [
        HbmTensor("X_hbm",         bs_packed, "input"),    # LN input
        HbmTensor("LN_SCALE_hbm",  bs_packed, "input"),    # LN scale (full-broadcast)
        HbmTensor("LN_BIAS_hbm",   bs_packed, "input"),    # LN bias  (full-broadcast)
        HbmTensor("LN_Y_hbm",      bs_packed, "scratch"),  # LN out  ==  modulate.X (alias)
        HbmTensor("MOD_X_hbm",     bs_packed, "alias_of:LN_Y_hbm"),
        HbmTensor("MOD_SCALE1P_hbm", bs_bshd, "input"),    # (1+scale)
        HbmTensor("MOD_SHIFT_hbm",   bs_bshd, "input"),
        HbmTensor("MOD_Y_hbm",       bs_bshd, "scratch"),  # modulate out (== all 4 linears' A)
        # ---- linear_q / linear_k / linear_v / linear_mlp ----
        # All four read the SAME x_mod (MOD_Y_hbm) as their A input.
        HbmTensor("LINQ_A_hbm",   bs_bshd, "alias_of:MOD_Y_hbm"),
        HbmTensor("LINQ_W_hbm",   qkv_w,   "input"),
        HbmTensor("LINQ_BIAS_hbm", qkv_b,  "input"),
        HbmTensor("Q_hbm",        qkv_y,   "scratch"),     # q proj out (M, HIDDEN_SIZE)
        HbmTensor("LINK_A_hbm",   bs_bshd, "alias_of:MOD_Y_hbm"),
        HbmTensor("LINK_W_hbm",   qkv_w,   "input"),
        HbmTensor("LINK_BIAS_hbm", qkv_b,  "input"),
        HbmTensor("K_hbm",        qkv_y,   "scratch"),     # k proj out
        HbmTensor("LINV_A_hbm",   bs_bshd, "alias_of:MOD_Y_hbm"),
        HbmTensor("LINV_W_hbm",   qkv_w,   "input"),
        HbmTensor("LINV_BIAS_hbm", qkv_b,  "input"),
        HbmTensor("V_hbm",        qkv_y,   "scratch"),     # v proj out
        HbmTensor("LINM_A_hbm",   bs_bshd, "alias_of:MOD_Y_hbm"),
        HbmTensor("LINM_W_hbm",   mlp_w,   "input"),
        HbmTensor("LINM_BIAS_hbm", mlp_b,  "input"),
        HbmTensor("MLP_hbm",      mlp_y,   "scratch"),     # mlp-in proj out
        # ---- QKNorm: RMSNorm(q) and RMSNorm(k) per head_dim ----
        # rmsnorm_min reads X_hbm as [batch, seq, head_count, hlen]; q's
        # compact [M, H*D] bytes ARE [1, S, H, D] (row stride H*D both
        # ways), so QKN_Q_X aliases Q_hbm with no copy.
        HbmTensor("QKN_Q_X_hbm",   bs_bshd, "alias_of:Q_hbm"),
        HbmTensor("QKN_Q_SCALE_hbm", bs_bshd, "input"),    # q_norm scale broadcast (S,H,D)
        HbmTensor("QKN_Q_Y_hbm",   bs_bshd, "scratch"),    # normed q
        HbmTensor("QKN_K_X_hbm",   bs_bshd, "alias_of:K_hbm"),
        HbmTensor("QKN_K_SCALE_hbm", bs_bshd, "input"),    # k_norm scale broadcast
        HbmTensor("QKN_K_Y_hbm",   bs_bshd, "scratch"),    # normed k
        # ---- RoPE on the normed q AND k ----
        # rope_min reads XQ_hbm as [batch, seq, head_count, hlen]; the
        # normed q/k bytes already ARE that BSHD shape, so ROPE_*_X
        # alias the QKNorm outputs with no copy. q and k share the same
        # rotary frequency (COS/SIN/NEG_SIN), matching the reference
        # tilelang_apply_rope(q, k, pe).
        HbmTensor("ROPE_Q_X_hbm",   bs_bshd, "alias_of:QKN_Q_Y_hbm"),
        HbmTensor("ROPE_COS_hbm",     bs_bshd, "input"),
        HbmTensor("ROPE_SIN_hbm",     bs_bshd, "input"),
        HbmTensor("ROPE_NEG_SIN_hbm", bs_bshd, "input"),
        HbmTensor("ROPE_Q_Y_hbm",     bs_bshd, "scratch"),  # rotated q
        HbmTensor("ROPE_K_X_hbm",   bs_bshd, "alias_of:QKN_K_Y_hbm"),
        HbmTensor("ROPE_K_Y_hbm",     bs_bshd, "scratch"),  # rotated k
        # ---- flash_attention: self-attn over rope_q / rope_k / v ----
        # Q from rope_q, K from rope_k (both rotated), V straight from
        # linear_v. Attention writes a COMPACT [S, HIDDEN_SIZE] output
        # tensor (no o_head_offset) — independently verifiable.
        HbmTensor("ATTN_Q_hbm",   bs_bshd, "alias_of:ROPE_Q_Y_hbm"),
        HbmTensor("ATTN_K_hbm",   bs_bshd, "alias_of:ROPE_K_Y_hbm"),
        HbmTensor("ATTN_V_hbm",   bs_bshd, "alias_of:V_hbm"),
        HbmTensor("ATTN_O_hbm",   bs_bshd, "scratch"),     # attention compact out
        # ---- GELU on the mlp branch ----
        # gelu activates the mlp-in projection; X aliases MLP_hbm. Its
        # output is a COMPACT [S, MLP_HIDDEN_DIM] tensor (no
        # o_head_offset) — independently verifiable.
        HbmTensor("GELU_X_hbm",   bs_bshd, "alias_of:MLP_hbm"),
        HbmTensor("GELU_OUT_hbm", bs_bshd, "scratch"),     # gelu compact out
        # ---- concat: join attn_out + gelu_out along the feature axis ----
        # concat_min copies the two compact tensors into the wide CONCAT
        # tensor: CONCAT_hbm[..., 0:HIDDEN_SIZE] = attn_out,
        # CONCAT_hbm[..., HIDDEN_SIZE:] = gelu_out.
        HbmTensor("CONCAT_A_hbm", bs_bshd,   "alias_of:ATTN_O_hbm"),
        HbmTensor("CONCAT_B_hbm", bs_bshd,   "alias_of:GELU_OUT_hbm"),
        HbmTensor("CONCAT_hbm",   bs_concat, "scratch"),   # wide concat out
        # ---- linear2: project the concat back to [M, H*D] ----
        # combined [M, CONCAT_DIM] @ W2[H*D, CONCAT_DIM].T + b2. A
        # aliases the CONCAT tensor.
        HbmTensor("LIN2_A_hbm",   bs_concat, "alias_of:CONCAT_hbm"),
        HbmTensor("LIN2_W_hbm",   lin2_w,  "input"),         # W2 (H*D, CONCAT_DIM)
        HbmTensor("LIN2_BIAS_hbm", qkv_b,  "input"),         # b2 broadcast (M, H*D)
        HbmTensor("LIN2_OUT_hbm", qkv_y,   "scratch"),       # linear2 output
        # ---- residual_gate: out = x_residual + gate * linear2_out ----
        # x_residual is the block's original LN input (X_hbm); gate is
        # the Modulation gate. Y aliases linear2's output. This single
        # gated residual matches the reference's final step.
        HbmTensor("GATE_X_hbm",    bs_bshd, "alias_of:X_hbm"),
        HbmTensor("GATE_G_hbm",    bs_bshd, "input"),        # gate
        HbmTensor("GATE_Y_hbm",    bs_bshd, "alias_of:LIN2_OUT_hbm"),
        HbmTensor("BLOCK_OUT_hbm", bs_bshd, "scratch"),      # FINAL block output
    ]


def _compute_address_plan(
    layout: list[HbmTensor], cfg: AddressAllocConfig,
) -> dict[str, int]:
    """Walk layout in order, assign each non-alias tensor a packed byte
    offset, and resolve aliases to their target's address. Returns
    {logical_name -> byte_address}.
    """
    plan: dict[str, int] = {}
    cur = 0
    for t in layout:
        if t.is_alias:
            target_addr = plan.get(t.alias_target)
            if target_addr is None:
                raise ValueError(
                    f"alias {t.name} -> {t.alias_target}: target must "
                    f"appear earlier in the layout list"
                )
            plan[t.name] = target_addr
            continue
        plan[t.name] = cur
        cur += _hbm_packed_byte_size(t.num_elements, cfg)
    return plan


# ---------------------------------------------------------------------------
# Step descriptor.
# ---------------------------------------------------------------------------
@dataclass
class Step:
    name: str
    compiled: CompiledKernel
    fpram_const_count: int
    output_buffer_name: str
    """Kernel-local HBM buffer name of this step's result. Used by the
    final stage_output block (_emit_output_staging) — differs per kernel
    (modulate: 'Y_hbm', linear: 'C_hbm')."""


# ---------------------------------------------------------------------------
# Hoisted-constant accounting — runs the hoist pre-pass in isolation to
# discover constant buffer names BEFORE the real compile, so the driver
# can pin them at globally-unique FPRAM addresses.
# ---------------------------------------------------------------------------
def _hoisted_const_names(prim_func) -> list[str]:
    from tilelang_tvm_compiler.frontend.passes import inline_let_stmts
    from tilelang_tvm_compiler.frontend.passes import lower_compound_fp_stores
    from tilelang_tvm_compiler.frontend.passes import hoist_float_constants

    fn = inline_let_stmts.run(prim_func)
    fn = lower_compound_fp_stores.run(fn)
    fn = hoist_float_constants.run(fn)
    consts = fn.attrs.get("plena.hoisted_constants") if fn.attrs else None
    if not consts:
        return []
    return [str(name) for name in consts.keys()]


# ---------------------------------------------------------------------------
# Per-step compile. Each takes the global address plan + FPRAM const
# cursor and returns the compiled kernel plus how many const slots it
# consumed.
# ---------------------------------------------------------------------------
def compile_layernorm_step(
    *, addr_plan: dict[str, int], fpram_const_base: int,
) -> Step:
    prim_func, _ = make_layernorm_min(
        rows=ROWS, hidden_size=HIDDEN_SIZE,
        num_s_blocks=NUM_S_BLOCKS, batch=BATCH, eps=EPS,
    )
    const_names = _hoisted_const_names(prim_func)
    fpram_overrides = {
        name: fpram_const_base + i for i, name in enumerate(const_names)
    }
    # Scratch lives in the fixed high segment, above ALL kernels'
    # pinned consts — NOT at const_base+const_count (that would put this
    # kernel's 256-slot scratch fragments on top of the next kernel's
    # pinned const slot, silently corrupting it).
    fpram_scratch_base = FPRAM_SCRATCH_BASE

    # Kernel-param-name -> global-layout-name mapping.
    hbm_overrides = {
        "X_hbm":     addr_plan["X_hbm"],
        "SCALE_hbm": addr_plan["LN_SCALE_hbm"],
        "BIAS_hbm":  addr_plan["LN_BIAS_hbm"],
        "Y_hbm":     addr_plan["LN_Y_hbm"],
    }

    addr_cfg = AddressAllocConfig(
        mlen=MLEN, blen=4, hlen=HLEN,
        hbm_address_overrides=hbm_overrides,
        fpram_address_overrides=fpram_overrides,
        fpram_base=fpram_scratch_base,
    )
    compiled = compile_kernel(
        prim_func, target=PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        name="layernorm_min", addr_config_override=addr_cfg,
    )
    return Step(name="layernorm_min", compiled=compiled,
                fpram_const_count=len(const_names),
                output_buffer_name="Y_hbm")


def compile_modulate_step(
    *, addr_plan: dict[str, int], fpram_const_base: int,
) -> Step:
    prim_func, _ = make_modulate_min(
        rows=ROWS, hlen=HLEN, head_count=HEAD_COUNT,
        num_s_blocks=NUM_S_BLOCKS, batch=BATCH,
    )
    const_names = _hoisted_const_names(prim_func)
    fpram_overrides = {
        name: fpram_const_base + i for i, name in enumerate(const_names)
    }
    # Scratch lives in the fixed high segment, above ALL kernels'
    # pinned consts — NOT at const_base+const_count (that would put this
    # kernel's 256-slot scratch fragments on top of the next kernel's
    # pinned const slot, silently corrupting it).
    fpram_scratch_base = FPRAM_SCRATCH_BASE

    hbm_overrides = {
        "X_hbm":       addr_plan["MOD_X_hbm"],   # aliases LN_Y_hbm
        "SCALE1P_hbm": addr_plan["MOD_SCALE1P_hbm"],
        "SHIFT_hbm":   addr_plan["MOD_SHIFT_hbm"],
        "Y_hbm":       addr_plan["MOD_Y_hbm"],
    }

    addr_cfg = AddressAllocConfig(
        mlen=MLEN, blen=4, hlen=HLEN,
        hbm_address_overrides=hbm_overrides,
        fpram_address_overrides=fpram_overrides,
        fpram_base=fpram_scratch_base,
    )
    compiled = compile_kernel(
        prim_func, target=PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        name="modulate_min", addr_config_override=addr_cfg,
    )
    return Step(name="modulate_min", compiled=compiled,
                fpram_const_count=len(const_names),
                output_buffer_name="Y_hbm")


def compile_linear_step(
    *, name: str, n_blocks: int,
    a_addr: int, w_addr: int, bias_addr: int, y_addr: int,
    fpram_const_base: int,
    k_blocks: int = LINEAR_K_BLOCKS,
) -> Step:
    """One GEMM step — generic over q / k / v / mlp / o / mlp_out
    projections.

    make_linear_min(with_bias=True) declares HBM buffers
    ``A_hbm, B_hbm, BIAS_hbm, C_hbm``:
      A_hbm    (1, M, 1, K)    <- x_mod (modulate output, M x K)
      B_hbm    (1, out, 1, K)  <- weight, nn.Linear (out, K) convention
      BIAS_hbm (1, M, 1, out)  <- bias broadcast to (M, out)
      C_hbm    (1, M, 1, out)  <- projection output, COMPACT [M, out]

    ``k_blocks`` defaults to LINEAR_K_BLOCKS (K = HIDDEN_SIZE, the
    q/k/v/mlp projections); linear2 passes its own K = CONCAT_DIM.
    """
    prim_func, _ = make_linear_min(
        m_blocks=LINEAR_M_BLOCKS,
        n_blocks=n_blocks,
        k_blocks=k_blocks,
        with_bias=True,
    )
    const_names = _hoisted_const_names(prim_func)
    fpram_overrides = {
        cn: fpram_const_base + i for i, cn in enumerate(const_names)
    }
    # Scratch lives in the fixed high segment, above ALL kernels'
    # pinned consts — NOT at const_base+const_count (that would put this
    # kernel's 256-slot scratch fragments on top of the next kernel's
    # pinned const slot, silently corrupting it).
    fpram_scratch_base = FPRAM_SCRATCH_BASE

    hbm_overrides = {
        "A_hbm":    a_addr,
        "B_hbm":    w_addr,
        "BIAS_hbm": bias_addr,
        "C_hbm":    y_addr,
    }

    addr_cfg = AddressAllocConfig(
        mlen=MLEN, blen=4, hlen=HLEN,
        hbm_address_overrides=hbm_overrides,
        fpram_address_overrides=fpram_overrides,
        fpram_base=fpram_scratch_base,
    )
    compiled = compile_kernel(
        prim_func, target=PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        name=name, addr_config_override=addr_cfg,
    )
    return Step(name=name, compiled=compiled,
                fpram_const_count=len(const_names),
                output_buffer_name="C_hbm")


def compile_rmsnorm_step(
    *, name: str, x_addr: int, scale_addr: int, y_addr: int,
    fpram_const_base: int,
) -> Step:
    """One RMSNorm step — generic over QKNorm-q / QKNorm-k.

    make_rmsnorm_min declares HBM buffers ``X_hbm, SCALE_hbm, Y_hbm``,
    all shaped (batch, seq_len, head_count, hlen). q's compact [M, H*D]
    bytes ARE that BSHD shape (row stride H*D both ways), so X_hbm just
    aliases the projection output with no copy.
    """
    prim_func, _ = make_rmsnorm_min(
        rows=ROWS, hlen=HLEN, head_count=HEAD_COUNT,
        num_s_blocks=NUM_S_BLOCKS, batch=BATCH, eps=EPS,
    )
    const_names = _hoisted_const_names(prim_func)
    fpram_overrides = {
        cn: fpram_const_base + i for i, cn in enumerate(const_names)
    }
    # Scratch lives in the fixed high segment, above ALL kernels'
    # pinned consts — NOT at const_base+const_count (that would put this
    # kernel's 256-slot scratch fragments on top of the next kernel's
    # pinned const slot, silently corrupting it).
    fpram_scratch_base = FPRAM_SCRATCH_BASE

    hbm_overrides = {
        "X_hbm":     x_addr,
        "SCALE_hbm": scale_addr,
        "Y_hbm":     y_addr,
    }

    addr_cfg = AddressAllocConfig(
        mlen=MLEN, blen=4, hlen=HLEN,
        hbm_address_overrides=hbm_overrides,
        fpram_address_overrides=fpram_overrides,
        fpram_base=fpram_scratch_base,
    )
    compiled = compile_kernel(
        prim_func, target=PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        name=name, addr_config_override=addr_cfg,
    )
    return Step(name=name, compiled=compiled,
                fpram_const_count=len(const_names),
                output_buffer_name="Y_hbm")


def compile_rope_step(
    *, name: str, xq_addr: int, cos_addr: int, sin_addr: int,
    neg_sin_addr: int, y_addr: int, fpram_const_base: int,
) -> Step:
    """RoPE step on the normed q.

    make_rope_min declares HBM buffers ``XQ_hbm, COS_hbm, SIN_hbm,
    NEG_SIN_hbm, Q_OUT_hbm``, all shaped (batch, seq_len, head_count,
    hlen). XQ aliases the QKNorm-q output with no copy.
    """
    prim_func, _ = make_rope_min(
        rows=ROWS, hlen=HLEN, head_count=HEAD_COUNT,
        half_dim=HLEN // 2, num_s_blocks=NUM_S_BLOCKS, batch=BATCH,
    )
    const_names = _hoisted_const_names(prim_func)
    fpram_overrides = {
        cn: fpram_const_base + i for i, cn in enumerate(const_names)
    }
    fpram_scratch_base = FPRAM_SCRATCH_BASE

    hbm_overrides = {
        "XQ_hbm":      xq_addr,
        "COS_hbm":     cos_addr,
        "SIN_hbm":     sin_addr,
        "NEG_SIN_hbm": neg_sin_addr,
        "Q_OUT_hbm":   y_addr,
    }

    addr_cfg = AddressAllocConfig(
        mlen=MLEN, blen=4, hlen=HLEN,
        hbm_address_overrides=hbm_overrides,
        fpram_address_overrides=fpram_overrides,
        fpram_base=fpram_scratch_base,
    )
    compiled = compile_kernel(
        prim_func, target=PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        name=name, addr_config_override=addr_cfg,
    )
    return Step(name=name, compiled=compiled,
                fpram_const_count=len(const_names),
                output_buffer_name="Q_OUT_hbm")


def compile_flash_attention_step(
    *, name: str, q_addr: int, k_addr: int, v_addr: int, o_addr: int,
    fpram_const_base: int,
) -> Step:
    """Self-attention step over the projected q/k/v.

    make_flash_attention_min declares HBM buffers ``Q_hbm, K_hbm,
    V_hbm, O_hbm``. q/k/v/O are all shaped (1, seq, head_count, hlen);
    O is a COMPACT [S, HIDDEN_SIZE] output tensor (o_head_count
    defaults to head_count, no o_head_offset). q/k/v alias the rope_q /
    rope_k / linear_v outputs with no copy. The concat_min kernel later
    joins this compact O with gelu's compact output.
    """
    prim_func, _ = make_flash_attention_min(
        rows=ROWS, hlen=HLEN, head_count=HEAD_COUNT,
        active_lane=ATTN_ACTIVE_LANE,
        num_kv_blocks=ATTN_NUM_KV_BLOCKS, num_q_blocks=ATTN_NUM_Q_BLOCKS,
    )
    const_names = _hoisted_const_names(prim_func)
    fpram_overrides = {
        cn: fpram_const_base + i for i, cn in enumerate(const_names)
    }
    fpram_scratch_base = FPRAM_SCRATCH_BASE

    hbm_overrides = {
        "Q_hbm": q_addr,
        "K_hbm": k_addr,
        "V_hbm": v_addr,
        "O_hbm": o_addr,
    }

    addr_cfg = AddressAllocConfig(
        mlen=MLEN, blen=4, hlen=HLEN,
        hbm_address_overrides=hbm_overrides,
        fpram_address_overrides=fpram_overrides,
        fpram_base=fpram_scratch_base,
    )
    compiled = compile_kernel(
        prim_func, target=PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        name=name, addr_config_override=addr_cfg,
    )
    return Step(name=name, compiled=compiled,
                fpram_const_count=len(const_names),
                output_buffer_name="O_hbm")


def compile_gelu_step(
    *, name: str, x_addr: int, y_addr: int, fpram_const_base: int,
) -> Step:
    """GELU activation step for the FFN back half.

    make_gelu_min declares HBM buffers ``X_hbm, Y_hbm``. X is shaped
    (batch, seq, MLP_HEAD_COUNT, hlen) — the mlp-in projection. Y is a
    COMPACT [S, MLP_HIDDEN_DIM] output tensor (o_head_count defaults to
    head_count, no o_head_offset). The concat_min kernel later joins
    this compact Y with attention's compact output. gelu hoists FIVE
    float constants (0.5, 1.0, 2.0, sqrt(2/pi), 0.044715) — the widest
    const footprint in the chain — so fpram_const_base must leave room
    for all five.
    """
    prim_func, _ = make_gelu_min(
        rows=ROWS, hlen=HLEN, head_count=MLP_HEAD_COUNT,
        num_s_blocks=GELU_NUM_S_BLOCKS, batch=BATCH,
    )
    const_names = _hoisted_const_names(prim_func)
    fpram_overrides = {
        cn: fpram_const_base + i for i, cn in enumerate(const_names)
    }
    fpram_scratch_base = FPRAM_SCRATCH_BASE

    hbm_overrides = {
        "X_hbm": x_addr,
        "Y_hbm": y_addr,
    }

    addr_cfg = AddressAllocConfig(
        mlen=MLEN, blen=4, hlen=HLEN,
        hbm_address_overrides=hbm_overrides,
        fpram_address_overrides=fpram_overrides,
        fpram_base=fpram_scratch_base,
    )
    compiled = compile_kernel(
        prim_func, target=PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        name=name, addr_config_override=addr_cfg,
    )
    return Step(name=name, compiled=compiled,
                fpram_const_count=len(const_names),
                output_buffer_name="Y_hbm")


def compile_concat_step(
    *, name: str, a_addr: int, b_addr: int, y_addr: int,
    fpram_const_base: int,
) -> Step:
    """Feature-axis concat step joining the attention + gelu outputs.

    make_concat_min declares HBM buffers ``A_hbm, B_hbm, Y_hbm``. A is
    attention's compact [S, HIDDEN_SIZE] output, B is gelu's compact
    [S, MLP_HIDDEN_DIM] output, Y is the WIDE [S, CONCAT_DIM] concat:
    Y[..., 0:HIDDEN_SIZE] = A, Y[..., HIDDEN_SIZE:] = B. A/B alias the
    attention / gelu outputs with no copy. Plain VRAM->VRAM copy — no
    hoisted constants.
    """
    prim_func, _ = make_concat_min(
        rows=ROWS, a_dim=HIDDEN_SIZE, b_dim=MLP_HIDDEN_DIM,
        num_s_blocks=NUM_S_BLOCKS, batch=BATCH,
    )
    const_names = _hoisted_const_names(prim_func)
    fpram_overrides = {
        cn: fpram_const_base + i for i, cn in enumerate(const_names)
    }
    fpram_scratch_base = FPRAM_SCRATCH_BASE

    hbm_overrides = {
        "A_hbm": a_addr,
        "B_hbm": b_addr,
        "Y_hbm": y_addr,
    }

    addr_cfg = AddressAllocConfig(
        mlen=MLEN, blen=4, hlen=HLEN,
        hbm_address_overrides=hbm_overrides,
        fpram_address_overrides=fpram_overrides,
        fpram_base=fpram_scratch_base,
    )
    compiled = compile_kernel(
        prim_func, target=PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        name=name, addr_config_override=addr_cfg,
    )
    return Step(name=name, compiled=compiled,
                fpram_const_count=len(const_names),
                output_buffer_name="Y_hbm")


def compile_residual_gate_step(
    *, name: str, x_addr: int, gate_addr: int, y_addr: int, out_addr: int,
    fpram_const_base: int,
) -> Step:
    """Residual-gate step: out = x + gate * y.

    make_residual_gate_min declares HBM buffers ``X_hbm, GATE_hbm,
    Y_hbm, OUT_hbm``, all shaped (batch, seq, head_count, hlen). X is
    the block's original LN input (residual), Y the out-projection
    result; both alias their producers with no copy.
    """
    prim_func, _ = make_residual_gate_min(
        rows=ROWS, hlen=HLEN, head_count=HEAD_COUNT,
        num_s_blocks=NUM_S_BLOCKS, batch=BATCH,
    )
    const_names = _hoisted_const_names(prim_func)
    fpram_overrides = {
        cn: fpram_const_base + i for i, cn in enumerate(const_names)
    }
    fpram_scratch_base = FPRAM_SCRATCH_BASE

    hbm_overrides = {
        "X_hbm":    x_addr,
        "GATE_hbm": gate_addr,
        "Y_hbm":    y_addr,
        "OUT_hbm":  out_addr,
    }

    addr_cfg = AddressAllocConfig(
        mlen=MLEN, blen=4, hlen=HLEN,
        hbm_address_overrides=hbm_overrides,
        fpram_address_overrides=fpram_overrides,
        fpram_base=fpram_scratch_base,
    )
    compiled = compile_kernel(
        prim_func, target=PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        name=name, addr_config_override=addr_cfg,
    )
    return Step(name=name, compiled=compiled,
                fpram_const_count=len(const_names),
                output_buffer_name="OUT_hbm")


# ---------------------------------------------------------------------------
# Merge FP preload — for each step, walk every global.fpram buffer that
# carries a constant_value (set by to_plena from the hoisted-constants
# attr), write it at the buffer's now-pinned address.
# ---------------------------------------------------------------------------
def merge_fp_preload(steps: list[Step]) -> torch.Tensor:
    entries: list[tuple[int, float]] = []
    for step in steps:
        for buf in step.compiled.hlir.buffers.values():
            # Hoisted const buffers carry constant_value; their scope is
            # collapsed to "fpram" by physical_scope before HLIR build,
            # so we match on the const_value presence, not on the
            # "global.fpram" string.
            if buf.constant_value is None:
                continue
            entries.append((int(buf.address), float(buf.constant_value)))

    if not entries:
        return torch.zeros(FPRAM_USER_BASE, dtype=torch.float16)

    max_addr = max(addr for addr, _ in entries)
    preload = torch.zeros(max_addr + 1, dtype=torch.float16)
    for addr, value in entries:
        preload[addr] = value
    return preload


# ---------------------------------------------------------------------------
# Golden precision model.
#
# Every chain stage's golden is computed in fp16: each HBM round-trip
# point is fp16-truncated via _q(). The hbm_shape argument is accepted
# for call-site compatibility but ignored — fp16 truncation is
# element-wise, so the tensor shape does not matter.
# ---------------------------------------------------------------------------
def _q(x: torch.Tensor, hbm_shape: tuple[int, ...] | None = None) -> torch.Tensor:
    """fp16-truncate a tensor (kept in fp32 storage). ``hbm_shape`` is
    ignored — fp16 rounding is element-wise."""
    del hbm_shape
    return x.to(torch.float16).to(torch.float32)


# ---------------------------------------------------------------------------
# Inputs + golden — host PyTorch reference. Returns hbm_inputs in the
# SAME order as the global HBM layout (alias entries excluded).
# ---------------------------------------------------------------------------
def build_inputs_and_golden(
    layout: list[HbmTensor], seed: int = 0,
) -> dict:
    torch.manual_seed(seed)

    # LN inputs.
    x = torch.randn(BATCH, SEQ_LEN, 1, HIDDEN_SIZE, dtype=torch.float32) * 0.5
    ln_scale = torch.randn(HIDDEN_SIZE, dtype=torch.float32) * 0.3 + 1.0
    ln_bias = torch.randn(HIDDEN_SIZE, dtype=torch.float32) * 0.1
    ln_scale_full = ln_scale.view(1, 1, 1, HIDDEN_SIZE).expand(
        BATCH, SEQ_LEN, 1, HIDDEN_SIZE).contiguous()
    ln_bias_full = ln_bias.view(1, 1, 1, HIDDEN_SIZE).expand(
        BATCH, SEQ_LEN, 1, HIDDEN_SIZE).contiguous()

    # Host LN reference (matches tvm_layernorm_min_test.py). The kernel
    # reads X / scale / bias from HBM (MX-E4M3) — _q() applies that, each
    # in its staged 4D HBM shape.
    _hd = (BATCH, SEQ_LEN, 1, HIDDEN_SIZE)
    x_q = _q(x, _hd)
    mu = x_q.mean(dim=-1, keepdim=True)
    xc = x_q - mu
    var = (xc * xc).mean(dim=-1, keepdim=True)
    inv = torch.rsqrt(var + EPS)
    # ln_scale_full / ln_bias_full are expand()'d (every row identical);
    # quantize the staged 4D tile, then take one row back to (HIDDEN_SIZE,).
    y_ln = xc * inv * _q(ln_scale_full, _hd)[0, 0, 0, :] + \
        _q(ln_bias_full, _hd)[0, 0, 0, :]

    # Modulate — host operates on the BSHD view of LN's output (same
    # bytes, different logical shape, see kernels/_head_layout.py). LN's
    # output went to HBM and is read back -> _q().
    _bshd = (BATCH, SEQ_LEN, HEAD_COUNT, HLEN)
    y_ln_bshd = _q(y_ln, _hd).view(BATCH, SEQ_LEN, HEAD_COUNT, HLEN)
    mod_scale = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.3
    mod_shift = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.3
    mod_scale_plus_one = 1.0 + mod_scale
    y_mod_golden = _q(mod_scale_plus_one, _bshd) * y_ln_bshd + _q(mod_shift, _bshd)

    # linear_{q,k,v,mlp} — x_mod [M, K] @ W [out, K].T + b. x_mod is the
    # modulate output's bytes reinterpreted as (M=SEQ_LEN, K=HIDDEN_SIZE):
    # a pure head-merge view (B,S,H,D -> B,S,1,H*D -> M x K). Modulate's
    # output went to HBM and is read back -> fp16-truncated via _q().
    x_mod_mk = _q(y_mod_golden, _bshd).reshape(SEQ_LEN, HIDDEN_SIZE)  # (M, K)

    def _proj(out_dim: int, a_mk: torch.Tensor = x_mod_mk, k_dim: int = HIDDEN_SIZE):
        """Random (W, bias) for one projection + its golden output.

        ``a_mk`` is the (M, K) input matrix — defaults to x_mod (the
        q/k/v/mlp projections); linear2 passes the concat([attn, gelu])
        tensor. ``k_dim`` is the K extent (a_mk's column count);
        defaults to HIDDEN_SIZE, linear2 passes CONCAT_DIM.

        W / bias are HBM inputs; a_mk has already been _q()'d by the
        caller. MM itself is fp32, so the matmul stays unquantized.
        W / bias are quantized in their staged 4D HBM shapes:
        w_hbm (1, out, 1, k_dim), b_hbm (1, SEQ_LEN, 1, out).
        """
        w = torch.randn(out_dim, k_dim, dtype=torch.float32) * 0.25       # (out, K)
        b = torch.randn(out_dim, dtype=torch.float32) * 0.1               # (out,)
        w_hbm = w.view(1, out_dim, 1, k_dim).contiguous()
        b_hbm = b.view(1, 1, 1, out_dim).expand(1, SEQ_LEN, 1, out_dim).contiguous()
        w_eff = _q(w_hbm, (1, out_dim, 1, k_dim)).view(out_dim, k_dim)
        # bias is broadcast (out,) across M rows — staged as (1,SEQ,1,out);
        # quantize that staged tile, then take one row back to (out,).
        b_eff = _q(b_hbm, (1, SEQ_LEN, 1, out_dim))[0, 0, 0, :]
        y = a_mk @ w_eff.T + b_eff                                        # (M, out)
        return w_hbm, b_hbm, y

    wq, bq, y_q = _proj(HIDDEN_SIZE)
    wk, bk, y_k = _proj(HIDDEN_SIZE)
    wv, bv, y_v = _proj(HIDDEN_SIZE)
    wm, bm, y_mlp = _proj(MLP_HIDDEN_DIM)

    # QKNorm — RMSNorm over head_dim (last axis) on q and k. The kernel
    # reads its input as BSHD; q's compact [M, H*D] bytes are exactly
    # [1, S, H, D] row-major, so reshape with no copy.
    def _rmsnorm(y_compact: torch.Tensor):
        """RMSNorm(y) over head_dim. y_compact: (M, HIDDEN_SIZE). The
        linear projection's output went to HBM as (1,SEQ,1,HIDDEN_SIZE)
        and is read back -> _q(); scale is staged as (B,S,H,D)."""
        x_bshd = _q(y_compact, (1, SEQ_LEN, 1, HIDDEN_SIZE)).reshape(
            BATCH, SEQ_LEN, HEAD_COUNT, HLEN)
        scale = torch.randn(HEAD_COUNT, HLEN, dtype=torch.float32) * 0.3 + 1.0
        scale_full = scale.view(1, 1, HEAD_COUNT, HLEN).expand(
            BATCH, SEQ_LEN, HEAD_COUNT, HLEN).contiguous()
        scale_eff = _q(scale_full, _bshd)[0, 0, :, :]
        ms = (x_bshd * x_bshd).mean(dim=-1, keepdim=True)
        inv = torch.rsqrt(ms + EPS)
        y = x_bshd * inv * scale_eff
        return scale_full, y

    qkn_q_scale, y_qkn_q = _rmsnorm(y_q)
    qkn_k_scale, y_qkn_k = _rmsnorm(y_k)

    # RoPE on the normed q AND k — same rotary frequency for both,
    # matching the reference tilelang_apply_rope(q, k, pe). Must match
    # rope_min.py's interleaved pair-swap EXACTLY (NOT GPT-NeoX
    # half-split): for each adjacent (even, odd) pair e=2i, o=2i+1,
    #   out[e] = x[e]*cos[e] + x[o]*neg_sin[e]
    #   out[o] = x[o]*cos[o] + x[e]*sin[o]
    # cos / sin / neg_sin are per-element (S,H,D) broadcast tensors.
    rope_cos = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32)
    rope_sin = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32)
    rope_neg_sin = -rope_sin
    # cos / sin / neg_sin are HBM inputs -> _q(), each staged (B,S,H,D).
    # neg_sin is staged as its own HBM tensor (ROPE_NEG_SIN_hbm), so
    # quantize it directly rather than negating the quantized sin.
    _cos_q = _q(rope_cos, _bshd)
    _sin_q = _q(rope_sin, _bshd)
    _neg_sin_q = _q(rope_neg_sin, _bshd)

    def _apply_rope(x_bshd: torch.Tensor) -> torch.Tensor:
        # The normed q/k went to HBM as (B,S,H,D) and is read back -> _q().
        x_bshd = _q(x_bshd, _bshd)
        out = torch.empty_like(x_bshd)
        for i in range(HLEN // 2):
            e, o = 2 * i, 2 * i + 1
            out[..., e] = x_bshd[..., e] * _cos_q[..., e] + x_bshd[..., o] * _neg_sin_q[..., e]
            out[..., o] = x_bshd[..., o] * _cos_q[..., o] + x_bshd[..., e] * _sin_q[..., o]
        return out

    y_rope_q = _apply_rope(y_qkn_q)
    y_rope_k = _apply_rope(y_qkn_k)

    # flash_attention — self-attention over the projected q/k/v. Q is
    # the rotated q, K the rotated k, V the raw v projection reshaped
    # to BSHD. Must match flash_attention_min's golden: per-head
    # scaled-dot-product softmax with scale = 1/sqrt(HLEN).
    # Q / K / V are all read back from HBM by the attention kernel -> _q().
    # rope_q/k were staged (B,S,H,D); the v projection was staged into
    # V_hbm as (1, SEQ_LEN, 1, HIDDEN_SIZE).
    attn_q = _q(y_rope_q, _bshd)                                   # (B, S, H, D)
    attn_k = _q(y_rope_k, _bshd)                                   # (B, S, H, D)
    attn_v = _q(y_v, (1, SEQ_LEN, 1, HIDDEN_SIZE)).reshape(
        BATCH, SEQ_LEN, HEAD_COUNT, HLEN)                          # v proj -> BSHD
    attn_scale = 1.0 / math.sqrt(HLEN)
    attn_score = torch.einsum("bihd,bjhd->bihj", attn_q, attn_k)
    y_attn = torch.empty(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32)
    for h in range(HEAD_COUNT):
        p = torch.softmax(attn_score[:, :, h, :] * attn_scale, dim=-1)
        y_attn[:, :, h, :] = torch.einsum("bij,bjd->bid", p, attn_v[:, :, h, :])

    # GELU on the mlp-in projection. gelu uses the tanh approximation
    # — must match gelu_min.py EXACTLY:
    #   GELU(x) = 0.5 * x * (1 + tanh(u)),
    #   u = sqrt(2/pi) * (x + 0.044715 * x^3),
    #   tanh(u) = 1 - 2 / (exp(2u) + 1)   [kernel has no native tanh]
    # mlp projection is read back from HBM by the gelu kernel -> _q();
    # it was staged into MLP_hbm as (1, SEQ_LEN, 1, MLP_HIDDEN_DIM).
    mlp_in_bshd = _q(y_mlp, (1, SEQ_LEN, 1, MLP_HIDDEN_DIM)).reshape(
        BATCH, SEQ_LEN, MLP_HEAD_COUNT, HLEN)
    _sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    _u = _sqrt_2_over_pi * (mlp_in_bshd + 0.044715 * mlp_in_bshd ** 3)
    _tanh_u = 1.0 - 2.0 / (torch.exp(2.0 * _u) + 1.0)
    y_gelu = 0.5 * mlp_in_bshd * (1.0 + _tanh_u)

    # concat([attn_out, mlp_out]) along the feature axis — the linear2
    # input and the golden for the concat step. attention's compact
    # output fills columns [0, HIDDEN_SIZE), gelu's compact output the
    # rest. This is exactly what concat_min copies into CONCAT_hbm.
    # attention's and gelu's outputs are read back from HBM by the
    # concat kernel -> _q(). attn staged (B,S,H,D); gelu staged
    # (B,S,MLP_HEAD_COUNT,HLEN).
    y_concat = torch.cat(
        [_q(y_attn, _bshd).reshape(SEQ_LEN, HIDDEN_SIZE),
         _q(y_gelu, (BATCH, SEQ_LEN, MLP_HEAD_COUNT, HLEN)).reshape(
             SEQ_LEN, MLP_HIDDEN_DIM)],
        dim=-1,
    )  # (M, CONCAT_DIM)

    # linear2 — project the concat back to [M, H*D]. combined [M,
    # CONCAT_DIM] @ W2[H*D, CONCAT_DIM].T + b2. K = CONCAT_DIM. The
    # concat output went to HBM as (B,S,1,CONCAT_DIM) and is read back.
    w2, b2, y_lin2 = _proj(
        HIDDEN_SIZE,
        a_mk=_q(y_concat, (BATCH, SEQ_LEN, 1, CONCAT_DIM)),
        k_dim=CONCAT_DIM,
    )

    # residual_gate — out = x_residual + gate * linear2_out. x_residual
    # is the block's original LN input (X); gate is the Modulation
    # gate. Must match residual_gate_min: tmp = gate*y; out = x + tmp.
    gate = torch.randn(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32) * 0.3
    # x_residual is the block's original X (HBM input) — quantize via
    # the same x_q used by the LN step; gate is staged (B,S,H,D);
    # linear2's output was staged (1,SEQ,1,HIDDEN_SIZE).
    x_residual = x_q.reshape(BATCH, SEQ_LEN, HEAD_COUNT, HLEN)
    lin2_bshd = _q(y_lin2, (1, SEQ_LEN, 1, HIDDEN_SIZE)).reshape(
        BATCH, SEQ_LEN, HEAD_COUNT, HLEN)
    x_out = x_residual + _q(gate, _bshd) * lin2_bshd

    # Map name -> tensor for non-alias layout entries.
    tensors_by_name = {
        "X_hbm":           x,
        "LN_SCALE_hbm":    ln_scale_full,
        "LN_BIAS_hbm":     ln_bias_full,
        "LN_Y_hbm":        torch.zeros_like(x),                          # scratch
        "MOD_SCALE1P_hbm": mod_scale_plus_one,
        "MOD_SHIFT_hbm":   mod_shift,
        "MOD_Y_hbm":       torch.zeros(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32),
        "LINQ_W_hbm": wq, "LINQ_BIAS_hbm": bq,
        "Q_hbm":   torch.zeros(1, SEQ_LEN, 1, HIDDEN_SIZE, dtype=torch.float32),
        "LINK_W_hbm": wk, "LINK_BIAS_hbm": bk,
        "K_hbm":   torch.zeros(1, SEQ_LEN, 1, HIDDEN_SIZE, dtype=torch.float32),
        "LINV_W_hbm": wv, "LINV_BIAS_hbm": bv,
        "V_hbm":   torch.zeros(1, SEQ_LEN, 1, HIDDEN_SIZE, dtype=torch.float32),
        "LINM_W_hbm": wm, "LINM_BIAS_hbm": bm,
        "MLP_hbm": torch.zeros(1, SEQ_LEN, 1, MLP_HIDDEN_DIM, dtype=torch.float32),
        "QKN_Q_SCALE_hbm": qkn_q_scale,
        "QKN_Q_Y_hbm": torch.zeros(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32),
        "QKN_K_SCALE_hbm": qkn_k_scale,
        "QKN_K_Y_hbm": torch.zeros(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32),
        "ROPE_COS_hbm":     rope_cos,
        "ROPE_SIN_hbm":     rope_sin,
        "ROPE_NEG_SIN_hbm": rope_neg_sin,
        "ROPE_Q_Y_hbm": torch.zeros(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32),
        "ROPE_K_Y_hbm": torch.zeros(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32),
        # attention + gelu compact output tensors (scratch).
        "ATTN_O_hbm":   torch.zeros(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32),
        "GELU_OUT_hbm": torch.zeros(BATCH, SEQ_LEN, MLP_HEAD_COUNT, HLEN, dtype=torch.float32),
        # concat([attn, mlp]) tensor — wide scratch, concat_min joins
        # the two compact tensors into it.
        "CONCAT_hbm": torch.zeros(BATCH, SEQ_LEN, 1, CONCAT_DIM, dtype=torch.float32),
        # linear2: weight + bias projecting the concat back to [M, H*D].
        "LIN2_W_hbm": w2, "LIN2_BIAS_hbm": b2,
        "LIN2_OUT_hbm": torch.zeros(1, SEQ_LEN, 1, HIDDEN_SIZE, dtype=torch.float32),
        # residual_gate: gate input + FINAL block output (scratch).
        "GATE_G_hbm": gate,
        "BLOCK_OUT_hbm": torch.zeros(BATCH, SEQ_LEN, HEAD_COUNT, HLEN, dtype=torch.float32),
    }

    hbm_inputs = {}
    for t in layout:
        if t.is_alias:
            continue
        hbm_inputs[t.name] = tensors_by_name[t.name]

    # Final golden = residual_gate output — the full SingleStreamBlock
    # (Open-Sora): x + gate * linear2(concat([attn, gelu(mlp)])),
    # flattened to (M, HIDDEN_SIZE).
    golden_flat = x_out.reshape(SEQ_LEN, HIDDEN_SIZE)

    # ---- per-step intermediates for diagnostic verification ----
    # Set SSB_VERIFY=<step> to stage+verify that step's output buffer
    # instead of the full block — pinpoints which kernel introduces
    # error. The DATA FLOW is unchanged: the whole chain still runs;
    # only the staged buffer + golden switch. Each entry:
    #   step name -> (golden_flat, output_buffer_name)
    #
    # NOTE: attention and gelu now each write their OWN compact output
    # tensor (no shared wide CONCAT, no o_head_offset). Each step is
    # therefore independently verifiable — flash_attention stages its
    # compact O_hbm, gelu its compact Y_hbm, concat its wide Y_hbm.
    intermediates = {
        "layernorm":      (y_ln.reshape(SEQ_LEN, HIDDEN_SIZE),       "Y_hbm"),
        "modulate":       (y_mod_golden.reshape(SEQ_LEN, HIDDEN_SIZE), "Y_hbm"),
        "linear_q":       (y_q.reshape(SEQ_LEN, HIDDEN_SIZE),        "C_hbm"),
        "linear_k":       (y_k.reshape(SEQ_LEN, HIDDEN_SIZE),        "C_hbm"),
        "linear_v":       (y_v.reshape(SEQ_LEN, HIDDEN_SIZE),        "C_hbm"),
        "linear_mlp":     (y_mlp.reshape(SEQ_LEN, MLP_HIDDEN_DIM),   "C_hbm"),
        "qknorm_q":       (y_qkn_q.reshape(SEQ_LEN, HIDDEN_SIZE),    "Y_hbm"),
        "qknorm_k":       (y_qkn_k.reshape(SEQ_LEN, HIDDEN_SIZE),    "Y_hbm"),
        "rope_q":         (y_rope_q.reshape(SEQ_LEN, HIDDEN_SIZE),   "Q_OUT_hbm"),
        "rope_k":         (y_rope_k.reshape(SEQ_LEN, HIDDEN_SIZE),   "Q_OUT_hbm"),
        "flash_attention": (y_attn.reshape(SEQ_LEN, HIDDEN_SIZE),    "O_hbm"),
        "gelu":           (y_gelu.reshape(SEQ_LEN, MLP_HIDDEN_DIM),  "Y_hbm"),
        "concat":         (y_concat.reshape(SEQ_LEN, CONCAT_DIM),    "Y_hbm"),
        "linear2":        (y_lin2.reshape(SEQ_LEN, HIDDEN_SIZE),     "C_hbm"),
        "residual_gate":  (golden_flat,                             "OUT_hbm"),
    }

    return {
        "hbm_inputs": hbm_inputs,
        "golden_flat": golden_flat,
        "intermediates": intermediates,
    }


# ---------------------------------------------------------------------------
# Driver entry.
# ---------------------------------------------------------------------------
def main() -> int:
    print("[single_stream_block] layernorm -> modulate -> "
          "linear_{q,k,v,mlp} -> qknorm_{q,k} -> rope_{q,k} -> "
          "flash_attention -+-> concat -> linear2 -> residual_gate")
    print("                  gelu(mlp) ---+")
    print()

    # ----- 1. plan HBM addresses -----
    addr_cfg_proto = AddressAllocConfig(mlen=MLEN, blen=4, hlen=HLEN)
    layout = _block_hbm_layout()
    addr_plan = _compute_address_plan(layout, addr_cfg_proto)
    print("[1/5] HBM address plan:")
    for t in layout:
        marker = "  (alias)" if t.is_alias else ""
        print(f"      {t.name:<22s} @ {addr_plan[t.name]:>8d} bytes  "
              f"({t.num_elements} elems){marker}")

    # ----- 2. compile each step with pinned addresses -----
    print("[2/5] Compiling chain ...")
    fpram_cursor = FPRAM_USER_BASE
    step_ln = compile_layernorm_step(
        addr_plan=addr_plan, fpram_const_base=fpram_cursor,
    )
    print(f"      layernorm_min: hoisted_consts={step_ln.fpram_const_count}, "
          f"FPRAM slots {fpram_cursor}..{fpram_cursor + step_ln.fpram_const_count}")
    fpram_cursor += step_ln.fpram_const_count

    step_mod = compile_modulate_step(
        addr_plan=addr_plan, fpram_const_base=fpram_cursor,
    )
    print(f"      modulate_min:  hoisted_consts={step_mod.fpram_const_count}, "
          f"FPRAM slots {fpram_cursor}..{fpram_cursor + step_mod.fpram_const_count}")
    fpram_cursor += step_mod.fpram_const_count

    # Four projection GEMMs — q / k / v share HIDDEN_SIZE out-dim,
    # mlp uses MLP_HIDDEN_DIM. Each writes a compact [M, out] HBM region.
    linear_specs = [
        ("linear_q",   QKV_N_BLOCKS, "LINQ_A_hbm", "LINQ_W_hbm", "LINQ_BIAS_hbm", "Q_hbm"),
        ("linear_k",   QKV_N_BLOCKS, "LINK_A_hbm", "LINK_W_hbm", "LINK_BIAS_hbm", "K_hbm"),
        ("linear_v",   QKV_N_BLOCKS, "LINV_A_hbm", "LINV_W_hbm", "LINV_BIAS_hbm", "V_hbm"),
        ("linear_mlp", MLP_N_BLOCKS, "LINM_A_hbm", "LINM_W_hbm", "LINM_BIAS_hbm", "MLP_hbm"),
    ]
    linear_steps = []
    for name, n_blocks, a_key, w_key, bias_key, y_key in linear_specs:
        st = compile_linear_step(
            name=name, n_blocks=n_blocks,
            a_addr=addr_plan[a_key], w_addr=addr_plan[w_key],
            bias_addr=addr_plan[bias_key], y_addr=addr_plan[y_key],
            fpram_const_base=fpram_cursor,
        )
        print(f"      {name:<12s} hoisted_consts={st.fpram_const_count}, "
              f"FPRAM slots {fpram_cursor}..{fpram_cursor + st.fpram_const_count}")
        fpram_cursor += st.fpram_const_count
        linear_steps.append(st)

    # QKNorm: RMSNorm(q) and RMSNorm(k) per head_dim. Each reads the
    # projection output in-place (X aliases Q_hbm / K_hbm) and writes a
    # normed copy to QKN_*_Y_hbm.
    qknorm_specs = [
        ("qknorm_q", "QKN_Q_X_hbm", "QKN_Q_SCALE_hbm", "QKN_Q_Y_hbm"),
        ("qknorm_k", "QKN_K_X_hbm", "QKN_K_SCALE_hbm", "QKN_K_Y_hbm"),
    ]
    qknorm_steps = []
    for name, x_key, scale_key, y_key in qknorm_specs:
        st = compile_rmsnorm_step(
            name=name,
            x_addr=addr_plan[x_key], scale_addr=addr_plan[scale_key],
            y_addr=addr_plan[y_key],
            fpram_const_base=fpram_cursor,
        )
        print(f"      {name:<12s} hoisted_consts={st.fpram_const_count}, "
              f"FPRAM slots {fpram_cursor}..{fpram_cursor + st.fpram_const_count}")
        fpram_cursor += st.fpram_const_count
        qknorm_steps.append(st)

    # RoPE on the normed q. XQ aliases QKN_Q_Y_hbm; writes rotated q to
    # ROPE_Q_Y_hbm. K-side RoPE is left out (rope_min is Q-only).
    # RoPE on the normed q.
    step_rope_q = compile_rope_step(
        name="rope_q",
        xq_addr=addr_plan["ROPE_Q_X_hbm"],
        cos_addr=addr_plan["ROPE_COS_hbm"],
        sin_addr=addr_plan["ROPE_SIN_hbm"],
        neg_sin_addr=addr_plan["ROPE_NEG_SIN_hbm"],
        y_addr=addr_plan["ROPE_Q_Y_hbm"],
        fpram_const_base=fpram_cursor,
    )
    print(f"      {'rope_q':<14s} hoisted_consts={step_rope_q.fpram_const_count}, "
          f"FPRAM slots {fpram_cursor}..{fpram_cursor + step_rope_q.fpram_const_count}")
    fpram_cursor += step_rope_q.fpram_const_count

    # RoPE on the normed k — same rotary frequency (COS/SIN/NEG_SIN) as
    # rope_q, matching the reference tilelang_apply_rope(q, k, pe).
    step_rope_k = compile_rope_step(
        name="rope_k",
        xq_addr=addr_plan["ROPE_K_X_hbm"],
        cos_addr=addr_plan["ROPE_COS_hbm"],
        sin_addr=addr_plan["ROPE_SIN_hbm"],
        neg_sin_addr=addr_plan["ROPE_NEG_SIN_hbm"],
        y_addr=addr_plan["ROPE_K_Y_hbm"],
        fpram_const_base=fpram_cursor,
    )
    print(f"      {'rope_k':<14s} hoisted_consts={step_rope_k.fpram_const_count}, "
          f"FPRAM slots {fpram_cursor}..{fpram_cursor + step_rope_k.fpram_const_count}")
    fpram_cursor += step_rope_k.fpram_const_count

    # flash_attention: self-attention over rope_q (Q), rope_k (K), v (V).
    # Writes a COMPACT [S, HIDDEN_SIZE] output tensor (ATTN_O_hbm).
    step_attn = compile_flash_attention_step(
        name="flash_attention",
        q_addr=addr_plan["ATTN_Q_hbm"],
        k_addr=addr_plan["ATTN_K_hbm"],
        v_addr=addr_plan["ATTN_V_hbm"],
        o_addr=addr_plan["ATTN_O_hbm"],
        fpram_const_base=fpram_cursor,
    )
    print(f"      {'flash_attention':<14s} hoisted_consts={step_attn.fpram_const_count}, "
          f"FPRAM slots {fpram_cursor}..{fpram_cursor + step_attn.fpram_const_count}")
    fpram_cursor += step_attn.fpram_const_count

    # gelu: activate the mlp-in projection; writes a COMPACT
    # [S, MLP_HIDDEN_DIM] output tensor (GELU_OUT_hbm).
    step_gelu = compile_gelu_step(
        name="gelu",
        x_addr=addr_plan["GELU_X_hbm"],
        y_addr=addr_plan["GELU_OUT_hbm"],
        fpram_const_base=fpram_cursor,
    )
    print(f"      {'gelu':<14s} hoisted_consts={step_gelu.fpram_const_count}, "
          f"FPRAM slots {fpram_cursor}..{fpram_cursor + step_gelu.fpram_const_count}")
    fpram_cursor += step_gelu.fpram_const_count

    # concat: join attention's + gelu's compact outputs into the wide
    # CONCAT tensor along the feature axis.
    step_concat = compile_concat_step(
        name="concat",
        a_addr=addr_plan["CONCAT_A_hbm"],
        b_addr=addr_plan["CONCAT_B_hbm"],
        y_addr=addr_plan["CONCAT_hbm"],
        fpram_const_base=fpram_cursor,
    )
    print(f"      {'concat':<14s} hoisted_consts={step_concat.fpram_const_count}, "
          f"FPRAM slots {fpram_cursor}..{fpram_cursor + step_concat.fpram_const_count}")
    fpram_cursor += step_concat.fpram_const_count

    # linear2: project concat([attn, mlp]) [M, CONCAT_DIM] back to
    # [M, H*D]. K = CONCAT_DIM (the concat width).
    step_lin2 = compile_linear_step(
        name="linear2", n_blocks=LINEAR2_N_BLOCKS,
        a_addr=addr_plan["LIN2_A_hbm"], w_addr=addr_plan["LIN2_W_hbm"],
        bias_addr=addr_plan["LIN2_BIAS_hbm"], y_addr=addr_plan["LIN2_OUT_hbm"],
        fpram_const_base=fpram_cursor,
        k_blocks=LINEAR2_K_BLOCKS,
    )
    print(f"      {'linear2':<14s} hoisted_consts={step_lin2.fpram_const_count}, "
          f"FPRAM slots {fpram_cursor}..{fpram_cursor + step_lin2.fpram_const_count}")
    fpram_cursor += step_lin2.fpram_const_count

    # residual_gate: out = x_residual + gate * linear2_out. The single
    # gated residual that closes the block (reference step 12).
    step_gate = compile_residual_gate_step(
        name="residual_gate",
        x_addr=addr_plan["GATE_X_hbm"],
        gate_addr=addr_plan["GATE_G_hbm"],
        y_addr=addr_plan["GATE_Y_hbm"],
        out_addr=addr_plan["BLOCK_OUT_hbm"],
        fpram_const_base=fpram_cursor,
    )
    print(f"      {'residual_gate':<14s} hoisted_consts={step_gate.fpram_const_count}, "
          f"FPRAM slots {fpram_cursor}..{fpram_cursor + step_gate.fpram_const_count}")
    fpram_cursor += step_gate.fpram_const_count

    steps = [step_ln, step_mod, *linear_steps, *qknorm_steps,
             step_rope_q, step_rope_k, step_attn, step_gelu,
             step_concat, step_lin2, step_gate]

    # Sanity: confirm each kernel's HBM buffers really landed at the
    # planned addresses (no driver/alloc-pass mismatch).
    _verify_step_addresses(step_ln, {
        "X_hbm":     addr_plan["X_hbm"],
        "SCALE_hbm": addr_plan["LN_SCALE_hbm"],
        "BIAS_hbm":  addr_plan["LN_BIAS_hbm"],
        "Y_hbm":     addr_plan["LN_Y_hbm"],
    })
    _verify_step_addresses(step_mod, {
        "X_hbm":       addr_plan["MOD_X_hbm"],
        "SCALE1P_hbm": addr_plan["MOD_SCALE1P_hbm"],
        "SHIFT_hbm":   addr_plan["MOD_SHIFT_hbm"],
        "Y_hbm":       addr_plan["MOD_Y_hbm"],
    })
    for st, (_n, _nb, a_key, w_key, bias_key, y_key) in zip(linear_steps, linear_specs):
        _verify_step_addresses(st, {
            "A_hbm":    addr_plan[a_key],
            "B_hbm":    addr_plan[w_key],
            "BIAS_hbm": addr_plan[bias_key],
            "C_hbm":    addr_plan[y_key],
        })
    for st, (_n, x_key, scale_key, y_key) in zip(qknorm_steps, qknorm_specs):
        _verify_step_addresses(st, {
            "X_hbm":     addr_plan[x_key],
            "SCALE_hbm": addr_plan[scale_key],
            "Y_hbm":     addr_plan[y_key],
        })
    _verify_step_addresses(step_rope_q, {
        "XQ_hbm":      addr_plan["ROPE_Q_X_hbm"],
        "COS_hbm":     addr_plan["ROPE_COS_hbm"],
        "SIN_hbm":     addr_plan["ROPE_SIN_hbm"],
        "NEG_SIN_hbm": addr_plan["ROPE_NEG_SIN_hbm"],
        "Q_OUT_hbm":   addr_plan["ROPE_Q_Y_hbm"],
    })
    _verify_step_addresses(step_rope_k, {
        "XQ_hbm":      addr_plan["ROPE_K_X_hbm"],
        "COS_hbm":     addr_plan["ROPE_COS_hbm"],
        "SIN_hbm":     addr_plan["ROPE_SIN_hbm"],
        "NEG_SIN_hbm": addr_plan["ROPE_NEG_SIN_hbm"],
        "Q_OUT_hbm":   addr_plan["ROPE_K_Y_hbm"],
    })
    _verify_step_addresses(step_attn, {
        "Q_hbm": addr_plan["ATTN_Q_hbm"],
        "K_hbm": addr_plan["ATTN_K_hbm"],
        "V_hbm": addr_plan["ATTN_V_hbm"],
        "O_hbm": addr_plan["ATTN_O_hbm"],
    })
    _verify_step_addresses(step_gelu, {
        "X_hbm": addr_plan["GELU_X_hbm"],
        "Y_hbm": addr_plan["GELU_OUT_hbm"],
    })
    _verify_step_addresses(step_concat, {
        "A_hbm": addr_plan["CONCAT_A_hbm"],
        "B_hbm": addr_plan["CONCAT_B_hbm"],
        "Y_hbm": addr_plan["CONCAT_hbm"],
    })
    _verify_step_addresses(step_lin2, {
        "A_hbm":    addr_plan["LIN2_A_hbm"],
        "B_hbm":    addr_plan["LIN2_W_hbm"],
        "BIAS_hbm": addr_plan["LIN2_BIAS_hbm"],
        "C_hbm":    addr_plan["LIN2_OUT_hbm"],
    })
    _verify_step_addresses(step_gate, {
        "X_hbm":    addr_plan["GATE_X_hbm"],
        "GATE_hbm": addr_plan["GATE_G_hbm"],
        "Y_hbm":    addr_plan["GATE_Y_hbm"],
        "OUT_hbm":  addr_plan["BLOCK_OUT_hbm"],
    })

    # ----- 3. concatenate ASM + final output staging -----
    print(f"[3/5] Concatenating ASM ...")
    isa_text = ""
    for step in steps:
        isa_text += (
            f"\n; ============================================================\n"
            f"; >>> STEP: {step.name}\n"
            f"; ============================================================\n"
        )
        isa_text += step.compiled.isa_text

    # Append the same "HBM -> VRAM[0..]" staging block that single-kernel
    # testbenches' --stage-output flag emits. view_mem.py compares
    # vram_dump.bin against golden_result.txt; without staging, the
    # final output sits in HBM and view_mem sees stale VRAM.
    #
    # Chain end is residual_gate: out = x + gate * linear2_out — the
    # full SingleStreamBlock (Open-Sora). We stage and verify that
    # step's OUT_hbm.
    #
    # DIAGNOSTIC MODE: set SSB_VERIFY=<step name> to truncate
    # verification at an intermediate kernel instead of the full block
    # — pinpoints which step introduces error. The staged buffer and
    # golden both switch to that step. Step names: layernorm, modulate,
    # linear_{q,k,v,mlp}, qknorm_{q,k}, rope_{q,k}, flash_attention,
    # gelu, linear2, residual_gate (default).
    from tilelang_tvm_compiler.__main__ import _emit_output_staging

    io = build_inputs_and_golden(layout, seed=0)

    _verify_name = os.environ.get("SSB_VERIFY", "residual_gate").strip()
    _steps_by_name = {s.name: s for s in steps}
    if _verify_name not in _steps_by_name:
        raise ValueError(
            f"SSB_VERIFY={_verify_name!r} not a step name. "
            f"Valid: {sorted(_steps_by_name)}"
        )
    verify_step = _steps_by_name[_verify_name]
    if _verify_name not in io["intermediates"]:
        raise ValueError(
            f"no golden intermediate for step {_verify_name!r}; "
            f"have: {sorted(io['intermediates'])}"
        )
    _verify_golden, _verify_buf = io["intermediates"][_verify_name]
    # Override golden + the buffer we stage to match the truncated step.
    io["golden_flat"] = _verify_golden
    if _verify_name != "residual_gate":
        print(f"  *** DIAGNOSTIC: verifying truncated chain at "
              f"'{_verify_name}' (stage {verify_step.name}.{_verify_buf})")

    staging_isa = _emit_output_staging(
        verify_step.compiled,
        PlenaTarget(mlen=MLEN, btmm_hlen=HLEN),
        _verify_buf,
    )
    isa_text = isa_text.rstrip() + staging_isa
    print(f"      total ISA: {isa_text.count(chr(10))} lines  "
          f"(incl. final stage_output for {verify_step.name}.{_verify_buf})")

    # ----- 4. merge FP preload + build inputs + golden -----
    print("[4/5] Merging FP preload + building inputs + golden ...")
    fp_preload = merge_fp_preload(steps)
    print(f"      fp_preload: shape={tuple(fp_preload.shape)}, "
          f"nonzero={int((fp_preload != 0).sum())}")

    print(f"      hbm_inputs (staging order): "
          f"{', '.join(io['hbm_inputs'].keys())}")
    print(f"      golden_flat: {tuple(io['golden_flat'].shape)}")

    # Per-step HLIR buffer dump — lets us inspect every buffer's scope /
    # address / constant_value without running the emulator. Written
    # below to build/hlir_buffers.json.
    hlir_dump = {}
    for step in steps:
        bufs = {}
        for bname, buf in step.compiled.hlir.buffers.items():
            bufs[bname] = {
                "scope": buf.scope,
                "shape": list(buf.shape),
                "dtype": buf.dtype,
                "address": buf.address,
                "constant_value": buf.constant_value,
                "is_pinned_global": buf.is_pinned_global,
            }
        hlir_dump[step.name] = {
            "fpram_const_count": step.fpram_const_count,
            "buffers": bufs,
        }

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
        data_size=256, mode="behave_sim", asm="tvm_single_stream_block",
        data=None, specified_data_order=list(input_feed),
        build_path=build_dir,
    )

    # HLIR buffer dump — debugging aid for the chained build.
    (build_dir / "hlir_buffers.json").write_text(json.dumps(hlir_dump, indent=2))
    print(f"      wrote hlir_buffers.json "
          f"({sum(len(s['buffers']) for s in hlir_dump.values())} buffers "
          f"across {len(hlir_dump)} steps)")

    # comparison_params — drives view_mem's compare. Shape follows the
    # staged buffer, which is the (truncated) verify_step's output:
    # (M, golden_cols). For the full block golden_cols == HIDDEN_SIZE;
    # under SSB_VERIFY it may differ (e.g. MLP_HIDDEN_DIM for gelu).
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
    (build_dir / "tvm_single_stream_block_generated_asm_code.asm").write_text(isa_text)

    print()
    print("=" * 60)
    print(f"final output: BLOCK_OUT_hbm @ byte {addr_plan['BLOCK_OUT_hbm']}  "
          f"shape ({final_rows}, {final_cols})  (staged to VRAM[0..])")
    print("build/ ready. Next:")
    print("  just build-emulator-debug tvm_single_stream_block")
    print("=" * 60)
    return 0


def _verify_step_addresses(step: Step, expected: dict[str, int]) -> None:
    """Assert each named HBM buffer landed at the planned address."""
    for name, want in expected.items():
        buf = step.compiled.hlir.buffers.get(name)
        if buf is None:
            raise RuntimeError(
                f"{step.name}: expected buffer {name!r} not in HLIR. "
                f"Have: {sorted(step.compiled.hlir.buffers.keys())}"
            )
        got = int(buf.address)
        if got != want:
            raise RuntimeError(
                f"{step.name}.{name}: pinned to {want} but landed at {got}. "
                f"AddressAllocationPass override path is broken."
            )


if __name__ == "__main__":
    sys.exit(main())
