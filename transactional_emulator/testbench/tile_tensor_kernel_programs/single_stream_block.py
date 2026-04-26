"""Top-level SingleStreamBlock builder — coordinates HBM layout across sub-kernels.

Each sub-kernel compiles independently with its own TileTensorProgram, but all
share a coordinated HBM address space:

  - Data regions (inputs, outputs, weights, intermediates): planned upfront,
    each kernel receives the correct hbm_addr for its inputs/outputs.
  - Scratch region: starts at a fixed base after all data regions. Every kernel
    reuses the same scratch range since they don't overlap in time.

The final ISA is the concatenation of all sub-kernel ISAs.
"""

from __future__ import annotations

import sys
from math import ceil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

_THIS_DIR = Path(__file__).resolve().parent
_TESTBENCH_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_TESTBENCH_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tile_tensor_program import TileTensorProgram


# ---------------------------------------------------------------------------
# HBM planner — assigns non-overlapping HBM regions for all block data
# ---------------------------------------------------------------------------

class HBMPlanner:
    """Bump-allocates HBM regions for an entire block before any kernel compiles."""

    def __init__(self, mlen: int, real_data_ratio: float = 1.125):
        self.mlen = mlen
        self.real_data_ratio = real_data_ratio
        self._next_addr = 0
        self.regions: Dict[str, dict] = {}

    def alloc(self, name: str, logical_shape: Tuple[int, ...]) -> int:
        """Allocate an HBM region for a tensor. Returns the base address."""
        # physical shape: tiles needed
        phys = _logical_to_physical(logical_shape, self.mlen)
        num_elems = phys[0] * phys[1]
        size = int(num_elems * self.real_data_ratio)
        addr = self._next_addr
        self._next_addr += size
        self.regions[name] = {
            "addr": addr,
            "logical_shape": logical_shape,
            "physical_shape": phys,
            "size": size,
        }
        return addr

    @property
    def scratch_base(self) -> int:
        """Address where scratch region starts (after all planned data)."""
        return self._next_addr


def _logical_to_physical(shape: Tuple[int, ...], mlen: int) -> Tuple[int, int]:
    """Convert a 4D logical shape to (rows, cols) in tile units."""
    if len(shape) == 4:
        b, s, h, d = shape
        rows = b * s
        cols = h * d
    elif len(shape) == 3:
        b, s, d = shape
        rows = b * s
        cols = d
    else:
        rows, cols = shape[0], shape[1]
    tile_rows = ceil(rows / mlen) * mlen
    tile_cols = ceil(cols / mlen) * mlen
    return (tile_rows, tile_cols)


# ---------------------------------------------------------------------------
# Sub-kernel builders — each returns compiled ISA string
# ---------------------------------------------------------------------------

def _make_prog(
    mlen: int, blen: int, scratch_base: int, hidden_size: int,
    fp_mem_start_addr: int = 32,
) -> TileTensorProgram:
    """Create a TileTensorProgram with shared scratch base.

    Args:
        fp_mem_start_addr: Starting FP_MEM address for this kernel's constants.
            All sub-kernels share one FP_MEM, so each must start where the
            previous one ended.  Default 32 (right after the reserved region).
    """
    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=hidden_size if mlen % hidden_size == 0 else mlen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=4096,
        hbm_base_addr=scratch_base,
    )
    # Advance fp_mem allocation pointer so this kernel's constants don't
    # collide with those of earlier kernels.
    if fp_mem_start_addr > 32:
        tm = prog.tensor_manager
        pad = fp_mem_start_addr - tm._next_fp_mem_addr  # how many slots to skip
        tm._fp_mem_values.extend([0.0] * pad)
        tm._next_fp_mem_addr = fp_mem_start_addr
    return prog


def _build_layernorm_isa(
    *, mlen, blen, scratch_base, seq_len, hidden_size,
    x_addr, out_addr, fp_mem_start_addr=32, eps=1e-6,
) -> tuple[str, TileTensorProgram]:
    batch_size = 1
    shape = (batch_size, seq_len, 1, hidden_size)
    prog = _make_prog(mlen, blen, scratch_base, hidden_size, fp_mem_start_addr=fp_mem_start_addr)

    x_in = prog.input("X_IN", shape, hbm_addr=x_addr)
    out_buf = prog.input("OUT", shape, hbm_addr=out_addr)

    recip_hidden = prog.constant(prog._auto_name("recip_hidden"), 1.0 / float(hidden_size))
    eps_scalar = prog.constant(prog._auto_name("eps"), float(eps))
    prog._fp_constant_end = prog.tensor_manager._next_fp_mem_addr

    x = prog.tensor("X", shape)
    y = prog.tensor("Y", shape)
    centered = prog.alloc_fragment(prog._auto_name("CENTERED"), shape)
    sq = prog.alloc_fragment(prog._auto_name("SQ"), shape)
    mean = prog.alloc_fragment(prog._auto_name("MEAN"), (batch_size, 1, seq_len))
    var = prog.alloc_fragment(prog._auto_name("VAR"), (batch_size, 1, seq_len))
    inv_std = prog.alloc_fragment(prog._auto_name("INV_STD"), (batch_size, 1, seq_len))

    prog.copy(x_in, x)
    prog.copy(x, centered)

    col_tiles = (hidden_size + mlen - 1) // mlen

    # Allocate a reusable partial-sum vector for multi-column reduce
    if col_tiles > 1:
        partial = prog.alloc_fragment(prog._auto_name("PARTIAL"), (batch_size, 1, seq_len))

    # ---- Mean: reduce_sum across all column tiles ----
    prog.fill(mean[0, 0, :], 0.0)
    for j in range(col_tiles):
        cs, ce = j * mlen, min((j + 1) * mlen, hidden_size)
        x_col = centered[0, :, 0:1, cs:ce]
        if col_tiles == 1:
            prog.row_op(x_col, op="reduce_sum", out=mean[0, 0, :], dim=-1)
        else:
            prog.fill(partial[0, 0, :], 0.0)
            prog.row_op(x_col, op="reduce_sum", out=partial[0, 0, :], dim=-1)
            prog.pure_fp_compute(mean[0, 0, :], mean[0, 0, :], src2=partial[0, 0, :],
                                 control="add", task_id=f"ln.mean_accum_{j}")
    prog.row_op(mean[0, 0, :], recip_hidden, "mul", dim=-1)

    # ---- Subtract mean from each column tile ----
    for j in range(col_tiles):
        cs, ce = j * mlen, min((j + 1) * mlen, hidden_size)
        prog.row_op(centered[0, :, 0:1, cs:ce], mean[0, 0, :], "sub", dim=-1)

    # ---- Variance: reduce_sum of squared centered values ----
    prog.atomic_mul(centered, centered, sq)
    prog.fill(var[0, 0, :], 0.0)
    for j in range(col_tiles):
        cs, ce = j * mlen, min((j + 1) * mlen, hidden_size)
        sq_col = sq[0, :, 0:1, cs:ce]
        if col_tiles == 1:
            prog.row_op(sq_col, op="reduce_sum", out=var[0, 0, :], dim=-1)
        else:
            prog.fill(partial[0, 0, :], 0.0)
            prog.row_op(sq_col, op="reduce_sum", out=partial[0, 0, :], dim=-1)
            prog.pure_fp_compute(var[0, 0, :], var[0, 0, :], src2=partial[0, 0, :],
                                 control="add", task_id=f"ln.var_accum_{j}")

    prog.row_op(var[0, 0, :], recip_hidden, "mul", dim=-1)
    prog.row_op(var[0, 0, :], eps_scalar, "add", dim=-1)
    prog.fp_sqrt(var[0, 0, :], inv_std[0, 0, :])
    prog.fp_reci(inv_std[0, 0, :], inv_std[0, 0, :])

    # ---- Normalize: multiply each column tile by inv_std ----
    for j in range(col_tiles):
        cs, ce = j * mlen, min((j + 1) * mlen, hidden_size)
        prog.row_op(centered[0, :, 0:1, cs:ce], inv_std[0, 0, :], "mul", dim=-1)

    prog.copy(centered, y)
    prog.copy(y, out_buf)
    return prog.compile(), prog


def _build_modulate_isa(
    *, mlen, blen, scratch_base, seq_len, hidden_size,
    x_addr, scale_addr, shift_addr, out_addr, fp_mem_start_addr=32,
) -> tuple[str, TileTensorProgram]:
    batch_size = 1
    shape = (batch_size, seq_len, 1, hidden_size)
    prog = _make_prog(mlen, blen, scratch_base, hidden_size, fp_mem_start_addr=fp_mem_start_addr)

    x_in = prog.input("X_IN", shape, hbm_addr=x_addr)
    scale_in = prog.input("SCALE_IN", shape, hbm_addr=scale_addr)
    shift_in = prog.input("SHIFT_IN", shape, hbm_addr=shift_addr)
    out_buf = prog.input("OUT", shape, hbm_addr=out_addr)

    x = prog.tensor("X", shape)
    scale = prog.tensor("SCALE", shape)
    shift = prog.tensor("SHIFT", shape)
    out = prog.tensor("OUT_T", shape)
    scale_plus_one = prog.alloc_fragment(prog._auto_name("SP1"), shape)
    tmp = prog.alloc_fragment(prog._auto_name("TMP"), shape)

    one = prog.constant(prog._auto_name("one"), 1.0)
    prog._fp_constant_end = prog.tensor_manager._next_fp_mem_addr

    prog.copy(x_in, x)
    prog.copy(scale_in, scale)
    prog.copy(shift_in, shift)
    prog.copy(scale, scale_plus_one)
    prog.row_op(scale_plus_one[0, :, 0:1, :], one, "add", dim=-1)
    prog.atomic_mul(scale_plus_one, x, tmp)
    prog.atomic_add(tmp, shift, tmp)
    prog.copy(tmp, out)
    prog.copy(out, out_buf)
    return prog.compile(), prog


def _build_linear_isa(
    *, mlen, blen, scratch_base, seq_len, in_features, out_features,
    x_addr, w_addr, out_addr, bias_addr=None, fp_mem_start_addr=32,
) -> tuple[str, TileTensorProgram]:
    batch_size = 1
    prog = _make_prog(mlen, blen, scratch_base, in_features, fp_mem_start_addr=fp_mem_start_addr)

    x_in = prog.input("X_IN", (batch_size, seq_len, 1, in_features), hbm_addr=x_addr)
    w_in = prog.input("W_IN", (batch_size, in_features, 1, out_features), hbm_addr=w_addr)
    if bias_addr is not None:
        bias_in = prog.input("BIAS_IN", (batch_size, seq_len, 1, out_features), hbm_addr=bias_addr)
    out_buf = prog.input("OUT", (batch_size, seq_len, 1, out_features), hbm_addr=out_addr)

    x = prog.tensor("X", (batch_size, seq_len, 1, in_features))
    w = prog.tensor("W", (batch_size, in_features, 1, out_features))
    if bias_addr is not None:
        bias = prog.tensor("BIAS", (batch_size, seq_len, 1, out_features))
    y = prog.tensor("Y", (batch_size, seq_len, 1, out_features))

    prog.copy(x_in, x)
    prog.copy(w_in, w)

    y_group = prog.alloc_fragment(prog._auto_name("Y_GROUP"), (batch_size, seq_len, 1, out_features))
    prog.matmul(x, w, y_group)

    if bias_addr is not None:
        prog.copy(bias_in, bias)
        prog.atomic_add(y_group, bias, y_group)

    prog.copy(y_group, y)
    prog.copy(y, out_buf)
    return prog.compile(), prog


def _build_gelu_isa(
    *, mlen, blen, scratch_base, seq_len, hidden_size,
    x_addr, out_addr, fp_mem_start_addr=32,
) -> tuple[str, TileTensorProgram]:
    """GELU with sigmoid approximation: x * sigmoid(1.702 * x)."""
    batch_size = 1
    shape = (batch_size, seq_len, 1, hidden_size)
    prog = _make_prog(mlen, blen, scratch_base, hidden_size, fp_mem_start_addr=fp_mem_start_addr)

    x_in = prog.input("X_IN", shape, hbm_addr=x_addr)
    out_buf = prog.input("OUT", shape, hbm_addr=out_addr)

    x = prog.tensor("X", shape)
    y = prog.tensor("Y", shape)
    scaled = prog.alloc_fragment(prog._auto_name("SCALED"), shape)

    scale_const = prog.constant(prog._auto_name("scale"), 1.702)
    one_const = prog.constant(prog._auto_name("one"), 1.0)
    neg_one_const = prog.constant(prog._auto_name("neg_one"), -1.0)
    prog._fp_constant_end = prog.tensor_manager._next_fp_mem_addr

    prog.copy(x_in, x)
    prog.copy(x, scaled)
    prog.row_op(scaled, scale_const, "mul", dim=-1)
    prog.row_op(scaled, neg_one_const, "mul", dim=-1)
    prog.row_op(scaled, op="exp", dim=-1)
    prog.row_op(scaled, one_const, "add", dim=-1)
    prog.row_op(scaled, op="reci", dim=-1)
    prog.atomic_mul(x, scaled, y)

    prog.copy(y, out_buf)
    return prog.compile(), prog


def _build_rmsnorm_isa(
    *, mlen, blen, scratch_base, seq_len, head_count, hidden_size,
    x_addr, scale_addr, out_addr, fp_mem_start_addr=32, eps=1e-6,
) -> tuple[str, TileTensorProgram]:
    """RMSNorm for QKNorm — operates on [B, S, H, hidden_size]."""
    batch_size = 1
    shape = (batch_size, seq_len, head_count, hidden_size)
    prog = _make_prog(mlen, blen, scratch_base, hidden_size, fp_mem_start_addr=fp_mem_start_addr)

    x_in = prog.input("X_IN", shape, hbm_addr=x_addr)
    scale_in = prog.input("SCALE_IN", shape, hbm_addr=scale_addr)
    out_buf = prog.input("OUT", shape, hbm_addr=out_addr)

    recip_hidden = prog.constant(prog._auto_name("recip_hidden"), 1.0 / float(hidden_size))
    eps_scalar = prog.constant(prog._auto_name("eps"), float(eps))
    prog._fp_constant_end = prog.tensor_manager._next_fp_mem_addr

    x = prog.tensor("X", shape)
    scale = prog.tensor("SCALE", shape)
    y = prog.tensor("Y", shape)
    work = prog.alloc_fragment(prog._auto_name("WORK"), shape)
    sq = prog.alloc_fragment(prog._auto_name("SQ"), shape)
    row_sq = prog.alloc_fragment(prog._auto_name("ROW_SQ"), (batch_size, 1, seq_len))
    inv_rms = prog.alloc_fragment(prog._auto_name("INV_RMS"), (batch_size, 1, seq_len))

    prog.copy(x_in, x)
    prog.copy(scale_in, scale)
    prog.copy(x, work)

    prog.atomic_mul(work, work, sq)
    for head_index in range(head_count):
        work_head = work[0, :, head_index:head_index + 1, :]
        sq_head = sq[0, :, head_index:head_index + 1, :]
        scale_head = scale[0, :, head_index:head_index + 1, :]
        prog.fill(row_sq[0, 0, :], 0.0)
        prog.row_op(sq_head, op="reduce_sum", out=row_sq[0, 0, :], dim=-1)
        prog.row_op(row_sq[0, 0, :], recip_hidden, "mul", dim=-1)
        prog.row_op(row_sq[0, 0, :], eps_scalar, "add", dim=-1)
        prog.fp_sqrt(row_sq[0, 0, :], inv_rms[0, 0, :])
        prog.fp_reci(inv_rms[0, 0, :], inv_rms[0, 0, :])
        prog.row_op(work_head, inv_rms[0, 0, :], "mul", dim=-1)
        prog.atomic_mul(work_head, scale_head, work_head)

    prog.copy(work, y)
    prog.copy(y, out_buf)
    return prog.compile(), prog


def _build_rope_isa(
    *, mlen, blen, scratch_base, seq_len, head_count, head_dim,
    q_addr, k_addr, cos_addr, sin_addr, neg_sin_addr,
    q_out_addr, k_out_addr, fp_mem_start_addr=32,
) -> tuple[str, TileTensorProgram]:
    """RoPE — operates on [B, S, H, D] layout."""
    batch_size = 1
    full_dim = head_dim
    shape = (batch_size, seq_len, head_count, full_dim)
    half_dim = full_dim // 2
    parallel_cache_slack = 1024

    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=mlen // blen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=max(4096, int(fp_mem_start_addr) + parallel_cache_slack),
        hbm_base_addr=scratch_base,
    )
    if fp_mem_start_addr > 32:
        tm = prog.tensor_manager
        pad = fp_mem_start_addr - tm._next_fp_mem_addr
        tm._fp_mem_values.extend([0.0] * pad)
        tm._next_fp_mem_addr = fp_mem_start_addr

    xq_in = prog.input("XQ_IN", shape, hbm_addr=q_addr)
    xk_in = prog.input("XK_IN", shape, hbm_addr=k_addr)
    cos_in = prog.input("COS_IN", shape, hbm_addr=cos_addr)
    sin_in = prog.input("SIN_IN", shape, hbm_addr=sin_addr)
    neg_sin_in = prog.input("NEG_SIN_IN", shape, hbm_addr=neg_sin_addr)
    q_out_buf = prog.input("Q_OUT", shape, hbm_addr=q_out_addr)
    k_out_buf = prog.input("K_OUT", shape, hbm_addr=k_out_addr)

    xq = prog.tensor("XQ", shape)
    xk = prog.tensor("XK", shape)
    cos_t = prog.tensor("COS", shape)
    sin_t = prog.tensor("SIN", shape)
    neg_sin_t = prog.tensor("NEG_SIN", shape)
    q_out = prog.tensor("Q", shape)
    k_out = prog.tensor("K", shape)

    prog.copy(xq_in, xq)
    prog.copy(xk_in, xk)
    prog.copy(cos_in, cos_t)
    prog.copy(sin_in, sin_t)
    prog.copy(neg_sin_in, neg_sin_t)

    with prog.parallel_region3d((seq_len, head_count, full_dim), name="rope_q") as (s, h, d):
        q_out[0, s, h, d] = prog.if_then_else(
            d % 2 == 0,
            xq[0, s, h, d] * cos_t[0, s, h, d] + xq[0, s, h, prog.pair(d)] * neg_sin_t[0, s, h, d],
            xq[0, s, h, prog.pair(d)] * sin_t[0, s, h, d] + xq[0, s, h, d] * cos_t[0, s, h, d],
        )

    with prog.parallel_region3d((seq_len, head_count, full_dim), name="rope_k") as (s, h, d):
        k_out[0, s, h, d] = prog.if_then_else(
            d % 2 == 0,
            xk[0, s, h, d] * cos_t[0, s, h, d] + xk[0, s, h, prog.pair(d)] * neg_sin_t[0, s, h, d],
            xk[0, s, h, prog.pair(d)] * sin_t[0, s, h, d] + xk[0, s, h, d] * cos_t[0, s, h, d],
        )

    prog.copy(q_out, q_out_buf)
    prog.copy(k_out, k_out_buf)
    return prog.compile(), prog


def _build_attention_isa(
    *, mlen, blen, scratch_base, seq_len, head_count, head_dim,
    q_addr, k_addr, v_addr, out_addr, causal=False, fp_mem_start_addr=32,
) -> tuple[str, TileTensorProgram]:
    """FlashAttention — [B, S, H, D] layout."""
    import math
    batch_size = 1
    hlen = head_dim
    group_heads = min(4, head_count)
    if head_count % group_heads != 0:
        group_heads = 1

    shape = (batch_size, seq_len, head_count, hlen)

    prog = TileTensorProgram(
        mlen=mlen,
        blen=blen,
        btmm_hlen=hlen,
        real_data_ratio=1.125,
        vram_tile_capacity=16,
        mram_tile_capacity=4,
        fpram_capacity=4096,
        hbm_base_addr=scratch_base,
    )
    if fp_mem_start_addr > 32:
        tm = prog.tensor_manager
        pad = fp_mem_start_addr - tm._next_fp_mem_addr
        tm._fp_mem_values.extend([0.0] * pad)
        tm._next_fp_mem_addr = fp_mem_start_addr

    q_in = prog.input("Q_IN", shape, hbm_addr=q_addr)
    k_in = prog.input("K_IN", shape, hbm_addr=k_addr)
    v_in = prog.input("V_IN", shape, hbm_addr=v_addr)
    out_buf = prog.input("OUT", shape, hbm_addr=out_addr)

    q = prog.tensor("Q", shape)
    k = prog.tensor("K", shape)
    v = prog.tensor("V", shape)
    o = prog.tensor("O", shape)
    prog.copy(q_in, q)
    prog.copy(k_in, k)
    prog.copy(v_in, v)

    scale_scalar = prog.constant(prog._auto_name("flash_scale"), 1.0 / math.sqrt(hlen))
    neg_inf_scalar = prog.constant(prog._auto_name("neg_inf"), -1.0e4)
    zero_scalar = prog.constant(prog._auto_name("zero"), 0.0)
    prog._fp_constant_end = prog.tensor_manager._next_fp_mem_addr
    scores_max = prog.alloc_fragment(prog._auto_name("scores_max"), (batch_size, group_heads, mlen))
    logsum = prog.alloc_fragment(prog._auto_name("logsum"), (batch_size, group_heads, mlen))
    scores_max_prev = prog.alloc_fragment(prog._auto_name("scores_max_prev"), (batch_size, 1, mlen))
    scores_scale = prog.alloc_fragment(prog._auto_name("scores_scale"), (batch_size, 1, mlen))
    scores_sum = prog.alloc_fragment(prog._auto_name("scores_sum"), (batch_size, 1, mlen))
    inv_l = prog.alloc_fragment(prog._auto_name("inv_l"), (batch_size, 1, mlen))
    mask_head = prog.alloc_fragment(prog._auto_name("mask_head"), (batch_size, mlen, 1, mlen))

    if causal:
        with prog.parallel_region3d((mlen, 1, mlen), name="causal_mask") as (q_local, head_local, k_local):
            mask_head[0, q_local, head_local, k_local] = prog.if_then_else(
                q_local < k_local, neg_inf_scalar, zero_scalar,
            )

    q_block_count = seq_len // mlen
    group_block_count = head_count // group_heads
    for q_block in prog.pipelined(q_block_count, num_stages=2):
        q_start = q_block * mlen
        q_end = q_start + mlen
        for group_block in prog.parallel(group_block_count):
            group_start = group_block * group_heads
            q_group = prog.alloc_fragment(prog._auto_name("Q_GROUP"), (batch_size, mlen, group_heads, hlen))
            score_group = prog.alloc_fragment(prog._auto_name("S_GROUP"), (batch_size, mlen, group_heads, mlen))
            out_group = prog.alloc_fragment(prog._auto_name("O_GROUP"), (batch_size, mlen, group_heads, hlen))
            pv_group = prog.alloc_fragment(prog._auto_name("PV_GROUP"), (batch_size, mlen, group_heads, hlen))

            prog.copy(q[0, q_start:q_end, group_start:group_start + group_heads, :], q_group)
            prog.clear(out_group)
            for local_head in prog.parallel(group_heads):
                prog.fill(logsum[0, local_head, :], 0.0)
                prog.fill(scores_max[0, local_head, :], neg_inf_scalar)

            for kv_block in prog.pipelined(q_block_count, num_stages=2):
                kv_start = kv_block * mlen
                kv_end = kv_start + mlen
                if causal and kv_start >= q_end:
                    continue

                k_group = prog.alloc_fragment(prog._auto_name("K_GROUP"), (batch_size, mlen, group_heads, hlen))
                v_group = prog.alloc_fragment(prog._auto_name("V_GROUP"), (batch_size, mlen, group_heads, hlen))
                prog.copy(k[0, kv_start:kv_end, group_start:group_start + group_heads, :], k_group)
                prog.copy(v[0, kv_start:kv_end, group_start:group_start + group_heads, :], v_group)

                prog.matmul(q_group, k_group.T, score_group)
                for local_head in prog.parallel(group_heads):
                    score_head = score_group[0, :, local_head:local_head + 1, :]
                    out_head = out_group[0, :, local_head:local_head + 1, :]
                    prog.row_op(score_head, scale_scalar, "mul", dim=-1)

                    if causal and kv_start < q_end and q_start < kv_end:
                        prog.atomic_add(score_head, mask_head, score_head)

                    prog.copy(scores_max[0, local_head, :], scores_max_prev[0, 0, :])
                    prog.fill(scores_max[0, local_head, :], neg_inf_scalar)
                    prog.row_op(score_head, op="reduce_max", out=scores_max[0, local_head, :], dim=-1)
                    prog.pure_fp_compute(
                        scores_max[0, local_head, :], scores_max[0, local_head, :],
                        src2=scores_max_prev[0, 0, :], control="max",
                        task_id=f"flash.max.q{q_start}.k{kv_start}.h{local_head}",
                    )
                    prog.copy(scores_max_prev[0, 0, :], scores_scale[0, 0, :])
                    prog.pure_fp_compute(
                        scores_scale[0, 0, :], scores_scale[0, 0, :],
                        src2=scores_max[0, local_head, :], control="sub",
                        task_id=f"flash.sub.q{q_start}.k{kv_start}.h{local_head}",
                    )
                    prog.pure_fp_compute(
                        scores_scale[0, 0, :], scores_scale[0, 0, :], control="exp",
                        task_id=f"flash.exp.q{q_start}.k{kv_start}.h{local_head}",
                    )
                    prog.row_op(score_head, scores_max[0, local_head, :], "sub", dim=-1)
                    prog.row_op(score_head, op="exp", dim=-1)
                    prog.fill(scores_sum[0, 0, :], 0.0)
                    prog.row_op(score_head, op="reduce_sum", out=scores_sum[0, 0, :], dim=-1)
                    prog.pure_fp_compute(
                        logsum[0, local_head, :], logsum[0, local_head, :],
                        src2=scores_scale[0, 0, :], control="mul",
                        task_id=f"flash.lscale.q{q_start}.k{kv_start}.h{local_head}",
                    )
                    prog.pure_fp_compute(
                        logsum[0, local_head, :], logsum[0, local_head, :],
                        src2=scores_sum[0, 0, :], control="add",
                        task_id=f"flash.ladd.q{q_start}.k{kv_start}.h{local_head}",
                    )
                    prog.row_op(out_head, scores_scale[0, 0, :], "mul", dim=-1)

                prog.matmul(score_group, v_group, pv_group)
                prog.atomic_add(out_group, pv_group, out_group)

            for local_head in prog.parallel(group_heads):
                out_head = out_group[0, :, local_head:local_head + 1, :]
                prog.pure_fp_compute(
                    logsum[0, local_head, :], inv_l[0, 0, :], control="reci",
                    task_id=f"flash.inv_l.q{q_start}.h{local_head}",
                )
                prog.row_op(out_head, inv_l[0, 0, :], "mul", dim=-1)

            prog.copy(out_group, o[0, q_start:q_end, group_start:group_start + group_heads, :])

    prog.copy(o, out_buf)
    return prog.compile(), prog


def _build_split_isa(
    *, mlen, blen, scratch_base, seq_len, total_cols,
    split_sizes, src_addr, dst_addrs, fp_mem_start_addr=32,
) -> tuple[str, TileTensorProgram]:
    """Split a [1, S, 1, total_cols] tensor along the last dim into separate HBM regions."""
    batch_size = 1
    prog = _make_prog(mlen, blen, scratch_base, total_cols, fp_mem_start_addr=fp_mem_start_addr)

    src_shape = (batch_size, seq_len, 1, total_cols)
    src_in = prog.input("SRC_IN", src_shape, hbm_addr=src_addr)
    src = prog.tensor("SRC", src_shape)
    prog.copy(src_in, src)

    col_offset = 0
    for i, width in enumerate(split_sizes):
        dst_shape = (batch_size, seq_len, 1, width)
        out_buf = prog.input(f"DST_{i}", dst_shape, hbm_addr=dst_addrs[i])
        dst = prog.tensor(f"DST_T_{i}", dst_shape)
        prog.copy(src[0, :, 0:1, col_offset:col_offset + width], dst)
        prog.copy(dst, out_buf)
        col_offset += width

    return prog.compile(), prog


def _build_cat_isa(
    *, mlen, blen, scratch_base, seq_len, part_widths,
    src_addrs, dst_addr, fp_mem_start_addr=32,
) -> tuple[str, TileTensorProgram]:
    """Cat multiple [1, S, 1, width_i] tensors along the last dim into one HBM region."""
    batch_size = 1
    total_cols = sum(part_widths)
    prog = _make_prog(mlen, blen, scratch_base, total_cols, fp_mem_start_addr=fp_mem_start_addr)

    parts = []
    for i, width in enumerate(part_widths):
        part_shape = (batch_size, seq_len, 1, width)
        part_in = prog.input(f"PART_{i}", part_shape, hbm_addr=src_addrs[i])
        part_t = prog.tensor(f"PART_T_{i}", part_shape)
        prog.copy(part_in, part_t)
        parts.append(part_t)

    out_shape = (batch_size, seq_len, 1, total_cols)
    out_buf = prog.input("OUT", out_shape, hbm_addr=dst_addr)
    dst = prog.tensor("DST", out_shape)

    col_offset = 0
    for i, (part_t, width) in enumerate(zip(parts, part_widths)):
        prog.copy(part_t, dst[0, :, 0:1, col_offset:col_offset + width])
        col_offset += width

    prog.copy(dst, out_buf)
    return prog.compile(), prog


def _build_residual_gate_isa(
    *, mlen, blen, scratch_base, seq_len, hidden_size,
    x_addr, gate_addr, y_addr, out_addr, fp_mem_start_addr=32,
) -> tuple[str, TileTensorProgram]:
    batch_size = 1
    shape = (batch_size, seq_len, 1, hidden_size)
    prog = _make_prog(mlen, blen, scratch_base, hidden_size, fp_mem_start_addr=fp_mem_start_addr)

    x_in = prog.input("X_IN", shape, hbm_addr=x_addr)
    gate_in = prog.input("GATE_IN", shape, hbm_addr=gate_addr)
    y_in = prog.input("Y_IN", shape, hbm_addr=y_addr)
    out_buf = prog.input("OUT", shape, hbm_addr=out_addr)

    x = prog.tensor("X", shape)
    gate = prog.tensor("GATE", shape)
    y = prog.tensor("Y", shape)
    out = prog.tensor("OUT_T", shape)
    gated = prog.alloc_fragment(prog._auto_name("GATED"), shape)

    prog.copy(x_in, x)
    prog.copy(gate_in, gate)
    prog.copy(y_in, y)
    prog.atomic_mul(gate, y, gated)
    prog.atomic_add(x, gated, gated)
    prog.copy(gated, out)
    prog.copy(out, out_buf)
    return prog.compile(), prog


# ---------------------------------------------------------------------------
# Top-level block builder
# ---------------------------------------------------------------------------

def build_single_stream_block(
    *,
    mlen: int,
    blen: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    mlp_hidden_dim: int,
    causal: bool = False,
    eps: float = 1e-6,
) -> dict:
    """Build the full SingleStreamBlock ISA with coordinated HBM layout.

    Returns a dict with:
      - "isa": concatenated ISA string
      - "planner": the HBMPlanner with all region info
      - "sub_isas": list of (name, isa) for each sub-kernel
    """
    batch_size = 1
    head_dim = hidden_size // num_heads
    act_shape = (batch_size, seq_len, 1, hidden_size)
    bshd_shape = (batch_size, seq_len, num_heads, head_dim)
    mlp_shape = (batch_size, seq_len, 1, mlp_hidden_dim)
    linear1_out_dim = 3 * hidden_size + mlp_hidden_dim
    linear1_out_shape = (batch_size, seq_len, 1, linear1_out_dim)
    linear2_in_dim = hidden_size + mlp_hidden_dim
    linear2_in_shape = (batch_size, seq_len, 1, linear2_in_dim)

    # ---- Step 1: Plan HBM layout ----
    planner = HBMPlanner(mlen)

    # Block input / conditioning (from host)
    x_addr = planner.alloc("x", act_shape)                           # block input / residual
    shift_addr = planner.alloc("shift", act_shape)                    # from Modulation (pre-expanded)
    scale_addr = planner.alloc("scale", act_shape)                    # from Modulation (pre-expanded)
    gate_addr = planner.alloc("gate", act_shape)                      # from Modulation (pre-expanded)
    cos_addr = planner.alloc("cos", bshd_shape)                       # RoPE cos (pre-expanded from host)
    sin_addr = planner.alloc("sin", bshd_shape)                       # RoPE sin
    neg_sin_addr = planner.alloc("neg_sin", bshd_shape)               # RoPE -sin

    # Intermediate activations
    ln_out_addr = planner.alloc("ln_out", act_shape)                  # LayerNorm output
    mod_out_addr = planner.alloc("mod_out", act_shape)                # modulate output
    linear1_out_addr = planner.alloc("linear1_out", linear1_out_shape)  # linear1 full output

    # After split from linear1_out (separate regions, need copy from linear1_out)
    q_addr = planner.alloc("q", bshd_shape)                           # q after split+reshape
    k_addr = planner.alloc("k", bshd_shape)                           # k after split+reshape
    v_addr = planner.alloc("v", bshd_shape)                           # v after split+reshape
    mlp_in_addr = planner.alloc("mlp_in", mlp_shape)                  # mlp branch after split

    # After QKNorm
    q_normed_addr = planner.alloc("q_normed", bshd_shape)
    k_normed_addr = planner.alloc("k_normed", bshd_shape)

    # After RoPE
    q_rope_addr = planner.alloc("q_rope", bshd_shape)
    k_rope_addr = planner.alloc("k_rope", bshd_shape)

    # Attention output and MLP output
    attn_out_addr = planner.alloc("attn_out", bshd_shape)             # attention output [B,S,H,D]
    mlp_out_addr = planner.alloc("mlp_out", mlp_shape)                # GELU output

    # linear2 input/output
    linear2_in_addr = planner.alloc("linear2_in", linear2_in_shape)   # cat(attn, mlp)
    linear2_out_addr = planner.alloc("linear2_out", act_shape)        # linear2 output
    block_out_addr = planner.alloc("block_out", act_shape)            # final: x + gate*output

    # Weights
    w_linear1_addr = planner.alloc("w_linear1", (batch_size, hidden_size, 1, linear1_out_dim))
    b_linear1_addr = planner.alloc("b_linear1", (batch_size, seq_len, 1, linear1_out_dim))
    w_linear2_addr = planner.alloc("w_linear2", (batch_size, linear2_in_dim, 1, hidden_size))
    b_linear2_addr = planner.alloc("b_linear2", (batch_size, seq_len, 1, hidden_size))
    # QKNorm scales: expanded to BSHD so the kernel can normalize per head directly.
    q_norm_scale_addr = planner.alloc("q_norm_scale", bshd_shape)
    k_norm_scale_addr = planner.alloc("k_norm_scale", bshd_shape)

    scratch_base = planner.scratch_base

    # ---- Step 2: Build each sub-kernel ISA ----
    # All sub-kernels share one FP_MEM that is loaded once at boot.
    # Only true preload constants participate in the global address chain;
    # kernel-local scratch must not shift the next kernel's constant base.
    common = dict(mlen=mlen, blen=blen, scratch_base=scratch_base)
    sub_isas: List[Tuple[str, str]] = []
    sub_progs: List[TileTensorProgram] = []  # keep progs to merge fp_preload
    fp_mem_next = 32  # first user-allocatable FP_MEM address

    def _append(name: str, result: tuple[str, TileTensorProgram]) -> None:
        nonlocal fp_mem_next
        isa, prog = result
        sub_isas.append((name, isa))
        sub_progs.append(prog)
        fp_mem_next = int(getattr(prog, "_fp_constant_end", prog.tensor_manager._next_fp_mem_addr))

    # 1. LayerNorm(x) → ln_out
    _append("layernorm", _build_layernorm_isa(
        **common, seq_len=seq_len, hidden_size=hidden_size,
        x_addr=x_addr, out_addr=ln_out_addr, eps=eps,
        fp_mem_start_addr=fp_mem_next,
    ))

    # 2. adaLN modulate: (1 + scale) * ln_out + shift → mod_out
    _append("modulate", _build_modulate_isa(
        **common, seq_len=seq_len, hidden_size=hidden_size,
        x_addr=ln_out_addr, scale_addr=scale_addr,
        shift_addr=shift_addr, out_addr=mod_out_addr,
        fp_mem_start_addr=fp_mem_next,
    ))

    # 3. linear1: mod_out → linear1_out [B, S, 1, 3H+mlp]
    _append("linear1", _build_linear_isa(
        **common, seq_len=seq_len,
        in_features=hidden_size, out_features=linear1_out_dim,
        x_addr=mod_out_addr, w_addr=w_linear1_addr,
        out_addr=linear1_out_addr, bias_addr=b_linear1_addr,
        fp_mem_start_addr=fp_mem_next,
    ))

    # 4. split linear1_out → q, k, v [B,S,H,D] + mlp_in [B,S,1,mlp]
    #    linear1_out layout: [..., q0..qH | k0..kH | v0..vH | mlp0..mlpN]
    #    Physical memory for [B,S,1,H*D] == [B,S,H,D], so split writes
    #    directly into the q/k/v/mlp_in HBM regions.
    split_sizes = [hidden_size, hidden_size, hidden_size, mlp_hidden_dim]
    _append("split", _build_split_isa(
        **common, seq_len=seq_len, total_cols=linear1_out_dim,
        split_sizes=split_sizes,
        src_addr=linear1_out_addr,
        dst_addrs=[q_addr, k_addr, v_addr, mlp_in_addr],
        fp_mem_start_addr=fp_mem_next,
    ))

    # 5. QKNorm: RMSNorm on q and k per head in native BSHD layout.
    _append("qknorm_q", _build_rmsnorm_isa(
        **common, seq_len=seq_len, head_count=num_heads, hidden_size=head_dim,
        x_addr=q_addr, scale_addr=q_norm_scale_addr,
        out_addr=q_normed_addr,
        fp_mem_start_addr=fp_mem_next,
    ))
    _append("qknorm_k", _build_rmsnorm_isa(
        **common, seq_len=seq_len, head_count=num_heads, hidden_size=head_dim,
        x_addr=k_addr, scale_addr=k_norm_scale_addr,
        out_addr=k_normed_addr,
        fp_mem_start_addr=fp_mem_next,
    ))

    # 6. RoPE: apply rotary position embedding to q_normed, k_normed
    _append("rope", _build_rope_isa(
        **common, seq_len=seq_len, head_count=num_heads, head_dim=head_dim,
        q_addr=q_normed_addr, k_addr=k_normed_addr,
        cos_addr=cos_addr, sin_addr=sin_addr, neg_sin_addr=neg_sin_addr,
        q_out_addr=q_rope_addr, k_out_addr=k_rope_addr,
        fp_mem_start_addr=fp_mem_next,
    ))

    # 7. FlashAttention: q_rope, k_rope, v → attn_out
    _append("attention", _build_attention_isa(
        **common, seq_len=seq_len, head_count=num_heads, head_dim=head_dim,
        q_addr=q_rope_addr, k_addr=k_rope_addr, v_addr=v_addr,
        out_addr=attn_out_addr, causal=causal,
        fp_mem_start_addr=fp_mem_next,
    ))

    # 8. GELU on mlp branch: mlp_in → mlp_out
    _append("gelu", _build_gelu_isa(
        **common, seq_len=seq_len, hidden_size=mlp_hidden_dim,
        x_addr=mlp_in_addr, out_addr=mlp_out_addr,
        fp_mem_start_addr=fp_mem_next,
    ))

    # 9. cat(attn_out, mlp_out) → linear2_in
    #    attn_out [B,S,H,D] in memory == [B,S,1,H*D], so cat directly
    #    with mlp_out [B,S,1,mlp] → [B,S,1,H*D+mlp]
    _append("cat", _build_cat_isa(
        **common, seq_len=seq_len,
        part_widths=[hidden_size, mlp_hidden_dim],
        src_addrs=[attn_out_addr, mlp_out_addr],
        dst_addr=linear2_in_addr,
        fp_mem_start_addr=fp_mem_next,
    ))

    # 10. linear2: linear2_in → linear2_out
    _append("linear2", _build_linear_isa(
        **common, seq_len=seq_len,
        in_features=linear2_in_dim, out_features=hidden_size,
        x_addr=linear2_in_addr, w_addr=w_linear2_addr,
        out_addr=linear2_out_addr, bias_addr=b_linear2_addr,
        fp_mem_start_addr=fp_mem_next,
    ))

    # 11. residual gate: x + gate * linear2_out → block_out
    _append("residual_gate", _build_residual_gate_isa(
        **common, seq_len=seq_len, hidden_size=hidden_size,
        x_addr=x_addr, gate_addr=gate_addr,
        y_addr=linear2_out_addr, out_addr=block_out_addr,
        fp_mem_start_addr=fp_mem_next,
    ))

    # ---- Step 3: Merge FP_MEM and concatenate ISA ----
    # Build a single fp_preload array covering all sub-kernels.
    # Each prog's _fp_mem_values has [0]*32 (reserved) + [0]*pad + [own constants].
    # We overlay each prog's segment onto one unified array.
    unified_fp = [0.0] * fp_mem_next
    for prog in sub_progs:
        values = prog.tensor_manager._fp_mem_values
        value_limit = int(getattr(prog, "_fp_constant_end", len(values)))
        for i in range(32, min(len(values), value_limit)):
            if values[i] != 0.0 or unified_fp[i] == 0.0:
                unified_fp[i] = values[i]
    fp_preload = unified_fp

    full_isa = "\n\n".join(
        f"; ---- {name} ----\n{isa}" for name, isa in sub_isas
    )

    return {
        "isa": full_isa,
        "planner": planner,
        "sub_isas": sub_isas,
        "sub_progs": sub_progs,
        "fp_preload": fp_preload,
    }


def _golden_single_stream_block(
    x, shift, scale, gate, cos, sin,
    w1, b1, w2, b2, q_norm_scale, k_norm_scale,
    num_heads, mlp_hidden_dim, causal=False, eps=1e-6,
):
    """PyTorch golden reference for the full SingleStreamBlock."""
    import math
    import torch.nn.functional as F

    hidden_size = x.shape[-1]
    head_dim = hidden_size // num_heads

    # 1. LayerNorm (no affine)
    ln_out = F.layer_norm(x, [hidden_size], eps=eps)

    # 2. Modulate: (1 + scale) * ln_out + shift
    mod_out = (1.0 + scale) * ln_out + shift

    # 3. Linear1
    linear1_out = mod_out @ w1 + b1

    # 4. Split
    q, k, v, mlp_in = torch.split(
        linear1_out, [hidden_size, hidden_size, hidden_size, mlp_hidden_dim], dim=-1
    )

    # Reshape to [B, S, H, D]
    B, S = q.shape[0], q.shape[1]
    q = q.reshape(B, S, num_heads, head_dim)
    k = k.reshape(B, S, num_heads, head_dim)
    v = v.reshape(B, S, num_heads, head_dim)

    # 5. QKNorm (RMSNorm per head, broadcast scale)
    def _rms_norm(t, sc, eps_):
        # t: [B, S, H, D]  sc: [1, 1, 1, D] broadcast
        rms = torch.sqrt(t.pow(2).mean(dim=-1, keepdim=True) + eps_)
        return t / rms * sc

    q_scale = q_norm_scale if q_norm_scale.dim() == 4 else q_norm_scale.reshape(1, 1, 1, head_dim)
    k_scale = k_norm_scale if k_norm_scale.dim() == 4 else k_norm_scale.reshape(1, 1, 1, head_dim)
    q = _rms_norm(q, q_scale, eps)
    k = _rms_norm(k, k_scale, eps)

    # 6. RoPE (interleaved)
    def _apply_rope(t, cos_, sin_):
        # t: [B,S,H,D]  cos_/sin_: [B,S,H,D]
        t_even = t[..., 0::2]
        t_odd = t[..., 1::2]
        out_even = t_even * cos_[..., 0::2] - t_odd * sin_[..., 0::2]
        out_odd = t_odd * cos_[..., 1::2] + t_even * sin_[..., 1::2]
        out = torch.stack([out_even, out_odd], dim=-1)
        return out.reshape(t.shape)

    q = _apply_rope(q, cos, sin)
    k = _apply_rope(k, cos, sin)

    # 7. Attention
    attn_scale = 1.0 / math.sqrt(head_dim)
    # [B, H, S, D]
    q_t = q.permute(0, 2, 1, 3)
    k_t = k.permute(0, 2, 1, 3)
    v_t = v.permute(0, 2, 1, 3)
    scores = (q_t @ k_t.transpose(-2, -1)) * attn_scale
    if causal:
        causal_mask = torch.triu(
            torch.full((S, S), float("-inf"), dtype=scores.dtype, device=scores.device),
            diagonal=1,
        )
        scores = scores + causal_mask.view(1, 1, S, S)
    attn_w = torch.softmax(scores, dim=-1)
    attn_out = (attn_w @ v_t).permute(0, 2, 1, 3)  # [B, S, H, D]

    # 8. GELU sigmoid approx: x * sigmoid(1.702 * x)
    mlp_out = mlp_in * torch.sigmoid(1.702 * mlp_in)

    # 9. Cat
    attn_flat = attn_out.reshape(B, S, 1, hidden_size)
    linear2_in = torch.cat([attn_flat, mlp_out], dim=-1)

    # 10. Linear2
    linear2_out = linear2_in @ w2 + b2

    # 11. Residual gate
    block_out = x + gate * linear2_out
    return block_out


if __name__ == "__main__":
    import json
    import numpy as np

    from tile_tensor_test_helper import stage_input_tensor_for_stride_compare
    from transactional_emulator.tools.create_sim_env import create_sim_env
    from compiler.sim_env_utils import create_mem_for_sim

    torch.manual_seed(42)
    build_dir = _TESTBENCH_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dimensions ----
    mlen = 64
    blen = 4
    seq_len = 128
    hidden_size = 128
    num_heads = 8
    mlp_hidden_dim = 512
    head_dim = hidden_size // num_heads
    eps = 1e-6
    causal = True
    batch_size = 1
    linear1_out_dim = 3 * hidden_size + mlp_hidden_dim

    # ---- Build block ----
    result = build_single_stream_block(
        mlen=mlen, blen=blen, seq_len=seq_len,
        hidden_size=hidden_size, num_heads=num_heads,
        mlp_hidden_dim=mlp_hidden_dim, causal=causal, eps=eps,
    )
    planner = result["planner"]
    full_isa = result["isa"]
    fp_preload = result["fp_preload"]
    sub_progs = result["sub_progs"]

    # ---- Generate random input data ----
    act_shape = (batch_size, seq_len, 1, hidden_size)
    bshd_shape = (batch_size, seq_len, num_heads, head_dim)
    mlp_shape = (batch_size, seq_len, 1, mlp_hidden_dim)

    x = torch.randn(*act_shape, dtype=torch.float32) * 0.1
    shift = torch.randn(*act_shape, dtype=torch.float32) * 0.01
    scale = torch.randn(*act_shape, dtype=torch.float32) * 0.01
    gate = torch.randn(*act_shape, dtype=torch.float32) * 0.01

    # RoPE tables: [B, S, H, D]
    cos_val = torch.randn(*bshd_shape, dtype=torch.float32) * 0.5
    sin_val = torch.randn(*bshd_shape, dtype=torch.float32) * 0.5
    neg_sin_val = -sin_val

    # Weights
    w1 = torch.randn(batch_size, hidden_size, 1, linear1_out_dim, dtype=torch.float32) * 0.02
    b1 = torch.randn(batch_size, seq_len, 1, linear1_out_dim, dtype=torch.float32) * 0.01
    w2 = torch.randn(batch_size, hidden_size + mlp_hidden_dim, 1, hidden_size, dtype=torch.float32) * 0.02
    b2 = torch.randn(batch_size, seq_len, 1, hidden_size, dtype=torch.float32) * 0.01
    q_norm_scale = torch.ones(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    k_norm_scale = torch.ones(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)

    # ---- Compute golden output ----
    golden = _golden_single_stream_block(
        x, shift, scale, gate, cos_val, sin_val,
        # linear1 weight: [B,H_in,1,H_out] → squeeze for matmul [H_in, H_out]
        w1.squeeze(0).squeeze(1),
        b1,
        w2.squeeze(0).squeeze(1),
        b2,
        q_norm_scale, k_norm_scale,
        num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim, causal=causal, eps=eps,
    )
    golden_flat = golden.reshape(seq_len, hidden_size)

    # ---- Prepare HBM data ----
    # Map planner region names to actual tensors.  Intermediates get zeros.
    region_data = {
        "x": x,
        "shift": shift,
        "scale": scale,
        "gate": gate,
        "cos": cos_val,
        "sin": sin_val,
        "neg_sin": neg_sin_val,
        "w_linear1": w1,
        "b_linear1": b1,
        "w_linear2": w2,
        "b_linear2": b2,
        "q_norm_scale": q_norm_scale,
        "k_norm_scale": k_norm_scale,
    }

    # Build .pt files for every planner region (in allocation order).
    input_feed = {}
    input_order = []
    for region_name, info in planner.regions.items():
        phys_rows, phys_cols = info["physical_shape"]
        logical_shape = info["logical_shape"]
        if region_name in region_data:
            data = region_data[region_name]
            # Reshape logical → physical (pad to tile boundaries)
            if len(logical_shape) == 4:
                b, s, h, d = logical_shape
                flat = data.contiguous().reshape(b * s, h * d)
            else:
                flat = data.contiguous().reshape(-1, logical_shape[-1])
            padded = flat.new_zeros(phys_rows, phys_cols)
            padded[:flat.shape[0], :flat.shape[1]] = flat
        else:
            # Intermediate buffer — zeros
            padded = torch.zeros(phys_rows, phys_cols, dtype=torch.float32)
        feed_name = f"{region_name}.hbm"
        input_feed[feed_name] = padded.contiguous().reshape(1, -1)
        input_order.append(feed_name)

    # ---- Write artifacts ----
    # Get staging ISA from the last prog (residual_gate) for output comparison
    last_prog = sub_progs[-1]  # residual_gate prog
    out_input = last_prog.tensor_manager.inputs["OUT"]
    comparison_params = stage_input_tensor_for_stride_compare(out_input)
    staging_isa = comparison_params.pop("staging_isa", "")

    gen_code = full_isa
    if staging_isa:
        gen_code = gen_code.rstrip() + "\n\n" + staging_isa

    create_sim_env(
        input_tensor=input_feed,
        generated_code=gen_code,
        golden_result={"original_output": golden_flat},
        fp_preload=fp_preload,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="single_stream_block",
        data=None,
        specified_data_order=input_order,
        build_path=build_dir,
    )

    with open(build_dir / "comparison_params.json", "w", encoding="utf-8") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "single_stream_block_generated_asm_code.asm", "w", encoding="utf-8") as f:
        f.write(gen_code)

    np.save(
        build_dir / "single_stream_block_golden_fp32.npy",
        golden_flat.detach().cpu().numpy().astype(np.float32),
    )

    # ---- Summary ----
    print("=" * 70)
    print("SingleStreamBlock testbench built")
    print("=" * 70)
    print(f"HBM regions: {len(planner.regions)}")
    for name, info in planner.regions.items():
        tag = "input" if name in region_data else "intermediate"
        print(f"  {name}: addr={info['addr']}, size={info['size']}  ({tag})")
    print(f"Scratch base: {planner.scratch_base}")
    print(f"Sub-kernels: {[name for name, _ in result['sub_isas']]}")
    print(f"Total ISA lines: {gen_code.count(chr(10))}")
    fp = fp_preload
    non_zero = [(i, v) for i, v in enumerate(fp) if v != 0.0]
    print(f"FP_MEM: {len(fp)} slots, {len(non_zero)} non-zero")
    for addr, val in non_zero:
        print(f"  [{addr}] = {val}")
    print(f"Golden output shape: {golden_flat.shape}")
    print(f"Artifacts written to: {build_dir}")
    print("=" * 70)
