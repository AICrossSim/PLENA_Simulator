"""Trace-specialized grouped BF16 MoE prototype, not a vLLM replacement.

Expert tile maps are prepared on the host from a known route trace. Timed kernels
include token gather, both grouped GEMMs, SiLU, and weighted scatter/reduction.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _gather(X, TOK, ROW, OUT, H: tl.constexpr, COUNT: tl.constexpr,
            BLOCK: tl.constexpr):
    r = tl.program_id(0)
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dst = tl.load(ROW + r)
    token = tl.load(TOK + dst)
    v = tl.load(X + token * H + j, j < H, 0)
    tl.store(OUT + dst * H + j, v, j < H)


def gather_rows(x, token_ids, row_indices, out):
    if row_indices.numel():
        _gather[(row_indices.numel(), triton.cdiv(x.shape[1], 256))](
            x, token_ids, row_indices, out, x.shape[1], row_indices.numel(), 256)


@triton.jit
def _grouped_mm(A, W, O, START, EXPERT, MSTART,
                K: tl.constexpr, N: tl.constexpr,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    tid = tl.program_id(0)
    expert = tl.load(EXPERT + tid)
    first = tl.load(START + expert)
    last = tl.load(START + expert + 1)
    mi = tl.load(MSTART + tid) + tl.arange(0, BM)
    row = first + mi
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for kb in range(tl.cdiv(K, BK)):
        k = kb * BK + kk
        a = tl.load(A + row[:, None] * K + k[None, :],
                    (row[:, None] < last) & (k[None, :] < K), 0)
        w = tl.load(W + expert * K * N + k[:, None] * N + col[None, :],
                    (k[:, None] < K) & (col[None, :] < N), 0)
        acc += tl.dot(a, w)
    tl.store(O + row[:, None] * N + col[None, :], acc,
             (row[:, None] < last) & (col[None, :] < N))


def grouped_matmul(a, weights, out, row_starts, tile_experts,
                   tile_mstarts, block_m=16):
    if tile_experts.numel():
        if not (a.is_contiguous() and weights.is_contiguous() and out.is_contiguous()):
            raise ValueError("grouped_matmul requires contiguous [row,K], [E,K,N], [row,N]")
        _grouped_mm[(tile_experts.numel(), triton.cdiv(weights.shape[2], 64))](
            a, weights, out, row_starts, tile_experts, tile_mstarts,
            weights.shape[1], weights.shape[2], block_m, 64, 64,
            num_warps=4, num_stages=3)


@triton.jit
def _silu_rows(GU, ROWS, Z, F: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.load(ROWS + tl.program_id(0))
    j = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    g = tl.load(GU + row * 2 * F + j, j < F, 0).to(tl.float32)
    u = tl.load(GU + row * 2 * F + F + j, j < F, 0).to(tl.float32)
    val = (g / (1.0 + tl.exp(-g))) * u
    tl.store(Z + row * F + j, val, j < F)


def silu_multiply_rows(gu, row_indices, out):
    if row_indices.numel():
        _silu_rows[(row_indices.numel(), triton.cdiv(out.shape[1], 256))](
            gu, row_indices, out, out.shape[1], 256)


@triton.jit
def _combine(Y, INDEX, SCORE, SHARED, OUT, H: tl.constexpr,
             TOPK: tl.constexpr, BT: tl.constexpr, BN: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    slots = tl.arange(0, BT)
    idx = tl.load(INDEX + token * TOPK + slots, slots < TOPK, 0)
    score = tl.load(SCORE + token * TOPK + slots, slots < TOPK, 0)
    values = tl.load(Y + idx[:, None] * H + col[None, :],
                     (slots[:, None] < TOPK) & (col[None, :] < H), 0).to(tl.float32)
    routed = tl.sum(values * score[:, None], axis=0)
    shared = tl.load(SHARED + token * H + col, col < H, 0).to(tl.float32)
    tl.store(OUT + token * H + col, routed + shared, col < H)


def combine_routed(routed_y, pack_indices, scores, shared_y, out):
    _combine[(out.shape[0], triton.cdiv(out.shape[1], 256))](
        routed_y, pack_indices, scores, shared_y, out, out.shape[1],
        pack_indices.shape[1], triton.next_power_of_2(pack_indices.shape[1]),
        256, num_warps=4)


@triton.jit
def _combine_subsets(SHARED, HOT, COLD, OUT, SIZE: tl.constexpr,
                     HAS_HOT: tl.constexpr, HAS_COLD: tl.constexpr,
                     BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(SHARED + i, i < SIZE, 0).to(tl.float32)
    if HAS_HOT:
        value += tl.load(HOT + i, i < SIZE, 0).to(tl.float32)
    if HAS_COLD:
        value += tl.load(COLD + i, i < SIZE, 0).to(tl.float32)
    tl.store(OUT + i, value, i < SIZE)


def combine_subsets(shared_y, hot_y, cold_y, out, has_hot=True, has_cold=True):
    _combine_subsets[(triton.cdiv(out.numel(), 256),)](
        shared_y, hot_y, cold_y, out, out.numel(), has_hot, has_cold, 256)
