#!/usr/bin/env python3
"""
ref_numerics.py - golden numerical reference for the PLENA-MoE v3 quantized datapath.

Covers
  * MXINT{2,3,4,8} block quantization (block 32 along the reduction axis, shared power-of-two scale)
  * low-rank quantization-error reconstruction: LQER (plain SVD), L2QER-style diagonal
    activation scaling, QERA-approx (diagonal RMS scaling), QERA-exact (full autocorrelation)
  * the rank-lane fused GEMM: per K segment s, the main dot product and the rank terms
    assigned to s are accumulated together (accumulation order is explicit)
  * a SwiGLU expert with compensation applied BEFORE SiLU and optional gate folding
  * gate-weighted runtime rank allocation with a byte-budget (lambda) controller

Conventions: weights are W[N, K] (output rows, reduction) as in compiler.py; y = x @ W.T.
Low-rank factors satisfy  W.T ~= Wq.T + A @ B,  A: (K, r),  B: (r, N),  U = x @ A.
All "hardware" accumulations are float32; references are float64.
This file is a reference for tests, not an optimized implementation.
"""
from __future__ import annotations

import numpy as np

K_SEG = 512
MX_BLOCK = 32


# --------------------------------------------------------------------------- bf16 helpers
def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 (round to nearest even), returned as float32."""
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    rounding = ((u >> 16) & 1) + 0x7FFF
    u = ((u + rounding) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)


# --------------------------------------------------------------------------- MX quantization
def mx_quantize(w: np.ndarray, bits: int, block: int = MX_BLOCK, axis: int = -1):
    """Symmetric MXINT quantization with a shared power-of-two scale per `block` elements along `axis`.

    Codes are integers in [-(2^(bits-1)-1), 2^(bits-1)-1]; scale = 2^e with
    e = ceil(log2(amax / qmax)) (clamped to the E8M0 range), so |w / scale| <= qmax.
    Returns (codes int8, exponents int16 per block, dequantized float32).
    """
    w = np.asarray(w, dtype=np.float32)
    w = np.moveaxis(w, axis, -1)
    n = w.shape[-1]
    pad = (-n) % block
    wp = np.pad(w, [(0, 0)] * (w.ndim - 1) + [(0, pad)])
    blocks = wp.reshape(*wp.shape[:-1], -1, block)
    qmax = 2 ** (bits - 1) - 1
    amax = np.max(np.abs(blocks), axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        e = np.ceil(np.log2(np.where(amax > 0, amax / qmax, 1.0)))
    e = np.clip(e, -127, 127)
    scale = np.exp2(e)
    q = np.clip(np.rint(blocks / scale), -qmax, qmax)
    deq = (q * scale).reshape(wp.shape)[..., :n]
    codes = q.reshape(wp.shape)[..., :n].astype(np.int8)
    return (np.moveaxis(codes, -1, axis), e[..., 0].astype(np.int16),
            np.moveaxis(deq.astype(np.float32), -1, axis))


def mx_tile_bytes(n_rows: int, k: int, bits: int) -> int:
    """Packed bytes of an MXINT tile with n_rows output rows and k reduction elements (32 B aligned)."""
    raw = n_rows * k * bits / 8 + n_rows * -(-k // MX_BLOCK)
    return int(-(-int(np.ceil(raw)) // 32) * 32)


# --------------------------------------------------------------------------- low-rank reconstruction
def _svd_factors(M: np.ndarray, r: int):
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :r], s, Vt[:r, :] * s[:r, None]


def lowrank_factors(W: np.ndarray, Wq: np.ndarray, X: np.ndarray | None, r: int, method: str, eps: float = 1e-6):
    """Return (A, B, singular_values_of_scaled_error) with  W.T ~= Wq.T + A @ B.

    method: 'lqer'          plain SVD of the weight error (Zhang et al., ICML 2024, LQER)
            'l2qer'         diagonal scaling by mean |x| per input channel (LQER's activation-induced scaling)
            'qera_approx'   diagonal scaling by sqrt(E[x^2]) (QERA closed form, uncorrelated inputs)
            'qera_exact'    full autocorrelation R = E[x x^T]: M = R^1/2 E,  A = R^-1/2 U_r
    """
    E = (np.asarray(W, np.float64) - np.asarray(Wq, np.float64)).T          # (K, N)
    K = E.shape[0]
    if r == 0:
        return np.zeros((K, 0)), np.zeros((0, E.shape[1])), np.linalg.svd(E, compute_uv=False)
    if method == "lqer":
        U, s, B = _svd_factors(E, r)
        return U, B, s
    X = np.asarray(X, np.float64)
    if method in ("l2qer", "qera_approx"):
        sc = np.mean(np.abs(X), axis=0) if method == "l2qer" else np.sqrt(np.mean(X * X, axis=0))
        sc = np.maximum(sc, eps * max(sc.max(), eps))
        U, s, B = _svd_factors(sc[:, None] * E, r)
        return U / sc[:, None], B, s
    if method == "qera_exact":
        R = X.T @ X / X.shape[0]
        R += eps * np.trace(R) / K * np.eye(K)
        lam, Q = np.linalg.eigh(R)
        lam = np.maximum(lam, eps * lam.max())
        Rh = (Q * np.sqrt(lam)) @ Q.T
        Rih = (Q / np.sqrt(lam)) @ Q.T
        U, s, B = _svd_factors(Rh @ E, r)
        return Rih @ U, B, s
    raise ValueError(method)


def quantize_factor(F: np.ndarray, fmt: str, red_axis: int) -> np.ndarray:
    """Quantize a factor along its reduction axis: A (K, r) -> axis 0; B (r, N) -> axis 0."""
    F = np.asarray(F, np.float32)
    if fmt == "bf16":
        return bf16_round(F)
    if fmt in ("mxint8", "mxint4"):
        bits = 8 if fmt == "mxint8" else 4
        block = min(MX_BLOCK, F.shape[red_axis])
        return mx_quantize(F, bits, block=block, axis=red_axis)[2]
    if fmt == "fp32":
        return F
    raise ValueError(fmt)


def mxint8_split(F: np.ndarray, red_axis: int = 0, block: int = MX_BLOCK):
    """MXINT8 factor for INT4-only PEs (precision P2): codes clipped to +-119 and split into two INT4 passes.

    q = 16*hi + lo with hi = floor((q + 8) / 16) in [-7, 7] and lo = q - 16*hi in [-8, 7] (two's-complement INT4,
    so the INT4 multiplier must accept -8). Pass 1 uses hi with exponent e+4, pass 2 uses lo with exponent e.
    Returns (hi, lo, e, deq) with deq = (16*hi + lo) * 2^e (exponent broadcast along red_axis blocks).
    """
    F = np.moveaxis(np.asarray(F, np.float32), red_axis, -1)
    n = F.shape[-1]
    block = min(block, n)
    pad = (-n) % block
    Fp = np.pad(F, [(0, 0)] * (F.ndim - 1) + [(0, pad)])
    blocks = Fp.reshape(*Fp.shape[:-1], -1, block)
    qmax = 119
    amax = np.max(np.abs(blocks), axis=-1, keepdims=True)
    with np.errstate(divide="ignore"):
        e = np.clip(np.ceil(np.log2(np.where(amax > 0, amax / qmax, 1.0))), -127, 127)
    q = np.clip(np.rint(blocks / np.exp2(e)), -qmax, qmax)
    hi = np.floor((q + 8) / 16)
    lo = q - 16 * hi
    deq = ((16 * hi + lo) * np.exp2(e)).reshape(Fp.shape)[..., :n]
    unblock = lambda a: np.moveaxis(a.reshape(Fp.shape)[..., :n].astype(np.int8), -1, red_axis)
    return unblock(hi), unblock(lo), e[..., 0].astype(np.int16), np.moveaxis(deq.astype(np.float32), -1, red_axis)


def quantize_b_per_segment(B: np.ndarray, segs, fmt: str) -> np.ndarray:
    """Quantize B (r, N) so that every scale block lies inside ONE segment's rank set R_s.

    A tile only carries the B rows of its own segment, so an MX block may not span ranks of different
    segments: the block along r is R_s itself (|R_s| <= L). BF16 is element-wise and unaffected.
    """
    B = np.asarray(B, np.float32)
    out = np.zeros_like(B)
    for R in segs:
        if R:
            out[R, :] = quantize_factor(B[R, :], fmt, 0)
    return out


# --------------------------------------------------------------------------- rank-lane fused GEMM
def rank_segments(r: int, K: int, L: int, placement: str = "interleave", slack: bool = False):
    """Assign rank indices to K segments. 'interleave': j -> j mod S; 'late': fill from the last segment.

    Raises ValueError if any segment would exceed its lane capacity (L, plus the idle K positions of
    the last segment when slack=True, which is legal only with BF16 PEs, i.e. precision P1).
    """
    S = -(-K // K_SEG)
    caps = [L] * S
    if slack:
        caps[-1] += S * K_SEG - K
    if r > sum(caps):
        raise ValueError(f"rank {r} exceeds lane capacity {sum(caps)} (L={L}, segments={S}, slack={slack})")
    segs = [[] for _ in range(S)]
    if placement == "interleave":
        order = [j % S for j in range(r)]
        for j, s in enumerate(order):
            if len(segs[s]) < caps[s]:
                segs[s].append(j)
            else:   # overflow (slack segment only) goes to the last segment
                segs[-1].append(j)
    elif placement == "late":
        j = 0
        for s in reversed(range(S)):
            while j < r and len(segs[s]) < caps[s]:
                segs[s].append(j); j += 1
    else:
        raise ValueError(placement)
    for s in range(S):
        if len(segs[s]) > caps[s]:
            raise ValueError("lane capacity violated")
    return segs


def fused_rank_lane_gemm(X: np.ndarray, Wq: np.ndarray, A: np.ndarray, B: np.ndarray, L: int,
                         placement: str = "interleave", slack: bool = False, u_bf16: bool = True):
    """Hardware-order reference: acc(fp32) += X[:,seg] @ Wq[:,seg].T + U[:,R_s] @ B[R_s,:] per segment.

    U = X @ A is computed by the pre-pass in fp32 and stored as BF16 (u_bf16=True).
    """
    X = np.asarray(X, np.float32)
    Wq = np.asarray(Wq, np.float32)
    K = X.shape[1]
    r = A.shape[1]
    U = (X @ np.asarray(A, np.float32)).astype(np.float32)
    if u_bf16:
        U = bf16_round(U)
    Bf = np.asarray(B, np.float32)
    segs = rank_segments(r, K, L, placement, slack)
    acc = np.zeros((X.shape[0], Wq.shape[0]), np.float32)
    for s, R in enumerate(segs):
        lo, hi = s * K_SEG, min(K, (s + 1) * K_SEG)
        part = X[:, lo:hi] @ Wq[:, lo:hi].T
        if R:
            part = part + U[:, R] @ Bf[R, :]
        acc = acc + part.astype(np.float32)
    return acc


def reference_compensated(X, Wq, A, B):
    X = np.asarray(X, np.float64)
    return X @ np.asarray(Wq, np.float64).T + (X @ np.asarray(A, np.float64)) @ np.asarray(B, np.float64)


# --------------------------------------------------------------------------- expert FFN
def silu(x):
    return x / (1.0 + np.exp(-x))


def expert_ffn(x, Wg, Wu, Wd, comp=None, gate=None, order="before_silu"):
    """SwiGLU expert. comp = dict(g=(A,B), u=(A,B), d=(A,B)) or None. gate: per-token routing weight.

    order='before_silu' (correct) applies the Gate/Up compensation before SiLU; 'after_silu' is the
    illegal ordering kept only to show that it changes the result.
    """
    x = np.asarray(x, np.float64)
    g = x @ Wg.T
    u = x @ Wu.T
    if comp is not None and order == "before_silu":
        g = g + (x @ comp["g"][0]) @ comp["g"][1]
        u = u + (x @ comp["u"][0]) @ comp["u"][1]
    z = silu(g) * u
    if comp is not None and order == "after_silu":
        z = z + silu((x @ comp["g"][0]) @ comp["g"][1]) * ((x @ comp["u"][0]) @ comp["u"][1])
    if gate is not None:
        z = z * np.asarray(gate, np.float64)[:, None]          # gate folding: g*(Z Wd) == (g*Z) Wd
    y = z @ Wd.T
    if comp is not None:
        y = y + (z @ comp["d"][0]) @ comp["d"][1]
    return y


def rel_err(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


# --------------------------------------------------------------------------- gate-weighted rank allocation
def tail_energy(singular_values: np.ndarray, ranks) -> np.ndarray:
    """T(r) = sum_{i >= r} sigma_i^2 of the activation-scaled error (QERA / L2QER objective)."""
    s2 = np.asarray(singular_values, np.float64) ** 2
    c = np.concatenate([[0.0], np.cumsum(s2)])
    return np.array([s2.sum() - c[min(r, len(s2))] for r in ranks])


def allocate_ranks(w: np.ndarray, T: np.ndarray, rank_bytes: np.ndarray, lam: float, cap_index: np.ndarray):
    """Per expert e choose candidate index i <= cap_index[e] minimizing w_e * T[e, i] + lam * rank_bytes[e, i]."""
    cost = w[:, None] * T + lam * rank_bytes
    idx = np.arange(T.shape[1])[None, :]
    cost = np.where(idx <= cap_index[:, None], cost, np.inf)
    return np.argmin(cost, axis=1)


def lambda_controller(steps, T, rank_bytes, cap_index, budget_bytes, lam0=1.0, eta=0.5):
    """Multiplicative update  lam <- lam * exp(eta * (bytes/budget - 1)) once per step (per layer).

    steps: iterable of (active_expert_indices, w_active). Returns (lam history, bytes history, choices).
    """
    lam, hist_l, hist_b, choices = lam0, [], [], []
    for active, w in steps:
        ch = allocate_ranks(np.asarray(w), T[active], rank_bytes[active], lam, cap_index[active])
        b = float(rank_bytes[active, ch].sum())
        hist_l.append(lam); hist_b.append(b); choices.append(ch)
        lam *= float(np.exp(eta * (b / budget_bytes - 1.0)))
    return np.array(hist_l), np.array(hist_b), choices

# --------------------------------------------------------------------------- v3 hardware contract

def _segment_matmul(x, w, segment=K_SEG):
    """FP32 segment partials accumulated in ascending reduction-segment order."""
    x, w = np.asarray(x, np.float32), np.asarray(w, np.float32)
    acc = np.zeros((x.shape[0], w.shape[1]), np.float32)
    for lo in range(0, x.shape[1], segment):
        acc = np.add(acc, x[:, lo:lo+segment] @ w[lo:lo+segment], dtype=np.float32)
    return acc


def fused_rank_lane_gemm_hw(X, Wq, A, B, L=8, placement="interleave", streamed=False, return_state=False):
    """FP32 ascending-K projection gold with BF16 U and explicit lane-only tails."""
    X, Wq = bf16_round(X), np.asarray(Wq, np.float32)
    A, B = np.asarray(A, np.float32), np.asarray(B, np.float32)
    K, r = X.shape[1], A.shape[1]
    if A.shape != (K, r) or B.shape != (r, Wq.shape[0]) or Wq.shape[1] != K:
        raise ValueError("projection dimensions disagree")
    if L <= 0 and r:
        raise ValueError("compensated lane projection needs positive L")
    U = bf16_round(_segment_matmul(X, A)) if r else np.zeros((X.shape[0], 0), np.float32)
    S = -(-K // K_SEG)
    fused = min(r, L if streamed else L*S)
    segs = [[] for _ in range(S)]
    if streamed:
        segs[-1] = list(range(fused))
    elif fused:
        segs = rank_segments(fused, K, L, placement)
    acc = np.zeros((X.shape[0], Wq.shape[0]), np.float32)
    for s, ranks in enumerate(segs):
        lo, hi = s*K_SEG, min(K, (s+1)*K_SEG)
        part = (X[:, lo:hi] @ Wq[:, lo:hi].T).astype(np.float32)
        if ranks:
            part = np.add(part, U[:, ranks] @ B[ranks, :], dtype=np.float32)
        acc = np.add(acc, part, dtype=np.float32)
    tails = [list(range(j, min(r, j+L))) for j in range(fused, r, max(1, L))]
    for ranks in tails:
        acc = np.add(acc, U[:, ranks] @ B[ranks, :], dtype=np.float32)
    state = {"U": U, "segments": segs, "tail_segments": tails, "tail_issues_per_n_tile":len(tails)}
    return (acc, state) if return_state else acc


def expert_ffn_hw(x, Wg, Wu, Wd, comp=None, gate=None, L=8, z_mode="full", return_state=False):
    """§3.8 gold: BF16 X/U/Z; FP32 K-ordered acc/SiLU/gating; correction before SiLU."""
    if z_mode not in ("full", "streamed"):
        raise ValueError("z_mode must be full or streamed")
    x = bf16_round(x)
    Ws = {"g":np.asarray(Wg,np.float32),"u":np.asarray(Wu,np.float32),"d":np.asarray(Wd,np.float32)}
    if comp is None:
        comp = {k:(np.zeros((w.shape[1],0),np.float32),np.zeros((0,w.shape[0]),np.float32)) for k,w in Ws.items()}
    g, gs = fused_rank_lane_gemm_hw(x, Ws["g"], *comp["g"], L=L, return_state=True)
    u, us = fused_rank_lane_gemm_hw(x, Ws["u"], *comp["u"], L=L, return_state=True)
    sigmoid = np.empty_like(g); pos = g >= 0
    sigmoid[pos] = 1 / (1 + np.exp(-g[pos]))
    eg = np.exp(g[~pos]); sigmoid[~pos] = eg / (1 + eg)
    z = (g * sigmoid * u).astype(np.float32)
    if gate is not None:
        gates = np.asarray(gate,np.float32)
        if gates.shape != (x.shape[0],): raise ValueError("one routing gate per token is required")
        z = (z*gates[:,None]).astype(np.float32)
    z = bf16_round(z)
    y, ds = fused_rank_lane_gemm_hw(z, Ws["d"], *comp["d"], L=L, streamed=z_mode=="streamed", return_state=True)
    state = {"X":x,"G":g,"Up":u,"Z":z,"U_g":gs["U"],"U_u":us["U"],"U_d":ds["U"],
             "rank_segments":{k:s["segments"] for k,s in (("g",gs),("u",us),("d",ds))},
             "tail_segments":{k:s["tail_segments"] for k,s in (("g",gs),("u",us),("d",ds))}}
    return (y,state) if return_state else y


def combine_hw(outputs, token_indices, completion_order, token_count=None, hidden=None):
    """FP32 scatter merge in externally supplied actual expert completion order."""
    keys = list(outputs.keys()) if isinstance(outputs,dict) else list(range(len(outputs)))
    if len(completion_order)!=len(keys) or set(completion_order)!=set(keys):
        raise ValueError("completion order must be a permutation of output experts")
    if token_count is None:
        token_count = 1+max((int(t) for k in keys for t in token_indices[k]),default=-1)
    if hidden is None: hidden = np.asarray(outputs[keys[0]]).shape[1] if keys else 0
    out = np.zeros((token_count,hidden),np.float32)
    for k in completion_order:
        y = np.asarray(outputs[k],np.float32); ids = token_indices[k]
        if y.shape != (len(ids),hidden) or len(set(ids))!=len(ids):
            raise ValueError("expert output and unique token indices must agree")
        for row,t in enumerate(ids):
            if not 0 <= t < token_count: raise ValueError("token outside combine buffer")
            out[t] = np.add(out[t],y[row],dtype=np.float32)
    return out


def down_partial_contributions_hw(Z, U_d, Wd, Bd, rank_segments, tail_segments):
    """Validation-only reconstruction of each Down delta before atomic Combine.

    The returned full matrices are host-side gold artifacts, not modeled private
    SRAM. Hardware holds an N-group delta and drains it after each K segment or
    lane-only tail. Inputs must be actual decoded W/B and stored BF16 Z/U.
    """
    Z, U_d = bf16_round(Z), bf16_round(U_d)
    Wd, Bd = np.asarray(Wd, np.float32), np.asarray(Bd, np.float32)
    if Wd.ndim != 2 or Wd.shape[1] != Z.shape[1] or Bd.shape != (U_d.shape[1], Wd.shape[0]):
        raise ValueError("Down partial dimensions disagree")
    if len(rank_segments) != -(-Z.shape[1] // K_SEG):
        raise ValueError("one Down rank placement per K segment is required")
    parts, sources = [], []
    for s, ranks in enumerate(rank_segments):
        lo, hi = s*K_SEG, min(Z.shape[1], (s+1)*K_SEG)
        part = (Z[:, lo:hi] @ Wd[:, lo:hi].T).astype(np.float32)
        if ranks:
            part = np.add(part, U_d[:, ranks] @ Bd[ranks], dtype=np.float32)
        parts.append(part)
        sources.append((True, s, tuple(ranks)))
    for ranks in tail_segments:
        parts.append((U_d[:, ranks] @ Bd[ranks]).astype(np.float32))
        # Lane-only TileSpec uses k_segment=0; ranks uniquely name its group.
        sources.append((False, 0, tuple(ranks)))
    out = np.zeros((Z.shape[0], Wd.shape[0]), np.float32)
    for part in parts:
        out = np.add(out, part, dtype=np.float32)
    return {"parts": parts, "source_keys": sources, "output": out}


def combine_partial_hw(contributions, token_indices, events, token_count=None, hidden=None, assert_complete=True, L=8, batch=None):
    """FP32 atomic scatter gold in actual timed Combine-event order.

    contributions[e] is down_partial_contributions_hw's result, a final
    Me-by-H ndarray, or {X,W:{g,u,d},factors:{g,u,d},gate} actual decoded operands.
    Raw operand states are projected independently with expert_ffn_hw. Events match timed_numeric.rs:
    expert_id, col, cols, partial, sources[{n,k_segment,main,ranks,m_start,valid_rows}]. Sources are
    ordered by actual tile issue and added into the pending delta with FP32
    rounding after each contribution. All K/tail column coverage and
    unique routed token indices are checked. Nonpartial events use final Y.
    """
    if batch is not None:
        if token_count is not None and token_count != batch:
            raise ValueError("batch and token_count disagree")
        token_count = batch
    events = list(events)
    decoded = {}
    for k, v in contributions.items():
        if isinstance(v, dict) and "X" in v and "W" in v:
            streamed = any(ev["expert_id"] == k and ev.get("partial", False) for ev in events)
            w, f = v["W"], v.get("factors")
            y, state = expert_ffn_hw(v["X"], w["g"], w["u"], w["d"], f, v.get("gate"), L=L,
                                     z_mode="streamed" if streamed else "full", return_state=True)
            Bd = f["d"][1] if f is not None else np.zeros((0, np.asarray(w["d"]).shape[0]), np.float32)
            c = down_partial_contributions_hw(state["Z"], state["U_d"], w["d"], Bd,
                                               state["rank_segments"]["d"], state["tail_segments"]["d"])
            c["output"] = y
            decoded[k] = c
        else:
            decoded[k] = v
    contributions = decoded
    keys = list(contributions)
    finals = {k: np.asarray(v["output"] if isinstance(v, dict) else v, np.float32)
              for k, v in contributions.items()}
    if token_count is None:
        token_count = 1 + max((int(t) for k in keys for t in token_indices[k]), default=-1)
    if hidden is None:
        hidden = finals[keys[0]].shape[1] if keys else 0
    out = np.zeros((token_count, hidden), np.float32)
    modes, coverage = {}, {}
    for k in keys:
        ids = token_indices[k]
        if finals[k].shape != (len(ids), hidden) or len(set(ids)) != len(ids):
            raise ValueError("expert output and unique token indices must agree")
        if any(not 0 <= int(t) < token_count for t in ids):
            raise ValueError("token outside combine buffer")
        v = contributions[k]
        count = len(v["parts"]) if isinstance(v, dict) else 1
        coverage[k] = np.zeros((count, len(ids), hidden), np.uint8)
    for event in events:
        k = event["expert_id"]
        if k not in contributions:
            raise ValueError("Combine event refers to an unknown expert")
        col, cols = int(event["col"]), int(event["cols"])
        if not 0 <= col < hidden or cols <= 0 or col+cols > hidden:
            raise ValueError("Combine column range is invalid")
        if "tokens" in event and list(event["tokens"]) != list(token_indices[k]):
            raise ValueError("Combine event token ownership changed")
        partial = bool(event.get("partial", False))
        if k in modes and modes[k] != partial:
            raise ValueError("one expert cannot mix final and partial Combine drains")
        modes[k] = partial
        delta = np.zeros((len(token_indices[k]), cols), np.float32)
        if not partial:
            if np.any(coverage[k][:, :, col:col+cols]):
                raise ValueError("expert output merged twice")
            coverage[k][:, :, col:col+cols] = 1
            delta[:] = finals[k][:, col:col+cols]
        else:
            v = contributions[k]
            if not isinstance(v, dict) or not event.get("sources"):
                raise ValueError("partial Combine needs decoded contribution sources")
            lookup = {tuple(s): i for i, s in enumerate(v["source_keys"])}
            for src in event["sources"]:
                n = int(src["n"])
                key = (bool(src["main"]), int(src["k_segment"]) if src["main"] else 0, tuple(src["ranks"]))
                if key not in lookup:
                    raise ValueError("Combine source differs from decoded Down placement")
                j = lookup[key]; lo, hi = max(col, n), min(col+cols, n+4, hidden)
                row_lo = int(src.get("m_start", 0)); row_hi = row_lo+int(src.get("valid_rows", len(token_indices[k])))
                if not 0 <= row_lo < row_hi <= len(token_indices[k]):
                    raise ValueError("Combine source row range is invalid")
                if lo >= hi or n < col or n >= col+cols:
                    raise ValueError("Combine source does not belong to this N group")
                if np.any(coverage[k][j, row_lo:row_hi, lo:hi]):
                    raise ValueError("Down delta merged twice")
                coverage[k][j, row_lo:row_hi, lo:hi] = 1
                delta[row_lo:row_hi, lo-col:hi-col] = np.add(delta[row_lo:row_hi, lo-col:hi-col], np.asarray(v["parts"][j], np.float32)[row_lo:row_hi, lo:hi], dtype=np.float32)
        for row, token in enumerate(token_indices[k]):
            out[token, col:col+cols] = np.add(out[token, col:col+cols], delta[row], dtype=np.float32)
    if assert_complete and any(not np.all(v == 1) for v in coverage.values()):
        raise ValueError("Combine did not drain every Down delta exactly once")
    return out
