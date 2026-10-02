#!/usr/bin/env python3
"""
budget_v3.py - closed budget, byte, rank-capacity and first-order timing reference
for the PLENA-MoE supply-first architecture (spec v3).

Purpose
-------
Give the compiler / Rust implementation exact expected values and invariants to
test against: tile and expert byte counts, rank-lane capacity, per-tile on-chip
service under each dataflow, storage and port closure, landing-pool and credit
sizing, compensation-scheme overheads, Shared split costs and a tile-level
single-core timeline.

It is NOT a simulator. Timing here is first order: per-tile on-chip time is the
max of pipelined stage times, cores overlap as a fluid, and HBM is a fixed
service rate. Report any number taken from here as an analytical estimate.

Units: bytes; cycles at 1 GHz (1 cycle = 1 ns); bandwidth in bytes/cycle (= GB/s).
Array notation M x N x K (token rows x output columns x reduction), N = 4, K = 512.
Weight layout W[N, K] (output rows, reduction rows) as in compiler.py.

Usage:  python3 budget_v3.py            # print every table (markdown)
        python3 budget_v3.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field, replace

K_TILE, N_TILE, GRAN = 512, 4, 32
MX_BLOCK = 32                      # MX shared-exponent block along K
CAPACITY_BYTES = 48 * 1024 + 12 * 1024 + 2 * 1024 ** 2   # existing W + X + accumulator arena = 2,158,592
LEGACY_ISSUE_NS = 30.4             # fitted from test_shared_first_all_results.csv (single cores, see prompt)
LEGACY_CREDIT_BW = 128.0           # 256 credits x 32 B / 64 ns
STREAM_ME_MAX = 4                  # stream core X register and accumulator are sized for Me <= 4
CONTROL_RESERVE = 16 * 1024        # descriptor ring + predictor/quota tables (existing controller already uses 4,288 B)
POOL_BANKS, BANK_B = 128, 16       # landing pool: 128 banks x 16 B/cycle = 2,048 B/cycle aggregate, 512 B per bank
SILU_LANES = 64                    # vector unit elements/cycle for SiLU(g)*u


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def align(n: float, g: int = GRAN) -> int:
    return int(cdiv(int(math.ceil(n)), g) * g)


# --------------------------------------------------------------------------- model dimensions
@dataclass(frozen=True)
class Dims:
    name: str
    d: int        # hidden
    I_r: int      # routed expert intermediate
    I_s: int      # shared intermediate (all shared experts merged)
    E: int
    k: int


MODELS = {
    "dsv2_lite": Dims("DeepSeek-V2-Lite (verify against config.json)", 2048, 1408, 2816, 64, 6),
    "qwen2_57b": Dims("Qwen2-57B-A14B", 3584, 2560, 20480, 64, 8),
    "glm45_air": Dims("GLM-4.5-Air", 4096, 1408, 1408, 128, 8),
}


def projections(dims: Dims, shared: bool):
    I = dims.I_s if shared else dims.I_r
    return [("gate", dims.d, I), ("up", dims.d, I), ("down", I, dims.d)]   # (name, K, N)


def nseg(K: int) -> int:
    return cdiv(K, K_TILE)


def seg_len(K: int, s: int) -> int:
    return min(K_TILE, K - K_TILE * s)


# --------------------------------------------------------------------------- formats and tile bytes
@dataclass(frozen=True)
class Fmt:
    name: str
    main_bits: int = 16          # 16 = BF16 (no scales); 4/3/2 = MXINT, block 32 along K, 8-bit shared exponent
    factor: str = "none"         # B format: 'none' | 'bf16' | 'mxint8' | 'mxint4'
    L: int = 0                   # rank lanes per K segment (0 = no fused lanes)
    factor_a: str = None         # A format (default = factor). P2 needs an INT format for A:
                                 # 'mxint4' (one pass) or 'mxint8s' (MXINT8 codes clipped to +-119, two INT4 passes)
    comp: str = "lanes"          # 'lanes' | 'separate' | 'kext' | 'offload' | 'none'
    slack_pack: bool = False     # P1 only: put extra ranks into the idle K positions of the last segment


BF16 = Fmt("BF16")
# v3 default (P1 and P2 use the same bytes, so mode comparisons are equal-byte): MXINT4 main, A MXINT4, B BF16, L=8
FMT_V3 = Fmt("W4+LR (A MXINT4, B BF16, L=8)", 4, "bf16", 8, factor_a="mxint4")
FMT_V3_A8S = Fmt("W4+LR (A MXINT8-split, B BF16, L=8)", 4, "bf16", 8, factor_a="mxint8s")
FMT_V3_W3 = Fmt("W3+LR (A MXINT4, B BF16, L=8)", 3, "bf16", 8, factor_a="mxint4")


def main_tile_bytes(kv: int, bits: int) -> int:
    if bits == 16:
        return N_TILE * kv * 2
    return align(N_TILE * kv * bits / 8 + N_TILE * cdiv(kv, MX_BLOCK))


def factor_block_bytes(n_red: int, n_cols: int, factor: str) -> float:
    """Bytes of an (n_red x n_cols) factor block whose reduction length is n_red."""
    if factor == "mxint8s":
        factor = "mxint8"
    if factor == "bf16":
        return n_red * n_cols * 2
    if factor == "mxint8":
        return n_red * n_cols + n_cols * cdiv(n_red, MX_BLOCK)
    if factor == "mxint4":
        return n_red * n_cols / 2 + n_cols * cdiv(n_red, MX_BLOCK)
    raise ValueError(factor)


def a_fmt(fmt: Fmt) -> str:
    return fmt.factor_a or fmt.factor


def a_passes(fmt: Fmt) -> int:
    return 2 if a_fmt(fmt) == "mxint8s" else 1


def ranks_on_segment(r_fused: int, S: int, s: int) -> int:
    """Interleaved placement j -> segment j mod S."""
    return r_fused // S + (1 if s < r_fused % S else 0)


def rank_capacity(K: int, L: int, slack_pack: bool = False) -> int:
    S = nseg(K)
    return L * S + ((S * K_TILE - K) if slack_pack else 0)


@dataclass
class ProjPlan:
    name: str
    K: int
    N: int
    r: int
    S: int
    n_tiles_n: int
    r_fused: int
    r_slack: int
    r_tail: int
    main_tiles: int
    main_bytes: int            # main tiles incl. embedded B sub-blocks
    a_bytes: int               # A factor tiles (for the U pre-pass)
    b_sep_bytes: int           # B bytes fetched as separate tiles (separate/offload/tail)
    seg_tile_bytes: list = field(default_factory=list)


def plan_projection(name: str, K: int, N: int, r: int, fmt: Fmt) -> ProjPlan:
    S, nt = nseg(K), cdiv(N, N_TILE)
    if fmt.main_bits == 16 or r == 0 or fmt.comp == "none":
        seg = [main_tile_bytes(seg_len(K, s), fmt.main_bits) for s in range(S)]
        return ProjPlan(name, K, N, 0, S, nt, 0, 0, 0, S * nt, nt * sum(seg), 0, 0, seg)
    if fmt.comp == "kext":
        # K-extension: [X | U] [W ; B]; B rows live in the extended K range (P1: BF16 PEs only)
        K2 = K + r
        S2 = nseg(K2)
        seg = []
        for s in range(S2):
            lo, hi = K_TILE * s, min(K_TILE * (s + 1), K2)
            kw = max(0, min(hi, K) - lo)                  # main weight rows in this segment
            kb = max(0, hi - max(lo, K))                   # B rows in this segment
            b = factor_block_bytes(kb, N_TILE, fmt.factor) if kb else 0
            seg.append(align((main_tile_bytes(kw, fmt.main_bits) if kw else 0) + b))
        a = sum(align(factor_block_bytes(seg_len(K, s), N_TILE, a_fmt(fmt))) for s in range(S)) * cdiv(r, N_TILE)
        return ProjPlan(name, K, N, r, S2, nt, r, 0, 0, S2 * nt, nt * sum(seg), a, 0, seg)
    lanes = fmt.L if fmt.comp == "lanes" else 0
    r_fused = min(r, lanes * S)
    r_slack = min(r - r_fused, (S * K_TILE - K)) if (fmt.slack_pack and fmt.comp == "lanes") else 0
    r_tail = r - r_fused - r_slack
    seg = []
    for s in range(S):
        kv = seg_len(K, s)
        nr = ranks_on_segment(r_fused, S, s) + (r_slack if s == S - 1 else 0)
        b = factor_block_bytes(nr, N_TILE, fmt.factor) if nr else 0
        seg.append(align(main_tile_bytes(kv, fmt.main_bits) + b))
    a = sum(align(factor_block_bytes(seg_len(K, s), N_TILE, a_fmt(fmt))) for s in range(S)) * cdiv(r, N_TILE)
    b_sep = nt * align(factor_block_bytes(r_tail, N_TILE, fmt.factor)) if r_tail else 0
    return ProjPlan(name, K, N, r, S, nt, r_fused, r_slack, r_tail, S * nt, nt * sum(seg), a, b_sep, seg)


DEFAULT_RANKS = {  # (gate, up, down) for L = 8; shared Down has 6 segments for DSV2-Lite
    8: {"routed": (32, 32, 24), "shared": (32, 32, 48)},
    16: {"routed": (64, 64, 48), "shared": (64, 64, 96)},
}


def expert_plan(dims: Dims, shared: bool, fmt: Fmt, ranks=None):
    if ranks is None:
        L = fmt.L if fmt.L in DEFAULT_RANKS else 8
        ranks = DEFAULT_RANKS[L]["shared" if shared else "routed"]
    plans = [plan_projection(n, K, N, r, fmt) for (n, K, N), r in zip(projections(dims, shared), ranks)]
    total = sum(p.main_bytes + p.a_bytes + p.b_sep_bytes for p in plans)
    bf16 = sum(N * K * 2 for _, K, N in projections(dims, shared))
    return plans, total, bf16


def issue_counts(dims: Dims, shared: bool, fmt: Fmt, Me: int, M: int, ranks=None):
    """Issues for one expert on a core with M rows: main, U pre-pass, short-K passes."""
    plans, _, _ = expert_plan(dims, shared, fmt, ranks)
    nb = cdiv(Me, M)
    main = nb * sum(p.main_tiles for p in plans)
    if fmt.main_bits == 16 or fmt.comp == "none":
        return dict(main=main, prepass=0, short=0)
    g, u, dn = plans
    prepass = a_passes(fmt) * nb * (nseg(g.K) * cdiv(g.r + u.r, N_TILE) + nseg(dn.K) * cdiv(dn.r, N_TILE))
    if fmt.comp in ("separate", "offload"):
        short = nb * sum(cdiv(p.r, K_TILE) * p.n_tiles_n for p in plans)
    else:   # ranks beyond lane capacity: lane-only issues, L ranks per issue
        short = nb * sum(cdiv(p.r_tail, max(1, fmt.L)) * p.n_tiles_n for p in plans if p.r_tail)
    return dict(main=main, prepass=prepass, short=short)


# --------------------------------------------------------------------------- per-tile on-chip service
@dataclass(frozen=True)
class Core:
    name: str
    M: int
    dataflow: str = "ws_group"   # 'ws_group' (dense) | 'is_stream' | 'legacy'
    G: int = 8                   # tiles held in the W operand register (ws_group)
    pool_rd: float = 512.0       # B/cycle landing pool -> core
    x_bw: float = 512.0          # B/cycle activation store -> X operand register
    dec_bw: float = 1024.0       # B/cycle of BF16 decoder output (P1 only)
    acc_bw: float = 0.0          # 0 = register-file accumulate; else SRAM RMW B/cycle
    lanes: int = 8
    legacy_issue_ns: float = LEGACY_ISSUE_NS


def t_onchip_tile(core: Core, Me: int, tile_bytes: int, prec: str, tiles_in_seg: int) -> float:
    """First-order per-tile on-chip service (cycles) with pipelined stages.

    prec: 'P0' BF16 weights; 'P1' compressed in pool, decoded on read to BF16 PEs;
          'P2' MX-native PEs (no decode, W stays compressed up to the PEs).
    """
    nb = cdiv(Me, core.M)
    if core.dataflow == "legacy":
        return nb * core.legacy_issue_ns
    if core.dataflow == "ws_group":
        w_reads = 1                                         # W held across all M blocks of the group
        x_bytes = nb * core.M * 1024 / core.G               # X block per (group, segment, M block), shared by G tiles
        dec = (4096 / core.dec_bw) if prec == "P1" else 0   # decode once into the (decoded) W register
    elif core.dataflow == "is_stream":
        if Me > STREAM_ME_MAX:
            return math.inf                                 # X register / accumulator sized for Me <= 4: hard limit
        w_reads = 1                                         # X for all M blocks of the segment held; W used nb times
        x_bytes = nb * core.M * 1024 / max(1, tiles_in_seg)
        dec = (4096 / core.dec_bw) if prec == "P1" else 0
    else:
        raise ValueError(core.dataflow)
    pool = w_reads * tile_bytes / core.pool_rd
    x = x_bytes / core.x_bw
    acc = 0.0 if core.acc_bw == 0 else nb * core.M * N_TILE * 4 * 2 / core.acc_bw
    return max(float(nb), pool, dec, x, acc)


def expert_onchip_cycles(dims: Dims, shared: bool, fmt: Fmt, Me: int, core: Core, prec: str) -> float:
    plans, _, _ = expert_plan(dims, shared, fmt)
    tot = 0.0
    for p in plans:
        for s, tb in enumerate(p.seg_tile_bytes):
            tot += p.n_tiles_n * t_onchip_tile(core, Me, tb, prec, p.n_tiles_n)
    if fmt.main_bits != 16 and fmt.comp != "none" and core.dataflow != "legacy":
        if core.dataflow == "is_stream" and Me > STREAM_ME_MAX:
            return math.inf
        ic = issue_counts(dims, shared, fmt, Me, core.M)
        nb = cdiv(Me, core.M)
        for p in plans:
            for s in range(nseg(p.K)):
                a_tile = align(factor_block_bytes(seg_len(p.K, s), N_TILE, a_fmt(fmt)))
                tot += cdiv(p.r, N_TILE) * max(nb * a_passes(fmt), a_tile / core.pool_rd)
        tot += ic["short"]
    return tot


# --------------------------------------------------------------------------- layer predictions
def task_list(dims: Dims, B: int, me_routed):
    return [("shared", True, B)] + [(f"e{i}", False, m) for i, m in enumerate(me_routed)]


def layer_bytes(dims: Dims, me_routed, fmt: Fmt) -> int:
    return expert_plan(dims, True, fmt)[1] + len(me_routed) * expert_plan(dims, False, fmt)[1]


def predict_layer(dims: Dims, B: int, me_routed, fmt: Fmt, cores, prec: str, bw: float, policy="affinity"):
    """Fluid estimate: time = max(bytes/bw, max_c sum of on-chip tile times on core c)."""
    tasks = task_list(dims, B, me_routed)
    cost = {(t[0], c.name): expert_onchip_cycles(dims, t[1], fmt, t[2], c, prec) for t in tasks for c in cores}
    load = {c.name: 0.0 for c in cores}
    small = min(cores, key=lambda c: c.M)
    big = max(cores, key=lambda c: c.M)
    order = sorted(tasks, key=lambda t: -max(cost[(t[0], c.name)] for c in cores))
    for name, sh, me in order:
        if len(cores) == 1:
            ch = cores[0]
        elif policy == "affinity" and sh:
            ch = big
        else:  # greedy earliest finish on on-chip load, ties -> affinity (Me <= M_small -> small core)
            ch = min((c for c in cores if math.isfinite(cost[(name, c.name)])),
                     key=lambda c: (load[c.name] + cost[(name, c.name)],
                                    0 if (me <= small.M) == (c is small) else 1))
        load[ch.name] += cost[(name, ch.name)]
    t_hbm = layer_bytes(dims, me_routed, fmt) / bw
    t_on = max(load.values())
    return dict(time=max(t_hbm, t_on), t_hbm=t_hbm, t_onchip=t_on, bound="HBM" if t_hbm >= t_on else "on-chip",
                loads=load)


def zipf_me(T: int, E: int = 64, k: int = 6, a: float = 0.8):
    """Deterministic Me list for T tokens: expected counts under Zipf(a) popularity, largest-remainder rounding."""
    p = [1.0 / (i + 1) ** a for i in range(E)]
    z = sum(p)
    exp = [T * k * x / z for x in p]
    base = [int(math.floor(x)) for x in exp]
    for i in sorted(range(E), key=lambda i: -(exp[i] - base[i]))[: T * k - sum(base)]:
        base[i] += 1
    return [m for m in base if m > 0]


# Me lists approximating the captured windows: distinct experts match; issue counts match the single cores
# (bfcl_b16 also matches 3+3 and 4+4). The implementation must use fixtures/workloads.json, not these lists.
ME_CASES = {
    "bfcl_b2": (2, [2] * 6),
    "bfcl_b16": (16, [1] * 21 + [2] * 16 + [3] * 4 + [4] * 4 + [5] * 3),
    "swe_b16": (16, [12, 11, 10, 9, 9, 7, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 1, 1]),
    "mixed_T64_zipf": (64, zipf_me(64)),
    "mixed_T96_zipf": (96, zipf_me(96)),
}


def v3_cores(org: str, prec: str):
    """Default v3 core configurations (see prompt section 3)."""
    if org == "4+2":
        return [Core("dense4", 4, "ws_group", G=8, pool_rd=512, x_bw=512),
                Core("stream2", 2, "is_stream", G=1, pool_rd=512, x_bw=128, acc_bw=128)]
    if org == "3+3":
        return [Core("c0_3", 3, "ws_group", G=4, pool_rd=512, x_bw=320),    # iso-port: X 640 total, WOR ~equal
                Core("c1_3", 3, "ws_group", G=4, pool_rd=512, x_bw=320)]
    if org == "6":   # iso-port single core: union of the 4+2 ports
        return [Core("single6", 6, "ws_group", G=8, pool_rd=1024, x_bw=640)]
    raise ValueError(org)


def legacy_cores(org: str):
    if org == "6":
        return [Core("legacy6", 6, "legacy")]
    if org == "4+2":
        return [Core("legacy4", 4, "legacy"), Core("legacy2", 2, "legacy")]
    if org == "3+3":
        return [Core("legacy3a", 3, "legacy"), Core("legacy3b", 3, "legacy")]
    raise ValueError(org)


# --------------------------------------------------------------------------- storage and ports
def route_state_bytes(T: int, k: int = 6, E: int = 64) -> int:
    """Per (token, expert) pair 16 B: token id, expert id, FP32 gate, scatter + combine slots; 64 B per expert."""
    return T * k * 16 + E * 64


def storage_plan(dims: Dims, T: int, prec: str = "P2", L: int = 8, z_mode: str = "auto",
                 me_dense_max: int = None, me_stream_max: int = STREAM_ME_MAX, k: int = 6):
    """Bytes per structure for the v3 4+2 organization; must fit CAPACITY_BYTES.

    The rule 'P1 only up to T=96; T=128 only P2 with L=8' is a result of this table, not an assumption.
    """
    me_dense_max = me_dense_max or T
    r_sh, r_rt = DEFAULT_RANKS[L]["shared"], DEFAULT_RANKS[L]["routed"]
    tile_c = align(main_tile_bytes(512, 4) + factor_block_bytes(L, N_TILE, "bf16"))
    full = z_mode == "full" or (z_mode == "auto" and T <= 64)
    z_dense = me_dense_max * dims.I_s * 2 if full else 2 * me_dense_max * K_TILE * 2
    w_slot = tile_c if prec == "P2" else 4096                            # P1 registers hold decoded BF16 tiles
    s = {
        "ingress_fifo": 8 * 1024,
        "landing_pool": POOL_BANKS * 512,                                # 128 banks x 512 B = 64 KiB
        "dense_W_register": 2 * 8 * w_slot,                              # G=8, double buffered
        "stream_W_register": 2 * w_slot,
        "dense_X_register": 2 * 4 * 1024,
        "stream_X_register": 2 * me_stream_max * 1024,
        "dense_acc_buffer": 2 * me_dense_max * 32 * 4,                   # Me x (G*4 cols) FP32, double buffered
        "stream_acc_sram": align(me_stream_max * max(2 * dims.I_r, dims.d) * 4, 4096),
        "U_buffers_bf16": me_dense_max * sum(r_sh) * 2 + me_stream_max * sum(r_rt) * 2,
        "U_d_partial_fp32": me_dense_max * r_sh[2] * 4 + me_stream_max * r_rt[2] * 4,   # U_d accumulates over Z segments
        "X_activation": T * dims.d * 2,
        "Z_dense": z_dense,
        "Z_stream": me_stream_max * dims.I_r * 2,
        "combine_fp32": T * dims.d * 4,
        "route_state": route_state_bytes(T, k),
        "control_reserve": CONTROL_RESERVE,
    }
    total = sum(s.values())
    return dict(structures=s, total=total, capacity=CAPACITY_BYTES, fits=total <= CAPACITY_BYTES,
                slack=CAPACITY_BYTES - total, z_mode="full" if full else "streamed")


def port_table(bw=256.0, tile_c=1152, Me_dense=16, M=4, G=8, Me_stream=2, dims=None):
    """Required (peak) vs provided bandwidth in B/cycle at the W4 @ 256 GB/s operating point (4+2)."""
    t_hbm = tile_c / bw                                                  # 4.5 cycles per tile when bytes-bound
    nb = cdiv(Me_dense, M)
    dense_x = min(512.0, nb * M * 1024 / G / max(t_hbm, nb))             # X block per (group, seg, M block) / G
    stream_x = Me_stream * 1024 / (352 * t_hbm)                         # X stationary for a whole K segment (352 tiles)
    zu = (Me_dense * N_TILE * 2 * 2) / t_hbm                            # Z write + U read per Gate/Up tile (small)
    comb = Me_dense * N_TILE * 4 * 2 / (3 * t_hbm)                       # FP32 RMW per Down column tile / 3 K segments
    pool_peak = bw + 512 + 512
    return [
        ("HBM -> ingress FIFO", bw, bw, "8 x 32 B grants per cycle"),
        ("ingress FIFO -> pool (write)", bw, 256, "16 of the 128 pool banks per cycle"),
        ("pool -> dense core (burst)", 512, 512, "one compressed tile per W-register slot"),
        ("pool -> stream core (burst)", 512, 512, "one compressed tile per issue"),
        ("pool aggregate (1 write + 2 read streams, peak)", pool_peak, POOL_BANKS * BANK_B,
         "128 banks x 16 B; bank conflicts must be simulated, not assumed away"),
        ("activation store -> dense X", dense_x, 512, f"Me={Me_dense}: X block per (group, segment, M block) / G"),
        ("activation store -> stream X", stream_x, 128, "X stationary for a whole K segment"),
        ("activation store Z write + U read", zu, 256, "SiLU output and rank-lane U operands"),
        ("activation store aggregate", dense_x + stream_x + zu, 1024, "64 banks x 16 B"),
        ("stream acc SRAM RMW", 2 * 4 * 4 * 2 / max(t_hbm, 1), 128, "per issue 32 B read + 32 B write"),
        ("combine buffer RMW", comb, 256, "FP32 Down output per (expert, column tile), gate folded"),
    ]


def credit_and_pool(bw: float, lat: float, t_ingress: float = 4, t_land: float = 8, tile_c: int = 1152,
                    margin: float = 64, cores: int = 2):
    """Little's law: credits cover lat + ingress; the pool covers lat + landing + a 64-cycle barrier margin
    (largest expected barrier, Shared b16 = 52 cycles) plus two tiles per core in flight to the PEs."""
    credits = cdiv(int(math.ceil(bw * (lat + t_ingress))), GRAN)
    pool_min = bw * (lat + t_land + margin) + cores * 2 * tile_c
    return dict(bw=bw, latency=lat, credits_32B=credits, inflight_bytes=credits * GRAN,
                pool_min_bytes=int(pool_min), pool_min_KiB=round(pool_min / 1024, 1))


# --------------------------------------------------------------------------- compensation alternatives
def compensation_costs(dims: Dims, Me: int, M_big: int = 4, M_small: int = 2, L: int = 8, ranks=None):
    out = []
    base_fmt = Fmt("W4", 4, "bf16", L, comp="none")
    base = issue_counts(dims, False, base_fmt, Me, M_big)["main"]
    for comp, label in [("none", "W4 only (no compensation)"), ("lanes", f"rank lanes L={L}"),
                        ("separate", "separate short-K pass, same core"), ("kext", "K-extension (P1 only)"),
                        ("offload", "offload X*A and U*B to small core")]:
        fmt = Fmt("W4", 4, "bf16", L if comp == "lanes" else 0, factor_a="mxint4", comp=comp)
        ic = issue_counts(dims, False, fmt, Me, M_big, ranks)
        plans, total, _ = expert_plan(dims, False, fmt, ranks)
        if comp == "offload":
            big = issue_counts(dims, False, Fmt("W4", 4, "bf16", 0, comp="none"), Me, M_big)["main"]
            sm = issue_counts(dims, False, fmt, Me, M_small, ranks)
            small_issues = sm["prepass"] + sm["short"]
            xbytes = Me * dims.d * 2                       # X to the small core
            zbytes = Me * dims.I_r * 2                     # Z to the small core (U_d = Z A_d needs it)
            ycomp = Me * (2 * dims.I_r + dims.d) * 4       # FP32 compensation terms back (gate/up before SiLU, down)
            out.append(dict(scheme=label, big_core_issues=big, small_core_issues=small_issues,
                            extra_issue_pct_on_big=0.0, small_core_busy_vs_big_pct=round(100 * small_issues / big, 1),
                            bytes_per_expert=total, cross_core_KiB=round((xbytes + zbytes + ycomp) / 1024, 1),
                            sync="per output column block, before SiLU (gate/up) and before combine (down)"))
            continue
        tot = ic["main"] + ic["prepass"] + ic["short"]
        out.append(dict(scheme=label, big_core_issues=tot, small_core_issues=0,
                        extra_issue_pct_on_big=round(100 * (tot - base) / base, 1), small_core_busy_vs_big_pct=0.0,
                        bytes_per_expert=total, cross_core_KiB=0.0,
                        sync="none" if comp in ("none", "lanes", "kext") else "accumulate before SiLU"))
    return out


def shared_split_costs(dims: Dims, Me: int, r_d: int = 48):
    return [
        dict(split="none (v3 default: whole Shared on dense core)", extra_onchip_bytes=0, sync="none"),
        dict(split="I-split (each core: Gate/Up columns + Down rows)",
             extra_onchip_bytes=Me * dims.d * 4 * 2,
             sync="Down partial outputs must be summed (extra combine RMW Me*d*FP32); each core must apply all r_d "
                  "rank terms with its own partial U_d, so per-core Down capacity L*S_c must be >= r_d"),
        dict(split="N-split (each core: half of every projection's output columns)",
             extra_onchip_bytes=Me * dims.d * 2 + Me * dims.I_s * 2 + Me * r_d * 2,
             sync="X broadcast, full Z broadcast before Down, U_d broadcast"),
    ]


# --------------------------------------------------------------------------- single-core tile timeline
def silu_barrier(dims: Dims, shared: bool, Me: int, core: Core) -> float:
    """Cycles between the last Gate/Up tile and the first trailing A_d tile.

    ws_group (dense): Gate/Up output columns complete group by group, so SiLU runs per column block and only the
    last block (Me x 16 elements) is left; A_d of Z segments 0..S-2 is interleaved into the Gate/Up stream
    (U_d accumulates in FP32), so only the last segment's A_d tiles remain on the critical path
    (Shared b16: 16*16/64 + 4*ceil(48/4) = 4 + 48 = 52 cycles including those A_d tiles).
    is_stream: outputs complete only after the last K segment, so SiLU covers all Me x I elements and every
    A_d tile follows it.
    """
    I = dims.I_s if shared else dims.I_r
    if core.dataflow == "ws_group":
        return Me * 16 / SILU_LANES
    return Me * I / SILU_LANES


def expert_items(dims: Dims, shared: bool, fmt: Fmt, Me: int, core: Core, prec: str):
    """Fetch/consume order for one expert on one core: A_gu, Gate/Up (+ early A_d on ws_group), A_d, Down."""
    plans, _, _ = expert_plan(dims, shared, fmt)
    g, u, dn = plans
    nb = cdiv(Me, core.M)
    comp = fmt.main_bits != 16 and fmt.comp != "none"

    def a_items(p, kind, segs, ncols):
        out = []
        for s in segs:
            a_tile = align(factor_block_bytes(seg_len(p.K, s), N_TILE, a_fmt(fmt)))
            out += [dict(kind=kind, bytes=a_tile, t=max(nb * a_passes(fmt), a_tile / core.pool_rd))] * cdiv(ncols, N_TILE)
        return out

    items = []
    if comp:
        items += a_items(g, "A_gu", range(nseg(g.K)), g.r + u.r)
    gu = []
    for p in (g, u):
        for s, tb in enumerate(p.seg_tile_bytes):
            gu += [dict(kind="GU", bytes=tb, t=t_onchip_tile(core, Me, tb, prec, p.n_tiles_n))] * p.n_tiles_n
    S_d = nseg(dn.K)
    if comp and core.dataflow == "ws_group" and S_d > 1:
        # interleave A_d of Z segment s once the Gate/Up column groups covering it have completed
        merged, step = [], len(gu) / S_d
        for s in range(S_d - 1):
            merged += gu[int(round(s * step)):int(round((s + 1) * step))]
            merged += a_items(dn, "A_d_early", [s], dn.r)
        merged += gu[int(round((S_d - 1) * step)):]
        items += merged
        items += a_items(dn, "A_d", [S_d - 1], dn.r)
    else:
        items += gu
        if comp:
            items += a_items(dn, "A_d", range(S_d), dn.r)
    for s, tb in enumerate(dn.seg_tile_bytes):
        items += [dict(kind="D", bytes=tb, t=t_onchip_tile(core, Me, tb, prec, dn.n_tiles_n))] * dn.n_tiles_n
    return items


def timeline(items, bw: float, lat: float, credit_bytes: int, pool_bytes: int, credit_release="ingress",
             t_land: float = 8, drain: float = 8, barrier_down: float = 0, vlen_silu: float = 0):
    """Sequential FIFO model: one consumer, HBM data bus at bw, credits and pool limit issue.

    Credits return at arrival (ingress) or after landing; pool space returns when the tile is consumed.
    Barriers: first Gate/Up tile waits for the U_gu pre-pass (last A_gu + drain);
              first A_d / Down tile waits barrier_down cycles after the last Gate/Up tile (SiLU / Z).
    """
    cred, pool = deque(), deque()      # (release_time, bytes), release times non-decreasing
    cred_used = pool_used = 0
    issue_prev = bus_free = core_free = 0.0
    last_kind = None
    u_ready = 0.0
    gu_end = None
    hbm_busy = 0.0
    peak_pool = 0
    stall_barrier = 0.0
    for it in items:
        b = it["bytes"]
        t_issue = issue_prev
        # credits
        while cred_used + b > credit_bytes:
            t0, bb = cred.popleft(); cred_used -= bb; t_issue = max(t_issue, t0)
        while pool_used + b > pool_bytes:
            t0, bb = pool.popleft(); pool_used -= bb; t_issue = max(t_issue, t0)
        start_bus = max(t_issue + lat, bus_free)
        arrive = start_bus + b / bw
        bus_free = arrive
        hbm_busy += b / bw
        rel_c = arrive if credit_release == "ingress" else arrive + t_land
        while cred and cred[0][0] <= t_issue:
            cred_used -= cred.popleft()[1]
        while pool and pool[0][0] is not None and pool[0][0] <= t_issue:
            pool_used -= pool.popleft()[1]
        cred.append((rel_c, b)); cred_used += b
        pool.append((None, b)); pool_used += b
        peak_pool = max(peak_pool, pool_used)
        ready = arrive + t_land
        if it["kind"] == "GU" and last_kind == "A_gu":
            u_ready = core_free + drain
        if it["kind"] in ("A_d", "D") and last_kind in ("GU", "A_d_early"):
            gu_end = core_free
        not_before = 0.0
        if it["kind"] == "GU":
            not_before = u_ready
        if it["kind"] in ("A_d", "D") and gu_end is not None:
            not_before = gu_end + barrier_down
        if it["kind"] == "D" and last_kind == "A_d":
            not_before = max(not_before, core_free + drain)
        start = max(ready, core_free, not_before)
        if not_before > max(ready, core_free):
            stall_barrier += not_before - max(ready, core_free)
        end = start + it["t"]
        core_free = end
        # pool release when consumed (refcount 1); replace placeholder
        pool[-1] = (end, b)
        issue_prev = t_issue
        last_kind = it["kind"]
    total = core_free
    return dict(cycles=round(total), hbm_busy_frac=round(hbm_busy / total, 3), peak_pool_bytes=peak_pool,
                barrier_stall_cycles=round(stall_barrier))


# --------------------------------------------------------------------------- report
def md_table(rows, cols=None):
    if not rows:
        return ""
    cols = cols or list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def build_report():
    rep = {}
    dsv2 = MODELS["dsv2_lite"]
    # 1. tile bytes
    rows = []
    for bits in (16, 4, 3, 2):
        for L in ((0,) if bits == 16 else (8, 16)):
            mt = main_tile_bytes(512, bits)
            aug = align(mt + (factor_block_bytes(L, 4, "bf16") if L else 0))
            rows.append(dict(main=("BF16" if bits == 16 else f"MXINT{bits}"), L=L, main_tile_B=mt,
                             augmented_tile_B=aug, requests_32B=aug // GRAN))
    rep["tile_bytes"] = rows
    # 2. rank capacity
    rows = []
    for key, dims in MODELS.items():
        for shared in (False, True):
            for name, K, N in projections(dims, shared):
                rows.append(dict(model=dims.name.split(" (")[0], expert="shared" if shared else "routed", proj=name,
                                 K=K, segments=nseg(K), cap_L8=rank_capacity(K, 8), cap_L16=rank_capacity(K, 16),
                                 P1_slack=nseg(K) * K_TILE - K))
    rep["rank_capacity"] = rows
    # 3. expert bytes
    rows = []
    for key, dims in MODELS.items():
        for shared in (False, True):
            for fmt in [Fmt("MXINT4 only", 4, "bf16", 8, comp="none"),
                        FMT_V3, FMT_V3_A8S,
                        Fmt("W4+LR (A BF16, B BF16, L=8) [P1 only]", 4, "bf16", 8),
                        Fmt("W4+LR (A MXINT4, B BF16, L=16)", 4, "bf16", 16, factor_a="mxint4"),
                        FMT_V3_W3,
                        Fmt("W3+LR (A MXINT4, B BF16, L=16)", 3, "bf16", 16, factor_a="mxint4")]:
                plans, tot, bf = expert_plan(dims, shared, fmt)
                emb_b = sum(p.n_tiles_n * (tb - main_tile_bytes(seg_len(p.K, s), fmt.main_bits))
                            for p in plans for s, tb in enumerate(p.seg_tile_bytes) if s < nseg(p.K))
                fac = sum(p.a_bytes + p.b_sep_bytes for p in plans) + emb_b
                rows.append(dict(model=dims.name.split(" (")[0], expert="shared" if shared else "routed",
                                 format=fmt.name, ranks="/".join(str(p.r) for p in plans) or "-",
                                 bytes=tot, bf16_bytes=bf, ratio=round(bf / tot, 3),
                                 factor_share=round(fac / tot, 3)))
    rep["expert_bytes"] = rows
    # 4. legacy evidence predictions (W4 bytes only on current on-chip path)
    rows = []
    fmt_lr = FMT_V3
    for case, (B, me) in ME_CASES.items():
        for org in ("6", "4+2"):
            bf_bytes = layer_bytes(dsv2, me, BF16)
            w4_bytes = layer_bytes(dsv2, me, fmt_lr)
            cores = legacy_cores(org)
            p_bf = predict_layer(dsv2, B, me, BF16, cores, "P0", LEGACY_CREDIT_BW)
            p_w4 = predict_layer(dsv2, B, me, BF16, cores, "P0", LEGACY_CREDIT_BW * bf_bytes / w4_bytes)
            sp = p_bf["time"] / p_w4["time"]
            rows.append(dict(case=case, org=org, BF16_us=round(p_bf["time"] / 1000, 1),
                             W4_bytes_only_us=round(p_w4["time"] / 1000, 1), speedup=round(sp, 2),
                             ideal=round(bf_bytes / w4_bytes, 2), R=round(sp / (bf_bytes / w4_bytes), 2),
                             bound_after=p_w4["bound"]))
    rep["legacy_w4_prediction"] = rows
    # 5. v3 predictions
    rows = []
    for case, (B, me) in ME_CASES.items():
        base = predict_layer(dsv2, B, me, BF16, legacy_cores("4+2"), "P0", LEGACY_CREDIT_BW)["time"]
        for org in ("6", "3+3", "4+2"):
            for label, fmt, prec, bw in [("BF16 @128", BF16, "P0", 128.0), ("BF16 @256", BF16, "P0", 256.0),
                                         ("W4+LR P1 @256", FMT_V3, "P1", 256.0), ("W4+LR P2 @256", FMT_V3, "P2", 256.0),
                                         ("W4+LR A8-split P2 @256", FMT_V3_A8S, "P2", 256.0),
                                         ("W3+LR P2 @256", FMT_V3_W3, "P2", 256.0),
                                         ("W4+LR P2 @512", FMT_V3, "P2", 512.0)]:
                p = predict_layer(dsv2, B, me, fmt, v3_cores(org, prec), prec, bw)
                rows.append(dict(case=case, org=org, point=label, time_us=round(p["time"] / 1000, 1),
                                 bound=p["bound"], onchip_over_hbm=round(p["t_onchip"] / p["t_hbm"], 2),
                                 speedup_vs_current_4p2=round(base / p["time"], 2)))
    rep["v3_prediction"] = rows
    # 6. per-tile on-chip service table
    rows = []
    tile_c = align(main_tile_bytes(512, 4) + factor_block_bytes(8, 4, "bf16"))
    for Me in (1, 2, 4, 8, 16, 32, 64):
        r = dict(Me=Me)
        r["legacy_any_M"] = round(cdiv(Me, 4) * LEGACY_ISSUE_NS, 1)
        for c in v3_cores("4+2", "P2") + v3_cores("6", "P2"):
            r[f"{c.name}_P2"] = round(t_onchip_tile(c, Me, tile_c, "P2", 352), 2)
            r[f"{c.name}_P1"] = round(t_onchip_tile(c, Me, tile_c, "P1", 352), 2)
        r["hbm_W4_256"] = round(tile_c / 256, 2)
        r["hbm_BF16_128"] = 32.0
        rows.append(r)
    rep["per_tile_onchip"] = rows
    # 7. storage
    rows = []
    for T, prec, L in [(t, p, l) for t in (16, 64, 96, 128) for p in ("P2", "P1") for l in (8, 16)]:
        sp = storage_plan(dsv2, T, prec=prec, L=L)
        rows.append(dict(T=T, prec=prec, L=L, z_mode=sp["z_mode"], total_KiB=round(sp["total"] / 1024, 1),
                         capacity_KiB=round(CAPACITY_BYTES / 1024, 1), slack_KiB=round(sp["slack"] / 1024, 1),
                         fits=sp["fits"],
                         **{k: round(v / 1024, 1) for k, v in sp["structures"].items()}))
    rep["storage"] = rows
    # 8. credits and pool
    rep["credits_pool"] = [credit_and_pool(bw, lat) for bw in (128.0, 256.0) for lat in (64.0, 150.0)]
    # 9. compensation alternatives
    rows = []
    for Me in (1, 2, 4, 16):
        for r in compensation_costs(dsv2, Me):
            rows.append(dict(Me=Me, **r))
    rep["compensation"] = rows
    # 10. shared split
    rep["shared_split_B16"] = shared_split_costs(dsv2, 16)
    # 11. timelines
    rows = []
    fmt = FMT_V3
    cores = {c.name: c for c in v3_cores("4+2", "P2")}
    scen = [
        ("routed Me=2 on stream core", False, 2, cores["stream2"]),
        ("routed Me=4 on stream core", False, 4, cores["stream2"]),
        ("Shared Me=16 on dense core", True, 16, cores["dense4"]),
        ("Shared Me=64 on dense core", True, 64, cores["dense4"]),
    ]
    for label, sh, Me, core in scen:
        bar = silu_barrier(dsv2, sh, Me, core)
        items = expert_items(dsv2, sh, fmt, Me, core, "P2")
        legacy_items = [dict(it, t=cdiv(Me, core.M) * LEGACY_ISSUE_NS) for it in items]
        for tag, its, bw_share, lat, cb, pb, rel in [
                ("v3", items, 256.0, 64.0, 544 * 32, 64 * 1024, "ingress"),
                ("v3", items, 256.0, 150.0, 1232 * 32, 64 * 1024, "ingress"),
                ("v3, pool 16 KiB", items, 256.0, 150.0, 1232 * 32, 16 * 1024, "ingress"),
                ("v3, credits 256 at landing", items, 256.0, 64.0, 256 * 32, 64 * 1024, "landing"),
                ("legacy feed 30.4 ns/issue", legacy_items, 256.0, 64.0, 544 * 32, 64 * 1024, "ingress")]:
            res = timeline(its, bw_share, lat, cb, pb, rel, barrier_down=bar)
            ideal = sum(i["bytes"] for i in its) / bw_share
            rows.append(dict(scenario=label, variant=tag, bw=bw_share, latency=lat, credits_B=cb, pool_B=pb,
                             silu_barrier=round(bar, 1),
                             credit_release=rel, cycles=res["cycles"], bytes_bound=round(ideal),
                             efficiency=round(ideal / res["cycles"], 3),
                             barrier_stall=res["barrier_stall_cycles"], peak_pool=res["peak_pool_bytes"]))
    rep["timeline"] = rows
    rep["ports_W4_256"] = [dict(path=a, required_Bpc=round(b, 1), provided_Bpc=c, note=d) for a, b, c, d in port_table()]
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", help="write all tables as JSON")
    a = ap.parse_args()
    rep = build_report()
    for k, rows in rep.items():
        print(f"\n## {k}\n")
        print(md_table(rows))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
