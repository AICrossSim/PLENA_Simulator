#!/usr/bin/env python3
"""Tests for ref_numerics.py and budget_v3.py.  Run:  python3 -m pytest -q test_ref.py  (or python3 test_ref.py)"""
import numpy as np

import budget_v3 as bv
import ref_numerics as rn

RNG = np.random.default_rng(1234)


def synthetic_layer(K=1408, N=512, T=256, outliers=8, seed=0):
    """Weights with heavy-ish tails; activations with per-channel scale spread, a few outlier channels
    and correlated structure (so QERA-exact has something to exploit)."""
    rng = np.random.default_rng(seed)
    W = (rng.standard_t(df=5, size=(N, K)) * 0.02).astype(np.float32)
    scale = np.exp(rng.normal(0, 0.6, size=K))
    scale[rng.choice(K, outliers, replace=False)] *= 12.0
    mix = rng.normal(size=(K, 32)) / np.sqrt(32)
    X = (rng.normal(size=(T, K)) + 1.5 * rng.normal(size=(T, 32)) @ mix.T) * scale
    return W, X.astype(np.float32)


# --------------------------------------------------------------------------- MX
def test_mx_roundtrip_bounds():
    W = RNG.normal(size=(64, 1408)).astype(np.float32)
    for bits in (2, 3, 4, 8):
        q, e, deq = rn.mx_quantize(W, bits)
        qmax = 2 ** (bits - 1) - 1
        assert np.abs(q).max() <= qmax
        blocks = np.pad(W, ((0, 0), (0, (-W.shape[1]) % 32))).reshape(64, -1, 32)
        sc = np.exp2(e.astype(np.float64))[..., None]
        err = np.abs(np.pad(W - deq, ((0, 0), (0, (-W.shape[1]) % 32))).reshape(64, -1, 32))
        assert np.all(err <= sc / 2 + 1e-6)
        assert np.all(np.abs(blocks) <= sc * qmax + 1e-6)


def test_mx_zero_block():
    W = np.zeros((4, 64), np.float32)
    q, e, deq = rn.mx_quantize(W, 4)
    assert np.all(q == 0) and np.all(deq == 0)


def test_tile_bytes_agree_with_budget():
    for bits in (2, 3, 4):
        assert rn.mx_tile_bytes(4, 512, bits) == bv.main_tile_bytes(512, bits)
    assert bv.main_tile_bytes(512, 4) == 1088
    assert bv.align(bv.main_tile_bytes(512, 4) + bv.factor_block_bytes(8, 4, "bf16")) == 1152


# --------------------------------------------------------------------------- low rank
def test_error_decreases_with_rank_and_qera_exact_is_best_on_calibration():
    W, X = synthetic_layer()
    Wq = rn.mx_quantize(W, 4)[2]
    y = X.astype(np.float64) @ W.T.astype(np.float64)
    errs = {}
    for method in ("lqer", "l2qer", "qera_approx", "qera_exact"):
        prev = None
        for r in (0, 8, 16, 32):
            A, B, _ = rn.lowrank_factors(W, Wq, X, r, method)
            e = rn.rel_err(rn.reference_compensated(X, Wq, A, B), y)
            if prev is not None:
                assert e <= prev + 1e-9, (method, r)
            prev = e
            errs[(method, r)] = e
    for r in (8, 16, 32):   # QERA-exact minimizes the calibration objective exactly
        assert errs[("qera_exact", r)] <= min(errs[(m, r)] for m in ("lqer", "l2qer", "qera_approx")) + 1e-9
    # with per-channel scale spread, activation-aware scaling beats plain weight-space SVD
    assert errs[("qera_approx", 32)] < errs[("lqer", 32)]


def test_factor_quantization_keeps_most_of_the_gain():
    W, X = synthetic_layer(seed=1)
    Wq = rn.mx_quantize(W, 4)[2]
    y = X.astype(np.float64) @ W.T.astype(np.float64)
    A, B, _ = rn.lowrank_factors(W, Wq, X, 32, "qera_approx")
    e0 = rn.rel_err(X.astype(np.float64) @ Wq.T.astype(np.float64), y)
    e_bf = rn.rel_err(rn.reference_compensated(X, Wq, rn.quantize_factor(A, "bf16", 0), rn.quantize_factor(B, "bf16", 0)), y)
    e_m8 = rn.rel_err(rn.reference_compensated(X, Wq, rn.quantize_factor(A, "mxint8", 0), rn.quantize_factor(B, "mxint8", 0)), y)
    assert e_bf < e0 and e_m8 < e0
    assert e_m8 <= e_bf * 1.25 + 1e-6


# --------------------------------------------------------------------------- rank lanes
def test_rank_segments_capacity():
    assert [len(s) for s in rn.rank_segments(24, 1408, 8)] == [8, 8, 8]
    assert [len(s) for s in rn.rank_segments(32, 2048, 8)] == [8, 8, 8, 8]
    try:
        rn.rank_segments(32, 1408, 8)
        raise AssertionError("capacity violation not detected")
    except ValueError:
        pass
    # P1 slack packing: Down K=1408 has 128 idle K positions in its last segment
    assert sum(len(s) for s in rn.rank_segments(32, 1408, 8, slack=True)) == 32
    assert bv.rank_capacity(1408, 8) == 24 and bv.rank_capacity(1408, 8, slack_pack=True) == 152


def test_fused_rank_lanes_match_unfused():
    W, X = synthetic_layer(K=1408, N=256, T=16, seed=2)
    Xb = rn.bf16_round(X)
    Wq = rn.mx_quantize(W, 4)[2]
    A, B, _ = rn.lowrank_factors(W, Wq, Xb, 24, "qera_approx")
    A = rn.quantize_factor(A, "bf16", 0); B = rn.quantize_factor(B, "bf16", 0)
    ref = rn.reference_compensated(Xb, Wq, A, B)
    for placement in ("interleave", "late"):
        fused = rn.fused_rank_lane_gemm(Xb, Wq, A, B, L=8, placement=placement)
        assert rn.rel_err(fused, ref) < 2e-3, placement           # U stored as BF16 + fp32 accumulation
        exact_u = rn.fused_rank_lane_gemm(Xb, Wq, A, B, L=8, placement=placement, u_bf16=False)
        assert rn.rel_err(exact_u, ref) < 1e-5, placement


def test_compensation_order_matters():
    rng = np.random.default_rng(3)
    d, I, T = 256, 128, 8
    Wg, Wu = rng.normal(size=(I, d)) * 0.05, rng.normal(size=(I, d)) * 0.05
    Wd = rng.normal(size=(d, I)) * 0.05
    x = rng.normal(size=(T, d))
    comp = {k: (rng.normal(size=(d if k != "d" else I, 8)) * 0.05, rng.normal(size=(8, I if k != "d" else d)) * 0.05)
            for k in ("g", "u", "d")}
    a = rn.expert_ffn(x, Wg, Wu, Wd, comp, order="before_silu")
    b = rn.expert_ffn(x, Wg, Wu, Wd, comp, order="after_silu")
    assert rn.rel_err(b, a) > 1e-3


def test_gate_folding_is_exact_in_real_arithmetic():
    rng = np.random.default_rng(4)
    d, I, T = 128, 64, 6
    Wg, Wu, Wd = rng.normal(size=(I, d)), rng.normal(size=(I, d)), rng.normal(size=(d, I))
    x = rng.normal(size=(T, d)); g = rng.uniform(0.05, 0.5, size=T)
    folded = rn.expert_ffn(x, Wg, Wu, Wd, gate=g)
    post = rn.expert_ffn(x, Wg, Wu, Wd) * g[:, None]
    assert rn.rel_err(folded, post) < 1e-12


# --------------------------------------------------------------------------- rank allocation
def _alloc_problem(E=64, seed=5):
    rng = np.random.default_rng(seed)
    ranks = [0, 8, 16, 24, 32]
    T = np.stack([rn.tail_energy(np.sort(rng.gamma(2.0, 1.0, size=64))[::-1] * rng.uniform(0.5, 2), ranks)
                  for _ in range(E)])
    per_rank_bytes = 2 * (2048 + 1408) * 2 + 2 * (1408 + 2048)   # gate+up+down factor bytes per unit rank (BF16)
    rank_bytes = np.array([[r * per_rank_bytes for r in ranks]] * E, dtype=np.float64)
    cap = np.full(E, len(ranks) - 1)
    return T, rank_bytes, cap, ranks


def test_allocation_is_monotone_in_lambda_and_respects_capacity():
    T, rb, cap, ranks = _alloc_problem()
    w = np.random.default_rng(6).gamma(1.0, 1.0, size=T.shape[0])
    cap2 = cap.copy(); cap2[:10] = 2
    prev = None
    for lam in (1e-9, 1e-7, 1e-5, 1e-3):
        ch = rn.allocate_ranks(w, T, rb, lam, cap2)
        assert np.all(ch <= cap2)
        b = rb[np.arange(len(ch)), ch].sum()
        if prev is not None:
            assert b <= prev + 1e-9
        prev = b


def test_weighted_allocation_beats_uniform_at_equal_bytes():
    T, rb, cap, ranks = _alloc_problem()
    rng = np.random.default_rng(7)
    w = rng.gamma(0.7, 1.0, size=T.shape[0])
    uni = np.full(T.shape[0], 2)                                   # rank 16 for everybody
    budget = rb[np.arange(len(uni)), uni].sum()
    lo, hi = 1e-12, 1e2
    for _ in range(200):                                           # bisection on lambda to meet the budget
        mid = np.sqrt(lo * hi)
        ch = rn.allocate_ranks(w, T, rb, mid, cap)
        if rb[np.arange(len(ch)), ch].sum() > budget:
            lo = mid
        else:
            hi = mid
    ch = rn.allocate_ranks(w, T, rb, hi, cap)
    assert rb[np.arange(len(ch)), ch].sum() <= budget + 1e-6
    obj_w = (w * T[np.arange(len(ch)), ch]).sum()
    obj_u = (w * T[np.arange(len(uni)), uni]).sum()
    assert obj_w <= obj_u


def test_lambda_controller_tracks_budget():
    T, rb, cap, ranks = _alloc_problem()
    rng = np.random.default_rng(8)
    steps = []
    for _ in range(200):
        active = rng.choice(T.shape[0], 40, replace=False)
        steps.append((active, rng.gamma(0.7, 1.0, size=40)))
    budget = 40 * rb[0, 2]                                          # average rank 16 over 40 active experts
    lam, b, _ = rn.lambda_controller(steps, T, rb, cap, budget, lam0=1e-6, eta=0.3)
    assert abs(np.mean(b[100:]) / budget - 1.0) < 0.1


# --------------------------------------------------------------------------- budget calculator invariants
def test_expert_bytes_and_capacity():
    dims = bv.MODELS["dsv2_lite"]
    _, tot, bf = bv.expert_plan(dims, False, bv.Fmt("x", 4, "bf16", 8, comp="lanes"))
    assert bf == 17301504 and tot == 5212160
    _, tot_s, bf_s = bv.expert_plan(dims, True, bv.Fmt("x", 4, "bf16", 8, comp="lanes"))
    assert bf_s == 34603008 and tot_s == 10280960
    for name, K, N in bv.projections(dims, False):
        assert bv.rank_capacity(K, 8) >= dict(gate=32, up=32, down=24)[name]


def test_storage_fits_and_issue_counts():
    dims = bv.MODELS["dsv2_lite"]
    for T in (16, 64, 96, 128):
        assert bv.storage_plan(dims, T, prec="P2", L=8)["fits"], T
    for T in (16, 64, 96):
        assert bv.storage_plan(dims, T, prec="P1", L=8)["fits"], T
    assert not bv.storage_plan(dims, 128, prec="P1", L=8)["fits"]      # P1 limited to T <= 96
    assert not bv.storage_plan(dims, 128, prec="P2", L=16)["fits"]     # T = 128 only with P2, L = 8
    ic = bv.issue_counts(dims, False, bv.Fmt("x", 4, "bf16", 8, comp="lanes"), 2, 2)
    assert ic == dict(main=4352, prepass=82, short=0)
    assert bv.issue_counts(dims, False, bv.FMT_V3_A8S, 2, 2)["prepass"] == 164   # two INT4 passes
    ic = bv.issue_counts(dims, False, bv.Fmt("x", 4, "bf16", 0, comp="separate"), 2, 2)
    assert ic["short"] == 1216


def test_v3_default_bytes():
    dims = bv.MODELS["dsv2_lite"]
    assert bv.expert_plan(dims, False, bv.FMT_V3)[1] == 4970112
    assert bv.expert_plan(dims, True, bv.FMT_V3)[1] == 9889920
    assert bv.expert_plan(dims, False, bv.FMT_V3_A8S)[1] == 5052544
    stream = bv.v3_cores("4+2", "P2")[1]
    assert bv.t_onchip_tile(stream, 5, 1152, "P2", 352) == float("inf")


def test_mxint8_split_is_exact_two_pass():
    rng = np.random.default_rng(11)
    A = (rng.standard_t(df=4, size=(1408, 24)) * 0.03).astype(np.float32)
    hi, lo, e, deq = rn.mxint8_split(A, red_axis=0)
    assert hi.min() >= -7 and hi.max() <= 7 and lo.min() >= -8 and lo.max() <= 7
    q = 16 * hi.astype(np.int32) + lo.astype(np.int32)
    assert np.abs(q).max() <= 119
    X = rn.bf16_round(rng.normal(size=(8, 1408)).astype(np.float32))
    sc = np.repeat(np.exp2(e.astype(np.float64)).T, 32, axis=0)[:1408]   # e is (r, K/32): broadcast along K
    two_pass = X.astype(np.float64) @ (16 * hi * sc) + X.astype(np.float64) @ (lo * sc)
    assert rn.rel_err(two_pass, X.astype(np.float64) @ deq.astype(np.float64)) < 1e-6
    # MXINT8-split is (much) closer to the original than MXINT4
    m4 = rn.quantize_factor(A, "mxint4", 0)
    assert rn.rel_err(deq, A) < 0.2 * rn.rel_err(m4, A)


def test_b_quantizer_respects_segments():
    rng = np.random.default_rng(12)
    B = rng.normal(size=(24, 64)).astype(np.float32)
    segs = rn.rank_segments(24, 1408, 8)
    q1 = rn.quantize_b_per_segment(B, segs, "mxint8")
    B2 = B.copy(); B2[segs[1], :] *= 1000.0                              # perturb another segment only
    q2 = rn.quantize_b_per_segment(B2, segs, "mxint8")
    assert np.array_equal(q1[segs[0], :], q2[segs[0], :])
    assert np.array_equal(rn.quantize_b_per_segment(B, segs, "bf16"), rn.bf16_round(B))


if __name__ == "__main__":
    import inspect, sys
    fails = 0
    for name, fn in sorted(inspect.getmembers(sys.modules[__name__], inspect.isfunction)):
        if name.startswith("test_"):
            try:
                fn(); print("PASS", name)
            except Exception as ex:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(ex))
    sys.exit(1 if fails else 0)
