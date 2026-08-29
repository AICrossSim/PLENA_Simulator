# ruff: noqa: E402
"""Numerical stage tests for the Mamba-2 lowering, on the transactional emulator.

Each case compiles ONE stage of the Mamba-2 mixer to ISA, assembles it, runs the
Rust emulator, and compares the result against a float32 PyTorch golden. What is
and is not covered:

    covered      the compiler emitters in aten/plena/program_mamba_common.py and
                 program_ssd.py, the two new opcodes (V_SOFTPLUS_V, S_MAP_FP_V)
                 end to end through assembler + emulator, and the bf16 rounding
                 the emulator applies on every vector write.

    NOT covered  the RTL's fixed-point exp model. The emulator implements
                 V_EXP_V as libtorch's exact exp, while
                 PLENA_Tools/plena_quant/quant_operations/exp.py is a range
                 reduction plus a 3-term Taylor with ~0.5-1.5% *systematic*
                 relative error. Attention's softmax normalises such a bias away;
                 Mamba applies exp on the critical path of a multiplicative
                 recurrence, where it compounds. Passing here bounds the
                 lowering, not the silicon.

Cases:
    dt        softplus(dt_raw + dt_bias) then clamp  -- exercises V_SOFTPLUS_V
    cumsum    cs = a @ lower_triangular_ones         -- the prefix-scan substitute
    decay     exp(min(cs_i - cs_j, 0)) * causal      -- exercises S_MAP_FP_V and
                                                       V_SUB_VF's rorder=1 form
    conv1d    causal depthwise conv, kernel 4

Run:  python3 mamba2_stage_test.py --case dt [--mlen 64] [--blen 4]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

# Prefer the pinned in-repo submodule (PLENA_Simulator/PLENA_Compiler) over a
# sibling checkout: a sibling may be on a different branch and would otherwise
# silently shadow the submodule on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
for _compiler_root in (_REPO_ROOT / "PLENA_Compiler", _REPO_ROOT.parent / "PLENA_Compiler"):
    if (_compiler_root / "aten" / "plena" / "compiler.py").exists():
        sys.path.insert(0, str(_compiler_root))
        break

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_mamba_common import Mamba2Shape
from compiler.aten.plena.program_ssd import HOST_STAGED
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env

#: Slot order of the Mamba FPRAM constant block, matching
#: ProgramMambaCommonMixin.mamba_fp_constants. Pinned here so a reordering there
#: fails this test rather than silently feeding the kernel wrong constants --
#: FPRAM contents are never checked at runtime.
FP_CONST_ORDER = ["zero", "one", "neg_one", "dt_min", "dt_max", "reci_group", "eps"]


def _mxfp8_exact(shape_, generator, *, hi: float = 2.0) -> torch.Tensor:
    """Draw values that MX-FP8 e4m3 represents exactly.

    Inputs staged through HBM are quantised to e4m3 (3 mantissa bits) with a
    shared e8m0 scale per block of 8. If the test drew arbitrary floats, the
    reported error would be dominated by that input quantisation rather than by
    the kernel, and a real lowering bug an order of magnitude smaller would hide
    inside it. Multiples of 1/8 with magnitude below 2 are exact in e4m3
    (spacing is 2^-3 relative to the exponent, i.e. 0.125 across [1,2) and finer
    below), so the quantisation becomes the identity and the remaining error is
    the kernel's. Same trick the shared-expert MoE test uses to get atol=rtol=0.
    """
    steps = int(hi * 8)
    raw = torch.randint(-steps, steps + 1, shape_, generator=generator)
    return raw.to(torch.float32) / 8.0


def _shape(args, *, seq_len: int) -> Mamba2Shape:
    mlen = args.mlen
    return Mamba2Shape(
        hidden_size=args.hidden_size or 2 * mlen,
        num_heads=args.num_heads,
        head_dim=mlen,
        state_size=mlen,
        n_groups=1,
        conv_kernel=4,
        chunk_size=mlen,
        seq_len=seq_len,
        time_step_min=args.dt_min,
        time_step_max=args.dt_max,
    )


def _write_comparison(build_dir: Path, prog, var, *, rows: int, cols: int, mlen: int):
    addr = prog.get_vram_addr(var.name)
    params = {
        "start_row_idx": addr // mlen,
        "num_rows": (rows * cols) // mlen,
        "num_batches": rows,
        "elements_per_batch": cols,
        "row_dim": mlen,
        "use_stride_mode": cols > mlen,
        # All four cases are bit-exact, so assert that rather than inheriting the
        # default atol=rtol=0.2 at a 90% match bar -- which would let a real
        # regression drift most of the way to wrong before going red, while the CI
        # comment claims a hard failure. Inputs are drawn exactly representable in
        # MX-FP8 precisely so this is achievable; see _mxfp8_exact.
        "atol": 0.0,
        "rtol": 0.0,
    }
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(params, f, indent=2)
    return addr


def _finish(build_dir: Path, prog, golden, input_tensors, fp_preload, order, case, args):
    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA")
    create_sim_env(
        input_tensors,
        gen_code,
        {"original_output": golden},
        fp_preload,
        build_dir=str(build_dir),
    )
    # Stage each tensor at the compiler's own HBM base. The allocator rounds a
    # tensor's footprint up to a whole 64-byte scale row (hbm_tensor_size), so a
    # tensor whose row count is not a multiple of 8 leaves a gap the contiguous
    # writer would not reproduce -- every following tensor would then be read
    # 32 bytes early, and the tail of the read lands in the scale stream, whose
    # 0x7f bytes decode to e4m3 NaN. Same call every other testbench makes.
    hbm_addrs = {name: prog.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    order = sorted(order, key=lambda name: hbm_addrs[name])
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=f"mamba2_{case}",
        data=None,
        specified_data_order=order,
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
    )
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)
    run_and_assert(build_dir, f"mamba2_{case}", mlen=args.mlen, blen=args.blen)


# ============================================================================
# Cases
# ============================================================================


def case_dt(args, build_dir, hw):
    """dt = clamp(softplus(dt_raw + dt_bias)). Exercises V_SOFTPLUS_V."""
    mlen = args.mlen
    shape = _shape(args, seq_len=mlen)
    rows = args.num_heads

    g = torch.Generator().manual_seed(args.seed)
    dt_raw = _mxfp8_exact((rows, mlen), g)
    # Force exact powers of two into the first lanes so the softplus tails are
    # still exercised: these are exact in e4m3 at any magnitude, and they are
    # where a naive log1p(exp(x)) overflows (+16) or underflows (-16).
    for col, val in enumerate([16.0, -16.0, 8.0, -8.0, 0.0]):
        dt_raw[:, col] = val
    bias_row = _mxfp8_exact((1, mlen), g, hi=1.0)

    golden = torch.nn.functional.softplus(dt_raw + (0.0 if args.no_bias else bias_row))
    golden = torch.clamp(golden, min=args.dt_min, max=args.dt_max)

    prog = PlenaCompiler(mlen=mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    consts = prog.mamba_fp_constants(shape)
    fp_preload = prog.mamba_fp_constant_values(shape) + [0.0] * 3

    dt_in = prog.input("DT", shape=(rows, mlen))
    # The bias is logically ONE row, but H_PREFETCH_V moves a fixed
    # HBM_V_Prefetch_Amount x VLEN granule, so a 1-row HBM tensor is read a whole
    # granule wide and the surplus rows come from whatever follows it in HBM.
    # Staging a full granule (the row replicated) and taking a 1-row VRAM view
    # keeps every byte the DMA touches defined -- what a real deployment does
    # anyway when the host lays out per-head parameters.
    #
    # NOTE: this is *not* what produced the NaN this case used to emit. That was
    # the HBM base mismatch fixed in _finish (BIAS was staged 32 B before the
    # address the ISA reads it from, so the tail of the read fell in the scale
    # stream and 0x7f decoded to an e4m3 NaN). With hbm_addrs passed, a plain
    # 1-row BIAS also passes bit-exactly; the granule staging is kept as defence,
    # not as a workaround for a hardware limit.
    prefetch_rows = prog.hbm_v_prefetch_amount
    bias_staged = bias_row.repeat(prefetch_rows, 1)
    bias_in = prog.input("BIAS", shape=(prefetch_rows, mlen))
    dt_v = prog.load_batch(dt_in, name="dt")
    bias_full = prog.load_batch(bias_in, name="dt_bias_full")
    bias_v = prog.alloc_at("dt_bias", 1, mlen, prog.get_vram_addr(bias_full.name))

    prog.mamba_dt_activation_v0(dt_v, None if args.no_bias else bias_v, consts, shape, rows=list(range(rows)))

    _write_comparison(build_dir, prog, dt_v, rows=rows, cols=mlen, mlen=mlen)
    _finish(build_dir, prog, golden, {"DT": dt_raw, "BIAS": bias_staged}, fp_preload, ["DT", "BIAS"], "dt", args)


def case_cumsum(args, build_dir, hw):
    """cs[h, t] = sum_{s<=t} a[h, s] via M_MM against lower-triangular ones."""
    mlen = args.mlen
    shape = _shape(args, seq_len=mlen)
    rows = args.num_heads

    torch.manual_seed(args.seed)
    # a = A_h * dt is strictly negative in the real kernel; keep it so, and keep
    # |cs| in a range where bf16 still resolves it (cs feeds an exponent).
    g = torch.Generator().manual_seed(args.seed)
    # Strictly negative (A < 0 in the real kernel) and exactly representable.
    a = -(torch.randint(1, 9, (rows, mlen), generator=g).to(torch.float32) / 128.0)
    golden = torch.cumsum(a, dim=1)

    prog = PlenaCompiler(mlen=mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    prog.mamba_fp_constants(shape)
    fp_preload = prog.mamba_fp_constant_values(shape) + [0.0] * 3

    tri = torch.tril(torch.ones(mlen, mlen)).T  # U[s, t] = 1 iff s <= t
    a_in = prog.input("A", shape=(rows, mlen))
    tri_in = prog.input("TRI", shape=(mlen, mlen))
    a_v = prog.load_batch(a_in, name="a_t")
    cs_v = prog.alloc("cs_t", rows, mlen, strict=False)

    prog.ssd_chunk_cumsum_v0(a_v, tri_in, cs_v, shape, precision=HOST_STAGED)

    _write_comparison(build_dir, prog, cs_v, rows=rows, cols=mlen, mlen=mlen)
    _finish(build_dir, prog, golden, {"A": a, "TRI": tri}, fp_preload, ["A", "TRI"], "cumsum", args)


def case_decay(args, build_dir, hw):
    """decay[i, j] = exp(min(cs_i - cs_j, 0)) * causal[i, j] for one head.

    The stage that needs S_MAP_FP_V (the whole cs row into FPRAM in one
    instruction) and V_SUB_VF's rorder=1 reverse form (cs_i - cs_j from a row of
    cs_j and a scalar cs_i).
    """
    mlen = args.mlen
    shape = _shape(args, seq_len=mlen)

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    # `cs` is a *staged* input here, so it takes the HBM MX-FP8 path and keeps
    # three mantissa bits per element. A cumsum is exactly the shape e4m3 cannot
    # hold -- 64 partial sums spread over a decade of dynamic range above a fixed
    # step -- so drawing `a` off the natural -k/128 grid makes the reported error
    # the *input* quantisation rather than the kernel's: measured 0.18 max abs
    # (90% of the atol budget) while the kernel itself was within 1e-3 of
    # exp(min(cs_i - cs_j, 0)) evaluated on the cs the hardware actually saw. In
    # the real pipeline cs never round-trips through HBM; it comes out of
    # ssd_chunk_cumsum_v0 as bf16 in VRAM.
    # So draw a cumsum e4m3 represents exactly: steps of -1/8, few enough that
    # every partial sum stays a multiple of 1/8 with |cs| <= 2 -- the same
    # exactness argument as `_mxfp8_exact`. decay then spans exp(-2)..exp(0).
    a = torch.zeros(1, mlen)
    a[0, torch.randperm(mlen, generator=g)[: min(mlen, 16)]] = -0.125
    cs = torch.cumsum(a, dim=1)  # [1, mlen], non-increasing

    d = cs[0][:, None] - cs[0][None, :]
    golden = torch.exp(torch.clamp(d, max=0.0)) * torch.tril(torch.ones(mlen, mlen))

    prog = PlenaCompiler(mlen=mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    consts = prog.mamba_fp_constants(shape)
    fp_preload = prog.mamba_fp_constant_values(shape) + [0.0] * 3

    cs_in = prog.input("CS", shape=(1, mlen))
    causal_in = prog.input("CAUSAL", shape=(mlen, mlen))
    cs_v = prog.load_batch(cs_in, name="cs_t")
    causal_v = prog.load_batch(causal_in, name="causal")
    decay_v = prog.alloc("decay", mlen, mlen, strict=False)
    cs_fp = prog.fp_var("cs_fp", size=mlen)

    prog.ssd_decay_mask_v0(cs_v, cs_fp, decay_v, causal_v, consts, shape, head_row=0)

    _write_comparison(build_dir, prog, decay_v, rows=mlen, cols=mlen, mlen=mlen)
    _finish(
        build_dir,
        prog,
        golden,
        {"CS": cs, "CAUSAL": torch.tril(torch.ones(mlen, mlen))},
        fp_preload,
        ["CS", "CAUSAL"],
        "decay",
        args,
    )


def case_conv1d(args, build_dir, hw):
    """Causal depthwise conv1d, kernel 4, sequence on the VRAM row axis."""
    mlen = args.mlen
    seq = getattr(args, "seq_len", None) or 8
    shape = _shape(args, seq_len=seq)
    k = shape.conv_kernel

    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)
    x = _mxfp8_exact((seq, mlen), g)
    w = _mxfp8_exact((k, mlen), g, hi=1.0)

    golden = torch.zeros(seq, mlen)
    for s in range(seq):
        for j in range(k):
            src = s - (k - 1) + j
            if src >= 0:
                golden[s] += x[src] * w[j]

    prog = PlenaCompiler(mlen=mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    prog.mamba_fp_constants(shape)
    fp_preload = prog.mamba_fp_constant_values(shape) + [0.0] * 3

    x_in = prog.input("X", shape=(seq, mlen))
    w_in = prog.input("W", shape=(k, mlen))
    x_v = prog.load_batch(x_in, name="x")
    w_v = prog.load_batch(w_in, name="conv_w")
    out_v = prog.alloc("conv_out", seq, mlen, strict=False)
    scratch = prog.alloc("conv_scratch", 4, mlen, strict=False)

    prog.mamba_conv1d_v0(x_v, w_v, None, out_v, scratch, shape, num_rows=seq)

    _write_comparison(build_dir, prog, out_v, rows=seq, cols=mlen, mlen=mlen)
    _finish(build_dir, prog, golden, {"X": x, "W": w}, fp_preload, ["X", "W"], "conv1d", args)


CASES = {"dt": case_dt, "cumsum": case_cumsum, "decay": case_decay, "conv1d": case_conv1d}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--case", default="dt", choices=sorted(CASES))
    parser.add_argument("--no-bias", action="store_true", help="skip dt_bias (isolates V_SOFTPLUS_V)")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dt-min", type=float, default=0.0)
    parser.add_argument("--dt-max", type=float, default=float("inf"))
    args = parser.parse_args()

    build_dir = Path(__file__).parent / "build" / f"mamba2_{args.case}"
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(f"Mamba-2 stage test: {args.case}  (mlen={args.mlen}, blen={args.blen})")
    print("=" * 80)

    CASES[args.case](args, build_dir, hw)
