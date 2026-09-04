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
import os
import sys
from pathlib import Path

import torch

# Prefer the pinned in-repo submodule (PLENA_Simulator/PLENA_Compiler) over a
# sibling checkout: a sibling may be on a different branch and would otherwise
# silently shadow the submodule on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_compiler_override = os.environ.get("PLENA_COMPILER_ROOT")
_compiler_candidates = (
    [Path(_compiler_override)]
    if _compiler_override
    else [_REPO_ROOT / "PLENA_Compiler", _REPO_ROOT.parent / "PLENA_Compiler"]
)
_ACTIVE_COMPILER_ROOT = None
for _compiler_root in _compiler_candidates:
    if (_compiler_root / "aten" / "plena" / "compiler.py").exists():
        sys.path.insert(0, str(_compiler_root))
        _ACTIVE_COMPILER_ROOT = _compiler_root
        break
if _ACTIVE_COMPILER_ROOT is None:
    raise RuntimeError("PLENA Compiler checkout was not found")

from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_mamba_common import Mamba2Shape
from compiler.aten.plena.program_ssm_recurrent import MambaDecodeInvocation
from compiler.aten.plena.program_ssd import HOST_STAGED, SPILLED_ACTIVATION
from transactional_emulator.testbench.aten.configurable import (
    add_hw_args,
    bf16_uniform,
    setup_hw,
    use_plain_bf16_precision_classes,
    use_uniform_bf16_hbm_precision,
)
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


def use_bf16_kv_precision(build_dir: Path) -> None:
    """Make every program-produced spill use the uniform BF16 data path."""

    use_plain_bf16_precision_classes(
        build_dir,
        "HBM_M_KV_TYPE",
        "HBM_V_KV_TYPE",
    )


def _write_comparison(
    build_dir: Path,
    prog,
    var,
    *,
    rows: int,
    cols: int,
    mlen: int,
    atol: float = 0.0,
    rtol: float = 0.0,
):
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
        "atol": atol,
        "rtol": rtol,
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
        compiler_root=_ACTIVE_COMPILER_ROOT,
    )
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)
    high_water = getattr(prog, "_next_hbm_addr", 0)
    if high_water:
        (build_dir / "hbm_size.txt").write_text(str(high_water + 64 * 1024))
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


def case_decode_batch(args, build_dir, hw):
    """B independent recurrent states with one reused FPRAM window."""

    from compiler.aten.models.mamba2.reference import mamba2_recurrent_reference

    mlen = args.mlen
    batch = args.batch_size or 4
    if batch <= 0:
        raise ValueError("batch_size must be positive")
    heads = args.num_heads
    groups = 1
    state_size = mlen
    shape = Mamba2Shape(
        hidden_size=heads * mlen,
        num_heads=heads,
        head_dim=mlen,
        state_size=state_size,
        n_groups=groups,
        conv_kernel=4,
        chunk_size=mlen,
        seq_len=1,
        batch_size=batch,
    )
    g = torch.Generator().manual_seed(args.seed)
    cases = []
    for request in range(batch):
        x = _mxfp8_exact((heads, mlen), g, hi=0.5)
        state = _mxfp8_exact((heads, state_size, mlen), g, hi=0.25)
        b = _mxfp8_exact((groups, state_size), g, hi=0.25)
        c = _mxfp8_exact((groups, state_size), g, hi=0.25)
        dt = _mxfp8_exact((heads,), g, hi=0.25).abs() + 0.125
        da = 0.5 + ((torch.arange(heads) + request) % 3).float() * 0.125
        a = torch.log(da) / dt
        d = _mxfp8_exact((heads,), g, hi=0.25)
        y_ref, state_ref = mamba2_recurrent_reference(
            x=x[None, None],
            dt=dt[None, None],
            A=a,
            B=b.reshape(1, 1, groups, state_size),
            C=c.reshape(1, 1, groups, state_size),
            D=d,
            initial_state=state[None],
        )
        cases.append(
            {
                "x": x,
                "state": state,
                "b": b,
                "c": c,
                "dt": dt,
                "da": da,
                "d": d,
                "y_ref": y_ref[0, 0],
                "state_ref": state_ref[0],
            }
        )

    prog = PlenaCompiler(mlen=mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    up = lambda n: ((n + mlen - 1) // mlen) * mlen  # noqa: E731
    inputs = {}
    items = []
    b_fp = prog.fp_var("batch_b_window", size=state_size)
    c_fp = prog.fp_var("batch_c_window", size=state_size)
    # S_MAP_FP_V transfers a whole Vector row, so these head-scalar windows are
    # one MLEN row each even when only ``heads`` leading slots are consumed.
    da_fp = prog.fp_var("batch_da_window", size=mlen)
    dt_fp = prog.fp_var("batch_dt_window", size=mlen)
    d_fp = prog.fp_var("batch_d_window", size=mlen)
    consts = prog.mamba_fp_constants()
    const_values = prog.mamba_fp_constant_values(shape)
    fp_preload = [0.0] * (consts.zero.address + len(const_values))
    fp_preload[consts.zero.address : consts.zero.address + len(const_values)] = const_values
    for i, case in enumerate(cases):
        state_rows = up(heads * state_size)
        state_tensor = torch.zeros(state_rows, mlen)
        state_tensor[: heads * state_size] = case["state"].reshape(-1, mlen)
        x_rows = up(heads)
        x_tensor = torch.zeros(x_rows, mlen)
        x_tensor[:heads] = case["x"]
        # Five live scalar rows are kept in ordinary tensor storage and mapped
        # into one shared FPRAM window immediately before this request runs.
        scalar_tensor = torch.zeros(up(5), mlen)
        scalar_tensor[0, :state_size] = case["b"].reshape(-1)
        scalar_tensor[1, :state_size] = case["c"].reshape(-1)
        scalar_tensor[2, :heads] = case["da"]
        scalar_tensor[3, :heads] = case["dt"]
        scalar_tensor[4, :heads] = case["d"]
        inputs[f"STATE{i}"] = state_tensor
        inputs[f"X{i}"] = x_tensor
        inputs[f"SCALARS{i}"] = scalar_tensor
        state_in = prog.input(f"STATE{i}", shape=tuple(state_tensor.shape))
        x_in = prog.input(f"X{i}", shape=tuple(x_tensor.shape))
        scalars_in = prog.input(f"SCALARS{i}", shape=tuple(scalar_tensor.shape))
        # The generic test image writer emits its input tensors through the
        # activation MX path. Values are chosen exactly representable there, so
        # load through the matching path; persistent BF16 HBM state is covered
        # separately by the state load/store contract tests.
        state_v = prog.load_batch(state_in, name=f"state{i}")
        x_v = prog.load_batch(x_in, name=f"x{i}")
        items.append(
            (
                MambaDecodeInvocation(
                    state=state_v,
                    x=x_v,
                    b_fp=b_fp,
                    c_fp=c_fp,
                    da_fp=da_fp,
                    dt_fp=dt_fp,
                    d_fp=d_fp,
                    y=prog.alloc(f"y{i}", x_rows, mlen),
                    scratch=prog.alloc(f"scratch{i}", mlen, mlen),
                ),
                prog.load_batch(scalars_in, name=f"scalars{i}"),
            )
        )

    for request, (item, scalars) in enumerate(items):
        prog.emit_comment(f"static streamed Mamba batch request={request}")
        prog.tile_row_to_fpram(scalars, b_fp, rows=[0])
        prog.tile_row_to_fpram(scalars, c_fp, rows=[1])
        prog.tile_row_to_fpram(scalars, da_fp, rows=[2])
        prog.tile_row_to_fpram(scalars, dt_fp, rows=[3])
        prog.tile_row_to_fpram(scalars, d_fp, rows=[4])
        prog.ssm_decode_step_v0(
            state=item.state,
            x=item.x,
            b_fp=item.b_fp,
            c_fp=item.c_fp,
            da_fp=item.da_fp,
            dt_fp=item.dt_fp,
            d_fp=item.d_fp,
            y=item.y,
            scratch=item.scratch,
            shape=shape.single_sequence(),
            consts=consts,
        )

    active_rows = batch * (heads + heads * state_size)
    packed = prog.alloc("batch_result", up(active_rows), mlen)
    golden_rows = []
    dst = 0
    for (item, _), case in zip(items, cases):
        for h in range(heads):
            prog.mamba_row_copy(packed, dst, item.y, h)
            golden_rows.append(case["y_ref"][h])
            dst += 1
        for row in range(heads * state_size):
            prog.mamba_row_copy(packed, dst, item.state, row)
            golden_rows.append(case["state_ref"].reshape(-1, mlen)[row])
            dst += 1
    golden = torch.stack(golden_rows)
    _write_comparison(
        build_dir,
        prog,
        packed,
        rows=active_rows,
        cols=mlen,
        mlen=mlen,
        atol=3e-2,
        rtol=3e-2,
    )
    _finish(
        build_dir,
        prog,
        golden,
        inputs,
        fp_preload,
        list(inputs),
        "decode_batch",
        args,
    )


def case_prefill_s128_full(args, build_dir, hw):
    """Execute both SSD chunks and read back every prompt output plus state."""

    from compiler.aten.models.mamba2.reference import ssd_chunk_reference

    mlen = args.mlen
    if mlen != 64:
        raise SystemExit(f"this S128 evidence requires MLEN=64, got {mlen}")
    chunks = 2
    total = chunks * mlen
    shape = Mamba2Shape(
        hidden_size=mlen,
        num_heads=1,
        head_dim=mlen,
        state_size=mlen,
        n_groups=1,
        conv_kernel=4,
        chunk_size=mlen,
        seq_len=total,
    )
    use_uniform_bf16_hbm_precision(build_dir)
    generator = torch.Generator().manual_seed(args.seed + 701)
    x = bf16_uniform((1, total, 1, mlen), generator, hi=0.125)
    b = bf16_uniform((1, total, 1, mlen), generator, hi=0.125)
    c = bf16_uniform((1, total, 1, mlen), generator, hi=0.125)
    state0 = bf16_uniform((1, 1, mlen, mlen), generator, hi=0.125)
    dt = torch.full((1, total, 1), 0.25)
    a = torch.tensor([-0.5])
    d = torch.tensor([0.125])
    output_ref, state_ref = ssd_chunk_reference(
        x,
        dt,
        a,
        b,
        c,
        d,
        mlen,
        initial_state=state0,
    )

    prog = PlenaCompiler(
        mlen=mlen,
        blen=args.blen,
        real_data_ratio=hw.real_data_ratio,
    )
    consts = prog.mamba_fp_constants(shape)
    inputs = {
        "S0": state0[0, 0],
        "TRI": torch.tril(torch.ones(mlen, mlen)).T,
        "CAUSAL": torch.tril(torch.ones(mlen, mlen)),
    }
    for chunk in range(chunks):
        start = chunk * mlen
        end = start + mlen
        x_chunk = x[0, start:end, 0]
        inputs[f"X{chunk}"] = x_chunk
        inputs[f"XDT{chunk}"] = (x_chunk * dt[0, start:end, 0, None]).to(torch.bfloat16).float()
        inputs[f"B{chunk}"] = b[0, start:end, 0]
        inputs[f"BT{chunk}"] = b[0, start:end, 0].T.contiguous()
        inputs[f"C{chunk}"] = c[0, start:end, 0]
        inputs[f"AT{chunk}"] = (dt[0, start:end].T * a[:, None]).to(torch.bfloat16).float().contiguous()

    staged = {
        name: prog.input(
            name,
            shape=tuple(value.shape),
            real_data_ratio=1.0,
            hbm_element_bytes=2,
        )
        for name, value in inputs.items()
    }
    state = prog.load_batch(staged["S0"], name="state")
    state_prev = staged["S0"]
    causal = prog.load_batch(staged["CAUSAL"], name="causal")
    outputs = []
    scale_vars = []
    decay_vars = []
    d_fp = prog.fp_var("skip_d", size=1)

    for chunk in range(chunks):
        a_t = prog.load_batch(staged[f"AT{chunk}"], name=f"a_t{chunk}")
        x_v = prog.load_batch(staged[f"X{chunk}"], name=f"x{chunk}")
        c_v = prog.load_batch(staged[f"C{chunk}"], name=f"c{chunk}")
        b_t = prog.load_batch(staged[f"BT{chunk}"], name=f"b_t{chunk}")
        cs = prog.alloc(f"cs{chunk}", 1, mlen, strict=False)
        decay = prog.alloc(f"decay{chunk}", mlen, mlen, strict=False)
        scores = prog.alloc(f"scores{chunk}", mlen, mlen, strict=False)
        output = prog.alloc(f"output{chunk}", mlen, mlen, strict=False)
        cs_fp = prog.fp_var(f"cs_fp{chunk}", size=mlen)

        prog.ssd_chunk_cumsum_v0(
            a_t,
            staged["TRI"],
            cs,
            shape,
            precision=SPILLED_ACTIVATION,
        )
        prog.ssd_decay_mask_v0(
            cs,
            cs_fp,
            decay,
            causal,
            consts,
            shape,
            head_row=0,
        )
        prog.ssd_chunk_head_v0(
            b_chunk=staged[f"B{chunk}"],
            c_chunk=c_v,
            x_chunk=staged[f"XDT{chunk}"],
            decay=decay,
            scores=scores,
            y_out=output,
            shape=shape,
            precision=SPILLED_ACTIVATION,
        )
        prog.ssd_inter_chunk_output_v0(
            c_chunk=c_v,
            state_prev=state_prev,
            y_out=output,
            cs_fp=cs_fp,
            shape=shape,
            precision=SPILLED_ACTIVATION,
        )
        prog.tile_row_fma_fp_broadcast(
            output,
            x_v,
            d_fp,
            dst_rows=range(mlen),
            src_rows=range(mlen),
        )
        outputs.append(output)

        start = chunk * mlen
        end = start + mlen
        cs_host = torch.cumsum((dt[0, start:end].T * a[:, None]), dim=1)[0]
        x_d_scale = torch.exp(cs_host[-1] - cs_host) * dt[0, start:end, 0]
        scale_fp = prog.fp_var(f"x_d_scale{chunk}", size=mlen)
        scale_vars.append((scale_fp, x_d_scale))
        x_d = prog.alloc(f"x_d{chunk}", mlen, mlen, strict=False)
        prog.mamba_block_copy(x_d, x_v, num_rows=mlen)
        prog.tile_row_mul_fp(x_d, scale_fp, rows=range(mlen))
        x_d_spill = prog.store(
            x_d,
            name=f"x_d_spill{chunk}",
            precision=1,
            hbm_element_bytes=2,
            real_data_ratio=1.0,
        )
        chunk_decay = prog.fp_var(f"chunk_decay{chunk}", size=1)
        decay_vars.append((chunk_decay, torch.exp(cs_host[-1]).item()))
        prog.ssd_state_update_v0(
            state=state,
            b_t_chunk=b_t,
            x_d_chunk=x_d_spill,
            decay_fp=chunk_decay,
            shape=shape,
            precision=SPILLED_ACTIVATION,
        )
        state_prev = prog.store(
            state,
            name=f"state_after_chunk{chunk}",
            precision=1,
            hbm_element_bytes=2,
            real_data_ratio=1.0,
        )

    packed = prog.alloc("prefill_full_result", total + mlen, mlen, strict=False)
    packed_row = 0
    for output in outputs:
        for row in range(mlen):
            prog.mamba_row_copy(packed, packed_row, output, row)
            packed_row += 1
    for row in range(mlen):
        prog.mamba_row_copy(packed, packed_row, state, row)
        packed_row += 1
    golden = torch.cat((output_ref[0, :, 0], state_ref[0, 0]), dim=0)

    tail = max(
        [consts.zero.address + len(prog.mamba_fp_constant_values(shape)), d_fp.address + 1]
        + [var.address + var.size for var, _ in scale_vars]
        + [var.address + 1 for var, _ in decay_vars]
    )
    fp = [0.0] * tail
    constants = prog.mamba_fp_constant_values(shape)
    fp[consts.zero.address : consts.zero.address + len(constants)] = constants
    fp[d_fp.address] = d.item()
    for var, values in scale_vars:
        fp[var.address : var.address + var.size] = values.tolist()
    for var, value in decay_vars:
        fp[var.address] = value

    _write_comparison(
        build_dir,
        prog,
        packed,
        rows=total + mlen,
        cols=mlen,
        mlen=mlen,
        atol=5e-3,
        rtol=3e-2,
    )
    _finish(
        build_dir,
        prog,
        golden,
        inputs,
        fp,
        list(inputs),
        "prefill_s128_full",
        args,
    )


def _prefill_decode_handoff(args, build_dir, hw, *, chunks: int):
    """One or more prefill chunks feed decode through packet addressing."""

    from compiler.aten.models.mamba2.reference import mamba2_recurrent_reference
    from compiler.aten.plena.affine_layout import AffineLayout, LayoutKind

    mlen = args.mlen
    shape = Mamba2Shape(
        hidden_size=mlen,
        num_heads=1,
        head_dim=mlen,
        state_size=mlen,
        n_groups=1,
        conv_kernel=4,
        chunk_size=mlen,
        seq_len=mlen * chunks,
    )
    g = torch.Generator().manual_seed(args.seed)
    state0 = _mxfp8_exact((mlen, mlen), g, hi=0.125)
    b_t_chunks = [_mxfp8_exact((mlen, mlen), g, hi=0.125) for _ in range(chunks)]
    x_d_chunks = [_mxfp8_exact((mlen, mlen), g, hi=0.125) for _ in range(chunks)]
    chunk_decays = [0.5 + 0.125 * (index % 2) for index in range(chunks)]
    state_after_prefill = state0
    for chunk_decay, b_t, x_d in zip(chunk_decays, b_t_chunks, x_d_chunks, strict=True):
        state_after_prefill = chunk_decay * state_after_prefill + b_t @ x_d

    x = _mxfp8_exact((1, mlen), g, hi=0.125)
    b = _mxfp8_exact((1, mlen), g, hi=0.125)
    c = _mxfp8_exact((1, mlen), g, hi=0.125)
    dt = torch.tensor([0.25])
    a = torch.tensor([-0.5])
    da = torch.exp(a * dt)
    d = torch.tensor([0.125])
    y_ref, state_ref = mamba2_recurrent_reference(
        x=x[None, None],
        dt=dt[None, None],
        A=a,
        B=b.reshape(1, 1, 1, mlen),
        C=c.reshape(1, 1, 1, mlen),
        D=d,
        initial_state=state_after_prefill.reshape(1, 1, mlen, mlen),
    )

    prog = PlenaCompiler(
        mlen=mlen,
        blen=args.blen,
        real_data_ratio=hw.real_data_ratio,
        stream_addressing=True,
        stream_packetized=True,
        stream_affine_alpha=1,
        stream_storage_atom=4,
        stream_packet_elements=mlen,
    )
    inputs = {
        "S0": state0,
        "X": x,
        "I": torch.eye(mlen),
        **{f"BT{index}": value for index, value in enumerate(b_t_chunks)},
        **{f"XD{index}": value for index, value in enumerate(x_d_chunks)},
    }
    staged = {name: prog.input(name, shape=tuple(value.shape)) for name, value in inputs.items()}
    state = prog.load_batch(staged["S0"], name="state")
    x_v = prog.load_batch(staged["X"], name="x")

    chunk_decay_fp = []
    for index in range(chunks):
        b_t_v = prog.load_batch(staged[f"BT{index}"], name=f"b_t{index}")
        decay_fp = prog.fp_var(f"chunk_decay{index}", size=1)
        chunk_decay_fp.append(decay_fp)
        prog.ssd_state_update_v0(
            state=state,
            b_t_chunk=b_t_v,
            x_d_chunk=staged[f"XD{index}"],
            decay_fp=decay_fp,
            shape=shape,
            precision=HOST_STAGED,
        )

    state_layout = AffineLayout(
        kind=LayoutKind.AFFINE_SKEW,
        groups=1,
        fields=1,
        majors=mlen,
        minors=mlen,
        alpha=1,
        major_packed=True,
    )
    affine_state = prog.alloc("affine_state", mlen, mlen, strict=False)
    prog.vram_identity_relayout_to(
        source=state,
        identity=staged["I"],
        out=affine_state,
        output_layout=state_layout,
        **HOST_STAGED,
    )
    state = affine_state

    b_fp = prog.fp_var("b", size=mlen)
    c_fp = prog.fp_var("c", size=mlen)
    da_fp = prog.fp_var("da", size=1)
    dt_fp = prog.fp_var("dt", size=1)
    d_fp = prog.fp_var("d", size=1)
    y = prog.alloc("y", 1, mlen, strict=False)
    scratch = prog.alloc("decode_scratch", mlen, mlen)
    consts = prog.mamba_fp_constants(shape)
    prog.ssm_decode_step_v0(
        state=state,
        x=x_v,
        b_fp=b_fp,
        c_fp=c_fp,
        da_fp=da_fp,
        dt_fp=dt_fp,
        d_fp=d_fp,
        y=y,
        scratch=scratch,
        shape=shape.single_sequence(),
        consts=consts,
    )

    state_result = prog.alloc("state_row_major", mlen, mlen, strict=False)
    state_copy_ones = prog.fp_var("state_copy_ones", size=mlen)
    rows = list(range(mlen))
    prog.vram_fill_zero(state_result, rows=rows)
    prog.tile_row_fma_fp_sweep(
        state_result,
        state,
        state_copy_ones,
        dst_rows=rows,
        src_rows=rows,
    )

    packed = prog.alloc("handoff_result", mlen + 1, mlen, strict=False)
    prog.mamba_row_copy(packed, 0, y, 0)
    for row in range(mlen):
        prog.mamba_row_copy(packed, row + 1, state_result, row)
    golden = torch.cat((y_ref[0, 0].reshape(1, mlen), state_ref[0, 0]), dim=0)

    fp = [0.0] * (consts.zero.address + len(prog.mamba_fp_constant_values(shape)))
    for decay_fp, value in zip(chunk_decay_fp, chunk_decays, strict=True):
        fp[decay_fp.address] = value
    fp[b_fp.address : b_fp.address + mlen] = b.flatten().tolist()
    fp[c_fp.address : c_fp.address + mlen] = c.flatten().tolist()
    fp[da_fp.address] = da.item()
    fp[dt_fp.address] = dt.item()
    fp[d_fp.address] = d.item()
    values = prog.mamba_fp_constant_values(shape)
    fp[consts.zero.address : consts.zero.address + len(values)] = values
    fp.extend([0.0] * max(0, state_copy_ones.address + state_copy_ones.size - len(fp)))
    fp[state_copy_ones.address : state_copy_ones.address + state_copy_ones.size] = [1.0] * state_copy_ones.size

    code = prog.compile()
    if "L_CFG" not in code:
        raise AssertionError("prefill/decode handoff did not exercise L-Compute packet addressing")
    _write_comparison(
        build_dir,
        prog,
        packed,
        rows=mlen + 1,
        cols=mlen,
        mlen=mlen,
        atol=1e-2,
        rtol=3e-2,
    )
    _finish(
        build_dir,
        prog,
        golden,
        inputs,
        fp,
        list(inputs),
        "prefill_decode_handoff",
        args,
    )


def case_prefill_decode_handoff(args, build_dir, hw):
    """One 64-token chunk, then one recurrent decode token."""

    _prefill_decode_handoff(args, build_dir, hw, chunks=1)


def case_prefill_s128_decode_handoff(args, build_dir, hw):
    """Two 64-token chunks, then one recurrent decode token."""

    if args.mlen != 64:
        raise SystemExit(f"this transactional S128 case requires mlen=64, got {args.mlen}")
    _prefill_decode_handoff(args, build_dir, hw, chunks=2)


CASES = {
    "dt": case_dt,
    "cumsum": case_cumsum,
    "decay": case_decay,
    "conv1d": case_conv1d,
    "decode_batch": case_decode_batch,
    "prefill_s128_full": case_prefill_s128_full,
    "prefill_decode_handoff": case_prefill_decode_handoff,
    "prefill_s128_decode_handoff": case_prefill_s128_decode_handoff,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--case", default="dt", choices=sorted(CASES))
    parser.add_argument("--no-bias", action="store_true", help="skip dt_bias (isolates V_SOFTPLUS_V)")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dt-min", type=float, default=0.0)
    parser.add_argument("--dt-max", type=float, default=float("inf"))
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="override the generated test directory (useful for disposable evidence runs)",
    )
    args = parser.parse_args()

    build_dir = args.build_dir or (Path(__file__).parent / "build" / f"mamba2_{args.case}")
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(f"Mamba-2 stage test: {args.case}  (mlen={args.mlen}, blen={args.blen})")
    print("=" * 80)

    CASES[args.case](args, build_dir, hw)
