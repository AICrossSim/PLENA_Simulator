# ruff: noqa: E402
"""Numerical stage tests for the KDA lowering, on the transactional emulator.

Every KDA test until now ran on `aten/tests/isa_interpreter.py`, which is
float64 and says so. That is the right oracle for "did the lowering emit the
arithmetic I meant", and the wrong one for "does it survive the hardware's
bf16". This file closes that gap: it compiles one stage, assembles it, runs the
Rust emulator, and compares against a float32 PyTorch golden.

    covered      the emitters in aten/plena/program_kda_chunk.py end to end
                 through assembler + emulator, V_FMA_VF as the emulator really
                 executes it, and the bf16 rounding applied on every vector
                 write.

    NOT covered  the RTL's fixed-point exp model -- same caveat as the Mamba
                 stage tests: the emulator uses libtorch's exact exp while
                 PLENA_Tools' quant model is a range reduction plus a 3-term
                 Taylor with systematic error. KDA applies exp on the critical
                 path of a multiplicative recurrence, where that compounds.

Cases:
    cumprod   A_t = prod_{s<=t} a_s, per key channel -- the cumulative decay
    ut        T = (I + tril(diag(beta) M, -1))^-1 diag(beta) -- exercises
              V_FMA_VF's accumulate inside a hardware loop, which is the whole
              reason the opcode exists

Run:  python3 kda_stage_test.py --case cumprod [--mlen 64] [--blen 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
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

from compiler.aten.models.kda.shape import KdaShape
from compiler.aten.plena import PlenaCompiler
from compiler.aten.plena.program_kda_chunk import kda_chunk_cols, kda_chunk_rows
from compiler.aten.plena.program_kda_prefill import (
    KdaPrefillBuffers,
    kda_prefill_state_transpose_shapes,
    kda_prefill_tile_shapes,
)
from transactional_emulator.testbench.aten.configurable import add_hw_args, setup_hw
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env

#: Slot order of the KDA FPRAM constant block. kda_fp_constants reuses
#: MambaFPConstants, so this mirrors FP_CONST_ORDER in the Mamba stage test.
#: Pinned here because FPRAM contents are never checked at runtime.
FP_CONST_ORDER = ["zero", "one", "neg_one", "dt_min", "dt_max", "reci_group", "eps"]


def use_bf16_kv_precision(build_dir: Path) -> None:
    """Rewrite the per-build TOML's KV classes to Plain BF16.

    The prefill path spills intermediates to HBM and reads them back as MRAM,
    because a matmul's second operand can only arrive that way. Those spills go
    through the ``keyvalue`` precision class, and the shipped
    ``[TRANSACTIONAL.PRECISION]`` makes that Mx/e4m3 with a separate scale
    stream. ``SPILLED_ACTIVATION`` sets ``set_scale=False``, so the read walks
    off into the scale stream, whose ``0x7f`` bytes decode to e4m3 NaN -- which
    is what this test produced before the rewrite: every output NaN.

    ``ProgramSSDMixin.require_bf16_kv_precision`` documents exactly this and is
    called by nothing; the Mamba prefill path has the same exposure. The state
    is a multiplicative accumulator carried across chunks, so three mantissa
    bits at every boundary compounds without bound even where it does not NaN.
    """
    path = Path(os.environ["PLENA_SETTINGS_TOML"])
    text = path.read_text()
    for key in ("HBM_M_KV_TYPE", "HBM_V_KV_TYPE"):
        head = f"[TRANSACTIONAL.PRECISION.{key}]"
        start = text.index(head)
        end = text.index("\n[", start + len(head))
        text = (
            text[:start]
            + (
                f'{head}\nformat = "Plain"\n'
                f"[TRANSACTIONAL.PRECISION.{key}.DATA_TYPE]\n"
                f'type = "Fp"\nsign = true\nexponent = 8\nmantissa = 7\n'
            )
            + text[end + 1 :]
        )
    path.write_text(text)


def _mxfp8_exact(shape_, generator, *, hi: float = 2.0) -> torch.Tensor:
    """Values MX-FP8 e4m3 represents exactly, so the reported error is the
    kernel's and not the input staging's. Same trick as the Mamba stage test."""
    steps = int(hi * 8)
    raw = torch.randint(-steps, steps + 1, shape_, generator=generator)
    return raw.to(torch.float32) / 8.0


def _shape(args, *, key_dim: int) -> KdaShape:
    return KdaShape(
        hidden_size=args.num_heads * args.mlen,
        num_heads=args.num_heads,
        key_dim=key_dim,
        value_dim=args.mlen,
        conv_kernel=4,
    )


def _write_comparison(
    build_dir: Path, prog, var, *, rows: int, cols: int, mlen: int, atol: float = 0.0, rtol: float = 0.0
):
    """Point the golden comparison at `rows x cols` of `var`, whatever its width.

    A VRAM matrix wider than `mlen` is stored column-block-major: block `c`
    starts `physical_rows * mlen` elements in, not `rows * mlen`. `check_mem`'s
    stride mode already knows this and takes the block pitch as
    `col_block_stride`, but only if `physical_rows` is handed to it -- it
    otherwise assumes `num_batches`, which is the logical row count and is
    smaller than the physical one for every padded tile. Passing it is what
    makes a multi-key-block state comparable at all; without it the second block
    is read from `rows * mlen` and every value past the first block is somebody
    else's.

    `num_rows` covers the whole tile rather than `rows * cols / mlen`, because
    the reorder has to reach the last block's live rows and those sit past the
    live rows of every block before it.
    """
    addr = prog.get_vram_addr(var.name)
    physical_rows = var.physical_shape[0]
    col_blocks = max(1, -(-cols // mlen))
    params = {
        "start_row_idx": addr // mlen,
        "num_rows": physical_rows * col_blocks if cols > mlen else (rows * cols) // mlen,
        "num_batches": rows,
        "elements_per_batch": cols,
        "physical_rows": physical_rows,
        "row_dim": mlen,
        "use_stride_mode": cols > mlen,
        "atol": atol,
        "rtol": rtol,
    }
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(params, f, indent=2)
    return addr


def _finish(
    build_dir: Path,
    prog,
    golden,
    input_tensors,
    fp_preload,
    order,
    case,
    args,
    *,
    tensor_layouts=None,
):
    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA")
    create_sim_env(
        input_tensors,
        gen_code,
        {"original_output": golden},
        fp_preload,
        build_dir=str(build_dir),
        tensor_layouts=tensor_layouts,
    )
    hbm_addrs = {name: prog.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    order = sorted(order, key=lambda name: hbm_addrs[name])
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=f"kda_{case}",
        data=None,
        specified_data_order=order,
        build_path=build_dir,
        input_tensors=input_tensors,
        hbm_addrs=hbm_addrs,
        tensor_layouts=tensor_layouts,
        compiler_root=_ACTIVE_COMPILER_ROOT,
    )
    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)
    # The prefill path spills intermediates to HBM (MRAM is writable only by
    # H_PREFETCH_M, so a matmul's second operand has to arrive that way), and
    # those allocations sit past everything create_mem_for_sim staged. Without
    # this the emulator indexes past its HBM vector and panics rather than
    # reporting a size problem. `run_emulator` reads this file.
    high_water = getattr(prog, "_next_hbm_addr", 0)
    if high_water:
        pad = 64 * 1024
        (build_dir / "hbm_size.txt").write_text(str(high_water + pad))
    run_and_assert(build_dir, f"kda_{case}", mlen=args.mlen, blen=args.blen)
    _assert_every_value_within_tolerance(build_dir, case)


def _assert_every_value_within_tolerance(build_dir: Path, case: str):
    """Fail unless *every* value is inside the tolerance the case asked for.

    `run_and_assert` goes through `check_mem`, whose `allclose_pass` is
    `match_rate >= 90.0`. That is a shared utility with many callers and this is
    not the place to change what it means -- but a tenth of the values being out
    of tolerance is not a pass for a kernel, and the gap is not academic: the
    output scale was applied to only the first value block, leaving half of every
    token 11x too large, and the case still reported PASSED at a 94.68% match
    rate because the data is O(1e-3) and the tolerance is 5e-2.

    So the threshold is restated here, on the same numbers, at 100%.
    """
    from transactional_emulator.testbench.emulator_runner import compare_emulator_output

    results, params = compare_emulator_output(build_dir)
    rate = results.get("allclose_match_rate")
    if rate is None:
        raise SystemExit(f"kda_{case}: no match rate to check")
    if rate < 100.0:
        raise SystemExit(
            f"\n[kda_{case} FAILED] {rate:.2f}% of values inside "
            f"atol={params.get('atol')} rtol={params.get('rtol')}; this check "
            f"wants all of them. check_mem's own bar is 90%, which is why "
            f"run_and_assert did not object."
        )
    print(f"[kda_{case}] every value inside tolerance ({rate:.2f}%)")


# ============================================================================
# Cases
# ============================================================================


def case_cumprod(args, build_dir, hw):
    """A_t = prod_{s<=t} a_s, per key channel.

    A running product and not exp of a running sum: the cumulative log-decay
    reaches chunk*gate_lower_bound = -80, where bf16's ulp is 0.31, so
    exponentiating a stored sum costs a 17% relative error on A. See
    program_kda_chunk's module docstring for the measurement.

    Drawn from exact e4m3 values in [1/2, 1] so the staging quantisation is the
    identity and the remaining error is the kernel's. Unlike the Mamba stage
    cases this cannot be bit-exact: the *inputs* are exact, but a product of
    sixteen 3-mantissa-bit values is not representable in bf16's 8, so each of
    the fifteen multiplies rounds. Observed error is one bf16 ulp (1.95e-3
    absolute, 0.6% relative), which is the floor, not a defect. The tolerance is
    set a few ulp above it -- a real error in the scan is orders of magnitude
    larger, as the mutation tests in test_kda_prefill.py show.
    """
    mlen = args.mlen
    chunk = args.chunk
    key_dim = args.key_dim or mlen
    shape = _shape(args, key_dim=key_dim)
    rows = kda_chunk_rows(shape, mlen, chunk)
    cols = kda_chunk_cols(shape, mlen)

    g = torch.Generator().manual_seed(args.seed)
    # Decays are in (0, 1]; keep them exactly representable and not so small
    # that a 16-fold product underflows out of the comparison's range.
    a = torch.randint(4, 9, (chunk, key_dim), generator=g).to(torch.float32) / 8.0
    golden = torch.cumprod(a, dim=0)

    prog = PlenaCompiler(mlen=mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    prog.kda_fp_constants()
    fp_preload = prog.kda_fp_constant_values() + [0.0] * 3

    # One row per timestep, key on lanes. At key_dim > mlen the staged tensor is
    # wider than a VRAM row, and load_batch lays it out column-block-major --
    # which is exactly how the scan reads it.
    a_in = prog.input("A", shape=(rows, cols))
    decay = prog.load_batch(a_in, name="decay")
    prev = prog.alloc("prev", rows, cols, strict=False)

    prog.kda_chunk_decay_cumprod_v0(decay=decay, prev=prev, chunk=chunk, shape=shape)

    # Rows past `chunk` are staged as 1.0, not 0: the scan does not touch them,
    # but a cumulative product that leaked into them would be visible.
    staged = torch.nn.functional.pad(a, (0, 0, 0, rows - chunk), value=1.0)
    _write_comparison(build_dir, prog, decay, rows=chunk, cols=cols, mlen=mlen, atol=0.0, rtol=2e-2)
    _finish(build_dir, prog, golden, {"A": staged}, fp_preload, ["A"], "cumprod", args)


def case_ut(args, build_dir, hw):
    """T = (I + tril(diag(beta) M, -1))^-1 diag(beta), by forward substitution.

    The stage that exercises V_FMA_VF's accumulate inside a hardware loop --
    T[i] = beta_i (e_i - sum_{j<i} M[i,j] T[j]) is one loop per row, and the
    accumulate is the instruction rather than a scratch round-trip.

    Not bit-exact: the substitution is a chain of length `chunk`, so bf16
    rounding compounds down the rows. The tolerance is on the result, which is
    what matters.
    """
    mlen = args.mlen
    chunk = args.chunk
    shape = _shape(args, key_dim=mlen)
    if chunk > mlen:
        raise SystemExit(f"chunk {chunk} must not exceed mlen {mlen}")

    g = torch.Generator().manual_seed(args.seed)
    m_dense = torch.tril(_mxfp8_exact((chunk, chunk), g, hi=0.5), -1)
    beta = _mxfp8_exact((chunk,), g, hi=1.0).abs().clamp(0.125, 1.0)
    golden = torch.linalg.inv(torch.eye(chunk) + torch.diag(beta) @ m_dense) @ torch.diag(beta)
    golden = torch.nn.functional.pad(golden, (0, mlen - chunk))

    prog = PlenaCompiler(mlen=mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    consts = prog.kda_fp_constants()
    fp_preload = prog.kda_fp_constant_values() + [0.0] * 3

    m_in = prog.input("M", shape=(chunk, mlen))
    i_in = prog.input("I", shape=(chunk, mlen))
    m_v = prog.load_batch(m_in, name="m")
    i_v = prog.load_batch(i_in, name="ident")
    t_out = prog.alloc("t_out", chunk, mlen, strict=False)
    m_fp = prog.fp_var("m_fp", size=mlen)
    beta_fp = prog.fp_var("beta_fp", size=mlen)

    prog.kda_ut_transform_v0(
        m=m_v,
        identity=i_v,
        beta_fp=beta_fp,
        t_out=t_out,
        m_fp=m_fp,
        consts=consts,
        chunk=chunk,
        shape=shape,
    )

    # beta lands in FPRAM after the constants; the preload is positional.
    fp_preload = fp_preload + [0.0] * (beta_fp.address - len(fp_preload)) + beta.tolist()

    m_padded = torch.nn.functional.pad(m_dense, (0, mlen - chunk))
    identity = torch.nn.functional.pad(torch.eye(chunk), (0, mlen - chunk))
    _write_comparison(build_dir, prog, t_out, rows=chunk, cols=mlen, mlen=mlen, atol=2e-2, rtol=2e-2)
    _finish(
        build_dir,
        prog,
        golden,
        {"M": m_padded, "I": identity},
        fp_preload,
        ["M", "I"],
        "ut",
        args,
    )


def _sequential_chunk(q, k, v, beta, per_step_decay, state0, scale):
    """The sequential gated delta rule, from reference.py::kda_step, one head.

    The oracle for prefill is not a chunked reference -- there is none;
    `kda_state_engine_prefill` is a loop of steps. It is *this*: chunked prefill
    must equal the recurrence it collapses.
    """
    s = state0.clone()
    outs = []
    for t in range(q.shape[0]):
        d = s * per_step_decay[t]
        e = beta[t] * (v[t] - d @ k[t])
        s = d + torch.outer(e, k[t])
        outs.append(scale * (s @ q[t]))
    return torch.stack(outs), s


def _prefill_case(args, build_dir, hw, *, want: str, chunks: int = 1):
    """``chunks`` chunks of prefill for one head, against the sequential recurrence.

    More than one matters for two reasons the single-chunk case cannot show.
    The spill regions are **reused** across chunks, so a second chunk's
    prefetch reads whatever the first left behind -- correct only because each
    spill zero-fills the rows past its live data before storing. And the carried
    state is the thing that could compound; it does not, but only running it
    proves that.
    """
    mlen, chunk = args.mlen, args.chunk
    key_dim = args.key_dim or mlen
    value_dim = args.value_dim or mlen
    shape = KdaShape(
        hidden_size=value_dim,
        num_heads=1,
        key_dim=key_dim,
        value_dim=value_dim,
        conv_kernel=4,
    )
    use_bf16_kv_precision(build_dir)
    total = chunk * chunks
    if chunk > mlen:
        raise SystemExit(f"chunk {chunk} must not exceed mlen {mlen}")

    g = torch.Generator().manual_seed(args.seed)
    # L2-normalised, because that is what reaches the recurrence: reference.py's
    # kda_step normalises q and k with rsqrt(sum + 1e-6), and the lowering does
    # the same in kda_mixer_step_v0. Feeding raw projections instead made |k| ~ 5
    # and drove cond(L) to the hundreds -- an artefact of the test, not a
    # property of the workload. Normalised, cond(L) is 1.0 at every chunk size.
    _norm = lambda t: t * torch.rsqrt(t.square().sum(-1, keepdim=True) + 1e-6)  # noqa: E731
    q = _norm(_mxfp8_exact((total, key_dim), g, hi=1.0))
    k = _norm(_mxfp8_exact((total, key_dim), g, hi=1.0))
    v = _mxfp8_exact((total, value_dim), g, hi=1.0)
    beta = _mxfp8_exact((total,), g, hi=1.0).abs().clamp(0.125, 1.0)
    # Per-step decays. exp(lower_bound * sigmoid(.)) with lower_bound = -5 puts
    # these across (e^-5, 1); the real distribution centres near exp(-2.5), which
    # makes the recurrence strongly contracting -- the reason errors do not
    # compound across chunks.
    a = torch.exp(shape.gate_lower_bound * torch.sigmoid(torch.randn((total, key_dim), generator=g)))
    state0 = _mxfp8_exact((value_dim, key_dim), g, hi=0.5)
    scale = 1.0 / key_dim**0.5
    out_ref, state_ref = _sequential_chunk(q, k, v, beta, a, state0, scale)

    prog = PlenaCompiler(mlen=mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    consts = prog.kda_fp_constants()
    fp_preload = prog.kda_fp_constant_values() + [0.0] * 3

    # Rows past `chunk` are POISON, not zero. Every spilled tile is prefetched
    # as a whole mlen x mlen MRAM block, so the rows past its live data are read
    # by the matmul whether or not anything wrote them -- and
    # kda_prefill_state_tail_v0's projection contracts over exactly those rows,
    # straight into the carried state. kda_prefill_spill_v0 zeroes them before
    # each store; with zeros here instead, deleting that zero-fill left every
    # case passing, because a fresh VRAM tile is already zero and nothing
    # dirties the tail. Poison is what makes the fill observable.
    POISON = 9.0

    # Every tile at the shape the emitter demands, from the one table -- both
    # the staged inputs below and the allocations further down. The old form
    # allocated mlen x mlen for all of them, which is the right shape only when
    # key_dim == value_dim == mlen, and the emitter's own idea of the shapes was
    # written separately so nothing compared the two.
    tile_shapes = kda_prefill_tile_shapes(shape, mlen, chunk)

    def _pad(t, *, poison=True, rows=None):
        rows = mlen if rows is None else rows
        have = t.shape[0]
        if have >= rows:
            return t
        tail = torch.full((rows - have, t.shape[1]), POISON if poison else 0.0)
        return torch.cat([t, tail], dim=0)

    def _chunkwise(t, c):
        """Slice chunk `c` out of a [total, *] tensor, padded to the tile's rows."""
        return _pad(t[c * chunk : (c + 1) * chunk], rows=tile_shapes["q"][0])

    ins = {
        # Padded to mlen rows: q and k are spilled (k_hat) and projected, and a
        # projection writes a whole mlen x mlen block. One staged tensor per
        # chunk, because each chunk consumes its own tokens.
        **{f"Q{c}": _chunkwise(q, c) for c in range(chunks)},
        **{f"K{c}": _chunkwise(k, c) for c in range(chunks)},
        # Time on lanes, zero-padded to mlen: every contraction here runs over a
        # whole block, and chunk <= mlen, so the padding contributes nothing and
        # keeps one uniform width across the [*, chunk] tiles.
        # v_t is [value_dim, chunk]: its *lanes* past chunk are padding, and the
        # projections contract over lanes, so those must stay zero.
        **{
            f"VT{c}": torch.nn.functional.pad(
                v[c * chunk : (c + 1) * chunk].T.contiguous(),
                (0, tile_shapes["v_t"][1] - chunk, 0, tile_shapes["v_t"][0] - value_dim),
            )
            for c in range(chunks)
        },
        **{f"A{c}": _chunkwise(a, c) for c in range(chunks)},
        "S0": state0,
        # t_mat is spilled, and it is built from `identity` -- so identity's tail
        # rows become t_mat's tail rows. Poison them too.
        "ID": _pad(torch.tensor(KdaPrefillBuffers.identity_values(chunk, mlen))),
        # t_mat is loaded, not allocated, so its tail rows start POISONED. That
        # is what makes the spill zero-fill observable: t_mat is spilled, and its
        # tail feeds err_t's lanes past `chunk`, which are the contraction index
        # of the state projection. With t_mat freshly allocated its tail is zero
        # and multiplies every other tail away -- one accidental zero masking the
        # whole class of over-read.
        "TM": _pad(torch.zeros(chunk, mlen)),
        "CI": _pad(torch.tensor(KdaPrefillBuffers.causal_mask_values(chunk, mlen)), poison=False),
    }
    v_ = {name: prog.input(name, shape=tuple(t.shape)) for name, t in ins.items()}
    state_v = prog.load_batch(v_["S0"], name="state")
    ident_v = prog.load_batch(v_["ID"], name="ident")
    ci_v = prog.load_batch(v_["CI"], name="causal_inclusive")

    alloc = lambda n: prog.alloc(n, *tile_shapes[n], strict=False)  # noqa: E731
    # q / k / v_t / decay are set per chunk in the loop below -- all four are
    # consumed in place, so each chunk gets a fresh load of its own tokens.
    _ph = prog.alloc("placeholder", mlen, mlen, strict=False)
    buffers = KdaPrefillBuffers(
        q=_ph,
        k=_ph,
        v_t=_ph,
        decay=_ph,
        k_tilde=alloc("k_tilde"),
        k_end=alloc("k_end"),
        gram=alloc("gram"),
        readout=alloc("readout"),
        t_mat=prog.load_batch(v_["TM"], name="t_mat"),
        identity=ident_v,
        causal_inclusive=ci_v,
        err_t=alloc("err_t"),
        state=state_v,
        contrib=alloc("contrib"),
        readout_contrib=alloc("readout_contrib"),
        state_contrib=alloc("state_contrib"),
        out=alloc("out"),
        prev=alloc("prev"),
        scale_scratch=alloc("scale_scratch"),
        beta_fp=prog.fp_var("beta_fp", size=mlen),
        m_fp=prog.fp_var("m_fp", size=mlen),
        output_scale_fp=prog.fp_var("out_scale", size=1),
        consts=consts,
    )
    outs, beta_vars = [], []
    for c in range(chunks):
        # Fresh q/k/v/decay per chunk: all four are consumed in place.
        buffers.q = prog.load_batch(v_[f"Q{c}"], name=f"q{c}")
        buffers.k = prog.load_batch(v_[f"K{c}"], name=f"k{c}")
        buffers.v_t = prog.load_batch(v_[f"VT{c}"], name=f"v_t{c}")
        buffers.decay = prog.load_batch(v_[f"A{c}"], name=f"decay{c}")
        buffers.out = prog.alloc(f"out{c}", *tile_shapes["out"], strict=False)
        buffers.beta_fp = prog.fp_var(f"beta_fp{c}", size=mlen)
        beta_vars.append(buffers.beta_fp)
        prog.kda_prefill_chunk_v0(buffers=buffers, chunk=chunk, shape=shape)
        outs.append(buffers.out)

    out_scale = buffers.output_scale_fp
    tail = max([out_scale.address + 1] + [bv.address + mlen for bv in beta_vars])
    fp = fp_preload + [0.0] * max(0, tail - len(fp_preload))
    fp[out_scale.address] = scale
    for c, bv in enumerate(beta_vars):
        fp[bv.address : bv.address + chunk] = beta[c * chunk : (c + 1) * chunk].tolist()

    if want == "out":
        # The last chunk's output, against the same span of the reference.
        target, golden, rows = outs[-1], out_ref[(chunks - 1) * chunk :], chunk
        cols = tile_shapes["out"][1]
        golden = torch.nn.functional.pad(golden, (0, cols - value_dim))
    else:
        target, golden, rows = buffers.state, state_ref, value_dim
        cols = tile_shapes["state"][1]
        golden = torch.nn.functional.pad(golden, (0, cols - key_dim))

    # Absolute, not relative: with normalised q/k the outputs are O(1e-3), so a
    # relative bar on values near zero measures nothing. 5e-2 absolute is ~10x
    # the observed 5e-3, and a real lowering error is orders of magnitude larger
    # -- the mutation tests in test_kda_prefill.py are what cover that.
    _write_comparison(build_dir, prog, target, rows=rows, cols=cols, mlen=mlen, atol=5e-2, rtol=5e-2)
    _finish(build_dir, prog, golden, ins, fp, list(ins), f"prefill_{want}", args)


def case_prefill_out(args, build_dir, hw):
    """The chunk's token outputs must equal the recurrence's, token for token."""
    _prefill_case(args, build_dir, hw, want="out")


def case_prefill_state(args, build_dir, hw):
    """And the carried state, which is what makes chunks chain."""
    _prefill_case(args, build_dir, hw, want="state")


def case_state_transpose(args, build_dir, hw):
    """The prefill -> decode state layout conversion.

    decode holds the state [key, value] and prefill holds it [value, key]. At
    Kimi K3 key_dim == value_dim, so the shapes match and handing one to the
    other is a finite plausible wrong answer. The conversion is one projection
    against a staged identity.
    """
    mlen = args.mlen
    key_dim = args.key_dim or mlen
    value_dim = args.value_dim or mlen
    shape = KdaShape(
        hidden_size=mlen,
        num_heads=1,
        key_dim=key_dim,
        value_dim=value_dim,
        conv_kernel=4,
    )
    use_bf16_kv_precision(build_dir)
    g = torch.Generator().manual_seed(args.seed)
    # Distinguishable under transposition *and* under a block swap: with
    # key_dim == value_dim the tile is square, so a converter that got the block
    # indices the wrong way round would return something of the right shape.
    state = _mxfp8_exact((value_dim, key_dim), g, hi=1.0)
    golden = state.T.contiguous()

    prog = PlenaCompiler(mlen=mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    prog._bf16_kv_checked = True
    prog.kda_fp_constants()
    fp_preload = prog.kda_fp_constant_values() + [0.0] * 3

    want = kda_prefill_state_transpose_shapes(shape, mlen)
    # The identity spans the whole key axis, not `chunk` and not `mlen`: the
    # transpose contracts over key, so a smaller one contracts over part of it.
    ins = {"S": state, "ID": torch.eye(want["identity"][0])}
    v_ = {n: prog.input(n, shape=tuple(t.shape)) for n, t in ins.items()}
    state_v = prog.load_batch(v_["S"], name="state")
    ident_v = prog.load_batch(v_["ID"], name="ident")
    out_v = prog.alloc("state_T", *want["out"], strict=False)

    from compiler.aten.plena.program_ssd import SPILLED_ACTIVATION

    prog.kda_prefill_state_to_decode_layout_v0(
        state=state_v,
        identity=ident_v,
        out=out_v,
        shape=shape,
        precision=SPILLED_ACTIVATION,
    )
    _write_comparison(
        build_dir,
        prog,
        out_v,
        rows=key_dim,
        cols=want["out"][1],
        mlen=mlen,
        atol=1e-3,
        rtol=1e-3,
    )
    _finish(build_dir, prog, golden, ins, fp_preload, list(ins), "state_transpose", args)


def _layer_reference(projected, conv_hist, conv_w, state, a_log, dt_bias, shape):
    """One layer of the state engine, from the reference."""
    from compiler.aten.models.kda.reference import (
        KdaConvWeights,
        KdaRecurrentState,
        kda_state_engine_step,
    )

    return kda_state_engine_step(
        projected,
        KdaRecurrentState(state.clone(), conv_hist.clone()),
        KdaConvWeights(conv_w["q"], conv_w["k"], conv_w["v"], None, None, None),
        a_log,
        dt_bias,
        shape,
        state_storage="fp32",
    )


def case_layer(args, build_dir, hw, *, layers: int = 1):
    """The assembled KDA layer on the emulator, against kda_state_engine_step.

    Gather, three convolutions, the gates and the recurrence -- every emitter
    this work added, running as one program through the assembler and the Rust
    emulator rather than through the float64 ISA interpreter.

    With ``layers > 1`` each layer gets its own projection, conv history,
    weights and recurrent state, and they run back to back in one program. That
    is what catches a layer whose emitters collide with the previous one's --
    reused spill names, an FPRAM window written twice, a VRAM view whose name
    repoints an earlier one. A single layer cannot see any of it.
    """
    import torch

    from compiler.aten.models.kda.shape import KdaShape
    from compiler.aten.plena.program_kda_common import (
        kda_state_row,
        kda_state_rows,
        kda_vector_row,
        kda_vector_rows,
    )
    from compiler.aten.plena.program_kda_conv import kda_conv_blocks, kda_conv_state_row
    from compiler.aten.plena.program_kda_gates import (
        kda_head_blocks,
        kda_key_blocks,
        kda_key_row,
    )
    from compiler.aten.plena.program_kda_layer import (
        kda_projection_features,
        kda_projection_sections,
    )
    from compiler.aten.plena.program_kda_mixer import KdaMixerBuffers

    mlen = args.mlen
    heads = args.num_heads
    key_dim = args.key_dim or mlen
    value_dim = mlen
    shape = KdaShape(
        hidden_size=heads * value_dim,
        num_heads=heads,
        key_dim=key_dim,
        value_dim=value_dim,
        conv_kernel=4,
    )
    key_width = shape.projection_size
    value_width = heads * value_dim
    kernel = shape.conv_kernel
    kb = kda_key_blocks(shape, mlen)
    vb = kda_vector_rows(shape, mlen) // heads

    g = torch.Generator().manual_seed(args.seed)
    shared_a_log = _mxfp8_exact((heads,), g, hi=0.5)
    per_layer = []
    for _ in range(layers):
        per_layer.append(
            dict(
                projected=_mxfp8_exact((1, kda_projection_features(shape)), g, hi=1.0),
                conv_hist=_mxfp8_exact((1, 2 * key_width + value_width, kernel), g, hi=0.5),
                conv_w={
                    n: _mxfp8_exact((w, kernel), g, hi=0.5)
                    for n, w in (("q", key_width), ("k", key_width), ("v", value_width))
                },
                state=_mxfp8_exact((1, heads, value_dim, key_dim), g, hi=0.5),
                a_log=shared_a_log,
                dt_bias=_mxfp8_exact((heads, key_dim), g, hi=0.5),
            )
        )
    golden_rows = []
    for L in per_layer:
        out, _ = _layer_reference(
            L["projected"],
            L["conv_hist"],
            L["conv_w"],
            L["state"],
            L["a_log"],
            L["dt_bias"],
            shape,
        )
        golden_rows.append(out[0])
    golden = torch.stack(golden_rows)
    golden = torch.nn.functional.pad(golden, (0, mlen * vb - golden.shape[1]))

    prog = PlenaCompiler(mlen=mlen, blen=args.blen, real_data_ratio=hw.real_data_ratio)
    consts = prog.kda_fp_constants()
    fp = prog.kda_fp_constant_values() + [0.0] * 3

    up = lambda n: ((n + mlen - 1) // mlen) * mlen  # noqa: E731
    a = lambda n, r: prog.alloc(n, up(r), mlen, strict=False)  # noqa: E731
    widths = {"q": key_width, "k": key_width, "v": value_width}
    offsets = {"q": 0, "k": key_width, "v": 2 * key_width}

    # One FPRAM window for every layer. FPRAM is per-layer *scratch*, not
    # per-layer storage: a program that allocates it per layer overflows even
    # the compiler's optimistic 1024 slots at four layers, let alone the
    # hardware's 512. That is the same reuse the mixer already does per head.
    ins, outs = {}, []
    for i, L in enumerate(per_layer):
        # Every layer stages its own tensors; nothing is shared but the
        # constants, which is what makes a collision between layers visible.
        flat = L["projected"][0]
        nblk = -(-flat.numel() // mlen)
        proj_rows = torch.zeros(prog.blen, nblk * mlen)
        for b in range(nblk):
            chunk = flat[b * mlen : (b + 1) * mlen]
            proj_rows[0, b * mlen : b * mlen + chunk.numel()] = chunk
        ins[f"PROJ{i}"] = proj_rows
        for n, w in widths.items():
            blocks = kda_conv_blocks(w, mlen)
            cs = torch.zeros(blocks * kernel, mlen)
            cw = torch.zeros(blocks * kernel, mlen)
            for cb in range(blocks):
                lo = cb * mlen
                for tap in range(kernel):
                    row = kda_conv_state_row(w, mlen, kernel, cb, tap)
                    cs[row] = L["conv_hist"][0, offsets[n] + lo : offsets[n] + lo + mlen, tap]
                    cw[row] = L["conv_w"][n][lo : lo + mlen, tap]
            ins[f"CS{i}_{n}"], ins[f"CW{i}_{n}"] = cs, cw
        dtb = torch.zeros(up(heads * kb), mlen)
        for h in range(heads):
            for b in range(kb):
                dtb[kda_key_row(shape, mlen, h, b)] = L["dt_bias"][h, b * mlen : (b + 1) * mlen]
        ins[f"DTB{i}"] = dtb
        st = torch.zeros(up(kda_state_rows(shape, mlen)), mlen)
        for h in range(heads):
            for blk in range(vb):
                for key in range(key_dim):
                    st[kda_state_row(shape, mlen, h, blk, key)] = L["state"][0, h, blk * mlen : (blk + 1) * mlen, key]
        ins[f"ST{i}"] = st

    v_ = {n: prog.input(n, shape=tuple(t.shape)) for n, t in ins.items()}
    result = a("layer_out", layers * vb)
    shared_fp = {
        "decay": prog.fp_var("decay_and_q_hat", size=key_dim),
        "k_hat": prog.fp_var("k_hat", size=key_dim),
        "beta": prog.fp_var("beta", size=kda_head_blocks(shape, mlen) * mlen),
        "part": prog.fp_var("part", size=kb),
        "acc": prog.fp_var("acc", size=1),
        "output_scale": prog.fp_var("output_scale", size=1),
        "rate": prog.fp_var("rate", size=heads),
        "lower_bound": prog.fp_var("lower_bound", size=1),
    }

    for i in range(layers):
        projected = prog.load_batch(v_[f"PROJ{i}"], name=f"projected{i}")
        sections = {n: (f, c) for n, f, c in kda_projection_sections(shape, mlen)}
        gathered = {n: a(f"g{i}_{n}", max(c, mlen)) for n, (_, c) in sections.items()}
        conv_state = {n: prog.load_batch(v_[f"CS{i}_{n}"], name=f"cs{i}_{n}") for n in widths}
        conv_weight = {n: prog.load_batch(v_[f"CW{i}_{n}"], name=f"cw{i}_{n}") for n in widths}
        conv_scratch = a(f"convsc{i}", kda_conv_blocks(max(widths.values()), mlen))
        decay = shared_fp["decay"]
        buffers = KdaMixerBuffers(
            q=a(f"q{i}", heads * kb),
            k=a(f"k{i}", heads * kb),
            v=a(f"v{i}", kda_vector_rows(shape, mlen)),
            gate=gathered["gate"],
            dt_bias=prog.load_batch(v_[f"DTB{i}"], name=f"dtb{i}"),
            beta_logit=gathered["beta"],
            state=prog.load_batch(v_[f"ST{i}"], name=f"st{i}"),
            out=a(f"out{i}", kda_vector_rows(shape, mlen)),
            pred=a(f"pred{i}", kda_vector_rows(shape, mlen)),
            err=a(f"err{i}", kda_vector_rows(shape, mlen)),
            sq_scratch=a(f"sq{i}", heads * kb),
            decay_fp=decay,
            q_hat_fp=decay,
            k_hat_fp=shared_fp["k_hat"],
            beta_fp=shared_fp["beta"],
            part_fp=shared_fp["part"],
            acc_fp=shared_fp["acc"],
            output_scale_fp=shared_fp["output_scale"],
            rate_fp=shared_fp["rate"],
            lower_bound_fp=shared_fp["lower_bound"],
            consts=consts,
        )
        prog.kda_layer_from_projected_v0(
            projected=projected,
            gathered=gathered,
            conv_state=conv_state,
            conv_weight=conv_weight,
            conv_bias={},
            conv_scratch=conv_scratch,
            mixer_buffers=buffers,
            shape=shape,
        )
        outs.append(buffers.out)
        # Pack every layer's output, for every head count. This was gated on
        # `heads == 1`, so under the only invocation the commit recorded
        # (`--num-heads 3`) it compared the LAST layer alone -- zeroing layers
        # 0..n-2's inputs left the verdict bit-identical.
        for h in range(heads):
            for blk in range(vb):
                prog.mamba_row_copy(
                    result,
                    (i * heads + h) * vb + blk,
                    buffers.out,
                    kda_vector_row(shape, mlen, h, blk),
                )
    target, rows = result, layers * heads * vb

    want = torch.zeros(rows, mlen)
    for i, row in enumerate(golden_rows):
        for h in range(heads):
            for blk in range(vb):
                lo = h * value_dim + blk * mlen
                want[(i * heads + h) * vb + blk] = row[lo : lo + mlen]
    golden = want

    tail = max(
        shared_fp["rate"].address + heads, shared_fp["lower_bound"].address + 1, shared_fp["output_scale"].address + 1
    )
    fp = fp + [0.0] * max(0, tail - len(fp))
    fp[shared_fp["output_scale"].address] = 1.0 / key_dim**0.5
    fp[shared_fp["lower_bound"].address] = shape.gate_lower_bound
    # One shared `rate` window, so every layer uses the same a_log. A real model
    # reloads it per layer from HBM; there is no emitter for that yet, and what
    # this case tests is that layers compose as programs.
    fp[shared_fp["rate"].address : shared_fp["rate"].address + heads] = torch.exp(per_layer[0]["a_log"]).tolist()

    # atol=0. An absolute bound is what made this case vacuous: the outputs are
    # O(5e-3) and atol was 2e-2, so every golden value sat inside it and writing
    # zeros into the tile scored 100%. Verified -- zeroing the output, skipping
    # the `v` convolution, running the mixer before the convolutions, and
    # swapping the `q` and `k` gather sections all passed.
    #
    # The bound is relative now and set from what bf16 delivers, not from what
    # makes the test green: the read-out contracts 128 terms, so the error grows
    # like sqrt(128) * 2^-9 = 2.2%, and the measured mean relative error is 4.3%.
    # `atol` is not 0. A pure relative bound cannot be met by the smallest
    # entries: the layer's outputs run from 1e-3 to 1e-2, bf16 delivers about
    # 1.8e-4 absolute on this contraction whatever the value, and on an entry of
    # 1e-3 that is 18% -- so three of the sixty-four always sat outside a 12%
    # bar. They were carried by `check_mem`'s 90% rule rather than by the
    # tolerance, which is not a bar at all. `layer_atol` states the absolute
    # floor instead, and the combined bound is checked at 100%.
    #
    # It stays a real check: an all-zero output scores 1.6% against this bound,
    # because every golden value is far above the floor. That is the property
    # the earlier `atol=2e-2` version lacked -- there, zeros scored 100%.
    _write_comparison(
        build_dir,
        prog,
        target,
        rows=rows,
        cols=mlen,
        mlen=mlen,
        atol=args.layer_atol,
        rtol=args.layer_rtol,
    )
    _finish(build_dir, prog, golden, ins, fp, list(ins), f"layer{layers}", args)


def case_layer_chain(args, build_dir, hw):
    """Four layers back to back in one program."""
    case_layer(args, build_dir, hw, layers=4)


def case_official_layer(args, build_dir, hw):
    """Eight projections through the final output projection on Rust.

    Unlike ``case_layer``, whose boundary starts at an already projected packed
    tensor and ends at the recurrent output, this case starts with hidden and
    executes the exact official Kimi K3 stage order.  The dimensions are small
    enough for a regression test, while the head/value/state layout and formulas
    are the production ones.
    """
    import math

    from compiler.aten.models.kda.reference import (
        KdaConvWeights,
        KdaOfficialLayerWeights,
        KdaRecurrentState,
        kda_official_layer_step,
    )
    from compiler.aten.models.kda.shape import KdaShape
    from compiler.aten.plena.program_kda_common import (
        kda_state_row,
        kda_state_rows,
        kda_vector_rows,
    )
    from compiler.aten.plena.program_kda_conv import kda_conv_blocks, kda_conv_state_row
    from compiler.aten.plena.program_kda_gates import (
        kda_head_blocks,
        kda_key_blocks,
        kda_key_row,
    )
    from compiler.aten.plena.program_kda_layer import (
        KdaOfficialLayerBuffers,
        KdaOfficialProjectionWeights,
    )
    from compiler.aten.plena.program_kda_mixer import KdaMixerBuffers

    use_bf16_kv_precision(build_dir)
    mlen = args.mlen
    heads = args.num_heads
    key_dim = args.key_dim or mlen
    value_dim = args.value_dim or mlen
    hidden_size = max(2 * mlen, heads * value_dim)
    decay_rank = mlen
    shape = KdaShape(hidden_size, heads, key_dim, value_dim, 4)
    if key_dim % mlen or value_dim % mlen:
        raise SystemExit("official_layer requires key_dim/value_dim multiples of mlen")

    key_width = shape.projection_size
    value_width = heads * value_dim
    kernel = shape.conv_kernel
    kb = kda_key_blocks(shape, mlen)
    vb = value_dim // mlen
    up = lambda n: math.ceil(n / mlen) * mlen  # noqa: E731
    g = torch.Generator().manual_seed(args.seed + 101)
    exact = lambda *size, hi=0.5: _mxfp8_exact(size, g, hi=hi)  # noqa: E731

    hidden = exact(1, hidden_size)
    matrix = {
        "W_Q": exact(hidden_size, key_width),
        "W_K": exact(hidden_size, key_width),
        "W_V": exact(hidden_size, value_width),
        "W_DECAY_A": exact(hidden_size, decay_rank),
        "W_DECAY_B": exact(decay_rank, key_width),
        "W_BETA": exact(hidden_size, heads),
        "W_OUTPUT_GATE": exact(hidden_size, value_width),
        "W_OUTPUT": exact(value_width, hidden_size),
    }
    conv_w = {
        "q": exact(key_width, kernel),
        "k": exact(key_width, kernel),
        "v": exact(value_width, kernel),
    }
    conv_hist = exact(1, 2 * key_width + value_width, kernel)
    recurrent = exact(1, heads, value_dim, key_dim, hi=0.25)
    a_log = exact(heads, hi=0.25)
    dt_bias = exact(heads, key_dim, hi=0.25)
    norm_weight = 1.0 + exact(value_dim, hi=0.125)

    reference_weights = KdaOfficialLayerWeights(
        q=matrix["W_Q"],
        k=matrix["W_K"],
        v=matrix["W_V"],
        decay_a=matrix["W_DECAY_A"],
        decay_b=matrix["W_DECAY_B"],
        beta=matrix["W_BETA"],
        output_gate=matrix["W_OUTPUT_GATE"],
        output=matrix["W_OUTPUT"],
        conv=KdaConvWeights(conv_w["q"], conv_w["k"], conv_w["v"]),
        norm_weight=norm_weight,
        a_log=a_log,
        dt_bias=dt_bias,
    )
    golden, _ = kda_official_layer_step(
        hidden,
        KdaRecurrentState(recurrent.clone(), conv_hist.clone()),
        reference_weights,
        shape,
        state_storage="bf16",
        conv_state_storage="bf16",
    )

    prog = PlenaCompiler(
        mlen=mlen,
        blen=args.blen,
        real_data_ratio=2.0,
        mram_tile_capacity=max(4, math.ceil(hidden_size / mlen)),
    )
    prog._bf16_kv_checked = True
    consts = prog.kda_fp_constants()
    norm_consts = prog.mamba_fp_constants(name_prefix="kda_output_norm")

    input_tensors: dict[str, torch.Tensor] = {}
    tensor_layouts: dict[str, dict] = {}

    def layout(source_rows, source_cols, storage_rows, storage_cols, precision):
        return {
            "source_shape": [source_rows, source_cols],
            "storage_shape": [storage_rows, storage_cols],
            "source_rows": source_rows,
            "storage_rows": storage_rows,
            "source_row_elements": source_cols,
            "storage_row_elements": storage_cols,
            "precision": precision,
        }

    def hbm_input(name, value, *, logical_shape=None, precision="HBM_V_KV_TYPE"):
        rows, cols = value.shape
        logical = logical_shape or (rows, cols)
        physical = (up(rows), up(cols))
        padded = torch.zeros(physical, dtype=torch.float32)
        padded[:rows, :cols] = value.float()
        input_tensors[name] = padded
        tensor_layouts[name] = layout(physical[0], physical[1], physical[0], physical[1], precision)
        return prog.input(
            name,
            logical,
            physical_shape=physical,
            real_data_ratio=2.0,
            hbm_element_bytes=2,
        )

    hidden_input = hbm_input("HIDDEN", hidden, logical_shape=(1, hidden_size))
    hidden_v = prog.load_batch(hidden_input, name="hidden", storage_precision=2, precision=1)
    projection_weights = KdaOfficialProjectionWeights(
        q=hbm_input("W_Q", matrix["W_Q"], precision="HBM_M_KV_TYPE"),
        k=hbm_input("W_K", matrix["W_K"], precision="HBM_M_KV_TYPE"),
        v=hbm_input("W_V", matrix["W_V"], precision="HBM_M_KV_TYPE"),
        decay_a=hbm_input("W_DECAY_A", matrix["W_DECAY_A"], precision="HBM_M_KV_TYPE"),
        decay_b=hbm_input("W_DECAY_B", matrix["W_DECAY_B"], precision="HBM_M_KV_TYPE"),
        beta=hbm_input("W_BETA", matrix["W_BETA"], precision="HBM_M_KV_TYPE"),
        output_gate=hbm_input("W_OUTPUT_GATE", matrix["W_OUTPUT_GATE"], precision="HBM_M_KV_TYPE"),
        output=hbm_input("W_OUTPUT", matrix["W_OUTPUT"], precision="HBM_M_KV_TYPE"),
    )

    widths = {"q": key_width, "k": key_width, "v": value_width}
    offsets = {"q": 0, "k": key_width, "v": 2 * key_width}
    conv_state, conv_weight = {}, {}
    for section, width in widths.items():
        blocks = kda_conv_blocks(width, mlen)
        state_tile = torch.zeros(blocks * kernel, mlen)
        weight_tile = torch.zeros(blocks * kernel, mlen)
        for block in range(blocks):
            lo = block * mlen
            for tap in range(kernel):
                row = kda_conv_state_row(width, mlen, kernel, block, tap)
                state_tile[row] = conv_hist[0, offsets[section] + lo : offsets[section] + lo + mlen, tap]
                weight_tile[row] = conv_w[section][lo : lo + mlen, tap]
        state_input = hbm_input(f"CONV_STATE_{section}", state_tile)
        weight_input = hbm_input(f"CONV_WEIGHT_{section}", weight_tile)
        conv_state[section] = prog.load_batch(
            state_input, name=f"conv_state_{section}", storage_precision=2, precision=1
        )
        conv_weight[section] = prog.load_batch(
            weight_input, name=f"conv_weight_{section}", storage_precision=2, precision=1
        )

    dt_tile = torch.zeros(up(heads * kb), mlen)
    for head in range(heads):
        for block in range(kb):
            dt_tile[kda_key_row(shape, mlen, head, block)] = dt_bias[head, block * mlen : (block + 1) * mlen]
    dt_v = prog.load_batch(
        hbm_input("DT_BIAS", dt_tile),
        name="dt_bias",
        storage_precision=2,
        precision=1,
    )

    state_tile = torch.zeros(up(kda_state_rows(shape, mlen)), mlen)
    for head in range(heads):
        for block in range(vb):
            for key in range(key_dim):
                state_tile[kda_state_row(shape, mlen, head, block, key)] = recurrent[
                    0, head, block * mlen : (block + 1) * mlen, key
                ]
    state_v = prog.load_batch(
        hbm_input("RECURRENT_STATE", state_tile),
        name="recurrent_state",
        storage_precision=2,
        precision=1,
    )

    norm_tile = torch.zeros(up(value_dim // mlen), mlen)
    for block in range(value_dim // mlen):
        norm_tile[block] = norm_weight[block * mlen : (block + 1) * mlen]
    norm_v = prog.load_batch(
        hbm_input("NORM_WEIGHT", norm_tile),
        name="norm_weight",
        storage_precision=2,
        precision=1,
    )

    a = lambda name, rows: prog.alloc(name, up(rows), mlen, strict=False)  # noqa: E731
    decay = a("decay", heads * kb)
    beta = a("beta", kda_head_blocks(shape, mlen))
    decay_or_q = prog.fp_var("decay_or_q", size=key_dim)
    mixer = KdaMixerBuffers(
        q=a("q", heads * kb),
        k=a("k", heads * kb),
        v=a("v", kda_vector_rows(shape, mlen)),
        gate=decay,
        dt_bias=dt_v,
        beta_logit=beta,
        state=state_v,
        out=a("mixed", kda_vector_rows(shape, mlen)),
        pred=a("pred", kda_vector_rows(shape, mlen)),
        err=a("err", kda_vector_rows(shape, mlen)),
        sq_scratch=a("mixer_sq", heads * kb),
        decay_fp=decay_or_q,
        q_hat_fp=decay_or_q,
        k_hat_fp=prog.fp_var("k_hat", size=key_dim),
        beta_fp=prog.fp_var("beta_fp", size=kda_head_blocks(shape, mlen) * mlen),
        part_fp=prog.fp_var("part", size=kb),
        acc_fp=prog.fp_var("acc", size=1),
        output_scale_fp=prog.fp_var("output_scale", size=1),
        rate_fp=prog.fp_var("rate", size=heads),
        lower_bound_fp=prog.fp_var("lower_bound", size=1),
        consts=consts,
    )
    buffers = KdaOfficialLayerBuffers(
        mixer=mixer,
        conv_state=conv_state,
        conv_weight=conv_weight,
        conv_bias={},
        conv_scratch=a("conv_scratch", max(kda_conv_blocks(width, mlen) for width in widths.values())),
        decay=decay,
        beta=beta,
        output_gate=a("output_gate", kda_vector_rows(shape, mlen)),
        norm_weight=norm_v,
        norm_sq_scratch=a("norm_sq", kda_vector_rows(shape, mlen)),
        norm_part_fp=prog.fp_var("norm_part", size=value_dim // mlen),
        norm_acc_fp=prog.fp_var("norm_acc", size=1),
        norm_consts=norm_consts,
        packed_output=prog.alloc(
            "packed_output",
            1,
            value_width,
            strict=False,
            physical_shape=(prog.blen, up(value_width)),
        ),
    )
    output = prog.kda_official_layer_decode_v0(
        hidden=hidden_v,
        weights=projection_weights,
        buffers=buffers,
        shape=shape,
    )
    debug_names = (
        "hidden",
        "kda_official_q",
        "kda_official_k",
        "kda_official_v",
        "kda_official_decay_a",
        "kda_official_decay_b",
        "kda_official_beta",
        "kda_official_output_gate",
        "q",
        "k",
        "v",
        "decay",
        "beta",
        "mixed",
        "output_gate",
        "packed_output",
        "kda_official_out",
    )
    debug_layout = {}
    for tensor_name in debug_names:
        tensor = prog._tensors.get(tensor_name)
        if tensor is None:
            continue
        debug_layout[tensor_name] = {
            "vram_addr": prog.get_vram_addr(tensor.name),
            "shape": list(tensor.shape),
            "physical_shape": list(tensor.physical_shape),
        }
    (build_dir / "official_layer_vram_layout.json").write_text(
        json.dumps(debug_layout, indent=2, sort_keys=True) + "\n"
    )

    fp = [0.0] * max(
        mixer.lower_bound_fp.address + 1,
        mixer.rate_fp.address + heads,
        norm_consts.eps.address + 1,
    )
    base_values = prog.kda_fp_constant_values()
    fp[consts.zero.address : consts.zero.address + len(base_values)] = base_values
    norm_values = [
        0.0,
        1.0,
        -1.0,
        0.0,
        65504.0,
        1.0 / value_dim,
        1.0e-5,
    ]
    fp[norm_consts.zero.address : norm_consts.zero.address + len(norm_values)] = norm_values
    fp[mixer.output_scale_fp.address] = 1.0 / math.sqrt(key_dim)
    fp[mixer.lower_bound_fp.address] = shape.gate_lower_bound
    fp[mixer.rate_fp.address : mixer.rate_fp.address + heads] = torch.exp(a_log).tolist()

    _write_comparison(
        build_dir,
        prog,
        output,
        rows=1,
        cols=hidden_size,
        mlen=mlen,
        atol=3.0e-2,
        rtol=0.20,
    )
    addrs = {name: prog.get_hbm_layout(name).hbm_base_addr for name in input_tensors}
    order = sorted(input_tensors, key=addrs.get)
    _finish(
        build_dir,
        prog,
        golden,
        input_tensors,
        fp,
        order,
        "official_layer",
        args,
        tensor_layouts=tensor_layouts,
    )


def case_prefill_chain_state(args, build_dir, hw):
    """Three chunks in sequence, against a 3*chunk-token recurrence.

    This is the case that proves the two memory fixes. The six spill regions are
    reused across chunks, so chunk 2's prefetch reads bytes chunk 1 wrote -- and
    each prefetch reads a whole mlen x mlen block, more than any chunk's live
    data. It is correct only because every spill zero-fills the rows past its
    live data first. With a single chunk that is invisible: unallocated HBM
    happens to be zero.

    It also answers whether the carried state's error compounds. It does not --
    the decay makes the recurrence strongly contracting, so old error is damped.
    """
    _prefill_case(args, build_dir, hw, want="state", chunks=3)


def case_prefill_chain_out(args, build_dir, hw):
    """The third chunk's token outputs, which depend on two chunks of carried state."""
    _prefill_case(args, build_dir, hw, want="out", chunks=3)


CASES = {
    "cumprod": case_cumprod,
    "ut": case_ut,
    "prefill_out": case_prefill_out,
    "prefill_state": case_prefill_state,
    "prefill_chain_out": case_prefill_chain_out,
    "prefill_chain_state": case_prefill_chain_state,
    "state_transpose": case_state_transpose,
    "layer": case_layer,
    "layer_chain": case_layer_chain,
    "official_layer": case_official_layer,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--case", default="cumprod", choices=sorted(CASES))
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=16)
    parser.add_argument("--key-dim", type=int, default=None)
    parser.add_argument("--value-dim", type=int, default=None)
    # Relative and honest. bf16 on the read-out's 128-term contraction
    # gives about sqrt(128)*2^-9 = 2.2%; measured mean is 4.3%. The
    # harness also passes at >= 90% of lanes inside the bound rather
    # than all of them -- see PLENA_Tools/verification/check_mem.py.
    parser.add_argument("--layer-rtol", type=float, default=0.12)
    parser.add_argument(
        "--layer-atol",
        type=float,
        default=2.5e-4,
        help="Absolute floor for the layer cases, above bf16's ~1.8e-4 on this "
        "contraction and ~40x below the smallest golden value.",
    )
    args = parser.parse_args()

    build_dir = Path(__file__).parent / "build" / f"kda_{args.case}"
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(f"KDA stage test: {args.case}  (mlen={args.mlen}, blen={args.blen}, chunk={args.chunk})")
    print("=" * 80)

    CASES[args.case](args, build_dir, hw)
