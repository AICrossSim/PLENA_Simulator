"""ATen-style Rotary Position Embedding (RoPE) Test.

    python rope_test.py [--mlen 128] [--blen 16] [--seq-len 4] [--head-dim 64]

SmolLM2 language model 1D PE step:
    Q_rotated = Q * cos + rotate_half(Q) * sin

Uses the PLENA ATen-style registry:
    import compiler.aten.ops as ops
    result = ops.rope(prog, Q_var, Q_rot_var, cos_var, sin_var)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden = ops.rope(Q, Q_rot, cos, sin)

Note: rotate_half(Q) and the cos/sin tables are precomputed on the CPU
before loading to PLENA VRAM.
"""

import argparse
import json
from pathlib import Path

import torch

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena import PlenaCompiler
from transactional_emulator.testbench.aten.golden import golden_rope
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.sim_env_utils import create_mem_for_sim
from transactional_emulator.tools.create_sim_env import create_sim_env
from transactional_emulator.testbench.aten.configurable import add_hw_args, resolve_rows, setup_hw


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """rotate_half: [-x[d//2:], x[:d//2]]"""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def make_rope_tables(seq_len: int, head_dim: int, theta: float = 10000.0):
    """Compute RoPE cos/sin tables, shape (seq_len, head_dim)."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)  # (seq_len, half)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    # Duplicate for both halves (standard RoPE)
    cos = torch.cat([cos_half, cos_half], dim=-1)  # (seq_len, head_dim)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_hw_args(parser)
    parser.add_argument("--head-dim", type=int, default=None, help="Head dimension (default: mlen, must be <= mlen)")
    args = parser.parse_args()

    mlen = args.mlen
    blen = args.blen
    # Total token rows = batch_size * seq_len (unified [batch, seq, hidden] interface).
    # RoPE is per-row (per-position) independent, so the rows are flattened to
    # [rows, head_dim]. Default rows == prior seq_len default (4).
    rows, batch_size, seq_len = resolve_rows(args, default_seq=4)
    head_dim = args.head_dim or mlen  # must equal mlen so one VRAM row = one position vector

    if head_dim > mlen:
        raise ValueError(f"head_dim ({head_dim}) must be <= MLEN ({mlen})")
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")

    build_dir = Path(__file__).parent / "build" / "rope"
    hw = setup_hw(args, build_dir)

    print("=" * 80)
    print(f"ATen-style RoPE Test  (mlen={mlen}, blen={blen}, batch={batch_size}, seq={seq_len}, rows={rows}, head_dim={head_dim})")
    print("=" * 80)

    torch.manual_seed(args.seed)

    # ========================================================================
    # Test data
    # ========================================================================
    Q = torch.randn(rows, head_dim)
    Q_rot = rotate_half(Q)
    cos, sin = make_rope_tables(rows, head_dim)

    print(f"\nQ:     {Q.shape}, range [{Q.min():.3f}, {Q.max():.3f}]")
    print(f"Q_rot: {Q_rot.shape}")
    print(f"cos:   {cos.shape}, range [{cos.min():.3f}, {cos.max():.3f}]")
    print(f"sin:   {sin.shape}, range [{sin.min():.3f}, {sin.max():.3f}]")

    # ========================================================================
    # Hardware-accurate golden reference
    # ========================================================================
    print("\n--- Hardware-Accurate Golden Reference ---")
    golden_out = golden_rope(Q, Q_rot, cos, sin)
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ========================================================================
    # PLENA backend
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=hw.real_data_ratio)

    # All four tensors loaded from HBM into VRAM
    q_input = prog.input("Q", shape=(rows, head_dim))
    qrot_input = prog.input("QROT", shape=(rows, head_dim))
    cos_input = prog.input("COS", shape=(rows, head_dim))
    sin_input = prog.input("SIN", shape=(rows, head_dim))

    Q_var = prog.load_batch(q_input, name="Q")
    Qrot_var = prog.load_batch(qrot_input, name="QROT")
    Cos_var = prog.load_batch(cos_input, name="COS")
    Sin_var = prog.load_batch(sin_input, name="SIN")

    # RoPE: Q_var is updated in-place
    result = ops.rope(prog, Q_var, Qrot_var, Cos_var, Sin_var)

    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    input_tensors = {"Q": Q, "QROT": Q_rot, "COS": cos, "SIN": sin}
    golden_result = {"original_output": golden_out}

    # Match the compiler's compact HBM layout exactly. Without this, a stale
    # tensor_layouts.json (or default rounding) can pad tensors so subsequent
    # inputs land off their compiler-assigned addresses.
    tensor_layouts = {
        name: {
            "logical_shape": [rows, head_dim],
            "physical_shape": [rows, head_dim],
            "source_rows": rows,
            "storage_rows": rows,
            "source_row_elements": head_dim,
            "storage_row_elements": head_dim,
        }
        for name in ("Q", "QROT", "COS", "SIN")
    }
    # Use the compiler's actual HBM byte offsets so each tensor is written
    # exactly where the prefetch ISA expects it (handles MLEN-alignment
    # padding the compiler inserts at MLEN>=128 that the HBM writer does not).
    hbm_addrs = {name: prog._compiler.get_hbm_layout(name).hbm_base_addr for name in ("Q", "QROT", "COS", "SIN")}

    create_sim_env(
        input_tensors,
        gen_code,
        golden_result,
        [],
        build_dir=str(build_dir),
        tensor_layouts=tensor_layouts,
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="rope_aten",
        data=None,
        specified_data_order=["Q", "QROT", "COS", "SIN"],
        build_path=build_dir,
        input_tensors=input_tensors,
        tensor_layouts=tensor_layouts,
        hbm_addrs=hbm_addrs,
    )

    # RoPE is in-place: result is at Q's VRAM location
    q_vram_addr = prog._compiler.get_vram_addr(Q_var.name)

    comparison_params = {
        "start_row_idx": q_vram_addr // mlen,
        "num_rows": (rows * head_dim) // mlen,
        "num_batches": rows,
        "elements_per_batch": head_dim,
        "row_dim": mlen,
        "use_stride_mode": head_dim > mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {q_vram_addr // mlen} (in-place on Q)")
    run_and_assert(build_dir, "rope", mlen=mlen, blen=blen)
