"""
ATen-style Rotary Position Embedding (RoPE) Test

SmolLM2 language model 1D PE step:
    Q_rotated = Q * cos + rotate_half(Q) * sin

Uses the PLENA ATen-style registry:
    import plena.ops as ops
    result = ops.rope(prog, Q_var, Q_rot_var, cos_var, sin_var)

CPU golden reference:
    registry.set_backend(Backend.CPU)
    golden = ops.rope(Q, Q_rot, cos, sin)

Note: rotate_half(Q) and the cos/sin tables are precomputed on the CPU
before loading to PLENA VRAM.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import json

from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops

from plena_program import PLENAProgram
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


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
    print("=" * 80)
    print("ATen-style RoPE Test  (plena.ops.rope)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    seq_len = 4
    head_dim = 64  # must equal mlen so one VRAM row = one position vector
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(42)

    # ========================================================================
    # Test data
    # ========================================================================
    Q = torch.randn(seq_len, head_dim)
    Q_rot = rotate_half(Q)
    cos, sin = make_rope_tables(seq_len, head_dim)

    print(f"\nQ:     {Q.shape}, range [{Q.min():.3f}, {Q.max():.3f}]")
    print(f"Q_rot: {Q_rot.shape}")
    print(f"cos:   {cos.shape}, range [{cos.min():.3f}, {cos.max():.3f}]")
    print(f"sin:   {sin.shape}, range [{sin.min():.3f}, {sin.max():.3f}]")

    # ========================================================================
    # CPU golden reference
    # ========================================================================
    print("\n--- CPU Golden Reference ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.CPU)
    golden_out = ops.rope(Q, Q_rot, cos, sin)
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ========================================================================
    # PLENA backend
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # All four tensors loaded from HBM into VRAM
    q_input = prog.input("Q", shape=(seq_len, head_dim))
    qrot_input = prog.input("QROT", shape=(seq_len, head_dim))
    cos_input = prog.input("COS", shape=(seq_len, head_dim))
    sin_input = prog.input("SIN", shape=(seq_len, head_dim))

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
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {"Q": Q, "QROT": Q_rot, "COS": cos, "SIN": sin}
    golden_result = {"original_output": golden_out}

    create_sim_env(input_tensor, gen_code, golden_result, [], build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="rope_aten",
        data=None,
        specified_data_order=["Q", "QROT", "COS", "SIN"],
        build_path=build_dir,
    )

    # RoPE is in-place: result is at Q's VRAM location
    q_vram_addr = prog._compiler.get_vram_addr(Q_var.name)

    comparison_params = {
        "start_row_idx": q_vram_addr // mlen,
        "num_rows": (seq_len * head_dim) // mlen,
        "num_batches": seq_len,
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
