"""
ATen-style FFN Test

Uses the PLENA ATen-style registry:
    import compiler.aten.ops as ops
    result = ops.ffn(prog, X_batch, w_gate_input, w_up_input, w_down_input)

CPU golden reference (hardware-accurate):
    Quantize inputs to MXFP8 (matching HBM storage), then compute with BF16 intermediates.

FFN formula: w_down @ (silu(w_gate @ x) * (w_up @ x))
"""

from pathlib import Path


import torch
import torch.nn.functional as F
import json

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena_compiler import PlenaCompiler
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.model_layer_test_builder import quantize_to_mxfp


if __name__ == "__main__":
    print("=" * 80)
    print("ATen-style FFN Test  (plena.ops.ffn)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    hidden_size = 128
    inter_dim = 256
    batch_size = 4
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(42)

    # ========================================================================
    # Test data
    # ========================================================================
    # Use Xavier-style initialization to keep values in a reasonable range
    scale_in = 1.0 / (hidden_size**0.5)
    scale_out = 1.0 / (inter_dim**0.5)

    X = torch.randn(batch_size, hidden_size)
    W_gate = torch.randn(hidden_size, inter_dim) * scale_in  # (hidden, inter_dim)
    W_up = torch.randn(hidden_size, inter_dim) * scale_in  # (hidden, inter_dim)
    W_down = torch.randn(inter_dim, hidden_size) * scale_out  # (inter_dim, hidden)

    print(f"\nInput X: {X.shape}")
    print(f"W_gate: {W_gate.shape}, W_up: {W_up.shape}, W_down: {W_down.shape}")

    # ========================================================================
    # Load ATen-style operator registry
    # ========================================================================
    registry = OpRegistry.load()
    print(f"\nLoaded ops: {registry.list_ops()}")

    # ========================================================================
    # Hardware-accurate golden reference (MXFP8 quantization + BF16 intermediates)
    # This matches what the hardware does: all tensors are stored in HBM as MXFP8,
    # and each stage output is stored in VRAM as BF16.
    # ========================================================================
    print("\n--- Hardware-Accurate Golden Reference (MXFP8 + BF16 intermediates) ---")
    X_q = quantize_to_mxfp(X)
    W_gate_q = quantize_to_mxfp(W_gate)
    W_up_q = quantize_to_mxfp(W_up)
    W_down_q = quantize_to_mxfp(W_down)

    # Stage 1 & 2: up and gate projections → store as BF16
    # Hardware order: up projection written to gp4 (SiLU input), gate to gp6
    up_out = torch.matmul(X_q.float(), W_up_q.float()).to(torch.bfloat16)
    gate_out = torch.matmul(X_q.float(), W_gate_q.float()).to(torch.bfloat16)

    # Stage 3: SiLU(up) * gate → store as BF16  (hardware applies SiLU to up, not gate)
    silu_gate = (F.silu(up_out.float()) * gate_out.float()).to(torch.bfloat16)

    # Stage 4: down projection → BF16 output
    golden_out = torch.matmul(silu_gate.float(), W_down_q.float()).to(torch.bfloat16)

    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ========================================================================
    # PLENA backend (via registry, Backend.PLENA)
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare inputs:
    #   activation → loaded to VRAM via load_batch
    #   weights    → remain in HBM (accessed block-by-block by ffn_asm)
    x_input = prog.input("X", shape=(batch_size, hidden_size))
    w_gate_input = prog.input("W_gate", shape=(hidden_size, inter_dim))
    w_up_input = prog.input("W_up", shape=(hidden_size, inter_dim))
    w_down_input = prog.input("W_down", shape=(inter_dim, hidden_size))

    X_batch = prog.load_batch(x_input, name="X")

    # ATen-style dispatch: ffn_plena() is called with (prog, X_batch, w_gate, w_up, w_down)
    result = ops.ffn(prog, X_batch, w_gate_input, w_up_input, w_down_input)

    # Compile to ISA
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build" / "ffn"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensor = {
        "X": X,
        "W_gate": W_gate,
        "W_up": W_up,
        "W_down": W_down,
    }
    golden_result = {"original_output": golden_out}

    # FP SRAM preload: [0]=0.0, [1]=1.0 (legacy), [5]=1.0 (for SiLU sigmoid via ffn_plena slot 5)
    fp_preload = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 4

    create_sim_env(input_tensor, gen_code, golden_result, fp_preload, build_dir=str(build_dir))

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="ffn_aten",
        data=None,
        specified_data_order=["X", "W_gate", "W_up", "W_down"],
        build_path=build_dir,
    )

    # FFN result overwrites activation area in VRAM (in-place)
    x_vram_addr = prog._compiler.get_vram_addr(X_batch.name)

    comparison_params = {
        "start_row_idx": x_vram_addr // mlen,
        "num_rows": (batch_size * hidden_size) // mlen,
        "num_batches": batch_size,
        "elements_per_batch": hidden_size,
        "row_dim": mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {x_vram_addr // mlen} (overwrites activation)")
    run_and_assert(build_dir, "ffn", mlen=mlen, blen=blen)
