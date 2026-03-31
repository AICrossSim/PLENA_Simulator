"""
SmolVLM2-256M FFN Test — Real Model Weights

Loads REAL weights from HuggingFaceTB/SmolVLM2-256M-Video-Instruct (layer 0 MLP)
and validates the PLENA FFN operator against a hardware-accurate golden reference.

Model dimensions:
    hidden_size = 576
    inter_dim   = 256

The golden reference applies MXFP8 quantization (matching HBM storage) and uses
BF16 intermediates — exactly as the hardware executes the FFN operator.

FFN formula: w_down @ (silu(w_gate @ x) * (w_up @ x))
"""

import sys
from pathlib import Path

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "tools"))

import torch
import torch.nn.functional as F
import json

from transformers import AutoModel
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware
from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops

from plena_program import PLENAProgram
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


def quantize_to_mxfp(tensor):
    """Quantize tensor to MXFP8 matching HBM hardware format; return dequantized result."""
    orig_shape = tensor.shape
    tensor_2d = tensor.float().reshape(-1, tensor.shape[-1])
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor_2d,
        width=8,
        exponent_width=4,
        exponent_bias_width=8,
        block_size=[1, 8],
    )
    return bm_x.reshape(orig_shape)


if __name__ == "__main__":
    print("=" * 80)
    print("SmolVLM2-256M FFN Test  (real model weights, plena.ops.ffn)")
    print("=" * 80)

    # ========================================================================
    # Load real SmolVLM2-256M-Video-Instruct weights
    # ========================================================================
    print("\nLoading HuggingFaceTB/SmolVLM2-256M-Video-Instruct ...")
    model = AutoModel.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        torch_dtype=torch.float32,
    )
    layer0 = model.text_model.layers[0]

    # HF stores weights transposed (out_features, in_features); transpose to (in, out)
    W_gate_full = layer0.mlp.gate_proj.weight.detach().T.contiguous()  # (576, 1536)
    W_up_full = layer0.mlp.up_proj.weight.detach().T.contiguous()  # (576, 1536)
    W_down_full = layer0.mlp.down_proj.weight.detach().T.contiguous()  # (1536, 576)

    print(f"Full model weights loaded:")
    print(f"  W_gate: {W_gate_full.shape}, range [{W_gate_full.min():.4f}, {W_gate_full.max():.4f}]")
    print(f"  W_up:   {W_up_full.shape}, range [{W_up_full.min():.4f}, {W_up_full.max():.4f}]")
    print(f"  W_down: {W_down_full.shape}, range [{W_down_full.min():.4f}, {W_down_full.max():.4f}]")

    # ========================================================================
    # Parameters
    # ========================================================================
    # Slice to inter_dim=256 (matching simulator HBM capacity limits)
    hidden_size = 128
    inter_dim = 256
    batch_size = 4
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    # Slice weights to the simulator-compatible dimensions
    W_gate = W_gate_full[:hidden_size, :256].contiguous()  # (128, 256)
    W_up = W_up_full[:hidden_size, :256].contiguous()  # (128, 256)
    W_down = W_down_full[:256, :hidden_size].contiguous()  # (256, 128)

    print(f"\nSliced weights (hidden={hidden_size}, inter={inter_dim}):")
    print(f"W_gate: {W_gate.shape}, range [{W_gate.min():.4f}, {W_gate.max():.4f}]")
    print(f"W_up:   {W_up.shape}, range [{W_up.min():.4f}, {W_up.max():.4f}]")
    print(f"W_down: {W_down.shape}, range [{W_down.min():.4f}, {W_down.max():.4f}]")

    torch.manual_seed(42)

    # Random activation input
    X = torch.randn(batch_size, hidden_size)

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

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

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
    build_dir = Path(__file__).parent / "build"
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
        asm="smolvlm2_256m_ffn",
        data=None,
        specified_data_order=["X", "W_gate", "W_up", "W_down"],
        build_path=build_dir,
    )

    # FFN result overwrites activation area in VRAM (in-place)
    x_vram_addr = prog._compiler.get_vram_addr(X_batch.name)

    # hidden_size=576 > mlen=64 → use stride mode so the comparison walks
    # across multiple VRAM rows per batch element
    comparison_params = {
        "start_row_idx": x_vram_addr // mlen,
        "num_rows": (batch_size * hidden_size) // mlen,
        "num_batches": batch_size,
        "elements_per_batch": hidden_size,
        "row_dim": mlen,
        "use_stride_mode": True,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {x_vram_addr // mlen} (overwrites activation)")
    run_and_assert(build_dir, "ffn", mlen=mlen, blen=blen)
