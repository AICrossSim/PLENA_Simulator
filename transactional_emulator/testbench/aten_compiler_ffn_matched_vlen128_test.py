"""
ATen Compiler Test: FFN (gate_proj + up_proj + down_proj with SiLU) -> PLENA ISA via compile_module.

Traces an FFN module with torch.export, detects the SiLU-gated FFN pattern,
fuses it into a single ffn_plena dispatch, runs the Rust emulator,
and checks numerical accuracy against the CPU golden reference.

FFN formula: down_proj(silu(gate_proj(x)) * up_proj(x))
Hardware:    w_down @ (silu(w_up @ x) * (w_gate @ x))
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from plena.compiler.aten_compiler import compile_module, quantize_to_mxfp
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


class FFN(nn.Module):
    """SiLU-gated FFN matching LLaMA MLP structure."""

    def __init__(self, hidden_size, inter_dim):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inter_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


if __name__ == "__main__":
    print("=" * 80)
    print("ATen Compiler Test: FFN (SiLU-gated, compile_module fusion) - Matched Dims")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    hidden_size = 128
    inter_dim = 256
    batch_size = 8
    mlen = 128
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    torch.manual_seed(42)

    # ========================================================================
    # Model + test data
    # ========================================================================
    model = FFN(hidden_size, inter_dim)
    x = torch.randn(batch_size, hidden_size)

    print(f"\nInput x: {x.shape}")
    print(f"gate_proj.weight: {model.gate_proj.weight.shape}")
    print(f"up_proj.weight:   {model.up_proj.weight.shape}")
    print(f"down_proj.weight: {model.down_proj.weight.shape}")

    # ========================================================================
    # CPU golden reference (MXFP8 quantized weights, BF16 intermediates)
    # Matches hardware execution order from ffn_plena:
    #   output = w_down @ (silu(w_up @ x) * (w_gate @ x))
    # ATen graph has silu on gate_proj, so:
    #   ATen silu'd branch (gate_proj) maps to w_up in ffn_plena
    #   ATen non-silu'd branch (up_proj) maps to w_gate in ffn_plena
    # ========================================================================
    print("\n--- Hardware-Accurate Golden Reference (MXFP8 + BF16 intermediates) ---")

    # Transpose weights to (in, out) as PLENA expects, then quantize
    W_gate_T = model.gate_proj.weight.T.contiguous()  # (hidden, inter)
    W_up_T = model.up_proj.weight.T.contiguous()  # (hidden, inter)
    W_down_T = model.down_proj.weight.T.contiguous()  # (inter, hidden)

    W_gate_q = quantize_to_mxfp(W_gate_T)
    W_up_q = quantize_to_mxfp(W_up_T)
    W_down_q = quantize_to_mxfp(W_down_T)

    # ffn_plena hardware order: up projection -> SiLU, gate projection -> mul
    # In ATen graph: gate_proj is the silu'd branch -> maps to "up" in hardware
    #                up_proj is the non-silu'd branch -> maps to "gate" in hardware
    up_out = torch.matmul(x.float(), W_gate_q.float()).to(torch.bfloat16)  # silu'd
    gate_out = torch.matmul(x.float(), W_up_q.float()).to(torch.bfloat16)  # non-silu'd

    silu_gate = (F.silu(up_out.float()) * gate_out.float()).to(torch.bfloat16)
    golden_out = torch.matmul(silu_gate.float(), W_down_q.float()).to(torch.bfloat16)

    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ========================================================================
    # Compile with ATen compiler (FFN fusion)
    # ========================================================================
    print("\n--- ATen Compiler (torch.export -> FFN fusion -> PLENA ISA) ---")
    isa_str, info = compile_module(
        model,
        (x,),
        mlen=mlen,
        blen=blen,
        real_data_ratio=real_data_ratio,
    )

    prog = info["prog"]
    tensor_map = info["tensor_map"]
    input_names = info["input_names"]
    hbm_input_order = info["hbm_input_order"]
    output_var = info["output_var"]
    state_dict_tensors = info["state_dict_tensors"]

    lines = isa_str.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")
    print(f"Input names (activations): {input_names}")
    print(f"HBM input order: {hbm_input_order}")
    print(f"State dict tensors: {list(state_dict_tensors.keys())}")
    print(f"Output var: {output_var}")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    # Build input_tensor dict matching hbm_input_order
    input_tensor = {}
    for name in hbm_input_order:
        if name in state_dict_tensors:
            input_tensor[name] = state_dict_tensors[name]
        else:
            input_tensor[name] = x

    golden_result = {"original_output": golden_out}

    # FP SRAM preload: [0]=0.0, [1]=1.0 (legacy), [5]=1.0 (for SiLU sigmoid via ffn_plena slot 5)
    fp_preload = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 4

    create_sim_env(
        input_tensor,
        isa_str,
        golden_result,
        fp_preload,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="aten_compiler_ffn_matched",
        data=None,
        specified_data_order=hbm_input_order,
        build_path=build_dir,
    )

    # FFN result overwrites activation area in VRAM (in-place)
    x_vram_addr = prog._compiler.get_vram_addr(output_var.name)

    comparison_params = {
        "start_row_idx": x_vram_addr // mlen,
        "num_rows": (batch_size * hidden_size) // mlen,
        "num_batches": batch_size,
        "elements_per_batch": hidden_size,
        "row_dim": mlen,
        "use_stride_mode": False,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(isa_str)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {x_vram_addr // mlen} (overwrites activation)")

    # ========================================================================
    # Run emulator and check
    # ========================================================================
    run_and_assert(build_dir, "aten_compiler_ffn_matched", mlen=mlen, blen=blen)
