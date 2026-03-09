"""
ATen Compiler Test: Decoder Layer (RMSNorm + Attention + Residual + FFN) -> PLENA ISA.

Traces a simplified LLaMA-style decoder layer through torch.export, compiles to
PLENA ISA via compile_module, runs the Rust emulator, and checks numerical accuracy.

Decoder structure:
    residual = x
    h = rms_norm(x)
    q, k, v = q_proj(h), k_proj(h), v_proj(h)
    attn = scaled_dot_product_attention(q, k, v, scale=1/sqrt(d))
    attn = o_proj(attn)
    x = residual + attn           <-- aten.add.Tensor (residual connection)
    residual = x
    h = rms_norm(x)
    ffn = down_proj(silu(gate_proj(h)) * up_proj(h))   <-- FFN fusion
    x = residual + ffn            <-- aten.add.Tensor (residual connection)

New ATen ops covered:
    - aten.add.Tensor             (element-wise VRAM add for residual connections)
    - aten.scaled_dot_product_attention.default  (flash attention via VRAM->HBM store)

Parameters:
    seq_len     = 64  (= mlen)
    hidden_size = 64  (= head_dim)
    inter_dim   = 128 (safe for VRAM layout: gate_result_max=20539 < O addr 28736)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from plena.compiler.aten_compiler import compile_module, quantize_to_mxfp
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class SimpleLlamaDecoder(nn.Module):
    """Simplified LLaMA decoder layer (no RoPE, single head)."""

    def __init__(self, hidden_size=64, inter_dim=128):
        super().__init__()
        self.hidden_size = hidden_size
        # Attention block
        self.input_norm = nn.RMSNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # FFN block
        self.post_norm = nn.RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, inter_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, hidden_size, bias=False)

    def forward(self, x):
        # === Attention block ===
        residual = x
        h = self.input_norm(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        attn = F.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.hidden_size)
        )
        attn = self.o_proj(attn)
        x = residual + attn
        # === FFN block ===
        residual = x
        h = self.post_norm(x)
        ffn = self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        x = residual + ffn
        return x


# ---------------------------------------------------------------------------
# Golden reference (hardware-accurate: MXFP8 weights + BF16 intermediates)
# ---------------------------------------------------------------------------

def golden_decoder(model, x, hidden_size):
    """Compute hardware-accurate golden reference matching PLENA execution.

    - All 2D weights are transposed to (in, out) and MXFP8-quantized (HBM format).
    - All intermediate activations are cast to bfloat16 (VRAM format).
    - Attention uses MXFP8-quantized K, V (stored to HBM then reloaded).
    - FFN follows hardware op order: silu on gate_proj's output maps to up in hardware.
    """
    # Transpose and quantize all 2D weights
    W_q = quantize_to_mxfp(model.q_proj.weight.T.contiguous())
    W_k = quantize_to_mxfp(model.k_proj.weight.T.contiguous())
    W_v = quantize_to_mxfp(model.v_proj.weight.T.contiguous())
    W_o = quantize_to_mxfp(model.o_proj.weight.T.contiguous())
    W_gate = quantize_to_mxfp(model.gate_proj.weight.T.contiguous())
    W_up = quantize_to_mxfp(model.up_proj.weight.T.contiguous())
    W_down = quantize_to_mxfp(model.down_proj.weight.T.contiguous())

    scale = 1.0 / math.sqrt(hidden_size)

    # === Attention block ===
    # rms_norm (bfloat16 intermediate)
    x_bf16 = x.to(torch.bfloat16)
    rms = torch.rsqrt(x_bf16.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    h = (x_bf16.float() * rms).to(torch.bfloat16)

    # Projections (float matmul, bfloat16 output)
    q = torch.matmul(h.float(), W_q.float()).to(torch.bfloat16)
    k = torch.matmul(h.float(), W_k.float()).to(torch.bfloat16)
    v = torch.matmul(h.float(), W_v.float()).to(torch.bfloat16)

    # SDPA: K and V are stored to HBM (MXFP8) then reloaded for flash attention
    k_hbm = quantize_to_mxfp(k)
    v_hbm = quantize_to_mxfp(v)

    # Flash attention (scale + softmax + matmul)
    scores = torch.matmul(q.float(), k_hbm.float().T) * scale
    attn_weights = F.softmax(scores, dim=-1).to(torch.bfloat16)
    attn_out = torch.matmul(attn_weights.float(), v_hbm.float()).to(torch.bfloat16)

    # O projection
    o_out = torch.matmul(attn_out.float(), W_o.float()).to(torch.bfloat16)

    # Residual add
    x_after_attn = (x_bf16.float() + o_out.float()).to(torch.bfloat16)

    # === FFN block ===
    # rms_norm
    rms2 = torch.rsqrt(x_after_attn.float().pow(2).mean(-1, keepdim=True) + 1e-5)
    h2 = (x_after_attn.float() * rms2).to(torch.bfloat16)

    # FFN: hardware order is silu(W_up @ x) * (W_gate @ x) — note the gate/up swap
    # In ATen graph: gate_proj is silu'd, which maps to W_up in hardware
    up_out = torch.matmul(h2.float(), W_gate.float()).to(torch.bfloat16)    # silu'd branch → hardware "up"
    gate_out = torch.matmul(h2.float(), W_up.float()).to(torch.bfloat16)    # non-silu'd → hardware "gate"
    silu_gate = (F.silu(up_out.float()) * gate_out.float()).to(torch.bfloat16)
    ffn_out = torch.matmul(silu_gate.float(), W_down.float()).to(torch.bfloat16)

    # Residual add
    x_final = (x_after_attn.float() + ffn_out.float()).to(torch.bfloat16)

    return x_final


if __name__ == "__main__":
    print("=" * 80)
    print("ATen Compiler Test: Decoder Layer (RMSNorm + SDPA + Residual + FFN)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    hidden_size = 64
    inter_dim = 128
    seq_len = 64
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    eps = 1e-5

    torch.manual_seed(42)

    # ========================================================================
    # Model + test data
    # ========================================================================
    model = SimpleLlamaDecoder(hidden_size=hidden_size, inter_dim=inter_dim)
    x = torch.randn(seq_len, hidden_size) * 0.5

    print(f"\nInput x: {x.shape}")
    print(f"  hidden_size={hidden_size}, inter_dim={inter_dim}, seq_len={seq_len}")

    # ========================================================================
    # CPU golden reference (MXFP8 quantized weights, BF16 intermediates)
    # ========================================================================
    print("\n--- Hardware-Accurate Golden Reference (MXFP8 + BF16 intermediates) ---")
    golden_out = golden_decoder(model, x, hidden_size)
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0,:4].tolist()}")

    # ========================================================================
    # Compile with ATen compiler
    # ========================================================================
    print("\n--- ATen Compiler (torch.export -> PLENA ISA) ---")

    # FPRAM slot config for decoder pipeline:
    #   slot 1 = attn_scale     (flash_attention)
    #   slot 2 = -inf           (flash_attention online softmax)
    #   slot 3 = eps            (rms_norm)
    #   slot 4 = 1/hidden       (rms_norm)
    #   slot 5 = 1.0            (FFN SiLU)
    fp_config = {"eps_offset": 3, "reci_hid_offset": 4}

    isa_str, info = compile_module(
        model, (x,),
        mlen=mlen, blen=blen, real_data_ratio=real_data_ratio,
        fp_config=fp_config,
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
            # User input (activation)
            input_tensor[name] = x

    golden_result = {"original_output": golden_out}

    scale = 1.0 / math.sqrt(hidden_size)

    # FP SRAM preload matching decoder pipeline slots:
    #   [0]=0.0, [1]=attn_scale, [2]=-inf, [3]=eps, [4]=1/hidden, [5]=1.0 (SiLU)
    fp_preload = [0.0, scale, float("-inf"), eps, 1.0 / hidden_size, 1.0] + [0.0] * 4

    create_sim_env(
        input_tensor, isa_str, golden_result, fp_preload,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="aten_compiler_decoder",
        data=None,
        specified_data_order=hbm_input_order,
        build_path=build_dir,
    )

    # Output location: the result of the second add (residual + FFN output).
    # This lives at the VRAM address of the FFN output (which overwrites the
    # rms_norm_1 copy's activation area, and then has residual added in-place).
    o_vram_addr = prog._compiler.get_vram_addr(output_var.name)

    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (seq_len * hidden_size) // mlen,
        "num_batches": seq_len,
        "elements_per_batch": hidden_size,
        "row_dim": mlen,
        "use_stride_mode": False,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(isa_str)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {o_vram_addr // mlen}")

    # ========================================================================
    # Run emulator and check
    # ========================================================================
    run_and_assert(build_dir, "aten_compiler_decoder", mlen=mlen, blen=blen)
