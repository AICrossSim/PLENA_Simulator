"""
SmolVLM2 Vision Encoder Pipeline Test — conv2d patch embed + decoder pipeline

Runs the following pipeline on PLENA hardware:
    conv2d (patch embed) -> embedding_add -> rms_norm -> flash_attention -> ffn -> rms_norm

Synthetic conv2d params (C_in=4, K=4) due to K_col=64 hardware constraint
(real SmolVLM2 needs K_col=588 = 3*14*14, exceeding VLEN=64).

Language model weights: real sliced SmolVLM2-256M-Video-Instruct weights
(W_gate, W_up, W_down from layer 0, sliced to hidden=64, inter=128).

Note: Uses RMSNorm+SwiGLU approximation; real SmolVLM2 SigLIP uses LayerNorm+GELU.
The purpose is profiling the instruction mix, not exact SigLIP accuracy.

Parameters:
    C_in        = 4       input channels (synthetic)
    K           = 4       kernel size (synthetic)
    H           = 67      input height -> OH = H - K + 1 = 64
    W           = 4       input width (= K, giving OW = W - K + 1 = 1)
    W_padded    = 64      padded to 64 for HBM alignment
    OH          = 64      output height = seq_len
    OW          = 1       output width
    M           = 64      total output positions = OH*OW = seq_len
    hidden_size = 64      = K_col = C_in*K*K = 4*4*4 (conv2d output channels = C_out)
    seq_len     = 64      = M = OH
    inter_dim   = 128     safe FFN intermediate size (avoids VRAM conflict)
    mlen        = 64
    blen        = 4
"""

import sys
import math
from pathlib import Path


import json
import torch
import torch.nn.functional as F

from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena_compiler import PlenaCompiler
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.emulator_runner import run_and_assert
from transactional_emulator.testbench.model_layer_test_builder import quantize_to_mxfp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def im2col(input_4d, kernel_size, stride=1, padding=0):
    """Transform input [B, C, H, W] -> [B*OH*OW, C*K*K] via torch.nn.Unfold."""
    unfold = torch.nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding)
    col = unfold(input_4d.float())  # [B, C*K*K, OH*OW]
    col = col.permute(0, 2, 1).reshape(-1, input_4d.shape[1] * kernel_size * kernel_size)
    return col


def _rms_norm_ref(x, eps):
    """CPU reference: RMS normalization (float32)."""
    x = x.float()
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms


def _flash_attn_ref(Q, K, V, scale):
    """CPU reference: scaled dot-product attention."""
    scores = (Q @ K.T) * scale
    attn = F.softmax(scores, dim=-1)
    return attn @ V


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("SmolVLM2 Vision Encoder Test")
    print("  conv2d (patch embed) -> embedding_add -> rms_norm -> flash_attention -> ffn -> rms_norm")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    # conv2d patch embed (synthetic — K_col constraint)
    B = 1
    C_in = 4  # input channels
    K_size = 4  # kernel size (K x K)
    H = 67  # input height -> OH = H - K + 1 = 64
    W = 4  # input width (= K, giving OW = 1)
    stride = 1
    padding = 0
    W_padded = 64  # padded to multiple of 64 for HBM alignment

    OH = (H - K_size + 2 * padding) // stride + 1  # 64
    OW = (W - K_size + 2 * padding) // stride + 1  # 1
    M = B * OH * OW  # 64 = seq_len
    K_col = C_in * K_size * K_size  # 64 = hidden_size = C_out

    # decoder params (must match conv2d output dims)
    C_out = K_col  # 64 — conv2d output channels = hidden_size
    hidden_size = K_col  # 64
    seq_len = M  # 64
    inter_dim = 128  # safe FFN intermediate (avoids VRAM conflict)
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)  # MX8 block format
    scale = 1.0 / math.sqrt(hidden_size)

    print(f"\nConv2d params: C_in={C_in}, H={H}, W={W}, K={K_size}")
    print(f"  OH={OH}, OW={OW}, M={M}, K_col={K_col}, C_out={C_out}")
    print(f"  W_padded={W_padded}")
    print(f"\nDecoder params: seq_len={seq_len}, hidden_size={hidden_size}, inter_dim={inter_dim}")
    print(f"  mlen={mlen}, blen={blen}, attn_scale={scale:.6f}")

    # ========================================================================
    # Load real SmolVLM2-256M weights for decoder part
    # ========================================================================
    print("\nLoading HuggingFaceTB/SmolVLM2-256M-Video-Instruct ...")
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        torch_dtype=torch.float32,
    )
    layer0 = model.text_model.layers[0]

    # HF stores weights transposed (out_features, in_features); transpose to (in, out)
    W_gate_full = layer0.mlp.gate_proj.weight.detach().T.contiguous()
    W_up_full = layer0.mlp.up_proj.weight.detach().T.contiguous()
    W_down_full = layer0.mlp.down_proj.weight.detach().T.contiguous()

    # Slice to simulator-compatible dimensions
    W_gate = W_gate_full[:hidden_size, :inter_dim].contiguous()  # (64, 128)
    W_up = W_up_full[:hidden_size, :inter_dim].contiguous()  # (64, 128)
    W_down = W_down_full[:inter_dim, :hidden_size].contiguous()  # (128, 64)

    # Get eps from input_layernorm
    norm = layer0.input_layernorm
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5))

    del model  # free memory

    print(f"\nSliced weights (hidden={hidden_size}, inter={inter_dim}):")
    print(f"  W_gate: {W_gate.shape}, range [{W_gate.min():.4f}, {W_gate.max():.4f}]")
    print(f"  W_up:   {W_up.shape},   range [{W_up.min():.4f}, {W_up.max():.4f}]")
    print(f"  W_down: {W_down.shape}, range [{W_down.min():.4f}, {W_down.max():.4f}]")
    print(f"  eps: {eps}")

    # ========================================================================
    # Generate test data
    # ========================================================================
    torch.manual_seed(42)

    # conv2d input and weights (synthetic — random)
    input_4d = torch.randn(B, C_in, H, W)
    weight_4d = torch.randn(C_out, C_in, K_size, K_size)

    # Raw input arranged for HBM: (C_in*H, W_padded)
    raw_input = torch.zeros(C_in * H, W_padded)
    for c in range(C_in):
        raw_input[c * H : (c + 1) * H, :W] = input_4d[0, c, :, :]

    # conv2d weight as 2D matrix: (K_col, C_out)
    W_2d = weight_4d.float().reshape(C_out, -1).T.contiguous()  # (K_col, C_out)

    # Position embeddings (random)
    pos_weight = torch.randn(seq_len, hidden_size)

    # K, V for flash_attention (random — no real KV cache in this test)
    K_mat = torch.randn(seq_len, hidden_size)
    V_mat = torch.randn(seq_len, hidden_size)

    print(f"\nraw_input (HBM layout): {raw_input.shape}")
    print(f"W_2d:     {W_2d.shape}")
    print(f"pos_weight: {pos_weight.shape}")
    print(f"K_mat:    {K_mat.shape}")
    print(f"V_mat:    {V_mat.shape}")

    # ========================================================================
    # CPU golden reference (MXFP8 quantized HBM tensors + BF16 intermediates)
    # ========================================================================
    print("\n--- CPU Golden Reference (MXFP8 + BF16 intermediates) ---")

    # Quantize all HBM-stored tensors to MXFP8
    W_2d_q = quantize_to_mxfp(W_2d)
    K_q = quantize_to_mxfp(K_mat)
    V_q = quantize_to_mxfp(V_mat)
    W_gate_q = quantize_to_mxfp(W_gate)
    W_up_q = quantize_to_mxfp(W_up)
    W_down_q = quantize_to_mxfp(W_down)

    # Step 1: conv2d golden (im2col -> matmul with MXFP8 weights)
    X_col = im2col(input_4d, K_size, stride=stride, padding=padding)  # (M, K_col)
    # im2col data comes from on-chip extraction (bfloat16 intermediate)
    X_col_bf16 = X_col.to(torch.bfloat16)
    conv_out = torch.matmul(X_col_bf16.float(), W_2d_q.float()).to(torch.bfloat16)
    print(f"  conv2d golden: {conv_out.shape}")
    print(f"  conv_out[0,:4]: {conv_out[0, :4].tolist()}")

    # Step 2: embedding_add golden (add position embeddings)
    # pos_weight is stored in HBM as MXFP8
    pos_q = quantize_to_mxfp(pos_weight)
    # After load_batch, pos is in VRAM as bfloat16
    X_gold = (conv_out.float() + pos_q.to(torch.bfloat16).float()).to(torch.bfloat16)
    print(f"  after embedding_add: {X_gold.shape}")

    # Step 3: rms_norm golden (bfloat16 intermediate)
    rms_gold = torch.rsqrt(X_gold.float().pow(2).mean(-1, keepdim=True) + eps).to(torch.bfloat16)
    X_gold = X_gold * rms_gold
    print(f"  after rms_norm_1: {X_gold.shape}")

    # Step 4: flash_attention golden (MXFP8 K/V)
    X_gold = _flash_attn_ref(X_gold.float(), K_q.float(), V_q.float(), scale)
    X_gold = X_gold.to(torch.bfloat16)
    print(f"  after flash_attn: {X_gold.shape}")

    # Step 5: FFN golden (MXFP8 weights + BF16 intermediates)
    up_out = torch.matmul(X_gold.float(), W_up_q.float()).to(torch.bfloat16)
    gate_out = torch.matmul(X_gold.float(), W_gate_q.float()).to(torch.bfloat16)
    silu_gate = (F.silu(up_out.float()) * gate_out.float()).to(torch.bfloat16)
    X_gold = torch.matmul(silu_gate.float(), W_down_q.float()).to(torch.bfloat16)
    print(f"  after ffn: {X_gold.shape}")

    # Step 6: rms_norm golden (final)
    X_gold = _rms_norm_ref(X_gold.float(), eps)
    print(f"  after rms_norm_2: {X_gold.shape}")

    golden_out = X_gold
    print(f"\n  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ========================================================================
    # PLENA backend — ISA generation
    # ========================================================================
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare HBM inputs — order determines HBM layout
    input_raw_var = prog.input("input_raw", shape=(C_in * H, W_padded))
    w_2d_var = prog.input("W_2d", shape=(K_col, C_out))
    pos_input = prog.input("POS", shape=(seq_len, hidden_size))
    k_input = prog.input("K", shape=(seq_len, hidden_size))
    v_input = prog.input("V", shape=(seq_len, hidden_size))
    wgate_input = prog.input("W_gate", shape=(hidden_size, inter_dim))
    wup_input = prog.input("W_up", shape=(hidden_size, inter_dim))
    wdown_input = prog.input("W_down", shape=(inter_dim, hidden_size))

    # Step 1: conv2d patch embedding (on-chip im2col + systolic matmul)
    # Y is a VRAMMatrixVar, shape (M, C_out) = (64, 64) — already in VRAM
    Y = ops.conv2d(
        prog,
        input_raw_var,
        w_2d_var,
        C_in=C_in,
        H=H,
        W=W,
        K=K_size,
        OH=OH,
        OW=OW,
        M=M,
        W_padded=W_padded,
        fp_one_reg=5,  # slot 5 = 1.0; slot 1 reserved for flash_attention attn_scale
    )

    # Step 2: embedding_add — need POS in VRAM
    POS_batch = prog.load_batch(pos_input, name="POS")
    ops.embedding_add(prog, Y, POS_batch)  # Y += POS (in-place)

    # Step 3: rms_norm (eps at FPRAM slot 3, 1/hidden at slot 4)
    prog.rms_norm(Y, eps_offset=3, reci_hid_offset=4)

    # Step 4: flash_attention (K, V from HBM)
    O = ops.flash_attention(prog, Y, k_input, v_input, scale)

    # Step 5: FFN (SwiGLU, real sliced SmolVLM2 weights)
    ops.ffn(prog, O, wgate_input, wup_input, wdown_input)

    # Step 6: rms_norm on O (final normalization)
    prog.rms_norm(O, eps_offset=3, reci_hid_offset=4)

    # Compile to ISA
    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build" / "smolvlm2_vision_encoder"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensors = {
        "input_raw": raw_input.float(),
        "W_2d": W_2d,
        "POS": pos_weight,
        "K": K_mat,
        "V": V_mat,
        "W_gate": W_gate,
        "W_up": W_up,
        "W_down": W_down,
    }
    golden_result = {"original_output": golden_out}

    # FPRAM layout:
    #   slot 0 = 0.0     (conv2d im2col zero init)
    #   slot 1 = 1.0     (conv2d basis vector construction)
    #   slot 2 = -inf    (flash_attention softmax init)
    #   slot 3 = eps     (rms_norm epsilon)
    #   slot 4 = 1/hid   (rms_norm reciprocal hidden)
    #   slot 5 = 1.0     (FFN SiLU)
    #   slots 6-9 = 0.0  (padding)
    fp_preload = [0.0, scale, float("-inf"), eps, 1.0 / hidden_size, 1.0] + [0.0] * 4
    # slot 0: 0.0          (hw constant)
    # slot 1: attn_scale   (flash_attention scale = 1/sqrt(hidden_size))
    # slot 2: -inf         (flash_attention softmax init)
    # slot 3: eps          (rms_norm epsilon)
    # slot 4: 1/hidden     (rms_norm reciprocal)
    # slot 5: 1.0          (conv2d fp_one_reg=5 + FFN SiLU)

    create_sim_env(
        input_tensors,
        gen_code,
        golden_result,
        fp_preload,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="smolvlm2_vision_encoder",
        data=None,
        specified_data_order=["input_raw", "W_2d", "POS", "K", "V", "W_gate", "W_up", "W_down"],
        build_path=build_dir,
    )

    # Result is at O's VRAM location (flash_attention allocates O, FFN + rms_norm in-place)
    o_vram_addr = prog._compiler.get_vram_addr(O.name)

    comparison_params = {
        "start_row_idx": o_vram_addr // mlen,
        "num_rows": (seq_len * hidden_size) // mlen,
        "num_batches": seq_len,
        "elements_per_batch": hidden_size,
        "row_dim": mlen,
        "use_stride_mode": hidden_size > mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {o_vram_addr // mlen} (O from flash_attention)")

    run_and_assert(build_dir, "smolvlm2_vision_encoder", mlen=mlen, blen=blen)
