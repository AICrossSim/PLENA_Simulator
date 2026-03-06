"""
Shared infrastructure for model-layer testbench tests.

Provides:
  - ModelDims     dataclass (model configuration)
  - get_model_dims()        probe HF config (no weight download)
  - slice_dims_for_sim()    clip to simulator HBM capacity
  - load_ffn_weights()      load + transpose MLP weights
  - quantize_to_mxfp()      MXFP8 round-trip (HBM format)
  - golden_ffn()            hardware-accurate golden (MXFP8 + BF16 intermediates)
  - build_and_run_ffn_test() end-to-end: load → golden → ISA → sim → assert
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "tools"))

import json
import torch
import torch.nn.functional as F

from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware
from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops

from plena_program import PLENAProgram
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIDDEN_SLICE    = 128
INTER_SLICE     = 256
MLEN            = 64
BLEN            = 4
REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)


# ---------------------------------------------------------------------------
# ModelDims
# ---------------------------------------------------------------------------
@dataclass
class ModelDims:
    """Configuration dimensions extracted from a HuggingFace model."""
    hidden_size:   int
    inter_dim:     int
    num_heads:     int
    num_kv_heads:  int
    head_dim:      int
    model_id:      str


def get_model_dims(model_id: str) -> ModelDims:
    """Probe model config (no weight download)."""
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model_id)
    # Resolve text config for VLMs that wrap a language model
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden    = text_cfg.hidden_size
    inter     = text_cfg.intermediate_size
    n_heads   = text_cfg.num_attention_heads
    n_kv      = getattr(text_cfg, "num_key_value_heads", n_heads)
    head_dim  = hidden // n_heads
    return ModelDims(
        hidden_size=hidden,
        inter_dim=inter,
        num_heads=n_heads,
        num_kv_heads=n_kv,
        head_dim=head_dim,
        model_id=model_id,
    )


def slice_dims_for_sim(
    dims: ModelDims,
    hidden_slice: int = HIDDEN_SLICE,
    inter_slice:  int = INTER_SLICE,
) -> ModelDims:
    """Return a new ModelDims clipped to simulator HBM capacity."""
    return ModelDims(
        hidden_size=min(dims.hidden_size, hidden_slice),
        inter_dim=min(dims.inter_dim, inter_slice),
        num_heads=dims.num_heads,
        num_kv_heads=dims.num_kv_heads,
        head_dim=dims.head_dim,
        model_id=dims.model_id,
    )


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------
def load_ffn_weights(
    model_id: str,
    layer_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, ModelDims]:
    """
    Load FFN weights from a HuggingFace model.

    Returns W_gate, W_up, W_down in float32 with shape (hidden, inter) / (inter, hidden),
    transposed from HF's (out_features, in_features) storage convention.

    Supports:
      - LlamaForCausalLM-style: model.model.layers[i].mlp
      - SmolVLM2-style: model.text_model.layers[i].mlp
    """
    from transformers import AutoModelForCausalLM, AutoModel

    # Try CausalLM first (Llama, SmolLM2, clm-60m…)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        layer = model.model.layers[layer_idx]
    except Exception:
        # Fall back to AutoModel (SmolVLM2, etc.)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
        layer = model.text_model.layers[layer_idx]

    # HF stores (out_features, in_features); PLENA expects (in_features, out_features)
    W_gate = layer.mlp.gate_proj.weight.detach().T.contiguous()  # (hidden, inter)
    W_up   = layer.mlp.up_proj.weight.detach().T.contiguous()    # (hidden, inter)
    W_down = layer.mlp.down_proj.weight.detach().T.contiguous()  # (inter, hidden)

    dims = get_model_dims(model_id)
    return W_gate, W_up, W_down, dims


# ---------------------------------------------------------------------------
# MXFP8 quantization
# ---------------------------------------------------------------------------
def quantize_to_mxfp(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to MXFP8 matching HBM hardware format; return dequantized result."""
    orig_shape = tensor.shape
    tensor_2d  = tensor.float().reshape(-1, tensor.shape[-1])
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor_2d,
        width=8,
        exponent_width=4,
        exponent_bias_width=8,
        block_size=[1, 8],
    )
    return bm_x.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Hardware-accurate golden reference
# ---------------------------------------------------------------------------
def golden_ffn(
    X:      torch.Tensor,
    W_gate: torch.Tensor,
    W_up:   torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """
    Hardware-accurate FFN golden reference.

    Applies MXFP8 quantization (HBM storage) + BF16 intermediates (VRAM).
    Formula: w_down @ (silu(w_up @ x) * (w_gate @ x))
    Hardware applies SiLU to the UP projection, not gate.
    """
    X_q      = quantize_to_mxfp(X)
    W_gate_q = quantize_to_mxfp(W_gate)
    W_up_q   = quantize_to_mxfp(W_up)
    W_down_q = quantize_to_mxfp(W_down)

    up_out    = torch.matmul(X_q.float(), W_up_q.float()).to(torch.bfloat16)
    gate_out  = torch.matmul(X_q.float(), W_gate_q.float()).to(torch.bfloat16)
    silu_gate = (F.silu(up_out.float()) * gate_out.float()).to(torch.bfloat16)
    return torch.matmul(silu_gate.float(), W_down_q.float()).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# End-to-end test runner
# ---------------------------------------------------------------------------
def build_and_run_ffn_test(
    model_id:  str,
    asm_name:  str,
    build_dir: Path,
    layer_idx: int = 0,
    batch_size: int = 4,
    mlen:      int  = MLEN,
    blen:      int  = BLEN,
    seed:      int  = 42,
) -> None:
    """
    Full end-to-end FFN test for any HuggingFace model:
      1. Load real weights → slice to sim dims
      2. Compute hardware-accurate golden
      3. Build PLENA ISA program
      4. Write sim artifacts + run emulator
      5. Assert numerical match

    Args:
        model_id:   HuggingFace model ID (e.g. 'HuggingFaceTB/SmolLM2-135M')
        asm_name:   Short identifier used for .asm file naming
        build_dir:  Directory for sim artifacts
        layer_idx:  Decoder layer index to test (default 0)
        batch_size: Number of tokens / batch elements
        mlen:       Matrix tile length (default 64)
        blen:       Batch tile length (default 4)
        seed:       Random seed for activation tensor
    """
    print("=" * 80)
    print(f"FFN Test — {model_id}  (layer {layer_idx})")
    print("=" * 80)

    # ------------------------------------------------------------------ dims
    dims = get_model_dims(model_id)
    sim  = slice_dims_for_sim(dims)

    print(f"\nFull model dims: hidden={dims.hidden_size}, inter={dims.inter_dim}, "
          f"heads={dims.num_heads}/{dims.num_kv_heads}, head_dim={dims.head_dim}")
    print(f"Sim dims:        hidden={sim.hidden_size}, inter={sim.inter_dim} (sliced)")

    # ---------------------------------------------------------------- weights
    print(f"\nLoading weights from {model_id} layer {layer_idx}...")
    W_gate_full, W_up_full, W_down_full, _ = load_ffn_weights(model_id, layer_idx)

    # Slice to simulator capacity
    W_gate = W_gate_full[:sim.hidden_size, :sim.inter_dim].contiguous()
    W_up   = W_up_full  [:sim.hidden_size, :sim.inter_dim].contiguous()
    W_down = W_down_full[:sim.inter_dim,   :sim.hidden_size].contiguous()

    print(f"W_gate: {W_gate.shape}, range [{W_gate.min():.4f}, {W_gate.max():.4f}]")
    print(f"W_up:   {W_up.shape},   range [{W_up.min():.4f},   {W_up.max():.4f}]")
    print(f"W_down: {W_down.shape}, range [{W_down.min():.4f}, {W_down.max():.4f}]")

    # ----------------------------------------------------------- activation
    torch.manual_seed(seed)
    X = torch.randn(batch_size, sim.hidden_size)
    print(f"\nInput X: {X.shape}")

    # ---------------------------------------------------------------- golden
    print("\n--- Hardware-Accurate Golden Reference (MXFP8 + BF16 intermediates) ---")
    golden_out = golden_ffn(X, W_gate, W_up, W_down)
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ---------------------------------------------------------- PLENA ISA
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    real_data_ratio = REAL_DATA_RATIO
    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    x_input      = prog.input("X",      shape=(batch_size, sim.hidden_size))
    w_gate_input = prog.input("W_gate", shape=(sim.hidden_size, sim.inter_dim))
    w_up_input   = prog.input("W_up",   shape=(sim.hidden_size, sim.inter_dim))
    w_down_input = prog.input("W_down", shape=(sim.inter_dim, sim.hidden_size))

    X_batch = prog.load_batch(x_input, name="X")
    ops.ffn(prog, X_batch, w_gate_input, w_up_input, w_down_input)

    gen_code = prog.compile()
    lines    = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ---------------------------------------------------------- sim env
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensors = {
        "X":      X,
        "W_gate": W_gate,
        "W_up":   W_up,
        "W_down": W_down,
    }
    golden_result = {"original_output": golden_out}

    # FPRAM: slot0=0.0, slot1=1.0 (legacy), slot5=1.0 (SiLU sigmoid)
    fp_preload = [0.0, 1.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 4

    create_sim_env(
        input_tensors, gen_code, golden_result, fp_preload,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=asm_name,
        data=None,
        specified_data_order=["X", "W_gate", "W_up", "W_down"],
        build_path=build_dir,
    )

    # Result overwrites X activation area (FFN is in-place)
    x_vram_addr = prog._compiler.get_vram_addr(X_batch.name)
    comparison_params = {
        "start_row_idx":      x_vram_addr // mlen,
        "num_rows":           (batch_size * sim.hidden_size) // mlen,
        "num_batches":        batch_size,
        "elements_per_batch": sim.hidden_size,
        "row_dim":            mlen,
        "use_stride_mode":    sim.hidden_size > mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {x_vram_addr // mlen} (overwrites activation)")

    run_and_assert(build_dir, asm_name, mlen=mlen, blen=blen)
