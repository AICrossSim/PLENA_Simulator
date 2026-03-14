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


# ---------------------------------------------------------------------------
# Decoder weight loading
# ---------------------------------------------------------------------------
def _load_decoder_weights_partial(
    model_id: str,
    layer_idx: int,
    hidden_slice: int,
    inter_slice: int,
    trust_remote_code: bool,
    head_dim: int,
    n_kv: int,
    eps: float,
    rope_theta: float,
) -> Dict[str, object]:
    """
    Partial-load path: use safetensors shard index to download and read only the
    specific layer tensors needed, avoiding loading the full model (~16GB for 8B models).

    Auto-detects two naming conventions:
      Standard LLaMA:  model.layers.{i}.mlp.{gate,up,down}_proj.weight
                       model.layers.{i}.self_attn.{k,v}_proj.weight
      LLaDA custom:    model.transformer.blocks.{i}.{ff_proj,up_proj,ff_out}.weight
                       model.transformer.blocks.{i}.{k,v}_proj.weight
    """
    import json as _json
    from huggingface_hub import hf_hub_download
    from safetensors.torch import safe_open

    # Download shard index
    index_path = hf_hub_download(
        repo_id=model_id,
        filename="model.safetensors.index.json",
    )
    with open(index_path) as f:
        index = _json.load(f)
    weight_map = index["weight_map"]  # key -> shard filename

    # Auto-detect naming convention by probing the index
    llama_gate = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    llada_gate = f"model.transformer.blocks.{layer_idx}.ff_proj.weight"

    if llama_gate in weight_map:
        # Standard LLaMA / Mistral / Qwen naming
        p = f"model.layers.{layer_idx}."
        key_map = {
            "gate": f"{p}mlp.gate_proj.weight",
            "up":   f"{p}mlp.up_proj.weight",
            "down": f"{p}mlp.down_proj.weight",
            "k":    f"{p}self_attn.k_proj.weight",
            "v":    f"{p}self_attn.v_proj.weight",
        }
    elif llada_gate in weight_map:
        # LLaDA-8B custom naming: ff_proj=gate, up_proj=up, ff_out=down
        p = f"model.transformer.blocks.{layer_idx}."
        key_map = {
            "gate": f"{p}ff_proj.weight",
            "up":   f"{p}up_proj.weight",
            "down": f"{p}ff_out.weight",
            "k":    f"{p}k_proj.weight",
            "v":    f"{p}v_proj.weight",
        }
    else:
        # Show available keys to help debug
        sample = [k for k in weight_map if f".{layer_idx}." in k][:10]
        raise RuntimeError(
            f"Unknown weight naming convention for {model_id}. "
            f"Sample keys with layer {layer_idx}: {sample}"
        )

    print(f"  [partial_load] Detected naming: {'LLaDA' if llada_gate in weight_map else 'LLaMA'}")
    needed_keys = list(key_map.values())

    # Find and download only the shards that contain our keys
    shards_needed = set(weight_map[k] for k in needed_keys if k in weight_map)
    missing = [k for k in needed_keys if k not in weight_map]
    if missing:
        raise RuntimeError(f"Keys not found in weight map: {missing}")

    print(f"  [partial_load] Downloading {len(shards_needed)} shard(s) for layer {layer_idx}...")
    shard_paths = {}
    for shard in shards_needed:
        shard_paths[shard] = hf_hub_download(repo_id=model_id, filename=shard)

    # Load only the specific tensors we need (safetensors lazy read per key)
    loaded = {}
    for role, key in key_map.items():
        shard = weight_map[key]
        with safe_open(shard_paths[shard], framework="pt", device="cpu") as f:
            loaded[role] = f.get_tensor(key).float()

    # Transpose (HF stores (out, in)) and slice to sim dims.
    # K/V column slice uses min(model_head_dim, hidden_slice) so Q and K/V
    # have matching dimensions in the sim (sim uses head_dim = hidden_size).
    sim_head_dim = min(head_dim, hidden_slice)
    W_gate = loaded["gate"].T.contiguous()[:hidden_slice, :inter_slice]
    W_up   = loaded["up"].T.contiguous()[:hidden_slice, :inter_slice]
    W_down = loaded["down"].T.contiguous()[:inter_slice, :hidden_slice]
    W_k    = loaded["k"].T.contiguous()[:hidden_slice, :sim_head_dim]
    W_v    = loaded["v"].T.contiguous()[:hidden_slice, :sim_head_dim]

    return dict(
        W_gate=W_gate, W_up=W_up, W_down=W_down,
        W_k=W_k, W_v=W_v,
        head_dim=sim_head_dim, eps=eps, rope_theta=rope_theta,
    )


def load_decoder_weights(
    model_id: str,
    layer_idx: int = 0,
    hidden_slice: int = 64,
    inter_slice: int = 256,
    trust_remote_code: bool = False,
    partial_load: bool = False,
) -> Dict[str, object]:
    """
    Load and slice all weights needed for a single-layer decoder pipeline test.

    Args:
        partial_load: If True, use safetensors shard index to load only the specific
                      layer tensors (avoids loading the full model — use for large
                      models like 7B+ where full load exceeds available RAM).

    Returns dict with:
        W_gate:     (hidden_slice, inter_slice)  float32
        W_up:       (hidden_slice, inter_slice)  float32
        W_down:     (inter_slice, hidden_slice)  float32
        W_k:        (hidden_slice, head_dim)     float32  — KV head 0
        W_v:        (hidden_slice, head_dim)     float32  — KV head 0
        head_dim:   int
        eps:        float  (from input_layernorm)
        rope_theta: float
    """
    from transformers import AutoModelForCausalLM, AutoModel, AutoConfig

    # Get config for dims (always needed, lightweight)
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    text_cfg = getattr(cfg, "text_config", cfg)
    n_heads = text_cfg.num_attention_heads
    n_kv = getattr(text_cfg, "num_key_value_heads", n_heads)
    head_dim = text_cfg.hidden_size // n_heads
    rope_theta = getattr(text_cfg, "rope_theta", 10000.0)
    eps = getattr(text_cfg, "rms_norm_eps", 1e-5)

    if partial_load:
        print(f"  [partial_load] Downloading only layer {layer_idx} tensors via safetensors shards...")
        return _load_decoder_weights_partial(
            model_id, layer_idx, hidden_slice, inter_slice,
            trust_remote_code, head_dim, n_kv, eps, rope_theta,
        )

    # Full model load (suitable for small models, e.g. <1B params)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)
        layer = model.model.layers[layer_idx]
    except Exception:
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)
        layer = model.text_model.layers[layer_idx]

    norm = layer.input_layernorm
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5))

    # FFN weights: HF stores (out, in) -> transpose to (in, out) -> slice
    W_gate = layer.mlp.gate_proj.weight.detach().T.contiguous()[:hidden_slice, :inter_slice]
    W_up   = layer.mlp.up_proj.weight.detach().T.contiguous()[:hidden_slice, :inter_slice]
    W_down = layer.mlp.down_proj.weight.detach().T.contiguous()[:inter_slice, :hidden_slice]

    # Attention: KV-head 0 (handles GQA: n_kv <= n_heads)
    # Cap K/V head_dim to hidden_slice so Q and K/V match in the sim
    sim_head_dim = min(head_dim, hidden_slice)
    W_k_full = layer.self_attn.k_proj.weight.detach().T.contiguous()
    W_v_full = layer.self_attn.v_proj.weight.detach().T.contiguous()
    W_k = W_k_full[:hidden_slice, :sim_head_dim].contiguous()
    W_v = W_v_full[:hidden_slice, :sim_head_dim].contiguous()
    head_dim = sim_head_dim

    return dict(
        W_gate=W_gate, W_up=W_up, W_down=W_down,
        W_k=W_k, W_v=W_v,
        head_dim=head_dim, eps=eps, rope_theta=rope_theta,
    )


# ---------------------------------------------------------------------------
# Decoder pipeline helpers
# ---------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """rotate_half: [-x[d//2:], x[:d//2]]"""
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def _make_rope_tables(seq_len: int, head_dim: int, theta: float = 10000.0):
    """Compute RoPE cos/sin tables, shape (seq_len, head_dim)."""
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half).float() / half))
    positions = torch.arange(seq_len).float()
    angles = torch.outer(positions, freqs)          # (seq_len, half)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat([cos_half, cos_half], dim=-1)   # (seq_len, head_dim)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin


def _ffn_ref(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    """CPU reference: SwiGLU FFN (SiLU applied to W_up projection)."""
    gate = x @ W_gate
    up = F.silu(x @ W_up)
    return (up * gate) @ W_down


def _flash_attn_ref(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
    """CPU reference: scaled dot-product attention."""
    scores = (Q @ K.T) * scale
    attn = F.softmax(scores, dim=-1)
    return attn @ V


def _rms_norm_ref(x: torch.Tensor, eps: float) -> torch.Tensor:
    """CPU reference: RMS normalization (float32)."""
    x = x.float()
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x / rms


# ---------------------------------------------------------------------------
# End-to-end decoder pipeline test
# ---------------------------------------------------------------------------
def build_and_run_decoder_test(
    model_id:    str,
    asm_name:    str,
    build_dir:   Path,
    layer_idx:   int = 0,
    seq_len:     int = 64,
    hidden_size: int = 64,
    inter_dim:   int = 128,
    mlen:        int = MLEN,
    blen:        int = BLEN,
    seed:        int = 42,
    trust_remote_code: bool = False,
    partial_load: bool = False,
) -> None:
    """
    Full end-to-end single-layer decoder pipeline test with real model weights:
        embedding_add -> rms_norm -> rope -> flash_attention -> ffn -> rms_norm

    K and V are precomputed from real W_k, W_v weights applied to random context.
    FFN uses real sliced weights.

    Args:
        model_id:    HuggingFace model ID (e.g. 'HuggingFaceTB/SmolLM2-135M')
        asm_name:    Short identifier used for .asm file naming
        build_dir:   Directory for sim artifacts
        layer_idx:   Decoder layer index to test (default 0)
        seq_len:     Sequence length (default 64, = mlen)
        hidden_size: Hidden dimension (default 64, = head_dim = mlen)
        inter_dim:   FFN intermediate dimension (default 128, sliced from full).
                     Must satisfy: 4096 + inter_dim*64 + (inter_dim//64 - 1)*4096 + 4155 < O_vram_addr.
                     With hidden=64 and O at row 449 (addr=28736), max safe inter_dim=192; use 128.
        mlen:        Matrix tile length (default 64)
        blen:        Batch tile length (default 4)
        seed:        Random seed
    """
    import math

    print("=" * 80)
    print(f"Decoder Pipeline Test — {model_id}  (layer {layer_idx})")
    print("  embedding_add -> rms_norm -> rope -> flash_attention -> ffn -> rms_norm")
    print("=" * 80)

    head_dim = hidden_size  # must equal mlen for flash_attention
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    scale = 1.0 / math.sqrt(head_dim)

    # ---------------------------------------------------------------- weights
    print(f"\nLoading weights from {model_id} layer {layer_idx}...")
    w = load_decoder_weights(model_id, layer_idx, hidden_slice=hidden_size, inter_slice=inter_dim, trust_remote_code=trust_remote_code, partial_load=partial_load)
    W_gate = w["W_gate"]
    W_up   = w["W_up"]
    W_down = w["W_down"]
    W_k    = w["W_k"]
    W_v    = w["W_v"]
    eps    = w["eps"]
    rope_theta = w["rope_theta"]

    print(f"  head_dim={w['head_dim']}, eps={eps}, rope_theta={rope_theta}")
    print(f"  W_gate: {W_gate.shape}, W_up: {W_up.shape}, W_down: {W_down.shape}")
    print(f"  W_k: {W_k.shape}, W_v: {W_v.shape}")

    # ----------------------------------------------------------- test data
    torch.manual_seed(seed)
    token_embeds = torch.randn(seq_len, hidden_size)
    pos_weight   = torch.randn(seq_len, hidden_size)
    X_ctx        = torch.randn(seq_len, hidden_size)

    # Precompute K, V from context
    K_mat = X_ctx @ W_k   # (seq_len, head_dim)
    V_mat = X_ctx @ W_v   # (seq_len, head_dim)

    cos, sin = _make_rope_tables(seq_len, head_dim, theta=rope_theta)

    # Precompute Q_rot from bfloat16-approximated intermediate
    # PLENA computes embedding_add + rms_norm in bfloat16; Q_rot must match
    X_embed_bf16 = token_embeds.to(torch.bfloat16) + pos_weight.to(torch.bfloat16)
    rms_bf16 = torch.rsqrt(
        X_embed_bf16.float().pow(2).mean(-1, keepdim=True) + eps
    ).to(torch.bfloat16)
    X_norm_bf16 = (X_embed_bf16 * rms_bf16)
    Q_rot = _rotate_half(X_norm_bf16.float())

    print(f"\ntoken_embeds: {token_embeds.shape}, range [{token_embeds.min():.3f}, {token_embeds.max():.3f}]")
    print(f"pos_weight:   {pos_weight.shape},   range [{pos_weight.min():.3f}, {pos_weight.max():.3f}]")
    print(f"Q_rot:        {Q_rot.shape},         range [{Q_rot.min():.3f}, {Q_rot.max():.3f}]")
    print(f"cos:          {cos.shape},            range [{cos.min():.3f}, {cos.max():.3f}]")
    print(f"sin:          {sin.shape},            range [{sin.min():.3f}, {sin.max():.3f}]")
    print(f"K_mat:        {K_mat.shape},          range [{K_mat.min():.3f}, {K_mat.max():.3f}]")
    print(f"V_mat:        {V_mat.shape},          range [{V_mat.min():.3f}, {V_mat.max():.3f}]")
    print(f"W_gate:       {W_gate.shape},         range [{W_gate.min():.3f}, {W_gate.max():.3f}]")
    print(f"W_up:         {W_up.shape},            range [{W_up.min():.3f}, {W_up.max():.3f}]")
    print(f"W_down:       {W_down.shape},          range [{W_down.min():.3f}, {W_down.max():.3f}]")
    print(f"\nattn_scale: {scale:.6f}")

    # ----------------------------------------------------------- golden ref
    # Apply MXFP8 quantization to all HBM-stored tensors (matching hardware storage).
    # K/V from real weights can have values up to ±15 — coarse quantization at that
    # scale causes large attention errors unless the golden accounts for it.
    K_q      = quantize_to_mxfp(K_mat)
    V_q      = quantize_to_mxfp(V_mat)
    W_gate_q = quantize_to_mxfp(W_gate)
    W_up_q   = quantize_to_mxfp(W_up)
    W_down_q = quantize_to_mxfp(W_down)

    print("\n--- CPU Golden Reference (MXFP8 quantized HBM tensors + BF16 intermediates) ---")

    X_gold = token_embeds.clone()
    X_gold = X_gold + pos_weight                                # embedding_add
    # Use bfloat16 rms_norm to match PLENA's quantised intermediate
    X_gold_bf16 = X_gold.to(torch.bfloat16)
    rms_gold = torch.rsqrt(
        X_gold_bf16.float().pow(2).mean(-1, keepdim=True) + eps
    ).to(torch.bfloat16)
    X_gold = (X_gold_bf16 * rms_gold).float()                  # rms_norm (bfloat16)
    Q_rot_gold = _rotate_half(X_gold)                           # consistent Q_rot
    X_gold = X_gold * cos + Q_rot_gold * sin                    # rope
    X_gold = _flash_attn_ref(X_gold, K_q, V_q, scale)          # flash_attn (MXFP8 K/V)
    # FFN with MXFP8 weights + BF16 intermediates (matches hardware VRAM storage)
    X_gold_attn = X_gold.to(torch.bfloat16)
    up_out    = torch.matmul(X_gold_attn.float(), W_up_q.float()).to(torch.bfloat16)
    gate_out  = torch.matmul(X_gold_attn.float(), W_gate_q.float()).to(torch.bfloat16)
    silu_gate = (F.silu(up_out.float()) * gate_out.float()).to(torch.bfloat16)
    X_gold    = torch.matmul(silu_gate.float(), W_down_q.float()).to(torch.bfloat16).float()
    X_gold = _rms_norm_ref(X_gold, eps)                         # final rms_norm

    golden_out = X_gold
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0,:4].tolist()}")

    # ----------------------------------------------------------- PLENA ISA
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare inputs — order determines HBM layout
    x_input     = prog.input("X",      shape=(seq_len, hidden_size))
    pos_input   = prog.input("POS",    shape=(seq_len, hidden_size))
    qrot_input  = prog.input("QROT",   shape=(seq_len, head_dim))
    cos_input   = prog.input("COS",    shape=(seq_len, head_dim))
    sin_input   = prog.input("SIN",    shape=(seq_len, head_dim))
    k_input     = prog.input("K",      shape=(seq_len, head_dim))
    v_input     = prog.input("V",      shape=(seq_len, head_dim))
    wgate_input = prog.input("W_gate", shape=(hidden_size, inter_dim))
    wup_input   = prog.input("W_up",   shape=(hidden_size, inter_dim))
    wdown_input = prog.input("W_down", shape=(inter_dim, hidden_size))

    # Load to VRAM
    X_batch    = prog.load_batch(x_input,    name="X")
    POS_batch  = prog.load_batch(pos_input,  name="POS")
    Qrot_batch = prog.load_batch(qrot_input, name="QROT")
    Cos_batch  = prog.load_batch(cos_input,  name="COS")
    Sin_batch  = prog.load_batch(sin_input,  name="SIN")

    # Pipeline
    ops.embedding_add(prog, X_batch, POS_batch)                    # X += POS (in-place)
    prog.rms_norm(X_batch, eps_offset=3, reci_hid_offset=4)       # normalize (in-place, slots 3,4)
    ops.rope(prog, X_batch, Qrot_batch, Cos_batch, Sin_batch)     # RoPE (in-place)
    O = ops.flash_attention(prog, X_batch, k_input, v_input, scale)  # attention -> new O var
    ops.ffn(prog, O, wgate_input, wup_input, wdown_input)          # ffn (in-place on O)
    prog.rms_norm(O, eps_offset=3, reci_hid_offset=4)             # final normalize (in-place on O)

    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # ----------------------------------------------------------- sim env
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensors = {
        "X":      token_embeds,
        "POS":    pos_weight,
        "QROT":   Q_rot,
        "COS":    cos,
        "SIN":    sin,
        "K":      K_mat,
        "V":      V_mat,
        "W_gate": W_gate,
        "W_up":   W_up,
        "W_down": W_down,
    }
    golden_result = {"original_output": golden_out}

    # FPRAM layout:
    #   slot 0 = 0.0        (reserved)
    #   slot 1 = attn_scale (flash_attention)
    #   slot 2 = -inf       (flash_attention softmax mask)
    #   slot 3 = eps        (rms_norm, offset=3)
    #   slot 4 = 1/hidden   (rms_norm, offset=4)
    #   slot 5 = 1.0        (FFN SiLU)
    #   slots 6-9 = 0.0     (padding)
    fp_preload = [0.0, scale, float("-inf"), eps, 1.0 / hidden_size, 1.0] + [0.0] * 4

    create_sim_env(
        input_tensors, gen_code, golden_result, fp_preload,
        build_dir=str(build_dir),
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=asm_name,
        data=None,
        specified_data_order=["X", "POS", "QROT", "COS", "SIN", "K", "V", "W_gate", "W_up", "W_down"],
        build_path=build_dir,
    )

    # Result is at O's VRAM location (flash_attention allocates O separately)
    o_vram_addr = prog._compiler.get_vram_addr(O.name)

    comparison_params = {
        "start_row_idx":      o_vram_addr // mlen,
        "num_rows":           (seq_len * hidden_size) // mlen,
        "num_batches":        seq_len,
        "elements_per_batch": hidden_size,
        "row_dim":            mlen,
        "use_stride_mode":    hidden_size > mlen,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"  Result location: VRAM row {o_vram_addr // mlen} (O from flash_attention)")

    run_and_assert(build_dir, asm_name, mlen=mlen, blen=blen)
