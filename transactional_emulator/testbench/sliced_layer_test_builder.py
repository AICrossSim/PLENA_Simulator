"""
Shared infrastructure for simulator-sliced layer testbench tests.

Provides:
  - ModelDims     dataclass (model configuration)
  - get_model_dims()        probe HF config (no weight download)
  - slice_dims_for_sim()    clip to simulator HBM capacity
  - load_ffn_weights()      load + transpose MLP weights
  - quantize_to_mxfp()      MXFP8 round-trip (HBM format)
  - golden_ffn()            hardware-accurate golden (MXFP8 + BF16 intermediates)
  - build_and_run_sliced_ffn_test() end-to-end: load → golden → ISA → sim → assert
"""

import sys
import os
from dataclasses import dataclass
from pathlib import Path


import tomlkit
import torch
import torch.nn.functional as F

from plena_quant.mxfp import _mx_fp_quantize_hardware
from compiler.aten.ops.registry import OpRegistry, Backend
import compiler.aten.ops as ops

from compiler.aten.plena import PlenaCompiler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def _skip_if_hf_unavailable(model_id: str, exc: BaseException) -> None:
    """If a HuggingFace download/load failed, print a friendly skip and exit 0.

    Catches the OSError / HTTP / connection errors that transformers raises
    when the model isn't cached locally and the network is unreachable
    (or HF Hub is down).  Tests should call this from their except branch
    so a missing model becomes a SKIP rather than a hard FAIL.
    """
    print()
    print("=" * 80)
    print(f"[SKIP] HuggingFace model '{model_id}' is not available")
    print(f"       Reason: {type(exc).__name__}: {exc}")
    print()
    print("       To download it, run with network access:")
    print(f"           python -c \"from transformers import AutoModel; AutoModel.from_pretrained('{model_id}')\"")
    print("       To stay offline and force-skip, set: export TRANSFORMERS_OFFLINE=1")
    print("=" * 80)
    sys.exit(0)


HIDDEN_SLICE = 128
INTER_SLICE = 256
MLEN = 64
BLEN = 4
REAL_DATA_RATIO = (8 * 8 + 8) / (8 * 8)


# ---------------------------------------------------------------------------
# ModelDims
# ---------------------------------------------------------------------------
@dataclass
class ModelDims:
    """Configuration dimensions extracted from a HuggingFace model."""

    hidden_size: int
    inter_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    model_id: str


def get_model_dims(model_id: str) -> ModelDims:
    """Probe model config (no weight download)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    # Resolve text config for VLMs that wrap a language model
    text_cfg = getattr(cfg, "text_config", cfg)
    hidden = text_cfg.hidden_size
    inter = text_cfg.intermediate_size
    n_heads = text_cfg.num_attention_heads
    n_kv = getattr(text_cfg, "num_key_value_heads", n_heads)
    head_dim = hidden // n_heads
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
    inter_slice: int = INTER_SLICE,
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ModelDims]:
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
    except (AttributeError, KeyError, ValueError) as e:
        print(f"  [INFO] CausalLM path failed ({type(e).__name__}: {e}), falling back to AutoModel")
        # Fall back to AutoModel (SmolVLM2, etc.)
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
        layer = model.text_model.layers[layer_idx]

    # HF stores (out_features, in_features); PLENA expects (in_features, out_features)
    W_gate = layer.mlp.gate_proj.weight.detach().T.contiguous()  # (hidden, inter)
    W_up = layer.mlp.up_proj.weight.detach().T.contiguous()  # (hidden, inter)
    W_down = layer.mlp.down_proj.weight.detach().T.contiguous()  # (inter, hidden)

    dims = get_model_dims(model_id)
    return W_gate, W_up, W_down, dims


# ---------------------------------------------------------------------------
# Active precision helpers
# ---------------------------------------------------------------------------


def _active_precision_settings():
    config_path = Path(os.environ.get("PLENA_SETTINGS_TOML", Path(__file__).parents[2] / "plena_settings.toml"))
    with open(config_path) as f:
        return tomlkit.load(f)["TRANSACTIONAL"]["PRECISION"]


def _quantize_plain_fp_no_specials(tensor: torch.Tensor, *, exponent: int, mantissa: int, sign: bool) -> torch.Tensor:
    x = tensor.float()
    out = torch.zeros_like(x)
    finite = torch.isfinite(x) & (x != 0)
    if not sign:
        finite &= x > 0
    if not torch.any(finite):
        return out

    values = x[finite].abs()
    exp_bias = (1 << exponent) // 2 - 1
    exp_min = -exp_bias
    exp_max = (1 << exponent) - 2 - exp_bias
    raw_exp = torch.floor(torch.log2(values + 1e-9))
    overflow = raw_exp > exp_max
    clamped_exp = torch.clamp(raw_exp, exp_min, exp_max)

    shift = 1 << mantissa
    scaled = values / torch.pow(torch.tensor(2.0, device=x.device), clamped_exp)
    subnormal = clamped_exp == exp_min
    shifted = torch.where(subnormal, scaled * shift, (scaled - 1.0) * shift)
    shifted = torch.round(shifted).clamp(0, shift - 1)
    shifted = torch.where(overflow, torch.full_like(shifted, shift - 1), shifted)

    exp_bits = clamped_exp + exp_bias
    decoded_exp = exp_bits - exp_bias
    decoded_base = torch.where(exp_bits == 0, shifted / shift, 1.0 + shifted / shift)
    decoded = decoded_base * torch.pow(torch.tensor(2.0, device=x.device), decoded_exp)
    decoded = torch.where(x[finite] < 0, -decoded, decoded)
    out[finite] = decoded
    return out


def quantize_to_vector_fp(tensor: torch.Tensor, precision=None) -> torch.Tensor:
    """Quantize through the active VECTOR_SRAM_TYPE plain FP format."""
    precision = precision or _active_precision_settings()
    data_type = precision["VECTOR_SRAM_TYPE"]["DATA_TYPE"]
    exponent = int(data_type["exponent"])
    mantissa = int(data_type["mantissa"])
    sign = bool(data_type.get("sign", True))
    if sign and exponent == 8 and mantissa == 7:
        return tensor.float().to(torch.bfloat16).float()
    if sign and exponent == 5 and mantissa == 10:
        return tensor.float().to(torch.float16).float()
    if sign and exponent == 8 and mantissa == 23:
        return tensor.float()
    return _quantize_plain_fp_no_specials(tensor, exponent=exponent, mantissa=mantissa, sign=sign)


def quantize_to_mxfp(tensor: torch.Tensor, precision_node=None) -> torch.Tensor:
    """Quantize tensor to the configured HBM MXFP format; return dequantized result."""
    if precision_node is None:
        width = 8
        exponent_width = 4
        exponent_bias_width = 8
        block_size = [1, 8]
    else:
        width = int(precision_node["ELEM"]["exponent"]) + int(precision_node["ELEM"]["mantissa"]) + 1
        exponent_width = int(precision_node["ELEM"]["exponent"])
        exponent_bias_width = int(precision_node["SCALE"]["exponent"])
        block_size = [1, int(precision_node["block"])]

    orig_shape = tensor.shape
    tensor_2d = tensor.float().reshape(-1, tensor.shape[-1])
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor_2d,
        width=width,
        exponent_width=exponent_width,
        exponent_bias_width=exponent_bias_width,
        block_size=block_size,
    )
    return bm_x.reshape(orig_shape)


def _load_to_vector_fp(tensor: torch.Tensor, hbm_precision, vector_precision) -> torch.Tensor:
    return quantize_to_vector_fp(quantize_to_mxfp(tensor, hbm_precision), vector_precision)


def _rms_norm_vector_ref(x: torch.Tensor, eps: float, precision) -> torch.Tensor:
    x_q = quantize_to_vector_fp(x, precision)
    rms = quantize_to_vector_fp(torch.rsqrt(x_q.float().pow(2).mean(-1, keepdim=True) + eps), precision)
    return quantize_to_vector_fp(x_q * rms, precision)


# ---------------------------------------------------------------------------
# Hardware-accurate golden reference
# ---------------------------------------------------------------------------
def golden_ffn(
    X: torch.Tensor,
    W_gate: torch.Tensor,
    W_up: torch.Tensor,
    W_down: torch.Tensor,
) -> torch.Tensor:
    """
    Hardware-accurate FFN golden reference.

    Applies MXFP8 quantization (HBM storage) + BF16 intermediates (VRAM).
    Formula: w_down @ (silu(w_up @ x) * (w_gate @ x))
    Hardware applies SiLU to the UP projection, not gate.
    """
    X_q = quantize_to_mxfp(X)
    W_gate_q = quantize_to_mxfp(W_gate)
    W_up_q = quantize_to_mxfp(W_up)
    W_down_q = quantize_to_mxfp(W_down)

    up_out = torch.matmul(X_q.float(), W_up_q.float()).to(torch.bfloat16)
    gate_out = torch.matmul(X_q.float(), W_gate_q.float()).to(torch.bfloat16)
    silu_gate = (F.silu(up_out.float()) * gate_out.float()).to(torch.bfloat16)
    return torch.matmul(silu_gate.float(), W_down_q.float()).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# End-to-end test runner
# ---------------------------------------------------------------------------
def compile_sliced_ffn(
    model_id: str,
    build_dir: Path,
    layer_idx: int = 0,
    batch_size: int = 4,
    mlen: int = MLEN,
    vlen: int | None = None,
    blen: int = BLEN,
    seed: int = 42,
) -> dict:
    """Compile sliced FFN test, return result dict (no emulation)."""
    if vlen is None:
        vlen = mlen
    print("=" * 80)
    print(f"FFN Test — {model_id}  (layer {layer_idx})")
    print("=" * 80)

    dims = get_model_dims(model_id)
    sim = slice_dims_for_sim(dims)

    print(
        f"\nFull model dims: hidden={dims.hidden_size}, inter={dims.inter_dim}, "
        f"heads={dims.num_heads}/{dims.num_kv_heads}, head_dim={dims.head_dim}"
    )
    print(f"Sim dims:        hidden={sim.hidden_size}, inter={sim.inter_dim} (sliced)")

    print(f"\nLoading weights from {model_id} layer {layer_idx}...")
    try:
        W_gate_full, W_up_full, W_down_full, _ = load_ffn_weights(model_id, layer_idx)
    except (OSError, ConnectionError) as exc:
        _skip_if_hf_unavailable(model_id, exc)
        return {}

    W_gate = W_gate_full[: sim.hidden_size, : sim.inter_dim].contiguous()
    W_up = W_up_full[: sim.hidden_size, : sim.inter_dim].contiguous()
    W_down = W_down_full[: sim.inter_dim, : sim.hidden_size].contiguous()

    torch.manual_seed(seed)
    X = torch.randn(batch_size, sim.hidden_size)

    golden_out = golden_ffn(X, W_gate, W_up, W_down)

    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=REAL_DATA_RATIO)

    x_input = prog.input("X", shape=(batch_size, sim.hidden_size))
    w_gate_input = prog.input("W_gate", shape=(sim.hidden_size, sim.inter_dim))
    w_up_input = prog.input("W_up", shape=(sim.hidden_size, sim.inter_dim))
    w_down_input = prog.input("W_down", shape=(sim.inter_dim, sim.hidden_size))

    X_batch = prog.load_batch(x_input, name="X")
    ops.ffn(prog, X_batch, w_gate_input, w_up_input, w_down_input)

    gen_code = prog.compile()
    print(f"\nGenerated {len(gen_code.splitlines())} lines of ISA code")

    x_vram_addr = prog._compiler.get_vram_addr(X_batch.name)

    return {
        "isa": gen_code,
        "input_tensors": {"X": X, "W_gate": W_gate, "W_up": W_up, "W_down": W_down},
        "golden_result": {"original_output": golden_out},
        "fp_preload": [0.0, 1.0, 0.0, 0.0, 0.0, 1.0] + [0.0] * 4,
        "data_order": ["X", "W_gate", "W_up", "W_down"],
        "comparison_params": {
            "start_row_idx": x_vram_addr // mlen,
            "num_rows": (batch_size * sim.hidden_size) // mlen,
            "num_batches": batch_size,
            "elements_per_batch": sim.hidden_size,
            "row_dim": mlen,
            "use_stride_mode": sim.hidden_size > mlen,
        },
        "tensor_layouts": None,
        "hbm_addrs": None,
    }


def build_and_run_sliced_ffn_test(
    model_id: str,
    asm_name: str,
    build_dir: Path,
    layer_idx: int = 0,
    batch_size: int = 4,
    mlen: int = MLEN,
    vlen: int | None = None,
    blen: int = BLEN,
    seed: int = 42,
) -> None:
    """End-to-end FFN test: compile + emulate + assert."""
    from transactional_emulator.testbench.emulator_runner import emulate_from_result

    result = compile_sliced_ffn(
        model_id,
        build_dir,
        layer_idx=layer_idx,
        batch_size=batch_size,
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        seed=seed,
    )
    emulate_from_result(result, build_dir, asm_name, mlen=mlen, blen=blen, vlen=vlen)


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
    n_heads: int,
    n_kv: int,
    eps: float,
    rope_theta: float,
) -> dict[str, object]:
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

    # Download shard index (or use single safetensors file for small models)
    try:
        index_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors.index.json",
        )
        with open(index_path) as f:
            index = _json.load(f)
        weight_map = index["weight_map"]  # key -> shard filename
    except Exception:
        # Single safetensors file (small models, VLMs, etc.)
        safetensors_path = hf_hub_download(
            repo_id=model_id,
            filename="model.safetensors",
        )
        # Build weight map from the single file: every key maps to model.safetensors
        import safetensors.torch

        with safetensors.safe_open(safetensors_path, framework="pt") as f:
            all_keys = list(f.keys())
        weight_map = {k: "model.safetensors" for k in all_keys}

    # Auto-detect naming convention by probing the index
    llama_gate = f"model.layers.{layer_idx}.mlp.gate_proj.weight"
    llada_gate = f"model.transformer.blocks.{layer_idx}.ff_proj.weight"
    vlm_gate = f"model.text_model.layers.{layer_idx}.mlp.gate_proj.weight"

    if llama_gate in weight_map:
        # Standard LLaMA / Mistral / Qwen naming
        p = f"model.layers.{layer_idx}."
        key_map = {
            "gate": f"{p}mlp.gate_proj.weight",
            "up": f"{p}mlp.up_proj.weight",
            "down": f"{p}mlp.down_proj.weight",
            "q": f"{p}self_attn.q_proj.weight",
            "k": f"{p}self_attn.k_proj.weight",
            "v": f"{p}self_attn.v_proj.weight",
            "o": f"{p}self_attn.o_proj.weight",
        }
    elif llada_gate in weight_map:
        # LLaDA-8B custom naming: ff_proj=gate, up_proj=up, ff_out=down, attn_out=o
        p = f"model.transformer.blocks.{layer_idx}."
        key_map = {
            "gate": f"{p}ff_proj.weight",
            "up": f"{p}up_proj.weight",
            "down": f"{p}ff_out.weight",
            "q": f"{p}q_proj.weight",
            "k": f"{p}k_proj.weight",
            "v": f"{p}v_proj.weight",
            "o": f"{p}attn_out.weight",
        }
    elif vlm_gate in weight_map:
        # VLM naming: model.text_model.layers.{i}.mlp.{gate,up,down}_proj.weight
        p = f"model.text_model.layers.{layer_idx}."
        key_map = {
            "gate": f"{p}mlp.gate_proj.weight",
            "up": f"{p}mlp.up_proj.weight",
            "down": f"{p}mlp.down_proj.weight",
            "q": f"{p}self_attn.q_proj.weight",
            "k": f"{p}self_attn.k_proj.weight",
            "v": f"{p}self_attn.v_proj.weight",
            "o": f"{p}self_attn.o_proj.weight",
        }
    else:
        # Show available keys to help debug
        sample = [k for k in weight_map if f".{layer_idx}." in k][:10]
        raise RuntimeError(
            f"Unknown weight naming convention for {model_id}. Sample keys with layer {layer_idx}: {sample}"
        )

    if vlm_gate in weight_map:
        naming = "VLM"
    elif llada_gate in weight_map:
        naming = "LLaDA"
    else:
        naming = "LLaMA"
    print(f"  [partial_load] Detected naming: {naming}")
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
    sim_head_dim = min(head_dim, hidden_slice)
    gqa_ratio = n_heads // n_kv
    n_packed_heads = min(gqa_ratio, hidden_slice // sim_head_dim) if sim_head_dim < hidden_slice else 1
    packed_q_dim = n_packed_heads * sim_head_dim

    W_gate = loaded["gate"].T.contiguous()[:hidden_slice, :inter_slice].contiguous()
    W_up = loaded["up"].T.contiguous()[:hidden_slice, :inter_slice].contiguous()
    W_down = loaded["down"].T.contiguous()[:inter_slice, :hidden_slice].contiguous()
    W_q = loaded["q"].T.contiguous()[:hidden_slice, :packed_q_dim].contiguous()
    W_k = loaded["k"].T.contiguous()[:hidden_slice, :sim_head_dim].contiguous()
    W_v = loaded["v"].T.contiguous()[:hidden_slice, :sim_head_dim].contiguous()
    W_o = loaded["o"].T.contiguous()[:packed_q_dim, :hidden_slice].contiguous()

    return dict(
        W_q=W_q,
        W_gate=W_gate,
        W_up=W_up,
        W_down=W_down,
        W_k=W_k,
        W_v=W_v,
        W_o=W_o,
        head_dim=sim_head_dim,
        n_packed_heads=n_packed_heads,
        eps=eps,
        rope_theta=rope_theta,
    )


def load_decoder_weights(
    model_id: str,
    layer_idx: int = 0,
    hidden_slice: int = 64,
    inter_slice: int = 256,
    trust_remote_code: bool = False,
    partial_load: bool = False,
) -> dict[str, object]:
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
            model_id,
            layer_idx,
            hidden_slice,
            inter_slice,
            trust_remote_code,
            head_dim,
            n_heads,
            n_kv,
            eps,
            rope_theta,
        )

    # Full model load (suitable for small models, e.g. <1B params)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, trust_remote_code=trust_remote_code
        )
        layer = model.model.layers[layer_idx]
    except (AttributeError, KeyError, ValueError) as e:
        print(f"  [INFO] CausalLM path failed ({type(e).__name__}: {e}), falling back to AutoModel")
        model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32, trust_remote_code=trust_remote_code)
        # VLMs nest the text model under .text_model (SmolVLMModel, etc.)
        # while bare CausalLM models use .model.layers
        text_model = getattr(model, "text_model", None)
        if text_model is None:
            text_model = getattr(getattr(model, "model", None), "text_model", None)
        if text_model is None:
            raise ValueError(f"Cannot find text_model on {type(model).__name__}")
        layer = text_model.layers[layer_idx]

    norm = layer.input_layernorm
    eps = getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5))

    # FFN weights: HF stores (out, in) -> transpose to (in, out) -> slice
    W_gate = layer.mlp.gate_proj.weight.detach().T.contiguous()[:hidden_slice, :inter_slice].contiguous()
    W_up = layer.mlp.up_proj.weight.detach().T.contiguous()[:hidden_slice, :inter_slice].contiguous()
    W_down = layer.mlp.down_proj.weight.detach().T.contiguous()[:inter_slice, :hidden_slice].contiguous()

    # Attention: pack GQA Q heads, single KV head
    sim_head_dim = min(head_dim, hidden_slice)
    gqa_ratio = n_heads // n_kv
    n_packed_heads = min(gqa_ratio, hidden_slice // sim_head_dim) if sim_head_dim < hidden_slice else 1
    packed_q_dim = n_packed_heads * sim_head_dim

    W_q_full = layer.self_attn.q_proj.weight.detach().T.contiguous()
    W_k_full = layer.self_attn.k_proj.weight.detach().T.contiguous()
    W_v_full = layer.self_attn.v_proj.weight.detach().T.contiguous()
    W_o_full = layer.self_attn.o_proj.weight.detach().T.contiguous()
    W_q = W_q_full[:hidden_slice, :packed_q_dim].contiguous()
    W_k = W_k_full[:hidden_slice, :sim_head_dim].contiguous()
    W_v = W_v_full[:hidden_slice, :sim_head_dim].contiguous()
    W_o = W_o_full[:packed_q_dim, :hidden_slice].contiguous()
    head_dim = sim_head_dim

    return dict(
        W_q=W_q,
        W_gate=W_gate,
        W_up=W_up,
        W_down=W_down,
        W_k=W_k,
        W_v=W_v,
        W_o=W_o,
        head_dim=head_dim,
        n_packed_heads=n_packed_heads,
        eps=eps,
        rope_theta=rope_theta,
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
    angles = torch.outer(positions, freqs)  # (seq_len, half)
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    cos = torch.cat([cos_half, cos_half], dim=-1)  # (seq_len, head_dim)
    sin = torch.cat([sin_half, sin_half], dim=-1)
    return cos, sin


def _ffn_ref(x: torch.Tensor, W_gate: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
    """CPU reference: SwiGLU FFN (SiLU applied to W_up projection)."""
    gate = x @ W_gate
    up = F.silu(x @ W_up)
    return (up * gate) @ W_down


def _flash_attn_ref(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float, precision=None) -> torch.Tensor:
    """CPU reference: scaled dot-product attention matching PLENA hardware precision.

    The emulator assembles softmax from individual ISA ops (max, sub, exp, sum, div),
    each quantized to the vector SRAM type (BF16).  We replicate that step-by-step
    quantization so the golden tracks the same rounding path.
    """

    def qvfp(t):
        return quantize_to_vector_fp(t, precision)

    scores = qvfp((Q @ K.T) * scale)
    row_max = qvfp(scores.max(dim=-1, keepdim=True).values)
    shifted = qvfp(scores - row_max)
    exp_shifted = qvfp(shifted.float().exp())
    row_sum = qvfp(exp_shifted.sum(dim=-1, keepdim=True))
    attn = qvfp(exp_shifted / row_sum)
    return attn @ V


# ---------------------------------------------------------------------------
# End-to-end decoder pipeline test
# ---------------------------------------------------------------------------
def compile_sliced_decoder_layer(
    model_id: str,
    build_dir: Path,
    layer_idx: int = 0,
    seq_len: int = 64,
    hidden_size: int = 64,
    inter_dim: int = 128,
    batch_size: int = 1,
    mlen: int = MLEN,
    vlen: int | None = None,
    blen: int = BLEN,
    seed: int = 42,
    trust_remote_code: bool = False,
    partial_load: bool = False,
) -> dict:
    """Compile single-layer sliced decoder test, return result dict (no emulation)."""
    import math

    if vlen is None:
        vlen = mlen
    total_seq = batch_size * seq_len
    print("=" * 80)
    print(f"Decoder Pipeline Test — {model_id}  (layer {layer_idx}, batch_size={batch_size})")
    print("  embedding_add -> rms_norm -> rope -> flash_attention -> ffn -> rms_norm")
    print("=" * 80)

    head_dim = hidden_size  # must equal mlen for flash_attention
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    scale = 1.0 / math.sqrt(head_dim)

    # ---------------------------------------------------------------- weights
    print(f"\nLoading weights from {model_id} layer {layer_idx}...")
    try:
        w = load_decoder_weights(
            model_id,
            layer_idx,
            hidden_slice=hidden_size,
            inter_slice=inter_dim,
            trust_remote_code=trust_remote_code,
            partial_load=partial_load,
        )
    except (OSError, ConnectionError) as exc:
        _skip_if_hf_unavailable(model_id, exc)
        return {}
    W_gate = w["W_gate"]
    W_up = w["W_up"]
    W_down = w["W_down"]
    W_k = w["W_k"]
    W_v = w["W_v"]
    eps = w["eps"]
    rope_theta = w["rope_theta"]

    print(f"  head_dim={w['head_dim']}, eps={eps}, rope_theta={rope_theta}")
    print(f"  W_gate: {W_gate.shape}, W_up: {W_up.shape}, W_down: {W_down.shape}")
    print(f"  W_k: {W_k.shape}, W_v: {W_v.shape}")

    # ----------------------------------------------------------- test data
    torch.manual_seed(seed)
    token_embeds = torch.randn(total_seq, hidden_size)
    pos_weight = torch.randn(total_seq, hidden_size)
    X_ctx = torch.randn(total_seq, hidden_size)

    # Precompute K, V from context
    K_mat = X_ctx @ W_k  # (total_seq, head_dim)
    V_mat = X_ctx @ W_v  # (total_seq, head_dim)

    # RoPE tables are per-position within a sequence, tiled across batches
    cos_1, sin_1 = _make_rope_tables(seq_len, head_dim, theta=rope_theta)
    cos = cos_1.repeat(batch_size, 1)  # (total_seq, head_dim)
    sin = sin_1.repeat(batch_size, 1)

    precision = _active_precision_settings()
    hbm_act_precision = precision["HBM_V_ACT_TYPE"]
    hbm_weight_precision = precision["HBM_M_WEIGHT_TYPE"]
    hbm_kv_precision = precision["HBM_M_KV_TYPE"]

    # Precompute Q_rot from the same vector precision path used by the simulator.
    X_embed_q = quantize_to_vector_fp(
        _load_to_vector_fp(token_embeds, hbm_act_precision, precision)
        + _load_to_vector_fp(pos_weight, hbm_act_precision, precision),
        precision,
    )
    X_norm_q = _rms_norm_vector_ref(X_embed_q, eps, precision)
    Q_rot = _rotate_half(X_norm_q.float())

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

    def _ceil_to_multiple(value: int, multiple: int) -> int:
        return ((value + multiple - 1) // multiple) * multiple

    def _layout(logical_shape: tuple[int, int], physical_shape: tuple[int, int]) -> dict[str, object]:
        return {
            "source_shape": list(logical_shape),
            "storage_shape": list(physical_shape),
            "logical_shape": list(logical_shape),
            "physical_shape": list(physical_shape),
            "source_row_elements": logical_shape[1],
            "storage_row_elements": physical_shape[1],
        }

    physical_seq_per_batch = max(mlen, _ceil_to_multiple(seq_len, blen))
    total_physical_seq = batch_size * physical_seq_per_batch
    act_physical_shape = (total_physical_seq, _ceil_to_multiple(hidden_size, mlen))
    head_physical_shape = (total_physical_seq, _ceil_to_multiple(head_dim, mlen))
    w_gate_physical_shape = (_ceil_to_multiple(hidden_size, mlen), _ceil_to_multiple(inter_dim, mlen))
    w_down_physical_shape = (_ceil_to_multiple(inter_dim, mlen), _ceil_to_multiple(hidden_size, mlen))

    # ----------------------------------------------------------- golden ref
    K_q = quantize_to_mxfp(K_mat, hbm_weight_precision)
    V_q = quantize_to_mxfp(V_mat, hbm_kv_precision)
    W_gate_q = quantize_to_mxfp(W_gate, hbm_weight_precision)
    W_up_q = quantize_to_mxfp(W_up, hbm_weight_precision)
    W_down_q = quantize_to_mxfp(W_down, hbm_weight_precision)

    print("\n--- CPU Golden Reference (active TOML HBM + vector precision) ---")

    X_gold = X_norm_q
    Q_rot_gold = _load_to_vector_fp(Q_rot, hbm_act_precision, precision)
    cos_q = _load_to_vector_fp(cos, hbm_act_precision, precision)
    sin_q = _load_to_vector_fp(sin, hbm_act_precision, precision)

    def qvfp(t):
        return quantize_to_vector_fp(t, precision)

    # RoPE: emulator quantizes each vmul result before vadd
    X_gold = qvfp(qvfp(X_gold * cos_q) + qvfp(Q_rot_gold * sin_q))  # rope
    # Attention is independent per batch — compute per-batch to avoid cross-batch mixing
    attn_outs = []
    for b in range(batch_size):
        s, e = b * seq_len, (b + 1) * seq_len
        attn_outs.append(_flash_attn_ref(X_gold[s:e], K_q[s:e], V_q[s:e], scale, precision=precision))
    X_gold = qvfp(torch.cat(attn_outs, dim=0))
    up_out = qvfp(torch.matmul(X_gold.float(), W_up_q.float()))
    gate_out = qvfp(torch.matmul(X_gold.float(), W_gate_q.float()))
    silu_gate = qvfp(qvfp(F.silu(up_out.float())) * gate_out.float())
    X_gold = quantize_to_vector_fp(torch.matmul(silu_gate.float(), W_down_q.float()), precision)
    X_gold = _rms_norm_vector_ref(X_gold, eps, precision)

    golden_out = X_gold
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ----------------------------------------------------------- PLENA ISA
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare inputs — order determines HBM layout
    x_input = prog.input("X", shape=(total_seq, hidden_size), physical_shape=act_physical_shape)
    pos_input = prog.input("POS", shape=(total_seq, hidden_size), physical_shape=act_physical_shape)
    qrot_input = prog.input("QROT", shape=(total_seq, head_dim), physical_shape=head_physical_shape)
    cos_input = prog.input("COS", shape=(total_seq, head_dim), physical_shape=head_physical_shape)
    sin_input = prog.input("SIN", shape=(total_seq, head_dim), physical_shape=head_physical_shape)
    k_input = prog.input("K", shape=(total_seq, head_dim), physical_shape=head_physical_shape)
    v_input = prog.input("V", shape=(total_seq, head_dim), physical_shape=head_physical_shape)
    wgate_input = prog.input("W_gate", shape=(hidden_size, inter_dim), physical_shape=w_gate_physical_shape)
    wup_input = prog.input("W_up", shape=(hidden_size, inter_dim), physical_shape=w_gate_physical_shape)
    wdown_input = prog.input("W_down", shape=(inter_dim, hidden_size), physical_shape=w_down_physical_shape)

    # Load to VRAM
    X_batch = prog.load_batch(x_input, name="X")
    POS_batch = prog.load_batch(pos_input, name="POS")
    Qrot_batch = prog.load_batch(qrot_input, name="QROT")
    Cos_batch = prog.load_batch(cos_input, name="COS")
    Sin_batch = prog.load_batch(sin_input, name="SIN")

    # Pipeline
    ops.embedding_add(prog, X_batch, POS_batch)  # X += POS (in-place)
    prog.rms_norm(X_batch, eps_offset=3, reci_hid_offset=4)  # normalize (in-place, slots 3,4)
    ops.rope(prog, X_batch, Qrot_batch, Cos_batch, Sin_batch)  # RoPE (in-place)
    O = ops.flash_attention(prog, X_batch, k_input, v_input, scale, batch_size=batch_size)  # attention -> new O var
    ops.ffn(prog, O, wgate_input, wup_input, wdown_input)  # ffn (in-place on O)
    prog.rms_norm(O, eps_offset=3, reci_hid_offset=4)  # final normalize (in-place on O)

    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    o_vram_addr = prog._compiler.get_vram_addr(O.name)

    data_order = ["X", "POS", "QROT", "COS", "SIN", "K", "V", "W_gate", "W_up", "W_down"]
    input_tensors = {
        "X": token_embeds,
        "POS": pos_weight,
        "QROT": Q_rot,
        "COS": cos,
        "SIN": sin,
        "K": K_mat,
        "V": V_mat,
        "W_gate": W_gate,
        "W_up": W_up,
        "W_down": W_down,
    }
    tensor_layouts = {
        "X": _layout((total_seq, hidden_size), act_physical_shape),
        "POS": _layout((total_seq, hidden_size), act_physical_shape),
        "QROT": _layout((total_seq, head_dim), head_physical_shape),
        "COS": _layout((total_seq, head_dim), head_physical_shape),
        "SIN": _layout((total_seq, head_dim), head_physical_shape),
        "K": _layout((total_seq, head_dim), head_physical_shape),
        "V": _layout((total_seq, head_dim), head_physical_shape),
        "W_gate": _layout((hidden_size, inter_dim), w_gate_physical_shape),
        "W_up": _layout((hidden_size, inter_dim), w_gate_physical_shape),
        "W_down": _layout((inter_dim, hidden_size), w_down_physical_shape),
    }

    return {
        "isa": gen_code,
        "input_tensors": input_tensors,
        "golden_result": {"original_output": golden_out},
        "fp_preload": [0.0, scale, float("-inf"), eps, 1.0 / hidden_size, 1.0] + [0.0] * 4,
        "data_order": data_order,
        "comparison_params": {
            "start_row_idx": o_vram_addr // mlen,
            "num_rows": (act_physical_shape[0] * act_physical_shape[1]) // mlen,
            "num_batches": total_seq,
            "elements_per_batch": hidden_size,
            "row_dim": mlen,
            "physical_rows": act_physical_shape[0],
            "use_stride_mode": hidden_size > mlen,
            "use_slice_mode": act_physical_shape[1] != hidden_size,
            "slice_per_row": hidden_size,
            "atol": 0.2,
            "rtol": 0.2,
            "min_allclose_match_rate": 90.0,
        },
        "tensor_layouts": tensor_layouts,
        "hbm_addrs": None,
    }


def build_and_run_sliced_decoder_layer_test(
    model_id: str,
    asm_name: str,
    build_dir: Path,
    layer_idx: int = 0,
    seq_len: int = 64,
    hidden_size: int = 64,
    inter_dim: int = 128,
    batch_size: int = 1,
    mlen: int = MLEN,
    vlen: int | None = None,
    blen: int = BLEN,
    seed: int = 42,
    trust_remote_code: bool = False,
    partial_load: bool = False,
) -> None:
    """End-to-end single-layer decoder test: compile + emulate + assert."""
    from transactional_emulator.testbench.emulator_runner import emulate_from_result

    result = compile_sliced_decoder_layer(
        model_id,
        build_dir,
        layer_idx=layer_idx,
        seq_len=seq_len,
        hidden_size=hidden_size,
        inter_dim=inter_dim,
        batch_size=batch_size,
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        seed=seed,
        trust_remote_code=trust_remote_code,
        partial_load=partial_load,
    )
    emulate_from_result(result, build_dir, asm_name, mlen=mlen, blen=blen, vlen=vlen)


# ---------------------------------------------------------------------------
# End-to-end multi-layer decoder pipeline test
# ---------------------------------------------------------------------------
def compile_sliced_decoder_chain(
    model_id: str,
    build_dir: Path,
    num_layers: int = 2,
    layer_idx_start: int = 0,
    seq_len: int = 64,
    hidden_size: int = 64,
    inter_dim: int = 128,
    batch_size: int = 1,
    mlen: int = MLEN,
    vlen: int | None = None,
    blen: int = BLEN,
    seed: int = 42,
    trust_remote_code: bool = False,
    partial_load: bool = False,
) -> dict:
    """Compile multi-layer sliced decoder chain test, return result dict (no emulation)."""
    import math

    if vlen is None:
        vlen = mlen
    total_seq = batch_size * seq_len
    print("=" * 80)
    print(f"Multi-Layer Decoder Test — {model_id}  ({num_layers} layers, batch_size={batch_size}, no RoPE)")
    print(
        "  embedding_add -> [rms_norm -> (Q/O proj) -> flash_attn -> residual -> rms_norm -> ffn -> residual] x N -> rms_norm"
    )
    print("=" * 80)

    # head_dim is auto-detected from model weights; if == hidden_size, Q/O projections are skipped
    real_data_ratio = REAL_DATA_RATIO

    # ---------------------------------------------------------------- weights
    print(f"\nLoading weights from {model_id} layers {layer_idx_start}..{layer_idx_start + num_layers - 1}...")
    all_weights = []
    for i in range(num_layers):
        try:
            w = load_decoder_weights(
                model_id,
                layer_idx_start + i,
                hidden_slice=hidden_size,
                inter_slice=inter_dim,
                trust_remote_code=trust_remote_code,
                partial_load=partial_load,
            )
        except (OSError, ConnectionError) as exc:
            _skip_if_hf_unavailable(model_id, exc)
            return
        all_weights.append(w)
        print(f"  Layer {i}: W_gate={w['W_gate'].shape}, W_k={w['W_k'].shape}, W_q={w['W_q'].shape}, eps={w['eps']}")

    eps = all_weights[0]["eps"]
    head_dim = all_weights[0]["head_dim"]
    n_packed_heads = all_weights[0].get("n_packed_heads", 1)
    packed_q_dim = n_packed_heads * head_dim
    use_qo_proj = head_dim != hidden_size
    scale = 1.0 / math.sqrt(head_dim)
    print(
        f"  head_dim={head_dim}, hidden_size={hidden_size}, use_qo_proj={use_qo_proj}, n_packed_heads={n_packed_heads}"
    )

    # ----------------------------------------------------------- test data
    torch.manual_seed(seed)
    token_embeds = torch.randn(total_seq, hidden_size)
    pos_weight = torch.randn(total_seq, hidden_size)

    K_mats = []
    V_mats = []
    for i in range(num_layers):
        X_ctx = torch.randn(total_seq, hidden_size)
        K_mats.append(X_ctx @ all_weights[i]["W_k"])
        V_mats.append(X_ctx @ all_weights[i]["W_v"])

    print(f"\ntoken_embeds: {token_embeds.shape} (batch_size={batch_size}, seq_len={seq_len})")
    print(f"pos_weight:   {pos_weight.shape}")
    for i in range(num_layers):
        print(f"  K_{i}: {K_mats[i].shape}, V_{i}: {V_mats[i].shape}")
    print(f"attn_scale: {scale:.6f}")

    # ----------------------------------------------------------- golden ref
    K_q_list = [quantize_to_mxfp(K_mats[i]) for i in range(num_layers)]
    V_q_list = [quantize_to_mxfp(V_mats[i]) for i in range(num_layers)]

    precision = _active_precision_settings()
    hbm_act_precision = precision["HBM_V_ACT_TYPE"]
    hbm_weight_precision = precision["HBM_M_WEIGHT_TYPE"]

    def qvfp(t):
        return quantize_to_vector_fp(t, precision)

    print("\n--- CPU Golden Reference (MXFP8 quantized HBM + vector FP intermediates) ---")

    X_gold = qvfp(
        _load_to_vector_fp(token_embeds, hbm_act_precision, precision)
        + _load_to_vector_fp(pos_weight, hbm_act_precision, precision)
    )

    for i in range(num_layers):
        w = all_weights[i]
        W_gate_q = quantize_to_mxfp(w["W_gate"], hbm_weight_precision)
        W_up_q = quantize_to_mxfp(w["W_up"], hbm_weight_precision)
        W_down_q = quantize_to_mxfp(w["W_down"], hbm_weight_precision)

        # --- Attention block ---
        residual = X_gold.clone()
        X_gold = _rms_norm_vector_ref(X_gold, eps, precision)
        if use_qo_proj:
            W_q_q = quantize_to_mxfp(w["W_q"], hbm_weight_precision)
            W_o_q = quantize_to_mxfp(w["W_o"], hbm_weight_precision)
            Q_gold = qvfp(torch.matmul(X_gold.float(), W_q_q.float()))
        else:
            Q_gold = X_gold
        attn_outs = []
        for b in range(batch_size):
            s, e = b * seq_len, (b + 1) * seq_len
            if n_packed_heads > 1:
                head_outs = []
                for h in range(n_packed_heads):
                    hs, he = h * head_dim, (h + 1) * head_dim
                    o_h = _flash_attn_ref(
                        Q_gold[s:e, hs:he], K_q_list[i][s:e], V_q_list[i][s:e], scale, precision=precision
                    )
                    head_outs.append(o_h)
                attn_outs.append(torch.cat(head_outs, dim=-1))
            else:
                attn_outs.append(
                    _flash_attn_ref(Q_gold[s:e], K_q_list[i][s:e], V_q_list[i][s:e], scale, precision=precision)
                )
        O_gold = qvfp(torch.cat(attn_outs, dim=0))
        if use_qo_proj:
            X_gold = qvfp(torch.matmul(O_gold.float(), W_o_q.float()))
        else:
            X_gold = O_gold
        X_gold = qvfp(X_gold + residual)

        # --- FFN block ---
        residual = X_gold.clone()
        X_gold = _rms_norm_vector_ref(X_gold, eps, precision)
        up_out = qvfp(torch.matmul(X_gold.float(), W_up_q.float()))
        gate_out = qvfp(torch.matmul(X_gold.float(), W_gate_q.float()))
        silu_gate = qvfp(qvfp(F.silu(up_out.float())) * gate_out.float())
        X_gold = qvfp(torch.matmul(silu_gate.float(), W_down_q.float()))
        X_gold = qvfp(X_gold + residual)

        print(f"  After layer {i}: X_gold[0,:4] = {X_gold[0, :4].tolist()}")

    # Final norm
    X_gold = _rms_norm_vector_ref(X_gold, eps, precision)

    golden_out = X_gold
    print(f"  golden_out: {golden_out.shape}")
    print(f"  golden_out[0,:4]: {golden_out[0, :4].tolist()}")

    # ----------------------------------------------------------- PLENA ISA
    print("\n--- PLENA Backend (ISA generation) ---")
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PlenaCompiler(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    def _cm(v, m=mlen):
        return ((v + m - 1) // m) * m

    physical_seq_per_batch = max(mlen, _cm(seq_len, blen))
    total_physical_seq = batch_size * physical_seq_per_batch
    act_phys = (total_physical_seq, _cm(hidden_size))
    head_phys = (total_physical_seq, _cm(head_dim))
    (total_physical_seq, _cm(packed_q_dim))
    wq_phys = (_cm(hidden_size), _cm(packed_q_dim))
    wo_phys = (_cm(packed_q_dim), _cm(hidden_size))
    wgate_phys = (_cm(hidden_size), _cm(inter_dim))
    wdown_phys = (_cm(inter_dim), _cm(hidden_size))

    # Shared inputs
    x_input = prog.input("X", shape=(total_seq, hidden_size), physical_shape=act_phys)
    pos_input = prog.input("POS", shape=(total_seq, hidden_size), physical_shape=act_phys)

    # Per-layer weight inputs (order determines HBM layout)
    layer_inputs = []
    for i in range(num_layers):
        li = {}
        if use_qo_proj:
            li["W_q"] = prog.input(f"W_q_{i}", shape=(hidden_size, packed_q_dim), physical_shape=wq_phys)
        li["K"] = prog.input(f"K_{i}", shape=(total_seq, head_dim), physical_shape=head_phys)
        li["V"] = prog.input(f"V_{i}", shape=(total_seq, head_dim), physical_shape=head_phys)
        if use_qo_proj:
            li["W_o"] = prog.input(f"W_o_{i}", shape=(packed_q_dim, hidden_size), physical_shape=wo_phys)
        li["W_gate"] = prog.input(f"W_gate_{i}", shape=(hidden_size, inter_dim), physical_shape=wgate_phys)
        li["W_up"] = prog.input(f"W_up_{i}", shape=(hidden_size, inter_dim), physical_shape=wgate_phys)
        li["W_down"] = prog.input(f"W_down_{i}", shape=(inter_dim, hidden_size), physical_shape=wdown_phys)
        layer_inputs.append(li)

    # Load activations to VRAM
    X_batch = prog.load_batch(x_input, name="X")
    POS_batch = prog.load_batch(pos_input, name="POS")
    ops.embedding_add(prog, X_batch, POS_batch)  # X += POS in-place

    # VRAM layout hazard: ffn_asm writes gate/up intermediates at absolute
    # address batch*hidden spanning up to that + 2*inter*batch.
    # The residual scratch buffer must be placed ABOVE this region.
    _ffn_intermediate_end = total_seq * hidden_size + 2 * inter_dim * total_seq
    _current_bump = 2 * total_seq * hidden_size  # X + POS already allocated
    if _current_bump < _ffn_intermediate_end:
        _pad_size = _ffn_intermediate_end - _current_bump
        _pad_rows = max(1, _pad_size // hidden_size)
        prog.alloc("_vram_padding", _pad_rows, hidden_size)

    # Allocate scratch buffer for residual save/restore (reused across layers)
    scratch = prog.alloc("residual_scratch", total_seq, hidden_size)

    # Chain layers
    current = X_batch  # VRAMMatrixVar tracking the current activation

    for i in range(num_layers):
        li = layer_inputs[i]

        # --- Attention block ---
        # Save residual: scratch = current (zero then add)
        prog.vram_fill_zero(scratch)
        prog.vram_add(scratch, current)

        # Norm (in-place on current)
        prog.rms_norm(current, eps_offset=3, reci_hid_offset=4)

        # Q projection (hidden_size → packed_q_dim) if needed
        if use_qo_proj:
            Q = prog.linear_projection(current, li["W_q"], name=f"Q_{i}")
        else:
            Q = current

        # Flash attention (no RoPE) — GQA when n_packed_heads > 1
        if n_packed_heads > 1:
            O = ops.flash_attention(
                prog, Q, li["K"], li["V"], scale, hq=n_packed_heads, hkv=1, h_qkv=head_dim, batch_size=batch_size
            )
        else:
            O = ops.flash_attention(prog, Q, li["K"], li["V"], scale, batch_size=batch_size)

        # O projection (head_dim → hidden_size) if needed
        if use_qo_proj:
            O_proj = prog.linear_projection(O, li["W_o"], name=f"O_proj_{i}")
        else:
            O_proj = O

        # Attention residual: O_proj += scratch
        prog.vram_add(O_proj, scratch)

        # --- FFN block ---
        # Save residual: scratch = O_proj (zero then add)
        prog.vram_fill_zero(scratch)
        prog.vram_add(scratch, O_proj)

        # Norm (in-place on O_proj)
        prog.rms_norm(O_proj, eps_offset=3, reci_hid_offset=4)

        # FFN (in-place on O_proj)
        ops.ffn(prog, O_proj, li["W_gate"], li["W_up"], li["W_down"])

        # FFN residual: O_proj += scratch
        prog.vram_add(O_proj, scratch)

        current = O_proj  # carry forward

    # Final norm
    prog.rms_norm(current, eps_offset=3, reci_hid_offset=4)

    gen_code = prog.compile()
    lines = gen_code.splitlines()
    print(f"\nGenerated {len(lines)} lines of ISA code")

    # Extract HBM addresses from compiler so create_mem_for_sim pads to match ISA
    hbm_addrs = {name: inp.hbm_addr for name, inp in prog._inputs.items() if hasattr(inp, "hbm_addr")}

    # ----------------------------------------------------------- sim env
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    input_tensors = {"X": token_embeds, "POS": pos_weight}
    data_order = ["X", "POS"]
    for i in range(num_layers):
        if use_qo_proj:
            input_tensors[f"W_q_{i}"] = all_weights[i]["W_q"]
            data_order.append(f"W_q_{i}")
        input_tensors[f"K_{i}"] = K_mats[i]
        input_tensors[f"V_{i}"] = V_mats[i]
        data_order.extend([f"K_{i}", f"V_{i}"])
        if use_qo_proj:
            input_tensors[f"W_o_{i}"] = all_weights[i]["W_o"]
            data_order.append(f"W_o_{i}")
        input_tensors[f"W_gate_{i}"] = all_weights[i]["W_gate"]
        input_tensors[f"W_up_{i}"] = all_weights[i]["W_up"]
        input_tensors[f"W_down_{i}"] = all_weights[i]["W_down"]
        data_order.extend([f"W_gate_{i}", f"W_up_{i}", f"W_down_{i}"])

    golden_result = {"original_output": golden_out}

    def _layout(logical, physical):
        return {
            "source_shape": list(logical),
            "storage_shape": list(physical),
            "logical_shape": list(logical),
            "physical_shape": list(physical),
            "source_row_elements": logical[1],
            "storage_row_elements": physical[1],
        }

    tensor_layouts = {
        "X": _layout((total_seq, hidden_size), act_phys),
        "POS": _layout((total_seq, hidden_size), act_phys),
    }
    for i in range(num_layers):
        if use_qo_proj:
            tensor_layouts[f"W_q_{i}"] = _layout((hidden_size, packed_q_dim), wq_phys)
        tensor_layouts[f"K_{i}"] = _layout((total_seq, head_dim), head_phys)
        tensor_layouts[f"V_{i}"] = _layout((total_seq, head_dim), head_phys)
        if use_qo_proj:
            tensor_layouts[f"W_o_{i}"] = _layout((packed_q_dim, hidden_size), wo_phys)
        tensor_layouts[f"W_gate_{i}"] = _layout((hidden_size, inter_dim), wgate_phys)
        tensor_layouts[f"W_up_{i}"] = _layout((hidden_size, inter_dim), wgate_phys)
        tensor_layouts[f"W_down_{i}"] = _layout((inter_dim, hidden_size), wdown_phys)

    o_vram_addr = prog._compiler.get_vram_addr(current.name)

    return {
        "isa": gen_code,
        "input_tensors": input_tensors,
        "golden_result": golden_result,
        "fp_preload": [0.0, scale, float("-inf"), eps, 1.0 / hidden_size, 1.0] + [0.0] * 4,
        "data_order": data_order,
        "comparison_params": {
            "start_row_idx": o_vram_addr // mlen,
            "num_rows": (act_phys[0] * act_phys[1]) // mlen,
            "num_batches": total_seq,
            "elements_per_batch": hidden_size,
            "row_dim": mlen,
            "physical_rows": act_phys[0],
            "use_stride_mode": hidden_size > mlen,
            "use_slice_mode": act_phys[1] != hidden_size,
            "slice_per_row": hidden_size,
        },
        "tensor_layouts": tensor_layouts,
        "hbm_addrs": hbm_addrs,
    }


def build_and_run_sliced_decoder_chain_test(
    model_id: str,
    asm_name: str,
    build_dir: Path,
    num_layers: int = 2,
    layer_idx_start: int = 0,
    seq_len: int = 64,
    hidden_size: int = 64,
    inter_dim: int = 128,
    batch_size: int = 1,
    mlen: int = MLEN,
    vlen: int | None = None,
    blen: int = BLEN,
    seed: int = 42,
    trust_remote_code: bool = False,
    partial_load: bool = False,
) -> None:
    """End-to-end multi-layer decoder chain test: compile + emulate + assert."""
    from transactional_emulator.testbench.emulator_runner import emulate_from_result

    result = compile_sliced_decoder_chain(
        model_id,
        build_dir,
        num_layers=num_layers,
        layer_idx_start=layer_idx_start,
        seq_len=seq_len,
        hidden_size=hidden_size,
        inter_dim=inter_dim,
        batch_size=batch_size,
        mlen=mlen,
        vlen=vlen,
        blen=blen,
        seed=seed,
        trust_remote_code=trust_remote_code,
        partial_load=partial_load,
    )
    emulate_from_result(result, build_dir, asm_name, mlen=mlen, blen=blen, vlen=vlen)


# Backwards-compatible aliases for older tests/scripts.
