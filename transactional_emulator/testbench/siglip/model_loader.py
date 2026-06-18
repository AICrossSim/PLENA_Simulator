"""Config loader and weight extractor for SigLIP full-model test.

Loads siglip-so400m-patch14-384.json config, instantiates real model from
Hugging Face, extracts per-layer weights in normalized format.
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModel
from transactional_emulator.testbench.siglip.utils.core import (
    resolve_position_embedding,
    resolve_vision_encoder_layer,
)


SIGLIP_VARIANT_CHOICES: tuple[str, ...] = (
    "so400m-patch14-384",
    "large-patch16-256",
    "patch14-8x8-smoke",
    "patch14-3x3-smoke",
)

_SIGLIP_VARIANT_ALIASES: dict[str, str] = {
    "so400m": "so400m-patch14-384",
    "patch14-384": "so400m-patch14-384",
    "large": "large-patch16-256",
    "patch16-256": "large-patch16-256",
    "8x8": "patch14-8x8-smoke",
    "3x3": "patch14-3x3-smoke",
    "so400m-patch14-8x8-smoke": "patch14-8x8-smoke",
    "so400m-patch14-3x3-smoke": "patch14-3x3-smoke",
}

_SIGLIP_VARIANT_SPECS: dict[str, dict[str, object]] = {
    "so400m-patch14-384": {
        "model_id": "google/siglip-so400m-patch14-384",
        "config_candidates": [
            "compiler/doc/Model_Lib/siglip-so400m-patch14-384.json",
            "PLENA_Compiler/doc/Model_Lib/siglip-so400m-patch14-384.json",
        ],
        "vision_defaults": {
            "image_size": 384,
            "patch_size": 14,
            "num_channels": 3,
            "hidden_size": 1152,
            "num_attention_heads": 16,
            "intermediate_size": 4304,
            "num_hidden_layers": 27,
        },
    },
    "large-patch16-256": {
        "model_id": "google/siglip-large-patch16-256",
        "config_candidates": [
            "PLENA_Compiler/doc/Model_Lib/siglip-large-patch16-256.json",
            "compiler/doc/Model_Lib/siglip-large-patch16-256.json",
        ],
        "vision_defaults": {
            "image_size": 256,
            "patch_size": 16,
            "num_channels": 3,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
            "num_hidden_layers": 24,
        },
    },
    "patch14-8x8-smoke": {
        "model_id": None,
        "config_candidates": [
            "PLENA_Compiler/doc/Model_Lib/siglip-so400m-patch14-8x8-smoke.json",
            "compiler/doc/Model_Lib/siglip-so400m-patch14-8x8-smoke.json",
        ],
        "vision_defaults": {
            "image_size": 112,
            "patch_size": 14,
            "num_channels": 3,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "num_hidden_layers": 27,
        },
    },
    "patch14-3x3-smoke": {
        "model_id": None,
        "config_candidates": [
            "PLENA_Compiler/doc/Model_Lib/siglip-so400m-patch14-3x3-smoke.json",
            "compiler/doc/Model_Lib/siglip-so400m-patch14-3x3-smoke.json",
        ],
        "vision_defaults": {
            "image_size": 42,
            "patch_size": 14,
            "num_channels": 3,
            "hidden_size": 128,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "num_hidden_layers": 27,
        },
    },
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _normalize_variant_name(variant: str | None) -> str:
    raw = (variant or os.environ.get("SIGLIP_VARIANT") or "so400m-patch14-384").strip().lower()
    normalized = _SIGLIP_VARIANT_ALIASES.get(raw, raw)
    if normalized not in _SIGLIP_VARIANT_SPECS:
        raise ValueError(
            f"Unsupported SigLIP variant '{raw}'. Supported variants: {', '.join(SIGLIP_VARIANT_CHOICES)}"
        )
    return normalized


def infer_siglip_model_id_from_config(config_path: str | Path) -> str | None:
    """Infer Hugging Face model ID from a SigLIP config file name."""
    name = Path(config_path).name.lower()
    if "large-patch16-256" in name:
        return "google/siglip-large-patch16-256"
    if "patch14-8x8-smoke" in name or "patch14-3x3-smoke" in name:
        return None
    if "so400m-patch14-384" in name:
        return "google/siglip-so400m-patch14-384"
    return None


def resolve_siglip_model_spec(
    *,
    variant: str | None = None,
    config_path: str | Path | None = None,
    model_id: str | None = None,
) -> tuple[str, str | None, str]:
    """Resolve config path + model ID from variant and optional explicit overrides.

    Returns:
        (config_path, model_id_or_none, variant_name)

    ``model_id`` is ``None`` for smoke variants (8x8, 3x3) that have no
    matching Hugging Face model; callers should generate random weights for
    those variants instead of loading from HF.
    """
    variant_name = _normalize_variant_name(variant)
    spec = _SIGLIP_VARIANT_SPECS[variant_name]

    resolved_config_path: str
    if config_path is not None and str(config_path).strip():
        resolved_config_path = str(config_path)
    else:
        repo_root = _repo_root()
        resolved_config_path = str(spec["config_candidates"][0])
        for candidate in spec["config_candidates"]:
            if (repo_root / str(candidate)).exists():
                resolved_config_path = str(candidate)
                break

    resolved_model_id = (model_id or "").strip() or None
    if not resolved_model_id:
        inferred = infer_siglip_model_id_from_config(resolved_config_path)
        resolved_model_id = inferred if inferred is not None else spec.get("model_id")

    return resolved_config_path, resolved_model_id, variant_name


def resolve_siglip_mlen_vlen_for_variant(
    *,
    variant: str | None,
    mlen: int,
    vlen: int,
    mlen_is_explicit: bool = True,
    vlen_is_explicit: bool = True,
) -> tuple[int, int, bool]:
    """Apply per-variant hardware defaults and return adjusted MLEN/VLEN.

    Returns:
        (resolved_mlen, resolved_vlen, was_overridden)
    """
    variant_name = _normalize_variant_name(variant)
    resolved_mlen = int(mlen)
    resolved_vlen = int(vlen)

    if variant_name == "large-patch16-256":
        if not mlen_is_explicit:
            resolved_mlen = 64
        if not vlen_is_explicit:
            resolved_vlen = 64

    changed = (resolved_mlen != int(mlen)) or (resolved_vlen != int(vlen))
    return resolved_mlen, resolved_vlen, changed


def _parse_model_dtype(model_dtype: str) -> torch.dtype:
    """Parse user/model config dtype string into torch dtype."""
    normalized = model_dtype.strip().lower()
    if normalized in {"float32", "fp32", "f32"}:
        return torch.float32
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported model_dtype '{model_dtype}'. Use one of: float32, bfloat16")


def load_siglip_config(config_path: str) -> dict:
    """Load and parse SigLIP config JSON.

    Args:
        config_path: Path to siglip-so400m-patch14-384.json

    Returns:
        dict with normalized config (vision_config merged to top level for convenience)
    """
    with open(config_path) as f:
        full_config = json.load(f)

    variant_name = _normalize_variant_name(Path(config_path).stem.replace("siglip-", ""))
    variant_spec = _SIGLIP_VARIANT_SPECS[variant_name]
    vision_defaults = dict(variant_spec.get("vision_defaults", {}))

    # Extract vision config (primary for vision-only encoder test)
    vision_cfg = full_config.get("vision_config", {})

    # Build normalized config dict
    config = {
        "image_size": int(vision_cfg.get("image_size", vision_defaults.get("image_size", 384))),
        "patch_size": int(vision_cfg.get("patch_size", vision_defaults.get("patch_size", 14))),
        "num_channels": int(vision_cfg.get("num_channels", vision_defaults.get("num_channels", 3))),
        "hidden_size": int(vision_cfg.get("hidden_size", vision_defaults.get("hidden_size", 1152))),
        "num_attention_heads": int(
            vision_cfg.get("num_attention_heads", vision_defaults.get("num_attention_heads", 16))
        ),
        # SigLIP configs often omit num_key_value_heads; treat omission as MHA.
        "num_key_value_heads": int(
            vision_cfg.get(
                "num_key_value_heads",
                vision_cfg.get(
                    "num_attention_heads",
                    vision_defaults.get("num_attention_heads", 16),
                ),
            )
        ),
        "intermediate_size": int(
            vision_cfg.get("intermediate_size", vision_defaults.get("intermediate_size", 4304))
        ),
        "num_hidden_layers": int(
            vision_cfg.get("num_hidden_layers", vision_defaults.get("num_hidden_layers", 27))
        ),
        # HF SigLIP vision defaults use a small epsilon for stable LN.
        "layer_norm_eps": float(vision_cfg.get("layer_norm_eps", 1e-6)),
        "hidden_act": str(vision_cfg.get("hidden_act", "gelu_pytorch_tanh")),
    }

    # Validate shape contract
    validate_config_shape_contract(config)

    return config


def validate_config_shape_contract(config: dict) -> None:
    """Validate that config has valid dimensions.

    Args:
        config: Config dict from load_siglip_config()

    Raises:
        ValueError if shape contract is violated
    """
    hidden = config["hidden_size"]
    heads = config["num_attention_heads"]
    kv_heads = config["num_key_value_heads"]
    inter = config["intermediate_size"]
    image_size = config["image_size"]
    patch_size = config["patch_size"]
    num_patches = (image_size // patch_size) ** 2

    # Check divisibility
    if hidden % heads != 0:
        raise ValueError(f"hidden_size ({hidden}) must be divisible by num_attention_heads ({heads})")

    if hidden % kv_heads != 0:
        raise ValueError(f"hidden_size ({hidden}) must be divisible by num_key_value_heads ({kv_heads})")

    if hidden <= 0 or heads <= 0 or inter <= 0:
        raise ValueError(f"Negative or zero dimensions: hidden={hidden}, heads={heads}, inter={inter}")

    if num_patches <= 0:
        raise ValueError(f"Invalid image/patch size: {image_size}/{patch_size} -> {num_patches} patches")

    print(f"✓ Config validated: hidden={hidden}, heads={heads}, patches={num_patches}, layers={config['num_hidden_layers']}")


def load_siglip_vision_model(
    model_id: str = "google/siglip-so400m-patch14-384",
    model_dtype: str = "float32",
) -> torch.nn.Module:
    """Load real SigLIP vision model from Hugging Face.

    Args:
        model_id: Model ID on Hugging Face Hub
        model_dtype: Model load dtype (float32 or bfloat16)

    Returns:
        Loaded model
    """
    torch_dtype = _parse_model_dtype(model_dtype)
    print(f"Loading {model_id} from Hugging Face...")
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch_dtype)
    print("✓ Model loaded successfully")
    return model


def extract_embedding_weights(model: torch.nn.Module, config: dict) -> dict:
    """Extract patch embedding and position embedding weights.

    Args:
        model: Real SigLIP model from load_siglip_vision_model()
        config: Config dict from load_siglip_config()

    Returns:
        dict with keys:
            - patch_weight: (hidden_size, num_channels*patch_size*patch_size)
            - patch_bias: (hidden_size,)
            - position_table: (num_patches, hidden_size)
    """
    vision_root = getattr(model, "vision_model", model)
    embeddings = vision_root.embeddings

    # Extract patch embedding weights
    patch_weight = embeddings.patch_embedding.weight.detach()  # (hidden_size, C_in, K, K)
    patch_bias = embeddings.patch_embedding.bias.detach() if hasattr(embeddings.patch_embedding, "bias") else None

    # Reshape weight to 2D: (hidden_size, C_in*K*K)
    if len(patch_weight.shape) == 4:
        patch_weight = patch_weight.reshape(patch_weight.shape[0], -1).T.contiguous()

    # Extract position embedding table
    position_table = resolve_position_embedding(vision_root).detach()  # (num_patches + 1, hidden_size) or (num_patches, hidden_size)

    # Remove class token if present (first row)
    image_size = config["image_size"]
    patch_size = config["patch_size"]
    num_patches = (image_size // patch_size) ** 2
    if position_table.shape[0] > num_patches:
        position_table = position_table[1:num_patches + 1, :]  # Skip class token

    weights = {
        "patch_weight": patch_weight.detach(),
        "patch_bias": patch_bias.detach() if patch_bias is not None else None,
        "position_table": position_table.detach(),
    }

    print(f"✓ Embedding weights extracted: patch_weight={weights['patch_weight'].shape}, position_table={weights['position_table'].shape}")

    return weights


def extract_layer_weights(model: torch.nn.Module, layer_idx: int) -> dict:
    """Extract weights for one encoder layer.

    Args:
        model: Real SigLIP model from load_siglip_vision_model()
        layer_idx: Layer index (0 to num_hidden_layers-1)

    Returns:
        dict with keys:
            - ln1_weight: Layer norm 1 weight
            - ln1_bias: Layer norm 1 bias
            - q_proj_weight, k_proj_weight, v_proj_weight, q_proj_bias, k_proj_bias, v_proj_bias
            - out_proj_weight, out_proj_bias
            - ln2_weight, ln2_bias
            - fc1_weight, fc1_bias (MLP up)
            - fc2_weight, fc2_bias (MLP down)
            - optional fc3_weight, fc3_bias for non-SigLIP variants
    """
    layer = resolve_vision_encoder_layer(model, layer_idx)

    weights = {}

    # Layer norm 1
    if hasattr(layer, "self_attn_layer_norm"):
        weights["ln1_weight"] = layer.self_attn_layer_norm.weight.detach()
        weights["ln1_bias"] = layer.self_attn_layer_norm.bias.detach() if hasattr(layer.self_attn_layer_norm, "bias") else None
    elif hasattr(layer, "layer_norm1"):
        weights["ln1_weight"] = layer.layer_norm1.weight.detach()
        weights["ln1_bias"] = layer.layer_norm1.bias.detach() if hasattr(layer.layer_norm1, "bias") else None
    else:
        raise AttributeError(f"Layer {layer_idx}: Could not locate LN1")

    # Self attention projections
    attn = layer.self_attn if hasattr(layer, "self_attn") else layer.attention
    weights["q_proj_weight"] = attn.q_proj.weight.detach()
    weights["q_proj_bias"] = attn.q_proj.bias.detach() if hasattr(attn.q_proj, "bias") else None
    weights["k_proj_weight"] = attn.k_proj.weight.detach()
    weights["k_proj_bias"] = attn.k_proj.bias.detach() if hasattr(attn.k_proj, "bias") else None
    weights["v_proj_weight"] = attn.v_proj.weight.detach()
    weights["v_proj_bias"] = attn.v_proj.bias.detach() if hasattr(attn.v_proj, "bias") else None

    # Output projection
    weights["out_proj_weight"] = attn.out_proj.weight.detach()
    weights["out_proj_bias"] = attn.out_proj.bias.detach() if hasattr(attn.out_proj, "bias") else None

    # Layer norm 2
    if hasattr(layer, "final_layer_norm"):
        weights["ln2_weight"] = layer.final_layer_norm.weight.detach()
        weights["ln2_bias"] = layer.final_layer_norm.bias.detach() if hasattr(layer.final_layer_norm, "bias") else None
    elif hasattr(layer, "layer_norm2"):
        weights["ln2_weight"] = layer.layer_norm2.weight.detach()
        weights["ln2_bias"] = layer.layer_norm2.bias.detach() if hasattr(layer.layer_norm2, "bias") else None
    else:
        raise AttributeError(f"Layer {layer_idx}: Could not locate LN2")

    # SigLIP MLP is two linear layers: fc1 (up) then fc2 (down).
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        # Extract standard two-layer MLP weights.
        if hasattr(mlp, "fc1"):
            weights["fc1_weight"] = mlp.fc1.weight.detach()  # Up projection
            weights["fc1_bias"] = mlp.fc1.bias.detach() if hasattr(mlp.fc1, "bias") else None
        if hasattr(mlp, "fc2"):
            weights["fc2_weight"] = mlp.fc2.weight.detach()  # Down projection
            weights["fc2_bias"] = mlp.fc2.bias.detach() if hasattr(mlp.fc2, "bias") else None
    else:
        raise AttributeError(f"Layer {layer_idx}: Could not locate MLP")

    print(f"✓ Layer {layer_idx} weights extracted")
    return weights


def extract_final_ln_weights(model: torch.nn.Module) -> dict | None:
    """Extract final layer norm weights if present (after all encoder layers).

    Args:
        model: Real SigLIP model from load_siglip_vision_model()

    Returns:
        dict with ln_weight and ln_bias, or None if not present
    """
    vision_root = getattr(model, "vision_model", model)

    if hasattr(vision_root, "post_layernorm"):
        ln = vision_root.post_layernorm
        return {
            "ln_weight": ln.weight.detach(),
            "ln_bias": ln.bias.detach() if hasattr(ln, "bias") else None,
        }

    if hasattr(vision_root, "layer_norm"):
        ln = vision_root.layer_norm
        return {
            "ln_weight": ln.weight.detach(),
            "ln_bias": ln.bias.detach() if hasattr(ln, "bias") else None,
        }

    print("✓ No final layer norm found")
    return None


def _xavier_weight(fan_out: int, fan_in: int) -> torch.Tensor:
    """Xavier-uniform-like initialization scaled for stable forward pass.

    Returns a tensor with std ≈ sqrt(2 / (fan_in + fan_out)) so that
    activations remain roughly unit-variance through linear layers.
    """
    std = (2.0 / (fan_in + fan_out)) ** 0.5
    return torch.randn(fan_out, fan_in, dtype=torch.float32) * std


def generate_random_embedding_weights(
    config: dict,
    seed: int = 42,
) -> dict:
    """Generate random embedding weights matching config dimensions.

    The shapes mirror what ``extract_embedding_weights`` produces but with
    Xavier-scaled random values instead of real HF weights.  Used for smoke
    variants that have no corresponding Hugging Face model.
    """
    torch.manual_seed(seed)
    hidden = int(config["hidden_size"])
    patch_size = int(config["patch_size"])
    num_channels = int(config["num_channels"])
    in_features = num_channels * patch_size * patch_size
    num_patches = (int(config["image_size"]) // patch_size) ** 2

    patch_weight = _xavier_weight(hidden, in_features).T.contiguous()  # (in_features, hidden)
    patch_bias = torch.zeros(hidden, dtype=torch.float32)
    position_table = torch.randn(num_patches, hidden, dtype=torch.float32) * 0.02

    return {
        "patch_weight": patch_weight,
        "patch_bias": patch_bias,
        "position_table": position_table,
    }


def generate_random_layer_weights(
    config: dict,
    layer_idx: int,
    seed: int = 42,
) -> dict:
    """Generate random encoder-layer weights matching config dimensions.

    The shapes mirror what ``extract_layer_weights`` produces but with
    Xavier-scaled random values.  Each layer gets a distinct seed derived
    from ``seed + layer_idx`` so layers are not identical.

    Biases are initialised to zero and LayerNorm weights to one to keep
    the forward pass well-conditioned.
    """
    torch.manual_seed(seed + layer_idx)
    hidden = int(config["hidden_size"])
    inter = int(config["intermediate_size"])

    return {
        "ln1_weight": torch.ones(hidden, dtype=torch.float32),
        "ln1_bias": torch.zeros(hidden, dtype=torch.float32),
        "q_proj_weight": _xavier_weight(hidden, hidden),
        "q_proj_bias": torch.zeros(hidden, dtype=torch.float32),
        "k_proj_weight": _xavier_weight(hidden, hidden),
        "k_proj_bias": torch.zeros(hidden, dtype=torch.float32),
        "v_proj_weight": _xavier_weight(hidden, hidden),
        "v_proj_bias": torch.zeros(hidden, dtype=torch.float32),
        "out_proj_weight": _xavier_weight(hidden, hidden),
        "out_proj_bias": torch.zeros(hidden, dtype=torch.float32),
        "ln2_weight": torch.ones(hidden, dtype=torch.float32),
        "ln2_bias": torch.zeros(hidden, dtype=torch.float32),
        "fc1_weight": _xavier_weight(inter, hidden),
        "fc1_bias": torch.zeros(inter, dtype=torch.float32),
        "fc2_weight": _xavier_weight(hidden, inter),
        "fc2_bias": torch.zeros(hidden, dtype=torch.float32),
    }


def generate_random_final_ln_weights(config: dict, seed: int = 42) -> dict:
    """Generate random final LayerNorm weights matching config dimensions."""
    torch.manual_seed(seed)
    hidden = int(config["hidden_size"])
    return {
        "ln_weight": torch.ones(hidden, dtype=torch.float32),
        "ln_bias": torch.zeros(hidden, dtype=torch.float32),
    }


if __name__ == "__main__":
    """Quick test of config loader."""
    config_path, model_id, variant = resolve_siglip_model_spec()

    print("=" * 80)
    print("SigLIP Config Loader Test")
    print("=" * 80)
    print(f"Variant: {variant}")
    print(f"Config: {config_path}")
    print(f"Model ID: {model_id}")

    # Test 1: Load config
    config = load_siglip_config(config_path)
    print("\nConfig loaded:")
    for key, val in config.items():
        print(f"  {key}: {val}")

    # Test 2: Load model
    model = load_siglip_vision_model(model_id=model_id)

    # Test 3: Extract embedding weights
    embed_weights = extract_embedding_weights(model, config)

    # Test 4: Extract layer 0 weights
    layer0_weights = extract_layer_weights(model, 0)

    # Test 5: Check final layer norm
    final_ln = extract_final_ln_weights(model)

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
