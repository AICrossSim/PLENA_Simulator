from types import SimpleNamespace

import pytest

from transactional_emulator.testbench.model_configs.loader import ModelArchConfig, load_model_config


def test_nemotron3_yaml_has_exact_hybrid_shape() -> None:
    config = load_model_config("nemotron3_nano_30b_a3b")
    arch = config.arch

    assert arch.head_dim == 128
    assert arch.gqa_ratio == 16
    assert arch.count_layers("mamba") == 23
    assert arch.count_layers("moe") == 23
    assert arch.count_layers("attention") == 6
    assert arch.mamba is not None
    assert arch.mamba.projection_size == 10304
    assert arch.mamba.state_elements * 4 == 2 * 1024 * 1024
    assert arch.mamba.conv_channels == 6144
    assert arch.moe is not None
    assert arch.moe.num_experts == 128
    assert arch.moe.experts_per_token == 6


def test_hf_config_uses_explicit_attention_head_dim() -> None:
    hf = SimpleNamespace(
        hidden_size=2688,
        intermediate_size=1856,
        num_attention_heads=32,
        num_key_value_heads=2,
        head_dim=128,
        num_hidden_layers=52,
        hybrid_override_pattern="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        mamba_num_heads=64,
        mamba_head_dim=64,
        ssm_state_size=128,
        n_groups=8,
        conv_kernel=4,
        chunk_size=128,
        mamba_ssm_cache_dtype="float32",
        n_routed_experts=128,
        num_experts_per_tok=6,
        moe_intermediate_size=1856,
        n_shared_experts=1,
        moe_shared_expert_intermediate_size=3712,
        layer_norm_epsilon=1e-5,
        rope_theta=10000,
        vocab_size=131072,
        model_type="nemotron_h",
    )

    arch = ModelArchConfig.from_hf_config(hf)
    assert arch.head_dim == 128
    assert arch.layer_types[0:6] == ["mamba", "moe", "mamba", "moe", "mamba", "attention"]


def test_hybrid_pattern_requires_matching_subconfigs() -> None:
    with pytest.raises(ValueError, match="no Mamba config"):
        ModelArchConfig(
            hidden_size=16,
            inter_dim=32,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
            num_layers=1,
            rope_theta=10000,
            rms_norm_eps=1e-5,
            layer_pattern="M",
        )
