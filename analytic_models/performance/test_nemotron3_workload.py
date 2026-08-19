from analytic_models.performance.nemotron3_workload import (
    InferencePhase,
    Nemotron3WorkloadModel,
    Precision,
    ScanStrategy,
    WorkloadScenario,
    affine_scan_pairs,
    formal_nemotron_nvfp4_weight_policy,
    storage_bytes,
)
from analytic_models.performance.b200_formal_campaign import PINNED_SUMMARY, build_report
from transactional_emulator.testbench.model_configs.loader import load_model_config


def _model(**kwargs) -> Nemotron3WorkloadModel:
    arch = load_model_config("nemotron3_nano_30b_a3b").arch
    return Nemotron3WorkloadModel(arch, **kwargs)


def test_decode_uses_real_layer_pattern_and_mamba_shapes() -> None:
    report = _model().build(
        WorkloadScenario(
            phase=InferencePhase.DECODE,
            context_length=2048,
            include_embedding=False,
            include_lm_head=False,
        )
    )

    assert len({stage.layer_id for stage in report.stages if stage.layer_id >= 0}) == 52
    assert sum(stage.name == "mamba_in_projection" for stage in report.stages) == 23
    assert sum(stage.name == "moe_routed_experts" for stage in report.stages) == 23
    assert sum(stage.name == "attention_qkv_projection" for stage in report.stages) == 6

    in_projection = next(stage for stage in report.stages if stage.name == "mamba_in_projection")
    out_projection = next(stage for stage in report.stages if stage.name == "mamba_out_projection")
    assert in_projection.macs == 2688 * 10304
    assert out_projection.macs == 4096 * 2688


def test_decode_state_traffic_is_one_read_and_write_per_mamba_layer() -> None:
    report = _model().build(
        WorkloadScenario(
            phase=InferencePhase.DECODE,
            context_length=2048,
            include_embedding=False,
            include_lm_head=False,
        )
    )
    state = report.total_traffic
    expected_per_layer = 64 * 64 * 128 * 4 + 6144 * 4 * 4
    assert state.state_read_bytes == 23 * expected_per_layer
    assert state.state_write_bytes == 23 * expected_per_layer
    assert expected_per_layer == 2 * 1024 * 1024 + 96 * 1024


def test_prefill_initializes_state_without_sequence_scaled_hbm_state_traffic() -> None:
    report = _model().build(
        WorkloadScenario(
            phase=InferencePhase.PREFILL,
            sequence_length=128,
            context_length=128,
            scan_strategy=ScanStrategy.SEQUENTIAL,
            include_embedding=False,
            include_lm_head=False,
        )
    )
    assert report.total_traffic.state_read_bytes == 0
    assert report.total_traffic.state_write_bytes == 23 * (2 * 1024 * 1024 + 96 * 1024)


def test_chunked_affine_scan_counts_real_128_token_chunk() -> None:
    assert affine_scan_pairs(128) == 769
    report = _model().build(
        WorkloadScenario(
            phase=InferencePhase.PREFILL,
            sequence_length=2048,
            context_length=2048,
            scan_strategy=ScanStrategy.CHUNKED_AFFINE,
            include_embedding=False,
            include_lm_head=False,
        )
    )
    scan = next(stage for stage in report.stages if stage.name == "mamba_chunk_scan_compose")
    intra_cb = next(stage for stage in report.stages if stage.name == "mamba_chunk_intra_cb")
    causal_pairs = 16 * (128 * 129 // 2)

    assert affine_scan_pairs(16) == 49
    assert scan.scan_compositions == 64 * 64 * 128 * 49
    assert scan.working_set_bytes == 16 * 2 * 1024 * 1024
    assert intra_cb.macs == causal_pairs * 64 * 128
    assert all(stage.name != "mamba_state_update" for stage in report.stages)


def test_mx8_storage_includes_one_scale_byte_per_128_elements() -> None:
    assert storage_bytes(128, Precision.MX8) == 129
    assert storage_bytes(129, Precision.MX8) == 131


def test_nvfp4_storage_includes_packed_values_and_one_fp8_scale_per_16() -> None:
    assert storage_bytes(16, Precision.NVFP4) == 9
    assert storage_bytes(17, Precision.NVFP4) == 11


def test_state_precision_changes_state_bytes_without_changing_work() -> None:
    scenario = WorkloadScenario(
        phase=InferencePhase.DECODE,
        include_embedding=False,
        include_lm_head=False,
    )
    fp32 = _model(state_precision=Precision.FP32).build(scenario)
    bf16 = _model(state_precision=Precision.BF16).build(scenario)
    assert fp32.total_macs == bf16.total_macs
    assert fp32.total_traffic.state_read_bytes == 2 * bf16.total_traffic.state_read_bytes


def test_formal_checkpoint_policy_applies_real_layer_specific_bf16_exclusions() -> None:
    arch = load_model_config("nemotron3_nano_30b_a3b").arch
    quantization = build_report(PINNED_SUMMARY)["nemotron"]["checkpoint_quantization"]
    policy = formal_nemotron_nvfp4_weight_policy(arch, quantization)
    report = Nemotron3WorkloadModel(
        arch,
        weight_precision=Precision.NVFP4,
        weight_precision_policy=policy,
    ).build(
        WorkloadScenario(
            phase=InferencePhase.DECODE,
            include_embedding=False,
            include_lm_head=False,
        )
    )

    def weight_bytes(layer_id: int, stage_name: str) -> int:
        return next(
            stage.traffic.weight_read_bytes
            for stage in report.stages
            if stage.layer_id == layer_id and stage.name == stage_name
        )

    mamba = arch.mamba
    moe = arch.moe
    assert mamba is not None and moe is not None
    in_projection_elements = arch.hidden_size * mamba.projection_size
    assert weight_bytes(0, "mamba_in_projection") == storage_bytes(in_projection_elements, Precision.NVFP4)
    assert weight_bytes(4, "mamba_in_projection") == storage_bytes(in_projection_elements, Precision.BF16)
    assert weight_bytes(0, "mamba_conv1d") == storage_bytes(
        mamba.conv_channels * mamba.conv_kernel,
        Precision.BF16,
    )
    attention_elements = arch.hidden_size * (
        arch.num_heads * arch.head_dim + 2 * arch.num_kv_heads * arch.head_dim
    )
    assert weight_bytes(5, "attention_qkv_projection") == storage_bytes(attention_elements, Precision.BF16)
    routed_elements = 6 * 2 * arch.hidden_size * moe.intermediate_size
    assert weight_bytes(1, "moe_routed_experts") == storage_bytes(routed_elements, Precision.NVFP4)
    assert report.to_dict()["weight_precision_policy"]["name"].endswith("mixed_v1")
