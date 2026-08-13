from analytic_models.performance.nemotron3_workload import (
    InferencePhase,
    Nemotron3WorkloadModel,
    Precision,
    ScanStrategy,
    WorkloadScenario,
    affine_scan_pairs,
    storage_bytes,
)
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
