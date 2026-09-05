from dataclasses import replace
from pathlib import Path

import pytest

from .hybrid_arch import load_nemotron3_arch
from .kimi_k3_workload import KimiK3HybridWorkloadModel
from .nemotron3_workload import (
    InferencePhase,
    Nemotron3WorkloadModel,
    Precision,
    PrecisionContract,
    StorageFormat,
    WorkloadScenario,
    storage_bytes,
)


def test_mx8_block8_counts_scales_without_changing_legacy_default():
    mx8 = StorageFormat.for_precision(Precision.MX8, mx8_block=8)
    assert mx8.storage_bytes(128) == 144
    assert storage_bytes(128, Precision.MX8) == 129
    assert mx8.storage_bytes(9) == 11
    assert replace(mx8, pad_to_block=True).storage_bytes(9) == 18
    assert replace(mx8, alignment_bytes=64).storage_bytes(9) == 64
    assert mx8.storage_bytes(0) == 0
    assert mx8.to_dict()["scale_format"] == "E8M0"


@pytest.mark.parametrize("precision,block", [(Precision.NVFP4, 8), (Precision.MX8, 0), (Precision.BF16, 8)])
def test_invalid_format_contract_is_rejected(precision, block):
    with pytest.raises(ValueError):
        StorageFormat(precision, block)


@pytest.mark.parametrize("name", ["nemotron3", "kimi_k3"])
def test_changing_weight_format_preserves_other_traffic_and_work(name):
    if name == "nemotron3":
        arch = load_nemotron3_arch(
            Path(__file__).resolve().parents[2] / "PLENA_Compiler/doc/Model_Lib/nemotron-3-nano-30b-a3b.json"
        )

        def build(contract):
            return Nemotron3WorkloadModel(arch, precision_contract=contract)
    else:

        def build(contract):
            return KimiK3HybridWorkloadModel(precision_contract=contract)

    contract = PrecisionContract.bf16_recurrence(Precision.MX8)
    old = replace(contract, weight=StorageFormat.for_precision(Precision.MX8, mx8_block=128))
    scenario = WorkloadScenario(InferencePhase.DECODE, batch_size=16, sequence_length=1, context_length=128)
    before, after = build(old).build(scenario), build(contract).build(scenario)
    assert after.total_traffic.weight_read_bytes > before.total_traffic.weight_read_bytes
    for a, b in zip(before.stages, after.stages, strict=True):
        assert replace(a.traffic, weight_read_bytes=0) == replace(b.traffic, weight_read_bytes=0)
        assert (a.macs, a.elementwise_ops, a.exp_ops) == (b.macs, b.elementwise_ops, b.exp_ops)
    kv_contract = replace(contract, kv=StorageFormat.for_precision(Precision.FP32))
    changed = build(kv_contract).build(scenario)
    assert changed.total_traffic.kv_read_bytes == 2 * after.total_traffic.kv_read_bytes
    assert changed.total_traffic.activation_read_bytes == after.total_traffic.activation_read_bytes
    assert changed.total_traffic.state_read_bytes == after.total_traffic.state_read_bytes


def test_nvfp4_exclusion_format_and_payload():
    contract = PrecisionContract.bf16_recurrence(Precision.NVFP4)
    assert contract.weight.storage_bytes(32) == 18
    assert contract.weight.to_dict()["scale_format"] == "E4M3"
    assert contract.weight_format(Precision.BF16).storage_bytes(32) == 64
    assert contract.activation == contract.kv == contract.state == StorageFormat(Precision.BF16)
