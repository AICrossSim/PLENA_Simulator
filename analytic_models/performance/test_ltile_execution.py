from dataclasses import replace
import numpy as np
import pytest
from .ltile_platform import ExecutionProfile, CodecResources, DmaResources
from .ltile_execution import apply_weight_codec
from .ltile_cost import ProgramCost, assembly_cost
from .weight_codec import decode_e4m3_scale, decode_nvfp4, WeightPacket, storage_bytes


def test_scale_bits_against_torch_float8():
    torch = pytest.importorskip("torch")
    bits = np.arange(256, dtype=np.uint8)
    expected = torch.from_numpy(bits).view(torch.float8_e4m3fn).float().numpy()
    np.testing.assert_equal(decode_e4m3_scale(bits), expected)


def test_nvfp4_all_codes_block_scale_and_global_order():
    packed = np.tile(np.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], np.uint8), (2, 2))
    scales = np.array([[0.5, 2.0], [4.0, 0.25]], np.float32)
    out = decode_nvfp4(packed, scales, 0.125).T
    table = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    expected = np.concatenate([table[:, None] * scales[:, 0] * 0.125, table[:, None] * scales[:, 1] * 0.125]).T
    np.testing.assert_array_equal(out.view(np.uint32), expected.view(np.uint32))


def test_codec_replaces_weights_only_and_charges_global_once():
    c = ProgramCost(memory_trace=[("r", 8192, 16384), ("r", 8192, 16384), ("w", 65536, 4096)])
    metrics = apply_weight_codec(c, [(8192, 16384)], CodecResources())
    reads = [x for x in c.memory_trace if x[0] == "r"]
    assert reads == [("r", 8192, 64), ("r", 8256, 4608), ("r", 8256, 4608)]
    assert metrics["decoded_bf16_bytes"] == 32768
    assert c.total == sum(a for op, a, _ in c.memory_trace if op == "d")
    assert c.transfers["write", 4096] == 1


def test_codec_tail_and_buffer_bounds():
    assert storage_bytes(3, 17) == dict(values=48, block_scales=6, tensor_scale=4, padding_elements=45, total=192)
    assert WeightPacket(8192).transfer_bytes == 4608
    with pytest.raises(ValueError, match="storage"):
        WeightPacket(8192).service(CodecResources(output_bytes=8192))


def test_profile_changes_every_resource_identity():
    p = ExecutionProfile()
    for q in (
        replace(p, hbm_controllers=16),
        replace(p, codec=CodecResources(lanes=128)),
        replace(p, dma=DmaResources(read_credits=1)),
        replace(p, state_rounding="sr"),
    ):
        assert q.identity != p.identity


def test_operator_markers_do_not_change_instruction_work():
    asm = "S_LUI_INT gp1, 0\nS_ADDI_INT gp2, gp1, 64\n"
    plain = assembly_cost(asm, trace_memory=True)
    marked = assembly_cost(
        "; @operator=a\n" + asm.splitlines(True)[0] + "; @operator=b\n" + asm.splitlines(True)[1], trace_memory=True
    )
    assert plain.components() == marked.components()
    assert sum(s["total"] for s in marked.sections) == plain.total
    assert [s["name"] for s in marked.sections] == ["a", "b"]


def test_decode_kv_write_is_inside_the_attention_extent():
    from pathlib import Path
    import os
    from .ltile_decode import workloads
    from .ltile_model import compose_peripheral
    from .nemotron3_workload import WorkloadScenario, InferencePhase

    root = Path(os.environ.get("PLENA_COMPILER_ROOT", Path(__file__).resolve().parents[2] / "PLENA_Compiler"))
    w = workloads(root)["nemotron3"]

    class Recorder:
        profile = ExecutionProfile()

        def __init__(self):
            self.matrices, self.writes = [], []

        def projection(self, b, k, n, weight):
            self.matrices.append((b, k, n))
            return {}

        def vector(self, *args, **kwargs):
            return {}

        def kv_append(self, *args):
            self.writes.append(args)
            return {}

    for context in (1, 4096, 32768):
        recorder = Recorder()
        report = w.build(WorkloadScenario(InferencePhase.DECODE, batch_size=2, context_length=context))
        for name in ("attention_qkv_projection", "attention_qk_softmax_pv"):
            stage = next(s for s in report.stages if s.name == name)
            compose_peripheral(stage, w, 2, context, recorder, {})
        assert recorder.writes[0][-1] == context - 1
        assert recorder.matrices[-2][2] == context
        assert recorder.matrices[-1][1] == context
