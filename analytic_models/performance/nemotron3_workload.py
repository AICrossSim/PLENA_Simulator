"""Architecture-independent work and traffic model for Nemotron 3 Nano.

This module deliberately does not turn work into cycles.  It describes the
model that must execute; ``nemotron3_dse`` separately applies a candidate PLENA
hardware design to that work.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from enum import StrEnum

from .hybrid_arch import ModelArchConfig


class InferencePhase(StrEnum):
    PREFILL = "prefill"
    DECODE = "decode"


class Precision(StrEnum):
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    MX8 = "mx8"
    MXFP4 = "mxfp4"
    NVFP4 = "nvfp4"


class ScanStrategy(StrEnum):
    SEQUENTIAL = "sequential"
    CHUNKED_AFFINE = "chunked_affine"


@dataclass(frozen=True)
class StagePrecisionOverride:
    stage_name: str
    layer_ids: tuple[int, ...]
    precision: Precision


@dataclass(frozen=True)
class WeightPrecisionPolicy:
    """Resolve checkpoint weight storage at stage and layer granularity."""

    name: str
    default_precision: Precision
    global_stage_precisions: tuple[tuple[str, Precision], ...] = ()
    layer_stage_precisions: tuple[StagePrecisionOverride, ...] = ()
    source: str = "unspecified"

    def __post_init__(self) -> None:
        global_names = [name for name, _ in self.global_stage_precisions]
        if len(global_names) != len(set(global_names)):
            raise ValueError("global weight precision stages must be unique")
        override_keys = [
            (override.stage_name, layer_id)
            for override in self.layer_stage_precisions
            for layer_id in override.layer_ids
        ]
        if len(override_keys) != len(set(override_keys)):
            raise ValueError("layer weight precision overrides must be unique")

    def precision_for(self, layer_id: int, stage_name: str) -> Precision:
        for override in self.layer_stage_precisions:
            if override.stage_name == stage_name and layer_id in override.layer_ids:
                return override.precision
        for name, precision in self.global_stage_precisions:
            if name == stage_name:
                return precision
        return self.default_precision

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "default_precision": self.default_precision,
            "global_stage_precisions": {name: precision for name, precision in self.global_stage_precisions},
            "layer_stage_precisions": [asdict(override) for override in self.layer_stage_precisions],
            "source": self.source,
        }


def formal_nemotron_nvfp4_weight_policy(
    arch: ModelArchConfig,
    quantization: Mapping[str, object],
) -> WeightPrecisionPolicy:
    """Build and validate the mixed checkpoint policy observed in the B200 run."""

    if quantization.get("default_linear_weight") != "nvfp4":
        raise ValueError("formal checkpoint must declare NVFP4 as its default linear weight")
    if quantization.get("group_size") != 16:
        raise ValueError("formal checkpoint must use NVFP4 group size 16")
    if quantization.get("excluded_modules_remain_model_dtype") is not True:
        raise ValueError("formal checkpoint exclusions must remain in model dtype")

    def layer_ids(key: str) -> tuple[int, ...]:
        value = quantization.get(key)
        if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
            raise ValueError(f"formal checkpoint field {key} must be a list of layer IDs")
        return tuple(value)

    mamba_layers = tuple(index for index, kind in enumerate(arch.layer_types) if kind == "mamba")
    attention_layers = tuple(index for index, kind in enumerate(arch.layer_types) if kind == "attention")
    projection_bf16 = layer_ids("mamba_projection_bf16_layers")
    attention_bf16 = layer_ids("attention_projection_bf16_layers")
    conv_bf16 = layer_ids("mamba_conv_bf16_layers")
    if not set(projection_bf16).issubset(mamba_layers):
        raise ValueError("Mamba projection exclusions contain a non-Mamba layer")
    if attention_bf16 != attention_layers:
        raise ValueError("formal checkpoint must exclude every Attention projection")
    if conv_bf16 != mamba_layers:
        raise ValueError("formal checkpoint must exclude every Mamba convolution")
    if quantization.get("lm_head_bf16") is not True:
        raise ValueError("formal checkpoint must keep lm_head in BF16")

    return WeightPrecisionPolicy(
        name="nemotron3_nano_30b_a3b_nvfp4_checkpoint_mixed_v1",
        default_precision=Precision.NVFP4,
        global_stage_precisions=(
            ("embedding_lookup", Precision.BF16),
            ("block_rms_norm", Precision.BF16),
            ("mamba_conv1d", Precision.BF16),
            ("mamba_gate_group_rms_norm", Precision.BF16),
            ("lm_head", Precision.BF16),
        ),
        layer_stage_precisions=(
            StagePrecisionOverride("mamba_in_projection", projection_bf16, Precision.BF16),
            StagePrecisionOverride("mamba_out_projection", projection_bf16, Precision.BF16),
            StagePrecisionOverride("attention_qkv_projection", attention_bf16, Precision.BF16),
            StagePrecisionOverride("attention_out_projection", attention_bf16, Precision.BF16),
        ),
        source="B200 complete campaign checkpoint quantization_config exclusions",
    )


def storage_bytes(elements: int, precision: Precision, block_size: int = 128) -> int:
    """Return logical payload bytes including block scales.

    NVFP4 uses packed E2M1 values plus one FP8 E4M3 scale per 16 values.
    Tensor-global FP32 scales and hardware-specific scale padding are excluded;
    callers that model physical checkpoint layout must add those separately.
    """
    if elements < 0:
        raise ValueError("elements must be non-negative")
    if precision == Precision.FP32:
        return elements * 4
    if precision in {Precision.BF16, Precision.FP16}:
        return elements * 2
    if precision == Precision.MX8:
        return elements + math.ceil(elements / block_size)
    if precision == Precision.MXFP4:
        # OCP MXFP4: two E2M1 values per byte and one E8M0 scale per
        # 32-value block. Tensor padding/alignment remains a separate physical
        # layout concern, as it is for NVFP4 below.
        return math.ceil(elements / 2) + math.ceil(elements / 32)
    if precision == Precision.NVFP4:
        return math.ceil(elements / 2) + math.ceil(elements / 16)
    raise ValueError(f"unsupported precision {precision}")


@dataclass(frozen=True)
class Traffic:
    """Logical transfers; on-chip counters do not prescribe a physical SRAM."""

    weight_read_bytes: int = 0
    activation_read_bytes: int = 0
    activation_write_bytes: int = 0
    kv_read_bytes: int = 0
    kv_write_bytes: int = 0
    state_read_bytes: int = 0
    state_write_bytes: int = 0
    on_chip_read_bytes: int = 0
    on_chip_write_bytes: int = 0

    def __post_init__(self) -> None:
        for field in fields(self):
            if getattr(self, field.name) < 0:
                raise ValueError(f"traffic field {field.name} must be non-negative")

    def __add__(self, other: Traffic) -> Traffic:
        return Traffic(**{field.name: getattr(self, field.name) + getattr(other, field.name) for field in fields(self)})

    @property
    def logical_hbm_read_bytes(self) -> int:
        return self.weight_read_bytes + self.activation_read_bytes + self.kv_read_bytes + self.state_read_bytes

    @property
    def logical_hbm_write_bytes(self) -> int:
        return self.activation_write_bytes + self.kv_write_bytes + self.state_write_bytes


@dataclass(frozen=True)
class StageWork:
    layer_id: int
    layer_type: str
    name: str
    resource: str
    macs: int = 0
    elementwise_ops: int = 0
    exp_ops: int = 0
    scan_compositions: int = 0
    traffic: Traffic = Traffic()
    working_set_bytes: int = 0

    def __post_init__(self) -> None:
        for name in ("macs", "elementwise_ops", "exp_ops", "scan_compositions", "working_set_bytes"):
            if getattr(self, name) < 0:
                raise ValueError(f"stage field {name} must be non-negative")

    @property
    def flops(self) -> int:
        # One MAC is reported as two floating-point operations.
        return 2 * self.macs + self.elementwise_ops + self.exp_ops

    def to_dict(self) -> dict:
        result = asdict(self)
        result["flops"] = self.flops
        result["logical_hbm_read_bytes"] = self.traffic.logical_hbm_read_bytes
        result["logical_hbm_write_bytes"] = self.traffic.logical_hbm_write_bytes
        return result


@dataclass(frozen=True)
class WorkloadScenario:
    phase: InferencePhase
    batch_size: int = 1
    sequence_length: int = 1
    context_length: int = 2048
    decode_tokens: int = 1
    scan_strategy: ScanStrategy = ScanStrategy.SEQUENTIAL
    continue_state: bool | None = None
    include_embedding: bool = True
    include_lm_head: bool = True
    moe_unique_experts: int | None = None

    def __post_init__(self) -> None:
        for name in ("batch_size", "sequence_length", "context_length", "decode_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.phase == InferencePhase.DECODE and self.sequence_length != 1:
            raise ValueError("decode scenario requires sequence_length=1")
        if self.moe_unique_experts is not None and self.moe_unique_experts <= 0:
            raise ValueError("moe_unique_experts must be positive when provided")

    @property
    def tokens(self) -> int:
        return self.batch_size * self.sequence_length

    @property
    def reads_initial_state(self) -> bool:
        if self.continue_state is not None:
            return self.continue_state
        return self.phase == InferencePhase.DECODE


@dataclass(frozen=True)
class WorkloadReport:
    scenario: WorkloadScenario
    activation_precision: Precision
    weight_precision: Precision
    state_precision: Precision
    stages: tuple[StageWork, ...]
    weight_precision_policy: WeightPrecisionPolicy | None = None

    @property
    def total_macs(self) -> int:
        return sum(stage.macs for stage in self.stages)

    @property
    def total_flops(self) -> int:
        return sum(stage.flops for stage in self.stages)

    @property
    def total_traffic(self) -> Traffic:
        total = Traffic()
        for stage in self.stages:
            total += stage.traffic
        return total

    def to_dict(self) -> dict:
        layer_counts: dict[str, set[int]] = {}
        for stage in self.stages:
            if stage.layer_id >= 0:
                layer_counts.setdefault(stage.layer_type, set()).add(stage.layer_id)
        return {
            "scenario": asdict(self.scenario),
            "precisions": {
                "activation": self.activation_precision,
                "weight": self.weight_precision,
                "state": self.state_precision,
            },
            "weight_precision_policy": (
                self.weight_precision_policy.to_dict() if self.weight_precision_policy is not None else None
            ),
            "layer_counts": {name: len(layer_ids) for name, layer_ids in layer_counts.items()},
            "totals": {
                "macs": self.total_macs,
                "flops": self.total_flops,
                "traffic": asdict(self.total_traffic),
                "logical_hbm_read_bytes": self.total_traffic.logical_hbm_read_bytes,
                "logical_hbm_write_bytes": self.total_traffic.logical_hbm_write_bytes,
            },
            "stages": [stage.to_dict() for stage in self.stages],
        }


def affine_scan_pairs(length: int) -> int:
    """Number of pair compositions in an inclusive Hillis-Steele scan."""
    if length <= 0:
        return 0
    pairs = 0
    offset = 1
    while offset < length:
        pairs += length - offset
        offset *= 2
    return pairs


class Nemotron3WorkloadModel:
    def __init__(
        self,
        arch: ModelArchConfig,
        *,
        activation_precision: Precision = Precision.BF16,
        weight_precision: Precision = Precision.BF16,
        state_precision: Precision = Precision.FP32,
        weight_precision_policy: WeightPrecisionPolicy | None = None,
    ) -> None:
        if arch.layer_pattern is None or arch.mamba is None or arch.moe is None:
            raise ValueError("Nemotron 3 workload requires hybrid, Mamba, and MoE configuration")
        self.arch = arch
        self.activation_precision = activation_precision
        self.weight_precision_policy = weight_precision_policy
        self.weight_precision = (
            weight_precision_policy.default_precision if weight_precision_policy is not None else weight_precision
        )
        self.state_precision = state_precision

    def build(self, scenario: WorkloadScenario) -> WorkloadReport:
        stages: list[StageWork] = []
        if scenario.include_embedding:
            stages.append(self._embedding(scenario))

        for layer_id, layer_type in enumerate(self.arch.layer_types):
            stages.append(self._block_norm(layer_id, layer_type, scenario))
            if layer_type == "mamba":
                stages.extend(self._mamba(layer_id, scenario))
            elif layer_type == "attention":
                stages.extend(self._attention(layer_id, scenario))
            elif layer_type == "moe":
                stages.extend(self._moe(layer_id, scenario))
            else:
                stages.extend(self._mlp(layer_id, scenario))
            stages.append(self._residual(layer_id, layer_type, scenario))

        if scenario.include_lm_head:
            stages.append(self._lm_head(scenario))
        return WorkloadReport(
            scenario=scenario,
            activation_precision=self.activation_precision,
            weight_precision=self.weight_precision,
            state_precision=self.state_precision,
            stages=tuple(stages),
            weight_precision_policy=self.weight_precision_policy,
        )

    def _a_bytes(self, elements: int) -> int:
        return storage_bytes(elements, self.activation_precision)

    def _w_bytes(self, elements: int, layer_id: int, stage_name: str) -> int:
        precision = self.weight_precision
        if self.weight_precision_policy is not None:
            precision = self.weight_precision_policy.precision_for(layer_id, stage_name)
        return storage_bytes(elements, precision)

    def _s_bytes(self, elements: int) -> int:
        return storage_bytes(elements, self.state_precision)

    def _embedding(self, scenario: WorkloadScenario) -> StageWork:
        elements = scenario.tokens * self.arch.hidden_size
        stage_name = "embedding_lookup"
        return StageWork(
            -1,
            "embedding",
            stage_name,
            "dma",
            traffic=Traffic(
                weight_read_bytes=self._w_bytes(elements, -1, stage_name),
                activation_write_bytes=self._a_bytes(elements),
            ),
            working_set_bytes=self._a_bytes(elements),
        )

    def _block_norm(self, layer_id: int, layer_type: str, scenario: WorkloadScenario) -> StageWork:
        elements = scenario.tokens * self.arch.hidden_size
        stage_name = "block_rms_norm"
        return StageWork(
            layer_id,
            layer_type,
            stage_name,
            "vector",
            elementwise_ops=5 * elements,
            traffic=Traffic(
                weight_read_bytes=self._w_bytes(self.arch.hidden_size, layer_id, stage_name),
                activation_read_bytes=self._a_bytes(elements),
                activation_write_bytes=self._a_bytes(elements),
            ),
            working_set_bytes=self._a_bytes(elements),
        )

    def _residual(self, layer_id: int, layer_type: str, scenario: WorkloadScenario) -> StageWork:
        elements = scenario.tokens * self.arch.hidden_size
        return StageWork(
            layer_id,
            layer_type,
            "block_residual",
            "vector",
            elementwise_ops=elements,
            traffic=Traffic(
                activation_read_bytes=2 * self._a_bytes(elements),
                activation_write_bytes=self._a_bytes(elements),
            ),
            working_set_bytes=2 * self._a_bytes(elements),
        )

    def _mamba(self, layer_id: int, scenario: WorkloadScenario) -> list[StageWork]:
        mamba = self.arch.mamba
        assert mamba is not None
        tokens = scenario.tokens
        projection_elements = tokens * mamba.projection_size
        conv_elements = tokens * mamba.conv_channels
        inner_elements = tokens * mamba.d_inner
        state_elements = scenario.batch_size * mamba.state_elements
        conv_state_elements = scenario.batch_size * mamba.conv_channels * mamba.conv_kernel

        stages = [
            StageWork(
                layer_id,
                "mamba",
                "mamba_in_projection",
                "matrix",
                macs=tokens * self.arch.hidden_size * mamba.projection_size,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(
                        self.arch.hidden_size * mamba.projection_size,
                        layer_id,
                        "mamba_in_projection",
                    ),
                    activation_read_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                    activation_write_bytes=self._a_bytes(projection_elements),
                    on_chip_write_bytes=self._a_bytes(projection_elements),
                ),
                working_set_bytes=self._a_bytes(projection_elements),
            ),
            StageWork(
                layer_id,
                "mamba",
                "mamba_conv1d",
                "conv",
                macs=conv_elements * mamba.conv_kernel,
                elementwise_ops=2 * conv_elements,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(
                        mamba.conv_channels * mamba.conv_kernel,
                        layer_id,
                        "mamba_conv1d",
                    ),
                    on_chip_read_bytes=self._a_bytes(conv_elements),
                    on_chip_write_bytes=self._a_bytes(conv_elements),
                    state_read_bytes=self._s_bytes(conv_state_elements) if scenario.reads_initial_state else 0,
                    state_write_bytes=self._s_bytes(conv_state_elements),
                ),
                working_set_bytes=self._s_bytes(conv_state_elements),
            ),
            StageWork(
                layer_id,
                "mamba",
                "mamba_dt_exp",
                "exp",
                elementwise_ops=4 * tokens * mamba.num_heads,
                exp_ops=2 * tokens * mamba.num_heads,
                traffic=Traffic(on_chip_read_bytes=self._a_bytes(tokens * mamba.num_heads)),
                working_set_bytes=self._a_bytes(tokens * mamba.num_heads),
            ),
        ]

        if scenario.phase == InferencePhase.PREFILL and scenario.scan_strategy == ScanStrategy.CHUNKED_AFFINE:
            stages.extend(self._mamba_chunked_ssd(layer_id, scenario))
        else:
            stages.extend(
                [
                    StageWork(
                        layer_id,
                        "mamba",
                        "mamba_state_update",
                        "state",
                        macs=tokens * mamba.state_elements,
                        elementwise_ops=2 * tokens * mamba.state_elements,
                        traffic=Traffic(
                            state_read_bytes=self._s_bytes(state_elements) if scenario.reads_initial_state else 0,
                            state_write_bytes=self._s_bytes(state_elements),
                            on_chip_read_bytes=self._a_bytes(
                                tokens * (mamba.d_inner + 2 * mamba.groups * mamba.state_dim + mamba.num_heads)
                            ),
                        ),
                        working_set_bytes=self._s_bytes(state_elements),
                    ),
                    StageWork(
                        layer_id,
                        "mamba",
                        "mamba_state_output",
                        "state",
                        macs=tokens * (mamba.state_elements + mamba.d_inner),
                        elementwise_ops=tokens * mamba.d_inner,
                        traffic=Traffic(
                            on_chip_read_bytes=self._a_bytes(tokens * (mamba.groups * mamba.state_dim + mamba.d_inner)),
                            on_chip_write_bytes=self._a_bytes(inner_elements),
                        ),
                        working_set_bytes=self._a_bytes(inner_elements),
                    ),
                ]
            )

        stages.extend(
            [
                StageWork(
                    layer_id,
                    "mamba",
                    "mamba_gate_group_rms_norm",
                    "vector",
                    elementwise_ops=8 * inner_elements,
                    traffic=Traffic(
                        weight_read_bytes=self._w_bytes(
                            mamba.d_inner,
                            layer_id,
                            "mamba_gate_group_rms_norm",
                        ),
                        on_chip_read_bytes=self._a_bytes(2 * inner_elements),
                        on_chip_write_bytes=self._a_bytes(inner_elements),
                    ),
                    working_set_bytes=self._a_bytes(2 * inner_elements),
                ),
                StageWork(
                    layer_id,
                    "mamba",
                    "mamba_out_projection",
                    "matrix",
                    macs=tokens * mamba.d_inner * self.arch.hidden_size,
                    traffic=Traffic(
                        weight_read_bytes=self._w_bytes(
                            mamba.d_inner * self.arch.hidden_size,
                            layer_id,
                            "mamba_out_projection",
                        ),
                        on_chip_read_bytes=self._a_bytes(inner_elements),
                        activation_write_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                    ),
                    working_set_bytes=self._a_bytes(inner_elements),
                ),
            ]
        )
        return stages

    def _mamba_chunked_ssd(self, layer_id: int, scenario: WorkloadScenario) -> list[StageWork]:
        """Model the four SSD blocks in Nemotron's Mamba-2 prefill path."""
        mamba = self.arch.mamba
        assert mamba is not None
        chunk_lengths = []
        remaining = scenario.sequence_length
        while remaining:
            length = min(remaining, mamba.chunk_size)
            chunk_lengths.append(length)
            remaining -= length

        tokens = scenario.tokens
        chunks = len(chunk_lengths)
        causal_pairs = sum(length * (length + 1) // 2 for length in chunk_lengths)
        scan_pairs = affine_scan_pairs(chunks)
        state_elements = scenario.batch_size * mamba.state_elements
        inner_elements = tokens * mamba.d_inner
        compositions = state_elements * scan_pairs
        chunk_states = state_elements * chunks

        return [
            StageWork(
                layer_id,
                "mamba",
                "mamba_chunk_intra_cb",
                "matrix",
                macs=scenario.batch_size * causal_pairs * mamba.num_heads * mamba.state_dim,
                traffic=Traffic(
                    on_chip_read_bytes=self._a_bytes(tokens * 2 * mamba.groups * mamba.state_dim),
                    on_chip_write_bytes=self._a_bytes(scenario.batch_size * causal_pairs * mamba.num_heads),
                ),
                working_set_bytes=self._a_bytes(
                    scenario.batch_size * max(length * length for length in chunk_lengths) * mamba.num_heads
                ),
            ),
            StageWork(
                layer_id,
                "mamba",
                "mamba_chunk_intra_y",
                "matrix",
                macs=scenario.batch_size * causal_pairs * mamba.num_heads * mamba.head_dim,
                elementwise_ops=scenario.batch_size * causal_pairs * mamba.num_heads,
                traffic=Traffic(
                    on_chip_read_bytes=self._a_bytes(
                        inner_elements + scenario.batch_size * causal_pairs * mamba.num_heads
                    ),
                    on_chip_write_bytes=self._a_bytes(inner_elements),
                ),
                working_set_bytes=self._a_bytes(inner_elements),
            ),
            StageWork(
                layer_id,
                "mamba",
                "mamba_chunk_state_build",
                "state",
                macs=tokens * mamba.state_elements,
                elementwise_ops=tokens * mamba.state_elements,
                traffic=Traffic(
                    on_chip_read_bytes=self._a_bytes(tokens * (mamba.d_inner + mamba.groups * mamba.state_dim))
                ),
                working_set_bytes=self._s_bytes(chunk_states),
            ),
            StageWork(
                layer_id,
                "mamba",
                "mamba_chunk_scan_compose",
                "state",
                macs=compositions,
                elementwise_ops=compositions,
                scan_compositions=compositions,
                traffic=Traffic(
                    state_read_bytes=self._s_bytes(state_elements) if scenario.reads_initial_state else 0,
                    state_write_bytes=self._s_bytes(state_elements),
                ),
                working_set_bytes=self._s_bytes(chunk_states),
            ),
            StageWork(
                layer_id,
                "mamba",
                "mamba_chunk_state_output",
                "state",
                macs=tokens * (mamba.state_elements + mamba.d_inner),
                elementwise_ops=2 * inner_elements,
                traffic=Traffic(
                    on_chip_read_bytes=self._a_bytes(tokens * (mamba.groups * mamba.state_dim + mamba.d_inner)),
                    on_chip_write_bytes=self._a_bytes(inner_elements),
                ),
                working_set_bytes=self._s_bytes(chunk_states),
            ),
        ]

    def _attention(self, layer_id: int, scenario: WorkloadScenario) -> list[StageWork]:
        tokens = scenario.tokens
        q_dim = self.arch.num_heads * self.arch.head_dim
        kv_dim = self.arch.num_kv_heads * self.arch.head_dim
        projection_width = q_dim + 2 * kv_dim
        if scenario.phase == InferencePhase.DECODE:
            score_values = scenario.batch_size * self.arch.num_heads * scenario.context_length
            attention_macs = score_values * self.arch.head_dim * 2
            kv_read_elements = 2 * scenario.batch_size * kv_dim * scenario.context_length
        else:
            causal_pairs = scenario.sequence_length * (scenario.sequence_length + 1) // 2
            score_values = scenario.batch_size * self.arch.num_heads * causal_pairs
            attention_macs = score_values * self.arch.head_dim * 2
            kv_read_elements = 0
        kv_write_elements = 2 * tokens * kv_dim

        return [
            StageWork(
                layer_id,
                "attention",
                "attention_qkv_projection",
                "matrix",
                macs=tokens * self.arch.hidden_size * projection_width,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(
                        self.arch.hidden_size * projection_width,
                        layer_id,
                        "attention_qkv_projection",
                    ),
                    activation_read_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                    on_chip_write_bytes=self._a_bytes(tokens * projection_width),
                    kv_write_bytes=self._a_bytes(kv_write_elements),
                ),
                working_set_bytes=self._a_bytes(tokens * projection_width),
            ),
            StageWork(
                layer_id,
                "attention",
                "attention_qk_softmax_pv",
                "matrix",
                macs=attention_macs,
                elementwise_ops=3 * score_values,
                exp_ops=score_values,
                traffic=Traffic(
                    kv_read_bytes=self._a_bytes(kv_read_elements),
                    on_chip_read_bytes=self._a_bytes(tokens * q_dim),
                    on_chip_write_bytes=self._a_bytes(tokens * q_dim),
                ),
                working_set_bytes=self._a_bytes(tokens * q_dim),
            ),
            StageWork(
                layer_id,
                "attention",
                "attention_out_projection",
                "matrix",
                macs=tokens * q_dim * self.arch.hidden_size,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(
                        q_dim * self.arch.hidden_size,
                        layer_id,
                        "attention_out_projection",
                    ),
                    on_chip_read_bytes=self._a_bytes(tokens * q_dim),
                    activation_write_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                ),
                working_set_bytes=self._a_bytes(tokens * q_dim),
            ),
        ]

    def _moe(self, layer_id: int, scenario: WorkloadScenario) -> list[StageWork]:
        moe = self.arch.moe
        assert moe is not None
        tokens = scenario.tokens
        assignments = tokens * moe.experts_per_token
        unique_experts = scenario.moe_unique_experts or min(moe.num_experts, assignments)
        unique_experts = min(unique_experts, moe.num_experts)
        routed_weight_elements = unique_experts * 2 * self.arch.hidden_size * moe.intermediate_size
        shared_weight_elements = moe.shared_experts * 2 * self.arch.hidden_size * moe.shared_intermediate_size

        return [
            StageWork(
                layer_id,
                "moe",
                "moe_router_topk",
                "vector",
                macs=tokens * self.arch.hidden_size * moe.num_experts,
                elementwise_ops=3 * tokens * moe.num_experts,
                exp_ops=tokens * moe.num_experts,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(
                        self.arch.hidden_size * moe.num_experts,
                        layer_id,
                        "moe_router_topk",
                    ),
                    activation_read_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                    on_chip_write_bytes=self._a_bytes(tokens * moe.num_experts),
                ),
                working_set_bytes=self._a_bytes(tokens * moe.num_experts),
            ),
            StageWork(
                layer_id,
                "moe",
                "moe_routed_experts",
                "matrix",
                macs=assignments * 2 * self.arch.hidden_size * moe.intermediate_size,
                elementwise_ops=assignments * moe.intermediate_size,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(
                        routed_weight_elements,
                        layer_id,
                        "moe_routed_experts",
                    ),
                    activation_read_bytes=self._a_bytes(assignments * self.arch.hidden_size),
                    activation_write_bytes=self._a_bytes(assignments * self.arch.hidden_size),
                ),
                working_set_bytes=self._w_bytes(
                    routed_weight_elements,
                    layer_id,
                    "moe_routed_experts",
                ),
            ),
            StageWork(
                layer_id,
                "moe",
                "moe_shared_expert",
                "matrix",
                macs=tokens * moe.shared_experts * 2 * self.arch.hidden_size * moe.shared_intermediate_size,
                elementwise_ops=tokens * moe.shared_experts * moe.shared_intermediate_size,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(
                        shared_weight_elements,
                        layer_id,
                        "moe_shared_expert",
                    ),
                    activation_read_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                    activation_write_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                ),
                working_set_bytes=self._w_bytes(
                    shared_weight_elements,
                    layer_id,
                    "moe_shared_expert",
                ),
            ),
            StageWork(
                layer_id,
                "moe",
                "moe_combine",
                "vector",
                elementwise_ops=(assignments + tokens) * self.arch.hidden_size,
                traffic=Traffic(
                    activation_read_bytes=self._a_bytes((assignments + tokens) * self.arch.hidden_size),
                    activation_write_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                ),
                working_set_bytes=self._a_bytes((assignments + tokens) * self.arch.hidden_size),
            ),
        ]

    def _mlp(self, layer_id: int, scenario: WorkloadScenario) -> list[StageWork]:
        tokens = scenario.tokens
        weights = 2 * self.arch.hidden_size * self.arch.inter_dim
        return [
            StageWork(
                layer_id,
                "mlp",
                "dense_mlp",
                "matrix",
                macs=tokens * weights,
                elementwise_ops=tokens * self.arch.inter_dim,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(weights, layer_id, "dense_mlp"),
                    activation_read_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                    activation_write_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                ),
                working_set_bytes=self._w_bytes(weights, layer_id, "dense_mlp"),
            )
        ]

    def _lm_head(self, scenario: WorkloadScenario) -> StageWork:
        tokens = scenario.batch_size if scenario.phase == InferencePhase.DECODE else scenario.tokens
        weight_elements = self.arch.hidden_size * (self.arch.vocab_size or 0)
        return StageWork(
            -1,
            "lm_head",
            "lm_head",
            "matrix",
            macs=tokens * weight_elements,
            traffic=Traffic(
                weight_read_bytes=self._w_bytes(weight_elements, -1, "lm_head"),
                activation_read_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                activation_write_bytes=self._a_bytes(tokens * (self.arch.vocab_size or 0)),
            ),
            working_set_bytes=self._w_bytes(weight_elements, -1, "lm_head"),
        )
