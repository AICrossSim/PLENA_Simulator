"""Kimi K3 KDA-only workload contract using the official text-model shapes.

This module intentionally stops at the 69 KDA mixer blocks. Kimi K3's 24 MLA
mixers, 92 LatentMoE blocks, first dense FFN, and AttnRes are not silently
approximated as ordinary Transformer layers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .nemotron3_workload import (
    InferencePhase,
    Precision,
    StageWork,
    Traffic,
    WeightPrecisionPolicy,
    WorkloadReport,
    WorkloadScenario,
    storage_bytes,
)


@dataclass(frozen=True)
class KdaShape:
    hidden_size: int
    num_heads: int
    key_dim: int
    value_dim: int
    conv_kernel: int
    chunk_size: int = 16
    gate_lower_bound: float = -5.0

    @classmethod
    def kimi_k3(cls) -> KdaShape:
        return cls(7168, 96, 128, 128, 4)

    @property
    def projection_size(self) -> int:
        return self.num_heads * self.key_dim

    @property
    def state_elements(self) -> int:
        return self.num_heads * self.value_dim * self.key_dim

    @property
    def conv_state_elements(self) -> int:
        return self.num_heads * (2 * self.key_dim + self.value_dim) * self.conv_kernel


def formal_kimi_k3_mxfp4_weight_policy(
    precision_contract: Mapping[str, object],
) -> WeightPrecisionPolicy:
    """Map the pinned Kimi checkpoint exclusions onto workload stage names.

    The official contract names modules, while this model prices stages.  Keep
    that translation explicit so an exclusion cannot silently fall back to the
    checkpoint default.  Routed latent experts are the only large stage left at
    MXFP4 by the pinned exclusion list.
    """

    if precision_contract.get("checkpoint_weight_default") != "mxfp4":
        raise ValueError("Kimi K3 checkpoint default must be MXFP4")
    exclusions = precision_contract.get("checkpoint_weight_exclusions")
    expected = {"self_attn", "shared_experts", "mlp gate/up/down", "lm_head"}
    if not isinstance(exclusions, list) or set(exclusions) != expected:
        raise ValueError("unexpected Kimi K3 checkpoint weight exclusions")
    bf16_stages = (
        # self_attn: KDA and MLA mixer parameters
        "kda_qkv_projection",
        "kda_short_conv",
        "kda_decay_beta_projection",
        "kda_output_gate_projection",
        "kda_output_gate_rmsnorm",
        "kda_out_projection",
        "mla_q_low_rank_projection",
        "mla_kv_latent_projection",
        "mla_compressed_kv_attention",
        "mla_output_gate",
        "mla_out_projection",
        # shared experts and the common latent MoE gate/up/down path
        "latent_moe_router_top16",
        "latent_moe_down_projection_norm",
        "latent_moe_up_projection",
        "latent_moe_shared_experts",
        "dense_situ_ffn",
        "lm_head",
        # Norm and AttnRes parameters remain in model dtype.
        "input_rms_norm",
        "post_attention_rms_norm",
        "final_rms_norm",
        "attn_res_before_mixer",
        "attn_res_before_ffn",
        "attn_res_output",
    )
    return WeightPrecisionPolicy(
        name="kimi_k3_mxfp4_checkpoint_mixed_v1",
        default_precision=Precision.MXFP4,
        global_stage_precisions=tuple((stage, Precision.BF16) for stage in bf16_stages),
        source="pinned Kimi K3 precision_contract checkpoint exclusions",
    )


@dataclass(frozen=True)
class KimiK3Architecture:
    num_layers: int = 93
    hidden_size: int = 7168
    vocab_size: int = 163_840
    kda: KdaShape = field(default_factory=KdaShape.kimi_k3)
    attn_res_block_size: int = 12
    num_experts: int = 896
    experts_per_token: int = 16
    shared_experts: int = 2
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    routed_expert_hidden_size: int = 3584
    moe_intermediate_size: int = 3072
    dense_intermediate_size: int = 33_792

    def __post_init__(self) -> None:
        if self.num_layers != 93:
            raise ValueError("the pinned Kimi K3 contract expects 93 text layers")
        if self.hidden_size != self.kda.hidden_size:
            raise ValueError("KDA and text hidden sizes must match")

    @property
    def kda_layer_numbers(self) -> tuple[int, ...]:
        # Official configuration uses one-based layer numbers.
        return tuple(layer for layer in range(1, 93) if layer % 4 != 0)

    @property
    def mla_layer_numbers(self) -> tuple[int, ...]:
        return (*range(4, 93, 4), 93)

    @property
    def dense_ffn_layer_numbers(self) -> tuple[int, ...]:
        return (1,)

    @property
    def moe_layer_numbers(self) -> tuple[int, ...]:
        return tuple(range(2, self.num_layers + 1))

    @property
    def kda_projection_width(self) -> int:
        # q, k, v are separate full-rank projections with equal 12,288 width.
        return 3 * self.kda.projection_size

    @property
    def mla_q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def mla_projection_size(self) -> int:
        return self.kda.num_heads * self.v_head_dim

    @property
    def mla_cache_elements_per_token(self) -> int:
        # Kimi's MLA keeps the normalized 512-wide latent plus the shared
        # 64-wide positional key, not 96 expanded K/V heads.
        return self.kv_lora_rank + self.qk_rope_head_dim

    def recurrent_state_bytes(self, precision: Precision = Precision.FP32, *, batch_size: int = 1) -> int:
        return batch_size * len(self.kda_layer_numbers) * storage_bytes(self.kda.state_elements, precision)

    def conv_state_bytes(self, precision: Precision = Precision.BF16, *, batch_size: int = 1) -> int:
        return batch_size * len(self.kda_layer_numbers) * storage_bytes(self.kda.conv_state_elements, precision)


class KimiK3KdaWorkloadModel:
    """Count architecture-independent work and traffic for all KDA mixers."""

    def __init__(
        self,
        arch: KimiK3Architecture | None = None,
        *,
        activation_precision: Precision = Precision.BF16,
        weight_precision: Precision = Precision.BF16,
        state_precision: Precision = Precision.FP32,
        conv_state_precision: Precision = Precision.BF16,
        weight_precision_policy: WeightPrecisionPolicy | None = None,
    ) -> None:
        self.arch = arch or KimiK3Architecture()
        self.activation_precision = activation_precision
        self.weight_precision_policy = weight_precision_policy
        self.weight_precision = (
            weight_precision_policy.default_precision if weight_precision_policy is not None else weight_precision
        )
        self.state_precision = state_precision
        self.conv_state_precision = conv_state_precision

    def _a_bytes(self, elements: int) -> int:
        return storage_bytes(elements, self.activation_precision)

    def _w_bytes(self, elements: int, layer_id: int, stage_name: str) -> int:
        precision = self.weight_precision
        if self.weight_precision_policy is not None:
            precision = self.weight_precision_policy.precision_for(layer_id, stage_name)
        return storage_bytes(elements, precision)

    def _s_bytes(self, elements: int) -> int:
        return storage_bytes(elements, self.state_precision)

    def _cs_bytes(self, elements: int) -> int:
        return storage_bytes(elements, self.conv_state_precision)

    def build(self, scenario: WorkloadScenario) -> WorkloadReport:
        stages = []
        for layer_number in self.arch.kda_layer_numbers:
            stages.extend(self._kda_layer(layer_number - 1, scenario))
        return WorkloadReport(
            scenario=scenario,
            activation_precision=self.activation_precision,
            weight_precision=self.weight_precision,
            state_precision=self.state_precision,
            stages=tuple(stages),
            weight_precision_policy=self.weight_precision_policy,
        )

    def _kda_layer(self, layer_id: int, scenario: WorkloadScenario) -> list[StageWork]:
        arch = self.arch
        kda = arch.kda
        tokens = scenario.tokens
        projection = kda.projection_size
        recurrent_state_elements = scenario.batch_size * kda.state_elements
        conv_state_elements = scenario.batch_size * kda.conv_state_elements
        qkv_elements = tokens * 3 * projection
        output_elements = tokens * projection

        qkv_weights = 3 * arch.hidden_size * projection
        conv_weights = 3 * projection * kda.conv_kernel
        decay_beta_weights = (
            arch.hidden_size * kda.key_dim + kda.key_dim * projection + arch.hidden_size * kda.num_heads
        )
        output_gate_weights = arch.hidden_size * projection
        output_projection_weights = projection * arch.hidden_size

        return [
            StageWork(
                layer_id,
                "kda",
                "kda_qkv_projection",
                "matrix",
                macs=tokens * qkv_weights,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(qkv_weights, layer_id, "kda_qkv_projection"),
                    activation_read_bytes=self._a_bytes(tokens * arch.hidden_size),
                    on_chip_write_bytes=self._a_bytes(qkv_elements),
                ),
                working_set_bytes=self._a_bytes(qkv_elements),
            ),
            StageWork(
                layer_id,
                "kda",
                "kda_short_conv",
                "conv",
                macs=tokens * conv_weights,
                elementwise_ops=2 * qkv_elements,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(conv_weights, layer_id, "kda_short_conv"),
                    state_read_bytes=self._cs_bytes(conv_state_elements) if scenario.reads_initial_state else 0,
                    state_write_bytes=self._cs_bytes(conv_state_elements),
                    on_chip_read_bytes=self._a_bytes(qkv_elements),
                    on_chip_write_bytes=self._a_bytes(qkv_elements),
                ),
                working_set_bytes=self._cs_bytes(conv_state_elements),
            ),
            StageWork(
                layer_id,
                "kda",
                "kda_decay_beta_projection",
                "matrix",
                macs=tokens * decay_beta_weights,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(decay_beta_weights, layer_id, "kda_decay_beta_projection"),
                    activation_read_bytes=self._a_bytes(tokens * arch.hidden_size),
                    on_chip_write_bytes=self._a_bytes(tokens * (projection + kda.num_heads)),
                ),
                working_set_bytes=self._a_bytes(tokens * (projection + kda.num_heads)),
            ),
            StageWork(
                layer_id,
                "kda",
                "kda_qk_l2norm",
                "vector",
                elementwise_ops=8 * tokens * kda.num_heads * kda.key_dim,
                traffic=Traffic(
                    on_chip_read_bytes=self._a_bytes(2 * tokens * projection),
                    on_chip_write_bytes=self._a_bytes(2 * tokens * projection),
                ),
                working_set_bytes=self._a_bytes(2 * tokens * projection),
            ),
            StageWork(
                layer_id,
                "kda",
                "kda_state_decay_prediction",
                "state",
                macs=tokens * kda.state_elements,
                elementwise_ops=tokens * kda.state_elements,
                exp_ops=tokens * kda.num_heads * kda.key_dim,
                traffic=Traffic(
                    state_read_bytes=self._s_bytes(recurrent_state_elements) if scenario.reads_initial_state else 0,
                    on_chip_read_bytes=self._a_bytes(tokens * (2 * projection + kda.num_heads)),
                ),
                working_set_bytes=self._s_bytes(recurrent_state_elements),
            ),
            StageWork(
                layer_id,
                "kda",
                "kda_delta_update_output",
                "state",
                macs=2 * tokens * kda.state_elements,
                elementwise_ops=3 * tokens * kda.num_heads * kda.value_dim,
                exp_ops=tokens * kda.num_heads,
                traffic=Traffic(
                    state_write_bytes=self._s_bytes(recurrent_state_elements),
                    on_chip_read_bytes=self._a_bytes(tokens * (projection + kda.num_heads * kda.value_dim)),
                    on_chip_write_bytes=self._a_bytes(output_elements),
                ),
                working_set_bytes=self._s_bytes(recurrent_state_elements),
            ),
            StageWork(
                layer_id,
                "kda",
                "kda_output_gate_projection",
                "matrix",
                macs=tokens * output_gate_weights,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(output_gate_weights, layer_id, "kda_output_gate_projection"),
                    activation_read_bytes=self._a_bytes(tokens * arch.hidden_size),
                    on_chip_write_bytes=self._a_bytes(output_elements),
                ),
                working_set_bytes=self._a_bytes(output_elements),
            ),
            StageWork(
                layer_id,
                "kda",
                "kda_output_gate_rmsnorm",
                "vector",
                elementwise_ops=8 * output_elements,
                traffic=Traffic(
                    on_chip_read_bytes=self._a_bytes(2 * output_elements),
                    on_chip_write_bytes=self._a_bytes(output_elements),
                ),
                working_set_bytes=self._a_bytes(2 * output_elements),
            ),
            StageWork(
                layer_id,
                "kda",
                "kda_out_projection",
                "matrix",
                macs=tokens * output_projection_weights,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(output_projection_weights, layer_id, "kda_out_projection"),
                    on_chip_read_bytes=self._a_bytes(output_elements),
                    activation_write_bytes=self._a_bytes(tokens * arch.hidden_size),
                ),
                working_set_bytes=self._a_bytes(output_elements),
            ),
        ]


class KimiK3HybridWorkloadModel(KimiK3KdaWorkloadModel):
    """Logical work/traffic for the complete 93-layer Kimi K3 text tower.

    This is a workload contract, not a calibrated PLENA latency model. MLA is
    counted with Kimi's compressed latent KV cache and phase-specific prefill
    versus decode dataflow. LatentMoE counts the shared down/up projections
    once per token and only the selected experts inside the 3,584-wide latent.
    """

    def build(self, scenario: WorkloadScenario) -> WorkloadReport:
        stages: list[StageWork] = []
        if scenario.include_embedding:
            stages.append(self._embedding(scenario))

        captured_blocks = 0
        kda_layers = set(self.arch.kda_layer_numbers)
        for layer_number in range(1, self.arch.num_layers + 1):
            layer_id = layer_number - 1
            mixer = "kda" if layer_number in kda_layers else "mla"
            if captured_blocks:
                stages.append(self._attn_res(layer_id, "before_mixer", captured_blocks, scenario))
            if layer_id % self.arch.attn_res_block_size == 0:
                captured_blocks += 1
                stages.append(self._capture_prefix(layer_id, captured_blocks, scenario))

            stages.append(self._rms_norm(layer_id, mixer, "input_rms_norm", scenario))
            if mixer == "kda":
                stages.extend(self._kda_layer(layer_id, scenario))
            else:
                stages.extend(self._mla_layer(layer_id, scenario))
            stages.append(self._prefix_sum(layer_id, mixer, "prefix_sum_after_mixer", scenario))

            stages.append(self._attn_res(layer_id, "before_ffn", captured_blocks, scenario))
            ffn = "dense" if layer_id == 0 else "latent_moe"
            stages.append(self._rms_norm(layer_id, ffn, "post_attention_rms_norm", scenario))
            if layer_id == 0:
                stages.append(self._dense_ffn(layer_id, scenario))
            else:
                stages.extend(self._latent_moe(layer_id, scenario))
            stages.append(self._prefix_sum(layer_id, ffn, "prefix_sum_after_ffn", scenario))

        stages.append(self._attn_res(-1, "output", captured_blocks, scenario))
        stages.append(self._rms_norm(-1, "output", "final_rms_norm", scenario))
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

    def _embedding(self, scenario: WorkloadScenario) -> StageWork:
        elements = scenario.tokens * self.arch.hidden_size
        return StageWork(
            -1,
            "embedding",
            "embedding_lookup",
            "dma",
            traffic=Traffic(
                weight_read_bytes=self._w_bytes(elements, -1, "embedding_lookup"),
                activation_write_bytes=self._a_bytes(elements),
            ),
            working_set_bytes=self._a_bytes(elements),
        )

    def _rms_norm(
        self,
        layer_id: int,
        layer_type: str,
        name: str,
        scenario: WorkloadScenario,
    ) -> StageWork:
        elements = scenario.tokens * self.arch.hidden_size
        return StageWork(
            layer_id,
            layer_type,
            name,
            "vector",
            elementwise_ops=5 * elements,
            traffic=Traffic(
                weight_read_bytes=self._w_bytes(self.arch.hidden_size, layer_id, name),
                on_chip_read_bytes=self._a_bytes(elements),
                on_chip_write_bytes=self._a_bytes(elements),
            ),
            working_set_bytes=self._a_bytes(elements),
        )

    def _capture_prefix(
        self,
        layer_id: int,
        captured_blocks: int,
        scenario: WorkloadScenario,
    ) -> StageWork:
        elements = scenario.tokens * self.arch.hidden_size
        return StageWork(
            layer_id,
            "attn_res",
            "attn_res_capture_prefix",
            "vector",
            elementwise_ops=elements,
            traffic=Traffic(
                on_chip_read_bytes=self._a_bytes(elements),
                on_chip_write_bytes=self._a_bytes(elements),
            ),
            working_set_bytes=self._a_bytes(captured_blocks * elements),
        )

    def _attn_res(
        self,
        layer_id: int,
        suffix: str,
        captured_blocks: int,
        scenario: WorkloadScenario,
    ) -> StageWork:
        elements = scenario.tokens * self.arch.hidden_size
        candidates = captured_blocks + 1  # block residual plus saved prefixes
        return StageWork(
            layer_id,
            "attn_res",
            f"attn_res_{suffix}",
            "matrix_vector",
            macs=2 * candidates * elements,
            elementwise_ops=6 * candidates * elements,
            exp_ops=scenario.tokens * candidates,
            traffic=Traffic(
                weight_read_bytes=self._w_bytes(2 * self.arch.hidden_size, layer_id, f"attn_res_{suffix}"),
                on_chip_read_bytes=self._a_bytes(candidates * elements),
                on_chip_write_bytes=self._a_bytes(elements),
            ),
            working_set_bytes=self._a_bytes(candidates * elements),
        )

    def _prefix_sum(
        self,
        layer_id: int,
        layer_type: str,
        name: str,
        scenario: WorkloadScenario,
    ) -> StageWork:
        elements = scenario.tokens * self.arch.hidden_size
        return StageWork(
            layer_id,
            layer_type,
            name,
            "vector",
            elementwise_ops=elements,
            traffic=Traffic(
                on_chip_read_bytes=2 * self._a_bytes(elements),
                on_chip_write_bytes=self._a_bytes(elements),
            ),
            working_set_bytes=2 * self._a_bytes(elements),
        )

    def _mla_layer(self, layer_id: int, scenario: WorkloadScenario) -> list[StageWork]:
        arch = self.arch
        tokens = scenario.tokens
        heads = arch.kda.num_heads
        q_width = heads * arch.mla_q_head_dim
        projection = arch.mla_projection_size
        q_weights = arch.hidden_size * arch.q_lora_rank + arch.q_lora_rank * q_width
        kv_weights = arch.hidden_size * arch.mla_cache_elements_per_token
        transform_weights = heads * arch.kv_lora_rank * (arch.qk_nope_head_dim + arch.v_head_dim)
        gate_weights = arch.hidden_size * projection
        output_weights = projection * arch.hidden_size
        if scenario.phase == InferencePhase.DECODE:
            pairs = scenario.batch_size * scenario.context_length
            attention_macs = pairs * heads * (2 * arch.kv_lora_rank + arch.qk_rope_head_dim)
            transform_macs = tokens * heads * arch.kv_lora_rank * (arch.qk_nope_head_dim + arch.v_head_dim)
            cache_read_elements = scenario.batch_size * scenario.context_length * arch.mla_cache_elements_per_token
        else:
            pairs = scenario.batch_size * scenario.sequence_length * (scenario.sequence_length + 1) // 2
            attention_macs = pairs * heads * (arch.mla_q_head_dim + arch.v_head_dim)
            transform_macs = tokens * transform_weights
            cache_read_elements = 0
        cache_write_elements = tokens * arch.mla_cache_elements_per_token
        return [
            StageWork(
                layer_id,
                "mla",
                "mla_q_low_rank_projection",
                "matrix_vector",
                macs=tokens * q_weights,
                elementwise_ops=5 * tokens * arch.q_lora_rank,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(q_weights, layer_id, "mla_q_low_rank_projection"),
                    on_chip_read_bytes=self._a_bytes(tokens * arch.hidden_size),
                    on_chip_write_bytes=self._a_bytes(tokens * q_width),
                ),
                working_set_bytes=self._a_bytes(tokens * q_width),
            ),
            StageWork(
                layer_id,
                "mla",
                "mla_kv_latent_projection",
                "matrix_vector",
                macs=tokens * kv_weights,
                elementwise_ops=5 * tokens * arch.kv_lora_rank,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(kv_weights, layer_id, "mla_kv_latent_projection"),
                    on_chip_read_bytes=self._a_bytes(tokens * arch.hidden_size),
                    kv_write_bytes=self._a_bytes(cache_write_elements),
                ),
                working_set_bytes=self._a_bytes(cache_write_elements),
            ),
            StageWork(
                layer_id,
                "mla",
                "mla_compressed_kv_attention",
                "matrix_vector",
                macs=attention_macs + transform_macs,
                exp_ops=pairs * heads,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(transform_weights, layer_id, "mla_compressed_kv_attention"),
                    kv_read_bytes=self._a_bytes(cache_read_elements),
                    on_chip_read_bytes=self._a_bytes(tokens * q_width),
                    on_chip_write_bytes=self._a_bytes(tokens * projection),
                ),
                working_set_bytes=self._a_bytes(cache_read_elements + tokens * q_width),
            ),
            StageWork(
                layer_id,
                "mla",
                "mla_output_gate",
                "matrix_vector",
                macs=tokens * gate_weights,
                elementwise_ops=2 * tokens * projection,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(gate_weights, layer_id, "mla_output_gate"),
                    on_chip_read_bytes=self._a_bytes(tokens * (arch.hidden_size + projection)),
                    on_chip_write_bytes=self._a_bytes(tokens * projection),
                ),
                working_set_bytes=self._a_bytes(tokens * projection),
            ),
            StageWork(
                layer_id,
                "mla",
                "mla_out_projection",
                "matrix",
                macs=tokens * output_weights,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(output_weights, layer_id, "mla_out_projection"),
                    on_chip_read_bytes=self._a_bytes(tokens * projection),
                    on_chip_write_bytes=self._a_bytes(tokens * arch.hidden_size),
                ),
                working_set_bytes=self._a_bytes(tokens * projection),
            ),
        ]

    def _dense_ffn(self, layer_id: int, scenario: WorkloadScenario) -> StageWork:
        tokens = scenario.tokens
        weights = 3 * self.arch.hidden_size * self.arch.dense_intermediate_size
        return StageWork(
            layer_id,
            "dense",
            "dense_situ_ffn",
            "matrix_vector",
            macs=tokens * weights,
            elementwise_ops=8 * tokens * self.arch.dense_intermediate_size,
            traffic=Traffic(
                weight_read_bytes=self._w_bytes(weights, layer_id, "dense_situ_ffn"),
                on_chip_read_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                on_chip_write_bytes=self._a_bytes(tokens * self.arch.hidden_size),
            ),
            working_set_bytes=self._w_bytes(weights, layer_id, "dense_situ_ffn"),
        )

    def _latent_moe(self, layer_id: int, scenario: WorkloadScenario) -> list[StageWork]:
        arch = self.arch
        tokens = scenario.tokens
        assignments = tokens * arch.experts_per_token
        unique_experts = scenario.moe_unique_experts or min(arch.num_experts, assignments)
        unique_experts = min(unique_experts, arch.num_experts)
        routed_weight_elements = unique_experts * 3 * arch.routed_expert_hidden_size * arch.moe_intermediate_size
        shared_intermediate = arch.shared_experts * arch.moe_intermediate_size
        shared_weight_elements = 3 * arch.hidden_size * shared_intermediate
        latent_elements = tokens * arch.routed_expert_hidden_size
        return [
            StageWork(
                layer_id,
                "latent_moe",
                "latent_moe_router_top16",
                "vector",
                macs=tokens * arch.hidden_size * arch.num_experts,
                elementwise_ops=3 * tokens * arch.num_experts,
                exp_ops=tokens * arch.num_experts,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(
                        arch.hidden_size * arch.num_experts,
                        layer_id,
                        "latent_moe_router_top16",
                    ),
                    on_chip_read_bytes=self._a_bytes(tokens * arch.hidden_size),
                    on_chip_write_bytes=self._a_bytes(tokens * arch.num_experts),
                ),
                working_set_bytes=self._a_bytes(tokens * arch.num_experts),
            ),
            StageWork(
                layer_id,
                "latent_moe",
                "latent_moe_down_projection_norm",
                "matrix_vector",
                macs=tokens * arch.hidden_size * arch.routed_expert_hidden_size,
                elementwise_ops=5 * latent_elements,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(
                        arch.hidden_size * arch.routed_expert_hidden_size,
                        layer_id,
                        "latent_moe_down_projection_norm",
                    ),
                    on_chip_read_bytes=self._a_bytes(tokens * arch.hidden_size),
                    on_chip_write_bytes=self._a_bytes(latent_elements),
                ),
                working_set_bytes=self._a_bytes(latent_elements),
            ),
            StageWork(
                layer_id,
                "latent_moe",
                "latent_moe_routed_experts",
                "matrix_vector",
                macs=(assignments * 3 * arch.routed_expert_hidden_size * arch.moe_intermediate_size),
                elementwise_ops=8 * assignments * arch.moe_intermediate_size,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(routed_weight_elements, layer_id, "latent_moe_routed_experts"),
                    on_chip_read_bytes=self._a_bytes(assignments * arch.routed_expert_hidden_size),
                    on_chip_write_bytes=self._a_bytes(latent_elements),
                ),
                working_set_bytes=self._w_bytes(routed_weight_elements, layer_id, "latent_moe_routed_experts"),
            ),
            StageWork(
                layer_id,
                "latent_moe",
                "latent_moe_up_projection",
                "matrix",
                macs=tokens * arch.routed_expert_hidden_size * arch.hidden_size,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(
                        arch.routed_expert_hidden_size * arch.hidden_size,
                        layer_id,
                        "latent_moe_up_projection",
                    ),
                    on_chip_read_bytes=self._a_bytes(latent_elements),
                    on_chip_write_bytes=self._a_bytes(tokens * arch.hidden_size),
                ),
                working_set_bytes=self._a_bytes(latent_elements),
            ),
            StageWork(
                layer_id,
                "latent_moe",
                "latent_moe_shared_experts",
                "matrix_vector",
                macs=tokens * shared_weight_elements,
                elementwise_ops=8 * tokens * shared_intermediate,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(shared_weight_elements, layer_id, "latent_moe_shared_experts"),
                    on_chip_read_bytes=self._a_bytes(tokens * arch.hidden_size),
                    on_chip_write_bytes=self._a_bytes(tokens * arch.hidden_size),
                ),
                working_set_bytes=self._w_bytes(shared_weight_elements, layer_id, "latent_moe_shared_experts"),
            ),
            StageWork(
                layer_id,
                "latent_moe",
                "latent_moe_combine",
                "vector",
                elementwise_ops=(arch.experts_per_token + arch.shared_experts) * tokens * arch.hidden_size,
                traffic=Traffic(
                    on_chip_read_bytes=self._a_bytes(
                        (arch.experts_per_token + arch.shared_experts) * tokens * arch.hidden_size
                    ),
                    on_chip_write_bytes=self._a_bytes(tokens * arch.hidden_size),
                ),
                working_set_bytes=self._a_bytes(
                    (arch.experts_per_token + arch.shared_experts) * tokens * arch.hidden_size
                ),
            ),
        ]

    def _lm_head(self, scenario: WorkloadScenario) -> StageWork:
        tokens = scenario.batch_size if scenario.phase == InferencePhase.DECODE else scenario.tokens
        weights = self.arch.hidden_size * self.arch.vocab_size
        return StageWork(
            -1,
            "lm_head",
            "lm_head",
            "matrix",
            macs=tokens * weights,
            traffic=Traffic(
                weight_read_bytes=self._w_bytes(weights, -1, "lm_head"),
                on_chip_read_bytes=self._a_bytes(tokens * self.arch.hidden_size),
                activation_write_bytes=self._a_bytes(tokens * self.arch.vocab_size),
            ),
            working_set_bytes=self._w_bytes(weights, -1, "lm_head"),
        )


def default_kimi_k3_scenario(
    phase: InferencePhase = InferencePhase.DECODE,
    *,
    batch_size: int = 1,
    sequence_length: int | None = None,
    context_length: int = 2048,
) -> WorkloadScenario:
    if sequence_length is None:
        sequence_length = 1 if phase == InferencePhase.DECODE else 2048
    return WorkloadScenario(
        phase=phase,
        batch_size=batch_size,
        sequence_length=sequence_length,
        context_length=context_length,
        include_embedding=False,
        include_lm_head=False,
    )
