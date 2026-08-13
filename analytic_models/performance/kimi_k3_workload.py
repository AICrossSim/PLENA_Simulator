"""Kimi K3 KDA-only workload contract using the official text-model shapes.

This module intentionally stops at the 69 KDA mixer blocks. Kimi K3's 24 MLA
mixers, 92 LatentMoE blocks, first dense FFN, and AttnRes are not silently
approximated as ordinary Transformer layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from analytic_models.reference.kimi_k3_kda import KdaShape

from .nemotron3_workload import (
    InferencePhase,
    Precision,
    StageWork,
    Traffic,
    WorkloadReport,
    WorkloadScenario,
    storage_bytes,
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

    def recurrent_state_bytes(self, precision: Precision = Precision.BF16, *, batch_size: int = 1) -> int:
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
        state_precision: Precision = Precision.BF16,
    ) -> None:
        self.arch = arch or KimiK3Architecture()
        self.activation_precision = activation_precision
        self.weight_precision = weight_precision
        self.state_precision = state_precision

    def _a_bytes(self, elements: int) -> int:
        return storage_bytes(elements, self.activation_precision)

    def _w_bytes(self, elements: int) -> int:
        return storage_bytes(elements, self.weight_precision)

    def _s_bytes(self, elements: int) -> int:
        return storage_bytes(elements, self.state_precision)

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
            arch.hidden_size * kda.key_dim
            + kda.key_dim * projection
            + arch.hidden_size * kda.num_heads
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
                    weight_read_bytes=self._w_bytes(qkv_weights),
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
                    weight_read_bytes=self._w_bytes(conv_weights),
                    state_read_bytes=self._s_bytes(conv_state_elements) if scenario.reads_initial_state else 0,
                    state_write_bytes=self._s_bytes(conv_state_elements),
                    on_chip_read_bytes=self._a_bytes(qkv_elements),
                    on_chip_write_bytes=self._a_bytes(qkv_elements),
                ),
                working_set_bytes=self._s_bytes(conv_state_elements),
            ),
            StageWork(
                layer_id,
                "kda",
                "kda_decay_beta_projection",
                "matrix",
                macs=tokens * decay_beta_weights,
                traffic=Traffic(
                    weight_read_bytes=self._w_bytes(decay_beta_weights),
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
                    weight_read_bytes=self._w_bytes(output_gate_weights),
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
                    weight_read_bytes=self._w_bytes(output_projection_weights),
                    on_chip_read_bytes=self._a_bytes(output_elements),
                    activation_write_bytes=self._a_bytes(tokens * arch.hidden_size),
                ),
                working_set_bytes=self._a_bytes(output_elements),
            ),
        ]


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
