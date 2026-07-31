"""Torch reference for the shared-expert MoE branch.

A shared expert is an FFN every token passes through, summed into the routed
output unweighted:

    y = shared(x) + sum_k route_weight_k * routed_expert_k(x)

This module mirrors what ``moe_shared_expert_v0`` emits, **operation by
operation**, including every intermediate BF16 rounding, so the comparison
against the emulator can be exact rather than tolerance-based.

Architectures covered
---------------------
``deepseek_shared_expert_golden``
    DeepSeek-V2/V3, Kimi K2, Llama-4, GLM-4.5. No gate; the shared output is added
    as-is. DeepSeek's ``n_shared_experts > 1`` needs no special handling here --
    see :func:`~compiler.aten.plena.program_moe_shared.fused_shared_intermediate`.

``qwen2_shared_expert_golden``
    Qwen2-MoE. Identical, then scaled by ``sigmoid(x @ w_shared_gate)`` with one
    scalar per token.

Both take a ``project`` callable rather than doing their own matmuls, because the
projection golden depends on hardware parameters (MX quantization, MRAM tile
capacity, K-split add order) that belong to the caller's test configuration.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

Project = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def swiglu_golden(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """BF16 SwiGLU matching ``standard_swiglu_activation_v0``'s emit sequence.

    The emitter builds ``silu(gate) * up`` as::

        sigmoid = reci(add_fp(exp(mul_fp(gate, -1.0)), 1.0))
        gate    = gate * sigmoid
        up      = up * gate

    Each of those is a separate vector op writing BF16 back to VRAM, so each
    rounds. ``torch.nn.functional.silu(gate) * up`` agrees to within a few ulp but
    not bit-exactly, which is the whole difference between this test asserting
    equality and asserting a tolerance.

    The exp clamp mirrors ``S_EXP_FP`` / the vector exp unit, which clamp the
    input to [-88, 88] before exponentiating.
    """
    neg_one = torch.tensor(-1.0, dtype=torch.bfloat16).float()

    sigmoid = _bf16(gate.float())
    sigmoid = _bf16(sigmoid.float() * neg_one)
    sigmoid = _bf16(torch.exp(torch.clamp(sigmoid.float(), -88.0, 88.0)))
    sigmoid = _bf16(sigmoid.float() + 1.0)
    sigmoid = _bf16(torch.reciprocal(sigmoid.float()))

    gated = _bf16(gate.float() * sigmoid.float())
    return _bf16(up.float() * gated.float())


def shared_gate_golden(x: torch.Tensor, gate_weight_row: torch.Tensor) -> torch.Tensor:
    """Qwen2-MoE ``sigmoid(x @ w_gate)``, one scalar per token, as emitted.

    ``moe_shared_gate_v0`` accumulates the dot product with ``V_MUL_VV`` +
    ``V_RED_SUM`` into a BF16 FP register, then evaluates the sigmoid in the scalar
    FP unit. ``V_RED_SUM`` reduces one MLEN-wide tile at a time and accumulates
    across tiles in BF16, so for hidden > MLEN the partial sums round between
    tiles -- but the caller passes ``hidden == MLEN`` shapes in the exact-value
    tests, where the whole dot product is one tile and this reduces to a single
    rounding.

    Returns ``[rows]`` BF16 gate scalars.
    """
    if gate_weight_row.ndim != 2 or gate_weight_row.shape[0] != 1:
        raise ValueError(f"gate_weight_row must be [1, hidden], got {tuple(gate_weight_row.shape)}")

    products = _bf16(x.float() * gate_weight_row.float())
    logits = _bf16(products.float().sum(dim=1))

    # sigmoid = 1 / (1 + exp(-logit)) -- S_SUB_FP, S_EXP_FP, S_ADD_FP, S_RECI_FP.
    neg = _bf16(0.0 - logits.float())
    exp = _bf16(torch.exp(torch.clamp(neg.float(), -88.0, 88.0)))
    denom = _bf16(exp.float() + 1.0)
    return _bf16(torch.reciprocal(denom.float()))


def deepseek_shared_expert_golden(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    *,
    project: Project,
    project_from_vram: Project,
) -> torch.Tensor:
    """Shared-expert output for the ungated architectures.

    ``project`` runs the HBM-weight projection golden for activations that came
    from HBM; ``project_from_vram`` for the post-activation hidden state, which is
    already BF16 in VRAM and must not be MX-quantized a second time.
    """
    gate = project(x, w_gate)
    up = project(x, w_up)
    hidden = swiglu_golden(gate, up)
    return project_from_vram(hidden, w_down)


def qwen2_shared_expert_golden(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
    w_shared_gate: torch.Tensor,
    *,
    project: Project,
    project_from_vram: Project,
) -> torch.Tensor:
    """Qwen2-MoE shared expert: the DeepSeek form scaled by a per-token sigmoid gate.

    Mirrors ``Qwen2MoeSparseMoeBlock``::

        shared = self.shared_expert(hidden_states)
        shared = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared

    The gate reads the *same* ``x`` the expert does -- not the expert's output --
    and is applied after the down projection.
    """
    out = deepseek_shared_expert_golden(x, w_gate, w_up, w_down, project=project, project_from_vram=project_from_vram)
    gate_scalars = shared_gate_golden(x, w_shared_gate)
    return _bf16(out.float() * gate_scalars.float().unsqueeze(1))


def combine_shared_and_routed_golden(routed: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
    """``routed + shared`` in BF16, matching the single ``V_ADD_VV`` pass."""
    return _bf16(routed.float() + shared.float())


__all__ = [
    "combine_shared_and_routed_golden",
    "deepseek_shared_expert_golden",
    "qwen2_shared_expert_golden",
    "shared_gate_golden",
    "swiglu_golden",
]
