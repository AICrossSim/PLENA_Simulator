"""Instruction latency expression loading and evaluation."""

from __future__ import annotations

import ast
import json
import math
import operator
from collections.abc import Mapping
from typing import Any


_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
}
_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_POWER_EXPONENT = 8


def build_latency_context(hardware_config: Any) -> dict[str, int | float]:
    """Build the variable context used by custom ISA latency expressions."""
    context = dict(hardware_config.model_dump())
    context["SA_ACC_CYCLES"] = int(math.log2(hardware_config.MLEN / hardware_config.BLEN) + 1)
    return context


def evaluate_latency_expression(expression: str, context: Mapping[str, int | float]) -> int:
    """Evaluate a restricted arithmetic latency expression."""
    expression = expression.strip()
    try:
        tree = ast.parse(expression, mode="eval")
        value = _evaluate_node(tree, context)
    except (SyntaxError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(f"Unsupported latency expression: {expression}") from exc

    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"Latency expression did not evaluate to an integer: {expression}")
        return int(value)
    return int(value)


def build_pipelined_latency_map(hardware_config: Any, custom_isa_path: str) -> dict[str, int]:
    """Load customISA_lib.json and evaluate pipelined latency expressions."""
    with open(custom_isa_path) as f:
        custom_isa_lib = json.load(f)

    context = build_latency_context(hardware_config)
    latencies = {}
    for instr_name, instr_data in custom_isa_lib.items():
        if "pipelined" not in instr_data:
            raise ValueError(f"Instruction '{instr_name}' missing 'pipelined' field.")
        latencies[instr_name] = evaluate_latency_expression(instr_data["pipelined"], context)
    return latencies


def _evaluate_node(node: ast.AST, context: Mapping[str, int | float]) -> int | float:
    if isinstance(node, ast.Expression):
        return _evaluate_node(node.body, context)

    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in context:
            raise ValueError(f"Unsupported latency expression variable: {node.id}")
        return context[node.id]

    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported latency expression unary operator: {type(node.op).__name__}")
        return op(_evaluate_node(node.operand, context))

    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Pow):
            left = _evaluate_node(node.left, context)
            right = _evaluate_node(node.right, context)
            if not isinstance(right, int | float) or int(right) != right:
                raise ValueError("Power exponent must be an integer")
            exponent = int(right)
            if exponent < 0 or exponent > _MAX_POWER_EXPONENT:
                raise ValueError(f"Power exponent must be between 0 and {_MAX_POWER_EXPONENT}")
            return operator.pow(left, exponent)

        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported latency expression operator: {type(node.op).__name__}")
        return op(_evaluate_node(node.left, context), _evaluate_node(node.right, context))

    raise ValueError(f"Unsupported latency expression node: {type(node).__name__}")
