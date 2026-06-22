"""Resource-utilisation area proxy for PLENA DSE.

This model is a lightweight analytic proxy migrated from the old PLENA
co-design flow. It is intended for ranking and filtering design points before
running RTL synthesis, not as a replacement for signoff area.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

DEFAULT_UNIT_INFO = Path(__file__).with_name("area_units.json")


class AreaFormulaError(ValueError):
    """Raised when an area proxy formula is invalid or unsafe."""


def _eval_arithmetic(expr: str, variables: dict[str, float | int]) -> float:
    tree = ast.parse(expr, mode="eval")

    def visit(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise AreaFormulaError(f"unknown variable in area formula: {node.id}")
            return float(variables[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
        raise AreaFormulaError(f"unsupported expression in area formula: {ast.dump(node)}")

    return visit(tree)


def load_area_units(path: str | Path = DEFAULT_UNIT_INFO) -> dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


def estimate_area(config: dict[str, Any], unit_info_path: str | Path = DEFAULT_UNIT_INFO) -> dict[str, Any]:
    """Estimate PLENA area proxy from hardware and precision parameters.

    Args:
        config: Flat dict containing hardware and precision scalar parameters.
        unit_info_path: JSON file containing per-unit coefficients and formulas.

    Returns:
        Dict with total proxy area, per-unit breakdown, and the evaluated inputs.
    """
    units = load_area_units(unit_info_path)
    base_config: dict[str, float | int] = {}
    for key, value in config.items():
        if isinstance(value, bool):
            base_config[key] = int(value)
        elif isinstance(value, int | float):
            base_config[key] = value
        else:
            raise TypeError(f"area proxy config value for {key} must be numeric, got {type(value).__name__}")

    breakdown: dict[str, float] = {}
    total = 0.0
    for unit, info in units.items():
        if "Coefficients" not in info or "Relationship" not in info:
            continue
        variables = dict(base_config)
        variables.update(info["Coefficients"])
        value = _eval_arithmetic(str(info["Relationship"]), variables)
        breakdown[unit] = value
        total += value

    return {
        "area": total,
        "area_proxy": total,
        "area_proxy_breakdown": breakdown,
        "area_proxy_inputs": base_config,
        "area_proxy_units": str(Path(unit_info_path)),
    }


__all__ = ["AreaFormulaError", "estimate_area", "load_area_units"]
