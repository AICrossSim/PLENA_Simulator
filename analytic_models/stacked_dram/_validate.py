"""Strict scalar validators shared by the stacked-DRAM configuration classes.

Ported from DeepStack (arXiv:2604.04750), tile-ai/DeepStack@8509061,
``src/deepstack/mosaic/arch/custom_profile.py`` (``_positive_int``,
``_nonnegative_int``, ``_finite_number``, ``_fraction``). Booleans are rejected
wherever a number is expected so that a JSON ``true`` cannot silently become 1.
"""

from __future__ import annotations

import math
from numbers import Integral, Real


def positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{field} must be greater than zero")
    return result


def nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field} must be non-negative")
    return result


def finite_number(value: object, field: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    if result < 0.0 or (result == 0.0 and not allow_zero):
        relation = "non-negative" if allow_zero else "greater than zero"
        raise ValueError(f"{field} must be {relation}")
    return result


def fraction(value: object, field: str, *, allow_zero: bool = True) -> float:
    result = finite_number(value, field, allow_zero=allow_zero)
    if result > 1.0:
        raise ValueError(f"{field} must be in [0, 1]")
    return result
