import numpy as np
import torch


def _tuple_is_equal(t1, t2):
    assert len(t1) == len(t2)
    is_equal = True
    for elem1, elem2 in zip(t1, t2):
        if elem1 == 0 or elem2 == 0:
            continue
        if elem1 != elem2:
            is_equal = False
            break
    return is_equal


def check_shape(x: torch.Tensor, shape: tuple[int]):
    assert x.ndim == len(shape), f"Expected {len(shape)} dimensions, got {x.ndim}"
    assert _tuple_is_equal(x.size(), shape), f"Expected shape {shape}, got {x.size()}"
