"""Inner products as computed on the platform that produced the reference results (numpy 2.2.6 with OpenBLAS 0.3.29
on AVX-512): the order of additions and fused multiply-adds of the ddot kernels, evaluated exactly in Python so that
the reference sums can be reproduced on any machine."""

import numpy as np


def _fma(a: float, b: float, c: float) -> float:
    na, da = a.as_integer_ratio()
    nb, db = b.as_integer_ratio()
    nc, dc = c.as_integer_ratio()
    return (na * nb * dc + nc * da * db) / (da * db * dc)


def _unit_stride(x: list, y: list) -> float:
    n, dot = len(x), 0.0
    n16 = n & -16
    if n16:
        n32 = n16 & ~31
        acc = [[0.0] * 8 for _ in range(4)]
        for i in range(0, n32, 32):
            for a in range(4):
                for lane in range(8):
                    k = i + 8 * a + lane
                    acc[a][lane] = _fma(x[k], y[k], acc[a][lane])
        acc4 = [[acc[a][lane] + acc[a][lane + 4] for lane in range(4)] for a in range(4)]
        for i in range(n32, n16, 16):
            for a in range(4):
                for lane in range(4):
                    k = i + 4 * a + lane
                    acc4[a][lane] = _fma(x[k], y[k], acc4[a][lane])
        s = [((acc4[0][lane] + acc4[1][lane]) + acc4[2][lane]) + acc4[3][lane] for lane in range(4)]
        dot = (s[0] + s[2]) + (s[1] + s[3])
    for i in range(n16, n):
        dot = _fma(y[i], x[i], dot)
    return dot


def _strided(x: list, y: list) -> float:
    n, t1, t2, i = len(x), 0.0, 0.0, 0
    while i < n & -4:
        t1 += _fma(y[i], x[i], y[i + 2] * x[i + 2])
        t2 += _fma(y[i + 1], x[i + 1], y[i + 3] * x[i + 3])
        i += 4
    while i < n:
        t1 = _fma(y[i], x[i], t1)
        i += 1
    return t1 + t2


def dot(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    unit = a.strides[0] == a.itemsize and b.strides[0] == b.itemsize
    return (_unit_stride if unit else _strided)([float(v) for v in a], [float(v) for v in b])
