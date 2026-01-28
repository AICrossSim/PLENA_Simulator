def ceil_div(m: int, n: int) -> int:
    assert isinstance(m, int) and isinstance(n, int), f"m={m}, n={n}"
    assert m >= 0 and n > 0, f"m={m}, n={n}"
    return (m + n - 1) // n
