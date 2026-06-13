# PLENA Quantization Formats

This document describes the microscaling (MX) quantization formats used in PLENA.

## Overview

PLENA uses block-based quantization where elements within a block share a common scale factor. Two formats are supported:

| Format | Element Type | Scale Type | Use Case |
|--------|--------------|------------|----------|
| MXINT  | Denormalized integer | Biased exponent | Weights, Activations |
| MXFP   | IEEE-like minifloat | Biased exponent | High-precision ops |

---

## MXINT Format

MXINT (Microscaling Integer) uses a **denormalized integer** representation for elements - there is no implicit leading 1, making all values "subnormal" in IEEE terms.

### Element Encoding

```
Element (N bits): [sign (1)] [magnitude (N-1)]
```

- **Sign**: 1 bit (0 = positive, 1 = negative)
- **Magnitude**: (N-1) bits, unsigned integer in range [0, 2^(N-1) - 1]
- **Encoding**: Sign-magnitude (NOT two's complement)

The magnitude represents a **normalized fraction** in [0, 1):
```
normalized_value = magnitude / 2^(N-1)
```

### Scale Encoding

```
Scale (S bits): biased exponent
```

- **Bias**: 2^(S-1) - 1 (e.g., 127 for 8-bit scale)
- **Actual exponent**: scale - bias
- **Range**: [-(2^(S-1) - 1), 2^(S-1)]

### Value Computation

```
value = (-1)^sign × (magnitude / 2^(N-1)) × 2^(scale - bias)
      = (-1)^sign × magnitude × 2^(scale - bias - (N-1))
```

### Example: MXINT8 with 8-bit Scale

| Parameter | Value |
|-----------|-------|
| Element width | 8 bits |
| Sign bits | 1 |
| Magnitude bits | 7 |
| Scale width | 8 bits |
| Scale bias | 127 |

**Example conversion:**
- Element = `0xB5` = `0b10110101`
  - sign = 1 (negative)
  - magnitude = 53
- Scale = `0xB4` = 180
  - actual_exp = 180 - 127 = 53

```
value = -1 × (53 / 128) × 2^53
      = -0.4140625 × 2^53
      ≈ -3.73 × 10^15
```

### Why "All Denorm"?

In IEEE floating-point, normalized numbers have an implicit leading 1:
```
IEEE normalized:   value = 1.mantissa × 2^exp
IEEE denormalized: value = 0.mantissa × 2^exp
```

MXINT always uses the denormalized form (no implicit 1), which:
- Simplifies hardware (no need to handle normalized/denormalized cases)
- Provides uniform precision across the range
- Maps directly to fixed-point arithmetic

---

## MXFP Format

MXFP (Microscaling Floating-Point) uses **IEEE-like minifloat** elements with a shared exponent bias.

### Element Encoding

```
Element (N bits): [sign (1)] [exponent (E)] [mantissa (M)]
```

Where N = 1 + E + M

- Standard IEEE-like encoding with implicit leading 1 for normalized values
- Per-element exponent allows larger dynamic range within a block

### Scale Encoding

```
Scale (S bits): shared exponent bias
```

- **Bias**: 2^(S-1) - 1
- Acts as a block-level exponent offset

### Value Computation

```
value = element_fp_value × 2^(scale - bias)
```

Where `element_fp_value` is the IEEE-decoded minifloat value.

---

## Block Structure

Both formats use blocks of elements sharing a single scale:

```
Block (k elements):
  ┌─────────────────────────────────────────┐
  │ element[0] | element[1] | ... | element[k-1] │
  └─────────────────────────────────────────┘
                      ↑
              shared scale factor
```

Typical block size: k = 8 or 32 elements

---

## Memory Layout

In HBM memory, MXINT data is organized as:

```
Address 0:     [element block 0] [element block 1] ...
Address N:     [element block N] ...
...
Scale offset:  [scale 0] [scale 1] [scale 2] ...
```

Elements are stored contiguously, followed by scales at a configured offset.

---

## RTL Parameters

When instantiating conversion modules:

| Parameter | MXINT Meaning |
|-----------|---------------|
| `MXINT_WIDTH` | Total element width (sign + magnitude) |
| `MXINT_SCALE_WIDTH` | Scale width in bits |

Note: MXINT has no `FRAC_WIDTH` parameter since it's always "all denorm" - all magnitude bits are fractional.

---

## References

- [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537)
- [Block Minifloat Arithmetic](https://openreview.net/forum?id=6zaTwpNSsQ2)
