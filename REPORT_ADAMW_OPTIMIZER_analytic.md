# PLENA Analytic Report — AdamW **Optimizer Step**

_Geometry: MLEN=1024, HLEN=128, **BLEN=8**, VLEN=1024 (MAC array = 8×1024 = 8,192); freq = 1.0 GHz_

_Power model: bottom-up 7 nm, **FP16 baseline** (MAC = 1 mW, vector lane = 0.5 mW, SRAM = 0.055 pJ/bit). The optimizer is part of training → same **fp32** tier as the backward, so Vector and SRAM are ×2 (the MCU is idle anyway — AdamW has no matmul). FP8 would be ×0.5 on vector/SRAM._

_HBM datatype: **fp32** (32-bit, no MX scale) — the optimizer runs in the training pass, keeping full fp32 precision for weights and moments, like the backward. HBM traffic stays a rounding error vs compute._

_Cycle formula: George performance customISA (M_BMM = BLEN·(MLEN/BLEN)² = 131,072, M_MM = BLEN = 8)_

_Method: the AdamW step compiled to real PLENA `.isa`
(`managerbuild_adamw/ir`, `adam_step_min` kernel) and measured by
`tools/power/plena_isa_energy.py`. compute = matmul + vector + scalar_fp
(scalar_int / ldst / control NOT counted). 1 cycle = 1 ns._

The AdamW step is the **optimizer** half of one training iteration (the
backward pass being the other half). It applies the decoupled-weight-decay
Adam update to every trainable weight:

```
m  = beta1*m + (1-beta1)*g
v  = beta2*v + (1-beta2)*g*g
mhat = m / (1 - beta1**t)            # bias correction (precomputed scalar)
vhat = v / (1 - beta2**t)
w  = w - lr*mhat/(sqrt(vhat)+eps) - lr*wd*w     # decoupled weight decay
```

It is **entirely vector-elementwise** — there is no matmul. The update
lowers to a single straight-line vector chain per weight tile (moment EMAs,
bias correction, the `1/(sqrt(vhat)+eps)` rescaling, and the
decoupled-weight-decay step); the per-tile opcode mix is **9× V_MUL_VF,
5× V_MUL_VV, 2× V_ADD_VV, 1× V_ADD_VF, 2× V_SUB_VV, 1× V_SUB_VF,
1× V_RECI_V** (no M_*).

Unlike the backward (whose cost scales with the activation **sequence /
batch**), the optimizer's cost scales with the **parameter count** — it
touches each weight once per step, independent of batch. The numbers below
are therefore reported per **parameter count**, scaled from the measured
per-tile cost.

---

## 1. Unit power (W, steady-state @100% utilisation, BLEN = 8)

| Unit | Power | Basis |
|---|---|---|
| Matrix core (MCU) | **16.4 W** | 1 mW/MAC × 8,192 × 2 (fp32) — **idle here (no matmul)** |
| Vector unit (1 thread) | 1.024 W | 0.5 mW/lane × 1024 × 2 (fp32) |
| Matrix SRAM | 1.802 W | 0.055 pJ/bit × 1024×16b × 1 GHz × 2 (fp32) |
| Vector SRAM | 1.802 W | 0.055 pJ/bit × 1024×16b × 1 GHz × 2 (fp32) |

> The MCU is **completely idle** for the optimizer — AdamW is pure vector
> work, so all energy sits in the Vector Unit and the Vector SRAM. The
> Matrix core and Matrix SRAM contribute nothing.

---

## 2. Measured per-tile cost (the compiled kernel)

One compiled tile covers **2,097,152 weights** (NSB=2 × MLEN=1024 ×
head_count=8 × HLEN=128), measured directly from its `.isa`:

| | Value |
|---|---|
| matmul | **0** (pure vector) |
| vector | 231,424 cyc |
| **compute** | **231.4 µs** |
| memory | 0.25 µs |
| **total** | **231.4 µs** (compute-bound) |
| **energy** | **0.65 mJ** |

Per element: **0.110 vector-cyc, 0.110 ns, 0.314 nJ** (fp32). Energy split: **VSRAM
63.8 %, Vector Unit 36.2 %** (MCU / MSRAM = 0). All scaling below is linear
from this point.

---

## 3. Per-block AdamW (one SSB's / one DSB's weights, batch-independent)

The optimizer step over the trainable weights of a single block, 1 device,
SIMT = 1:

| Block | Trainable params | Latency | Energy | VSRAM | Vector |
|---|---|---|---|---|---|
| **SSB** | 113.29 M | **12.50 ms** | **35.6 mJ** | 22.7 mJ | 12.9 mJ |
| **DSB** | 226.62 M | **25.00 ms** | **71.4 mJ** | 45.5 mJ | 25.9 mJ |

The **DSB is exactly 2× the SSB** — its two streams hold twice the weights.
Note this 2:1 ratio differs from the forward / backward per-block share
(63 % / 37 %, 60 % / 40 %): those scale with the activation sequence (DSB's
text stream is only 1/9), whereas the optimizer scales with **weights**,
which are symmetric across the two streams.

> SSB trainable params = 113.29 M (4 big linears WQ/WK/WV 9.4 M each, WM
> 37.7 M, W2 47.2 M, plus norm/modulate/gate affines). DSB = 226.62 M (two
> streams: WQ/WK/WV/WPROJ 9.4 M each + WMI/WMO 37.7 M each, ×2).

---

## 4. Full MMDiT AdamW (all weights, one step)

A full MMDiT is **38 SSB + 19 DSB**. Total trainable parameters:

$$38 \times 113.29\,\text{M} + 19 \times 226.62\,\text{M} = \mathbf{8.611\ B}$$

split **50 % / 50 %** between SSB and DSB weights (38 single-stream blocks ≈
19 double-stream blocks in parameter mass). One AdamW step over all 8.611 B
weights:

| | Value |
|---|---|
| trainable params | **8.611 B** |
| matmul | **0** |
| **compute / latency (1 dev)** | **950.1 ms** |
| **energy** | **2.710 J** |
| VSRAM | 1.729 J (63.8 %) |
| Vector | 0.981 J (36.2 %) |
| MCU / MSRAM | 0 |

### 4.1 Multi-device scaling (ideal linear, energy fixed)

| Devices | Latency | Energy | Avg power |
|---|---|---|---|
| 1 | 950.1 ms | 2.710 J | 2.9 W |
| 4 | 237.5 ms | 2.710 J | 11.4 W |
| **12** | **79.2 ms** | **2.710 J** | **34.2 W** |

(12 devices = the deployment matched to one A100's MAC budget used
throughout the forward / backward reports. Energy is independent of device
count.)

### 4.2 Batch independence

The optimizer step is **independent of batch**. It updates each weight once
per training step regardless of how large a batch the forward / backward
ran — there is no `× batch` factor (unlike the backward, whose latency and
energy scale linearly with batch). One step = one pass over the 8.611 B
weights, full stop.

---

## 5. Optimizer vs backward — the headline

Placed against the backward pass of the **same** full MMDiT (38 SSB + 19
DSB backward, batch = 4, 12 devices — from the SSB/DSB backward reports):

| Phase | Latency (12 dev) | Energy | matmul share |
|---|---|---|---|
| **Backward** (batch = 4, fp32) | **11.35 s** | **2,437 J** | 87–89 % |
| **AdamW optimizer** (fp32) | **0.079 s** | **2.71 J** | **0 %** |
| optimizer as fraction | **0.69 %** | **0.11 %** | — |

The optimizer is a **negligible tail** on the training step: under **1 % of
the latency and ~0.1 % of the energy** of the backward it follows. This is
exactly the expected profile — AdamW is a handful of element-wise vector
ops per weight (vector-bound, MCU idle), while the backward is a chain of
large MLEN³ GEMMs (matmul-bound, MCU ≈ 80 % of energy). The two are
structural opposites, and the GEMM-heavy backward dominates training cost.

---

## 6. Per-opcode breakdown (one tile, compute basis)

| class | cycles | share |
|---|---|---|
| **vector** | 231,424 | **100 %** |
| matmul | 0 | 0 % |
| scalar_fp | 0 | 0 % |

All compute is in the vector unit. The opcode chain per weight tile:

| opcode | count | role |
|---|---|---|
| V_MUL_VF | 9 | β₁·M, (1−β₁)·G, β₂·V, (1−β₂)·g², bc1, bc2, lr, weight-decay terms |
| V_MUL_VV | 5 | g² and the rescaling products |
| V_ADD_VV | 2 | m accumulate, v accumulate |
| V_ADD_VF | 1 | rescaling-term add |
| V_SUB_VV | 2 | Adam step, weight-decay subtract |
| V_SUB_VF | 1 | rescaling-term subtract |
| V_RECI_V | 1 | reciprocal in the `1/(sqrt(vhat)+eps)` rescaling |

The whole AdamW update is a single straight-line vector chain — no matmul,
no compiler change.

---

## 7. Caveats

- Single-thread (SIMT = 1), 1 device. SIMT-N divides the vector leg by N
  (there is no matmul leg to be unaffected); device scaling divides latency
  by N at fixed energy. Batch does **not** scale the optimizer.
- Per-element cost is scaled from one measured 2.10 M-weight tile; the full
  8.611 B figure assumes the same per-element cost across all weight tiles
  (exact for this straight-line vector kernel).
- This is the **BLEN = 8** geometry. The optimizer is matmul-free, so the
  MCU power (which scales with BLEN) is irrelevant here — a BLEN = 128
  build would give identical optimizer latency/energy (vector unit and
  VSRAM are VLEN-bound, not BLEN-bound).
