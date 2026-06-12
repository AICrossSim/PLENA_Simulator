# PLENA Analytic Report — DoubleStreamBlock (DSB) **Backward Pass**

_Geometry: MLEN=1024, HLEN=128, **BLEN=8**, VLEN=1024 (MAC array = 8×1024 = 8,192); freq = 1.0 GHz_

_Power model: bottom-up 7 nm, **FP16 baseline** (MAC = 1 mW, vector lane = 0.5 mW, SRAM = 0.055 pJ/bit). Datatype scales power on two legs: **MCU(matmul)** by accumulation precision (FP8/FP16 = 1× since FP8 accumulates into FP16; FP32 = 2×) and **Vector/SRAM** by element bit-width (FP8 = 0.5×, FP16 = 1×, FP32 = 2×). **Training uses fp32 → all units ×2.** See §2 for the precision comparison._

_HBM datatype: **fp32** (32-bit, no MX scale) — the backward/training pass keeps full fp32 precision for the gradients, unlike the forward inference pass which uses MX-E4M3. HBM traffic is therefore larger per element, but memory stays a rounding error vs compute._

_Cycle formula: George performance customISA (M_BMM = BLEN·(MLEN/BLEN)² = 131,072, M_MM = BLEN = 8)_

_Method: every backward kernel compiled to real PLENA `.isa`
(`managerbuild_bwd_double/ir`, 47 kernels) and measured by
`tools/power/plena_isa_energy.py`. compute = matmul + vector + scalar_fp
(scalar_int / ldst / control NOT counted). 1 cycle = 1 ns._

The DoubleStreamBlock runs **two parallel streams** — an **image** stream
(S = 9216, NSB = 9) and a **text** stream (S = 1024, NSB = 1) — each with
its own norm / modulate / linear-qkv / qknorm / rope / mlp / proj / res.
Their q/k/v are concatenated into one **joint sequence (10240, NSB = 10)**,
run through a **single flash attention**, then split back per stream. The
backward reuses the exact same operator-class kernels as the SSB backward;
the chain just has each operator's backward **once per stream** plus **one
joint flash-attention backward**. concat/split backward (= split/concat) is
~free and omitted.

---

## 1. Unit power (W, steady-state @100% utilisation, BLEN = 8)

| Unit | Power | Basis |
|---|---|---|
| Matrix core (MCU) | **16.4 W** | 1 mW/MAC × 8,192 MAC × 2 (fp32) |
| Vector unit (1 thread) | 1.024 W | 0.5 mW/lane × 1024 lanes × 2 (fp32) |
| Matrix SRAM | 1.802 W | 0.055 pJ/bit × 1024×16b × 1 GHz × 2 (fp32) |
| Vector SRAM | 1.802 W | 0.055 pJ/bit × 1024×16b × 1 GHz × 2 (fp32) |

> MCU power scales with the MAC array (BLEN·MLEN): 16.4 W at BLEN=8 (fp32), 262 W at
> BLEN=128. Energy = power × active time (MCU active = matmul cycles; SRAM
> powered for the whole run).

---

## 2. Chain latency & energy (batch = 1, SIMT = 1, BLEN = 8)

| | Value |
|---|---|
| **compute** | **724.72 ms** |
| memory | 0.36 ms |
| **total** | **724.72 ms** (compute-bound) |
| **energy** | **13.09 J** |
| avg power | 18.1 W |

Energy breakdown:

| Unit | Energy | Share |
|---|---|---|
| MCU | 10,549.6 mJ | **80.6 %** |
| VSRAM | 1,306.2 mJ | 10.0 % |
| MSRAM | 1,160.4 mJ | 8.9 % |
| Vector | 78.6 mJ | 0.6 % |

> Scope: one DoubleStreamBlock, batch = 1, one PLENA device. A full MMDiT
> backward is 38 SSB + 19 DSB blocks; scale accordingly.

### Precision: low (FP8) vs high (fp32) — training uses high

Power scales on two legs (relative to the FP16 baseline): **MCU(matmul)**
by accumulation precision (FP8/FP16 = 1×, since FP8 accumulates into FP16;
FP32 = 2×), **Vector/SRAM** by element bit-width (FP8 = 0.5×, FP16 = 1×,
FP32 = 2×). Cycle count is unchanged → latency identical; only energy and
power move:

| Precision | MCU× | Vec/SRAM× | Latency | Energy | Avg power |
|---|---|---|---|---|---|
| FP8 | 1× | 0.5× | 724.72 ms | 5.91 J | 8.2 W |
| FP16 (baseline) | 1× | 1× | 724.72 ms | 6.55 J | 9.0 W |
| **fp32 (training)** | **2×** | **2×** | **724.72 ms** | **13.09 J** | **18.1 W** |

**Training uses the high-precision (fp32) tier** (gradients need the
dynamic range). Memory stays a rounding error vs the 724.72 ms compute in
all tiers — compute-bound either way.

### 2.1 Multi-device scaling (ideal linear)

| Devices | Latency | Energy | Avg power |
|---|---|---|---|
| 1 | 724.72 ms | 13.09 J | 18.1 W |
| 4 | 181.18 ms | 13.09 J | 72.3 W |
| **12** | **60.39 ms** | **13.09 J** | **216.8 W** |

### 2.2 Multi-batch scaling (linear)

| Batch | Latency (1 dev) | Energy | Latency (12 dev) |
|---|---|---|---|
| 1 | 724.72 ms | 13.09 J | 60.39 ms |
| 4 | 2,898.9 ms | 52.38 J | 241.6 ms |
| **12** (real MMDiT: 3 CFG × 4) | **8,696.6 ms** | **157.14 J** | **724.7 ms** |

`latency = 724.72 ms × batch ÷ devices`, `energy = 13.09 J × batch`,
`avg power = 18.1 W × devices`.

---

## 3. Matrix vs vector time share (the headline)

| class | cycles | time | **share** |
|---|---|---|---|
| **matmul** | 643,891,860 | 643.89 ms | **88.85 %** |
| **vector** | 76,713,984 | 76.71 ms | **10.59 %** |
| scalar_fp | 4,116,480 | 4.12 ms | 0.57 % |

The DSB backward is **even more matmul-bound than the SSB backward**
(88.9 % vs 87.1 %): it carries two streams' worth of linear layers plus a
larger joint flash attention, so the matrix datapath dominates the
critical path while the vector unit sits in its shadow. (Both numbers rose
after the linear `dW` fix — dW now contracts over M as a transpose-A matmul
`M_TMM_A`, equal in cost to dX's M_MM, not a cheaper btmm — see §6.)

---

## 4. Per-kernel ISA cycles (raw, pre-SIMT) — grouped

### Joint attention

| Kernel | matmul | vector | comp µs | E (mJ) |
|---|---|---|---|---|
| flash_attention_bwd (joint, NSB=10) | 211,354,200 | 13,578,240 | 224,932 | 2,132 |

### Image stream (S = 9216, NSB = 9)

| Kernel | matmul | vector | comp µs | E (mJ) |
|---|---|---|---|---|
| I_proj_dx | 59,719,680 | 691,200 | 60,411 | 598 |
| I_proj_dw | 59,719,680 | 506,880 | 60,227 | 598 |
| I_mlpout_dx | 47,775,744 | 552,960 | 48,329 | 478 |
| I_mlpin_dx | 47,775,744 | 387,072 | 48,163 | 478 |
| I_mlpout_dw | 47,775,744 | 405,504 | 48,181 | 478 |
| I_mlpin_dw | 47,775,744 | 405,504 | 48,181 | 478 |
| I_linq/k/v_dx (×3) | 11,943,936 ea | 138,240 ea | 12,082 ea | 120 ea |
| I_linq/k/v_dw (×3) | 11,943,936 ea | 101,376 ea | 12,045 ea | 120 ea |
| I_gelu | 0 | 41,361,408 | 41,361 | 58 |
| I_rope_q/k (×2) | 3,538,971 ea | 304,128 ea | 3,843 ea | 36 ea |
| I_qknorm_q/k (×2) | 0 | 4,174,848 ea | 5,944 ea | 7 ea |
| I_norm1/2 (×2) | 0 | 1,244,160 ea | 1,327 ea | 2 ea |
| I_mod1/2, I_res1/2 (×4) | 0 | 27,648 ea | 28 ea | 0.04 ea |

### Text stream (S = 1024, NSB = 1) — ~1/9 the image stream

| Kernel | matmul | vector | comp µs | E (mJ) |
|---|---|---|---|---|
| T_proj_dx | 6,635,520 | 76,800 | 6,712 | 66 |
| T_proj_dw | 6,635,520 | 138,240 | 6,774 | 67 |
| T_mlpout/in_dx (×2) | 5,308,416 | — | ~5,360 | ~53 |
| T_mlpout/in_dw (×2) | 5,308,416 | 110,592 | 5,419 | 53 |
| T_linq/k/v_dx (×3) | 1,327,104 ea | 15,360 ea | 1,343 ea | 13 ea |
| T_linq/k/v_dw (×3) | 1,327,104 ea | 27,648 ea | 1,355 ea | 13 ea |
| T_gelu | 0 | 4,595,712 | 4,596 | 6 |
| T_rope_q/k (×2) | 393,219 ea | 33,792 ea | 427 ea | 4 ea |
| T_qknorm_q/k (×2) | 0 | 463,872 ea | 661 ea | 0.8 ea |
| T_norm1/2 (×2) | 0 | 138,240 ea | 148 ea | 0.2 ea |
| T_mod1/2, T_res1/2 (×4) | 0 | 3,072 ea | 3 ea | 0.00 ea |

The text stream is roughly **1/9** the image stream throughout (its
sequence is 1024 vs 9216), so the image stream and the joint attention
account for essentially all of the cost.

> The `E (mJ)` column is at the **FP16 baseline** for per-kernel
> comparison; at the fp32 training tier each value is **×2** (the §2 chain
> total, 13.09 J, is already the fp32 figure).

---

## 5. Per-opcode cycle share (whole chain, compute basis)

| opcode | class | cycles | % of compute |
|---|---|---|---|
| **M_MM** | matmul | 267,386,880 | **36.90 %** |
| **M_TMM_A** | matmul | 228,065,280 | **31.47 %** |
| **M_BTMM** | matmul | 86,507,520 | **11.94 %** |
| M_MM_WO | matmul | 61,931,520 | 8.55 % |
| V_MUL_VF | vector | 24,041,472 | 3.32 % |
| V_ADD_VF | vector | 13,780,992 | 1.90 % |
| V_EXP_V | vector | 10,813,440 | 1.49 % |
| V_RED_SUM | vector | 9,830,400 | 1.36 % |
| V_RECI_V | vector | 5,898,240 | 0.81 % |
| V_SUB_VF | vector | 5,038,080 | 0.70 % |
| V_ADD_VV | vector | 4,853,760 | 0.67 % |
| S_MUL_FP | scalar_fp | 2,560,000 | 0.35 % |
| V_MUL_VV | vector | 2,334,720 | 0.32 % |
| S_ADD_FP | scalar_fp | 512,000 | 0.07 % |
| S_SQRT_FP | scalar_fp | 512,000 | 0.07 % |
| S_RECI_FP | scalar_fp | 512,000 | 0.07 % |
| V_SUB_VV | vector | 122,880 | 0.02 % |
| S_SUB_FP | scalar_fp | 20,480 | 0.00 % |
| M_BMM_WO | matmul | 660 | 0.00 % |
| **COMPUTE TOTAL** | | **724,722,324** | 100 % |

### By class

| class | cycles | share |
|---|---|---|
| **matmul** | 643,891,860 | **88.85 %** |
| **vector** | 76,713,984 | **10.59 %** |
| scalar_fp | 4,116,480 | 0.57 % |

The four matmul opcodes **M_MM + M_TMM_A + M_BTMM + M_MM_WO = 88.9 %** of all
compute. The two largest are the linear backwards across both streams:
**M_MM (36.9 %)** is every `dX = dY·W` (a plain, non-transposed contraction
over N), and **M_TMM_A (31.5 %)** is every `dW = dYᵀ·X` — a real
**transpose-A** GEMM in which the matrix core transposes the VRAM (A) operand
on the fly (the A-side counterpart of M_TMM's MRAM-side transpose). Together
they are 68.4 % of compute (the same total the earlier plain-matmul mapping
reported as one M_MM line; M_TMM_A and M_MM share the cycle formula, so only
the opcode split changed, not the cost). Each is drained by a per-block
`M_MM_WO` (no hardware accumulator, partials summed by hand — hence the 8.6 %
M_MM_WO share). M_TMM_A also carries the joint attention's `dQ = dS·K`, the
other transpose-A point. M_BTMM (11.9 %) carries the attention score-grads
(Sᵀ, dPᵀ) and rope. Vector (10.6 %) is the exact gelu′ and the RMS/LayerNorm
Jacobians; scalar_fp (0.6 %) is the norms' sqrt/reciprocal.

> Note: this §5 compute total (724.7 M) matches the §2 energy-tool total
> exactly — both read the real `plena_settings.toml` cycle constants and
> the `performance/customISA_lib.json` pipelined formulas. The matmul leg
> (the critical path) is the headline figure.

---

## 6. DSB vs SSB backward, and forward rules

| | SSB backward | **DSB backward** |
|---|---|---|
| compute | **533.9 ms** | **724.7 ms** |
| energy | 9.48 J | **13.09 J** |
| matmul share | 87.1 % | **88.9 %** |
| vector share | 12.2 % | **10.6 %** |
| MCU energy share | 80.3 % | **80.6 %** |

The backward FLOP rules per operator class are unchanged from the SSB
report: **linear = 2× forward matmul** (dX + dW), **attention = 2.5×**
(measured), **rope = 1.0×** (P self-inverse), and the vector-only kernels
(gelu / norm / affine) stay vector-only with mathematically exact
gradients. DSB simply instances the per-stream operators twice and runs a
larger joint attention, pushing the matmul share up. (Both blocks' matmul
shares rose vs the previous revision after the linear `dW` fix — dW now
contracts over M as a transpose-A matmul `M_TMM_A`, equal in cost to dX's
M_MM, not a cheaper btmm — see §7.)

### Where the backward compute concentrates (matmul, the critical path)

| Group | matmul (M) | share of bwd matmul |
|---|---|---|
| joint flash_attention | 211.4 | 32.8 % |
| image-stream linears (proj+mlp+qkv, dx+dw) | ~382 | 59.4 % |
| text-stream linears | ~42 | 6.6 % |
| rope (both streams) | ~7.9 | 1.2 % |

**Joint attention + the image-stream linears ≈ 92 %** of the backward
matmul work.

---

## 7. PLENA-specific backward notes

- **Transpose-A** (attention `dQ = dS·K` and every linear `dW = dYᵀ·X`):
  these contract over an axis that sits on the VRAM (A) operand, while the
  MRAM transpose unit (M_TMM) only transposes the B operand. PLENA handles
  this with a dedicated **transpose-A datapath**: the matrix core transposes
  the VRAM (A) tile on the fly as it streams into the array — the symmetric
  counterpart of M_TMM's MRAM-side transpose. It lowers to the **`M_TMM_A`**
  opcode, which shares M_MM's exact cycle formula, so the contraction is
  mathematically exact at no extra cost (no corner-turn copy, no transpose
  buffer).

- **The linear GEMMs use M_MM (dX) and M_TMM_A (dW).** `dX = dY·W` contracts
  N with a plain (non-transposed) matmul `M_MM`; `dW = dYᵀ·X` contracts M with
  a transpose-A matmul `M_TMM_A` — dY is stored `[M,N]` and fed as the A
  operand with `transpose_A=True`, transposed to `[N,M]` on the fly. Both
  hand-accumulate their partials (one M_MM_WO drain per block). Because
  M_TMM_A costs the same as M_MM, dW equals dX in matmul cost, giving the
  **88.9 %** matmul share. Only the attention score-grads (Sᵀ, dPᵀ) and rope
  keep transpose-B (M_TMM); attention `dV`/`dK` are plain matmuls.
  **rope / gelu / rmsnorm / layernorm / affine** backward are exact (gelu′
  exact derivative; norms use the full Jacobian).

---

## 8. Caveats

- Single-thread (SIMT = 1), batch = 1; SIMT-N divides the vector/scalar
  legs by N (matmul unaffected), batch scales linearly. Ratios invariant.
- The two transpose-A points (attention dQ, linear dW) use the on-the-fly
  transpose-A datapath (`M_TMM_A`), which shares M_MM's cycle cost — no
  separate corner-turn copy or transpose buffer.
- This is the **BLEN = 8** geometry. A BLEN = 128 version (MCU = 131 W,
  M_MM = 128) follows the same structure with the MCU power and matmul
  cycle constants rescaled.
- Compile-only build: kernel inputs/outputs share HBM buffer pools to fit
  the int32 address space. This changes only tensor addresses, not the
  per-kernel ISA, so the cycle/energy analysis is unaffected.
