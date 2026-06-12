# PLENA Analytic Report — SingleStreamBlock (SSB) **Backward Pass**

_Geometry: MLEN=1024, HLEN=128, **BLEN=8**, VLEN=1024 (MAC array = 8×1024 = 8,192); freq = 1.0 GHz_

_Power model: bottom-up 7 nm, **FP16 baseline** (MAC = 1 mW, vector lane = 0.5 mW, SRAM = 0.055 pJ/bit). Datatype scales power on two legs: **MCU(matmul)** by accumulation precision (FP8/FP16 = 1× since FP8 accumulates into FP16; FP32 = 2×) and **Vector/SRAM** by element bit-width (FP8 = 0.5×, FP16 = 1×, FP32 = 2×). **Training uses fp32 → all units ×2.** See §2 for the precision comparison._

_HBM datatype: **fp32** (32-bit, no MX scale) — the backward/training pass keeps full fp32 precision for the gradients, unlike the forward inference pass which uses MX-E4M3. HBM traffic is therefore larger per element, but memory stays a rounding error vs compute._

_Cycle formula: George performance customISA (pipelined column; M_BMM = BLEN·(MLEN/BLEN)² = 131,072, M_MM = BLEN = 8)_

_Method: every backward kernel compiled to real PLENA `.isa`
(`managerbuild_bwd_chain/ir`) and measured by `tools/power/plena_isa_energy.py`.
compute = matmul + vector + scalar_fp (scalar_int / ldst / control NOT counted).
1 cycle = 1 ns._

All 19 backward kernels are compiled end-to-end (not estimated); every
kernel's gradient is the exact analytic backward of its forward operator.

---

## 1. Unit power (W, steady-state @100% utilisation, BLEN = 8)

| Unit | Power | Basis |
|---|---|---|
| Matrix core (MCU) | **16.4 W** | 1 mW/MAC × 8,192 MAC × 2 (fp32) |
| Vector unit (1 thread) | 1.024 W | 0.5 mW/lane × 1024 lanes × 2 (fp32) |
| Matrix SRAM | 1.802 W | 0.055 pJ/bit × 1024×16b × 1 GHz × 2 (fp32) |
| Vector SRAM | 1.802 W | 0.055 pJ/bit × 1024×16b × 1 GHz × 2 (fp32) |

> The MCU power scales with the MAC array (BLEN·MLEN): at BLEN=8 it is
> 16.4 W (fp32); at BLEN=128 it would be 262 W. Energy =
> power × active time (MCU active = matmul cycles; SRAM powered for the
> whole run).

---

## 2. Chain latency & energy (batch = 1, SIMT = 1, BLEN = 8)

| | Value |
|---|---|
| **compute** | **533.88 ms** |
| memory | 0.25 ms |
| **total** | **533.88 ms** (compute-bound) |
| **energy** | **9.48 J** |
| avg power | 17.8 W |

Energy breakdown:

| Unit | Energy | Share |
|---|---|---|
| MCU | 7,617.4 mJ | **80.3 %** |
| VSRAM | 962.2 mJ | 10.1 % |
| MSRAM | 837.9 mJ | 8.8 % |
| Vector | 66.9 mJ | 0.7 % |

The block is **matmul-bound**: the MCU (80 %) plus the two SRAMs (19 %)
dominate; the vector unit is < 1 % of energy.

### Precision: low (FP8) vs high (fp32) — training uses high

Datatype scales the power on **two separate legs** (relative to the FP16
baseline): the **MCU (matmul)** scales with the *accumulation* precision —
FP8 multiplies accumulate into FP16, so the MAC array stays at the FP16
power level (FP8 MCU = 1×, **not** halved); only FP32 accumulation doubles
it. The **vector unit and SRAMs** scale with the *element bit-width* (FP8 =
0.5×, FP16 = 1×, FP32 = 2×). The cycle count is unchanged, so latency is
identical across precisions; only energy and average power move:

| Precision | MCU× | Vec/SRAM× | Latency | Energy | Avg power |
|---|---|---|---|---|---|
| FP8 | 1× | 0.5× | 533.88 ms | 4.28 J | 8.0 W |
| FP16 (baseline) | 1× | 1× | 533.88 ms | 4.74 J | 8.9 W |
| **fp32 (training)** | **2×** | **2×** | **533.88 ms** | **9.48 J** | **17.8 W** |

**Training uses the high-precision (fp32) tier** — gradients need the
dynamic range, so the backward keeps fp32 elements (MCU and vector/SRAM
both ×2 vs FP16). Because the backward is ~80 % MCU energy, the
FP8→FP16→FP32 spread is driven mostly by the MCU leg (FP8 saves only on the
small vector/SRAM legs, hence 4.28 J vs 4.74 J; FP32 doubles everything to
9.48 J). The fp32 row is the headline throughout; the lower-precision rows
isolate the datatype effect. Memory stays a rounding error vs the
533.88 ms compute in all tiers — compute-bound either way.

> Scope: these numbers are **one SingleStreamBlock**, batch = 1, one
> PLENA device. A full MMDiT backward is 38 SSB + 19 DSB blocks; scale
> accordingly.

---

## 2.1 Multi-device scaling (ideal linear parallelism)

Splitting one block across D devices is ideal-linear: wall-clock ÷ D,
**total energy unchanged**, average power × D (the work and its energy are
fixed; D devices just finish it D× faster at D× the instantaneous draw).

| Devices | Latency | Energy | Avg power |
|---|---|---|---|
| 1 | 533.88 ms | 9.48 J | 17.8 W |
| 4 | 133.47 ms | 9.48 J | 71.1 W |
| **12** | **44.49 ms** | **9.48 J** | **213.2 W** |

(12 devices = the deployment matched to one A100's MAC budget used
throughout the forward report.)

## 2.2 Multi-batch scaling (linear)

Each batch element is an independent SSB invocation, processed serially on
one device, so latency and energy scale linearly with batch; average power
is unchanged.

| Batch | Latency (1 dev) | Energy | Latency (12 dev) |
|---|---|---|---|
| 1 | 533.88 ms | 9.48 J | 44.49 ms |
| 4 | 2,135.5 ms | 37.94 J | 178.0 ms |
| **12** (real MMDiT: 3 CFG × 4) | **6,406.6 ms** | **113.81 J** | **533.9 ms** |

So at the real inference shape (batch = 12) one SSB backward block is
**6.41 s on a single PLENA device, or 0.53 s across 12 devices**, for
**113.8 J** of energy (geometry BLEN = 8, fp32). The combined effect is simply
`latency × batch ÷ devices`, `energy × batch`, `avg power × devices`.

---

## 3. Per-kernel ISA cycles (raw, pre-SIMT)

| Kernel | matmul | vector | scalarFP | comp µs | E (mJ) | bwd class |
|---|---|---|---|---|---|---|
| flash_attention_bwd | 171,196,902 | 11,003,904 | 0 | 182,200.8 | 1,727 | attention (5 GEMM) |
| linear2_dx | 59,719,680 | 691,200 | 0 | 60,410.9 | 598 | proj dX |
| linear2_dw | 59,719,680 | 506,880 | 0 | 60,226.6 | 598 | proj dW (transpose-A) |
| linear_mlp_dx | 47,775,744 | 387,072 | 0 | 48,162.8 | 478 | mlp dX |
| linear_mlp_dw | 47,775,744 | 405,504 | 0 | 48,181.2 | 478 | mlp dW (transpose-A) |
| linear_k_dx | 11,943,936 | 138,240 | 0 | 12,082.2 | 120 | k dX |
| linear_q_dx | 11,943,936 | 138,240 | 0 | 12,082.2 | 120 | q dX |
| linear_v_dx | 11,943,936 | 138,240 | 0 | 12,082.2 | 120 | v dX |
| linear_k_dw | 11,943,936 | 101,376 | 0 | 12,045.3 | 120 | k dW (transpose-A) |
| linear_q_dw | 11,943,936 | 101,376 | 0 | 12,045.3 | 120 | q dW (transpose-A) |
| linear_v_dw | 11,943,936 | 101,376 | 0 | 12,045.3 | 120 | v dW (transpose-A) |
| gelu_bwd | 0 | 41,361,408 | 0 | 41,361.4 | 58 | gelu′ (exact) |
| rope_k_bwd | 3,538,971 | 304,128 | 0 | 3,843.1 | 36 | rope (P self-inverse) |
| rope_q_bwd | 3,538,971 | 304,128 | 0 | 3,843.1 | 36 | rope |
| qknorm_k_bwd | 0 | 4,174,848 | 1,769,472 | 5,944.3 | 7 | RMSNorm dX |
| qknorm_q_bwd | 0 | 4,174,848 | 1,769,472 | 5,944.3 | 7 | RMSNorm dX |
| layernorm_bwd | 0 | 1,244,160 | 82,944 | 1,327.1 | 2 | LayerNorm dX |
| modulate_bwd | 0 | 27,648 | 0 | 27.6 | 0.04 | affine |
| residual_gate_bwd | 0 | 27,648 | 0 | 27.6 | 0.04 | affine |

(`concat` backward = split, ~free, omitted.)

The per-kernel `comp µs` ≈ its matmul cycles: every GEMM kernel is
matmul-bound, the vector-only kernels (gelu/norm/affine) are small.

> The `E (mJ)` column is shown at the **FP16 baseline** for per-kernel
> comparison; at the fp32 training tier each value is **×2** (the chain
> total in §2 is already the fp32 figure, 9.48 J).

---

## 4. Per-opcode cycle share (whole chain, compute basis)

| opcode | class | cycles | % of compute |
|---|---|---|---|
| **M_MM** | matmul | 191,102,976 | **35.79 %** |
| **M_TMM_A** | matmul | 159,252,480 | **29.83 %** |
| **M_BTMM** | matmul | 70,778,880 | **13.26 %** |
| M_MM_WO | matmul | 43,794,432 | 8.20 % |
| V_MUL_VF | vector | 21,067,776 | 3.95 % |
| V_ADD_VF | vector | 12,026,880 | 2.25 % |
| V_EXP_V | vector | 9,289,728 | 1.74 % |
| V_RED_SUM | vector | 7,962,624 | 1.49 % |
| V_RECI_V | vector | 5,308,416 | 0.99 % |
| V_SUB_VF | vector | 4,036,608 | 0.76 % |
| V_ADD_VV | vector | 3,621,888 | 0.68 % |
| S_MUL_FP | scalar_fp | 2,257,920 | 0.42 % |
| V_MUL_VV | vector | 1,935,360 | 0.36 % |
| S_ADD_FP | scalar_fp | 451,584 | 0.08 % |
| S_SQRT_FP | scalar_fp | 451,584 | 0.08 % |
| S_RECI_FP | scalar_fp | 451,584 | 0.08 % |
| V_SUB_VV | vector | 82,944 | 0.02 % |
| S_SUB_FP | scalar_fp | 9,216 | 0.00 % |
| M_BMM_WO | matmul | 540 | 0.00 % |
| **COMPUTE TOTAL** | | **533,883,420** | 100 % |

### By class

| class | cycles | share |
|---|---|---|
| **matmul** | 464,929,308 | **87.08 %** |
| **vector** | 65,332,224 | **12.24 %** |
| scalar_fp | 3,621,888 | 0.68 % |

The four matmul opcodes **M_MM + M_TMM_A + M_BTMM + M_MM_WO = 87.1 %** of all
compute. The two largest are the linear backwards: **M_MM (35.8 %)** is the
`dX = dY·W` contraction over N (a plain, non-transposed matmul), and
**M_TMM_A (29.8 %)** is the `dW = dYᵀ·X` contraction over M — a real
**transpose-A** GEMM in which the matrix core transposes the VRAM (A) operand
on the fly (the A-side counterpart of M_TMM's MRAM-side transpose). Together
they are 65.6 % of compute (the same total the earlier plain-matmul mapping
reported as a single M_MM line; only the opcode split changed, not the cost —
M_TMM_A and M_MM share the cycle formula). Each is drained by a per-block
`M_MM_WO` (no hardware accumulator, partials summed by hand). M_TMM_A also
carries attention's `dQ = dS·K`, the other transpose-A point. M_BTMM (13.3 %)
carries the attention score-grads (Sᵀ, dPᵀ) and rope. M_MM_WO (8.2 %) is the
drain count from all the hand-accumulated dX/dW partials. Vector (12.2 %) is
the exact gelu′ and the RMS/LayerNorm Jacobians; scalar_fp (0.7 %) is the
norms' sqrt/reciprocal.

> Note: this §4 `compute` total (533.9 M) now matches the §2/§3 energy-tool
> total exactly (both read the real `plena_settings.toml` cycle constants).
> Earlier revisions differed because the energy tool divides the
> vector leg by the SIMT thread count and applies a small per-op rounding;
> the matmul leg (the critical path) is identical.

---

## 5. Backward vs forward (the structural rules)

| Operator class | backward matmul vs fwd | why |
|---|---|---|
| **Linear (+bias)** ×5 | **2.0×** | fwd 1 GEMM (X·Wᵀ); bwd 2 (dX=dY·W, dW=dYᵀ·X) |
| **Attention** | **2.5×** (measured) | fwd 2 GEMM; bwd 5 (Sᵀ, dPᵀ, dV, dK, dQ) |
| **RoPE** ×2 | **1.0×** | P is a symmetric permutation (P = Pᵀ = P⁻¹) |
| **GELU** | matmul 0 | dX = dY·gelu′(x): vector ≈ 3× fwd (recompute tanh + sech²) |
| **RMS/LayerNorm** | matmul 0 | dX needs the normalize Jacobian: ~2× fwd vector |
| **affine** | matmul 0 | dX = dY·factor: ~1× fwd vector |

Each linear's dX and dW are the two GEMMs of the exact 2× rule. **dX and dW
cost the same** (e.g. q/k/v_dx = q/k/v_dw = 11.94 M): dX is a plain
(non-transposed) matmul contracting N (M_MM), and dW is a transpose-A matmul
contracting M (M_TMM_A) — the matrix core transposes the activation A operand
on the fly. M_TMM_A and M_MM share the cycle formula, so the two GEMMs are
exactly equal in cost, each accumulating its partials by hand with one M_MM_WO
drain per block. dW = dYᵀ·X is the same transpose-A mapping as attention's dQ
(see §6). RoPE backward equals forward (3.54 M) because the rotation is its
own inverse.

### Where the backward compute concentrates (matmul, the critical path)

| Kernel | matmul (M) | share of bwd matmul |
|---|---|---|
| flash_attention_bwd | 171.2 | 36.8 % |
| linear2 (dx+dw) | 119.4 | 25.7 % |
| linear_mlp (dx+dw) | 95.6 | 20.6 % |
| linear_q/k/v (dx+dw) | 71.6 | 15.4 % |
| rope_q/k | 7.1 | 1.5 % |

**Attention + the four big linears ≈ 98 %** of the backward matmul work —
the same kernels that dominate the forward.

---

## 6. PLENA-specific backward notes

- **Transpose-A** shows up in exactly two backward GEMMs: attention's
  `dQ = dS·K` and the linear `dW = dYᵀ·X`. Both need to contract over an axis
  that sits on the VRAM (A) operand, while the MRAM transpose unit (M_TMM)
  only transposes the B operand. PLENA handles this with a dedicated
  **transpose-A datapath**: the matrix core transposes the VRAM (A) tile on
  the fly as it streams into the array — the symmetric counterpart of M_TMM's
  MRAM-side transpose. It lowers to the **`M_TMM_A`** opcode, which shares the
  exact cycle formula of `M_MM`, so the contraction is mathematically exact at
  no extra cost (no corner-turn copy, no transpose buffer).

- **The two linear GEMMs use M_MM and M_TMM_A; only attention's score-grads
  use transpose-B.** The linear **`dX = dY·W`** contracts N with a *plain*
  (non-transposed) matmul `M_MM` — W is stored `[N,K]` so a non-transposed B
  contracts its first dim (N) correctly; the `n_blocks` N-partials are
  accumulated by hand (`dX += dY·W` per block). The linear **`dW = dYᵀ·X`**
  contracts M and is the **transpose-A point** above: dY is stored `[M,N]`
  (its natural forward layout) and fed as the A operand with `transpose_A=True`,
  so the core transposes it to `[N,M]` on the fly (**`M_TMM_A`**), giving the
  exact `[N,M]·[M,K]` contraction with M-partials hand-accumulated — exactly
  like attention's dQ. Because M_TMM_A costs the same as M_MM, dW's matmul cost
  equals dX's; the chain is **87.1 %** matmul-bound. Attention `Sᵀ`/`dPᵀ` and
  rope keep transpose-B (M_TMM); attention `dV`/`dK` are plain matmuls.

- **All vector backward kernels are mathematically exact**: gelu′ is the
  exact derivative of the tanh-GELU (tanh expanded via exp/reci as in the
  forward); RMSNorm/LayerNorm use the full input-gradient Jacobian
  (recompute rms / μ,σ plus the cross-term reductions). They stay
  vector-only and off the matmul critical path.

---

## 7. Caveats

- Single-thread (SIMT = 1), batch = 1. SIMT-N divides the vector/scalar
  legs by N (matmul unaffected); batch scales everything linearly. The
  bwd/fwd ratios are invariant to both.
- The attention `dQ` and linear `dW` transpose-A contractions use the
  on-the-fly transpose-A datapath (`M_TMM_A`), which shares M_MM's cycle
  cost — no separate corner-turn copy or transpose buffer.

- This report is the **BLEN = 8** geometry. A BLEN = 128 version (MCU =
  131 W, M_MM = 128) follows the same structure with the MCU power and the
  matmul cycle constants rescaled.
