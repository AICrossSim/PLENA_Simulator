# On-chip Memory Layout Convention

There are four on-chip SRAM in total.
- Matrix SRAM: Used to store the weight matrix from HBM only, cannot be written by matrix sram. And only the matrix machine can read weight datafrom this memory.
- Vector SRAM: Used as the scatchpad for activations and intermediate outputs (Not Weight Data), it has two ports:
 - Port A (RW): Used to prefetch and write back data from HBM or Used to load data to vector machine for V_* ops
 - Port B (RW): Used to load data to vector machine for V_* ops and matrix machine for M_* ops, and write back the computed result from matrix and vector sram.
- Scalar INT SRAM: Used as the extended register file for S_*_INT ops
- Scalar FP SRAM: Used as the extended register file for S_*_FP ops, we have preloaded file to this memory, it will store some FP constant for the computation.

All the on-chip address are specified in the gp register, and the address are in the unit of data elements (not related to the data type).

# On-chip Vector SRAM Memory Layout Convention

By default, the memory layout follows this format:

Given an input tensor with shape `[b, s, h]` where:
- `b` = batch size
- `s` = sequence length
- `h` = hidden size

The data stored in Vector SRAM is reshaped to `[h // VLEN, b, s, VLEN]`.

**Rationale:** The hidden dimension is split along the `VLEN` boundary to enable efficient multi-batch GEMM operations. This layout allows parallel processing across batches while maintaining vector-length alignment requirements.


### Example: Activation Tensor

For an activation tensor with shape `[batch=4, hidden=128]` and `VLEN=64`:

- Reshaped to: `[128//64, 4, 64]` = `[2, 4, 64]`
- Vector SRAM layout:
  - Address 0: First 64 elements of hidden dim for all batches
  - Address 256: Second 64 elements of hidden dim for all batches

### Output Tensor Location by Layer Type

Different layer types store their final outputs at different locations in Vector SRAM. Assuming the activations are stored from the Vector SRAM starting from address 0. All the intermediate outputs are stored after the activations, also in the same shape format.

| Layer Type | Output Location | Reason |
|------------|-----------------|--------|
| **Linear** | AFTER activations | Input needed for gradient computation |
| **RMS Norm** | REPLACE activations | Element-wise operation, input no longer needed |
| **FFN** | REPLACE activations| Multiple stages; final output overwrites input |
| **Attention** | REPLACE activations | Multiple stages; final output overwrites input |

### Linear Layer Output Location

**CRITICAL:** When writing output with `M_MM_WO`, you must write to addresses AFTER the activations to avoid overwriting input data that may still be needed.

For a linear layer Y = X @ W with X shape `[batch, hidden]`:
- Activations occupy: `batch * hidden` elements = addresses `[0, batch*hidden)`
- **Output must start at:** `batch * hidden` (e.g., address 512 for batch=4, hidden=128)

Example assuming BLEN = 4, VLEN = 64, and MLEN = 64.

For every M_MM_WO, the accumulated results from the matrix unit with shape (BLEN, BLEN) are written to the Vector SRAM. The write address is composed of a column offset plus a row offset: every column offset is a multiple of BLEN, and every row offset is a multiple of MLEN × BLEN.

For example, if you want to write the accumulated result to the second (BLEN, BLEN) block within a (BLEN, MLEN) region of the Vector SRAM, the row offset stays the same, but the column offset becomes BLEN.

# On-chip Matrix SRAM Memory Layout Convention

For a high-dimensional matrix stored in HBM, `(MLEN, MLEN)` tiles are extracted and stored in the On-chip Matrix SRAM using a **row-major tile layout**.

**Note:** Addresses are specified in units of data elements (not bytes).

## Tile Addressing

Each `(MLEN, MLEN)` tile is stored contiguously in row-major order. Tiles are addressed sequentially.

Consider a large matrix of shape `(2*MLEN, 2*MLEN)`, logically divided into four `(MLEN, MLEN)` tiles arranged as `[[0, 1], [2, 3]]`. The tiles are mapped as follows:
- **Address 0..MLEN-1:** Tile 0 of Matrix[0][0]
- **Address MLEN..2*MLEN-1:** Tile 1 of Matrix[0][1]
- **Address 2*MLEN..3*MLEN-1:** Tile 2 of Matrix[1][0]
- **Address 3*MLEN..4*MLEN-1:** Tile 3 of Matrix[1][1]

**Rationale:** The matrix is tiled and stored in the On-chip Matrix SRAM to enable efficient matrix multiplication operations. This row-major tile layout allows parallel processing across tiles while maintaining matrix-length alignment requirements.

---

# Scalar FP Memory (FP_MEM)

FP_MEM is a small memory for scalar floating-point constants. It is preloaded before execution with layer-specific constants. Use `S_LD_FP` to load values into FP registers.

**Example:**
```asm
S_LD_FP f1, gp0, 1    ; Load FP_MEM[1] into f1
S_LD_FP f3, gp0, 2    ; Load FP_MEM[2] into f3
```

The exact contents of FP_MEM depend on the layer type and are provided via the `fp_sram_layout` field in the workload configuration.

---


# Off-chip HBM Memory Convention

## MXFP Format and Address Calculation

Data in HBM uses MXFP (Microscaling) format where each block of 8 elements has an associated scale factor. This means the **actual HBM size is larger than the logical tensor size**.

**MXFP overhead ratio:** `(8 elements + 1 scale) / 8 elements = 1.125`

### Computing HBM Base Addresses

**Critical:** When setting up HBM address registers (`a0`, `a1`, `a2`), you must account for MXFP overhead:

```python
mxfp_ratio = 1.125  # (8 + 1) / 8

# For activation tensor [batch, hidden]
act_hbm_size = int(batch * hidden * mxfp_ratio)

# For weight tensor [hidden, hidden]
weight_hbm_size = int(hidden * hidden * mxfp_ratio)

# HBM address register setup
a0 = 0                           # activation base
a1 = act_hbm_size                # weight base (NOT batch * hidden!)
a2 = act_hbm_size + weight_hbm_size  # output base
```

### Example: Linear Layer [4, 128] @ [128, 128]

```
batch=4, hidden=128, mxfp_ratio=1.125

act_hbm_size = int(4 * 128 * 1.125) = 576
weight_hbm_size = int(128 * 128 * 1.125) = 18432

Assembly setup:
  S_ADDI_INT gp1, gp0, 576        ; weight base = 576 (NOT 512!)
  C_SET_ADDR_REG a1, gp0, gp1
```

**Common mistake:** Using `batch * hidden = 512` instead of `576` causes weight reads to fetch garbage data.

## Scale Register Setup

To make the prefetch process aware of MXFP layout, use `C_SET_SCALE_REG` to set the distance between data blocks and their scale factors:

```asm
; Before prefetching activations
S_ADDI_INT gp1, gp0, 512         ; logical size: batch * hidden
C_SET_SCALE_REG gp1

; Before prefetching weights
S_ADDI_INT gp1, gp0, 16384       ; logical size: hidden * hidden
C_SET_SCALE_REG gp1
```

The accelerator automatically converts MXFP data to the correct FP layout in on-chip SRAM.

## Stride Register Setup

Use `C_SET_STRIDE_REG` to set the row stride for matrix prefetch operations (typically `hidden_size`) and is not related to the MX data type.

---

# Test Environment Data Layout

In the test environment:
- **Activations are in HBM** at address 0. You MUST use `H_PREFETCH_V` to load them into Vector SRAM before use.
- **FP constants are pre-loaded** into FP_MEM (see `fp_sram_layout`).
- **Weights (if needed) are in HBM** after activations. Use `H_PREFETCH_M` to load them.

| Data | Location | Pre-loaded? | Action Required |
|------|----------|-------------|-----------------|
| Activations | HBM address 0 | No | Use `H_PREFETCH_V` to load to VRAM |
| Weights | HBM after activations | No | Use `H_PREFETCH_M` to load to MSRAM |
| FP constants | FP_MEM | Yes | Use `S_LD_FP` to load into FP registers |

**Important**: All activation and weight data must be explicitly prefetched from HBM before computation.

---

# Prefetch-Compute Pattern

The key principle for efficient computation is: **data must be in SRAM before it can be used**.

## Understanding SRAM as a Working Buffer

Think of Matrix SRAM and Vector SRAM as working buffers:
- `H_PREFETCH_M` copies data from HBM → Matrix SRAM at a specified SRAM address
- `H_PREFETCH_V` copies data from HBM → Vector SRAM at a specified SRAM address
- `M_MM` reads from SRAM addresses (not HBM!)

**Critical insight:** The first argument of `H_PREFETCH_*` is the **SRAM destination address**. If you prefetch multiple tiles, they must go to different SRAM addresses, otherwise they overwrite each other.

## Example: Why Multiple Prefetches Are Needed

For a matrix multiply that accumulates across K dimension:
- If K=128 and MLEN=64, you need 2 weight tiles (K/MLEN = 2)
- The inner loop does 2 M_MM operations reading from Matrix SRAM[0] and Matrix SRAM[4096]
- **Before** the inner loop, you must prefetch **both** tiles:
  ```
  H_PREFETCH_M to Matrix SRAM[0] from HBM[offset1]
  H_PREFETCH_M to Matrix SRAM[4096] from HBM[offset2]
  ```
- Then the inner loop reads from addresses that already have data

## Mental Model

Before writing any M_MM instruction, ask: "What SRAM address does this read from? When was data written there?"

If you can't trace back to a prefetch that wrote to that exact SRAM address, the data won't be there.

---

# Activation Prefetch Pattern (H_PREFETCH_V)

## HBM vs VRAM Layout Difference

Activations in HBM and VRAM use **different layouts**:

| Memory | Layout | Description |
|--------|--------|-------------|
| HBM | `[batch, hidden]` row-major | Each batch is a contiguous row of `hidden` elements |
| VRAM | `[hidden//VLEN, batch, VLEN]` | Tiled by VLEN along hidden dimension |

This means the **VRAM destination address** and **HBM source offset** are computed differently.

## H_PREFETCH_V Operands

`H_PREFETCH_V rd, rs1, rs2, rstride, precision`

| Operand | Meaning | Computed From |
|---------|---------|---------------|
| `rd` | VRAM destination address | VRAM tile index × batch × VLEN |
| `rs1` | HBM source offset | Column offset = tile index × VLEN |
| `rs2` | HBM base address register | Usually `a0` |
| `rstride` | Use STRIDE_REG if 1 | Set to 1 for multi-batch loads |

**Critical:** `rd` and `rs1` are independent. Do NOT use the same value for both.

## Stride Mode Behavior

With `rstride=1`, H_PREFETCH_V loads `HBM_V_Prefetch_Amount` (typically `batch`) consecutive chunks, each spaced by `STRIDE_REG`:

```
Starting at HBM[rs1], loads:
  HBM[rs1 + 0*STRIDE_REG : +VLEN]  → VRAM[rd + 0*VLEN]
  HBM[rs1 + 1*STRIDE_REG : +VLEN]  → VRAM[rd + 1*VLEN]
  HBM[rs1 + 2*STRIDE_REG : +VLEN]  → VRAM[rd + 2*VLEN]
  ...
```

For activations, set `STRIDE_REG = hidden` (the row stride in HBM).

## Address Calculation Formula

For activation `[batch, hidden]` split into `num_tiles = hidden // VLEN` tiles:

```
For tile j (j = 0, 1, ..., num_tiles-1):
  VRAM destination (rd) = j × batch × VLEN
  HBM offset (rs1)      = j × VLEN
  STRIDE_REG            = hidden
```

**Key insight:**
- VRAM address grows by `batch × VLEN` per tile (accounts for all batches)
- HBM offset grows by just `VLEN` per tile (column offset within each row)

## Common Mistake

```asm
; WRONG - using same value for rd and rs1
S_ADDI_INT gp4, gp0, 256
H_PREFETCH_V gp4, gp4, a0, 1, 0    ; rd=256, rs1=256 ← BUG!

; CORRECT - rd and rs1 are different
S_ADDI_INT gp3, gp0, 256           ; VRAM dest for tile 1
S_ADDI_INT gp4, gp0, 64            ; HBM offset for column slice 1
H_PREFETCH_V gp3, gp4, a0, 1, 0    ; rd=256, rs1=64 ✓
```

---

# Vector Operation Loop Requirement

## Prefetch vs Compute Granularity

`H_PREFETCH_V` and `V_*` instructions operate at different granularities:

| Operation | Granularity |
|-----------|-------------|
| `H_PREFETCH_V` | `HBM_V_Prefetch_Amount × VLEN` elements |
| `V_*` instructions | `VLEN` elements |

After prefetching N elements, you need `N / VLEN` vector operations to process all data.

## Loop Pattern

```asm
; After prefetch loads N elements to VRAM starting at base_addr:
S_ADDI_INT gp1, gp0, <base_addr>
C_LOOP_START gp2, <N / VLEN>
  V_<op> gp1, ...                  ; process VLEN elements
  S_ADDI_INT gp1, gp1, <VLEN>      ; advance pointer
C_LOOP_END gp2
```

Without the loop, only the first VLEN elements are processed.