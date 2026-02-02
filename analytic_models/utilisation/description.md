# Resource Utilisation Cost Model

## Components to Be Synthesised Individually
- Matrix Machine
- Matrix SRAM
- Vector Machine
- Vector SRAM
- Scalar Machine
- HBM System
- Control (including Pipeline Control and Dataflow Control, which are fixed)

---

## Matrix Machine
**Relevant Parameters:**
- `MLEN`, `BLEN`: Define the size of the systolic MCU. Resource utilisation for the systolic array computation is expected to scale with `MLEN * BLEN`.
- `WT_MX_MANT_WIDTH`, `WT_MX_EXP_WIDTH`, `ACT_MXFP_MANT_WIDTH`, `ACT_MXFP_EXP_WIDTH`: Define the bitwidth of the left and top inputs of the systolic array.
- `MXFP_SCALE_WIDTH`: Determines the bitwidth of the MXFP scale input.
- `BLOCK_DIM`: Controls the number of blocks (`MLEN / BLOCK_DIM`) in the systolic array, which also defines the number of input and output ports for MXFP scale.
- `V_FP_EXP_WIDTH`, `V_FP_MANT_WIDTH`: Define the bitwidth of the systolic array output.
- `M_FP_EXP_WIDTH`, `M_FP_MANT_WIDTH`: Define the bitwidth for accumulation and the adder tree within the systolic array.

---

## Matrix SRAM
**Relevant Parameters:**
- `MLEN`
- `WT_MX_EXP_WIDTH`, `WT_MX_MANT_WIDTH`, `MXFP_SCALE_WIDTH`: Define the bitwidth of the data stored in the Matrix SRAM.
- `BLOCK_DIM`: Can be ignored for SRAM sizing, as MXFP scales will be duplicated to match `MLEN`.
- `SRAM_DEPTH`: Defines the depth of the Matrix SRAM.

---

## Vector Machine
**Relevant Parameters:**
- `VLEN`: Currently set equal to `MLEN`.
- `V_FP_EXP_WIDTH`, `V_FP_MANT_WIDTH`: Define the bitwidth of the data processed by the Vector Machine.

---

## Vector SRAM
**Relevant Parameters:**
- `VLEN`: Currently set equal to `MLEN`.
- `V_FP_EXP_WIDTH`, `V_FP_MANT_WIDTH`: Define the bitwidth for storage in the Vector SRAM.
- `WT_MX_MANT_WIDTH`, `WT_MX_EXP_WIDTH`, `ACT_MXFP_MANT_WIDTH`, `ACT_MXFP_EXP_WIDTH`, `KV_MX_EXP_WIDTH`, `KV_MX_MANT_WIDTH`: These parameters have only minor impact on resource utilisation for Vector SRAM, as they determine the input/output interface and will all be converted to the standard `V_FP_EXP_WIDTH` and `V_FP_MANT_WIDTH` bitwidths in SRAM.

---

## Scalar Machine
**Relevant Parameters:**
- `INT_DATA_WIDTH`: Defines the bitwidth of integer data processed by the Scalar Machine.
- `S_FP_EXP_WIDTH`, `S_FP_MANT_WIDTH`: Define the bitwidth of floating-point data.
- `FP_SRAM_WIDTH`: Depth of the floating-point SRAM.
- `INT_SRAM_WIDTH`: Depth of the fixed-point SRAM.

---

## HBM System 
(Might not need to be systematically analysed)

**Relevant Parameters:**
- `HBM_ELE_WIDTH`: Defines the MXFP block bitwidth for HBM data.
- `HBM_SCALE_WIDTH`: Defines the MXFP scale bitwidth for HBM data.
