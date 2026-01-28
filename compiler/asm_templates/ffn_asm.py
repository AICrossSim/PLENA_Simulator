import os
from typing import Dict, List, Any, Optional
from pathlib import Path
import math
IMM2_BOUND = 2**18

def ffn_asm(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,

    alive_registers: List[int],
    gate_weight_hbm_offset_reg: int,
    up_weight_hbm_offset_reg: int,
    down_weight_hbm_offset_reg: int,
    const_one_fp_address: int,

    activation_base_address: int,
    use_loop_instructions: bool = False,
    use_fused_up_gate: bool = False
) -> str:
    """
    Generates assembly code for a FFN operation.

    Set use_loop_instructions=True to use C_LOOP_START/END for compact code.
    Set use_fused_up_gate=True to fuse upsize and gate projections (requires 12 registers).
    """
    if use_fused_up_gate:
        return _ffn_asm_fused_up_gate(
            mlen, vlen, blen, batch, seq_len, hidden_size, intermediate_size,
            alive_registers, gate_weight_hbm_offset_reg, up_weight_hbm_offset_reg,
            down_weight_hbm_offset_reg, const_one_fp_address, activation_base_address
        )
    elif use_loop_instructions:
        return _ffn_asm_with_loops(
            mlen, vlen, blen, batch, seq_len, hidden_size, intermediate_size,
            alive_registers, gate_weight_hbm_offset_reg, up_weight_hbm_offset_reg,
            down_weight_hbm_offset_reg, const_one_fp_address, activation_base_address
        )
    else:
        return _ffn_asm_unrolled(
            mlen, vlen, blen, batch, seq_len, hidden_size, intermediate_size,
            alive_registers, gate_weight_hbm_offset_reg, up_weight_hbm_offset_reg,
            down_weight_hbm_offset_reg, const_one_fp_address, activation_base_address
        )


def _ffn_asm_unrolled(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,

    alive_registers: List[int],
    gate_weight_hbm_offset_reg: int,
    up_weight_hbm_offset_reg: int,
    down_weight_hbm_offset_reg: int,
    const_one_fp_address: int,
    
    activation_base_address: int
) -> str:
    """
    Generates assembly code for a FFN operation.

    Args:
        mlen (int): The number of rows in the matrix.
        vlen (int): The number of columns in the matrix.
        blen (int): The number of columns in the second matrix.
        batch (int): The number of batches.
        hidden_size (int): The number of rows in the hidden size.
        intermediate_size (int): The number of rows in the intermediate size.
        alive_registers (List[int]): List of registers that are alive.
        gate_weight_hbm_offset_reg (int): index for the address mapper pointing to the base addr of the gate weight matrix.
        up_weight_hbm_offset_reg (int): index for the address mapper pointing to the base addr of the up weight matrix.
        down_weight_hbm_offset_reg (int): index for the address mapper pointing to the base addr of the down weight matrix.
        activation_base_address (int): index for the address mapper pointing to the base addr of the activation matrix.
        result_base_address (int): index for the address mapper pointing to the base addr of the result matrix.
    Functionality:
        Upsize linear   (b, s, hidden_size) @ (hidden_size, intermediate_size) - > (b, s, intermediate_size)
        Gate Projection (b, s, hidden_size) @ (hidden_size, intermediate_size) -> (b, s, intermediate_size)
        SILU Activation (b, s, intermediate_size) -> (b, s, intermediate_size)
        Downsize linear (b, s, intermediate_size) @ (intermediate_size, hidden_size) -> (b, s, hidden_size)
    """
    

    # memory assignment
    # 0 -> activation
    # b * s * hidden_size -> upsize intermediate results
    # b * s * (hidden_size + intermediate_size) -> gate projection results

    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    a_actual_register = alive_registers[2]
    up_result_register = alive_registers[3]
    intermediate_register = alive_registers[4]
    gate_result_register = alive_registers[5]
    w_hbm_offset_register = alive_registers[6]
    m_stride_register = alive_registers[7]

    # reset the registers
    generated_code = "; FFN Generation \n"

    # Settings for up and gate weight matrices prefetching
    assert hidden_size * intermediate_size < IMM2_BOUND, f"hidden_size * hidden_size must be less than {IMM2_BOUND}"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size * intermediate_size} \n"
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {intermediate_size} \n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
    assert hidden_size * batch * seq_len < IMM2_BOUND, f"hidden_size * batch * seq_len must be less than {IMM2_BOUND}"
    # Set the address for on-chip sram
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size} \n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{up_result_register}, {intermediate_size * batch * seq_len} \n"

    generated_code += " ; FFN Upsize Linear Generation \n"
    for weight_row in range (intermediate_size // blen):
        if weight_row % (mlen // blen) == 0:
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
            generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, {weight_row * blen} \n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0 \n"
            
            for weight_col in range (hidden_size // mlen):
                generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{w_hbm_offset_register}, a{up_weight_hbm_offset_reg}, 1, 0 \n"
                generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} \n"
                generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen * intermediate_size} \n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
        else:
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {(weight_row % (mlen // blen)) * blen} \n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, {(weight_row % (mlen // blen)) * blen} \n"
        for act_col in range ((batch * seq_len) // blen):
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address + act_col * mlen * blen} \n"
            generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0 \n"
            for inner_loop_index in range (hidden_size // mlen):
                generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register} \n"
                generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen} \n"
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len} \n"
            generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0 \n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen} \n"    # generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {activation_base_address} \n"
        if (weight_row + 1) % (mlen // blen) == 0 and weight_row != intermediate_size // blen - 1:
            generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len} \n"

    generated_code += " ; FFN Gate Projection Generation \n"
    for weight_row in range (intermediate_size // blen):
        if weight_row % (mlen // blen) == 0:
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
            generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, {weight_row * blen} \n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0 \n"
            
            for weight_col in range (hidden_size // mlen):
                generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{w_hbm_offset_register}, a{gate_weight_hbm_offset_reg}, 1, 0 \n"
                generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} \n"
                generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen * intermediate_size} \n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
        else:
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {(weight_row % (mlen // blen)) * blen} \n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, {(weight_row % (mlen // blen)) * blen} \n"
        for act_col in range ((batch * seq_len) // blen):
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address + act_col * mlen * blen} \n"
            generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0 \n"
            for inner_loop_index in range (hidden_size // mlen):
                generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register} \n"
                generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen} \n"
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len} \n"
            generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0 \n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen} \n"    # generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {activation_base_address} \n"
        if (weight_row + 1) % (mlen // blen) == 0 and weight_row != intermediate_size // blen - 1:
            generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {mlen * batch * seq_len} \n"


    # Intermediate Dim SILU Activation Generation, now x in shape of (b, s, intermediate_size)
    generated_code += "; SILU Generation \n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address} \n"
    # Reset the addr for up and gate result
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size} \n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{up_result_register}, {intermediate_size * batch * seq_len} \n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp0, {activation_base_address} \n" 
    
    # Treat the original activation region as the place for scratchpad.
    for b in range(batch * seq_len):
        for i in range(intermediate_size // vlen):
            # 0 : -x
            generated_code += f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1 \n"
            # 1 : exp(-x)
            generated_code += f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0 \n"
            # 2 : 1 + exp(-x)
            generated_code += f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0 \n"
            # 3 : 1 / (1 + exp(-x))
            generated_code += f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0 \n"
            # 4 : (1 / (1 + exp(-x))) * gate_result
            generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0 \n"
            # 5: multiply by gate result and store to the up result region
            generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0 \n"
            generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen} \n"
            generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen} \n"

    generated_code += "; FFN Downsize Linear Generation \n"
    # Reset the addr for up and gate result
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size * intermediate_size} \n"
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size} \n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register} \n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
    generated_code += f"S_ADDI_INT gp{m_stride_register}, gp0, {((batch * seq_len) // blen)} \n"
    # Storing the results to the activation base region
    act_result_register = gate_result_register
    generated_code += f"S_ADDI_INT gp{act_result_register}, gp0, {activation_base_address} \n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size} \n"
    for weight_row in range (hidden_size // blen):
        if weight_row % (mlen // blen) == 0:
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
            generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, {weight_row * blen} \n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0 \n"
            for weight_col in range (intermediate_size // mlen):
                generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{w_hbm_offset_register}, a{down_weight_hbm_offset_reg}, 1, 0 \n"
                generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen} \n"
                generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen * hidden_size} \n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0 \n"
        else:
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {(weight_row % (mlen // blen)) * blen} \n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, {(weight_row % (mlen // blen)) * blen} \n"
        for act_col in range (batch * seq_len // blen):
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {act_col * mlen * blen} \n"
            generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0 \n"
            for inner_loop_index in range (intermediate_size // mlen):
                generated_code += f"M_MM 0, gp{w_actual_register}, gp{a_actual_register} \n"
                generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen} \n"
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len} \n"
            generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0 \n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {mlen * blen} \n"    # generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {activation_base_address} \n"
        if (weight_row + 1) % (mlen // blen) == 0 and weight_row != intermediate_size // blen - 1:
            generated_code += f"S_ADDI_INT gp{act_result_register}, gp{act_result_register}, {mlen * batch * seq_len} \n"
    return generated_code


def ffn_up_silu_asm(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,

    alive_registers: List[int],
    up_weight_hbm_offset_reg: int,
    const_one_fp_address: int,

    activation_base_address: int
) -> str:
    """
    Generates assembly code for up projection + SILU activation only.
    
    Computes: SILU(up_proj(x)) = silu(w1(x))
    Stops before gate projection and down projection.
    Uses loop instructions for compact code.
    """
    # Register allocation
    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    a_actual_register = alive_registers[2]
    up_result_register = alive_registers[3]
    intermediate_register = alive_registers[4]
    w_hbm_offset_register = alive_registers[5]

    # Need extra registers for loop counters and temp save
    assert len(alive_registers) >= 10, "Loop version requires 10 registers (9 minimum + 1 temp)"
    loop_outer_reg = alive_registers[6]
    loop_inner_reg = alive_registers[7]
    loop_inner2_reg = alive_registers[8]
    temp_save_reg = alive_registers[9]  # Use this as temp save for a_actual_register

    generated_code = "; FFN Up Projection + SILU Generation\n"

    # === SETUP PHASE ===
    assert hidden_size * intermediate_size < IMM2_BOUND
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size * intermediate_size}\n"
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {intermediate_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base address for up result
    assert hidden_size * batch * seq_len < IMM2_BOUND
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"

    # === UPSIZE LINEAR (loop version) ===
    generated_code += "; FFN Upsize Linear Generation (Loop)\n"

    # Outer loop: weight_row from 0 to intermediate_size // mlen (MLEN blocks)
    num_mlen_blocks = intermediate_size // mlen
    tiles_per_mlen = mlen // blen
    num_weight_tiles = hidden_size // mlen
    num_act_cols = (batch * seq_len) // blen

    # w_hbm_offset_register tracks the START offset for each MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * intermediate_size}\n"

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Inner loop: act_col iterations
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    # Copy weight pointer
    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual_register value before inner accumulation modifies it
    generated_code += f"S_ADDI_INT gp{temp_save_reg}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    # Write output
    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    
    # Restore a_actual_register and advance to next activation column
    act_col_advance = mlen * blen
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{temp_save_reg}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance up_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # # === SILU ACTIVATION (loop version) ===
    # generated_code += "; SILU Generation (Loop)\n"
    # generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"

    # # Reset addresses - up_result_register points to up projection results
    # generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    # generated_code += f"S_ADDI_INT gp{intermediate_register}, gp0, {activation_base_address}\n"  # Use activation_base_address as scratchpad

    # Loop over batch * seq_len * (intermediate_size // vlen)
    # num_silu_iters = batch * seq_len * (intermediate_size // vlen)
    # generated_code += f"; SILU loop: {num_silu_iters} iterations\n"
    # generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_silu_iters}\n"

    # # SILU computation: sigmoid(x) * x (no gate multiplication)
    # # Step 1: -x (negate)
    # generated_code += f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1\n"
    # # Step 2: exp(-x)
    # generated_code += f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    # # Step 3: 1 + exp(-x)
    # generated_code += f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0\n"
    # # Step 4: 1 / (1 + exp(-x)) = sigmoid(x)
    # generated_code += f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    # # Step 5: sigmoid(x) * x = silu(x), store in-place in up_result_register
    # generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{up_result_register}, 0\n"
    # generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen}\n"

    # generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # Note: Result is stored in up_result_register at base address batch * seq_len * hidden_size
    generated_code += "; Result (up projection only) stored at up_result_register location\n"

    return generated_code


def ffn_intermediate_asm(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,

    alive_registers: List[int],
    gate_weight_hbm_offset_reg: int,
    up_weight_hbm_offset_reg: int,
    const_one_fp_address: int,

    activation_base_address: int
) -> str:
    """
    Generates assembly code for FFN intermediate operations (up + gate + SILU only).
    
    Stops before down projection to allow checking intermediate results.
    Uses loop instructions for compact code.
    """
    # Register allocation
    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    a_actual_register = alive_registers[2]
    up_result_register = alive_registers[3]
    intermediate_register = alive_registers[4]
    gate_result_register = alive_registers[5]
    w_hbm_offset_register = alive_registers[6]

    # Need extra registers for loop counters
    assert len(alive_registers) >= 10, "Loop version requires 10 registers"
    loop_outer_reg = alive_registers[7]
    loop_inner_reg = alive_registers[8]
    loop_inner2_reg = alive_registers[9]

    generated_code = "; FFN Intermediate Generation (Up + Gate + SILU only)\n"

    # === SETUP PHASE ===
    assert hidden_size * intermediate_size < IMM2_BOUND
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size * intermediate_size}\n"
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {intermediate_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base addresses for results
    assert hidden_size * batch * seq_len < IMM2_BOUND
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{up_result_register}, {intermediate_size * batch * seq_len}\n"

    # === UPSIZE LINEAR (loop version) ===
    generated_code += "; FFN Upsize Linear Generation (Loop)\n"

    # Outer loop: weight_row from 0 to intermediate_size // mlen (MLEN blocks)
    num_mlen_blocks = intermediate_size // mlen
    tiles_per_mlen = mlen // blen
    num_weight_tiles = hidden_size // mlen
    num_act_cols = (batch * seq_len) // blen

    # w_hbm_offset_register tracks the START offset for each MLEN block
    # It starts at 0 and increments by mlen after each outer loop iteration
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    # Use a_actual_register temporarily to track running HBM offset during prefetch
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * intermediate_size}\n"

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Inner loop: act_col iterations - track activation column base in a_actual_register
    # Reset activation base at start of each middle loop iteration
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    # Copy weight and activation pointers, iterate through weight tiles
    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual_register value before inner accumulation modifies it (use gate_result_register as temp)
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    act_col_advance = mlen * blen
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{gate_result_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance up_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # === GATE PROJECTION (loop version) ===
    generated_code += "; FFN Gate Projection Generation (Loop)\n"

    # Reset base addresses
    # up_result_register = where upsize results start
    # gate_result_register = where gate results should be written (after upsize results)
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{up_result_register}, {intermediate_size * batch * seq_len}\n"
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"

    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    # Use a_actual_register temporarily to track running HBM offset during prefetch
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{gate_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * intermediate_size}\n"

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Inner loop: act_col iterations - track activation column base in a_actual_register
    # Reset activation base at start of each middle loop iteration
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    # Copy weight and activation pointers, iterate through weight tiles
    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual_register value before inner accumulation modifies it (use up_result_register as temp)
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance gate_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # === SILU ACTIVATION (loop version) ===
    generated_code += "; SILU Generation (Loop)\n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"

    # Reset addresses
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{up_result_register}, {intermediate_size * batch * seq_len}\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp0, {activation_base_address}\n"

    # Loop over batch * seq_len * (intermediate_size // vlen)
    num_silu_iters = batch * seq_len * (intermediate_size // vlen)
    generated_code += f"; SILU loop: {num_silu_iters} iterations\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_silu_iters}\n"

    # SILU computation: sigmoid(x) * x * gate
    generated_code += f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1\n"
    generated_code += f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    generated_code += f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0\n"
    generated_code += f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0\n"
    generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen}\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # Note: Result is stored in up_result_register at base address batch * seq_len * hidden_size
    generated_code += "; Intermediate result (up + gate + SILU) stored at up_result_register location\n"

    return generated_code


def _ffn_asm_with_loops(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,

    alive_registers: List[int],
    gate_weight_hbm_offset_reg: int,
    up_weight_hbm_offset_reg: int,
    down_weight_hbm_offset_reg: int,
    const_one_fp_address: int,

    activation_base_address: int
) -> str:
    """
    Generates assembly code for FFN using C_LOOP_START/END instructions.

    Uses nested loops to reduce code size at cost of some latency overhead.
    """

    # Register allocation
    w_actual_register = alive_registers[0]
    w_temp_register = alive_registers[1]
    a_actual_register = alive_registers[2]
    up_result_register = alive_registers[3]
    intermediate_register = alive_registers[4]
    gate_result_register = alive_registers[5]
    w_hbm_offset_register = alive_registers[6]

    # Need extra registers for loop counters
    assert len(alive_registers) >= 10, "Loop version requires 10 registers"
    loop_outer_reg = alive_registers[7]
    loop_inner_reg = alive_registers[8]
    loop_inner2_reg = alive_registers[9]

    generated_code = "; FFN Generation (Loop-Optimized)\n"

    # === SETUP PHASE ===
    assert hidden_size * intermediate_size < IMM2_BOUND
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size * intermediate_size}\n"
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {intermediate_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base addresses for results
    assert hidden_size * batch * seq_len < IMM2_BOUND
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{up_result_register}, {intermediate_size * batch * seq_len}\n"

    # === UPSIZE LINEAR (loop version) ===
    generated_code += "; FFN Upsize Linear Generation (Loop)\n"

    # Outer loop: weight_row from 0 to intermediate_size // mlen (MLEN blocks)
    num_mlen_blocks = intermediate_size // mlen
    tiles_per_mlen = mlen // blen
    num_weight_tiles = hidden_size // mlen
    num_act_cols = (batch * seq_len) // blen

    # w_hbm_offset_register tracks the START offset for each MLEN block
    # It starts at 0 and increments by mlen after each outer loop iteration
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    # Use a_actual_register temporarily to track running HBM offset during prefetch
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * intermediate_size}\n"

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Inner loop: act_col iterations - track activation column base in a_actual_register
    # Reset activation base at start of each middle loop iteration
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    # Copy weight and activation pointers, iterate through weight tiles
    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual_register value before inner accumulation modifies it (use gate_result_register as temp)
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    act_col_advance = mlen * blen
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{gate_result_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance up_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # === GATE PROJECTION (loop version) ===
    generated_code += "; FFN Gate Projection Generation (Loop)\n"

    # Reset base addresses
    # up_result_register = where upsize results start
    # gate_result_register = where gate results should be written (after upsize results)
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{up_result_register}, {intermediate_size * batch * seq_len}\n"
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"

    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    # Use a_actual_register temporarily to track running HBM offset during prefetch
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{gate_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * intermediate_size}\n"

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Inner loop: act_col iterations - track activation column base in a_actual_register
    # Reset activation base at start of each middle loop iteration
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"; Inner loop: {num_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    # Copy weight and activation pointers, iterate through weight tiles
    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual_register value before inner accumulation modifies it (use up_result_register as temp)
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance gate_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # === SILU ACTIVATION (loop version) ===
    generated_code += "; SILU Generation (Loop)\n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"

    # Reset addresses
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{up_result_register}, {intermediate_size * batch * seq_len}\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp0, {activation_base_address}\n"

    # Loop over batch * seq_len * (intermediate_size // vlen)
    num_silu_iters = batch * seq_len * (intermediate_size // vlen)
    generated_code += f"; SILU loop: {num_silu_iters} iterations\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_silu_iters}\n"

    # SILU computation: sigmoid(x) * x * gate
    generated_code += f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1\n"
    generated_code += f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    generated_code += f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0\n"
    generated_code += f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
    generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0\n"
    generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen}\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # === DOWNSIZE LINEAR (loop version) ===
    generated_code += "; FFN Downsize Linear Generation (Loop)\n"

    # Setup scale and stride for downsize
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size * intermediate_size}\n"
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Result goes to activation base region
    act_result_register = gate_result_register
    generated_code += f"S_ADDI_INT gp{act_result_register}, gp0, {activation_base_address}\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"

    # Downsize: (b*s, intermediate_size) @ (intermediate_size, hidden_size) -> (b*s, hidden_size)
    num_down_mlen_blocks = hidden_size // mlen
    num_down_weight_tiles = intermediate_size // mlen
    down_act_col_advance = mlen * blen

    # w_hbm_offset_register tracks the START offset for each MLEN block
    # It starts at 0 and increments by mlen after each outer loop iteration
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_down_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_down_mlen_blocks}\n"

    # Prefetch weights for this MLEN block
    # Use a_actual_register temporarily to track running HBM offset during prefetch
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
    for weight_col in range(num_down_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{down_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * hidden_size}\n"

    # Reset for compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"

    # Middle loop: tiles within MLEN block
    generated_code += f"; Middle loop: {tiles_per_mlen} tiles per MLEN block\n"
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    # Inner loop: act_col iterations - track activation column base in a_actual_register
    # Reset activation base at start of each middle loop iteration
    # Note: up_result_register will be used as temp save in inner loop, so recompute it here
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, 0\n"
    num_down_act_cols = (batch * seq_len) // blen
    generated_code += f"; Inner loop: {num_down_act_cols} activation columns\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_down_act_cols}\n"

    # Copy weight and activation pointers
    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save a_actual_register value before inner accumulation modifies it
    # Use up_result_register as temp save (it's recomputed at each middle loop iteration start)
    down_act_save_reg = up_result_register
    generated_code += f"S_ADDI_INT gp{down_act_save_reg}, gp{a_actual_register}, 0\n"

    # Innermost accumulation (unrolled)
    for inner_loop_index in range(num_down_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_loop_index < num_down_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore a_actual_register and advance to next activation column
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{down_act_save_reg}, {down_act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    # After inner loop: advance weight offset within MLEN block
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    # Reset intermediate back for next tile row
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
    # Add offset for current tile
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # After middle loop: advance w_hbm_offset_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    # Advance act_result_register for next MLEN block
    generated_code += f"S_ADDI_INT gp{act_result_register}, gp{act_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    return generated_code


def _ffn_asm_fused_up_gate(
    mlen: int,
    vlen: int,
    blen: int,
    batch: int,
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,

    alive_registers: List[int],
    gate_weight_hbm_offset_reg: int,
    up_weight_hbm_offset_reg: int,
    down_weight_hbm_offset_reg: int,
    const_one_fp_address: int,

    activation_base_address: int
) -> str:
    """
    Optimized FFN: Fuses upsize and gate projections to reduce HBM prefetch overhead.

    Key optimization: Both up and gate weights are prefetched into MRAM for each MLEN block,
    then both projections are computed before moving to the next block. This reduces
    the number of times we need to refetch weights.

    Requires 12 registers for optimal performance.
    """

    # Register allocation for fused version
    assert len(alive_registers) >= 12, "Fused version requires 12 registers"

    w_actual_register = alive_registers[0]         # Weight MRAM offset (shared)
    w_temp_register = alive_registers[1]           # Weight temp pointer
    a_actual_register = alive_registers[2]         # Activation VRAM pointer
    up_result_register = alive_registers[3]        # Upsize result base
    intermediate_register = alive_registers[4]     # Output write pointer
    gate_result_register = alive_registers[5]      # Gate result base
    w_hbm_offset_register = alive_registers[6]     # HBM block offset for prefetch
    loop_outer_reg = alive_registers[7]            # Outer loop counter
    loop_inner_reg = alive_registers[8]            # Middle loop counter
    loop_inner2_reg = alive_registers[9]           # Inner loop counter
    # Extra registers for fused version
    a_save_register = alive_registers[10]          # Activation save
    w_gate_base_register = alive_registers[11]     # Gate weight base in MRAM

    generated_code = "; FFN Generation (Fused Up+Gate Optimized)\n"

    # === SETUP PHASE ===
    assert hidden_size * intermediate_size < IMM2_BOUND
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size * intermediate_size}\n"
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {intermediate_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"

    # Set base addresses for results
    assert hidden_size * batch * seq_len < IMM2_BOUND
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{up_result_register}, {intermediate_size * batch * seq_len}\n"

    # === FUSED UP + GATE LINEAR with overlapped prefetch ===
    generated_code += "; Fused Up+Gate Linear (overlapped prefetch optimization)\n"

    num_mlen_blocks = intermediate_size // mlen
    tiles_per_mlen = mlen // blen
    num_weight_tiles = hidden_size // mlen
    num_act_cols = (batch * seq_len) // blen
    act_col_advance = mlen * blen
    gate_mram_offset = num_weight_tiles * mlen * mlen  # Gate weights start after up weights in MRAM

    # Calculate how to spread GATE prefetches across UP computation
    # UP projection has tiles_per_mlen * num_act_cols iterations of inner work
    total_up_inner_iters = tiles_per_mlen * num_act_cols
    gate_prefetch_interval = max(1, total_up_inner_iters // num_weight_tiles)

    # HBM offset tracking
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, 0\n"
    generated_code += f"; Outer loop: {num_mlen_blocks} MLEN blocks\n"
    generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_mlen_blocks}\n"

    # Prefetch UP weights only (GATE will be prefetched during UP compute)
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"

    # Prefetch up weights (to MRAM at offset 0)
    for weight_col in range(num_weight_tiles):
        generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{up_weight_hbm_offset_reg}, 1, 0\n"
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * intermediate_size}\n"

    # Setup for UP compute and GATE prefetch overlap
    generated_code += f"S_ADDI_INT gp{w_gate_base_register}, gp0, {gate_mram_offset}\n"

    # Reset for UP compute phase
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"

    # === UP PROJECTION with interleaved GATE prefetch ===
    generated_code += f"; Up projection for MLEN block (with GATE prefetch every {gate_prefetch_interval} iters)\n"

    # Unroll to interleave GATE prefetches during UP computation
    # NOTE: We compute GATE HBM offset directly from w_hbm_offset_register + offset
    # instead of tracking it in a_save_register (which is reused for weight offset in inner loop)
    gate_prefetch_count = 0
    gate_mram_ptr = gate_mram_offset

    for tile_idx in range(tiles_per_mlen):
        # Reset activation base for this tile
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"

        for act_col in range(num_act_cols):
            iter_num = tile_idx * num_act_cols + act_col

            # Check if we should insert a GATE prefetch
            if iter_num % gate_prefetch_interval == 0 and gate_prefetch_count < num_weight_tiles:
                generated_code += f"; Prefetch GATE weight tile {gate_prefetch_count} during UP compute\n"
                # Save current a_actual_register (activation pointer) to w_temp_register
                generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{a_actual_register}, 0\n"
                # Compute GATE HBM offset directly: base_offset + prefetch_count * stride
                gate_hbm_offset = gate_prefetch_count * mlen * intermediate_size
                # Set MRAM destination
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {gate_mram_ptr}\n"
                # Set HBM source: w_hbm_offset_register + gate_hbm_offset
                generated_code += f"S_ADDI_INT gp{a_save_register}, gp{w_hbm_offset_register}, {gate_hbm_offset}\n"
                generated_code += f"H_PREFETCH_M gp{a_actual_register}, gp{a_save_register}, a{gate_weight_hbm_offset_reg}, 1, 0\n"
                gate_mram_ptr += mlen * mlen
                gate_prefetch_count += 1
                # Restore activation pointer
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_temp_register}, 0\n"

            # Save activation column base before weight tile loop modifies a_actual_register
            generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{a_actual_register}, 0\n"

            # UP weight accumulation
            generated_code += f"S_ADDI_INT gp{a_save_register}, gp{w_actual_register}, 0\n"  # save weight offset

            for inner_idx in range(num_weight_tiles):
                generated_code += f"M_MM 0, gp{a_save_register}, gp{a_actual_register}\n"
                generated_code += f"S_ADDI_INT gp{a_save_register}, gp{a_save_register}, {mlen * mlen}\n"
                if inner_idx < num_weight_tiles - 1:
                    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

            generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
            generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"

            # Restore activation and advance to next column
            if act_col < num_act_cols - 1:
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_temp_register}, {act_col_advance}\n"

        # After all act_cols for this tile, advance weight offset
        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
        generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{up_result_register}, 0\n"
        generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    # === GATE PROJECTION for this block (weights already prefetched) ===
    generated_code += f"; Gate projection for MLEN block (weights pre-fetched during UP)\n"
    generated_code += f"S_ADDI_INT gp{a_save_register}, gp0, 0\n"  # tile offset tracker for output
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_gate_base_register}, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"

    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen}\n"

    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, {activation_base_address}\n"
    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    # Save activation pointer - use w_gate_base_register
    generated_code += f"S_ADDI_INT gp{w_gate_base_register}, gp{a_actual_register}, 0\n"

    for inner_idx in range(num_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_idx < num_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    # Restore activation from saved base + advance to next column
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_gate_base_register}, {act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    generated_code += f"S_ADDI_INT gp{a_save_register}, gp{a_save_register}, {blen}\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{gate_result_register}, 0\n"
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{a_save_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # Advance for next MLEN block
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {mlen * batch * seq_len}\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {mlen * batch * seq_len}\n"

    generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    # === SILU ACTIVATION with overlapped DOWN weight prefetch ===
    # Key optimization: Prefetch first block of DOWN weights DURING SILU computation
    num_down_mlen_blocks = hidden_size // mlen
    num_down_weight_tiles = intermediate_size // mlen
    num_silu_iters = batch * seq_len * (intermediate_size // vlen)

    generated_code += "; SILU Generation (with overlapped DOWN prefetch)\n"
    generated_code += f"S_LD_FP f1, gp0, {const_one_fp_address}\n"

    # Set up DOWN weight prefetch parameters
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size * intermediate_size}\n"
    generated_code += f"C_SET_SCALE_REG gp{w_actual_register}\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, {hidden_size}\n"
    generated_code += f"C_SET_STRIDE_REG gp{w_actual_register}\n"

    # Initialize SILU pointers
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{up_result_register}, {intermediate_size * batch * seq_len}\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp0, {activation_base_address}\n"

    # Initialize DOWN prefetch pointers (w_actual_register=MRAM offset, a_actual_register=HBM offset)
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp0, 0\n"

    # Compute how many SILU iters per prefetch op to spread prefetches across SILU loop
    # We have num_down_weight_tiles prefetches to do for the first block
    # Spread them evenly across the SILU loop
    prefetch_interval = max(1, num_silu_iters // num_down_weight_tiles)

    generated_code += f"; SILU loop: {num_silu_iters} iterations (prefetch every {prefetch_interval} iters)\n"
    generated_code += f"; Prefetching {num_down_weight_tiles} DOWN weight tiles during SILU\n"

    # Unroll SILU loop to interleave prefetch operations
    for silu_iter in range(num_silu_iters):
        # SILU computation
        generated_code += f"V_SUB_VF gp{intermediate_register}, gp{up_result_register}, f0, 0, 1\n"
        generated_code += f"V_EXP_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
        generated_code += f"V_ADD_VF gp{intermediate_register}, gp{intermediate_register}, f1, 0\n"
        generated_code += f"V_RECI_V  gp{intermediate_register}, gp{intermediate_register}, 0\n"
        generated_code += f"V_MUL_VV gp{intermediate_register}, gp{intermediate_register}, gp{up_result_register}, 0\n"
        generated_code += f"V_MUL_VV gp{up_result_register}, gp{intermediate_register}, gp{gate_result_register}, 0\n"
        generated_code += f"S_ADDI_INT gp{gate_result_register}, gp{gate_result_register}, {vlen}\n"
        generated_code += f"S_ADDI_INT gp{up_result_register}, gp{up_result_register}, {vlen}\n"

        # Insert prefetch at appropriate intervals
        prefetch_idx = silu_iter // prefetch_interval
        if silu_iter % prefetch_interval == 0 and prefetch_idx < num_down_weight_tiles:
            generated_code += f"; Prefetch DOWN weight tile {prefetch_idx}\n"
            generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{down_weight_hbm_offset_reg}, 1, 0\n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * hidden_size}\n"

    # === DOWNSIZE LINEAR (first block already prefetched) ===
    generated_code += "; FFN Downsize Linear Generation (first block pre-fetched during SILU)\n"

    act_result_register = gate_result_register
    generated_code += f"S_ADDI_INT gp{act_result_register}, gp0, {activation_base_address}\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"

    down_act_col_advance = mlen * blen

    # First block: weights already prefetched, just do computation
    generated_code += f"; First DOWN block (weights pre-fetched during SILU)\n"
    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp0, {mlen}\n"  # Next block HBM offset

    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
    tiles_per_mlen_down = mlen // blen

    # First block computation
    generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen_down}\n"

    generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, 0\n"
    num_down_act_cols = (batch * seq_len) // blen

    generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_down_act_cols}\n"

    generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
    generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

    for inner_idx in range(num_down_weight_tiles):
        generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
        if inner_idx < num_down_weight_tiles - 1:
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

    generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
    generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {down_act_col_advance}\n"

    generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

    generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
    generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
    generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

    generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

    # Advance to second block base
    generated_code += f"S_ADDI_INT gp{act_result_register}, gp{act_result_register}, {mlen * batch * seq_len}\n"

    # Remaining blocks (if any) - standard prefetch then compute
    if num_down_mlen_blocks > 1:
        generated_code += f"; Remaining {num_down_mlen_blocks - 1} DOWN blocks\n"
        generated_code += f"C_LOOP_START gp{loop_outer_reg}, {num_down_mlen_blocks - 1}\n"

        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{w_hbm_offset_register}, 0\n"
        for weight_col in range(num_down_weight_tiles):
            generated_code += f"H_PREFETCH_M gp{w_actual_register}, gp{a_actual_register}, a{down_weight_hbm_offset_reg}, 1, 0\n"
            generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {mlen * mlen}\n"
            generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * hidden_size}\n"

        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp0, 0\n"
        generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"

        generated_code += f"; Middle loop: {tiles_per_mlen_down} tiles per MLEN block\n"
        generated_code += f"C_LOOP_START gp{loop_inner_reg}, {tiles_per_mlen_down}\n"

        generated_code += f"S_ADDI_INT gp{up_result_register}, gp0, {batch * seq_len * hidden_size}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, 0\n"

        generated_code += f"C_LOOP_START gp{loop_inner2_reg}, {num_down_act_cols}\n"

        generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_actual_register}, 0\n"
        generated_code += f"S_ADDI_INT gp{up_result_register}, gp{a_actual_register}, 0\n"

        for inner_idx in range(num_down_weight_tiles):
            generated_code += f"M_MM 0, gp{w_temp_register}, gp{a_actual_register}\n"
            generated_code += f"S_ADDI_INT gp{w_temp_register}, gp{w_temp_register}, {mlen * mlen}\n"
            if inner_idx < num_down_weight_tiles - 1:
                generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{a_actual_register}, {mlen * batch * seq_len}\n"

        generated_code += f"M_MM_WO gp{intermediate_register}, gp0, 0\n"
        generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{intermediate_register}, {blen * mlen}\n"
        generated_code += f"S_ADDI_INT gp{a_actual_register}, gp{up_result_register}, {down_act_col_advance}\n"

        generated_code += f"C_LOOP_END gp{loop_inner2_reg}\n"

        generated_code += f"S_ADDI_INT gp{w_actual_register}, gp{w_actual_register}, {blen}\n"
        generated_code += f"S_ADDI_INT gp{intermediate_register}, gp{act_result_register}, 0\n"
        generated_code += f"S_ADD_INT gp{intermediate_register}, gp{intermediate_register}, gp{w_actual_register}\n"

        generated_code += f"C_LOOP_END gp{loop_inner_reg}\n"

        generated_code += f"S_ADDI_INT gp{w_hbm_offset_register}, gp{w_hbm_offset_register}, {mlen}\n"
        generated_code += f"S_ADDI_INT gp{act_result_register}, gp{act_result_register}, {mlen * batch * seq_len}\n"

        generated_code += f"C_LOOP_END gp{loop_outer_reg}\n"

    return generated_code
