`timescale 1ns / 1ps

`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"

// Thin control adapter around the production MatrixMachine.  Operand streams
// pass through its real alignment buffers and the observed out_valid includes
// the production result buffer, unlike the MCU-only calibration harness.
module matrix_machine_timing_wrapper
    import precision_pkg::*;
    import configuration_pkg::*;
    import instruction_pkg::*;
(
    input  logic clk,
    input  logic rst,
    input  logic [3:0] matrix_op,
    input  logic [ON_CHIP_ADDR_WIDTH-1:0] result_waddr,
    input  logic result_waddr_update,
    input  logic [MLEN-1:0][WT_ELEMENT_WIDTH-1:0] matrix_element,
    input  logic [MLEN-1:0][MX_SCALE_WIDTH-1:0] matrix_scale,
    input  logic matrix_valid,
    input  logic [MLEN-1:0][ACT_ELEMENT_WIDTH-1:0] vector_element,
    input  logic [MLEN/BLOCK_DIM-1:0][MX_SCALE_WIDTH-1:0] vector_scale,
    input  logic vector_valid,
    output logic result_ready,
    output logic backend_busy,
    output logic complete_operand_load,
    output logic [1:0] write_mode,
    output logic [ON_CHIP_ADDR_WIDTH-1:0] write_addr
);
    OP_BUNDLE exe_stage_op;
    logic [MLEN-1:0][V_FP_EXP_WIDTH + V_FP_MANT_WIDTH:0] result_data;

    // Assign each field separately.  Besides documenting the exact opcode
    // boundary under test, this avoids a Verilator packed-struct lowering bug
    // triggered by assigning the complete OP_BUNDLE from an integer literal.
    assign exe_stage_op.m_op              = M_OP'(matrix_op);
    assign exe_stage_op.v_ele_op          = STALL_V_ELEMENT;
    assign exe_stage_op.v_reduct_op       = STALL_V_REDUCT;
    assign exe_stage_op.s_fp_op           = STALL_S_FP;
    assign exe_stage_op.c_op              = STALL_C;
    assign exe_stage_op.h_op              = STALL_H;
    assign exe_stage_op.m_transposed_read = 1'b0;
    assign exe_stage_op.v_broadcast_en    = 1'b0;
    assign exe_stage_op.fps1              = '0;
    assign exe_stage_op.fps2              = '0;
    assign exe_stage_op.fpd               = '0;
    assign exe_stage_op.gp_reg1           = '0;
    assign exe_stage_op.gp_reg2           = '0;
    assign exe_stage_op.gp_rstride        = '0;
    assign exe_stage_op.gp_rd             = '0;
    assign exe_stage_op.addr_1            = '0;
    assign exe_stage_op.addr_2            = result_waddr;
    assign exe_stage_op.update_m_waddr    = result_waddr_update;
    assign exe_stage_op.update_v_waddr    = 1'b0;
    assign exe_stage_op.pc_tag            = '0;

    matrix_machine dut (
        .clk                    (clk),
        .rst                    (rst),
        .exe_stage_op           (exe_stage_op),
        .mcu_active             (backend_busy),
        .m_element              (matrix_element),
        .m_scale                (matrix_scale),
        .m_valid                (matrix_valid),
        .v_element              (vector_element),
        .v_scale                (vector_scale),
        .v_valid                (vector_valid),
        .out_v_fp               (result_data),
        .out_valid              (result_ready),
        .m_waddr                (write_addr),
        .m_wreq                 (write_mode)
`ifdef SIMULATION
        , .dbg_complete_v1_load (),
          .dbg_complete_v2_load (),
          .dbg_complete_loading_q (),
          .dbg_gebm_result_0 ()
`endif
    );

    // matrix_machine's legacy debug outputs are declared but not driven.  The
    // production MXFP MCU latch is therefore observed directly here.  This is
    // an observation-only tap and adds no logic to the measured datapath.
    if (!WT_MX_INT_ENABLE) begin : gen_mxfp_load_observer
        assign complete_operand_load =
            dut.gen_mx_systolic_mcu.matrix_compute_unit.complete_loading_q;
    end else begin : gen_mxint_load_observer
        assign complete_operand_load = 1'b0;
    end
endmodule
