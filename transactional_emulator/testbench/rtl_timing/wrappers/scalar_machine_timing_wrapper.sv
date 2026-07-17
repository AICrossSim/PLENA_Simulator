`timescale 1ns / 1ps

`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"

// Simulation-only control adapter for the production ScalarMachine.  It keeps
// the measured boundary at OP_BUNDLE acceptance and exposes existing internal
// result-valid signals without modifying the implementation under test.
module scalar_machine_timing_wrapper
    import precision_pkg::*;
    import configuration_pkg::*;
    import instruction_pkg::*;
(
    input  logic clk,
    input  logic rst,
    input  logic [3:0] scalar_fp_op,
    input  logic [FP_OPERAND_WIDTH-1:0] fps1,
    input  logic [FP_OPERAND_WIDTH-1:0] fps2,
    input  logic [FP_OPERAND_WIDTH-1:0] fpd,
    input  logic [ON_CHIP_ADDR_WIDTH-1:0] map_addr,
    input  logic [S_FP_EXP_WIDTH + S_FP_MANT_WIDTH:0] external_fp_in,
    input  logic external_fp_in_valid,
    input  logic [FP_OPERAND_WIDTH-1:0] external_fp_wtarget,
    output logic compute_result_ready,
    output logic map_result_ready,
    output logic backend_busy,
    output logic scalar_sram_busy
);
    OP_BUNDLE exe_stage_op;
    logic [INT_DATA_WIDTH-1:0] gp_out_1, gp_out_2;
    logic [S_FP_EXP_WIDTH + S_FP_MANT_WIDTH:0] fp_out;
    logic [VLEN-1:0][S_FP_EXP_WIDTH + S_FP_MANT_WIDTH:0] fp_vector_out;
    logic received_v_reduct_result;
    logic loop_counter_zero;

    assign exe_stage_op.m_op              = STALL_M;
    assign exe_stage_op.v_ele_op          = STALL_V_ELEMENT;
    assign exe_stage_op.v_reduct_op       = STALL_V_REDUCT;
    assign exe_stage_op.s_fp_op           = S_FP_OP'(scalar_fp_op);
    assign exe_stage_op.c_op              = STALL_C;
    assign exe_stage_op.h_op              = STALL_H;
    assign exe_stage_op.m_transposed_read = 1'b0;
    assign exe_stage_op.v_broadcast_en    = 1'b0;
    assign exe_stage_op.fps1              = fps1;
    assign exe_stage_op.fps2              = fps2;
    assign exe_stage_op.fpd               = fpd;
    assign exe_stage_op.gp_reg1           = '0;
    assign exe_stage_op.gp_reg2           = '0;
    assign exe_stage_op.gp_rstride        = '0;
    assign exe_stage_op.gp_rd             = '0;
    assign exe_stage_op.addr_1            = map_addr;
    assign exe_stage_op.addr_2            = '0;
    assign exe_stage_op.update_m_waddr    = 1'b0;
    assign exe_stage_op.update_v_waddr    = 1'b0;
    assign exe_stage_op.pc_tag            = '0;

    scalar_machine dut (
        .clk                       (clk),
        .rst                       (rst),
        .exe_stage_op              (exe_stage_op),
        .assigned_int_op           (STALL_S_INT),
        .rs1                       ('0),
        .rs2                       ('0),
        .rd                        ('0),
        .imm_in                    ('0),
        .gp_out_1                  (gp_out_1),
        .gp_out_2                  (gp_out_2),
        .external_fp_in            (external_fp_in),
        .external_fp_in_valid      (external_fp_in_valid),
        .external_fp_wtarget       (external_fp_wtarget),
        .fp_out                    (fp_out),
        .fp_vector_out             (fp_vector_out),
        .fp_vector_out_valid       (map_result_ready),
        .received_v_reduct_result  (received_v_reduct_result),
        .fp_stall_req              (backend_busy),
        .fp_sram_stall_req         (scalar_sram_busy),
        .loop_counter_zero         (loop_counter_zero)
    );

    assign compute_result_ready = dut.fp_alu_valid | dut.fp_sfu_valid;
endmodule
