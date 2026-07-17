`timescale 1ns / 1ps

`include "precision.svh"
`include "configuration.svh"
`include "operation.svh"

// Simulation-only flat adapter around the production pipeline controller.
// No register or combinational stage is added. Internal observation taps are
// exposed solely so the cocotb harness can distinguish the raw hazard from the
// registered one-cycle recovery requested by b1_pipeline_stall.
module pipeline_control_timing_wrapper
    import precision_pkg::*;
    import configuration_pkg::*;
    import instruction_pkg::*;
(
    input logic clk,
    input logic rst,

    input logic [3:0] decode_m_op,
    input logic [3:0] decode_v_element_op,
    input logic [2:0] decode_v_reduction_op,
    input logic [3:0] decode_s_fp_op,
    input logic [2:0] decode_c_op,
    input logic [2:0] decode_h_op,
    input logic decode_v_broadcast_en,

    input logic hbm_m_prefetch_in_progress,
    input logic hbm_v_prefetch_in_progress,
    input logic continuous_write_to_v_sram,
    input logic fp_stall_req,
    input logic fp_sram_stall_req,
    input logic m_load_in_process,
    input logic m_mcu_active,
    input logic s_received_v_reduct_result,

    input logic mem_wreq_m_sram,
    input logic mem_wreq_s_sram_port_a,
    input logic mem_wreq_s_sram_port_b,
    input logic [1:0] mem_wreq_from_m,

    output logic pipeline_stall_req,
    output logic raw_pipeline_stall,
    output logic recovery_stall,
    output logic vector_reduction_in_process,
    output logic [3:0] determine_m_op,
    output logic [3:0] determine_v_element_op,
    output logic [2:0] determine_v_reduction_op,
    output logic [3:0] determine_s_fp_op,
    output logic [3:0] execute_m_op,
    output logic [3:0] execute_v_element_op,
    output logic [2:0] execute_v_reduction_op,
    output logic [3:0] execute_s_fp_op
);
    OP_BUNDLE decode_stage_op;
    OP_BUNDLE exe_stage_op;
    MEM_WREQ_INFO mem_write_req;
    MEM_WEN_INFO mem_write_control;

    assign decode_stage_op.m_op              = M_OP'(decode_m_op);
    assign decode_stage_op.v_ele_op          = V_ELEMENT_OP'(decode_v_element_op);
    assign decode_stage_op.v_reduct_op       = V_REDUCT_OP'(decode_v_reduction_op);
    assign decode_stage_op.s_fp_op           = S_FP_OP'(decode_s_fp_op);
    assign decode_stage_op.c_op              = C_OP'(decode_c_op);
    assign decode_stage_op.h_op              = H_OP'(decode_h_op);
    assign decode_stage_op.m_transposed_read = 1'b0;
    assign decode_stage_op.v_broadcast_en    = decode_v_broadcast_en;
    assign decode_stage_op.fps1              = '0;
    assign decode_stage_op.fps2              = '0;
    assign decode_stage_op.fpd               = '0;
    assign decode_stage_op.gp_reg1           = '0;
    assign decode_stage_op.gp_reg2           = '0;
    assign decode_stage_op.gp_rstride        = '0;
    assign decode_stage_op.gp_rd             = '0;
    assign decode_stage_op.addr_1            = '0;
    assign decode_stage_op.addr_2            = '0;
    assign decode_stage_op.update_m_waddr    = 1'b0;
    assign decode_stage_op.update_v_waddr    = 1'b0;
    assign decode_stage_op.pc_tag            = '0;

    assign mem_write_req.wreq_m_sram        = mem_wreq_m_sram;
    assign mem_write_req.wreq_s_sram_port_a = mem_wreq_s_sram_port_a;
    assign mem_write_req.wreq_s_sram_port_b = mem_wreq_s_sram_port_b;
    assign mem_write_req.wreq_from_m         = mem_wreq_from_m;

    pipeline_control dut (
        .clk                         (clk),
        .rst                         (rst),
        .decode_stage_op             (decode_stage_op),
        .gp_addr_1                   ('0),
        .gp_addr_2                   ('0),
        .v_sram_wen_a                (1'b0),
        .v_sram_addr_a               ('0),
        .v_sram_wen_b                (1'b0),
        .v_sram_addr_b               ('0),
        .hbm_m_prefetch_in_progress  (hbm_m_prefetch_in_progress),
        .hbm_v_prefetch_in_progress  (hbm_v_prefetch_in_progress),
        .continuous_write_to_v_sram  (continuous_write_to_v_sram),
        .mem_write_req               (mem_write_req),
        .hbm_in_used                 (1'b0),
        .fp_stall_req                (fp_stall_req),
        .fp_sram_stall_req           (fp_sram_stall_req),
        .m_load_in_process           (m_load_in_process),
        .m_mcu_active                (m_mcu_active),
        .s_received_v_reduct_result  (s_received_v_reduct_result),
        .pipeline_stall_req          (pipeline_stall_req),
        .exe_stage_op                (exe_stage_op),
        .mem_write_control           (mem_write_control)
    );

    assign raw_pipeline_stall            = dut.pipeline_stall;
    assign recovery_stall                = dut.b1_pipeline_stall;
    assign vector_reduction_in_process   = dut.vector_reduct_in_process;
    assign determine_m_op                = dut.determine_stage_op.m_op;
    assign determine_v_element_op        = dut.determine_stage_op.v_ele_op;
    assign determine_v_reduction_op      = dut.determine_stage_op.v_reduct_op;
    assign determine_s_fp_op             = dut.determine_stage_op.s_fp_op;
    assign execute_m_op                  = exe_stage_op.m_op;
    assign execute_v_element_op          = exe_stage_op.v_ele_op;
    assign execute_v_reduction_op        = exe_stage_op.v_reduct_op;
    assign execute_s_fp_op               = exe_stage_op.s_fp_op;
endmodule
