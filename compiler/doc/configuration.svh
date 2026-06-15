`ifndef CONFIGURATION_SVH
`define CONFIGURATION_SVH
`include "global_define.vh"
`include "precision.svh"

import precision_pkg::*;

package configuration_pkg;
    // Compute Unit Related 
    parameter   BLEN = 4;
    parameter   HLEN = 8
    parameter   MLEN = 16;
    parameter   Matrix_Parallel_Rd_Dim = 1;
    parameter   VLEN = 16;
    parameter   INST_BUFF_DEPTH = 16;
    parameter   ON_CHIP_ADDR_WIDTH = precision_pkg::INT_DATA_WIDTH;
    parameter   SourceWidth = 1;
    parameter   SinkWidth = 1;
    // Memory Related
    parameter   MATRIX_SRAM_WIDTH = (precision_pkg::WT_MX_MANT_WIDTH + precision_pkg::WT_MX_EXP_WIDTH + 1 + precision_pkg::MX_SCALE_WIDTH) * MLEN;
    parameter   MATRIX_SRAM_DEPTH = 1024;
    parameter   VECTOR_SRAM_WIDTH = (precision_pkg::V_FP_MANT_WIDTH + precision_pkg::V_FP_EXP_WIDTH + 1) * VLEN;
    parameter   VECTOR_SRAM_DEPTH = 1024;
    parameter   VECTOR_RESET_AMOUNT = 8;            // Need to be the same as Head_Dim for assembly code.
    parameter   INT_SRAM_WIDTH      = precision_pkg::INT_DATA_WIDTH;
    parameter   INT_SRAM_DEPTH      = 32;
    parameter   FP_SRAM_WIDTH       = (precision_pkg::S_FP_MANT_WIDTH + precision_pkg::S_FP_EXP_WIDTH + 1);
    parameter   FP_SRAM_DEPTH       = 512;
    parameter   HBM_ADDR_WIDTH      = 128;

    // HBM Related
    parameter   HBM_M_Prefetch_Amount   = 16;
    parameter   HBM_V_Prefetch_Amount   = 16;
    parameter   HBM_V_Writeback_Amount  = 4;
    parameter   HBM_ELE_WIDTH           = 512;
    parameter   HBM_SCALE_WIDTH         = 512;
    parameter   HBM_WIDTH               = 512;
endpackage

package instruction_pkg;
    parameter INT_OPERAND_WIDTH     = 4;
    parameter FP_OPERAND_WIDTH      = 3;
    parameter HBM_ADR_OPERAND_WIDTH = 3;
    parameter STRIDE_OPERAND_WIDTH  = 3;
    parameter OPERAND_WIDTH         = 4;
    parameter FUNCT_WIDTH           = 4;
    parameter OPCODE_WIDTH          = 6;
    parameter IMM_WIDTH             = 22;
    parameter IMM_2_WIDTH           = 18;
    parameter INSTRUCTION_LENGTH    = 32;
endpackage

package simulation_pkg;
    parameter   FAKE_HBM_ADDR_WIDTH             = 16;
endpackage

`ifdef DC_LIB_EN // Define for DC Library Enabled, the pipeline stage lib changed accordingly.

    package pipeline_pkg;
        parameter   MAX_PIPELINE_STAGE             = 10;   
        parameter   SYSTOLIC_PROCESSING_OVERHEAD   = 0;
        parameter   VECTOR_LONGEST_OPERATE_CYCLES  = 10;
        parameter   VECTOR_ADD_CYCLES              = 2;
        parameter   VECTOR_MUL_CYCLES              = 1;
        parameter   VECTOR_EXP_CYCLES              = 1;
        parameter   VECTOR_PREFIX_SCAN_CYCLES      = 9;
        parameter   VECTOR_SHIFT_CYCLES            = 1;
        parameter   VECTOR_RECI_CYCLES             = 2;
        parameter   VECTOR_MAX_CYCLES              = 4;
        parameter   VECTOR_SUM_CYCLES              = 8;
        parameter   SCALAR_FP_LONGEST_OPERATE_CYCLES = 4;
        parameter   SCALAR_FP_BASIC_CYCLES         = 1;
        parameter   SCALAR_FP_EXP_CYCLES           = 1;
        parameter   SCALAR_FP_SQRT_CYCLES          = 1;
        parameter   SCALAR_FP_RECI_CYCLES          = 1;
        parameter   SCALAR_INT_BASIC_CYCLES        = 1;
        parameter   HADAMARD_TRANSFORM_CYCLES      = $clog2(configuration_pkg::MLEN) + 1;
    endpackage

`else

    package pipeline_pkg;
        parameter   MAX_PIPELINE_STAGE             = 10;   
        parameter   SYSTOLIC_PROCESSING_OVERHEAD   = 0;
        parameter   VECTOR_LONGEST_OPERATE_CYCLES  = 20;
        parameter   VECTOR_ADD_CYCLES              = 7;
        parameter   VECTOR_MUL_CYCLES              = 5;
        parameter   VECTOR_PREFIX_SCAN_CYCLES      = 9;
        parameter   VECTOR_EXP_CYCLES              = 6;
        parameter   VECTOR_SHIFT_CYCLES            = 1;
        parameter   VECTOR_RECI_CYCLES             = 7;
        parameter   VECTOR_MAX_CYCLES              = 4;
        parameter   VECTOR_SUM_CYCLES              = 20;
        parameter   SCALAR_FP_LONGEST_OPERATE_CYCLES = 4;
        parameter   SCALAR_FP_BASIC_CYCLES         = 1;
        parameter   SCALAR_FP_EXP_CYCLES           = 2;
        parameter   SCALAR_FP_SQRT_CYCLES          = 2;
        parameter   SCALAR_FP_RECI_CYCLES          = 2;
        parameter   SCALAR_INT_BASIC_CYCLES        = 1;
    endpackage

`endif

`endif