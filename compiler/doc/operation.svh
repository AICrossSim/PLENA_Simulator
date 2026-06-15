`ifndef OPERATION_SVH
`define OPERATION_SVH

`include "configuration.svh"
import configuration_pkg::*;
import instruction_pkg::*;

typedef struct {
    logic w_m_sram_en;
    logic w_s_sram_port_a_en;
    logic w_s_sram_port_b_en;
    logic [1:0] w_from_m; // 2'b00: No write, 2'b01: write for M_MM_WO (MLEN, MLEN), 2'b10: write for M_MV_WO (MLEN, 1)
} MEM_WEN_INFO;

typedef struct {
    logic wreq_m_sram;
    logic wreq_s_sram_port_a;
    logic wreq_s_sram_port_b;
    logic [1:0] wreq_from_m;
} MEM_WREQ_INFO;

typedef enum logic [3:0] {
    MV_IC           = 3'h1,
    MV_BIC          = 3'h2,
    MV_WO           = 3'h3,
    BMV_WO          = 3'h4,
    MM_IC           = 3'h5,
    MM_BIC          = 3'h6,
    MM_WO           = 3'h7,
    BMM_WO          = 3'h8,
    STALL_M         = 3'h0
} M_OP;

typedef enum logic [3:0] {
    STALL_V_ELEMENT = 4'h0,
    ADD_V_ELEMENT   = 4'h1,
    SUB_V_ELEMENT   = 4'h2,
    MUL_V_ELEMENT   = 4'h3,
    EXP_V_ELEMENT   = 4'h4,
    RECI_V_ELEMENT  = 4'h5,
    INNER_HADAMARD_TRANSFORM    = 4'h6,
    PREFIX_SCAN_V_ELEMENT       = 4'h7,
    SHIFT_V_LANES_ELEMENT       = 4'h8   // renamed, use this everywhere
} V_ELEMENT_OP;

typedef enum logic [2:0] {
    STALL_V_REDUCT  = 3'h0,
    SUM_V_REDUCT    = 3'h1,
    MAX_V_REDUCT    = 3'h2
} V_REDUCT_OP;

typedef enum logic [3:0] {
    STALL_S_FP  = 4'h0,
    ADD_FP      = 4'h1,
    SUB_FP      = 4'h2,
    MAX_FP      = 4'h3,
    MUL_FP      = 4'h4,
    EXP_FP      = 4'h5,
    RECI_FP     = 4'h6,
    SQRT_FP     = 4'h7,
    LD_REG_FP   = 4'h8,
    LD_OUT_FP   = 4'h9,
    ST_REG_FP   = 4'hA,
    ST_IN_FP    = 4'hB,
    MV_FP       = 4'hC,
    MAP_V_FP    = 4'hD
} S_FP_OP;

typedef enum logic [3:0] {
    ADD_INT       = 4'h1,
    ADDI_INT      = 4'h2,
    SUB_INT       = 4'h3,
    MUL_INT       = 4'h4,
    LUI_INT       = 4'h5,
    LD_INT        = 4'h6,
    ST_INT        = 4'h7,
    PASS_ADDR     = 4'h8,
    PASS_ADDR_2   = 4'h9, // addr_port_2: rd and addr_port_1: rs1 adress.
    COMP_ADDR     = 4'hA,
    COMP_ADDR_2   = 4'hB, // addr_port_2: rd and addr_port_1: rs1 + imm
    STALL_S_INT   = 4'h0
} S_INT_OP;

typedef enum logic [2:0] {
    STALL_C             = 3'h0,
    SET_ADDR_REG        = 3'h1,
    SET_V_STRIDE_SIZE   = 3'h2,
    SET_M_STRIDE_SIZE   = 3'h3,
    SET_V_SCALE_REG     = 3'h4,
    SET_M_SCALE_REG     = 3'h5,
    BREAK               = 3'h6
} C_OP;

typedef enum logic [2:0] {
    STALL_H         = 4'h0,
    PREFETCH_M_H    = 4'h1,
    PREFETCH_M_L    = 4'h2,
    PREFETCH_V_H    = 4'h3,
    PREFETCH_V_L    = 4'h4,
    STORE_V_H       = 4'h5,
    STORE_V_L       = 4'h6
} H_OP;

function automatic int max(input int a, input int b);
    return (a > b) ? a : b;
endfunction

typedef enum logic [instruction_pkg::OPCODE_WIDTH - 1:0] {
    // Invalid
    INVALID_OPCODE         = 6'h00,

    // Matrix Operations
    M_MM                   = 6'h01,
    M_TMM                  = 6'h02,
    M_BMM                  = 6'h03,
    M_BTMM                 = 6'h04,
    M_BMM_WO               = 6'h05,
    M_MM_WO                = 6'h06,
    M_MV                   = 6'h07,
    M_TMV                  = 6'h08,
    M_BMV                  = 6'h09,
    M_BTMV                 = 6'h0A,
    M_MV_WO                = 6'h0B,
    M_BMV_WO               = 6'h0C,

    // Vector Operations
    V_ADD_VV               = 6'h0D,
    V_ADD_VF               = 6'h0E,
    V_SUB_VV               = 6'h0F,
    V_SUB_VF               = 6'h10,
    V_MUL_VV               = 6'h11,
    V_MUL_VF               = 6'h12,
    V_EXP_V                = 6'h13,
    V_RECI_V               = 6'h14,
    V_RED_SUM              = 6'h15,
    V_RED_MAX              = 6'h16,

    // Scalar Operations (Floating-Point)
    S_ADD_FP               = 6'h17,
    S_SUB_FP               = 6'h18,
    S_MAX_FP               = 6'h19,
    S_MUL_FP               = 6'h1A,
    S_EXP_FP               = 6'h1B,
    S_RECI_FP              = 6'h1C,
    S_SQRT_FP              = 6'h1D,
    S_LD_FP                = 6'h1E,
    S_ST_FP                = 6'h1F,
    S_MAP_V_FP             = 6'h20,

    // Scalar Operations (INT)
    S_ADD_INT              = 6'h21,
    S_ADDI_INT             = 6'h22,
    S_SUB_INT              = 6'h23,
    S_MUL_INT              = 6'h24,
    S_LUI_INT              = 6'h25,
    S_LD_INT               = 6'h26,
    S_ST_INT               = 6'h27,

    // Memory Operations
    H_PREFETCH_M           = 6'h28,
    H_PREFETCH_V           = 6'h29,
    H_STORE_V              = 6'h2A,

    // CSR Setting
    C_SET_ADDR_REG         = 6'h2B,
    C_SET_SCALE_REG        = 6'h2C,
    C_SET_STRIDE_REG       = 6'h2D,
    C_SET_V_MASK_REG       = 6'h2E,
    C_LOOP_START           = 6'h2F,
    C_LOOP_END             = 6'h30,

    // Extensions
    V_PS_V                 = 6'h31,
    V_SHFT_V               = 6'h32,
    C_HADAMARD_TRANSFORM   = 6'h33,
    C_BREAK                = 6'h34
} CUSTOM_ISA_OPCODE;


typedef enum logic [2:0] {
    INVALID_TYPE = 3'h0,
    M            = 3'h1,
    V            = 3'h2,
    S_INT        = 3'h3,
    S_FP         = 3'h4,
    C            = 3'h5,
    H            = 3'h6
} CUSTOM_ISA_TYPE;

typedef struct {
    logic [instruction_pkg::OPCODE_WIDTH  - 1 : 0]      opcode;
    logic [instruction_pkg::OPERAND_WIDTH - 1 : 0]      rs1;
    logic [instruction_pkg::OPERAND_WIDTH - 1 : 0]      rs2;
    logic [instruction_pkg::OPERAND_WIDTH - 1 : 0]      rstride;
    logic [instruction_pkg::OPERAND_WIDTH - 1 : 0]      rd;
    logic [instruction_pkg::IMM_WIDTH - 1 : 0]          imm;
    logic [instruction_pkg::FUNCT_WIDTH - 1 : 0]        funct1;
    CUSTOM_ISA_TYPE instruction_type;
} INSTR_INFO;

typedef struct {
    M_OP            m_op;
    V_ELEMENT_OP    v_ele_op;
    V_REDUCT_OP     v_reduct_op;
    S_FP_OP         s_fp_op;
    C_OP            c_op;
    H_OP            h_op;
    logic           m_transposed_read;
    logic           v_broadcast_en;
    logic [instruction_pkg::FP_OPERAND_WIDTH - 1:0]         fps1;
    logic [instruction_pkg::FP_OPERAND_WIDTH - 1:0]         fps2;
    logic [instruction_pkg::FP_OPERAND_WIDTH - 1:0]         fpd;
    logic [instruction_pkg::INT_OPERAND_WIDTH - 1:0]        gp_reg1;
    logic [instruction_pkg::INT_OPERAND_WIDTH - 1:0]        gp_reg2;
    logic [instruction_pkg::INT_OPERAND_WIDTH - 1:0]        gp_rstride;
    logic [instruction_pkg::INT_OPERAND_WIDTH - 1:0]        gp_rd;
    logic [configuration_pkg::ON_CHIP_ADDR_WIDTH - 1:0]     addr_1;
    logic [configuration_pkg::ON_CHIP_ADDR_WIDTH - 1:0]     addr_2;
    logic update_m_waddr;
    logic update_v_waddr;
} OP_BUNDLE;

`endif