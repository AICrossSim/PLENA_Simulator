from assembler.parser import load_isa_definitions, load_isa_settings, parse_asm_file
from utils.load_config import load_svh_settings
import torch
from cfl_tools import PROJECT_PATH
from pathlib import Path
import argparse

class AssemblyToBinary:
    def __init__(self, isa_definition_file: str, config_file: str):
        """
        Initialize the Assembler with the ISA file.

        :param isa_definition_file: Path to the ISA file
        """
        self.isa_definitions = load_isa_definitions(isa_definition_file)
        self.isa_definition_file = isa_definition_file
        config_settings = load_svh_settings(config_file)
        self.opcode_width    = config_settings.get("OPCODE_WIDTH", 0)
        self.operands_width  = config_settings.get("OPERAND_WIDTH", 0)
        self.imm_width       = config_settings.get("IMM_WIDTH", 0)
        self.imm2_width      = config_settings.get("IMM_2_WIDTH", 0)
        self.instruction_length = config_settings.get("INSTRUCTION_LENGTH", 0)
        self.funct_width = config_settings.get("FUNCT_WIDTH", 0)
        self.funct_dist = self.instruction_length - 2 * self.funct_width


    def _convert_to_binary(self, instruction):
        """
        Convert an instruction to its binary representation.

        :param instruction: Instruction object
        :return: Binary representation of the instruction
        """
        # Example conversion logic (to be replaced with actual logic)
        opcode = self.isa_definitions[instruction.opcode]
        rd =  instruction.rd
        rs1 = instruction.rs1
        rs2 = instruction.rs2
        rstride = instruction.rstride
        funct1 = instruction.funct1
        funct2 = instruction.funct2
        imm = instruction.imm
        rmask = instruction.rmask
        binary_instruction = 0
        # print(f"Converting instruction: {instruction.opcode} with opcode={hex(opcode)}, rd={rd}, rs1={rs1}, rs2={rs2}, rstride={rstride}, funct1={funct1}, funct2={funct2}, imm={imm}")
        ow = self.operands_width
        opw = self.opcode_width

        if instruction.opcode in ["S_ADDI_INT",  "M_MM_WO", "S_LD_FP", "S_ST_FP", "S_LD_INT", "S_ST_INT", "S_MAP_V_FP", "V_RED_MAX", "V_RECI_V", "V_EXP_V"]:
            binary_instruction = (
                (imm << (opw + 2 * ow)) +
                (rs1 << (opw + ow)) +
                (rd << opw) +
                opcode
            )
        elif instruction.opcode in ["S_LUI_INT", "M_MV_WO", "M_BMM_WO", "M_BMV_WO"]:
            binary_instruction = (
                (imm << (opw + ow)) +
                (rd << opw) +
                opcode
            )
        elif instruction.opcode in [ "S_MV_FP", "S_RECI_FP", "S_EXP_FP", "S_SQRT_FP", "V_EXP_V", "V_RED_SUM"]:
            binary_instruction = (
                (rs1 << (opw + ow)) +
                (rd << opw) +
                opcode
            )
        elif instruction.opcode in [ "C_SET_SCALE_REG", "C_SET_STRIDE_REG", "C_SET_V_MASK_REG", "C_LOOP_END"]:
            binary_instruction = (
                (rd << opw) +
                opcode
            )
        elif instruction.opcode in ["C_LOOP_START"]:
            # C_LOOP_START rd, imm - uses 22-bit immediate like S_LUI_INT
            binary_instruction = (
                (imm << (opw + ow)) +
                (rd << opw) +
                opcode
            )
        elif instruction.opcode in ["H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V", "V_SUB_VF"]:
            binary_instruction = (
                (funct1 << (opw + 4 * ow)) +
                (rstride << (opw + 3 * ow)) +
                (rs2 << (opw + 2 * ow)) +
                (rs1 << (opw + ow)) +
                (rd << opw) +
                opcode
            )
        elif instruction.opcode in ["V_ADD_VV", "V_ADD_VF", "V_MUL_VV", "V_SUB_VV", "V_MUL_VF", "V_EXP_V", "V_RECI_V", "V_RED_SUM", "V_RED_MAX"]:
            binary_instruction = (
                (rmask << (opw + 3 * ow)) +
                (rs2 << (opw + 2 * ow)) +
                (rs1 << (opw + ow)) +
                (rd << opw) +
                opcode
            )
        else:
            binary_instruction = (
                (rs2 << (opw + 2 * ow)) +
                (rs1 << (opw + ow)) +
                (rd << opw) +
                opcode
            )

        # Print in hex with fixed 16-bit width
        return binary_instruction
    
    def write_binary_to_file(self, binary_instructions, output_file: str):
        with open(output_file, 'w') as file:
            for instruction in binary_instructions:
                file.write(f"0x{instruction:08X}\n")
    
    def generate_binary(self, asm_file: str, output_file: str):
        """
        Generate binary instructions from the assembled instructions.
        """
        instructions = parse_asm_file(asm_file)
        binary_instructions = []
        for instruction in instructions:
            # Convert each instruction to binary format
            binary_instruction = self._convert_to_binary(instruction)
            binary_instructions.append(binary_instruction)
        # Write the binary instructions to a file
        self.write_binary_to_file(binary_instructions, output_file)
        return binary_instructions
    

# if __name__ == "__main__":
#     import os
#     from pathlib import Path
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--layer', type=str, required=True, help='Input file name')
#     parser.add_argument('--test_type', type=str, default='Layerwise_Benchmark', help='Input file name (default: basic)')
#     args = parser.parse_args()

#     isa_file_path = '../../src/definitions/operation.svh'
#     config_file_path = '../../src/definitions/configuration.svh'
#     asm_file_path = f'../../test/{args.test_type}/{args.layer}.asm'
#     print(f'Assembling {asm_file_path} to {args.layer}.mem')
#     output_file_path = f'../../test/{args.test_type}/{args.layer}.mem'
#     assembler = AssemblyToBinary(isa_file_path, config_file_path)
#     assembler.generate_binary(asm_file_path, output_file_path)