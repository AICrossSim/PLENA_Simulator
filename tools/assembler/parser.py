import re
from typing import List, Optional

def load_isa_definitions(file_path: str) -> dict:
    """
    Parse a SystemVerilog enum from a .svh file and return it as a dictionary.
    """
    enum_dict = {}
    inside_enum = False
    pattern = re.compile(r'(\w+)\s*=\s*(\d+)\'h([0-9A-Fa-f]+)')

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()

            # Detect the start of the enum
            if line.startswith(f'typedef enum') and 'OPCODE_WIDTH' in line:
                inside_enum = True
                continue

            if inside_enum:
                # End of enum
                if line.endswith('} CUSTOM_ISA_OPCODE;'):
                    break

                # Match line like: S_ADD_FP = 6'h0E,
                match = pattern.search(line)
                if match:
                    name = match.group(1)
                    value = int(match.group(3), 16)
                    enum_dict[name] = value

    return enum_dict

def load_isa_settings(file_path: str) -> dict:
    param_pattern = re.compile(r'parameter\s+(\w+)\s*=\s*([^;]+);')
    param_dict = {}
    isa_settings_param = ["OPERAND_WIDTH", "OPCODE_WIDTH", "IMM_WIDTH", "IMM_2_WIDTH"]
    # First pass: collect simple constant values
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith('//') or not line or 'parameter' not in line:
            continue

        match = param_pattern.match(line)
        if match:
            key = match.group(1)
            value = match.group(2).strip()

            if key not in isa_settings_param:
                continue

            # Try to resolve constant integer values
            try:
                param_dict[key] = int(value)
            except ValueError:
                param_dict[key] = value  # Expression, to evaluate later
    return param_dict


class Instruction:
    def __init__(self, opcode: str, rd: str, rs1: Optional[str], rs2: Optional[str], rstride: Optional[str], funct1: Optional[int], funct2: Optional[int], imm: Optional[int] = None, rflag: Optional[int] = None):

        self.opcode = opcode
        self.rd = rd
        self.rs1 = rs1
        self.rs2 = rs2
        self.rstride = rstride
        self.funct1 = funct1
        self.funct2 = funct2
        self.imm = imm
        self.rmask = rstride

    def __repr__(self):
        return f"Instruction(opcode='{self.opcode}', rd='{self.rd}', rs1='{self.rs1}', rs2='{self.rs2}', rstride = '{self.rstride}', funct1={self.funct1}, funct2={self.funct2}, imm={self.imm}, rflag={self.rflag})"


def parse_asm_file(file_path: str) -> List[Instruction]:
    """
    Parse an ASM file into a list of Instruction objects.

    Supported formats:
    - opcode rd, rs1, rs2, rs3, funct1, funct2;
    - opcode rd, rs1, rs2, funct1, funct2;
    - opcode rd, rs1, rs2;
    - opcode rd, rs1, imm;
    - opcode rd, rs1;
    - opcode rd;

    :param file_path: Path to the .asm file
    :return: List of Instruction objects
    """
    instructions = []

    with open(file_path, 'r') as file:
        for line in file:
            # Remove comments and strip whitespace
            # Handle both // and ; style comments
            if line.startswith('//') or line.strip().startswith(';'):
                continue
            line = line.split('//')[0]  # Remove // comments
            line = line.split(';')[0]   # Remove ; comments
            line = line.strip()
            if not line:
                continue

            # Split the opcode and operands
            parts = line.split()
            if len(parts) < 2 or ';' in parts[0]:
                continue  # Invalid line
            opcode = parts[0]
            operands = [part.strip() for part in ' '.join(parts[1:]).split(',')]
            # print(f"Parsing instruction: {line}", "operand length:", len(operands), "operands:", operands)
            
            # Decode based on number of operands, case-structure by length
            rd = None
            rs1 = None
            rs2 = None
            rstride = None
            funct1 = None
            funct2 = None
            imm = None

            # Helper to parse a register or int operand
            def parse_reg_or_int(operand):
                operand = operand.strip()
                if operand.endswith(';'):
                    operand = operand[:-1]
                if operand.startswith('gp'):
                    return int(operand[2:])  # decimal, not hex
                elif operand.startswith('f'):
                    return int(operand[1:])  # decimal, not hex
                elif operand.startswith('a'):
                    return int(operand[1:])  # decimal, not hex
                else:
                    try:
                        return int(operand)
                    except ValueError:
                        return None

            if len(operands) == 1:
                operand_0 = operands[0]
                rd = parse_reg_or_int(operand_0)
            elif len(operands) == 2:
                operand_0 = operands[0]
                operand_1 = operands[1]
                rd = parse_reg_or_int(operand_0)
                # rs1 is a register, imm is a number
                # Heuristics: if it looks like a reg, it's rs1; else, it's imm
                if operand_1.strip().startswith(('gp','f','a')):
                    rs1 = parse_reg_or_int(operand_1)
                else:
                    try:
                        imm = int(operand_1)
                    except ValueError:
                        imm = None
            elif len(operands) == 3:
                operand_0, operand_1, operand_2 = operands
                rd = parse_reg_or_int(operand_0)
                # If looks like register, rs1; else, imm
                if operand_1.strip().startswith(('gp','f','a')):
                    rs1 = parse_reg_or_int(operand_1)
                else:
                    try:
                        imm = int(operand_1)
                    except ValueError:
                        imm = None
                # If it looks like register, rs2; else, imm (overwrites imm if rs1 not present)
                if operand_2.strip().startswith(('gp','f','a')):
                    rs2 = parse_reg_or_int(operand_2)
                else:
                    try:
                        imm = int(operand_2)
                    except ValueError:
                        pass
            elif len(operands) == 4:
                operand_0, operand_1, operand_2, operand_3 = operands
                rd = parse_reg_or_int(operand_0)
                if operand_1.strip().startswith(('gp','f','a')):
                    rs1 = parse_reg_or_int(operand_1)
                else:
                    try:
                        imm = int(operand_1)
                    except ValueError:
                        imm = None
                if operand_2.strip().startswith(('gp','f','a')):
                    rs2 = parse_reg_or_int(operand_2)
                else:
                    try:
                        imm = int(operand_2)
                    except ValueError:
                        pass
                # Interpret 4th operand as rstride if int
                try:
                    rstride = int(operand_3)
                except ValueError:
                    rstride = None
            elif len(operands) == 5:
                operand_0, operand_1, operand_2, operand_3, operand_4 = operands
                rd = parse_reg_or_int(operand_0)
                if operand_1.strip().startswith(('gp','f','a')):
                    rs1 = parse_reg_or_int(operand_1)
                else:
                    try:
                        imm = int(operand_1)
                    except ValueError:
                        imm = None
                if operand_2.strip().startswith(('gp','f','a')):
                    rs2 = parse_reg_or_int(operand_2)
                else:
                    try:
                        imm = int(operand_2)
                    except ValueError:
                        pass
                try:
                    rstride = int(operand_3)
                except ValueError:
                    rstride = None
                funct1_raw = operand_4.strip()
                if funct1_raw.endswith(';'):
                    funct1_raw = funct1_raw[:-1]
                try:
                    funct1 = int(funct1_raw)
                except ValueError:
                    funct1 = funct1_raw  # fallback, if not int, keep as string
            elif len(operands) == 6:
                operand_0, operand_1, operand_2, operand_3, operand_4, operand_5 = operands
                rd = parse_reg_or_int(operand_0)
                if operand_1.strip().startswith(('gp','f','a')):
                    rs1 = parse_reg_or_int(operand_1)
                else:
                    try:
                        imm = int(operand_1)
                    except ValueError:
                        imm = None
                if operand_2.strip().startswith(('gp','f','a')):
                    rs2 = parse_reg_or_int(operand_2)
                else:
                    try:
                        imm = int(operand_2)
                    except ValueError:
                        pass
                try:
                    rstride = int(operand_3)
                except ValueError:
                    rstride = None
                funct1_raw = operand_4.strip()
                if funct1_raw.endswith(';'):
                    funct1_raw = funct1_raw[:-1]
                try:
                    funct1 = int(funct1_raw)
                except ValueError:
                    funct1 = funct1_raw  # fallback, if not int, keep as string
                funct2_raw = operand_5.strip()
                if funct2_raw.endswith(';'):
                    funct2_raw = funct2_raw[:-1]
                try:
                    funct2 = int(funct2_raw)
                except ValueError:
                    funct2 = funct2_raw  # fallback, if not int, keep as string


            instructions.append(Instruction(opcode, rd, rs1, rs2, rstride, funct1, funct2, imm))

    return instructions



if __name__ == "__main__":
    # Example usage
    # file_path = '/home/george/Coprocessor_for_Llama/src/definitions/operation.svh'
    # enum_dict = load_isa_definitions(file_path)
    # print(enum_dict)
    

    asm_file_path = '/home/george/Coprocessor_for_Llama/src/system/test/benchmarks/fixed.asm'
    loaded_instr = parse_asm_file(asm_file_path)
    for instr in loaded_instr:
        print(instr)