import os
from pathlib import Path

from memory_mapping import generate_hbm
from assembler.assembly_to_binary import AssemblyToBinary

# Derive project root: this file is at PLENA_Tools/sim_env_utils/build_sys_tools.py
PROJECT_PATH = Path(__file__).resolve().parent.parent.parent

# Default instruction storage offset (from configuration.svh)
INSTRUCTION_STORAGE_OFFSET = 8192


def calculate_instr_offset_after_data(
    tensor_data: list,
    element_width: int,
    bias_width: int,
    hbm_row_width: int = 256,
) -> int:
    """
    Calculate the instruction storage offset to place instructions right after data.

    This computes the byte offset where the instruction section should start,
    immediately after all element and scale data rows (in interleaved format).

    Args:
        tensor_data: List of (blocks, bias) tuples, one per tensor
        element_width: Bit width of each element
        bias_width: Bit width of each scale value
        hbm_row_width: HBM row width in bits (default: 256)

    Returns:
        Byte offset for instruction storage
    """
    import math

    bytes_per_row = hbm_row_width // 8
    total_rows = 0

    for blocks, bias in tensor_data:
        # Calculate number of element rows for this tensor
        if blocks:
            block_width = element_width * len(blocks[0])
            num_blocks_per_row = hbm_row_width // block_width if block_width > 0 else 1
            num_element_rows = math.ceil(len(blocks) / num_blocks_per_row)
        else:
            num_element_rows = 0

        # Calculate number of scale rows for this tensor
        if bias:
            num_bias_per_row = hbm_row_width // bias_width if bias_width > 0 else 1
            num_scale_rows = math.ceil(len(bias) / num_bias_per_row)
        else:
            num_scale_rows = 0

        total_rows += num_element_rows + num_scale_rows

    return total_rows * bytes_per_row


def read_instructions_from_mem(mem_file_path: Path) -> list:
    """
    Read 32-bit instructions from a .mem file.

    Args:
        mem_file_path: Path to the instruction .mem file

    Returns:
        List of instruction words as integers
    """
    instructions = []
    if not mem_file_path.exists():
        return instructions

    with open(mem_file_path) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            # Parse hex value (format: 0xDEADBEEF or just DEADBEEF)
            if line.startswith("0x") or line.startswith("0X"):
                try:
                    instructions.append(int(line, 16))
                except ValueError:
                    pass
    return instructions


def env_setup(
    memory_data_manager,
    build_path: str,
    data_config,
    quant_config,
    hbm_row_width=256,
    test_file_name=None,
    instr_storage_offset=None,
):
    """
    Setup environment for simulation using MemoryDataManager.
    Generates a single hbm.mem file with interleaved element and scale data per tensor.

    Memory layout (interleaved per tensor):
        [tensor0_elements][tensor0_scales][tensor1_elements][tensor1_scales]...[instructions]

    Args:
        memory_data_manager: MemoryDataManager instance
        build_path: Path to build directory
        data_config: Data configuration dictionary
        quant_config: Quantization configuration dictionary
        hbm_row_width: HBM row width in bits
        test_file_name: Optional test file name
        instr_storage_offset: Byte offset for instruction storage. If None, uses default
            INSTRUCTION_STORAGE_OFFSET (8192). Pass "auto" or -1 to automatically
            calculate offset to place instructions right after data.
    """
    isa_file_path = PROJECT_PATH / "compiler" / "doc" / "operation.svh"
    config_file_path = PROJECT_PATH / "compiler" / "doc" / "configuration.svh"

    # Determine instruction file path
    if test_file_name is None:
        instr_file = build_path / "generated_machine_code.mem"
        assembler = AssemblyToBinary(str(isa_file_path), str(config_file_path))
        assembler.generate_binary(build_path / "generated_asm_code.asm", instr_file)
    else:
        instr_file = build_path / f"{test_file_name}.mem"
        assembler = AssemblyToBinary(str(isa_file_path), str(config_file_path))
        assembler.generate_binary(build_path / f"{test_file_name}.asm", instr_file)

    # Read instructions from the generated .mem file
    instructions = read_instructions_from_mem(instr_file)

    entries = memory_data_manager.get_all_entries()

    # Collect tensor data as list of (blocks, bias) tuples for interleaved output
    tensor_data = []

    for entry in entries:
        if entry["type"] == "mx":
            tensor_data.append((entry["blocks"], entry["bias"]))
        elif entry["type"] == "int":
            # For int data, treat as raw data blocks with no scales
            data = entry["data"]
            int_blocks = [[val] for val in data]
            tensor_data.append((int_blocks, []))

    # Determine instruction storage offset
    # Calculate element width based on format
    if quant_config.get("format") == "mxint":
        # MXINT: element is just the integer width
        element_width = quant_config["man_width"]
    else:
        # MXFP: element is sign + exp + mantissa
        element_width = quant_config["exp_width"] + quant_config["man_width"] + 1
    bias_width = quant_config["exp_bias_width"]

    if instr_storage_offset == "auto" or instr_storage_offset == -1:
        # Calculate offset to place instructions right after data
        final_offset = calculate_instr_offset_after_data(tensor_data, element_width, bias_width, hbm_row_width)
    elif instr_storage_offset is None:
        # Use default constant
        final_offset = INSTRUCTION_STORAGE_OFFSET
    else:
        # Use provided offset
        final_offset = instr_storage_offset

    # Generate single combined hbm.mem file for RTL simulation (with instructions)
    # Using tensor_data for interleaved format: [elements][scales] per tensor
    generate_hbm(
        blocks=None,
        bias=None,
        element_width=element_width,
        bias_width=bias_width,
        directory=build_path,
        hbm_row_width=hbm_row_width,
        mode="rtl",
        instructions=instructions,
        instr_storage_offset=final_offset,
        tensor_data=tensor_data,
    )

    # Also generate hbm_for_behave_sim.bin for the Rust transactional emulator
    generate_hbm(
        blocks=None,
        bias=None,
        element_width=element_width,
        bias_width=bias_width,
        directory=build_path,
        hbm_row_width=hbm_row_width,
        mode="sim",
        tensor_data=tensor_data,
    )


def init_mem(build_path):
    """Initialize memory files and environment variables for simulation."""
    build_path.mkdir(parents=True, exist_ok=True)

    hbm_file = build_path / "hbm.mem"  # Combined HBM file with data and instructions
    fp_sram_file = build_path / "fp_sram.mem"
    int_sram_file = build_path / "int_sram.mem"
    vector_mem_result_file = build_path / "vector_result.mem"

    # Touch essential output files
    hbm_file.touch()
    fp_sram_file.touch()
    int_sram_file.touch()
    vector_mem_result_file.touch()

    os.environ["FAKE_HBM_INIT_FILE"] = str(hbm_file)
    os.environ["FP_MEM_INIT_FILE"] = str(fp_sram_file)
    os.environ["INT_MEM_INIT_FILE"] = str(int_sram_file)
    os.environ["VECTOR_MEM_RESULT_FILE"] = str(vector_mem_result_file)
