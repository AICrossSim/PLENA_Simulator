import argparse
import os
from pathlib import Path

import torch
from memory_mapping.memory_map import (
    map_mx_data_to_hbm_for_behave_sim,
    map_normal_data_to_hbm_for_behave_sim,
)
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware
from utils.torch_fp_conversion import fp_2_bin, pack_fp_to_bin

from compiler.assembler.assembly_to_binary import AssemblyToBinary

# Project root is 3 levels up from compiler/sim_env_utils/
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def generate_golden_result(data, logger, precision_settings, data_config):
    qdata, pbexp, pbmant, pbbias = _mx_fp_quantize_hardware(
        data,
        width=precision_settings["ACT_MXFP_EXP_WIDTH"] + precision_settings["ACT_MXFP_MANT_WIDTH"] + 1,
        exponent_width=precision_settings["ACT_MXFP_EXP_WIDTH"],
        exponent_bias_width=precision_settings["MX_SCALE_WIDTH"],
        block_size=data_config["block_size"],
    )
    qele = pbmant * 2**pbexp
    logger.debug("---- mxfp_input ----")
    logger.debug(f"data: {data}")
    logger.debug(f"pbexp: {pbexp}")
    logger.debug(f"pbmant: {pbmant}")
    logger.debug(f"qele: {qele}")
    bin_ele = pack_fp_to_bin(
        pbexp, pbmant, precision_settings["ACT_MXFP_EXP_WIDTH"], precision_settings["ACT_MXFP_MANT_WIDTH"]
    )
    bin_bias = pbbias
    logger.debug("-- hardware bin --")
    logger.debug(f"bin_ele: {bin_ele}")
    logger.debug(f"bin_bias: {bin_bias}")

    logger.debug("---- fp_input ----")
    qdata_fp, bin_fp = fp_2_bin(qdata, precision_settings["V_FP_EXP_WIDTH"], precision_settings["V_FP_MANT_WIDTH"])
    logger.debug(f"qdata_fp: {qdata_fp}")
    logger.debug("--hardware bin--")
    logger.debug(f"bin_fp: {bin_fp}")

    logger.debug("---- exp_out ----")
    exp_fp = torch.exp(qdata_fp)
    qexp_fp, bin_exp_fp = fp_2_bin(exp_fp, precision_settings["V_FP_EXP_WIDTH"], precision_settings["V_FP_MANT_WIDTH"])
    logger.debug(f"exp_fp: {qexp_fp}")
    logger.debug("--hardware bin--")
    logger.debug(f"bin_exp_fp: {bin_exp_fp}")

    logger.debug("---- 1 + exp(x) ----")
    q1_exp_fp, bin_1_exp_fp = fp_2_bin(
        1 + qexp_fp, precision_settings["V_FP_EXP_WIDTH"], precision_settings["V_FP_MANT_WIDTH"]
    )
    logger.debug(f"1_exp_fp: {q1_exp_fp}")
    logger.debug("--hardware bin--")
    logger.debug(f"bin_1_exp_fp: {bin_1_exp_fp}")

    return qdata


def env_setup(
    memory_data_manager,
    build_path: str,
    data_config,
    quant_config,
    hbm_row_width=256,
    test_file_name=None,
    config_settings=None,
):
    """
    Setup environment for simulation using MemoryDataManager.
    Each pt file entry is processed based on its type (mx or int).

    Args:
        memory_data_manager: MemoryDataManager instance or dict (for backward compatibility)
        build_path: Path to build directory
        data_config: Data configuration dictionary
        quant_config: Quantization configuration dictionary
        hbm_row_width: HBM row width
        test_file_name: Optional test file name
        config_settings: Full CONFIG dict from plena_settings.toml.  When
            provided, the per-tensor element region is padded up to
            ``max(HBM_V_Writeback_Amount, HBM_V_Prefetch_Amount) * VLEN``
            bytes so the simulator's H_STORE_V / H_PREFETCH_V iter loop
            (which unconditionally writes AMOUNT rows of VLEN elements per
            call) doesn't clobber the per-tensor scale region.  Required to
            stay in sync with tilelang codegen's
            ``memory_planner/hbm_frame.py::_collect_vector_burst_buffers``.
    """
    isa_file_path = _PROJECT_ROOT / "compiler" / "doc" / "operation.svh"
    config_file_path = _PROJECT_ROOT / "compiler" / "doc" / "configuration.svh"

    if test_file_name is None:
        assembler = AssemblyToBinary(str(isa_file_path), str(config_file_path))
        assembler.generate_binary(build_path / "generated_asm_code.asm", build_path / "generated_machine_code.mem")
    else:
        assembler = AssemblyToBinary(str(isa_file_path), str(config_file_path))
        assembler.generate_binary(build_path / f"{test_file_name}.asm", build_path / f"{test_file_name}.mem")

    entries = memory_data_manager.get_all_entries()

    # Cross-side contract with tilelang codegen
    # (``tilelang/plena/memory_planner/hbm_frame.py::_collect_vector_burst_buffers``).
    # H_PREFETCH_V / H_STORE_V iterate STORE_V_AMOUNT / PREFETCH_V_AMOUNT rows of
    # VLEN elements per call inside the simulator.  When the kernel's actual
    # batch is smaller than AMOUNT, codegen bumps the per-tensor element
    # allocation to ``max(actual, AMOUNT * VLEN)`` so iter N's element writes
    # don't overlap iter 0's scale region.  The Python packer must apply the
    # same bump to keep tensor base offsets aligned between codegen-asm and
    # ``hbm_for_behave_sim.bin``.
    min_element_bytes = 0
    if config_settings is not None:
        burst_amount = max(
            config_settings["HBM_V_Writeback_Amount"]["value"],
            config_settings["HBM_V_Prefetch_Amount"]["value"],
        )
        min_element_bytes = burst_amount * config_settings["VLEN"]["value"]

    # Process each entry based on its type
    for entry in entries:
        if entry["type"] == "mx":
            blocks = entry["blocks"]
            bias = entry["bias"]
            # print("blocks", blocks)
            # print("bias", bias)
            # print(f"Processing mx file: {entry.get('filename', 'unknown')}")
            # map_data_to_fake_hbm_for_rtl_sim(
            #                     blocks          =blocks,
            #                     element_width   =quant_config["exp_width"] + quant_config["man_width"] + 1,
            #                     block_width     =data_config["block_size"][1],
            #                     bias            =bias,
            #                     bias_width      =quant_config["exp_bias_width"],
            #                     combined_blk_dim=hbm_row_width // data_config["block_size"][1],
            #                     directory       =build_path,
            #                     append          =True,
            #                     hbm_row_width   =hbm_row_width)

            map_mx_data_to_hbm_for_behave_sim(
                blocks=blocks,
                element_width=quant_config["exp_width"] + quant_config["man_width"] + 1,
                block_width=data_config["block_size"][1],
                bias=bias,
                bias_width=quant_config["exp_bias_width"],
                directory=build_path,
                append=True,
                hbm_row_width=hbm_row_width,
                min_element_bytes=min_element_bytes,
            )
        elif entry["type"] == "int":
            data = entry["data"]
            # print(f"Processing int file: {entry.get('filename', 'unknown')}")
            map_normal_data_to_hbm_for_behave_sim(
                data=data,
                data_width=quant_config["int_width"],
                directory=build_path,
                append=True,
                hbm_row_width=hbm_row_width,
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the test assembly file")
    args = parser.parse_args()
    return args


def init_mem(build_path):
    """Initialize memory files and environment variables for simulation."""
    build_path.mkdir(parents=True, exist_ok=True)
    hbm_element_file = build_path / "hbm_ele.mem"
    hbm_scale_file = build_path / "hbm_scale.mem"
    hbm_file_for_behave_sim = build_path / "hbm_for_behave_sim.mem"
    instr_file = build_path / "machine_code.mem"

    os.environ["HBM_ELEMENT_FILE"] = str(hbm_element_file)
    os.environ["HBM_SCALE_FILE"] = str(hbm_scale_file)
    os.environ["HBM_FOR_BEHAVE_SIM_FILE"] = str(hbm_file_for_behave_sim)
    os.environ["INSTR_FILE"] = str(instr_file)

    hbm_write_element_m_file = build_path / "hbm_write_m_ele.mem"
    hbm_write_element_v_file = build_path / "hbm_write_v_ele.mem"
    hbm_write_scale_m_file = build_path / "hbm_write_m_scale.mem"
    hbm_write_scale_v_file = build_path / "hbm_write_v_scale.mem"
    vector_mem_result_file = build_path / "vector_result.mem"

    hbm_write_element_m_file.touch()
    hbm_write_element_v_file.touch()
    hbm_write_scale_m_file.touch()
    hbm_write_scale_v_file.touch()
    vector_mem_result_file.touch()

    addr_mapper_file = build_path / "hbm_addr_mapper.mem"
    addr_mapper_file.touch()

    os.environ["VECTOR_MEM_RESULT_FILE"] = str(vector_mem_result_file)
    os.environ["HBM_ADDR_MAPPER_FILE"] = str(addr_mapper_file)
    os.environ["FAKE_HBM_ELEMENT_WRITE_M_FILE"] = str(hbm_write_element_m_file)
    os.environ["FAKE_HBM_ELEMENT_WRITE_V_FILE"] = str(hbm_write_element_v_file)
    os.environ["FAKE_HBM_SCALE_WRITE_M_FILE"] = str(hbm_write_scale_m_file)
    os.environ["FAKE_HBM_SCALE_WRITE_V_FILE"] = str(hbm_write_scale_v_file)
    os.environ["ASM_FILE"] = str(instr_file)


def init_vector_sram():
    vector_mem_file = Path("test/Instr_Level_Benchmark/build/vector_fp_add/vector_result.mem")
    vector_mem_file.touch()
    with open(vector_mem_file, "w") as f:
        f.write("@0\n")
        f.write("3F800000\n")  # 1.0 in IEEE 754
        f.write("40000000\n")  # 2.0
        f.write("40400000\n")  # 3.0
        f.write("40800000\n")  # 4.0
    print(f"Initialized vector SRAM file: {vector_mem_file}")


def init_vector_hbm_for_test():
    build_dir = Path("test/Instr_Level_Benchmark/build/vector_fp_add")
    build_dir.mkdir(parents=True, exist_ok=True)
    hbm_ele = build_dir / "hbm_ele.mem"

    # Initialize vector elements at address 0 (32-bit IEEE754 values shown)
    with open(hbm_ele, "w") as f:
        f.write("@0\n")
        f.write("3F800000\n")  # 1.0
        f.write("40000000\n")  # 2.0
        f.write("40400000\n")  # 3.0
        f.write("40800000\n")  # 4.0
        # add more lines if your VLEN/scale expects more

    print(f"Initialized HBM element file: {hbm_ele}")
