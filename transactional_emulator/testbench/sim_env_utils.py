"""Build simulator memory artifacts from testbench tensor files."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_PATH = str(REPO_ROOT / "tools")
if TOOLS_PATH not in sys.path:
    sys.path.insert(0, TOOLS_PATH)

from compiler.assembler.assembly_to_binary import AssemblyToBinary  # noqa: E402
from memory_mapping.memory_map import (  # noqa: E402
    map_mx_data_to_hbm_for_behave_sim,
    map_normal_data_to_hbm_for_behave_sim,
)
from memory_mapping.rand_gen import RandomMxfpTensorGenerator  # noqa: E402
from utils.load_config import load_toml_config  # noqa: E402
from utils.logger import get_logger  # noqa: E402


logger = get_logger("testbench")
logger.setLevel(logging.DEBUG)


class MemoryDataManager:
    """Collect MX and integer memory payloads for HBM setup."""

    def __init__(self) -> None:
        self.mx_entries = []
        self.int_entries = []

    def add_mx_file(self, filename, blocks, bias, quant_config) -> None:
        self.mx_entries.append(
            {
                "filename": filename,
                "type": "mx",
                "blocks": blocks,
                "bias": bias,
                "quant_config": quant_config,
            }
        )

    def add_int_file(self, filename, data) -> None:
        self.int_entries.append({"filename": filename, "type": "int", "data": data})

    def get_all_entries(self):
        return [*self.mx_entries, *self.int_entries]


def _mx_quant_config(precision_node, precision_settings):
    return {
        "exp_width": precision_node["ELEM"]["exponent"],
        "man_width": precision_node["ELEM"]["mantissa"],
        "exp_bias_width": precision_node["SCALE"]["exponent"],
        "block_size": [1, precision_node["block"]],
        "int_width": precision_settings["HBM_V_INT_TYPE"]["DATA_TYPE"]["width"],
        "skip_first_dim": False,
    }


def _precision_for_tensor(stem: str, precision_settings):
    if stem == "V" or stem.startswith("V_"):
        return precision_settings["HBM_M_KV_TYPE"]
    if stem == "K" or stem.startswith(("K_", "W_")):
        return precision_settings["HBM_M_WEIGHT_TYPE"]
    return precision_settings["HBM_V_ACT_TYPE"]


def create_mem_for_sim(
    data_size=256,
    mode="behave_sim",
    asm="attn",
    data=None,
    specified_data_order=None,
    build_path=None,
):
    plena_toml_path = str(REPO_ROOT / "plena_settings.toml")
    config_settings = load_toml_config(plena_toml_path, "CONFIG")
    precision_settings = load_toml_config(plena_toml_path, "PRECISION")

    if mode == "behave_sim":
        target_dir = Path(build_path) if build_path is not None else REPO_ROOT / "transactional_emulator/testbench/build"
        asm_file = target_dir / "generated_asm_code.asm"
    else:
        asm_file = REPO_ROOT / "test" / "Instr_Level_Benchmark" / f"{asm}.asm"
        target_dir = asm_file.parent

    init_mem(asm_file.parent)

    data_config = {
        "tensor_size": [1, data_size],
        "block_size": [1, precision_settings["HBM_M_WEIGHT_TYPE"]["block"]],
    }
    quant_config = {
        "exp_width": precision_settings["HBM_V_ACT_TYPE"]["ELEM"]["exponent"],
        "man_width": precision_settings["HBM_V_ACT_TYPE"]["ELEM"]["mantissa"],
        "exp_bias_width": precision_settings["HBM_V_ACT_TYPE"]["SCALE"]["exponent"],
        "block_size": data_config["block_size"],
        "int_width": precision_settings["HBM_V_INT_TYPE"]["DATA_TYPE"]["width"],
        "skip_first_dim": False,
    }

    memory_data_manager = MemoryDataManager()
    if mode != "behave_sim":
        raw_data = RandomMxfpTensorGenerator(
            shape=tuple(data_config["tensor_size"]),
            quant_config=quant_config,
            config_settings=config_settings,
            directory=asm_file.parent,
            filename=Path(f"{asm}/fake_test_raw_data.pt"),
        )
        raw_data.tensor_gen()
        raw_tensor = raw_data.tensor_load()
        blocks, bias = raw_data.quantize_tensor(raw_tensor)
        memory_data_manager.add_mx_file("fake_test_raw_data.pt", blocks, bias, quant_config)
    else:
        if specified_data_order is not None:
            pt_files = [target_dir / f"{name}.pt" for name in specified_data_order]
        else:
            pt_files = list(target_dir.glob("*.pt")) + list(target_dir.glob("*.pth"))

        for pt_file in pt_files:
            if pt_file.stem == "int":
                memory_data_manager.add_int_file(pt_file.name, torch.load(pt_file))
                continue

            file_quant_config = _mx_quant_config(_precision_for_tensor(pt_file.stem, precision_settings), precision_settings)
            file_raw_data = RandomMxfpTensorGenerator(
                shape=tuple(data_config["tensor_size"]),
                quant_config=file_quant_config,
                config_settings=config_settings,
                directory=asm_file.parent,
                filename=pt_file,
            )
            file_tensor = file_raw_data.tensor_load()
            blocks, bias = file_raw_data.quantize_tensor(file_tensor)
            memory_data_manager.add_mx_file(pt_file.name, blocks, bias, file_quant_config)

    env_setup(
        memory_data_manager,
        asm_file.parent,
        data_config,
        quant_config,
        hbm_row_width=config_settings["HBM_WIDTH"]["value"],
        logical_row_elements=config_settings["MLEN"]["value"],
    )


def env_setup(
    memory_data_manager,
    build_path: Path,
    data_config,
    quant_config,
    hbm_row_width=256,
    logical_row_elements=None,
) -> None:
    isa_file_path = REPO_ROOT / "PLENA_Compiler" / "doc" / "operation.svh"
    config_file_path = REPO_ROOT / "PLENA_Compiler" / "doc" / "configuration.svh"

    assembler = AssemblyToBinary(str(isa_file_path), str(config_file_path))
    assembler.generate_binary(build_path / "generated_asm_code.asm", build_path / "generated_machine_code.mem")

    for entry in memory_data_manager.get_all_entries():
        if entry["type"] == "mx":
            entry_quant_config = entry.get("quant_config", quant_config)
            map_mx_data_to_hbm_for_behave_sim(
                blocks=entry["blocks"],
                element_width=entry_quant_config["exp_width"] + entry_quant_config["man_width"] + 1,
                block_width=entry_quant_config["block_size"][1],
                bias=entry["bias"],
                bias_width=entry_quant_config["exp_bias_width"],
                directory=build_path,
                append=True,
                hbm_row_width=hbm_row_width,
                logical_row_elements=logical_row_elements,
            )
        elif entry["type"] == "int":
            map_normal_data_to_hbm_for_behave_sim(
                data=entry["data"],
                data_width=quant_config["int_width"],
                directory=build_path,
                append=True,
                hbm_row_width=hbm_row_width,
            )


def init_mem(build_path: Path) -> None:
    build_path.mkdir(parents=True, exist_ok=True)

    hbm_bin_file = build_path / "hbm_for_behave_sim.bin"
    if hbm_bin_file.exists():
        hbm_bin_file.unlink()

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

    os.environ["VECTOR_MEM_RESULT_FILE"] = str(vector_mem_result_file)
    os.environ["FAKE_HBM_ELEMENT_WRITE_M_FILE"] = str(hbm_write_element_m_file)
    os.environ["FAKE_HBM_ELEMENT_WRITE_V_FILE"] = str(hbm_write_element_v_file)
    os.environ["FAKE_HBM_SCALE_WRITE_M_FILE"] = str(hbm_write_scale_m_file)
    os.environ["FAKE_HBM_SCALE_WRITE_V_FILE"] = str(hbm_write_scale_v_file)
    os.environ["ASM_FILE"] = str(instr_file)
