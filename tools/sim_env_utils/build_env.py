from .build_sys_tools import *
import logging
from cfl_cocotb import SRC_PATH
from cfl_tools.logger import get_logger
from memory_mapping.rand_gen import Random_MXFP_Tensor_Generator
from utils.load_config import load_toml_config
from pathlib import Path
import torch

logger = get_logger("testbench")
logger.setLevel(logging.DEBUG)

class MemoryDataManager:
    """Manages memory data from pt files, supporting multiple mx and int entries."""
    def __init__(self):
        self.mx_entries = []  # Can have multiple mx entries
        self.int_entries = []  # Can have multiple int entries
    
    def add_mx_file(self, filename, blocks, bias):
        """Add an mx type data entry."""
        self.mx_entries.append({
            "filename": filename,
            "type": "mx",
            "blocks": blocks,
            "bias": bias
        })
    
    def add_int_file(self, filename, data):
        """Add an int type data entry."""
        self.int_entries.append({
            "filename": filename,
            "type": "int",
            "data": data
        })
    
    def get_all_entries(self):
        """Get all entries as a list for iteration."""
        entries = []
        entries.extend(self.mx_entries)
        entries.extend(self.int_entries)
        return entries
    
    def to_dict(self):
        """Convert to dictionary format for backward compatibility if needed."""
        result = {}
        if self.mx_entries:
            result["mx"] = {
                "blocks": [entry["blocks"] for entry in self.mx_entries],
                "bias": [entry["bias"] for entry in self.mx_entries]
            }
        if self.int_entries:
            # For backward compatibility, use "normal" key
            # If multiple int entries, combine them or use the last one
            if len(self.int_entries) == 1:
                result["normal"] = {
                    "data": self.int_entries[0]["data"]
                }
            else:
                # If multiple int entries, use the last one (or could combine)
                result["normal"] = {
                    "data": self.int_entries[-1]["data"]
                }
        return result

def create_mem_for_sim(data_size=256, mode="behave_sim", asm="attn", data=None, specified_data_order = None, build_path = None):

    plena_toml_path = str(SRC_PATH / "definitions" / "plena_settings.toml")
    config_settings = load_toml_config(plena_toml_path, "CONFIG")
    precision_settings = load_toml_config(plena_toml_path, "PRECISION")
    if mode == "behave_sim":
        if build_path is not None:
            asm_file = Path(build_path) / "generated_asm_code.asm"
        else:
            asm_file = Path(PROJECT_PATH / "behavioral_simulator" / "testbench" / "build" / "generated_asm_code.asm")
    else:
        asm_file = Path(PROJECT_PATH / "test" / "Instr_Level_Benchmark" / f"{asm}.asm")

    init_mem(Path(asm_file.parent))

    data_config = {
        "tensor_size": [1, data_size],
        "block_size" : [1, precision_settings["HBM_M_WEIGHT_TYPE"]["block"]],
    }

    quant_config = {
            "exp_width": precision_settings["HBM_V_ACT_TYPE"]["ELEM"]["exponent"],
            "man_width": precision_settings["HBM_V_ACT_TYPE"]["ELEM"]["mantissa"],
            "exp_bias_width": precision_settings["HBM_V_ACT_TYPE"]["SCALE"]["exponent"],
            "block_size": data_config["block_size"],
            "int_width": precision_settings["HBM_V_INT_TYPE"]["DATA_TYPE"]["width"],
            "skip_first_dim": False,
        }

    if mode != "behave_sim":
        grp_blocks = []
        grp_bias = []
        raw_data = Random_MXFP_Tensor_Generator(
            shape=tuple(data_config["tensor_size"]),
            quant_config=quant_config,
            config_settings=config_settings,
            directory=Path(asm_file).parent,
            filename= Path(f"{asm}/fake_test_raw_data.pt")
        )
        raw_data.tensor_gen()
        data = raw_data.tensor_load()
        blocks, bias = raw_data.quantize_tensor(data)
        grp_blocks.append(blocks)
        grp_bias.append(bias)
    else:
        # The provided path (args.data) is a directory. Enumerate all .pt and .pth files within,
        # then load and quantize all of them. Collect the results in a MemoryDataManager.
        if build_path is not None:
            target_dir = Path(build_path)
        else:
            target_dir = PROJECT_PATH / "behavioral_simulator" / "testbench" / "build"
        if specified_data_order is not None:
            pt_files = [target_dir / f"{data}.pt" for data in specified_data_order]
        else:
            pt_files = list(target_dir.glob("*.pt")) + list(target_dir.glob("*.pth"))
        
        memory_data_manager = MemoryDataManager()
        for pt_file in pt_files:
            if pt_file.stem != "int":
                # print("loading file", pt_file)
                file_raw_data = Random_MXFP_Tensor_Generator(
                    shape           =   tuple(data_config["tensor_size"]),
                    quant_config    =   quant_config,
                    config_settings =   config_settings,
                    directory       =   Path(asm_file).parent,
                    filename        =   pt_file
                )
                file_tensor = file_raw_data.tensor_load()
                blocks, bias = file_raw_data.quantize_tensor(file_tensor)
                # Multiple mx files are all kept
                memory_data_manager.add_mx_file(pt_file.name, blocks, bias)
            else:
                # print("loading file", pt_file)
                int_data = torch.load(pt_file)
                memory_data_manager.add_int_file(pt_file.name, int_data)
    # generate_golden_result(data, logger, precision_settings, data_config)
    env_setup(memory_data_manager, asm_file.parent, data_config, quant_config, hbm_row_width=config_settings["HBM_WIDTH"]["value"])

if __name__ == "__main__":
    create_mem_for_sim()
    pass