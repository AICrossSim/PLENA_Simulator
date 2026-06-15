import logging
import os
from pathlib import Path

import torch
from memory_mapping.rand_gen import RandomMxfpTensorGenerator
from utils.load_config import load_toml_config
from utils.logger import get_logger

from .build_sys_tools import env_setup, init_mem

# Project root is 3 levels up from compiler/sim_env_utils/
_PROJECT_ROOT = Path(__file__).parent.parent.parent


def _resolve_plena_toml_path(default: Path) -> str:
    """Mirror the Rust simulator's PLENA_CONFIG-aware toml resolution
    (``transactional_emulator/src/load_config.rs::load_config``).

    Without this the Python packer keeps reading ``<sim_root>/plena_settings.toml``
    even when the user runs the kernel against a different config like
    ``configs/config_2.toml`` — yielding mismatched byte layouts between the
    codegen-asm (parameterized by ``$PLENA_CONFIG``) and the packed HBM bytes
    (parameterized by the project-root default toml).  Lookup order:

      1. ``$PLENA_CONFIG`` env var (file path, never a config name)
      2. Fallback to the project-root ``plena_settings.toml``
    """
    env_value = os.environ.get("PLENA_CONFIG", "").strip()
    if env_value:
        return env_value
    return str(default)

logger = get_logger("testbench")
logger.setLevel(logging.DEBUG)


class MemoryDataManager:
    """Manages memory data from pt files, supporting multiple mx and int entries."""

    def __init__(self):
        self.mx_entries = []  # Can have multiple mx entries
        self.int_entries = []  # Can have multiple int entries

    def add_mx_file(self, filename, blocks, bias):
        """Add an mx type data entry."""
        self.mx_entries.append({"filename": filename, "type": "mx", "blocks": blocks, "bias": bias})

    def add_int_file(self, filename, data):
        """Add an int type data entry."""
        self.int_entries.append({"filename": filename, "type": "int", "data": data})

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
                "bias": [entry["bias"] for entry in self.mx_entries],
            }
        if self.int_entries:
            # For backward compatibility, use "normal" key
            # If multiple int entries, combine them or use the last one
            if len(self.int_entries) == 1:
                result["normal"] = {"data": self.int_entries[0]["data"]}
            else:
                # If multiple int entries, use the last one (or could combine)
                result["normal"] = {"data": self.int_entries[-1]["data"]}
        return result


def create_mem_for_sim(
    data_size=256, mode="behave_sim", asm="attn", data=None, specified_data_order=None, build_path=None
):

    plena_toml_path = _resolve_plena_toml_path(_PROJECT_ROOT / "plena_settings.toml")
    config_settings = load_toml_config(plena_toml_path, "CONFIG")
    precision_settings = load_toml_config(plena_toml_path, "PRECISION")
    if mode == "behave_sim":
        if build_path is not None:
            asm_file = Path(build_path) / "generated_asm_code.asm"
        else:
            asm_file = Path(_PROJECT_ROOT / "transactional_emulator" / "testbench" / "build" / "generated_asm_code.asm")
    else:
        asm_file = Path(_PROJECT_ROOT / "test" / "Instr_Level_Benchmark" / f"{asm}.asm")

    init_mem(Path(asm_file.parent))

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

    if mode != "behave_sim":
        grp_blocks = []
        grp_bias = []
        raw_data = RandomMxfpTensorGenerator(
            shape=tuple(data_config["tensor_size"]),
            quant_config=quant_config,
            config_settings=config_settings,
            directory=Path(asm_file).parent,
            filename=Path(f"{asm}/fake_test_raw_data.pt"),
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
            target_dir = _PROJECT_ROOT / "transactional_emulator" / "testbench" / "build"
        if specified_data_order is not None:
            pt_files = [target_dir / f"{data}.pt" for data in specified_data_order]
        else:
            pt_files = list(target_dir.glob("*.pt")) + list(target_dir.glob("*.pth"))

        memory_data_manager = MemoryDataManager()
        for pt_file in pt_files:
            if pt_file.stem != "int":
                # print("loading file", pt_file)
                file_raw_data = RandomMxfpTensorGenerator(
                    shape=tuple(data_config["tensor_size"]),
                    quant_config=quant_config,
                    config_settings=config_settings,
                    directory=Path(asm_file).parent,
                    filename=pt_file,
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
    env_setup(
        memory_data_manager,
        asm_file.parent,
        data_config,
        quant_config,
        hbm_row_width=config_settings["HBM_WIDTH"]["value"],
        # Pass full config_settings so env_setup can pad the element region to
        # max(STORE_V, PREFETCH_V) * VLEN bytes (cross-side contract with
        # tilelang codegen's ``memory_planner/hbm_frame.py``).
        config_settings=config_settings,
    )


if __name__ == "__main__":
    create_mem_for_sim()
    pass
