import logging
from pathlib import Path

import torch

from plena_utils import Random_MXFP_Tensor_Generator, Random_MXINT_Tensor_Generator
from .build_sys_tools import PROJECT_PATH, init_mem, env_setup

logger = logging.getLogger("testbench")
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
    precision_settings: dict,
    data_size: int = 256,
    mode: str = "behave_sim",
    asm: str = "attn",
    data=None,
    specified_data_order=None,
    build_path=None,
    hbm_row_width: int = 256,
    mx_format: str | None = None,
    instr_storage_offset: int | None = None,
):
    """
    Create memory files for simulation.

    Args:
        precision_settings: Dict with quantization settings, must contain:
            - block_size: int (e.g., 8)
            - exp_width: int (e.g., 4)
            - man_width: int (e.g., 3)
            - scale_exp_width: int (e.g., 8)
            - int_width: int (e.g., 32)
            - mxint_enable: int (0 or 1, optional)
            - mxint_width: int (e.g., 8, optional)
        data_size: Size of data tensor
        mode: "behave_sim" or other
        asm: Assembly name
        data: Optional data
        specified_data_order: List of data file names in order
        build_path: Path to build directory
        hbm_row_width: HBM row width in bits
        mx_format: Force format to use ("mxfp" or "mxint"). If None, uses mxint_enable flag.
        instr_storage_offset: Byte offset for instruction storage. If None, uses default
            from build_sys_tools (8192). Pass "auto" or -1 to automatically calculate
            offset to place instructions right after data.
    """
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
        "block_size": [1, precision_settings["block_size"]],
    }

    # Determine format: use mx_format if specified, otherwise check mxint_enable flag
    use_mxint = False
    if mx_format is not None:
        use_mxint = mx_format.lower() == "mxint"
    else:
        use_mxint = precision_settings.get("mxint_enable", 0) == 1

    if use_mxint:
        # MXINT quantization config
        mxint_width = precision_settings.get("mxint_width", 8)
        quant_config = {
            "exp_width": precision_settings["scale_exp_width"],  # Shared exponent width
            "man_width": mxint_width,  # Total integer width (including sign)
            "exp_bias_width": precision_settings["scale_exp_width"],
            "block_size": data_config["block_size"],
            "int_width": precision_settings["int_width"],
            "skip_first_dim": False,
            "format": "mxint",
        }
        logger.info(
            f"Using MXINT format: element_width={mxint_width}, scale_width={precision_settings['scale_exp_width']}"
        )
    else:
        # MXFP quantization config
        quant_config = {
            "exp_width": precision_settings["exp_width"],
            "man_width": precision_settings["man_width"],
            "exp_bias_width": precision_settings["scale_exp_width"],
            "block_size": data_config["block_size"],
            "int_width": precision_settings["int_width"],
            "skip_first_dim": False,
            "format": "mxfp",
        }
        element_width = precision_settings["exp_width"] + precision_settings["man_width"] + 1
        logger.info(
            f"Using MXFP format: element_width={element_width}, scale_width={precision_settings['scale_exp_width']}"
        )

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
            if use_mxint:
                file_raw_data = Random_MXINT_Tensor_Generator(
                    shape=tuple(data_config["tensor_size"]),
                    quant_config=quant_config,
                    directory=pt_file.parent,
                    filename=pt_file.name,
                )
            else:
                file_raw_data = Random_MXFP_Tensor_Generator(
                    shape=tuple(data_config["tensor_size"]),
                    quant_config=quant_config,
                    config_settings={},
                    directory=pt_file.parent,
                    filename=pt_file.name,
                )
            file_tensor = file_raw_data.tensor_load()
            blocks, bias = file_raw_data.quantize_tensor(file_tensor)
            memory_data_manager.add_mx_file(pt_file.name, blocks, bias)
        else:
            int_data = torch.load(pt_file)
            memory_data_manager.add_int_file(pt_file.name, int_data)

    env_setup(
        memory_data_manager,
        asm_file.parent,
        data_config,
        quant_config,
        hbm_row_width=hbm_row_width,
        instr_storage_offset=instr_storage_offset,
    )
