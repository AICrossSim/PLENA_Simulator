import re

# TODO: use the lib in tools.utils
def load_svh_settings(file_path):
    """
    Parse SystemVerilog `parameter` definitions in an .svh/.sv file
    """
    param_pattern = re.compile(r'\s*parameter\s+(\w+)\s*=\s*([^;]+);')
    hardware_settings = {}

    with open(file_path, "r") as f:
        for line in f:
            match = param_pattern.match(line)
            if match:
                name, value_str = match.groups()
                value_str = value_str.strip()
                # Try integer conversion first
                try:
                    value = int(value_str)
                except ValueError:
                    # Fallback to raw string (could be expression or real number)
                    continue
                hardware_settings[name] = value
    return hardware_settings


def HardwareParser(config_file, precision_file):
    """
    Parse SystemVerilog `parameter` definitions in an .svh/.sv file
    """

    hardware_settings = load_svh_settings(config_file)
    precision_settings = load_svh_settings(precision_file)
    hardware_settings["wt_block_width"]     = (precision_settings.get("WT_MX_MANT_WIDTH", 3) + precision_settings.get("WT_MX_EXP_WIDTH", 4) + 1) * hardware_settings.get("BLOCK_DIM", 4)
    hardware_settings["kv_block_width"]     = (precision_settings.get("KV_MX_MANT_WIDTH", 3) + precision_settings.get("KV_MX_EXP_WIDTH", 4) + 1) * hardware_settings.get("BLOCK_DIM", 4)
    hardware_settings["act_block_width"]    = (precision_settings.get("ACT_MX_MANT_WIDTH", 3) + precision_settings.get("ACT_MX_EXP_WIDTH", 4) + 1) * hardware_settings.get("BLOCK_DIM", 4)
    hardware_settings["scale_width"]        = precision_settings.get("MX_SCALE_WIDTH", 3) + precision_settings.get("SCALE_MX_EXP_WIDTH", 4) + 1
    hardware_settings["block_dim"]         = hardware_settings.get("BLOCK_DIM", 4)

    return hardware_settings
