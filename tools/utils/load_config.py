import re
import toml

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


def load_json(file_path):
    """
    Load machine learning model configuration from a JSON file.
    """
    import json
    with open(file_path, "r") as f:
        ml_config = json.load(f)
    return ml_config


def load_toml_config(file_path, section_to_load=None, mode="BEHAVIOR"):
    """
    Load configuration from TOML file.

    Args:
        file_path: Path to the TOML file
        section_to_load: Section name (CONFIG, PRECISION, LATENCY)
        mode: Top-level mode section (BEHAVIOR or ANALYTIC)

    Returns:
        dict: The requested configuration section
    """
    with open(file_path, "r") as f:
        full_toml = toml.load(f)

    # Get the mode section (BEHAVIOR or ANALYTIC)
    mode_section = full_toml.get(mode, {})
    return mode_section.get(section_to_load, {})