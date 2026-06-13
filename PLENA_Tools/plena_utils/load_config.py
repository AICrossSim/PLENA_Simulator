import re

import toml


def load_svh_settings(file_path):
    """
    Parse SystemVerilog `parameter` definitions in an .svh/.sv file
    """
    param_pattern = re.compile(r"\s*(?:localparam|parameter)\s+(\w+)\s*=\s*([^;]+);")
    hardware_settings = {}

    with open(file_path) as f:
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

    with open(file_path) as f:
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
    with open(file_path) as f:
        full_toml = toml.load(f)

    # Get the mode section (BEHAVIOR or ANALYTIC)
    mode_section = full_toml.get(mode, {})
    return mode_section.get(section_to_load, {})


def load_precision_from_toml(toml_path, mode="TRANSACTIONAL", data_type="act"):
    """
    Load precision settings from plena_settings.toml.

    Args:
        toml_path: Path to plena_settings.toml
        mode: "TRANSACTIONAL" or "ANALYTIC"
        data_type: Type of data to load precision for ("act", "kv", or "wt")

    Returns:
        dict: precision_settings with block_size, exp_width, man_width,
              scale_exp_width, int_width
    """

    with open(toml_path) as f:
        config = toml.load(f)

    mode_config = config.get(mode, {})
    precision_config = mode_config.get("PRECISION", {})

    # Map data_type to HBM type key
    type_map = {
        "act": "HBM_V_ACT_TYPE",
        "kv": "HBM_V_KV_TYPE",
        "wt": "HBM_M_WEIGHT_TYPE",
    }
    hbm_key = type_map.get(data_type.lower(), "HBM_M_WEIGHT_TYPE")
    hbm_type = precision_config.get(hbm_key, {})
    int_type = precision_config.get("HBM_V_INT_TYPE", {}).get("DATA_TYPE", {})

    elem = hbm_type.get("ELEM", {})
    scale = hbm_type.get("SCALE", {})

    return {
        "block_size": hbm_type.get("block", 8),
        "exp_width": elem.get("exponent", 4),
        "man_width": elem.get("mantissa", 3),
        "scale_exp_width": scale.get("exponent", 8),
        "int_width": int_type.get("width", 32),
    }


def load_precision_from_svh(definitions_path, data_type="act"):
    """
    Load precision settings from hardware SVH files.

    Args:
        definitions_path: Path to the definitions directory containing
                         precision.svh and configuration.svh
        data_type: Type of data to load precision for ("act", "kv", or "wt")

    Returns:
        tuple: (precision_settings, config_settings)
            precision_settings: dict with block_size, exp_width, man_width,
                              scale_exp_width, int_width, mxint_enable, mxint_width
            config_settings: dict with raw config from configuration.svh
    """
    from pathlib import Path

    definitions_path = Path(definitions_path)

    precision_svh = definitions_path / "precision.svh"
    config_svh = definitions_path / "configuration.svh"

    precision = load_svh_settings(str(precision_svh))
    config = load_svh_settings(str(config_svh))

    # Map data_type to parameter prefixes
    prefix_map = {
        "act": "ACT",
        "kv": "KV",
        "wt": "WT",
    }
    prefix = prefix_map.get(data_type.lower(), "ACT")

    # MXFP settings
    mxfp_exp_key = f"{prefix}_MXFP_EXP_WIDTH" if prefix == "ACT" else f"{prefix}_MX_EXP_WIDTH"
    mxfp_man_key = f"{prefix}_MXFP_MANT_WIDTH" if prefix == "ACT" else f"{prefix}_MX_MANT_WIDTH"

    precision_settings = {
        "block_size": precision.get("BLOCK_DIM", 8),
        "exp_width": precision.get(mxfp_exp_key, 4),
        "man_width": precision.get(mxfp_man_key, 3),
        "scale_exp_width": precision.get(f"{prefix}_MX_SCALE_WIDTH", 8),
        "int_width": precision.get("INT_DATA_WIDTH", 32),
        # MXINT settings
        "mxint_enable": precision.get(f"{prefix}_MX_INT_ENABLE", 0),
        "mxint_width": precision.get(f"{prefix}_MX_INT_WIDTH", 8),
    }

    return precision_settings, config


def load_hardware_tile_sizes(definitions_path=None):
    """
    Load hardware tile size parameters (MLEN, BLEN, VLEN, etc.) from configuration.svh.

    Args:
        definitions_path: Path to the definitions directory containing configuration.svh.
                         If None, uses the default src/definitions path.

    Returns:
        dict: Hardware tile sizes with keys:
            - MLEN: Matrix tile size
            - BLEN: Vector/batch tile size
            - VLEN: Vector length
            - HLEN: Head length (if defined)
    """
    from pathlib import Path

    if definitions_path is None:
        # Default to project's src/definitions directory
        definitions_path = Path(__file__).resolve().parents[2] / "src" / "definitions"
    else:
        definitions_path = Path(definitions_path)

    config_svh = definitions_path / "configuration.svh"
    config = load_svh_settings(str(config_svh))

    return {
        "MLEN": config.get("MLEN", 16),
        "BLEN": config.get("BLEN", 8),
        "VLEN": config.get("VLEN", 16),
        "HLEN": config.get("HLEN", 8),
        "HBM_V_Prefetch_Amount": config.get("HBM_V_Prefetch_Amount", 4),
    }


def get_quant_config_for_format(mx_format: str, precision_settings: dict) -> dict:
    """Get the appropriate quant_config based on mx_format.

    Args:
        mx_format: "mxfp", "mxint", or None (auto-detect from precision_settings)
        precision_settings: Dict with precision settings from precision.svh

    Returns:
        quant_config dict with appropriate parameters for the format
    """
    # Determine format
    use_mxint = False
    if mx_format is not None:
        use_mxint = mx_format.lower() == "mxint"
    else:
        use_mxint = precision_settings.get("mxint_enable", 0) == 1

    block_size = precision_settings.get("block_size", 8)

    if use_mxint:
        # MXINT: element is [sign(1)][magnitude(int_width-1)]
        mxint_width = precision_settings.get("mxint_width", 8)
        return {
            "exp_width": precision_settings.get("scale_exp_width", 8),  # Shared scale width
            "man_width": mxint_width,  # Total integer width (including sign)
            "exp_bias_width": precision_settings.get("scale_exp_width", 8),
            "block_size": [1, block_size],
            "int_width": precision_settings.get("int_width", 32),
            "skip_first_dim": False,
            "format": "mxint",
        }
    else:
        # MXFP: element is [sign(1)][exp(exp_width)][man(man_width)]
        return {
            "exp_width": precision_settings.get("exp_width", 4),
            "man_width": precision_settings.get("man_width", 3),
            "exp_bias_width": precision_settings.get("scale_exp_width", 8),
            "block_size": [1, block_size],
            "int_width": precision_settings.get("int_width", 32),
            "skip_first_dim": False,
            "format": "mxfp",
        }
