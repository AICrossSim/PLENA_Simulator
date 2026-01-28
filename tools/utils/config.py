import toml
import re
import os
import argparse
import sys

def update_global_define(file_path, selected_mode):
    modes = ["SIMULATION", "ASIC", "FPGA"]
    if selected_mode not in modes:
        print(f"Error: Unsupported mode '{selected_mode}'. Must be one of {modes}.")
        return

    with open(file_path, 'w') as f:
        f.write("`ifndef GLOBAL_DEFINE_VH\n")
        f.write(f"`define {selected_mode}\n")
        f.write("`endif\n")

    print(f"Updated {file_path} with mode {selected_mode}.")



def patch_config_svh_from_toml(
    toml_path: str,
    section: str,
    svh_path: str
):
    """Configures the SystemVerilog header file based on the TOML [active] configuration."""
    pkg_name = {"CONFIG": "configuration_pkg", "PRECISION": "precision_pkg", "INSTR": "instruction_pkg"}.get(section, None)

    with open(toml_path, "r") as f:
        data = toml.load(f)
    toml_config = data.get(section, {})

    if not toml_config:
        raise ValueError(f"No {section} section found in TOML")
    mode = "active"
    hardware_settings = {
        param: values.get(mode)
        for param, values in toml_config.items()
        if mode in values
    } 
    print("svh_path:", svh_path)
    with open(svh_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    in_configuration_pkg = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(f"package {pkg_name}"):
            in_configuration_pkg = True
        elif stripped.startswith("endpackage") and in_configuration_pkg:
            in_configuration_pkg = False

        if in_configuration_pkg:
            match = re.match(r'\s*parameter\s+(\w+)\s*=.*;', line)
            if match:
                param_name = match.group(1)
                if param_name in hardware_settings:
                    new_value = hardware_settings[param_name]
                    indent = re.match(r'^(\s*)', line).group(1)
                    new_line = f"{indent}parameter   {param_name} = {new_value};\n"
                    new_lines.append(new_line)
                    continue

        new_lines.append(line)

    with open(svh_path, "w") as f:
        f.writelines(new_lines)

def parse_config_string(config_str):
    param_dict = {}
    if config_str:
        pairs = config_str.strip().split()
        for pair in pairs:
            key, val = pair.split('=')
            param_dict[key.strip()] = int(val.strip())
    return param_dict


def modify_toml_file(
    mode: str = None,
    toml_path: str = "plena_settings.toml",
    section: str = "CONFIG",
    config_params: dict = None
):
    with open(toml_path, "r") as f:
        data = toml.load(f)
        toml_config = data.get(section, {})

        if not toml_config:
            raise ValueError(f"No {section} section found in TOML")

        if mode is not None:
            found_any = False
            for param, values in toml_config.items():
                if mode in values:
                    found_any = True
                    # Copy mode value to active
                    toml_config[param]['active'] = values[mode]
            if not found_any:
                raise ValueError(f"Mode '{mode}' not found in any parameters.")
        else:
            for param, values in toml_config.items():
                if mode in values:
                    found_any = True
                    # Copy mode default to active first
                    toml_config[param]['active'] = values["default"]
            
        if config_params is not None:
            for param, value in config_params.items():
                if param in toml_config:
                    toml_config[param]['active'] = value
        
        # Write back the modified toml
        data[section] = toml_config
        with open(toml_path, "w") as f:
            toml.dump(data, f)
        print(f"Updated 'active' values in {toml_path} with mode '{mode}'.")

def auto_config (
    config_svh_path: str = "default",
    precision_svh_path: str = "default",
    toml_path: str = "config/plena_settings.toml",
    settings: dict = None
):
    modify_toml_file(
        toml_path=toml_path,
        section="CONFIG",
        config_params=settings
    )
    patch_config_svh_from_toml(
        toml_path=toml_path,
        section="CONFIG",
        svh_path=config_svh_path
    )
    
    modify_toml_file(
        toml_path=toml_path,
        section="PRECISION",
        config_params=settings
    )
    patch_config_svh_from_toml(
        toml_path=toml_path,
        section="PRECISION",
        svh_path=precision_svh_path
    )



def main():
    parser = argparse.ArgumentParser(description="Update TOML active values.")
    parser.add_argument("--config", default=None, help="Parameter to update or '*' for all")
    parser.add_argument("--precision", default=None, help="Parameter to update or '*' for all")
    parser.add_argument("--mode",   default=None, help="Mode to use for copying (e.g. ASIC, SIMULATION, etc.)")
    args = parser.parse_args()
    config_settings = parse_config_string(args.config) if args.config else None
    precision_settings = parse_config_string(args.precision) if args.precision else None
    parent_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(parent_path, "plena_settings.toml")
    config_svh_path    = os.path.join(parent_path, "configuration.svh")
    precision_svh_path = os.path.join(parent_path, "precision.svh")

    if args.mode is not None:
        update_global_define(
            file_path=os.path.join(parent_path, "global_define.vh"),
            selected_mode=args.mode
        )

    if config_settings is not None:
        modify_toml_file(
            mode=args.mode,
            toml_path=config_path,
            section="CONFIG",
            config_params=config_settings
        )
        patch_config_svh_from_toml(
            toml_path=config_path,
            section="CONFIG",
            svh_path=config_svh_path
        )
    
    if precision_settings is not None:
        modify_toml_file(
            mode=args.mode,
            toml_path=config_path,
            section="PRECISION",
            config_params=precision_settings
        )
        patch_config_svh_from_toml(
            toml_path=config_path,
            section="PRECISION",
            svh_path=precision_svh_path
        )

if __name__ == "__main__":
    main()