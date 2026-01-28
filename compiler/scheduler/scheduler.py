import json

def try_eval(expr, vars_):
    try:
        return eval(expr, {}, vars_)
    except Exception as e:
        return f"Error: {e}"

def generate_mem_layout(
    hardware_config: dict,
    model_config: dict,
    mem_layout_lib: str = "mem_layout_lib.json"
) -> None:
    """
    Generate memory layout based on hardware and model configurations.
    
    Args:
        hardware_config (dict): Hardware configuration settings.
        model_config (dict): Model configuration settings.
        output_file (str): Path to save the generated memory layout.
    """
    with open(mem_layout_lib, "r") as f:
        mem_layout = json.load(f)
    ref_info = hardware_config | model_config
    out = {}

    for cat, vals in mem_layout.items():
        out[cat] = {}
        for k, v in vals.items():
            if isinstance(v, dict) and "addr" in v:
                expr = v["addr"].strip()
                out[cat][k] = try_eval(expr, ref_info) if expr else None
            elif isinstance(v, str):
                expr = v.strip()
                out[cat][k] = try_eval(expr, ref_info) if expr else None
    return out
    
def gen_scheduler(
    hardware_config: dict,
    model_config: dict,
    mem_layout_lib: str = "mem_layout_lib.json",
    reg_assignment_lib: str = "reg_assignment_lib.json"
) -> dict:
    """
    Generate scheduler based on hardware and model configurations.
    
    Args:
        hardware_config (dict): Hardware configuration settings.
        model_config (dict): Model configuration settings.
        mem_layout_lib (str): Path to the memory layout library JSON file.
    
    Returns:
        dict: Generated scheduler configuration.
    """
    mem_layout = generate_mem_layout(hardware_config, model_config, mem_layout_lib)
    with open(reg_assignment_lib, "r") as f:
        register_assignment = json.load(f)

    scheduler = {
        "memory_layout": mem_layout,
        "register_assignment": register_assignment
    }
    
    return scheduler