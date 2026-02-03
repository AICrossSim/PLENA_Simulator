import json
import os
from pathlib import Path
from math import log2
from utils import load_svh_settings, load_toml_config
from overall_inference_estimation import model_config


def load_custom_isa_lib(
        json_path: str
):
    with open(json_path, "r") as f:
        custom_isa_lib = json.load(f)
    return custom_isa_lib


class instr_info:
    def __init__(self, name, alone, pipelined, configs):
        self.name       = name
        self.alone      = eval(alone, {}, configs)
        self.pipelined  = eval(pipelined, {}, configs)

def build_instr_model(
    hardware_settings_file: str = "configuration.svh",
    custom_isa_lib_file:    str = "customISA_lib.json"
):
    hardware_settings = load_svh_settings(hardware_settings_file)
    hardware_settings["SA_ACC_CYCLES"] = int(log2(hardware_settings["MLEN"] / hardware_settings["BLEN"]) + 1)
    custom_isa_lib = load_custom_isa_lib(custom_isa_lib_file)

    instr_latency_model = {}
    for instr_name, instr_data in custom_isa_lib.items():
        if "alone" in instr_data and "pipelined" in instr_data:
            alone = instr_data["alone"]
            pipelined = instr_data["pipelined"]
            instr_latency_model[instr_name] = instr_info(instr_name, alone, pipelined, hardware_settings)
        else:
            raise ValueError(f"Instruction '{instr_name}' does not have 'alone' or 'pipelined' fields.")
    
    return instr_latency_model


class instr_latency_model:
    def __init__(self, hardware_settings_file: str = "plena_settings.toml", custom_isa_lib_file: str = "customISA_lib.json", model_config_file: str = "model_config.json"):
        self.instr_model = build_instr_model(hardware_settings_file, custom_isa_lib_file)
        self.model_config_file = model_config_file
        self.hardware_config = load_svh_settings(hardware_settings_file)
        # print(f"hardware config: {self.hardware_config}")
        # print(f"model config: {self.model_config_file}")

    def get_instr_info(self, instr_name):
        return self.model.get(instr_name, None)
    
    def obtain_per_instr_pipelined_latency(self, output_file: str = "instr_latency_model.json"):
        pipelined_latency = {
            instr_name: {
                "pipelined": instr_info.pipelined
            }
            for instr_name, instr_info in self.instr_model.items()
        }
        
        with open(output_file, "w") as f:
            json.dump(pipelined_latency, f, indent=4)
        
        print(f"Average latency model saved to {output_file}")
    
    def obtain_per_instr_alone_latency(self, output_file: str = "instr_alone_latency_model.json"):
        alone_latency = {
            instr_name: {
                "alone": instr_info.alone
            }
            for instr_name, instr_info in self.instr_model.items()
        }
        
        with open(output_file, "w") as f:
            json.dump(alone_latency, f, indent=4)
        
        print(f"Alone latency model saved to {output_file}")

    def obtain_overall_latency(self, updated_config):
        batch_size = 4
        input_seq_len = 2048
        output_seq_len = 1024
        device_num = 1
        hardware_settings = self.hardware_config
        for key, value in updated_config.items():
            hardware_settings[key] = value
        model = model_config(self.model_config_file, hardware_settings, batch_size, input_seq_len, output_seq_len, device_num)
        ttft, tps = model.compute_overall_perf()
        print(f"TTFT: {ttft}")
        print(f"TPS: {tps}")



if __name__ == "__main__":
    import toml
    import time
    start_time = time.time()
    # Project root is 2 levels up from analytic_models/latency/
    config_parent_path      = Path(__file__).resolve().parents[2]
    config_path             = os.path.join(config_parent_path, "src/definitions/configuration.svh")
    toml_path               = os.path.join(config_parent_path, "src/definitions/plena_settings.toml")
    custom_isa_parent_path  = os.path.dirname(os.path.abspath(__file__))
    custom_isa_path         = os.path.join(custom_isa_parent_path, "customISA_lib.json")
    # model_config_path       = os.path.join(config_parent_path, "doc/Model_Lib/qwen2_5_7b.json")
    # model_config_path       = os.path.join(config_parent_path, "doc/Model_Lib/llama-3.3-70b.json")
    model_config_path       = os.path.join(config_parent_path, "doc/Model_Lib/llama-3.1-8b.json")

    model = instr_latency_model(config_path, custom_isa_path, model_config_path)
    test_from_toml = load_toml_config(toml_path, "active")
    model.obtain_overall_latency(test_from_toml)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")