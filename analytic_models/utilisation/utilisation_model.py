
import toml
import os
import json
from pathlib import Path
from utils import load_toml_config, load_json, load_svh_settings

from attainable import attn_model_config

class utilisation_model:
    def __init__(self, hardware_settings_file: str = "plena_settings.toml", precision_settings_file: str = "precision.toml", unit_info_file: str = "unit_info.json"):
        self.unit_info = load_json(unit_info_file)

        config_settings = load_svh_settings(hardware_settings_file)
        precision_settings = load_svh_settings(precision_settings_file)
        self.hardware_settings = {**config_settings, **precision_settings}
    def obtain_resource_utilisation(self, updated_config):
        resource_utilisation = 0
        hardware_settings = self.hardware_settings
        #TODOs
        # for key, value in updated_config.items():
        #     hardware_settings[key] = value
        custom_config= {
            "MLEN": 1024,
            "BLEN": 64,
            "VLEN": 1024,
            "WT_MX_MANT_WIDTH": 1,
            "WT_MX_EXP_WIDTH": 2,
            "KV_ELEMENT_WIDTH": 8,
            "BLOCK_DIM": 8,
            "MX_SCALE_WIDTH" : 8,
            # "ACT_MXFP_MANT_WIDTH": 3,
            # "ACT_MXFP_EXP_WIDTH": 4,
            "ACT_ELEMENT_WIDTH" : 8,
            "FP_EXP_WIDTH": 3,
            "FP_MANT_WIDTH": 4,
            "HBM_ELE_WIDTH": 512,
            "HBM_SCALE_WIDTH": 512,
            "MATRIX_SRAM_DEPTH": 4096,
            "VECTOR_SRAM_DEPTH": 4096,
            "INT_DATA_WIDTH": 32,
            "INT_SRAM_DEPTH": 256,
            "FP_SRAM_DEPTH": 256,
            
        }

        for unit, info in self.unit_info.items():
            if "Coefficients" in info and "Relationship" in info:     
                custom_config.update(info["Coefficients"])
                relationship = info["Relationship"]
                resource_utilisation += eval(relationship, {}, custom_config)
        return resource_utilisation


class attainable_GEMM_model:
    def __init__(self, hardware_settings_file: str = "plena_settings.toml", model_config_file: str = "model_config.json"):
        self.model_config_file = model_config_file
        self.hardware_config = load_svh_settings(hardware_settings_file)
        # print(f"hardware config: {self.hardware_config}")
        # print(f"model config: {self.model_config_file}")

    def obtain_overall_utilization(self, updated_config):
        overall_latency = 0
        batch_size = 16
        input_seq_len = 5600
        output_seq_len = 80000
        device_num = 4
        hardware_settings = self.hardware_config
        for key, value in updated_config.items():
            hardware_settings[key] = value
        model = attn_model_config(self.model_config_file, hardware_settings, batch_size, input_seq_len, output_seq_len, device_num)
        utilization = model.compute_overall_perf()
        print(f"Overall utilization: {utilization} FLOPS")
        return utilization




if __name__ == "__main__":
    import toml
    # Project root is 2 levels up from analytic_models/utilisation/
    project_root = Path(__file__).resolve().parents[2]
    # JSON files are in the same directory as this file
    current_dir = Path(__file__).resolve().parent

    config_path     = os.path.join(project_root, "src/definitions/configuration.svh")
    precision_path  = os.path.join(project_root, "src/definitions/precision.svh")
    toml_path       = os.path.join(project_root, "plena_settings.toml")
    unit_info_file  = os.path.join(current_dir, "individual_units_lib.json")
    model_config_path  = os.path.join(project_root, "doc/Model_Lib/llama-3.1-8b.json")
    # utilisation = utilisation_model(config_path, precision_path, unit_info_file)
    # test_from_toml = load_toml_config(toml_path, "active")
    # print(f"Resource Utilisation: {utilisation.obtain_resource_utilisation(test_from_toml)}")

    model = attainable_GEMM_model(config_path, model_config_path)
    test_from_toml = load_toml_config(toml_path, "active")
    print("test_from_toml", test_from_toml)
    overall_utilization = model.obtain_overall_utilization(test_from_toml)
    print(f"Overall utilization: {overall_utilization} FLOPS")