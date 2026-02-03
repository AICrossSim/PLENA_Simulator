from utils import auto_config
from utils import load_json
from pathlib import Path
import os

def experiment_module_auto_config (
    module_name: str,
    testcase:   str,
    toml_path:  str,
    config_svh_path:   str,
    precision_svh_path: str,
    experiment_samples: str = "single_unit_experiment.json"
):
    """
    Automatically configure the module based on the provided module name and testcase.
    
    Args:
        module_name (str): Name of the module to configure.
        testcase (str): Test case to use for configuration. Defaults to "default".
    """
    config_data = load_json(experiment_samples)
    testcase_settings = config_data.get(module_name, {}).get(testcase, {}).get("Config", {})
    # Modify the TOML file with the loaded configuration
    if not testcase_settings:
        raise ValueError(f"No settings found for module '{module_name}' and testcase '{testcase}'.")
    auto_config(
        config_svh_path      = config_svh_path,
        precision_svh_path   = precision_svh_path,
        toml_path     = toml_path,
        settings      = testcase_settings
    )

if __name__ == "__main__":
    # Project root is 2 levels up from analytic_models/utilisation/
    project_root = Path(__file__).resolve().parents[2]
    # JSON files are in the same directory as this file
    current_dir = Path(__file__).resolve().parent

    toml_path = os.path.join(project_root, "plena_settings.toml")
    config_svh_path = os.path.join(project_root, "src/definitions/configuration.svh")
    precision_svh_path = os.path.join(project_root, "src/definitions/precision.svh")
    experiment_samples = os.path.join(current_dir, "single_unit_experiment.json")
    
    # Example usage
    experiment_module_auto_config(
        module_name = "MatrixMachine",
        testcase    = "Testcase 3",
        toml_path   = toml_path,
        config_svh_path    = config_svh_path,
        precision_svh_path = precision_svh_path,
        experiment_samples = experiment_samples
    )