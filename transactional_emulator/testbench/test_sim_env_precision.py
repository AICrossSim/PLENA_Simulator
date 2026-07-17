from transactional_emulator.testbench.sim_env_utils import (
    _mx_element_width,
    _mx_quant_config,
)


def _settings(element: dict, block: int = 64) -> dict:
    return {
        "HBM_V_ACT_TYPE": {
            "format": "Mx",
            "block": block,
            "ELEM": element,
            "SCALE": {"type": "Fp", "exponent": 8, "mantissa": 0},
        },
        "HBM_V_INT_TYPE": {
            "format": "Plain",
            "DATA_TYPE": {"type": "Int", "width": 32},
        },
    }


def test_mxint_quant_config_uses_integer_element_width() -> None:
    settings = _settings({"type": "Int", "width": 4})
    config = _mx_quant_config(settings["HBM_V_ACT_TYPE"], settings)

    assert config["format"] == "mxint"
    assert config["man_width"] == 4
    assert config["exp_width"] == 8
    assert _mx_element_width(config) == 4


def test_mxfp_quant_config_preserves_exp_and_mantissa() -> None:
    settings = _settings(
        {"type": "Fp", "sign": True, "exponent": 4, "mantissa": 3},
        block=8,
    )
    config = _mx_quant_config(settings["HBM_V_ACT_TYPE"], settings)

    assert config["format"] == "mxfp"
    assert config["exp_width"] == 4
    assert config["man_width"] == 3
    assert _mx_element_width(config) == 8
