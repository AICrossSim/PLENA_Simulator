"""Bridge a local PLENA_RTL configuration into the simulator TOML."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path
import re

import tomlkit


_PARAM_RE = re.compile(r"\s*(?:localparam|parameter)\s+(?:[\w]+\s+)*(?P<name>\w+)\s*=\s*(?P<value>[^;]+);")


def _load_svh_ints(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        line = line.split("//", 1)[0]
        match = _PARAM_RE.match(line)
        if not match:
            continue
        value = match.group("value").strip()
        try:
            values[match.group("name")] = int(value, 0)
        except ValueError:
            continue
    return values


def _set_value(config, section: str, name: str, value: int) -> None:
    if name in config["BEHAVIOR"][section]:
        config["BEHAVIOR"][section][name]["value"] = value


def _set_plain_fp(node, *, sign: bool, exponent: int, mantissa: int) -> None:
    node["format"] = "Plain"
    node["DATA_TYPE"]["type"] = "Fp"
    node["DATA_TYPE"]["sign"] = sign
    node["DATA_TYPE"]["exponent"] = exponent
    node["DATA_TYPE"]["mantissa"] = mantissa


def _set_mx_fp(node, *, block: int, elem_exp: int, elem_mant: int, scale_exp: int) -> None:
    node["format"] = "Mx"
    node["block"] = block
    node["ELEM"]["type"] = "Fp"
    node["ELEM"]["sign"] = True
    node["ELEM"]["exponent"] = elem_exp
    node["ELEM"]["mantissa"] = elem_mant
    node["SCALE"]["type"] = "Fp"
    node["SCALE"]["sign"] = False
    node["SCALE"]["exponent"] = scale_exp
    node["SCALE"]["mantissa"] = 0


def _set_scalar_fp(node, *, sign: bool, exponent: int, mantissa: int) -> None:
    node["type"] = "Fp"
    node["sign"] = sign
    node["exponent"] = exponent
    node["mantissa"] = mantissa


def default_rtl_root(repo_root: Path) -> Path:
    env_path = os.environ.get("PLENA_RTL_LOCAL") or os.environ.get("PLENA_RTL_ROOT")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return (repo_root.parent / "PLENA_RTL").resolve()


def apply_rtl_settings_to_toml(config, rtl_root: Path) -> dict[str, int]:
    definitions = rtl_root / "src" / "definitions"
    config_svh = definitions / "configuration.svh"
    precision_svh = definitions / "precision.svh"
    if not config_svh.exists() or not precision_svh.exists():
        raise FileNotFoundError(f"Missing RTL configuration files under {definitions}")

    rtl_config = _load_svh_ints(config_svh)
    rtl_precision = _load_svh_ints(precision_svh)

    mlen = rtl_config["MLEN"]
    vlen = rtl_config["VLEN"]
    blen = rtl_config["BLEN"]
    hlen = rtl_config.get("HLEN", config["BEHAVIOR"]["CONFIG"]["HLEN"]["value"])
    broadcast_amount = rtl_config.get("BROADCAST_AMOUNT", mlen // hlen)

    _set_value(config, "CONFIG", "MLEN", mlen)
    _set_value(config, "CONFIG", "VLEN", vlen)
    _set_value(config, "CONFIG", "BLEN", blen)
    _set_value(config, "CONFIG", "HLEN", hlen)
    _set_value(config, "CONFIG", "BROADCAST_AMOUNT", broadcast_amount)
    _set_value(config, "CONFIG", "HBM_M_Prefetch_Amount", rtl_config.get("HBM_M_Prefetch_Amount", 16))
    _set_value(config, "CONFIG", "HBM_V_Prefetch_Amount", rtl_config.get("HBM_V_Prefetch_Amount", 16))
    _set_value(config, "CONFIG", "HBM_V_Writeback_Amount", rtl_config.get("HBM_V_Writeback_Amount", 4))
    _set_value(config, "CONFIG", "MATRIX_SRAM_SIZE", rtl_config.get("MATRIX_SRAM_DEPTH", 1024))
    _set_value(config, "CONFIG", "VECTOR_SRAM_SIZE", rtl_config.get("VECTOR_SRAM_DEPTH", 1024))

    wt_exp = rtl_precision.get("WT_MX_EXP_WIDTH", 4)
    wt_mant = rtl_precision.get("WT_MX_MANT_WIDTH", 3)
    kv_exp = rtl_precision.get("KV_MX_EXP_WIDTH", wt_exp)
    kv_mant = rtl_precision.get("KV_MX_MANT_WIDTH", wt_mant)
    act_exp = rtl_precision.get("ACT_MXFP_EXP_WIDTH", 4)
    act_mant = rtl_precision.get("ACT_MXFP_MANT_WIDTH", 3)
    scale_exp = rtl_precision.get("MX_SCALE_WIDTH", 8)
    block = rtl_precision.get("BLOCK_DIM", 4)

    raw_hbm_width = (wt_mant + wt_exp + 1) * mlen
    hbm_width = 1 << ((raw_hbm_width * 2 - 1).bit_length())
    _set_value(config, "CONFIG", "HBM_WIDTH", rtl_config.get("HBM_WIDTH", hbm_width))

    precision = config["BEHAVIOR"]["PRECISION"]
    _set_plain_fp(
        precision["MATRIX_SRAM_TYPE"],
        sign=True,
        exponent=rtl_precision.get("M_FP_EXP_WIDTH", 8),
        mantissa=rtl_precision.get("M_FP_MANT_WIDTH", 7),
    )
    _set_plain_fp(
        precision["VECTOR_SRAM_TYPE"],
        sign=True,
        exponent=rtl_precision.get("V_FP_EXP_WIDTH", 8),
        mantissa=rtl_precision.get("V_FP_MANT_WIDTH", 7),
    )
    _set_mx_fp(precision["HBM_M_WEIGHT_TYPE"], block=block, elem_exp=wt_exp, elem_mant=wt_mant, scale_exp=scale_exp)
    _set_mx_fp(precision["HBM_M_KV_TYPE"], block=block, elem_exp=kv_exp, elem_mant=kv_mant, scale_exp=scale_exp)
    _set_mx_fp(precision["HBM_V_ACT_TYPE"], block=block, elem_exp=act_exp, elem_mant=act_mant, scale_exp=scale_exp)
    _set_mx_fp(precision["HBM_V_KV_TYPE"], block=block, elem_exp=kv_exp, elem_mant=kv_mant, scale_exp=scale_exp)
    _set_scalar_fp(
        precision["SCALAR_FP"],
        sign=True,
        exponent=rtl_precision.get("S_FP_EXP_WIDTH", 8),
        mantissa=rtl_precision.get("S_FP_MANT_WIDTH", 7),
    )

    return {
        "MLEN": mlen,
        "VLEN": vlen,
        "BLEN": blen,
        "HLEN": hlen,
        "BROADCAST_AMOUNT": broadcast_amount,
        "M_FP_EXP_WIDTH": rtl_precision.get("M_FP_EXP_WIDTH", 8),
        "M_FP_MANT_WIDTH": rtl_precision.get("M_FP_MANT_WIDTH", 7),
        "V_FP_EXP_WIDTH": rtl_precision.get("V_FP_EXP_WIDTH", 8),
        "V_FP_MANT_WIDTH": rtl_precision.get("V_FP_MANT_WIDTH", 7),
        "S_FP_EXP_WIDTH": rtl_precision.get("S_FP_EXP_WIDTH", 8),
        "S_FP_MANT_WIDTH": rtl_precision.get("S_FP_MANT_WIDTH", 7),
    }


@contextmanager
def rtl_plena_settings(plena_toml: Path, rtl_root: Path) -> Iterator[dict[str, int]]:
    """Temporarily rewrite plena_settings.toml from RTL .svh files and restore it."""
    original = plena_toml.read_text()
    previous_env = os.environ.get("PLENA_SETTINGS_TOML")
    try:
        config = tomlkit.loads(original)
        summary = apply_rtl_settings_to_toml(config, rtl_root)
        plena_toml.write_text(tomlkit.dumps(config))
        os.environ["PLENA_SETTINGS_TOML"] = str(plena_toml)
        yield summary
    finally:
        plena_toml.write_text(original)
        if previous_env is None:
            os.environ.pop("PLENA_SETTINGS_TOML", None)
        else:
            os.environ["PLENA_SETTINGS_TOML"] = previous_env
