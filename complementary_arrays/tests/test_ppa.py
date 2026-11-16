"""Area and energy model, write-back ratios and the Ramulator configuration of the HBM cross-check."""

import json

import numpy as np
import pytest

from complementary_arrays.ppa import chip as pc
from complementary_arrays.ppa import model as pm
from complementary_arrays.ppa.writeback import ratios
from complementary_arrays.xcheck.run import HERE, ramulator_config


@pytest.mark.parametrize(
    "arrays,mm2",
    [
        ([(4, 352, 128), (8, 208, 256)], 9.0819562048),  # pair of the designs for batch 128
        ([(4, 256, 192)] * 2, 9.786617624576001),  # two identical arrays
        ([(4, 384, 256)], 9.691916763647999),  # single array
    ],
)
def test_array_area(arrays, mm2):
    assert sum(pm.array_area(*a) for a in arrays) / 1e6 == pytest.approx(mm2, rel=1e-12)
    assert pc.chip_area(arrays, "w4g32p")["array"] / 1e6 == pytest.approx(mm2, rel=1e-12)


def test_energy_components_add_up():
    arrays = [(4, 352, 128), (8, 208, 256)]
    area = pc.chip_area(arrays, "w4g32p")
    cls = lambda v: [[v, v / 4, v / 16] for _ in arrays]  # noqa: E731
    counts = dict(
        cycles=10000.0,
        hbm_bytes=4.0e7,
        port_bytes=6.0e7,
        sram_X=1.0e5,
        sram_Y=1.0e5,
        cls_decoded=cls(3.0e7),
        cls_reread_bytes=cls(1.0e6),
        cls_hbm_bytes=cls(1.3e7),
        cls_wport_bytes=cls(2.0e7),
        cls_useful_macs=cls(1.0e8),
        arrays=[
            dict(shape=list(a), matrix_phase_cycles=6000.0, useful_macs=1.0e8, lanes=1024, st_vector=500.0)
            for a in arrays
        ],
    )
    wb = {"B128": {"4": {"ratio": 0.5}, "8": {"ratio": 0.8}}}
    e = pc.energy_layer(counts, arrays, "w4g32p", area, wb, 128)
    on = sum(e[k] for k in ("array_busy", "array_idle", "array_periph", "vector", "decoder", "dataplane", "control"))
    assert e["onchip_logic"] == pytest.approx(on)
    for lv in ("low", "mid", "high"):
        assert e[f"total_{lv}"] == pytest.approx(on + e[f"sram_{lv}"] + e[f"hbm_{lv}"])
    assert e["total_low"] < e["total_mid"] < e["total_high"]
    assert e["array_busy_rowgate"] < e["array_busy"]
    packed = pc.energy_layer(counts, arrays, "w4g32p", area, wb, 128, packed=True)
    assert packed["sram_mid"] < e["sram_mid"]  # no write-back of re-read blocks


def test_writeback_ratios():
    # one window, batch 4, top-2: expert 0 gets 4 tokens, expert 1 gets 3, expert 2 gets 1
    routes = np.array([[[0, 1], [0, 1], [0, 1], [0, 2]]])
    r = ratios(routes, np.array([4]), experts=8)
    assert r["2"]["ratio"] == pytest.approx(2 / 2)  # experts 0 and 1 need one extra pass each
    assert r["4"]["ratio"] is None and r["4"]["reread_experts_per_window"] == 0


def test_ramulator_config():
    c = json.loads(ramulator_config(HERE / "hbm2_2134.json"))
    ms = c["memory_system"]
    assert len(ms["controllers"]) == 512 and all(x == ms["controllers"][0] for x in ms["controllers"])
    assert ms["controllers"][0]["dram"]["impl"] == "HBM2" and ms["controllers"][0]["dram"]["timing"][0] == 2134
    assert list(ms) == ["impl", "clock_ratio", "controllers", "channel_mapper"]
