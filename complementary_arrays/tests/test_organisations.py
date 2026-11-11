"""Organisations, dispatch policies and the tuning space."""

from complementary_arrays.dse.evaluate import converter_rate, sram_mib
from complementary_arrays.dse.organisations import FAMILIES, ORGS, expand
from complementary_arrays.dse.tune import policies, tune_space


def test_streams_per_array():
    het22 = ORGS["het22"]
    assert expand(het22, [[4, 256, 128], [16, 256, 256]]) == [
        [4, 256, 128],
        [4, 256, 128],
        [16, 256, 256],
        [16, 256, 256],
    ]
    assert expand(ORGS["single4"], [[8, 176, 512]]) == [[8, 176, 512]] * 4
    assert expand(ORGS["hom11"], [[8, 192, 256], [8, 192, 256]]) == [[8, 192, 256], [8, 192, 256]]
    assert [ORGS[o].shared_array for o in FAMILIES[2] + FAMILIES[4]] == [True, False, False, True, True, True]


def test_policies_and_tuning_space():
    assert policies(ORGS["single2"]) == [(2, None), (13, None)]
    assert policies(ORGS["het11"]) == [(2, None), (13, None), (14, 0), (14, 1), (14, 2)]
    assert policies(ORGS["single4"]) == [(13, None), (14, None)]
    assert policies(ORGS["hom22"]) == [(13, None), (14, 0), (14, 1), (14, 2)]
    assert [name for name, _ in tune_space(ORGS["het22"])] == [
        "share",
        "share2",
        "lookahead",
        "slots",
        "stage_slots",
        "vslots",
    ]
    assert "share2" not in dict(tune_space(ORGS["het11"]))


def test_converter_and_sram():
    assert [converter_rate(f, 8000) for f in ("bf16", "w8g128", "w4g32")] == [4096, 8192, 16384]
    assert sram_mib(98304) == sram_mib(196608) == 16 and sram_mib(524288) == 64
