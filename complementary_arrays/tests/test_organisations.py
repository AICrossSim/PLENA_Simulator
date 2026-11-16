"""Organisations, weight formats, dispatch policies and the tuning space."""

from complementary_arrays.dse.evaluate import sram_mib
from complementary_arrays.dse.formats import FORMATS, converter_rate, kappa
from complementary_arrays.dse.organisations import FAMILY, ORGS, expand
from complementary_arrays.dse.tune import PACKED, policies, tune_space


def test_streams_per_array():
    assert expand(ORGS["single2"], [[4, 384, 256]]) == [[4, 384, 256]] * 2
    assert expand(ORGS["het11"], [[2, 256, 128], [8, 256, 256]]) == [[2, 256, 128], [8, 256, 256]]
    assert [ORGS[o].shared_array for o in FAMILY] == [True, False, False]


def test_policies_and_tuning_space():
    assert policies(ORGS["single2"]) == [(2, None), (13, None), (15, None), (16, None)]
    assert policies(ORGS["het11"]) == [(2, None), (13, None), (14, 0), (14, 1), (14, 2), (15, None), (16, 2)]
    assert [name for name, _ in tune_space(PACKED)] == ["share", "lookahead", "slots", "stage_slots", "vslots", "redeliv_packed"]


def test_formats():
    assert [converter_rate(f, 8000) for f in ("bf16", "w8g128", "w4g32", "w4g32p", "fp8b128")] == [4096, 8192, 16384, 16384, 8192]
    assert FORMATS["w4g32p"]["bytes_per_weight"] == 0.5625 and FORMATS["w4g32p"]["group"] == 64
    assert kappa("w8g128") == 0.54 and abs(kappa("fp8b128") - 4192 / 8000) < 1e-12
    assert sram_mib(98304) == sram_mib(196608) == 16 and sram_mib(524288) == 64
