"""Promotion of chips short of the multiplier budget."""

from complementary_arrays.dse.candidates import base_macs, promote

P = 98304


def test_promote_to_the_whole_budget():
    assert promote("single", [[4, 352, 256]], P) == [[[4, 384, 256]], [[4, 512, 192]]]
    assert promote("single", [[8, 272, 320]], P) == [[[8, 256, 384]], [[8, 384, 256]]]
    assert promote("hom", [[2, 176, 256], [2, 176, 256]], P) == [[[2, 192, 256], [2, 192, 256]], [[2, 256, 192], [2, 256, 192]]]
    assert promote("het", [[4, 352, 128], [8, 160, 256]], P) == [
        [[4, 448, 128], [8, 160, 256]],
        [[4, 256, 224], [8, 160, 256]],
        [[4, 352, 128], [8, 208, 256]],
        [[4, 352, 128], [8, 128, 416]],
    ]
    assert promote("single", [[16, 352, 512]], 196608) == [[[16, 384, 512]], [[16, 512, 384]]]


def test_full_budget_chips_stay():
    for kind, base in (("single", [[4, 384, 256]]), ("hom", [[4, 256, 192]] * 2), ("het", [[2, 256, 128], [8, 256, 256]])):
        assert base_macs(base, kind) == P and promote(kind, base, P) == []
    assert promote("single", [[6, 240, 768]], 196608) == []
