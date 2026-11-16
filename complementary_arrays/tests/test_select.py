"""Final selection: fastest valid chip on the select windows, earlier candidate on ties, equal-multiplier rule."""

import json

import pytest

from complementary_arrays.dse.select import select


def _chip(base, select_main, valid=True):
    return dict(base=base, shapes=base, policy=[13, None], env={}, select={}, select_busy={}, valid=valid,
                select_main=select_main)


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return path


def test_fastest_valid_and_ties(tmp_path):
    pair = [[2, 256, 128], [8, 256, 256]]
    a = _write(tmp_path / "a.json", dict(best_main=_chip(pair, 100.0)))
    b = _write(tmp_path / "b.json", dict(best_main=_chip(pair, 90.0, valid=False)))
    c = _write(tmp_path / "c.json", dict(best_main=_chip(pair, 100.0)))
    res = select("het11", [("a", a), ("b", b)], as_is=[("c", c)])
    assert res["selected_from"] == "a" and res["candidates_valid"] == dict(a=True, b=False, c=True)
    none = _write(tmp_path / "none.json", dict(best_main=None))
    assert select("het11", [("none", none), ("c", c)])["selected_from"] == "c"


def test_budget_rule(tmp_path):
    short = _write(tmp_path / "short.json", dict(best_main=_chip([[4, 352, 256]], 50.0)))
    full = _write(tmp_path / "full.json", dict(best_main=_chip([[4, 384, 256]], 60.0)))
    assert select("single2", [("full", full)], as_is=[("short", short)], budget=98304)["selected_from"] == "full"
    with pytest.raises(ValueError):
        select("single2", [("short", short)], budget=98304)
