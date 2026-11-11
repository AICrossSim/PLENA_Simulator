"""The simulator reproduces outputs of the reference engine bit for bit on small windows (data/sim_reference.npz)."""

import json
from pathlib import Path

import numpy as np
import pytest

from complementary_arrays.sim.engine import Environment, counters, drain_latency, program_metrics, simulate, trace

DATA = np.load(Path(__file__).with_name("data") / "sim_reference.npz")
MANIFEST = json.loads(str(DATA["manifest"]))
CASES = [m for m in MANIFEST if isinstance(m["case"], int)]
TRACE = next(m for m in MANIFEST if m["case"] == "trace")
METRICS = next(m for m in MANIFEST if m["case"] == "metrics")["tasks"]
PROVISION = next(m for m in MANIFEST if m["case"] == "provision")


def _label(m):
    env = m["env"]
    return f"{m['cell']}-{m['org']}-B{m['batch']}-systolic{env.get('systolic', 0)}-{env.get('drain_form', 'linear')}-port{env['port_mode']}"


@pytest.mark.parametrize("case", CASES, ids=[_label(m) for m in CASES])
def test_counters(case):
    i, nc = case["case"], len(case["shapes"])
    got = counters(
        simulate(DATA[f"{i}/routes"], case["shapes"], case["workload"], Environment(**case["env"]), threads=1)
    )
    flat, o = DATA[f"{i}/expected"], 0
    for name, shape in case["fields"]:
        n = int(np.prod(shape))
        value = got[name][:, :nc] if name.startswith("ctx_") else got[name]
        assert np.array_equal(value, flat[o : o + n].reshape(shape)), name
        o += n


def test_trace():
    _, rec, decisions = trace(DATA["trace/ids"], TRACE["shapes"], TRACE["workload"], Environment(**TRACE["env"]))
    names = dict(
        state="state",
        task="task",
        ntask="prefetch_task",
        hcur_B="hbm_current",
        hnext_B="hbm_prefetch",
        w_B="port_first_bytes",
        redeliv_w="port_reread_weights",
    )
    ledger = DATA["trace/ledger"]
    assert len(rec["cycle"]) == len(ledger)
    for j, col in enumerate(TRACE["ledger_columns"]):
        if col == "cycle":
            value = rec["cycle"]
        elif col == "port_used_B":
            value = rec["port_used"]
        else:
            value = rec[names[col[3:]]][:, int(col[1])]
        assert np.array_equal(value, ledger[:, j]), col
    assert np.array_equal(decisions, DATA["trace/decisions"])


@pytest.mark.parametrize("task", METRICS)
def test_program_metrics(task):
    env = Environment(**task["env"])
    got = program_metrics(
        task["shape"], task["tokens"], task["shared"], task["workload"], env, task["capacity"], task["lanes"]
    )
    assert got == task["expected"]


def test_provision_tokens():
    """Tail mode: a window simulated alone, and provisioned for its whole window group as in the stored full-group run."""
    case, ids = PROVISION, DATA["provision/ids"][None]
    env = Environment(**case["env"])
    alone = counters(simulate(ids, case["shapes"], case["workload"], env, threads=1))
    group = counters(simulate(ids, case["shapes"], case["workload"], env, threads=1, provision_tokens=case["tokens"]))
    assert alone["cycles"][0] == case["cycles_alone"]
    assert group["cycles"][0] == case["cycles_group"]


def test_drain_latency():
    env = Environment(systolic=3, drain_form="tree", drain_segment=16)
    assert [drain_latency(n, env) for n in (16, 17, 64, 256, 512)] == [15, 16, 17, 19, 20]
    assert drain_latency(256, Environment(drain_form="express", drain_segment=16)) == 31
    assert drain_latency(256, Environment()) == 255
