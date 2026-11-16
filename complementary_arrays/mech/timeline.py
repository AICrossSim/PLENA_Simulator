"""Event timeline of one MoE layer window on the chips of one set: per expert task its dispatch, HBM, port and compute
intervals, and the exact per-cycle run lengths of the array states, the HBM users and the weight-port users.

Window: the eval snapshot of the batch whose MoE step time on the pair is the lower median of the snapshots, and the
layer of that snapshot whose time on the pair is the lower median (from the evaluation files of the chip set)."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

from ..dse import results as rs
from ..dse import snapshots as ss
from ..dse.evaluate import chip_setup
from ..dse.formats import FORMATS, ROUTES
from ..dse.organisations import FAMILY, ORGS
from ..sim import workload as wl
from ..sim.engine import ARRAY_STATES, core_map, counters, trace

MERGE_GAP = 32
STATES = {
    0: "idle",
    1: "compute",
    2: "wait_hbm",
    3: "wait_port",
    4: "wait_port",
    5: "wait_other",
    6: "wait_other",
    7: "none",
}
TASK_STATES = ((1, "compute"), (2, "wait_hbm"), (3, "wait_port_reread"), (4, "wait_port_first"), (6, "wait_other"))
FIELDS = dict(
    array_states="per task stream, [state, t0, t1] exact per-cycle runs; compute = the array advanced its phase this cycle "
    "(pipeline fill and drain included); wait_hbm = first-pass weights not yet in SRAM; wait_port = weights in SRAM but not yet "
    "delivered by the shared port (first pass or re-read); wait_other = converter, pipeline latency, activations, output, merge "
    "and commit; idle = no task",
    core_states="per physical array, [state, t0, t1] runs of the engine's array states",
    hbm_runs="[users, t0, t1, bytes]; users = arrays receiving HBM bytes this cycle (cur = the phase being computed, next = prefetch)",
    port_runs="[users, t0, t1, first_pass_bytes, reread_bytes]; first = first pass (stored bytes), reread = re-read from SRAM "
    "(2 B per BF16 weight, stored bytes for packed re-reads)",
    hbm_bytes_attribution="HBM bytes of the layer by the task they land for (current task or prefetched next task)",
    experts="intervals merged across gaps of <= merge_gap_cycles; exact counts in compute_cycles, reread_weights and hbm_bytes; "
    "passes = ceil(tokens / array height); passes_measured = 1 + re-delivered weights / first-pass weights",
)


def merge(cycles, gap: int) -> list[list[int]]:
    """Sorted cycles -> [[t0, t1], ...], joining runs separated by at most gap missing cycles."""
    out = []
    for t in cycles:
        if out and t <= out[-1][1] + 1 + gap:
            out[-1][1] = t
        else:
            out.append([t, t])
    return out


def runs(label, cycles) -> list[list]:
    """Per-cycle labels -> [[label, t0, t1], ...]."""
    out = []
    for t, x in zip(cycles, label):
        if out and out[-1][0] == x and t == out[-1][2] + 1:
            out[-1][2] = t
        else:
            out.append([x, t, t])
    return out


def labels(org: str, shapes, cmap) -> list[str]:
    if ORGS[org].kind == "het":
        tallest = max(int(s[0]) for s in shapes)
        return ["tall" if int(s[0]) == tallest else "short" for s in shapes]
    if ORGS[org].kind == "single":
        return [f"A0/s{c}" for c in range(len(shapes))]
    return [f"A{cmap[c]}" for c in range(len(shapes))]


def pick(eval_dir: Path, batch: int, set_name: str) -> dict:
    """The median window of the pair (lower medians) and the step and layer times of every chip there."""
    per = {}
    for o in FAMILY:
        with np.load(Path(eval_dir) / f"{set_name}_{o}.npz") as z:
            per[o] = (z[f"B{batch}_cycles"], z[f"B{batch}_unit"], z[f"B{batch}_layer"])
    cyc, unit, layer = per["het11"]
    S = int(unit.max()) + 1
    step = np.bincount(unit, weights=cyc, minlength=S)
    snap = int(np.argsort(step, kind="stable")[(S - 1) // 2])
    ix = np.flatnonzero(unit == snap)
    lay = int(layer[ix][np.argsort(cyc[ix], kind="stable")[(len(ix) - 1) // 2]])
    return dict(
        batch=batch,
        snapshot=snap,
        layer=lay,
        step_cycles={o: float(per[o][0][per[o][1] == snap].sum()) for o in per},
        layer_cycles={o: float(per[o][0][(per[o][1] == snap) & (per[o][2] == lay)].sum()) for o in per},
        step_cycles_quantiles_het11={q: float(np.percentile(step, q)) for q in (5, 25, 50, 75, 95)},
    )


def timeline(root: Path, model: str, fmt: str, org: str, chip: dict, batch: int, snapshot: int, layer: int) -> dict:
    shapes, env = chip_setup(chip, org)
    cmap = core_map(env, len(shapes))
    lab = labels(org, shapes, cmap)
    rows = ss.windows(root, model, ROUTES[fmt], batch, "eval")
    idx = [i for i, r in enumerate(rows) if r["snapshot"] == snapshot and r["layer"] == layer]
    if len(idx) != 1:
        raise ValueError(f"window not unique: {idx}")
    win, E = rows[idx[0]], wl.MODELS[model]["experts"]
    tokens = np.bincount(np.asarray(win["ids"]).ravel(), minlength=E)
    o, ledger, decisions = trace(np.asarray(win["ids"]), shapes, wl.MODELS[model], env)
    c = counters(o)
    cyc, K = ledger["cycle"], min(len(shapes), 2)
    state, task, nxt = (ledger[f][:, :K].T for f in ("state", "task", "prefetch_task"))
    hcur, hnext, port_bytes, bf16_w, packed = (
        ledger[f][:, :K].T
        for f in ("hbm_current", "hbm_prefetch", "port_first_bytes", "port_reread_weights", "port_packed_bytes")
    )
    stored = FORMATS[fmt]["bytes_per_weight"]
    first_bytes = port_bytes - packed  # the port beats of packed re-reads are counted with the first passes
    reread_w = bf16_w + packed / stored
    tasks = {}
    plan = [
        dict(cycle=int(t), ctx=int(k), field_a=int(a), field_b=int(b), planned_makespan=int(pred))
        for t, k, a, b, _, kind, _, pred in decisions
        if kind == 5
    ]
    for t, k, tk, expert, M, kind, _, _ in decisions:
        if kind not in (0, 1, 2, 3, 4):
            continue
        r = tasks.setdefault(tk, dict(task=int(tk), expert=int(expert), tokens=int(M)))
        if kind == 1 and "queued_as_next" not in r:
            r["queued_as_next"] = int(t)
        if kind in (0, 2, 3) and "start" not in r:
            r["start"], r["ctx"] = int(t), int(k)
        if kind == 4:
            r["done"] = int(t)
    experts = []
    for tk, r in tasks.items():
        k = r.get("ctx")
        if k is None or k >= K:
            continue
        hmask, hbytes = np.zeros(len(cyc), bool), 0
        for q in range(K):
            a, b = (task[q] == tk) & (hcur[q] > 0), (nxt[q] == tk) & (hnext[q] > 0)
            hmask |= a | b
            hbytes += int(hcur[q][a].sum() + hnext[q][b].sum())
        own = task[k] == tk
        comp, rer = cyc[own & (state[k] == 1)], cyc[own & (reread_w[k] > 0)]
        fpass, rwt = int(first_bytes[k][own].sum()), float(reread_w[k][own].sum())
        hc = cyc[hmask]
        experts.append(
            dict(
                kind="router" if r["expert"] < 0 else "shared" if r["expert"] >= E else "routed",
                id="router" if r["expert"] < 0 else "shared" if r["expert"] >= E else int(r["expert"]),
                task=r["task"],
                tokens=r["tokens"],
                array=lab[k],
                ctx=k,
                passes=math.ceil(r["tokens"] / int(shapes[k][0])) if r["expert"] >= 0 else None,
                passes_measured=(1 + rwt / (fpass / stored)) if fpass > 0 else None,
                queued_as_next=r.get("queued_as_next"),
                start=r.get("start"),
                done=r.get("done"),
                hbm_first=[int(hc.min()), int(hc.max())] if len(hc) else None,
                hbm_bytes=hbytes,
                hbm_runs=merge(hc.tolist(), MERGE_GAP),
                reread=merge(rer.tolist(), MERGE_GAP),
                reread_weights=rwt,
                port_first_bytes=fpass,
                compute=merge(comp.tolist(), MERGE_GAP),
                compute_cycles=len(comp),
                state_cycles_while_current={name: int((own & (state[k] == s)).sum()) for s, name in TASK_STATES},
            )
        )
    routed = sorted([x for x in experts if isinstance(x["id"], int)], key=lambda x: (-x["tokens"], x["id"]))
    for i, x in enumerate(routed):
        x["rank_hot_to_cold"] = i
    t = cyc.tolist()
    hbm_users = [
        "+".join(
            [f"{lab[q]}:cur" for q in range(K) if hcur[q][i] > 0]
            + [f"{lab[q]}:next" for q in range(K) if hnext[q][i] > 0]
        )
        or "idle"
        for i in range(len(cyc))
    ]
    port_users = [
        "+".join(
            [f"first:{lab[q]}" for q in range(K) if first_bytes[q][i] > 0]
            + [f"reread:{lab[q]}" for q in range(K) if reread_w[q][i] > 0]
        )
        or "idle"
        for i in range(len(cyc))
    ]
    hbm_runs, port_runs = runs(hbm_users, t), runs(port_users, t)
    hbm_total, port_first, port_reread = (hcur + hnext).sum(0), first_bytes.sum(0), (2 * bf16_w + packed).sum(0)
    attribution = dict(total=int(hbm_total.sum()))
    for kind in ("routed", "shared", "router"):
        attribution[kind] = sum(x["hbm_bytes"] for x in experts if x["kind"] == kind)
    attribution["not_attributed"] = attribution["total"] - attribution["routed"] - attribution["shared"] - attribution["router"]
    pos = {x: i for i, x in enumerate(t)}
    for r in hbm_runs:
        r.append(int(hbm_total[pos[r[1]] : pos[r[2]] + 1].sum()))
    for r in port_runs:
        r += [int(port_first[pos[r[1]] : pos[r[2]] + 1].sum()), int(port_reread[pos[r[1]] : pos[r[2]] + 1].sum())]
    return dict(
        chip=org,
        base=chip["base"],
        stream_shapes=[list(map(int, s)) for s in shapes],
        labels=lab,
        policy=chip["policy"],
        window=dict(
            batch=batch,
            snapshot=snapshot,
            layer=layer,
            live=int(win["live"]),
            routed_experts_active=int((tokens > 0).sum()),
            token_assignments=int(tokens.sum()),
        ),
        cycles=int(c["cycles"]),
        startup_cycles=int(c["startup"]),
        policy14_plan_records=plan,
        ledger_first_cycle=int(cyc[0]),
        ledger_last_cycle=int(cyc[-1]),
        merge_gap_cycles=MERGE_GAP,
        hbm_peak_B_per_cycle=env.H,
        port_W_B_per_cycle=env.W,
        stored_B_per_weight=stored,
        experts=routed + [x for x in experts if not isinstance(x["id"], int)],
        array_states={lab[k]: runs([STATES[int(s)] for s in state[k]], t) for k in range(K)},
        core_states={f"array{a}": runs([ARRAY_STATES[int(s)] for s in ledger["array_state"][:, a]], t)
                     for a in range(ledger["array_state"].shape[1])},
        hbm_bytes_attribution=attribution,
        redeliv_packed=int(env.redeliv_packed),
        hbm_runs=hbm_runs,
        port_runs=port_runs,
        fields=FIELDS,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt")
    ap.add_argument("P", type=int, help="multiplier budget")
    ap.add_argument("--chips", type=Path, required=True, help="directory with <org>.json chip records")
    ap.add_argument("--version", required=True, help="name of the chip set (its folder under eval/)")
    ap.add_argument("--set", default="s1_eval", help="evaluation set holding the batch (e.g. s1_b128)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--orgs", default=",".join(FAMILY))
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    cell = rs.cell_name(a.model, a.fmt, a.P)
    folder = Path(a.out) / "mech" / "timeline" / a.version
    window = pick(rs.eval_dir(a.out, a.version, cell), a.batch, a.set)
    rs.write_json(folder / f"{cell}_step.json", window)
    for org in a.orgs.split(","):
        chip = rs.read_json(a.chips / f"{org}.json")["best_main"]
        res = timeline(a.data, a.model, a.fmt, org, chip, a.batch, window["snapshot"], window["layer"])
        if res["cycles"] != window["layer_cycles"][org]:
            raise ValueError(f"{org}: traced window takes {res['cycles']} cycles, the evaluation {window['layer_cycles'][org]}")
        rs.write_json(folder / f"{cell}_{org}.json", dict(res, cell=cell, version=a.version), indent=None)
        print(cell, a.version, org, "cycles", res["cycles"], "experts", len(res["experts"]), flush=True)


if __name__ == "__main__":
    main()
