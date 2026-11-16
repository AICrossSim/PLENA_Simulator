"""Where the multiplier-cycles of a chip go, on the eval snapshots of the given batches (window means).

Per array, its n*k multipliers times the layer cycles split into: useful (MACs of routed experts, shared expert and
router), empty_rows (row slots without a token and partial-tile padding), wait_first_pass (weights not yet landed from
HBM, port, converter or in flight), wait_redelivery (re-reads of weights already on chip), vector_control_drain (vector
phases, control, drain and tail hand-off, activation waits, output and commit stalls, startup, pipeline fill) and
no_work (idle before the last task and at the tail). The chip fractions weigh the arrays by their multipliers.

Per array the engine counts the cycles of each array state (array_state_cycles, ARRAY_STATES), the cycles with MAC
inputs, empty row slots, passes and tokens; idle_while_other_works counts the cycles with one array idle while the other
works.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from ..dse import results as rs
from ..dse import snapshots as ss
from ..dse.evaluate import chip_setup
from ..dse.formats import FORMATS, ROUTES
from ..dse.organisations import FAMILY
from ..sim import workload as wl
from ..sim.engine import ARRAY_STATES, CapacityError, core_map, counters, simulate

ARRAY_COUNTERS = (
    "mac_inputs",
    "empty_row_slots",
    "matrix_phase_cycles",
    "passes",
    "pass_tokens",
    "routed_passes",
    "routed_tokens",
    "routed_tasks",
)
CATEGORIES = ("useful", "empty_rows", "wait_first_pass", "wait_redelivery", "vector_control_drain", "no_work")


def _columns(c: dict, K: int, n: int) -> dict:
    cols = dict(
        T=c["cycles"],
        hbm_service=c["hbm_service"],
        delivered=c["ctx_delivered_weights"].sum(-1),
        port_used=c["port_bytes"],
        port_saturated=c["port_saturated_cycles"],
        idle_other=c["idle_while_other_works"],
        merge=c["ctx_merge_stall"].sum(-1),
        share=c["ctx_share_stall"].sum(-1),
        output=c["ctx_output_stall"].sum(-1),
    )
    for name in ("wport_bytes", "hbm_bytes", "local_read_bytes", "decoded", "reread_bytes"):
        a = np.asarray(c[name])[:, :K]
        cols[name] = a.reshape(n, -1).sum(1)
        for k in range(K):
            cols[f"{name}{k}"] = a[:, k].sum(-1)
    states = np.asarray(c["array_state_cycles"])[:, :K]
    for i, f in enumerate(ARRAY_STATES):
        cols[f] = states[:, :, i]
    for f in ARRAY_COUNTERS:
        cols[f] = np.asarray(c["array_" + f])[:, :K]
    cols["useful_macs"] = np.asarray(c["useful_macs"])[:, :K].sum(-1)
    return cols


def account(root: Path, model: str, fmt: str, org: str, chip: dict, batches, threads: int = 3, eval_dir=None) -> dict:
    shapes, env = chip_setup(chip, org)
    cmap = core_map(env, len(shapes))
    K = max(cmap) + 1
    arrays = [shapes[cmap.index(k)] for k in range(K)]
    bpw, route, res = FORMATS[fmt]["bytes_per_weight"], ROUTES[fmt], {}
    for B in batches:
        rows = ss.windows(root, model, route, B, "eval")
        try:
            out = simulate(np.stack([r["ids"] for r in rows]), shapes, wl.MODELS[model], env, threads=threads)
        except (CapacityError, RuntimeError, ValueError) as ex:
            res[f"B{B}"] = dict(infeasible=True, why=str(ex)[:200])
            continue
        cols = _columns(counters(out), K, len(rows))
        w = wl.window_weights(rows)
        W = float(w.sum())

        def wm(x):
            return wl.dot(w, x) / W

        T = cols["T"]
        wT = wl.dot(w, T)
        Tm = wT / W
        chip_cat, macs_total, per_array = dict.fromkeys(CATEGORIES, 0.0), 0, []
        for k, (m, n, kk) in enumerate(arrays):
            g = {f: wm(cols[f][:, k]) for f in ARRAY_STATES + ARRAY_COUNTERS + ("useful_macs",)}
            macs = n * kk
            macs_total += macs
            pad = macs * g["mac_inputs"] - g["useful_macs"] - g["empty_row_slots"]
            cat = dict(
                useful=g["useful_macs"],
                empty_rows=g["empty_row_slots"] + pad,
                wait_first_pass=macs * (g["wait_port"] + g["wait_hbm"] + g["wait_decode"] + g["wait_latency"]),
                wait_redelivery=macs * g["wait_redelivery"],
                vector_control_drain=macs * (g["vector"] + g["control"] + g["drain"] + g["wait_x"] + g["stall"]
                                             + g["tail"] + g["startup"] + g["matrix"] - g["mac_inputs"]),
                no_work=macs * (g["idle"] + g["idle_end"]),
            )
            total = macs * Tm
            if abs(sum(cat.values()) - total) > 1e-6 * total:
                raise ValueError(f"accounting of array {k} at B{B} does not close")
            for c in CATEGORIES:
                chip_cat[c] += cat[c]
            share = {s: macs * g[s] / total for s in ARRAY_STATES}
            per_array.append(dict(
                array=k,
                shape=[int(m), int(n), int(kk)],
                macs=macs,
                fractions={c: cat[c] / total for c in CATEGORIES},
                detail=dict(
                    empty_rows_without_token=g["empty_row_slots"] / total,
                    partial_tile_padding=pad / total,
                    wait_first_port=share["wait_port"],
                    wait_first_hbm_landing=share["wait_hbm"],
                    wait_converter=share["wait_decode"],
                    wait_inflight_latency=share["wait_latency"],
                    idle_before_last_task=share["idle"],
                    idle_tail=share["idle_end"],
                    vector=share["vector"],
                    control=share["control"],
                    systolic_drain_or_tail_handoff=share["drain"],
                    matrix_fill_drain_reload=macs * (g["matrix"] - g["mac_inputs"]) / total,
                    wait_activation=share["wait_x"],
                    stall_output_lock_commit=share["stall"],
                    tail_slot_only=share["tail"],
                    startup=share["startup"],
                ),
                state_cycles={s: g[s] for s in ARRAY_STATES},
                passes_per_routed_expert=g["routed_passes"] / g["routed_tasks"] if g["routed_tasks"] else None,
                routed_tasks=g["routed_tasks"],
                row_occupancy_routed=g["routed_tokens"] / (m * g["routed_passes"]) if g["routed_passes"] else None,
                weights_per_cycle=dict(
                    hbm_first_pass=wm(cols[f"hbm_bytes{k}"]) / bpw / Tm,
                    delivered_to_array=wm(cols[f"local_read_bytes{k}"]) / 2 / Tm,
                    converter=wm(cols[f"decoded{k}"]) / Tm,
                ),
            ))
        d = dict(
            layer_cycles=Tm,
            windows=len(rows),
            chip_fractions={c: chip_cat[c] / (macs_total * Tm) for c in CATEGORIES},
            arrays=per_array,
            hbm_busy_frac=wl.dot(w, cols["hbm_service"]) / (wT * env.H),
            port=dict(
                W=env.W,
                first_pass_bytes_frac=wl.dot(w, cols["wport_bytes"]) / (wT * env.W),
                redelivery_bytes_frac=2 * wl.dot(w, cols["delivered"]) / (wT * env.W),
                used_frac=wl.dot(w, cols["port_used"]) / (wT * env.W),
                saturated_cycles_frac=wl.dot(w, cols["port_saturated"]) / wT,
            ),
            idle_while_other_works=float(cols["idle_other"].sum()) / float(T.sum()),
            weights_per_cycle=dict(
                hbm_first_pass=wm(cols["hbm_bytes"]) / bpw / Tm,
                delivered_to_arrays=wm(cols["local_read_bytes"]) / 2 / Tm,
                converter=wm(cols["decoded"]) / Tm,
            ),
            limits=dict(hbm_weights_per_cycle=env.H / bpw, port_stored_weights_per_cycle=env.W / bpw,
                        port_bf16_weights_per_cycle=env.W / 2),
            redeliv_packed=int(env.redeliv_packed),
        )
        if eval_dir is not None:
            for name in (f"s1_b{B}_{org}.npz", f"s1_eval_{org}.npz"):
                p = Path(eval_dir) / name
                if p.exists():
                    with np.load(p) as z:
                        if f"B{B}_cycles" in z.files:
                            d.update(cycles_equal_eval=bool(np.array_equal(T, z[f"B{B}_cycles"])), eval_file=name)
                            break
        res[f"B{B}"] = d
    return dict(arrays=[list(map(int, a)) for a in arrays], batches=res)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=sorted(wl.MODELS))
    ap.add_argument("fmt")
    ap.add_argument("P", type=int, help="multiplier budget")
    ap.add_argument("--chips", type=Path, required=True, help="directory with <org>.json chip records")
    ap.add_argument("--version", required=True, help="name of the chip set (also its folder under eval/)")
    ap.add_argument("--orgs", default=",".join(FAMILY))
    ap.add_argument("--batches", type=int, nargs="+", default=list(wl.MAIN_BATCHES))
    ap.add_argument("--data", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_DATA"), help="data root")
    ap.add_argument("--out", type=Path, default=os.environ.get("COMPLEMENTARY_ARRAYS_OUT"), help="output root")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args(argv)
    if a.data is None or a.out is None:
        ap.error("--data and --out (or COMPLEMENTARY_ARRAYS_DATA / COMPLEMENTARY_ARRAYS_OUT) are required")
    cell = rs.cell_name(a.model, a.fmt, a.P)
    for org in a.orgs.split(","):
        chip = rs.read_json(a.chips / f"{org}.json")["best_main"]
        res = account(a.data, a.model, a.fmt, org, chip, a.batches, a.threads, rs.eval_dir(a.out, a.version, cell))
        res.update(cell=cell, org=org, version=a.version, base=chip["base"], policy=chip["policy"], env=chip["env"])
        rs.write_json(Path(a.out) / "mech" / a.version / f"{cell}_{org}.json", res)
        print(cell, a.version, org, {k: round(v["layer_cycles"], 1) for k, v in res["batches"].items() if "layer_cycles" in v},
              flush=True)


if __name__ == "__main__":
    main()
