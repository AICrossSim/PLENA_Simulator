"""Chip area and energy per MoE layer from the block model (model.py) and the simulator's activity counters, at 1 GHz.

A chip is a list of physical (m, n, k) arrays with its dispatch policy, converter rate (dequantized weights per cycle)
and slot count. The activity counters of one batch are window-weighted means of the simulator's extended outputs (layer
cycles, HBM and SRAM bytes, port bytes, decoded weights, and per array its shape, matrix-phase and vector cycles, useful
MACs, vector lanes; per-class lists [routed, shared, router] per array), as mech/ records them.

usage: python -m complementary_arrays.ppa.chip M,N,K [M,N,K ...] [--format w4g32p] [--slots 32] [--converter 16384]
"""

import argparse
import json

from . import model as pm

LEVELS = ("low", "mid", "high")
CLOCK_HZ = 1.0e9


def physical_arrays(org, base):
    """Physical arrays of an organisation from its base shapes: one array for single2, two for hom11 and het11."""
    base = [tuple(s) for s in base]
    if org == "single2":
        return base[:1]
    return base * 2 if len(base) == 1 else base


def chip_area(arrays, fmt, policy=16, converter_rate=16384, slots=32):
    """um^2 per block, logic total and chip totals with the 16 MiB SRAM and the streaming buffers at SRAM density."""
    each = [pm.array_area(*a) for a in arrays]
    d = dict(
        array=sum(each),
        array_each=each,
        vector=pm.VEC_LANES * pm.VEC_LANE,
        decoder=converter_rate / 64 * pm.dec_area(fmt),
        dataplane=pm.dataplane_area(arrays),
        wport=pm.WPORT,
        control=pm.CTRL,
        dispatch=pm.DISP[pm.policy_dispatcher(policy)],
    )
    d["logic"] = sum(d[k] for k in ("array", "vector", "decoder", "dataplane", "wport", "control", "dispatch"))
    d["buffer_bytes"] = sum(pm.stream_buffer_bytes(m, n, k, slots) for (m, n, k) in arrays)
    for lv, sram in pm.SRAM.items():
        d[f"buffer_{lv}"] = d["buffer_bytes"] / 2**20 * pm.SRAM_MM2_PER_MIB[lv] * 1e6
        d[f"total_{lv}"] = d["logic"] + sram + pm.PORT_REP + d[f"buffer_{lv}"]
    d["mults"] = sum(n * k for (m, n, k) in arrays)
    return d


def efficiency(arrays, counts):
    """Peak TOPS and attained TFLOPS (two operations per useful MAC) per mm^2 of array, and multiplier utilization."""
    area_mm2 = sum(pm.array_area(*a) for a in arrays) / 1e6
    mults = sum(n * k for (m, n, k) in arrays)
    macs = sum(sum(x) for x in counts["cls_useful_macs"])
    return dict(
        array_area_mm2=area_mm2,
        peak_tops_per_mm2=2 * mults * CLOCK_HZ / 1e12 / area_mm2,
        attained_tflops_per_mm2=2 * macs / (counts["cycles"] / CLOCK_HZ) / 1e12 / area_mm2,
        mac_utilization=macs / (mults * counts["cycles"]),
    )


def energy_layer(b, arrays, fmt, area, writeback=None, batch=None, packed=False):
    """pJ per MoE layer by component for one batch record of activity counters `b`. `writeback` holds the write-back
    ratios of the batch (writeback.py), used when re-delivered BF16 blocks are written back to the main SRAM; with
    `packed` re-deliveries stay packed and nothing is written back."""
    T = b["cycles"]
    e = {}
    eb = eb_rowgate = eidle = eper = evec = 0.0
    shapes = [tuple(a["shape"]) for a in b["arrays"]]
    phys = arrays if len(shapes) == len(arrays) else arrays[: len(shapes)]
    for a, (m, n, k) in zip(b["arrays"], phys, strict=True):
        assert tuple(a["shape"]) == (m, n, k), (a["shape"], (m, n, k))
        q = k // m
        cells = m * n
        busy = a["matrix_phase_cycles"]
        pb = pm.dpu_busy_pj(q, fmt)[0]
        pz = pb * pm.zeroact_factor(q)[0]  # a row without a token computes on zeros
        pi = pm.dpu_idle_pj(q)[0]
        tok_cells = a["useful_macs"] / q  # cell-cycles with a real token
        empty = max(0.0, cells * busy - tok_cells)
        eb += tok_cells * pb + empty * pz  # one valid bit per array
        eb_rowgate += tok_cells * pb + empty * pi  # with per-row valid gating
        eidle += cells * (T - busy) * pi
        eper += (pm.array_area(m, n, k) - cells * pm.DPU_AREA[q]) * pm.periph_density(fmt, q) * busy
        evec += a["lanes"] * a["st_vector"] * pm.VEC_LANE * pm.DPU_DENSITY
    e.update(array_busy=eb, array_busy_rowgate=eb_rowgate, array_idle=eidle, array_periph=eper, vector=evec)
    e["decoder"] = sum(sum(x) for x in b["cls_decoded"]) * pm.DEC_PJ_PER_WEIGHT[fmt]

    # Main SRAM: HBM landing writes, weight-port reads of packed records, write-back of the BF16 blocks of re-read
    # experts (once per re-read expert), re-delivered reads, activations and outputs. First-pass decoded weights go
    # through the per-array streaming buffer instead (written and read once).
    def total(name):
        return sum(sum(x) for x in b["cls_" + name])

    reread = total("reread_bytes")
    wback = 0.0
    if not packed:
        for i, a in enumerate(b["arrays"]):
            m = a["shape"][0]
            r = writeback[f"B{batch}"].get(str(m), {}).get("ratio") if writeback else None
            wback += b["cls_reread_bytes"][i][0] * (r if r is not None else 1.0)  # routed experts
            passes = -(-batch // m) - 1  # shared expert: all tokens, ceil(B / m) passes
            wback += b["cls_reread_bytes"][i][1] / passes if passes > 0 else 0.0
    main_bytes = total("hbm_bytes") + total("wport_bytes") + wback + reread + b["sram_X"] + b["sram_Y"]
    buf_bytes = 2.0 * total("decoded")
    t_layer = T / CLOCK_HZ
    capacity = (16 * 2**20 + area["buffer_bytes"]) / (16 * 2**20)
    e["sram_leak"] = pm.SRAM_LEAK_W * capacity * t_layer * 1e12
    for lv in LEVELS:
        e[f"sram_{lv}"] = (
            main_bytes * pm.E_SRAM[lv]
            + b["port_bytes"] * pm.E_WIRE[lv]
            + e["sram_leak"]
            + buf_bytes * (2 * pm.E_SRAM[lv] + pm.E_BUF_WIRE[lv])
        )
        e[f"hbm_{lv}"] = b["hbm_bytes"] * 8 * pm.E_HBM_BIT[lv]
    dp_bytes = b["hbm_bytes"] + b["port_bytes"]
    e["dataplane"] = dp_bytes * pm.DPU_DENSITY * area["dataplane"] / (8000 + 16384)
    e["control"] = (area["control"] + area["dispatch"] + area["wport"]) * pm.DPU_DENSITY * T
    on = sum(e[k] for k in ("array_busy", "array_idle", "array_periph", "vector", "decoder", "dataplane", "control"))
    for lv in LEVELS:
        e[f"total_{lv}"] = on + e[f"sram_{lv}"] + e[f"hbm_{lv}"]
    e["total_mid_rowgate"] = on - e["array_busy"] + e["array_busy_rowgate"] + e["sram_mid"] + e["hbm_mid"]
    e["clock_tree"] = pm.CLOCK_TREE_FACTOR * on
    e["total_mid_clock"] = e["total_mid"] + e["clock_tree"]
    e["onchip_logic"] = on
    e["cycles"] = T
    return e


def main():
    p = argparse.ArgumentParser(description="Area of a chip from its array shapes (um^2 per block, mm^2 totals).")
    p.add_argument("arrays", nargs="+", help="physical arrays as m,n,k")
    p.add_argument("--format", default="w4g32p", choices=sorted(pm.DEC))
    p.add_argument("--policy", type=int, default=16)
    p.add_argument("--slots", type=int, default=32)
    p.add_argument("--converter", type=int, default=16384, help="dequantized weights per cycle")
    a = p.parse_args()
    arrays = [tuple(int(v) for v in s.split(",")) for s in a.arrays]
    d = chip_area(arrays, a.format, a.policy, a.converter, a.slots)
    out = {k: (v / 1e6 if k not in ("buffer_bytes", "mults", "array_each") else v) for k, v in d.items()}
    out["array_each"] = [v / 1e6 for v in d["array_each"]]
    print(json.dumps(dict(arrays=arrays, mm2=out), indent=1))


if __name__ == "__main__":
    main()
