"""B200 summary: MoE-layer time per decode step, MoE share of the step and MoE energy per token.

MoE-layer wall time (us per layer and decode step) = (decode-only step of the full model - that of the MoE-zeroed
model) / MoE layers, from stream.py; the router-and-routed part uses the shared-expert-zeroed run, the shared expert is
the rest. The MoE-zeroed step of an engine and model is the median over all its MoE-zeroed runs (BF16). A wall cell is
void when the engine's zeroed step exceeds the other engine's by more than 1.2x (the step became host-bound once the MoE
work was removed, so the difference underestimates the MoE) or when it is 5% below the cell of the next smaller batch
(MoE time does not shrink with the batch); a void cell is replaced by the engine's kernel time (kernel_time.py) when
that is smaller than the other candidates. Per row (weight format and shared-expert precision) the fastest engine is
taken per cell, and per format family the fastest row: the B200 times of the comparison.

usage: python -m complementary_arrays.gpu.bench.report OUT_STEM --stream DIR [--kernels DIR] [--energy DIR]
  Under --stream every stream_<engine>_<model>_<format>_full[_noshared].jsonl must be unique; MoE-zeroed runs may
  repeat. Writes OUT_STEM.json, OUT_STEM.md and, with --energy, OUT_STEM_gpu_energy_<method>.json (lowest MoE energy per
  model, format family and batch; the --gpu input of ppa.energy).
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from .formats import LAYERS

BATCHES = (2, 8, 32, 64, 128)
HOST_BOUND = 1.2
SHRINK = 0.95
PARTS = ("no_shared", "shared", "full")
# row: (shared-expert precision on the B200, candidates of the full layer, candidates without the shared expert)
ROWS = {
    "fp8_w8a8": ("FP8", [("vllm", "fp8_w8a8"), ("sglang", "fp8_w8a8")], None),
    "fp8_w8a16": ("FP8", [("vllm", "fp8_w8a16")], None),
    "nvfp4_w4a16": ("NVFP4", [("vllm", "nvfp4_w4a16")], [("vllm", "nvfp4_w4a16"), ("sglang", "nvfp4_w4a16")]),
    "nvfp4_w4a4": (
        "BF16",
        [("vllm", "nvfp4_w4a4"), ("sglang", "nvfp4_w4a4")],
        [("vllm", "nvfp4_w4a4_sh"), ("sglang", "nvfp4_w4a4")],
    ),
    "nvfp4_w4a4_sh": ("NVFP4", [("vllm", "nvfp4_w4a4_sh")], [("vllm", "nvfp4_w4a4_sh"), ("sglang", "nvfp4_w4a4")]),
}
FAMILIES = {"FP8": ("fp8_w8a8", "fp8_w8a16"), "NVFP4": ("nvfp4_w4a16", "nvfp4_w4a4", "nvfp4_w4a4_sh")}
PPA_MODEL = {"qwen35": "qwen", "deepseek": "deepseek"}
NAME = re.compile(r"^(stream|energy)_(vllm|sglang)_(qwen35|deepseek)_(\w+?)_(full|no_moe)(_noshared)?\.jsonl$")


def rows(path):
    """Records of a JSONL file by batch (a later record of the same batch replaces an earlier one)."""
    return {r["batch"]: r for r in map(json.loads, Path(path).read_text().splitlines()) if r}


def scan(root, prefix):
    """{(engine, model, format, moe, no_shared): [files]} under root."""
    found = {}
    for p in sorted(Path(root).rglob(f"{prefix}_*.jsonl")):
        m = NAME.match(p.name)
        if m and m.group(1) == prefix:
            found.setdefault((m.group(2), m.group(3), m.group(4), m.group(5), bool(m.group(6))), []).append(p)
    return found


class Stream:
    def __init__(self, root):
        self.files = scan(root, "stream")
        self.zeroed = {}
        for (st, kind, mode, moe, ns), ps in self.files.items():
            if moe == "no_moe" and mode == "bf16" and not ns:
                vals = {}
                for p in ps:
                    for b, r in rows(p).items():
                        if r.get("decode_only_ms_mean"):
                            vals.setdefault(b, []).append(r["decode_only_ms_mean"])
                self.zeroed[(st, kind)] = {b: statistics.median(v) for b, v in vals.items()}

    def single(self, st, kind, mode, no_shared=False):
        ps = self.files.get((st, kind, mode, "full", no_shared), [])
        if len(ps) > 1:
            raise ValueError(f"several runs of {st} {kind} {mode} no_shared={no_shared}: {[str(p) for p in ps]}")
        return rows(ps[0]) if ps else {}

    def wall(self, st, kind, mode):
        """{batch: (no_shared, shared, full)} in us per layer; None where the shared-expert-zeroed run is missing."""
        f, n, z = self.single(st, kind, mode), self.single(st, kind, mode, True), self.zeroed.get((st, kind), {})
        t = lambda d, b: d.get(b, {}).get("decode_only_ms_mean")  # noqa: E731
        out = {}
        for b in BATCHES:
            if t(f, b) and b in z:
                full = 1e3 * (t(f, b) - z[b]) / LAYERS[kind]
                if t(n, b):
                    out[b] = (1e3 * (t(n, b) - z[b]) / LAYERS[kind], 1e3 * (t(f, b) - t(n, b)) / LAYERS[kind], full)
                else:
                    out[b] = (None, None, full)
        return out

    def host_bound(self, st, kind, b):
        other = self.zeroed.get(("vllm" if st == "sglang" else "sglang", kind), {}).get(b)
        own = self.zeroed.get((st, kind), {}).get(b)
        return bool(other and own and own > HOST_BOUND * other)


def kernel_times(root, st, kind, mode):
    p = Path(root) / f"profile_{st}_{kind}_{mode}.jsonl" if root else None
    if not p or not p.exists():
        return {}
    return {
        b: (r.get("no_shared_us_per_layer"), r.get("shared_us_per_layer"), r["moe_us_per_layer"])
        for b, r in rows(p).items()
        if r.get("moe_us_per_layer")
    }


def b200_row(stream, kernels, kind, candidates, part):
    """{batch: dict(wall=(us, engine), kernel=(us, engine), void=[...], wall_without_kernel=...)}."""
    out = {b: {} for b in BATCHES}
    for st, mode in candidates:
        k = {b: v[part] for b, v in kernel_times(kernels, st, kind, mode).items() if v[part] is not None}
        w = {b: v[part] for b, v in stream.wall(st, kind, mode).items() if v[part] is not None}
        for i, b in enumerate(BATCHES):
            if b in k and ("kernel" not in out[b] or k[b] < out[b]["kernel"][0]):
                out[b]["kernel"] = (k[b], st)
            if b not in w:
                continue
            prev = BATCHES[i - 1] if i else None
            shrinks = prev in w and w[b] < SHRINK * w[prev]
            if stream.host_bound(st, kind, b) or shrinks:
                out[b].setdefault("void", []).append(f"{st} {mode} wall {w[b]:.1f}")
                if b in k:
                    out[b].setdefault("kernel_for_void", []).append((k[b], f"{st}_kernel"))
                continue
            if "wall" not in out[b] or w[b] < out[b]["wall"][0]:
                out[b]["wall"] = (w[b], st)
    for b in BATCHES:
        subst = out[b].pop("kernel_for_void", [])
        if subst:
            out[b]["wall_without_kernel"] = out[b].get("wall")
            for v in subst:
                if "wall" not in out[b] or v[0] < out[b]["wall"][0]:
                    out[b]["wall"] = v
    return out


def moe_share(stream):
    """1 - MoE-zeroed step / full step for every full run."""
    res = {}
    for (st, kind, mode, moe, ns), _ in stream.files.items():
        if moe != "full" or ns:
            continue
        f, z = stream.single(st, kind, mode), stream.zeroed.get((st, kind), {})
        for b, r in f.items():
            if r.get("decode_only_ms_mean") and b in z:
                res[f"{kind}|{st}|{mode}|B{b}"] = 1 - z[b] / r["decode_only_ms_mean"]
    return res


def moe_energy(root):
    """MoE energy per token (mJ, all MoE layers) = full run - MoE-zeroed run of the same engine and method in the same
    directory."""
    res = {}
    for (st, kind, mode, moe, ns), ps in scan(root, "energy").items():
        if moe != "full" or ns:
            continue
        for p in ps:
            zp = p.parent / f"energy_{st}_{kind}_bf16_no_moe.jsonl"
            if not zp.exists():
                continue
            f, z = rows(p), rows(zp)
            for b, r in f.items():
                e, ez = r.get("energy_per_token_mJ"), z.get(b, {}).get("energy_per_token_mJ")
                if e and ez:
                    method = "closed" if r.get("method", "steady").startswith("closed") else "steady"
                    res[f"{kind}|{st}|{mode}|{method}|B{b}"] = dict(
                        moe_mJ_per_token=e - ez,
                        step_ms=r["decode_only_ms_mean"],
                        power_W=r["decode_power_W"],
                        energy_per_token_mJ=e,
                        file=str(p),
                    )
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--stream", type=Path, required=True)
    p.add_argument("--kernels", type=Path)
    p.add_argument("--energy", type=Path)
    a = p.parse_args()
    stream = Stream(a.stream)
    table, md = {}, []
    for kind in LAYERS:
        md += [
            f"\n### {kind}: MoE layer, us per layer and decode step (wall clock, engine)\n",
            "| row | shared expert | " + " | ".join(f"B{b}" for b in BATCHES) + " |",
            "|---|---|" + "---|" * len(BATCHES),
        ]
        for row, (shared, full_cands, nosh_cands) in ROWS.items():
            for part, cands in ((2, full_cands), (0, nosh_cands or full_cands)):
                res = b200_row(stream, a.kernels, kind, cands, part)
                for b in BATCHES:
                    table[f"{kind}|{row}|{PARTS[part]}|B{b}"] = dict(res[b], shared_expert=shared)
                cells = [f"{res[b]['wall'][0]:.1f} ({res[b]['wall'][1]})" if "wall" in res[b] else "-" for b in BATCHES]
                md.append(f"| {row} | {shared if part == 2 else 'excluded'} | " + " | ".join(cells) + " |")
    fastest = {}
    for kind in LAYERS:
        for fam, members in FAMILIES.items():
            for b in BATCHES:
                walls = [table[f"{kind}|{r}|full|B{b}"].get("wall") for r in members]
                walls = [(v[0], v[1], r) for v, r in zip(walls, members, strict=True) if v]
                if walls:
                    us, st, r = min(walls)
                    fastest[f"{kind}|{fam}|B{b}"] = dict(b200_us=us, engine=st, row=r)
    md += [
        "\n### Fastest B200 time per format family (full MoE layer, us)\n",
        "| model | family | " + " | ".join(f"B{b}" for b in BATCHES) + " |",
        "|---|---|" + "---|" * len(BATCHES),
    ]
    for kind in LAYERS:
        for fam in FAMILIES:
            cells = [f"{fastest[k]['b200_us']:.1f}" if (k := f"{kind}|{fam}|B{b}") in fastest else "-" for b in BATCHES]
            md.append(f"| {kind} | {fam} | " + " | ".join(cells) + " |")
    share = moe_share(stream)
    if share:
        md.append(f"\nMoE share of the decode step: {100 * min(share.values()):.0f}-{100 * max(share.values()):.0f}%")
    energy = moe_energy(a.energy) if a.energy else {}
    lowest = {}  # per method, model, family and batch: the lowest MoE energy over the measured formats and engines
    for key, v in energy.items():
        kind, _, mode, method, b = key.split("|")
        fam = next((f for f, members in FAMILIES.items() if mode in members), None)
        k = (method, kind, fam, int(b[1:]))
        if fam and (k not in lowest or v["moe_mJ_per_token"] < lowest[k]):
            lowest[k] = v["moe_mJ_per_token"]
    gpu_energy = {}
    for (method, kind, fam, b), v in sorted(lowest.items()):
        gpu_energy.setdefault(method, []).append(dict(model=PPA_MODEL[kind], format=fam, batch=b, mJ_per_token=v))
    out = dict(
        rows=table,
        fastest=fastest,
        moe_share=share,
        moe_energy=energy,
        zeroed_step_ms={f"{st}|{kind}": v for (st, kind), v in stream.zeroed.items()},
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(out, indent=1))
    a.out.with_suffix(".md").write_text("\n".join(md) + "\n")
    for method, entries in gpu_energy.items():  # inputs of ppa.energy --gpu
        a.out.with_name(f"{a.out.name}_gpu_energy_{method}.json").write_text(json.dumps(entries, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
