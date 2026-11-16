"""Run sampled windows (sample.py) with the closed-loop HBM service of hbm_ramulator.hpp.

simulate() with ram_config (Ramulator configuration text; None = ideal HBM stream) uses the closed-loop build of the
simulator, linked against libramulator in $RAMULATOR_LIB; ram_extra_latency, ram_issue_cap, ram_layout and ram_look set
the service and ram_log the file prefix of the per-window logs. Each window is its own call, so the logs are named
<OUT>_w0.json and <OUT>_w0_req.csv next to the output.

usage: python -m complementary_arrays.xcheck.run SAMPLE_DIR TAG WINDOW OUT.npz [--engine-dir DIR]
       [--config hbm2_2134.json] [--extra 65] [--cap 0] [--layout 0] [--look 512] [--ideal]
  --engine-dir loads engine.py from DIR instead of complementary_arrays.sim.engine. --ideal runs without Ramulator
  (must reproduce the sampled reference).
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def ramulator_config(path):
    """Full Ramulator configuration text from the compact form: one controller repeated for every channel."""
    c = json.loads(Path(path).read_text())
    ms = dict(c["memory_system"])
    mapper = ms.pop("channel_mapper")
    ms["controllers"] = [c["controller"]] * c["channels"]
    ms["channel_mapper"] = mapper
    return json.dumps(dict(frontend=c["frontend"], memory_system=ms), separators=(",", ":"))


def engine(engine_dir):
    if engine_dir is None:
        from ..sim import engine as e

        return e
    spec = importlib.util.spec_from_file_location("xcheck_engine", Path(engine_dir) / "engine.py")
    e = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e)
    return e


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("samples", type=Path)
    p.add_argument("tag", help="<tag>_B<batch> written by sample.py")
    p.add_argument("window", type=int)
    p.add_argument("out", type=Path)
    p.add_argument("--engine-dir", type=Path)
    p.add_argument("--config", type=Path, default=HERE / "hbm2_2134.json")
    p.add_argument("--extra", type=int, default=65, help="on-chip cycles after the last returned transfer")
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--layout", type=int, default=0)
    p.add_argument("--look", type=int, default=512)
    p.add_argument("--ideal", action="store_true")
    a = p.parse_args()
    if a.out.exists():
        return
    eng = engine(a.engine_dir)
    info = json.loads((a.samples / f"{a.tag}.json").read_text())
    with np.load(a.samples / f"{a.tag}.npz") as z:
        ids, ref = z["ids"][a.window : a.window + 1], z["out"][a.window : a.window + 1]
    env = eng.Environment.from_dict(info["env"])
    kw = {"provision_tokens": info["provision_tokens"]} if "provision_tokens" in info else {}
    if not a.ideal:
        kw.update(
            ram_config=ramulator_config(a.config),
            ram_extra_latency=a.extra,
            ram_issue_cap=a.cap,
            ram_layout=a.layout,
            ram_look=a.look,
            ram_log=str(a.out.with_suffix("")),
        )
    t0 = time.time()
    out = eng.simulate(ids, info["shapes"], info["workload"], env, threads=1, **kw)
    out = out[0] if isinstance(out, tuple) else out
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name("." + a.out.name + ".tmp.npz")
    np.savez(tmp, out=out, seconds=time.time() - t0)
    os.replace(tmp, a.out)
    print(
        a.tag,
        a.window,
        "cycles",
        int(out[0, 0]),
        "ideal",
        int(ref[0, 0]),
        "equal",
        bool(np.array_equal(out, ref)),
        flush=True,
    )


if __name__ == "__main__":
    main()
