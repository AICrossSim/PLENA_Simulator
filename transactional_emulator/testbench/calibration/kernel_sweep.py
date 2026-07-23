"""Effective-bandwidth calibration sweep for the decode chip.

Runs the decode-step testbench across (kv_size x hbm_gen x channels) points
with the emulator in --blocking-prefetch mode, so every H_PREFETCH_M/V and
H_STORE_V --op-stats record is the exact DMA service time of its transfer
(no compute overlap). Per point and per op class this yields: transfer count,
bytes moved, total DMA time, and achieved bandwidth — the measurements the
analytic effective-bandwidth model (analytic_models/disagg_serve/memory.py)
is fitted on.

The decode-step assembly depends only on kv_size, so each kv_size is
generated once (PyTorch golden + asm + memory images) and the emulator is
then re-run on the same build dir for every (gen, channels) point.

Usage (from the PLENA_Simulator repo root, using its .venv python):
  python transactional_emulator/testbench/calibration/kernel_sweep.py \
      --kv-sizes 128,256,512,1024,2048 --channels 8,16,32 --gens HBM2,HBM3 \
      --out analytic_models/disagg_serve/calibration_bw.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_TESTBENCH = _HERE.parent
_EMULATOR = _TESTBENCH.parent
_REPO = _EMULATOR.parent
sys.path.insert(0, str(_TESTBENCH))

from emulator_runner import run_emulator  # noqa: E402

# Memory-traffic op classes the model distinguishes. H_STORE_V is a
# read-modify-write on the emulator (it reads the 64B lines it will
# partially overwrite), so its bytes are rd+wr.
MEM_OPS = ("H_PREFETCH_M", "H_PREFETCH_V", "H_STORE_V")


def patch_settings(base_toml: str, gen: str, channels: int) -> str:
    """Return the TOML text with HBM_GEN/HBM_CHANNELS values replaced."""
    out = re.sub(
        r'(\[TRANSACTIONAL\.CONFIG\.HBM_GEN\][^\[]*?value = )"[A-Za-z0-9]+"',
        rf'\1"{gen}"',
        base_toml,
        flags=re.S,
    )
    out = re.sub(
        r"(\[TRANSACTIONAL\.CONFIG\.HBM_CHANNELS\][^\[]*?value = )\d+",
        rf"\g<1>{channels}",
        out,
        flags=re.S,
    )
    return out


def generate_build(kv_size: int, python: str) -> Path:
    """Generate the decode-step build artifacts for one kv_size (slow: torch)."""
    build_dir = _TESTBENCH / "build" / "decoder_decode"
    # The generator imports the assembler from compiler/ and the testbench
    # helpers by package path from the repo root.
    env = {**os.environ}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO / "compiler"), str(_REPO), env.get("PYTHONPATH", "")]
    )
    # Generation is emulator-independent, but decoder_decode_test also runs the
    # emulator once; let that first run use the default settings.
    proc = subprocess.run(
        [python, str(_TESTBENCH / "misc" / "decoder_decode_test.py"), "--kv-size", str(kv_size)],
        cwd=_REPO,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + "\n" + proc.stderr[-4000:] + "\n")
        raise RuntimeError(f"decoder_decode_test failed for kv_size={kv_size}")
    return build_dir


def parse_op_stats(path: Path) -> dict:
    """Return {op: {count, dt_ps, hbm_rd, hbm_wr}} from the aggregate line."""
    agg = None
    with open(path) as f:
        for line in f:
            if '"aggregate"' in line:
                agg = json.loads(line)
    if agg is None:
        raise RuntimeError(f"no aggregate line in {path}")
    per_op = {o["op"]: o for o in agg["ops"]}
    per_op["_total"] = {
        "count": 1,
        "dt_ps": agg["total_dt_ps"],
        "hbm_rd": agg["total_hbm_rd"],
        "hbm_wr": agg["total_hbm_wr"],
    }
    return per_op


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kv-sizes", default="128,256,512,1024")
    ap.add_argument("--channels", default="8,16,32")
    ap.add_argument("--gens", default="HBM2,HBM3")
    ap.add_argument("--out", default=str(_REPO / "analytic_models" / "disagg_serve" / "calibration_bw.csv"))
    ap.add_argument("--python", default=sys.executable,
                    help="python used for workload generation (needs torch)")
    args = ap.parse_args()

    kv_sizes = [int(x) for x in args.kv_sizes.split(",")]
    channels = [int(x) for x in args.channels.split(",")]
    gens = [g.strip() for g in args.gens.split(",")]

    base_toml = (_REPO / "plena_settings.toml").read_text()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    tmpdir = Path(tempfile.mkdtemp(prefix="plena_calib_"))

    for kv in kv_sizes:
        print(f"[calib] generating decode build for kv_size={kv} ...", flush=True)
        build_dir = generate_build(kv, args.python)

        for gen in gens:
            for ch in channels:
                toml_path = tmpdir / f"settings_{gen}_{ch}.toml"
                toml_path.write_text(patch_settings(base_toml, gen, ch))
                stats_path = tmpdir / f"op_stats_kv{kv}_{gen}_{ch}.jsonl"

                os.environ["PLENA_SETTINGS_TOML"] = str(toml_path)
                os.environ["PLENA_EMU_EXTRA_ARGS"] = (
                    f"--blocking-prefetch --op-stats {stats_path}"
                )
                try:
                    run_stats = run_emulator(build_dir)
                finally:
                    os.environ.pop("PLENA_SETTINGS_TOML", None)
                    os.environ.pop("PLENA_EMU_EXTRA_ARGS", None)

                per_op = parse_op_stats(stats_path)
                for op in (*MEM_OPS, "_total"):
                    o = per_op.get(op)
                    if not o:
                        continue
                    bytes_moved = o["hbm_rd"] + o["hbm_wr"]
                    dt_s = o["dt_ps"] / 1e12
                    rows.append({
                        "kv_size": kv,
                        "hbm_gen": gen,
                        "channels": ch,
                        "op": op,
                        "count": o["count"],
                        "bytes": bytes_moved,
                        "dt_ps": o["dt_ps"],
                        "achieved_gbps": (bytes_moved / dt_s / 1e9) if dt_s > 0 else 0.0,
                        "sim_latency_ns": run_stats.get("sim_latency_ns"),
                    })
                print(
                    f"[calib] kv={kv} gen={gen} ch={ch}: "
                    + ", ".join(
                        f"{op}={r['achieved_gbps']:.1f}GB/s"
                        for op in MEM_OPS
                        for r in rows[-4:]
                        if r["op"] == op and r["kv_size"] == kv
                        and r["hbm_gen"] == gen and r["channels"] == ch
                    ),
                    flush=True,
                )

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[calib] wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
