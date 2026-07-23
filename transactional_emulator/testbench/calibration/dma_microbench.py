"""Pure-DMA transfer-size microbenchmark for the effective-bandwidth model.

Measures the H_PREFETCH_M service time as a function of transfer size by
running programs that contain nothing but back-to-back prefetches (no compute,
no correctness constraints), with the emulator in --blocking-prefetch mode so
each instruction's --op-stats record is its exact DMA time. Transfer size is
swept via HBM_M_Prefetch_Amount (rows per prefetch): bytes/transfer =
amount x MLEN x element_bits/8 (+ scales). Each prefetch reads a fresh HBM
region so Ramulator sees realistic row/bank behavior rather than one hot row.

The output CSV feeds the alpha-beta fit in analytic_models/disagg_serve/
memory.py: time/transfer = t0 + bytes/BW_inf, which lets the analytic model
price transfers of any size (large-MLEN chips issue MB-scale transfers) from
measured physics instead of clamping to the decode-testbench transfer size.

Usage (repo root, .venv python inside the nix shell):
  python transactional_emulator/testbench/calibration/dma_microbench.py \
      --amounts 4,8,16,32,64,128,256,512,1024,2048,4096 \
      --channels 8,16,32 --gens HBM2,HBM3 \
      --out analytic_models/disagg_serve/calibration_dma.csv
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

N_PREFETCH = 32  # transfers per point; enough for a stable mean
MLEN = 64        # emulator tile width (fixed by the 512-bit HBM burst design)


def build_asm(amount: int, elem_bits: float) -> str:
    """Program: address setup, then N_PREFETCH H_PREFETCH_M ops, each reading a
    fresh HBM region (offset advanced by the transfer's element count)."""
    from asm_templates._imm import load_large_int_str

    transfer_elems = amount * MLEN
    lines = [
        "; DMA microbenchmark: back-to-back H_PREFETCH_M, fresh region each",
        "S_ADDI_INT gp1, gp0, 0",
        "C_SET_ADDR_REG a1, gp0, gp1 ",
        # Scales live far past the element region touched by the sweep
        # (16M elements; the largest sweep touches ~8.4M).
        load_large_int_str(1, 16_777_216).rstrip("\n"),
        "C_SET_SCALE_REG gp1 ",
        f"S_ADDI_INT gp2, gp0, {MLEN}",
        "C_SET_STRIDE_REG gp2 ",
        "S_ADDI_INT gp3, gp0, 0",
    ]
    for _ in range(N_PREFETCH):
        lines.append("H_PREFETCH_M gp0, gp3, a1, 0, 0 ")
        lines.append(load_large_int_str(4, transfer_elems).rstrip("\n"))
        lines.append("S_ADD_INT gp3, gp3, gp4")
    lines.append("C_BREAK")
    return "\n".join(lines) + "\n"


def patch_settings(base: str, gen: str, channels: int, amount: int) -> str:
    out = re.sub(r'(\[TRANSACTIONAL\.CONFIG\.HBM_GEN\][^\[]*?value = )"[A-Za-z0-9]+"',
                 rf'\1"{gen}"', base, flags=re.S)
    out = re.sub(r"(\[TRANSACTIONAL\.CONFIG\.HBM_CHANNELS\][^\[]*?value = )\d+",
                 rf"\g<1>{channels}", out, flags=re.S)
    out = re.sub(r"(\[TRANSACTIONAL\.CONFIG\.HBM_M_Prefetch_Amount\][^\[]*?value = )\d+",
                 rf"\g<1>{amount}", out, flags=re.S)
    # The destination SRAM must hold one full transfer.
    out = re.sub(r"(\[TRANSACTIONAL\.CONFIG\.MATRIX_SRAM_SIZE\][^\[]*?value = )\d+",
                 r"\g<1>8192", out, flags=re.S)
    return out


def parse_prefetch_stats(path: Path) -> dict:
    agg = None
    with open(path) as f:
        for line in f:
            if '"aggregate"' in line:
                agg = json.loads(line)
    for o in agg["ops"]:
        if o["op"] == "H_PREFETCH_M":
            return o
    raise RuntimeError(f"no H_PREFETCH_M in {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--amounts", default="4,8,16,32,64,128,256,512,1024,2048,4096")
    ap.add_argument("--channels", default="8,16,32")
    ap.add_argument("--gens", default="HBM2,HBM3")
    ap.add_argument("--out", default=str(_REPO / "analytic_models" / "disagg_serve" / "calibration_dma.csv"))
    args = ap.parse_args()

    amounts = [int(x) for x in args.amounts.split(",")]
    channels = [int(x) for x in args.channels.split(",")]
    gens = [g.strip() for g in args.gens.split(",")]

    sys.path.insert(0, str(_REPO / "compiler"))
    from assembler.assembly_to_binary import AssemblyToBinary

    base_toml = (_REPO / "plena_settings.toml").read_text()
    tmpdir = Path(tempfile.mkdtemp(prefix="plena_dma_"))
    binary = _EMULATOR / "target" / "release" / "transactional_emulator"

    # Zero-filled memory images; data values are irrelevant to DMA timing.
    hbm_bin = tmpdir / "hbm.bin"
    hbm_bin.write_bytes(b"\x00" * (128 * 1024 * 1024))
    # Scalar SRAMs are 1024 entries; a token zero preload is enough.
    (tmpdir / "fp_sram.bin").write_bytes(b"\x00" * 8)
    (tmpdir / "int_sram.bin").write_bytes(b"\x00" * 8)

    # libtorch runtime path for the tch-linked binary.
    libtorch = list((_EMULATOR / "target" / "release" / "build").glob("torch-sys-*/out/libtorch/libtorch/lib"))
    env = {**os.environ}
    if libtorch:
        env["LD_LIBRARY_PATH"] = f"{libtorch[0]}:{env.get('LD_LIBRARY_PATH', '')}"
    env.setdefault("OMP_NUM_THREADS", "1")

    assembler = AssemblyToBinary(
        str(_REPO / "compiler" / "doc" / "operation.svh"),
        str(_REPO / "compiler" / "doc" / "configuration.svh"),
    )

    rows = []
    for amount in amounts:
        asm_path = tmpdir / f"dma_{amount}.asm"
        mem_path = tmpdir / f"dma_{amount}.mem"
        asm_path.write_text(build_asm(amount, elem_bits=4))
        assembler.generate_binary(str(asm_path), str(mem_path))

        for gen in gens:
            for ch in channels:
                toml_path = tmpdir / f"s_{gen}_{ch}_{amount}.toml"
                toml_path.write_text(patch_settings(base_toml, gen, ch, amount))
                stats_path = tmpdir / f"ops_{gen}_{ch}_{amount}.jsonl"
                cmd = [
                    str(binary),
                    "--opcode", str(mem_path),
                    "--hbm", str(hbm_bin),
                    "--fpsram", str(tmpdir / "fp_sram.bin"),
                    "--intsram", str(tmpdir / "int_sram.bin"),
                    "--hbm-size", "256M",
                    "--settings", str(toml_path),
                    "--log-level", "warn",
                    "--blocking-prefetch",
                    "--op-stats", str(stats_path),
                ]
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
                if proc.returncode != 0:
                    sys.stderr.write(proc.stderr[-3000:] + "\n")
                    raise RuntimeError(f"emulator failed: gen={gen} ch={ch} amount={amount}")
                o = parse_prefetch_stats(stats_path)
                bytes_per = o["hbm_rd"] / o["count"]
                dt_per_ps = o["dt_ps"] / o["count"]
                rows.append({
                    "hbm_gen": gen, "channels": ch, "amount": amount,
                    "count": o["count"],
                    "bytes_per_transfer": bytes_per,
                    "dt_ps_per_transfer": dt_per_ps,
                    "achieved_gbps": bytes_per / (dt_per_ps / 1e12) / 1e9,
                })
                print(f"[dma] gen={gen} ch={ch} amount={amount}: "
                      f"{bytes_per/1024:.1f} KiB/transfer @ {rows[-1]['achieved_gbps']:.1f} GB/s",
                      flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[dma] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
