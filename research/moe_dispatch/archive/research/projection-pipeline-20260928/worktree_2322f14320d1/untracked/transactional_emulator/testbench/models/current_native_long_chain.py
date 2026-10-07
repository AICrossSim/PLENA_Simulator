"""Captured raw Mamba gates -> current Compiler -> native Rust recurrence.

This is a producer/recurrence long-chain diagnostic, NOT a hidden-to-hidden
layer or K1024 projection execution. The historical capture contains no hidden
inputs. Initial state and source B/x/q are explicitly converted to BF16; raw
dt/bias/A execute the current BF16 softplus/rational-delta producer. The current
native coefficient descriptors consume compact B/C. Scalar dt/skip packing
executes ordinary gather instructions. FP32 update intermediates, BF16 tree
leaves/merges and per-physical-lane LFSR SR match the pinned Rust contract.

Only existing native checkpoints and the final state are retained. Per-token
raw outputs are retained; no 2048-token full-state tensor is allocated.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from analytic_models.performance.ltile_platform import ExecutionProfile
from compiler.aten.plena.ltile_v2 import Options, lower_group
from compiler.aten.plena.recurrent_coefficients import (
    GATE_CONSTANTS, MambaGateRow, lower_bf16_gather, lower_mamba_gate_rows,
)
from transactional_emulator.testbench.aten.recurrent_conv_test import read, run_program
from transactional_emulator.testbench.aten.recurrent_gate_test import Arena, bf, delta, metric, softplus


STREAMS = ("0_bfcl_v3", "1_gpqa_diamond", "2_swebench_verified")
LAYERS = (0, 21, 46)
SEED = 0x13579BDF
LANES = 256


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def same_bits(actual, expected):
    return np.array_equal(np.ascontiguousarray(actual, np.float32).view(np.uint32),
                          np.ascontiguousarray(expected, np.float32).view(np.uint32))


class LaneRandom:
    """Independent host implementation of accepted-launch lane RNG ordering.

    Rust launches rows in ascending order, then 256-element subgroups. The
    physical lane index restarts at zero for each subgroup. State DMA is head
    major, but random consumption is row/head/channel major. No random number
    is consumed by readout, packing, stalls, or gate generation.
    """

    def __init__(self, lanes=LANES, seed=SEED):
        self.lanes = lanes
        i = np.arange(lanes, dtype=np.uint32)
        self.state = np.uint32(seed) + i * np.uint32(0x9E3779B9)
        self.state[1:] |= np.uint32(1)
        self.state[0] = np.uint32(seed)
        self.draws_per_lane = 0

    def thresholds(self, heads, rows, width):
        if heads * width % self.lanes:
            raise ValueError("reference expects complete physical lane groups")
        random = np.empty((rows, heads * width // self.lanes, self.lanes), dtype=np.uint16)
        for r in range(rows):
            for chunk in range(random.shape[1]):
                random[r, chunk] = (self.state & np.uint32(65535)).astype(np.uint16)
                self.state = (self.state >> np.uint32(1)) ^ np.where(
                    self.state & np.uint32(1), np.uint32(0x80200003), np.uint32(0)
                )
                self.draws_per_lane += 1
        return random.reshape(rows, heads, width).transpose(1, 0, 2)


def stochastic_bf16(value, thresholds):
    value = np.ascontiguousarray(value, dtype=np.float32)
    if value.shape != thresholds.shape or not np.isfinite(value).all():
        raise ValueError("SR requires finite inputs and one threshold per value")
    bits = value.view(np.uint32)
    upper = (bits >> np.uint32(16)).astype(np.uint16)
    upper += (thresholds < (bits & np.uint32(65535))).astype(np.uint16)
    return (upper.astype(np.uint32) << np.uint32(16)).view(np.float32)


def pairwise_tree(values, axis):
    values = np.moveaxis(bf(values), axis, -1)
    while values.shape[-1] > 1:
        values = bf(values[..., ::2] + values[..., 1::2])
    return values[..., 0]


def update_reference(state, decay_delta, b, x, dt, c, skip, rounding, rng):
    """Separate RN32 operations, one state store, independent BF16 tree."""
    scaled = bf(x * dt[:, None])
    product = np.multiply(state, decay_delta[:, None, None], dtype=np.float32)
    decayed = np.subtract(state, product, dtype=np.float32)
    outer = np.multiply(b[:, :, None], scaled[:, None, :], dtype=np.float32)
    value = np.add(decayed, outer, dtype=np.float32)
    if rounding == "rn":
        updated = bf(value)
    else:
        # A complete 32-head group finishes before the second group starts.
        updated = np.empty_like(value)
        for first in (0, 32):
            updated[first:first + 32] = stochastic_bf16(
                value[first:first + 32], rng.thresholds(32, 128, 64)
            )
    leaves = bf(np.multiply(updated, c[:, :, None], dtype=np.float32))
    reduced = pairwise_tree(leaves, axis=1)
    # SCALE_ACCUM keeps D*x in FP32; only its final output is BF16.
    output = bf(np.add(reduced, np.multiply(x, skip[:, None], dtype=np.float32), dtype=np.float32))
    return updated, output


def padded(values):
    values = np.asarray(values, np.float32).ravel()
    if len(values) > 2048:
        raise ValueError("expected a compact producer row")
    return np.pad(values, (0, 2048 - len(values)))


def gather_scalar(source, first, destination, zero, onehot, *, ones=None):
    mapping = [None] * 2048
    for h in range(32):
        mapping[2 * h] = None if ones is None else (ones, 0)
        mapping[2 * h + 1] = (source, first + h)
    return lower_bf16_gather(mapping, destination, zero, onehot, strategy="cached")


def capture_tokens(folder, layer, count, pins):
    """Load one 32-token shard at a time; no long-chain state materialization."""
    expected = 0
    for start in range(0, count, 32):
        path = folder / f"model.layers.{layer}.mixer_inputs_{start:04d}.npz"
        pins[str(path)] = sha256(path)
        with np.load(path) as archive:
            take = min(count - start, len(archive["token_indices"]))
            indices = archive["token_indices"][:take]
            if not np.array_equal(indices, np.arange(start, start + take)):
                raise ValueError("capture tokens are not a continuous prefix")
            checkpoints = {}
            if "checkpoint_states" in archive:
                checkpoints = dict(zip(archive["checkpoint_indices"].tolist(), archive["checkpoint_states"]))
            for t in range(take):
                raw_dt = archive["source_dt"][t, 0]
                bias = archive["source_dt_bias"][t]
                a = archive["source_A"][t]
                if not (np.all(raw_dt == raw_dt[:, :1]) and np.all(bias == bias[:, :1])
                        and np.all(a == a[:, :1, :1])):
                    raise ValueError("raw Mamba scalar fields are not head-invariant")
                q = archive["q"][t]
                if not np.array_equal(q, np.repeat(q[::8], 8, axis=0)):
                    raise ValueError("captured C is not shared by groups of eight heads")
                yield dict(
                    index=start + t, raw_dt=bf(raw_dt[:, 0]), bias=bf(bias[:, 0]),
                    a=bf(a[:, 0, 0]), b=bf(archive["source_B"][t, 0]),
                    x=bf(archive["source_x"][t, 0]), c=bf(q[::8]),
                    skip=bf(archive["d"][t]), native_output=archive["native_output"][t].copy(),
                    native_checkpoint=checkpoints.get(start + t),
                )
                expected += 1
    if expected != count:
        raise ValueError("capture has fewer tokens than requested")


def build_case(args):
    folder = args.capture_root / args.stream
    initial_path = folder / f"model.layers.{args.layer}.mixer_initial.npy"
    initial = np.load(initial_path)
    if initial.shape != (64, 128, 64) or not np.isfinite(initial).all():
        raise ValueError("unexpected native initial state")
    pins = {str(initial_path): sha256(initial_path)}
    manifest = folder / "manifest.json"
    if manifest.exists():
        pins[str(manifest)] = sha256(manifest)
    arena = Arena()
    zero = arena.add(np.zeros(2048, np.float32))
    onehot = arena.add(np.eye(1, 2048, dtype=np.float32).ravel())
    ones = arena.add(np.ones(2048, np.float32))
    constants = arena.add(np.repeat(np.asarray(GATE_CONSTANTS, np.float32)[:, None], 2048, axis=1))
    states = [arena.add(initial[g:g + 32], output=True) for g in (0, 32)]
    # Each token reuses these scratch HBM rows. Their lifetimes do not overlap.
    scalar = [arena.add(np.zeros(2048, np.float32), output=True) for _ in range(2)]
    skip_scalar = [arena.add(np.zeros(2048, np.float32), output=True) for _ in range(2)]
    expected_state = bf(initial)
    rng = LaneRandom()
    expected_raw = np.empty((args.tokens, 64, 64), np.float32)
    native_raw = np.empty_like(expected_raw)
    expected_dt = np.empty((args.tokens, 64), np.float32)
    expected_delta = np.empty_like(expected_dt)
    addresses = []
    checkpoint_refs = {}
    native_states = {}
    lines = []
    # Static sources are interned as immutable rows; values are never host-
    # prepared dt/delta, only actual model constants/raw producer inputs.
    static = {}

    def static_row(value):
        value = padded(value)
        key = value.tobytes()
        if key not in static:
            static[key] = arena.add(value)
        return static[key]

    for source in capture_tokens(folder, args.layer, args.tokens, pins):
        token = source["index"]
        raw = arena.add(padded(source["raw_dt"]))
        bias = static_row(source["bias"])
        negative_a = static_row(source["a"])
        skip = static_row(source["skip"])
        dt_out = arena.add(np.full(2048, 7, np.float32), output=True)
        delta_out = arena.add(np.full(2048, 7, np.float32), output=True)
        lines.append(f"; @stage=token{token}/current_raw_gate\n")
        lines.append(lower_mamba_gate_rows([MambaGateRow(raw, bias, negative_a, dt_out, delta_out)], constants))
        dt = softplus(bf(source["raw_dt"] + source["bias"]))
        decay_delta = delta(bf(dt * source["a"]))
        expected_dt[token], expected_delta[token] = dt, decay_delta
        bc = arena.add(np.concatenate((source["b"].ravel(), source["c"].ravel())))
        x_base = arena.add(source["x"])
        raw_out = arena.add(np.full(4096, 7, np.float32), output=True)
        snapshot_bases = []
        checkpoint = source["native_checkpoint"] is not None or token == args.tokens - 1
        for group in range(2):
            first = group * 32
            lines.append(f"; @stage=token{token}/scalar_gather{group}\n")
            lines.append(gather_scalar(dt_out, first, scalar[group], zero, onehot))
            if token == 0:
                lines.append(gather_scalar(skip, first, skip_scalar[group], zero, onehot, ones=ones))
            elif source["skip"].tobytes() != first_skip:
                raise ValueError("D changed across captured tokens")
            memory = dict(
                states=[states[group]],
                native=[(delta_out, first, 1, 0, 0),
                        (bc, group * 512, 128, 1, 3),
                        (bc, 1024 + group * 512, 128, 1, 3)],
                input=x_base + group * 4096, scalar=scalar[group], skip=skip_scalar[group],
                output=raw_out + group * 4096,
            )
            if checkpoint:
                dst = arena.add(np.full((32, 128, 64), 7, np.float32), output=True)
                memory["snapshots"] = [dst]
                snapshot_bases.append(dst)
            lines.append(f"; @stage=token{token}/native_recurrence{group}\n")
            lines.append("\n".join(lower_group(Options("mamba"), memory).lines) + "\n")
        if token == 0:
            first_skip = source["skip"].tobytes()
        b = source["b"][np.arange(64) // 8]
        c = source["c"][np.arange(64) // 8]
        expected_state, expected_raw[token] = update_reference(
            expected_state, decay_delta, b, source["x"], dt, c, source["skip"], args.rounding, rng
        )
        native_raw[token] = source["native_output"]
        if checkpoint:
            checkpoint_refs[token] = (snapshot_bases, expected_state.copy())
        if source["native_checkpoint"] is not None:
            native_states[token] = source["native_checkpoint"].copy()
        addresses.append((dt_out, delta_out, raw_out))
        if token % 32 == 31:
            print(json.dumps(dict(phase="build", stream=args.stream, layer=args.layer,
                                  rounding=args.rounding, tokens=token + 1)), flush=True)
    return dict(arena=arena, assembly="".join(lines), input_pins=pins, addresses=addresses,
                expected_dt=expected_dt, expected_delta=expected_delta, expected_raw=expected_raw,
                checkpoints=checkpoint_refs, native_states=native_states, native_raw=native_raw,
                final_state=expected_state, state_addresses=states, draws_per_lane=rng.draws_per_lane)


def archive_run(work, target):
    """Retain audit material; remove only this run's regenerable HBM dumps."""
    hashes = {p.name: sha256(p) for p in work.iterdir() if p.is_file()}
    json_write(target / "execution_file_sha256.json", hashes)
    for path in work.iterdir():
        if not path.is_file() or path.name in ("hbm_for_behave_sim.bin", "hbm_dump.bin"):
            continue
        if path.suffix in (".asm", ".mem", ".log"):
            with path.open("rb") as source, gzip.open(target / (path.name + ".gz"), "wb", compresslevel=6) as dest:
                shutil.copyfileobj(source, dest)
        else:
            shutil.copy2(path, target / path.name)
    # The directory is generated under the named temporary root and never
    # contains source/captures; no broad project cleanup is performed.
    shutil.rmtree(work)


def run_case(args):
    torch.set_num_threads(1)
    name = f"{args.stream}_layer{args.layer}_{args.tokens}_{args.rounding}"
    target = args.output / name
    target.mkdir(parents=True, exist_ok=False)
    work = args.temporary_root / name
    if work.exists():
        raise FileExistsError(f"unreviewed temporary run already exists: {work}")
    start = time.monotonic()
    profile = ExecutionProfile(hbm_controllers=16, state_rounding=args.rounding,
                               projection_schedule="transposed", projection_k_tile=1024)
    json_write(target / "status.json", dict(status="building", scope=__doc__, profile=asdict(profile)))
    fixture = build_case(args)
    json_write(target / "input_sha256.json", fixture["input_pins"])
    import inspect
    implementation_files = {
        str(Path(__file__).resolve()),
        inspect.getfile(lower_mamba_gate_rows), inspect.getfile(lower_bf16_gather),
        inspect.getfile(lower_group), inspect.getfile(run_program), inspect.getfile(ExecutionProfile),
    }
    from compiler.aten.plena.ltile_native import lower_native_group
    implementation_files.add(inspect.getfile(lower_native_group))
    json_write(target / "implementation_sha256.json", {p: sha256(p) for p in sorted(implementation_files)})
    json_write(target / "status.json", dict(status="executing", tokens=args.tokens,
                hbm_image_bytes=len(fixture["arena"].data), assembly_bytes=len(fixture["assembly"]),
                runtime_sha256=sha256(args.runtime), scope=__doc__))
    image, result = run_program(work, args.runtime, args.memory_root, fixture["arena"], fixture["assembly"],
                                profile=profile)
    actual_dt = np.empty_like(fixture["expected_dt"])
    actual_delta = np.empty_like(actual_dt)
    actual_raw = np.empty_like(fixture["expected_raw"])
    for t, (dt_base, delta_base, out) in enumerate(fixture["addresses"]):
        actual_dt[t] = read(image, dt_base, 64)
        actual_delta[t] = read(image, delta_base, 64)
        actual_raw[t] = read(image, out, 4096).reshape(64, 64)
    implementation = {"dt": metric(actual_dt, fixture["expected_dt"]),
                      "delta": metric(actual_delta, fixture["expected_delta"]),
                      "raw_output": metric(actual_raw, fixture["expected_raw"])}
    exact = {"dt": same_bits(actual_dt, fixture["expected_dt"]),
             "delta": same_bits(actual_delta, fixture["expected_delta"]),
             "raw_output": same_bits(actual_raw, fixture["expected_raw"])}
    state_native, saved = [], {}
    for token, (bases, expected) in fixture["checkpoints"].items():
        actual = np.concatenate([read(image, b, 32 * 128 * 64).reshape(32, 128, 64) for b in bases])
        key = f"state_token{token}"
        implementation[key] = metric(actual, expected)
        exact[key] = same_bits(actual, expected)
        saved[key] = actual
        saved[key + "_implementation_reference"] = expected
        if token in fixture["native_states"]:
            state_native.append(dict(token=token, **metric(actual, fixture["native_states"][token])))
    final = np.concatenate([read(image, b, 32 * 128 * 64).reshape(32, 128, 64)
                            for b in fixture["state_addresses"]])
    exact["persistent_final_state"] = same_bits(final, fixture["final_state"])
    implementation["persistent_final_state"] = metric(final, fixture["final_state"])
    output_metrics = [dict(token=t, **metric(actual_raw[t], fixture["native_raw"][t]))
                      for t in range(args.tokens)]
    result.update(
        status="passed" if all(exact.values()) else "failed_implementation_exact", scope=__doc__,
        stream=args.stream, layer=args.layer, tokens=args.tokens, state_rounding=args.rounding,
        projection_executed=False,
        projection_profile_note="K1024 software projection profile is shared metadata; this program issues no Matrix projection instruction",
        cycles_scope="numerical diagnostic, including extra checkpoint DMA; not the no-diagnostic performance program",
        rng_algorithm="per_physical_lane_32bit_LFSR_0x80200003" if args.rounding == "sr" else None,
        rng_seed=SEED if args.rounding == "sr" else None,
        rng_draws_per_lane=fixture["draws_per_lane"] if args.rounding == "sr" else 0,
        native_repeat_evidence="pinned capture driver reports exact repeats; not rerun by this CPU diagnostic",
        implementation_exact=exact, implementation_exact_definition="float32 bits after BF16 commit, including signed zero",
        implementation_errors=implementation,
        native_state_checkpoints=state_native,
        native_state_peak_rel_l2_at_saved_checkpoints=max((x["rel_l2"] for x in state_native), default=None),
        native_state_final=next((x for x in state_native if x["token"] == args.tokens - 1), None),
        native_raw_output=metric(actual_raw, fixture["native_raw"]),
        native_raw_peak_rel_l2=max(x["rel_l2"] for x in output_metrics),
        elapsed_seconds=time.monotonic() - start,
        inputs_sha256=fixture["input_pins"], profile_identity=profile.identity,
        reproduce=shlex.join([sys.executable, "-m",
                             "transactional_emulator.testbench.models.current_native_long_chain", *sys.argv[1:]]),
        exclusions=["projection/conv/norm/output projection", "full model logits or task quality",
                    "FP32 context", "Philox trajectory equivalence", "per-token peak state outside saved checkpoints",
                    "RTL/PPA certification"],
    )
    np.savez_compressed(target / "compact_values.npz", **saved, dt=actual_dt, delta=actual_delta,
                        raw_output=actual_raw, native_raw_output=fixture["native_raw"])
    json_write(target / "raw_output_per_token.json", output_metrics)
    json_write(target / "result.json", result)
    archive_run(work, target)
    json_write(target / "status.json", dict(status=result["status"], elapsed_seconds=result["elapsed_seconds"]))
    json_write(target / "artifact_sha256.json", {p.name: sha256(p) for p in target.iterdir()
               if p.is_file() and p.name != "artifact_sha256.json"})
    print(json.dumps(dict(case=name, status=result["status"], cycles=result["observed"]["total"],
                          raw_rel_l2=result["native_raw_output"]["rel_l2"],
                          final_state=result["native_state_final"])), flush=True)
    if not all(exact.values()):
        raise AssertionError("Rust differs from the independent implementation reference")


def campaign(args):
    """Persistent, bounded CPU queue. Short exact gates precede long runs."""
    args.output.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, "-m", "transactional_emulator.testbench.models.current_native_long_chain",
            "--capture-root", str(args.capture_root), "--runtime", str(args.runtime),
            "--memory-root", str(args.memory_root), "--output", str(args.output),
            "--temporary-root", str(args.temporary_root)]
    status = {}

    def execute(stream, layer, tokens, rounding):
        name = f"{stream}_layer{layer}_{tokens}_{rounding}"
        result = args.output / name / "result.json"
        if result.exists():
            previous = json.loads(result.read_text())
            profile = ExecutionProfile(hbm_controllers=16, state_rounding=rounding,
                                       projection_schedule="transposed", projection_k_tile=1024)
            if (previous["status"] == "passed" and previous["runtime_sha256"] == sha256(args.runtime)
                    and previous["profile_identity"] == profile.identity):
                return name, "already_passed"
            raise ValueError(f"cannot reuse existing result with another runtime/profile: {name}")
        command = base + ["--stream", stream, "--layer", str(layer), "--tokens", str(tokens), "--rounding", rounding]
        with (args.output / (name + ".worker.log")).open("w") as log:
            done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                  env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        return name, "passed" if done.returncode == 0 else "failed"

    for rounding in ("rn", "sr"):
        name, outcome = execute(STREAMS[0], LAYERS[0], 32, rounding)
        status[name] = outcome
        json_write(args.output / "campaign_status.json", status)
        if outcome == "failed":
            raise RuntimeError("short exact diagnostic failed; long queue not launched")
    jobs = [(stream, layer, 2048, rounding) for stream in STREAMS for layer in LAYERS for rounding in ("rn", "sr")]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(execute, *job) for job in jobs]
        for future in as_completed(futures):
            name, outcome = future.result()
            status[name] = outcome
            json_write(args.output / "campaign_status.json", status)
            print(json.dumps(dict(case=name, status=outcome)), flush=True)
    if any(value == "failed" for value in status.values()):
        raise RuntimeError("at least one long-chain case failed")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("capture-root", "runtime", "memory-root", "output", "temporary-root"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--stream", choices=STREAMS, default=STREAMS[0])
    p.add_argument("--layer", type=int, choices=LAYERS, default=0)
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--rounding", choices=("rn", "sr"), default="rn")
    p.add_argument("--campaign", action="store_true")
    p.add_argument("--workers", type=int, choices=(1, 2), default=1)
    args = p.parse_args()
    if not 1 <= args.tokens <= 2048:
        p.error("tokens must be in 1..2048")
    args.output = args.output.resolve()
    args.temporary_root = args.temporary_root.resolve()
    # Prevent a typo from permitting cleanup inside the workspace or home.
    if not str(args.temporary_root).startswith("/tmp/plena-current-native-chain-"):
        p.error("temporary-root must be a dedicated /tmp/plena-current-native-chain-* directory")
    args.temporary_root.mkdir(parents=True, exist_ok=True)
    if args.campaign:
        campaign(args)
    else:
        try:
            run_case(args)
        except Exception as error:
            target = args.output / f"{args.stream}_layer{args.layer}_{args.tokens}_{args.rounding}"
            if target.exists():
                json_write(target / "status.json", dict(status="failed", error_type=type(error).__name__,
                           reason=str(error), temporary_path=str(args.temporary_root / target.name)))
            raise


if __name__ == "__main__":
    main()
