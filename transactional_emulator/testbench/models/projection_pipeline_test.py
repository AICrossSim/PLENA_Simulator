"""Execute finite projection schedules on historical layer inputs; no predicted cycles used.

Same-arithmetic full-array oracle from prior machine-code executions. Batch
members repeat the captured input, but own distinct state/intermediate addresses.
Projection panels are shared by the real batch compiler. BF16 weights execute;
this experiment does not include NVFP4 runtime decoding or whole-model inference.
"""

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import inspect
from itertools import pairwise
import json
import os
import re
from pathlib import Path
import shutil
import subprocess

from analytic_models.performance import ltile_layers as layers
from analytic_models.performance.ltile_platform import ExecutionProfile
from analytic_models.performance.ltile_program import ShapeArena


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


class RecordedArena(ShapeArena):
    """Associate each allocation with its compiler call site and occurrence.

    This is fixture relocation only. Data never changes after machine launch.
    Every allocation's size must match the B1 fixture before it can be reused.
    """

    def __init__(self):
        super().__init__()
        self.records, self.counts = [], {}
        self.request = -1

    def add(self, count):
        stack = []
        contexts = []
        frame = inspect.currentframe().f_back
        while frame and Path(frame.f_code.co_filename).name == "ltile_layers.py":
            stack.append((frame.f_code.co_name, frame.f_lineno))
            contexts.extend(inspect.getframeinfo(frame).code_context or [])
            if frame.f_code.co_name == "build_layer":
                break
            frame = frame.f_back
        first = inspect.getframeinfo(inspect.currentframe().f_back).code_context[0]
        if "self.zero =" in first:
            self.request += 1
            self.counts = {}
        key = tuple(stack)
        index = self.counts.get(key, 0)
        self.counts[key] = index + 1
        base = super().add(count)
        # Unused legacy packing allocations preserve the old fixture address ABI.
        unused = any(
            s in "".join(contexts)
            for s in ("p.output(len(update))", "p.output(len(dot))", "p.output(len(umap))", "p.output(len(dmap))")
        )
        self.records.append((base, count * 2, (key, index), unused, self.request))
        return base


def marked(plan):
    names = {
        "projection": "mamba_in_proj",
        "coefficient_layout": "gather",
        "recurrence": "mamba_state_update",
        "conv": "mamba_conv1d",
        "norm": "mamba_gated_norm",
        "gate_prepare": "mamba_dt",
    }
    result = []
    for s in plan.stages:
        text = re.sub(r"(?m)^\s*;\s*@stage=.*$", "", s.assembly)
        result.append(f"; @stage={names[s.category]}\n; operator={s.name}\n{text}")
    return "".join(result)


def run(args):
    if args.handoff_shared and not args.handoff:
        raise ValueError("--handoff-shared requires --handoff")
    layers.ShapeArena = RecordedArena
    profile = ExecutionProfile(hbm_controllers=16, projection_schedule=args.schedule,
        projection_vector_rows=48 if args.handoff and not args.handoff_shared else 58,
        gather_vector_rows=48 if args.handoff else 64)
    profile = replace(profile, matrix=replace(profile.matrix, weight_replay=args.replay))
    profile = replace(
        profile, machine=replace(profile.machine, native_read_width=args.width, native_result_slots=args.credits)
    )
    kwargs = dict(control="fsm", gather="cached", projection_schedule=args.schedule, native_coefficients=True,
        vector_rows=profile.projection_vector_rows, gather_vector_rows=profile.gather_vector_rows)
    reference_plan = layers.build_batch(args.kind, 1, args.compiler, **kwargs)
    from compiler.assembler.assembly_to_binary import AssemblyToBinary

    original = args.fixtures / (
        "mamba_complete_bf16_candidate" if args.kind == "mamba" else "kda_complete_bf16_candidate_v2"
    )
    with gzip.open(original / "hbm_for_behave_sim.bin.gz", "rb") as f:
        initial = f.read()
    with gzip.open(original / "hbm_dump.bin.gz", "rb") as f:
        expected = f.read()
    assert len(initial) == len(expected) == reference_plan.arena.size
    lookup = {key: (base, size) for base, size, key, _, _ in reference_plan.arena.records}
    plan = layers.build_batch(args.kind, args.batch, args.compiler, **kwargs)
    folder = args.output.resolve()
    if (folder / "manifest.json").exists():
        raise FileExistsError("refusing to overwrite an executed or running experiment")
    folder.mkdir(parents=True, exist_ok=True)
    input_path = folder / "hbm_for_behave_sim.bin"
    with input_path.open("wb") as f:
        f.truncate(plan.arena.size)
        for base, size, key, _, _ in plan.arena.records:
            source, n = lookup[key]
            assert size == n
            f.seek(base)
            f.write(initial[source : source + n])
    del initial
    asm = folder / "program.asm"
    program = marked(plan)
    handoff_report = None
    if args.handoff:
        from compiler.aten.plena.projection_handoff import retain_vector_handoffs
        program, handoff_report = retain_vector_handoffs(program,
            projection_stages=[s.name for s in plan.stages if s.matrix_shape] if args.handoff_shared else ())
    asm.write_text(program)
    if handoff_report is not None:
        (folder / "handoff.json").write_text(json.dumps(handoff_report, indent=2))
    binary = folder / "program.mem"
    AssemblyToBinary(
        str(args.compiler / "doc/operation.svh"), str(args.compiler / "doc/configuration.svh")
    ).generate_binary(str(asm), str(binary))
    for name in ("fp_sram.bin", "int_sram.bin", "plena_settings.toml"):
        shutil.copyfile(original / name, folder / name)
    (folder / "matrix_profile.json").write_text(json.dumps(asdict(profile.matrix), indent=2))
    env = dict(
        os.environ,
        **profile.runtime_environment(),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        RUST_LOG="warn",
        PLENA_MATRIX_SERVICE_PROFILE=str(folder / "matrix_profile.json"),
    )
    command = [
        str(args.runtime.resolve()),
        "--opcode",
        str(binary),
        "--hbm",
        str(input_path),
        "--fpsram",
        str(folder / "fp_sram.bin"),
        "--intsram",
        str(folder / "int_sram.bin"),
        "--fpsram-bf16",
        "--settings",
        str(folder / "plena_settings.toml"),
        "--hbm-size",
        str(plan.arena.size),
        "--hbm-dump",
        str(folder / "hbm_dump.bin"),
        "--stage-profile-asm",
        str(asm),
        "--stage-profile-out",
        str(folder / "stage_profile.json"),
    ]
    manifest = dict(
        scope=__doc__,
        kind=args.kind,
        batch=args.batch,
        schedule=args.schedule, replay=args.replay, handoff=args.handoff, handoff_shared=args.handoff_shared,
        profile=asdict(profile),
        width=args.width,
        credits=args.credits,
        command=command,
        environment=profile.runtime_environment()
        | {k: env[k] for k in ("PLENA_NATIVE_READ_WIDTH", "PLENA_NATIVE_RESULT_SLOTS", "PLENA_MATRIX_SERVICE_PROFILE")},
        runtime_sha256=sha(args.runtime),
        inputs={
            p.name: sha(p) for p in (asm, binary, input_path, folder / "fp_sram.bin", folder / "plena_settings.toml")
        },
        fixture_input_sha256=sha(original / "hbm_for_behave_sim.bin.gz"),
        fixture_output_sha256=sha(original / "hbm_dump.bin.gz"),
    )
    manifest["fixture_directory"] = str(original.resolve())
    manifest["allocation_map"] = [
        dict(destination=base, bytes=size, source=lookup[key][0], unused_packing=unused, request=request)
        for base, size, key, unused, request in plan.arena.records
    ]
    allocations = sorted((base, base + size) for base, size, _, _, _ in plan.arena.records)
    assert all(end <= next_start for (_, end), (next_start, _) in pairwise(allocations))
    projection_weights = {(stage.weight_base, stage.weight_bytes) for stage in plan.stages if stage.matrix_shape}
    manifest["storage_validation"] = dict(
        disjoint_allocations=True,
        hbm_image_bytes=plan.arena.size,
        shared_projection_tensor_count=len(projection_weights),
        shared_projection_bytes=sum(size for _, size in projection_weights),
        request_count=batch_count if (batch_count := len({r[4] for r in plan.arena.records})) else 0,
        matrix_capacity_bytes=profile.matrix.matrix_capacity_bytes,
        vector_capacity_bytes=profile.matrix.vector_capacity_bytes,
        fixture_scope="repeated captured request; heterogeneous recurrent state checked separately",
    )
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(dict(status="running", kind=args.kind, batch=args.batch, hbm_bytes=plan.arena.size)), flush=True)
    with (folder / "run.log").open("w") as log:
        subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    checked = 0
    with (folder / "hbm_dump.bin").open("rb") as f:
        for base, size, key, unused, request in plan.arena.records:
            if unused:
                continue
            source, n = lookup[key]
            f.seek(base)
            actual = f.read(n)
            if actual != expected[source : source + n]:
                raise AssertionError(f"numerical mismatch request={request} base={base} key={key}")
            checked += n // 2
    t = json.loads((folder / "execution_timing.json").read_text())
    c = t["counters"]
    result = dict(
        status="passed",
        kind=args.kind,
        batch=args.batch,
        schedule=args.schedule, replay=args.replay, handoff=args.handoff, handoff_shared=args.handoff_shared,
        width=args.width,
        credits=args.credits,
        checked_bf16_values=checked,
        exact=True,
        calibration="not yet an analytical calibration",
        total_cycles=t["total_picos"] / t["period_picos"],
        issue=c["issue_cycles"],
        scalar=t["scalar_and_control_cycles"],
        sram=c["bank_service_cycles"],
        arithmetic=c["arithmetic_cycles"],
        dependency=c["dependency_cycles"],
        dma=t["dma_and_memory_wait_picos"] / t["period_picos"],
        hbm_read_bytes=t["hbm_read_bytes"],
        hbm_write_bytes=t["hbm_write_bytes"],
        output_sha256=sha(folder / "hbm_dump.bin"),
        manifest_sha256=sha(folder / "manifest.json"),
    )
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if getattr(args, "mapped_archive", False):
        # Lossless reconstruction against immutable historical fixtures. Check
        # the entire byte image, including holes, before discarding raw files.
        with gzip.open(original / "hbm_for_behave_sim.bin.gz", "rb") as stream:
            source_input = stream.read()
        mapped = {}
        for name, output in (("hbm_for_behave_sim.bin", False), ("hbm_dump.bin", True)):
            h = hashlib.sha256()
            cursor = 0
            for record in manifest["allocation_map"]:
                start, n, source = record["destination"], record["bytes"], record["source"]
                h.update(bytes(start - cursor))
                data = expected if output and not record["unused_packing"] else source_input
                h.update(memoryview(data)[source : source + n])
                cursor = start + n
            h.update(bytes(plan.arena.size - cursor))
            if h.hexdigest() != sha(folder / name):
                raise AssertionError("mapped archive roundtrip failed: " + name)
            mapped[name] = h.hexdigest()
        (folder / "mapped_archive.json").write_text(
            json.dumps(
                dict(
                    format="fixture-relocation-v1",
                    bytes=plan.arena.size,
                    sha256=mapped,
                    fixture_directory=str(original.resolve()),
                    source_input_sha256=manifest["fixture_input_sha256"],
                    source_output_sha256=manifest["fixture_output_sha256"],
                    segments=manifest["allocation_map"],
                    roundtrip_exact=True,
                ),
                indent=2,
            )
            + "\n"
        )
        for name in mapped:
            (folder / name).unlink()
        print(json.dumps(result), flush=True)
        return
    # Retain lossless evidence without keeping both compressed and raw images.
    for name in ("hbm_for_behave_sim.bin", "hbm_dump.bin"):
        p = folder / name
        with p.open("rb") as src, gzip.open(str(p) + ".gz", "wb", compresslevel=1) as dst:
            shutil.copyfileobj(src, dst, 4 * 1024**2)
        p.unlink()
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("compiler", "fixtures", "runtime", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--kind", choices=["mamba", "kda"], required=True)
    p.add_argument("--batch", type=int, choices=[1, 2, 4, 8, 16], default=1)
    p.add_argument("--schedule", choices=["resident", "compact", "batch"], default="batch")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--handoff", action="store_true")
    p.add_argument("--handoff-shared", action="store_true")
    p.add_argument("--width", type=int, choices=[256, 512, 2048], default=512)
    p.add_argument("--credits", type=int, choices=[1, 2, 4, 8], default=4)
    p.add_argument(
        "--mapped-archive",
        action="store_true",
        help="verify and retain fixture relocation maps instead of duplicate weight images",
    )
    run(p.parse_args())
