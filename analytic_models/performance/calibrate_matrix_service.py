"""Validate closed-form Matrix costs against an independent Rust tick oracle.

This is a reference-model gate, NOT full Matrix ISA or RTL calibration. Every
prediction is saved before the probe runs; measured cycles never feed the model.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .matrix_service import MatrixService, matrix_cost


def bf(value):
    x = np.ascontiguousarray(value, dtype=np.float32)
    u = x.view(np.uint32)
    return ((u + np.uint32(32767) + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def oracle(a, b, h):
    """Vectorized numerical oracle; independent of Rust PE clock/control code."""
    a, b = bf(a), bf(b)
    m, k = a.shape
    _, n = b.shape
    e, kl = h.edge, h.reduction_lanes
    rounded = bf if h.accumulator == "BF16" else lambda x: np.asarray(x, dtype=np.float32)
    answer = np.zeros((m, n), np.float32)
    for row in range(0, m, e):
        for col in range(0, n, e):
            tile = np.zeros((e, e), np.float32)
            for start in range(0, k, kl):
                aa, bb = np.zeros((e, kl), np.float32), np.zeros((kl, e), np.float32)
                ar, bc = a[row : row + e, start : start + kl], b[start : start + kl, col : col + e]
                aa[: ar.shape[0], : ar.shape[1]] = ar
                bb[: bc.shape[0], : bc.shape[1]] = bc
                aa, bb = aa.reshape(e, h.groups, e), bb.reshape(h.groups, e, e)
                partial = np.zeros((e, h.groups, e), np.float32)
                for reduction in range(e):
                    partial = rounded(partial + aa[:, :, reduction, None] * bb[None, :, reduction, :])
                while partial.shape[1] > 1:
                    partial = rounded(partial[:, ::2] + partial[:, 1::2])
                tile = rounded(tile + partial[:, 0])
            answer[row : row + e, col : col + e] = bf(tile)[: min(e, m - row), : min(e, n - col)]
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = MatrixService()
    cases = [
        ("definition", (4, 4, 1024), base, 0),
        ("tail_all_axes", (3, 7, 1031), base, 0),
        ("decode_b1", (1, 16, 2688), base, 0),
        ("decode_b2", (2, 16, 2688), base, 0),
        ("decode_b8", (8, 16, 2688), base, 0),
        ("decode_b16", (16, 16, 2688), base, 0),
        ("backpressure", (5, 5, 31), base, 9),
        (
            "port_limited",
            (3, 9, 19),
            replace(
                base,
                edge=2,
                reduction_lanes=32,
                matrix_read_elements=7,
                vector_read_elements=5,
                vector_write_elements=3,
            ),
            2,
        ),
        ("feedback_limited", (3, 9, 19), replace(base, edge=2, reduction_lanes=32, mac_latency=5), 0),
        ("launch_limited", (3, 9, 19), replace(base, edge=2, reduction_lanes=32, mac_latency=1, mac_ii=4), 0),
        ("fp32_partials", (5, 9, 1025), replace(base, accumulator="FP32"), 3),
    ]
    records = []
    rng = np.random.Generator(np.random.PCG64(20260922))
    for name, (m, n, k), hardware, stall in cases:
        path = args.output / name
        path.mkdir()
        a, b = rng.normal(0, 0.2, (m, k)).astype(np.float32), rng.normal(0, 0.2, (k, n)).astype(np.float32)
        # A nonzero input/answer ensures early-zero/first-nonzero termination
        # cannot masquerade as complete writeback. Inspect every final output.
        prediction = matrix_cost(m, n, k, hardware, write_stall=stall)
        (path / "prediction.json").write_text(json.dumps(prediction, indent=2) + "\n")
        request = dict(
            m=m, n=n, k=k, hardware=asdict(hardware), write_stall=stall, a=a.ravel().tolist(), b=b.ravel().tolist()
        )
        wire = json.dumps(request)
        completed = subprocess.run(
            [str(args.probe)], input=wire, text=True, capture_output=True, check=True, timeout=120
        )
        observed = json.loads(completed.stdout)
        ref = oracle(a, b, hardware)
        bits = ((ref.view(np.uint32) >> 16).astype(np.uint16)).ravel().tolist()
        if bits != observed.pop("output_bf16_bits"):
            raise AssertionError(f"{name}: final numerical outputs differ")
        errors = {key: abs(prediction[key] - value) for key, value in observed.items()}
        if any(errors.values()):
            raise AssertionError(f"{name}: component mismatch {errors}")
        true = bf(a).astype(np.float64) @ bf(b).astype(np.float64)
        rel = float(np.linalg.norm(ref - true) / max(np.linalg.norm(true), 1e-30))
        record = dict(
            name=name,
            shape=[m, n, k],
            hardware=asdict(hardware),
            write_stall=stall,
            prediction=prediction,
            observed=observed,
            component_errors=errors,
            final_output_bitwise=True,
            output_count=m * n,
            versus_fp64_rel_l2=rel,
            request_sha256=hashlib.sha256(wire.encode()).hexdigest(),
        )
        (path / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        records.append(record)
    report = dict(
        status="passed_reference_model_only",
        cases=records,
        held_out=len(cases) - 1,
        probe_sha256=hashlib.sha256(args.probe.read_bytes()).hexdigest(),
        seed=20260922,
        data="deterministic synthetic BF16 operands, not checkpoint layer weights",
        excluded=matrix_cost(1, 1, 1)["excluded"],
        acceptance="does not satisfy projection/full-layer evidence gate",
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(cases)} Matrix reference cases passed; ISA/RTL integration remains unverified")


if __name__ == "__main__":
    main()
