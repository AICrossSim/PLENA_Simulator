"""Signal-relative agreement check for the decode testbench.

`decoder_decode_test.py` gates on `|err| <= atol + rtol*|golden|` with a fixed
`atol = 0.2`. That constant was chosen at MLEN=64 against the residual stream,
whose elements are order 1. It is not a property of the hardware, and it does
not follow the signal: attention output elements shrink as roughly `1/sqrt(kv)`
because softmax averages over more keys, so at kv=1024 the fixed atol is an
order of magnitude larger than the entire attention signal and the gate admits
any value at all for that stage.

The scaling rule used here is per-stage and derived from the arithmetic rather
than tuned. A dot product of length K in block-scaled MXFP8 accumulates a
relative error that is a property of the format and the reduction length, not of
the absolute value of the result, so the admissible absolute error must scale
with the RMS of the quantity being checked:

    tolerance(stage) = ABSOLUTE_FLOOR + RELATIVE_LIMIT * rms(golden_stage)

`ABSOLUTE_FLOOR` covers exactly-zero regions (zero padding) where a relative
bound is undefined. `RELATIVE_LIMIT` is the fraction of the signal's own RMS
that low-precision arithmetic may consume. Because the bound tracks the stage's
own RMS, one rule holds at every geometry and every cache length, which a fixed
atol cannot.

The bound is per stage. A stage at the end of a long chain also carries every
upstream stage's error, so its end-to-end residue is not a statement about its
own arithmetic. The tool therefore also reports *local* agreement for stages
whose input survives in the dump: the same reference recomputed from the
emulator's own input, which isolates the stage from what it inherits.

This tool reports agreement; it does not replace the testbench's gate.

It reads the VRAM dump left in the build directory by `decoder_decode_test.py`,
which holds one configuration at a time. The reference below draws `k_old` and
`v_old` at sizes derived from `kv_size` before it draws the RoPE tables, so the
whole tail of the random stream — and therefore Q itself — moves with the cache
length. Checking a dump against a golden built for a different `kv_size` reports
every stage as broken for a reason that has nothing to do with the hardware, so
the cache length is taken from the manifest the generator writes and a
contradicting `--kv-size` is refused rather than silently believed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_runner import (  # noqa: E402
    acquire_build_directory,
    acquire_emulator_execution,
    validate_emulator_run_receipt,
)
from runtime_paths import settings_path  # noqa: E402
from misc.decoder_decode_asm_gen import decode_geometry  # noqa: E402
from misc.decoder_decode_test import (  # noqa: E402
    gqa_sdpa_ref,
    rms_norm_ref,
    rope_ref,
)

BUILD = Path(__file__).resolve().parents[1] / "build" / "decoder_decode"

# Fraction of a stage's own RMS that low-precision arithmetic may consume.
RELATIVE_LIMIT = 0.25
# Absolute floor for zero-valued regions of the golden signal.
ABSOLUTE_FLOOR = 1e-3
# RMSNorm epsilon, matching the constant the testbench seeds into FP SRAM.
RMS_NORM_EPS = 1e-5


def vram_f32(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def vram_addresses(asm: str) -> dict[str, int]:
    return {
        name: int(addr)
        for name, addr in re.findall(
            r"; Allocate VRAM Matrix (\S+): .*? at VRAM\[(\d+)\]", asm
        )
    }


def agreement(golden: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """Signal-relative agreement between a golden stage and its measurement."""
    golden = golden.astype(np.float64).ravel()
    measured = measured.astype(np.float64).ravel()[: golden.size]
    rms = float(np.sqrt(np.mean(golden**2)))
    tolerance = ABSOLUTE_FLOOR + RELATIVE_LIMIT * rms
    error = np.abs(golden - measured)
    within = error <= tolerance
    spread = golden.std() * measured.std()
    correlation = (
        float(np.mean((golden - golden.mean()) * (measured - measured.mean())) / spread)
        if spread > 0
        else float("nan")
    )
    return {
        "rms": rms,
        "tolerance": tolerance,
        "match": 100.0 * float(within.mean()),
        "relative_error": float(error.mean() / rms) if rms > 0 else float("nan"),
        "correlation": correlation,
        "passed": bool(within.all()),
    }


def build_reference(kv_size: int, inter: int, vocab: int) -> dict[str, torch.Tensor]:
    """Recompute every decode stage with the testbench's inputs."""
    geometry = decode_geometry()
    mlen = geometry["mlen"]
    rows = geometry["batch"]
    heads = geometry["query_heads"]
    kv_heads = geometry["kv_heads"]
    head_dim = geometry["head_dim"]
    hidden = geometry["hidden"]
    inter = inter or 2 * mlen
    query_width = heads * head_dim
    kv_width = kv_heads * head_dim
    scale = 1.0 / math.sqrt(head_dim)
    torch.manual_seed(42)
    x = torch.randn(rows, hidden, dtype=torch.bfloat16) * 0.5
    w_q = torch.randn(hidden, query_width, dtype=torch.bfloat16) * 0.1
    w_k = torch.randn(hidden, kv_width, dtype=torch.bfloat16) * 0.1
    w_v = torch.randn(hidden, kv_width, dtype=torch.bfloat16) * 0.1
    w_o = torch.randn(query_width, hidden, dtype=torch.bfloat16) * 0.1
    k_old = torch.randn(kv_size - rows, kv_width, dtype=torch.bfloat16) * 0.5
    v_old = torch.randn(kv_size - rows, kv_width, dtype=torch.bfloat16) * 0.5
    w_gate = torch.randn(hidden, inter, dtype=torch.bfloat16) * 0.1
    w_up = torch.randn(hidden, inter, dtype=torch.bfloat16) * 0.1
    w_down = torch.randn(inter, hidden, dtype=torch.bfloat16) * 0.1
    cos = torch.rand(rows, hidden // 2, dtype=torch.bfloat16).repeat_interleave(2, 1)
    sin = torch.rand(rows, hidden // 2, dtype=torch.bfloat16).repeat_interleave(2, 1)
    # The LM head weight keeps the checkpoint's (vocab_size, hidden_size) layout.
    w_lm_head = torch.randn(vocab, hidden, dtype=torch.bfloat16) * 0.1

    normed = rms_norm_ref(x.float(), eps=RMS_NORM_EPS).to(torch.bfloat16)
    query = (normed.float() @ w_q.float()).to(torch.bfloat16)
    key = (normed.float() @ w_k.float()).to(torch.bfloat16)
    value = (normed.float() @ w_v.float()).to(torch.bfloat16)
    query = rope_ref(query, cos, sin)
    key = rope_ref(key, cos[:, :kv_width], sin[:, :kv_width])
    key_cache = torch.cat([k_old, key])
    value_cache = torch.cat([v_old, value])
    attention = gqa_sdpa_ref(
        query.reshape(rows, heads, head_dim).float(),
        key_cache.reshape(-1, kv_heads, head_dim).float(),
        value_cache.reshape(-1, kv_heads, head_dim).float(),
        scale, heads, kv_heads,
    ).to(torch.bfloat16)
    projected = (attention.float() @ w_o.float()).to(torch.bfloat16)
    residual = x.to(torch.bfloat16) + projected
    pre_ffn = rms_norm_ref(residual.float(), eps=RMS_NORM_EPS).to(torch.bfloat16)
    gate = (pre_ffn.float() @ w_gate.float()).to(torch.bfloat16)
    up = (pre_ffn.float() @ w_up.float()).to(torch.bfloat16)
    activated = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.bfloat16)
    final = residual + (activated.float() @ w_down.float()).to(torch.bfloat16)
    final_normed = rms_norm_ref(final.float(), eps=RMS_NORM_EPS).to(torch.bfloat16)
    logits = (final_normed.float() @ w_lm_head.float().T).to(torch.bfloat16)

    return {
        "geometry": geometry,
        "inter": inter,
        "Q": query,
        "K_new": key,
        "V_new": value,
        "O": attention,
        "ffn_residual": residual,
        "O_proj": final,
        "logits": logits,
        "lm_head_weight": w_lm_head,
        "ffn_weights": (w_gate, w_up, w_down),
    }


def run_manifest() -> dict:
    """Configuration the artifacts in the build directory were generated for."""
    path = BUILD / "decode_run_manifest.json"
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing: run misc/decoder_decode_test.py --kv-size N to "
            "generate a dump before checking it."
        )
    return json.loads(path.read_text())


def _run_check(args: argparse.Namespace) -> int:
    try:
        validate_emulator_run_receipt(BUILD, settings_file=settings_path())
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    manifest = run_manifest()
    if manifest.get("kv_head_reuse") is not False:
        raise SystemExit(
            "the signal checker requires a default-schedule dump; regenerate "
            "without --kv-head-reuse"
        )
    kv_size = manifest["kv_size"]
    inter = manifest["inter"]
    if args.kv_size and args.kv_size != kv_size:
        raise SystemExit(
            f"build directory holds kv_size={kv_size} but --kv-size={args.kv_size} "
            f"was requested. Regenerate with "
            f"misc/decoder_decode_test.py --kv-size {args.kv_size} first."
        )
    if args.inter and args.inter != inter:
        raise SystemExit(
            f"build directory holds inter={inter} but --inter={args.inter} was "
            f"requested. Regenerate with that --inter first."
        )

    vocab = manifest["vocab"]
    reference = build_reference(kv_size, inter, vocab)
    geometry = reference["geometry"]
    if geometry != manifest["geometry"]:
        raise SystemExit(
            "settings geometry has changed since the dump was generated:\n"
            f"  generated with {manifest['geometry']}\n"
            f"  checking with  {geometry}\n"
            "Regenerate with misc/decoder_decode_test.py."
        )
    mlen = geometry["mlen"]
    rows = geometry["batch"]
    widths = {
        "Q": geometry["query_heads"] * geometry["head_dim"],
        "K_new": geometry["kv_heads"] * geometry["head_dim"],
        "V_new": geometry["kv_heads"] * geometry["head_dim"],
        "O": geometry["query_heads"] * geometry["head_dim"],
        "ffn_residual": geometry["hidden"],
        "O_proj": geometry["hidden"],
        "logits": vocab,
    }

    address = vram_addresses((BUILD / "generated_asm_code.asm").read_text())
    memory = vram_f32(BUILD / "vram_dump.bin")

    print(
        f"kv={kv_size} MLEN={mlen} BLEN={geometry['blen']} "
        f"HLEN={geometry['hlen']} rows={rows} "
        f"tolerance = {ABSOLUTE_FLOOR} + {RELATIVE_LIMIT} * rms(stage)"
    )
    header = (
        f"{'stage':<16}{'rms':>10}{'tolerance':>11}{'match':>9}"
        f"{'mean|err|/rms':>15}{'corr':>8}  verdict"
    )
    print(header)
    print("-" * len(header))
    failures = []
    def read_stage(stage: str, width: int) -> np.ndarray:
        """A VRAM matrix wider than MLEN is stored as MLEN-wide column blocks,
        each a whole `rows`-row tile, so the blocks are concatenated back."""
        base = address[stage]
        blocks = [
            memory[start: start + rows * mlen].reshape(rows, mlen)
            for start in range(
                base, base + -(-width // mlen) * rows * mlen, rows * mlen
            )
        ]
        return np.concatenate(blocks, axis=1)[:, :width]

    for stage, width in widths.items():
        measured = read_stage(stage, width)
        result = agreement(reference[stage].float().numpy(), measured)
        verdict = "PASS" if result["passed"] else "FAIL"
        if not result["passed"]:
            failures.append(stage)
        print(
            f"{stage:<16}{result['rms']:>10.5f}{result['tolerance']:>11.5f}"
            f"{result['match']:>8.2f}%{result['relative_error']:>14.1%}"
            f"{result['correlation']:>8.3f}  {verdict}"
        )
    print("-" * len(header))

    def print_local(name: str, golden: np.ndarray, stage: str, width: int) -> None:
        local = agreement(golden, read_stage(stage, width))
        if not local["passed"]:
            failures.append(name)
        print(
            f"{name:<16}{local['rms']:>10.5f}{local['tolerance']:>11.5f}"
            f"{local['match']:>8.2f}%{local['relative_error']:>14.1%}"
            f"{local['correlation']:>8.3f}  "
            f"{'PASS' if local['passed'] else 'FAIL'}"
        )

    # Local agreement, stage by stage: each of these recomputes one stage from
    # the emulator's own measurement of the stage before it rather than from the
    # pure-PyTorch chain. It separates a stage's own arithmetic from the error
    # it inherits, which a stage at the end of a long chain is dominated by.
    w_gate, w_up, w_down = reference["ffn_weights"]
    emulator_residual = torch.tensor(
        read_stage("ffn_residual", geometry["hidden"])
    ).float()
    local_pre_ffn = rms_norm_ref(emulator_residual, eps=RMS_NORM_EPS).to(torch.bfloat16)
    local_gate = (local_pre_ffn.float() @ w_gate.float()).to(torch.bfloat16)
    local_up = (local_pre_ffn.float() @ w_up.float()).to(torch.bfloat16)
    local_activated = (
        torch.nn.functional.silu(local_gate.float()) * local_up.float()
    ).to(torch.bfloat16)
    local_o_proj = (
        emulator_residual.to(torch.bfloat16)
        + (local_activated.float() @ w_down.float()).to(torch.bfloat16)
    ).float().numpy()
    print_local("O_proj (local)", local_o_proj, "O_proj", geometry["hidden"])

    emulator_final = read_stage("O_proj", geometry["hidden"])
    local_logits = (
        rms_norm_ref(torch.tensor(emulator_final).float(), eps=RMS_NORM_EPS)
        .to(torch.bfloat16)
        .float()
        @ reference["lm_head_weight"].float().T
    ).to(torch.bfloat16).float().numpy()
    print_local("logits (local)", local_logits, "logits", vocab)
    print("-" * len(header))

    if failures:
        print(f"stages outside the signal-relative bound: {', '.join(failures)}")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kv-size", type=int, default=0,
        help="Cache length to check. Defaults to the generated configuration; "
             "a value disagreeing with it is refused.",
    )
    parser.add_argument("--inter", type=int, default=0)
    args = parser.parse_args()
    build_lease = acquire_build_directory(BUILD)
    try:
        execution_lease = acquire_emulator_execution()
        try:
            return _run_check(args)
        finally:
            execution_lease.release()
    finally:
        build_lease.release()


if __name__ == "__main__":
    raise SystemExit(main())
