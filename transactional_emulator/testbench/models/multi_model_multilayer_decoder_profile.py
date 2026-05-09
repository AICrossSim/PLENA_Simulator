"""
Multi-Model Multi-Layer Decoder ASM Profile Generator

Tiles single-layer decoder ASM (from misc/decoder_asm_gen.py) into a
full inference trace for roofline profiling.

Two model archetypes are wired in:

| Model    | Layers (default) | Steps | LM head per step |
|----------|------------------|-------|------------------|
| smolvlm2 | 30               | 1     | no               |
| llada    | 32               | 64    | yes (full-seq)   |

LLaDA uses denoising inference: each of T steps re-runs the full N-layer
stack + LM head over the full sequence (no autoregressive decode).
SmolVLM2 (and any standard decoder) is just N layers, no per-step LM head.

Output layout:
    header + ((transformer_body * n_layers) + lm_head_body) * n_steps

For smolvlm2 lm_head_body == "" and n_steps == 1 → standard N-layer stack.
Add a new model by appending to MODEL_PRESETS.

Usage:
    python multi_model_multilayer_decoder_profile.py --model smolvlm2
    python multi_model_multilayer_decoder_profile.py --model llada --layers 32 --steps 64
    python multi_model_multilayer_decoder_profile.py --model smolvlm2 --no-profile
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Per-model defaults.  CLI args override.
MODEL_PRESETS = {
    "smolvlm2": {
        "layers": 30,
        "steps": 1,
        "with_lm_head": False,
        "asm_name": "smolvlm2_multilayer_decoder_profile",
    },
    "llada": {
        "layers": 32,
        "steps": 64,
        "with_lm_head": True,
        "asm_name": "llada_multilayer_decoder_profile",
    },
}


def _split_header_body(asm_text: str, marker: str = "; === SECTION:") -> tuple[str, str]:
    """Split ASM at the first marker line; lines before = header, from-marker = body."""
    lines = asm_text.splitlines(keepends=True)
    body_start = next((i for i, line in enumerate(lines) if line.strip().startswith(marker)), 0)
    return "".join(lines[:body_start]), "".join(lines[body_start:])


def _generate_decoder_asm(build_dir: Path, seq_len: int, hidden: int, inter_dim: int, env: dict) -> str:
    print("Generating single-layer decoder ASM...")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent.parent / "misc" / "decoder_asm_gen.py"),
            "--seq-len",
            str(seq_len),
            "--hidden",
            str(hidden),
            "--inter",
            str(inter_dim),
            "--build-dir",
            str(build_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError("decoder_asm_gen.py failed")
    return (build_dir / "generated_asm_code.asm").read_text()


def _generate_lm_head_asm(build_dir: Path, seq_len: int, hidden: int, vocab: int, env: dict) -> str:
    print(f"Generating LM head ASM (seq={seq_len}, hidden={hidden}, vocab={vocab})...")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "llada_lm_head_asm_gen.py"),
            "--seq-len",
            str(seq_len),
            "--hidden",
            str(hidden),
            "--vocab",
            str(vocab),
            "--build-dir",
            str(build_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError("llada_lm_head_asm_gen.py failed")
    _, body = _split_header_body((build_dir / "lm_head_asm.asm").read_text(), marker="; === SECTION: lm_head")
    return body


def build_multilayer_asm(
    model: str,
    n_layers: int,
    n_steps: int,
    with_lm_head: bool,
    asm_name: str,
    seq_len: int = 64,
    hidden: int = 64,
    inter_dim: int = 128,
    vocab: int = 128,
) -> Path:
    root = Path(__file__).parent.parent.parent
    build_dir = Path(__file__).parent / "build" / asm_name
    build_dir.mkdir(parents=True, exist_ok=True)
    asm_path = build_dir / "generated_asm_code.asm"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{root}/tools:{env.get('PYTHONPATH', '')}"

    decoder_asm = _generate_decoder_asm(build_dir, seq_len, hidden, inter_dim, env)
    header, transformer_body = _split_header_body(decoder_asm)

    lm_head_body = ""
    if with_lm_head:
        lm_head_body = _generate_lm_head_asm(build_dir, seq_len, hidden, vocab, env)

    print(
        f"\nTiling [{model}]: {n_layers} layers x {n_steps} step(s)" + (" + LM head per step" if with_lm_head else "")
    )
    one_step = (transformer_body * n_layers) + lm_head_body
    tiled_asm = header + (one_step * n_steps)

    asm_path.write_text(tiled_asm)
    total_lines = len(tiled_asm.splitlines())
    print(f"\nWrote {asm_name} ASM to {asm_path}")
    print(f"  Total: {total_lines} lines")
    return asm_path


def _run_profiler(asm_path: Path) -> None:
    root = Path(__file__).parent.parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{root}/tools:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [sys.executable, str(root / "analytic_models" / "roofline" / "asm_profiler.py"), str(asm_path)],
        env=env,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-500:])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Model Multi-Layer Decoder ASM Profile Generator")
    parser.add_argument("--model", choices=list(MODEL_PRESETS), required=True, help="Model preset")
    parser.add_argument("--layers", type=int, help="Override default layer count")
    parser.add_argument("--steps", type=int, help="Override default step count (LLaDA denoising steps)")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--inter", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=128, help="Vocabulary size (only used when LM head is on)")
    parser.add_argument("--no-profile", action="store_true", help="Skip profiler, just generate ASM")
    args = parser.parse_args()

    preset = MODEL_PRESETS[args.model]
    layers = args.layers if args.layers is not None else preset["layers"]
    steps = args.steps if args.steps is not None else preset["steps"]

    asm_path = build_multilayer_asm(
        model=args.model,
        n_layers=layers,
        n_steps=steps,
        with_lm_head=preset["with_lm_head"],
        asm_name=preset["asm_name"],
        seq_len=args.seq_len,
        hidden=args.hidden,
        inter_dim=args.inter,
        vocab=args.vocab,
    )

    if not args.no_profile:
        print(f"\nRunning ASM profiler on {args.model} ASM ({layers} layers x {steps} step(s))...")
        _run_profiler(asm_path)
