"""
LLaDA Multi-Layer Decoder ASM Profile Generator

Generates ASM for LLaDA inference: N transformer layers x T denoising steps + full-seq LM head.
LLaDA uses the same transformer body as a standard decoder (flash_attention_plena is already
bidirectional -- no causal mask). The LM head runs over all seq_len positions every step.

Structure: header + ((transformer_body x n_layers) + lm_head_body) x n_steps

Usage:
    python llada_multilayer_decoder_profile.py --layers 32 --steps 64
    python llada_multilayer_decoder_profile.py --layers 32 --steps 1 --no-profile
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import subprocess
import os
import argparse


def build_llada_asm(
    n_layers: int = 32, n_steps: int = 64, seq_len: int = 64, hidden: int = 64, inter_dim: int = 128, vocab: int = 128
) -> Path:
    """
    Build tiled LLaDA ASM:
        header + ((transformer_body x n_layers) + lm_head_body) x n_steps
    """
    root = Path(__file__).parent.parent.parent
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True)
    asm_path = build_dir / "generated_asm_code.asm"

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{root}/tools:{env.get('PYTHONPATH', '')}"

    # ---------------------------------------------------------------
    # Step 1: Generate single-layer decoder ASM
    # ---------------------------------------------------------------
    print("Generating single-layer decoder ASM...")
    # decoder_asm_gen.py generates decoder ISA directly (no weights, no simulator run).
    # LLaDA-8B uses identical transformer ops (RMSNorm+RoPE+Attention+FFN) — no model-specific changes needed.
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "decoder_asm_gen.py"),
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
    print(result.stdout[-500:] if result.stdout else "(no stdout)")

    # Read generated decoder ASM
    decoder_asm = asm_path.read_text()
    decoder_lines = decoder_asm.splitlines(keepends=True)

    # Split into header + body at first "; === SECTION:" marker
    body_start = 0
    for i, line in enumerate(decoder_lines):
        if line.strip().startswith("; === SECTION:"):
            body_start = i
            break

    header = "".join(decoder_lines[:body_start])
    transformer_body = "".join(decoder_lines[body_start:])

    print(f"Decoder ASM: {body_start} header lines, {len(decoder_lines) - body_start} body lines")

    # ---------------------------------------------------------------
    # Step 2: Generate LM head ASM
    # ---------------------------------------------------------------
    print(f"\nGenerating LM head ASM (seq={seq_len}, hidden={hidden}, vocab={vocab})...")
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
        raise RuntimeError("LM head ASM generation failed")
    print(result.stdout[-500:] if result.stdout else "(no stdout)")

    # Read LM head ASM and extract body (from "; === SECTION: lm_head" to end)
    lm_head_asm = (build_dir / "lm_head_asm.asm").read_text()
    lm_head_lines = lm_head_asm.splitlines(keepends=True)

    lm_head_body_start = 0
    for i, line in enumerate(lm_head_lines):
        if line.strip().startswith("; === SECTION: lm_head"):
            lm_head_body_start = i
            break

    lm_head_body = "".join(lm_head_lines[lm_head_body_start:])

    print(
        f"LM head ASM: {lm_head_body_start} header lines (skipped), "
        f"{len(lm_head_lines) - lm_head_body_start} body lines"
    )

    # ---------------------------------------------------------------
    # Step 3: Tile ASM for LLaDA inference
    # Structure: header + ((transformer_body x n_layers) + lm_head_body) x n_steps
    # ---------------------------------------------------------------
    print(f"\nTiling: {n_layers} layers x {n_steps} steps + LM head per step...")

    one_step = (transformer_body * n_layers) + lm_head_body
    tiled_asm = header + (one_step * n_steps)

    # Write tiled ASM
    asm_path.write_text(tiled_asm)

    # Summary
    total_lines = len(tiled_asm.splitlines())
    transformer_body_lines = len(transformer_body.splitlines())
    lm_head_body_lines = len(lm_head_body.splitlines())

    print(f"\nWrote LLaDA ASM to {asm_path}")
    print(f"  Header:           {body_start} lines")
    print(f"  Transformer body: {transformer_body_lines} lines/layer")
    print(f"  LM head body:     {lm_head_body_lines} lines/step")
    print(f"  Layers:           {n_layers}")
    print(f"  Denoising steps:  {n_steps}")
    print(f"  Total:            {total_lines} lines")

    return asm_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaDA Multi-Layer Decoder ASM Profile Generator")
    parser.add_argument("--layers", type=int, default=32, help="Number of transformer layers")
    parser.add_argument("--steps", type=int, default=64, help="Number of denoising steps")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden size")
    parser.add_argument("--inter", type=int, default=128, help="FFN intermediate dimension")
    parser.add_argument("--vocab", type=int, default=128, help="Vocabulary size (sliced)")
    parser.add_argument("--no-profile", action="store_true", help="Skip profiler, just generate ASM")
    args = parser.parse_args()

    asm_path = build_llada_asm(
        n_layers=args.layers,
        n_steps=args.steps,
        seq_len=args.seq_len,
        hidden=args.hidden,
        inter_dim=args.inter,
        vocab=args.vocab,
    )

    if not args.no_profile:
        print(f"\nRunning ASM profiler on LLaDA ASM ({args.layers} layers x {args.steps} steps)...")
        root = Path(__file__).parent.parent.parent
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{root}:{root}/tools:{env.get('PYTHONPATH', '')}"
        result = subprocess.run(
            [sys.executable, str(root / "analytic_models/roofline/asm_profiler.py"), str(asm_path)],
            env=env,
            capture_output=True,
            text=True,
        )
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr[-500:])
