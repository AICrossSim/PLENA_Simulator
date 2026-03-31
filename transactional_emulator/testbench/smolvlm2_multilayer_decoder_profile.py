"""
SmolVLM2-256M Multi-Layer Decoder ASM Profile Generator

Generates ASM for N stacked decoder layers by tiling single-layer output.
SmolVLM2-256M has 30 text decoder layers (SmolLM2-135M backbone).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import subprocess
import os


def build_multilayer_asm(n_layers: int = 30) -> Path:
    """Run single-layer decoder test, tile body N times, write to build dir."""
    root = Path(__file__).parent.parent.parent
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True)
    asm_path = build_dir / "generated_asm_code.asm"

    # Step 1: Generate single-layer ASM by running existing decoder test
    print("Generating single-layer decoder ASM...")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}:{root}/tools:{env.get('PYTHONPATH', '')}"
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parent / "decoder_asm_gen.py"),
            "--seq-len",
            "64",
            "--hidden",
            "64",
            "--inter",
            "128",
            "--build-dir",
            str(build_dir),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError("Decoder ASM generation failed")
    print(result.stdout[-500:] if result.stdout else "(no stdout)")

    # Step 2: Read generated ASM
    asm_text = asm_path.read_text()

    # Step 3: Split into header + body
    # Header = everything before first "; === SECTION:" marker
    # Body = from first "; === SECTION:" to end
    lines = asm_text.splitlines(keepends=True)
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("; === SECTION:"):
            body_start = i
            break

    header = "".join(lines[:body_start])
    body = "".join(lines[body_start:])

    # Step 4: Tile body N times
    print(f"Tiling body {n_layers} times...")
    tiled_asm = header + (body * n_layers)

    # Step 5: Write tiled ASM
    asm_path.write_text(tiled_asm)
    print(f"Wrote {n_layers}-layer ASM to {asm_path}")
    print(f"  Header: {body_start} lines, Body: {len(lines) - body_start} lines")
    print(f"  Total tiled: {len(tiled_asm.splitlines())} lines")
    return asm_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=30, help="Number of decoder layers to simulate")
    parser.add_argument("--no-profile", action="store_true", help="Skip profiler, just generate ASM")
    args = parser.parse_args()

    asm_path = build_multilayer_asm(n_layers=args.layers)

    if not args.no_profile:
        print(f"\nRunning ASM profiler on {args.layers}-layer ASM...")
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
