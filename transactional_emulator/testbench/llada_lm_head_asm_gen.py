"""
LLaDA LM Head ASM Generator

Generates PLENA ISA for the LM head linear projection:
    output = X @ W_lm_head    (seq_len, hidden) @ (hidden, vocab)

The generated ASM includes a "; === SECTION: lm_head" marker so the
multilayer profile script can extract the body.

Usage:
    python llada_lm_head_asm_gen.py --seq-len 64 --hidden 64 --vocab 128
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops
from plena_program import PLENAProgram


def generate_lm_head_asm(seq_len: int, hidden: int, vocab: int,
                         build_dir: str = "./build") -> str:
    """Generate LM head linear projection ASM and write to build_dir."""
    build_path = Path(build_dir)
    build_path.mkdir(parents=True, exist_ok=True)

    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)

    # Set up PLENA backend
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare inputs
    x_input = prog.input("X", shape=(seq_len, hidden))
    w_input = prog.input("W_lm_head", shape=(hidden, vocab))

    # Load activation into VRAM
    X_batch = prog.load_batch(x_input, name="X")

    # Inject section marker before the linear op
    prog._compiler.generated_code += "; === SECTION: lm_head\n"

    # LM head projection: (seq_len, hidden) @ (hidden, vocab)
    Y = ops.linear(prog, X_batch, w_input)

    # Compile to ISA
    gen_code = prog.compile()

    # Write to file
    asm_path = build_path / "lm_head_asm.asm"
    asm_path.write_text(gen_code)
    print(f"LM head ASM written to {asm_path}")
    print(f"  seq_len={seq_len}, hidden={hidden}, vocab={vocab}")
    print(f"  Lines: {len(gen_code.splitlines())}")

    return gen_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaDA LM Head ASM Generator")
    parser.add_argument("--seq-len", type=int, default=64, help="Sequence length")
    parser.add_argument("--hidden", type=int, default=64, help="Hidden size")
    parser.add_argument("--vocab", type=int, default=128, help="Vocabulary size (sliced)")
    parser.add_argument("--build-dir", type=str, default="./build", help="Build directory")
    args = parser.parse_args()

    generate_lm_head_asm(
        seq_len=args.seq_len,
        hidden=args.hidden,
        vocab=args.vocab,
        build_dir=args.build_dir,
    )
