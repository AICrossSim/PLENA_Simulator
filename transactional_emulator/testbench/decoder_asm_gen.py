"""
Lightweight decoder layer ASM generator for profiling.

Generates a single transformer decoder layer ISA:
    embedding_add -> rms_norm -> rope -> flash_attention -> ffn -> rms_norm

Does NOT load model weights or run the simulator — ASM only.
This is a fast alternative to smollm2_135m_decoder_test.py for profiling use cases.
The instruction mix is architecturally identical to LLaDA-8B and LLaMA-3 decoders.

Usage:
    python decoder_asm_gen.py --seq-len 64 --hidden 64 --inter 128 --head-dim 64 --build-dir ./build
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import math

from plena.ops.registry import OpRegistry, Backend
import plena.ops as ops
from plena_program import PLENAProgram


def generate_decoder_asm(seq_len: int, hidden: int, inter: int, head_dim: int,
                         build_dir: str = "./build") -> str:
    """Generate a single decoder layer ASM and write to build_dir."""
    build_path = Path(build_dir)
    build_path.mkdir(parents=True, exist_ok=True)

    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    scale = 1.0 / math.sqrt(head_dim)

    # Set PLENA backend
    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    # Declare inputs — order determines HBM layout
    x_input     = prog.input("X",      shape=(seq_len, hidden))
    pos_input   = prog.input("POS",    shape=(seq_len, hidden))
    qrot_input  = prog.input("QROT",   shape=(seq_len, head_dim))
    cos_input   = prog.input("COS",    shape=(seq_len, head_dim))
    sin_input   = prog.input("SIN",    shape=(seq_len, head_dim))
    k_input     = prog.input("K",      shape=(seq_len, head_dim))
    v_input     = prog.input("V",      shape=(seq_len, head_dim))
    wgate_input = prog.input("W_gate", shape=(hidden, inter))
    wup_input   = prog.input("W_up",   shape=(hidden, inter))
    wdown_input = prog.input("W_down", shape=(inter, hidden))

    # Load activations to VRAM
    X_batch    = prog.load_batch(x_input,    name="X")
    POS_batch  = prog.load_batch(pos_input,  name="POS")
    Qrot_batch = prog.load_batch(qrot_input, name="QROT")
    Cos_batch  = prog.load_batch(cos_input,  name="COS")
    Sin_batch  = prog.load_batch(sin_input,  name="SIN")

    # Pipeline
    ops.embedding_add(prog, X_batch, POS_batch)                    # X += POS (in-place)
    prog.rms_norm(X_batch, eps_offset=3, reci_hid_offset=4)       # normalize (slots 3,4)
    ops.rope(prog, X_batch, Qrot_batch, Cos_batch, Sin_batch)     # RoPE (in-place)
    O = ops.flash_attention(prog, X_batch, k_input, v_input, scale)  # attention -> new O
    ops.ffn(prog, O, wgate_input, wup_input, wdown_input)          # ffn (in-place on O)
    prog.rms_norm(O, eps_offset=3, reci_hid_offset=4)             # final normalize (in-place)

    gen_code = prog.compile()
    lines = gen_code.splitlines()

    asm_path = build_path / "generated_asm_code.asm"
    asm_path.write_text(gen_code)

    print(f"Generated {len(lines)} lines of ISA code")
    print(f"Written to {asm_path}")

    return gen_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lightweight decoder layer ASM generator")
    parser.add_argument("--seq-len",  type=int, default=64,  help="Sequence length")
    parser.add_argument("--hidden",   type=int, default=64,  help="Hidden size")
    parser.add_argument("--inter",    type=int, default=128, help="FFN intermediate dimension")
    parser.add_argument("--head-dim", type=int, default=64,  help="Attention head dimension")
    parser.add_argument("--build-dir", type=str, default="./build", help="Build directory")
    args = parser.parse_args()

    generate_decoder_asm(
        seq_len=args.seq_len,
        hidden=args.hidden,
        inter=args.inter,
        head_dim=args.head_dim,
        build_dir=args.build_dir,
    )
