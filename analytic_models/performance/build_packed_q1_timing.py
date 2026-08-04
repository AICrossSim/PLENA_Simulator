"""Build exact PackedKV q_len=1 compiler-count timing evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

try:
    from .decode_timing import RTL_SERIALIZED, TIMING_MODES
    from .packed_q1_timing import (
        COMPILER_TIMING_SOURCE_PATHS,
        RTL_TIMING_SOURCE_PATHS,
        PackedQ1TimingContract,
        PackedQ1TracePoint,
        _source_hashes,
    )
except ImportError:
    from decode_timing import RTL_SERIALIZED, TIMING_MODES
    from packed_q1_timing import (
        COMPILER_TIMING_SOURCE_PATHS,
        RTL_TIMING_SOURCE_PATHS,
        PackedQ1TimingContract,
        PackedQ1TracePoint,
        _source_hashes,
    )

SIMULATOR_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_COMPILER_ROOT = SIMULATOR_ROOT / "compiler"
DEFAULT_RTL_ROOT = SIMULATOR_ROOT.parent / "PLENA_RTL"
DEFAULT_LATENCY_LIBRARY = Path(__file__).resolve().parent / "customISA_lib.json"


def _compiler_modules(compiler_root: Path):
    root = str(compiler_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from compiler.assembler.assembly_to_binary import AssemblyToBinary
    from compiler.assembler.parser import parse_asm_file
    from compiler.aten.plena import PlenaCompiler

    return AssemblyToBinary, parse_asm_file, PlenaCompiler


def _compile_point(
    *,
    compiler_root: Path,
    cache_tokens: int,
    batch: int,
    mlen: int,
    blen: int,
    hlen: int,
    query_heads: int,
    kv_heads: int,
    element_bits: int,
    block_size: int,
    scale_bits: int,
) -> PackedQ1TracePoint:
    AssemblyToBinary, parse_asm_file, PlenaCompiler = _compiler_modules(
        compiler_root
    )
    rows_per_batch = mlen
    physical_rows = batch * rows_per_batch
    cache_rows_per_batch = ((cache_tokens + mlen - 1) // mlen) * mlen
    cache_physical_rows = batch * cache_rows_per_batch
    broadcast_amount = mlen // hlen
    group_heads = query_heads // kv_heads
    q_width = kv_heads * mlen

    compiler = PlenaCompiler(
        mlen=mlen,
        blen=blen,
        hbm_element_width=element_bits,
        hbm_block_size=block_size,
        hbm_scale_width=scale_bits,
    )
    compiler.hlen = hlen
    compiler.broadcast_amount = broadcast_amount
    query = compiler.alloc(
        "Q",
        batch,
        q_width,
        strict=False,
        physical_shape=(physical_rows, q_width),
    )
    output = compiler.alloc(
        "O",
        batch,
        q_width,
        strict=False,
        physical_shape=(physical_rows, q_width),
    )
    scratch = compiler.alloc(
        "S",
        mlen * broadcast_amount * 2,
        mlen,
        strict=True,
    )
    key = compiler.input(
        "K_packed",
        shape=(batch * cache_tokens, mlen),
        physical_shape=(cache_physical_rows, mlen),
        hbm_element_width=element_bits,
        hbm_block_size=block_size,
        hbm_scale_width=scale_bits,
        precision_role="key",
    )
    value = compiler.input(
        "V_packed",
        shape=(batch * cache_tokens, mlen),
        physical_shape=(cache_physical_rows, mlen),
        hbm_element_width=element_bits,
        hbm_block_size=block_size,
        hbm_scale_width=scale_bits,
        precision_role="value",
    )
    compiler.flash_attention_packed_cache(
        query,
        key,
        value,
        num_kv_heads=kv_heads,
        group_heads=group_heads,
        head_slot_dim=hlen,
        output_base_address=compiler.get_vram_addr(output.name),
        scratch_base_address=compiler.get_vram_addr(scratch.name),
        broadcast_amount=broadcast_amount,
        causal_mask=False,
        valid_cols=cache_tokens,
        cache_position=cache_tokens - 1,
        batch_size=batch,
        rows_per_batch=rows_per_batch,
        query_rows_per_batch=1,
        cache_rows_per_batch=cache_rows_per_batch,
    )
    artifact = compiler.compile_with_trace()
    assembly = artifact.assembly
    assembly_bytes = assembly.encode("utf-8")

    with tempfile.TemporaryDirectory() as temporary:
        assembly_path = Path(temporary) / "packed-q1.asm"
        assembly_path.write_text(assembly, encoding="utf-8")
        instructions = parse_asm_file(str(assembly_path))
    assembler = AssemblyToBinary(
        str(compiler_root / "doc" / "operation.svh"),
        str(compiler_root / "doc" / "configuration.svh"),
    )
    words = tuple(
        assembler._convert_to_binary(instruction)
        for instruction in instructions
    )
    machine_code = "".join(f"0x{word:08X}\n" for word in words).encode("ascii")
    # Machine code remains the deployment identity, while timing counts come
    # from the compiler's algebraic trace so compact hardware loops contribute
    # their full dynamic multiplicity.
    histogram = Counter(artifact.execution_trace.opcode_histogram)
    return PackedQ1TracePoint(
        cache_tokens=cache_tokens,
        opcode_histogram=tuple(sorted(histogram.items())),
        assembly_sha256=hashlib.sha256(assembly_bytes).hexdigest(),
        machine_code_sha256=hashlib.sha256(machine_code).hexdigest(),
    )


def build_contract(
    *,
    cache_tokens: Sequence[int],
    batch: int,
    mlen: int,
    blen: int,
    hlen: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    timing_mode: str = RTL_SERIALIZED,
    compiler_root: Path = DEFAULT_COMPILER_ROOT,
    rtl_root: Path = DEFAULT_RTL_ROOT,
    latency_library_path: Path = DEFAULT_LATENCY_LIBRARY,
    element_bits: int = 4,
    block_size: int = 8,
    scale_bits: int = 8,
) -> PackedQ1TimingContract:
    """Compile every requested cache point and seal its exact opcode histogram."""

    unique_cache_tokens = tuple(sorted(set(int(value) for value in cache_tokens)))
    if not unique_cache_tokens:
        raise ValueError("at least one cache length is required")
    if head_dim != hlen:
        raise ValueError("head_dim must equal HLEN")
    if query_heads <= 0:
        query_heads = kv_heads * (mlen // hlen)
    if query_heads % kv_heads:
        raise ValueError("query_heads must be divisible by kv_heads")
    if query_heads // kv_heads > mlen // hlen:
        raise ValueError("the GQA ratio exceeds the hardware broadcast width")
    points = tuple(
        _compile_point(
            compiler_root=compiler_root,
            cache_tokens=value,
            batch=batch,
            mlen=mlen,
            blen=blen,
            hlen=hlen,
            query_heads=query_heads,
            kv_heads=kv_heads,
            element_bits=element_bits,
            block_size=block_size,
            scale_bits=scale_bits,
        )
        for value in unique_cache_tokens
    )
    return PackedQ1TimingContract(
        timing_mode=timing_mode,
        mlen=mlen,
        blen=blen,
        hlen=hlen,
        query_heads=query_heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        batch=batch,
        points=points,
        compiler_source_hashes=_source_hashes(
            compiler_root,
            COMPILER_TIMING_SOURCE_PATHS,
        ),
        rtl_source_hashes=_source_hashes(
            rtl_root,
            RTL_TIMING_SOURCE_PATHS,
        ),
        latency_library_sha256=hashlib.sha256(
            latency_library_path.read_bytes()
        ).hexdigest(),
    )


def _write_atomic(path: Path, value: dict[str, object]) -> None:
    payload = json.dumps(
        value,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _cache_schedule(args: argparse.Namespace) -> tuple[int, ...]:
    if args.cache_tokens:
        return tuple(
            sorted(
                {
                    int(token)
                    for token in args.cache_tokens.split(",")
                    if token.strip()
                }
            )
        )
    if args.input_seq <= 0 or args.output_seq <= 0 or args.stride <= 0:
        raise ValueError("input-seq, output-seq, and stride must be positive")
    values = {
        args.input_seq + offset
        for offset in range(0, args.output_seq, args.stride)
    }
    values.add(args.input_seq + args.output_seq // 2)
    return tuple(sorted(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-tokens", default="")
    parser.add_argument("--input-seq", type=int, default=512)
    parser.add_argument("--output-seq", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--mlen", type=int, default=1024)
    parser.add_argument("--blen", type=int, default=8)
    parser.add_argument("--hlen", type=int, default=128)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--query-heads", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--element-bits", type=int, choices=(2, 4, 8), default=4)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--scale-bits", type=int, default=8)
    parser.add_argument("--timing-mode", choices=TIMING_MODES, default=RTL_SERIALIZED)
    parser.add_argument("--compiler-root", type=Path, default=DEFAULT_COMPILER_ROOT)
    parser.add_argument("--rtl-root", type=Path, default=DEFAULT_RTL_ROOT)
    parser.add_argument(
        "--latency-library",
        type=Path,
        default=DEFAULT_LATENCY_LIBRARY,
    )
    args = parser.parse_args()

    contract = build_contract(
        cache_tokens=_cache_schedule(args),
        batch=args.batch,
        mlen=args.mlen,
        blen=args.blen,
        hlen=args.hlen,
        query_heads=args.query_heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        timing_mode=args.timing_mode,
        compiler_root=args.compiler_root,
        rtl_root=args.rtl_root,
        latency_library_path=args.latency_library,
        element_bits=args.element_bits,
        block_size=args.block_size,
        scale_bits=args.scale_bits,
    )
    _write_atomic(args.output.resolve(), contract.to_dict())
    print(contract.contract_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
