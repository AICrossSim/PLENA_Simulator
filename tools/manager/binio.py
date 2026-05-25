"""Seek-based MX pack/unpack against the HBM bin.

The legacy packer (memory_map.map_mx_data_to_hbm_for_behave_sim) is append-only:
it streams tensors back-to-back and trusts that append order == the compiler's
address plan. Deleting/reordering one tensor mis-aligns everything downstream.

This module produces the *same bytes* but writes them at an explicit byte
offset via ``seek``, so a single tensor is independently overwritable.

Real per-tensor layout (measured empirically, see
reference_hbm_packer_real_layout):
    [elem bytes, contiguous]          # num_elements * elem_bytes
    [scale bytes, contiguous]         # (num_elements // block_size) * scale_bytes
    [pad to a multiple of 64 bytes]
i.e. packed = align64(elem_bytes + scale_bytes). Elem and scale regions are NOT
individually padded to hbm_row_width — they're concatenated, then the whole
tensor is padded to 64. (This is why the manager does not reuse the compiler's
_hbm_packed_byte_size, which pads each region separately.)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from .geometry import BehaviorSettings

# tools/manager/ -> PLENA_Simulator/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CHECK_MEM_PATH = _PROJECT_ROOT / "transactional_emulator" / "tools" / "check_mem.py"
_check_mem = None


def _load_check_mem():
    """Load check_mem.py by absolute path (it lives under transactional_emulator/
    tools/, which is not a package on sys.path)."""
    global _check_mem
    if _check_mem is None:
        spec = importlib.util.spec_from_file_location("plena_check_mem", _CHECK_MEM_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _check_mem = mod
    return _check_mem

# tools/ must be on sys.path for these (PYTHONPATH=tools, as the existing
# pipeline does). Imported lazily inside functions to keep import-time light.


def _align_up(n: int, mul: int) -> int:
    return ((n + mul - 1) // mul) * mul


def _row_unit(s: BehaviorSettings) -> int:
    """The byte granularity the real packer pads each region to.

    map_mx_data_to_hbm_for_behave_sim flushes/pads in chunks of
    ``hbm_row_width // element_width`` BYTES (element_width is in bits, =
    elem_bits; hbm_row_width is in bytes). So unit = hbm_row_width // elem_bits:
    8 at row_width=64, 64 at row_width=512. Measured empirically — verified
    byte-identical at both row widths. See [[reference_hbm_packer_real_layout]].
    """
    return s.hbm_row_width // s.elem_bits


def scale_region_offset(num_elements: int, s: BehaviorSettings) -> int:
    """Byte offset from a tensor's start to its scale region. The element
    region is padded up to ``_row_unit`` before the scale region begins."""
    elem_bytes = num_elements * s.elem_bytes
    return _align_up(elem_bytes, _row_unit(s))


def packed_byte_size(num_elements: int, s: BehaviorSettings) -> int:
    """Bytes a tensor occupies in the bin — matches the real packer.

    Real layout: [elem bytes padded to row_unit][scale bytes padded to
    row_unit][whole thing padded to 64]. elem and scale regions are each padded
    to row_unit = hbm_row_width//elem_bits, NOT contiguous and NOT each padded
    to a fixed 64 (that was a row_width=64 coincidence)."""
    unit = _row_unit(s)
    elem_bytes = _align_up(num_elements * s.elem_bytes, unit)
    num_scales = num_elements // s.block_size
    scale_bytes = _align_up(num_scales * s.scale_bytes, unit)
    return _align_up(elem_bytes + scale_bytes, 64)


def _quant_blocks_bias(tensor, s: BehaviorSettings):
    """MX-quantize ``tensor`` -> (elem_values_1d, scale_values_1d) as numpy
    uint8 arrays, VECTORIZED (no per-block Python loop).

    rand_gen.quantize_tensor loops over every block calling pack_fp_to_bin +
    .tolist() — ~40s for a 1M-elem tensor (131072 blocks). But pack_fp_to_bin
    is already vectorized over any shape, so we call _mx_fp_quantize_hardware +
    pack_fp_to_bin ONCE over the whole tensor. With E4M3 elem (8 bits -> 1 byte)
    and E8M0 scale (8 bits -> 1 byte) the packed ints ARE the bytes. Verified
    byte-identical to the per-block path (see _validate_quant_fast)."""
    import numpy as np
    from quant.quantizer.hardware_quantizer import _mx_fp_quantize_hardware
    from utils.torch_fp_conversion import pack_fp_to_bin

    flat = tensor.contiguous().reshape(1, -1)
    _, exp, man, scaling = _mx_fp_quantize_hardware(
        flat,
        width=s.elem_exp + s.elem_man + 1,
        exponent_width=s.elem_exp,
        exponent_bias_width=s.scale_exp,
        block_size=[1, s.block_size],
        skip_first_dim=False,
    )
    # exp/man: (num_blocks, block_size). Pack the WHOLE thing at once.
    packed = pack_fp_to_bin(exp, man, s.elem_exp, s.elem_man)  # int tensor, block layout
    # -> numpy uint8 without needing torch at module scope (packed/scaling are
    # torch tensors; .cpu().numpy() then cast)
    elem = packed.reshape(-1).cpu().numpy().astype(np.uint8)
    scale = scaling.reshape(-1).cpu().numpy().astype(np.uint8)
    return elem, scale


def _mx_payload(tensor, s: BehaviorSettings) -> bytes:
    """Quantize ``tensor`` to MX and serialise to the exact bin byte image
    (element region padded to row_unit, scale region padded to row_unit, whole
    padded to 64). Vectorized — byte-identical to the legacy per-block packer.
    """
    elem, scale = _quant_blocks_bias(tensor, s)  # uint8 arrays

    unit = _row_unit(s)
    buf = bytearray()
    # element region (1 byte/elem for E4M3), padded to row_unit
    buf.extend(elem.tobytes())
    buf.extend(b"\x00" * (_align_up(len(buf), unit) - len(buf)))
    # scale region (1 byte/scale for E8M0), padded to row_unit
    scale_start = len(buf)
    buf.extend(scale.tobytes())
    buf.extend(b"\x00" * (_align_up(len(buf) - scale_start, unit) - (len(buf) - scale_start)))
    # whole tensor padded to a multiple of 64 bytes
    buf.extend(b"\x00" * (_align_up(len(buf), 64) - len(buf)))
    return bytes(buf)


def _mx_payload_OLD_unused(tensor, s: BehaviorSettings) -> bytes:
    """(kept for reference) legacy per-block hex packer."""
    from memory_mapping.rand_gen import Random_MXFP_Tensor_Generator
    from memory_mapping.memory_map import (
        map_block_to_value, map_scale_to_value, hex_to_bytes,
    )
    quant_config = {
        "exp_width": s.elem_exp, "man_width": s.elem_man,
        "exp_bias_width": s.scale_exp, "block_size": [1, s.block_size],
        "int_width": 32, "skip_first_dim": False,
    }
    gen = Random_MXFP_Tensor_Generator(
        shape=tuple(tensor.shape), quant_config=quant_config,
        config_settings={}, directory=None, filename=None,
    )
    flat = tensor.contiguous().reshape(1, -1)
    blocks, bias = gen.quantize_tensor(flat)
    elem_width_bits = s.elem_bits
    scale_width_bits = s.scale_bits

    unit = _row_unit(s)
    buf = bytearray()
    for block in blocks:
        buf.extend(hex_to_bytes(map_block_to_value(block, elem_width_bits)))
    buf.extend(b"\x00" * (_align_up(len(buf), unit) - len(buf)))
    scale_start = len(buf)
    for sc in bias:
        buf.extend(hex_to_bytes(map_scale_to_value(sc, scale_width_bits)))
    buf.extend(b"\x00" * (_align_up(len(buf) - scale_start, unit) - (len(buf) - scale_start)))
    # whole tensor padded to a multiple of 64 bytes
    buf.extend(b"\x00" * (_align_up(len(buf), 64) - len(buf)))
    return bytes(buf)


def write_tensor(bin_path: str | Path, addr: int, tensor, s: BehaviorSettings) -> int:
    """Quantize ``tensor`` and seek-write its MX bytes at ``addr``.

    Single tensor independently overwritable; neighbouring bytes untouched.
    Returns the number of bytes written. The file must already be large enough
    (zero-filled) up to addr+packed_bytes; callers / the manager ensure that.
    """
    payload = _mx_payload(tensor, s)
    p = Path(bin_path)
    # 'r+b' keeps existing content; create empty if missing.
    if not p.exists():
        p.write_bytes(b"")
    with open(p, "r+b") as f:
        f.seek(int(addr))
        f.write(payload)
    return len(payload)


def read_tensor(bin_path: str | Path, addr: int, shape, s: BehaviorSettings):
    """Read a tensor back from the bin at ``addr`` and reshape.

    Wraps check_mem.read_hbm_bin_file_as_array (the same routine view_mem uses
    to extract a tensor from any bin offset). Returns a numpy array of FP32.
    """
    read_hbm_bin_file_as_array = _load_check_mem().read_hbm_bin_file_as_array

    num_elements = 1
    for d in shape:
        num_elements *= int(d)

    arr = read_hbm_bin_file_as_array(
        str(bin_path),
        exp_width=s.elem_exp,
        man_width=s.elem_man,
        start_byte_offset=int(addr),
        num_elements=num_elements,
        element_bytes=s.elem_bytes,
        scale_width=s.scale_bits,
        block_size=s.block_size,
        scale_offset=scale_region_offset(num_elements, s),
    )
    return arr.reshape(tuple(int(d) for d in shape))
