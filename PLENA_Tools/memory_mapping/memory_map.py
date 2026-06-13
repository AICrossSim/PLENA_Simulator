"""Memory mapping utilities for generating HBM files.

This module provides functions to convert quantized MXFP data into
memory files compatible with RTL simulation (.mem) or behavioral simulation (.bin).
"""

from pathlib import Path


def _map_block_to_hex(block: list[int], element_width: int) -> str:
    """Convert a block of elements to hex string.

    Elements are packed in little-endian order within the block:
    element[0] at LSB (rightmost), element[N-1] at MSB (leftmost).
    This matches the RTL memory layout where lower indices are at lower addresses.
    """
    if element_width % 4 != 0:
        raise ValueError("element_width must be a multiple of 4")
    hex_digits = element_width // 4
    # Reverse the block so element[0] ends up at LSB (rightmost in hex string)
    return "".join(f"{element:0{hex_digits}X}" for element in reversed(block))


def _map_scale_to_hex(scale: int, scale_width: int) -> str:
    """Convert a scale value to hex string."""
    if scale_width % 4 != 0:
        raise ValueError("scale_width must be a multiple of 4")
    hex_digits = scale_width // 4
    return f"{scale:0{hex_digits}X}"


def _hex_to_bytes(hex_str: str) -> bytes:
    """Convert hex string to bytes."""
    hex_str = hex_str.strip()
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    if len(hex_str) % 2 != 0:
        hex_str = "0" + hex_str
    return bytes.fromhex(hex_str)


def generate_hbm(
    blocks: list[list[int]],
    bias: list[int],
    element_width: int,
    bias_width: int,
    directory: str | Path,
    hbm_row_width: int = 256,
    mode: str = "rtl",
    instructions: list[int] | None = None,
    instr_storage_offset: int = 8192,
    tensor_data: list[tuple] | None = None,
) -> Path:
    """Generate HBM memory file from quantized MXFP data.

    Args:
        blocks: List of quantized element blocks, each block is a list of ints
            (used when tensor_data is None for backward compatibility)
        bias: List of scale/bias values
            (used when tensor_data is None for backward compatibility)
        element_width: Bit width of each element
        bias_width: Bit width of each scale value
        directory: Output directory path
        hbm_row_width: HBM row width in bits (default: 256)
        mode: "rtl" for .mem hex format, "sim" for .bin binary format
        instructions: Optional list of 32-bit instruction words
        instr_storage_offset: Byte offset for instruction storage (default: 8192)
        tensor_data: List of (blocks, bias) tuples, one per tensor. When provided,
            data is written in interleaved format: [tensor0_elements][tensor0_scales]
            [tensor1_elements][tensor1_scales]...

    Returns:
        Path to the generated file
    """
    if mode == "rtl":
        return _generate_hbm_mem(
            blocks,
            bias,
            element_width,
            bias_width,
            directory,
            hbm_row_width,
            instructions,
            instr_storage_offset,
            tensor_data,
        )
    elif mode == "sim":
        return _generate_hbm_bin(blocks, bias, element_width, bias_width, directory, hbm_row_width, tensor_data)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'rtl' or 'sim'.")


def _write_blocks_to_file(f, blocks, element_width, hbm_row_width):
    """Write element blocks to file, returns number of rows written."""
    if not blocks:
        return 0
    block_width = element_width * len(blocks[0])
    num_blocks_per_row = hbm_row_width // block_width if block_width > 0 else 1
    rows_written = 0
    row_hex = ""
    count = 0
    for block in blocks:
        row_hex = _map_block_to_hex(block, element_width) + row_hex
        count += 1
        if count >= num_blocks_per_row:
            f.write(f"0x{row_hex}\n")
            row_hex = ""
            count = 0
            rows_written += 1
    if row_hex:
        padding_bits = (num_blocks_per_row - count) * block_width
        row_hex = "0" * (padding_bits // 4) + row_hex
        f.write(f"0x{row_hex}\n")
        rows_written += 1
    return rows_written


def _write_scales_to_file(f, bias, bias_width, hbm_row_width):
    """Write scale values to file, returns number of rows written."""
    if not bias:
        return 0
    num_bias_per_row = hbm_row_width // bias_width if bias_width > 0 else 1
    rows_written = 0
    row_hex = ""
    count = 0
    for b in bias:
        row_hex = _map_scale_to_hex(b, bias_width) + row_hex
        count += 1
        if count >= num_bias_per_row:
            f.write(f"0x{row_hex}\n")
            row_hex = ""
            count = 0
            rows_written += 1
    if row_hex:
        padding_bits = (num_bias_per_row - count) * bias_width
        row_hex = "0" * (padding_bits // 4) + row_hex
        f.write(f"0x{row_hex}\n")
        rows_written += 1
    return rows_written


def _generate_hbm_mem(
    blocks: list[list[int]],
    bias: list[int],
    element_width: int,
    bias_width: int,
    directory: str | Path,
    hbm_row_width: int,
    instructions: list[int] | None = None,
    instr_storage_offset: int = 8192,
    tensor_data: list[tuple] | None = None,
) -> Path:
    """Generate HBM .mem file (hex text format for RTL simulation).

    Format (when tensor_data is provided - interleaved):
        0xTENSOR0_DATA_ROW_0
        ...
        0xTENSOR0_SCALE_ROW_0
        ...
        0xTENSOR1_DATA_ROW_0
        ...
        0xTENSOR1_SCALE_ROW_0
        ...
        0xINSTR_0
        ...

    Format (legacy - when tensor_data is None):
        0xDATA_ROW_0
        ...
        0xSCALE_ROW_0
        ...
        0xINSTR_0
        ...
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output_file = directory / "hbm.mem"

    with open(output_file, "w") as f:
        if tensor_data is not None:
            # Interleaved format: for each tensor, write elements then scales
            for tensor_blocks, tensor_bias in tensor_data:
                _write_blocks_to_file(f, tensor_blocks, element_width, hbm_row_width)
                _write_scales_to_file(f, tensor_bias, bias_width, hbm_row_width)
        else:
            # Legacy format: all elements first, then all scales
            _write_blocks_to_file(f, blocks, element_width, hbm_row_width)
            _write_scales_to_file(f, bias, bias_width, hbm_row_width)

        # Write instruction section (32-bit instructions packed into rows)
        if instructions:
            instr_width = 32  # Each instruction is 32 bits
            num_instr_per_row = hbm_row_width // instr_width
            row_hex = ""
            count = 0
            for instr in instructions:
                row_hex = f"{instr:08X}" + row_hex
                count += 1
                if count >= num_instr_per_row:
                    f.write(f"0x{row_hex}\n")
                    row_hex = ""
                    count = 0
            if row_hex:
                # Pad remaining row with zeros
                padding_bits = (num_instr_per_row - count) * instr_width
                row_hex = "0" * (padding_bits // 4) + row_hex
                f.write(f"0x{row_hex}\n")

    return output_file


def _build_blocks_bytes(blocks, element_width, bytes_per_row):
    """Build byte array from element blocks.

    Matches the byte layout of _write_blocks_to_file for little-endian memory:
    - _write_blocks_to_file PREPENDS blocks: final hex = H3 + H2 + H1 + H0
    - In .mem, block0 is at the RIGHT (low address in LE)
    - For LE memory: blocks stay in order 0,1,2,3, but each block's bytes reverse

    Within each block:
    - _map_block_to_hex puts element[0] at the rightmost position (LSB)
    - _hex_to_bytes reads left-to-right, producing [MSB,...,LSB]
    - For LE, we reverse each block so element[0] is at lowest address
    """
    data = bytearray()
    row_buffer = bytearray()
    for block in blocks:
        hex_str = _map_block_to_hex(block, element_width)
        block_bytes = _hex_to_bytes(hex_str)
        # Reverse each block for little-endian: element[0] at lowest address
        row_buffer.extend(block_bytes[::-1])
        if len(row_buffer) >= bytes_per_row:
            row_bytes = row_buffer[:bytes_per_row]
            data.extend(row_bytes)
            row_buffer = row_buffer[bytes_per_row:]
    if row_buffer:
        padding = bytes_per_row - len(row_buffer)
        row_buffer.extend(b"\x00" * padding)
        data.extend(row_buffer)
    return data


def _build_scales_bytes(bias, bias_width, bytes_per_row):
    """Build byte array from scale values.

    Matches the byte layout of _write_scales_to_file for little-endian memory:
    - _write_scales_to_file PREPENDS scales: final hex = S_n + ... + S_1 + S_0
    - In .mem, scale[0] is at the RIGHT (low address in LE)
    - For LE memory: scales stay in order, but each scale's bytes reverse

    For 8-bit scales, there's only 1 byte, so reversal is identity.
    """
    data = bytearray()
    row_buffer = bytearray()
    for b in bias:
        hex_str = _map_scale_to_hex(b, bias_width)
        bias_bytes = _hex_to_bytes(hex_str)
        # Reverse each scale for little-endian (no-op for 8-bit scales)
        row_buffer.extend(bias_bytes[::-1])
        if len(row_buffer) >= bytes_per_row:
            row_bytes = row_buffer[:bytes_per_row]
            data.extend(row_bytes)
            row_buffer = row_buffer[bytes_per_row:]
    if row_buffer:
        padding = bytes_per_row - len(row_buffer)
        row_buffer.extend(b"\x00" * padding)
        data.extend(row_buffer)
    return data


def _generate_hbm_bin(
    blocks: list[list[int]],
    bias: list[int],
    element_width: int,
    bias_width: int,
    directory: str | Path,
    hbm_row_width: int,
    tensor_data: list[tuple] | None = None,
) -> Path:
    """Generate HBM .bin file (binary format for behavioral simulation).

    Format (interleaved when tensor_data is provided):
        - No header for interleaved format
        - [tensor0_elements][tensor0_scales][tensor1_elements][tensor1_scales]...

    Format (legacy when tensor_data is None):
        - 8-byte header containing scale data byte offset (little-endian)
        - Element data (packed bytes)
        - Scale data (packed bytes)
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    output_file = directory / "hbm_for_behave_sim.bin"

    bytes_per_row = hbm_row_width // 8

    with open(output_file, "wb") as f:
        if tensor_data is not None:
            # Interleaved format: for each tensor, write elements then scales
            for tensor_blocks, tensor_bias in tensor_data:
                element_data = _build_blocks_bytes(tensor_blocks, element_width, bytes_per_row)
                scale_data = _build_scales_bytes(tensor_bias, bias_width, bytes_per_row)
                f.write(element_data)
                f.write(scale_data)
        else:
            # Legacy format: header + all elements + all scales
            element_data = _build_blocks_bytes(blocks, element_width, bytes_per_row)
            scale_data = _build_scales_bytes(bias, bias_width, bytes_per_row)
            # Scale offset = 8 (header) + element data size
            scale_offset = 8 + len(element_data)
            header = scale_offset.to_bytes(8, byteorder="little")
            f.write(header)
            f.write(element_data)
            f.write(scale_data)

    return output_file
