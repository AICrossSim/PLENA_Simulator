try:
    from memory_mapping.rand_gen import Random_MXFP_Tensor_Generator
except ImportError:
    Random_MXFP_Tensor_Generator = None
from bitstring import BitArray
import torch
import os
import numpy as np


def print_outputfile_contents(output_file):
    print("\n--- File content hex dump: ---")
    if not os.path.exists(output_file):
        print(f"File {output_file} does not exist!")
        return
    with open(output_file, "rb") as f:
        data = f.read()
        # Print as 16 bytes per line, hex values
        for i in range(0, len(data), 16):
            chunk = data[i : i + 16]
            hex_bytes = " ".join(f"{b:02X}" for b in chunk)
            print(f"{i:08X}: {hex_bytes}")


def map_block_to_value(block, data_width):
    if data_width % 4 != 0:
        raise ValueError("data_width must be a multiple of 4 for hex representation.")

    hex_digits = data_width // 4  # e.g., 32 bits = 8 hex digits
    return "".join(f"{element:0{hex_digits}X}" for element in block)


def map_scale_to_value(scale, data_width):
    if data_width % 4 != 0:
        raise ValueError("data_width must be a multiple of 4 for hex representation.")

    hex_digits = data_width // 4  # e.g., 32 bits = 8 hex digits
    return f"{scale:0{hex_digits}X}"


def map_fp_data_to_fake_hbm(packed_input, element_width, path):
    assert len(packed_input.shape) == 2, "packed_input must be a 2D tensor"
    with open(os.path.join(path, "hbm.mem"), "a") as f:
        row = ""
        index_in_row = 0
        for i, vector in enumerate(packed_input):
            for j, element in enumerate(vector):
                row = row + BitArray(uint=int(element), length=element_width).hex
            f.write("0x" + row + "\n")
            row = ""
            index_in_row += 1


def hex_to_bytes(hex_str):
    """Convert hex string (with or without 0x prefix) to bytes"""
    hex_str = hex_str.strip()
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    # Ensure even length
    if len(hex_str) % 2 != 0:
        hex_str = "0" + hex_str
    return bytes.fromhex(hex_str)


def pack_values_to_bytes(values, data_width):
    """Pack fixed-width integer values into bytes, least-significant element first."""
    data = 0
    bits_left = 0
    out = bytearray()
    mask = (1 << data_width) - 1

    for value in values:
        data |= (int(value) & mask) << bits_left
        bits_left += data_width
        while bits_left >= 8:
            out.append(data & 0xFF)
            data >>= 8
            bits_left -= 8

    if bits_left > 0:
        out.append(data & 0xFF)

    return bytes(out)


def pack_byte_aligned_values_to_bytes(values, data_width):
    """Pack fixed-width integer values when each value is byte-aligned."""
    if data_width % 8 != 0:
        raise ValueError("data_width must be byte-aligned")
    arr = np.asarray(values)
    if arr.size == 0:
        return b""

    mask = (1 << data_width) - 1
    arr = np.bitwise_and(arr.astype(np.int64, copy=False), mask)
    if data_width == 8:
        return arr.astype(np.uint8, copy=False).tobytes()
    if data_width == 16:
        return arr.astype("<u2", copy=False).tobytes()
    if data_width == 32:
        return arr.astype("<u4", copy=False).tobytes()
    if data_width == 64:
        return arr.astype("<u8", copy=False).tobytes()

    item_bytes = data_width // 8
    out = bytearray(arr.size * item_bytes)
    offset = 0
    for value in arr.reshape(-1):
        out[offset : offset + item_bytes] = int(value).to_bytes(item_bytes, "little", signed=False)
        offset += item_bytes
    return bytes(out)


def map_data_to_fake_hbm_for_rtl_sim(
    blocks, element_width, block_width, bias, bias_width, directory, combined_blk_dim, append=True, hbm_row_width=64
):
    """
    Maps the quantized blocks and bias to two memory files as the fake HBM memory.
    """
    num_blocks_per_row = hbm_row_width // block_width
    num_bias_per_row = hbm_row_width // bias_width

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not append:
        # Clear existing files if not appending
        with open(os.path.join(directory, "hbm_ele.mem"), "w") as f:
            f.write("")
        with open(os.path.join(directory, "hbm_scale.mem"), "w") as f:
            f.write("")

    with open(os.path.join(directory, "hbm_ele.mem"), "a") as f:
        insert_block_row = ""
        combined_blk = ""
        index_in_row = 0
        # for i, block in enumerate(reversed(blocks)):
        for i, block in enumerate(blocks):
            combined_blk = combined_blk + map_block_to_value(block, element_width)
            if i % combined_blk_dim == combined_blk_dim - 1:
                insert_block_row = combined_blk + insert_block_row
                combined_blk = ""
            index_in_row += 1
            if index_in_row == num_blocks_per_row:
                f.write("0x" + insert_block_row + "\n")
                insert_block_row = ""
                index_in_row = 0
        if 0 < index_in_row < num_blocks_per_row:
            # If the last row is not full, pad it with zeros
            insert_block_row = "0" * (num_blocks_per_row - index_in_row) * (element_width // 4) + insert_block_row
            f.write("0x" + insert_block_row + "\n")

    # Save Bias to HBM file
    with open(os.path.join(directory, "hbm_scale.mem"), "a") as f:
        insert_bias_row = ""
        combined_bias = ""
        index_in_row = 0
        # for i, b in enumerate(reversed(bias)):
        for i, b in enumerate(bias):
            combined_bias = combined_bias + map_scale_to_value(b, bias_width)
            if i % combined_blk_dim == combined_blk_dim - 1:
                insert_bias_row = combined_bias + insert_bias_row
                combined_bias = ""
            index_in_row += 1
            if index_in_row == num_bias_per_row:
                f.write("0x" + insert_bias_row + "\n")
                insert_bias_row = ""
                index_in_row = 0
        if 0 < index_in_row < num_bias_per_row:
            # If the last row is not full, pad it with zeros
            insert_bias_row = "0" * (num_bias_per_row - index_in_row) * (bias_width // 4) + insert_bias_row
            f.write("0x" + insert_bias_row + "\n")


def map_mx_data_to_hbm_for_behave_sim(
    blocks,
    element_width,
    block_width,
    bias,
    bias_width,
    directory,
    append=True,
    hbm_row_width=64,
    logical_row_elements=None,
    source_row_elements=None,
    logical_rows=None,
    source_rows=None,
    hbm_addr: int | None = None,
):
    """
    Maps the quantized blocks and bias to binary memory file for fake HBM memory.
    Writes raw bytes instead of ASCII hex text.
    blocks: list of blocks, each block is a list of elements
    bias: list of biases

    Ensures overall data size (blocks + bias) is a multiple of 64 bytes.

    If hbm_addr is provided, pads the file with zeros to reach that byte offset
    before writing. This matches the compiler's HBM address allocation when
    _allocate_hbm inserts gaps for tile alignment.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    output_file = os.path.join(directory, "hbm_for_behave_sim.bin")
    mode = "ab" if append else "wb"

    # Pad to compiler-assigned HBM address if needed
    if hbm_addr is not None and append:
        try:
            current_pos = os.path.getsize(output_file)
        except OSError:
            current_pos = 0
        if hbm_addr > current_pos:
            with open(output_file, "ab") as pad_f:
                pad_f.write(b"\x00" * (hbm_addr - current_pos))

    if logical_row_elements is None:
        logical_row_elements = hbm_row_width // element_width
    if source_row_elements is None:
        source_row_elements = logical_row_elements
    if source_row_elements > logical_row_elements:
        raise ValueError(
            f"source_row_elements ({source_row_elements}) cannot exceed "
            f"logical_row_elements ({logical_row_elements})"
        )
    physical_hbm_row_bytes = (hbm_row_width + 7) // 8
    element_row_bits = logical_row_elements * element_width
    hbm_row_bytes = max(physical_hbm_row_bytes, (element_row_bits + 7) // 8)
    blocks_per_source_row = (source_row_elements + block_width - 1) // block_width
    blocks_per_logical_row = (logical_row_elements + block_width - 1) // block_width
    inferred_source_rows = (len(blocks) + blocks_per_source_row - 1) // blocks_per_source_row
    if source_rows is None:
        source_rows = inferred_source_rows
    if logical_rows is None:
        logical_rows = source_rows
    if source_rows > logical_rows:
        raise ValueError(f"source_rows ({source_rows}) cannot exceed logical_rows ({logical_rows})")

    scale_row_bits = blocks_per_logical_row * bias_width
    scale_row_bytes = (scale_row_bits + 7) // 8

    if element_width % 8 == 0 and bias_width % 8 == 0:
        _map_mx_byte_aligned_data_to_hbm_for_behave_sim(
            blocks=blocks,
            element_width=element_width,
            bias=bias,
            bias_width=bias_width,
            output_file=output_file,
            mode=mode,
            blocks_per_source_row=blocks_per_source_row,
            hbm_row_bytes=hbm_row_bytes,
            scale_row_bytes=scale_row_bytes,
            source_rows=source_rows,
            logical_rows=logical_rows,
        )
        return

    with open(output_file, mode) as f:
        # Track total bytes written
        total_bytes_written = 0
        blocks_bytes_written = 0
        bias_bytes_written = 0

        # Process blocks
        row_buffer = bytearray()

        for row_idx in range(logical_rows):
            row_start = row_idx * blocks_per_source_row
            row_buffer = bytearray()
            if row_idx < source_rows:
                for block in blocks[row_start : row_start + blocks_per_source_row]:
                    row_buffer.extend(pack_values_to_bytes(block, element_width))

            if len(row_buffer) > hbm_row_bytes:
                raise ValueError(
                    f"Packed element row ({len(row_buffer)} bytes) exceeds HBM row width ({hbm_row_bytes} bytes)"
                )
            row_buffer.extend(b"\x00" * (hbm_row_bytes - len(row_buffer)))
            f.write(row_buffer)
            total_bytes_written += len(row_buffer)
            blocks_bytes_written += len(row_buffer)

        # Process bias
        row_buffer = bytearray()

        for row_idx in range(logical_rows):
            row_start = row_idx * blocks_per_source_row
            row_buffer = bytearray()
            if row_idx < source_rows:
                for b in bias[row_start : row_start + blocks_per_source_row]:
                    row_buffer.extend(pack_values_to_bytes([b], bias_width))

            if len(row_buffer) > scale_row_bytes:
                raise ValueError(
                    f"Packed scale row ({len(row_buffer)} bytes) exceeds scale row width ({scale_row_bytes} bytes)"
                )
            row_buffer.extend(b"\x00" * (scale_row_bytes - len(row_buffer)))
            f.write(row_buffer)
            total_bytes_written += len(row_buffer)
            bias_bytes_written += len(row_buffer)

        print("\n  [Bias]")
        print(f"    Bytes written: {bias_bytes_written}")
        print("    Row padding added: 0 bytes")

        # Ensure overall data size is a multiple of 64 bytes
        remainder = total_bytes_written % 64
        final_padding = 0
        if remainder != 0:
            final_padding = 64 - remainder
            f.write(b"\x00" * final_padding)
            total_bytes_written += final_padding

    # print_outputfile_contents(output_file)  # Muted for cleaner output


def _map_mx_byte_aligned_data_to_hbm_for_behave_sim(
    *,
    blocks,
    element_width,
    bias,
    bias_width,
    output_file,
    mode,
    blocks_per_source_row,
    hbm_row_bytes,
    scale_row_bytes,
    source_rows,
    logical_rows,
):
    blocks_array = np.asarray(blocks)
    if blocks_array.ndim == 1:
        blocks_array = blocks_array.reshape(-1, 1)
    bias_array = np.asarray(bias).reshape(-1)

    total_bytes_written = 0
    blocks_bytes_written = 0
    bias_bytes_written = 0
    block_rows = blocks_array.shape[0]

    with open(output_file, mode) as f:
        for row_idx in range(logical_rows):
            row_start = row_idx * blocks_per_source_row
            if row_idx < source_rows:
                row_blocks = blocks_array[row_start : row_start + blocks_per_source_row]
                row_bytes = pack_byte_aligned_values_to_bytes(row_blocks.reshape(-1), element_width)
            else:
                row_bytes = b""
            if len(row_bytes) > hbm_row_bytes:
                raise ValueError(
                    f"Packed element row ({len(row_bytes)} bytes) exceeds HBM row width ({hbm_row_bytes} bytes)"
                )
            row_bytes += b"\x00" * (hbm_row_bytes - len(row_bytes))
            f.write(row_bytes)
            total_bytes_written += len(row_bytes)
            blocks_bytes_written += len(row_bytes)

        for row_idx in range(logical_rows):
            row_start = row_idx * blocks_per_source_row
            if row_idx < source_rows:
                row_bias = bias_array[row_start : row_start + blocks_per_source_row]
                row_bytes = pack_byte_aligned_values_to_bytes(row_bias, bias_width)
            else:
                row_bytes = b""
            if len(row_bytes) > scale_row_bytes:
                raise ValueError(
                    f"Packed scale row ({len(row_bytes)} bytes) exceeds scale row width ({scale_row_bytes} bytes)"
                )
            row_bytes += b"\x00" * (scale_row_bytes - len(row_bytes))
            f.write(row_bytes)
            total_bytes_written += len(row_bytes)
            bias_bytes_written += len(row_bytes)

        final_padding = 0
        remainder = total_bytes_written % 64
        if remainder != 0:
            final_padding = 64 - remainder
            f.write(b"\x00" * final_padding)
            total_bytes_written += final_padding

    print("\n  [Bias]")
    print(f"    Bytes written: {bias_bytes_written}")
    print("    Row padding added: 0 bytes")


def map_normal_data_to_hbm_for_behave_sim(data, data_width, directory, append=True, hbm_row_width=64):
    """
    Maps the normal data to binary memory file for fake HBM memory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    output_file = os.path.join(directory, "hbm_for_behave_sim.bin")
    mode = "ab" if append else "wb"
    with open(output_file, mode) as f:
        row_buffer = bytearray()
        for i, element in enumerate(data):
            hex_str = map_scale_to_value(element, data_width)
            data_bytes = hex_to_bytes(hex_str)
            row_buffer.extend(data_bytes)
            if len(row_buffer) >= hbm_row_width:
                f.write(row_buffer[:hbm_row_width])
                row_buffer = bytearray()
        if len(row_buffer) > 0:
            f.write(row_buffer)
    print_outputfile_contents(output_file)


if __name__ == "__main__":
    directory = "../../test/weight"
    fake_hbm_dir = "../../test/load_mem"
    filename = "test_projection_data.pt"
    torch.manual_seed(52)
    quant_config_high = {
        "exp_width": 1,
        "man_width": 2,
        "exp_bias_width": 8,
        "block_size": [1, 4],
        "skip_first_dim": False,
    }
    rand_gen_high = Random_MXFP_Tensor_Generator(
        shape=(16, 8), directory=directory, filename=filename, quant_config=quant_config_high
    )

    # Expect shape, blocks.shape = (32, 4), bias.shape = (32, 1)
    rand_gen_high.tensor_gen()
    weight = rand_gen_high.tensor_load()
    blocks, bias = rand_gen_high.quantize_tensor(weight[:8, :])
    map_data_to_fake_hbm_for_rtl_sim(
        blocks=blocks,
        element_width=quant_config_high["exp_width"] + quant_config_high["man_width"] + 1,
        block_width=(quant_config_high["exp_width"] + quant_config_high["man_width"] + 1) * 4,
        bias=bias,
        bias_width=quant_config_high["exp_bias_width"],
        combined_blk_dim=2,
        directory=fake_hbm_dir,
        append=False,
        hbm_row_width=256,
    )

    quant_config_low = {
        "exp_width": 4,
        "man_width": 3,
        "exp_bias_width": 8,
        "block_size": [1, 4],
        "skip_first_dim": False,
    }
    rand_gen_low = Random_MXFP_Tensor_Generator(
        shape=(8, 8), directory=directory, filename=filename, quant_config=quant_config_low
    )
    blocks, bias = rand_gen_low.quantize_tensor(weight[8:, :])
    map_data_to_fake_hbm_for_rtl_sim(
        blocks=blocks,
        element_width=quant_config_low["exp_width"] + quant_config_low["man_width"] + 1,
        block_width=(quant_config_low["exp_width"] + quant_config_low["man_width"] + 1) * 4,
        bias=bias,
        bias_width=quant_config_low["exp_bias_width"],
        combined_blk_dim=2,
        directory=fake_hbm_dir,
        hbm_row_width=256,
    )
