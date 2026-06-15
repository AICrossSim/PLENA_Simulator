import os

import torch
from bitstring import BitArray
from memory_mapping.rand_gen import RandomMxfpTensorGenerator


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
    min_element_bytes=0,
):
    """
    Maps the quantized blocks and bias to binary memory file for fake HBM memory.
    Writes raw bytes instead of ASCII hex text.
    blocks: list of blocks, each block is a list of elements
    bias: list of biases

    Ensures overall data size (blocks + bias) is a multiple of 64 bytes.

    When ``min_element_bytes > 0``, the element region (blocks) is padded with
    0x00 up to that minimum BEFORE the scale region (bias) is appended.  This
    keeps the on-disk layout consistent with the tilelang codegen, which sets
    the per-tensor element footprint to
    ``max(logical_element_count, max(HBM_V_Writeback_Amount,
    HBM_V_Prefetch_Amount) * VLEN)`` so the simulator's H_STORE_V /
    H_PREFETCH_V iter loop (which always writes AMOUNT rows of VLEN elements
    per call) doesn't overlap the scale region.  See
    ``tilelang/plena/memory_planner/hbm_frame.py::_collect_vector_burst_buffers``
    in the TileLang PLENA backend for the cross-side contract.
    """

    if not os.path.exists(directory):
        os.makedirs(directory)

    output_file = os.path.join(directory, "hbm_for_behave_sim.bin")
    mode = "ab" if append else "wb"

    for row_idx, row in enumerate(blocks):
        _ = " ".join(f"0x{val:02X}" for val in row)

    hbm_row_elem_num = hbm_row_width // (element_width)
    hbm_row_bias_num = hbm_row_width // (bias_width)

    with open(output_file, mode) as f:
        # Track total bytes written
        total_bytes_written = 0
        blocks_bytes_written = 0
        bias_bytes_written = 0

        # Process blocks
        row_buffer = bytearray()

        for i, block in enumerate(blocks):
            hex_str = map_block_to_value(block, element_width)
            block_bytes = hex_to_bytes(hex_str)
            row_buffer.extend(block_bytes)

            # Write when row is full
            if len(row_buffer) >= hbm_row_elem_num:
                f.write(row_buffer[:hbm_row_elem_num])
                total_bytes_written += hbm_row_elem_num
                blocks_bytes_written += hbm_row_elem_num
                row_buffer = bytearray()  # Reset buffer after writing

        # Flush any remaining block data
        blocks_row_padding = 0
        if len(row_buffer) > 0:
            # Pad to row width
            blocks_row_padding = hbm_row_elem_num - len(row_buffer)
            row_buffer.extend(b"\x00" * blocks_row_padding)
            f.write(row_buffer)
            total_bytes_written += len(row_buffer)
            blocks_bytes_written += len(row_buffer)

        # Element-region padding for STORE_V_AMOUNT-aligned layouts.  When the
        # caller requested a minimum element-region byte count (typical when
        # tilelang's effective_batch > actual_batch), pad the element region
        # up to that minimum with 0x00 before writing scales.  The padded
        # bytes are written by iter 1..AMOUNT-1 of H_STORE_V but never read
        # back as data, so 0x00 is safe.
        element_burst_padding = 0
        if min_element_bytes > 0 and blocks_bytes_written < min_element_bytes:
            element_burst_padding = min_element_bytes - blocks_bytes_written
            f.write(b"\x00" * element_burst_padding)
            total_bytes_written += element_burst_padding
            blocks_bytes_written += element_burst_padding

        # Process bias
        row_buffer = bytearray()

        for i, b in enumerate(bias):
            hex_str = map_scale_to_value(b, bias_width)
            bias_bytes = hex_to_bytes(hex_str)
            row_buffer.extend(bias_bytes)

            # Write when row is full
            if len(row_buffer) >= hbm_row_bias_num:
                f.write(row_buffer[:hbm_row_bias_num])
                total_bytes_written += hbm_row_bias_num
                bias_bytes_written += hbm_row_bias_num
                row_buffer = bytearray()

        # # For Little Endian Purpose
        # if len(row_buffer) > 0:
        #     # Calculate padding needed
        #     padding_needed = hbm_row_bias_num - len(row_buffer)
        #     # Insert zeros at the beginning
        #     row_buffer = bytearray(b'\x00' * padding_needed) + row_buffer
        #     f.write(row_buffer)

        # For Big Endian Purpose
        bias_row_padding = 0
        if len(row_buffer) > 0:
            # Calculate padding needed
            bias_row_padding = hbm_row_bias_num - len(row_buffer)
            row_buffer.extend(b"\x00" * bias_row_padding)
            f.write(row_buffer)
            total_bytes_written += len(row_buffer)
            bias_bytes_written += len(row_buffer)

        # Scale-region burst padding (mirror of element_burst_padding above).
        # The tilelang planner sets per-tensor frame size to
        #   element_alloc + element_alloc / MXFP_GROUP_SIZE
        # where ``element_alloc = max(logical, PREFETCH_V_AMOUNT * VLEN)``
        # (see ``tilelang/plena/memory_planner/hbm_frame.py::plan_hbm_frame_layout``
        # line 347, ``scale_offset = element_alloc``).  Without padding the
        # scale region too, every "small 1D tensor" (e.g. rms_norm weight
        # ``(HD=1024,)`` on config_2 where ``PREFETCH_V_AMOUNT * VLEN = 8192``)
        # is short by ``(min_element_bytes - logical_bytes) / 8`` scale bytes,
        # so the NEXT tensor's frame slides forward and the kernel's
        # ``C_SET_ADDR_REG a0, ...`` lands mid-data.  Concretely for rms_norm
        # at config_2: InvHidden frame planner-expected at byte 18432 but
        # packer-actual at byte 17536 → prefetch reads ``InvHidden[896..]``
        # (all zeros) → ``f1 = sum = 0 → var = 0 → sqrt = 0 → 1/sqrt = inf``,
        # silently corrupting RMSNorm / LayerNorm output to ±inf.
        # MXFP_GROUP_SIZE = 8 (1 scale byte per 8 element bytes) is hardcoded
        # because every config we ship uses BLOCK = 8; if a future config
        # changes this, plumb the value through here from the caller's
        # ``data_config["block_size"]`` instead.
        scale_burst_padding = 0
        if min_element_bytes > 0:
            target_scale_bytes = blocks_bytes_written // 8
            if bias_bytes_written < target_scale_bytes:
                scale_burst_padding = target_scale_bytes - bias_bytes_written
                f.write(b"\x00" * scale_burst_padding)
                total_bytes_written += scale_burst_padding
                bias_bytes_written += scale_burst_padding

        print("\n  [Bias]")
        print(f"    Bytes written: {bias_bytes_written}")
        print(f"    Row padding added: {bias_row_padding} bytes")
        if scale_burst_padding:
            print(f"    Burst-aligned scale padding: {scale_burst_padding} bytes")

        # Ensure overall data size is a multiple of 64 bytes
        remainder = total_bytes_written % 64
        final_padding = 0
        if remainder != 0:
            final_padding = 64 - remainder
            f.write(b"\x00" * final_padding)
            total_bytes_written += final_padding

    # print_outputfile_contents(output_file)  # Muted for cleaner output


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
    rand_gen_high = RandomMxfpTensorGenerator(
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
    rand_gen_low = RandomMxfpTensorGenerator(
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
