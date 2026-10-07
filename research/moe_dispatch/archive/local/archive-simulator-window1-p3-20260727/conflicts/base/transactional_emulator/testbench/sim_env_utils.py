"""Build simulator memory artifacts from testbench tensor files."""

from __future__ import annotations

import logging
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compiler.assembler.assembly_to_binary import AssemblyToBinary
from memory_mapping.rand_gen import RandomMxfpTensorGenerator
from plena_utils.load_config import load_toml_config
from plena_utils.logger import get_logger

REPO_ROOT = Path(__file__).resolve().parents[2]

logger = get_logger("testbench")
logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# HBM binary writers (moved from tools/memory_mapping/memory_map.py)
# ---------------------------------------------------------------------------


def _pack_values_to_bytes(values, data_width):
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


def _pack_byte_aligned_values_to_bytes(values, data_width):
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


def _map_mx_byte_aligned(
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

    with open(output_file, mode) as f:
        for row_idx in range(logical_rows):
            row_start = row_idx * blocks_per_source_row
            if row_idx < source_rows:
                row_blocks = blocks_array[row_start : row_start + blocks_per_source_row]
                row_bytes = _pack_byte_aligned_values_to_bytes(row_blocks.reshape(-1), element_width)
            else:
                row_bytes = b""
            if len(row_bytes) > hbm_row_bytes:
                raise ValueError(f"Packed element row ({len(row_bytes)} B) > HBM row ({hbm_row_bytes} B)")
            row_bytes += b"\x00" * (hbm_row_bytes - len(row_bytes))
            f.write(row_bytes)

        for row_idx in range(logical_rows):
            row_start = row_idx * blocks_per_source_row
            if row_idx < source_rows:
                row_bias = bias_array[row_start : row_start + blocks_per_source_row]
                row_bytes = _pack_byte_aligned_values_to_bytes(row_bias, bias_width)
            else:
                row_bytes = b""
            if len(row_bytes) > scale_row_bytes:
                raise ValueError(f"Packed scale row ({len(row_bytes)} B) > scale row ({scale_row_bytes} B)")
            row_bytes += b"\x00" * (scale_row_bytes - len(row_bytes))
            f.write(row_bytes)

        # NOTE: no per-tensor pad to a 64-byte boundary. Each tensor is written at
        # its compiler-assigned hbm_addr (forward-padded above) and packed at its
        # exact element+scale byte length, matching the compiler's tight HBM base
        # allocation (advances by hbm_size, not rounded to 64). A per-tensor tail
        # pad to 64 would shift the NEXT tensor past its compiler base whenever a
        # tensor's size is not a 64-multiple (only happens at sub-64 MLEN, e.g.
        # X(16,16) = 288 B -> padded 320 -> next tensor read at the wrong addr ->
        # zero weights). The emulator's MemoryBacked capacity (hbm-size, a 64-byte
        # multiple sized to ~2x the preload) zero-covers the final aligned-block
        # read past the file end. At MLEN>=64 tensor sizes are already 64-multiples,
        # so dropping this pad is a no-op.


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
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, "hbm_for_behave_sim.bin")
    mode = "ab" if append else "wb"

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
        raise ValueError(f"source_row_elements ({source_row_elements}) > logical_row_elements ({logical_row_elements})")

    # Pack each element row tightly at its real byte width. The compiler
    # addresses HBM rows at the logical element stride (e.g. MLEN bytes for 8-bit
    # elements) and places scales at the tight element-region offset, so the
    # writer must match that stride rather than padding rows up to a 64-byte HBM
    # burst. The emulator reads aligned 64-byte words and extracts the real bytes
    # (transfer_mx_from_hbm), so a tight layout is read correctly. At MLEN>=64 the
    # element row is already >=64 bytes, so this equals the previous value
    # (max with the 64-byte burst floor was a no-op there).
    element_row_bits = logical_row_elements * element_width
    hbm_row_bytes = (element_row_bits + 7) // 8
    blocks_per_source_row = (source_row_elements + block_width - 1) // block_width
    blocks_per_logical_row = (logical_row_elements + block_width - 1) // block_width
    inferred_source_rows = (len(blocks) + blocks_per_source_row - 1) // blocks_per_source_row
    if source_rows is None:
        source_rows = inferred_source_rows
    if logical_rows is None:
        logical_rows = source_rows
    if source_rows > logical_rows:
        raise ValueError(f"source_rows ({source_rows}) > logical_rows ({logical_rows})")

    scale_row_bits = blocks_per_logical_row * bias_width
    scale_row_bytes = (scale_row_bits + 7) // 8

    if element_width % 8 == 0 and bias_width % 8 == 0:
        _map_mx_byte_aligned(
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

    total_bytes_written = 0
    with open(output_file, mode) as f:
        for row_idx in range(logical_rows):
            row_start = row_idx * blocks_per_source_row
            row_buffer = bytearray()
            if row_idx < source_rows:
                for block in blocks[row_start : row_start + blocks_per_source_row]:
                    row_buffer.extend(_pack_values_to_bytes(block, element_width))
            if len(row_buffer) > hbm_row_bytes:
                raise ValueError(f"Packed element row ({len(row_buffer)} B) > HBM row ({hbm_row_bytes} B)")
            row_buffer.extend(b"\x00" * (hbm_row_bytes - len(row_buffer)))
            f.write(row_buffer)
            total_bytes_written += len(row_buffer)

        for row_idx in range(logical_rows):
            row_start = row_idx * blocks_per_source_row
            row_buffer = bytearray()
            if row_idx < source_rows:
                for b in bias[row_start : row_start + blocks_per_source_row]:
                    row_buffer.extend(_pack_values_to_bytes([b], bias_width))
            if len(row_buffer) > scale_row_bytes:
                raise ValueError(f"Packed scale row ({len(row_buffer)} B) > scale row ({scale_row_bytes} B)")
            row_buffer.extend(b"\x00" * (scale_row_bytes - len(row_buffer)))
            f.write(row_buffer)
            total_bytes_written += len(row_buffer)

        remainder = total_bytes_written % 64
        if remainder != 0:
            f.write(b"\x00" * (64 - remainder))


def map_normal_data_to_hbm_for_behave_sim(data, data_width, directory, append=True, hbm_row_width=64):
    os.makedirs(directory, exist_ok=True)
    output_file = os.path.join(directory, "hbm_for_behave_sim.bin")
    mode = "ab" if append else "wb"
    hex_digits = data_width // 4
    (data_width + 7) // 8
    with open(output_file, mode) as f:
        row_buffer = bytearray()
        for element in data:
            hex_str = f"{element:0{hex_digits}X}"
            if len(hex_str) % 2 != 0:
                hex_str = "0" + hex_str
            row_buffer.extend(bytes.fromhex(hex_str))
            if len(row_buffer) >= hbm_row_width:
                f.write(row_buffer[:hbm_row_width])
                row_buffer = row_buffer[hbm_row_width:]
        if row_buffer:
            f.write(row_buffer)


class MemoryDataManager:
    """Collect MX and integer memory payloads for HBM setup."""

    def __init__(self) -> None:
        self.mx_entries = []
        self.int_entries = []

    def add_mx_file(
        self,
        filename,
        blocks,
        bias,
        quant_config,
        source_rows=None,
        storage_rows=None,
        source_row_elements=None,
        storage_row_elements=None,
        hbm_addr: int | None = None,
    ) -> None:
        self.mx_entries.append(
            {
                "filename": filename,
                "type": "mx",
                "blocks": blocks,
                "bias": bias,
                "quant_config": quant_config,
                "source_rows": source_rows,
                "storage_rows": storage_rows,
                "source_row_elements": source_row_elements,
                "storage_row_elements": storage_row_elements,
                "hbm_addr": hbm_addr,
            }
        )

    def add_int_file(self, filename, data) -> None:
        self.int_entries.append({"filename": filename, "type": "int", "data": data})

    def get_all_entries(self):
        return [*self.mx_entries, *self.int_entries]


def _mx_quant_config(precision_node, precision_settings):
    return {
        "exp_width": precision_node["ELEM"]["exponent"],
        "man_width": precision_node["ELEM"]["mantissa"],
        "exp_bias_width": precision_node["SCALE"]["exponent"],
        "block_size": [1, precision_node["block"]],
        "int_width": precision_settings["HBM_V_INT_TYPE"]["DATA_TYPE"]["width"],
        "skip_first_dim": False,
    }


def _precision_for_tensor(stem: str, precision_settings):
    if stem == "V" or stem.startswith("V_"):
        return precision_settings["HBM_M_KV_TYPE"]
    if stem == "K" or stem.startswith(("K_", "W_")):
        return precision_settings["HBM_M_WEIGHT_TYPE"]
    return precision_settings["HBM_V_ACT_TYPE"]


def create_mem_for_sim(
    data_size=256,
    mode="behave_sim",
    asm="attn",
    data=None,
    specified_data_order=None,
    build_path=None,
    input_tensors: Mapping[str, Any] | None = None,
    tensor_layouts: dict | None = None,
    hbm_addrs: dict[str, int] | None = None,
):
    plena_toml_path = os.environ.get("PLENA_SETTINGS_TOML", str(REPO_ROOT / "plena_settings.toml"))
    config_settings = load_toml_config(plena_toml_path, "CONFIG", mode="TRANSACTIONAL")
    precision_settings = load_toml_config(plena_toml_path, "PRECISION", mode="TRANSACTIONAL")

    if mode == "behave_sim":
        target_dir = (
            Path(build_path) if build_path is not None else REPO_ROOT / "transactional_emulator/testbench/build"
        )
        asm_file = target_dir / "generated_asm_code.asm"
    else:
        asm_file = REPO_ROOT / "test" / "Instr_Level_Benchmark" / f"{asm}.asm"
        target_dir = asm_file.parent

    init_mem(asm_file.parent)
    if tensor_layouts is None:
        tensor_layouts = _load_tensor_layouts(target_dir)

    data_config = {
        "tensor_size": [1, data_size],
        "block_size": [1, precision_settings["HBM_M_WEIGHT_TYPE"]["block"]],
    }
    quant_config = {
        "exp_width": precision_settings["HBM_V_ACT_TYPE"]["ELEM"]["exponent"],
        "man_width": precision_settings["HBM_V_ACT_TYPE"]["ELEM"]["mantissa"],
        "exp_bias_width": precision_settings["HBM_V_ACT_TYPE"]["SCALE"]["exponent"],
        "block_size": data_config["block_size"],
        "int_width": precision_settings["HBM_V_INT_TYPE"]["DATA_TYPE"]["width"],
        "skip_first_dim": False,
    }

    memory_data_manager = MemoryDataManager()
    if mode != "behave_sim":
        raw_data = RandomMxfpTensorGenerator(
            shape=tuple(data_config["tensor_size"]),
            quant_config=quant_config,
            config_settings=config_settings,
            directory=asm_file.parent,
            filename=Path(f"{asm}/fake_test_raw_data.pt"),
        )
        raw_data.tensor_gen()
        raw_tensor = raw_data.tensor_load()
        blocks, bias = raw_data.quantize_tensor(raw_tensor)
        memory_data_manager.add_mx_file("fake_test_raw_data.pt", blocks, bias, quant_config)
    else:
        if specified_data_order is None:
            specified_data_order = _load_data_order(target_dir)

        if input_tensors is not None:
            tensor_names = list(specified_data_order) if specified_data_order is not None else list(input_tensors)
            for name in tensor_names:
                if name not in input_tensors:
                    raise KeyError(f"Tensor {name!r} is in data_order but was not provided")

                tensor = materialize_tensor_spec(input_tensors[name])
                if name == "int":
                    memory_data_manager.add_int_file(f"{name}.pt", tensor)
                    continue

                file_quant_config = _mx_quant_config(
                    _precision_for_tensor(name, precision_settings), precision_settings
                )
                file_raw_data = RandomMxfpTensorGenerator(
                    shape=tuple(tensor.shape),
                    quant_config=file_quant_config,
                    config_settings=config_settings,
                    directory=asm_file.parent,
                    filename=Path(f"{name}.pt"),
                )
                blocks, bias = file_raw_data.quantize_tensor(tensor)
                layout = _layout_for_tensor(tensor_layouts, name, tensor)
                memory_data_manager.add_mx_file(
                    f"{name}.pt",
                    blocks,
                    bias,
                    file_quant_config,
                    source_rows=layout.get("source_rows"),
                    storage_rows=layout.get("storage_rows"),
                    source_row_elements=layout.get("source_row_elements"),
                    storage_row_elements=layout.get("storage_row_elements"),
                    hbm_addr=hbm_addrs.get(name) if hbm_addrs else None,
                )
        elif specified_data_order is not None:
            pt_files = [target_dir / f"{name}.pt" for name in specified_data_order]
        else:
            pt_files = list(target_dir.glob("*.pt")) + list(target_dir.glob("*.pth"))

        if input_tensors is None:
            for pt_file in pt_files:
                if pt_file.stem == "int":
                    memory_data_manager.add_int_file(pt_file.name, torch.load(pt_file))
                    continue

                file_quant_config = _mx_quant_config(
                    _precision_for_tensor(pt_file.stem, precision_settings), precision_settings
                )
                file_raw_data = RandomMxfpTensorGenerator(
                    shape=tuple(data_config["tensor_size"]),
                    quant_config=file_quant_config,
                    config_settings=config_settings,
                    directory=asm_file.parent,
                    filename=pt_file,
                )
                file_tensor = file_raw_data.tensor_load()
                blocks, bias = file_raw_data.quantize_tensor(file_tensor)
                layout = _layout_for_tensor(tensor_layouts, pt_file, file_tensor)
                memory_data_manager.add_mx_file(
                    pt_file.name,
                    blocks,
                    bias,
                    file_quant_config,
                    source_rows=layout.get("source_rows"),
                    storage_rows=layout.get("storage_rows"),
                    source_row_elements=layout.get("source_row_elements"),
                    storage_row_elements=layout.get("storage_row_elements"),
                )

    env_setup(
        memory_data_manager,
        asm_file.parent,
        data_config,
        quant_config,
        hbm_row_width=config_settings["HBM_WIDTH"]["value"],
        logical_row_elements=config_settings["MLEN"]["value"],
    )


def _load_tensor_layouts(target_dir: Path) -> dict:
    layout_path = target_dir / "tensor_layouts.json"
    if not layout_path.exists():
        return {}
    with layout_path.open() as f:
        return json.load(f)


def _load_data_order(target_dir: Path) -> list[str] | None:
    order_path = target_dir / "data_order.json"
    if not order_path.exists():
        return None
    with order_path.open() as f:
        order = json.load(f)
    if not isinstance(order, list) or not all(isinstance(name, str) for name in order):
        raise ValueError(f"{order_path} must contain a JSON list of tensor names")
    return order


def materialize_tensor_spec(value: Any) -> torch.Tensor:
    materialize = getattr(value, "materialize", None)
    if callable(materialize):
        value = materialize()
    if not torch.is_tensor(value):
        raise TypeError(f"Expected torch.Tensor input for memory image, got {type(value).__name__}")
    return value


def _layout_for_tensor(tensor_layouts: dict, tensor_name: Path | str, tensor) -> dict:
    if isinstance(tensor_name, Path):
        stem = tensor_name.stem
        name = tensor_name.name
    else:
        stem = str(tensor_name)
        name = f"{stem}.pt"
    layout = tensor_layouts.get(stem, tensor_layouts.get(name, {}))
    if not layout:
        return {}

    source_row_elements = layout.get("source_row_elements")
    if source_row_elements is None and hasattr(tensor, "shape") and len(tensor.shape) > 0:
        source_row_elements = int(tensor.shape[-1])
    source_rows = layout.get("source_rows")
    if source_rows is None:
        source_shape = layout.get("source_shape") or layout.get("logical_shape")
        if source_shape:
            source_rows = int(source_shape[0])
        elif hasattr(tensor, "shape") and len(tensor.shape) > 1:
            source_rows = int(tensor.shape[0])

    storage_row_elements = layout.get("storage_row_elements")
    if storage_row_elements is None:
        storage_shape = layout.get("storage_shape") or layout.get("physical_shape")
        if storage_shape:
            storage_row_elements = int(storage_shape[-1])
    storage_rows = layout.get("storage_rows")
    if storage_rows is None:
        storage_shape = layout.get("storage_shape") or layout.get("physical_shape")
        if storage_shape:
            storage_rows = int(storage_shape[0])

    out = {}
    if source_rows is not None:
        out["source_rows"] = int(source_rows)
    if storage_rows is not None:
        out["storage_rows"] = int(storage_rows)
    if source_row_elements is not None:
        out["source_row_elements"] = int(source_row_elements)
    if storage_row_elements is not None:
        out["storage_row_elements"] = int(storage_row_elements)
    return out


def env_setup(
    memory_data_manager,
    build_path: Path,
    data_config,
    quant_config,
    hbm_row_width=256,
    logical_row_elements=None,
) -> None:
    isa_file_path = REPO_ROOT / "PLENA_Compiler" / "doc" / "operation.svh"
    config_file_path = REPO_ROOT / "PLENA_Compiler" / "doc" / "configuration.svh"

    assembler = AssemblyToBinary(str(isa_file_path), str(config_file_path))
    assembler.generate_binary(build_path / "generated_asm_code.asm", build_path / "generated_machine_code.mem")

    for entry in memory_data_manager.get_all_entries():
        if entry["type"] == "mx":
            entry_quant_config = entry.get("quant_config", quant_config)
            map_mx_data_to_hbm_for_behave_sim(
                blocks=entry["blocks"],
                element_width=entry_quant_config["exp_width"] + entry_quant_config["man_width"] + 1,
                block_width=entry_quant_config["block_size"][1],
                bias=entry["bias"],
                bias_width=entry_quant_config["exp_bias_width"],
                directory=build_path,
                append=True,
                hbm_row_width=hbm_row_width,
                logical_row_elements=entry.get("storage_row_elements") or logical_row_elements,
                source_row_elements=entry.get("source_row_elements"),
                logical_rows=entry.get("storage_rows"),
                source_rows=entry.get("source_rows"),
                hbm_addr=entry.get("hbm_addr"),
            )
        elif entry["type"] == "int":
            map_normal_data_to_hbm_for_behave_sim(
                data=entry["data"],
                data_width=quant_config["int_width"],
                directory=build_path,
                append=True,
                hbm_row_width=hbm_row_width,
            )


def init_mem(build_path: Path) -> None:
    build_path.mkdir(parents=True, exist_ok=True)

    hbm_bin_file = build_path / "hbm_for_behave_sim.bin"
    if hbm_bin_file.exists():
        hbm_bin_file.unlink()

    hbm_element_file = build_path / "hbm_ele.mem"
    hbm_scale_file = build_path / "hbm_scale.mem"
    hbm_file_for_behave_sim = build_path / "hbm_for_behave_sim.mem"
    instr_file = build_path / "machine_code.mem"

    os.environ["HBM_ELEMENT_FILE"] = str(hbm_element_file)
    os.environ["HBM_SCALE_FILE"] = str(hbm_scale_file)
    os.environ["HBM_FOR_BEHAVE_SIM_FILE"] = str(hbm_file_for_behave_sim)
    os.environ["INSTR_FILE"] = str(instr_file)

    hbm_write_element_m_file = build_path / "hbm_write_m_ele.mem"
    hbm_write_element_v_file = build_path / "hbm_write_v_ele.mem"
    hbm_write_scale_m_file = build_path / "hbm_write_m_scale.mem"
    hbm_write_scale_v_file = build_path / "hbm_write_v_scale.mem"
    vector_mem_result_file = build_path / "vector_result.mem"

    hbm_write_element_m_file.touch()
    hbm_write_element_v_file.touch()
    hbm_write_scale_m_file.touch()
    hbm_write_scale_v_file.touch()
    vector_mem_result_file.touch()

    os.environ["VECTOR_MEM_RESULT_FILE"] = str(vector_mem_result_file)
    os.environ["FAKE_HBM_ELEMENT_WRITE_M_FILE"] = str(hbm_write_element_m_file)
    os.environ["FAKE_HBM_ELEMENT_WRITE_V_FILE"] = str(hbm_write_element_v_file)
    os.environ["FAKE_HBM_SCALE_WRITE_M_FILE"] = str(hbm_write_scale_m_file)
    os.environ["FAKE_HBM_SCALE_WRITE_V_FILE"] = str(hbm_write_scale_v_file)
    os.environ["ASM_FILE"] = str(instr_file)
