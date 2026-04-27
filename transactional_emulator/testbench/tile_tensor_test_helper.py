from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from compiler.asm_templates import preload_act_asm, preload_addr_reg_asm, reset_reg_asm

from tile_tensor_program import (
    Input,
    TileTensorProgram,
    _logical_shape_to_hbm_stride,
    _logical_shape_to_physical_shape,
    _tile_coord_to_hbm_offset,
)


def build_input_feed(
    prog: TileTensorProgram,
    full_tensors: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """Build create_sim_env-compatible HBM input feed from logical tensors."""
    input_feed: Dict[str, Any] = {}
    input_order: List[str] = []
    for tensor_name, data in full_tensors.items():
        input_obj = prog.tensor_manager.inputs.get(tensor_name)
        if input_obj is None:
            raise KeyError(f"Unknown tensor: {tensor_name}")
        rows, cols = _logical_shape_to_physical_shape(input_obj.logical_shape)
        logical_shape = tuple(input_obj.logical_shape)
        logical_layout = input_obj.metadata.get("logical_layout")
        if logical_layout is None:
            logical_layout = "bshd" if len(logical_shape) == 4 else "2d"

        if logical_layout == "bshd":
            data_shape = tuple(data.shape)
            b, s, h, d = logical_shape
            logical_flat_shape = (b * s, h * d)
            flat_shape = (rows, cols)
            canonical_shape = (b, s, h, d)
            shd_shape = (s, h, d) if b == 1 else None
            if data_shape == flat_shape:
                data = data.contiguous()
            elif data_shape == logical_flat_shape:
                padded = data.new_zeros(flat_shape)
                padded[: logical_flat_shape[0], : logical_flat_shape[1]] = data.contiguous()
                data = padded
            elif data_shape == canonical_shape:
                flat = data.contiguous().reshape(logical_flat_shape)
                padded = flat.new_zeros(flat_shape)
                padded[: logical_flat_shape[0], : logical_flat_shape[1]] = flat
                data = padded
            elif shd_shape is not None and data_shape == shd_shape:
                flat = data.contiguous().reshape(logical_flat_shape)
                padded = flat.new_zeros(flat_shape)
                padded[: logical_flat_shape[0], : logical_flat_shape[1]] = flat
                data = padded
            elif b == 1 and data_shape == (h, d, s):
                flat = data.permute(2, 0, 1).contiguous().reshape(logical_flat_shape)
                padded = flat.new_zeros(flat_shape)
                padded[: logical_flat_shape[0], : logical_flat_shape[1]] = flat
                data = padded
            else:
                raise ValueError(
                    f"Input shape mismatch for {tensor_name}: got {data_shape}, "
                    f"expected one of {flat_shape} / {logical_flat_shape} / {canonical_shape}"
                    + (f" / {shd_shape} / {(h, d, s)}" if b == 1 else "")
                )
        elif logical_layout == "shd_dup_scatter_rows":
            scatter_rows = int(input_obj.metadata.get("scatter_rows", 0))
            if scatter_rows <= 0 or prog.mlen % scatter_rows != 0:
                raise ValueError(f"Invalid scatter_rows metadata for {tensor_name}: {scatter_rows}")
            data_shape = tuple(data.shape)
            if data_shape == (rows, cols):
                data = data.contiguous()
            elif data_shape == (scatter_rows, prog.mlen):
                repeat_n = prog.mlen // scatter_rows
                data = data.repeat(repeat_n, 1).contiguous()
            else:
                raise ValueError(
                    f"Input shape mismatch for {tensor_name}: got {data_shape}, "
                    f"expected {(rows, cols)} or {(scatter_rows, prog.mlen)}"
                )
        else:
            if tuple(data.shape) != (rows, cols):
                raise ValueError(
                    f"Input shape mismatch for {tensor_name}: got {tuple(data.shape)}, expected {(rows, cols)}"
                )

        group_obj = input_obj.metadata.get("hbm_group_obj", f"{tensor_name}.hbm")
        if not isinstance(data, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor for input '{tensor_name}', got {type(data)}")
        input_feed[group_obj] = data.contiguous().reshape(1, -1)
        input_order.append(group_obj)
    return input_feed, input_order


def stage_input_tensor_for_stride_compare(
    tensor: Input,
) -> Dict[str, int | bool]:
    """Build tail ISA to load one input/output tensor from HBM into VRAM[0..] for compare."""
    prog = tensor.program
    rows, cols = _logical_shape_to_physical_shape(tensor.logical_shape)
    col_blocks = (cols + prog.mlen - 1) // prog.mlen
    tile_elems = prog.mlen * prog.mlen
    staging_code = [f"; compare staging input tensor {tensor.name} from HBM to VRAM[0]"]
    vram_addr = 0
    addr_reg = 0
    addr_gp = [1]
    preload_gp = [2, 3, 4, 5, 6]

    row_blocks = (rows + prog.mlen - 1) // prog.mlen
    # Stage tiles in col-block-major order so the flattened VRAM layout matches
    # reorder_stride_mode(): all batches' first 64 columns first, then all
    # batches' next 64 columns, etc.
    for j in range(col_blocks):
        for i in range(row_blocks):
            tile = tensor.tiles[(i, j)]
            hbm_group_obj = tensor.metadata.get("hbm_group_obj", f"{tensor.name}.hbm")
            layout = prog.compiler.sub_matrix_manager.hbm_matrices.get(str(hbm_group_obj))
            if layout is None:
                raise RuntimeError(
                    f"Compare staging tile HBM object is not registered: {tensor.name}[{i},{j}] -> {hbm_group_obj}"
                )
            layout_rows, layout_cols = layout.full_shape
            tile_row_count = min(prog.mlen, rows - i * prog.mlen)
            tile_col_count = min(prog.mlen, cols - j * prog.mlen)
            object_base = getattr(layout, "hbm_base_addr", None)
            if object_base is None:
                raise RuntimeError(
                    f"Compare staging HBM layout missing base address: {tensor.name}[{i},{j}] -> {hbm_group_obj}"
                )
            tile_hbm_stride = _logical_shape_to_hbm_stride(tensor.logical_shape)
            hbm_rel = _tile_coord_to_hbm_offset(tile.coord, tensor.logical_shape, prog.mlen)
            tile_hbm_addr = int(object_base) + int(hbm_rel)
            scale_size = int(layout_rows * layout_cols)
            staging_code.append(
                f"; compare stage tile {tensor.name}[{i},{j}] HBM[{tile_hbm_addr}] -> VRAM[{vram_addr}]"
            )
            staging_code.append(
                preload_addr_reg_asm(
                    addr_reg_to_set=[addr_reg],
                    available_registers=addr_gp,
                    addr_reg_val=[object_base],
                ).rstrip()
            )
            staging_code.append(reset_reg_asm(alive_registers=preload_gp).rstrip())
            staging_code.append(
                preload_act_asm(
                    vlen=prog.mlen,
                    preload_len=prog.blen,
                    batch=tile_row_count,
                    hidden_size=tile_col_count,
                    act_vram_offset=vram_addr,
                    alive_registers=preload_gp,
                    activation_offset_reg=addr_reg,
                    stride_size=(int(tile_hbm_stride) if tile_hbm_stride is not None else prog.mlen),
                    scale_size=scale_size,
                    hbm_start_offset=hbm_rel,
                ).rstrip()
            )
            vram_addr += tile_elems

    return {
        "start_row_idx": 0,
        "num_rows": rows * col_blocks,
        "num_batches": rows,
        "elements_per_batch": cols,
        "row_dim": prog.mlen,
        "use_stride_mode": True,
        "use_slice_mode": False,
        "compare_mode": "tiled_matrix",
        "logical_rows": rows,
        "logical_cols": cols,
        "tile_rows": prog.mlen,
        "tile_cols": prog.mlen,
        "stage_order": "col_major",
        "staging_isa": "\n".join(staging_code) + "\n",
    }
