"""Unit contracts for the opt-in tile-major HBM matrix layout."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Import the compiler submodule belonging to this checkout.  Test results must
# not depend on an unrelated compiler path inherited through PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPILER_ROOT = _REPO_ROOT / "PLENA_Compiler"
sys.path.insert(0, str(_COMPILER_ROOT))

from compiler.aten.plena.memory import MatrixBlockLayout
from transactional_emulator.testbench.sim_env_utils import map_mx_data_to_hbm_for_behave_sim


def test_compiler_tile_major_offsets_and_stride() -> None:
    row_major = MatrixBlockLayout(
        name="W",
        full_shape=(128, 192),
        physical_shape=(128, 192),
        block_size=64,
        storage_order="row_major",
    )
    tile_major = MatrixBlockLayout(
        name="W",
        full_shape=(128, 192),
        physical_shape=(128, 192),
        block_size=64,
        storage_order="tile_major",
    )

    assert row_major.get_sub_block(0, 1).hbm_offset == 64
    assert row_major.get_sub_block(1, 0).hbm_offset == 64 * 192
    assert row_major.hbm_row_stride_elements == 192

    assert tile_major.get_sub_block(0, 1).hbm_offset == 64 * 64
    assert tile_major.get_sub_block(1, 0).hbm_offset == 3 * 64 * 64
    assert tile_major.hbm_row_stride_elements == 64


def test_tile_major_writer_orders_elements_and_scales_by_tile(tmp_path) -> None:
    elements = np.arange(64, dtype=np.uint8).reshape(8, 8)
    blocks = elements.reshape(8, 4, 2).reshape(-1, 2)
    scales = np.arange(32, dtype=np.uint8).reshape(8, 4)

    map_mx_data_to_hbm_for_behave_sim(
        blocks=blocks,
        element_width=8,
        block_width=2,
        bias=scales.reshape(-1),
        bias_width=8,
        directory=tmp_path,
        append=False,
        logical_row_elements=8,
        source_row_elements=8,
        logical_rows=8,
        source_rows=8,
        storage_order="tile_major",
        tile_size=4,
    )

    payload = (tmp_path / "hbm_for_behave_sim.bin").read_bytes()
    expected_elements = b"".join(
        elements[row : row + 4, col : col + 4].reshape(-1).tobytes()
        for row in (0, 4)
        for col in (0, 4)
    )
    expected_scales = b"".join(
        scales[row : row + 4, col // 2 : col // 2 + 2].reshape(-1).tobytes()
        for row in (0, 4)
        for col in (0, 4)
    )
    assert payload == expected_elements + expected_scales
