"""
Utility functions for managing hardware configuration across tests.
"""

import tomlkit
from pathlib import Path
from typing import Optional


def update_plena_config(
    vlen: Optional[int] = None,
    mlen: Optional[int] = None,
    blen: Optional[int] = None,
    verbose: bool = True
) -> None:
    """
    Update plena_settings.toml with test-specific hardware parameters.

    Args:
        vlen: Vector length. If None, keeps current value.
        mlen: Matrix tile length. If None, keeps current value.
        blen: Batch tile length. If None, keeps current value.
        verbose: If True, print the updated configuration.

    Example:
        # Update both vlen and mlen
        update_plena_config(vlen=128, mlen=128)

        # Update only vlen
        update_plena_config(vlen=64)
    """
    plena_settings_path = Path(__file__).parent.parent.parent / "src" / "definitions" / "plena_settings.toml"

    with open(plena_settings_path, 'r') as f:
        config = tomlkit.load(f)

    updated = []
    if vlen is not None:
        config['CONFIG']['VLEN']['value'] = vlen
        updated.append(f"VLEN={vlen}")
    if mlen is not None:
        config['CONFIG']['MLEN']['value'] = mlen
        updated.append(f"MLEN={mlen}")
    if blen is not None:
        config['CONFIG']['BLEN']['value'] = blen
        updated.append(f"BLEN={blen}")

    with open(plena_settings_path, 'w') as f:
        tomlkit.dump(config, f)

    if verbose and updated:
        print(f"Updated plena_settings.toml: {', '.join(updated)}")


def get_comparison_params(
    vlen: int,
    batch_size: int,
    hidden_size: int,
    result_vram_offset: int = 0,
    use_stride_mode: Optional[bool] = None
) -> dict:
    """
    Generate comparison parameters for view_mem.py based on test configuration.

    Args:
        vlen: Vector length used in the test
        batch_size: Number of batches
        hidden_size: Hidden dimension size (elements per batch)
        result_vram_offset: Starting address in VRAM where results are stored
        use_stride_mode: If None, automatically determined based on vlen vs hidden_size.
                        If True/False, uses that value explicitly.

    Returns:
        Dictionary of comparison parameters for view_mem.py

    Note:
        Stride mode is used when vlen < hidden_size (multiple vectors per batch).
        Batch-wise mode is used when vlen >= hidden_size (one vector per batch).
    """
    result_start_row = result_vram_offset // vlen
    num_result_rows = (batch_size * hidden_size) // vlen

    if use_stride_mode is None:
        use_stride_mode = (hidden_size > vlen)

    return {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": batch_size,
        "elements_per_batch": hidden_size,
        "row_dim": vlen,
        "use_stride_mode": use_stride_mode
    }
