import hashlib
import os

import torch
from torch import nn


def generate_and_save_random_weights(
    input_dim, output_dim, filename="model_weights.pth"
):
    """
    Generates random weights for a Linear layer with the given input and output dimensions,
    and saves them in .pth format so they can be loaded with torch.load and used with
    m.load_state_dict(saved_weights).
    """
    # Create a Linear layer
    model = nn.Linear(input_dim, output_dim)
    # Get the state dict (contains 'weight' and 'bias')
    state_dict = model.state_dict()
    # Save the state dict to the specified file
    torch.save(state_dict, filename)
    print(f"Random weights saved to {filename}")


def get_weights_path(filename="model_weights.pth"):
    """
    Returns the absolute path to the weights file in the current directory.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def test_comparison_rejects_mismatched_cardinality():
    """A matching prefix cannot substitute for a complete result."""
    import numpy as np

    from verification.verify_rtl_sim import compare_results

    result = compare_results(
        np.zeros(15, dtype=np.float32),
        np.zeros(16, dtype=np.float32),
    )
    assert not result["passed"]
    assert result["num_compared"] == 0
    assert "cardinality mismatch" in result["error"]


def test_vram_verification_rejects_truncated_result_and_golden(tmp_path):
    """The configured row window must be present in both compared artifacts."""
    from verification.verify_rtl_sim import verify_vram

    result_path = tmp_path / "vector_result.mem"
    golden_path = tmp_path / "golden_vram_result.pt"
    result_path.write_bytes(b"0\n" * 16)
    torch.save(torch.zeros((16, 16), dtype=torch.float32), golden_path)
    params = {
        "golden_vram_file": golden_path.name,
        "vram_start_row_idx": 0,
        "vram_num_rows": 16,
        "row_dim": 16,
        "vram_compare_start_row": 0,
        "vram_compare_num_rows": 16,
        "vram_total_rows": 16,
    }

    exact = verify_vram(tmp_path, params, verbose=False, save_fp_result=False)
    assert exact["passed"], exact
    assert exact["num_compared"] == 256

    original_result = result_path.read_bytes()
    result_md5 = hashlib.md5(original_result).hexdigest()
    result_path.write_bytes(b"0\n" * 15)
    truncated_result = verify_vram(
        tmp_path, params, verbose=False, save_fp_result=False
    )
    assert not truncated_result["passed"]
    assert "row cardinality mismatch" in truncated_result["error"]
    result_path.write_bytes(original_result)
    assert hashlib.md5(result_path.read_bytes()).hexdigest() == result_md5
    assert verify_vram(tmp_path, params, verbose=False, save_fp_result=False)[
        "passed"
    ]

    original_golden = golden_path.read_bytes()
    golden_md5 = hashlib.md5(original_golden).hexdigest()
    torch.save(torch.zeros(255, dtype=torch.float32), golden_path)
    truncated_golden = verify_vram(
        tmp_path, params, verbose=False, save_fp_result=False
    )
    assert not truncated_golden["passed"]
    assert "golden cardinality mismatch" in truncated_golden["error"]
    golden_path.write_bytes(original_golden)
    assert hashlib.md5(golden_path.read_bytes()).hexdigest() == golden_md5
    assert verify_vram(tmp_path, params, verbose=False, save_fp_result=False)[
        "passed"
    ]

    invalid_window = verify_vram(
        tmp_path,
        params | {"vram_compare_num_rows": 17},
        verbose=False,
        save_fp_result=False,
    )
    assert not invalid_window["passed"]
    assert "comparison window is outside" in invalid_window["error"]


def test_hbm_verification_rejects_truncated_result_and_golden(tmp_path):
    """Missing HBM rows and shortened comparison references fail closed."""
    from verification.verify_rtl_sim import verify_hbm

    result_path = tmp_path / "hbm_result.mem"
    golden_path = tmp_path / "golden_hbm_result.pt"
    result_path.write_bytes(b"0x0\n")
    torch.save(torch.zeros(8, dtype=torch.float32), golden_path)
    params = {
        "golden_hbm_file": golden_path.name,
        "result_hbm_start_byte": 0,
        "num_elements": 8,
        "scale_width": 8,
        "block_size": 8,
        "scale_offset": 8,
        "mx_format": "mxint",
        "man_width": 8,
        "hbm_compare_start_row": 0,
        "hbm_compare_num_rows": 1,
        "hbm_elements_per_row": 8,
        "hbm_total_rows": 1,
    }

    exact = verify_hbm(tmp_path, params, verbose=False, save_translated=False)
    assert exact["passed"], exact
    assert exact["num_compared"] == 8

    original_result = result_path.read_bytes()
    result_md5 = hashlib.md5(original_result).hexdigest()
    result_path.write_bytes(b"")
    truncated_result = verify_hbm(
        tmp_path, params, verbose=False, save_translated=False
    )
    assert not truncated_result["passed"]
    assert "source row is missing" in truncated_result["error"]
    result_path.write_bytes(original_result)
    assert hashlib.md5(result_path.read_bytes()).hexdigest() == result_md5
    assert verify_hbm(tmp_path, params, verbose=False, save_translated=False)[
        "passed"
    ]

    missing_scale = verify_hbm(
        tmp_path,
        params
        | {
            "num_elements": 32,
            "scale_offset": 32,
            "hbm_compare_num_rows": 4,
            "hbm_total_rows": 4,
        },
        verbose=False,
        save_translated=False,
    )
    assert not missing_scale["passed"]
    assert "scale source row is missing" in missing_scale["error"]

    original_golden = golden_path.read_bytes()
    golden_md5 = hashlib.md5(original_golden).hexdigest()
    torch.save(torch.zeros(7, dtype=torch.float32), golden_path)
    truncated_golden = verify_hbm(
        tmp_path, params, verbose=False, save_translated=False
    )
    assert not truncated_golden["passed"]
    assert "golden cardinality mismatch" in truncated_golden["error"]
    golden_path.write_bytes(original_golden)
    assert hashlib.md5(golden_path.read_bytes()).hexdigest() == golden_md5
    assert verify_hbm(tmp_path, params, verbose=False, save_translated=False)[
        "passed"
    ]

    invalid_window = verify_hbm(
        tmp_path,
        params | {"hbm_compare_num_rows": 2},
        verbose=False,
        save_translated=False,
    )
    assert not invalid_window["passed"]
    assert "comparison window is outside" in invalid_window["error"]


if __name__ == "__main__":
    generate_and_save_random_weights(
        128, 128, get_weights_path("model_weights.pth")
    )
