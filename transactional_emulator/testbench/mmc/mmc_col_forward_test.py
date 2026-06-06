import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware

from compiler.asm_templates import columnwise_scan_stride_asm, preload_addr_reg_asm, reset_reg_asm
from compiler.sim_env_utils import create_mem_for_sim
from transactional_emulator.testbench.config_utils import update_plena_config
from transactional_emulator.testbench.emulator_runner import compare_emulator_output, run_emulator
from transactional_emulator.testbench.mmc.padding_utils import align_up, choose_kernel_grid, pad_sequence_to_kernel
from transactional_emulator.tools.check_mem import print_comparison_results
from transactional_emulator.tools.create_sim_env import create_sim_env


def quantize_to_mxfp(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to MXFP format matching hardware."""
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor,
        width=8,
        exponent_width=4,
        exponent_bias_width=8,
        block_size=[8],
    )
    return bm_x.reshape(orig_shape)


if __name__ == "__main__":
    # Forward-column prefetch test parameters (B=1, logical 27x27 sequence grid)
    rows_valid = 27
    cols_valid = 27
    seq_len_valid = rows_valid * cols_valid
    feature_dim = 2048
    vlen = 128
    mlen = 128
    blen = 4

    # Match full-model runtime policy: pad sequence length to MLEN boundary.
    seq_len_kernel = align_up(seq_len_valid, mlen)
    rows_kernel, cols_kernel = choose_kernel_grid(
        seq_len_kernel=seq_len_kernel,
        target_rows=rows_valid,
        target_cols=cols_valid,
        blen=blen,
    )

    update_plena_config(vlen=vlen, mlen=mlen, blen=blen, verbose=False)

    torch.manual_seed(42)
    input_tensor_valid = torch.randn(seq_len_valid, feature_dim, dtype=torch.bfloat16)
    input_mxfp_valid = quantize_to_mxfp(input_tensor_valid).to(torch.bfloat16)
    input_mxfp_kernel = pad_sequence_to_kernel(input_mxfp_valid, seq_len_kernel)

    # Golden over padded kernel grid. This mirrors full-model behavior where padded
    # sequence tokens are carried through kernel execution.
    golden_output = (
        input_mxfp_kernel.view(rows_kernel, cols_kernel, feature_dim)
        .transpose(0, 1)
        .contiguous()
        .view(seq_len_kernel, feature_dim)
    )
    golden_result = {"input_tensor": input_mxfp_kernel, "original_output": golden_output}

    gen_assembly_code = "; MMC Col Forward (Columnwise H_PREFETCH_V) Test Generation\n"
    gen_assembly_code += (
        f"; Shape: (B=1, rows={rows_kernel}, cols={cols_kernel}, D={feature_dim}) "
        "columnwise token-order prefetch\n"
    )
    gen_assembly_code += f"; seq_len_valid={seq_len_valid}, seq_len_kernel={seq_len_kernel}\n"
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[0],
        available_registers=[0],
        addr_reg_val=[0],
    )
    gen_assembly_code += reset_reg_asm(alive_registers=[1, 2, 3, 4])
    gen_assembly_code += columnwise_scan_stride_asm(
        mlen=mlen,
        vlen=vlen,
        rows=rows_kernel,
        cols=cols_kernel,
        feature_dim=feature_dim,
        prefetch_block_size=blen,
        alive_registers=[1, 2, 3, 4],
        input_hbm_base_addr_reg=0,
        output_vram_base=0,
    )

    build_path = Path(__file__).parent / "build" / "mmc_col_forward_test"
    create_sim_env(
        {"input_tensor": input_mxfp_kernel},
        gen_assembly_code,
        golden_result,
        [0.0, 1e-6],
        build_dir=build_path,
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="mmc_col_forward",
        data=None,
        specified_data_order=["input_tensor"],
        build_path=build_path,
    )

    comparison_params = {
        "start_row_idx": 0,
        "num_rows": seq_len_kernel * feature_dim // vlen,
        "num_batches": seq_len_kernel,
        "elements_per_batch": feature_dim,
        "row_dim": vlen,
        "use_stride_mode": False,
        "use_chunk_major_mode": True,
        "seq_len": seq_len_kernel,
        "hidden_dim": feature_dim,
        "mlen": vlen,
        "chunk_major_valid_seq_len": seq_len_kernel,
    }
    with open(build_path / "comparison_params.json", "w", encoding="utf-8") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating mmc_col_forward_test assembly code")
    print(
        "Kernel layout: "
        f"valid=({rows_valid}x{cols_valid}={seq_len_valid}), "
        f"kernel=({rows_kernel}x{cols_kernel}={seq_len_kernel}), "
        f"mlen={mlen}, blen={blen}"
    )
    print(f"Comparison params: {comparison_params}")
    print("================================================")

    print("\n--- Running Rust transactional emulator ---")
    run_emulator(build_path)

    print("\n--- Comparing emulator output vs golden ---")
    results, params = compare_emulator_output(build_path)
    print_comparison_results(results, verbose=True, comparison_params=params)

    simulated_values = results.get("simulated_values")
    if simulated_values is not None:
        nan_mask = torch.isnan(simulated_values)
        nan_count = int(nan_mask.sum().item())
        if nan_count > 0:
            print(f"\n[mmc_col_forward_test ERROR] Found {nan_count} NaN(s) in compared simulator output")
            first_nan_indices = torch.nonzero(nan_mask, as_tuple=False).flatten()[:10].tolist()
            for flat_idx in first_nan_indices:
                token = flat_idx // feature_dim
                feat = flat_idx % feature_dim
                row = token % rows_kernel
                col = token // rows_kernel
                tile = feat // vlen
                off = feat % vlen
                print(
                    "  "
                    f"idx={flat_idx}, token={token}, row={row}, col={col}, "
                    f"feat={feat}, tile={tile}, off={off}"
                )
            raise SystemExit(1)

    if results.get("allclose_pass", False):
        print("\n[mmc_col_forward_test PASSED - emulator numerical check passed]")
    else:
        print("\n[mmc_col_forward_test FAILED - emulator numerical check failed]")
        raise SystemExit(1)
