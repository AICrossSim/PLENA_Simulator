"""
ATen-style Batched Matrix Multiply Test

Tests:  (B, M, K) @ (B, K, N) -> (B, M, N)
with B=4, M=64, K=128, N=128, mlen=64, blen=4.

The assembly uses batched_matmul_asm which calls M_MM_WO with
stride_len = N // mlen = 2.  With two n_groups the even/odd VRAM rows
are both filled, so all B*M*(N//mlen) = 512 consecutive rows starting
at result_base_address // mlen are populated in row-major order.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import torch

from compiler.asm_templates import batched_matmul_asm, preload_addr_reg_asm, reset_reg_asm
from transactional_emulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from emulator_runner import run_and_assert


if __name__ == "__main__":
    print("=" * 80)
    print("ATen-style Batched Matrix Multiply Test  (batched_matmul_asm)")
    print("=" * 80)

    # ========================================================================
    # Parameters
    # ========================================================================
    m = 64
    k = 128
    n = 128
    batch_size = 4
    mlen = 64
    blen = 4
    real_data_ratio = (8 * 8 + 8) / (8 * 8)
    result_base_address = 2048  # past activation area (rows 0-31)

    # ========================================================================
    # Golden reference
    # ========================================================================
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, m, k)
    weight_tensor = torch.randn(batch_size, k, n)
    original_output = torch.bmm(input_tensor, weight_tensor)

    print(f"\nInput:  {input_tensor.shape}")
    print(f"Weight: {weight_tensor.shape}")
    print(f"Output: {original_output.shape}")

    # ========================================================================
    # ISA generation
    # ========================================================================
    gen_assembly_code = "; BMM ATen Test\n"

    # a1 = HBM offset of weight tensor (past activation data)
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1],
        available_registers=[1],
        addr_reg_val=[int(m * k * batch_size * real_data_ratio)],
    )

    gen_assembly_code += reset_reg_asm(alive_registers=[1])

    gen_assembly_code += batched_matmul_asm(
        mlen=mlen,
        blen=blen,
        b=batch_size,
        m=m,
        k=k,
        n=n,
        alive_registers=[1, 2, 3],
        w_base_hbm_offset_reg=1,   # a1 holds weight HBM offset
        w_prefetch_amount=k,
        a_base_hbm_offset_reg=0,   # a0 = 0 (activation starts at HBM byte 0)
        a_prefetch_amount=4,
        result_base_address=result_base_address,
    )

    # ========================================================================
    # Build simulation environment
    # ========================================================================
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    input_data = {"input_tensor": input_tensor, "model_weights": weight_tensor}
    golden_result = {
        "input_tensor": input_tensor,
        "weights": weight_tensor,
        "original_output": original_output,
    }

    create_sim_env(
        input_data, gen_assembly_code, golden_result,
        fp_preload=[0.0],
        build_dir=build_dir,
    )

    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm="linear",
        data=None,
        specified_data_order=["input_tensor", "model_weights"],
        build_path=build_dir,
    )

    # ========================================================================
    # Comparison parameters
    # ========================================================================
    # M_MM_WO with stride_len = N//mlen = 2 interleaves two n_groups:
    #   n_group 0 (cols 0..63)  fills even VRAM rows (32, 34, 36, ...)
    #   n_group 1 (cols 64..127) fills odd  VRAM rows (33, 35, 37, ...)
    # Together they pack the full (B, M, N) result row-major into
    # B * M * (N // mlen) = 4 * 64 * 2 = 512 consecutive rows from start_row.
    result_start_row = result_base_address // mlen   # 32
    num_result_rows = batch_size * m * (n // mlen)   # 512

    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": batch_size,
        "elements_per_batch": m * n,
        "row_dim": mlen,
        "use_stride_mode": False,
        "row_stride": 1,
    }

    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    with open(build_dir / "generated_asm_code.asm", "w") as f:
        f.write(gen_assembly_code)

    print(f"\nSimulation environment created: {build_dir}")
    print(f"Result at VRAM rows {result_start_row}..{result_start_row + num_result_rows - 1}")

    run_and_assert(build_dir, "bmm", mlen=mlen, blen=blen)
