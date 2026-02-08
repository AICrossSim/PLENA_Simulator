import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from torch import nn
from compiler.asm_templates import projection_asm, projection_T_asm, preload_act_asm, reset_reg_asm, preload_addr_reg_asm
from behavioral_simulator.tools.create_sim_env import create_sim_env
from compiler.sim_env_utils import create_mem_for_sim
from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware
from check_mem import compare_with_golden
import json
import os


def quantize_to_mxfp(tensor):
    """
    Quantize tensor to MXFP format matching hardware (E4M3 with 8-bit scale per block of 8).
    Uses the same quantizer as the behavioral simulator's memory loader.
    Returns the dequantized tensor (what hardware sees after HBM->VRAM load).
    """
    orig_shape = tensor.shape
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor, width=8, exponent_width=4, exponent_bias_width=8, block_size=[8]
    )
    return bm_x.reshape(orig_shape)


if __name__ == "__main__":
    # Testing rectangular linear: (batch, in_features) @ (in_features, out_features) -> (batch, out_features)
    in_features = 64
    out_features = 64  # Rectangular matrix test
    batch_size = 8
    real_data_ratio = (8*8 + 8) / (8 * 8)
    fp_preload = [0.0, 1e-6, 1/in_features]

    # 使用更简单的测试数据，便于调试和验证
    # Activation: 每行都是相同的简单序列 [1, 2, 3, ..., in_features]
    act_tensor = torch.zeros(batch_size, in_features, dtype=torch.float32)
    for i in range(batch_size):
        for j in range(in_features):
            act_tensor[i, j] = j + 1  # 每行都是 [1, 2, 3, ..., in_features]
    
    # Weight: 使用单位矩阵（对角线为1，其他为0）
    # 注意：权重是 (out_features, in_features)
    # 使用单位矩阵，这样 act @ weight^T 的结果就是 act 本身（如果 out_features == in_features）
    weight_tensor = torch.eye(out_features, in_features, dtype=torch.float32)
    
    # 计算 golden result
    # 使用 projection_T: act @ weight^T
    # 等价于: act @ weight.T
    original_output = torch.matmul(act_tensor, weight_tensor.T)  # (batch, in_features) @ (in_features, out_features)
    
    # 转换为 bfloat16 以匹配硬件精度
    act_mxfp = act_tensor.to(torch.bfloat16)
    weights_mxfp = weight_tensor.to(torch.bfloat16)

    print("="*60)
    print("Projection_T Test (with simple test data)")
    print("="*60)
    print(f"Activation shape: {act_tensor.shape}")
    print(f"  Activation pattern: 每行都是 [1, 2, 3, ..., {in_features}]")
    print(f"  Activation sample (first row, first 8): {act_tensor[0, :8].tolist()}")
    print(f"\nWeight shape: {weight_tensor.shape} (will be transposed automatically)")
    print(f"  Weight pattern: 单位矩阵 (identity matrix)")
    print(f"  Weight sample (first row, first 8): {weight_tensor[0, :8].tolist()}")
    print(f"\nExpected output shape: {original_output.shape}")
    print(f"Operation: (batch, in_features) @ (out_features, in_features)^T")
    print(f"         = (batch, in_features) @ (in_features, out_features)")
    print(f"         = (batch, out_features)")
    print(f"\nGolden output (first 3x8):\n{original_output[:3, :8]}")
    if in_features == out_features:
        print(f"\nNote: Since in_features == out_features, with identity weight matrix,")
        print(f"      the output should be the same as input (each row = [1, 2, 3, ..., {in_features}])")

    # Weight is stored as (out_features, in_features) in PyTorch, we transpose for our layout
    # Our layout: (in_features, out_features) for matmul: act @ weight
    # Store original (non-quantized) tensors - they will be quantized when loaded to HBM
    input_tensor = {
        "act_tensor": act_mxfp,  # Use MXFP-quantized to match simulator
        "weights": weights_mxfp,  # Use MXFP-quantized to match simulator
    }

    golden_result = {
        "input_tensor": input_tensor,
        "original_output": original_output
    }

    gen_assembly_code = "; Projection_T Test Generation (Simple Test Data)\n"
    gen_assembly_code += f"; Shape: ({batch_size}, {in_features}) @ ({in_features}, {out_features}) -> ({batch_size}, {out_features})\n"

    # Calculate HBM offsets
    # Layout in HBM: [activations | weights]
    act_hbm_size = int(in_features * batch_size * real_data_ratio)
    weight_hbm_offset = act_hbm_size
    weight_hbm_end = int((in_features * batch_size + in_features * out_features) * real_data_ratio)

    # Set the addr offset for weight
    gen_assembly_code += preload_addr_reg_asm(
        addr_reg_to_set=[1, 2],
        available_registers=[1, 2],
        addr_reg_val=[weight_hbm_offset, weight_hbm_end]
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3]
    )

    # Gen Activation Preload
    gen_assembly_code += preload_act_asm(
        vlen=64,
        preload_len=4,
        batch=batch_size,
        hidden_size=in_features,
        alive_registers=[1,2,3,4,5],
        act_vram_offset=0,
        activation_offset_reg=0,
        stride_size=in_features
    )

    # Reset the registers
    gen_assembly_code += reset_reg_asm(
        alive_registers=[1,2,3,4]
    )

    # Result is stored after activation in VRAM
    result_vram_offset = in_features * batch_size

    gen_assembly_code += projection_T_asm(
        mlen=64,
        blen=4,
        batch=batch_size,
        hidden_size=in_features,      # in_features (input dimension)
        out_features=out_features,     # out_features (output dimension) - rectangular support!
        alive_registers=[1,2,3,4,5,6],
        w_base_hbm_offset_reg=1,
        activation_base_address=0,
        result_base_address=result_vram_offset,
        rope_enabled=False
    )

    create_sim_env(input_tensor, gen_assembly_code, golden_result, fp_preload)
    create_mem_for_sim(data_size=256, mode="behave_sim", asm="linear", data=None, specified_data_order=["act_tensor", "weights"])

    # Save comparison parameters for view_mem.py
    result_start_row = result_vram_offset // 64  # Row where results start
    num_result_rows = (batch_size * out_features) // 64
    comparison_params = {
        "start_row_idx": result_start_row,
        "num_rows": num_result_rows,
        "num_batches": batch_size,
        "elements_per_batch": out_features
    }
    build_dir = Path(__file__).parent / "build"
    with open(build_dir / "comparison_params.json", "w") as f:
        json.dump(comparison_params, f, indent=2)

    print("================================================")
    print("Finished generating assembly code")
    print(f"Result location: row {result_start_row}, {num_result_rows} rows")
    print(f"Comparison params: {comparison_params}")
    print("================================================")
    
    # Try to compare results with golden output
    print("\n" + "="*60)
    print("Comparing Results with Golden Output")
    print("="*60)
    
    # Find VRAM dump file
    script_dir = Path(__file__).parent.parent.parent
    vram_file = script_dir / "behavioral_simulator" / "vram_dump.bin"
    golden_file = build_dir / "golden_result.txt"
    output_file = build_dir / "golden_vs_simulated.txt"
    
    # Always create/update the comparison file if golden file exists
    if golden_file.exists():
        if vram_file.exists():
            # Both files exist, do full comparison
            print(f"✓ Found VRAM dump: {vram_file}")
            print(f"✓ Found golden file: {golden_file}")
            
            # Compare results
            results = compare_with_golden(
                str(vram_file),
                str(golden_file),
                exp_width=8,
                man_width=7,
                num_bytes_per_val=2,
                row_dim=64,
                start_row_idx=result_start_row,
                num_rows=num_result_rows,
                num_batches=batch_size,
                use_stride_mode=True,
                elements_per_batch=out_features,
                atol=0.1,
                rtol=0.1
            )
            
            # Write comparison results to file
            with open(output_file, "w") as f:
                f.write("="*80 + "\n")
                f.write("Golden Output vs Simulated Output Comparison\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Test Configuration:\n")
                f.write(f"  Batch size: {batch_size}\n")
                f.write(f"  Input features: {in_features}\n")
                f.write(f"  Output features: {out_features}\n")
                f.write(f"  Start row: {result_start_row}\n")
                f.write(f"  Num rows: {num_result_rows}\n")
                f.write(f"  Elements per batch: {out_features}\n\n")
                
                f.write("Comparison Metrics:\n")
                f.write(f"  Mean Squared Error (MSE):     {results['mse']:.6e}\n")
                f.write(f"  Mean Absolute Error (MAE):    {results['mae']:.6e}\n")
                f.write(f"  Max Absolute Error:           {results['max_error']:.6f}\n")
                f.write(f"  Mean Relative Error:          {results['relative_error']:.6f}\n")
                f.write(f"  Match Rate (allclose):        {results['allclose_match_rate']:.2f}%\n")
                f.write(f"  All Values Pass:              {'PASS' if results['allclose_pass'] else 'FAIL'}\n\n")
                
                f.write("="*80 + "\n")
                f.write("Detailed Comparison (First 100 values)\n")
                f.write("="*80 + "\n")
                f.write(f"{'Index':<8} {'Golden':<15} {'Simulated':<15} {'Error':<15} {'Relative Error':<15}\n")
                f.write("-"*80 + "\n")
                
                golden_vals = results['golden_values']
                simulated_vals = results['simulated_values']
                errors = results['errors']
                abs_golden = torch.abs(golden_vals)
                relative_errors = torch.where(
                    abs_golden > 1e-10,
                    errors / abs_golden,
                    errors
                )
                
                num_to_show = min(100, len(golden_vals))
                for i in range(num_to_show):
                    f.write(f"{i:<8} {golden_vals[i].item():<15.6f} {simulated_vals[i].item():<15.6f} "
                           f"{errors[i].item():<15.6f} {relative_errors[i].item():<15.6f}\n")
                
                if len(golden_vals) > num_to_show:
                    f.write(f"\n... (showing first {num_to_show} of {len(golden_vals)} values)\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("Full Golden Output (reshaped to batch-wise layout)\n")
                f.write("="*80 + "\n")
                golden_reshaped = golden_vals.reshape(batch_size, out_features)
                for i in range(batch_size):
                    f.write(f"Batch {i}:\n")
                    for j in range(0, out_features, 8):
                        end_j = min(j + 8, out_features)
                        values_str = " ".join([f"{golden_reshaped[i, k].item():10.6f}" for k in range(j, end_j)])
                        f.write(f"  [{values_str}]\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("Full Simulated Output (reshaped to batch-wise layout)\n")
                f.write("="*80 + "\n")
                simulated_reshaped = simulated_vals.reshape(batch_size, out_features)
                for i in range(batch_size):
                    f.write(f"Batch {i}:\n")
                    for j in range(0, out_features, 8):
                        end_j = min(j + 8, out_features)
                        values_str = " ".join([f"{simulated_reshaped[i, k].item():10.6f}" for k in range(j, end_j)])
                        f.write(f"  [{values_str}]\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("Error Analysis (Top 20 Largest Errors)\n")
                f.write("="*80 + "\n")
                f.write(f"{'Index':<8} {'Batch':<8} {'Col':<8} {'Golden':<15} {'Simulated':<15} {'Error':<15}\n")
                f.write("-"*80 + "\n")
                
                # Find top 20 largest errors
                top_errors_idx = torch.argsort(errors, descending=True)[:20]
                for idx in top_errors_idx:
                    batch_idx = idx.item() // out_features
                    col_idx = idx.item() % out_features
                    f.write(f"{idx.item():<8} {batch_idx:<8} {col_idx:<8} "
                           f"{golden_vals[idx].item():<15.6f} {simulated_vals[idx].item():<15.6f} "
                           f"{errors[idx].item():<15.6f}\n")
            
            print(f"✓ Comparison results saved to: {output_file}")
            print(f"  Full path: {output_file.absolute()}")
            print(f"\nComparison Summary:")
            print(f"  MSE: {results['mse']:.6e}")
            print(f"  MAE: {results['mae']:.6e}")
            print(f"  Max Error: {results['max_error']:.6f}")
            print(f"  Match Rate: {results['allclose_match_rate']:.2f}%")
            print(f"  All Pass: {'PASS' if results['allclose_pass'] else 'FAIL'}")
        else:
            # Only golden file exists, save golden output only
            print(f"✓ Found golden file: {golden_file}")
            print(f"⚠️  VRAM dump not found: {vram_file}")
            print("  Saving golden output only (no comparison available)")
            
            # Parse golden output
            from check_mem import parse_golden_output
            golden_np = parse_golden_output(str(golden_file))
            golden_vals = torch.from_numpy(golden_np).float()
            
            # Write golden output to file
            with open(output_file, "w") as f:
                f.write("="*80 + "\n")
                f.write("Golden Output (Simulated output not available)\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Test Configuration:\n")
                f.write(f"  Batch size: {batch_size}\n")
                f.write(f"  Input features: {in_features}\n")
                f.write(f"  Output features: {out_features}\n")
                f.write(f"  Start row: {result_start_row}\n")
                f.write(f"  Num rows: {num_result_rows}\n")
                f.write(f"  Elements per batch: {out_features}\n\n")
                
                f.write("Note: VRAM dump not found. Run the simulator to generate comparison.\n\n")
                
                f.write("="*80 + "\n")
                f.write("Full Golden Output (reshaped to batch-wise layout)\n")
                f.write("="*80 + "\n")
                golden_reshaped = golden_vals.reshape(batch_size, out_features)
                for i in range(batch_size):
                    f.write(f"Batch {i}:\n")
                    for j in range(0, out_features, 8):
                        end_j = min(j + 8, out_features)
                        values_str = " ".join([f"{golden_reshaped[i, k].item():10.6f}" for k in range(j, end_j)])
                        f.write(f"  [{values_str}]\n")
            
            print(f"✓ Golden output saved to: {output_file}")
            print(f"  Full path: {output_file.absolute()}")
    else:
        print("⚠️  Golden file not found. Cannot create comparison file.")
        print(f"  Missing: {golden_file}")
        print(f"\nNote: golden_vs_simulated.txt will be created in: {build_dir.absolute()}")
