#!/usr/bin/env python3
"""
Automatically compare Input Tensor from golden_result.txt with VRAM dump from simulation.log
Considering stride mode data layout
"""

import re
import os
import sys

def parse_golden_input_tensor(filename):
    """Parse Input Tensor from golden_result.txt"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find Input Tensor section
    match = re.search(r'Input Tensor:\s*\[(.*?)\]', content, re.DOTALL)
    if not match:
        print("❌ Cannot find Input Tensor section")
        sys.exit(1)
    
    tensor_text = match.group(1)
    
    # Parse each line (each batch)
    batches = []
    for line in tensor_text.strip().split('\n'):
        if line.strip():
            # Extract all numbers (including negative and decimal)
            numbers = re.findall(r'-?\d+\.\d+', line)
            if numbers:
                batches.append([float(x) for x in numbers])
    
    return batches

def parse_simulation_vram_rows(filename, start_row=0, end_row=8):
    """Parse VRAM dump from simulation.log"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    rows = []
    for line in lines:
        # Match "Row   N:   number list"
        match = re.match(r'Row\s+(\d+):\s+(.*)', line)
        if match:
            row_num = int(match.group(1))
            if start_row <= row_num < end_row:
                # Extract all floating point numbers
                numbers = re.findall(r'-?\d+\.\d+', match.group(2))
                if numbers:
                    rows.append([float(x) for x in numbers])
    
    return rows

def parse_golden_weights(filename):
    """Parse Weights from golden_result.txt"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find Weights section
    match = re.search(r'weight:\s*\[(.*?)\]', content, re.DOTALL)
    if not match:
        print("⚠️  Cannot find Weights section")
        return None
    
    weight_text = match.group(1)
    
    # Parse each line (each row of weight matrix)
    weights = []
    for line in weight_text.strip().split('\n'):
        if line.strip():
            # Extract all numbers
            numbers = re.findall(r'-?\d+\.\d+', line)
            if numbers:
                weights.append([float(x) for x in numbers])
    
    return weights

def parse_golden_output(filename):
    """Parse Original Output from golden_result.txt"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find Original Output section
    match = re.search(r'Original Output:\s*\[(.*?)\]', content, re.DOTALL)
    if not match:
        print("⚠️  Cannot find Original Output section")
        return None
    
    output_text = match.group(1)
    
    # Parse each line (each batch output)
    outputs = []
    for line in output_text.strip().split('\n'):
        if line.strip():
            # Extract all numbers
            numbers = re.findall(r'-?\d+\.\d+', line)
            if numbers:
                outputs.append([float(x) for x in numbers])
    
    return outputs

def parse_simulation_mram_rows(filename, num_rows=8):
    """Parse MRAM dump from simulation.log"""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Find "Viewing MRAM dump" section
    mram_match = re.search(r'Viewing MRAM dump 0 to.*?\n((?:Row.*?\n)+)', content, re.DOTALL)
    if not mram_match:
        return []
    
    mram_text = mram_match.group(1)
    rows = []
    
    for line in mram_text.split('\n'):
        if line.strip().startswith('Row'):
            match = re.match(r'Row\s+\d+:\s+(.*)', line)
            if match:
                numbers = re.findall(r'-?\d+\.\d+', match.group(1))
                if numbers:
                    rows.append([float(x) for x in numbers])
                if len(rows) >= num_rows:
                    break
    
    return rows

def compare_with_tolerance(golden, sim, tolerance=0.1):
    """Compare two values, considering quantization error"""
    diff = abs(golden - sim)
    return diff < tolerance, diff

def main():
    golden_file = "behavioral_simulator/testbench/build/golden_result.txt"
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Change golden_file and sim_file to absolute paths
    golden_file = os.path.join(script_dir, "behavioral_simulator/testbench/build/golden_result.txt")
    sim_file = os.path.join(script_dir, "simulation.log")
    
    print("=" * 80)
    print("Input Tensor Match Rate Check")
    print("=" * 80)
    print(f"Reading files:")
    print(f"  Golden: {golden_file}")
    print(f"  Simulation: {sim_file}")
    print()
    
    # Read data
    try:
        golden_batches = parse_golden_input_tensor(golden_file)
        sim_input_rows = parse_simulation_vram_rows(sim_file, start_row=0, end_row=8)
        golden_weights = parse_golden_weights(golden_file)
        sim_mram_rows = parse_simulation_mram_rows(sim_file, num_rows=8)
        golden_outputs = parse_golden_output(golden_file)
        sim_output_rows = parse_simulation_vram_rows(sim_file, start_row=8, end_row=16)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    
    if len(golden_batches) != 4:
        print(f"⚠️  Abnormal number of Golden batches: {len(golden_batches)}")
    
    if len(sim_input_rows) < 8:
        print(f"⚠️  Insufficient number of Simulation input rows: {len(sim_input_rows)}")
        sys.exit(1)
    
    print(f"✓ Golden input batches: {len(golden_batches)} batches")
    print(f"✓ Simulation VRAM input: {len(sim_input_rows)} rows")
    if golden_weights:
        print(f"✓ Golden weights: {len(golden_weights)}x{len(golden_weights[0]) if golden_weights else 0} matrix")
    if sim_mram_rows:
        print(f"✓ Simulation MRAM: {len(sim_mram_rows)} rows")
    if golden_outputs:
        print(f"✓ Golden output: {len(golden_outputs)} batches x {len(golden_outputs[0]) if golden_outputs else 0} elements")
    if sim_output_rows:
        print(f"✓ Simulation VRAM output: {len(sim_output_rows)} rows")
    print()
    
    # VRAM layout (stride mode):
    # Row 0-3: First 64 elements of Batch 0-3
    # Row 4-7: Last 64 elements of Batch 0-3
    
    total_matches = 0
    total_elements = 0
    batch_results = []
    
    for batch_idx in range(4):
        print("=" * 80)
        print(f"Batch {batch_idx} ({batch_idx+1}th batch of 4)")
        print("=" * 80)
        
        golden_batch = golden_batches[batch_idx]
        
        if len(golden_batch) != 128:
            print(f"⚠️  Golden batch {batch_idx} has abnormal number of elements: {len(golden_batch)}")
            continue
        
        # First 64 elements: Row batch_idx
        sim_row_first = sim_input_rows[batch_idx]
        # Last 64 elements: Row (batch_idx + 4)
        sim_row_second = sim_input_rows[batch_idx + 4]
        
        batch_matches = 0
        batch_total = 128
        significant_diffs = []
        
        # Compare first 64 elements
        for i in range(64):
            golden_val = golden_batch[i]
            sim_val = sim_row_first[i] if i < len(sim_row_first) else 0.0
            match, diff = compare_with_tolerance(golden_val, sim_val)
            
            if match:
                batch_matches += 1
            
            if diff >= 0.15:  # Record significant differences
                significant_diffs.append((i, golden_val, sim_val, diff, f"Row {batch_idx}"))
        
        # Compare last 64 elements
        for i in range(64):
            golden_val = golden_batch[64 + i]
            sim_val = sim_row_second[i] if i < len(sim_row_second) else 0.0
            match, diff = compare_with_tolerance(golden_val, sim_val)
            
            if match:
                batch_matches += 1
            
            if diff >= 0.15:  # Record significant differences
                significant_diffs.append((64+i, golden_val, sim_val, diff, f"Row {batch_idx+4}"))
        
        match_rate = batch_matches * 100.0 / batch_total
        total_matches += batch_matches
        total_elements += batch_total
        
        print(f"First 64 elements (Golden[0:64]   ↔ Sim Row {batch_idx})")
        print(f"Last 64 elements  (Golden[64:128] ↔ Sim Row {batch_idx+4})")
        print()
        
        if significant_diffs:
            print(f"Significant differences (diff >= 0.15): {len(significant_diffs)}/{batch_total} elements")
            if len(significant_diffs) <= 10:
                for idx, g_val, s_val, diff, row_info in significant_diffs[:10]:
                    print(f"  [{idx:3d}] Golden: {g_val:7.3f}  Sim: {s_val:7.3f}  "
                          f"Diff: {diff:6.3f}  ({row_info})")
            else:
                print(f"  (Showing first 10...)")
                for idx, g_val, s_val, diff, row_info in significant_diffs[:10]:
                    print(f"  [{idx:3d}] Golden: {g_val:7.3f}  Sim: {s_val:7.3f}  "
                          f"Diff: {diff:6.3f}  ({row_info})")
        
        print()
        status = "✅" if match_rate >= 95 else "⚠️" if match_rate >= 90 else "❌"
        print(f"{status} Batch {batch_idx} match rate: {batch_matches}/{batch_total} ({match_rate:.1f}%)")
        print()
        
        batch_results.append({
            'batch': batch_idx,
            'matches': batch_matches,
            'total': batch_total,
            'rate': match_rate
        })
    
    # ================================================================================
    # Compare Weights
    # ================================================================================
    print("\n" + "=" * 80)
    print("Weight Matrix Comparison")
    print("=" * 80)
    
    if golden_weights and sim_mram_rows:
        # Sample first few rows for comparison (full comparison would be too verbose)
        sample_rows = min(3, len(golden_weights), len(sim_mram_rows))
        weight_all_match = True
        weight_issue_count = 0
        
        for row_idx in range(sample_rows):
            if row_idx >= len(golden_weights) or row_idx >= len(sim_mram_rows):
                break
            
            golden_row = golden_weights[row_idx]
            sim_row = sim_mram_rows[row_idx]
            
            # Sample first 10 elements
            sample_size = min(10, len(golden_row), len(sim_row))
            row_match_count = 0
            
            for i in range(sample_size):
                if i < len(golden_row) and i < len(sim_row):
                    match, diff = compare_with_tolerance(golden_row[i], sim_row[i], tolerance=0.15)
                    if match:
                        row_match_count += 1
                    elif abs(sim_row[i]) > 0.001:  # Not zero
                        weight_issue_count += 1
            
            match_rate = row_match_count * 100.0 / sample_size if sample_size > 0 else 0
            print(f"Weight Row {row_idx} (first {sample_size} elements): {row_match_count}/{sample_size} ({match_rate:.1f}%)")
            
            if match_rate < 50:
                weight_all_match = False
        
        # Check if MRAM is all zeros (common issue)
        mram_is_zeros = all(
            all(abs(val) < 0.001 for val in row)
            for row in sim_mram_rows[:sample_rows]
        )
        
        if mram_is_zeros:
            print("\n❌ WARNING: MRAM appears to be all zeros!")
            print("   This suggests H_PREFETCH_M did not load weights correctly.")
        elif weight_all_match:
            print("\n✅ Weight matrix loaded correctly!")
        else:
            print(f"\n⚠️  Weight matrix has some mismatches (sampled {sample_rows} rows)")
    else:
        print("⚠️  Unable to compare weights (missing data)")
    
    # ================================================================================
    # Compare Output Results  
    # ================================================================================
    print("\n" + "=" * 80)
    print("Output Result Comparison")
    print("=" * 80)
    
    output_results = []
    
    if golden_outputs and sim_output_rows:
        # Output is stored in stride mode similar to input
        # Row 8-11: Batch 0-3's first 64 elements
        # Row 12-15: Batch 0-3's last 64 elements (if exists)
        
        for batch_idx in range(min(4, len(golden_outputs))):
            if batch_idx >= len(golden_outputs):
                break
            
            golden_output = golden_outputs[batch_idx]
            
            if len(golden_output) != 128:
                print(f"⚠️  Golden output batch {batch_idx} has abnormal size: {len(golden_output)}")
                continue
            
            # Get corresponding simulation rows
            if batch_idx < len(sim_output_rows) and (batch_idx + 4) < len(sim_output_rows):
                sim_first = sim_output_rows[batch_idx]
                sim_second = sim_output_rows[batch_idx + 4] if (batch_idx + 4) < len(sim_output_rows) else []
            else:
                sim_first = []
                sim_second = []
            
            batch_matches = 0
            batch_total = 128
            
            # Compare first 64 elements
            for i in range(64):
                if i < len(golden_output) and i < len(sim_first):
                    match, diff = compare_with_tolerance(golden_output[i], sim_first[i], tolerance=0.2)
                    if match:
                        batch_matches += 1
            
            # Compare last 64 elements
            for i in range(64):
                if (64 + i) < len(golden_output) and i < len(sim_second):
                    match, diff = compare_with_tolerance(golden_output[64 + i], sim_second[i], tolerance=0.2)
                    if match:
                        batch_matches += 1
            
            match_rate = batch_matches * 100.0 / batch_total
            status = "✅" if match_rate >= 95 else "⚠️" if match_rate >= 50 else "❌"
            print(f"{status} Output Batch {batch_idx}: {batch_matches}/{batch_total} ({match_rate:.1f}%)")
            
            output_results.append({
                'batch': batch_idx,
                'matches': batch_matches,
                'total': batch_total,
                'rate': match_rate
            })
        
        # Check if output is all zeros (common issue)
        output_is_zeros = all(
            all(abs(val) < 0.001 for val in row)
            for row in sim_output_rows[:4] if row
        )
        
        if output_is_zeros:
            print("\n❌ WARNING: Output appears to be all zeros!")
            print("   This suggests M_MM/M_MM_WO did not compute/write results correctly.")
        elif output_results and sum(r['matches'] for r in output_results) == sum(r['total'] for r in output_results):
            print("\n🎉 Output results match perfectly!")
    else:
        print("⚠️  Unable to compare output results (missing data)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Overall Summary")
    print("=" * 80)
    
    print("\n📊 Input Tensor:")
    for result in batch_results:
        status = "✅" if result['rate'] >= 95 else "⚠️" if result['rate'] >= 90 else "❌"
        print(f"  {status} Batch {result['batch']}: {result['matches']}/{result['total']} "
              f"({result['rate']:.1f}%)")
    
    input_overall_rate = total_matches * 100.0 / total_elements if total_elements > 0 else 0
    input_status = "✅" if input_overall_rate >= 95 else "⚠️" if input_overall_rate >= 90 else "❌"
    print(f"\n  {input_status} Input Overall: {total_matches}/{total_elements} ({input_overall_rate:.1f}%)")
    
    if output_results:
        print("\n📊 Output Results:")
        for result in output_results:
            status = "✅" if result['rate'] >= 95 else "⚠️" if result['rate'] >= 50 else "❌"
            print(f"  {status} Batch {result['batch']}: {result['matches']}/{result['total']} "
                  f"({result['rate']:.1f}%)")
        
        output_total_matches = sum(r['matches'] for r in output_results)
        output_total_elements = sum(r['total'] for r in output_results)
        output_overall_rate = output_total_matches * 100.0 / output_total_elements if output_total_elements > 0 else 0
        output_status = "✅" if output_overall_rate >= 95 else "⚠️" if output_overall_rate >= 50 else "❌"
        print(f"\n  {output_status} Output Overall: {output_total_matches}/{output_total_elements} ({output_overall_rate:.1f}%)")
    
    print("\n" + "=" * 80)
    
    # Memory layout explanation
    print()
    print("Memory Layout Reference:")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ VRAM (Input & Output):                                  │")
    print("│   Row 0-3:   Input Batch 0-3 [0:64]                     │")
    print("│   Row 4-7:   Input Batch 0-3 [64:128]                   │")
    print("│   Row 8-11:  Output Batch 0-3 [0:64]                    │")
    print("│   Row 12-15: Output Batch 0-3 [64:128]                  │")
    print("│                                                          │")
    print("│ MRAM (Weights):                                          │")
    print("│   Row 0+: Weight Matrix (128x128)                       │")
    print("└─────────────────────────────────────────────────────────┘")
    print()
    
    # Final conclusion
    issues = []
    if input_overall_rate < 95:
        issues.append("Input loading")
    
    if golden_weights and sim_mram_rows:
        mram_is_zeros = all(
            all(abs(val) < 0.001 for val in row)
            for row in sim_mram_rows[:min(3, len(sim_mram_rows))]
        )
        if mram_is_zeros:
            issues.append("Weight loading (MRAM all zeros)")
    
    if output_results:
        output_total_matches = sum(r['matches'] for r in output_results)
        output_total_elements = sum(r['total'] for r in output_results)
        output_rate = output_total_matches * 100.0 / output_total_elements if output_total_elements > 0 else 0
        if output_rate < 50:
            issues.append("Output computation")
    
    if not issues:
        print("🎉 All checks passed! Simulation is working correctly!")
        return 0
    else:
        print(f"❌ Issues detected in: {', '.join(issues)}")
        print("\nDebugging suggestions:")
        if "Input loading" in issues:
            print("  - Check H_PREFETCH_V instruction and preload_act_asm")
        if "Weight loading" in issues:
            print("  - Check H_PREFETCH_M instruction and weight HBM addressing")
        if "Output computation" in issues:
            print("  - Check M_MM and M_MM_WO instructions")
            print("  - Verify m_accum accumulator is working")
        return 1

if __name__ == "__main__":
    sys.exit(main())

