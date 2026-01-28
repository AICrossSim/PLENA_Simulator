#!/usr/bin/env python3

import sys
from pathlib import Path
from parser import LLMModelParser, HardwareParser
from passes.code_gen import code_gen_pass
from passes.utilization_report import analyse_overall_utilization
from scheduler import gen_scheduler


def run():
    if len(sys.argv) < 3:
        print("Usage: python runner.py <model_name_or_path> <output_file.asm>")
        print("Example: python runner.py AICrossSim/clm-60m output.asm")
        return
    mode = sys.argv[1]
    model_path = sys.argv[2]
    output_file = sys.argv[3]
    hardware_config_path = Path(__file__).resolve().parents[1] / "src" / "definitions" / "configuration.svh"
    precision_config_path = Path(__file__).resolve().parents[1] / "src" / "definitions" / "precision.svh"
    mem_layout_lib_path = Path(__file__).resolve().parents[0] / "scheduler" / "mem_layout_lib.json"
    reg_assignment_lib_path = Path(__file__).resolve().parents[0] / "scheduler" / "reg_assignment_lib.json"
    # Validate that output file ends with .asm
    if not output_file.endswith(".asm"):
        print("Error: Output file must end with .asm extension")
        print("Example: python runner.py AICrossSim/clm-60m output.asm")
        return

    print(f"Loading model: {model_path}")
    parser = LLMModelParser(model_path)

    parser.load_model()
    parser.print_summary()

    # Create symbolic graph
    symbolic_graph = parser.create_symbolic_graph()

    dimensions = parser.extract_critical_dimensions()

    # Print detailed symbolic graph
    parser.print_symbolic_graph_details()

    # Prepare model info for code generation
    model_info = {
        "model_name": model_path,
        "architecture": getattr(parser.config, "architectures", ["Unknown"])[0] if parser.config else "Unknown",
        "batch_size": 4,
        "context_length" : dimensions.get("max_position_embeddings", "Unknown"),
        "vocab_size": dimensions.get("vocab_size", "Unknown"),
        "hidden_size": dimensions.get("hidden_size", "Unknown"),
        "intermediate_size": dimensions.get("ffn", {}).get("intermediate_size", 4096),
        "num_key_value_heads": dimensions.get("attention", {}).get("num_key_value_heads", "Unknown"),
        "num_attention_heads": dimensions.get("attention", {}).get("num_attention_heads", "Unknown"),
        "num_layers": dimensions.get("num_hidden_layers", "Unknown"),
        "head_dim" : dimensions.get("hidden_size", "Unknown") // dimensions.get("num_attention_heads", 1),
        "eps": dimensions.get("rms_norm", {}).get("eps", 1e-6)
    }

    # Run code generation pass
    if mode == "utilization":
        M = 64
        K = 64
        N = 64
        print(f"\nRunning utilization analysis...")
        utilization_report = analyse_overall_utilization(symbolic_graph, model_info, M, K, N)
        print(f"Utilization Report:\n{utilization_report}")
        return
    
    hardware_config = HardwareParser(hardware_config_path, precision_config_path)
    scheduler = gen_scheduler(hardware_config, model_info, mem_layout_lib_path, reg_assignment_lib_path)
    print(f"\nRunning code generation pass...")
    generated_asm = code_gen_pass(symbolic_graph, model_info, hardware_config, scheduler)

    # Save generated code
    with open(output_file, "w") as f:
        f.write(generated_asm)

    print(f"Generated assembly code saved to: {output_file}")

    # Print a preview of the generated code
    print(f"\nGenerated code preview (first 20 lines):")
    print("=" * 50)
    lines = generated_asm.split("\n")
    for i, line in enumerate(lines[:20]):
        print(f"{i + 1:3d}: {line}")
    if len(lines) > 20:
        print(f"... and {len(lines) - 20} more lines")
    print("=" * 50)


if __name__ == "__main__":
    sys.exit(run())
