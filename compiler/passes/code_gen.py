"""
Code generation pass for LLM symbolic graph to assembly transformation.

This module transforms the symbolic graph representation of a LLM model
into assembly code using predefined templates for different operation types.
"""

import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from asm_templates import (
    projection_asm,
    # flash_attn_asm,
    ffn_asm,
    rms_norm_asm,
    elementwise_add_asm,
    embedding_asm
)


def _load_template(template_name: str) -> str:
    """Load assembly template from file."""
    templates_dir = Path(__file__).parent.parent / "asm_templates"
    template_path = templates_dir / f"{template_name}.asm"

    if not template_path.exists():
        raise FileNotFoundError(f"Template {template_name}.asm not found in {templates_dir}")

    with open(template_path, "r") as f:
        return f.read()


def _generate_embedding_code(node: Dict[str, Any], model_info: Dict[str, Any], hardware_config: Dict[str, Any], scheduler: Dict[str, Any]) -> str:
    """Generate assembly code for embedding operations."""
    vocab_size = model_info["vocab_size"]
    dim = node["dimensions"]
    # TODO need to add a dot product at the end.
    code = f"""
; Embedding lookup: vocab_size={vocab_size}
; Input: token_ids, Output: embedded_vectors
"""
    code += embedding_asm(
        mlen    = hardware_config.get("mlen", 16),
        blen    = hardware_config.get("blen", 16),
        batch                   = model_info.get("batch_size", 1),
        hidden_size             = dim["hidden_size"],
        alive_registers         = hardware_config.get("alive_registers", [1, 2, 3, 4]),
        voc_table_row_size      = vocab_size,
        activation_base_address = scheduler.get("activation_base_address", 0),
        voc_table_base_addr_reg_index = scheduler.get("register_assignment", {}).get("hbm_addr_reg", {}).get("token_table_offset", 0),
        input_ids = [1 for _ in range(model_info.get("batch_size", 1))]
    )

    return code.strip()


def _generate_attention_code(node: Dict[str, Any], model_info: Dict[str, Any], hardware_config: Dict[str, Any], scheduler: Dict[str, Any]) -> str:
    """Generate assembly code for attention operations."""


    dims = node["dimensions"]
    hidden_size = dims["hidden_size"]
    num_heads = dims["num_attention_heads"]
    head_dim = dims["head_dim"]

    # # TODO: break flash attention down into multiple smaller templates for loop
    # # TODO: Templates in asm_templates/flash_attention_tr_loop.asm + asm_templates/flash_attention_tc_loop.asm
    code = f"""
    # ; Self-attention: hidden_size={hidden_size}, num_heads={num_heads}, head_dim={head_dim}
    # ; Q, K, V projections and attention computation
    # """
    code += projection_asm(
        mlen = hardware_config.get("MLEN", 16),
        blen = hardware_config.get("BLEN", 16),
        batch = model_info.get("batch", 1),
        hidden_size = hidden_size,
        alive_registers = [1,2,3,4,5,6,7,8],
        head_dim = head_dim,
        w_base_hbm_offset_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("q_weight_offset", 0),
        rope_hbm_offset_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("rope_params_offset", 0),
        rope_on_chip_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block3", 0),
        activation_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block1", 0),
        result_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block2", 0),
        rope_enabled = True
    )

    code += projection_asm(
        mlen = hardware_config.get("MLEN", 16),
        blen = hardware_config.get("BLEN", 16),
        batch = model_info.get("batch", 1),
        hidden_size = hidden_size,
        alive_registers = [1,2,3,4,5,6,7,8],
        head_dim = head_dim,
        w_base_hbm_offset_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("k_weight_offset", 0),
        rope_hbm_offset_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("rope_params_offset", 0),
        rope_on_chip_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block3", 0),
        activation_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block1", 0),
        result_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block2", 0),
        rope_enabled = True
    )

    code += projection_asm(
        mlen = hardware_config.get("MLEN", 16),
        blen = hardware_config.get("BLEN", 16),
        batch = model_info.get("batch", 1),
        hidden_size = hidden_size,
        alive_registers = [1,2,3,4,5,6,7,8],
        head_dim = head_dim,
        w_base_hbm_offset_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("v_weight_offset", 0),
        rope_hbm_offset_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("rope_params_offset", 0),
        rope_on_chip_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block3", 0),
        activation_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block1", 0),
        result_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block2", 0),
        rope_enabled = False
    )
    
    # code += flash_attn_asm()
    
    return code.strip()


def _generate_ffn_code(node: Dict[str, Any], model_info: Dict[str, Any], hardware_config: Dict[str, Any], scheduler: Dict[str, Any]) -> str:
    """Generate assembly code for FFN/MLP operations."""

    dims = node["dimensions"]
    hidden_size = dims["hidden_size"]
    intermediate_size = dims["intermediate_size"]
    activation = dims["activation"]

    code = f"""
    ; FFN/MLP: hidden_size={hidden_size}, intermediate_size={intermediate_size}, activation={activation}
    ; Gate and Up projections
    """

    code += ffn_asm(
        mlen = hardware_config.get("MLEN", 16),
        blen = hardware_config.get("BLEN", 16),
        batch = model_info.get("batch", 1),
        hidden_size = hidden_size,
        alive_registers = [1, 2, 3, 4, 5, 6],
        weight_hbm_offset_reg = scheduler["register_assignment"].get("hbm_addr_reg", {}).get("ffn_weight_offset", 0),
        intermediate_size = model_info.get("intermediate_size", 4096),
        activation_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block1", 0),
        const_address = scheduler["memory_layout"].get("fp_sram", {}).get("silu_e", 0),
        result_base_address = scheduler["memory_layout"].get("vector_sram_addr", {}).get("block5", 0),
    )
    return code.strip()


def _generate_normalization_code(node: Dict[str, Any], model_info: Dict[str, Any], hardware_config: Dict[str, Any], scheduler: Dict[str, Any]) -> str:
    """Generate assembly code for normalization operations."""

    dims = node["dimensions"]
    hidden_size = dims["normalized_shape"]
    eps_offset = scheduler.get("fp_sram", {}).get("eps", 0)
    reci_hid_offset = scheduler.get("fp_sram", {}).get("hid_reciprocal", 0)
    code = f"""
; Normalization: hidden_size={hidden_size}`
; Layer normalization
"""
    code += rms_norm_asm(
        _eps_offset = eps_offset,
        reci_hid_offset = reci_hid_offset,
        alive_registers = [1, 2, 3],
        activation_base_address = scheduler.get("vector_sram_addr", {}).get("block1", 0),
        scratchpad_base_address = scheduler.get("vector_sram_addr", {}).get("block2", 0),
        vlen = hardware_config.get("vlen", 16),
        hidden_dim = hidden_size
    )

    return code.strip()


def _generate_elementwise_add_code(node: Dict[str, Any], model_info: Dict[str, Any], hardware_config: Dict[str, Any], scheduler: Dict[str, Any]) -> str:
    """Generate assembly code for elementwise addition (residual connections)."""
    dims = node["dimensions"]
    shape = dims["shape"]

    code = f"""
    ; Elementwise addition (residual connection): shape={shape}
    """
    code += elementwise_add_asm(
        vlen=hardware_config.get("VLEN", 16),
        hidden_size=model_info["hidden_size"],
        batch=model_info.get("batch", 1),
        alive_registers=hardware_config.get("alive_registers", [1, 2, 3]),
        stored_activation_base_address=scheduler.get("vector_sram_addr", {}).get("block1", 0),
        previous_activation_base_address=scheduler.get("vector_sram_addr", {}).get("block2", 0),
        previous_act_on_chip_addr_reg_index=scheduler["register_assignment"].get("hbm_addr_reg", {}).get("previous_activation_offset", 0)
    )
    return code.strip()


def _generate_node_code(node: Dict[str, Any], model_info: Dict[str, Any], hardware_config: Dict[str, Any], scheduler: Dict[str, Any]) -> str:
    """Generate assembly code for a single symbolic graph node."""
    operation_type = node["operation_type"]
    node_name = node["name"]

    header = f"\n; === {node_name} ({operation_type}) ===\n"

    if operation_type == "embedding":
        return header + _generate_embedding_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "attention":
        return header + _generate_attention_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "ffn":
        return header + _generate_ffn_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "normalization":
        return header + _generate_normalization_code(node, model_info, hardware_config, scheduler)
    elif operation_type == "elementwise_add":
        return header + _generate_elementwise_add_code(node, model_info, hardware_config, scheduler)
    else:
        raise ValueError(f"Unknown operation type: {operation_type}")


def _generate_program_header(model_info: Dict[str, Any]) -> str:
    """Generate program header with model information."""
    return f"""
; Generated assembly code for LLM model
; Model: {model_info.get("model_name", "Unknown")}
; Architecture: {model_info.get("architecture", "Unknown")}
; Hidden size: {model_info.get("hidden_size", "Unknown")}
; Number of layers: {model_info.get("num_layers", "Unknown")}
; Generated by LLM Compiler
"""


def _generate_program_footer() -> str:
    """Generate program footer."""
    return """
    ; Program termination
"""


def code_gen_pass(symbolic_graph: Dict[str, Any], model_info: Dict[str, Any], hardware_config: Dict[str, Any], scheduler: Dict[str, Any]) -> str:
    """
    Transform the complete symbolic graph into assembly code.

    Args:
        symbolic_graph: The symbolic graph from LLMModelParser
        model_info: Model metadata for header generation

    Returns:
        Complete assembly program as string
    """
    # Generate program header
    asm_code = [_generate_program_header(model_info)]

    # Process each node in execution order
    nodes = symbolic_graph["nodes"]
    execution_order = symbolic_graph["execution_order"]

    # Create a mapping from node names to nodes for efficient lookup
    node_map = {node["name"]: node for node in nodes}

    # Generate code for each node in execution order
    for node_name in execution_order:
        if node_name in node_map:
            node = node_map[node_name]
            node_code = _generate_node_code(node, model_info, hardware_config, scheduler)
            asm_code.append(node_code)

    # Add program footer
    
    asm_code.append(_generate_program_footer())
    return "\n".join(asm_code)