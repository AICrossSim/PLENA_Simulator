import os
from typing import Dict, List, Any, Optional

def _report_flash_attn_utilization(node: Dict[str, Any], model_info: Dict[str, Any], context_len: int, M: int, N: int, K: int) -> None:
    """
    Report the utilization of flash attention for a given node.
    """
    dims = node["dimensions"]
    batch_size = model_info["batch_size"]
    hidden_size = dims["hidden_size"]
    num_attn_heads = dims["num_attention_heads"]
    num_kv_heads = dims["num_key_value_heads"]

    head_dim = dims["head_dim"]
    input_token_size = context_len
    theoretical_operation = 0
    attainable_operation = 0
    overall_operation_amount = 0
    
    # Decoding
    # Projection
    operation_amount = ((head_dim * num_attn_heads)  // M) * ( hidden_size // K) + ((head_dim * num_kv_heads) // M) * ( hidden_size// K) * 2
    overall_operation_amount    += operation_amount
    attainable_operation        += operation_amount * (M * K * batch_size)
    theoretical_operation       += operation_amount * (M * K * N)
    breakpoint()
    # QKT
    operation_amount =  batch_size * num_attn_heads * (head_dim // K) * (input_token_size // N)
    overall_operation_amount    += operation_amount
    attainable_operation        += operation_amount * (M * K)
    theoretical_operation       += operation_amount * (M * K * N)

    # PV
    operation_amount =  batch_size * num_attn_heads * (input_token_size // K) * (head_dim // N)
    overall_operation_amount    += operation_amount
    attainable_operation        += operation_amount * (M * K)
    theoretical_operation       += operation_amount * (M * K * N)
    
    return [operation_amount, attainable_operation, theoretical_operation]



def _report_embedding_utilization(node: Dict[str, Any], model_info: Dict[str, Any], M: int, N: int, K: int) -> None:
    """
    Report the utilization of flash attention for a given node.
    """
    
    dims = node["dimensions"]
    batch_size = model_info["batch_size"]
    hidden_size = model_info["hidden_size"]

    theoretical_operation = 0
    attainable_operation = 0

    # Assuming Decoding only
    operation_amount = (hidden_size // M) * (hidden_size // K)
    attainable_operation += operation_amount * (M * K * batch_size)
    theoretical_operation += operation_amount * (M * K * N)

    return [operation_amount, attainable_operation, theoretical_operation]



def _report_ffn_utilization(node: Dict[str, Any], model_info: Dict[str, Any], M: int, N: int, K: int) -> None:
    """
    Report the utilization of flash attention for a given node.
    """

    dims = node["dimensions"]
    batch_size = model_info["batch_size"]
    hidden_size = dims["hidden_size"]
    intermediate_size = dims["intermediate_size"]
    overall_operation_amount = 0
    theoretical_operation = 0
    attainable_operation = 0

    # Up Projection
    operation_amount = (intermediate_size // M) * (hidden_size // K)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (M * K * batch_size)
    theoretical_operation += operation_amount * (M * K * N)

    # Gate Projection
    operation_amount = (intermediate_size // M) * (hidden_size // K)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (M * K * batch_size)
    theoretical_operation += operation_amount * (M * K * N)

    # Down Projection
    operation_amount = (hidden_size // M) * (intermediate_size // K)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (M * K * batch_size)
    theoretical_operation += operation_amount * (M * K * N)

    return [overall_operation_amount, attainable_operation, theoretical_operation]


def _report_utilization(node: Dict[str, Any], model_info: Dict[str, Any], M: int, K: int, N: int) -> str:
    """Generate assembly code for a single symbolic graph node."""
    operation_type = node["operation_type"]
    node_name = node["name"]
    gemm_operation = 0

    if operation_type == "embedding":
        return _report_embedding_utilization(node, model_info, M, K, N)
    elif operation_type == "attention":
        return _report_flash_attn_utilization(node, model_info, 1024, M, K, N)
    elif operation_type == "ffn":
        return _report_ffn_utilization(node, model_info, M, K, N)
    else:
        return [0, 0, 0]
    
def _report_lm_head_utilization(model_info: Dict[str, Any], M: int, K: int, N: int) -> str:
    """
    Report the utilization of LM head for a given node.
    """
    batch_size = model_info["batch_size"]
    vocab_size = model_info.get("vocab_size", 128256)
    hidden_size = model_info["hidden_size"]

    theoretical_operation = 0
    attainable_operation = 0

    # Assuming Decoding only
    operation_amount = (vocab_size // M) * (hidden_size // K)
    attainable_operation += operation_amount * (M * K * batch_size)
    theoretical_operation += operation_amount * (M * K * N)

    return [operation_amount, attainable_operation, theoretical_operation]
    


def analyse_overall_utilization(symbolic_graph: Dict[str, Any], model_info: Dict[str, Any], M: int, K: int, N: int) -> str:
    """
    Transform the complete symbolic graph into assembly code.

    Args:
        symbolic_graph: The symbolic graph from LLMModelParser
        model_info: Model metadata for header generation

    Returns:
        Complete assembly program as string
    """
    # Process each node in execution order
    nodes = symbolic_graph["nodes"]
    execution_order = symbolic_graph["execution_order"]

    # Create a mapping from node names to nodes for efficient lookup
    node_map = {node["name"]: node for node in nodes}

    overall_operations = {"embedding": 0, "attention": 0, "ffn": 0, "lm_head": 0}
    overall_attainable_FLOPS = {"embedding": 0, "attention": 0, "ffn": 0, "lm_head": 0}
    overall_theoretical_FLOPS = {"embedding": 0, "attention": 0, "ffn": 0, "lm_head": 0}

    # Generate code for each node in execution order
    for node_name in execution_order:
        if node_name in node_map:
            node = node_map[node_name]
            single_op_operation = _report_utilization(node, model_info, M, K, N)
            operation_type = node["operation_type"]
            if operation_type in ["embedding", "attention", "ffn"]:
                overall_operations[operation_type] += single_op_operation[0]
                overall_attainable_FLOPS[operation_type] += single_op_operation[1]
                overall_theoretical_FLOPS[operation_type] += single_op_operation[2]

    single_op_operation = _report_lm_head_utilization(model_info, M, K, N)
    overall_operations["lm_head"] += single_op_operation[0]
    overall_attainable_FLOPS["lm_head"] += single_op_operation[1]
    overall_theoretical_FLOPS["lm_head"] += single_op_operation[2]

    return {
        "operations": overall_operations,
        "attainable_FLOPS": overall_attainable_FLOPS,
        "theoretical_FLOPS": overall_theoretical_FLOPS
    }