from typing import Any


def _report_flash_attn_utilization(
    node: dict[str, Any], model_info: dict[str, Any], context_len: int, m: int, n: int, k: int
) -> None:
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
    operation_amount = ((head_dim * num_attn_heads) // m) * (hidden_size // k) + ((head_dim * num_kv_heads) // m) * (
        hidden_size // k
    ) * 2
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)
    breakpoint()
    # QKT
    operation_amount = batch_size * num_attn_heads * (head_dim // k) * (input_token_size // n)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k)
    theoretical_operation += operation_amount * (m * k * n)

    # PV
    operation_amount = batch_size * num_attn_heads * (input_token_size // k) * (head_dim // n)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k)
    theoretical_operation += operation_amount * (m * k * n)

    return [operation_amount, attainable_operation, theoretical_operation]


def _report_embedding_utilization(node: dict[str, Any], model_info: dict[str, Any], m: int, n: int, k: int) -> None:
    """
    Report the utilization of flash attention for a given node.
    """

    batch_size = model_info["batch_size"]
    hidden_size = model_info["hidden_size"]

    theoretical_operation = 0
    attainable_operation = 0

    # Assuming Decoding only
    operation_amount = (hidden_size // m) * (hidden_size // k)
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)

    return [operation_amount, attainable_operation, theoretical_operation]


def _report_ffn_utilization(node: dict[str, Any], model_info: dict[str, Any], m: int, n: int, k: int) -> None:
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
    operation_amount = (intermediate_size // m) * (hidden_size // k)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)

    # Gate Projection
    operation_amount = (intermediate_size // m) * (hidden_size // k)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)

    # Down Projection
    operation_amount = (hidden_size // m) * (intermediate_size // k)
    overall_operation_amount += operation_amount
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)

    return [overall_operation_amount, attainable_operation, theoretical_operation]


def _report_utilization(node: dict[str, Any], model_info: dict[str, Any], m: int, k: int, n: int) -> str:
    """Generate assembly code for a single symbolic graph node."""
    operation_type = node["operation_type"]

    if operation_type == "embedding":
        return _report_embedding_utilization(node, model_info, m, k, n)
    elif operation_type == "attention":
        return _report_flash_attn_utilization(node, model_info, 1024, m, k, n)
    elif operation_type == "ffn":
        return _report_ffn_utilization(node, model_info, m, k, n)
    else:
        return [0, 0, 0]


def _report_lm_head_utilization(model_info: dict[str, Any], m: int, k: int, n: int) -> str:
    """
    Report the utilization of LM head for a given node.
    """
    batch_size = model_info["batch_size"]
    vocab_size = model_info.get("vocab_size", 128256)
    hidden_size = model_info["hidden_size"]

    theoretical_operation = 0
    attainable_operation = 0

    # Assuming Decoding only
    operation_amount = (vocab_size // m) * (hidden_size // k)
    attainable_operation += operation_amount * (m * k * batch_size)
    theoretical_operation += operation_amount * (m * k * n)

    return [operation_amount, attainable_operation, theoretical_operation]


def analyse_overall_utilization(
    symbolic_graph: dict[str, Any], model_info: dict[str, Any], m: int, k: int, n: int
) -> str:
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
    overall_attainable_flops = {"embedding": 0, "attention": 0, "ffn": 0, "lm_head": 0}
    overall_theoretical_flops = {"embedding": 0, "attention": 0, "ffn": 0, "lm_head": 0}

    # Generate code for each node in execution order
    for node_name in execution_order:
        if node_name in node_map:
            node = node_map[node_name]
            single_op_operation = _report_utilization(node, model_info, m, k, n)
            operation_type = node["operation_type"]
            if operation_type in ["embedding", "attention", "ffn"]:
                overall_operations[operation_type] += single_op_operation[0]
                overall_attainable_flops[operation_type] += single_op_operation[1]
                overall_theoretical_flops[operation_type] += single_op_operation[2]

    single_op_operation = _report_lm_head_utilization(model_info, m, k, n)
    overall_operations["lm_head"] += single_op_operation[0]
    overall_attainable_flops["lm_head"] += single_op_operation[1]
    overall_theoretical_flops["lm_head"] += single_op_operation[2]

    return {
        "operations": overall_operations,
        "attainable_FLOPS": overall_attainable_flops,
        "theoretical_FLOPS": overall_theoretical_flops,
    }
