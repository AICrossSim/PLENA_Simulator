import sys
import os
import torch
from transformers import LlamaConfig, LlamaForCausalLM
from typing import Dict, Any, List

# Add the parent directory to the path so we can import the parser
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parser.llm_parser import LLMModelParser


def get_actual_execution_shapes(model, input_ids, num_layers, hidden_size):
    """Execute the Llama model and capture intermediate tensor shapes using hooks"""
    shapes = {}

    # Hook to capture shapes
    def make_hook(layer_name):
        def hook(module, input, output=None):
            if isinstance(input, tuple) and len(input) > 0 and hasattr(input[0], "shape"):
                input_shape = list(input[0].shape)
            elif hasattr(input, "shape"):
                input_shape = list(input.shape)
            else:
                input_shape = []

            if output is not None:
                if isinstance(output, tuple) and len(output) > 0:
                    output_shape = list(output[0].shape)
                elif hasattr(output, "shape"):
                    output_shape = list(output.shape)
                else:
                    output_shape = []
            else:
                output_shape = []

            shapes[layer_name] = {"input_shape": input_shape, "output_shape": output_shape}

        return hook

    # Register hooks
    hooks = []

    # Embedding hook
    hooks.append(model.model.embed_tokens.register_forward_hook(make_hook("embed_tokens")))

    # Layer hooks
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]

        # Input layernorm
        hooks.append(layer.input_layernorm.register_forward_hook(make_hook(f"layer_{layer_idx}_input_layernorm")))

        # Self-attention - hook individual projection layers instead of the whole attention module
        # Hook q_proj to get the input to attention
        hooks.append(layer.self_attn.q_proj.register_forward_hook(make_hook(f"layer_{layer_idx}_q_proj")))

        # Hook o_proj to get the output of attention (before residual)
        hooks.append(layer.self_attn.o_proj.register_forward_hook(make_hook(f"layer_{layer_idx}_o_proj")))

        # Also hook the full attention module with special handling
        def make_attention_hook(layer_name):
            def hook(module, input, output):
                # For attention, we'll manually set the input shape to match the hidden state
                # since it's called with keyword arguments
                batch_size, seq_len = input_ids.shape
                input_shape = [batch_size, seq_len, hidden_size]

                if isinstance(output, tuple) and len(output) > 0:
                    output_shape = list(output[0].shape)
                else:
                    output_shape = []

                shapes[layer_name] = {"input_shape": input_shape, "output_shape": output_shape}

            return hook

        hooks.append(layer.self_attn.register_forward_hook(make_attention_hook(f"layer_{layer_idx}_self_attn")))

        # Post-attention layernorm
        hooks.append(
            layer.post_attention_layernorm.register_forward_hook(
                make_hook(f"layer_{layer_idx}_post_attention_layernorm")
            )
        )

        # MLP
        hooks.append(layer.mlp.register_forward_hook(make_hook(f"layer_{layer_idx}_mlp")))

    # Final layer norm
    hooks.append(model.model.norm.register_forward_hook(make_hook("final_layernorm")))

    # Forward pass
    with torch.no_grad():
        output = model(input_ids)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Add residual connection shapes manually (since they're not modules)
    batch_size, seq_len = input_ids.shape
    residual_shape = [batch_size, seq_len, hidden_size]

    for layer_idx in range(num_layers):
        shapes[f"layer_{layer_idx}_attn_residual"] = {
            "input_shape": [residual_shape, residual_shape],
            "output_shape": residual_shape,
        }
        shapes[f"layer_{layer_idx}_ffn_residual"] = {
            "input_shape": [residual_shape, residual_shape],
            "output_shape": residual_shape,
        }

    return shapes


def test_model():
    """Test that symbolic graph shapes match actual LlamaForCausalLM execution shapes"""
    print("Testing create_symbolic_graph against actual LlamaForCausalLM execution...")

    # Test parameters
    vocab_size = 1000
    hidden_size = 512
    num_layers = 2
    num_heads = 8
    intermediate_size = 1376
    eps = 1e-6
    batch_size = 2
    seq_len = 128

    # Create LlamaConfig
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=2048,
        rms_norm_eps=eps,
        hidden_act="silu",
        torch_dtype=torch.float32,
    )

    # Create actual LlamaForCausalLM model
    print("Creating LlamaForCausalLM model...")
    model = LlamaForCausalLM(config)
    model.eval()

    # Create parser using the same config
    parser = LLMModelParser("dummy-model")
    parser.config = config
    parser.model = model

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Get symbolic graph
    print(f"Creating symbolic graph for {num_layers}-layer model...")
    symbolic_graph = parser.create_symbolic_graph(batch_size=batch_size, seq_len=seq_len)

    # Execute actual model and track intermediate shapes
    print("Executing LlamaForCausalLM model and tracking shapes...")
    actual_shapes = get_actual_execution_shapes(model, input_ids, num_layers, hidden_size)

    # Verify that symbolic graph shapes match actual execution
    print("Verifying shapes match...")
    nodes = symbolic_graph["nodes"]

    all_passed = True
    for i, node in enumerate(nodes):
        node_name = node["name"]

        # Special handling for embed_tokens since it's not captured by hooks
        if node_name == "embed_tokens":
            # Manually verify embedding shapes
            symbolic_input_shape = node["input_shape"]
            symbolic_output_shape = node["output_shape"]
            expected_input_shape = [batch_size, seq_len]
            expected_output_shape = [batch_size, seq_len, hidden_size]

            if symbolic_input_shape == expected_input_shape:
                print(f"✅ Input shape match for {node_name}: {symbolic_input_shape}")
            else:
                print(
                    f"❌ Input shape mismatch for {node_name}: symbolic={symbolic_input_shape}, expected={expected_input_shape}"
                )
                all_passed = False

            if symbolic_output_shape == expected_output_shape:
                print(f"✅ Output shape match for {node_name}: {symbolic_output_shape}")
            else:
                print(
                    f"❌ Output shape mismatch for {node_name}: symbolic={symbolic_output_shape}, expected={expected_output_shape}"
                )
                all_passed = False
            continue

        # Special handling for self_attn since it's called with keyword arguments
        if node_name.endswith("_self_attn"):
            # Use the q_proj input shape as the attention input shape
            layer_prefix = node_name.replace("_self_attn", "")
            q_proj_name = f"{layer_prefix}_q_proj"

            if q_proj_name in actual_shapes:
                # TODO: why do we assess q_proj as input but node_name as output?
                actual_input_shape = actual_shapes[q_proj_name]["input_shape"]
                actual_output_shape = actual_shapes[node_name]["output_shape"]

                symbolic_input_shape = node["input_shape"]
                symbolic_output_shape = node["output_shape"]

                if symbolic_input_shape == actual_input_shape:
                    print(f"✅ Input shape match for {node_name}: {symbolic_input_shape}")
                else:
                    print(
                        f"❌ Input shape mismatch for {node_name}: symbolic={symbolic_input_shape}, actual={actual_input_shape}"
                    )
                    all_passed = False

                if symbolic_output_shape == actual_output_shape:
                    print(f"✅ Output shape match for {node_name}: {symbolic_output_shape}")
                else:
                    print(
                        f"❌ Output shape mismatch for {node_name}: symbolic={symbolic_output_shape}, actual={actual_output_shape}"
                    )
                    all_passed = False
                continue
            else:
                print(f"⚠️  No q_proj data found for {node_name}")
                continue

        # Skip if we don't have actual shapes for this node
        if node_name not in actual_shapes:
            print(f"⚠️  Skipping {node_name} - no actual shape captured")
            continue

        actual_input_shape = actual_shapes[node_name]["input_shape"]
        actual_output_shape = actual_shapes[node_name]["output_shape"]

        symbolic_input_shape = node["input_shape"]
        symbolic_output_shape = node["output_shape"]

        # Check input shapes
        if isinstance(symbolic_input_shape[0], list):
            # Multiple inputs (like residual connections)
            if symbolic_input_shape != actual_input_shape:
                print(
                    f"❌ Input shape mismatch for {node_name}: symbolic={symbolic_input_shape}, actual={actual_input_shape}"
                )
                all_passed = False
            else:
                print(f"✅ Input shape match for {node_name}: {symbolic_input_shape}")
        else:
            # Single input - handle case where hooks might not capture input correctly
            if not actual_input_shape:
                # For self-attention, the input should be the same as the previous layer's output
                expected_input_shape = [batch_size, seq_len, hidden_size]
                if symbolic_input_shape == expected_input_shape:
                    print(f"✅ Input shape match for {node_name}: {symbolic_input_shape} (hook missed input)")
                else:
                    print(
                        f"❌ Input shape mismatch for {node_name}: symbolic={symbolic_input_shape}, expected={expected_input_shape}"
                    )
                    all_passed = False
            elif symbolic_input_shape != actual_input_shape:
                print(
                    f"❌ Input shape mismatch for {node_name}: symbolic={symbolic_input_shape}, actual={actual_input_shape}"
                )
                all_passed = False
            else:
                print(f"✅ Input shape match for {node_name}: {symbolic_input_shape}")

        # Check output shapes
        if symbolic_output_shape != actual_output_shape:
            print(
                f"❌ Output shape mismatch for {node_name}: symbolic={symbolic_output_shape}, actual={actual_output_shape}"
            )
            all_passed = False
        else:
            print(f"✅ Output shape match for {node_name}: {symbolic_output_shape}")

    # Test execution order
    print("\nTesting execution order...")
    execution_order = symbolic_graph["execution_order"]
    expected_sequence = [
        "embed_tokens",
        "layer_0_input_layernorm",
        "layer_0_self_attn",
        "layer_0_attn_residual",
        "layer_0_post_attention_layernorm",
        "layer_0_mlp",
        "layer_0_ffn_residual",
        "layer_1_input_layernorm",
        "layer_1_self_attn",
        "layer_1_attn_residual",
        "layer_1_post_attention_layernorm",
        "layer_1_mlp",
        "layer_1_ffn_residual",
        "final_layernorm",
    ]

    if execution_order == expected_sequence:
        print("✅ Execution order is correct")
    else:
        print(f"❌ Execution order mismatch:")
        print(f"   Expected: {expected_sequence}")
        print(f"   Actual:   {execution_order}")
        all_passed = False

    # Test end-to-end model execution
    print("\nTesting end-to-end model execution...")
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits
        expected_shape = [batch_size, seq_len, vocab_size]
        if list(logits.shape) == expected_shape:
            print(f"✅ End-to-end output shape correct: {expected_shape}")
        else:
            print(f"❌ End-to-end output shape mismatch: expected={expected_shape}, actual={list(logits.shape)}")
            all_passed = False

    # Test different batch sizes and sequence lengths
    print("\nTesting different input dimensions...")
    test_cases = [
        (1, 64),  # Small batch, short sequence
        (4, 256),  # Medium batch, medium sequence
    ]

    for test_batch_size, test_seq_len in test_cases:
        symbolic_graph = parser.create_symbolic_graph(batch_size=test_batch_size, seq_len=test_seq_len)

        # Verify embedding shapes
        embed_node = symbolic_graph["nodes"][0]
        if embed_node["input_shape"] == [test_batch_size, test_seq_len]:
            print(f"✅ Embedding input shape correct for ({test_batch_size}, {test_seq_len})")
        else:
            print(f"❌ Embedding input shape incorrect for ({test_batch_size}, {test_seq_len})")
            all_passed = False

        if embed_node["output_shape"] == [test_batch_size, test_seq_len, hidden_size]:
            print(f"✅ Embedding output shape correct for ({test_batch_size}, {test_seq_len})")
        else:
            print(f"❌ Embedding output shape incorrect for ({test_batch_size}, {test_seq_len})")
            all_passed = False

    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL TESTS PASSED! create_symbolic_graph is correct for LlamaForCausalLM architecture")
    else:
        print("❌ SOME TESTS FAILED! create_symbolic_graph needs fixes")
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    test_model()