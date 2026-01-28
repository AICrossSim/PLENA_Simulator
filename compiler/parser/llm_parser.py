from typing import Dict, Any, Optional, List
import torch
import torch.fx as fx
from transformers import AutoModel, AutoConfig


class LLMModelParser:
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.config = None
        self.model = None
        self.symbolic_graph = None

    def load_model(self):
        """Load the model and configuration from HuggingFace"""
        try:
            self.config = AutoConfig.from_pretrained(self.model_name_or_path)
            self.model = AutoModel.from_pretrained(self.model_name_or_path, torch_dtype=torch.float32)
            self.model.eval()
        except Exception as e:
            raise ValueError(f"Failed to load model {self.model_name_or_path}: {e}")

    def extract_critical_dimensions(self) -> Dict[str, Any]:
        """Extract dimensions for attention, RMSNorm, FFN operations"""
        if self.config is None:
            self.load_model()

        dimensions = {}

        # Common dimensions
        dimensions["vocab_size"] = getattr(self.config, "vocab_size", None)
        dimensions["hidden_size"] = getattr(self.config, "hidden_size", None)
        dimensions["num_hidden_layers"] = getattr(self.config, "num_hidden_layers", None)
        dimensions["max_position_embeddings"] = getattr(self.config, "max_position_embeddings", None)

        # Attention dimensions
        dimensions["attention"] = self._extract_attention_dimensions()

        # FFN dimensions
        dimensions["ffn"] = self._extract_ffn_dimensions()

        # RMSNorm dimensions
        dimensions["rms_norm"] = self._extract_rms_norm_dimensions()

        return dimensions

    def _extract_attention_dimensions(self) -> Dict[str, Any]:
        """Extract attention-specific dimensions"""
        attention_dims = {}

        # Multi-head attention parameters
        attention_dims["num_attention_heads"] = getattr(self.config, "num_attention_heads", None)
        attention_dims["num_key_value_heads"] = getattr(
            self.config, "num_key_value_heads", getattr(self.config, "num_attention_heads", None)
        )

        hidden_size = getattr(self.config, "hidden_size", 0)
        num_heads = getattr(self.config, "num_attention_heads", 1)

        if hidden_size and num_heads:
            attention_dims["head_dim"] = hidden_size // num_heads
            attention_dims["key_value_head_dim"] = hidden_size // attention_dims["num_key_value_heads"]

        return attention_dims

    def _extract_ffn_dimensions(self) -> Dict[str, Any]:
        """Extract FFN (Feed-Forward Network) dimensions"""
        ffn_dims = {}

        hidden_size = getattr(self.config, "hidden_size", 0)
        intermediate_size = getattr(self.config, "intermediate_size", hidden_size * 4)

        ffn_dims["hidden_size"] = hidden_size
        ffn_dims["intermediate_size"] = intermediate_size
        ffn_dims["activation"] = getattr(self.config, "hidden_act", "silu")

        return ffn_dims

    def _extract_rms_norm_dimensions(self) -> Dict[str, Any]:
        """Extract RMSNorm dimensions"""
        rms_dims = {}

        hidden_size = getattr(self.config, "hidden_size", 0)

        rms_dims["normalized_shape"] = hidden_size
        rms_dims["eps"] = getattr(self.config, "rms_norm_eps", 1e-6)

        return rms_dims

    def create_symbolic_graph(self, batch_size: int = 1, seq_len: int = 512) -> Dict[str, Any]:
        """Create a symbolic graph with execution orders"""
        # TODO: this is in fixed ordering and thus would only support only LlamaForCausalLM architecture such as AICrossSim/clm-60m that we know the detail
        # TODO: Additional work is needed to make it more flexible (maybe use MASEGraph or torch.fx)
        if self.model is None:
            self.load_model()

        symbolic_nodes = []
        execution_order = []
        order_counter = 0

        # Start with input embedding
        # Check for embed_tokens in different locations (model.embed_tokens or model.model.embed_tokens)
        has_embed_tokens = hasattr(self.model, "embed_tokens") or (
            hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens")
        )

        if has_embed_tokens:
            embed_info = {
                "name": "embed_tokens",
                "operation_type": "embedding",
                "operation_category": "embedding",
                "execution_order": order_counter,
                "input_shape": [batch_size, seq_len],  # input_ids shape
                "output_shape": [batch_size, seq_len, self.config.hidden_size],  # embedded tokens
                "dimensions": {"num_embeddings": self.config.vocab_size, "embedding_dim": self.config.hidden_size},
                "is_data_placeholder": True,
            }
            symbolic_nodes.append(embed_info)
            execution_order.append("embed_tokens")
            order_counter += 1

        # Process transformer layers
        num_layers = getattr(self.config, "num_hidden_layers", 0)

        for layer_idx in range(num_layers):
            current_shape = [batch_size, seq_len, self.config.hidden_size]

            # Input layer norm
            norm_info = {
                "name": f"layer_{layer_idx}_input_layernorm",
                "operation_type": "normalization",
                "operation_category": "normalization",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,  # normalization preserves shape
                "dimensions": {
                    "normalized_shape": self.config.hidden_size,
                    "eps": getattr(self.config, "rms_norm_eps", 1e-6),
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(norm_info)
            execution_order.append(f"layer_{layer_idx}_input_layernorm")
            order_counter += 1

            # Self-attention block (fused)
            attn_info = {
                "name": f"layer_{layer_idx}_self_attn",
                "operation_type": "attention",
                "operation_category": "attention",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,  # attention preserves shape
                "dimensions": {
                    "hidden_size": self.config.hidden_size,
                    "num_attention_heads": self.config.num_attention_heads,
                    "num_key_value_heads": getattr(self.config, "num_key_value_heads", self.config.num_attention_heads),
                    "head_dim": self.config.hidden_size // self.config.num_attention_heads,
                    "q_proj": {"in_features": self.config.hidden_size, "out_features": self.config.hidden_size},
                    "k_proj": {"in_features": self.config.hidden_size, "out_features": self.config.hidden_size},
                    "v_proj": {"in_features": self.config.hidden_size, "out_features": self.config.hidden_size},
                    "o_proj": {"in_features": self.config.hidden_size, "out_features": self.config.hidden_size},
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(attn_info)
            execution_order.append(f"layer_{layer_idx}_self_attn")
            order_counter += 1

            # Residual connection (attention)
            residual_info = {
                "name": f"layer_{layer_idx}_attn_residual",
                "operation_type": "elementwise_add",
                "operation_category": "elementwise_add",
                "execution_order": order_counter,
                "input_shape": [current_shape, current_shape],  # two inputs of same shape
                "output_shape": current_shape,  # elementwise add preserves shape
                "dimensions": {"shape": [self.config.hidden_size]},
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(residual_info)
            execution_order.append(f"layer_{layer_idx}_attn_residual")
            order_counter += 1

            # Post-attention layer norm
            post_norm_info = {
                "name": f"layer_{layer_idx}_post_attention_layernorm",
                "operation_type": "normalization",
                "operation_category": "normalization",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,  # normalization preserves shape
                "dimensions": {
                    "normalized_shape": self.config.hidden_size,
                    "eps": getattr(self.config, "rms_norm_eps", 1e-6),
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(post_norm_info)
            execution_order.append(f"layer_{layer_idx}_post_attention_layernorm")
            order_counter += 1

            # MLP/FFN block (fused)
            mlp_info = {
                "name": f"layer_{layer_idx}_mlp",
                "operation_type": "ffn",
                "operation_category": "ffn",
                "execution_order": order_counter,
                "input_shape": current_shape,
                "output_shape": current_shape,  # FFN preserves shape
                "dimensions": {
                    "hidden_size": self.config.hidden_size,
                    "intermediate_size": getattr(self.config, "intermediate_size", self.config.hidden_size * 4),
                    "activation": getattr(self.config, "hidden_act", "silu"),
                    "gate_proj": {
                        "in_features": self.config.hidden_size,
                        "out_features": getattr(self.config, "intermediate_size", self.config.hidden_size * 4),
                    },
                    "up_proj": {
                        "in_features": self.config.hidden_size,
                        "out_features": getattr(self.config, "intermediate_size", self.config.hidden_size * 4),
                    },
                    "down_proj": {
                        "in_features": getattr(self.config, "intermediate_size", self.config.hidden_size * 4),
                        "out_features": self.config.hidden_size,
                    },
                },
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(mlp_info)
            execution_order.append(f"layer_{layer_idx}_mlp")
            order_counter += 1

            # Residual connection (FFN)
            ffn_residual_info = {
                "name": f"layer_{layer_idx}_ffn_residual",
                "operation_type": "elementwise_add",
                "operation_category": "elementwise_add",
                "execution_order": order_counter,
                "input_shape": [current_shape, current_shape],  # two inputs of same shape
                "output_shape": current_shape,  # elementwise add preserves shape
                "dimensions": {"shape": [self.config.hidden_size]},
                "is_data_placeholder": False,
            }
            symbolic_nodes.append(ffn_residual_info)
            execution_order.append(f"layer_{layer_idx}_ffn_residual")
            order_counter += 1

        # Final layer norm
        final_shape = [batch_size, seq_len, self.config.hidden_size]
        final_norm_info = {
            "name": "final_layernorm",
            "operation_type": "normalization",
            "operation_category": "normalization",
            "execution_order": order_counter,
            "input_shape": final_shape,
            "output_shape": final_shape,  # normalization preserves shape
            "dimensions": {
                "normalized_shape": self.config.hidden_size,
                "eps": getattr(self.config, "rms_norm_eps", 1e-6),
            },
            "is_data_placeholder": False,
        }
        symbolic_nodes.append(final_norm_info)
        execution_order.append("final_layernorm")
        order_counter += 1

        self.symbolic_graph = {
            "nodes": symbolic_nodes,
            "execution_order": execution_order,
            "total_nodes": len(symbolic_nodes),
        }

        return self.symbolic_graph

    def print_summary(self):
        """Print a summary of the model dimensions and structure"""
        dims = self.extract_critical_dimensions()

        print(f"Model: {self.model_name_or_path}")
        print(f"Architecture: {getattr(self.config, 'architectures', ['Unknown'])[0]}")
        print("\n=== Critical Dimensions ===")

        print(f"Vocabulary Size: {dims['vocab_size']}")
        print(f"Hidden Size: {dims['hidden_size']}")
        print(f"Number of Layers: {dims['num_hidden_layers']}")
        print(f"Max Position Embeddings: {dims['max_position_embeddings']}")

        print("\n=== Attention Dimensions ===")
        att_dims = dims["attention"]
        print(f"Number of Attention Heads: {att_dims['num_attention_heads']}")
        print(f"Number of Key-Value Heads: {att_dims['num_key_value_heads']}")
        print(f"Head Dimension: {att_dims['head_dim']}")
        print(f"Key-Value Head Dimension: {att_dims['key_value_head_dim']}")

        print("\n=== FFN Dimensions ===")
        ffn_dims = dims["ffn"]
        print(f"Hidden Size: {ffn_dims['hidden_size']}")
        print(f"Intermediate Size: {ffn_dims['intermediate_size']}")
        print(f"Activation: {ffn_dims['activation']}")

        print("\n=== RMSNorm Dimensions ===")
        rms_dims = dims["rms_norm"]
        print(f"Normalized Shape: {rms_dims['normalized_shape']}")
        print(f"Epsilon: {rms_dims['eps']}")

        # Print symbolic graph summary
        if self.symbolic_graph:
            print(f"\n=== Symbolic Graph ===")
            print(f"Total Operations: {self.symbolic_graph['total_nodes']}")

            # Group operations by category
            categories = {}
            for node in self.symbolic_graph["nodes"]:
                cat = node.get("operation_category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1

            for cat, count in sorted(categories.items()):
                print(f"{cat}: {count}")

    def print_symbolic_graph_details(self):
        """Print detailed symbolic graph with execution order"""
        if not self.symbolic_graph:
            self.create_symbolic_graph()

        print("\n=== Symbolic Graph Execution Order ===")
        for node in self.symbolic_graph["nodes"]:
            name = node["name"]
            op_type = node["operation_type"]
            category = node.get("operation_category", "unknown")
            is_placeholder = node.get("is_data_placeholder", False)

            placeholder_marker = " [DATA PLACEHOLDER]" if is_placeholder else ""
            print(f"{node['execution_order']:3d}. {name} [{op_type}] -> {category}{placeholder_marker}")

            # Print input/output shapes
            if node.get("input_shape"):
                input_shape = node["input_shape"]
                if isinstance(input_shape[0], list):  # multiple inputs
                    print(f"     Input shapes: {input_shape}")
                else:  # single input
                    print(f"     Input shape: {input_shape}")

            if node.get("output_shape"):
                print(f"     Output shape: {node['output_shape']}")

            # Print operation-specific details
            if node.get("dimensions"):
                dims = node["dimensions"]
                if category == "attention":
                    print(
                        f"     Attention: heads={dims.get('num_attention_heads')}, kv_heads={dims.get('num_key_value_heads')}, head_dim={dims.get('head_dim')}"
                    )
                elif category == "normalization":
                    print(f"     Norm: shape={dims.get('normalized_shape')}, eps={dims.get('eps')}")
                elif category == "embedding":
                    print(f"     Embedding: {dims.get('num_embeddings')} x {dims.get('embedding_dim')}")
                elif category == "ffn":
                    print(
                        f"     FFN: {dims.get('hidden_size')} -> {dims.get('intermediate_size')} -> {dims.get('hidden_size')}, activation={dims.get('activation')}"
                    )
                elif category == "elementwise_add":
                    print(f"     Add: shape={dims.get('shape')}")

            print()