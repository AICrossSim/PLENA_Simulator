"""
Integrated Utilization Model for PLENA Simulator.

Architecture:
- PLENA_Utilization: Per-layer hardware utilization modeling
- LLaMA_Utilization_Model: LLM architecture-level utilization model (attainable vs theoretical FLOPS)

Usage:
    python utilisation_model.py --model llama-3.1-8b
    python utilisation_model.py --model llama-3.1-8b --batch-size 4 --input-seq 2048
    python utilisation_model.py --list-models
"""

import argparse
import json
import math
from pathlib import Path

try:
    import toml
except ImportError:
    toml = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_LIB_PATH = PROJECT_ROOT / "compiler" / "doc" / "Model_Lib"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "plena_settings.toml"


# =============================================================================
# Hardware Configuration Loading
# =============================================================================

def load_hardware_config_from_toml(toml_path: str) -> dict:
    """
    Load hardware configuration from plena_settings.toml.
    Extracts values from CONFIG and LATENCY sections.
    """
    if toml is None:
        raise ImportError("toml package required. Install with: pip install toml")

    with open(toml_path, "r") as f:
        data = toml.load(f)

    hardware_config = {}

    if "CONFIG" in data:
        for param_name, val in data["CONFIG"].items():
            if isinstance(val, dict) and "value" in val:
                hardware_config[param_name] = val["value"]

    if "LATENCY" in data:
        for param_name, val in data["LATENCY"].items():
            if isinstance(val, dict):
                if "dc_lib_en" in val:
                    hardware_config[param_name] = val["dc_lib_en"]
                elif "value" in val:
                    hardware_config[param_name] = val["value"]

    return hardware_config


# =============================================================================
# PLENA_Utilization: Per-Layer Hardware Utilization Model
# =============================================================================

class PLENA_Utilization:
    """
    Per-layer hardware utilization model for PLENA accelerator.

    Computes attainable vs theoretical FLOPS for different operations
    based on hardware configuration (MLEN, BLEN, VLEN).

    Attributes:
        mlen, blen, vlen: Hardware dimensions
        bubble_ratio: Overhead ratio for memory operations (default 1.2)
    """

    def __init__(self, hardware_config: dict, bubble_ratio: float = 1.2):
        self.hardware_config = hardware_config
        self.mlen = hardware_config["MLEN"]
        self.blen = hardware_config["BLEN"]
        self.vlen = hardware_config["VLEN"]
        self.bubble_ratio = bubble_ratio

    def projection_utilization(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill"
    ) -> tuple:
        """
        Compute projection layer utilization.

        Returns:
            tuple: (attainable_ops, theoretical_ops)
        """
        attainable = 0
        theoretical = 0

        if mode == "prefill":
            # Q projection
            gemm_ops = (seq_len // self.blen) * (hidden_size / self.mlen) * self.blen * ((head_dim * num_attention_heads) // self.blen)
            # K, V projection
            gemm_ops += ((head_dim * num_kv_heads) // self.blen) * (hidden_size / self.mlen) * self.blen * (seq_len // self.blen) * 2

            attainable = gemm_ops * batch_size
            theoretical = gemm_ops * self.bubble_ratio * batch_size

        elif mode == "decode":
            gemm_ops = ((head_dim * num_attention_heads) // self.blen) * (hidden_size // self.mlen) * self.blen * math.ceil(batch_size / self.blen)
            gemm_ops += ((head_dim * num_kv_heads) // self.blen) * (hidden_size // self.mlen) * self.blen * math.ceil(batch_size / self.blen) * 2

            theoretical = gemm_ops * self.bubble_ratio
            if batch_size > self.blen:
                attainable = gemm_ops
            else:
                attainable = gemm_ops * (batch_size / self.blen)

        return attainable, theoretical

    def flash_attention_utilization(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
        partitioned_matrix: bool = True
    ) -> tuple:
        """
        Compute flash attention utilization.

        Returns:
            tuple: (attainable_ops, theoretical_ops)
        """
        attainable = 0
        theoretical = 0
        max_head_per_mlen = math.ceil(self.mlen / head_dim)
        num_head_groups = num_attention_heads // num_kv_heads

        if mode == "prefill":
            if partitioned_matrix:
                for _ in range(batch_size):
                    for _ in range(math.ceil(num_attention_heads / max_head_per_mlen)):
                        for _ in range(math.ceil(seq_len / self.mlen)):
                            for _ in range(math.ceil(seq_len / self.mlen)):
                                # QKT
                                gemm_ops = head_dim
                                theoretical += gemm_ops * self.bubble_ratio
                                if num_attention_heads > max_head_per_mlen:
                                    attainable += gemm_ops
                                else:
                                    attainable += gemm_ops * (head_dim * num_attention_heads / self.mlen)
                                # Softmax
                                theoretical += self.mlen * 10
                                # PV
                                gemm_ops = math.ceil(self.mlen / self.blen) * self.blen * math.ceil(head_dim / self.blen) * num_head_groups
                                theoretical += gemm_ops * self.bubble_ratio
                                if head_dim > self.blen:
                                    attainable += gemm_ops
                                else:
                                    attainable += gemm_ops * (head_dim / self.blen)
                                theoretical += self.mlen * num_head_groups * (head_dim // self.blen) * self.bubble_ratio
                                # Compute O
                                theoretical += self.mlen * 5 + 4
            else:
                for _ in range(batch_size):
                    for _ in range(num_attention_heads):
                        for _ in range(math.ceil(seq_len / self.mlen)):
                            for _ in range(math.ceil(seq_len / self.mlen)):
                                # QKT
                                gemm_ops = (self.mlen // self.blen) * math.ceil(head_dim / self.mlen) * self.blen * (self.mlen // self.blen)
                                theoretical += gemm_ops * self.bubble_ratio
                                if head_dim > self.mlen:
                                    attainable += gemm_ops
                                else:
                                    attainable += gemm_ops * (head_dim / self.mlen)
                                # Softmax
                                theoretical += self.mlen * 10
                                # PV
                                gemm_ops = (self.mlen // self.blen) * math.ceil(head_dim / self.blen) * self.blen
                                theoretical += gemm_ops * self.bubble_ratio
                                if head_dim > self.blen:
                                    attainable += gemm_ops
                                else:
                                    attainable += gemm_ops * (head_dim / self.blen)
                                # Compute O
                                theoretical += self.mlen * 5 + 4

        elif mode == "decode":
            if partitioned_matrix:
                for _ in range(batch_size):
                    for _ in range(math.ceil(num_kv_heads / max_head_per_mlen)):
                        for _ in range(math.ceil(kv_size / self.mlen)):
                            # QKT
                            gemm_ops = math.ceil(num_head_groups / self.blen) * head_dim * math.ceil(self.mlen / self.blen)
                            theoretical += gemm_ops * self.bubble_ratio
                            if num_kv_heads > (self.mlen // head_dim):
                                if num_head_groups > self.blen:
                                    attainable += gemm_ops
                                else:
                                    attainable += gemm_ops * (num_head_groups / self.blen)
                            else:
                                if num_head_groups > self.blen:
                                    attainable += gemm_ops * (num_kv_heads / (self.mlen // head_dim))
                                else:
                                    attainable += gemm_ops * (num_head_groups / self.blen) * (num_kv_heads / (self.mlen // head_dim))
                            # Softmax
                            theoretical += self.mlen * 10
                            # PV
                            gemm_ops = math.ceil(num_head_groups / self.blen) * math.ceil(head_dim / self.blen) * num_head_groups
                            theoretical += gemm_ops * self.bubble_ratio
                            if num_head_groups > self.blen:
                                attainable += gemm_ops
                            else:
                                attainable += gemm_ops * (num_head_groups / self.blen)
                            # Compute O
                            theoretical += self.mlen * 5 + 4
            else:
                for _ in range(batch_size):
                    for _ in range(num_attention_heads):
                        for _ in range(math.ceil(kv_size / self.mlen)):
                            # QKT
                            gemm_ops = math.ceil(head_dim / self.mlen) * math.ceil(self.mlen / self.blen) * self.blen
                            theoretical += gemm_ops * self.bubble_ratio
                            attainable += gemm_ops * (1 / self.blen) * (head_dim / self.mlen)
                            # Softmax
                            theoretical += self.mlen * 10
                            # PV
                            gemm_ops = (head_dim // self.blen) * self.blen
                            theoretical += gemm_ops * self.bubble_ratio
                            attainable += gemm_ops * (1 / self.blen)
                            # Compute O
                            theoretical += self.mlen * 5 + 4

        return attainable, theoretical

    def ffn_utilization(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill"
    ) -> tuple:
        """
        Compute feed-forward network utilization.

        Returns:
            tuple: (attainable_ops, theoretical_ops)
        """
        attainable = 0
        theoretical = 0

        if mode == "prefill":
            # Up Projection
            ops = (intermediate_size // self.blen) * (hidden_size // self.mlen) * (seq_len // self.blen)
            attainable += ops
            theoretical += ops * self.bubble_ratio

            # Gate Projection
            ops = (intermediate_size // self.blen) * (hidden_size // self.mlen) * (seq_len // self.blen)
            attainable += ops
            theoretical += ops * self.bubble_ratio

            # SiLU
            theoretical += (intermediate_size // self.vlen) * 5

            # Down Projection
            ops = (hidden_size // self.blen) * (intermediate_size // self.mlen) * (seq_len // self.blen)
            attainable += ops
            theoretical += ops * self.bubble_ratio

        elif mode == "decode":
            # Up projection
            fill_bubble = self.blen * (intermediate_size // self.blen) * math.ceil(batch_size / self.blen)
            ops = (intermediate_size // self.blen) * self.blen * (hidden_size // self.mlen) * math.ceil(batch_size / self.blen)
            attainable += ops * (min(batch_size, self.blen) / self.blen)
            theoretical += ops + fill_bubble

            # Gate Projection
            fill_bubble = self.blen * (intermediate_size // self.blen) * math.ceil(batch_size / self.blen)
            ops = (intermediate_size // self.blen) * self.blen * (hidden_size // self.mlen) * math.ceil(batch_size / self.blen)
            attainable += ops * (min(batch_size, self.blen) / self.blen)
            theoretical += ops + fill_bubble

            # SiLU
            theoretical += (intermediate_size // self.vlen) * 5

            # Down Projection
            fill_bubble = self.blen * (hidden_size // self.blen) * math.ceil(batch_size / self.blen)
            ops = (hidden_size // self.blen) * self.blen * (intermediate_size // self.mlen) * math.ceil(batch_size / self.blen)
            attainable += ops * (min(batch_size, self.blen) / self.blen)
            theoretical += ops + fill_bubble

        return attainable, theoretical

    def embedding_utilization(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill"
    ) -> tuple:
        """
        Compute embedding layer utilization.

        Returns:
            tuple: (attainable_ops, theoretical_ops)
        """
        if mode == "prefill":
            ops = (hidden_size // self.blen) * (hidden_size // self.mlen) * (seq_len // self.blen)
            attainable = ops * (self.blen * self.mlen * self.blen)
            theoretical = ops * (self.blen * self.mlen * self.blen)
        else:
            ops = (hidden_size // self.blen) * (hidden_size // self.mlen)
            attainable = ops * (self.blen * self.mlen * min(batch_size, self.blen))
            theoretical = ops * (self.blen * self.mlen * self.blen)

        return attainable, theoretical


# =============================================================================
# LLaMA_Utilization_Model: LLM Utilization Model
# =============================================================================

class LLaMA_Utilization_Model:
    """
    LLaMA architecture utilization model.
    Uses PLENA_Utilization for per-layer utilization and computes
    overall inference utilization (attainable vs theoretical FLOPS).
    """

    def __init__(
        self,
        model_config_path: str,
        hardware_config: dict,
        batch_size: int = 1,
        input_seq_len: int = 2048,
        output_seq_len: int = 128,
        device_num: int = 1,
        partitioned_matrix: bool = True
    ):
        with open(model_config_path, "r") as f:
            model_param = json.load(f)

        self.hidden_size = model_param["hidden_size"]
        self.num_attention_heads = model_param["num_attention_heads"]
        self.num_hidden_layers = model_param["num_hidden_layers"]
        self.intermediate_size = model_param["intermediate_size"]
        self.num_key_value_heads = model_param["num_key_value_heads"]
        self.vocab_size = model_param["vocab_size"]

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.hardware_config = hardware_config
        self.batch_size = batch_size
        self.device_num = device_num
        self.partitioned_matrix = partitioned_matrix

        # Initialize PLENA hardware utilization model
        self.plena = PLENA_Utilization(hardware_config)

    def print_config(self):
        """Print model and hardware configuration."""
        print("=" * 50)
        print("Model Configuration")
        print("=" * 50)
        print(f"Hidden size:          {self.hidden_size}")
        print(f"Num attention heads:  {self.num_attention_heads}")
        print(f"Num KV heads:         {self.num_key_value_heads}")
        print(f"Num hidden layers:    {self.num_hidden_layers}")
        print(f"Intermediate size:    {self.intermediate_size}")
        print(f"Head dim:             {self.head_dim}")
        print(f"Vocab size:           {self.vocab_size}")
        print("-" * 50)
        print("Inference Settings")
        print("-" * 50)
        print(f"Batch size:           {self.batch_size}")
        print(f"Input seq len:        {self.input_seq_len}")
        print(f"Output seq len:       {self.output_seq_len}")
        print(f"Partitioned matrix:   {self.partitioned_matrix}")
        print("-" * 50)
        print("Hardware Config")
        print("-" * 50)
        print(f"MLEN: {self.plena.mlen}, BLEN: {self.plena.blen}, VLEN: {self.plena.vlen}")
        print("=" * 50)

    def compute_prefill_utilization(self, verbose: bool = True) -> dict:
        """Compute prefill phase utilization."""
        mode = "prefill"

        # Attention (projection + flash attention)
        proj_att, proj_theo = self.plena.projection_utilization(
            self.hidden_size, self.num_attention_heads, self.num_key_value_heads,
            self.head_dim, self.input_seq_len, self.batch_size, mode
        )
        fa_att, fa_theo = self.plena.flash_attention_utilization(
            self.num_attention_heads, self.num_key_value_heads, self.head_dim,
            self.input_seq_len, self.input_seq_len, self.batch_size, mode, self.partitioned_matrix
        )

        attention_attainable = proj_att + fa_att
        attention_theoretical = proj_theo + fa_theo

        # FFN
        ffn_attainable, ffn_theoretical = self.plena.ffn_utilization(
            self.hidden_size, self.intermediate_size, self.input_seq_len, self.batch_size, mode
        )

        result = {
            "attention": {
                "attainable": attention_attainable,
                "theoretical": attention_theoretical,
                "utilization": attention_attainable / attention_theoretical if attention_theoretical > 0 else 0
            },
            "ffn": {
                "attainable": ffn_attainable,
                "theoretical": ffn_theoretical,
                "utilization": ffn_attainable / ffn_theoretical if ffn_theoretical > 0 else 0
            }
        }

        if verbose:
            print("\nPrefill Utilization:")
            print(f"  Attention: {result['attention']['utilization']*100:.2f}%")
            print(f"  FFN:       {result['ffn']['utilization']*100:.2f}%")

        return result

    def compute_decode_utilization(self, kv_size: int = None, verbose: bool = True) -> dict:
        """Compute decode phase utilization."""
        mode = "decode"
        if kv_size is None:
            kv_size = self.input_seq_len

        # Attention (projection + flash attention)
        proj_att, proj_theo = self.plena.projection_utilization(
            self.hidden_size, self.num_attention_heads, self.num_key_value_heads,
            self.head_dim, 1, self.batch_size, mode
        )
        fa_att, fa_theo = self.plena.flash_attention_utilization(
            self.num_attention_heads, self.num_key_value_heads, self.head_dim,
            1, kv_size, self.batch_size, mode, self.partitioned_matrix
        )

        attention_attainable = proj_att + fa_att
        attention_theoretical = proj_theo + fa_theo

        # FFN
        ffn_attainable, ffn_theoretical = self.plena.ffn_utilization(
            self.hidden_size, self.intermediate_size, 1, self.batch_size, mode
        )

        result = {
            "attention": {
                "attainable": attention_attainable,
                "theoretical": attention_theoretical,
                "utilization": attention_attainable / attention_theoretical if attention_theoretical > 0 else 0
            },
            "ffn": {
                "attainable": ffn_attainable,
                "theoretical": ffn_theoretical,
                "utilization": ffn_attainable / ffn_theoretical if ffn_theoretical > 0 else 0
            }
        }

        if verbose:
            print(f"\nDecode Utilization (kv_size={kv_size}):")
            print(f"  Attention: {result['attention']['utilization']*100:.2f}%")
            print(f"  FFN:       {result['ffn']['utilization']*100:.2f}%")

        return result

    def compute_overall_utilization(self, verbose: bool = True) -> dict:
        """
        Compute overall inference utilization.

        Returns:
            dict: Overall utilization metrics
        """
        prefill = self.compute_prefill_utilization(verbose)
        decode = self.compute_decode_utilization(verbose=verbose)

        total_attainable = (
            prefill["attention"]["attainable"] + prefill["ffn"]["attainable"] +
            decode["attention"]["attainable"] + decode["ffn"]["attainable"]
        )
        total_theoretical = (
            prefill["attention"]["theoretical"] + prefill["ffn"]["theoretical"] +
            decode["attention"]["theoretical"] + decode["ffn"]["theoretical"]
        )

        overall_utilization = total_attainable / total_theoretical if total_theoretical > 0 else 0

        result = {
            "prefill": prefill,
            "decode": decode,
            "overall_utilization": overall_utilization
        }

        if verbose:
            print("\n" + "=" * 50)
            print(f"Overall Utilization: {overall_utilization*100:.2f}%")
            print("=" * 50)

        return result


# =============================================================================
# Model Library Utilities
# =============================================================================

def list_available_models() -> list:
    """List all available model configs in Model_Lib."""
    if not MODEL_LIB_PATH.exists():
        return []
    return sorted([f.stem for f in MODEL_LIB_PATH.glob("*.json")])


def resolve_model_path(model_name: str) -> Path:
    """Resolve model name to full path."""
    model_path = MODEL_LIB_PATH / f"{model_name}.json"
    if not model_path.exists():
        available = list_available_models()
        raise FileNotFoundError(
            f"Model '{model_name}' not found.\nAvailable models: {', '.join(available)}"
        )
    return model_path


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PLENA Utilization Model - Compute hardware utilization for LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  {', '.join(list_available_models())}

Examples:
  python utilisation_model.py --model llama-3.1-8b
  python utilisation_model.py --model llama-3.3-70b --batch-size 8
  python utilisation_model.py --model-path ./custom_model.json
"""
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", "-m", help="Model name from Model_Lib")
    model_group.add_argument("--model-path", help="Full path to model config JSON")
    model_group.add_argument("--list-models", "-l", action="store_true", help="List available models")

    parser.add_argument("--config", "-c", default=str(DEFAULT_CONFIG_PATH), help="Path to hardware config TOML")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--input-seq", "-i", type=int, default=2048, help="Input sequence length (default: 2048)")
    parser.add_argument("--output-seq", "-o", type=int, default=128, help="Output sequence length (default: 128)")
    parser.add_argument("--no-partition", action="store_true", help="Disable partitioned matrix optimization")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    if args.list_models:
        print("Available models:")
        for model in list_available_models():
            print(f"  {model}")
        return

    # Resolve model path
    if args.model:
        model_path = str(resolve_model_path(args.model))
    else:
        model_path = args.model_path

    # Load hardware config
    hardware_config = load_hardware_config_from_toml(args.config)

    # Create and run model
    model = LLaMA_Utilization_Model(
        model_config_path=model_path,
        hardware_config=hardware_config,
        batch_size=args.batch_size,
        input_seq_len=args.input_seq,
        output_seq_len=args.output_seq,
        partitioned_matrix=not args.no_partition
    )

    if not args.quiet:
        model.print_config()

    result = model.compute_overall_utilization(verbose=not args.quiet)

    if args.json:
        output = {
            "model": args.model or args.model_path,
            "batch_size": args.batch_size,
            "input_seq_len": args.input_seq,
            "output_seq_len": args.output_seq,
            "partitioned_matrix": not args.no_partition,
            "prefill_attention_util": result["prefill"]["attention"]["utilization"],
            "prefill_ffn_util": result["prefill"]["ffn"]["utilization"],
            "decode_attention_util": result["decode"]["attention"]["utilization"],
            "decode_ffn_util": result["decode"]["ffn"]["utilization"],
            "overall_utilization": result["overall_utilization"]
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
