"""
GPT-OSS Performance Model for PLENA Simulator.

Uses PerfModel from perf_model.py for per-layer cycle counting and computes
overall inference performance (prefill time, decode time, TPS, TTFT).

Supports:
- Mixture of Experts (MoE) layers
- Mixed attention types (sliding_attention and full_attention per layer)

Usage:
    python gpt_oss_model.py --model gpt-oss-20b --model-lib ./Model_Lib --config ./plena_settings.toml --isa-lib ./customISA_lib.json
    python gpt_oss_model.py --list-models --model-lib ./Model_Lib
"""

import argparse
import json
from pathlib import Path

from perf_model import PerfModel, load_hardware_config_from_toml


class GPTOssModel:
    """
    GPT-OSS architecture performance model.
    Uses PerfModel for per-layer cycle counting and computes
    overall inference performance (prefill time, decode time, TPS, TTFT).

    Features:
    - MoE (Mixture of Experts) FFN layers
    - Mixed attention: sliding_attention and full_attention per layer
    """

    def __init__(
        self,
        model_config_path: str,
        hardware_config: dict,
        custom_isa_path: str,
        batch_size: int = 1,
        input_seq_len: int = 2048,
        output_seq_len: int = 128,
        device_num: int = 1,
        frequency_hz: float = 1e9,
    ):
        with open(model_config_path) as f:
            model_param = json.load(f)

        self.hidden_size = model_param["hidden_size"]
        self.num_attention_heads = model_param["num_attention_heads"]
        self.num_hidden_layers = model_param["num_hidden_layers"]
        self.intermediate_size = model_param["intermediate_size"]
        self.num_key_value_heads = model_param["num_key_value_heads"]
        self.vocab_size = model_param["vocab_size"]
        self.head_dim = model_param.get("head_dim", self.hidden_size // self.num_attention_heads)

        # MoE parameters
        self.num_experts = model_param.get("num_local_experts", 1)
        self.experts_per_token = model_param.get("experts_per_token", model_param.get("num_experts_per_tok", 1))

        # Per-layer MLP types: "ffn" or "moe"
        # Default: all "moe" if num_experts > 1, else all "ffn"
        default_mlp_type = "moe" if self.num_experts > 1 else "ffn"
        self.mlp_types = model_param.get("mlp_types", [default_mlp_type] * self.num_hidden_layers)
        self.num_moe_layers = sum(1 for mt in self.mlp_types if mt == "moe")
        self.num_ffn_layers = self.num_hidden_layers - self.num_moe_layers

        # Sliding window attention parameters
        self.sliding_window = model_param.get("sliding_window", 0)
        self.layer_types = model_param.get("layer_types", ["full_attention"] * self.num_hidden_layers)

        # Ensure layer_types matches num_hidden_layers
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(self.layer_types)}) != num_hidden_layers ({self.num_hidden_layers})"
            )

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        self.frequency = frequency_hz
        self.hardware_config = hardware_config
        self.batch_size = batch_size
        self.device_batch_size = batch_size // device_num
        self.device_num = device_num

        # Initialize PLENA hardware latency model
        self.perf = PerfModel(hardware_config, custom_isa_path)

    def print_config(self):
        """Print model and hardware configuration."""
        print("=" * 60)
        print("GPT-OSS Model Configuration")
        print("=" * 60)
        print(f"Hidden size:          {self.hidden_size}")
        print(f"Num attention heads:  {self.num_attention_heads}")
        print(f"Num KV heads:         {self.num_key_value_heads}")
        print(f"Num hidden layers:    {self.num_hidden_layers}")
        print(f"Intermediate size:    {self.intermediate_size}")
        print(f"Head dim:             {self.head_dim}")
        print(f"Vocab size:           {self.vocab_size}")
        print("-" * 60)
        print("MLP Configuration")
        print("-" * 60)
        print(f"FFN layers:           {self.num_ffn_layers}")
        print(f"MoE layers:           {self.num_moe_layers}")
        print(f"Num experts:          {self.num_experts}")
        print(f"Experts per token:    {self.experts_per_token}")
        print("-" * 60)
        print("Attention Configuration")
        print("-" * 60)
        print(f"Sliding window size:  {self.sliding_window}")
        sliding_layers = sum(1 for lt in self.layer_types if lt == "sliding_attention")
        full_layers = sum(1 for lt in self.layer_types if lt == "full_attention")
        print(f"Sliding attn layers:  {sliding_layers}")
        print(f"Full attn layers:     {full_layers}")
        print("-" * 60)
        print("Inference Settings")
        print("-" * 60)
        print(f"Batch size:           {self.batch_size}")
        print(f"Input seq len:        {self.input_seq_len}")
        print(f"Output seq len:       {self.output_seq_len}")
        print(f"Device num:           {self.device_num}")
        print("-" * 60)
        print("Hardware Config")
        print("-" * 60)
        print(f"MLEN: {self.perf.mlen}, BLEN: {self.perf.blen}, VLEN: {self.perf.vlen}")
        print("=" * 60)

    def _compute_attention(self, layer_idx: int, seq_len: int, kv_size: int, mode: str) -> int:
        """Compute attention cycles for a layer based on its type."""
        layer_type = self.layer_types[layer_idx]

        if layer_type == "sliding_attention":
            return self.perf.sliding_window_attention(
                num_attention_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                seq_len=seq_len,
                kv_size=kv_size,
                batch_size=self.device_batch_size,
                sliding_window_size=self.sliding_window,
                num_sink_tokens=1,
                mode=mode,
            )
        else:  # full_attention
            return self.perf.flash_attention(
                num_attention_heads=self.num_attention_heads,
                num_kv_heads=self.num_key_value_heads,
                head_dim=self.head_dim,
                seq_len=seq_len,
                kv_size=kv_size,
                batch_size=self.device_batch_size,
                mode=mode,
            )

    def compute_prefill_time(self, verbose: bool = True) -> float:
        """Compute prefill phase execution time in seconds."""
        mode = "prefill"
        kv_size = self.input_seq_len
        overall_exe_cycle = 0

        # Embedding layer
        emb_cycles = self.perf.embeddings(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
        overall_exe_cycle += emb_cycles

        # Per-layer cycle tracking
        rms_total = 0
        proj_total = 0
        attn_total = 0
        attn_sliding_total = 0
        attn_full_total = 0
        res_total = 0
        moe_total = 0

        for layer_idx in range(self.num_hidden_layers):
            # RMS normalization (pre-attention)
            rms = self.perf.rms_layer(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
            rms_total += rms

            # QKV projection
            proj = self.perf.projection(
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                self.input_seq_len,
                self.device_batch_size,
                mode,
            )
            proj_total += proj

            # Attention (sliding or full based on layer type)
            attn = self._compute_attention(layer_idx, self.input_seq_len, kv_size, mode)
            attn_total += attn
            if self.layer_types[layer_idx] == "sliding_attention":
                attn_sliding_total += attn
            else:
                attn_full_total += attn

            # Residual connection
            res = self.perf.residual(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
            res_total += res

            # RMS normalization (pre-FFN)
            rms = self.perf.rms_layer(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
            rms_total += rms

            # FFN or MoE (based on mlp_types)
            if self.mlp_types[layer_idx] == "moe":
                moe = self.perf.mlp_moe(
                    hidden_size=self.hidden_size,
                    seq_len=self.input_seq_len,
                    batch_size=self.device_batch_size,
                    num_experts=self.num_experts,
                    expert_per_token=self.experts_per_token,
                    intermediate_size=self.intermediate_size,
                    mode=mode,
                )
                moe_total += moe
            else:
                ffn = self.perf.feed_forward(
                    self.hidden_size, self.intermediate_size, self.input_seq_len, self.device_batch_size, mode
                )
                moe_total += ffn  # Add to same counter for simplicity

            # Second residual connection
            res = self.perf.residual(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
            res_total += res

        transformer_cycles = rms_total + proj_total + attn_total + res_total + moe_total
        overall_exe_cycle += transformer_cycles

        # LM head
        lm_head_cycles = self.perf.lm_head(self.hidden_size, self.vocab_size, self.device_batch_size)
        overall_exe_cycle += lm_head_cycles

        execution_time = overall_exe_cycle / self.frequency

        if verbose:
            print("\nPrefill Execution Distribution:")
            print(f"  RMS Layer:          {rms_total / transformer_cycles * 100:.2f}%")
            print(f"  Projection:         {proj_total / transformer_cycles * 100:.2f}%")
            print(f"  Attention (total):  {attn_total / transformer_cycles * 100:.2f}%")
            print(f"    - Sliding Attn:   {attn_sliding_total / transformer_cycles * 100:.2f}%")
            print(f"    - Full Attn:      {attn_full_total / transformer_cycles * 100:.2f}%")
            print(f"  Residual:           {res_total / transformer_cycles * 100:.2f}%")
            print(f"  MLP (FFN/MoE):      {moe_total / transformer_cycles * 100:.2f}%")
            print(f"\n  Total cycles: {overall_exe_cycle:,}")

        return execution_time

    def compute_decode_time(self, output_token_size: int, verbose: bool = True) -> float:
        """Compute decode phase execution time in seconds."""
        mode = "decode"
        kv_size = self.input_seq_len

        rms_count = 0
        projection_count = 0
        attn_count = 0
        attn_sliding_count = 0
        attn_full_count = 0
        residual_count = 0
        moe_count = 0

        for _ in range(output_token_size):
            for layer_idx in range(self.num_hidden_layers):
                # RMS normalization (pre-attention)
                rms_count += self.perf.rms_layer(self.hidden_size, 1, self.device_batch_size, mode)

                # QKV projection
                projection_count += self.perf.projection(
                    self.hidden_size,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    1,
                    self.device_batch_size,
                    mode,
                )

                # Attention (sliding or full based on layer type)
                attn = self._compute_attention(layer_idx, 1, kv_size, mode)
                attn_count += attn
                if self.layer_types[layer_idx] == "sliding_attention":
                    attn_sliding_count += attn
                else:
                    attn_full_count += attn

                # Residual connection
                residual_count += self.perf.residual(self.hidden_size, 1, self.device_batch_size, mode)

                # RMS normalization (pre-FFN)
                rms_count += self.perf.rms_layer(self.hidden_size, 1, self.device_batch_size, mode)

                # FFN or MoE (based on mlp_types)
                if self.mlp_types[layer_idx] == "moe":
                    moe_count += self.perf.mlp_moe(
                        hidden_size=self.hidden_size,
                        seq_len=1,
                        batch_size=self.device_batch_size,
                        num_experts=self.num_experts,
                        expert_per_token=self.experts_per_token,
                        intermediate_size=self.intermediate_size,
                        mode=mode,
                    )
                else:
                    moe_count += self.perf.feed_forward(
                        self.hidden_size, self.intermediate_size, 1, self.device_batch_size, mode
                    )

                # Second residual connection
                residual_count += self.perf.residual(self.hidden_size, 1, self.device_batch_size, mode)

            kv_size += 1

        overall_inst_num = rms_count + projection_count + attn_count + residual_count + moe_count
        overall_exe_cycle = overall_inst_num * 2
        execution_time = overall_exe_cycle / self.frequency

        if verbose:
            print("\nDecode Execution Distribution:")
            print(f"  RMS Layer:          {rms_count / overall_inst_num * 100:.2f}%")
            print(f"  Projection:         {projection_count / overall_inst_num * 100:.2f}%")
            print(f"  Attention (total):  {attn_count / overall_inst_num * 100:.2f}%")
            print(f"    - Sliding Attn:   {attn_sliding_count / overall_inst_num * 100:.2f}%")
            print(f"    - Full Attn:      {attn_full_count / overall_inst_num * 100:.2f}%")
            print(f"  Residual:           {residual_count / overall_inst_num * 100:.2f}%")
            print(f"  MLP (FFN/MoE):      {moe_count / overall_inst_num * 100:.2f}%")

        return execution_time

    def compute_performance(self, verbose: bool = True) -> tuple:
        """
        Compute overall inference performance.

        Returns:
            tuple: (TTFT in seconds, TPS)
        """
        prefill_time = self.compute_prefill_time(verbose)
        first_token_decode = self.compute_decode_time(1, verbose=False)
        ttft = (prefill_time + first_token_decode) / self.device_num

        decode_time = self.compute_decode_time(self.output_seq_len, verbose)
        tps = (self.batch_size * self.output_seq_len) / decode_time

        return ttft, tps


# =============================================================================
# Model Library Utilities
# =============================================================================


def list_available_models(model_lib_path: Path) -> list:
    """List all available model configs in Model_Lib."""
    if not model_lib_path.exists():
        return []
    return sorted([f.stem for f in model_lib_path.glob("*.json")])


def resolve_model_path(model_name: str, model_lib_path: Path) -> Path:
    """Resolve model name to full path."""
    model_path = model_lib_path / f"{model_name}.json"
    if not model_path.exists():
        available = list_available_models(model_lib_path)
        raise FileNotFoundError(f"Model '{model_name}' not found.\nAvailable models: {', '.join(available)}")
    return model_path


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="GPT-OSS Performance Model - Compute TPS and TTFT for MoE LLM inference on PLENA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gpt_oss_model.py --model gpt-oss-20b --model-lib ./Model_Lib --config ./plena_settings.toml --isa-lib ./customISA_lib.json
  python gpt_oss_model.py --model-path ./custom_model.json --config ./plena_settings.toml --isa-lib ./customISA_lib.json
  python gpt_oss_model.py --list-models --model-lib ./Model_Lib
""",
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", "-m", help="Model name from Model_Lib")
    model_group.add_argument("--model-path", help="Full path to model config JSON")
    model_group.add_argument("--list-models", "-l", action="store_true", help="List available models")
    model_group.add_argument(
        "--task-file", "-t", help="Path to task JSON file specifying model, batch, input_seq, output_seq"
    )

    parser.add_argument(
        "--model-lib", required=False, help="Path to Model_Lib directory (required for --model and --list-models)"
    )
    parser.add_argument("--config", "-c", required=False, help="Path to hardware config TOML (required for inference)")
    parser.add_argument("--isa-lib", required=False, help="Path to customISA_lib.json (required for inference)")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--input-seq", "-i", type=int, default=2048, help="Input sequence length (default: 2048)")
    parser.add_argument("--output-seq", "-o", type=int, default=128, help="Output sequence length (default: 128)")
    parser.add_argument("--device-num", "-d", type=int, default=1, help="Number of devices (default: 1)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    if args.list_models:
        if not args.model_lib:
            parser.error("--model-lib is required for --list-models")
        model_lib_path = Path(args.model_lib)
        print("Available models:")
        for model in list_available_models(model_lib_path):
            print(f"  {model}")
        return

    # Handle task file: load parameters from JSON
    if args.task_file:
        with open(args.task_file) as f:
            task = json.load(f)
        args.model = task.get("model")
        args.batch_size = task.get("batch_size", args.batch_size)
        args.input_seq = task.get("input_seq", args.input_seq)
        args.output_seq = task.get("output_seq", args.output_seq)
        args.device_num = task.get("device_num", args.device_num)
        if not args.model:
            parser.error("Task file must specify 'model'")

    # Validate required args for inference
    if not args.config:
        parser.error("--config is required for inference")
    if not args.isa_lib:
        parser.error("--isa-lib is required for inference")

    # Resolve model path
    if args.model or args.task_file:
        if not args.model_lib:
            parser.error("--model-lib is required when using --model")
        model_lib_path = Path(args.model_lib)
        model_path = str(resolve_model_path(args.model, model_lib_path))
    else:
        model_path = args.model_path

    # Load hardware config
    hardware_config = load_hardware_config_from_toml(args.config)

    # Create and run model
    model = GPTOssModel(
        model_config_path=model_path,
        hardware_config=hardware_config,
        custom_isa_path=args.isa_lib,
        batch_size=args.batch_size,
        input_seq_len=args.input_seq,
        output_seq_len=args.output_seq,
        device_num=args.device_num,
    )

    if not args.quiet:
        model.print_config()

    ttft, tps = model.compute_performance(verbose=not args.quiet)

    if args.json:
        result = {
            "model": args.model or args.model_path,
            "batch_size": args.batch_size,
            "input_seq_len": args.input_seq,
            "output_seq_len": args.output_seq,
            "device_num": args.device_num,
            "ttft_seconds": ttft,
            "ttft_ms": ttft * 1000,
            "tps": tps,
        }
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 60)
        print("Performance Results")
        print("=" * 60)
        print(f"TTFT (Time to First Token): {ttft:.6f} s ({ttft * 1000:.3f} ms)")
        print(f"TPS (Tokens Per Second):    {tps:.2f}")
        print("=" * 60)


if __name__ == "__main__":
    main()
