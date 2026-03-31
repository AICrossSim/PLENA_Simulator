"""
LLaMA Performance Model for PLENA Simulator.

Uses PerfModel from perf_model.py for per-layer cycle counting and computes
overall inference performance (prefill time, decode time, TPS, TTFT).

Usage:
    python llama_model.py --model llama-3.1-8b --model-lib ./Model_Lib --config ./plena_settings.toml --isa-lib ./customISA_lib.json
    python llama_model.py --list-models --model-lib ./Model_Lib
"""

import argparse
import json
from pathlib import Path

from perf_model import PerfModel, load_hardware_config_from_toml


class LLaMAModel:
    """
    LLaMA architecture performance model.
    Uses PerfModel for per-layer cycle counting and computes
    overall inference performance (prefill time, decode time, TPS, TTFT).
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

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.frequency = frequency_hz
        self.hardware_config = hardware_config
        self.batch_size = batch_size
        self.device_batch_size = batch_size // device_num
        self.device_num = device_num

        # Initialize PLENA hardware latency model
        self.perf = PerfModel(hardware_config, custom_isa_path)

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
        print(f"Device num:           {self.device_num}")
        print("-" * 50)
        print("Hardware Config")
        print("-" * 50)
        print(f"MLEN: {self.perf.mlen}, BLEN: {self.perf.blen}, VLEN: {self.perf.vlen}")
        print("=" * 50)

    def compute_prefill_time(self, verbose: bool = True) -> float:
        """Compute prefill phase execution time in seconds."""
        mode = "prefill"
        kv_size = self.input_seq_len
        overall_exe_cycle = 0

        overall_exe_cycle += self.perf.embeddings(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)

        rms = self.perf.rms_layer(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
        proj = self.perf.projection(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.input_seq_len,
            self.device_batch_size,
            mode,
        )
        attn = self.perf.flash_attention(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.input_seq_len,
            kv_size,
            self.device_batch_size,
            mode,
        )
        res = self.perf.residual(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
        ffn = self.perf.feed_forward(
            self.hidden_size, self.intermediate_size, self.input_seq_len, self.device_batch_size, mode
        )
        transformer_block_cycles = rms + proj + attn + res + rms + ffn
        overall_exe_cycle += transformer_block_cycles * self.num_hidden_layers

        execution_time = overall_exe_cycle / self.frequency

        if verbose:
            print("\nPrefill Execution Distribution:")
            print(f"  RMS Layer:       {rms / transformer_block_cycles * 100:.2f}%")
            print(f"  Projection:      {proj / transformer_block_cycles * 100:.2f}%")
            print(f"  Flash Attention: {attn / transformer_block_cycles * 100:.2f}%")
            print(f"  Residual:        {res / transformer_block_cycles * 100:.2f}%")
            print(f"  Feed Forward:    {ffn / transformer_block_cycles * 100:.2f}%")

        return execution_time

    def compute_decode_time(self, output_token_size: int, verbose: bool = True) -> float:
        """Compute decode phase execution time in seconds."""
        mode = "decode"
        kv_size = self.input_seq_len

        rms_count = 0
        projection_count = 0
        flash_attention_count = 0
        residual_count = 0
        feed_forward_count = 0

        for _ in range(output_token_size):
            for _ in range(self.num_hidden_layers):
                rms_count += self.perf.rms_layer(self.hidden_size, 1, self.device_batch_size, mode) * 2
                projection_count += self.perf.projection(
                    self.hidden_size,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    1,
                    self.device_batch_size,
                    mode,
                )
                flash_attention_count += self.perf.flash_attention(
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.head_dim,
                    1,
                    kv_size,
                    self.device_batch_size,
                    mode,
                )
                residual_count += self.perf.residual(self.hidden_size, 1, self.device_batch_size, mode)
                feed_forward_count += self.perf.feed_forward(
                    self.hidden_size, self.intermediate_size, 1, self.device_batch_size, mode
                )
            kv_size += 1

        overall_inst_num = rms_count + projection_count + flash_attention_count + residual_count + feed_forward_count
        # Factor of 2 accounts for instruction issue + memory access pipeline stages
        # in decode mode (single-token generation with sequential KV cache updates).
        overall_exe_cycle = overall_inst_num * 2
        execution_time = overall_exe_cycle / self.frequency

        if verbose:
            print("\nDecode Execution Distribution:")
            print(f"  RMS Layer:       {rms_count / overall_inst_num * 100:.2f}%")
            print(f"  Projection:      {projection_count / overall_inst_num * 100:.2f}%")
            print(f"  Flash Attention: {flash_attention_count / overall_inst_num * 100:.2f}%")
            print(f"  Residual:        {residual_count / overall_inst_num * 100:.2f}%")
            print(f"  Feed Forward:    {feed_forward_count / overall_inst_num * 100:.2f}%")

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

    def compute_llada_step_time(self, seq_len: int, verbose: bool = False) -> float:
        """
        Compute cost of one LLaDA denoising step in seconds.

        One step = full-sequence prefill (bidirectional attn) + full-seq LM head + full-seq softmax.
        LLaDA has NO autoregressive decode — every step processes the entire sequence.
        """
        mode = "prefill"
        kv_size = seq_len
        overall_exe_cycle = 0

        overall_exe_cycle += self.perf.embeddings(self.hidden_size, seq_len, self.device_batch_size, mode)

        rms = self.perf.rms_layer(self.hidden_size, seq_len, self.device_batch_size, mode)
        proj = self.perf.projection(
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            seq_len,
            self.device_batch_size,
            mode,
        )
        attn = self.perf.flash_attention(
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            seq_len,
            kv_size,
            self.device_batch_size,
            mode,
        )
        res = self.perf.residual(self.hidden_size, seq_len, self.device_batch_size, mode)
        ffn = self.perf.feed_forward(
            self.hidden_size, self.intermediate_size, seq_len, self.device_batch_size, mode
        )

        transformer_block_cycles = rms + proj + attn + res + rms + ffn
        overall_exe_cycle += transformer_block_cycles * self.num_hidden_layers

        # LLaDA: LM head and softmax over ALL positions (not just last token)
        lm = self.perf.lm_head_full_seq(self.hidden_size, self.vocab_size, seq_len, self.device_batch_size)
        smax = self.perf.softmax_full_seq(self.vocab_size, seq_len, self.device_batch_size)
        overall_exe_cycle += lm + smax

        if verbose:
            total = transformer_block_cycles * self.num_hidden_layers + lm + smax
            print("\nPer-step execution distribution:")
            print(f"  Transformer body: {transformer_block_cycles * self.num_hidden_layers / total * 100:.1f}%")
            print(f"    RMS Norm:        {rms * 2 * self.num_hidden_layers / total * 100:.1f}%")
            print(f"    Projection:      {proj * self.num_hidden_layers / total * 100:.1f}%")
            print(f"    Flash Attention: {attn * self.num_hidden_layers / total * 100:.1f}%")
            print(f"    Residual:        {res * self.num_hidden_layers / total * 100:.1f}%")
            print(f"    Feed Forward:    {ffn * self.num_hidden_layers / total * 100:.1f}%")
            print(f"  LM head (full seq): {lm / total * 100:.1f}%")
            print(f"  Softmax (full seq): {smax / total * 100:.1f}%")

        return overall_exe_cycle / self.frequency

    def compute_llada_inference(self, seq_len: int, diffusion_steps: int, verbose: bool = True) -> tuple:
        """
        Compute total LLaDA inference cost for T denoising steps.

        LLaDA inference: T steps, each processing the full sequence (prompt + output together).
        No autoregressive decode. Output tokens are unmasked progressively across steps.

        Returns:
            tuple: (total_time_seconds, output_tokens_per_second)
        """
        step_time = self.compute_llada_step_time(seq_len, verbose=verbose)
        total_time = (diffusion_steps * step_time) / self.device_num
        # Throughput: output tokens generated per second (batch * seq_len across all steps)
        output_tokens_per_second = (self.batch_size * seq_len) / total_time

        return total_time, output_tokens_per_second


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
        description="LLaMA Performance Model - Compute TPS and TTFT for LLM inference on PLENA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python llama_model.py --model llama-3.1-8b --model-lib ./Model_Lib --config ./plena_settings.toml --isa-lib ./customISA_lib.json
  python llama_model.py --model-path ./custom_model.json --config ./plena_settings.toml --isa-lib ./customISA_lib.json
  python llama_model.py --list-models --model-lib ./Model_Lib
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
    parser.add_argument("--output-seq", "-o", type=int, default=1024, help="Output sequence length (default: 1024)")
    parser.add_argument("--device-num", "-d", type=int, default=1, help="Number of devices (default: 1)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")
    parser.add_argument("--llada", action="store_true", help="Run LLaDA diffusion inference model instead of AR")
    parser.add_argument("--diffusion-steps", type=int, default=64, help="Number of LLaDA denoising steps (default: 64)")
    parser.add_argument("--seq-len", type=int, default=None, help="Total sequence length for LLaDA (prompt+output, default: input_seq)")

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
    model = LLaMAModel(
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

    if args.llada:
        seq_len = args.seq_len if args.seq_len else args.input_seq
        if not args.quiet:
            print(f"\nLLaDA mode: {args.diffusion_steps} denoising steps, seq_len={seq_len}")
        total_time, tps = model.compute_llada_inference(seq_len, args.diffusion_steps, verbose=not args.quiet)
        if args.json:
            result = {
                "model": args.model or args.model_path,
                "mode": "llada_diffusion",
                "batch_size": args.batch_size,
                "seq_len": seq_len,
                "diffusion_steps": args.diffusion_steps,
                "total_time_seconds": total_time,
                "total_time_ms": total_time * 1000,
                "output_tokens_per_second": tps,
            }
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "=" * 50)
            print("LLaDA Diffusion Performance Results")
            print("=" * 50)
            print(f"Denoising steps (T):        {args.diffusion_steps}")
            print(f"Sequence length:            {seq_len}")
            print(f"Total inference time:       {total_time:.6f} s ({total_time * 1000:.3f} ms)")
            print(f"Output tokens/sec:          {tps:.2f}")
            print("=" * 50)
        return

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
        print("\n" + "=" * 50)
        print("Performance Results")
        print("=" * 50)
        print(f"TTFT (Time to First Token): {ttft:.6f} s ({ttft * 1000:.3f} ms)")
        print(f"TPS (Tokens Per Second):    {tps:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
