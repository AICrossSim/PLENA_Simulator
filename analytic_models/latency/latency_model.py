"""
Integrated Latency Model for PLENA Simulator.

Architecture:
- PLENA_Latency: Per-layer hardware latency modeling using instruction latencies
- LLaMA_Perf_Model: LLM architecture-level performance model (prefill, decode, TPS, TTFT)

Usage:
    python latency_model.py --model llama-3.1-8b
    python latency_model.py --model llama-3.1-8b --batch-size 4 --input-seq 2048 --output-seq 1024
    python latency_model.py --list-models
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
CUSTOM_ISA_LIB_PATH = Path(__file__).parent / "customISA_lib.json"


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

    # Extract CONFIG section values
    if "CONFIG" in data:
        for param_name, val in data["CONFIG"].items():
            if isinstance(val, dict) and "value" in val:
                hardware_config[param_name] = val["value"]

    # Extract LATENCY section values
    if "LATENCY" in data:
        for param_name, val in data["LATENCY"].items():
            if isinstance(val, dict):
                if "dc_lib_en" in val:
                    hardware_config[param_name] = val["dc_lib_en"]
                elif "value" in val:
                    hardware_config[param_name] = val["value"]

    return hardware_config


# =============================================================================
# Instruction Latency Model (Pipelined)
# =============================================================================

def build_pipelined_latency(hardware_config: dict, custom_isa_path: str = None) -> dict:
    """
    Build pipelined instruction latency dict from customISA_lib.json.
    Evaluates expressions using hardware config values.

    Args:
        hardware_config: Dict with MLEN, BLEN, VLEN, and LATENCY params

    Returns:
        dict: Instruction name -> pipelined latency cycles
              e.g., {"M_MM": 8, "V_ADD_VV": 2, ...}
    """
    if custom_isa_path is None:
        custom_isa_path = str(CUSTOM_ISA_LIB_PATH)

    with open(custom_isa_path, "r") as f:
        custom_isa_lib = json.load(f)

    # Build config dict for eval
    configs = hardware_config.copy()
    if "MLEN" in configs and "BLEN" in configs:
        configs["SA_ACC_CYCLES"] = int(math.log2(configs["MLEN"] / configs["BLEN"]) + 1)

    instr_latency = {}
    for instr_name, instr_data in custom_isa_lib.items():
        if "pipelined" in instr_data:
            instr_latency[instr_name] = eval(instr_data["pipelined"], {}, configs)
        else:
            raise ValueError(f"Instruction '{instr_name}' missing 'pipelined' field.")

    return instr_latency


# =============================================================================
# PLENA_Latency: Per-Layer Hardware Latency Model
# =============================================================================

class PLENA_Latency:
    """
    Per-layer hardware latency model for PLENA accelerator.

    On init, builds pipelined instruction latency dict from customISA_lib.json
    using hardware config values (MLEN, BLEN, VLEN, latency params).

    Attributes:
        hardware_config: Dict with MLEN, BLEN, VLEN, etc.
        mlen, blen, vlen: Hardware dimensions
        instr: Dict mapping instruction name -> pipelined latency cycles

    Usage:
        plena = PLENA_Latency(hardware_config)

        # Access instruction latencies:
        plena.instr["M_MM"]       # Matrix multiply latency
        plena.instr["V_ADD_VV"]   # Vector add latency
        plena.instr["V_EXP_V"]    # Vector exp latency

        # Use in layer computation:
        total_cycles = num_mm * plena.instr["M_MM"] + num_vadd * plena.instr["V_ADD_VV"]
    """

    def __init__(self, hardware_config: dict):
        """
        Initialize PLENA_Latency model.

        Args:
            hardware_config: Hardware configuration dict from plena_settings.toml
        """
        self.hardware_config = hardware_config
        self.mlen = hardware_config["MLEN"]
        self.blen = hardware_config["BLEN"]
        self.vlen = hardware_config["VLEN"]

        # Build instruction latency dict
        self.instr = build_pipelined_latency(hardware_config)

    def print_instr_latency(self):
        """Print all instruction latencies."""
        print(f"\nInstruction Latencies (MLEN={self.mlen}, BLEN={self.blen}, VLEN={self.vlen}):")
        print("-" * 40)
        for name, latency in sorted(self.instr.items()):
            print(f"  {name:20s}: {latency} cycles")
        print()

    def get(self, instr_name: str) -> int:
        """Get latency for a specific instruction."""
        if instr_name not in self.instr:
            raise KeyError(f"Unknown instruction: {instr_name}. Available: {list(self.instr.keys())}")
        return self.instr[instr_name]

    # -------------------------------------------------------------------------
    # Layer-level latency computation methods
    # TODO: Fill in with actual instruction sequences for each layer
    # -------------------------------------------------------------------------

    def rms_layer(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill"
    ) -> int:
        """
        RMSNorm layer cycle count.

        TODO: Implement using self.instr for instruction latencies
        Example:
            cycles = 0
            cycles += num_loads * self.instr["H_PREFETCH_V"]
            cycles += num_mul * self.instr["V_MUL_VV"]
            cycles += num_add * self.instr["V_ADD_VV"]
            ...
            return cycles
        """
        # Placeholder implementation - replace with instruction-level counting
        setting_inst_num = 10
        loop_inst_num = 8
        loop_num = hidden_size // self.vlen

        if mode == "prefill":
            instruction_num = setting_inst_num + loop_num * loop_inst_num * seq_len
        else:  # decode
            instruction_num = setting_inst_num + loop_num * loop_inst_num

        return instruction_num * batch_size

    def projection(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill"
    ) -> int:
        """
        Q, K, V projection + RoPE cycle count.

        TODO: Implement using self.instr for instruction latencies
        """
        overall_inst_num = 0

        if mode == "prefill":
            overall_inst_num += batch_size * (
                math.ceil(hidden_size / self.blen) *
                (math.ceil(hidden_size / self.mlen) * (math.ceil(seq_len / self.blen) * 2 + 10))
            )
            overall_inst_num += batch_size * (num_attention_heads * math.ceil(seq_len / self.vlen)) * 3

            overall_inst_num += batch_size * (
                math.ceil((num_kv_heads * head_dim) / self.blen) *
                (math.ceil(hidden_size / self.mlen) * (math.ceil(seq_len / self.blen) * 2 + 10))
            )
            overall_inst_num += batch_size * (num_kv_heads * math.ceil(seq_len / self.vlen)) * 3

            overall_inst_num += batch_size * (
                math.ceil((num_kv_heads * head_dim) / self.blen) *
                (math.ceil(hidden_size / self.mlen) * (math.ceil(seq_len / self.blen) * 2 + 10))
            )
        else:  # decode
            overall_inst_num += math.ceil(hidden_size / self.blen) * (math.ceil(hidden_size / self.mlen) * 2 + 10)
            overall_inst_num += batch_size * num_attention_heads * 4

            overall_inst_num += math.ceil((num_kv_heads * head_dim) / self.blen) * (math.ceil(hidden_size / self.mlen) * 2 + 10)
            overall_inst_num += batch_size * num_kv_heads * 4

            overall_inst_num += math.ceil((num_kv_heads * head_dim) / self.blen) * (math.ceil(hidden_size / self.mlen) * 2 + 10)

        return overall_inst_num

    def flash_attention(
        self,
        num_attention_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill",
        partitioned_optimized: bool = True
    ) -> int:
        """
        Flash attention cycle count.

        TODO: Implement using self.instr for instruction latencies
        """
        prefill_len = min(seq_len, self.mlen)
        decode_len = min(kv_size, self.mlen)
        max_head_per_mlen = math.ceil(self.mlen / head_dim)
        per_head_iter = math.ceil(num_attention_heads / max_head_per_mlen)

        overall_inst_num = 0

        if mode == "prefill":
            if partitioned_optimized:
                for _ in range(math.ceil(seq_len / self.mlen)):
                    overall_inst_num += self.mlen * max_head_per_mlen
                    for _ in range(math.ceil(seq_len / self.mlen)):
                        overall_inst_num += (2 + prefill_len * 5) * max_head_per_mlen
                        overall_inst_num += math.ceil(self.mlen / self.blen) * self.blen * math.ceil(head_dim / self.blen) * max_head_per_mlen
                        overall_inst_num += (prefill_len * 5 + 4) * max_head_per_mlen
                        overall_inst_num += 8 * max_head_per_mlen
                return overall_inst_num * per_head_iter * batch_size
            else:
                for _ in range(math.ceil(seq_len / self.mlen)):
                    for _ in range(math.ceil(seq_len / self.mlen)):
                        overall_inst_num += math.ceil(prefill_len / self.blen) * math.ceil(head_dim / self.mlen) * self.blen * math.ceil(prefill_len / self.blen)
                        overall_inst_num += 2 + prefill_len * 5
                        overall_inst_num += math.ceil(self.mlen / self.blen) * self.blen * math.ceil(head_dim / self.blen)
                        overall_inst_num += prefill_len * 5 + 4
                        overall_inst_num += 8
                return overall_inst_num * batch_size * num_attention_heads
        else:  # decode
            if partitioned_optimized:
                overall_inst_num = decode_len
                overall_inst_num += (2 + decode_len * 3) * max_head_per_mlen
                overall_inst_num += math.ceil(decode_len / self.mlen) * self.blen * math.ceil(head_dim / self.blen) * max_head_per_mlen
                overall_inst_num += (decode_len * 3 + 4) * max_head_per_mlen
                overall_inst_num += 8 * max_head_per_mlen
                return overall_inst_num * batch_size * math.ceil(kv_size / self.mlen) * per_head_iter
            else:
                overall_inst_num = math.ceil(self.mlen / self.blen) * math.ceil(head_dim / self.mlen) * self.blen
                overall_inst_num += 2 + decode_len * 3
                overall_inst_num += math.ceil(head_dim / self.blen) * (4 + math.ceil(decode_len / self.blen) * 4)
                overall_inst_num += decode_len * 3 + 4
                overall_inst_num += 8
                return overall_inst_num * batch_size * math.ceil(kv_size / self.mlen) * num_attention_heads

    def residual(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill"
    ) -> int:
        """Residual connection cycle count."""
        iteration = hidden_size // self.vlen

        if mode == "prefill":
            overall_inst_num = (5 * iteration + 3) * seq_len
        else:
            overall_inst_num = 5 * iteration + 3

        return overall_inst_num * batch_size

    def feed_forward(
        self,
        hidden_size: int,
        intermediate_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill"
    ) -> int:
        """
        Feed-forward (MLP) layer cycle count.

        TODO: Implement using self.instr for instruction latencies
        """
        overall_inst_num = 0

        if mode == "prefill":
            overall_inst_num += 2 * math.ceil(intermediate_size / self.blen) * math.ceil(seq_len / self.blen) * math.ceil(hidden_size / self.mlen) * self.blen
            overall_inst_num += math.ceil(intermediate_size / self.vlen) * 3 * seq_len
            overall_inst_num += math.ceil(intermediate_size / self.blen) * math.ceil(hidden_size / self.mlen) * math.ceil(seq_len / self.blen) * self.blen
            overall_inst_num *= batch_size
        else:
            overall_inst_num += 2 * math.ceil(intermediate_size / self.blen) * math.ceil(hidden_size / self.mlen) * math.ceil(batch_size / self.blen) * self.blen * math.ceil(batch_size / self.blen)
            overall_inst_num += math.ceil(intermediate_size / self.vlen) * 5 * batch_size
            overall_inst_num += math.ceil(intermediate_size / self.blen) * math.ceil(hidden_size / self.mlen) * math.ceil(batch_size / self.blen) * self.blen * math.ceil(batch_size / self.blen)

        return overall_inst_num

    def embeddings(
        self,
        hidden_size: int,
        seq_len: int,
        mode: str = "prefill"
    ) -> int:
        """Embedding layer cycle count."""
        overall_inst_num = 3
        if mode == "prefill":
            overall_inst_num += seq_len * math.ceil(hidden_size / self.blen) * math.ceil(hidden_size / self.mlen) * (self.blen * 2 + 1) + 4
        else:
            overall_inst_num += math.ceil(hidden_size / self.blen) * math.ceil(hidden_size / self.mlen) * (self.blen * 2 + 1) + 4

        return overall_inst_num

    def lm_head(self, hidden_size: int, vocab_size: int) -> int:
        """LM head cycle count."""
        overall_inst_num = 3
        overall_inst_num += math.ceil(hidden_size / self.blen) * math.ceil(vocab_size / self.mlen) * (self.blen * 2 + 1) + 4
        return overall_inst_num


# =============================================================================
# LLaMA_Perf_Model: LLM Performance Model
# =============================================================================

class LLaMA_Perf_Model:
    """
    LLaMA architecture performance model.
    Uses PLENA_Latency for per-layer cycle counting and computes
    overall inference performance (prefill time, decode time, TPS, TTFT).
    """

    def __init__(
        self,
        model_config_path: str,
        hardware_config: dict,
        batch_size: int = 1,
        input_seq_len: int = 2048,
        output_seq_len: int = 128,
        device_num: int = 1,
        frequency_hz: float = 1e9
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

        self.frequency = frequency_hz
        self.hardware_config = hardware_config
        self.batch_size = batch_size
        self.device_batch_size = batch_size // device_num
        self.device_num = device_num

        # Initialize PLENA hardware latency model
        self.plena = PLENA_Latency(hardware_config)

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
        print(f"MLEN: {self.plena.mlen}, BLEN: {self.plena.blen}, VLEN: {self.plena.vlen}")
        print("=" * 50)

    def compute_prefill_time(self, verbose: bool = True) -> float:
        """Compute prefill phase execution time in seconds."""
        mode = "prefill"
        kv_size = self.input_seq_len
        overall_exe_cycle = 0

        overall_exe_cycle += self.plena.embeddings(self.hidden_size, self.input_seq_len, mode)

        rms = self.plena.rms_layer(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
        proj = self.plena.projection(self.hidden_size, self.num_attention_heads, self.num_key_value_heads, self.head_dim, self.input_seq_len, self.device_batch_size, mode)
        attn = self.plena.flash_attention(self.num_attention_heads, self.head_dim, self.input_seq_len, kv_size, self.device_batch_size, mode)
        res = self.plena.residual(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
        ffn = self.plena.feed_forward(self.hidden_size, self.intermediate_size, self.input_seq_len, self.device_batch_size, mode)

        transformer_block_cycles = rms + proj + attn + res + rms + ffn
        overall_exe_cycle += transformer_block_cycles * self.num_hidden_layers
        lm_head = self.plena.lm_head(self.hidden_size, self.vocab_size)
        overall_exe_cycle += lm_head

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
                rms_count += self.plena.rms_layer(self.hidden_size, 1, self.device_batch_size, mode) * 2
                projection_count += self.plena.projection(self.hidden_size, self.num_attention_heads, self.num_key_value_heads, self.head_dim, 1, self.device_batch_size, mode)
                flash_attention_count += self.plena.flash_attention(self.num_attention_heads, self.head_dim, 1, kv_size, self.device_batch_size, mode)
                residual_count += self.plena.residual(self.hidden_size, 1, self.device_batch_size, mode)
                feed_forward_count += self.plena.feed_forward(self.hidden_size, self.intermediate_size, 1, self.device_batch_size, mode)
            kv_size += 1

        overall_inst_num = rms_count + projection_count + flash_attention_count + residual_count + feed_forward_count
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
        description="PLENA Latency Model - Compute TPS and TTFT for LLM inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  {', '.join(list_available_models())}

Examples:
  python latency_model.py --model llama-3.1-8b
  python latency_model.py --model llama-3.3-70b --batch-size 8
  python latency_model.py --model-path ./custom_model.json
"""
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", "-m", help="Model name from Model_Lib")
    model_group.add_argument("--model-path", help="Full path to model config JSON")
    model_group.add_argument("--list-models", "-l", action="store_true", help="List available models")

    parser.add_argument("--config", "-c", default=str(DEFAULT_CONFIG_PATH), help="Path to hardware config TOML")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--input-seq", "-i", type=int, default=2048, help="Input sequence length (default: 2048)")
    parser.add_argument("--output-seq", "-o", type=int, default=1024, help="Output sequence length (default: 1024)")
    parser.add_argument("--device-num", "-d", type=int, default=1, help="Number of devices (default: 1)")
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
    model = LLaMA_Perf_Model(
        model_config_path=model_path,
        hardware_config=hardware_config,
        batch_size=args.batch_size,
        input_seq_len=args.input_seq,
        output_seq_len=args.output_seq,
        device_num=args.device_num
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
            "tps": tps
        }
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 50)
        print("Performance Results")
        print("=" * 50)
        print(f"TTFT (Time to First Token): {ttft:.6f} s ({ttft*1000:.3f} ms)")
        print(f"TPS (Tokens Per Second):    {tps:.2f}")
        print("=" * 50)


if __name__ == "__main__":
    main()
