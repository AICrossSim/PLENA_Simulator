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
        self.vector_sram_size = hardware_config["VECTOR_SRAM_SIZE"] * self.vlen
        self.prefetch_v_amount = hardware_config["HBM_V_Prefetch_Amount"]

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
        """
        # Placeholder implementation - replace with instruction-level counting
        setting_inst_num = 10
        loop_inst_num = 8
        compute_cycle = 0
        loop_num = hidden_size // self.vlen
        overall_cycles = 0

        if mode == "prefill":
            compute_cycle = setting_inst_num * self.instr["S_BASIC"] + loop_num * loop_inst_num * seq_len * self.instr["V_BASIC"] * batch_size
            if hidden_size * seq_len * batch_size > self.vector_sram_size:
                # Can't hold everything in vector SRAM.
                overall_cycles += compute_cycle + ((hidden_size * seq_len * batch_size - self.vector_sram_size) // (self.vlen * self.prefetch_v_amount)) * self.instr["H_PREFETCH_V"] * 2
            else:
                overall_cycles += compute_cycle
        else:  # decode
            overall_cycles = setting_inst_num * self.instr["S_BASIC"] + loop_num * loop_inst_num * self.instr["V_BASIC"] * batch_size

        return overall_cycles

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
        """
        compute_cycle = 0
        overall_cycles = 0
        # Batch and Sequence Dim are combined in Projection.
        bs_dim = seq_len * batch_size

        if mode == "prefill":
            # Projection of Q
            compute_cycle += math.ceil(bs_dim / self.blen) * (math.ceil(hidden_size / self.mlen) * (math.ceil(hidden_size / self.blen))) * self.instr["M_MM"]
            # RoPE of Q
            compute_cycle += num_attention_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of K
            compute_cycle += math.ceil(bs_dim / self.blen) * (math.ceil((num_kv_heads * head_dim) / self.mlen) * (math.ceil((num_kv_heads * head_dim) / self.blen))) * self.instr["M_MM"]
            # RoPE of K
            compute_cycle += num_kv_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of V
            compute_cycle += math.ceil(bs_dim / self.blen) * (math.ceil((num_kv_heads * head_dim) / self.mlen) * (math.ceil((num_kv_heads * head_dim) / self.blen))) * self.instr["M_MM"]
            # Activation (Q) Manipulation
            if hidden_size * seq_len * batch_size > self.vector_sram_size:
                # Can't hold everything in vector SRAM.
                overall_cycles += compute_cycle + ((hidden_size * seq_len * batch_size - self.vector_sram_size) // (self.vlen * self.prefetch_v_amount)) * self.instr["H_PREFETCH_V"] * 2
            else:
                overall_cycles += compute_cycle
            # Store K,V Cache
            overall_cycles += 2 * ((batch_size * seq_len * num_kv_heads * head_dim) // (self.vlen * self.prefetch_v_amount)) * self.instr["H_STORE_V"]
        else:  # decode
            # Projection of Q
            compute_cycle += math.ceil(batch_size / self.blen) * math.ceil(hidden_size / self.mlen) * math.ceil(hidden_size / self.blen) * self.instr["M_MM"]
            # RoPE of Q
            compute_cycle += num_attention_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of K
            compute_cycle += math.ceil(batch_size / self.blen) * (math.ceil((num_kv_heads * head_dim) / self.mlen) * (math.ceil((num_kv_heads * head_dim) / self.blen))) * self.instr["M_MM"]
            # RoPE of K
            compute_cycle += num_kv_heads * math.ceil(bs_dim / self.vlen) * self.instr["V_BASIC"]
            # Projection of V
            compute_cycle += math.ceil(batch_size / self.blen) * (math.ceil((num_kv_heads * head_dim) / self.mlen) * (math.ceil((num_kv_heads * head_dim) / self.blen))) * self.instr["M_MM"]
            overall_cycles += compute_cycle
            # Store K,V Cache
            overall_cycles += 2 * ((batch_size * num_kv_heads * head_dim) // (self.vlen * self.prefetch_v_amount)) * self.instr["H_STORE_V"]
        return overall_cycles

    def flash_attention(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        kv_size: int,
        batch_size: int,
        mode: str = "prefill"
    ) -> int:
        """
        Flash attention cycle count.
        This assumes the GQA mode Attention.
        """

        compute_cycle = 0
        inner_compute_cycles = 0
        overall_cycles = 0
        # Loop Settings

        kv_head_loop = num_kv_heads
        inner_q_head_loop = num_attention_heads // num_kv_heads
        tr = math.ceil(seq_len / self.mlen)
        tc = math.ceil(kv_size / self.mlen)

        if mode == "prefill":
            # inner loop
            # QKT ( per KV head and Grouped Q heads)
            inner_compute_cycles += 4 + self.instr["M_BTMM"] + self.instr["H_PREFETCH_M"]
            # online softmax
            inner_compute_cycles += self.mlen *  (8 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"]) * inner_q_head_loop
            # Compute PV
            inner_compute_cycles += 4 + math.ceil(head_dim / self.blen) * (math.ceil(self.mlen / self.blen)) * self.instr["M_MM"] * inner_q_head_loop
            # Compute O
            inner_compute_cycles += self.mlen * (2 * self.instr["V_BASIC"] + 4) * inner_q_head_loop
            # Compute Scaling
            inner_compute_cycles += self.mlen * (1 * self.instr["V_BASIC"] + 4 + self.instr["S_RECI_FP"]) * inner_q_head_loop
            overall_cycles = inner_compute_cycles * tr * tc * kv_head_loop * batch_size
        else:  # decode
            # inner loop
            # QKT ( per KV head and Grouped Q heads)
            inner_compute_cycles += 4 + self.instr["M_BTMV"] + self.instr["H_PREFETCH_M"]
            # online softmax
            inner_compute_cycles += (8 * self.instr["V_BASIC"] + 2 * self.instr["S_BASIC"] + self.instr["S_EXP_FP"]) * inner_q_head_loop
            # Compute PV
            inner_compute_cycles += 4 + math.ceil(head_dim / self.blen) * (self.instr["M_MV"] + 2*self.instr["S_ADDI_INT"]) * inner_q_head_loop
            # Compute O
            inner_compute_cycles +=  (2 * self.instr["V_BASIC"] + 1) * inner_q_head_loop
            # Compute Scaling
            inner_compute_cycles +=  (1 * self.instr["V_BASIC"] + self.instr["S_RECI_FP"]) * inner_q_head_loop
            overall_cycles = inner_compute_cycles * tr * tc * kv_head_loop * batch_size
        
        return overall_cycles

    def residual(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill"
    ) -> int:
        """Residual connection cycle count."""
        iteration = hidden_size // self.vlen
        compute_cycle = 0
        overall_cycles = 0

        if mode == "prefill":
            compute_cycle = (self.instr["V_ADD_VV"] + 3) * seq_len * iteration * batch_size
            if hidden_size * seq_len * batch_size > self.vector_sram_size:
                # Can't hold everything in vector SRAM.
                overall_cycles += compute_cycle + ((hidden_size * seq_len * batch_size - self.vector_sram_size) // (self.vlen * self.prefetch_v_amount)) * self.instr["H_PREFETCH_V"] * 2
            else:
                overall_cycles += compute_cycle
        else:
            compute_cycle = (self.instr["V_ADD_VV"] + 3) * iteration * batch_size
            overall_cycles += compute_cycle
        return overall_cycles

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
        overall_cycles = 0

        if mode == "prefill":
            # Upsize Linear and Gate
            overall_cycles += 2 * math.ceil(intermediate_size / self.blen) * math.ceil((seq_len * batch_size) / self.blen) * math.ceil(hidden_size / self.mlen) * self.instr["M_MM"]
            # SiLU
            overall_cycles += math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"] * seq_len * batch_size
            # Downsize Linear
            overall_cycles += math.ceil(intermediate_size / self.blen) * math.ceil(hidden_size / self.mlen) * math.ceil((seq_len * batch_size) / self.blen) * self.instr["M_MM"]
        else:
            # Upsize Linear and Gate
            overall_cycles += 2 * math.ceil(intermediate_size / self.blen) * math.ceil(hidden_size / self.mlen) * math.ceil(batch_size / self.blen) * self.instr["M_MM"]
            # SiLU
            overall_cycles += math.ceil(intermediate_size / self.vlen) * 6 * self.instr["V_BASIC"] * batch_size
            # Downsize Linear
            overall_cycles += math.ceil(intermediate_size / self.blen) * math.ceil(hidden_size / self.mlen) * math.ceil(batch_size / self.blen) * self.instr["M_MM"]

        return overall_cycles

    def embeddings(
        self,
        hidden_size: int,
        seq_len: int,
        batch_size: int,
        mode: str = "prefill"
    ) -> int:
        """Embedding layer cycle count."""
        setting_inst_num = 3
        overall_cycles = setting_inst_num * self.instr["S_BASIC"]

        if mode == "prefill":
            # Prefetch embedding vectors for each token
            overall_cycles += seq_len * batch_size * math.ceil(hidden_size / self.vlen) * self.instr["H_PREFETCH_V"]
        else:  # decode
            # Single token embedding lookup per batch
            overall_cycles += batch_size * math.ceil(hidden_size / self.vlen) * self.instr["H_PREFETCH_V"]

        return overall_cycles

    def lm_head(
        self,
        hidden_size: int,
        vocab_size: int,
        batch_size: int
    ) -> int:
        """LM head cycle count (linear projection to vocab)."""
        setting_inst_num = 3
        overall_cycles = setting_inst_num * self.instr["S_BASIC"]

        # Matrix multiply: [batch_size, hidden_size] x [hidden_size, vocab_size]
        overall_cycles += math.ceil(batch_size / self.blen) * math.ceil(hidden_size / self.mlen) * math.ceil(vocab_size / self.blen) * self.instr["M_MM"]

        return overall_cycles


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

        overall_exe_cycle += self.plena.embeddings(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)

        rms = self.plena.rms_layer(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
        print(f"RMS: {rms}")
        proj = self.plena.projection(self.hidden_size, self.num_attention_heads, self.num_key_value_heads, self.head_dim, self.input_seq_len, self.device_batch_size, mode)
        print(f"Projection: {proj}")
        attn = self.plena.flash_attention(self.num_attention_heads, self.num_key_value_heads, self.head_dim, self.input_seq_len, kv_size, self.device_batch_size, mode)
        print(f"Flash Attention: {attn}")
        res = self.plena.residual(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
        print(f"Residual: {res}")
        ffn = self.plena.feed_forward(self.hidden_size, self.intermediate_size, self.input_seq_len, self.device_batch_size, mode)
        print(f"Feed Forward: {ffn}")

        transformer_block_cycles = rms + proj + attn + res + rms + ffn
        overall_exe_cycle += transformer_block_cycles * self.num_hidden_layers
        lm_head = self.plena.lm_head(self.hidden_size, self.vocab_size, self.device_batch_size)
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
                flash_attention_count += self.plena.flash_attention(self.num_attention_heads, self.num_key_value_heads, self.head_dim, 1, kv_size, self.device_batch_size, mode)
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
