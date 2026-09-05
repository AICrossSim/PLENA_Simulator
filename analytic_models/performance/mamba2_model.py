"""
Mamba-2 Performance Model for PLENA Simulator.

Uses PerfModel from perf_model.py for per-layer cycle counting and computes
overall inference performance (prefill time, decode time, TPS, TTFT).

Structure of one Mamba-2 mixer block (following HF `Mamba2Mixer`):

    x_norm            = RMSNorm(x)
    z, xBC, dt        = in_proj(x_norm)          hidden -> 2*d_inner + 2*n_groups*N + H
    xBC               = SiLU(causal_conv1d(xBC)) over conv_dim = d_inner + 2*n_groups*N
    dt                = clamp(softplus(dt + dt_bias), dt_min, dt_max)
    y                 = SSD(x, dt, A, B, C)      chunked scan in prefill,
                                                 single-step recurrence in decode
    y                 = RMSNorm(y * SiLU(z))     gated norm
    out               = out_proj(y)              d_inner -> hidden
    x                 = x + out                  residual

There is no attention and no KV cache: the entire history is compressed into a
fixed `num_heads * head_dim * state_size` recurrent state (plus a
`conv_dim * (conv_kernel-1)` conv window), so decode cost is O(1) in context
length. Whether that translates into a measured win depends on the memory
bandwidth term in perf_model.py — with no bandwidth term the comparison is
meaningless, because the growing KV cache costs nothing to read.

Usage:
    python mamba2_model.py --model mamba2-2.7b --model-lib ./Model_Lib \
        --config ./plena_settings.toml --isa-lib ./customISA_lib.json
    python mamba2_model.py --list-models --model-lib ./Model_Lib
"""

import argparse
import json
from pathlib import Path

try:  # script-style invocation (matches llama_model.py / gpt_oss_model.py)
    from perf_model import PerfModel, load_hardware_config_from_toml
except ImportError:  # package-style import
    from analytic_models.performance.perf_model import PerfModel, load_hardware_config_from_toml


class Mamba2Model:
    """
    Mamba-2 architecture performance model.

    Uses PerfModel for per-layer cycle counting and computes overall inference
    performance (prefill time, decode time, TPS, TTFT).
    """

    def __init__(
        self,
        model_config_path: str,
        hardware_config,
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
        self.num_hidden_layers = model_param["num_hidden_layers"]
        self.vocab_size = model_param["vocab_size"]

        # SSM geometry. HF's Mamba2Config names these exactly this way.
        self.state_size = model_param.get("state_size", 128)
        self.expand = model_param.get("expand", 2)
        self.d_inner = model_param.get("intermediate_size", self.expand * self.hidden_size)
        self.head_dim = model_param.get("head_dim", 64)
        self.num_heads = model_param.get("num_heads", self.d_inner // self.head_dim)
        self.n_groups = model_param.get("n_groups", 1)
        self.conv_kernel = model_param.get("conv_kernel", 4)
        self.chunk_size = model_param.get("chunk_size", 256)

        if self.num_heads * self.head_dim != self.d_inner:
            raise ValueError(
                f"num_heads*head_dim ({self.num_heads}*{self.head_dim}) != d_inner ({self.d_inner}); "
                "check head_dim / expand / intermediate_size in the model config"
            )

        # in_proj emits [z | x | B | C | dt]
        self.in_proj_out = 2 * self.d_inner + 2 * self.n_groups * self.state_size + self.num_heads
        # conv1d runs over [x | B | C] only
        self.conv_dim = self.d_inner + 2 * self.n_groups * self.state_size

        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len

        self.frequency = frequency_hz
        self.hardware_config = hardware_config
        self.batch_size = batch_size
        self.device_batch_size = max(1, batch_size // device_num)
        self.device_num = device_num

        self.perf = PerfModel(hardware_config, custom_isa_path)

    # -------------------------------------------------------------------------
    # Reporting helpers
    # -------------------------------------------------------------------------

    @property
    def state_elems_per_layer(self) -> int:
        """Recurrent SSM state size per sequence, per layer (elements)."""
        return self.num_heads * self.head_dim * self.state_size

    @property
    def conv_state_elems_per_layer(self) -> int:
        return self.conv_dim * (self.conv_kernel - 1)

    def print_config(self):
        """Print model and hardware configuration."""
        state_bytes = (self.state_elems_per_layer + self.conv_state_elems_per_layer) * self.perf.state_bytes
        print("=" * 60)
        print("Mamba-2 Model Configuration")
        print("=" * 60)
        print(f"Hidden size:          {self.hidden_size}")
        print(f"Num hidden layers:    {self.num_hidden_layers}")
        print(f"d_inner (expand={self.expand}):  {self.d_inner}")
        print(f"Num heads:            {self.num_heads}")
        print(f"Head dim:             {self.head_dim}")
        print(f"State size (N):       {self.state_size}")
        print(f"Num groups:           {self.n_groups}")
        print(f"Conv kernel:          {self.conv_kernel}")
        print(f"Conv dim:             {self.conv_dim}")
        print(f"Chunk size:           {self.chunk_size}")
        print(f"in_proj out features: {self.in_proj_out}")
        print(f"Vocab size:           {self.vocab_size}")
        print("-" * 60)
        print("Recurrent State (context-independent)")
        print("-" * 60)
        print(f"SSM state / layer:    {self.state_elems_per_layer:,} elems")
        print(f"Conv state / layer:   {self.conv_state_elems_per_layer:,} elems")
        print(f"Total state (all L):  {state_bytes * self.num_hidden_layers / 1e6:.2f} MB per sequence")
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
        print(f"HBM bandwidth: {self.perf.hbm_bytes_per_cycle:.1f} B/cycle")
        print(f"bytes/elem  weight={self.perf.weight_bytes}  act={self.perf.act_bytes}  state={self.perf.state_bytes}")
        print("=" * 60)

    # -------------------------------------------------------------------------
    # One mixer block
    # -------------------------------------------------------------------------

    def _block_cycles(self, seq_len: int, mode: str) -> dict:
        """Cycle counts for one Mamba-2 block, broken down by stage."""
        b = self.device_batch_size
        parts = {}

        # pre-mixer RMSNorm
        parts["rms"] = self.perf.rms_layer(self.hidden_size, seq_len, b, mode)

        # in_proj: hidden -> 2*d_inner + 2*n_groups*N + num_heads
        parts["in_proj"] = self.perf.linear(self.hidden_size, self.in_proj_out, seq_len, b, mode)

        # causal depthwise conv1d + SiLU over [x | B | C]
        parts["conv1d"] = self.perf.causal_conv1d(self.conv_dim, self.conv_kernel, seq_len, b, mode)

        # dt = clamp(softplus(dt + dt_bias))
        parts["dt"] = self.perf.dt_activation(self.num_heads, seq_len, b, mode)

        # the selective scan itself
        if mode == "prefill":
            parts["ssd"] = self.perf.ssd_chunk_scan(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                state_size=self.state_size,
                n_groups=self.n_groups,
                chunk_size=self.chunk_size,
                seq_len=seq_len,
                batch_size=b,
            )
        else:
            parts["ssd"] = self.perf.ssd_recurrence_decode(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                state_size=self.state_size,
                n_groups=self.n_groups,
                batch_size=b,
            )

        # gated RMSNorm on the d_inner-wide output
        parts["gated_norm"] = self.perf.gated_rms_norm(self.d_inner, seq_len, b, mode)

        # out_proj: d_inner -> hidden
        parts["out_proj"] = self.perf.linear(self.d_inner, self.hidden_size, seq_len, b, mode)

        # residual
        parts["residual"] = self.perf.residual(self.hidden_size, seq_len, b, mode)

        return parts

    # -------------------------------------------------------------------------
    # Phase-level timing
    # -------------------------------------------------------------------------

    def compute_prefill_time(self, verbose: bool = True) -> float:
        """Compute prefill phase execution time in seconds."""
        mode = "prefill"
        overall_exe_cycle = self.perf.embeddings(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)

        parts = self._block_cycles(self.input_seq_len, mode)
        block_cycles = sum(parts.values())
        overall_exe_cycle += block_cycles * self.num_hidden_layers

        # final norm + LM head
        overall_exe_cycle += self.perf.rms_layer(self.hidden_size, self.input_seq_len, self.device_batch_size, mode)
        overall_exe_cycle += self.perf.lm_head(self.hidden_size, self.vocab_size, self.device_batch_size)

        execution_time = overall_exe_cycle / self.frequency

        if verbose:
            print("\nPrefill Execution Distribution:")
            for name, cyc in parts.items():
                print(f"  {name:<12} {cyc / block_cycles * 100:6.2f}%")
            print(f"\n  Total cycles: {overall_exe_cycle:,}")

        return execution_time

    def compute_decode_time(self, output_token_size: int, verbose: bool = True) -> float:
        """Compute decode phase execution time in seconds.

        Unlike an attention model there is no kv_size to advance: every decode
        step costs exactly the same regardless of how much context precedes it.
        We therefore evaluate one step and multiply.
        """
        mode = "decode"
        # Every layer is evaluated explicitly (rather than one block x num_layers)
        # so that PerfModel's HBM traffic counters hold one full token's traffic
        # after this call — that is what makes the state-vs-KV comparison auditable.
        parts: dict[str, int] = {}
        for _ in range(self.num_hidden_layers):
            for name, cyc in self._block_cycles(1, mode).items():
                parts[name] = parts.get(name, 0) + cyc
        per_step = sum(parts.values())
        per_step += self.perf.rms_layer(self.hidden_size, 1, self.device_batch_size, mode)
        per_step += self.perf.lm_head(self.hidden_size, self.vocab_size, self.device_batch_size)

        overall_inst_num = per_step * output_token_size
        # Factor of 2 matches llama_model.py / gpt_oss_model.py: instruction issue +
        # memory access pipeline stages in decode mode. Kept identical so the two
        # architectures are compared on the same footing.
        overall_exe_cycle = overall_inst_num * 2
        execution_time = overall_exe_cycle / self.frequency

        if verbose:
            block_total = sum(parts.values())
            print("\nDecode Execution Distribution (mixer blocks):")
            for name, cyc in parts.items():
                print(f"  {name:<12} {cyc / block_total * 100:6.2f}%")

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
        description="Mamba-2 Performance Model - Compute TPS and TTFT for SSM inference on PLENA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mamba2_model.py --model mamba2-2.7b --model-lib ./Model_Lib --config ./plena_settings.toml --isa-lib ./customISA_lib.json
  python mamba2_model.py --model-path ./mamba2-2.7b.json --config ./plena_settings.toml --isa-lib ./customISA_lib.json
  python mamba2_model.py --list-models --model-lib ./Model_Lib
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

    args = parser.parse_args()

    if args.list_models:
        if not args.model_lib:
            parser.error("--model-lib is required for --list-models")
        print("Available models:")
        for model in list_available_models(Path(args.model_lib)):
            print(f"  {model}")
        return

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

    if not args.config:
        parser.error("--config is required for inference")
    if not args.isa_lib:
        parser.error("--isa-lib is required for inference")

    if args.model or args.task_file:
        if not args.model_lib:
            parser.error("--model-lib is required when using --model")
        model_path = str(resolve_model_path(args.model, Path(args.model_lib)))
    else:
        model_path = args.model_path

    hardware_config = load_hardware_config_from_toml(args.config)

    model = Mamba2Model(
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
            "architecture": "mamba2",
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
