"""
Qwen3 dense performance model for PLENA Simulator.

This entrypoint reuses the LLaMA-style dense decoder analytic model and adds
Qwen3-specific QK-Norm overhead. RoPE is not added here because PerfModel's
projection() already includes Q/K RoPE costs.
"""

import argparse
import json
import math
from pathlib import Path

from llama_model import LLaMAModel, list_available_models, resolve_model_path
from perf_model import load_hardware_config_from_toml


class Qwen3DenseModel(LLaMAModel):
    """Qwen3 dense analytic model with QK-Norm overhead."""

    latency_model_name = "qwen3_dense_analytic"
    base_model_name = "llama_style_dense"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_qwen3_overheads = {
            "qk_norm_cycles": 0,
            "qk_norm_seconds": 0.0,
            "qk_norm_ms": 0.0,
        }

    def _qk_norm_cycles(self, seq_len: int, batch_size: int, mode: str = "prefill") -> int:
        """Estimate per-layer QK-Norm cycles using per-head RMSNorm semantics."""
        head_groups = self.num_attention_heads + self.num_key_value_heads
        loop_num = max(1, math.ceil(self.head_dim / self.perf.vlen))
        setting_inst_num = 10
        loop_inst_num = 8

        if mode == "prefill":
            per_group_cycles = (
                setting_inst_num * self.perf.instr["S_BASIC"]
                + loop_num * loop_inst_num * seq_len * self.perf.instr["V_BASIC"] * batch_size
            )
            qk_norm_cycles = head_groups * per_group_cycles

            total_qk_elems = head_groups * self.head_dim * seq_len * batch_size
            spill_elems = max(0, total_qk_elems - self.perf.vector_sram_size)
            if spill_elems:
                chunk_elems = max(1, self.perf.vlen * self.perf.prefetch_v_amount)
                qk_norm_cycles += math.ceil(spill_elems / chunk_elems) * self.perf.instr["H_PREFETCH_V"] * 2
            return qk_norm_cycles

        per_group_cycles = (
            setting_inst_num * self.perf.instr["S_BASIC"]
            + loop_num * loop_inst_num * self.perf.instr["V_BASIC"] * batch_size
        )
        return head_groups * per_group_cycles

    def _record_qwen3_overheads(self, qk_norm_cycles: int) -> None:
        seconds = qk_norm_cycles / self.frequency
        self.last_qwen3_overheads = {
            "qk_norm_cycles": qk_norm_cycles,
            "qk_norm_seconds": seconds,
            "qk_norm_ms": seconds * 1000,
        }

    def compute_prefill_time(self, verbose: bool = True) -> float:
        """Compute Qwen3 dense prefill phase execution time in seconds."""
        mode = "prefill"
        kv_size = self.input_seq_len
        overall_exe_cycle = 0

        overall_exe_cycle += self.perf.embeddings(
            self.hidden_size, self.input_seq_len, self.device_batch_size, mode
        )

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
        qk_norm = self._qk_norm_cycles(self.input_seq_len, self.device_batch_size, mode)
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
        transformer_block_cycles = rms + proj + qk_norm + attn + res + rms + ffn
        total_qk_norm_cycles = qk_norm * self.num_hidden_layers
        overall_exe_cycle += transformer_block_cycles * self.num_hidden_layers
        self._record_qwen3_overheads(total_qk_norm_cycles)

        execution_time = overall_exe_cycle / self.frequency

        if verbose:
            print("\nQwen3 Prefill Execution Distribution:")
            print(f"  RMS Layer:       {rms / transformer_block_cycles * 100:.2f}%")
            print(f"  Projection:      {proj / transformer_block_cycles * 100:.2f}%")
            print(f"  QK-Norm:         {qk_norm / transformer_block_cycles * 100:.2f}%")
            print(f"  Flash Attention: {attn / transformer_block_cycles * 100:.2f}%")
            print(f"  Residual:        {res / transformer_block_cycles * 100:.2f}%")
            print(f"  Feed Forward:    {ffn / transformer_block_cycles * 100:.2f}%")

        return execution_time

    def compute_decode_time(self, output_token_size: int, verbose: bool = True) -> float:
        """Compute Qwen3 dense decode phase execution time in seconds."""
        mode = "decode"
        kv_size = self.input_seq_len

        rms_count = 0
        projection_count = 0
        qk_norm_count = 0
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
                qk_norm_count += self._qk_norm_cycles(1, self.device_batch_size, mode)
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

        overall_inst_num = (
            rms_count
            + projection_count
            + qk_norm_count
            + flash_attention_count
            + residual_count
            + feed_forward_count
        )
        overall_exe_cycle = overall_inst_num * 2
        self._record_qwen3_overheads(qk_norm_count * 2)
        execution_time = overall_exe_cycle / self.frequency

        if verbose:
            print("\nQwen3 Decode Execution Distribution:")
            print(f"  RMS Layer:       {rms_count / overall_inst_num * 100:.2f}%")
            print(f"  Projection:      {projection_count / overall_inst_num * 100:.2f}%")
            print(f"  QK-Norm:         {qk_norm_count / overall_inst_num * 100:.2f}%")
            print(f"  Flash Attention: {flash_attention_count / overall_inst_num * 100:.2f}%")
            print(f"  Residual:        {residual_count / overall_inst_num * 100:.2f}%")
            print(f"  Feed Forward:    {feed_forward_count / overall_inst_num * 100:.2f}%")

        return execution_time


def _base_result(args, model_path: str, phase: str) -> dict:
    return {
        "model": args.model or model_path,
        "phase": phase,
        "batch_size": args.batch_size,
        "input_seq_len": args.input_seq,
        "output_seq_len": args.output_seq,
        "device_num": args.device_num,
        "latency_model": Qwen3DenseModel.latency_model_name,
        "base_model": Qwen3DenseModel.base_model_name,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3 Dense Performance Model - LLaMA-style dense decoder plus QK-Norm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
    parser.add_argument(
        "--phase",
        choices=["ttft", "prefill"],
        default="ttft",
        help="Output full TTFT/TPS or prefill-only latency (default: ttft)",
    )

    args = parser.parse_args()

    if args.list_models:
        if not args.model_lib:
            parser.error("--model-lib is required for --list-models")
        model_lib_path = Path(args.model_lib)
        print("Available models:")
        for model in list_available_models(model_lib_path):
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
        model_lib_path = Path(args.model_lib)
        model_path = str(resolve_model_path(args.model, model_lib_path))
    else:
        model_path = args.model_path

    hardware_config = load_hardware_config_from_toml(args.config)
    model = Qwen3DenseModel(
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

    if args.phase == "prefill":
        prefill_time = model.compute_prefill_time(verbose=not args.quiet)
        if args.json:
            result = _base_result(args, model_path, "prefill")
            result.update(
                {
                    "prefill_seconds": prefill_time,
                    "prefill_ms": prefill_time * 1000,
                    "qwen3_overheads": model.last_qwen3_overheads,
                }
            )
            print(json.dumps(result, indent=2))
        else:
            print("\n" + "=" * 50)
            print("Qwen3 Prefill Performance Results")
            print("=" * 50)
            print(f"Prefill time: {prefill_time:.6f} s ({prefill_time * 1000:.3f} ms)")
            print(f"QK-Norm overhead: {model.last_qwen3_overheads['qk_norm_ms']:.3f} ms")
            print("=" * 50)
        return

    ttft, tps = model.compute_performance(verbose=not args.quiet)
    if args.json:
        result = _base_result(args, model_path, "ttft")
        result.update(
            {
                "ttft_seconds": ttft,
                "ttft_ms": ttft * 1000,
                "tps": tps,
                "qwen3_overheads": model.last_qwen3_overheads,
            }
        )
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 50)
        print("Qwen3 Performance Results")
        print("=" * 50)
        print(f"TTFT (Time to First Token): {ttft:.6f} s ({ttft * 1000:.3f} ms)")
        print(f"TPS (Tokens Per Second):    {tps:.2f}")
        print(f"Latest QK-Norm overhead:    {model.last_qwen3_overheads['qk_norm_ms']:.3f} ms")
        print("=" * 50)


if __name__ == "__main__":
    main()
