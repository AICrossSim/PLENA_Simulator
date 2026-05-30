"""SigLIP vision performance model for PLENA Simulator.

Uses PerfModel for instruction-level latency primitives and estimates end-to-end
vision encoder latency for image embedding workloads.
"""

import argparse
import json
import math
from pathlib import Path

from perf_model import PerfModel, load_hardware_config_from_toml


class SigLIPVisionModel:
    """SigLIP vision encoder performance model."""

    def __init__(
        self,
        model_config_path: str,
        hardware_config: dict,
        custom_isa_path: str,
        batch_size: int = 1,
        device_num: int = 1,
        frequency_hz: float = 1e9,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if device_num <= 0:
            raise ValueError(f"device_num must be > 0, got {device_num}")
        if batch_size % device_num != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by device_num ({device_num}) "
                "for per-device analytic estimates"
            )

        with open(model_config_path) as f:
            model_param = json.load(f)

        vision_cfg = model_param.get("vision_config", model_param)

        self.hidden_size = int(vision_cfg["hidden_size"])
        self.intermediate_size = int(vision_cfg["intermediate_size"])
        self.num_attention_heads = int(vision_cfg["num_attention_heads"])
        self.num_hidden_layers = int(vision_cfg["num_hidden_layers"])
        self.image_size = int(vision_cfg["image_size"])
        self.patch_size = int(vision_cfg["patch_size"])
        self.num_channels = int(vision_cfg.get("num_channels", 3))

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.patch_dim = self.num_channels * self.patch_size * self.patch_size
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.frequency = frequency_hz
        self.batch_size = batch_size
        self.device_num = device_num
        self.device_batch_size = batch_size // device_num

        self.perf = PerfModel(hardware_config, custom_isa_path)

    def print_config(self):
        """Print model and hardware configuration."""
        print("=" * 60)
        print("SigLIP Vision Configuration")
        print("=" * 60)
        print(f"Hidden size:          {self.hidden_size}")
        print(f"Intermediate size:    {self.intermediate_size}")
        print(f"Num attention heads:  {self.num_attention_heads}")
        print(f"Num hidden layers:    {self.num_hidden_layers}")
        print(f"Image size:           {self.image_size}")
        print(f"Patch size:           {self.patch_size}")
        print(f"Num patches:          {self.num_patches}")
        print(f"Patch dim:            {self.patch_dim}")
        print(f"Batch size:           {self.batch_size}")
        print(f"Device num:           {self.device_num}")
        print("-" * 60)
        print("Hardware Config")
        print("-" * 60)
        print(f"MLEN: {self.perf.mlen}, BLEN: {self.perf.blen}, VLEN: {self.perf.vlen}")
        print("=" * 60)

    def _linear_cycles(self, in_features: int, out_features: int, token_count: int) -> int:
        """Cycle estimate for one linear projection over token rows."""
        return (
            math.ceil(token_count / self.perf.blen)
            * math.ceil(in_features / self.perf.mlen)
            * math.ceil(out_features / self.perf.blen)
            * self.perf.instr["M_MM"]
        )

    def _patch_embedding_cycles(self) -> int:
        """Patch projection + position add cycle estimate."""
        token_count = self.num_patches * self.device_batch_size

        patch_proj = self._linear_cycles(self.patch_dim, self.hidden_size, token_count)

        # Position embedding add is the same elementwise add pattern as residual.
        pos_add = self.perf.residual(
            hidden_size=self.hidden_size,
            seq_len=self.num_patches,
            batch_size=self.device_batch_size,
            mode="prefill",
        )

        return patch_proj + pos_add

    def _layer_norm_cycles(self, seq_len: int, batch_size: int, use_affine: bool = False) -> int:
        """LayerNorm estimate aligned to layer_norm_asm instruction structure.

        The estimate mirrors two vector loops (stats pass + normalize pass),
        scalar mean/variance epilogue, and optional affine gamma/beta pass.
        """
        row_count = seq_len * batch_size
        hidden_tiles = math.ceil(self.hidden_size / self.perf.vlen)

        # Constant setup outside the per-row loop:
        # S_LD_FP (epsilon), S_ADD_FP x2 (zero accumulators), S_LD_FP (1/hidden)
        total = 2 * self.perf.instr["S_LD_FP"] + 2 * self.perf.instr["S_ADD_FP"]

        # First vector loop per row:
        # V_RED_SUM, V_MUL_VV, V_RED_SUM, S_ADDI_INT
        first_pass_per_tile = (
            self.perf.instr["V_RED_SUM"]
            + self.perf.instr["V_MUL_VV"]
            + self.perf.instr["V_RED_SUM"]
            + self.perf.instr["S_ADDI_INT"]
        )

        # Scalar epilogue per row:
        # mean/var/std/inv-std + reset accumulators for next row
        scalar_per_row = (
            3 * self.perf.instr["S_MUL_FP"]
            + self.perf.instr["S_SUB_FP"]
            + self.perf.instr["S_ADD_FP"]
            + self.perf.instr["S_SQRT_FP"]
            + self.perf.instr["S_RECI_FP"]
            + 2 * self.perf.instr["S_ADD_FP"]
        )

        # Second vector loop per row:
        # V_SUB_VF, V_MUL_VF, S_ADDI_INT
        second_pass_per_tile = (
            self.perf.instr["V_SUB_VF"]
            + self.perf.instr["V_MUL_VF"]
            + self.perf.instr["S_ADDI_INT"]
        )

        total += row_count * (hidden_tiles * first_pass_per_tile + scalar_per_row + hidden_tiles * second_pass_per_tile)

        if use_affine:
            # Optional affine pass in layer_norm_asm: y = y * gamma + beta
            affine_per_tile = (
                self.perf.instr["V_MUL_VV"]
                + self.perf.instr["V_ADD_VV"]
                + 3 * self.perf.instr["S_ADDI_INT"]
            )
            total += row_count * hidden_tiles * affine_per_tile

        # Keep the same spill-aware prefetch heuristic.
        working_set = self.hidden_size * row_count
        if working_set > self.perf.vector_sram_size:
            spill = working_set - self.perf.vector_sram_size
            total += math.ceil(spill / (self.perf.vlen * self.perf.prefetch_v_amount)) * self.perf.instr["H_PREFETCH_V"] * 2

        return total

    def _gelu_mlp_cycles(self, seq_len: int, batch_size: int) -> int:
        """SigLIP MLP estimate (GELU path).

        This follows the gelu_asm sigmoid approximation:
          x * sigmoid(1.702 * x)
        with explicit vector op accounting for the GELU middle stage.
        """
        token_count = seq_len * batch_size
        up_proj = (
            math.ceil(token_count / self.perf.blen)
            * math.ceil(self.hidden_size / self.perf.mlen)
            * math.ceil(self.intermediate_size / self.perf.blen)
            * self.perf.instr["M_MM"]
        )

        # Per-vector GELU op mix from gelu_asm:
        # V_MUL_VF, V_SUB_VF, V_EXP_V, V_ADD_VF, V_RECI_V, V_MUL_VV, S_ADDI_INT.
        gelu_vectors = token_count * math.ceil(self.intermediate_size / self.perf.vlen)
        gelu = gelu_vectors * (
            self.perf.instr["V_MUL_VF"]
            + self.perf.instr["V_SUB_VF"]
            + self.perf.instr["V_EXP_V"]
            + self.perf.instr["V_ADD_VF"]
            + self.perf.instr["V_RECI_V"]
            + self.perf.instr["V_MUL_VV"]
            + self.perf.instr["S_ADDI_INT"]
        )

        # Constant setup in gelu_asm: load 1.0 and 1.702 once.
        gelu += 2 * self.perf.instr["S_LD_FP"]

        down = (
            math.ceil(token_count / self.perf.blen)
            * math.ceil(self.intermediate_size / self.perf.mlen)
            * math.ceil(self.hidden_size / self.perf.blen)
            * self.perf.instr["M_MM"]
        )

        # Spill-aware prefetch heuristic for intermediate activations.
        working_set = self.intermediate_size * token_count
        prefetch = 0
        if working_set > self.perf.vector_sram_size:
            spill = working_set - self.perf.vector_sram_size
            prefetch = math.ceil(spill / (self.perf.vlen * self.perf.prefetch_v_amount)) * self.perf.instr["H_PREFETCH_V"] * 2

        return up_proj + gelu + down + prefetch

    def _encoder_layer_breakdown(self) -> dict:
        """Per-layer cycle breakdown for SigLIP encoder."""
        seq_len = self.num_patches
        batch_size = self.device_batch_size

        ln1 = self._layer_norm_cycles(seq_len, batch_size)

        # SigLIP attention has q/k/v/out projections but no RoPE or KV-cache writes.
        token_count = seq_len * batch_size
        q_proj = self._linear_cycles(self.hidden_size, self.hidden_size, token_count)
        k_proj = self._linear_cycles(self.hidden_size, self.hidden_size, token_count)
        v_proj = self._linear_cycles(self.hidden_size, self.hidden_size, token_count)
        proj = q_proj + k_proj + v_proj

        attn = self.perf.flash_attention(
            num_attention_heads=self.num_attention_heads,
            num_kv_heads=self.num_attention_heads,
            head_dim=self.head_dim,
            seq_len=seq_len,
            kv_size=seq_len,
            batch_size=batch_size,
            mode="prefill",
        )

        out_proj = self._linear_cycles(self.hidden_size, self.hidden_size, token_count)
        res1 = self.perf.residual(self.hidden_size, seq_len, batch_size, mode="prefill")

        ln2 = self._layer_norm_cycles(seq_len, batch_size)
        mlp = self._gelu_mlp_cycles(seq_len, batch_size)
        res2 = self.perf.residual(self.hidden_size, seq_len, batch_size, mode="prefill")

        total = ln1 + proj + attn + out_proj + res1 + ln2 + mlp + res2
        return {
            "ln1": ln1,
            "qkv_proj": proj,
            "flash_attention": attn,
            "out_proj": out_proj,
            "residual1": res1,
            "ln2": ln2,
            "mlp": mlp,
            "residual2": res2,
            "total": total,
        }

    def compute_latency(self, verbose: bool = True) -> tuple[float, dict]:
        """Compute end-to-end vision encoder latency in seconds."""
        patch_embedding = self._patch_embedding_cycles()
        layer_breakdown = self._encoder_layer_breakdown()
        final_ln = self._layer_norm_cycles(self.num_patches, self.device_batch_size)

        total_cycles = patch_embedding + layer_breakdown["total"] * self.num_hidden_layers + final_ln
        latency_s = total_cycles / self.frequency / self.device_num

        breakdown = {
            "patch_embedding": patch_embedding,
            "encoder_per_layer": layer_breakdown,
            "encoder_total": layer_breakdown["total"] * self.num_hidden_layers,
            "final_post_layernorm": final_ln,
            "total_cycles": total_cycles,
            "latency_s": latency_s,
            "images_per_second": self.batch_size / latency_s,
        }

        if verbose:
            encoder_total = breakdown["encoder_total"]
            print("\nExecution Distribution:")
            print(f"  Patch embedding:    {patch_embedding / total_cycles * 100:.2f}%")
            print(f"  Encoder stack:      {encoder_total / total_cycles * 100:.2f}%")
            print(f"  Final post-LN:      {final_ln / total_cycles * 100:.2f}%")
            print("\nPer-layer Encoder Distribution:")
            for key in ["ln1", "qkv_proj", "flash_attention", "out_proj", "residual1", "ln2", "mlp", "residual2"]:
                value = layer_breakdown[key]
                print(f"  {key:16s} {value / layer_breakdown['total'] * 100:6.2f}%")

        return latency_s, breakdown


# =============================================================================
# Model Library Utilities
# =============================================================================


def list_available_models(model_lib_path: Path) -> list[str]:
    """List available SigLIP model configs in Model_Lib."""
    if not model_lib_path.exists():
        return []
    return sorted([f.stem for f in model_lib_path.glob("*siglip*.json")])


def resolve_model_path(model_name: str, model_lib_path: Path) -> Path:
    """Resolve model name to full JSON path."""
    model_path = model_lib_path / f"{model_name}.json"
    if not model_path.exists():
        available = list_available_models(model_lib_path)
        raise FileNotFoundError(f"Model '{model_name}' not found. Available: {', '.join(available)}")
    return model_path


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="SigLIP Vision Performance Model - Estimate vision encoder latency on PLENA",
    )

    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--model", "-m", help="Model name from Model_Lib")
    model_group.add_argument("--model-path", help="Full path to model config JSON")
    model_group.add_argument("--list-models", "-l", action="store_true", help="List available SigLIP models")

    parser.add_argument("--model-lib", help="Path to Model_Lib directory")
    parser.add_argument("--config", "-c", help="Path to hardware config TOML")
    parser.add_argument("--isa-lib", help="Path to customISA_lib.json")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--device-num", "-d", type=int, default=1, help="Number of devices (default: 1)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress detailed output")

    args = parser.parse_args()

    if args.list_models:
        if not args.model_lib:
            parser.error("--model-lib is required for --list-models")
        model_lib_path = Path(args.model_lib)
        print("Available SigLIP models:")
        for model in list_available_models(model_lib_path):
            print(f"  {model}")
        return

    if not args.config:
        parser.error("--config is required for inference")
    if not args.isa_lib:
        parser.error("--isa-lib is required for inference")

    if args.model:
        if not args.model_lib:
            parser.error("--model-lib is required when using --model")
        model_path = resolve_model_path(args.model, Path(args.model_lib))
    else:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model config not found: {model_path}")

    hardware_config = load_hardware_config_from_toml(args.config)

    model = SigLIPVisionModel(
        model_config_path=str(model_path),
        hardware_config=hardware_config,
        custom_isa_path=args.isa_lib,
        batch_size=args.batch_size,
        device_num=args.device_num,
    )

    if not args.quiet:
        model.print_config()

    latency_s, breakdown = model.compute_latency(verbose=not args.quiet)

    if args.json:
        result = {
            "model": model_path.stem,
            "batch_size": args.batch_size,
            "latency_s": latency_s,
            "images_per_second": breakdown["images_per_second"],
            "cycles": breakdown,
        }
        print(json.dumps(result, indent=2))
    else:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Model:                {model_path.stem}")
        print(f"Batch size:           {args.batch_size}")
        print(f"Latency:              {latency_s * 1000:.2f} ms")
        print(f"Images/sec:           {breakdown['images_per_second']:.2f}")


if __name__ == "__main__":
    main()
