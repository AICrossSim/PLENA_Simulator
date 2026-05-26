"""Unified model runner — compile and/or emulate any model config.

Usage:
    python run_model.py <nickname> [--config <preset>] [--compile-only] [--case <case>] [...]

Examples:
    python run_model.py smollm2 --config sliced_64x64x16_b1
    python run_model.py llada-8b --config native_256x256x64_b1 --compile-only
    python run_model.py smolvlm2 --case vision-layers --layers 5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (
    REPO_ROOT,
    REPO_ROOT / "PLENA_Compiler",
    REPO_ROOT / "PLENA_Tools",
    REPO_ROOT / "transactional_emulator" / "testbench",
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from transactional_emulator.testbench.model_configs.loader import (  # noqa: E402
    HardwarePreset,
    ModelConfig,
    load_model_config_by_nickname,
)
from transactional_emulator.testbench.aten.configurable import HardwareConfig  # noqa: E402


SLICED_CASES = ("ffn", "decoder-layer", "decoder-chain")
NATIVE_CASES = ("decoder", "vision-layers", "vision-connector", "vlm-e2e")


def _build_dir(nickname: str, config_name: str, case: str) -> Path:
    base = Path(__file__).parent / "build" / f"{nickname}_{config_name}_{case}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_toml(preset: HardwarePreset, build_dir: Path) -> Path:
    hw = HardwareConfig(
        mlen=preset.mlen,
        vlen=preset.vlen,
        blen=preset.blen,
        hlen=preset.hlen,
        broadcast_amount=preset.broadcast,
        dc_en=None,
        latency_profile=None,
        hbm_m_prefetch_amount=None,
        hbm_v_prefetch_amount=None,
        hbm_v_writeback_amount=None,
    )
    toml_path = hw.write_toml(build_dir)
    os.environ["PLENA_SETTINGS_TOML"] = str(toml_path)
    return toml_path


def compile_sliced(mc: ModelConfig, preset: HardwarePreset, args) -> tuple[dict, Path]:
    from transactional_emulator.testbench.sliced_layer_test_builder import (
        compile_sliced_ffn,
        compile_sliced_decoder_layer,
        compile_sliced_decoder_chain,
    )

    hidden = args.hidden_size or 64
    inter = args.inter_dim or 128
    seq_len = args.seq_len or preset.mlen
    layers = args.layers or 1
    batch_size = args.batch_size or preset.batch_size
    build_dir = _build_dir(mc.nickname, args.config or "default", args.case or "decoder-chain")

    _write_toml(preset, build_dir)

    case = args.case or ("decoder-layer" if layers == 1 else "decoder-chain")
    common = dict(
        model_id=mc.model_id,
        build_dir=build_dir,
        mlen=preset.mlen,
        vlen=preset.vlen,
        blen=preset.blen,
        seed=args.seed,
    )

    if case == "ffn":
        result = compile_sliced_ffn(batch_size=batch_size, **common)
    elif case == "decoder-layer":
        result = compile_sliced_decoder_layer(
            seq_len=seq_len,
            hidden_size=hidden,
            inter_dim=inter,
            batch_size=batch_size,
            trust_remote_code=mc.trust_remote_code,
            partial_load=args.partial_load,
            **common,
        )
    elif case == "decoder-chain":
        result = compile_sliced_decoder_chain(
            num_layers=layers,
            seq_len=seq_len,
            hidden_size=hidden,
            inter_dim=inter,
            batch_size=batch_size,
            trust_remote_code=mc.trust_remote_code,
            partial_load=args.partial_load,
            **common,
        )
    else:
        raise ValueError(f"Unknown sliced case: {case}")

    return result, build_dir


def compile_native(mc: ModelConfig, preset: HardwarePreset, args) -> tuple[dict, Path]:
    import torch
    from compiler.aten.plena_frontend import compile_native_hf_decoder, compile_native_hf_vision_encoder
    from transformers import AutoModel

    case = args.case or "decoder"
    layers = args.layers
    seq_len = args.seq_len or 64
    batch_size = args.batch_size or preset.batch_size
    build_dir = _build_dir(mc.nickname, args.config or "default", case)

    _write_toml(preset, build_dir)

    print(f"Loading model {mc.model_id}...")
    model = AutoModel.from_pretrained(
        mc.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=mc.trust_remote_code,
    )

    common = dict(
        model=model,
        seq_len=seq_len,
        batch_size=batch_size,
        mlen=preset.mlen,
        blen=preset.blen,
        mram_tile_capacity=preset.mram_tile_capacity,
        seed=args.seed,
    )

    if case == "decoder":
        result = compile_native_hf_decoder(
            hlen=preset.hlen,
            broadcast_amount=preset.broadcast,
            num_layers=layers,
            **common,
        )
    elif case == "vision-layers":
        result = compile_native_hf_vision_encoder(
            num_layers=layers,
            include_connector=False,
            **common,
        )
    elif case == "vision-connector":
        result = compile_native_hf_vision_encoder(
            num_layers=layers,
            include_connector=True,
            **common,
        )
    elif case == "vlm-e2e":
        vision_layers = args.vision_layers or 1
        text_layers = args.text_layers or 1
        vision_result = compile_native_hf_vision_encoder(
            num_layers=vision_layers,
            include_connector=True,
            **common,
        )
        decoder_input = vision_result.get("padded_golden_output", vision_result.get("golden_output"))
        result = compile_native_hf_decoder(
            hlen=preset.hlen,
            broadcast_amount=preset.broadcast,
            num_layers=text_layers,
            decoder_input_embeds=decoder_input,
            **common,
        )
    else:
        raise ValueError(f"Unknown native case: {case}")

    return result, build_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("nickname", help="Model nickname (e.g. smollm2, llada-8b, smolvlm2)")
    parser.add_argument("--config", default=None, help="Hardware preset name (e.g. sliced_64x64x16_b1)")
    parser.add_argument("--compile-only", action="store_true", help="Generate ASM only, skip emulator")
    parser.add_argument(
        "--case",
        default=None,
        help="Test case (sliced: ffn/decoder-layer/decoder-chain; native: decoder/vision-layers/vision-connector/vlm-e2e)",
    )
    parser.add_argument("--layers", type=int, default=None, help="Number of decoder layers")
    parser.add_argument("--vision-layers", type=int, default=None, help="Vision encoder layers (vlm-e2e)")
    parser.add_argument("--text-layers", type=int, default=None, help="Text decoder layers (vlm-e2e)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length")
    parser.add_argument("--hidden-size", type=int, default=None, help="Sliced hidden dim (sliced mode only)")
    parser.add_argument("--inter-dim", type=int, default=None, help="Sliced FFN intermediate dim (sliced mode only)")
    parser.add_argument(
        "--partial-load", action="store_true", help="Use safetensors partial download (for large models)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mc = load_model_config_by_nickname(args.nickname)

    if args.config:
        preset = mc.get_preset(args.config)
    else:
        preset = mc.hardware

    mode = preset.mode
    print(f"Model: {mc.nickname} ({mc.model_id})")
    print(f"Config: {args.config or 'default'} (mode={mode})")
    print(
        f"Hardware: mlen={preset.mlen} vlen={preset.vlen} blen={preset.blen} hlen={preset.hlen} broadcast={preset.broadcast}"
    )

    if mode == "sliced":
        result, build_dir = compile_sliced(mc, preset, args)
    elif mode == "native":
        result, build_dir = compile_native(mc, preset, args)
    else:
        raise ValueError(f"Unknown mode '{mode}' in preset")

    asm_path = build_dir / "generated_asm_code.asm"
    asm_path.write_text(result["isa"])
    print(f"\nASM written to: {asm_path}")
    print(f"ISA lines: {len(result['isa'].splitlines())}")

    if args.compile_only:
        print("\n[compile-only] Skipping emulator run.")
        return

    from transactional_emulator.testbench.emulator_runner import emulate_from_result

    asm_name = f"{mc.nickname}_{args.case or 'decoder'}"
    emulate_from_result(result, build_dir, asm_name, mlen=preset.mlen, blen=preset.blen, vlen=preset.vlen)


if __name__ == "__main__":
    main()
