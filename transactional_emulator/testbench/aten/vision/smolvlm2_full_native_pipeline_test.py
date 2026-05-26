"""Native SmolVLM2 module and boundary isolation tests.

This harness runs compiler-generated ISA through the Rust transactional
emulator at explicit VLM boundaries:

    vision encoder stages -> connector stages -> text decoder layers

It is intentionally stage-oriented.  Use it to find the first bad boundary
before attempting full vision-to-text end-to-end runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPILER_ROOT = REPO_ROOT / "PLENA_Compiler"
TOOLS_ROOT = REPO_ROOT / "tools"
for path in (REPO_ROOT, COMPILER_ROOT, TOOLS_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from compiler.aten.model_extract import (  # noqa: E402
    embedding_module,
    extract_model_config,
    extract_vision_config,
    extract_vision_connector_weights,
    find_model_root,
)
from compiler.aten.plena import PlenaCompiler  # noqa: E402
from compiler.aten.plena_frontend import (  # noqa: E402
    REAL_DATA_RATIO,
    _ceil_to_multiple,
    _fix_large_immediates,
    _pad_batched_sequence_storage,
    _pad_optional_vector,
    _pad_vision_connector_weight_for_tiles,
    _register_vision_connector_inputs,
    _run_vision_connector_reference,
    _emit_vision_connector,
    _lazy_repeat_feature_vector_storage,
    _tensor_layout_metadata,
    compile_native_hf_decoder,
    compile_native_hf_vision_encoder,
)
from compiler.aten.reference import ReferencePrecision  # noqa: E402
from compiler.aten.ops.registry import Backend, OpRegistry  # noqa: E402
from transactional_emulator.testbench.aten.configurable import (  # noqa: E402
    AtenTemplateTestbench,
    HardwareConfig,
)
from transactional_emulator.testbench.emulator_runner import (  # noqa: E402
    compare_emulator_output,
    run_and_assert,
)
from transactional_emulator.testbench.sim_env_utils import (  # noqa: E402
    create_mem_for_sim,
    materialize_tensor_spec,
)
from transactional_emulator.tools.create_sim_env import create_sim_env  # noqa: E402


MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=[
            "patch-im2col",
            "patch",
            "patch-bias",
            "zero-layer-encoder",
            "vision-layer-ln1",
            "vision-layer-attn-o-full",
            "vision-layer-attn-proj",
            "vision-layer-attn",
            "vision-layer-ln2",
            "vision-layer-fc1",
            "vision-layer-gelu",
            "vision-layer-fc2",
            "vision-layer-mlp",
            "vision-layers",
            "vision-connector-shuffle",
            "vision-connector",
            "connector-only",
            "text-decoder",
            "vlm-e2e",
            "all-smoke",
        ],
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument(
        "--model-key",
        default=None,
        help="Known model key (llada_8b, smolvlm2_256m, smollm2_135m). "
        "Loads hlen/broadcast/trust_remote_code from YAML config.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--vision-layers", type=int, default=None)
    parser.add_argument("--text-layers", type=int, default=None)
    parser.add_argument("--layer-idx-start", type=int, default=0)
    parser.add_argument("--stage-layer", type=int, default=0)
    parser.add_argument("--text-seq-len", type=int, default=64)
    parser.add_argument("--mram-tile-capacity", type=int, default=4)
    parser.add_argument("--golden-precision", default="hardware")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--dump-input-tensors",
        action="store_true",
        help="Write every input tensor as a .pt sidecar. Disabled by default to keep large model builds small.",
    )
    AtenTemplateTestbench.add_common_args(
        parser,
        default_build_dir=Path(__file__).parent / "build" / "smolvlm2_native_module",
    )
    parser.set_defaults(min_allclose_rate=99.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.case == "all-smoke":
        for case in ("patch-im2col", "patch", "patch-bias", "zero-layer-encoder", "connector-only"):
            child_args = argparse.Namespace(**vars(args))
            child_args.case = case
            child_args.build_dir = args.build_dir / case
            print(f"\n{'=' * 80}\nSmolVLM2 native smoke case: {case}\n{'=' * 80}")
            run_case(child_args)
        return

    run_case(args)


def run_case(args: argparse.Namespace) -> None:
    build_dir = args.build_dir.resolve()

    # Load model YAML config if --model-key is set
    if args.model_key:
        from transactional_emulator.testbench.model_configs.loader import (
            load_model_config,
        )

        # Only override model_id if user explicitly set --model-id, otherwise use YAML default
        model_id_override = args.model_id if args.model_id != MODEL_ID else None
        mc = load_model_config(args.model_key, model_id_override=model_id_override)
        if not model_id_override:
            args.model_id = mc.model_id
        if not args.trust_remote_code:
            args.trust_remote_code = mc.trust_remote_code
        # Override hlen/broadcast from config if not explicitly set on CLI

        # Auto-apply hardware from model config
        print(
            f"  [model-key] {args.model_key}: hlen={mc.hardware.hlen}, broadcast={mc.hardware.broadcast}, "
            f"trust_remote_code={mc.trust_remote_code}"
        )
        hlen_override = mc.hardware.hlen
        broadcast_override = mc.hardware.broadcast
    else:
        hlen_override = None
        broadcast_override = None

    hw = _configure_hw(args, build_dir, default_hlen=hlen_override, default_broadcast_amount=broadcast_override)
    model = _load_model(args.model_id, local_files_only=args.local_files_only, trust_remote_code=args.trust_remote_code)

    # Validate hardware against actual model architecture
    if args.model_key:
        from transactional_emulator.testbench.model_configs.loader import ModelArchConfig

        actual_arch = ModelArchConfig.from_hf_config(model.config)
        print(
            f"  [arch] hidden={actual_arch.hidden_size}, inter={actual_arch.inter_dim}, "
            f"heads={actual_arch.num_heads}/{actual_arch.num_kv_heads}, "
            f"head_dim={actual_arch.head_dim}, GQA={actual_arch.gqa_ratio}"
        )
        if hw.hlen < actual_arch.head_dim:
            raise ValueError(f"hlen={hw.hlen} < head_dim={actual_arch.head_dim}")
        if hw.broadcast_amount < actual_arch.gqa_ratio:
            raise ValueError(f"broadcast={hw.broadcast_amount} < GQA={actual_arch.gqa_ratio}")
        if hw.hlen * hw.broadcast_amount != hw.mlen:
            raise ValueError(f"hlen*broadcast={hw.hlen * hw.broadcast_amount} != MLEN={hw.mlen}")

    if args.case == "connector-only":
        result = build_connector_only_case(model, args, hw)
        _prepare_and_maybe_run(build_dir, result, "smolvlm2_connector_only", args, hw)
        return

    if args.case == "vlm-e2e":
        run_vlm_e2e_case(model, args, hw, build_dir)
        return

    if args.case == "text-decoder":
        layers = args.layers if args.layers is not None else 1
        result = compile_native_hf_decoder(
            model,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_layers=layers,
            layer_idx_start=args.layer_idx_start,
            mlen=hw.mlen,
            blen=hw.blen,
            hlen=hw.hlen,
            broadcast_amount=hw.broadcast_amount,
            mram_tile_capacity=args.mram_tile_capacity,
            seed=args.seed,
            golden_precision=args.golden_precision,
            component="decoder",
        )
        _prepare_and_maybe_run(build_dir, result, "smolvlm2_text_decoder", args, hw)
        return

    stop_after, include_connector, layers = _vision_case_config(args)
    result = compile_native_hf_vision_encoder(
        model,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_layers=layers,
        layer_idx_start=args.layer_idx_start,
        mlen=hw.mlen,
        blen=hw.blen,
        mram_tile_capacity=args.mram_tile_capacity,
        seed=args.seed,
        golden_precision=args.golden_precision,
        include_connector=include_connector,
        stop_after=stop_after,
    )
    _prepare_and_maybe_run(build_dir, result, "smolvlm2_native_vision", args, hw)


def _vision_case_config(args: argparse.Namespace) -> tuple[str, bool, int]:
    if args.case == "patch":
        return "patch", False, 0
    if args.case == "patch-im2col":
        return "patch_im2col", False, 0
    if args.case == "patch-bias":
        return "patch_bias", False, 0
    if args.case == "zero-layer-encoder":
        return "post_ln", False, 0
    if args.case == "vision-layer-attn":
        layer = int(args.stage_layer)
        layers = args.layers if args.layers is not None else layer + 1
        return f"layer{layer}_attn_residual", False, layers
    if args.case == "vision-layer-ln1":
        layer = int(args.stage_layer)
        layers = args.layers if args.layers is not None else layer + 1
        return f"layer{layer}_ln1", False, layers
    if args.case == "vision-layer-attn-o-full":
        layer = int(args.stage_layer)
        layers = args.layers if args.layers is not None else layer + 1
        return f"layer{layer}_attn_o_full", False, layers
    if args.case == "vision-layer-attn-proj":
        layer = int(args.stage_layer)
        layers = args.layers if args.layers is not None else layer + 1
        return f"layer{layer}_attn_proj", False, layers
    if args.case == "vision-layer-ln2":
        layer = int(args.stage_layer)
        layers = args.layers if args.layers is not None else layer + 1
        return f"layer{layer}_ln2", False, layers
    if args.case == "vision-layer-fc1":
        layer = int(args.stage_layer)
        layers = args.layers if args.layers is not None else layer + 1
        return f"layer{layer}_fc1", False, layers
    if args.case == "vision-layer-gelu":
        layer = int(args.stage_layer)
        layers = args.layers if args.layers is not None else layer + 1
        return f"layer{layer}_gelu", False, layers
    if args.case == "vision-layer-fc2":
        layer = int(args.stage_layer)
        layers = args.layers if args.layers is not None else layer + 1
        return f"layer{layer}_fc2", False, layers
    if args.case == "vision-layer-mlp":
        layer = int(args.stage_layer)
        layers = args.layers if args.layers is not None else layer + 1
        return f"layer{layer}_mlp_residual", False, layers
    if args.case == "vision-layers":
        return "post_ln", False, args.layers if args.layers is not None else 1
    if args.case == "vision-connector-shuffle":
        return "connector_shuffle", True, args.layers if args.layers is not None else 0
    if args.case == "vision-connector":
        return "connector", True, args.layers if args.layers is not None else 0
    raise ValueError(f"Unhandled vision case {args.case!r}")


def build_connector_only_case(model, args: argparse.Namespace, hw: HardwareConfig) -> dict:
    if args.batch_size != 1:
        raise NotImplementedError("connector-only currently supports batch_size=1")

    model_cfg = extract_vision_config(model)
    connector_weights = extract_vision_connector_weights(model, model_cfg)
    if connector_weights is None:
        raise ValueError("Model does not expose a vision connector")

    grid = int(args.seq_len**0.5)
    if grid * grid != args.seq_len:
        raise ValueError(f"connector-only seq_len must be square, got {args.seq_len}")
    if grid % connector_weights.scale_factor != 0:
        raise ValueError(f"connector scale_factor={connector_weights.scale_factor} must divide grid={grid}")

    hidden = model_cfg.hidden_size
    padded_hidden = _ceil_to_multiple(hidden, hw.mlen)
    rows_per_batch = max(hw.mlen, _ceil_to_multiple(args.seq_len, hw.blen))
    connector_seq_len = args.seq_len // (connector_weights.scale_factor**2)
    connector_rows = max(hw.mlen, _ceil_to_multiple(connector_seq_len, hw.blen))
    connector_storage_dim = padded_hidden * (connector_weights.scale_factor**2)
    padded_output_dim = _ceil_to_multiple(connector_weights.output_dim, hw.mlen)

    torch.manual_seed(args.seed)
    encoder_out = torch.randn(args.seq_len, hidden)
    encoder_storage = _pad_batched_sequence_storage(
        encoder_out,
        batch_size=1,
        seq_len=args.seq_len,
        rows_per_batch=rows_per_batch,
        cols=padded_hidden,
    )

    precision = ReferencePrecision.from_mode(args.golden_precision)
    golden_out = _run_vision_connector_reference(
        encoder_out,
        connector_weights,
        precision=precision,
    )
    padded_golden_output = _pad_batched_sequence_storage(
        golden_out,
        batch_size=1,
        seq_len=connector_seq_len,
        rows_per_batch=connector_rows,
        cols=padded_output_dim,
    )

    registry = OpRegistry.load()
    registry.set_backend(Backend.PLENA)
    prog = PlenaCompiler(
        mlen=hw.mlen,
        blen=hw.blen,
        real_data_ratio=REAL_DATA_RATIO,
        mram_tile_capacity=args.mram_tile_capacity,
    )
    encoder_input = prog.input(
        "V_ENCODER_OUT",
        shape=(args.seq_len, hidden),
        physical_shape=(rows_per_batch, padded_hidden),
    )
    connector_inputs = _register_vision_connector_inputs(
        prog,
        connector_weights,
        storage_input_dim=connector_storage_dim,
        padded_output_dim=padded_output_dim,
        connector_rows=connector_rows,
        has_bias=connector_weights.bias is not None,
    )
    current = prog.load_batch(encoder_input, name="V_ENCODER_OUT")
    current = _emit_vision_connector(
        prog,
        current,
        connector_inputs,
        seq_len=args.seq_len,
        hidden=hidden,
        padded_hidden=padded_hidden,
        scale_factor=connector_weights.scale_factor,
        connector_rows=connector_rows,
        connector_storage_dim=connector_storage_dim,
        padded_output_dim=padded_output_dim,
    )
    isa = _fix_large_immediates(prog.compile())

    compile_connector_weight = _pad_vision_connector_weight_for_tiles(
        connector_weights,
        hidden=hidden,
        padded_hidden=padded_hidden,
        padded_output_dim=padded_output_dim,
    )
    input_tensors = {
        "V_ENCODER_OUT": encoder_storage,
        "V_CONNECTOR_W": compile_connector_weight,
    }
    data_order = ["V_ENCODER_OUT", "V_CONNECTOR_W"]
    if connector_weights.bias is not None:
        input_tensors["V_CONNECTOR_B"] = _lazy_repeat_feature_vector_storage(
            _pad_optional_vector(connector_weights.bias, padded_output_dim),
            batch_size=1,
            seq_len=connector_seq_len,
            rows_per_batch=connector_rows,
            cols=padded_output_dim,
        )
        data_order.append("V_CONNECTOR_B")

    tensor_layouts = _tensor_layout_metadata(prog, input_tensors)
    o_vram_addr = prog.get_vram_addr(current.name)
    comparison_params = {
        "start_row_idx": o_vram_addr // hw.mlen,
        "num_rows": (connector_seq_len * padded_output_dim) // hw.mlen,
        "num_batches": connector_seq_len,
        "elements_per_batch": padded_output_dim,
        "row_dim": hw.mlen,
        "physical_rows": current.physical_shape[0],
        "use_stride_mode": padded_output_dim > hw.mlen,
    }
    info = {
        "component": "connector_only",
        "seq_len": args.seq_len,
        "output_seq_len": connector_seq_len,
        "hidden_size": hidden,
        "padded_hidden_size": padded_hidden,
        "connector_input_dim": connector_weights.input_dim,
        "connector_storage_input_dim": connector_storage_dim,
        "connector_output_dim": connector_weights.output_dim,
        "padded_output_hidden_size": padded_output_dim,
        "mlen": hw.mlen,
        "blen": hw.blen,
        "isa_lines": len(isa.splitlines()),
    }
    hbm_addrs = {}
    for name, inp in prog._inputs.items():
        if hasattr(inp, "hbm_addr"):
            hbm_addrs[name] = inp.hbm_addr
    return {
        "isa": isa,
        "golden_output": golden_out,
        "padded_golden_output": padded_golden_output,
        "hf_ground_truth": golden_out,
        "input_tensors": input_tensors,
        "tensor_layouts": tensor_layouts,
        "data_order": data_order,
        "fp_preload": [0.0, 0.0, float("-inf"), 1e-6, 1.0 / hidden, 1.0, 1.702] + [0.0] * 3,
        "comparison_params": comparison_params,
        "info": info,
        "hbm_addrs": hbm_addrs,
        "sim_golden_result": {
            "original_output": padded_golden_output,
            "tensor_layouts": tensor_layouts,
            "data_order": data_order,
            "compile_info": info,
        },
    }


def run_vlm_e2e_case(
    model,
    args: argparse.Namespace,
    hw: HardwareConfig,
    build_dir: Path,
) -> None:
    if args.batch_size != 1:
        raise NotImplementedError("vlm-e2e currently supports batch_size=1")

    vision_layers = (
        args.vision_layers if args.vision_layers is not None else (args.layers if args.layers is not None else 0)
    )
    decoder_layers = (
        args.text_layers if args.text_layers is not None else (args.layers if args.layers is not None else 1)
    )

    vision_result = compile_native_hf_vision_encoder(
        model,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_layers=vision_layers,
        layer_idx_start=args.layer_idx_start,
        mlen=hw.mlen,
        blen=hw.blen,
        mram_tile_capacity=args.mram_tile_capacity,
        seed=args.seed,
        golden_precision=args.golden_precision,
        include_connector=True,
        stop_after="connector",
    )
    vision_args = argparse.Namespace(**vars(args))
    vision_args.case = "vlm-e2e-vision-connector"
    vision_build_dir = build_dir / "vision_connector"
    _prepare_and_maybe_run(
        vision_build_dir,
        vision_result,
        "smolvlm2_vlm_e2e_vision_connector",
        vision_args,
        hw,
    )
    if args.no_run:
        connector_output = vision_result["golden_output"]
        connector_source = "golden_output_no_run"
    else:
        connector_output = _extract_emulated_connector_output(vision_build_dir, vision_result)
        connector_source = "emulator_vram"

    text_result = build_vlm_e2e_text_decoder_case(
        model,
        args,
        hw,
        connector_output=connector_output,
        connector_source=connector_source,
        decoder_layers=decoder_layers,
    )
    text_args = argparse.Namespace(**vars(args))
    text_args.case = "vlm-e2e-text-decoder"
    text_build_dir = build_dir / "text_decoder"
    _prepare_and_maybe_run(
        text_build_dir,
        text_result,
        "smolvlm2_vlm_e2e_text_decoder",
        text_args,
        hw,
    )

    _write_vlm_e2e_sidecars(
        build_dir,
        vision_result=vision_result,
        text_result=text_result,
        vision_build_dir=vision_build_dir,
        text_build_dir=text_build_dir,
    )


def _extract_emulated_connector_output(build_dir: Path, vision_result: dict) -> torch.Tensor:
    results, _params = compare_emulator_output(build_dir)
    if not results.get("finite_pass", False):
        raise RuntimeError(
            "Vision connector emulator output contains non-finite values: "
            f"golden={results.get('nonfinite_golden_count', 0)}, "
            f"simulated={results.get('nonfinite_simulated_count', 0)}"
        )
    if not results.get("test_pass", False):
        raise RuntimeError("Vision connector emulator output did not satisfy the configured comparison threshold")

    info = vision_result["info"]
    output_seq_len = int(info["output_seq_len"])
    output_hidden = int(info["output_hidden_size"])
    padded_output_hidden = int(info["padded_output_hidden_size"])
    expected_values = output_seq_len * padded_output_hidden

    simulated = results["simulated_values"].detach().cpu().float()
    if simulated.numel() < expected_values:
        raise RuntimeError(
            f"Connector VRAM extraction produced {simulated.numel()} values, expected at least {expected_values}"
        )
    simulated = simulated[:expected_values].reshape(output_seq_len, padded_output_hidden)
    return simulated[:, :output_hidden].contiguous()


def build_vlm_e2e_text_decoder_case(
    model,
    args: argparse.Namespace,
    hw: HardwareConfig,
    *,
    connector_output: torch.Tensor,
    connector_source: str = "golden_output",
    decoder_layers: int | None = None,
) -> dict:
    if args.batch_size != 1:
        raise NotImplementedError("vlm-e2e currently supports batch_size=1")
    if args.text_seq_len <= 0:
        raise ValueError(f"text_seq_len must be positive, got {args.text_seq_len}")

    model_cfg = extract_model_config(model)
    connector_output = _normalize_connector_output(connector_output, hidden_size=model_cfg.hidden_size)
    connector_seq_len = int(connector_output.shape[1])
    if args.text_seq_len < connector_seq_len + 1:
        raise ValueError(f"text_seq_len={args.text_seq_len} must fit BOS plus {connector_seq_len} image tokens")

    input_ids = _build_vlm_e2e_input_ids(
        model,
        text_seq_len=args.text_seq_len,
        connector_seq_len=connector_seq_len,
    )
    token_embeds = _embed_text_tokens(model, input_ids)
    merged_embeds = _merge_smolvlm_inputs(
        model,
        input_ids=input_ids,
        inputs_embeds=token_embeds,
        image_hidden_states=connector_output.to(token_embeds.device),
    )

    layers = decoder_layers if decoder_layers is not None else (args.layers if args.layers is not None else 1)
    result = compile_native_hf_decoder(
        model,
        seq_len=args.text_seq_len,
        batch_size=args.batch_size,
        num_layers=layers,
        layer_idx_start=args.layer_idx_start,
        mlen=hw.mlen,
        blen=hw.blen,
        hlen=hw.hlen,
        broadcast_amount=hw.broadcast_amount,
        mram_tile_capacity=args.mram_tile_capacity,
        seed=args.seed,
        golden_precision=args.golden_precision,
        component="decoder",
        decoder_input_embeds=merged_embeds,
    )
    image_positions = list(range(1, 1 + connector_seq_len))
    result["info"].update(
        {
            "component": "vlm_e2e_text_decoder",
            "text_seq_len": args.text_seq_len,
            "connector_seq_len": connector_seq_len,
            "image_token_positions": image_positions,
            "image_token_id": int(_required_config_token_id(model, "image_token_id")),
            "bos_token_id": int(input_ids[0, 0].item()),
            "connector_source_for_text": connector_source,
        }
    )
    result["sim_golden_result"]["compile_info"] = result["info"]
    result["vlm_e2e"] = {
        "connector_output": connector_output.detach().cpu().contiguous(),
        "input_ids": input_ids.detach().cpu().contiguous(),
        "token_embeds": token_embeds.detach().cpu().contiguous(),
        "merged_inputs_embeds": merged_embeds.detach().cpu().contiguous(),
        "image_positions": image_positions,
        "connector_source": connector_source,
    }
    return result


def _normalize_connector_output(connector_output: torch.Tensor, *, hidden_size: int) -> torch.Tensor:
    connector_output = connector_output.detach().float()
    if connector_output.dim() == 2:
        connector_output = connector_output.unsqueeze(0)
    if connector_output.dim() != 3:
        raise ValueError(f"Expected connector output to be 2D or 3D, got shape {tuple(connector_output.shape)}")
    if connector_output.shape[0] != 1:
        raise NotImplementedError("vlm-e2e currently supports one connector block")
    if connector_output.shape[-1] < hidden_size:
        raise ValueError(
            f"connector output hidden dim {connector_output.shape[-1]} is smaller than text hidden {hidden_size}"
        )
    return connector_output[..., :hidden_size].contiguous()


def _build_vlm_e2e_input_ids(
    model,
    *,
    text_seq_len: int,
    connector_seq_len: int,
) -> torch.Tensor:
    image_token_id = int(_required_config_token_id(model, "image_token_id"))
    bos_token_id = int(_config_token_id(model, "bos_token_id", default=1))
    pad_token_id = _config_token_id(model, "pad_token_id", default=None)
    vocab_size = int(extract_model_config(model).vocab_size or _embedding_vocab_size(model))
    embedding_vocab_size = _embedding_vocab_size(model)
    normal_vocab_size = min(vocab_size, embedding_vocab_size)
    if image_token_id >= embedding_vocab_size:
        raise ValueError(f"image_token_id={image_token_id} is outside embedding vocab size {embedding_vocab_size}")

    forbidden = {image_token_id, bos_token_id}
    if pad_token_id is not None:
        forbidden.add(int(pad_token_id))
    if normal_vocab_size <= len(forbidden):
        raise ValueError(f"Cannot choose deterministic text tokens from vocab size {normal_vocab_size}")

    input_ids = torch.empty((1, text_seq_len), dtype=torch.long)
    input_ids[0, 0] = bos_token_id
    for pos in range(1, text_seq_len):
        token_id = (17 + pos * 37) % normal_vocab_size
        while token_id in forbidden:
            token_id = (token_id + 1) % normal_vocab_size
        input_ids[0, pos] = token_id
    input_ids[0, 1 : 1 + connector_seq_len] = image_token_id
    return input_ids.contiguous()


def _embed_text_tokens(model, input_ids: torch.Tensor) -> torch.Tensor:
    root = find_model_root(model)
    embed = embedding_module(root)
    if embed is None:
        raise ValueError("Model does not expose text embed_tokens for vlm-e2e")
    device = next(embed.parameters()).device
    with torch.no_grad():
        return embed(input_ids.to(device)).detach().float().contiguous()


def _merge_smolvlm_inputs(
    model,
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_hidden_states: torch.Tensor,
) -> torch.Tensor:
    merger_owner = _find_inputs_merger_owner(model)
    if merger_owner is not None:
        with torch.no_grad():
            return (
                merger_owner.inputs_merger(
                    input_ids=input_ids.to(inputs_embeds.device),
                    inputs_embeds=inputs_embeds,
                    image_hidden_states=image_hidden_states,
                )
                .detach()
                .float()
                .contiguous()
            )

    image_token_id = int(_required_config_token_id(model, "image_token_id"))
    return _local_smolvlm_inputs_merger(
        input_ids=input_ids.to(inputs_embeds.device),
        inputs_embeds=inputs_embeds,
        image_hidden_states=image_hidden_states,
        image_token_id=image_token_id,
    )


def _local_smolvlm_inputs_merger(
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    image_hidden_states: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    _, patch_size, _ = image_hidden_states.shape
    image_mask = input_ids == image_token_id
    num_image_tokens = image_mask.sum(dim=1)
    if not torch.all(num_image_tokens % patch_size == 0):
        raise ValueError("At least one sample has <image> tokens not divisible by connector_seq_len")

    blocks_per_sample = num_image_tokens // patch_size
    expected_blocks = int(blocks_per_sample.sum().item())
    if expected_blocks != int(image_hidden_states.shape[0]):
        raise ValueError(
            f"Image token blocks ({expected_blocks}) do not match connector blocks ({image_hidden_states.shape[0]})"
        )

    offsets = torch.nn.functional.pad(blocks_per_sample.cumsum(dim=0), (1, 0), value=0)
    block_offset = offsets[:-1]
    row_cum = image_mask.cumsum(dim=-1)
    chunk_idx = (row_cum - 1) // patch_size
    local_idx = (row_cum - 1) % patch_size
    block_idx = block_offset.unsqueeze(1) + chunk_idx

    image_embeds = torch.zeros_like(inputs_embeds)
    image_embeds[image_mask] = image_hidden_states[block_idx[image_mask], local_idx[image_mask], :]
    return torch.where(image_mask.unsqueeze(-1), image_embeds, inputs_embeds).detach().float().contiguous()


def _find_inputs_merger_owner(model):
    for candidate in (model, getattr(model, "model", None)):
        if candidate is not None and hasattr(candidate, "inputs_merger"):
            return candidate
    return None


def _config_token_id(model, name: str, *, default):
    for config in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if config is not None:
            value = getattr(config, name, None)
            if value is not None:
                return value
    return default


def _required_config_token_id(model, name: str):
    value = _config_token_id(model, name, default=None)
    if value is None:
        raise ValueError(f"Model config does not expose {name}")
    return value


def _embedding_vocab_size(model) -> int:
    root = find_model_root(model)
    embed = embedding_module(root)
    if embed is None or not hasattr(embed, "num_embeddings"):
        raise ValueError("Model does not expose an embedding vocab size")
    return int(embed.num_embeddings)


def _write_vlm_e2e_sidecars(
    build_dir: Path,
    *,
    vision_result: dict,
    text_result: dict,
    vision_build_dir: Path,
    text_build_dir: Path,
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    sidecars = text_result["vlm_e2e"]
    connector_path = build_dir / "connector_output_for_text.pt"
    input_ids_path = build_dir / "text_input_ids.pt"
    merged_path = build_dir / "merged_inputs_embeds.pt"
    torch.save(sidecars["connector_output"], connector_path)
    torch.save(sidecars["input_ids"], input_ids_path)
    torch.save(sidecars["merged_inputs_embeds"], merged_path)

    manifest = {
        "vision_build_dir": str(vision_build_dir),
        "text_build_dir": str(text_build_dir),
        "connector_output_path": str(connector_path),
        "text_input_ids_path": str(input_ids_path),
        "merged_inputs_embeds_path": str(merged_path),
        "connector_info": vision_result.get("info", {}),
        "text_info": text_result.get("info", {}),
        "image_positions": sidecars["image_positions"],
        "connector_source_for_text": sidecars.get("connector_source", "unknown"),
    }
    with (build_dir / "vlm_e2e_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _prepare_and_maybe_run(
    build_dir: Path,
    result: dict,
    asm_name: str,
    args: argparse.Namespace,
    hw: HardwareConfig,
) -> None:
    input_tensors = result["input_tensors"]
    if args.dump_input_tensors:
        env_input_tensors = {name: materialize_tensor_spec(tensor) for name, tensor in input_tensors.items()}
        mem_input_tensors = None
    else:
        env_input_tensors = {}
        mem_input_tensors = input_tensors

    create_sim_env(
        env_input_tensors,
        result["isa"],
        result["sim_golden_result"],
        result["fp_preload"],
        build_dir=str(build_dir),
        tensor_layouts=result.get("tensor_layouts"),
    )
    create_mem_for_sim(
        data_size=256,
        mode="behave_sim",
        asm=asm_name,
        data=None,
        specified_data_order=result["data_order"],
        build_path=build_dir,
        input_tensors=mem_input_tensors,
        tensor_layouts=result.get("tensor_layouts"),
        hbm_addrs=result.get("hbm_addrs"),
    )
    comparison_params = dict(result["comparison_params"])
    comparison_params.setdefault("atol", float(args.atol))
    comparison_params.setdefault("rtol", float(args.rtol))
    comparison_params.setdefault(
        "min_allclose_match_rate",
        100.0 if args.strict_allclose else float(args.min_allclose_rate),
    )
    with (build_dir / "comparison_params.json").open("w") as f:
        json.dump(comparison_params, f, indent=2)
    with (build_dir / "generated_asm_code.asm").open("w") as f:
        f.write(result["isa"])
    # Write hbm_size.txt so the emulator allocates enough HBM tiles.
    # The heuristic (2 * preload_bytes) in run_emulator is correct,
    # but for large models the preload file may be misidentified.
    hbm_preload = build_dir / "hbm_for_behave_sim.bin"
    if hbm_preload.exists():
        preload_bytes = hbm_preload.stat().st_size
        hbm_bytes = (((2 * preload_bytes) + 63) // 64) * 64
        (build_dir / "hbm_size.txt").write_text(str(hbm_bytes))

    print(f"\nPrepared {args.case} in {build_dir}")
    print(f"  info: {json.dumps(result.get('info', {}), indent=2, sort_keys=True)}")
    if not args.no_run:
        run_and_assert(build_dir, args.case, mlen=hw.mlen, blen=hw.blen)


def _configure_hw(
    args: argparse.Namespace,
    build_dir: Path,
    default_hlen: int | None = None,
    default_broadcast_amount: int | None = None,
) -> HardwareConfig:
    hw = HardwareConfig.from_args(
        args,
        default_hlen=default_hlen if default_hlen is not None else min(int(args.mlen), 64),
        default_broadcast_amount=default_broadcast_amount
        if default_broadcast_amount is not None
        else max(1, int(args.mlen) // 64),
    )
    settings_path = hw.write_toml(build_dir)
    os.environ["PLENA_SETTINGS_TOML"] = str(settings_path)
    if args.latency_profile:
        os.environ["PLENA_LATENCY_PROFILE"] = args.latency_profile
    return hw


def _load_model(model_id: str, *, local_files_only: bool, trust_remote_code: bool = False):
    from transformers import AutoModel

    kwargs = {"torch_dtype": torch.bfloat16}
    if local_files_only:
        kwargs["local_files_only"] = True
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    print(f"Loading {model_id} ...")
    return AutoModel.from_pretrained(model_id, **kwargs)


if __name__ == "__main__":
    main()
