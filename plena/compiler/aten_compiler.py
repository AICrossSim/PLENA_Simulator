"""
ATen Compiler for PLENA ISA.

Traces an nn.Module via torch.export, walks the ATen graph, and dispatches
to existing PLENA backends to produce ISA code.

Usage:
    isa_str, info = compile_module(model, (x,))
"""

import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Set, Tuple

_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "tools"))

import torch
import torch.nn as nn
from torch.export.graph_signature import InputKind

from plena_program import PLENAProgram
from plena.ops.plena.linear_ops import linear_plena
from plena.ops.plena.norm_ops import rms_norm_plena, layer_norm_plena
from plena.ops.plena.ffn_ops import ffn_plena

from quant.quantizer.hardware_quantizer.mxfp import _mx_fp_quantize_hardware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quantize_to_mxfp(tensor: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to MXFP8 matching HBM hardware format; return dequantized result."""
    orig_shape = tensor.shape
    tensor_2d = tensor.float().reshape(-1, tensor.shape[-1])
    bm_x, _, _, _ = _mx_fp_quantize_hardware(
        tensor_2d,
        width=8,
        exponent_width=4,
        exponent_bias_width=8,
        block_size=[1, 8],
    )
    return bm_x.reshape(orig_shape)


# ---------------------------------------------------------------------------
# ATen op handlers
# ---------------------------------------------------------------------------

def _handle_linear(prog, node, tensor_map):
    """Handle aten.linear.default(input, weight, bias=None)."""
    args = node.args
    input_node = args[0]
    weight_node = args[1]

    input_var = tensor_map[input_node.name]
    weight_var = tensor_map[weight_node.name]

    # input_var should be a VRAMMatrixVar (loaded batch), weight_var an InputVar (HBM)
    out_var = linear_plena(prog, input_var, weight_var)
    tensor_map[node.name] = out_var
    return out_var


def _handle_mm(prog, node, tensor_map):
    """Handle aten.mm.default(input, other) — same as linear without bias."""
    args = node.args
    input_node = args[0]
    weight_node = args[1]

    input_var = tensor_map[input_node.name]
    weight_var = tensor_map[weight_node.name]

    out_var = linear_plena(prog, input_var, weight_var)
    tensor_map[node.name] = out_var
    return out_var


def _handle_rms_norm(prog, node, tensor_map):
    """Handle aten.rms_norm.default(input, normalized_shape, weight=None, eps=None).

    The weight parameter is registered as an HBM input but PLENA's rms_norm_plena
    does not use it directly (weight scaling is handled via FPRAM preload).
    eps defaults to 1e-6 when not provided by the ATen node.
    """
    args = node.args
    input_node = args[0]
    # args[1] = normalized_shape (list), args[2] = weight node (may be absent)
    # args[3] = eps (float, may be absent)
    eps = args[3] if len(args) > 3 and args[3] is not None else 1e-6

    input_var = tensor_map[input_node.name]
    out_var = rms_norm_plena(prog, input_var, eps=eps)
    tensor_map[node.name] = out_var
    return out_var


def _handle_layer_norm(prog, node, tensor_map):
    """Handle aten.layer_norm.default(input, normalized_shape, weight, bias, eps, cudnn_enable).

    Weight and bias are registered as HBM inputs but PLENA's layer_norm_plena
    does not apply them directly (handled via FPRAM preload).
    eps defaults to 1e-5 when not provided by the ATen node.
    """
    args = node.args
    input_node = args[0]
    # args[1] = normalized_shape (list)
    # args[2] = weight node (may be absent/None)
    # args[3] = bias node (may be absent/None)
    # args[4] = eps (float, may be absent)
    eps = args[4] if len(args) > 4 and args[4] is not None else 1e-5

    input_var = tensor_map[input_node.name]
    out_var = layer_norm_plena(prog, input_var, eps=eps)
    tensor_map[node.name] = out_var
    return out_var


# ---------------------------------------------------------------------------
# FFN fusion pre-pass
# ---------------------------------------------------------------------------

class _FFNPattern(NamedTuple):
    """Detected FFN pattern in the ATen graph."""
    x_node_name: str        # activation input node name
    w_gate_node_name: str   # gate weight placeholder name
    w_up_node_name: str     # up-projection weight placeholder name
    w_down_node_name: str   # down-projection weight placeholder name
    output_node_name: str   # final linear node that produces the FFN output
    fused_node_names: Set[str]  # all intermediate node names consumed by the fusion


def _detect_ffn_patterns(graph) -> List[_FFNPattern]:
    """Detect SiLU-gated FFN patterns in the ATen graph.

    Pattern:
        gate_linear = aten.linear(x, w_gate)
        silu_out    = aten.silu(gate_linear)
        up_linear   = aten.linear(x, w_up)
        mul_out     = aten.mul(silu_out, up_linear)
        down_linear = aten.linear(mul_out, w_down)

    The silu input can be either the gate or up projection; the other projection
    is the non-silu'd branch fed to mul. We identify gate as the silu'd branch
    and up as the non-silu'd branch (matching ffn_plena semantics where
    output = w_down @ (silu(w_gate @ x) * (w_up @ x)), but note that ffn_plena
    internally applies silu to the *up* weight's projection — so we must map
    the silu'd ATen branch to w_up and the non-silu'd branch to w_gate).

    Returns list of _FFNPattern tuples.
    """
    aten = torch.ops.aten
    patterns = []

    # Build a dict for quick node lookup by name
    node_by_name = {}
    for n in graph.nodes:
        node_by_name[n.name] = n

    # Find all silu nodes as FFN anchors
    for node in graph.nodes:
        if node.op != "call_function" or node.target != aten.silu.default:
            continue

        silu_node = node
        silu_input = silu_node.args[0]  # should be a linear node

        # silu input must be aten.linear
        if silu_input.op != "call_function" or silu_input.target != aten.linear.default:
            continue

        silu_linear_node = silu_input
        x_node = silu_linear_node.args[0]        # activation
        w_silu_node = silu_linear_node.args[1]    # weight of the silu'd branch

        # Find the mul node that uses silu output
        mul_node = None
        for user in silu_node.users:
            if user.op == "call_function" and user.target == aten.mul.Tensor:
                mul_node = user
                break
        if mul_node is None:
            continue

        # mul has two args: silu_out and the other linear
        other_arg = None
        for arg in mul_node.args:
            if hasattr(arg, 'name') and arg.name != silu_node.name:
                other_arg = arg
                break
        if other_arg is None:
            continue

        # The other arg must be aten.linear with the same x input
        if other_arg.op != "call_function" or other_arg.target != aten.linear.default:
            continue
        other_linear_node = other_arg
        if other_linear_node.args[0].name != x_node.name:
            continue  # different activation input — not an FFN pattern

        w_other_node = other_linear_node.args[1]  # weight of the non-silu'd branch

        # Find the down-projection linear that consumes mul output
        down_linear_node = None
        for user in mul_node.users:
            if user.op == "call_function" and user.target == aten.linear.default:
                down_linear_node = user
                break
        if down_linear_node is None:
            continue

        w_down_node = down_linear_node.args[1]

        # ffn_plena semantics: output = w_down @ (silu(w_up @ x) * (w_gate @ x))
        # Hardware applies silu to the "up" projection internally.
        # In the ATen graph, silu is applied to w_silu_node's projection.
        # Therefore: w_up (silu'd in hardware) = w_silu_node,
        #            w_gate (non-silu'd)       = w_other_node
        fused = {
            silu_linear_node.name,   # gate/up linear (silu'd)
            silu_node.name,          # silu
            other_linear_node.name,  # the other linear
            mul_node.name,           # element-wise mul
            down_linear_node.name,   # down projection
        }

        patterns.append(_FFNPattern(
            x_node_name=x_node.name,
            w_gate_node_name=w_other_node.name,
            w_up_node_name=w_silu_node.name,
            w_down_node_name=w_down_node.name,
            output_node_name=down_linear_node.name,
            fused_node_names=fused,
        ))

    return patterns


# Map from ATen op to handler
_OP_TABLE = {}


def _register_ops():
    """Populate op table — called once at import time."""
    aten = torch.ops.aten
    _OP_TABLE[aten.linear.default] = _handle_linear
    _OP_TABLE[aten.mm.default] = _handle_mm
    _OP_TABLE[aten.rms_norm.default] = _handle_rms_norm
    _OP_TABLE[aten.layer_norm.default] = _handle_layer_norm


_register_ops()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compile_module(
    model: nn.Module,
    example_inputs: tuple,
    mlen: int = 64,
    blen: int = 4,
    real_data_ratio: float = (8 * 8 + 8) / (8 * 8),
    fp_preload: list = None,
) -> Tuple[str, dict]:
    """
    Trace model with torch.export and compile to PLENA ISA.

    Returns:
        (isa_str, info_dict) where info_dict has:
          - 'prog': the PLENAProgram
          - 'tensor_map': Dict[node_name, TensorVar]
          - 'input_names': list of placeholder node names (in order)
          - 'output_var': the final output TensorVar
          - 'state_dict_tensors': Dict[name, tensor] (original float weights for sim env)
    """
    model.eval()

    # Export the model to ATen graph
    ep = torch.export.export(model, example_inputs)
    graph = ep.graph
    state_dict = ep.state_dict

    # Build lookup: placeholder_name -> InputSpec (to distinguish params from user inputs)
    spec_by_name = {}
    for spec in ep.graph_signature.input_specs:
        spec_by_name[spec.arg.name] = spec

    prog = PLENAProgram(mlen=mlen, blen=blen, real_data_ratio=real_data_ratio)

    tensor_map: Dict[str, object] = {}
    input_names = []                          # user-input placeholder names (activation)
    hbm_input_order = []                      # ALL prog.input() names in declaration order
    state_dict_tensors: Dict[str, torch.Tensor] = {}
    user_input_idx = 0
    output_var = None

    # --- FFN fusion pre-pass ---
    ffn_patterns = _detect_ffn_patterns(graph)
    fused_nodes: Set[str] = set()
    # Map: output_node_name -> _FFNPattern for triggering ffn_plena dispatch
    ffn_triggers: Dict[str, _FFNPattern] = {}
    for pat in ffn_patterns:
        fused_nodes |= pat.fused_node_names
        ffn_triggers[pat.output_node_name] = pat

    for node in graph.nodes:
        if node.op == "placeholder":
            spec = spec_by_name[node.name]

            if spec.kind == InputKind.PARAMETER:
                # Weight parameter — spec.target is the state_dict key (e.g. "weight")
                weight_tensor = state_dict[spec.target]  # shape (out_features, in_features)
                if weight_tensor.dim() < 2:
                    # 1-D bias/scale parameters (e.g. RMSNorm weight) are not loaded into
                    # HBM by PLENA ops that use FPRAM preload instead. Skip HBM registration
                    # but record the raw tensor so callers can inspect it if needed.
                    state_dict_tensors[node.name] = weight_tensor.clone()
                    # Leave tensor_map[node.name] unset; handlers that need it will handle None.
                else:
                    # Transpose: PLENA linear expects (in_features, out_features)
                    weight_T = weight_tensor.T.contiguous()
                    # Store original float tensor for sim env setup
                    state_dict_tensors[node.name] = weight_T.clone()
                    # Register as HBM input with transposed shape
                    input_var = prog.input(node.name, shape=tuple(weight_T.shape))
                    tensor_map[node.name] = input_var
                    hbm_input_order.append(node.name)

            elif spec.kind == InputKind.USER_INPUT:
                # Activation input
                input_names.append(node.name)
                x = example_inputs[user_input_idx]
                user_input_idx += 1
                input_var = prog.input(node.name, shape=tuple(x.shape))
                hbm_input_order.append(node.name)
                # Load activation to VRAM
                batch_var = prog.load_batch(input_var, name=node.name)
                tensor_map[node.name] = batch_var

            elif spec.kind == InputKind.BUFFER:
                # Buffer (e.g. running_mean) — treat like parameter
                buf_tensor = state_dict[spec.target]
                state_dict_tensors[node.name] = buf_tensor.clone()
                input_var = prog.input(node.name, shape=tuple(buf_tensor.shape))
                tensor_map[node.name] = input_var
                hbm_input_order.append(node.name)

        elif node.op == "call_function":
            # --- FFN fusion handling ---
            if node.name in ffn_triggers:
                # This is the output node of a fused FFN — dispatch whole pattern
                pat = ffn_triggers[node.name]
                x_var = tensor_map[pat.x_node_name]
                w_gate_var = tensor_map[pat.w_gate_node_name]
                w_up_var = tensor_map[pat.w_up_node_name]
                w_down_var = tensor_map[pat.w_down_node_name]
                out = ffn_plena(prog, x_var, w_gate_var, w_up_var, w_down_var)
                tensor_map[node.name] = out
                output_var = out
            elif node.name in fused_nodes:
                # Intermediate node already consumed by FFN fusion — skip
                pass
            elif node.target in _OP_TABLE:
                handler = _OP_TABLE[node.target]
                out = handler(prog, node, tensor_map)
                output_var = out
            else:
                raise NotImplementedError(
                    f"ATen op not supported: {node.target}. "
                    f"Supported ops: {list(_OP_TABLE.keys())}"
                )

        elif node.op == "output":
            # The output node's args reference the final result node(s)
            # For single-output models, args is ((result_node,),)
            pass

        # Skip 'get_attr' and other node types

    # Compile to ISA
    isa_str = prog.compile()

    info = {
        "prog": prog,
        "tensor_map": tensor_map,
        "input_names": input_names,
        "hbm_input_order": hbm_input_order,
        "output_var": output_var,
        "state_dict_tensors": state_dict_tensors,
    }

    return isa_str, info
