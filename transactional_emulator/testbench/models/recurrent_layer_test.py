"""Connected first recurrent sublayers: Compiler -> machine code -> Rust.

Both checkpoints are decoded OFFLINE into BF16 weights; all intermediate
operators execute in Rust. This is not compressed-weight codec performance,
a transformer residual/MoE block, full-model execution, or RTL certification.
KDA uses captured hidden inputs; Mamba uses a real first-token embedding.
"""

import argparse
import json
from pathlib import Path
import shlex
import sys
import time
import numpy as np
import torch
from safetensors import safe_open
from transactional_emulator.testbench.aten.recurrent_gate_test import (
    Arena,
    bf,
    delta,
    sigmoid,
    digest,
    metric,
    softplus,
    COMPILER_ROOT,
)
from transactional_emulator.testbench.aten.recurrent_conv_test import run_program, read
from transactional_emulator.testbench.aten.matrix_projection_test import reference as matrix_reference
from transactional_emulator.testbench.aten.recurrent_prepare_test import norm_reference
from compiler.aten.plena.isa_matrix_projection import Projection, lower_b1_projection
from compiler.aten.plena.recurrent_coefficients import (
    GATE_CONSTANTS,
    ConvStep,
    KdaGateRow,
    L2NormRows,
    lower_conv_steps,
    lower_kda_gate_rows,
    lower_l2norm_rows,
    lower_bf16_gather,
    lower_global_rms,
    lower_pointwise_rows,
    MambaGateRow,
    lower_mamba_gate_rows,
)
from compiler.aten.plena.ltile_v2 import Options, lower_group
from analytic_models.performance.ltile_cost import Machine
from analytic_models.performance.matrix_service import MatrixService


KDA_PREFIX = "language_model.model.layers.0."


def tree(values, axis=-1):
    values = np.moveaxis(bf(values), axis, -1)
    while values.shape[-1] > 1:
        values = bf(values[..., ::2] + values[..., 1::2])
    return values[..., 0]


class Program:
    def __init__(self, hardware):
        self.arena = Arena()
        self.lines = []
        self.references = []
        self.stages = []
        self.h = hardware
        self.zero = self.arena.add(np.zeros(2048))
        self.onehot = self.arena.add(np.eye(1, 2048).ravel())
        self.constants = self.arena.add(np.repeat(np.asarray(GATE_CONSTANTS)[:, None], 2048, axis=1))
        masks = np.zeros((16, 2048), np.float32)
        for i in range(16):
            masks[i, i * 128 : (i + 1) * 128] = 1
        self.masks = self.arena.add(masks)

    def output(self, count, extra=0):
        # The meaningful output is poisoned; owned overread padding is zero.
        n = (count + extra + 2047) // 2048 * 2048
        x = np.zeros(n, np.float32)
        x[:count] = 7
        return self.arena.add(x, output=True)

    def emit(self, name, text):
        self.stages.append((name, text))
        self.lines.append(text)

    def check(self, name, address, value):
        self.references.append((name, address, np.asarray(value, np.float32).ravel().copy()))

    def projection(self, name, source, x, w):
        k, n = w.shape
        padded = np.zeros(((k + 31) // 32 * 32, (n + 31) // 32 * 32), np.float32)
        padded[:k, :n] = w
        packed = padded.reshape(len(padded), -1, 32).transpose(1, 0, 2).copy().ravel()
        weights = self.arena.add(packed)
        out = self.output(n, 2048)
        config = Projection(source, weights, out, self.zero, k, n, 256)
        self.emit(name, lower_b1_projection(config))
        expected = matrix_reference(x, w, 256, self.h)
        self.check(name, out, expected)
        print(json.dumps({"stage": name, "built": True, "weight_bytes": config.weight_bytes}), flush=True)
        return out, expected

    def gather(self, name, mapping, expected):
        out = self.output(len(mapping))
        self.emit(name, lower_bf16_gather(mapping, out, self.zero, self.onehot))
        self.check(name, out, expected)
        return out


def lane(base, index):
    return base + index // 2048 * 4096, index % 2048


def load_weight(f, name):
    w = f.get_tensor(KDA_PREFIX + "self_attn." + name + "_proj.weight").float().numpy()
    scale = f.get_tensor(KDA_PREFIX + "self_attn." + name + "_proj.weight_scale").numpy().squeeze(1).squeeze(-1)
    expected = ((w.shape[0] + 127) // 128, (w.shape[1] + 127) // 128)
    if scale.shape != expected:
        raise ValueError("FP8 block scale shape differs")
    for row in range(0, len(w), 128):
        for col in range(0, w.shape[1], 128):
            w[row : row + 128, col : col + 128] *= scale[row // 128, col // 128]
    return bf(w).T.copy()


from analytic_models.performance.weight_codec import decode_nvfp4


MAMBA_PREFIX = "backbone.layers.0."


def run_kda(args):
    data = np.load(args.capture / "kda_inputs_0000.npz")
    h = MatrixService(accumulator=args.accumulator)
    p = Program(h)
    if not 0 <= args.token < len(data["hidden_input"]):
        raise ValueError("token outside captured contiguous input sequence")
    x = bf(data["hidden_input"][args.token].ravel())
    if x.size != 7168:
        raise ValueError("only actual Kimi-K3 first-layer shape accepted")
    carried = np.load(args.state_input) if args.state_input else None
    if not 0 <= args.token < len(data["hidden_input"]):
        raise ValueError("token outside captured contiguous input sequence")
    if carried is not None:
        initial_state = bf(carried["state"])
    elif args.token == 0:
        initial_state = bf(np.load(args.capture / "kda_initial.npy")).reshape(96, 128, 128)
    else:
        indices = list(data["checkpoint_indices"])
        if args.token - 1 not in indices:
            raise ValueError("missing native incoming state checkpoint for this token")
        initial_state = bf(data["checkpoint_states"][indices.index(args.token - 1)])
    if initial_state.shape != (96, 128, 128):
        raise ValueError("incoming KDA state shape differs")
    state_bases = []
    conv_bases = {}
    fp = [7168 * 1e-5, 7168**0.5, 1e-6, 128**-0.5, 1.0, 128 * 1e-5, 128**0.5]
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as f:
        norm_w = bf(f.get_tensor(KDA_PREFIX + "input_layernorm.weight").float().numpy())
        source = p.arena.add(np.pad(x, (0, 2048 * (-(-len(x) // 2048)) - len(x))))
        nw = p.arena.add(np.pad(norm_w, (0, 8192 - len(norm_w))))
        norm = p.output(7168, 2048)
        p.emit("input_norm", lower_global_rms(source, nw, norm, 7168))
        squares = np.pad(bf(x * x), (0, 8192 - len(x))).reshape(-1, 2048)
        total = np.float32(0)
        for row in squares:
            total = bf(total + tree(row))
        inv = bf(bf(1 / bf(np.sqrt(bf(total + bf(fp[0]))))) * bf(fp[1]))
        normalized = bf(bf(x * inv) * norm_w)
        p.check("input_norm", norm, normalized)
        projections = {}
        for name in ("q", "k", "v", "f_a", "b", "g"):
            projections[name] = p.projection(name + "_projection", norm, normalized, load_weight(f, name))
        gbase, gref = p.projection("f_b_projection", *projections["f_a"], load_weight(f, "f_b"))
        conv = {}
        for name in ("q", "k", "v"):
            weights = bf(f.get_tensor(KDA_PREFIX + "self_attn." + name + "_conv1d.weight").float().numpy()[:, 0, :].T)
            wb = p.arena.add(weights)
            if carried is not None:
                initial_conv = bf(carried["conv_" + name])
            elif args.token:
                stage = {"q": 4, "k": 5, "v": 6}[name]
                initial_conv = bf(data[f"stage_{stage}_out_1"][args.token - 1, 0].T)
            else:
                initial_conv = np.zeros_like(weights)
            if initial_conv.shape != (4, 12288):
                raise ValueError("incoming convolution history shape differs")
            hist = p.arena.add(initial_conv, output=True)
            out = p.output(12288)
            conv_bases[name] = hist
            p.emit(
                name + "_conv", lower_conv_steps([ConvStep(projections[name][0], hist, wb, out, 12288)], p.constants)
            )
            history = np.vstack([initial_conv[1:], projections[name][1]])
            products = bf(history * weights)
            pre = bf(bf(products[0] + products[1]) + bf(products[2] + products[3]))
            value = bf(pre * sigmoid(pre))
            p.check(name + "_conv", out, value)
            p.check(name + "_conv_state", hist, history)
            conv[name] = (out, value)
        vectors = {}
        for name, slot in [("q", 3), ("k", 4)]:
            out = p.output(12288)
            p.emit(name + "_norm", lower_l2norm_rows(L2NormRows(conv[name][0], out, 12288, p.masks, p.zero, 2, slot)))
            value = norm_reference(conv[name][1], fp[slot])
            p.check(name + "_norm", out, value)
            vectors[name] = (out, value)
        dt_bias = bf(f.get_tensor(KDA_PREFIX + "self_attn.dt_bias").numpy())
        a_log = bf(f.get_tensor(KDA_PREFIX + "self_attn.A_log").numpy()[:96])
        biasbase = p.arena.add(dt_bias)
        abase = p.arena.add(np.repeat(a_log, 128))
        dbase = p.output(12288)
        betabases = [p.output(2048) for _ in range(6)]
        rows = [
            KdaGateRow(
                gbase + i * 4096,
                biasbase + i * 4096,
                abase + i * 4096,
                projections["b"][0],
                dbase + i * 4096,
                betabases[i],
            )
            for i in range(6)
        ]
        p.emit("decay_beta", lower_kda_gate_rows(rows, p.constants))
        from transactional_emulator.testbench.aten.recurrent_gate_test import exp

        dref = delta(bf(-5 * sigmoid(bf(exp(np.repeat(a_log, 128)) * bf(gref + dt_bias)))))
        beta = sigmoid(projections["b"][1])
        p.check("delta", dbase, dref)
        p.check("beta", betabases[0], beta)
        raw_out = p.output(12288, 2048)
        raw_ref = np.empty((96, 128), np.float32)
        options = Options("kda", args.control)
        for group in range(6):
            first = group * 16
            index = np.arange(first * 128, (first + 16) * 128).reshape(16, 128).T.ravel()
            updates = []
            updref = []
            dots = []
            dotref = []
            for ix in index:
                updates.extend([lane(dbase, int(ix)), lane(vectors["k"][0], int(ix))])
                updref.extend([dref[ix], vectors["k"][1][ix]])
                dots.extend([lane(vectors["q"][0], int(ix)), None])
                dotref.extend([vectors["q"][1][ix], 0.0])
            update = p.gather("pack_update_" + str(group), updates, updref)
            dot = p.gather("pack_dot_" + str(group), dots, dotref)
            scalar_map = []
            scalar_ref = []
            for head in range(first, first + 16):
                scalar_map.extend([lane(betabases[0], head), None])
                scalar_ref.extend([beta[head], 0.0])
            scalar = p.gather("pack_beta_" + str(group), scalar_map, scalar_ref)
            state = p.arena.add(initial_state[first : first + 16], output=True)
            state_bases.append(state)
            memory = dict(
                states=[state],
                update=[update],
                dot=[dot],
                input=conv["v"][0] + group * 4096,
                scalar=scalar,
                output=raw_out + group * 4096,
            )
            p.emit("recurrence_" + str(group), "\n".join(lower_group(options, memory).lines) + "\n")
            key = vectors["k"][1].reshape(96, 128)[first : first + 16]
            query = vectors["q"][1].reshape(96, 128)[first : first + 16]
            value = conv["v"][1].reshape(96, 128)[first : first + 16]
            old = initial_state[first : first + 16]
            decay = dref.reshape(96, 128)[first : first + 16]
            decayed = old - decay[:, :, None] * old
            prediction = tree(bf(decayed * key[:, :, None]), axis=1)
            residual = bf(beta[first : first + 16, None] * (value - prediction))
            updated = bf(decayed + key[:, :, None] * residual[:, None, :])
            raw_ref[first : first + 16] = tree(bf(updated * query[:, :, None]), axis=1)
            p.check("state_" + str(group), state, updated)
        p.check("recurrent_output", raw_out, raw_ref)
        rms = p.output(12288)
        p.emit("output_rms", lower_l2norm_rows(L2NormRows(raw_out, rms, 12288, p.masks, p.zero, 5, 6)))
        sumsq = tree(bf(raw_ref * raw_ref))
        scale = bf(bf(1 / bf(np.sqrt(bf(sumsq + bf(fp[5]))))) * bf(fp[6]))
        normalized_out = bf(raw_ref * scale[:, None]).ravel()
        p.check("output_rms", rms, normalized_out)
        norm_weight = np.tile(bf(f.get_tensor(KDA_PREFIX + "self_attn.o_norm.weight").float().numpy()), 96)
        wb = p.arena.add(norm_weight)
        weighted = p.output(12288)
        p.emit("output_norm_weight", lower_pointwise_rows(rms, wb, weighted, 6))
        weighted_ref = bf(normalized_out * norm_weight)
        p.check("output_norm_weight", weighted, weighted_ref)
        gated = p.output(12288, 2048)
        p.emit(
            "output_gate",
            lower_pointwise_rows(
                projections["g"][0], weighted, gated, 6, sigmoid_input=True, constants_base=p.constants
            ),
        )
        gated_ref = bf(sigmoid(projections["g"][1]) * weighted_ref)
        p.check("output_gate", gated, gated_ref)
        output, _ = p.projection("output_projection", gated, gated_ref, load_weight(f, "o"))
    machine = Machine(
        sfu_lanes=32,
        vector_exp_cycles=8,
        vector_softplus_cycles=16,
        vector_reciprocal_cycles=8,
        reduction_tree_bf16=True,
    )
    print(json.dumps({"phase": "execute", "hbm_bytes": len(p.arena.data), "stages": len(p.stages)}), flush=True)
    start = time.monotonic()
    image, result = run_program(
        args.output,
        args.runtime,
        args.memory_root,
        p.arena,
        "".join(p.lines),
        machine,
        h,
        fp_constants=fp + [0.0] * 25,
        recheck_only=args.recheck_only,
        references=p.references,
    )
    checked = 0
    errors = {}
    for name, address, reference in p.references:
        actual = read(image, address, len(reference))
        errors[name] = metric(actual, reference)
        checked += len(reference)
        if not np.array_equal(actual, reference):
            raise AssertionError("connected stage mismatch: " + name + " " + str(errors[name]))
    np.savez_compressed(
        args.output / "state_handoff.npz",
        state=np.concatenate([read(image, b, 16 * 128 * 128).reshape(16, 128, 128) for b in state_bases]),
        **{"conv_" + name: read(image, b, 4 * 12288).reshape(4, 12288) for name, b in conv_bases.items()},
    )
    result.update(
        status="passed_complete_kda_attention_candidate",
        scope=__doc__,
        heads=96,
        key_dim=128,
        value_dim=128,
        batch=1,
        tokens=1,
        token_index=args.token,
        control=args.control,
        checked_values=checked,
        stages=errors,
        elapsed_seconds=time.monotonic() - start,
        versus_native=metric(read(image, output, 7168), data["native_layer_output"][args.token].ravel()),
        source_sha256=args.source_sha256,
        reproduce=shlex.join(
            [sys.executable, "-m", "transactional_emulator.testbench.models.recurrent_layer_test", *sys.argv[1:]]
        ),
        incoming_state_source=str(args.state_input)
        if args.state_input
        else ("initial" if args.token == 0 else "native_checkpoint"),
        formal_full_layer_passed=False,
        formal_decode_passed=False,
        exclusions=[
            "NVFP4/FP8 hardware codec",
            "outer residual and MoE",
            "long token chain",
            "optimized software baseline",
            "RTL timing certification",
        ],
        limitations=[
            "BF16 offline-dequantized weight diagnostic with candidate Matrix/SFU/norm services",
            "software gather is correctness reference; its cost must not manufacture baseline speedup",
            "native comparison also changes FP8 activation quantization and arithmetic boundaries",
        ],
    )
    (args.output / ("rechecked_result.json" if args.recheck_only else "result.json")).write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "cycles": result["observed"]["total"],
                "relative_error": result["versus_native"]["rel_l2"],
            }
        ),
        flush=True,
    )


def run_mamba(args):
    h = MatrixService(accumulator=args.accumulator)
    p = Program(h)
    fp = [2688 * 1e-5, np.sqrt(2688), 512 * 1e-5, np.sqrt(512)]
    carried = np.load(args.state_input) if args.state_input else None
    initial_state = np.zeros((64, 128, 64), np.float32) if carried is None else carried["state"]
    initial_conv = np.zeros((4, 6144), np.float32) if carried is None else carried["conv_state"]
    if initial_state.shape != (64, 128, 64) or initial_conv.shape != (4, 6144):
        raise ValueError("carried state shape does not match Nemotron layer 0")
    state_bases = []
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as f:

        def weight(name):
            return f.get_tensor(MAMBA_PREFIX + name).float().numpy()

        def projection_weight(name):
            prefix = MAMBA_PREFIX + "mixer." + name
            return decode_nvfp4(
                f.get_tensor(prefix + ".weight").numpy(),
                f.get_tensor(prefix + ".weight_scale").float().numpy(),
                f.get_tensor(prefix + ".weight_scale_2").item(),
            )

        hidden = bf(f.get_tensor("input_hidden").float().numpy())
        if hidden.shape != (2688,):
            raise ValueError("fixture is not the real Nemotron hidden shape")
        source = p.arena.add(np.pad(hidden, (0, 4096 - len(hidden))))
        norm_weight = bf(weight("norm.weight"))
        nw = p.arena.add(np.pad(norm_weight, (0, 4096 - len(hidden))))
        normalized = p.output(2688, 2048)
        p.emit("input_norm", lower_global_rms(source, nw, normalized, 2688))
        squares = np.pad(bf(hidden * hidden), (0, 4096 - len(hidden))).reshape(-1, 2048)
        total = np.float32(0)
        for row in squares:
            total = bf(total + tree(row))
        inv = bf(bf(1 / bf(np.sqrt(bf(total + bf(fp[0]))))) * bf(fp[1]))
        norm_ref = bf(bf(hidden * inv) * norm_weight)
        p.check("input_norm", normalized, norm_ref)
        in_weight = projection_weight("in_proj")
        projected, projection_ref = p.projection("input_projection", normalized, norm_ref, bf(in_weight))
        conv_weights = bf(weight("mixer.conv1d.weight")[:, 0, :].T)
        bias = bf(weight("mixer.conv1d.bias"))
        wb, bb = p.arena.add(conv_weights), p.arena.add(bias)
        history = p.arena.add(initial_conv, output=True)
        convolved = p.output(6144)
        p.emit(
            "convolution", lower_conv_steps([ConvStep(projected + 8192, history, wb, convolved, 6144, bb)], p.constants)
        )
        updated_conv = np.vstack([initial_conv[1:], projection_ref[4096:10240]])
        products = bf(updated_conv * conv_weights)
        pre = bf(bf(bf(products[0] + products[1]) + bf(products[2] + products[3])) + bias)
        conv_ref = bf(pre * sigmoid(pre))
        p.check("convolution", convolved, conv_ref)
        p.check("conv_state", history, updated_conv)
        dt_bias = bf(weight("mixer.dt_bias"))
        negative_a = bf(-np.exp(weight("mixer.A_log")))
        db = p.arena.add(np.pad(dt_bias, (0, 1984)))
        ab = p.arena.add(np.pad(negative_a, (0, 1984)))
        dt_base, delta_base = p.output(2048), p.output(2048)
        p.emit(
            "gate_production",
            lower_mamba_gate_rows([MambaGateRow(projected + 10240 * 2, db, ab, dt_base, delta_base)], p.constants),
        )
        dt = softplus(bf(projection_ref[10240:] + dt_bias))
        d = delta(bf(dt * negative_a))
        p.check("dt", dt_base, dt)
        p.check("delta", delta_base, d)
        skip_weight = bf(weight("mixer.D"))
        skipbase = p.arena.add(np.pad(skip_weight, (0, 1984)))
        onebase = p.arena.add(np.ones(2048))
        raw_out = p.output(4096)
        raw_ref = np.empty((64, 64), np.float32)
        for group in range(2):
            first = group * 32
            updates, update_ref, dots, dot_ref = [], [], [], []
            for row in range(128):
                for head in range(first, first + 32):
                    b_index = 4096 + head // 8 * 128 + row
                    c_index = 5120 + head // 8 * 128 + row
                    updates.extend([lane(delta_base, head), lane(convolved, b_index)])
                    update_ref.extend([d[head], conv_ref[b_index]])
                    dots.extend([lane(convolved, c_index), None])
                    dot_ref.extend([conv_ref[c_index], 0.0])
            update = p.gather("pack_update_" + str(group), updates, update_ref)
            dot = p.gather("pack_dot_" + str(group), dots, dot_ref)
            scalar_map, scalar_ref, skip_map, skip_ref = [], [], [], []
            for head in range(first, first + 32):
                scalar_map.extend([None, lane(dt_base, head)])
                scalar_ref.extend([0.0, dt[head]])
                skip_map.extend([lane(onebase, 0), lane(skipbase, head)])
                skip_ref.extend([1.0, skip_weight[head]])
            scalar = p.gather("pack_dt_" + str(group), scalar_map, scalar_ref)
            skip = p.gather("pack_skip_" + str(group), skip_map, skip_ref)
            state = p.arena.add(initial_state[first : first + 32], output=True)
            state_bases.append(state)
            memory = dict(
                states=[state],
                update=[update],
                dot=[dot],
                input=convolved + group * 4096,
                scalar=scalar,
                skip=skip,
                output=raw_out + group * 4096,
            )
            p.emit(
                "recurrence_" + str(group), "\n".join(lower_group(Options("mamba", args.control), memory).lines) + "\n"
            )
            x = conv_ref[:4096].reshape(64, 64)[first : first + 32]
            b = conv_ref[4096:5120].reshape(8, 128)[np.arange(first, first + 32) // 8]
            c = conv_ref[5120:6144].reshape(8, 128)[np.arange(first, first + 32) // 8]
            scaled = bf(x * dt[first : first + 32, None])
            old = bf(initial_state[first : first + 32])
            updated = bf((old - d[first : first + 32, None, None] * old) + b[:, :, None] * scaled[:, None, :])
            reduced = tree(bf(updated * c[:, :, None]), axis=1)
            raw_ref[first : first + 32] = bf(reduced + x * skip_weight[first : first + 32, None])
            p.check("state_" + str(group), state, updated)
        p.check("recurrent_output", raw_out, raw_ref)
        silu = p.output(4096)
        p.emit(
            "silu_gate",
            lower_pointwise_rows(projected, projected, silu, 2, sigmoid_input=True, constants_base=p.constants),
        )
        gate_ref = bf(projection_ref[:4096] * sigmoid(projection_ref[:4096]))
        p.check("silu_gate", silu, gate_ref)
        gated = p.output(4096)
        p.emit("gate_product", lower_pointwise_rows(raw_out, silu, gated, 2))
        gated_ref = bf(raw_ref.ravel() * gate_ref)
        p.check("gate_product", gated, gated_ref)
        masks = np.zeros((4, 2048), np.float32)
        for i in range(4):
            masks[i, i * 512 : (i + 1) * 512] = 1
        masksbase = p.arena.add(masks)
        rms = p.output(4096)
        p.emit("gated_rms", lower_l2norm_rows(L2NormRows(gated, rms, 4096, masksbase, p.zero, 2, 3, 512)))
        grouped = gated_ref.reshape(8, 512)
        ss = tree(bf(grouped * grouped))
        scale = bf(bf(1 / bf(np.sqrt(bf(ss + bf(fp[2]))))) * bf(fp[3]))
        rms_ref = bf(grouped * scale[:, None]).ravel()
        p.check("gated_rms", rms, rms_ref)
        mixer_norm = bf(weight("mixer.norm.weight"))
        mn = p.arena.add(mixer_norm)
        weighted = p.output(4096, 2048)
        p.emit("norm_weight", lower_pointwise_rows(rms, mn, weighted, 2))
        weighted_ref = bf(rms_ref * mixer_norm)
        p.check("norm_weight", weighted, weighted_ref)
        out_weight = projection_weight("out_proj")
        output, _ = p.projection("output_projection", weighted, weighted_ref, bf(out_weight))

        # Independent mathematical FP32 diagnostic, from the same raw weights
        # and its separately carried FP32 state. No producer intermediates substituted.
        x32 = hidden / np.sqrt(np.mean(hidden * hidden) + 1e-5) * norm_weight
        projected32 = x32 @ in_weight
        conv32 = np.zeros((4, 6144), np.float32) if carried is None else carried["conv_state_fp32"]
        conv32 = np.vstack([conv32[1:], projected32[4096:10240]])
        cv32 = np.sum(conv32 * conv_weights, axis=0) + bias
        cv32 *= 1 / (1 + np.exp(-cv32))
        dt32 = np.logaddexp(0, projected32[10240:] + dt_bias)
        xx = cv32[:4096].reshape(64, 64)
        bb32 = cv32[4096:5120].reshape(8, 128)[np.arange(64) // 8]
        cc32 = cv32[5120:6144].reshape(8, 128)[np.arange(64) // 8]
        s32 = np.zeros((64, 128, 64), np.float32) if carried is None else carried["state_fp32"]
        decay32 = np.exp(-np.exp(weight("mixer.A_log")) * dt32)
        s32 = s32 * decay32[:, None, None] + bb32[:, :, None] * (xx * dt32[:, None])[:, None, :]
        y32 = np.sum(s32 * cc32[:, :, None], axis=1) + xx * skip_weight[:, None]
        gate32 = projected32[:4096] / (1 + np.exp(-projected32[:4096]))
        y32 = (y32.ravel() * gate32).reshape(8, 512)
        y32 /= np.sqrt(np.mean(y32 * y32, axis=1, keepdims=True) + 1e-5)
        reference32 = (y32.ravel() * mixer_norm) @ out_weight
    machine = Machine(
        sfu_lanes=32,
        vector_exp_cycles=8,
        vector_softplus_cycles=16,
        vector_reciprocal_cycles=8,
        reduction_tree_bf16=True,
    )
    print(json.dumps({"phase": "execute", "hbm_bytes": len(p.arena.data), "stages": len(p.stages)}), flush=True)
    image, result = run_program(
        args.output,
        args.runtime,
        args.memory_root,
        p.arena,
        "".join(p.lines),
        machine,
        h,
        fp_constants=fp + [0.0] * 28,
        recheck_only=args.recheck_only,
        references=p.references,
    )
    checks = {}
    for name, address, reference in p.references:
        actual = read(image, address, len(reference))
        checks[name] = metric(actual, reference)
        if not np.array_equal(actual, reference):
            raise AssertionError("connected stage mismatch: " + name + " " + str(checks[name]))
    np.savez_compressed(
        args.output / "state_handoff.npz",
        state=np.concatenate([read(image, b, 32 * 128 * 64).reshape(32, 128, 64) for b in state_bases]),
        conv_state=read(image, history, 4 * 6144).reshape(4, 6144),
        state_fp32=s32,
        conv_state_fp32=conv32,
    )
    result.update(
        status="passed_complete_mamba_mixer_candidate",
        scope=__doc__,
        heads=64,
        state_shape=[64, 128, 64],
        batch=1,
        tokens=1,
        stages=checks,
        checked_values=sum(len(x[2]) for x in p.references),
        versus_fp32_formula=metric(read(image, output, 2688), reference32),
        source_sha256=args.source_sha256,
        reproduce=shlex.join(
            [sys.executable, "-m", "transactional_emulator.testbench.models.recurrent_layer_test", *sys.argv[1:]]
        ),
        formal_full_layer_passed=False,
        formal_decode_passed=False,
        incoming_state="zero" if carried is None else str(args.state_input),
        control=args.control,
        comparison_scope="row vs FSM shares extended arithmetic, Matrix access, packing and all surrounding operators",
        incoming_state_sha256=None if carried is None else digest(args.state_input),
        sequence_note="real first prompt embedding; carried-state follow-up repeats that embedding",
        exclusions=[
            "NVFP4 codec and native activation quantization",
            "outer residual/MoE",
            "long chain",
            "optimized baseline",
            "RTL timing certification",
        ],
    )
    result_path = args.output / ("rechecked_result.json" if args.recheck_only else "result.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "cycles": result["observed"]["total"],
                "relative_error": result["versus_fp32_formula"]["rel_l2"],
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["mamba", "kda"], required=True)
    for name in ("checkpoint", "runtime", "memory-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--token", type=int, default=0)
    parser.add_argument("--accumulator", choices=["BF16", "FP32"], default="BF16")
    parser.add_argument(
        "--control",
        choices=["row", "fsm"],
        default="fsm",
        help="Same extended arithmetic/access hardware; row is not old ISA",
    )
    parser.add_argument("--state-input", type=Path)
    parser.add_argument("--recheck-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists() and not args.recheck_only:
        raise FileExistsError(args.output)
    sources = [
        args.checkpoint,
        Path(__file__),
        COMPILER_ROOT / "aten/plena/recurrent_coefficients.py",
        COMPILER_ROOT / "aten/plena/isa_matrix_projection.py",
        COMPILER_ROOT / "aten/plena/ltile_v2.py",
    ]
    if args.capture:
        sources.extend([args.capture / "kda_inputs_0000.npz", args.capture / "kda_initial.npy"])
    if args.state_input:
        sources.append(args.state_input)
    args.source_sha256 = {str(path): digest(path) for path in sources}
    if args.kind == "kda":
        if args.capture is None:
            parser.error("--capture is required for KDA")
        run_kda(args)
    else:
        if args.token:
            parser.error("Mamba follow-up repeats the first embedding; use --state-input")
        run_mamba(args)


if __name__ == "__main__":
    main()
