"""Hybrid-model performance driver: one cost per layer, by that layer's type.

`llama_model.py`, `gpt_oss_model.py` and `mamba2_model.py` all assume one block
shape repeated `num_hidden_layers` times -- the assumption is literally the line
`overall_exe_cycle += block_cycles * self.num_hidden_layers`. A hybrid model has
no such block. Nemotron-3 interleaves Mamba-2, attention and MLP/MoE blocks;
Kimi Linear interleaves KDA and full attention three-to-one inside every layer.

`PerfModel` was already operator-level and model-agnostic, so what was missing
was only this: a driver that reads a per-layer type list and dispatches.

Why it matters beyond running two more models
---------------------------------------------
The claim a hybrid accelerator argument rests on is that some operator
contributes a small share of a model's FLOPs and a large share of its decode
latency. Neither `llama_model.py` nor `mamba2_model.py` can express that,
because in a homogeneous stack both shares are 100% by construction. This driver
reports **FLOPs, cycles and HBM bytes per operator**, which is the form that
claim has to be read off.

FLOPs are counted here rather than taken from `PerfModel`, deliberately. They are
a property of the model, the cost model is a property of the machine, and the
whole point is to compare the two -- deriving both from the same code would make
the comparison circular.

Layer-type encodings, both read from the config and both verified against the
model's own source
------------------------------------------------------------------------------
**NemotronH** (`nemotron_h`) carries `hybrid_override_pattern`, one character per
layer, each layer being a *single* block -- a mixer or a feed-forward, not both.
From `configuration_nemotron_h.py`::

    pattern_mapping = {"M": "mamba", "E": "moe", "*": "attention"}   # and "-": "mlp"

**Kimi Linear** (`kimi_linear`) carries `linear_attn_config` with `kda_layers`
and `full_attn_layers`. Those indices are **1-based against a 0-based
`layer_idx`** -- `is_kda_layer` is `(layer_idx + 1) in kda_layers` -- and every
layer additionally has a feed-forward, dense for the first
`first_k_dense_replace` layers and MoE after::

    num_experts is not None and layer_idx >= first_k_dense_replace
                             and layer_idx % moe_layer_freq == 0

Both are checked for completeness when parsed: an index that appears twice, or a
layer that appears in neither list, raises rather than being silently dropped.

Two things this driver does not fix
-----------------------------------
**The `MLEN` disagreement.** `plena_settings.toml` declares
`[ANALYTIC.CONFIG.MLEN] = 2048` and `[TRANSACTIONAL.CONFIG.MLEN] = 64`, with
`BLEN` 128 against 4. Those describe different machines. `--sweep` walks between
them rather than picking one, but the absolute cycle counts still belong to
whichever point is being read.

**And width is free in this model.** `MLEN`/`VLEN` enter cycles only through
instruction counts -- `ceil(work / MLEN)` -- while the latencies are constants:
`V_FMA_VF` bills `VECTOR_MUL_CYCLES` whatever `VLEN` is. So a wider machine here
covers 32x the data per instruction at the same cost, with no area term anywhere.
That makes absolute speedups against width near-tautological and they should not
be quoted as findings. What survives is the *comparison between operators*, which
is priced by the same idealisation on both sides: when one saturates on width and
another does not, that difference is a property of the operators. Every number
this driver is used for should be of that shape -- a ratio, or a turning point,
never an absolute speedup.

**The clock.** Cycles are cycles. `runtime_config.rs` hard-codes `PERIOD` at 1 ns
with no stated basis, so this driver reports no times at all rather than
multiplying by a number it cannot defend.

Usage::

    python hybrid_model.py --model kimi-linear-48b-a3b --model-lib ./doc/Model_Lib \\
        --config ./plena_settings.toml --isa-lib ./customISA_lib.json
    python hybrid_model.py --list-models --model-lib ./doc/Model_Lib
"""

from __future__ import annotations

import argparse
import json
from itertools import pairwise
from dataclasses import dataclass, field
from pathlib import Path

try:  # script-style invocation, matching the sibling drivers
    from perf_model import PerfModel, load_hardware_config_from_toml
except ImportError:  # package-style import
    from analytic_models.performance.perf_model import PerfModel, load_hardware_config_from_toml

#: `hybrid_override_pattern` characters, from NemotronH's own `pattern_mapping`
#: plus the `-` the docstring names ("M: Mamba2, *: Attention, -: MLP").
_NEMOTRON_PATTERN = {"M": "mamba", "*": "attention", "-": "mlp", "E": "moe"}

#: Operators a layer's mixer slot can hold. `None` means the layer has no mixer,
#: which is the normal case for NemotronH's feed-forward-only layers.
MIXERS = ("mamba", "attention", "kda")
FFNS = ("mlp", "moe")


@dataclass(frozen=True)
class LayerSpec:
    """One layer: at most one mixer and at most one feed-forward.

    Both slots are optional because the two families disagree about what a layer
    is. A NemotronH layer is exactly one of the two; a Kimi Linear layer is one
    of each. Modelling that with two optional slots rather than a single "type"
    string is what keeps the per-operator totals honest -- a Kimi layer really
    does run a mixer *and* an FFN, and folding it into one label would lose one
    of them.
    """

    index: int
    mixer: str | None
    ffn: str | None

    def __post_init__(self) -> None:
        if self.mixer is not None and self.mixer not in MIXERS:
            raise ValueError(f"layer {self.index}: unknown mixer {self.mixer!r}")
        if self.ffn is not None and self.ffn not in FFNS:
            raise ValueError(f"layer {self.index}: unknown ffn {self.ffn!r}")
        if self.mixer is None and self.ffn is None:
            raise ValueError(f"layer {self.index} has neither a mixer nor a feed-forward")


def layer_plan(config: dict) -> list[LayerSpec]:
    """One :class:`LayerSpec` per layer, from whichever encoding the config uses.

    Raises on any config it cannot read rather than falling back to a
    homogeneous stack: a hybrid model silently modelled as uniform is exactly
    the failure this driver exists to remove.
    """
    model_type = config.get("model_type")
    num_layers = config["num_hidden_layers"]

    if model_type == "nemotron_h":
        pattern = config["hybrid_override_pattern"]
        if len(pattern) != num_layers:
            raise ValueError(
                f"hybrid_override_pattern is {len(pattern)} characters but "
                f"num_hidden_layers is {num_layers}; one of them is stale"
            )
        unknown = sorted(set(pattern) - set(_NEMOTRON_PATTERN))
        if unknown:
            raise ValueError(f"unknown hybrid_override_pattern characters: {unknown}")
        plan = []
        for i, ch in enumerate(pattern):
            kind = _NEMOTRON_PATTERN[ch]
            if kind in FFNS:
                plan.append(LayerSpec(i, None, kind))
            else:
                plan.append(LayerSpec(i, kind, None))
        return plan

    if model_type == "kimi_linear":
        linear = config.get("linear_attn_config")
        if linear is None:
            raise ValueError("kimi_linear config without linear_attn_config")
        kda = set(linear["kda_layers"])
        full = set(linear["full_attn_layers"])
        both = sorted(kda & full)
        if both:
            raise ValueError(f"layers listed as both KDA and full attention: {both}")
        # 1-based, per `is_kda_layer`: `(layer_idx + 1) in kda_layers`.
        missing = sorted(set(range(1, num_layers + 1)) - kda - full)
        if missing:
            raise ValueError(
                f"layers in neither kda_layers nor full_attn_layers: {missing}. "
                f"Defaulting them to either would silently mis-attribute the mixer"
            )
        first_dense = config.get("first_k_dense_replace", 0)
        freq = config.get("moe_layer_freq", 1) or 1
        has_experts = config.get("num_experts") is not None
        plan = []
        for i in range(num_layers):
            mixer = "kda" if (i + 1) in kda else "attention"
            is_moe = has_experts and i >= first_dense and i % freq == 0
            plan.append(LayerSpec(i, mixer, "moe" if is_moe else "mlp"))
        return plan

    raise ValueError(
        f"model_type {model_type!r} carries no layer-type encoding this driver "
        f"knows. Add it here rather than assuming a uniform stack"
    )


@dataclass
class OperatorCost:
    """Accumulated cost of one operator across every layer that runs it."""

    cycles: int = 0
    bytes_: float = 0.0
    flops: float = 0.0
    layers: int = 0


@dataclass
class HybridModel:
    """Per-operator cost of one hybrid model on the analytic PLENA model."""

    config: dict
    perf: PerfModel
    seq_len: int = 4096
    batch_size: int = 1
    kda_chunk: int = 16
    #: Price a prefetch that takes a row count instead of a whole-block count.
    #: No such instruction exists; this exists to put a number on not having it.
    row_granular_prefetch: bool = False
    plan: list[LayerSpec] = field(init=False)

    def __post_init__(self) -> None:
        self.plan = layer_plan(self.config)
        c = self.config
        self.hidden_size = c["hidden_size"]
        self.num_layers = c["num_hidden_layers"]
        self.vocab_size = c["vocab_size"]

    # -- shapes, per family --------------------------------------------------

    @property
    def _attn(self) -> dict:
        c = self.config
        heads = c["num_attention_heads"]
        kv_heads = c.get("num_key_value_heads", heads)
        head_dim = c.get("head_dim") or c.get("attention_head_dim") or self.hidden_size // heads
        return {"heads": heads, "kv_heads": kv_heads, "head_dim": head_dim}

    @property
    def _kda(self) -> dict:
        linear = self.config["linear_attn_config"]
        return {
            "heads": linear["num_heads"],
            "head_dim": linear["head_dim"],
            "conv_kernel": linear["short_conv_kernel_size"],
        }

    @property
    def _mamba(self) -> dict:
        c = self.config
        heads = c["mamba_num_heads"]
        head_dim = c.get("mamba_head_dim") or (c["hidden_size"] * c.get("expand", 2)) // heads
        return {
            "heads": heads,
            "head_dim": head_dim,
            "state_size": c.get("ssm_state_size") or c["state_size"],
            "n_groups": c.get("n_groups", 1),
            "conv_kernel": c.get("conv_kernel", 4),
            "chunk_size": c.get("chunk_size", 256),
        }

    @property
    def _ffn(self) -> dict:
        c = self.config
        # NemotronH's MLP is `down(act(up(x)))` -- two matmuls, no gate. Kimi's is
        # `down(act(gate(x)) * up(x))` -- three. Read from the source of each,
        # not assumed: a gated count applied to NemotronH overstates every
        # feed-forward layer in the model by 50%.
        gated = c.get("model_type") != "nemotron_h"
        return {
            "intermediate": c["intermediate_size"],
            "moe_intermediate": c.get("moe_intermediate_size", c["intermediate_size"]),
            "num_experts": c.get("num_experts") or c.get("n_routed_experts") or 0,
            "top_k": c.get("num_experts_per_token") or c.get("num_experts_per_tok") or 0,
            "shared": c.get("num_shared_experts", 0) or 0,
            "matmuls": 3 if gated else 2,
        }

    # -- FLOPs, counted independently of the cost model ----------------------

    @staticmethod
    def _gemm_flops(m: int, n: int, k: int) -> float:
        """Multiply-accumulate counted as two flops, the usual convention."""
        return 2.0 * m * n * k

    def _mixer_flops(self, kind: str, tokens: int, kv_len: int) -> float:
        h = self.hidden_size
        if kind == "attention":
            a = self._attn
            q_out = a["heads"] * a["head_dim"]
            kv_out = a["kv_heads"] * a["head_dim"]
            flops = self._gemm_flops(tokens, q_out + 2 * kv_out, h)  # qkv
            flops += self._gemm_flops(tokens, h, q_out)  # o proj
            # scores and the value read, both over the whole visible context
            flops += 2 * self._gemm_flops(tokens, kv_len, q_out)
            return flops
        if kind == "kda":
            k = self._kda
            width = k["heads"] * k["head_dim"]
            flops = self._gemm_flops(tokens, 3 * width, h)  # q, k, v
            flops += self._gemm_flops(tokens, h, width)  # out proj
            flops += 2.0 * tokens * width * k["conv_kernel"]  # causal conv
            # The recurrence: predict, error and update each touch the whole
            # [key, value] state once per token.
            flops += 3 * 2.0 * tokens * k["heads"] * k["head_dim"] * k["head_dim"]
            return flops
        if kind == "mamba":
            m = self._mamba
            inner = m["heads"] * m["head_dim"]
            proj_out = 2 * inner + 2 * m["n_groups"] * m["state_size"] + m["heads"]
            flops = self._gemm_flops(tokens, proj_out, h)  # in_proj
            flops += self._gemm_flops(tokens, h, inner)  # out_proj
            conv_dim = inner + 2 * m["n_groups"] * m["state_size"]
            flops += 2.0 * tokens * conv_dim * m["conv_kernel"]
            flops += 3 * 2.0 * tokens * m["heads"] * m["head_dim"] * m["state_size"]
            return flops
        raise ValueError(f"no FLOP model for mixer {kind!r}")

    def _ffn_flops(self, kind: str, tokens: int) -> float:
        f = self._ffn
        h = self.hidden_size
        if kind == "mlp":
            return f["matmuls"] * self._gemm_flops(tokens, f["intermediate"], h)
        active = f["top_k"] + f["shared"]
        flops = active * f["matmuls"] * self._gemm_flops(tokens, f["moe_intermediate"], h)
        flops += self._gemm_flops(tokens, f["num_experts"], h)  # router
        return flops

    # -- dispatch ------------------------------------------------------------

    def _stage(self, fn, *args, **kwargs) -> tuple[int, float]:
        """Run one PerfModel stage and return its (cycles, HBM bytes).

        `PerfModel` accumulates traffic globally, so the counters are reset
        around each call. That is the only way to get a per-operator byte count
        out of it without changing its interface.
        """
        before = self.perf.traffic_bytes
        cycles = fn(*args, **kwargs)
        return cycles, self.perf.traffic_bytes - before

    def _mixer_cost(self, kind: str, mode: str, tokens: int, kv_len: int) -> tuple[int, float]:
        p, b = self.perf, self.batch_size
        if kind == "attention":
            a = self._attn
            c1, b1 = self._stage(
                p.projection, self.hidden_size, a["heads"], a["kv_heads"], a["head_dim"], tokens, b, mode
            )
            c2, b2 = self._stage(p.flash_attention, a["heads"], a["kv_heads"], a["head_dim"], tokens, kv_len, b, mode)
            return c1 + c2, b1 + b2
        if kind == "kda":
            k = self._kda
            width = k["heads"] * k["head_dim"]
            cycles, byts = self._stage(
                p.projection, self.hidden_size, 3 * k["heads"], k["heads"], k["head_dim"], tokens, b, mode
            )
            c, bb = self._stage(p.causal_conv1d, width, k["conv_kernel"], tokens, b, mode)
            cycles, byts = cycles + c, byts + bb
            if mode == "prefill":
                c, bb = self._stage(
                    p.kda_chunk_prefill,
                    k["heads"],
                    k["head_dim"],
                    k["head_dim"],
                    self.kda_chunk,
                    tokens,
                    b,
                    row_granular_prefetch=self.row_granular_prefetch,
                )
            else:
                c, bb = self._stage(p.kda_recurrence_decode, k["heads"], k["head_dim"], k["head_dim"], b)
            cycles, byts = cycles + c, byts + bb
            c, bb = self._stage(p.gated_rms_norm, width, tokens, b, mode)
            return cycles + c, byts + bb
        if kind == "mamba":
            m = self._mamba
            inner = m["heads"] * m["head_dim"]
            conv_dim = inner + 2 * m["n_groups"] * m["state_size"]
            cycles, byts = self._stage(
                p.linear,
                self.hidden_size,
                2 * inner + 2 * m["n_groups"] * m["state_size"] + m["heads"],
                tokens,
                b,
                mode,
            )
            c, bb = self._stage(p.causal_conv1d, conv_dim, m["conv_kernel"], tokens, b, mode)
            cycles, byts = cycles + c, byts + bb
            c, bb = self._stage(p.dt_activation, m["heads"], tokens, b, mode)
            cycles, byts = cycles + c, byts + bb
            if mode == "prefill":
                c, bb = self._stage(
                    p.ssd_chunk_scan,
                    m["heads"],
                    m["head_dim"],
                    m["state_size"],
                    m["n_groups"],
                    m["chunk_size"],
                    tokens,
                    b,
                )
            else:
                c, bb = self._stage(
                    p.ssd_recurrence_decode, m["heads"], m["head_dim"], m["state_size"], m["n_groups"], b
                )
            cycles, byts = cycles + c, byts + bb
            c, bb = self._stage(p.gated_rms_norm, inner, tokens, b, mode)
            cycles, byts = cycles + c, byts + bb
            c, bb = self._stage(p.linear, inner, self.hidden_size, tokens, b, mode)
            return cycles + c, byts + bb
        raise ValueError(f"no cost model for mixer {kind!r}")

    def _ffn_cost(self, kind: str, mode: str, tokens: int) -> tuple[int, float]:
        f, p, b = self._ffn, self.perf, self.batch_size
        if kind == "mlp":
            return self._stage(p.feed_forward, self.hidden_size, f["intermediate"], tokens, b, mode)
        return self._stage(
            p.mlp_moe,
            self.hidden_size,
            tokens,
            b,
            f["num_experts"],
            f["top_k"] + f["shared"],
            f["moe_intermediate"],
            mode,
        )

    def per_operator(self, mode: str = "decode") -> dict[str, OperatorCost]:
        """FLOPs, cycles and HBM bytes for every operator, summed over layers."""
        if mode not in ("prefill", "decode"):
            raise ValueError(f"mode must be prefill or decode, got {mode!r}")
        tokens = self.seq_len if mode == "prefill" else 1
        # In decode the attention read spans the whole context; the recurrent
        # mixers do not, which is the asymmetry the whole comparison is about.
        kv_len = self.seq_len

        out: dict[str, OperatorCost] = {}

        def add(name: str, cycles: int, byts: float, flops: float) -> None:
            e = out.setdefault(name, OperatorCost())
            e.cycles += cycles
            e.bytes_ += byts
            e.flops += flops
            e.layers += 1

        for layer in self.plan:
            if layer.mixer is not None:
                c, bb = self._mixer_cost(layer.mixer, mode, tokens, kv_len)
                add(layer.mixer, c, bb, self._mixer_flops(layer.mixer, tokens * self.batch_size, kv_len))
            if layer.ffn is not None:
                c, bb = self._ffn_cost(layer.ffn, mode, tokens)
                add(layer.ffn, c, bb, self._ffn_flops(layer.ffn, tokens * self.batch_size))
            # Two norms and two residuals per layer whichever slots it fills.
            c, bb = self._stage(self.perf.rms_layer, self.hidden_size, tokens, self.batch_size, mode)
            c2, bb2 = self._stage(self.perf.residual, self.hidden_size, tokens, self.batch_size, mode)
            add("norm+residual", c + c2, bb + bb2, 0.0)

        c, bb = self._stage(self.perf.lm_head, self.hidden_size, self.vocab_size, self.batch_size)
        add("lm_head", c, bb, self._gemm_flops(self.batch_size, self.vocab_size, self.hidden_size))
        return out

    # -- reporting -----------------------------------------------------------

    def layer_census(self) -> dict[str, int]:
        census: dict[str, int] = {}
        for layer in self.plan:
            for slot in (layer.mixer, layer.ffn):
                if slot is not None:
                    census[slot] = census.get(slot, 0) + 1
        return census

    def report(self, mode: str = "decode") -> str:
        costs = self.per_operator(mode)
        total_c = sum(e.cycles for e in costs.values()) or 1
        total_f = sum(e.flops for e in costs.values()) or 1.0
        total_b = sum(e.bytes_ for e in costs.values()) or 1.0

        lines = [
            f"{'operator':<16}{'layers':>7}{'GFLOP':>12}{'%FLOP':>8}"
            f"{'Mcycles':>12}{'%cycles':>9}{'MB':>12}{'%bytes':>8}"
        ]
        lines.append("-" * 85)
        for name, e in sorted(costs.items(), key=lambda kv: -kv[1].cycles):
            lines.append(
                f"{name:<16}{e.layers:>7}{e.flops / 1e9:>12.3f}{100 * e.flops / total_f:>7.1f}%"
                f"{e.cycles / 1e6:>12.3f}{100 * e.cycles / total_c:>8.1f}%"
                f"{e.bytes_ / 1e6:>12.3f}{100 * e.bytes_ / total_b:>7.1f}%"
            )
        lines.append("-" * 85)
        lines.append(
            f"{'total':<16}{'':>7}{total_f / 1e9:>12.3f}{100.0:>7.1f}%"
            f"{total_c / 1e6:>12.3f}{100.0:>8.1f}%{total_b / 1e6:>12.3f}{100.0:>7.1f}%"
        )
        return "\n".join(lines)


#: Coherent machine points between the two shapes `plena_settings.toml` declares.
#:
#: `MLEN`/`BLEN`/`HLEN`/`VLEN` are not independent -- `MLEN % BLEN == 0`,
#: `VLEN >= BLEN`, and `flash_attention` divides by `MLEN // HLEN`, which is a
#: crash rather than a wrong answer when it rounds to zero. Overriding one field
#: and leaving the rest is what makes that happen, so the sweep moves them
#: together and the endpoints are the two declared configurations:
#: `TRANSACTIONAL` at 64/4/16/64 and `ANALYTIC` at 2048/128/128/2048.
MACHINE_POINTS: tuple[tuple[int, int, int, int], ...] = (
    (64, 4, 16, 64),
    (256, 16, 32, 256),
    (512, 32, 64, 512),
    (1024, 64, 128, 1024),
    (2048, 128, 128, 2048),
)

#: HBM bandwidth in bytes per cycle. The shipped `HBM_WIDTH` of 512 is one point
#: on this axis, not a measurement -- `perf_model.py` derives bandwidth from it
#: as "one HBM row per cycle", which is an assumption about the interface rather
#: than a number anyone has read off silicon.
BANDWIDTH_POINTS: tuple[int, ...] = (64, 128, 256, 512, 1024, 2048, 4096)


def sweep(
    config: dict,
    isa_lib: Path,
    settings: Path,
    *,
    seq_len: int,
    batch_size: int,
    kda_chunk: int,
    mode: str,
    operator: str,
    row_granular_prefetch: bool = False,
) -> list[dict]:
    """`operator`'s FLOP share against its cycle share, over machine and bandwidth.

    The ratio of the two is the quantity a hybrid accelerator argument turns on:
    a value near 1 says the operator costs what its arithmetic says it should,
    and a value far from 1 says the machine is spending time somewhere the FLOP
    count cannot see.

    Both axes are swept because neither is known. `MLEN` is declared twice in
    `plena_settings.toml` with a 32x disagreement, and bandwidth is derived from
    a row width rather than measured. Reporting a single ratio at one arbitrary
    point on that plane would be reporting the plane's shape as if it were a
    property of the design.
    """
    rows = []
    for mlen, blen, hlen, vlen in MACHINE_POINTS:
        for bw in BANDWIDTH_POINTS:
            hw = load_hardware_config_from_toml(str(settings))
            hw.MLEN, hw.BLEN, hw.HLEN, hw.VLEN = mlen, blen, hlen, vlen
            hw.HBM_BANDWIDTH_BYTES_PER_CYCLE = float(bw)
            model = HybridModel(
                config=config,
                perf=PerfModel(hw, str(isa_lib)),
                seq_len=seq_len,
                batch_size=batch_size,
                kda_chunk=kda_chunk,
                row_granular_prefetch=row_granular_prefetch,
            )
            costs = model.per_operator(mode)
            if operator not in costs:
                raise SystemExit(f"{operator!r} is not an operator of this model; it runs {sorted(costs)}")
            total_f = sum(e.flops for e in costs.values()) or 1.0
            total_c = sum(e.cycles for e in costs.values()) or 1
            e = costs[operator]
            traffic = model.perf.traffic_summary()
            mem_bound = traffic["memory_only_cycles"] / max(1, traffic["compute_only_cycles"])
            rows.append(
                {
                    "mlen": mlen,
                    "bandwidth": bw,
                    "flop_share": e.flops / total_f,
                    "cycle_share": e.cycles / total_c,
                    "ratio": (e.cycles / total_c) / max(1e-12, e.flops / total_f),
                    "memory_over_compute": mem_bound,
                    "total_cycles": total_c,
                    "operator_cycles": e.cycles,
                }
            )
    return rows


def format_sweep(rows: list[dict], operator: str) -> str:
    """The ratio as a grid, plus how memory-bound the whole model is at each point.

    `mem/compute` is the ratio of the two sides of the roofline summed over every
    stage. Above 1 the model is memory-bound overall, and on-chip width stops
    changing anything -- which is the reading that decides whether a wider
    machine is worth anything at all at that bandwidth.
    """
    bws = sorted({r["bandwidth"] for r in rows})
    mlens = sorted({r["mlen"] for r in rows})
    grid = {(r["mlen"], r["bandwidth"]): r for r in rows}

    out = [f"cycle share / FLOP share for `{operator}`  (1.00 = costs what its arithmetic says)", ""]
    out.append("  MLEN " + "".join(f"{b:>9}" for b in bws) + "   B/cycle")
    out.append("  " + "-" * (5 + 9 * len(bws)))
    for m in mlens:
        out.append(f"{m:>6} " + "".join(f"{grid[(m, b)]['ratio']:>9.2f}" for b in bws))
    out.append("")
    out.append("whole-model memory cycles / compute cycles  (>1 = memory-bound overall)")
    out.append("")
    out.append("  MLEN " + "".join(f"{b:>9}" for b in bws) + "   B/cycle")
    out.append("  " + "-" * (5 + 9 * len(bws)))
    for m in mlens:
        out.append(f"{m:>6} " + "".join(f"{grid[(m, b)]['memory_over_compute']:>9.2f}" for b in bws))
    return "\n".join(out)


def format_width_return(rows: list[dict], operator: str) -> str:
    """What another doubling of machine width is worth, and where it stops being worth anything.

    The ratio grid says an operator's *share* of the time gets worse on a wider
    machine. It does not say whether the machine is faster, which is the question
    a width decision actually turns on -- a wider machine helps the dense
    operators even while it wastes lanes on the recurrent one, and the two have to
    be weighed against each other rather than read separately.

    So this reports total decode cycles against width, and the marginal return of
    each doubling. There is no area model here, so "wider is better" is trivially
    true wherever it still helps at all; what is not trivial, and what this
    reports, is the width past which it stops helping. Beyond that point the extra
    lanes are spent on an operand that cannot fill them.
    """
    bws = sorted({r["bandwidth"] for r in rows})
    mlens = sorted({r["mlen"] for r in rows})
    grid = {(r["mlen"], r["bandwidth"]): r for r in rows}

    out = ["total decode cycles, in millions", ""]
    out.append("  MLEN " + "".join(f"{b:>9}" for b in bws) + "   B/cycle")
    out.append("  " + "-" * (5 + 9 * len(bws)))
    for m in mlens:
        out.append(f"{m:>6} " + "".join(f"{grid[(m, b)]['total_cycles'] / 1e6:>9.2f}" for b in bws))

    out += ["", "marginal return of doubling MLEN: % of total cycles removed", ""]
    out.append("  MLEN " + "".join(f"{b:>9}" for b in bws) + "   B/cycle")
    out.append("  " + "-" * (5 + 9 * len(bws)))
    for prev, m in pairwise(mlens):
        cells = []
        for b in bws:
            before = grid[(prev, b)]["total_cycles"]
            after = grid[(m, b)]["total_cycles"]
            cells.append(f"{100 * (before - after) / max(1, before):>8.1f}%")
        out.append(f"{prev:>4}->{m:<1}" + "".join(cells))

    out += ["", f"share of that width spent on `{operator}`, which cannot use it", ""]
    out.append("  MLEN " + "".join(f"{b:>9}" for b in bws) + "   B/cycle")
    out.append("  " + "-" * (5 + 9 * len(bws)))
    for m in mlens:
        out.append(f"{m:>6} " + "".join(f"{100 * grid[(m, b)]['cycle_share']:>8.1f}%" for b in bws))
    return "\n".join(out)


def load_model_config(name: str, model_lib: Path) -> dict:
    path = model_lib / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"no config {path}; --list-models shows what is available")
    return json.loads(path.read_text())


def list_hybrid_models(model_lib: Path) -> list[tuple[str, str, dict[str, int]]]:
    """Every config in the library this driver can read, with its layer census."""
    found = []
    for path in sorted(model_lib.glob("*.json")):
        try:
            config = json.loads(path.read_text())
            plan = layer_plan(config)
        except Exception:
            continue
        census: dict[str, int] = {}
        for layer in plan:
            for slot in (layer.mixer, layer.ffn):
                if slot is not None:
                    census[slot] = census.get(slot, 0) + 1
        found.append((path.stem, config.get("model_type", "?"), census))
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="config stem in --model-lib, e.g. kimi-linear-48b-a3b")
    ap.add_argument("--model-lib", type=Path, default=Path("doc/Model_Lib"))
    ap.add_argument("--config", type=Path, default=Path("plena_settings.toml"))
    ap.add_argument("--isa-lib", type=Path, default=Path("customISA_lib.json"))
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--kda-chunk", type=int, default=16, help="bf16 range bounds this at 17")
    ap.add_argument("--mode", choices=("prefill", "decode", "both"), default="both")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument(
        "--sweep",
        metavar="OPERATOR",
        help="sweep MLEN and HBM bandwidth, reporting OPERATOR's cycle share "
        "against its FLOP share. Neither axis is known: MLEN is declared twice "
        "with a 32x disagreement and bandwidth is derived from a row width, so "
        "a single point would report the plane rather than the design.",
    )
    ap.add_argument(
        "--row-granular-prefetch",
        action="store_true",
        help="price an instruction set whose prefetch takes a row count rather "
        "than a whole-block count. No such instruction exists; this exists to "
        "put a number on not having it.",
    )
    args = ap.parse_args()

    if args.list_models:
        rows = list_hybrid_models(args.model_lib)
        if not rows:
            raise SystemExit(f"no hybrid configs found under {args.model_lib}")
        for stem, model_type, census in rows:
            census_str = ", ".join(f"{k}x{v}" for k, v in sorted(census.items()))
            print(f"  {stem:<34}{model_type:<16}{census_str}")
        return

    if not args.model:
        raise SystemExit("--model is required (or --list-models)")

    config = load_model_config(args.model, args.model_lib)
    hw = load_hardware_config_from_toml(str(args.config))
    perf = PerfModel(hw, str(args.isa_lib))
    model = HybridModel(
        config=config,
        perf=perf,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        kda_chunk=args.kda_chunk,
        row_granular_prefetch=args.row_granular_prefetch,
    )

    census = ", ".join(f"{k} x{v}" for k, v in sorted(model.layer_census().items()))
    print(f"\n{args.model}  ({config.get('model_type')})")
    print(f"  hidden {model.hidden_size}, {model.num_layers} layers: {census}")
    print(f"  seq_len {args.seq_len}, batch {args.batch_size}, MLEN {perf.mlen}, VLEN {perf.vlen}")
    print(
        "  NOTE cycles are uncalibrated and come from the ANALYTIC config, whose "
        f"MLEN ({perf.mlen}) disagrees with the TRANSACTIONAL one. Read the ratios."
    )
    if args.sweep:
        mode = "decode" if args.mode == "both" else args.mode
        print(f"\n--- sweep, {mode} ---\n")
        rows = sweep(
            config,
            args.isa_lib,
            args.config,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            kda_chunk=args.kda_chunk,
            mode=mode,
            operator=args.sweep,
            row_granular_prefetch=args.row_granular_prefetch,
        )
        print(format_sweep(rows, args.sweep))
        print()
        print(format_width_return(rows, args.sweep))
        print()
        return

    for mode in ("prefill", "decode") if args.mode == "both" else (args.mode,):
        print(f"\n--- {mode} ---")
        print(model.report(mode))
    print()


if __name__ == "__main__":
    main()
