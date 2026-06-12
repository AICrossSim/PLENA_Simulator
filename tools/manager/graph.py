"""Compute-graph driver (MANAGER_DESIGN.md §8).

A lightweight declarative graph (dict/JSON) describes a kernel chain:
  * tensors: name -> {shape, role}
  * nodes:   one kernel each, in/out mapping kernel-buffer-name -> tensor-name

The manager reads it and auto-derives everything: HBM placement, topological
execution order (= bin-relay order), and each node's tensor_map. Edges are
shared tensor names (producer.out["Y_hbm"] == consumer.in["X_hbm"] both name
the same tensor -> same address -> data relays through the bin).

This replaces hand-written KernelStep lists. The graph is the single source of
truth; KernelStep becomes an internal product.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np

from .tensor import Role


_ROLE = {
    "io": Role.IO,
    "weight": Role.WEIGHT,
    "activation": Role.ACTIVATION,
    "scratch": Role.SCRATCH,
}


@dataclass
class _Node:
    name: str
    kernel: str
    kwargs: Mapping[str, object]
    ins: Mapping[str, str]    # buffer name -> tensor name
    outs: Mapping[str, str]


@dataclass
class ComputeGraph:
    tensors: Dict[str, dict]          # name -> {shape, role}
    nodes: List[_Node]

    @classmethod
    def load(cls, graph) -> "ComputeGraph":
        if isinstance(graph, (str, Path)):
            graph = json.loads(Path(graph).read_text())
        tensors = dict(graph["tensors"])
        nodes = [
            _Node(
                name=n["name"],
                kernel=n["kernel"],
                kwargs=dict(n.get("kwargs", {})),
                ins=dict(n.get("in", {})),
                outs=dict(n.get("out", {})),
            )
            for n in graph["nodes"]
        ]
        return cls(tensors=tensors, nodes=nodes)

    # ---- validation + ordering ----
    def _topo_order(self, allow_shared_io: bool = False) -> List[_Node]:
        """Order nodes so every tensor is written by its producer before any
        consumer reads it. Producer of a tensor = the node that lists it in
        out; consumers list it in in. io/weight tensors are pre-written in the
        prep phase (no producer node) and are available from the start.

        ``allow_shared_io`` (compile-only): in compile-only mode there is no
        real dataflow, so several kernels may write the same pre-available
        io/weight pool as a scratch output. The single-producer uniqueness
        check is then relaxed for io/weight pools only. In a real run/validate
        this stays False, so every output (including io) is still enforced
        unique and real data races are caught."""
        _shared = set()
        if allow_shared_io:
            _shared = {name for name, spec in self.tensors.items()
                       if spec.get("role") in ("io", "weight")}
        produced_by: Dict[str, str] = {}
        for nd in self.nodes:
            for t in nd.outs.values():
                if t in _shared:
                    continue
                if t in produced_by:
                    raise ValueError(
                        f"tensor {t!r} written by two nodes "
                        f"({produced_by[t]}, {nd.name}); not supported")
                produced_by[t] = nd.name

        # pre-available = io/weight (and anything with no producer node)
        available = {
            name for name, spec in self.tensors.items()
            if spec.get("role") in ("io", "weight")
        }
        pending = list(self.nodes)
        ordered: List[_Node] = []
        while pending:
            progressed = False
            for nd in list(pending):
                needs = set(nd.ins.values())
                if needs <= available:
                    ordered.append(nd)
                    available |= set(nd.outs.values())
                    pending.remove(nd)
                    progressed = True
            if not progressed:
                missing = {nd.name: sorted(set(nd.ins.values()) - available)
                          for nd in pending}
                raise ValueError(
                    f"compute graph has a cycle or unmet inputs; "
                    f"unresolved nodes -> missing tensors: {missing}")
        return ordered

    # ---- materialize into manager + steps ----
    def to_steps(self, mgr, *, data: Optional[Mapping[str, object]] = None,
                compare: Optional[Mapping[str, np.ndarray]] = None):
        """Place every tensor on the manager's HBM layout and return the ordered
        KernelStep list.

        ``data``: {tensor_name -> torch.Tensor} for io/weight tensors (their
        actual values, written into the bin in the prep phase). activation/
        scratch get data=None (filled at run time by their producer).
        ``compare``: {tensor_name -> golden} attached to the producing node.
        """
        from .pipeline import KernelStep   # local import (avoid cycle)
        from .binio import packed_byte_size

        data = data or {}
        compare = compare or {}

        # 1a. place IO + ACTIVATION + SCRATCH in the data-flow region (bump;
        #     distinct addresses; preserved across kernels for the bin relay).
        for name, spec in self.tensors.items():
            role = _ROLE[spec["role"]]
            if role is Role.WEIGHT:
                continue
            payload = data.get(name) if role is Role.IO else None
            mgr.place(name, spec["shape"], role, data=payload)

        # 1b. weights go in ONE shared reused region above the data-flow region.
        #     Each node lays its weights out from weight_base (reset per node),
        #     so different kernels' weights overwrite the same bytes
        #     ([[feedback_weights_just_in_time]]). Region size = the largest
        #     single node's weight footprint.
        ordered = self._topo_order(allow_shared_io=getattr(mgr, "compile_only", False))
        weight_base = max(mgr.layout.total_bytes(), mgr.layout.base)

        def _is_weight(tname: str) -> bool:
            return _ROLE[self.tensors[tname]["role"]] is Role.WEIGHT

        # Place each DISTINCT weight tensor once, bump-allocated in the weight
        # region (above all activations, so weights never clobber the data
        # flow). A weight shared by several nodes is placed once; each
        # consuming node lists it and writes it just-in-time before running
        # (same address, same value — harmless redundant write). This is the
        # user's "load the weights this kernel needs, don't overwrite other
        # important values" rule; we don't reuse weight bytes across nodes
        # (HBM is plentiful in the sim) which keeps it simple and correct.
        cur = weight_base
        for nd in ordered:
            for t in nd.ins.values():
                if _is_weight(t) and t not in mgr.layout.tensors:
                    spec = self.tensors[t]
                    mgr.layout.pin(t, spec["shape"], Role.WEIGHT, cur, data=data.get(t))
                    cur += packed_byte_size(mgr.layout.tensors[t].num_elements, mgr.s)

        node_weight_names = {
            nd.name: tuple(t for t in nd.ins.values() if _is_weight(t))
            for nd in ordered
        }

        # 2. per-node tensor_map + compares + just-in-time weight list
        steps = []
        for nd in ordered:
            tmap = {**nd.ins, **nd.outs}
            node_compare = {t: compare[t] for t in nd.outs.values() if t in compare}
            steps.append(KernelStep(
                kernel=nd.kernel,
                asm_name=nd.name,
                kernel_kwargs=nd.kwargs,
                tensor_map=tmap,
                weight_tensors=node_weight_names[nd.name],
                compare=node_compare or None,
            ))
        return steps
