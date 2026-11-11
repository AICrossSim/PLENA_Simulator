"""Cycle-level simulator of one MoE decode layer on one or two systolic arrays that share HBM, SRAM and a weight port.

simulate() runs layer windows (the routed experts of B tokens) and returns one int64 row of counters per window;
counters() names them. trace() runs one window with a per-cycle record and the dispatch decision log.
"""

import ctypes
import hashlib
import os
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent / "src"
FLAGS = ("-O3", "-std=c++17", "-fopenmp", "-shared", "-fPIC")
MAX_CONTEXTS, MAX_ARRAYS, NCLASS = 4, 2, 3
CLASSES = ("routed", "shared", "router")

CONFIG_FIELDS = (
    "batch",
    "topk",
    "experts",
    "hidden",
    "routed_width",
    "shared_width",
    "shared_gate",
    "contexts",
    "shared_array",
    "ctx_per_core",
    "core_caps",
    "W",
    "X",
    "Y",
    "H",
    "capacity",
    "latency",
    "hlatency",
    "control_cycles",
    "bits",
    "group",
    "converter_rate",
    "converter_latency",
    "port_mode",
    "kappa_milli",
    "policy",
    "p14_steal",
    "lookahead",
    "slots",
    "stage_slots",
    "vslots",
    "ybanks",
    "share",
    "share2",
    "token_locks",
    "drain_inner",
    "drain_last",
    "tile_bubble",
    "tail_slot",
)
SCALAR_FIELDS = (
    "status",
    "cycles",
    "startup",
    "rows",
    "peak_sram",
    "hbm_idle_cycles",
    "next_reservations",
    "prefetch_same_task_bytes",
    "prefetch_next_task_bytes",
    "band_plan_cycle",
    "band_planned_makespan",
    "band_steals",
    "band_plan_fallback",
    "delivery_limited_cycles",
    "port_bytes",
    "port_saturated_cycles",
    "tail_fallbacks",
    "multi_group_tasks",
    "weight_bytes",
)
CTX_FIELDS = (
    "core",
    "sram_bytes",
    "enabled",
    "input_stall",
    "output_stall",
    "merge_stall",
    "commit_wait",
    "idle",
    "share_stall",
    "hbm_service",
    "delivery_wait",
    "delivered_weights",
    "wait_hbm",
    "wait_wport",
    "wait_x",
    "wait_latency",
    "wait_decode",
    "finish",
    "reserved_bytes",
    "drain_wait",
    "drain_waits",
    "independent_phases",
    "drain_wait_last",
    "tail_handoffs",
    "tail_wait",
    "tail_busy",
    "tail_blocked",
)
ARRAY_FIELDS = ("lanes", "drain_idle")
CLASS_FIELDS = (
    "tasks",
    "unique_weights",
    "decoded",
    "hbm_bytes",
    "wport_bytes",
    "expanded_write_bytes",
    "local_read_bytes",
    "busy",
    "matrix_busy",
    "useful_macs",
    "reread_bytes",
)
TRACE_FIELDS = ("cycle", "port_need_reread", "port_need_first", "port_used", "port_first_budget")
TRACE_CTX_FIELDS = (
    "state",
    "task",
    "phase",
    "prefetch_task",
    "prefetch_phase",
    "hbm_bytes",
    "hbm_current",
    "hbm_prefetch",
    "port_first_bytes",
    "port_reread_weights",
    "decoded",
    "x_bytes",
    "y_bytes",
)
# Totals over the contexts that counters() adds under the field name without the ctx_ prefix.
TOTALS = (
    "hbm_service",
    "wait_hbm",
    "wait_wport",
    "wait_x",
    "wait_latency",
    "wait_decode",
    "drain_wait",
    "drain_waits",
    "independent_phases",
    "drain_wait_last",
    "tail_handoffs",
    "tail_wait",
    "tail_busy",
    "tail_blocked",
)
STATES = ("idle", "compute", "wait_hbm", "wait_redelivery", "wait_port", "wait_other", "stall")
NOUT = (
    len(SCALAR_FIELDS)
    + MAX_CONTEXTS * len(CTX_FIELDS)
    + MAX_ARRAYS * len(ARRAY_FIELDS)
    + MAX_ARRAYS * NCLASS * len(CLASS_FIELDS)
)
NTRACE = len(TRACE_FIELDS) + MAX_CONTEXTS * len(TRACE_CTX_FIELDS)

# Systolic timing modes -> (drain_inner, drain_last, tile_bubble, tail_slot). drain_inner / drain_last: a phase that reads
# a matrix phase's outputs waits for the array's drain L(n), inside a task / after its last matrix phase; tile_bubble:
# one idle cycle between tiles (broadcast arrays); tail_slot: a task's trailing phases run in a one-deep slot of the
# context, which starts its next task at once (the tail pays the drain after the last matrix phase).
SYSTOLIC = {
    0: (0, 0, 1, 0),
    1: (1, 1, 1, 0),
    2: (1, 1, 0, 0),
    3: (1, 1, 0, 1),
    4: (0, 0, 0, 0),
    5: (0, 0, 0, 1),
    6: (1, 0, 1, 0),
    7: (0, 1, 1, 0),
}
DRAIN_FORMS = ("linear", "tree", "express", "table")

# Settings of earlier model variants that this model fixes; stored settings must hold these values.
FIXED = dict(
    packet=1,
    overlap=1,
    gate_split=0,
    execution_mode=1,
    retention=3,
    ndiv=1,
    kfactor=1,
    mac_ii=1,
    tree_ii=1,
    kappa_scope=0,
    ctx_map=(),
    ctx_share=(),
    core_modes=(),
    core_caps_per_core=(),
    expert_bits=(),
    core_dp_bits=(),
)

ERRORS = {
    -1: "needs 2 or 4 contexts and more SRAM than the token buffers",
    -2: "contexts do not map onto one or two arrays",
    -3: "contexts of one array have different shapes",
    -4: "unknown policy, or policy 2 without exactly two contexts",
    -5: "more than 2**20 multipliers",
    -6: "expert ids out of range",
    -7: "decision log overflow",
}


class CapacityError(ValueError):
    """A task fits no context: the chip does not fit its SRAM."""


@dataclass(frozen=True)
class Environment:
    """Chip and runtime settings. Bandwidths in bytes per cycle (1 GHz), sizes in bytes."""

    W: int = 512  # weight port
    X: int = 4096  # activation reads
    Y: int = 4096  # output writes
    capacity: int = 4 * 2**20  # SRAM
    latency: int = 4  # SRAM read latency (cycles)
    slots: int = 2  # weight / activation FIFO depth of a matrix phase
    lookahead: int = 8  # policy 2: dispatch window (tasks)
    token_locks: bool = True  # combines into one token are serialised
    control_cycles: int = 8  # per task
    vslots: int = 4  # FIFO depth of a vector phase
    ybanks: int = 2  # output buffer banks
    share: int = 500  # SRAM permille of context 0 (four contexts: of contexts 0 and 1)
    policy: int = 13  # 2 regret list scheduling, 13 pincer (hot end / cold end), 14 band dispatch
    H: int = 174720  # HBM
    hlatency: int = 4  # HBM latency (cycles)
    stage_slots: int = 2  # HBM landing slots per context
    bits: int = 16  # stored weight bits (the router stays BF16)
    group: int = 128  # quantization group along K (one FP16 scale per 16 rows x group)
    converter_rate: int = 4096  # weights decoded per cycle
    converter_latency: int = 4
    share2: int = 500  # four contexts: SRAM permille of the even context of each pair
    ctx_per_core: int = 2  # shared_array: contexts per array
    core_caps: int = 0  # contexts of a shared array share its weight-port and converter intake
    kappa_reread: float = 0.0  # port_mode 0: re-delivery budget in units of the HBM weight rate (0: unlimited)
    port_mode: int = 0  # 1: one W-byte port carries first passes and re-deliveries (half-port rule)
    p14_steal: int = 1  # policy 14: 2 steals only tasks the thief finishes before the band's owner; 0, 1 always
    shared_array: bool = False  # context c runs on array c // ctx_per_core; else every context has its own array
    systolic: int = 0  # timing mode of the arrays' outputs, see SYSTOLIC
    drain_form: str = "linear"  # L(n): linear n - 1; tree / express over segments of drain_segment columns; table
    drain_segment: int = 16
    drain_table: tuple = ()  # ((n, L), ...) for drain_form 'table'

    def __post_init__(self):
        object.__setattr__(self, "drain_table", tuple(tuple(int(v) for v in x) for x in self.drain_table))

    @classmethod
    def from_dict(cls, d):
        """Environment from stored settings, which may carry FIXED fields at their fixed values."""
        d = dict(d)
        for k, v in FIXED.items():
            if k in d and (tuple(d[k]) if isinstance(d[k], (list, tuple)) else d[k]) != v:
                raise ValueError(f"{k}={d[k]!r}: this model implements only {k}={v!r}")
            d.pop(k, None)
        return cls(**d)


def core_map(env, contexts):
    """Array of each context."""
    return [c // env.ctx_per_core if env.shared_array else c for c in range(contexts)]


def drain_latency(n, env):
    """Cycles from a phase's last output row at column 0 until column n - 1 has it, under env's drain form."""
    s, form = env.drain_segment, env.drain_form
    if form == "table":
        table = dict(env.drain_table)
        if n not in table:
            raise ValueError(f"drain_table has no entry for n = {n}")
        return table[n]
    if form == "linear" or n <= s:
        return n - 1
    segments = -(-n // s)
    return s - 1 + ((segments - 1).bit_length() if form == "tree" else segments)


def library(build_dir=None):
    """The compiled simulator, built on first use into build_dir (default: $SIM_BUILD_DIR or ~/.cache)."""
    d = build_dir or os.environ.get("SIM_BUILD_DIR") or Path.home() / ".cache" / "complementary_arrays" / "sim"
    return _load(str(Path(d).resolve()))


@lru_cache(None)
def _load(build_dir):
    sources = sorted(SRC.glob("*.cpp")) + sorted(SRC.glob("*.hpp"))
    h = hashlib.sha256(" ".join(FLAGS).encode())
    for p in sources:
        h.update(p.name.encode() + b"\0" + p.read_bytes())
    so = Path(build_dir) / f"sim-{h.hexdigest()[:16]}.so"
    if not so.exists():
        so.parent.mkdir(parents=True, exist_ok=True)
        tmp = so.with_name(f"{so.name}.{os.getpid()}.tmp")
        subprocess.run(["g++", *FLAGS, *(str(p) for p in sources if p.suffix == ".cpp"), "-o", str(tmp)], check=True)
        os.replace(tmp, so)
    lib = ctypes.CDLL(str(so))
    I, I64, P = ctypes.c_int, ctypes.c_int64, ctypes.c_void_p
    lib.sim_layout.restype = ctypes.c_char_p
    lib.sim_run.argtypes = [P, P, I, P, P, I, P]
    lib.sim_trace.argtypes = [P, P, P, P, P, P, I64, P, I, P]
    lib.sim_program.argtypes = [P, P, I, I, I, P]
    layout = dict(part.split(":") for part in lib.sim_layout().decode().split(";"))
    expected = dict(
        config=CONFIG_FIELDS,
        scalar=SCALAR_FIELDS,
        context=CTX_FIELDS,
        array=ARRAY_FIELDS,
        cls=CLASS_FIELDS,
        trace=TRACE_FIELDS,
        trace_context=TRACE_CTX_FIELDS,
    )
    layout["cls"] = layout.pop("class")
    assert all(tuple(layout[k].split(",")) == v for k, v in expected.items()), (
        "engine.py and src/model.hpp layouts differ"
    )
    assert layout["dims"] == f"{MAX_CONTEXTS},{MAX_ARRAYS},{NCLASS}", "engine.py and src/model.hpp layouts differ"
    return lib


def _config(env, workload, batch, contexts):
    if (
        env.policy not in (2, 13, 14)
        or env.p14_steal not in (0, 1, 2)
        or env.port_mode not in (0, 1)
        or env.kappa_reread < 0
    ):
        raise ValueError("policy, p14_steal, port_mode or kappa_reread out of range")
    if env.port_mode and env.kappa_reread:
        raise ValueError("port_mode 1 replaces kappa_reread")
    if env.systolic not in SYSTOLIC or env.drain_form not in DRAIN_FORMS or env.drain_segment < 1:
        raise ValueError("systolic mode or drain form out of range")
    if min(env.W, env.X, env.Y) < 64 or env.bits not in (2, 4, 8, 16) or env.group not in (32, 128):
        raise ValueError("ports below one 64-byte beat, or unknown weight format")
    inner, last, bubble, tail = SYSTOLIC[env.systolic]
    v = dict(
        asdict(env),
        batch=batch,
        topk=workload["topk"],
        experts=workload["experts"],
        hidden=workload["hidden"],
        routed_width=workload["routed_width"],
        shared_width=workload["shared_width"],
        shared_gate=int(workload["shared_gate"]),
        contexts=contexts,
        kappa_milli=round(env.kappa_reread * 1000),
        drain_inner=inner,
        drain_last=last,
        tile_bubble=bubble,
        tail_slot=tail,
    )
    return np.array([int(v[k]) for k in CONFIG_FIELDS], np.int64)


def _inputs(routes, shapes, workload, env):
    routes = np.ascontiguousarray(routes, np.int32)
    if routes.ndim == 2:
        routes = routes[None]
    _, B, topk = routes.shape
    if (
        topk != workload["topk"]
        or routes.min() < 0
        or routes.max() >= workload["experts"]
        or np.any(np.diff(np.sort(routes, 2), axis=2) <= 0)
    ):
        raise ValueError("routes: every token needs topk distinct experts")
    shapes = np.array(shapes, np.int64).reshape(-1, 3)
    if np.any(shapes[:, 1] % shapes[:, 0]) or np.any(shapes[:, 2] % shapes[:, 0]):
        raise ValueError("array shapes need n and k multiples of m")
    cfg = _config(env, workload, B, len(shapes))
    if 6 * B * workload["hidden"] >= env.capacity:
        raise CapacityError("the token buffers alone fill the SRAM")
    drain = [drain_latency(int(n), env) for n in shapes[:, 1]]
    return routes, np.ascontiguousarray(np.column_stack([shapes, drain]), np.int32), cfg


def _provision(tokens, batch):
    mask = np.zeros(batch, np.int8)
    t = np.asarray(sorted(tokens if tokens is not None else ()), np.int64)
    if t.size and (t[0] < 1 or t[-1] > batch):
        raise ValueError("provision_tokens must lie in 1..B")
    mask[t - 1] = 1
    return mask


def token_counts(routes, experts):
    """Token counts of the experts that receive tokens in some window of routes (windows, B, topk)."""
    r = np.asarray(routes)
    counts = np.stack([np.bincount(w.ravel(), minlength=experts) for w in r.reshape(-1, *r.shape[-2:])])
    return sorted(set(counts[counts > 0].tolist()))


def _check(ret, out, B, topk, env):
    if ret:
        raise ValueError(ERRORS.get(ret, f"engine error {ret}"))
    c = counters(out)
    if np.any(c["status"] == 1):
        raise CapacityError("a task fits no context")
    if np.any(c["status"] == 2):
        raise RuntimeError("deadlock: no progress for 100,000 cycles")
    assert np.all(c["rows"] == B * (topk + 1)) and np.all(c["peak_sram"] <= env.capacity), "inconsistent engine output"


def simulate(routes, shapes, workload, env, threads=4, provision_tokens=None):
    """Simulate windows of one layer. routes: (windows, B, topk) expert ids; shapes: (m, n, k) of each context's array.
    SRAM is provisioned for the largest program over the token counts in routes and provision_tokens (e.g. token_counts()
    of a whole window group), which decides tail-slot eligibility in systolic modes 3 and 5. Returns an int64 array
    (windows, NOUT)."""
    routes, shp, cfg = _inputs(routes, shapes, workload, env)
    provision = _provision(provision_tokens, routes.shape[1])
    out = np.zeros((len(routes), NOUT), np.int64)
    lib = library()
    ret = lib.sim_run(
        cfg.ctypes.data,
        shp.ctypes.data,
        len(routes),
        routes.ctypes.data,
        provision.ctypes.data,
        int(threads),
        out.ctypes.data,
    )
    libc = ctypes.CDLL(None)  # return freed program memory to the system between candidates of a search
    if hasattr(libc, "malloc_trim"):
        libc.malloc_trim(0)
    _check(ret, out, routes.shape[1], routes.shape[2], env)
    return out


def program_metrics(shape, tokens, shared, workload, env, capacity, lanes=1024):
    """Compiled program of one expert task (tokens rows; the shared expert if shared) on an array (m, n, k) with the
    given vector lanes and SRAM bytes: dict(estimate, cycles, hbm_bytes) with the dispatch estimate, the compute cycles of
    its phases and its HBM bytes; None if it does not fit."""
    m, n, k = (int(v) for v in shape)
    shp = np.array([m, n, k, int(lanes), drain_latency(n, env)], np.int32)
    out = np.zeros(3)
    if library().sim_program(
        _config(env, workload, 1, 2).ctypes.data,
        shp.ctypes.data,
        int(tokens),
        int(bool(shared)),
        int(capacity),
        out.ctypes.data,
    ):
        return None
    return dict(estimate=float(out[0]), cycles=int(out[1]), hbm_bytes=int(out[2]))


def counters(out):
    """Name the fields of simulate() output; leading axes are those of out. Per-context fields 'ctx_*' end in a context
    axis, per-array fields 'array_*' in an array axis, task-class counters in (array, class) axes (classes in CLASSES
    order). Adds the TOTALS over contexts and array_busy, the enabled cycles of each array's contexts."""
    out = np.asarray(out)
    lead, o, d = out.shape[:-1], 0, {}
    for n in SCALAR_FIELDS:
        d[n] = out[..., o]
        o += 1
    for n in CTX_FIELDS:
        d["ctx_" + n] = out[..., o : o + MAX_CONTEXTS]
        o += MAX_CONTEXTS
    for n in ARRAY_FIELDS:
        d["array_" + n] = out[..., o : o + MAX_ARRAYS]
        o += MAX_ARRAYS
    for n in CLASS_FIELDS:
        d[n] = out[..., o : o + MAX_ARRAYS * NCLASS].reshape((*lead, MAX_ARRAYS, NCLASS))
        o += MAX_ARRAYS * NCLASS
    for n in TOTALS:
        d[n] = d["ctx_" + n].sum(-1)
    d["array_busy"] = np.stack(
        [np.where(d["ctx_core"] == k, d["ctx_enabled"], 0).sum(-1) for k in range(MAX_ARRAYS)], -1
    )
    return d


def trace(ids, shapes, workload, env, provision_tokens=None):
    """One window (ids: B x topk) with a per-cycle record and the dispatch decision log: (out row, record, decisions).
    record maps TRACE_FIELDS to (cycles,) and TRACE_CTX_FIELDS to (cycles, contexts) arrays, from cycle `startup` on;
    state codes are STATES indices. A decision is (cycle, context, task, expert (-1 router, `experts` the shared expert),
    tokens, kind, policy, predicted cycles) with kind 0 dispatch to a free context, 1 next-task reservation, 2 reserved
    task started, 3 router started, 4 task finished, 6 tail handoff; kind 5 is a band plan: (cycle, array, first task,
    last task, rank, 5, policy, planned makespan)."""
    out = simulate(np.asarray(ids)[None], shapes, workload, env, threads=1, provision_tokens=provision_tokens)
    routes, shp, cfg = _inputs(np.asarray(ids)[None], shapes, workload, env)
    provision = _provision(provision_tokens, routes.shape[1])
    c = counters(out)
    T, nc = int(c["cycles"][0] - c["startup"][0]), len(shp)
    rows, n = np.zeros((T, NTRACE), np.int64), ctypes.c_int()
    dec = np.zeros((4 * (workload["experts"] + 2) + MAX_ARRAYS, 8), np.int64)
    again = np.zeros_like(out)
    ret = library().sim_trace(
        cfg.ctypes.data,
        shp.ctypes.data,
        routes.ctypes.data,
        provision.ctypes.data,
        again.ctypes.data,
        rows.ctypes.data,
        T,
        dec.ctypes.data,
        len(dec),
        ctypes.byref(n),
    )
    if ret:
        raise ValueError(ERRORS.get(ret, f"engine error {ret}"))
    assert np.array_equal(again, out)
    k = len(TRACE_FIELDS)
    rec = {f: rows[:, i] for i, f in enumerate(TRACE_FIELDS)}
    for i, f in enumerate(TRACE_CTX_FIELDS):
        rec[f] = rows[:, k + i * MAX_CONTEXTS : k + i * MAX_CONTEXTS + nc]
    return out[0], rec, dec[: n.value].copy()
