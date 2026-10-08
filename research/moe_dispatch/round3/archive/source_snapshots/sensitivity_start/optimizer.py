"""Isolated round-two exact allocation solver on the enlarged round-three model.

No frozen module is mutated. Assignment minimizes the conservative resource
relaxation; replay is a feasible finite LPT schedule, not a temporal optimum.
The original solver's grouped enumeration / deterministic CP-SAT machinery is
retained verbatim. Old fixed-category regional bounds are deliberately unused.
"""
from __future__ import annotations
from pathlib import Path
import types


def _static_cp_api(original):
    """Same installed CP model, with only the used legacy aliases defined once.

    The frozen solver text still calls CamelCase methods. OR-Tools normally
    rebuilds a large compatibility wrapper table on each CpModel instance;
    these direct aliases use the identical official snake_case implementations.
    This proxy is local to the isolated solver: no installed/global module or
    frozen round-two source is modified.
    """
    class StaticModel(original.CpModel):
        def _add_pre_pep8_methods(self):
            pass

        NewIntVar = original.CpModel.new_int_var
        AddHint = original.CpModel.add_hint
        Add = original.CpModel.add

    class StaticSolver(original.CpSolver):
        Solve = original.CpSolver.solve
        StatusName = original.CpSolver.status_name
        Value = original.CpSolver.value

        def ResponseProto(self):
            return self.response_proto

    class API:
        CpModel = StaticModel
        CpSolver = StaticSolver

        def __getattr__(self, name):
            return getattr(original, name)

    return API()


def _load():
    source = Path(__file__).resolve().parents[1] / "round2" / "optimizer.py"
    text = source.read_text()
    start = text.index("try:\n    from .model import")
    end = text.index("\n\ndef _down", start)
    # Route physical owner replay through the repaired finite runtime. The
    # costs, solver coefficient rounding and assignment domains stay identical.
    replacement = (
        "from .model import Design, Parameters, Core, task_cost, ceildiv, storage_chunks\n"
        "from .runtime import simulate\n"
    )
    text = text[:start] + replacement + text[end:]
    module = types.ModuleType(__name__ + "._isolated")
    module.__package__ = __package__
    exec(compile(text, str(source), "exec"), module.__dict__)
    module.cp_model = _static_cp_api(module.cp_model)
    return module


_solver = _load()
solve_assignment = _solver.solve_assignment
allocation_objective = _solver.allocation_objective
unique_hbm_bytes = _solver.unique_hbm_bytes
_costs = _solver._costs


def evaluate_design(workload, design, params, *, max_seconds=10.0, detail=False):
    solution = solve_assignment(workload, design, params, max_seconds=max_seconds)
    if solution["owners"] is None:
        return {"legal": False, "assignment": solution}
    from .runtime import simulate
    physical = simulate(workload, design, params, owners=solution["owners"], detail=detail)
    lower = solution["lb_cycles"]
    if lower > physical["cycles"] + max(1e-7, abs(lower) * 1e-10):
        raise AssertionError("allocation lower bound exceeds physical replay")
    return {"legal": True, "lb_cycles": lower, "assignment": solution,
            "milp_sched": physical,
            "gap_sched_pct": 100 * (physical["cycles"] / lower - 1) if lower else None}


def universal_bound(workload, params):
    """Mandatory service floor valid with cross-category cuts and shared W.

    It does not assume max Z=384 KiB or max retained W=40 KiB. The storage
    variables now exchange bytes inside one 532 KiB aggregate envelope.
    The W floor follows the inherited projection accounting (three services),
    and is tested against every physical evaluation.
    """
    unique = unique_hbm_bytes(workload)
    useful = sum(3 * e["Me"] * e.get("H", 2048) * e.get("F", 1408)
                 for e in workload["experts"])
    wtotal = 4096 / 30.4 if params.onchip_mode == "port_tight" else 64 * params.bank_Bpc
    if hasattr(params, "weight_tile_service_cycles"):
        wtotal = min(wtotal, 4096 / params.weight_tile_service_cycles)
    xmin = sum(e["Me"] * (2 * e.get("H", 2048) + 4 * e.get("F", 1408))
               for e in workload["experts"])
    amin = sum(e["Me"] * (16 * e.get("F", 1408) + 16 * e.get("H", 2048))
               for e in workload["experts"])
    vector = sum(e["Me"] * (3 * e.get("F", 1408) + 2 * e.get("H", 2048))
                 for e in workload["experts"])
    terms = {"hbm_unique": unique / params.hbm_bandwidth,
             "mac": useful / 12288,
             "W_mandatory": 3 * unique / wtotal,
             "X_mandatory": xmin / (24 * params.bank_Bpc),
             "acc_mandatory": amin / (12 * params.bank_Bpc),
             "vector_mandatory": vector / (64 * params.vector_scale)}
    # Both simultaneous maxima are granted to every task, an optimistic
    # relaxation of the single 532-KiB total. The unchanged full-row GU->Down
    # lifetime requires every row chunk to pass all expert weights.
    terms["HBM_rowchunk_optimistic532"] = _solver.rowchunk_hbm_floor_bytes(
        workload, max_z_bytes=532*1024, max_retained_w_bytes=532*1024
    ) / params.hbm_bandwidth
    name = max(terms, key=terms.get)
    return {"lb_cycles": terms[name], "binding_term": name, "terms": terms,
            "scope": "mandatory services + optimistic simultaneous532KiBZ/W full-row lifetime; no old category-limited rowchunk term"}


def region_bound(workloads, params, geometries):
    """Optimistic all-global-port + longest-task issue/dependency floor."""
    values = []
    for w in workloads:
        base = universal_bound(w, params)["lb_cycles"]
        longest = 0.0
        for e in w["experts"]:
            m, h, f = e["Me"], e.get("H", 2048), e.get("F", 1408)
            fastest = float("inf")
            from .model import ceildiv
            for cores in geometries:
                for c in cores:
                    issues = (2 * ceildiv(m, c.pm) * ceildiv(f, c.pn) * ceildiv(h, c.pk)
                              + ceildiv(m, c.pm) * ceildiv(h, c.pn) * ceildiv(f, c.pk))
                    dependency = max(ceildiv(h, c.pk), ceildiv(f, c.pk)) * (params.dot_latency(c) + 1)
                    fastest = min(fastest, max(issues * params.issue_interval, dependency))
            longest = max(longest, fastest)
        values.append(max(base, longest) / 1e6)
    return values
