"""Full saved-source before/after comparisons; no cached solver answers."""
from __future__ import annotations
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

from research.moe_dispatch.round3.common import ROOT, inputs, frozen_designs, canonical, digest, write_json, write_csv, sha
from research.moe_dispatch.round3.config import parameters
from research.moe_dispatch.round3.sensitivity import SensitivityParameters
from research.moe_dispatch.round3 import model, runtime, optimizer
from research.moe_dispatch.round2.predictors import Predictor
from research.moe_dispatch.round2.dispatch_fix.predictor_ablation_20261008.nominal_control import NominalPredictor


def baseline_modules():
    out = ROOT / "diagnostics/performance/before"
    prefix = "research.moe_dispatch.round3.diagnostics.performance.baseline_"
    loaded = {}
    for name in ("model", "runtime", "optimizer"):
        module = types.ModuleType(prefix + name)
        module.__package__ = "research.moe_dispatch.round3"
        # Preserve frozen adapter's relative path to round2/optimizer.py.
        module.__file__ = str(ROOT / (name + ".py"))
        sys.modules[module.__name__] = module
        source = (out / (name + ".py")).read_text()
        if name != "optimizer":
            source = source.replace("from .model import", "from " + prefix + "model import")
            source = source.replace("from .runtime import", "from " + prefix + "runtime import")
        exec(compile(source, str(out / (name + ".py")), "exec"), module.__dict__)
        if name == "optimizer":
            # Preserve the literal import-boundary search in the old adapter,
            # then route its fresh solver's cost globals to the before model.
            for key in ("Design", "Parameters", "Core", "task_cost", "ceildiv", "storage_chunks"):
                setattr(module._solver, key, getattr(loaded["model"], key))
            module._solver.simulate = loaded["runtime"].simulate
        loaded[name] = module
    return loaded


def run():
    before = baseline_modules()
    out = ROOT / "diagnostics/performance"
    ws = inputs()
    windows = [ws["development"][0], ws["development"][3], ws["development"][-1],
               next(w for w in ws["heldout"] if w["id"] == "v3_captured_mixed_heldout_gpqa_t128_l13")]
    points = [parameters(credits=256), parameters(credits=520),
              SensitivityParameters(weight_tile_service_cycles=15, bank_Bpc=16, dotstagecycles=2, credits=520, vector_scale=1),
              SensitivityParameters(weight_tile_service_cycles=30.4, bank_Bpc=8, dotstagecycles=4, credits=390, vector_scale=.5)]
    rows = []; started = time.monotonic()
    for mode in ("pipelined", "port_tight"):
        ds = frozen_designs(mode)
        designs = [(name, ds[name]) for name in ("B1", "B2", "best_hetero")]
        h = ds["best_hetero"]
        designs.append(("shared_hetero", replace(h, w_bytes=(0, 0), landing_mode="shared", landing_pool_bytes=sum(h.w_bytes))))
        for point_index, point in enumerate(points):
            p = replace(point, onchip_mode=mode)
            for name, d in designs:
                for w in windows:
                    old_sol = before["optimizer"].solve_assignment(w, d, p)
                    new_sol = optimizer.solve_assignment(w, d, p)
                    assert canonical(old_sol) == canonical(new_sol), ("solver", mode, point_index, name, w["id"])
                    rows.append({"component": "solver", "mode": mode, "point": point_index, "design": name,
                                 "window_id": w["id"], "detail": "all", "method": old_sol["assignment_backend"],
                                 "before_digest": digest(old_sol), "after_digest": digest(new_sol), "exact": True})
                    for detail in (False, True):
                        cases = [("milp", None, old_sol["owners"]), ("ours", Predictor, None),
                                 ("nominal", NominalPredictor, None)]
                        for method, factory, owners in cases:
                            old_pred = factory("ours") if factory is Predictor else factory() if factory else None
                            new_pred = factory("ours") if factory is Predictor else factory() if factory else None
                            old = before["runtime"].simulate(w, d, p, dispatch="fixed", predictor=old_pred, owners=owners, detail=detail)
                            new = runtime.simulate(w, d, p, dispatch="fixed", predictor=new_pred, owners=new_sol["owners"] if owners is not None else None, detail=detail)
                            assert canonical(old) == canonical(new), ("physical", mode, point_index, name, w["id"], detail, method)
                            rows.append({"component": "physical", "mode": mode, "point": point_index, "design": name,
                                         "window_id": w["id"], "detail": detail, "method": method,
                                         "before_digest": digest(old), "after_digest": digest(new), "exact": True})
                print({"mode": mode, "point": point_index, "design": name, "checked": len(rows)}, flush=True)
    write_csv(out / "equivalence.csv", rows)
    write_json(out / "EQUIVALENCE.json", {"cases": len(rows), "all_canonical_bitexact": True,
        "old_source_sha256": json.loads((out / "before/MANIFEST.json").read_text()),
        "new_source_sha256": {name: sha(ROOT / name) for name in ("model.py", "runtime.py", "optimizer.py")},
        "elapsed_seconds": time.monotonic() - started,
        "changes": ["detail=False skips discarded phase rows", "same-order static per-phase attribution reason", "same official CP methods via static isolated aliases"],
        "independence": "Each side constructs a fresh solver and executes its own full physical replay; no owner/solution/result caching."})
    print({"cases": len(rows), "all_bitexact": True, "seconds": time.monotonic() - started}, flush=True)


if __name__ == "__main__":
    run()
