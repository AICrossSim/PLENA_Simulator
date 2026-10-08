"""Local one-condition variant: do not poll a physically inadmissible queue.

The established dispatcher source and all frozen evidence remain byte exact.
Load a private namespace with the single displayed guard addition. Resource
release, phase transitions and actual progress already reconsider waiting work.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from .. import runtime as original

ORIGINAL_SOURCE = Path(original.__file__).read_text()
FROM = "if outstanding and earliest-at>params.binding_lead_cycles+1e-7:"
TO = "if can_bind and outstanding and earliest-at>params.binding_lead_cycles+1e-7:"
assert ORIGINAL_SOURCE.count(FROM) == 1
PATCHED_SOURCE = ORIGINAL_SOURCE.replace(FROM, TO)
ORIGINAL_SOURCE_SHA256 = hashlib.sha256(ORIGINAL_SOURCE.encode()).hexdigest()
PATCHED_SOURCE_SHA256 = hashlib.sha256(PATCHED_SOURCE.encode()).hexdigest()

# Relative imports and dataclass annotations retain the original package
# context; no original module globals or source files are modified.
_namespace = dict(vars(original))
exec(compile(PATCHED_SOURCE, str(Path(__file__)), "exec"), _namespace)
simulate = _namespace["simulate"]


def patch_metadata():
    return {"original_source_sha256": ORIGINAL_SOURCE_SHA256,
            "patched_source_sha256": PATCHED_SOURCE_SHA256,
            "from": FROM, "to": TO,
            "scope": "all predictors/organizations; no ETA wakeups when physical capacity_to_bind is false",
            "resource_wakeup": "existing actual completion, phase transition and progress events",
            "original_source_mutated": False}
