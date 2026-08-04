"""The PackedKV ablation mode names are one contract declared in two places.

The compiler (`compiler/aten/plena/packed_kv.py`) and the analytic traffic
model (`analytic_models/disagg_serve/packed_kv.py`) each declare the four
mode strings; a drift between them would silently mislabel traffic evidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SIMULATOR_ROOT = Path(__file__).resolve().parents[2]
if str(_SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIMULATOR_ROOT))

from analytic_models.disagg_serve.packed_kv import PACKED_KV_MODES
from compiler.aten.plena.packed_kv import PackedKVAblation


def test_packed_kv_mode_names_match_the_compiler_contract() -> None:
    compiler_modes = tuple(member.value for member in PackedKVAblation)
    assert set(PACKED_KV_MODES) == set(compiler_modes)
    assert len(PACKED_KV_MODES) == len(compiler_modes)
