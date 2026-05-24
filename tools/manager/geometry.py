"""Geometry + MX precision, read from plena_settings.toml [BEHAVIOR].

Single source of truth: the emulator reads this same toml, so the manager must
read it too — never hardcode. In particular ``HBM_WIDTH`` is 64 under BEHAVIOR
(the AddressAllocConfig default of 512 is the ANALYTIC value and would
mis-align every address). See MANAGER_DESIGN.md §6.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import toml

# Project root: tools/manager/geometry.py -> PLENA_Simulator/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_TOML = _PROJECT_ROOT / "plena_settings.toml"


@dataclass(frozen=True)
class BehaviorSettings:
    """Everything the manager needs from the toml's [BEHAVIOR] section.

    Geometry mirrors the compiler's HardwareSizes; precision mirrors the MX
    element/scale widths used to pack the HBM bin. We read the representative
    ``HBM_M_WEIGHT_TYPE`` MX type — all HBM MX types in the toml share the same
    E4M3 element + E8M0 scale + block=8 layout.
    """

    # geometry -- ALL of these scale with the hardware and must never be
    # hardcoded; they are read live from the toml. Today's small test HW
    # (mlen=64 hlen=16 blen=4 broadcast=4) will roughly double for the real
    # part (mlen=1024 hlen=128 blen=8 broadcast=8). Anything derived from
    # these must stay a function of them, not a baked-in constant.
    mlen: int
    hlen: int
    blen: int
    vlen: int
    hbm_row_width: int        # bytes -- BEHAVIOR.CONFIG.HBM_WIDTH (64, not 512)

    # MX precision (HBM_M_WEIGHT_TYPE)
    elem_exp: int             # element exponent bits (E4M3 -> 4)
    elem_man: int             # element mantissa bits (E4M3 -> 3)
    scale_exp: int            # scale exponent bits (E8M0 -> 8)
    scale_man: int            # scale mantissa bits (E8M0 -> 0)
    block_size: int           # elements per scale block (8)

    @property
    def hardware_lane_count(self) -> int:
        """BTMM head lanes per MLEN vector. Equals the toml BROADCAST_AMOUNT
        (broadcast_amount == mlen // hlen always holds), so we derive it rather
        than read a redundant field."""
        return self.mlen // self.hlen

    @property
    def elem_bits(self) -> int:
        # sign + exp + man
        return 1 + self.elem_exp + self.elem_man

    @property
    def elem_bytes(self) -> int:
        return self.elem_bits // 8

    @property
    def scale_bits(self) -> int:
        # E8M0 scale has no sign bit
        return self.scale_exp + self.scale_man

    @property
    def scale_bytes(self) -> int:
        return self.scale_bits // 8


def load_behavior_settings(toml_path: str | Path | None = None) -> BehaviorSettings:
    path = Path(toml_path) if toml_path is not None else _DEFAULT_TOML
    full = toml.load(path)
    b = full["BEHAVIOR"]          # always BEHAVIOR; [MODE].active is dead config
    cfg = b["CONFIG"]
    elem_t = b["PRECISION"]["HBM_M_WEIGHT_TYPE"]
    elem = elem_t["ELEM"]
    scale = elem_t["SCALE"]

    # Consistency guard: broadcast_amount must equal mlen//hlen. If the toml
    # ever violates this, derived lane counts would silently disagree with the
    # configured broadcast width — fail loud instead.
    _mlen, _hlen = int(cfg["MLEN"]["value"]), int(cfg["HLEN"]["value"])
    _bcast = int(cfg["BROADCAST_AMOUNT"]["value"])
    if _bcast != _mlen // _hlen:
        raise ValueError(
            f"plena_settings.toml [BEHAVIOR]: BROADCAST_AMOUNT={_bcast} != "
            f"MLEN//HLEN={_mlen}//{_hlen}={_mlen // _hlen}. The manager derives "
            f"lane_count as mlen//hlen; reconcile the toml."
        )

    return BehaviorSettings(
        mlen=int(cfg["MLEN"]["value"]),
        hlen=int(cfg["HLEN"]["value"]),
        blen=int(cfg["BLEN"]["value"]),
        vlen=int(cfg["VLEN"]["value"]),
        hbm_row_width=int(cfg["HBM_WIDTH"]["value"]),
        elem_exp=int(elem["exponent"]),
        elem_man=int(elem["mantissa"]),
        scale_exp=int(scale["exponent"]),
        scale_man=int(scale["mantissa"]),
        block_size=int(elem_t["block"]),
    )


def addr_cfg_from_toml(toml_path: str | Path | None = None):
    """Build a compiler ``AddressAllocConfig`` from the toml [BEHAVIOR].

    Imported lazily so the manager's data layer doesn't hard-depend on the
    compiler package being importable (venv split). Callers that only need
    geometry can use ``load_behavior_settings`` instead.
    """
    from tilelang_tvm_compiler.address_alloc import AddressAllocConfig

    s = load_behavior_settings(toml_path)
    return AddressAllocConfig(
        mlen=s.mlen,
        blen=s.blen,
        hlen=s.hlen,
        hbm_row_width=s.hbm_row_width,     # 64 under BEHAVIOR, NOT the 512 default
        hbm_elem_bits=s.elem_bits,         # 8
        hbm_scale_bits=s.scale_bits,       # 8
        hbm_block_size=s.block_size,       # 8
    )
