"""Scalar FP SRAM depth as a decode co-design axis.

The emitted per-head schedule loops over KV heads on the outside and prefetches
the packed KV row inside that loop, so it reads every cached token `kv_heads`
times. At the headline geometry, its element plane is 4,096 B per token-tensor
against 512 B for one packed row. Including the E8M0 scale plane for MXINT4 with
block size 8, the corresponding row-aligned physical totals are 5,120 B and
640 B. The opt-in lowering hoists the KV-head loop inside the key-tile loop and
uses `M_BTMM`'s head selector to consume one resident row.

The instruction structure is pinned and the multi-head mechanism is exercised
end to end in the transactional emulator at MLEN=64. This module still reports
the Qwen headline as an analytic traffic sensitivity: MLEN=1024/hkv=8 has not
run in RTL, and the active MXFP RTL profile does not implement the selector.

What stops it is the online softmax. Keeping `g` KV groups live means their
running state must be live together, so a query-row tile of `t` rows holds

    constants + 3 * t * (MLEN / HLEN) * g

scalar slots, and the depth caps `g`. That is the trade this module reports: the
KV read plane falls as `kv_heads / g`, and `g` is bought with scalar SRAM.

The query-row tile is not itself a traffic term. Each packed query row is a
different sequence holding its own cache, so splitting the batch into row tiles
re-reads nothing -- it only needs `t >= BLEN`, and `t` a multiple of BLEN, for
the M_BTMM query block to be fully occupied. The binding cost is therefore one
full block of query rows, not the whole MLEN tile.

Usage:
    fp_sram_sweep.py --model qwen3-32b --mlen 1024 --hlen 128 --blen 4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))

from disagg_decode import load_model_dims, resolve_model_path  # noqa: E402

#: Shared constants the lowering seeds below the softmax state.
CONSTANT_SLOTS = 6

#: Running max, its exponentiated residual, and the running sum.
SCALARS_PER_ROW = 3

#: Depth declared in PLENA_RTL/src/definitions/configuration.svh.
RTL_FP_SRAM_DEPTH = 512

#: One scalar slot, from the RTL's S_FP widths (sign + exponent + mantissa).
FP_SLOT_BITS = 1 + 6 + 5


def state_slots(query_rows: int, broadcast_heads: int, groups_live: int = 1) -> int:
    """Scalar slots the online softmax holds live across one key sweep."""
    return CONSTANT_SLOTS + SCALARS_PER_ROW * query_rows * broadcast_heads * groups_live


def depth_for_reuse(broadcast_heads: int, blen: int, groups_live: int) -> int:
    """Depth that keeps `groups_live` groups live over one full query block."""
    return state_slots(blen, broadcast_heads, groups_live)


@dataclass(frozen=True)
class ReusePoint:
    """What one scalar FP SRAM depth buys on the KV read plane."""

    depth: int
    groups_live: int
    query_tile: int
    kv_read_factor: int
    live_slots: int

    @property
    def scalar_sram_kib(self) -> float:
        return self.depth * FP_SLOT_BITS / 8 / 1024


def evaluate(
    depth: int,
    *,
    broadcast_heads: int,
    kv_heads: int,
    blen: int,
) -> ReusePoint | None:
    """Best KV-head reuse a depth affords, or None if one query block will not fit.

    Groups are taken in powers of two so each key tile is consumed by a whole
    number of head-selector passes.
    """
    best = None
    groups = 1
    while groups <= kv_heads:
        if state_slots(blen, broadcast_heads, groups) <= depth:
            best = groups
        groups *= 2
    if best is None:
        return None
    return ReusePoint(
        depth=depth,
        groups_live=best,
        query_tile=blen,
        kv_read_factor=kv_heads // best,
        live_slots=state_slots(blen, broadcast_heads, best),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen3-32b")
    parser.add_argument(
        "--model-lib", default=str(_HERE.parent.parent / "compiler" / "doc" / "Model_Lib")
    )
    parser.add_argument("--mlen", type=int, default=1024)
    parser.add_argument("--hlen", type=int, default=0, help="0 follows the model's head_dim")
    parser.add_argument("--blen", type=int, default=4, help="systolic block length")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    dims = load_model_dims(resolve_model_path(args.model, args.model_lib))
    hlen = args.hlen or dims["head_dim"]
    if args.mlen % hlen:
        raise SystemExit(f"HLEN {hlen} must divide MLEN {args.mlen}")
    broadcast_heads = args.mlen // hlen
    kv_heads = dims["kv_heads"]

    print(
        f"{args.model}: MLEN={args.mlen} HLEN={hlen} BLEN={args.blen} -> "
        f"{broadcast_heads} head lanes, {kv_heads} KV heads"
    )
    print(
        f"one query block of one KV group holds "
        f"{SCALARS_PER_ROW * args.blen * broadcast_heads} slots; "
        f"the RTL provides {RTL_FP_SRAM_DEPTH}\n"
    )

    header = (
        f"{'KV groups live':>15}{'slots needed':>14}{'scalar KiB':>12}"
        f"{'KV reads/token':>16}{'vs RTL depth':>14}"
    )
    print(header)
    print("-" * len(header))
    points = []
    groups = 1
    while groups <= kv_heads:
        needed = depth_for_reuse(broadcast_heads, args.blen, groups)
        points.append(
            {
                "groups_live": groups,
                "slots_needed": needed,
                "kv_read_factor": kv_heads // groups,
            }
        )
        print(
            f"{groups:>15}{needed:>14,}{needed * FP_SLOT_BITS / 8 / 1024:>12.2f}"
            f"{kv_heads // groups:>15}x"
            f"{needed / RTL_FP_SRAM_DEPTH:>13.2f}x"
        )
        groups *= 2
    print("-" * len(header))

    on_die = evaluate(
        RTL_FP_SRAM_DEPTH,
        broadcast_heads=broadcast_heads,
        kv_heads=kv_heads,
        blen=args.blen,
    )
    if on_die is None:
        raise SystemExit("the RTL depth cannot hold one query block at this geometry")
    print(
        f"  At the RTL's {RTL_FP_SRAM_DEPTH} slots, {on_die.groups_live} groups stay live "
        f"({on_die.live_slots} slots), so each token is read "
        f"{on_die.kv_read_factor}x instead of {kv_heads}x."
    )
    one_read = depth_for_reuse(broadcast_heads, args.blen, kv_heads)
    print(
        f"  One read per token needs {one_read:,} slots, "
        f"{one_read / RTL_FP_SRAM_DEPTH:.2f}x the RTL depth "
        f"({(one_read - RTL_FP_SRAM_DEPTH) * FP_SLOT_BITS / 8:,.0f} B more scalar SRAM)."
    )
    print(
        "  Query-row tiling adds no KV-cache reread traffic: packed rows are "
        "separate\n  sequences. Its execution effects must be measured separately."
    )

    if args.json:
        args.json.write_text(json.dumps(points, indent=2) + "\n")
        print(f"\npoints -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
