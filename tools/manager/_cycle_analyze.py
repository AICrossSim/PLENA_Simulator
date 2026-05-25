#!/usr/bin/env python3
"""Static cycle analysis of manager-built kernel ISA.

The transactional_emulator has NO instruction-level parallelism: total latency =
sum over executed instructions of that opcode's cycle cost. So we can compute the
exact cycle count by parsing each kernel's .isa, expanding C_LOOP_START/END by
their static immediate trip counts (nested loops multiply), and summing per-opcode
costs taken from transactional_emulator/src/main.rs under the ACTIVE
plena_settings.toml (MODE.active = "analytic", DC_EN=1).

Active hw constants (analytic, dc_lib_en):
  MLEN=VLEN=1024, HLEN=128, BLEN=128, SYSTOLIC_PROCESSING_OVERHEAD=0
Cost table below is transcribed directly from main.rs cycle! sites.
"""
import re
import sys
from pathlib import Path
from collections import defaultdict

# --- active hw constants (analytic profile, dc_en=1) ---
MLEN = 1024
VLEN = 1024
SYS_OVH = 0  # SYSTOLIC_PROCESSING_OVERHEAD

# vector (dc_lib_en)
V_ADD, V_MUL, V_EXP, V_RECI, V_MAX, V_SUM = 1, 1, 1, 2, 4, 8
# scalar
S_FP_BASIC, S_FP_EXP, S_FP_SQRT, S_FP_RECI, S_INT = 1, 1, 1, 1, 1

# opcode -> (cost, category). category buckets for the breakdown.
MATMUL = "matmul(M_*)"
VEC = "vector(V_*)"
SMAP = "v<->fp map(S_MAP_*)"
SCAL_FP = "scalar_fp(S_*_FP)"
SCAL_INT = "scalar_int/addr(S_*_INT)"
CTRL = "control(C_*/loop)"
HBM = "hbm_dma(H_*)"

COST = {
    # matmul: SYS_OVH + MLEN
    "M_MM": (SYS_OVH + MLEN, MATMUL),
    "M_TMM": (SYS_OVH + MLEN, MATMUL),
    "M_BMM": (SYS_OVH + MLEN, MATMUL),
    "M_BTMM": (SYS_OVH + MLEN, MATMUL),
    "M_MV": (MLEN, MATMUL),
    "M_TMV": (MLEN, MATMUL),
    # matmul drains / batched-vector = 1
    "M_MM_WO": (1, MATMUL),
    "M_BMM_WO": (1, MATMUL),
    "M_MV_WO": (1, MATMUL),
    "M_BMV_WO": (1, MATMUL),
    "M_BMV": (SYS_OVH + 1, MATMUL),
    "M_BTMV": (SYS_OVH + 1, MATMUL),
    # vector
    "V_ADD_VV": (V_ADD, VEC), "V_ADD_VF": (V_ADD, VEC),
    "V_SUB_VV": (V_ADD, VEC), "V_SUB_VF": (V_ADD, VEC),
    "V_MUL_VV": (V_MUL, VEC), "V_MUL_VF": (V_MUL, VEC),
    "V_EXP_V": (V_EXP, VEC),
    "V_RECI_V": (V_RECI, VEC),
    "V_RED_SUM": (V_SUM, VEC),
    "V_RED_MAX": (V_MAX, VEC),
    # v<->fpram map = VLEN
    "S_MAP_V_FP": (VLEN, SMAP),
    "S_MAP_FP_V": (VLEN, SMAP),
    # scalar fp
    "S_ADD_FP": (S_FP_BASIC, SCAL_FP), "S_SUB_FP": (S_FP_BASIC, SCAL_FP),
    "S_MAX_FP": (S_FP_BASIC, SCAL_FP), "S_MUL_FP": (S_FP_BASIC, SCAL_FP),
    "S_EXP_FP": (S_FP_EXP, SCAL_FP), "S_RECI_FP": (S_FP_RECI, SCAL_FP),
    "S_SQRT_FP": (S_FP_SQRT, SCAL_FP),
    "S_LD_FP": (1, SCAL_FP), "S_ST_FP": (1, SCAL_FP),
    # scalar int / address
    "S_ADD_INT": (S_INT, SCAL_INT), "S_ADDI_INT": (S_INT, SCAL_INT),
    "S_SUB_INT": (S_INT, SCAL_INT), "S_MUL_INT": (S_INT, SCAL_INT),
    "S_LUI_INT": (S_INT, SCAL_INT),
    "S_SLL_INT": (S_INT, SCAL_INT), "S_SLLI_INT": (S_INT, SCAL_INT),
    "S_SRL_INT": (S_INT, SCAL_INT), "S_SRLI_INT": (S_INT, SCAL_INT),
    "S_LD_INT": (S_INT, SCAL_INT), "S_ST_INT": (S_INT, SCAL_INT),
    # control
    "C_SET_ADDR_REG": (1, CTRL), "C_SET_SCALE_REG": (1, CTRL),
    "C_SET_STRIDE_REG": (1, CTRL), "C_SET_V_MASK_REG": (1, CTRL),
    "C_LOOP_START": (1, CTRL), "C_LOOP_END": (1, CTRL), "C_BREAK": (1, CTRL),
    # hbm dma = 0 cycle
    "H_PREFETCH_M": (0, HBM), "H_PREFETCH_V": (0, HBM), "H_STORE_V": (0, HBM),
}

# rd==0 forms of these scalar/vector ops are no-ops (=> {}) costing 0 in main.rs.
NOP_IF_RD0 = {"S_ADD_FP", "S_SUB_FP", "S_MAX_FP", "S_MUL_FP", "S_EXP_FP",
              "S_RECI_FP", "S_SQRT_FP", "V_RED_SUM", "V_RED_MAX"}

OPC_RE = re.compile(r"^([A-Z][A-Z0-9_]+)\b(.*)$")


def parse_lines(path):
    """Return list of (opcode, first_operand_token) ignoring comments/blank."""
    out = []
    for raw in Path(path).read_text().splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        m = OPC_RE.match(line)
        if not m:
            continue
        opc = m.group(1)
        rest = m.group(2).strip()
        # operands separated by comma
        ops = [o.strip() for o in rest.split(",")] if rest else []
        out.append((opc, ops))
    return out


def analyze(path):
    instrs = parse_lines(path)
    # build loop structure via a stack walk to attach multiplier to each instr.
    # We compute by recursive expansion using indices.
    n = len(instrs)
    # match C_LOOP_START to C_LOOP_END by register, respecting nesting.
    # Pre-pair using a stack.
    start_to_end = {}
    stack = []
    for i, (opc, ops) in enumerate(instrs):
        if opc == "C_LOOP_START":
            stack.append(i)
        elif opc == "C_LOOP_END":
            s = stack.pop()
            start_to_end[s] = i
    assert not stack, f"unbalanced loops in {path}"

    cat_cycles = defaultdict(int)
    opc_cycles = defaultdict(int)
    opc_count = defaultdict(int)

    def trip(ops):
        # C_LOOP_START gpX, N  -> N is ops[1]
        return int(ops[1])

    def run(lo, hi, mult):
        i = lo
        while i < hi:
            opc, ops = instrs[i]
            if opc == "C_LOOP_START":
                end = start_to_end[i]
                t = trip(ops)
                # the C_LOOP_START itself executes once per entry into this scope (mult),
                # the END executes t times (jump-back), body executes mult*t.
                cost, cat = COST["C_LOOP_START"]
                cat_cycles[cat] += cost * mult
                opc_cycles[opc] += cost * mult
                opc_count[opc] += mult
                # body
                run(i + 1, end, mult * t)
                # the END
                eopc, _ = instrs[end]
                ecost, ecat = COST["C_LOOP_END"]
                cat_cycles[ecat] += ecost * mult * t
                opc_cycles[eopc] += ecost * mult * t
                opc_count[eopc] += mult * t
                i = end + 1
                continue
            if opc == "C_LOOP_END":
                # handled by its START; skip
                i += 1
                continue
            if opc not in COST:
                raise KeyError(f"unknown opcode {opc} in {path}")
            cost, cat = COST[opc]
            if opc in NOP_IF_RD0 and ops and ops[0] in ("f0",):
                # rd==0: f0 is the zero fp reg / gp0 zero. main.rs no-ops rd:0.
                cost = 0
            # also gp0-dest int? int ops to gp0 still cost; only fp/red rd0 no-op.
            cat_cycles[cat] += cost * mult
            opc_cycles[opc] += cost * mult
            opc_count[opc] += mult
            i += 1

    run(0, n, 1)
    total = sum(cat_cycles.values())
    return total, cat_cycles, opc_cycles, opc_count


if __name__ == "__main__":
    irdir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parents[2] / "managerbuild" / "ir"
    kernels = sorted(p.name for p in irdir.iterdir() if p.is_dir())
    grand = 0
    rows = []
    allcats = [MATMUL, VEC, SMAP, SCAL_FP, SCAL_INT, CTRL, HBM]
    per_kernel_cat = {}
    for k in kernels:
        isa = irdir / k / f"{k}.isa"
        if not isa.exists():
            continue
        total, cats, opcs, cnts = analyze(isa)
        grand += total
        rows.append((k, total))
        per_kernel_cat[k] = cats
        print(f"\n=== {k}: {total:,} cycles (ns) ===")
        for cat in allcats:
            if cats.get(cat):
                print(f"  {cat:24s} {cats[cat]:>14,}  {100*cats[cat]/total:5.1f}%")
        # top opcodes
        top = sorted(opcs.items(), key=lambda x: -x[1])[:6]
        print("  top opcodes:", ", ".join(f"{o}={c:,}(x{cnts[o]:,})" for o, c in top))

    print("\n" + "=" * 60)
    print("PER-KERNEL TOTAL (cycles == ns, 1 cycle = 1 ns):")
    for k, t in sorted(rows, key=lambda x: -x[1]):
        print(f"  {k:18s} {t:>16,} ns  {100*t/grand:5.1f}%")
    print(f"  {'CHAIN TOTAL':18s} {grand:>16,} ns  ({grand/1e6:.3f} ms)")


def chain_category_table(irdir):
    """Aggregate per-category cycles across the whole chain for the report."""
    from collections import defaultdict
    cat_total = defaultdict(int)
    grand = 0
    for p in sorted(irdir.iterdir()):
        isa = p / f"{p.name}.isa"
        if not isa.exists():
            continue
        total, cats, _, _ = analyze(isa)
        grand += total
        for c, v in cats.items():
            cat_total[c] += v
    return cat_total, grand
