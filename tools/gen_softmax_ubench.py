"""S->P (masked online-softmax) microbenchmark for the fp-scoreboard +
multi-vector-unit A/B.

Pattern per S-tile row r (256 rows, VLEN=1024, BC=8 head segments):
  fk = f{1 + r%7}                       # fp register, round-robin over rows
  S_LD_FP fk, gp0, 0                    # seed
  for h in 0..8:                        # per-head masked max (the BC ops
      C_SET_V_MASK_REG (1<<h)           #  per VLEN row the flash S->P needs)
      V_RED_MAX fk, row_r, rmask=1
  V_SUB_VF row_r, row_r, fk             # x - max  (await fk via scoreboard)
  V_EXP_V  row_r, row_r                 # exp
  V_RED_SUM fk, row_r                   # denom
  V_MUL_VF row_r, row_r, fk             # scale (recip folded away - timing only)

Dependence shape: within a row the 8 reduces chain through fk (seed read);
ACROSS rows everything is independent except the 7-deep fp register file -
which caps cross-row ILP at 7. That cap is a real ISA constraint and part
of what this bench measures.

Old 2.2c semantics: every V_RED_* drains the whole machine -> ~2.3k full
pipeline flushes. Scoreboard: no drains; multi-unit: rows spread round-robin.
"""
import os
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ubench_softmax")
OUT.mkdir(parents=True, exist_ok=True)

VLEN = 1024
ROWS = 256
HEADS = 8

asm = []
emit = asm.append

for r in range(ROWS):
    row = r * VLEN
    fk = 1 + (r % 7)
    hi, lo = divmod(row, 1 << 12)
    emit(f"S_LUI_INT gp2, {hi}")
    if lo:
        emit(f"S_ADDI_INT gp2, gp2, {lo}")
    emit(f"S_LD_FP f{fk}, gp0, 0 ; row {r} seed")
    for h in range(HEADS):
        emit(f"S_ADDI_INT gp1, gp0, {1 << h}")
        emit("C_SET_V_MASK_REG gp1")
        emit(f"V_RED_MAX f{fk}, gp2, 1 ; row {r} head {h} masked max")
    emit(f"V_SUB_VF gp2, gp2, f{fk}, 0, 0 ; row {r} x-max")
    emit(f"V_EXP_V gp2, gp2, 0 ; row {r} exp")
    emit(f"S_LD_FP f{fk}, gp0, 0 ; row {r} sum seed")
    emit(f"V_RED_SUM f{fk}, gp2, 0 ; row {r} denom")
    emit(f"V_MUL_VF gp2, gp2, f{fk}, 0 ; row {r} scale")

asm_text = "\n".join(asm) + "\n"
(OUT / "ubench.asm").write_text(asm_text)
print(f"ASM: {len(asm)} instructions -> {OUT/'ubench.asm'}")

# Tiny zero HBM (kernel never touches HBM) + zero fp/int srams.
for name, size in [("hbm.bin", 4096), ("fp_sram.bin", 16384 * 2), ("int_sram.bin", 1024 * 4)]:
    with open(OUT / name, "wb") as f:
        f.truncate(size)

sim_root = Path(os.environ["PLENA_SIM"])
sys.path.insert(0, str(sim_root))
sys.path.insert(0, str(sim_root / "tools"))
os.chdir(OUT)
from compiler.assembler.assembly_to_binary import AssemblyToBinary  # noqa: E402

isa = sim_root / "compiler" / "doc" / "operation.svh"
cfg = sim_root / "compiler" / "doc" / "configuration.svh"
a2b = AssemblyToBinary(str(isa), str(cfg))
a2b.generate_binary(OUT / "ubench.asm", OUT / "ubench.mem")
print(f"MEM:  {OUT/'ubench.mem'}")

# ---------------------------------------------------------------------------
# Measured (2026-06-10, config_2 8ch/1GHz, 256 rows x 8 masked heads,
# 8,128 instructions, 2,304 masked V_RED_MAX):
#
#   (a) blocking baseline (yw/online_emulator)      16,064 ns
#   (b) 2.2c V_RED-drain (pre-scoreboard ooo)       15,553 ns
#   (c) fp-scoreboard, NUM_VECTOR_UNITS=1           11,969 ns   1.34x vs (a)
#   (d-g) NUM_VECTOR_UNITS = 2 / 4 / 8 / 16         11,969 ns   FLAT
#
# (b)->(c): de-draining V_RED_* alone buys 23% — every reduce used to
# flush the whole machine.
#
# The flat multi-unit curve is NOT fp-register pressure and NOT config
# failure (the unit count is logged at build_accelerator): it is the
# in-order dispatcher PARKING AT ISSUE. Per row, V_SUB_VF's write-acquire
# conflicts with the row's 8 in-flight masked reduces, so the dispatcher
# stalls until they finish — and while stalled it cannot issue the NEXT
# row's fully independent reduces. One stalled op blocks all younger
# independent ops: classic CDC6600-scoreboard stall-at-issue semantics.
# Units 2..16 therefore never see more than ~1 outstanding op each.
#
# Next step (2.2e, reservation-station semantics): move tracker
# acquisition into the spawned task with ticket-ordered grants
# (age-ordered wakeup) and pass fp operands as futures, so the
# dispatcher never blocks and independent rows pipeline across units —
# at which point NUM_VECTOR_UNITS should scale until the 7-deep fp
# register file caps cross-row ILP.
# ---------------------------------------------------------------------------

# 2.2e (Tomasulo reservation-station issue) re-measurement, same artifact:
#
#   scoreboard (stall-at-issue), any V:    11,969 ns   (flat)
#   tomasulo V=1:                          11,012 ns   (1.09x)
#   tomasulo V=2:                           6,416 ns   (1.87x)
#   tomasulo V=4:                           5,092 ns   (2.35x)
#   tomasulo V=8 / V=16:                    5,087 / 5,084 ns  (saturated)
#
# The numbers reconcile exactly: V=1 ~= 256 rows x 43 cy/row (the
# per-row reduce/softmax chain is serial; rows now PIPELINE across
# units). Saturation ~5,085 ns ~= the ~5K-instruction scalar/control
# stream at 1 op/cycle — with the back-end unblocked, the FRONT-END
# issue bandwidth becomes the next bottleneck (multi-issue decode is
# the follow-up), well before fp-register pressure binds.
#
# Two bugs found by the deadlock detector during bring-up, both now
# regression-covered by this bench's V=1 case:
#   * pump() same-pass co-grant: requests granted within one scan were
#     invisible to each other -> a row's V_SUB/V_EXP/V_MUL (mutually
#     write-conflicting) co-granted the moment its reduces released.
#   * operand-wait-under-unit-lock: V_MUL_VF held the (single) vector
#     unit while awaiting V_RED_SUM's fp future; the producer was
#     queued on the same unit. Operands now resolve after grant,
#     BEFORE the unit lock.
