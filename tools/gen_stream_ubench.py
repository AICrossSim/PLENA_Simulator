"""Generate a streaming-GEMM microbenchmark for the OOO A/B comparison.

Shape: 16 weight chunks (1 MRAM tile = 1024x1024 MXFP8 each) stream from
HBM through the 4-tile MRAM round-robin. Per chunk: H_PREFETCH_M then
8x (M_MM col-slice) + 1x M_MM_WO. X operand comes from VRAM zeros (no
V prefetch needed - we only measure simulated latency, data is zeros).

Sequential simulator: each H_PREFETCH_M blocks until HBM data lands ->
T = sum(T_pref + T_mm). OOO (Step1 fanout + 2.2c): prefetch of chunk i+1
overlaps M_MM of chunk i -> T ~ max-bound. The gap quantifies the
runtime's automatic double-buffering.

Usage:
  PLENA_SIM=... python3 /tmp/gen_stream_ubench.py /tmp/ubench_stream
"""
import os
import sys
from pathlib import Path

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ubench_stream")
OUT.mkdir(parents=True, exist_ok=True)

MLEN = 1024
TILE = MLEN * MLEN              # elements per MRAM tile
N_CHUNKS = 16
MRAM_TILES = 4
CHUNK_BYTES = TILE              # MXFP8: 1 byte per element
SCALE_BYTES_PER_CHUNK = TILE // 8   # E8M0, block=8
ELEM_REGION = N_CHUNKS * CHUNK_BYTES            # 16 MiB
SCALE_REGION = N_CHUNKS * SCALE_BYTES_PER_CHUNK  # 2 MiB

asm = []
emit = asm.append

# Scale base register: scales live AFTER all elements.
# C_SET_SCALE_REG reads a gp reg; value = ELEM_REGION.
hi, lo = divmod(ELEM_REGION, 1 << 12)
emit(f"S_LUI_INT gp1, {hi}")
if lo:
    emit(f"S_ADDI_INT gp1, gp1, {lo}")
emit("C_SET_SCALE_REG gp1 ; scale region starts after 16MiB of elements")

for i in range(N_CHUNKS):
    tile_dst = (i % MRAM_TILES) * TILE
    # HBM address register a0 = element base of chunk i.
    addr = i * CHUNK_BYTES
    hi, lo = divmod(addr, 1 << 12)
    emit(f"S_LUI_INT gp2, {hi}")
    emit(f"S_ADDI_INT gp3, gp0, 0")
    if lo:
        emit(f"S_ADDI_INT gp2, gp2, {lo}")
    emit(f"C_SET_ADDR_REG a0, gp3, gp2 ; chunk {i} elements @ {addr}")
    # Dest MRAM tile (must be multiple of MLEN*MLEN).
    hi, lo = divmod(tile_dst, 1 << 12)
    emit(f"S_LUI_INT gp4, {hi}")
    if lo:
        emit(f"S_ADDI_INT gp4, gp4, {lo}")
    emit("S_ADDI_INT gp5, gp0, 0")
    emit(f"H_PREFETCH_M gp4, gp5, a0, 0, 0 ; W chunk {i} -> mram tile {i % MRAM_TILES}")
    # Consume the tile: 8 column-slices of 8 (BLEN) + one writeback.
    for j in range(8):
        col = j * 8
        if col:
            emit(f"S_ADDI_INT gp6, gp4, {col}")
        else:
            emit(f"S_ADDI_INT gp6, gp4, 0")
        emit("S_ADDI_INT gp7, gp0, 0")
        emit(f"M_MM 0, gp6, gp7 ; chunk {i} slice {j}")
    # Write accumulator out to VRAM rows far from anything else (row 256).
    wb = 256 * MLEN
    hi, lo = divmod(wb, 1 << 12)
    emit(f"S_LUI_INT gp8, {hi}")
    if lo:
        emit(f"S_ADDI_INT gp8, gp8, {lo}")
    emit(f"M_MM_WO gp8, gp0, 0 ; chunk {i} writeback")

asm_text = "\n".join(asm) + "\n"
(OUT / "ubench.asm").write_text(asm_text)
print(f"ASM: {len(asm)} instructions -> {OUT/'ubench.asm'}")

# Zero HBM image: elements + scales (zeros decode to 0.0 cleanly).
hbm = OUT / "hbm.bin"
with open(hbm, "wb") as f:
    f.truncate(ELEM_REGION + SCALE_REGION)
print(f"HBM:  {(ELEM_REGION + SCALE_REGION) >> 20} MiB zeros -> {hbm}")

# fp/int sram zero images (sizes per config_2: FP_SRAM 16384 f16, INT 1024 u32
# - exact size only needs to be <= simulator's; zeros are fine).
with open(OUT / "fp_sram.bin", "wb") as f:
    f.truncate(16384 * 2)
with open(OUT / "int_sram.bin", "wb") as f:
    f.truncate(1024 * 4)

# Assemble via the PLENA assembler.
sim_root = Path(os.environ["PLENA_SIM"])
sys.path.insert(0, str(sim_root))
sys.path.insert(0, str(sim_root / "tools"))  # `utils.load_config` lives here
os.chdir(OUT)
from compiler.assembler.assembly_to_binary import AssemblyToBinary  # noqa: E402

isa = sim_root / "compiler" / "doc" / "operation.svh"
cfg = sim_root / "compiler" / "doc" / "configuration.svh"
a2b = AssemblyToBinary(str(isa), str(cfg))
a2b.generate_binary(OUT / "ubench.asm", OUT / "ubench.mem")
print(f"MEM:  {OUT/'ubench.mem'}")

# ---------------------------------------------------------------------------
# Measured results (2026-06-09, config_2, 16 chunks x 1MiB, artifacts above):
#
#   baseline  yw/online_emulator (blocking prefetch, in-order):  428,180 ns
#   step1     bbe1b51 (per-tile fanout, in-order dispatch):      428,148 ns
#   ooo       yw/ooo_arch HEAD (fanout + 2.2c dispatch-ahead):   303,973 ns
#                                                  speedup:  1.41x (-29%)
#
# step1 ~= baseline is the ablation insight: the per-tile Err(rx) machinery
# alone doesn't help because the in-order dispatcher immediately blocks on
# the first M_MM's mram.read() of the still-in-flight tile. Only with the
# 2.2c dispatch-ahead (compute ops spawned, dispatcher free to issue the
# NEXT chunk's prefetch) does the channel machinery turn into overlap.
# Mechanism (Step 1) x scheduling (2.2c) — neither suffices alone.
#
# Repro:
#   PLENA_SIM=$PWD python3 tools/gen_stream_ubench.py /tmp/ubench_stream
#   DYLD_LIBRARY_PATH=<ramulator>:<torch> PLENA_CONFIG=$PWD/configs/config_2.toml \
#     transactional_emulator --opcode /tmp/ubench_stream/ubench.mem \
#       --hbm /tmp/ubench_stream/hbm.bin --fpsram /tmp/ubench_stream/fp_sram.bin \
#       --intsram /tmp/ubench_stream/int_sram.bin --quiet 2>&1 | grep Latency
# ---------------------------------------------------------------------------

# Memory-bound ablation (compute cut to 1 M_MM/chunk via range(8)->range(1)):
#
#   baseline  428,180 -> 313,157 ns   (less compute to serialize)
#   ooo       303,973 -> 296,805 ns   speedup collapses 1.41x -> 1.055x
#
# Confirms T = max(T_mem, T_compute): once memory dominates, OOO hides
# only the small compute slice and cannot accelerate the HBM drain
# itself (the WithTiming serial 64B chain is untouched). Fixing
# effective HBM bandwidth is an orthogonal memory-model change; OOO's
# role after that fix is keeping multiple prefetches outstanding so
# the widened pipe stays fed.
