# Shared Expert DSE at the PLENA 65,536-multiplier budget

This experiment implements the user's Shared-only, compute-first study. It is
an analytical model anchored to the published PLENA topology, **not a measured
native PLENA simulation and not an end-to-end MoE inference benchmark**. The
earlier 12,288-multiplier results are neither inputs nor baselines here.

## Published configuration and the necessary precision distinction

The source is [PLENA v3](https://arxiv.org/html/2509.09505v3), Sections III.B,
V.A and Table III. The primary evaluated Matrix Unit is BLEN=32, MLEN=2048;
VLEN=2048 describes the separate Vector Unit and is not added to the Matrix
multiplier budget. Its 64 square 32-by-32 systolic subarrays contain exactly
65,536 multipliers. Table III lists BLEN powers of two from 2 to 64 and MLEN
powers of two from 2 to 1024. This study also includes the primary evaluated
MLEN=2048 point explicitly. The paper's main evaluated precision is W/A/KV
4/4/4; this requested BF16-input/weight, FP32-partial experiment is a precision
variant of its topology. Equal multiplier counts do not establish equal area
across precisions, shapes, replication networks or separate controllers.

The original physical expansion is PM=32, PN=32, G=64. Here G is the number
of spatial reduction groups; logical reduction width R=b*G=2048, and each
frame is fed over b=32 temporal cycles. The triple is **PM x PN x G**. It is
not a fully spatial 32 x 32 x 2048 multiplication array. For every candidate:

```
PM = b * RM; PN = b * RN; G = R / b
multipliers = PM * PN * G = RM * RN * b * R
```

## Workload and objective

Only the DeepSeek-V2-Lite Shared Expert is evaluated: H=2048, merged
intermediate F=2816, B in {2,4,8,16,64,96,128}. Gate and Up each require
B*H*F MAC operations; Down requires the same. Total useful MACs are 3*B*H*F,
useful FLOPs are twice that, and unique BF16 weights are exactly 33 MiB
(34,603,008 bytes). The equal-batch geometric mean of seven completion
times selects one frozen geometry for each family. Runtime row/column cuts
may change with B; the geometry does not. Repeating routing windows does not
create independent Shared matrix dimensions or statistical confidence.

SiLU, elementwise gating, router, Vector Unit service, memory/port delays,
dispatch overhead and task-descriptor execution are excluded from compute
time. Gate/Up may share a scheduled phase; Down begins after completed Z.
This is a three-GEMM compute time, not complete FFN or model latency.

## Two separate geometry domains

1. **Native topology domain:** RM=RN=1, exactly the paper's square-subarray
   organization. Exact-budget single cores and identical/shape-distinct dual
   cores are enumerated. Power-of-two native core budgets cannot form a
   65,536-total unequal two-core split. Calling such a point 3:1 would be false.
2. **Output-replicated extension:** independently replicate native groups in
   M and N, with PM powers of two 2..64, PN a positive multiple of four, and
   R powers of two b..2048. PN has only the physical budget bound, no
   workload-derived cap. This preserves the square subarrays but adds
   operand broadcast, M_SUM outputs and control. It is an architectural
   extension and is reported separately. No claim of equal area or 1-GHz
   layout closure is made. Exact 2:1 cannot divide an integer 65,536 budget;
   all representable ratios, including 3:1, are considered.

Single cores have exactly 65,536 multipliers; homogeneous cores have 32,768
each; heterogeneous pairs total exactly 65,536 and may have equal budgets
with different shapes. The full extension domain has 188 single geometries,
176 identical-dual geometries and 784,402 unordered nonidentical-dual
geometries, drawn from 53,040 legal individual cores. Alternative native
frames with the same physical triple remain distinct because timing differs.

## Explicit timing contract and its limitations

The repository's analytic ISA (`customISA_lib.json`) specifies steady-state
M_MM initiation interval b and isolated M_MM latency 2*b+9. It does not price
the complete M_SUM consistently with the current Rust path. Therefore these
paths are not silently combined or labeled native calibration. This model
declares the missing pipeline behavior:

```
records = factor * ceil(M/PM) * ceil(N/PN)
feed = ceil(K/R) * b
drain = b + 9 + log2(G) * tree_stage + 2
pipelined_phase = records * feed + drain
serialized_phase = records * (feed + drain)
```

`factor=2` jointly schedules Gate and Up output records. G groups retain
separate PE partials across the complete logical K loop and M_SUM reduces
them only at completed output. The default profile assumes a fully pipelined
FP32 tree, one cycle per level, and a separate shadow PE-partial bank that
allows output draining to overlap the next record. Tree registers, shadow
storage and operand staging are charged in the resource ledger. One-GHz
FP32 tree closure and simultaneous context access remain uncalibrated.

Three profiles are searched separately: `pipelined_tree1`,
`pipelined_tree2`, and `serialized_output`. A separate frozen minimax-regret
choice checks sensitivity to these timing contracts. Tree-two capacity
requires an adjusted register ledger; the main ledger is tree-one only.

For two cores, the search exactly solves both declared execution contracts:
contiguous token-row partitioning of the whole expert, or separate contiguous
output-column cuts for paired Gate/Up and for Down, with a shared-Z barrier.
No K split or cross-core partial summation is used. The default vectorized
backend evaluates every geometry at every batch and all three timing
profiles, without pruning. An independent scalar backend (`--backend pruned`)
uses valid fractional lower bounds that relax only integer M/N tiling. Both
algorithms are checked against complete small-domain enumeration. This proves the
optimum **inside this finite geometry, timing and partition contract**, not
an unrestricted program scheduling optimum.

## What OS/WS/IS mean in this implementation

All three use the same original OS PEs and useful arithmetic. OS uses
`m,n,k` and retains one output tile's G-group partials through K. WS uses
`k,n,m` and retains one weight tile over M. IS uses `k,m,n` and retains one X
tile over N. WS/IS save and restore G-group partial contexts between K visits;
this additional scratch capacity and traffic are explicitly counted.
These are operand-retention loop extensions, **not an implemented native
WS/IS PE mode switch**. With all supply/context ports ideal, compute time is
identical for these loop variants. Their distinction is operand/context
traffic and storage, not a manufactured compute speedup.

The report evaluates every frozen single geometry in all three flows, every
dual geometry in all nine ordered flow pairs, and all seven batches. It
separates these boundaries:

- Unique external weights into a unified resident pool: always 33 MiB.
- Pool-to-local operand refills: depend on loop order and reuse.
- Local-to-array operand delivery: includes retained operands presented again.
- A separate **no-resident-pool streaming** scenario: repeated operand and
  context transfers cross the external boundary. It is not added to the
  resident-pool scenario.

Payload storage peaks are explicit conservative bounds, not fitted minimum
SRAM areas. Weight residency and local staging are different capacities.
M_SUM registers and per-PE partial storage cannot be replaced with a small
PM*PN final-output buffer. WS/IS contexts multiply output slots by G.

## Bandwidth and crossover definitions

The requested reference channel is 256 GB/s (decimal), at 1 ns/cycle. Weight
service time is W/256 GB/s = **0.135168 ms**, independently of B. Required
external weight bandwidth is W/compute time. This is an average necessary
rate, not sufficient proof of latency hiding: bursts, independent local
ports, bank conflicts and capacity must subsequently be checked. SRAM
refill bandwidth and context rates are reported separately.

The ratio of compute time and ideal weight-service time is a roofline
diagnostic, not a measured HBM runtime. No finite-credit HBM simulation is
included in this compute-first task. The crossover table scans exact model
completion times through B4096; thresholds depend on the selected frozen
geometry and the explicit timing profile. It must not infer that B128 is
compute-bound merely because it is the largest requested B.

## Reproduction

Run from the simulator repository, with a Python environment containing
NumPy, openpyxl and (for figures) Matplotlib:

```
python -m pip install -r research/moe_dispatch/plena65536_shared/requirements.txt
python -m pytest research/moe_dispatch/plena65536_shared -q
python -m research.moe_dispatch.plena65536_shared.run --out OUTPUT_DIR
```

The search, report and plot modules can also be invoked separately, each
with `--out OUTPUT_DIR`. Exact-score NPZ files contain all geometries and
all three profile GM scores; `core_domain.json` resolves their indices.
Among exactly co-optimal heterogeneous designs, the main frozen selection
prefers the closest multiplier balance, rather than interpreting an arbitrary
first index as a uniquely beneficial extreme ratio. A 3:1 co-optimal witness
is also evaluated in every dataflow combination.

The report evaluates the frozen points twice and compares canonical results.
The full domain, lower-bound certificates, frozen selections, per-batch
tables, all dataflow combinations, traffic/capacity ledgers and explicit
completion manifest are delivered. Native timing/area calibration remains a
documented next validation step; it is not claimed to have been completed.
