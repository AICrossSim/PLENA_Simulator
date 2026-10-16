# 65,536-multiplier Dedicated Shared/Routed Compute DSE

This is a separate experiment from the preserved general whole-expert study.
It implements **absolute hard affinity**, not a preference:

- Core 0 executes exactly the Shared expert.
- Core 1 executes every active routed expert, including hot experts.
- Neither core steals, substitutes, migrates or slices the other's work.
- Dual-core compute makespan is `max(Shared_C0_cycles, Routed_C1_cycles)`.
- A single executes both populations serially.

All operands are instantaneously ready, external bandwidth is infinite, SRAM
port delay is zero, and the reference clock is 1 GHz (one cycle = one ns).
Internal Matrix Unit feed, drain and reduction-tree computation is still
charged. The study excludes router, SiLU, gating/combine, memory and dispatch
execution. BF16 inputs and FP32 partials are a precision variant of the paper
geometry; this is not a newly measured full-model inference or native RTL run.

## Workload, objective and hardware freezing

The archived true DeepSeek-V2-Lite routes contain 18 development and 135 heldout
windows at B2/B4/B8/B16/B64/B96/B128. H=2048, routed F=1408, Shared F=2816,
Top-6 over 64 routed experts. Each token contributes to Shared once and to six
routed experts. Every active expert executes Gate, Up and Down exactly once.
All file hashes, token/expert membership and useful MACs are checked.

For every window, compute the integer role loads and their maximum first.
The objective is a window GM within each batch, followed by an equal-weight
GM over seven batches. Geometry is selected using development payloads only
and frozen before heldout replay. Historical heldout data has been accessed
in earlier studies; this is exploratory validation on archived routing.

## Geometry and what the certificate proves

Each design has exactly 65,536 physical multipliers. The paper geometry
baseline is BLEN=32, MLEN=2048: 64 native32x32 subarrays.

A declared core `(b,R,RM,RN)` has `PM=b*RM`, `PN=b*RN`, `G=R/b`, and
`MACs=PM*PN*G`. G is a physical subarray-group count, not logical GEMM K or
MLEN. Logical reduction R is streamed over b feed cycles.

The retained frame domain has BLEN>=2, PM powers2 through64, R powers2 up to
2048, and PN alignment4. To honour the requested PM=1 exploration, an explicit
new **BLEN=1 physical scalar1x1 subarray** is added in a second domain. This is
not merely turning off and reclaiming multipliers from a32x32 array; unused
lanes are not counted as free. The scalar extension uses the same conservative
feed/drain formula. It is permitted equally for the optimised single,
homogeneous and dedicated designs. It has not been synthesised or calibrated.

Dedicated Routed cores have PM in{1,2}; the retained-frame check has PM=2.
Shared cores explore all declared PM values. Homogeneous baselines explore
all legal identical32,768-multiplier cores under the same hard affinity.

Two budget-orientation domains are reported:

1. **Primary large-Shared/small-Routed:** Core0 multipliers >= Core1 multipliers,
   including the requested1:1 boundary; the two dedicated frames are distinct.
2. **Free-ratio diagnostic:** exact role binding remains unchanged, but the
   routed core may have more multipliers. This is explicitly not substituted
   for the primary large-Shared claim.

All legal ordered complementary-budget geometries are exhaustively evaluated;
there is no heuristic assignment or bound pruning. Integer makespans are exact.
Near-tied floating GM scores are resolved through an exact integer objective
product from rational batch weights. Among primary-objective ties, the selected
Shared core minimises Shared GM and then operand-buffer bytes; this tie-break
does not change makespan. Full search is repeated byte-for-byte at the array
level. Independent code rebuilds geometry coordinates and integer timing.

Global optimality is restricted to this declared finite geometry/timing domain.
PN alignment4 is an inherited constraint; wider alignment/unsupported shapes
are not silently claimed explored. The proposed43,690+21,846 split is invalid
under this alignment; it is also only approximately2:1, because65,536 is not
divisible by3. Area equality and actual1GHz timing are not certified by equal
multiplier counts. Three internal timing profiles are independently searched
and frozen-hardware sensitivity is reported.

## Structural workload implication

For every captured window, Routed useful MACs are exactly **3x Shared**.
The large-Shared constraint allocates at most half the chip to75% of useful
work. Consequently whole-chip wall-clock useful utilisation cannot exceed
66.67% even before padding/drain. A compute-balanced unrestricted split would
start near Shared:Routed=1:3; this is diagnosed rather than assumed to win.

Equivalent Fetch is unique cold weight bytes/256GB/s. Required bandwidth is
bytes/compute makespan. These are service/rate references, not measured HBM
time or complete end-to-end latency; resident weights require separate local
bandwidth and capacity verification.

## Reproduction

Python3.11+, NumPy, matplotlib, openpyxl and pytest are required. From the
repository or extracted portable archive root:

```sh
python -m research.moe_dispatch.plena65536_dedicated.run --out new_results
```

The runner repeats full search, executes the independent audit, replays all
frozen points twice, generates the user's two tables, alignment, bootstrap,
workbook, standalone PNG/PDF and source/input/results archive. Old general
dispatch and Shared-only outputs are not modified.

Paper reference: https://arxiv.org/html/2509.09505v3
