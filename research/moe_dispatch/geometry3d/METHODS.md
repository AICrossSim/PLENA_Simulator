# Three-dimensional geometry study: execution and resource contract

This directory implements a prospective analytical comparison of physical
`PM × PN × PK` cores. It is separate from the frozen v3 simulator and the
earlier fixed-PK Round A study. This file defines the model and its limits;
selection results belong in the generated study report.

## Dimensions, precision and execution boundary

Logical `M`, `N` and `K` come from the captured post-Router expert workload.
Physical `PM`, `PN` and `PK` specify the installed array. The native transport
descriptors historically cover four output columns and 512 reduction elements;
those descriptor sizes do not fix physical `PN` or `PK`.

The declared search domain is `PM=1..16`, `PN=1..192`, and
`PK ∈ {32,64,128,256,512,1024}`. One-core and canonical two-core designs contain
exactly 12,288 main multipliers. The two cores may have different values on
every axis. Exhaustiveness refers to this bounded domain and the primary
settings. The subsequent allocation, dataflow, prefetch and dispatch searches
are finalist refinements, not a joint exhaustive hardware/runtime optimum.

Inputs and weights are BF16; partial sums and combined output are FP32. For
an expert with population `Me`, hidden width `H` and intermediate width `F`,
the required work is exactly `3 × Me × H × F` useful MACs: two `Me × F × H`
Gate/Up projections and one `Me × H × F` Down projection. Each physical issue
occupies `PM × PN × PK` multipliers. M/N/K tails add issued padding MACs but
never add useful work. Changing PK changes FP32 reduction grouping, so matching
precision does not establish bitwise equality across geometries.

The timing boundary begins with real routes, expert populations and routed X
available. The model charges clearing FP32 combined Y, dispatch, cold expert
weights, Gate/Up, SiLU/product, Down, route scaling and combine. It excludes
Router computation, attention, upstream gathering and full autoregressive
generation. Cold weights are refetched for each layer invocation; this study
does not grant a persistent inter-layer weight cache.

## Fixed storage and physical endpoint budgets

Every family and batch uses the following installed capacity. Workload size
changes live occupancy and legal execution, not installed SRAM.

| Structure | Installed bytes | Purpose |
|---|---:|---|
| Global X | 524,288 | Resident input activations |
| Global FP32 combined Y | 1,048,576 | Output accumulation across routes |
| Z pool | 393,216 | Finite BF16 intermediate row chunks |
| Private W arena | 40,960 | Physical weight operand slots |
| Shared ingress | 8,192 | Native 32-byte responses and landing |
| Local X arena | 12,288 | Physical operand slices |
| Local accumulator/RF arena | 98,304 | Bounded live FP32 output tiles |
| Control | 16,384 | Controller, descriptors and slot state |
| Routes | 16,384 | Runtime route metadata |
| **Total** | **2,158,592** | **Same capacity for all designs** |

Private W/X/accumulator/Z quotas partition their single aggregate arenas.
The primary partition follows multiplier share, with integer alignment;
an equal partition is a sensitivity. Each core receives an 8-KiB minimum Z
quota inside the same 384-KiB total. W capacity is counted once: native
sectors land through the bounded ingress into the physical operand allocation,
with explicit splice/copy service. There is no additional uncharged native
tile pool or alternative packed HBM tensor.

W has 64 × 16-byte endpoint banks, X has 24 × 16-byte banks, and local
accumulator/RF has 12 × 16-byte banks. Their aggregate byte budgets are
1,024, 384 and 192 bytes/cycle, respectively. Integer per-core bank allocations
sum to these totals. The global activation path has a shared 384-byte/cycle
read/write service constraint covering X fetches, Z stores and Y read-modify-
writes. X fetches also obey their private X endpoint allocation; the two
constraints do not grant an additional independent free activation service.
The shared abstract vector engine supplies 64 modeled operations/cycle and
the shared control engine supplies one control service cycle per cycle.

A physical W slot costs `align32(2 × PN × PK)` bytes. Available slots are the
private W quota divided by that cost, capped by the configured prefetch limit
and the installed 32 slot descriptors/core. Each slot descriptor reserves
24 bytes of control state. Local X can hold up to two full `2 × PM × PK`
slices. At least one physical W and X slice must fit. A single-buffer design
remains eligible; its W last-use window limits refill overlap. Tail rows and
columns still occupy their full physical operand/partial-sum allocation.

`Settings.records` defaults to eight **output descriptors** per core. A Down
descriptor retains one FP32 `PM × PN` plane; a paired Gate/Up descriptor
retains two planes. Thus eight paired descriptors can retain sixteen spatial
FP32 tiles. The actual bound is the smaller of the descriptor limit and the
private accumulator quota divided by the descriptor's physical byte cost.
The control allowance is 256 bytes per output descriptor, plus all installed
W slot descriptors and 4,288 bytes of controller state. Layer execution rejects
a configuration when that sum exceeds the same 16-KiB control reserve.

The byte ledger includes operand/RF storage. Equal multiplier count and this
closed capacity ledger do not establish equal chip area, wiring, power or
energy.

## Native BF16 addresses and finite row chunks

The primary weight layout is the existing `W[N,K]` row layout with
`row_stride = align32(2 × K)`. Only real N rows exist in HBM. A projection's
unique native footprint is `N × row_stride`. Physical N padding and unused K
lanes are supplied locally. A logical K tail transfers its actual 32-byte
sectors rather than a full padded 512-element region; for example,
`N=10,K=1025` occupies 20,800 native bytes.

`NativeWeightLayout.tile` maps any physical PN/PK slice to those exact native
32-byte addresses. Adjacent requested sectors can coalesce when their live
references allow it. All supported PK values are multiples of 16 BF16 elements,
so partitioning K into their slices does not by itself duplicate a boundary
sector. The optional `v3_tiles` layout is an explicitly different compiler-v3
layout sensitivity, not the primary BF16 address contract.

For each owner, Z admits `floor(Z_quota / (2 × F))` logical rows. When the
whole expert cannot fit, full row chunks round down to a PM multiple where
possible; the last chunk may be short. Every chunk executes paired Gate/Up,
consumes each bounded group into BF16 Z, then executes Down and combine before
reusing its Z allocation. All K segments stay on that owner. Chunking repeats
the relevant weight passes and their real HBM bytes; no zero-cost spill or
unbudgeted full-expert Z allocation is assumed.

## Bounded groups, traffic and timing hypotheses

`model.py` uses the declared N-group, M-chunk, K-segment, M-block and N-tile
loops. Bounded WS admits as many M blocks as the descriptor limit allows;
bounded OS admits one M block. N groups obey both the remaining descriptor
capacity and the physical W slots, reserving a lookahead slot when possible.
With a single W slot, a paired group admits one M block and one N tile and
streams Gate and Up sequentially.

The principal model **interleaves Gate and Up at each K segment**. A group
with `q` paired output descriptors schedules `2q` independent FP32 planes
through its `s` K segments. All paired planes survive until SiLU/product
consumes the group. The ready-operand completion cost is

`(s−1) × max(2q × II, L) + (2q−1) × II + L`,

where `L` includes dot latency and commit. Down uses `q` in the same formula.
This differs from `compute.py::paired_gate_up`, which provides a separate
**Gate-group then Up-group** ready-operand helper. That helper is useful for
its explicit serial contract; its timing must not silently replace the
K-interleaved principal phase model.

W transfers repeat for each admitted M chunk, while a loaded group serves its
M blocks. X is fetched again for each N group, and the matching Gate/Up pair
reuses that X slice. Traffic includes native-sector splice reads, physical W
fills, W/X operand reads, and physical accumulator writes and follow-up K
reads. Group consumers add `8MeF` local FP32 read bytes and `2MeF` global Z
write bytes for Gate/Up; Down adds `4MeH` local read bytes and `8MeH` global
Y read-modify-write bytes. Modeled vector work is `3MeF` and `2MeH`,
respectively. These are declared abstract service counts, not an instruction-
accurate nonlinear-function implementation.

The default hypothetical clock is 1 GHz: one cycle is one ns and cycles/10⁶
is milliseconds. Dot latency is
`20 + 2 × log2(PK/512)` cycles, with one-cycle initiation and one-cycle commit.
The PK=512 dot=20 anchor is inherited; other PK values, routing delay and the
common frequency have not been physically validated. Sensitivities use one,
two or four cycles per changed reduction-tree stage, plus a flat 20-cycle
dot profile. These are explicit timing hypotheses. They are not synthesized
clock closure, and a changed timing profile may change geometry rankings.

## Shared supply, fluid progress and what the result means

HBM has nominal shared bandwidth 256 bytes/ns, response latency 64 ns and 256
credits, each covering one 32-byte transaction. The credit window gives a
128-byte/ns upper bound before landing. Credits remain occupied through a
minimum one-cycle landing commit, so the principal continuous service cap is

`min(256, 256 × 32 / (64+1)) = 126.03 bytes/ns`.

Private W slots impose an additional refill/last-use window using actual
native request bytes. Let `a` be a phase's average native bytes per loaded
slice, `c` its current W slots, `s` its spare slots, and `R` the last-read
service per current slice. With spare slots the lookahead cap is
`s × a / 65`. With no spare slots, a current group carries `B = c × a`
native bytes and has cap `B / (65 + B / HBM_cap + c × R)`: response and
landing, native payload transfer, and the group's last operand reads execute
serially. Physical tail padding occupies W capacity and read service, but
does not provide extra native requests or useful lookahead sectors. The
average-slice formulation remains a phase approximation for unequal tails.
One shared capacity serves all active cores; adding a core does not multiply
HBM bandwidth or credits.

The all-space comparison uses **phase-level fluid progress**. Each phase has
a local compute/dependency/endpoint bound and demand for shared HBM,
activation, vector and control service. Weighted max-min allocation uses the
inverse native-byte demand as its progress weight, providing equal native-
byte service as the reference fairness rule. A phase that reaches its local
bound stops receiving additional share, and available shared service goes to
other eligible phases. The simulation advances to phase completions and
scheduled events. A no-spare phase sets `serialized_supply`: its service
already includes every current group's response startup, so phase entry adds
no second 64-cycle delay. A phase with lookahead charges one explicit
64-cycle startup before fluid progress. The ready-operand group sum includes
its final dot/commit latency `L`; the phase's compute component subtracts that
last `L`, then every phase charges `final_drain_cycles=L` once after fluid
completion, including an HBM-limited final response. The runtime's private
estimate uses the same serialized/startup convention and final drain. A
layer dispatch barrier precedes streamed phase control work to avoid an
uncharged second controller.

This relaxation averages group consumers and SRAM service over each phase.
It is not a native per-cycle request trace, Ramulator execution, RTL execution,
or a guaranteed bound on the existing simulator. In particular, satisfying
average SRAM endpoint bandwidth does not prove that one physical bank can
perform every individual commit or that every simultaneous landing/read is
conflict-free. NoC topology, extra burst/packet overhead, detailed bank address
conflicts, decoder routing, arbitration implementation and clock closure are
not modeled. Limiter counters describe this analytical allocation and are not
measured native stall counts; startup can overlap idle attribution.

`SharedHBM` separately validates small finite-request experiments: per-32-byte
round-robin grants, credit retention through landing commit, finite ingress
with response backpressure, private landing quotas and last-use leases. Its
tests establish these mechanisms for the micro cases, not cycle-exact
validation of every full-space design. `HBMEndpoint` is an optional continuous-
credit endpoint approximation; bulk projection reservations there are not
the final ranking engine.

The `compute_only` study switch removes HBM, SRAM-port and controller service
constraints. It retains installed capacity, the same bounded groups and Z
chunks, K dependencies, group consumers and the **shared vector engine**.
Consequently it is a resource-disable diagnostic, not a pure unlimited-array
MAC-count benchmark. Resource-disable comparisons can freeze owner bindings
to separate supply effects from changed dispatch decisions.

## Compiler and runtime responsibilities

The compiler knows tensor dimensions, native addresses, legal physical
geometries, finite quotas, descriptor formats and the parametric loop rules.
It can generate owner-compatible phase plans and bounds. It does not predict
or invent future Router outcomes.

After Router supplies actual routes, the runtime forms each expert's real
`Me`, checks legal owners, chooses whole-expert FIFO bindings, and instantiates
the bounded loop/chunk counts. EFT, idle, round-robin and declared threshold
policies are alternative runtime choices. An expert keeps its chosen owner
through all chunks and all K segments; there is no free expert splitting,
cross-core partial-sum transfer or hidden migration. The principal model
assumes the route list is already available and charges its own binding
service and route-state capacity.

The study selects on development captures, freezes selected hardware and
runtime settings, then evaluates held-out captures and declared sensitivities.
The held-out data had previously been exposed in older v3/Round A work, so
this split is respected for the present selection but is not a pristine blind
dataset. The generated contract and hashes identify the exact inputs,
settings and scope used for any reported result.
