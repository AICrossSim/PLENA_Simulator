# Execution and comparison contract

## Fixed resources

Dimensions are M x N x K. Source operands are BF16; accumulation is FP32.

| Resource | Single 6 | Homogeneous 3+3 | Heterogeneous 4+2 |
|---|---|---|---|
| Core shape | 6x4x512 | 3x4x512 twice | 4x4x512 + 2x4x512 |
| Physical multipliers | 12,288 | 12,288 | 12,288 |
| Private 4 KiB weight slots | 10 | 5+5 | 5+5 |
| X staging | 12 KiB | 6+6 KiB | 8+4 KiB |
| Private work/result/control bytes | 2,097,152 | 1,048,576 each | 1,398,080 + 699,072 |
| Aggregate W/X/workspace banks | 64/24/12 | 64/24/12 | 64/24/12 |

Shared weight-return capacity is 8 KiB, making 48 KiB total weight storage.
The 4 KiB control reservation is inside the 2 MiB arena. Input, route records,
output inboxes and combined outputs are reserved before execution; workspaces
are checked per core. Equal capacities/MAC counts do not establish equal area.
X staging is an experimental private operand buffer, not a claim that original
PLENA already had these independent SRAM instances.

## Control and dataflow

The default window contains four unassigned experts. Each core runs at most one
expert at a time. Whole-expert ownership lasts through gate, up, activation,
down and retirement. The dynamic selector compares bounded candidate assignments
using a shape/service estimate; it may wait for a busy core. It does not learn
from past completions or inspect future event times.

Workspace is reserved at admission; finite weight slots/credits are acquired
before actual requests. Prefetch is independent of X operand staging. Up to G=4
N bands reuse a staged X tile. Independent outputs may pipeline; each output's
K segments wait for ordered accumulator commit. The optional paired-N mode
requires both cores, explicit Z copies and a barrier before down; it is a
different mechanism, not a free migration of partial sums.

Retirement copies results to pre-reserved inboxes before freeing expert storage.
Final route combination follows captured rank/score order. Both local and remote
copies pay source, bus and destination service; no input source is infinite.

## Timing assumptions, not physical-device measurements

One model cycle is 1 ns (1 GHz). Nominal HBM issue bandwidth is 256 B/ns, sector
size 32 B, response delay 64 ns and global credits 256. Credits remain occupied
through SRAM landing. Even before landing, credits/latency limit throughput to
128 B/ns. Row/bank/channel timing and refresh are not modeled.

Banks are 16-byte 1RW, with 2-cycle reads and a locked 5-cycle FP32 RMW. The
candidate dot tree has a 20 ns tail and eight finite result contexts/core.
Copy service is 384 B/ns; vector service is 32 elements/ns with a 16 ns tail.
Source-read/service/destination-write phases are conservatively serialized.
These hypotheses require native-backend and hardware calibration.

Start: route table and original layer input are already resident. End: every
expert and ordered output combination has drained. Router, attention, preceding
layers and full autoregressive generation are excluded. Front-end state counters
partition one core's observations and overlap other activity; do not add them
across cores or to arithmetic service to obtain wall time.

## Evidence and limitations

The analytical model uses captured route/shape metadata and explicit Compiler
addresses, not the original trained numerical weights. The NumPy audit separately
executes small complete MoEs through byte-addressed SRAM and checks BF16/FP32
rounding, K ordering, aliasing and copies. It is not a timing/payload unified test.

Historical sweeps contain 351 points x 2 repeats plus 9 trace points x 2 repeats.
Their frozen results are imported in `results/`; branch packaging is separately
validated with tests and 12 points x 2 repeats compared by raw report SHA-256.
Identical repeats establish determinism, not statistical generalization.

The imported best-per-window table selects from G={1,2,4} and fixed/dynamic/
adaptive policies. It is an exploratory search, not held-out policy performance.
Use `default_dynamic.csv` when comparing a single common policy. Online feedback,
multi-workload evaluation, native HBM, synthesis and numerical/timing integration
remain future work.
