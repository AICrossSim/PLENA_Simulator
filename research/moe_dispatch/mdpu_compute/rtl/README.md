# M-DPU control RTL prototype

These modules implement the control path exercised by `mdpu_compute/runtime.py`.
They do not implement a PLENA matrix datapath, measure native hardware, or prove
timing closure at 1 GHz. Operand supply remains ideal in this experiment.

```
Router descriptor: expert ID, token count, Shared flag
           |
Compiler coefficients --> ceil(Me/PM) service estimator (both core shapes)
           |
remaining drain + service, affinity tie break, capacity admission
           |
       +---+---+
       |       |
   DTQ, 8    STQ, 8       80-bit descriptors; ready/valid backpressure
       |       |
       +--head-only bypass--+   CURRENT/NEXT never migrate
       |       |
   NEXT_D   NEXT_S        independent four-word context setup
       |       |
   issue_D  issue_S       independent core ready/valid ports
       |       |
    Dense    Stream       current countdown feedback from each core
```

The ingress threshold supplies an affinity preference, not an unconditional
assignment. M-DPU compares the estimated completion clocks; Shared can migrate
when its alternative completion is earlier. The hard-threshold ablation
disables this bypass. A separate head-bypass enable distinguishes greedy EFT
from M-DPU after enqueue. `wait_best` compares completion before capacity; when
the winning queue is full, it backpressures the source instead of forcing the
descriptor into a slower queue. It does not wait when the other core is actually
predicted to finish earlier.

`mdpu_queue_engine.sv` maintains two queued-service sums, FIFO pointers/counts,
and two shadow contexts. It computes remaining drain from these state values
and the cores' current execution countdowns. Both own-queue claims are resolved
before any cross-queue claim. A head may move only when its requesting core's
completion is strictly earlier; current and shadow tasks are never moved.
The decision incorporates a same-cycle issue before considering a refill or
steal. Ingress observes same-cycle FIFO pops, allowing a full FIFO to receive
while its head is removed.

The default context latency is four cycles, a specified controller design cost:
four setup words per task. With preconfiguration enabled these cycles overlap
the active arithmetic task; otherwise claim waits until that core is idle.
This is instruction/context preparation, not an HBM prefetch speedup. A task
accepted in cycle 0 is visible for claim in cycle 1 and can first issue in cycle
5. Core execution and all charged arithmetic drain are retained.

The coefficient estimator exactly implements the analytical service formula
for the frozen shapes:

```
service = ceil(Me / PM) * coefficient[Shared/Routed] + tail
coefficient = 2*ceil(F/PN)*ceil(H/MLEN)*BLEN
              + ceil(H/PN)*ceil(F/MLEN)*BLEN
tail = 2 * phase_drain
```

PM is a power of two, so its ceiling is add/shift. PN need not be a power of two;
its division is performed by the compiler when creating coefficients. The RTL
estimator has a generic 16-by-32-bit multiplier. Only a version with coefficients
hardwired at synthesis can replace that multiply with constant arithmetic. No
one-cycle Fmax claim follows from this functional implementation.

The tested queue-engine parameters are depth 8 and context latency 4. FIFO depth
must be a power of two; context latency must be positive. Runtime sensitivity
points with context latency zero are not covered by this RTL. The queued-work
counters are 32 bits, sufficient for all frozen tasks and depth-8 queues in this
experiment. Changing the task domain requires checking these bounds again.

## Validation and accounting

Run from the simulator repository using Python 3.11:

```
PYTHONDONTWRITEBYTECODE=1 /tmp/plena-round2-venv/bin/python \
  research/moe_dispatch/mdpu_compute/rtl/validate_rtl.py \
  --output /home/mcl123/plena-outputs/mdpu_compute_20261013/rtl
```

The parent output directory must contain the development-frozen
`FROZEN_SELECTION.json`. Builds occur in temporary directories under `/tmp`.
The script emits source hashes, Verilator logs, exact runtime replay vectors,
`validation.json`, and `hardware_resources.json`.

Validation includes 200,000 selector vectors (ties, full queues, valid gating,
33-bit completion sums and wait-for-best), 43,008 estimator vectors, randomized
dual-queue stress, and all 135 heldout windows on three frozen two-core designs.
For every replay it compares ingress binding/time and every expert's core,
start, end and layer makespan against the Python event model. It covers M-DPU
with and without context preconfiguration, hard threshold, and the two guarded
admission variants. Exact replay does not calibrate the underlying arithmetic
service model against a physical matrix array.

The queue/context/queued-work implementation declares 1,532 state bits
(191.5 bytes). Two core execution countdowns and coefficient/configuration
registers add 242 bits if counted in the integrated controller, for 1,774 bits
(221.75 bytes). This excludes the matrix cores' existing current contexts and
operand/accumulator storage. Queues can become flops or RAM macros; these are
source-level bits, not post-synthesis cells. No technology library, synthesis,
place-and-route, area, watts or GHz timing closure result is claimed.
