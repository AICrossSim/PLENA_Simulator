# Controlled BF16 recurrence execution

Every A/B/D row executes Compiler machine words in Rust. A and B are newly
materialized packed ordinary-VV controls, with explicit addresses and static
address reuse respectively; they are **not** the historical analytic
original/Arlo instruction census. The tests use the same logical BF16 initial
state and prepared scalar values, and the same SRAM topology for every variant.
No checkpoint weights, projection, routing or full model are executed here.

A/B explicitly expand coefficients into HBM lanes; D transfers compact fields.
The measured DMA difference is part of this dataflow comparison, not an isolated
bank-only gain. A/B round after each VV instruction. D performs local FP32
reductions before writing BF16 state/output. Each sequence must exactly match
its independent rounding reference; a separate common recurrence budget is
relative L2 <= 1%, plus the declared element bound. A blank speedup cell means
that common budget failed and no speedup claim is qualified.

Each request has a private persistent HBM state arena. The whole batch runs
in a single Rust invocation, token-major, without host state updates. All
non-output/non-state bytes, including guard regions and input fields, must be
unchanged. Distinct request inputs and per-request reference results also catch
writes to the wrong live state arena.

`timing_components.csv` is a dependency-safe serial model: one issue cycle per
instruction; ordinary VV rows pay two single-port bank reads and one write;
viewed operations pay actual packet bank service; both execute their arithmetic
latency code. DMA/memory wait is residual dispatch waiting, **not** an independent
HBM-engine busy counter. Components add to total elapsed virtual time. No ideal
overlap credit, RTL frequency or PPA is claimed. The opt-in timing contract
rejects opcodes outside the implemented recurrence subset.
