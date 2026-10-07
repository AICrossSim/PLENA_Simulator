# MoE event-ready controller: step 1

This experiment replaces output-pool polling with event notifications and a
ready bitmap. It retains the existing weight loader, physical P/R/Mt, full M
cohort, three packed+decoded weight slots, two operand stages, arithmetic order,
accumulator feedback, local data ports and native memory configuration.

The implementation is optional and defaults off:

```json
"refinement": {
  "stream_ctrl": {"event_ready": true, "selection": "rotating"}
}
```

This is a fragment of the existing per-core refinement object; its other
required fields remain unchanged. `selection` also accepts `lowest`. An absent
configuration or `event_ready: false` executes the frozen output-pool path.
No later streaming, local-sum, K-segmentation or ECT mechanism is enabled.

## Execution and dependencies

1. Admit a complete M cohort into Q records. A band occupies `ceil(Me/Mt)`
   records and fetches one shared weight tile for all of them.
2. The unchanged loader reads paired element/scale bytes, decodes them and
   writes decoded weight SRAM. Its completion produces a tile-decoded event.
3. The operand actor waits for an existing operand stage and reads through the
   existing local weight port. A second phase of that event marks the tile as
   installed in stationary operands.
4. The event actor updates one M-context descriptor per cycle. A context is
   selectable only when its operand data and previous K writeback are ready.
5. The issue actor selects from the bitmap, reads the old accumulator when
   needed and performs the original ascending-global-K FP32 additions. It
   retains the original activation service and MAC feedback timing.
6. After the original accumulator writeback service, a K-done event restores
   that context's dependency bit. Its band retires only after all completions
   and the final operand/weight-slot release. Slot release and writeback may
   arrive in either order.
7. The original projection-wide finalization remains: accumulator read,
   vector conversion/copy and BF16 rounding. Experts and projections remain
   serialized within a core.

The event, operand and issue actors are concurrent futures. Their metadata
accesses share one control service port; weight and accumulator data accesses
retain their existing separate arbitration. A full W-entry event FIFO blocks
producers. Pending completion payloads remain in bounded, already charged
slot/context records. There is no overflow-event queue or speculative next-K
weight slot. Loading is admitted only for the band's current K and when at
least one previous-K context can advance, as in the frozen output pool.

## Storage and timing accounting

Control state remains inside **accumulator SRAM**, alongside
`128*Q + 64*(weight_slots + operand_stages)` bytes of existing records.

Additional bytes are `24*ceil(Q/64) + 16*W + 8 + 64 + 64`. At Q=32 and W=3,
this is 208 B/core, changing control reservation from 4,416 B to 4,624 B.
Weight, vector, pending FP32-result and pipeline reservations are unchanged.
No additional SRAM capacity or data-port throughput is supplied.

| Control operation | Charged cycles at the 1 ns core clock |
|---|---:|
| Admit one N band with C M contexts | C+1 |
| Bind a tile to a band and slot | 2 |
| Consume decoded-slot event | 1 |
| Bind operand stage and slot | 2 |
| Expose tile to all M contexts | C, one context per cycle |
| Select/issue a context and update its band | 2 |
| K completion: context and band update | 2 |
| Release weight slot, operand stage and advance band | 3 |

Selection logic occupies one cycle within the paired issue-record accesses.
The raw `selector_busy_ps` counter includes both accesses; it is the inclusive
selection/issue-control service, not the encoder alone. All control visits,
including fan-out and this paired update, contribute to the original
`output_pool.scheduler_busy_ps` and `scheduler_visits` fields. There is at most
one descriptor write per cycle. No simultaneous actor service is subtracted
from the scheduler counter.

For T tiles, I context issues and admitted cohorts C_b, scheduler cycles are:

`sum_b(C_b+1) + 8*T + 5*I`.

For Me32 single Q32, T=768, I=6,144 and 384 bands each have C=8: the resulting
40,320 cycles are independently reconstructible from the work, rather than
obtained by discounting old scan counts.

Queue and port waits sum over waiting producers/actors and can overlap wall
time. The inherited weight/feedback idle counters are condition-based issue
wait classifications; they are not an additive or unique critical-path
attribution. Use the serialized service counters and total measured runtime
for the acceptance decisions.

## Validation and scope

`scripts/moe_stream_ctrl/step1.py` repeats the complete frozen step0 controls,
the restored work-conserving B32 HBM-clock/DMA-2x control, and both selectors for
Me32 single Q32 and that B32 configuration. Every point runs twice. Validation
checks BF16 and both FP32 representations, native bytes/drain, finite budgets,
port service, event counts, context fan-out and exact repeatability.

Unit tests additionally exercise the 208 B budget boundary, lowest/rotating
selection across bitmap words, short/tail cohorts, one-slot backpressure and
simultaneous completion during long event fan-out. First packed-tile arrival
timestamps are observational and cannot affect scheduling.

Only the specified B32 diagnostic changes HBM-clock/DMA timing relative to
native; its baseline and candidates use the same already frozen configuration.
Native Ramulator code, its library, Compiler exports and HBM layout are unchanged.
Historical demand-aware cases are compatibility checks only; new controller
points use `per_channel`. Demand-aware is excluded from the new architecture
comparison because the proposed step2 six-slot reservation exceeds 44 KiB.

The acceptance report is in the UUID output directory under
`outputs/moe_stream_ctrl_20260914/5cec1f6917944c7d8d96cb5ac7f79bc7/step1/`.
Any failed step1 criterion stops progression to step2.
