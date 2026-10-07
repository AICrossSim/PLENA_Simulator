# Phase-1 source integration gate

Supported current source tests pass. This is a source-validation gate, not evidence that any new hardware design is faster. Exact counts, skipped optional archive checks, source hashes and immutable receipts are in PHASE1_GATE.json.

| Scope | Result |
|---|---|
| Simulator analytical current collection | 469 unique passes, 9 optional archive skips, 478/478 node IDs covered |
| Simulator MoE Python | 281 passed + 282 subtests; no skips |
| Fresh main Rust | 408 passed |
| Fresh isolated MoE Rust | 89 passed |
| Compiler ordinary current support | 1,143 passed + 11 subtests; 29 explicit retired-profile skips |
| Compiler current TVM | 143 passed; 5 retired demos; 13 standalone entrypoints also pass |
| Compiler MoE research | 33 passed + 3 subtests |
| Compiler final lifetime and affected integration reruns | 41 and 45 passed (overlap, not additive) |
| Compiler slow supported connected programs | 3 passed; one retired X_STATE-profile skip |
| Historical BF16 exact replay | 945 + 405 = 1,350 records, repeated twice, all exact |

Compiler pin: 80ac775c91e1b9384ec40382090e525dd24e732a. Broad analytical coverage is the exact current node-ID union of two partial broad runs and the full repaired Matrix/formal files; the two superseded failing expectations are preserved with causal explanations. Failed environment/collection attempts remain in their original receipt directories and are not counted as passes. Archived tests are excluded from current discovery while their source is preserved.

The nine optional skips are three raw GPU archive rebuilds and six historical-E-Compiler component oracles; they are not current supported-feature failures. No new native HBM calibration, full-model inference or synthesized PPA is implied.
