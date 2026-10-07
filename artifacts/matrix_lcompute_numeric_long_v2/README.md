# Extended BF16 recurrence regression

All listed cases execute Compiler-generated machine words in Rust with the
same physical Matrix SRAM point as the four-token fixtures. State snapshots
are actual extra HBM stores after each completed token. Initial states and
prepared coefficients are deterministic synthetic BF16 data, not checkpoint
weights. `summary.json` records seeds, every intermediate state error/hash,
output hashes, input HBM/machine-code hashes and the explicit acceptance policy.

Phased deterministic fixtures require bit-exact results. Fixed uses the
existing 1% relative-L2 budget, element atol/rtol 0.01, and 1e-7 near-zero RMS
floor. These observations do not prove a tolerance for every possible input.
Snapshot DMA is diagnostic overhead; the timing counters here are not used in
the execution-comparison or whole-model speedup tables.
