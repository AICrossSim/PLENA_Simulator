# Memory calibration provenance

What the memory-bandwidth model was fitted to, how strong the evidence behind
each dataset is, and what would be needed to strengthen the weakest part.

"Provenance" here means the ability to replay a measurement exactly: the command
that produced it, the inputs it consumed, the binaries it ran, and a checksum
binding all of them to the recorded result. A dataset without that record can
still be a valid analysis, but it cannot be cited as a reproducible measurement.

## Retained datasets

Two historical aggregate tables and one structured per-request table:

- `calibration_bw.csv` — 120 rows spanning KV sizes 128–2048, HBM2 and HBM3,
  8/16/32 channels, and four operation classes.
- `calibration_dma.csv` — 42 rows spanning transfer amounts 64–4096, HBM2 and
  HBM3, and 8/16/32 channels.
- `calibration_dma_requests.csv` — 11,520 isolated Ramulator2 observations
  spanning opcode, transfer size, stride, alignment, MX precision, HBM
  generation, and 8/16/32/128 channels. Every row keeps the compiler-visible
  descriptor, the physical read and write byte counts, and the measured service
  time. The 16-channel plane was added so every headline HBM ranking point
  (8, 16, and 32 interface units) rests on a receipted measurement.
- `calibration_dma_requests.receipt.json` — a single immutable receipt holding
  all 11,520 process invocations with their exit statuses, standard output and
  error, raw op-statistics JSONL, artifact checksums, sweep inputs, and
  toolchain and host metadata.

Verify the contents, the completeness of the Cartesian grids, the stored Git
objects, and the current reproduction harnesses with:

```bash
python -m analytic_models.disagg_serve.calibration_provenance \
  --output results/calibration/memory-calibration-audit.json
```

## The two aggregate tables: valid analysis, incomplete receipt

Both aggregate CSVs were added to this repository on 2026-07-23 in commit
`8ec7964` ("analytic: disagg_serve package"). The harnesses that reproduce them
were added the same day in commit `d91f0ea` ("testbench: effective-bandwidth
calibration harnesses"). Use `git show <short-sha>` to inspect either.

The harness source checksums match those commits, so the code is accounted for.
What is missing is a receipt: nothing binds a specific harness invocation, its
settings, the compiler state, or the emulator binary to the aggregate rows that
were produced. The run cannot be replayed exactly.

The audit therefore grades these two tables
`aggregate_csv_without_raw_run_receipts` — aggregate results whose individual
runs were not recorded. The fitted model and its holdout results remain valid
aggregate analyses; they are simply not a complete publication calibration
receipt.

## The structured request dataset: complete receipt

The 8,640-row request dataset is graded separately as
`ramulator2_structured_csv_with_process_receipts`, meaning every underlying
process was recorded and can be replayed.

Its holdout split is deterministic and descriptor-aware: rows sharing an
identical physical descriptor are kept together on one side of the split, so the
model is never scored on a descriptor it was trained on. That gives 6,900
training and held-out observations at a 20.1% descriptor-aware split, with
absolute latency error of 6.27% median, 23.22% at P95, and 45.63% at P99
on the four-plane dataset.

The hardest retained group is HBM2 matrix prefetch at 32 channels, at 47.43%
P95 (16 channels: 20.42% P95). The audit deliberately preserves this tail
rather than reporting only the aggregate figure; with 32 channels now a
headline ranking point, that group's error band must be quoted wherever
32-channel results are reported. All of these are simulator-calibrated model
errors, not measured-silicon errors.

The calibration rows labelled HBM3 in every retained dataset were measured at
the emulator's 2 Gb/s pin rate, not a production HBM3 rate; they exist for
emulator-consistency checks and are never used for headline pricing. Faster
generations reach reports only through the labelled sensitivity path.

The structured CSV, its validation JSON, the harness, the emulator binary, the
compiler source state, the settings and the receipt are all checksum-bound. The
audit parses every raw op-statistic, confirms all 8,640 process IDs are present,
successful and unique, and checks that each instruction's latency and physical
issued bytes reproduce its CSV row exactly.

## Why the overall grade is still incomplete

The top-level calibration grade is limited by its weakest input. The structured
sweep has a complete receipt; the two historical aggregate tables do not, so the
combined grade stays incomplete.

Closing that gap means rerunning the aggregate sweep while retaining, for every
point: the argument vector, the settings and input checksums, the op-statistics
JSONL, standard output and error, the exit status, the emulator binary checksum,
the compiler revision, toolchain versions, host and environment metadata, and a
checksum of the final aggregate table.
