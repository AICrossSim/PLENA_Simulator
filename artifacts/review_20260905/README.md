# Fresh review execution evidence — 2026-09-05

See [validation](../../docs/REVIEW_VALIDATION_20260905.md) for commands and limits.
These summaries and CSVs are unmodified outputs from the paired review trees;
`manifest.json` records their hashes and executed source hashes. Original
campaigns remain separate.

All four connected command invocations exited 0. **KDA A/B fail the common
1% output error budget** despite exactly matching their own BF16 rounding
oracles. Their qualified speedup is correctly blank. Mamba's qualified ratio
is only for the prepared recurrent core. No full-model/GPU speedup is implied.

The projection run observed zero output error; its acceptance check uses
atol=rtol=0.2, not exact comparison. Affine recurrence cases require exact
comparison; fixed recurrence cases use the declared error budget.
