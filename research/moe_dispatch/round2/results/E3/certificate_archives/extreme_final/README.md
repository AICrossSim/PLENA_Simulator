# Final CMA verification payload

The required `workload_extreme.json` is an explicitly derived compact summary. The exact 137 MiB actual-driver output, with its original SHA-256, is preserved as the `workload_extreme.json` member of this independent archive. A standalone final delta=0 checkpoint includes the exact saved synthetic workload and all original resume/frontier fields. Every archived member passed byte-for-byte roundtrip and engine/workload/parameter validation.

From the E3 directory, extract only the standalone checkpoint:

```sh
tar -xzf certificate_archives/extreme_final/part_000.tar.gz -C . cma_verification/final_delta0.json
```

From the repository root, continue without refitting or regenerating routing:

```sh
python -m research.moe_dispatch.round2.resume --certificate research/moe_dispatch/round2/results/E3/cma_verification/final_delta0.json --seconds 3600
```

To restore the exact full original driver output too, extract the `workload_extreme.json` archive member. The original hash and derived-summary hash are recorded separately in `manifest.json`; raw files over 50 MiB are listed explicitly and are not committed.
