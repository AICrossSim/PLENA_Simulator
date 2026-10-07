# GPU capture evidence

This is the data entry point for `feat/matrix-sram-lcompute`. The paired
Compiler research branch is also `feat/matrix-sram-lcompute`; the mechanism
review PRs remain [Compiler #79](https://github.com/AICrossSim/PLENA_Compiler/pull/79)
and [Simulator #116](https://github.com/AICrossSim/PLENA_Simulator/pull/116).

| Capture | What actually ran |
| --- | --- |
| [Nemotron agentic, B200, September 3](tables/nemotron_agentic_20260903/) | Real Nemotron-3 Nano NVFP4 checkpoint; 48 prompts from BFCL v3, GPQA Diamond and SWE-bench Verified; timed B1/2/4/8/16. Routing was collected separately at B1. |
| [Hybrid formal, B200, August 19](tables/hybrid_formal_b200_20260819/) | Nemotron NVFP4 timing/routing and Nsight profiles; the KDA single-layer projection/recurrence/output/gate stages use synthetic tensors and random weights. |
| [KDA Stage 2, B200, August 17](tables/kda_stage2_20260817/) | Official-shape KDA prefill/decode, stage timing, numerical comparison and Nsight evidence. |
| [Kimi components and precision, B200, August 19](tables/kimi_components_precision_20260819/) | MLA/MoE component latency and profiles, plus Mamba state-precision experiments; synthetic inputs/random weights and an explicitly bounded MoE proxy. |
| [KDA numerical diagnostic, B200, September 5](tables/kda_numeric_20260905/) | B1, three seeds, 32 recurrent steps and six numerical modes; includes the nine completed-run tensor snapshots. |
| [Nemotron Mamba, RTX 5090](tables/nemotron_mamba_rtx5090/) | Historical official-shape Mamba mixer profiling with random BF16 weights; prefill and B1/4/8/16 decode. |

Kimi's complete 93-layer checkpoint was **not** executed in these captures.
The GPU measurements and PLENA cycle estimates remain separate evidence.
Existing [agentic estimates](../matrix_lcompute_agentic_v2/),
[whole-model estimates](../matrix_lcompute_e2e_v6/) and
[Rust recurrence comparisons](../matrix_lcompute_execution_v1/) are retained.

## Get the data

Small result tables and publication manifests are in `tables/` and
`manifests/`. Full measurement streams, captured scripts, native Nsight reports
and synthetic tensor snapshots are distributed as six assets in the
[GPU evidence archive](https://github.com/AICrossSim/PLENA_Simulator/releases/tag/gpu-captures-20260908).
They are release attachments so ordinary source checkouts do not download the
large profiler/tensor payloads. `catalog.json` pins every asset by SHA-256 and size.

Python 3.10 or newer, using only the standard library:

```bash
# List captures and download sizes.
python artifacts/gpu_capture_v1/download.py
# Download, verify, and unpack one complete public capture.
python artifacts/gpu_capture_v1/download.py --campaign nemotron_agentic_20260903 --output /tmp/plena-gpu --extract
# Download all six captures.
python artifacts/gpu_capture_v1/download.py --all --output /tmp/plena-gpu --extract
```

Every extracted capture has its own `SHA256SUMS` and publication manifest,
recording original and published file hashes, field changes and exclusions.
Original checksum lists, where retained, refer to original source bytes.
Capture scripts preserve their historical environment assumptions; inspect the
included environment metadata before starting a new GPU run.

## Scope of the public copies

Nemotron vocabulary token IDs and prompt/answer content are replaced by hashes
and lengths. Expert IDs, expert weights/counts, request/group membership,
timing observations and power measurements are preserved. Trace comparisons
record full-continuation and first-32-token agreement without publishing text.
The original strict `agentic_campaign` importer expects the original private
token arrays; these public-schema copies do not satisfy that private schema.
The public files support inspection and reanalysis of the measurement and
routing data; original archive hashes are retained for authorized source audits.

The September 3 energy values retain the archived sampling/window defect;
they are not corrected measured energy. See the
[energy reanalysis](../gpu_energy_reanalysis_v1/) for the qualified offline
approximation. Timing and routing came from separate runs, and B2–B16 routing
is grouped B1 replay, not a second batched routing capture.

Cache binaries, aborted numeric snapshots and unrelated process-list artifacts
are excluded as recorded in the manifests. Five Stage 2 Nsight reports containing
credentials in captured process metadata are also excluded; their exported
measurement tables remain available. Retained native profiler files and all nine
completed synthetic tensor snapshots preserve their original bytes.

The historical Mamba state-precision experiment uses block-128 MX8. It is not
the corrected block-8 weight traffic model or the BF16 recurrence contract.
