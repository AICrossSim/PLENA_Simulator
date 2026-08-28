# GPU evidence contracts

This directory keeps the small, reviewable outputs used to constrain the
Mamba/KDA workload models. Raw Nsight reports and checkpoints are deliberately
not committed.

| Data | What it establishes | What it does not establish |
|---|---|---|
| B200 formal campaign | Real Nemotron NVFP4 end-to-end baseline, six NCU layer-type cases, routing skew, real-shape KDA stages | PLENA cycles or RTL performance |
| RTX 5090 Mamba | Official Nemotron Mamba mixer shape, clean CUDA-event latency and NSYS stage split | Full-model latency; the NCU files are concurrency-qualified |
| B200 supplemental | Long-sequence Mamba state precision and real-shape Kimi MLA/LatentMoE component behavior | Language quality or a full Kimi checkpoint baseline |

`gpu_sources.json` pins four source archives and every imported file hash.
Collection-machine paths are removed from the checked-in CSV/JSON files.

Validate the checked-in contracts:

```bash
nix develop --command bash -c "just test-gpu-evidence"
```

Rebuild the compact files when the raw archives are available:

```bash
nix develop --command bash -c \
  "python3 -m analytic_models.performance.gpu_evidence_import /path/to/gpu/artifacts"
PLENA_GPU_ARTIFACT_ROOT=/path/to/gpu/artifacts \
  nix develop --command bash -c "just test-gpu-evidence"
```

The formal B200 profile and compact routing trace are rebuilt from an extracted
campaign by `b200_campaign_raw.py`. The RTX 5090 and supplemental contracts are
rebuilt directly from tar archives by `gpu_evidence_import.py`.

GPU time is a comparison baseline. It is never converted into PLENA cycles.
Until RTL supplies frequency, area, power, and calibrated operator latency,
these files cannot support a PLENA-vs-GPU speedup or token/J claim.
