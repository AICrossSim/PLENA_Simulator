# Captured-route analytical results

Times are **milliseconds**, at an assumed 1 GHz. These are analytical MoE-layer
times with supplied routing/input, not native HBM, GPU, silicon or full-model
measurements. Metadata was captured from DeepSeek-V2-Lite-Chat BFCL, first MoE
layer, last-token prefill: H=2048, routed F=1408, shared F=2816, top-k=6.

## Same policy: G4, dynamic whole-expert dispatch

| Batch | Single 6 | Homogeneous 3+3 | Heterogeneous 4+2 |
|---:|---:|---:|---:|
| 2 | 1.248675 | 1.2467 | 1.246672 |
| 4 | 1.391709 | 1.476915 | 1.509512 |
| 8 | 2.039253 | 2.095387 | 1.867423 |
| 16 | 3.402242 | 3.537016 | 3.379115 |

## Best tested point per batch and organization

Each organization receives the same G/policy search. This selects the best point
after observing this window; it is not one deployable policy or a held-out result.

| Batch | Single 6 | Homogeneous 3+3 | Heterogeneous 4+2 |
|---:|---:|---:|---:|
| 2 | 1.248121 | 1.242879 | 1.245761 |
| 4 | 1.391057 | 1.476915 | 1.451762 |
| 8 | 2.038503 | 1.908529 | 1.867423 |
| 16 | 3.401492 | 3.537016 | 3.379115 |

B8 best-tested 4+2 reduces latency by 2.15% versus best-tested 3+3 and 8.39%
versus single. B4 favors single; B16 hetero versus single is below 1%. A memory
response-latency change can reverse the organization ranking. These results do
**not** establish that heterogeneous cores are universally necessary.

B8 all organizations read 181.5 MiB of weights and stage 66 MiB of X with G4.
Shape utilization and layer wall time are different metrics; waiting/service
counters overlap and must not be summed into an invented latency breakdown.

## Evidence files

- `default_dynamic.csv`: the common policy, 12 points.
- `architecture_best.csv`: selected configurations and corresponding counters.
- `b8_ablation.csv`: same-G policy/splitting ablation.
- `sensitivity_key.csv`: bandwidth/latency assumptions and ranking counterexamples.
- `reference_reports.json`: historical SHA-256 of the full raw report for every
  common-policy point; each original pair was verified identical during packaging.
- `provenance.json`: original CSV hashes and historical campaign counts.
- `reproduction.json`: fresh branch-packaging replay receipt, generated only after
  all 12 points x 2 repeats match the frozen full reports.

The original campaign had 351 main points x 2 and 9 trace points x 2. The present
branch does not claim to rerun all 720 simulations during packaging. Reproduce
the 24-run parity check with the commands in the parent README; run the unfiltered
suites for a new full campaign. The full original raw archive remains local and
is not required to execute the checked-in common-policy regression.
