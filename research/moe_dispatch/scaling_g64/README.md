# G=64 fixed-configuration compute scaling experiment

This experiment preserves all earlier frozen results. It evaluates Shared +
all active Routed experts in 135 archived real DeepSeek-V2-Lite windows.
There is no hardware search or per-batch hardware selection in this study.
The 21 fixed configurations include the requested compact single/homogeneous,
P/E heterogeneous, alternative P, independently executing E lanes, and controls
giving homogeneous hardware the same stream-lane capability.

## Contract and corrections to the proposed experiment

* Total multipliers are exactly 65,536 / 131,072 / 262,144. Every two-group
  configuration allocates exactly half to each group. This is iso-MAC,
  not a synthesized equal-area comparison.
* OS arithmetic with ideal zero-latency operand supply; finite descriptor
  issue and four-cycle next-context setup are retained. 1 GHz converts
  modeled cycles to nanoseconds, and is not a proven implementation Fmax.
* `G=MLEN/BLEN`. The explicit physical expansion is `BLEN=PM`,
  `MLEN=PM*64`, column replicas `PN/PM`. Thus `G=64` does NOT guarantee
  all K dimensions are unpadded: for example, PM=4 implies MLEN=256,
  which does not divide 1408. A 64-input reduction tree has six binary
  levels; the model charges pipeline stages, not a guaranteed single cycle.
* `PN<=256` is a declared constraint for compact cores. A PN512 stream
  core has different interconnect/control assumptions. The extra homogeneous
  stream controls expose this difference instead of attributing it to
  heterogeneity alone.
* A monolithic `4x512x64` E core has PM=4 and violates the narrow-PM<=2
  condition. It is retained as the requested diagnostic alternative. The
  separate `2*(2x512x64)` variant has independently executing lanes, not
  merely twice as many rows consuming one expert. Likewise at128K,
  `2x512x64` differs from `2*(1x512x64)`.
* There is one depth8 FIFO per logical P/E group. Independent lanes add
  current/shadow contexts but do not silently increase FIFO capacity. The
  pooled queue supplies at most one lane per clock. Its completion estimator
  greedily projects at most eight queued descriptors onto worker timelines;
  that extension has not been synthesized or validated by the old two-worker
  RTL. It is separately checked as a software timing model.
* The initial run exposed a legacy Python control bug: an own-queue claim
  followed by a cross-pull could consume two heads from one FIFO in a cycle.
  RTL permits one. The old replay bench read expected claim clocks but did
  not compare them. The new runtime charges one FIFO dequeue per clock in
  every organization, and the new RTL bench checks each claim explicitly.
  Initial results remain preserved and marked SUPERSEDED; only the corrected
  run is a deliverable. The 64K regression found32 changed claim traces among
  2160 comparisons, with all2160 assignments/start/end/makespans unchanged.
* Whole experts are the scheduling unit. All three GEMMs of one expert run
  on its assigned worker; Shared is included once and has Me=Batch.
  There is no unmodeled within-expert partition or cross-expert packing.
* The affinity threshold is fixed at2. Full M-DPU compares predicted finish
  times; affinity breaks a tie. It may assign Shared to E or Routed to P.
  This is the existing guarded M-DPU contract, not hard affinity.

## Timing and metrics

For `(M,K)*(K,N)`, a frame has
`records=ceil(M/PM)*ceil(N/PN)`,
`feed=ceil(K/MLEN)*BLEN`,
`drain=BLEN+9+log2(G)*tree_stage+2`.
The primary pipelined model uses `records*feed+drain`.
Gate/Up form one doubled-record phase; Down is a second phase.
The sensitivity models double the registered tree-stage latency, or drain
after each output record. These are declared analytical timing contracts,
not native datapath measurements or gate-level timing closure.

Each window is repeated twice and its entire serialized result must match.
Latency is GM within each batch, followed by equal-weight GM of the seven
batches. Speedup is baseline/candidate on paired windows. Whole-chip useful
utilization is useful MACs/(total multipliers*layer makespan), including all
padding, idle, arithmetic drain and controller bubbles. Group useful-work
shares and idle rates are averaged within batch then equally across batches.
For independent E lanes, idle is MAC-weighted and group finish is the maximum
lane finish, not the sum of busy cycles.

The four scheduling policies are fixed static affinity, old forced spill,
guarded M-DPU without context preconfiguration, and guarded M-DPU full.
All four run in the primary timing model; the two timing sensitivity models
run the full policy. No result is required to favor a particular organization.

## Reproduce

Run from the repository root with Python3.11 and numpy/openpyxl installed:

```bash
export TMPDIR=/tmp PYTHONDONTWRITEBYTECODE=1
python3 -m unittest research.moe_dispatch.scaling_g64.test_runtime -v
python3 -m research.moe_dispatch.scaling_g64.run --output /path/to/new/results
python3 -m research.moe_dispatch.scaling_g64.audit --output /path/to/new/results
python3 -m research.moe_dispatch.scaling_g64.validate_rtl --output /path/to/new/results
python3 -m research.moe_dispatch.scaling_g64.report --output /path/to/new/results
python3 -m research.moe_dispatch.scaling_g64.mechanism --output /path/to/new/results
python3 -m research.moe_dispatch.scaling_g64.plot --output /path/to/new/results
```

Use a new output directory: the runner refuses to overwrite records.
The deliverable uses `/home/mcl123/plena-outputs/scaling_g64_20261015_corrected` because
scratch filled during task startup. Old archived session files were moved
with SHA256 verification and original-path symlinks, without deleting history.

Source: [PLENA v3, III-B flattened systolic array](https://arxiv.org/html/2509.09505v3#S3.SS2).
The paper does not establish this custom BF16 scaling matrix's physical
feasibility, and these budgets are not asserted to equal specific TPU products.
