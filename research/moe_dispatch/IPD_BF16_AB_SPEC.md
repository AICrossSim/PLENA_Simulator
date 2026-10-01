# BF16 fixed-budget dispatch study and multi-layer baseline

Status: implemented research candidate on `research/moe-ipd-bf16-ab` in Simulator
and Compiler. The proposal is not the full IPD-PF design in the earlier PDF.

## A: dispatch-only comparison

Question: with the existing fixed physical layouts and BF16 weights/inputs,
does a bounded high/low intensity policy beat both the strongest same-hardware
controller and the equal-budget single and homogeneous organizations?

The primary M6 layouts are exactly the prior frozen `6`, `3+3`, and `4+2`
configurations. Every policy uses 12,288 multipliers, 48 KiB W/return storage,
12 KiB X, 2 MiB arena, a shared 256 B/ns HBM link, 64 ns response, 256 shared
32 B credits, private W slots and banks, the same Shared-first descriptor
stream, and the same 4,352 B control reserve inside the arena. No W pooling,
credit expansion, I slicing, cross-layer prefetch or quantization is included.
The timing scope is a full analytical FFN from resident inputs/routes through
ordered combine, excluding Router, Attention, native Ramulator and silicon.

`dispatch="ipd"` reuses the current eight-descriptor charged snapshot, age
limit, potential-versus-due masks, atomic Next slot reservation, late binding,
service feedback and Current/Next protocol. It anchors the visible Shared task
or highest-Me task to the wider legal core; a lowest-Me visible companion goes
to the other core. An aged legal descriptor overrides the anchor rule after
eight bypass rounds. Equal-sized cores compare both orientations by predicted
pair finish. Decisions scan only arrived descriptors and pay the same bounded
read/comparison/commit service as joint. No unseen route or actual future
completion is read. Accepted DMA is neither canceled nor migrated.

The optional `ipd_credit_quotas` switch divides the same 256 credits across
the two cores. For a core with a Current or Next expert, predicted demand in
32 B requests is `ceil(128 * L / (16 * ceil(Me/M)))`, using the 4 KiB tile and
at least 16 operand-feed cycles per M group. Shares are proportional, each at
least 32 credits, and sum to 256. If the peer has no legal request, its unused
share can be borrowed. Quota limits are stored in bounded registers and
recomputed only when the selected task identity changes: Current takes
precedence, and Next is used only while Current is absent. Binding Next while
Current still runs therefore does not refresh the quota. Each update costs
eight cycles on the shared control port before new DMA requests may be picked.
A valid request stalled by DMA backpressure retains its identity; a credit is
released only after W landing. The formula is an
admission heuristic, not a bandwidth guarantee.

Five policies are paired on the same twelve archived DeepSeek-V2-Lite decode
route windows: dynamic, window LPT/ECT, joint, IPD without quotas and full IPD.
They run on all three M6 organizations, twice per point: 180 points/360 Rust
executions. Those twelve windows have already been examined in earlier work;
this is a controlled reanalysis, not a new held-out result. A later paper test
requires frozen request-disjoint inputs selected before tuning.

Required report fields are per-window FFN latency, configured HBM byte
utilization `weight_bytes / (256 * cycles)`, utilization relative to the
128 B/ns credit roof, useful MACs divided by nominal multiplier-cycle
capacity, spatial useful/issued MAC ratio, per-core arithmetic-active observer
fraction, per-core MAC-issue/HBM-accept intersection, per-core arithmetic-window/
HBM-accept intersection, W-not-ready time, control service, credit peaks and
core finish gap.
Arithmetic-active may overlap operand feed and is not by itself a MAC
utilization. Every policy must preserve useful MACs and weight bytes; all DMA
must drain and all capacities must remain valid. The research hypothesis is
met only if 4+2 beats both 6 and 3+3 under the same IPD policy on a declared
test set; an algorithmic gain over joint alone does not establish heterogeneity.

Run the archived comparison from the Simulator repository root:

```bash
export PLENA_DISPATCH_COMPILER=/absolute/path/to/PLENA_Compiler/research/moe_dispatch
cargo build --release --locked --manifest-path research/moe_dispatch/rust/Cargo.toml
python3 research/moe_dispatch/ipd_ab_study.py run --workers 6 --output /tmp/plena-ipd-ab
python3 research/moe_dispatch/ipd_ab_study.py report --output /tmp/plena-ipd-ab
```

## B: sequential multi-layer baseline profile

The Rust `--multilayer-baseline input.json output.json` entry point accepts at
least two individually compiled FFN workloads, a common fixed configuration,
and optional caller-supplied gap cycles/HBM bytes between layers. It resets W,
credits and execution state at each boundary, so this is a **no cross-layer
prefetch** baseline. Gap cycles/traffic are external inputs: the tool does not
pretend to execute Attention or Router. A zero gap means back-to-back FFNs,
not full-model latency. The fixed HBM byte-bandwidth validity of supplied gap
traffic is checked.

For each layer it emits absolute start/end cycles, full single-layer report
and 1,024-cycle profile bins by default. Bins contain accepted HBM bytes,
spare configured-byte capacity, credit occupancy/full cycles, per-core free W
slots, Current occupancy, arithmetic-active observer cycles and weight waits.
A per-core slot+credit+bandwidth intersection in the layer tail is an
**opportunity upper bound**, not proof that prefetch is harmless or useful.
The tail horizon defaults to 65,536 cycles and can be changed in the input.
Reports are repeated twice byte-for-byte by the preparation driver.

`multilayer_profile.py prepare --capture CAPTURE.npz --layer-ids 12,13,14`
selects one deterministic, request-disjoint decode window and extracts those
adjacent layers for the same requests. It requires the original multi-layer
capture file, which is not included in the published Git archive. The driver
also accepts `--workloads-json` for protocol tests; caller-supplied workloads
are not automatically certified as true adjacent layers. A supplied gap must
be justified by separate layer/Attention measurements before any full-model
performance claim. The current worktree has only archived layer-13 windows,
so the committed small two-layer example is a protocol validation, not a
measured multi-layer DeepSeek profile.

The self-contained protocol replay is:

```bash
python3 research/moe_dispatch/multilayer_profile.py prepare --toy --shape 4+2 --policy ipd --output /tmp/plena-multilayer-toy
python3 research/moe_dispatch/multilayer_profile.py run --output /tmp/plena-multilayer-toy
```

Neither A nor B enables cross-layer prefetch. A future C stage must use the
same multi-layer inputs and hardware on every organization, compare no-prefetch
and prefetch cases, account for competition with current-layer and intervening
HBM traffic, and keep routed-expert-ID prediction out of the deterministic
Shared-only variant.
