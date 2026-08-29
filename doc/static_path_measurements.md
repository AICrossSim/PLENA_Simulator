# Static Mamba/KDA path — measurements

Everything here was measured on `feat/static-kda`. The two kinds of number are
kept apart on purpose, because only one of them decides anything.

* **Hard facts** — static instruction counts, dynamic instruction counts, HBM
  bytes, memory footprints. Properties of the compiled artifact. Reproducible,
  and admissible as CI gates.
* **Uncalibrated model output** — cycles, µs, TPOT. Produced by a timing model
  that has never been calibrated against silicon. Labelled as such on every
  line, and **not compared against any other uncalibrated model output**: two
  numbers from two uncalibrated models decide nothing between them.

There is a second split inside the hard facts, and this document got it wrong
four times before 2026-08-27. A **static** count is the size of the program
image. A **dynamic** count is the number of instructions issued, with every
hardware loop expanded by its trip count. Both are exact and reproducible;
only the second says anything about cost. A sweep the `V_FMA_VF` conversion
collapsed into a hardware loop is a handful of words in the image and runs its
body up to 192 times.

Every claim here about what something *costs*, about which kernel dominates,
or about what an alternative lowering would save is now made against the
dynamic count. The four that were not:

| claim | from the image | from the issue stream |
|---|---|---|
| projection gather, share of a KDA layer | 0.07% | **0.94%** |
| convolutions vs mixer, share of a layer | 58% / 42% | **12% / 87%** |
| what `V_FMA_VF` bought | 97× | **3.6×** |
| Kimi K3 head vs `64 × 64`, for 4× state | 1.6× | **2.3×** |

The image counts are not retracted — they are the right instrument for the CI
gates, which exist to catch a sweep falling off the hardware-loop path, and
that is an image failure no numeric test can see. What is retracted is reading
any of them as a cost.

---

## 1. Hard facts

### The instructions that were added, and what they bought

Three, all ordinary fixed-function ops with their operands named in the
instruction word:

| | | |
|---|---|---|
| `V_SOFTPLUS_V` | `0x39` | `log(1 + exp(x))` over a vector row, for Mamba's `dt` |
| `S_MAP_FP_V` | `0x3A` | one VRAM row into `VLEN` consecutive FPRAM slots |
| `V_FMA_VF` | `0x3B` | `Vector[rd] += Vector[rs1] * fp_reg<rs2>` |

`V_FMA_VF` is the one the numbers below are about. It replaced the
`copy + multiply + add` triple that both recurrences used.

Instructions for one head, image and issue stream side by side. The image
columns are the ones this document used to carry alone:

| | image before | image after | issued after |
|---|---|---|---|
| KDA decode, `key_dim` 8 | 505 | **76** | 174 |
| KDA decode, `key_dim` 16 | 961 | **76** | 302 |
| KDA decode, `key_dim` 64 | 3,697 | **76** | 1,070 |
| KDA decode, `key_dim` 128 | 7,345 | **76** | 2,094 |
| Mamba decode, `state_size` 4 | 208 | **54** | 77 |
| Mamba decode, `state_size` 8 | 376 | **54** | 121 |
| Mamba decode, `state_size` 16 | 712 | **54** | 209 |
| Mamba decode, `state_size` 32 | 1,384 | **54** | 385 |

The reason the image collapsed is not the saved multiply. The triple staged
every row through a scratch row, and because it was the *same* row each
iteration the destination never formed an arithmetic progression — so the sweep
could not become a hardware loop and was unrolled in full.

**The cost did not go from linear to constant.** This document said it did, and
put a factor of 97 on it; both statements are about the image. Expanded, the two
kernels are still exactly linear in the contracted dimension, which they have to
be — a recurrence touches every state row and an encoding cannot change how many
rows there are:

```
KDA:    issued = 16 * key_dim    + 46
Mamba:  issued = 11 * state_size + 33
```

Both fit their four measured points exactly. The pre-conversion slopes were
about 57 and 42, so **what `V_FMA_VF` bought is a 3.6× smaller coefficient on a
line that stays a line** — together with an image that stops growing, which is
what makes a whole-model program image tractable and is worth having on its own
terms.

Both facts are real and they are not the same fact. 97× is the image; 3.6× is
the time. `test_the_conversion_reduced_the_slope_not_the_linearity` in
`test_instruction_budget.py` pins the fits and asserts the two never converge
again.

### One KDA layer, Kimi K3 shape, `mlen` 64

| | static (image) | dynamic (issued) | dynamic share |
|---|---|---|---|
| three convolutions | 53,757 | 59,409 | 12.1% |
| mixer (gates + normalisation + recurrence) | 39,526 | 428,622 | 87.0% |
| projection gather | 70 | 4,650 | 0.94% |
| **layer** | **93,353** | **492,681** | |
| **× 93 layers** | **8,681,829** | **45,819,333** | |

Two columns because they answer different questions and were conflated here
until 2026-08-27. **Static** is the program image, and it is what the
`V_FMA_VF` conversion bought — an image that does not grow with `key_dim`. It
is the right number for a budget gate against accidental unrolling, and the
wrong number for pricing a kernel, because a hardware loop whose body runs 192
times costs 192 times what the image shows.

The larger correction is the ordering. The convolutions are 58% of the image and
**12% of the work**; the mixer is 42% of the image and **87% of the work**. This
document and the compiler's budget file both called the convolutions "the larger
half of the layer" on the strength of the image, which points optimisation effort
at the smaller kernel by a factor of seven. The cause is structural: the
convolutions are a four-tap FIR whose taps are separate weights, so the image
carries an instruction per tap per block and almost nothing loops, while the
mixer is the recurrence the `V_FMA_VF` conversion collapsed into hardware loops
running once per key block per head — a small image over a large trip count.

The gather is the case where the difference bites: 70 static is 0.07% of the
layer, and the earlier version of this table reported that as the gather's
cost. **The gather issues 4,650 instructions, 0.94%** — thirteen times more.
The conclusion the 0.07% was reaching for still holds at 0.94% (no mechanism
that only makes the split faster can move the layer by more than a percent),
but 0.94% is a cost, not a rounding error, and it is the number any alternative
must be compared against.

### The gather is not forced by packing, and is not forced at all

The 0.94% invites the question of whether a different projection lowering
avoids it: five separate projections, each writing straight to where its
consumer wants it, instead of one packed tile plus a split. Both were built and
counted, at `mlen` 64, `blen` 4, through `linear_projection`:

| | matmul static | matmul issued | `M_MM` | gather issued |
|---|---:|---:|---:|---:|
| packed | 1,090,417 | 13,876,267 | 21,560 | 4,650 |
| separate | 1,107,395 | 13,893,245 | 21,560 | 4,650 |

Identical matmul work, 0.12% worse for the separate form from the extra
per-projection setup, and the same gather. **There is no crossover in `blen` to
find** — the premise was wrong.

The reason is the writeback granularity. `M_MM_WO` writes a `blen × blen`
sub-tile and the loops around it cover `mlen / blen` column groups, so the
smallest thing a projection can lay down is `blen` token-rows by `mlen` lanes:
column block `c` lands at row `c * blen` however the weight matrices are
arranged. The consumers want one token's blocks as consecutive rows. **The
mismatch is exactly `blen`, and it is a property of the matrix writeback rather
than of the packing**, so no rearrangement of weights touches it.

What removes it is not moving the data. Consumers take explicit row indices,
any arithmetic progression collapses into one hardware loop, and the step is an
`S_ADDI_INT` immediate — so a step of `blen` costs what a step of 1 costs.
`kda_conv_step_v0` now takes `x_new_row_base` / `x_new_row_stride`, and feeding
it the tall projection view removes **4,041 of the 4,650** issued instructions,
the q/k/v share, for no instructions and no hardware. Gate and beta reach a
different consumer and still gather.

What stays is gate and beta: **1,176 issued, 0.24% of the state-engine path.**
They reach `kda_decay_scalars_v0` and `kda_beta_scalars_v0` rather than the
convolution, and the two are not symmetric. beta is single-tile work throughout
— sigmoid in place, then `tile_row_to_fpram`, which addresses FPRAM by position
in the row list rather than by row index — so a stride drops straight in, for 18
instructions. gate opens with `tile_row_add(gate, dt_bias)`, and underneath
`_emit_tile_row_vector_op` derives one address progression and steps both
operands by it; a strided gate against a dense `dt_bias` needs that emitter to
take `(dst, src)` row pairs the way `_emit_tile_row_fma` already takes triples.
That is a change to a row-op emitter every kernel on the branch shares, for
0.24%. Laying `dt_bias` out at the same stride instead needs no ISA change but
pads it from 192 rows to 768 — 74 KiB per layer, 6.8 MiB more HBM traffic per
token across 93 layers, to save 1,176 instructions. Both trades are bad, so
they stay and the test records the price.

### The input projection, which no budget covered

Building it turned up the largest kernel in the layer and the fact that nothing
gated it — `test_kda_layer.py` loads the projection's result instead of
computing it, so it was never in the lowering. At Kimi K3, `mlen` 64, and the
capacity the shipped transactional config implies:

| | static | issued |
|---|---:|---:|
| input projection | 389,717 | 4,967,367 |
| state-engine path (conv + mixer + gather) | 93,353 | 492,681 |

**Those two rows must not be added, ranked, or turned into a percentage.** An
`M_MM` is a `64 × 64 × 64` matmul and a `V_FMA_VF` is a 64-lane vector
operation; counting them in one unit says nothing about time, and doing so
would be the same error as §1's first four. The row is here because a kernel
that size should not sit outside every budget, and it now has one. Comparing
the gather against the state-engine path stays legitimate — both are vector
work — and that comparison is the 0.94% above.

A second thing fell out. **The compiler's `mram_tile_capacity` defaults to 4
and the configured machine has 64.** Nothing derives one from the other:
`MatrixSram::new(tile_size = MLEN, depth = MATRIX_SRAM_SIZE)` keeps
`depth / tile_size` tiles, so MLEN 64 over `MATRIX_SRAM_SIZE` 4096 is 64 tiles,
while `PlenaCompiler` takes `mram_tile_capacity: int = 4` from its own
signature. Kimi K3's projection contracts over 112 k-tiles and
`linear_projection` re-streams the weights once per capacity-sized chunk:

| capacity | | static | issued |
|---|---|---:|---:|
| 4 | 32 KiB — compiler default | 1,090,417 | 13,876,267 |
| 16 | 128 KiB | 524,467 | 6,680,617 |
| 64 | 512 KiB — configured machine | 389,717 | 4,967,367 |
| 112 | 896 KiB — whole contraction | 362,767 | 4,624,717 |

**A caller who did not name a capacity got 2.79× the projection the configured
machine would run**, and nothing said so. Fixed: the default is now
`MATRIX_SRAM_SIZE // mlen`, the same arithmetic on the same two numbers as the
emulator's `MatrixSram::new`.

Chasing that turned up why the configuration had never reached the compiler at
all. `PlenaCompiler`'s docstring said HLEN, BROADCAST_AMOUNT and the prefetch
amounts default to `plena_settings.toml`; none of them ever did, for two
independent reasons at once:

1. The reader asked for `[BEHAVIOR].CONFIG`, and the file has no such table —
   it ships `[MODE]`, `[ANALYTIC.*]` and `[TRANSACTIONAL.*]`.
2. `load_toml_config` required the third-party `toml` package and raised
   `ImportError` without it. `toml` is not a declared dependency and CI
   installs `pytest pyyaml`, so it failed there too — and the `except
   Exception` around the call turned "cannot read the config" into "the default
   is correct".

Either alone would have been enough, and together they made a silent failure
that looked exactly like a working default. The reader now goes through
`tomllib`, and searches `BEHAVIOR` first — so a file that does define one still
wins — then the section whose `MLEN` is the one being compiled for.

**Matching on `MLEN` rather than on `[MODE].active` is deliberate.** `active`
selects the machine the *simulator* models, not the one the compiler emits for,
and the two differ today: the shipped file is `analytic` (MLEN 2048) while every
program on this branch is compiled at 64 and executed by the transactional
emulator, whose section declares 64. Reading `active` would hand a 64-wide
compilation the 2048-wide machine's numbers.

At `mlen` 64 the fix also moves HLEN from 64 to 16 and BROADCAST_AMOUNT from 1
to 4. Neither reaches an emitted program — `hlen` is read only by a
packed-attention precondition whose one test sets both attributes by hand at
`mlen` 256, and `broadcast_amount` is read nowhere outside the constructor. The
prefetch and writeback amounts do reach the emitters, and are 4 in that section,
which is what they already were. Every count in this document is unchanged: the
state-engine path does not go through MRAM matmuls.

At an `mlen` no section declares — the 8 the budget tests use — the capacity
stays 4 and nothing changes. At `mlen` 2048 the `ANALYTIC` config gives
`256 / 2048 = 0` tiles and the constructor now raises naming both figures,
rather than substituting a number nobody chose: a contraction split into
zero-tile chunks does not terminate. Nothing reads that mode today, but the
number in the file is still not a capacity anyone picked.

This is the **state-engine path alone** — the projections, MoE blocks, norms and
embeddings are not in it. Any figure that includes those is measuring a
different thing and the two must not be put side by side.

### Where the issue slots go, and what an addressing mode would buy

The layer table above says a Kimi K3 layer issues 492,681 instructions. Per
opcode, **124,428 of them compute and 368,253 do not** — 25.3%.

The shape of the recurrent sweep is what does it. Every `V_FMA_VF` sits in a
body of five, of which one computes:

```
S_LD_FP     f1, gp3, 0        <- the scalar for this row
V_FMA_VF    gp1, gp2, f1, 0   <- the only instruction that computes
S_ADDI_INT  gp1, gp1, mlen    <- advance the destination
S_ADDI_INT  gp2, gp2, mlen    <- advance the source
C_LOOP_END  gp4               <- not counted; zero-overhead loop boundary
```

| | conv ×3 | mixer | gather | **layer** |
|---|---:|---:|---:|---:|
| issued | 59,409 | 428,622 | 4,650 | **492,681** |
| arithmetic | 18,432 | 104,456 | 1,540 | **124,428** |
| share | 31.0% | 24.4% | 33.1% | **25.3%** |

That 25% is the figure a dedicated recurrence engine beats by construction —
one instruction covering a whole head has no instruction overhead at all. So it
is worth knowing how much of the gap is *encoding* rather than *architecture*.

A **post-increment operand** — the pointer carrying its own stride, the way
`_emit_tile_row_fma` already carries independent row progressions — folds
`S_ADDI_INT gpN, gpN, step` into the instruction that consumes the pointer. It
only works where the register has a consumer in the same loop body, so
`self_advance_counts` splits them and prices the saving against the ones that
can really disappear. **Every one of them qualifies: 215,634 foldable, 0 not.**
86% fold into a single consumer; the rest are read more than once in the body
and would need the increment on the last use.

| | issued | arithmetic share | vs today |
|---|---:|---:|---:|
| today | 492,681 | 25.3% | — |
| **P1** post-inc on the vector operands | 277,047 | 44.9% | **1.78×** |
| **P2** P1 + the FPRAM scalar auto-advancing | 174,129 | 71.5% | **2.83×** |

Over 93 layers: 45.8M issued instructions falling to 16.2M.

P2 is an addressing field on instructions that already exist. What it competes
with is a dedicated recurrence datapath with its own banked state SRAM plus a
layout engine to feed it — the two are not close in cost, though neither has
been synthesised so neither has an area number.

**What neither proposal touches**, and this is the limit that matters: KDA's
decay is channel-wise on the key axis, so the operand is `key_dim` lanes wide —
6.2% of `VLEN` 2048 (§5). That is a **width** mismatch, not an encoding one,
and no addressing mode reaches it. The two inefficiencies are independent and
they multiply: at `VLEN` 2048, 71.5% of instructions computing on 6.2% of the
lanes is 4.4% of the machine doing work; at a `VLEN` that matches the operand
it is 71.5%. **That is the one place a purpose-built engine wins on
architecture rather than on encoding**, and an addressing mode does not answer
it — choosing the width, or changing the state layout, would.

### Memory

| | |
|---|---|
| FPRAM, one KDA layer, Kimi K3 | **492 slots** of 512 (`FP_SRAM_DEPTH`) |
| FPRAM, 93 layers | **492** — it is scratch, reused per layer |
| HBM, per layer | **2,336,832 bytes (2,282 KiB)** |
| HBM, 93 layers | **217,325,376 bytes (0.202 GiB)** |
| emulator `HBM_SIZE` | 16 GiB |

The footprint fits the emulator's flat allocation with three orders of magnitude
to spare, which is why the program executes rather than being described. That is
a property worth stating on its own: the emulator preloads HBM from a flat file
starting at offset 0, so a layout whose regions are spread across a wide address
space needs an allocation and a file as large as the span, whatever the live
data comes to.

The HBM figure counts everything a layer stages — conv history and weights, the
recurrent state, `dt_bias` and the projection tile — and excludes only the
projection *weights*, which any implementation needs and which are
model-inherent. A first version of this number was 331,776 bytes, seven times
too low, because the measuring helper put the state and the projection in VRAM
while the emulator test staged them from HBM. The two now agree.

The FPRAM number is tight and was nearly wrong: the first accounting came to 620
slots because `beta` was counted as one slot when `S_MAP_FP_V` writes a padded
row (128 at Kimi), and `rate` was omitted entirely. `decay` and `q_hat` are never
live at the same time, and sharing one window is what brings it to 492.

### Numerical agreement, on the transactional emulator

Against the CPU reference, compiled and assembled and executed as machine code:

| | max abs error |
|---|---|
| cumulative decay | 1.95e-03 (one bf16 ulp) |
| UT transform | 1.95e-03 (one bf16 ulp) |
| prefill, one chunk of 16 | 1.13e-03 out / 1.32e-02 state |
| prefill, three chunks chained | 6.71e-04 out / 8.30e-03 state |
| state layout transpose | **0.00** |
| **assembled layer** | **2.14e-04 abs / 4.3% mean relative** |
| **four layers back to back** | **2.75e-04 abs / 4.4% mean relative** |

The layer figures are quoted both ways on purpose. Its outputs are O(5e-3), so
an absolute error of 2e-4 is **4% relative** — and an earlier version of this
table quoted only the absolute number against a test whose `atol` was 2e-2,
which every golden value already satisfied. That test passed on an all-zero
output tile. It is relative now, and 4.3% is what bf16 delivers on the
read-out's 128-term contraction (`sqrt(128) * 2^-9 = 2.2%` before the rest of
the chain). The lowering itself is right to **5e-4 in float64**, which the ISA
interpreter checks separately — the 4% is precision, not a defect.

---

## 2. The banking question

**Answer: stalls ≈ 0. The vector path is limited by instruction issue, not by
SRAM.**

`analytic_models/performance/vector_sram_banking.py` extracts every hardware
loop's pointer stride from the *emitted assembly* — not from a description of
the algorithm — and models row-interleaved banking, where row `r` lives in bank
`r % B` and a stride-`s` sweep visits `B / gcd(s, B)` distinct banks.

| kernel | loops | B=2 | B=4 | B=8 | B=16 | B=32 |
|---|---|---|---|---|---|---|
| KDA recurrence (Kimi K3) | 20 | 0.6% | 0.9% | 1.0% | 1.1% | 1.1% |
| KDA mixer, one head | 48 | 1.2% | 1.8% | 2.1% | 2.3% | 2.3% |
| Mamba-2 decode step | 7 | 1.0% | 1.5% | 1.8% | 1.9% | 2.0% |

The dominant loops are **stride 1 over 128 trips** (KDA) and stride 1 over 64
(Mamba). A stride-1 walk visits every bank in turn, so it never conflicts. The
1–2% residue is entirely single-trip loops with a pinned pointer.

So there is no evidence for a Vector SRAM bank-mapping mode. Note also that the
emulator does not model banking at all — `VectorSram` is a flat row vector — so
this is analytic, and a word-interleaved mapping would make the question moot
regardless.

### What the follow-up actually is

Half the dynamic instruction stream is pointer arithmetic:

| kernel | dynamic | `S_ADDI_INT` | `S_LD_FP` | `V_FMA_VF` |
|---|---|---|---|---|
| KDA recurrence (Kimi K3) | 4,179 | **50.0%** | 24.6% | 18.4% |
| KDA mixer, one head | 4,398 | **50.2%** | 23.9% | 17.5% |
| Mamba-2 decode step | 730 | **46.4%** | 26.6% | 17.7% |

Under a fifth of the stream is arithmetic. Every hardware loop advances its
pointers with an explicit `S_ADDI_INT` per pointer per iteration, and reloads its
FPRAM scalar with `S_LD_FP`. **Post-increment addressing on the vector operand
pointers, and an auto-advancing FP pointer, would together remove up to
three-quarters of the dynamic instructions on the recurrent path.** That is the
next instruction-set question, and it is a much larger effect than banking.

---

## 3. Uncalibrated model output

**Every number in this section comes from a timing model that has never been
calibrated against silicon. They are not performance claims.**

| | UNCALIBRATED |
|---|---|
| KDA chunked prefill, one 16-token chunk, one head | 173,939 cycles |

No `µs` or TPOT figure is given, because converting cycles to time requires a
clock the model does not have, and quoting one would give the number a
credibility it has not earned.

---

## 3b. Prefill past one block: `key_dim` and `value_dim` above `mlen`

Chunked prefill used to refuse `key_dim > mlen` outright, which is why Kimi K3's
128-wide head could not prefill at `mlen` 64. It now runs, and so does
`value_dim > mlen`, which nothing had ever exercised.

### What changed

The key axis moved from **rows** onto **lanes**. A `[chunk, key_dim]` tile is now
`chunk` rows spread across `key_dim / mlen` column blocks, instead of
`chunk * key_blocks` rows one block wide.

That is what the projections want. Five of the seven products contract over key,
and `vram_sub_projection_asm_impl` already walks a VRAM operand's column blocks
at stride `physical_rows * mlen`, accumulating inside the systolic array -- so a
key axis on lanes contracts across as many blocks as it needs with **no new
instruction and no explicit sum**. No projection call changed. The row ops pay
instead, one sweep per column block rather than one over the folded rows, and
rows within a block are still consecutive so each sweep is still a hardware loop.

### Measured on the transactional emulator, as machine code

Every case at 100% of values inside tolerance -- see the note on the 90% rule
below for why that phrasing matters.

| case | `key_dim` × `value_dim` | max abs error |
|---|---|---|
| cumulative decay | 64 × 64 | 1.95e-03 |
| cumulative decay | 128 × 64 | 1.95e-03 |
| state layout transpose | 64 × 64 | **0.00** |
| state layout transpose | 128 × 128 | **0.00** |
| prefill, out | 64 × 64 | 1.13e-03 |
| prefill, out | 128 × 64 | 6.83e-04 |
| prefill, out | 128 × 128 | 6.71e-04 |
| prefill, state | 64 × 64 | 1.32e-02 |
| prefill, state | 128 × 64 | 5.13e-03 |
| prefill, state | 128 × 128 | 9.28e-03 |
| three chunks chained, out | 128 × 128 | 6.10e-04 |
| three chunks chained, state | 128 × 128 | 7.32e-03 |

`128 × 128` is Kimi K3's head at `mlen` 64: two key blocks and two value blocks.

### Instructions, one head, `mlen` 64

Image, then issue stream:

| chunk | 64 × 64 | 128 × 64 | 128 × 128 |
|---|---|---|---|
| 4 | 818 / 18,904 | 1,237 / 25,898 | 1,470 / 43,851 |
| 8 | 1,039 / 19,246 | 1,547 / 26,366 | 1,792 / 44,343 |
| 16 | **1,476** / **20,120** | 2,157 / 27,490 | **2,426** / **45,515** |

The single-block column is **unchanged** by the layout move (1,476 against a
1,500 measurement before it), so lifting the limit cost nothing at the shape that
already worked.

Kimi K3's head against a `64 × 64` one, for 4× the state: **1.6× by image,
2.3× by issue stream**. An earlier version of this note gave only the first and
read it as the cost of quadrupling the state. The kernel is sublinear in the
state either way, which was the point being made, but by 2.3× rather than 1.6×.

The `chunk` column carries a second divergence worth having. Going from chunk 4
to chunk 16 costs **1.65× of the image and 1.04× of the issue stream** — four
times the tokens per chunk for 4% more work. Prefill is spill-dominated, so the
per-chunk cost is the fixed traffic of filling and storing `mlen`-sized tiles
and barely moves with how many tokens ride along. That makes the `chunk` cap of
17 — set by bf16 range on `1/A`, not by the lowering, see §5 — more expensive
than the image suggests: every token that cannot join a chunk pays the fixed
cost again. `test_kda_prefill_structure.py` pins both ratios.

### One real defect, and the tolerance that nearly hid it

`out = scale * (...)` ended with a single `tile_row_mul_fp_broadcast`, and a row
op reaches **one column block per call**. At `value_dim` 128 that left the upper
half of every token unscaled -- `sqrt(key_dim)` = 11.3x too large.

It reported **PASSED**. `check_mem`'s `allclose_pass` is `match_rate >= 90.0`,
and 94.68% of the values sat inside the case's `atol` of 5e-2 because the data is
of order 1e-3: a tolerance fifty times the signal. Splitting the comparison by
value block is what showed it -- block 0 at 6.4e-04, block 1 at 1.4e-01, a mean
relative error of 11.28 against a predicted 11.31.

The stage test now restates the bar at **100%** on the same numbers
(`_assert_every_value_within_tolerance`). `check_mem` is shared by many callers
and its 90% rule is left alone. Under the new bar, four mutations of the
column-block loops die:

| mutation | match rate | would the 90% rule have caught it? |
|---|---|---|
| output scale on block 0 only *(the real defect)* | 94.68% | **no** |
| `k_hat`/`q_hat` on block 0 only | 0.00% | yes |
| `A_C * state` on block 0 only | 55.79% | yes |
| `k / A` on block 0 only | 85.80% | yes |

Only one of the four needed the stricter bar, and it is the one that actually
happened.

The layer cases moved to `atol` 2.5e-04 with the same `rtol` 0.12 for the same
reason from the other side: three of their sixty-four outputs are near 1e-03,
bf16 delivers about 1.8e-04 absolute whatever the value, and a **pure relative**
bound cannot be met there. They had been passing at 95.31% on the 90% rule rather
than on the tolerance. The floor keeps the check real: an all-zero output scores
**1.6%** against it.

### Still refused

`key_dim` and `value_dim` must each be a whole multiple of `mlen`
(`kda_key_blocks` and `kda_blocks` enforce it), and `chunk` must still fit one
block. The state transpose additionally requires its output to be exactly
`key_dim` rows when there is more than one value block, because that is what
makes its column-block layout coincide with the row indexing decode uses.

---

## 4. What is not measured

* **No real checkpoint.** Every number above uses synthetic weights. Binding to
  published Kimi K3 or Nemotron-3 weights needs a model config that this
  repository does not have.
* **No whole model.** The projections, MoE blocks, norms and embeddings are
  shared code, unchanged by this work and not instantiated here. "One layer ×
  93" is a per-layer figure multiplied out, not a compiled 93-layer program.
* **No RTL.** `PLENA_RTL`'s `operation.svh` stops at `0x34`. `V_FMA_VF` joins six
  opcodes (`0x35`–`0x3A`) that the compiler and emulator implement and the
  hardware does not.
* **The exp model.** The emulator uses libtorch's exact `exp`;
  `PLENA_Tools/plena_quant` implements a range reduction plus a 3-term Taylor
  with systematic 0.5–1.5% relative error. KDA applies `exp` on the critical path
  of a multiplicative recurrence, where that compounds. Passing here bounds the
  lowering, not the silicon.

---

## 5. Hybrid models: cost per operator

`analytic_models/performance/hybrid_model.py` reads a per-layer type list out of
a model config and dispatches each layer to the operator it actually runs, then
reports **FLOPs, cycles and HBM bytes per operator**. The three sibling drivers
cannot do this: each assumes one block shape repeated `num_hidden_layers` times,
which is the line `overall_exe_cycle += block_cycles * self.num_hidden_layers`.

### The configs are real, and were fetched rather than written

| `doc/Model_Lib` entry | source | layers |
|---|---|---|
| `kimi-linear-48b-a3b` | `moonshotai/Kimi-Linear-48B-A3B-Instruct` | 20 KDA + 7 attention, 1 dense MLP + 26 MoE |
| `nemotron-3-nano-4b` | `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` | 21 Mamba + 4 attention + 17 MLP |
| `nemotron-3-super-120b-a12b` | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | 40 Mamba + 8 attention + 40 MoE |

Verbatim, except that the Super's 5.7 MB `quantization_config` (per-tensor NVFP4
scales) is stripped. Both layer-type encodings were read out of each model's own
source rather than inferred: NemotronH's `pattern_mapping = {"M": "mamba",
"E": "moe", "*": "attention"}` with `-` for MLP, and Kimi's `is_kda_layer` being
`(layer_idx + 1) in kda_layers`, one-based against a zero-based index. The
parser refuses a config it cannot read rather than falling back to a uniform
stack.

**`KdaShape.kimi_k3()` does not match any of these.** It is `hidden 7168, 96
heads, 93 layers`; the public Kimi Linear is `hidden 2304, 32 heads, 27 layers`.
The *per-head* shape does match — `head_dim` 128 and `short_conv_kernel_size` 4
are exactly what the real config carries — so the kernel-level measurements in
sections 1 and 3b stand. Everything in this repository labelled "Kimi K3" at
model scale is a synthetic shape.

### The KDA cost model is calibrated against the compiler

Every other stage in `perf_model.py` is a formula with no oracle. `kda_chunk_prefill`
and `kda_recurrence_decode` have one: set every opcode latency to 1 and the
formula counts instructions issued, which the compiler can be asked for directly.

| | modelled / measured |
|---|---|
| prefill, nine shapes (chunk 4-16, key and value 64-128) | **0.87 - 1.14** |
| decode, one step at 128 x 128 | **0.98** |

Calibrating found a real error rather than confirming the formula. The model
scaled with `key_dim` at half the compiler's rate, worst at `key_dim 128,
value_dim 64` (0.75): a spill's zero-fill and store walk every **column block**
of a tile wider than `mlen`, and the term billed one. Fixing that moved the range
from 0.75-0.96 to 0.87-1.14.

`test_kda_stage_calibration.py` pins it, and states what it cannot see: at
instruction-count granularity, removing the seven matmuls or the UT forward
substitution moves the count by only 9% — each matmul is one `M_TMM` issue — so
those terms stay as unvalidated as every other stage, while being large in
*cycles*, which is what the driver reports.

### The roofline mismatch is real, and it has a mechanism

**An earlier version of this section said the opposite.** It reported that the
FLOP share and the cycle share track each other and concluded that this model
shows no mismatch. That was measured at one point -- the shipped `ANALYTIC`
config, `MLEN` 2048 with bandwidth derived from `HBM_WIDTH` 512 -- on a plane
neither of whose axes is known. Sweeping both changes the answer.

`hybrid_model.py --sweep <operator>` reports an operator's cycle share divided by
its FLOP share. 1.00 means it costs what its arithmetic says; above 1 means the
machine spends time the FLOP count cannot see. Kimi Linear, decode, 4k context:

| operator | `MLEN` 64, any bandwidth | `MLEN` 2048, 2048 B/cycle |
|---|---|---|
| **KDA** | 0.96 | **1.80** |
| MoE | 1.11 | 0.76 |
| attention | 0.64 | 0.63 |

On a narrow machine the three sit within 0.6-1.1 of each other and nothing stands
out. On a wide one KDA costs **2.9x** what attention does per unit of arithmetic.
The mismatch is not a property of the workload; it is a property of the workload
**on a wide machine**.

### The mechanism is KDA's channel-wise decay, and Mamba-2 does not share it

The cause is vector-lane utilisation, and it is arithmetic rather than
speculation. Each vector op covers `ceil(width / VLEN) * VLEN` lanes and uses
`width` of them:

| | operand width per vector op | at `VLEN` 64 | at `VLEN` 2048 |
|---|---|---|---|
| KDA, one key row | `value_dim` = 128 | 100% | **6.2%** |
| Mamba-2, one state block | `head_dim * state_size` = 8,192 | 100% | **100%** |
| Kimi hidden activation | 2,304 | 100% | 56.2% |
| MoE intermediate | 1,024 | 100% | 50.0% |

**KDA's decay is per key channel**, so every key row carries its own scalar and
has to be its own vector operation -- 128 lanes wide, whatever the machine is.
Mamba-2's decay is one scalar per head, so its whole `[head_dim, state_size]`
state scales in a single op and fills any width.

That is the same fact this project has turned on from the beginning: the
channel-wise decay is what makes KDA more expressive than a scalar-gated SSM, and
it is *also* what makes it waste a wide vector unit. The two are the same
property seen from two sides.

Nemotron-3 Super confirms the negative half. Its recurrent operator is Mamba-2,
and its ratio moves only 0.60 to 0.84 across the whole plane while its MoE sits
flat at 1.09. **"Recurrent operators are disproportionately expensive" is not
the finding**; "an operator whose vector operand is narrower than the machine is"
is.

### Which configuration is real still decides the number

| context | KDA ratio, `ANALYTIC` MLEN 2048 | KDA ratio, `TRANSACTIONAL` MLEN 64 |
|---|---|---|
| 4,096 | 1.27 | **0.96** |
| 1,048,576 | 1.37 | 1.46 |

The sweep above is the answer to this, not a workaround for it: the design
question is not "which of the two declared configurations is correct" but "how
wide should the machine be, given that width past the recurrent state's own
dimension buys the recurrent operator nothing". That is a number the sweep
produces and a single configuration cannot.

Mixing the two configurations is still not survivable: overriding `MLEN` to 64
while leaving `HLEN` at 128 makes `flash_attention` divide by `mlen // hlen` = 0,
which is why `MACHINE_POINTS` moves all four fields together.

### What the earlier reading got wrong

### The single-point reading, and why it was wrong

Kimi Linear, decode, at the shipped `ANALYTIC` configuration only:

| context | KDA %FLOP | %cycles | ratio | attention %FLOP | %cycles | ratio |
|---|---|---|---|---|---|---|
| 4,096 | 24.7% | 31.5% | 1.27 | 8.8% | 7.1% | 0.80 |
| 65,536 | 15.2% | 20.0% | 1.31 | 43.8% | 41.1% | 0.94 |
| 1,048,576 | 2.1% | 2.9% | 1.37 | 92.1% | 91.4% | 0.99 |

Read alone, this says the two shares track each other and no operator is
disproportionately expensive. **Sweeping context was the wrong axis.** It changes
which operator dominates without changing what any of them costs per unit of
arithmetic, so the ratios barely move and the plane the answer actually lives on
is never visited.

Even here the signal was present — KDA at 1.27–1.37 against attention at
0.80–0.99, a 1.6x spread — and it was read as noise around 1.0 rather than as the
narrow end of a trend that reaches 2.9x. Reporting the number that came out is
necessary but not sufficient; a number from one point on an unexplored plane is
not yet a result.

### The two declared configurations, as the endpoints of that plane

The same context sweep on the two configurations `plena_settings.toml` declares:

| context | KDA ratio, `ANALYTIC` MLEN 2048 | KDA ratio, `TRANSACTIONAL` MLEN 64 |
|---|---|---|
| 4,096 | 1.27 | **0.96** |
| 65,536 | 1.31 | 1.12 |
| 1,048,576 | 1.37 | 1.46 |

At short context the wide machine makes KDA look disproportionately expensive and
the narrow one makes it exactly proportionate. That is a **qualitative**
disagreement produced by a config field, not a modelling subtlety, and it is why
`MLEN` 2048 against 64 (and `BLEN` 128 against 4, `HLEN` 128 against 16) has to
be settled before any of these numbers mean anything. Mixing the two is not
survivable either: overriding `MLEN` to 64 while leaving `HLEN` at 128 makes
`flash_attention` divide by `mlen // hlen` = 0.

No times are reported anywhere in this section. `runtime_config.rs` hard-codes
`PERIOD` at 1 ns with no stated basis, so cycles stay cycles.

### What "wide" and "narrow" mean here, and the assumption inside them

`MLEN`, `BLEN`, `HLEN` and `VLEN` are the tile sizes `doc/plena_isa_spec.md`
defines: the matrix machine's tile, the systolic array's, the partitioned
systolic array's, and the vector machine's. `M_MM` fetches a `(BLEN, MLEN)` tile
from Vector SRAM against an `(MLEN, BLEN)` tile from Matrix SRAM. So `MLEN` is
the edge one matrix instruction addresses in and `VLEN` is how many elements one
vector instruction covers.

"Narrow" is `TRANSACTIONAL`: 64 / 4 / 16 / 64. "Wide" is `ANALYTIC`:
2048 / 128 / 128 / 2048.

**The assumption this bakes in has to be stated, because it decides how the
tables below may be read.** In `perf_model.py` those fields enter cycles *only*
through instruction counts:

    cycles = ceil(work / MLEN) x instruction latency

and the latency is a constant. `V_MUL_VF` and `V_FMA_VF` bill
`VECTOR_MUL_CYCLES` from the TOML whatever `VLEN` is; `M_MM` bills `BLEN`. So a
wider machine here means **one instruction covering 32x the data at the same
cost** -- 32x the throughput, free, and with no area term anywhere in the model.

That makes the *absolute* speedups below near-tautological. "Widening from 64 to
2048 makes decode 15x faster" is mostly a restatement of the assumption, not a
finding, and it should not be quoted as one.

**What survives the assumption is the comparison between operators.** KDA and
Mamba-2 are priced by the same idealisation, so when one saturates on width at
23.9% marginal return and the other is still at 44.0%, the difference is a
property of the two operators and not of the free throughput both were granted.
Every result in this section is of that shape: a ratio between operators, or a
turning point, never an absolute speedup.

**A second inconsistency, found while writing this.** The transactional
emulator's matrix core is a hard-coded `big_4x1024` -- 4,096 PEs, in
`matrix_core.rs`, independent of `MLEN` entirely. So the two halves model width
by different mechanisms: the analytic side by instruction count, the emulator
side by a fixed PE array. That is separate from the `MLEN` disagreement in
`plena_settings.toml` and equally unresolved.

### How wide should the machine be

The ratio grid says an operator's *share* of the time gets worse on a wider
machine. It does not say whether the machine is faster, which is what a width
decision turns on. `--sweep` reports that too.

**Width and bandwidth have to scale together, or width buys nothing.** Marginal
return of doubling `MLEN`, as a percentage of total decode cycles removed, Kimi
Linear:

| doubling | 64 B/cyc | 256 | 1024 | 4096 |
|---|---|---|---|---|
| 64 → 256 | 14.4% | 74.6% | 75.5% | 75.5% |
| 256 → 512 | 0.0% | 6.2% | 43.0% | 43.0% |
| 512 → 1024 | 0.0% | 0.0% | 36.8% | 36.8% |
| 1024 → 2048 | 0.0% | 0.0% | 13.4% | 23.9% |

The zeros are the finding. Below the diagonal a wider machine removes **no**
cycles at all: the memory server is already the limit and the extra lanes idle.
On Nemotron-3 Super the same cells go slightly **negative** (−0.1% to −0.5%) --
a wider machine is fractionally slower when bandwidth does not follow it.

**Where width stops paying depends on which recurrent operator the model uses.**
At matched bandwidth, the last doubling (1024 → 2048) returns:

| | marginal return | recurrent operator's share of cycles, 64 → 2048 |
|---|---|---|
| Kimi Linear (KDA) | **23.9%** | 23.9% → **44.5%** |
| Nemotron-3 Super (Mamba-2) | **44.0%** | 10.7% → 15.1% |

The Mamba model is still returning nearly half per doubling at the widest point
tested. The KDA model has decayed to a quarter, and the reason is visible in the
second column: as the machine widens, KDA goes from a quarter of decode time to
**nearly half**, while contributing 24.7% of the FLOPs and running at **6.2%
vector-lane utilisation**. Mamba-2's share barely moves, because its state block
fills any width.

So at `MLEN` 2048, about **45% of Kimi Linear's decode time is spent by an
operator using 6.2% of the lanes it is charged for**. That is the concrete
inefficiency a heterogeneous-operator substrate would target, and it is the
number this driver exists to produce.

**What this does not say.** There is no area model here, so "wider is better"
wherever it still helps is trivially true and not a recommendation. What is not
trivial is the shape: the returns decay at a rate set by the narrowest operand in
the model, and two hybrid models with the same layer census and different
recurrent operators sit at different points on that curve.

---

### Prefill and decode want different machines, and by a lot

The width question has a different answer in each mode. Total cycles at matched
bandwidth, Kimi Linear:

| `MLEN` | decode (Mcycles) | prefill (Mcycles) |
|---|---|---|
| 256 | 14.20 | 3,831 |
| 512 | 8.10 | 1,856 |
| 1024 | 5.12 | **1,742** |
| 2048 | **3.89** | 2,650 |

**Decode's optimum is the widest point tested; prefill's is 1024, and 2048 is
52% worse than that.** A single fixed width cannot be right for both, which is a
direct measurement of the thing a mode-switching fabric would be for -- and, read
the other way, a direct cost of not having one.

**Nemotron-3 Super does not have this problem.** Same sweep, same axes:

| `MLEN` | decode, marginal return | prefill, total (Mcycles) |
|---|---|---|
| 1024 | | 2,808 |
| 2048 | **+44.0%** | **1,145** (+59.2%) |

Both modes still want the widest machine tested. So the divergence is not a
property of hybrid models, or of recurrent mixers, or of prefill: it is a
property of **KDA**, and the next section says why.

### The mechanism is `chunk` against `mlen`, and it is a numerical constraint

Decomposing KDA prefill by machine width, one chunk and one head, instructions
issued:

| `MLEN` | total | without the spill term | spill share |
|---|---|---|---|
| 64 | 2,450 | 1,696 | 30.8% |
| 512 | 4,390 | 1,072 | 75.6% |
| 2048 | 14,173 | **1,063** | **92.5%** |

**The real work gets cheaper as the machine widens** -- 1,696 down to 1,063,
which is what wider tiles should do. The spill term grows from 754 instructions
to 13,110, and swamps it.

Every spill zero-fills the rows past its live data, because `load_sub_matrix_*`
prefetches a whole `mlen x mlen` MRAM block unconditionally: `k_block_count`
selects whole blocks and cannot trim a partial one. So the fill scales with
`mlen` while the live data scales with `chunk` -- and `chunk` **cannot grow to
meet it**. `kda_chunk_check_range` caps it at 17, because `1/A` reaches
`exp(chunk * |gate_lower_bound|)` and overflows bf16 past that.

At `mlen` 2048 that is 2,048 rows zeroed to protect 16 rows of data, six times
per chunk. The bf16 range of the reciprocal decay is what sets the numerator and
the machine's tile size sets the denominator, and nothing in the lowering can
reconcile them.

Mamba-2's chunk is **256** -- `nemotron-3-super-120b-a12b.json` says so, and
nothing bounds it numerically, because the SSD scan has no reciprocal decay to
overflow. Sixteen times more of every tile is live, which is why its prefill
keeps improving where KDA's turns around.

So both of KDA's costs on a wide machine come from the same design choice this
project has circled from the start. The decay is **channel-wise on the key
axis**, which makes each decode sweep 128 lanes wide (6.2% utilisation at `VLEN`
2048) and forces the chunk form to divide by a per-channel cumulative decay,
whose reciprocal caps the chunk at 17 and leaves prefill's tiles 99% padding. The
same property that makes KDA more expressive than a scalar-gated SSM is what
costs it in both modes, by two different routes.

### The ISA consequence, priced rather than asserted

A prefetch taking a row count instead of a whole-block count would remove the
fill and store only the live rows. `kda_chunk_prefill(row_granular_prefetch=True)`
models exactly that -- no such instruction exists, and the switch is there to
price it, not to assume it. `--row-granular-prefetch` carries it through the
driver.

Instructions issued, one chunk and one head:

| `mlen` | whole-block prefetch | row-granular | factor |
|---|---|---|---|
| 64 | 2,450 | 1,801 | 1.4x |
| 512 | 4,390 | 1,146 | 3.8x |
| 2048 | 14,173 | 1,137 | **12.5x** |

**The proposal is worth almost nothing on a narrow machine and a great deal on a
wide one**, which is the shape you would expect from a term that scales with
`mlen`. Anyone reading 12.5x as the value of this instruction on the machine they
have should check which column they are in first.

*(An earlier draft of this section said 13x, from 14,173 down to 1,063. That was
the count with the whole spill term zeroed. Row-granular prefetch still stores
the live rows, so the floor is 1,137 and the factor is 12.5x. Predicting a number
and then measuring it is how the difference showed up.)*

### And it makes the two modes agree on width

Total prefill cycles, Kimi Linear, at matched bandwidth:

| `MLEN` | whole-block prefetch | row-granular |
|---|---|---|
| 512 | 1,856 | 1,325 |
| 1024 | **1,742** (optimum) | 676 |
| 2048 | 2,650 (52% worse) | **515** (optimum) |

The optimum **moves from 1024 to 2048**, and the last doubling returns **+23.9%**
instead of −52.1%. Decode's return at that same doubling is also +23.9% -- with
the fill removed, both modes are limited by the same dense work, and the width
disagreement disappears.

So the mode divergence measured above is not a fact about KDA's algorithm. It is
a fact about **KDA's algorithm on this instruction set**: the chunk is capped at
17 by bf16 range, the prefetch granularity is a whole block, and the product of
those two is the whole effect. One of them is a numerical constraint that cannot
be moved. The other is an instruction-set choice that can.

This is the opposite shape from a descriptor-driven state engine: it asks for
*less* to be implied by one instruction, not more. Nothing here implements it.

---

---

## 6. The clock, and what it is not

`runtime_config.rs` carried `pub(crate) const PERIOD: Duration =
Duration::from_nanos(1)` — a bare constant, in a source file, with no stated
basis. Every second, microsecond, TPOT and token-per-joule figure the emulator
can produce is that number's consequence.

It is now `[<MODE>.CONFIG.CLOCK_PERIOD_PS]`, defaulted to the same 1000 ps so
nothing moves, with the assumption written where it can be read:

> THIS IS AN ASSUMPTION, NOT A MEASUREMENT. No RTL has been synthesised, so no
> critical path has set a frequency.

Three things changed beyond moving the value.

**It is printed at startup, with its provenance.** Every emulator run now begins:

```
Clock: 1000 ps (1.000 GHz) from CLOCK_PERIOD_PS -- an assumption, not a
synthesised frequency. DRAM model: 1000 ps (1x).
```

A number that appears in a report and a number that appears in a log line have
different half-lives. This one is now impossible to use without seeing where it
came from.

**The relationship to the DRAM clock is asserted, not assumed.**
`stage_profile.rs` had already recorded the hazard in a comment: a cycle-domain
comparison there *"only held because the DRAM tCK happened to equal PERIOD; any
preset or frequency change made it fail"*. It does happen to be equal — the HBM2
preset at 2000 MBPS is a 1 ns command clock — and that is a property of the
preset, not of the design. `runner` now requires one period to be a whole
multiple of the other and refuses to start otherwise:

```
accelerator clock is 700 ps (1.429 GHz, from CLOCK_PERIOD_PS) and the DRAM
model's is 1000 ps; neither divides the other, so ticks never line up and any
cycle count derived from both is off by a fraction nothing reports
```

Verified by setting it to 700 and watching it refuse.

**No times are reported anywhere in this document.** Making the clock
configurable does not make it known. Until a synthesis run sets a frequency, a
microsecond figure here would be a cycle count multiplied by a number someone
chose, and section 5's sweep is the shape of that: cycles are the quantity the
model produces, and cycles are what it reports.

### What is still open on this

`analytic_models/performance/` has no clock at all — it reports cycles and stops,
which is the right behaviour for now but means there is no single place where a
frequency would be applied when one exists. The two halves also still disagree
about bandwidth: the analytic side derives it from `HBM_WIDTH` 512 as one row per
cycle, and the transactional side runs an HBM2 preset. Section 5 sweeps the
former; nothing yet reconciles the two.
