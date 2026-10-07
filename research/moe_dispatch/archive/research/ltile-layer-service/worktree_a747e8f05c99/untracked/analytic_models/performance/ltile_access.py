"""Candidate access-path experiment; NOT a calibrated replacement for R3.

The experiment isolates an on-chip fused UPDATE pass. It does not execute an
ISA, projection, DMA, prediction/readout dot, or a complete recurrent layer.
It holds arithmetic constant and enumerates real fixed-diagonal bank words.
No empirical speedup, random bank-conflict probability or unlimited FIFO is
used. Native coefficients remain in their producer's head-major order.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, asdict
import math


BANKS = 64
WORD_VALUES = 32  # BF16 values, NOT bits
VALUE_BYTES = 2
MATRIX_BYTES = 1024 * 1024


@dataclass(frozen=True)
class CoefficientView:
    """Candidate 64-bit local-SRAM descriptor; not an assigned ISA encoding.

    base[19], head_stride[13], row_stride[9], repeat_log2[3], heads-1[6],
    extent-1[12], reserved[2]. Addresses/strides are BF16 element indices.
    All three fields can use the same AGU; none contains a model identifier.
    """

    base: int
    head_stride: int
    row_stride: int
    repeat_log2: int
    heads: int
    extent: int

    def __post_init__(self):
        fields = ((self.base, 19), (self.head_stride, 13), (self.row_stride, 9),
                  (self.repeat_log2, 3), (self.heads - 1, 6), (self.extent - 1, 12))
        if any(type(v) is not int or not 0 <= v < 1 << bits for v, bits in fields):
            raise ValueError("coefficient view field exceeds its encoding")
        if self.base + self.extent > MATRIX_BYTES // VALUE_BYTES:
            raise ValueError("coefficient view exceeds existing SRAM")

    def pack(self):
        return (self.base | self.head_stride << 19 | self.row_stride << 32 | self.repeat_log2 << 41
                | (self.heads - 1) << 44 | (self.extent - 1) << 50)

    @classmethod
    def unpack(cls, word):
        if type(word) is not int or not 0 <= word < 1 << 62:
            raise ValueError("reserved or out-of-range coefficient view bits")
        return cls(word & ((1 << 19) - 1), (word >> 19) & 8191, (word >> 32) & 511,
                   (word >> 41) & 7, ((word >> 44) & 63) + 1, ((word >> 50) & 4095) + 1)

    def location(self, row, head):
        if type(row) is not int or type(head) is not int or row < 0 or not 0 <= head < self.heads:
            raise ValueError("invalid coefficient index")
        offset = (head >> self.repeat_log2) * self.head_stride + row * self.row_stride
        if offset >= self.extent:
            raise ValueError("coefficient address exceeds view extent")
        index = self.base + offset
        return physical_word(index // WORD_VALUES), index % WORD_VALUES


def physical_word(linear_word: int) -> tuple[int, int]:
    if not 0 <= linear_word < MATRIX_BYTES // (WORD_VALUES * VALUE_BYTES):
        raise ValueError("word outside the existing 1 MiB Matrix SRAM")
    address, column = divmod(linear_word, BANKS)
    return (column + address) % BANKS, address


def bank_waves(words):
    """One distinct address per bank per cycle; same-word fanout is shared."""
    pending = {}
    for bank, address in dict.fromkeys(words):
        if not 0 <= bank < BANKS or not 0 <= address < 256:
            raise ValueError("invalid physical bank word")
        pending.setdefault(bank, deque()).append(address)
    waves = []
    while pending:
        waves.append(tuple((bank, addresses.popleft()) for bank, addresses in sorted(pending.items())))
        pending = {bank: addresses for bank, addresses in pending.items() if addresses}
    return tuple(waves)


@dataclass(frozen=True)
class Shape:
    kind: str
    rows: int = 128

    def __post_init__(self):
        if self.kind not in ("mamba", "kda") or type(self.rows) is not int or self.rows < 1:
            raise ValueError("expected Mamba/KDA and positive state rows")
        if self.state_bytes > MATRIX_BYTES // 2:
            raise ValueError("state tile exceeds the candidate's 512 KiB allocation")

    @property
    def width(self):
        return 64 if self.kind == "mamba" else 128

    @property
    def heads(self):
        return 2048 // self.width

    @property
    def coefficient_heads(self):
        return self.heads // 8 if self.kind == "mamba" else self.heads

    @property
    def state_bytes(self):
        return self.rows * self.heads * self.width * VALUE_BYTES

    def state_word(self, row, head, word, layout):
        if not (0 <= row < self.rows and 0 <= head < self.heads and 0 <= word < self.width // WORD_VALUES):
            raise ValueError("state index outside tile")
        if layout == "head_major":
            index = (head * self.rows + row) * self.width // WORD_VALUES + word
        elif layout == "head_phase":
            index = (row * self.heads + head) * self.width // WORD_VALUES + word
        else:
            raise ValueError("unknown layout")
        return physical_word(index)

    def stripe(self, row, first_value, values, layout):
        if first_value % WORD_VALUES or values % WORD_VALUES or first_value + values > 2048:
            raise ValueError("stripe must contain whole bank words inside one row")
        return tuple(
            self.state_word(row, logical // self.width, logical % self.width // WORD_VALUES, layout)
            for logical in range(first_value, first_value + values, WORD_VALUES)
        )


class Coefficients:
    """Bounded native source views, or existing row-packed UPDATE pairs.

    One 32-row sector per field/head is explicitly latched. This is a
    compiler-addressed staging register file, not a tagged runtime cache.
    The full prediction/readout design additionally needs c/q, not used here.
    Mamba x means already-prepared BF16(dt*x); KDA x means BF16 residual.
    Their production is OUTSIDE this update-only experiment.
    """

    def __init__(self, shape: Shape, mode: str):
        if mode not in ("native", "packed"):
            raise ValueError("unknown coefficient supply")
        self.shape, self.mode = shape, mode
        cursor = math.ceil(shape.state_bytes / 4096) * 2048
        self.fields = {}

        def allocate(name, values):
            nonlocal cursor
            cursor = math.ceil(cursor / WORD_VALUES) * WORD_VALUES
            self.fields[name] = (cursor, values)
            cursor += math.ceil(values / WORD_VALUES) * WORD_VALUES

        allocate("input", 2048)
        if mode == "packed":
            allocate("pairs", shape.rows * shape.heads * 2)
        else:
            allocate("delta", shape.heads if shape.kind == "mamba" else shape.heads * shape.rows)
            allocate("b", shape.coefficient_heads * shape.rows)
        if cursor * VALUE_BYTES > MATRIX_BYTES:
            raise ValueError("state, input and coefficients exceed existing Matrix SRAM")
        self.allocated_bytes = cursor * VALUE_BYTES
        self.views = {}
        if mode == "native":
            for name in ("delta", "b"):
                base, values = self.fields[name]
                scalar_delta = name == "delta" and shape.kind == "mamba"
                self.views[name] = CoefficientView(
                    base, 1 if scalar_delta else shape.rows, 0 if scalar_delta else 1,
                    3 if name == "b" and shape.kind == "mamba" else 0, shape.heads, values,
                )

    def word(self, field, index):
        base, values = self.fields[field]
        if not 0 <= index < values:
            raise ValueError("coefficient index outside field")
        return physical_word((base + index) // WORD_VALUES)

    def scalar_location(self, field, row, head):
        s = self.shape
        if field not in ("delta", "b") or not 0 <= row < s.rows or not 0 <= head < s.heads:
            raise ValueError("invalid update scalar")
        if self.mode == "packed":
            index = 2 * (row * s.heads + head) + (field == "b")
            name = "pairs"
        else:
            return self.views[field].location(row, head)
        base, _ = self.fields[name]
        return self.word(name, index), (base + index) % WORD_VALUES

    def initial_waves(self):
        waves = list(bank_waves(self.word("input", i) for i in range(0, 2048, WORD_VALUES)))
        if self.mode == "native" and self.shape.kind == "mamba":
            waves.extend(bank_waves(self.word("delta", h) for h in range(self.shape.heads)))
        return tuple(waves)

    def epoch(self, row):
        return row if self.mode == "packed" else row // WORD_VALUES

    def refill(self, row):
        s = self.shape
        if self.mode == "packed":
            return bank_waves(self.word("pairs", 2 * (row * s.heads + h) + f) for h in range(s.heads) for f in (0, 1))
        first, last = row // WORD_VALUES * WORD_VALUES, min(s.rows, (row // WORD_VALUES + 1) * WORD_VALUES)
        waves = []
        # Separate fields are separate address-generator traversals, never a
        # free multi-base vector load. Non-word-aligned tails may need 2 words.
        for field, heads in (("delta", s.heads), ("b", s.coefficient_heads)):
            if s.kind == "mamba" and field == "delta":
                continue
            waves.extend(bank_waves(self.word(field, h * s.rows + r) for h in range(heads) for r in range(first, last)))
        return tuple(waves)

    @property
    def staging_bytes(self):
        if self.mode == "packed":
            return 2 * self.shape.heads * VALUE_BYTES
        # Each head can span two native words for a non-word-aligned N.
        words = max(sum(len(w) for w in self.refill(r)) for r in range(0, self.shape.rows, WORD_VALUES))
        return words * WORD_VALUES * VALUE_BYTES + (self.shape.heads * VALUE_BYTES if self.shape.kind == "mamba" else 0)


@dataclass(frozen=True)
class AccessConfig:
    lanes: int = 256
    read_values: int = 2048
    input_slots: int = 2
    output_slots: int = 4
    ii: int = 2
    latency: int = 6
    alignment_latency: int = 1
    port_policy: str = "shared"
    blocked_every: int = 0

    def __post_init__(self):
        integers = (self.lanes, self.read_values, self.input_slots, self.output_slots,
                    self.ii, self.latency, self.alignment_latency, self.blocked_every)
        if any(type(v) is not int for v in integers):
            raise ValueError("resource geometry and cycles must be integers")
        if self.lanes not in (128, 256, 512):
            raise ValueError("unsupported update width")
        if self.read_values not in (128, 256, 512, 1024, 2048) or self.read_values % self.lanes:
            raise ValueError("read width must be a supported multiple of compute width")
        if min(self.input_slots, self.output_slots, self.ii, self.latency) < 1 or self.alignment_latency < 0:
            raise ValueError("positive bounded slots and pipeline required")
        if self.input_slots > 8 or self.output_slots > 64 or max(self.latency, self.ii, self.alignment_latency) > 64:
            raise ValueError("resource budget outside the bounded candidate search")
        if self.port_policy not in ("shared", "1r1w") or self.blocked_every not in (0, 4, 8):
            raise ValueError("unsupported port/arbitration policy")

    def storage(self, coefficients):
        return dict(
            state_input_bytes=self.input_slots * self.read_values * VALUE_BYTES,
            result_hold_bytes=self.output_slots * self.lanes * VALUE_BYTES,
            input_vector_bytes=2048 * VALUE_BYTES,
            coefficient_sector_bytes=coefficients.staging_bytes,
            # Each entry: 40-bit destination, 64-bit bank mask, row/mode bits.
            tag_bytes=(self.input_slots + self.output_slots) * 16,
            coefficient_descriptor_bytes=3 * 8 if coefficients.mode == "native" else 0,
        )


def simulate_update(shape: Shape, config=AccessConfig(), *, layout="head_phase", coefficients="native", trace=False):
    """Deterministic cycle simulation with finite accepted-result credits.

    One Matrix transfer per cycle for shared, at most 1R and 1W for 1r1w.
    Writes win shared-port arbitration; coefficient refills precede state
    reads. Slots are freed only after acceptance/last consumption. All state
    words are unique within a pass, so in-place writes cannot corrupt a later
    read. Across-token overlap and DMA are intentionally not modeled.
    """
    cfg, src = config, Coefficients(shape, coefficients)
    packets = [(r, first) for r in range(shape.rows) for first in range(0, 2048, cfg.read_values)]
    buffers, pipeline, outputs = deque(), deque(), deque()
    initial = deque(src.initial_waves())
    refill = deque()
    loaded_epoch, target_epoch, coefficient_ready, input_ready = -1, -1, 0, 0
    fetch, launched, completed, credits, next_launch = 0, 0, 0, 0, 0
    total_chunks = shape.rows * 2048 // cfg.lanes
    counts, events = Counter(), []
    occupancy = dict(input_slots=0, output_slots=0)
    credit_release = Counter()
    cycle = 0

    def event(kind, **data):
        if trace:
            events.append(dict(cycle=cycle, kind=kind, **data))

    def transfer(kind, words, **data):
        assert len({b for b, _ in words}) == len(words)
        counts[kind + "_waves"] += 1
        counts[kind + "_words"] += len(words)
        event(kind, words=words, **data)

    while completed < total_chunks:
        if cycle > 1000 * total_chunks + 10000:
            raise RuntimeError("access experiment deadlocked")
        credits -= credit_release.pop(cycle, 0)
        while pipeline and pipeline[0][0] <= cycle:
            due, tag, words = pipeline.popleft()
            outputs.append(dict(tag=tag, waves=deque(bank_waves(words))))
        row = min(shape.rows - 1, launched * cfg.lanes // 2048)
        epoch = src.epoch(row)
        if not initial and loaded_epoch != epoch and target_epoch != epoch:
            refill = deque(src.refill(row))
            target_epoch = epoch
        # Slots consumed at the preceding edge can now receive a new stripe.
        while buffers and buffers[0]["consumed"] == cfg.read_values:
            buffers.popleft()
        if fetch < len(packets) and len(buffers) < cfg.input_slots:
            r, first = packets[fetch]
            buffers.append(dict(id=fetch, row=r, first=first, consumed=0, ready=math.inf,
                                waves=deque(bank_waves(shape.stripe(r, first, cfg.read_values, layout)))))
            fetch += 1
        occupancy["input_slots"] = max(occupancy["input_slots"], len(buffers))

        # Launch is at the beginning of a cycle. A write accepted this cycle
        # returns its credit at the next edge, not combinationally for free.
        if (buffers and buffers[0]["ready"] <= cycle and loaded_epoch == epoch
                and coefficient_ready <= cycle and input_ready <= cycle
                and cycle >= next_launch and credits < cfg.output_slots):
            packet = buffers[0]
            offset = packet["first"] + packet["consumed"]
            words = shape.stripe(packet["row"], offset, cfg.lanes, layout)
            tag = launched
            event("launch", tag=tag, packet=packet["id"], row=packet["row"], offset=offset, words=words)
            pipeline.append((cycle + cfg.latency, tag, words))
            credits += 1
            occupancy["output_slots"] = max(occupancy["output_slots"], credits)
            packet["consumed"] += cfg.lanes
            launched += 1
            next_launch = cycle + cfg.ii
        if cfg.blocked_every and cycle % cfg.blocked_every == 0:
            counts["external_port_block_cycles"] += 1
            cycle += 1
            continue
        wrote = False
        if outputs:
            output = outputs[0]
            transfer("write", output["waves"].popleft(), tag=output["tag"])
            wrote = True
            if not output["waves"]:
                event("commit", tag=output["tag"])
                outputs.popleft()
                completed += 1
                credit_release[cycle + 1] += 1
        if cfg.port_policy == "1r1w" or not wrote:
            if initial:
                transfer("coefficient_read", initial.popleft(), epoch=-1)
                if not initial:
                    input_ready = cycle + 1 + cfg.alignment_latency
            elif refill:
                transfer("coefficient_read", refill.popleft(), epoch=target_epoch)
                if not refill:
                    loaded_epoch = target_epoch
                    coefficient_ready = cycle + 1 + cfg.alignment_latency
            else:
                packet = next((p for p in buffers if p["waves"]), None)
                if packet is not None:
                    transfer("state_read", packet["waves"].popleft(), packet=packet["id"])
                    if not packet["waves"]:
                        packet["ready"] = cycle + 1 + cfg.alignment_latency
        cycle += 1
    assert launched == completed == total_chunks
    expected_words = shape.state_bytes // (WORD_VALUES * VALUE_BYTES)
    assert counts["state_read_words"] == counts["write_words"] == expected_words
    compute_floor = (total_chunks - 1) * cfg.ii + cfg.latency
    port_floor = (counts["state_read_waves"] + counts["coefficient_read_waves"] + counts["write_waves"]
                  if cfg.port_policy == "shared" else max(counts["state_read_waves"] + counts["coefficient_read_waves"], counts["write_waves"]))
    result = dict(
        status="uncalibrated_update_access_candidate", shape=asdict(shape), config=asdict(cfg),
        layout=layout, coefficients=coefficients, cycles=cycle, launches=launched,
        counters=dict(counts), peak_occupancy=occupancy, storage=cfg.storage(src),
        matrix_allocation_bytes=src.allocated_bytes, compute_lower_bound=compute_floor,
        port_lower_bound=port_floor, excluded="ISA issue, producer, DMA, dot/tree/residual, whole layer, PPA",
    )
    assert cycle >= max(compute_floor, port_floor)
    if trace:
        result["events"] = events
    return result


def validate_trace(result):
    """Independent ledger checks, not a second implementation of scheduling."""
    cfg = AccessConfig(**result["config"])
    ports, reads, writes, live, launches = Counter(), {}, Counter(), {}, {}
    last_launch = -cfg.ii
    for e in result["events"]:
        t, kind = e["cycle"], e["kind"]
        if kind in ("state_read", "coefficient_read", "write"):
            direction = "write" if kind == "write" else "read"
            ports[t, direction] += 1
            assert ports[t, direction] <= 1
            if cfg.port_policy == "shared":
                assert ports[t, "read"] + ports[t, "write"] <= 1
            assert not cfg.blocked_every or t % cfg.blocked_every != 0
            assert len({b for b, _ in e["words"]}) == len(e["words"])
        if kind == "state_read":
            for word in e["words"]:
                assert word not in reads
                reads[word] = t
        elif kind == "launch":
            assert t - last_launch >= cfg.ii
            last_launch = t
            for word in e["words"]:
                assert reads[word] + 1 + cfg.alignment_latency <= t
            # Commits at t retain a reservation through the end of cycle t.
            live = {tag: end for tag, end in live.items() if end >= t}
            assert len(live) < cfg.output_slots
            live[e["tag"]] = math.inf
            launches[e["tag"]] = (t, set(e["words"]))
        elif kind == "write":
            start, expected = launches[e["tag"]]
            assert t >= start + cfg.latency
            for word in e["words"]:
                assert word in expected
                writes[word] += 1
        elif kind == "commit":
            live[e["tag"]] = t
    assert len(launches) == result["launches"]
    assert set(reads) == set(writes) and set(writes.values()) == {1}
    return True


def verify_native_update(shape: Shape, seed=0, tokens=4):
    """Bitwise Python check against independently packed BF16 operands.

    Writes producer arrays through the fixed physical SRAM mapping, fetches
    through native descriptors, and checks each coefficient and updated state.
    This validates addressing/rounding, not real-input accuracy or an ISA path.
    """
    import numpy as np

    if tokens < 1:
        raise ValueError("positive token chain required")
    rng = np.random.default_rng(seed)
    s, coeff = shape, Coefficients(shape, "native")
    memory = np.zeros((BANKS, 256, WORD_VALUES), dtype=np.uint16)

    def bf16_bits(x):
        bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

    def floats(bits):
        return (bits.astype(np.uint32) << 16).view(np.float32)

    def write_field(field, data):
        base, count = coeff.fields[field]
        assert data.size == count
        indices = base + np.arange(count)
        words, lanes = indices // 32, indices % 32
        addresses, columns = words // 64, words % 64
        memory[(columns + addresses) % 64, addresses, lanes] = data.ravel()

    def update(state, delta, b, x):
        # Separate multiply/subtract/add; numpy float32 arrays force each
        # boundary, without a host compiler contracting an FMA.
        a = floats(state)
        p = np.multiply(floats(delta)[..., None], a, dtype=np.float32)
        q = np.multiply(floats(b)[..., None], floats(x)[None, ...], dtype=np.float32)
        u = np.subtract(a, p, dtype=np.float32)
        return bf16_bits(np.add(u, q, dtype=np.float32))

    reference = bf16_bits(rng.normal(0, 0.1, (s.rows, s.heads, s.width)))
    actual = reference.copy()
    compared = 0
    for _ in range(tokens):
        delta = bf16_bits(rng.uniform(0.0001, 0.05, s.heads if s.kind == "mamba" else (s.heads, s.rows)))
        b = bf16_bits(rng.normal(0, 0.1, (s.coefficient_heads, s.rows)))
        x = bf16_bits(rng.normal(0, 0.1, (s.heads, s.width)))
        for name, data in (("delta", delta), ("b", b), ("input", x)):
            write_field(name, data)
        expected_delta = np.broadcast_to(delta[None, :], (s.rows, s.heads)) if s.kind == "mamba" else delta.T
        expected_b = np.repeat(b.T, 8, axis=1) if s.kind == "mamba" else b.T
        # Independent packing reference corresponds to original row/head pairs.
        prepared = np.stack((expected_delta, expected_b), axis=-1)
        fetched = np.empty_like(prepared)
        for row in range(s.rows):
            for head in range(s.heads):
                for field_index, field in enumerate(("delta", "b")):
                    (bank, address), lane = coeff.scalar_location(field, row, head)
                    fetched[row, head, field_index] = memory[bank, address, lane]
        np.testing.assert_array_equal(prepared, fetched)
        reference = update(reference, prepared[..., 0], prepared[..., 1], x)
        actual = update(actual, fetched[..., 0], fetched[..., 1], x)
        np.testing.assert_array_equal(reference, actual)
        compared += prepared.size + actual.size
    return dict(model=s.kind, rows=s.rows, seed=seed, tokens=tokens, compared_bf16_values=compared,
                coefficient_bits_exact=True, update_bits_exact=True,
                rng=rng.bit_generator.__class__.__name__, numpy_version=np.__version__,
                scope="synthetic Python native-SRAM fetch vs prepared coefficients; no machine code or task-quality claim")
