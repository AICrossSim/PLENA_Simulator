//! Standalone architectural feasibility probe; NOT instruction-engine overlap.
//!
//! Fixed producer/consumer contexts use a shared 1R/1W Matrix port calendar
//! and one shared Vector port. Four K64 sectors refill a bounded K256 replay
//! latch before one unchanged K256 reduction. Payload becomes readable only
//! after write completion; a slot is released only after its final read.
//! This small numeric protocol test does not model a complete layer or PPA.
use crate::matrix_service::MatrixService;
use half::bf16;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
struct Tag {
    request: usize,
    token: usize,
    group: usize,
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum Owner {
    Empty,
    Filling(Tag),
    Ready(Tag),
    Reading(Tag),
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
enum Port {
    MatrixRead,
    MatrixWrite,
    Vector,
}
#[derive(Clone, Copy, Debug, Serialize)]
struct Reservation {
    port: Port,
    start: u64,
    end: u64,
    producer: bool,
    tag: Tag,
}
#[derive(Clone)]
struct Pending {
    phase: usize,
    end: u64,
}
struct Context {
    tag: Tag,
    slot: usize,
    phase: usize,
    pending: Option<Pending>,
    payload: Vec<f32>,
}
#[derive(Serialize)]
struct ProbeResult {
    cycles: u64,
    jobs: usize,
    slots: usize,
    producer_backpressure: u64,
    port_wait: u64,
    max_live_slots: usize,
    exact: bool,
    preserved_k_reduction: usize,
    sector_rows: (usize, usize),
    reservations: Vec<Reservation>,
}
fn rn(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn hardware() -> MatrixService {
    MatrixService {
        edge: 4,
        reduction_lanes: 1024,
        mac_latency: 2,
        mac_ii: 1,
        tree_add_latency: 2,
        matrix_read_elements: 2048,
        vector_read_elements: 2048,
        vector_write_elements: 2048,
        matrix_capacity_bytes: 1048576,
        vector_capacity_bytes: 262144,
        accumulator: "BF16".into(),
        weight_replay: true,
        projection_segments: 1,
    }
}
fn weights(tag: Tag) -> Vec<f32> {
    (0..256 * 32)
        .map(|i| rn(((i * 13 + tag.group * 5) % 47) as f32 / 256.0 - 0.09))
        .collect()
}
fn inputs(tag: Tag) -> Vec<f32> {
    (0..256)
        .map(|i| rn(((i * 3 + tag.request * 11 + tag.token * 7) % 31) as f32 / 64.0 - 0.2))
        .collect()
}
fn project(x: &[f32], w: &[f32]) -> Vec<f32> {
    let h = hardware();
    (0..32)
        .map(|c| h.column(x, &(0..256).map(|k| w[k * 32 + c]).collect::<Vec<_>>(), 0.0))
        .collect()
}
fn tree(mut values: Vec<f32>) -> f32 {
    while values.len() > 1 {
        values = values.chunks_exact(2).map(|p| rn(p[0] + p[1])).collect();
    }
    values[0]
}
/// Small Mamba/KDA recurrent cells, solely to expose stale/mixed operands.
/// Same mathematical function is used in the serial oracle; arithmetic itself
/// is verified by the production recurrence and Matrix execution tests.
fn consume(state: &mut [f32], coeff: &[f32], kda: bool) -> Vec<f32> {
    let old = state.to_vec();
    let residual: Vec<f32> = (0..4)
        .map(|j| {
            if kda {
                let pred = tree(
                    (0..4)
                        .map(|i| rn((old[i * 4 + j] - 0.25 * old[i * 4 + j]) * coeff[i]))
                        .collect(),
                );
                rn(0.5 * (coeff[8 + j] - pred))
            } else {
                coeff[8 + j]
            }
        })
        .collect();
    for i in 0..4 {
        for j in 0..4 {
            let s = old[i * 4 + j];
            let b = coeff[i];
            let x = residual[j];
            state[i * 4 + j] = rn((s - 0.25 * s) + b * x);
        }
    }
    (0..4)
        .map(|j| {
            tree(
                (0..4)
                    .map(|i| rn(state[i * 4 + j] * coeff[4 + i]))
                    .collect(),
            )
        })
        .collect()
}
fn phase_port(producer: bool, phase: usize) -> Option<Port> {
    if producer {
        match phase {
            0 | 10 => Some(Port::Vector),
            1..=8 => Some(if phase % 2 == 1 {
                Port::MatrixWrite
            } else {
                Port::MatrixRead
            }),
            _ => None,
        }
    } else {
        match phase {
            0 => Some(Port::MatrixRead),
            1..=4 => Some(Port::Vector),
            6 => Some(Port::MatrixWrite),
            _ => None,
        }
    }
}
fn index(port: Port) -> usize {
    match port {
        Port::MatrixRead => 0,
        Port::MatrixWrite => 1,
        Port::Vector => 2,
    }
}
fn probe(slots: usize, slow_consumer: u64, kda: bool) -> ProbeResult {
    assert!((1..=4).contains(&slots));
    let tags: Vec<_> = (0..3)
        .flat_map(|token| {
            (0..4).map(move |request| Tag {
                request,
                token,
                group: request % 2,
            })
        })
        .collect();
    let initial: Vec<_> = (0..4)
        .map(|r| {
            (0..16)
                .map(|j| rn((j + r) as f32 / 64.0))
                .collect::<Vec<_>>()
        })
        .collect();
    let mut states = initial.clone();
    let mut expected = initial;
    let mut expected_outputs = Vec::new();
    for &tag in &tags {
        expected_outputs.push(consume(
            &mut expected[tag.request],
            &project(&inputs(tag), &weights(tag)),
            kda,
        ));
    }
    let mut owners = vec![Owner::Empty; slots];
    let mut slot_payload = vec![vec![0_f32; 32]; slots];
    let mut producer: Option<Context> = None;
    let mut consumer: Option<Context> = None;
    let mut next = 0;
    let mut retired = 0;
    let mut tick = 0;
    let mut ready_at = [0_u64; 3];
    let mut reservations = Vec::new();
    let mut producer_backpressure = 0;
    let mut port_wait = 0;
    let mut max_live_slots = 0;
    let mut outputs = Vec::new();
    // Physical upper Matrix rows160..223 are reused only after sector read.
    // State resides in lower rows. No re-association into four K64 sums.
    let mut upper_sector = vec![0_f32; 64 * 32];
    let mut weight_replay = vec![0_f32; 256 * 32];
    let mut input_latch = vec![0_f32; 256];
    let mut consumer_state = vec![0_f32; 16];
    let mut consumer_result = Vec::new();
    while retired < tags.len() {
        assert!(tick < 100000, "bounded handoff deadlock");
        // Complete operations first: no writes become visible at issue.
        if let Some(p) = producer.as_mut()
            && p.pending.as_ref().is_some_and(|x| x.end == tick)
        {
            let phase = p.pending.take().unwrap().phase;
            match phase {
                0 => input_latch = inputs(p.tag),
                1 | 3 | 5 | 7 => {
                    let sector = (phase - 1) / 2;
                    let w = weights(p.tag);
                    upper_sector.copy_from_slice(&w[sector * 64 * 32..(sector + 1) * 64 * 32]);
                }
                2 | 4 | 6 | 8 => {
                    let sector = (phase - 2) / 2;
                    weight_replay[sector * 64 * 32..(sector + 1) * 64 * 32]
                        .copy_from_slice(&upper_sector);
                }
                9 => p.payload = project(&input_latch, &weight_replay),
                10 => {
                    assert!(owners[p.slot] == Owner::Filling(p.tag));
                    slot_payload[p.slot].clone_from(&p.payload);
                    owners[p.slot] = Owner::Ready(p.tag);
                }
                _ => unreachable!(),
            }
            p.phase += 1;
        }
        if producer.as_ref().is_some_and(|p| p.phase == 11) {
            producer = None;
        }
        if let Some(c) = consumer.as_mut()
            && c.pending.as_ref().is_some_and(|x| x.end == tick)
        {
            let phase = c.pending.take().unwrap().phase;
            match phase {
                0 => consumer_state.clone_from(&states[c.tag.request]),
                1..=4 => {
                    assert!(
                        owners[c.slot] == Owner::Reading(c.tag),
                        "slot overwritten before final read"
                    );
                    let first = (phase - 1) * 8;
                    c.payload[first..first + 8]
                        .copy_from_slice(&slot_payload[c.slot][first..first + 8]);
                    if phase == 4 {
                        owners[c.slot] = Owner::Empty;
                    }
                }
                5 => consumer_result = consume(&mut consumer_state, &c.payload, kda),
                6 => {
                    states[c.tag.request].clone_from(&consumer_state);
                    outputs.push(consumer_result.clone());
                    retired += 1;
                }
                _ => unreachable!(),
            }
            c.phase += 1;
        }
        if consumer.as_ref().is_some_and(|c| c.phase == 7) {
            consumer = None;
        }
        if producer.is_none() && next < tags.len() {
            if let Some(slot) = owners.iter().position(|x| *x == Owner::Empty) {
                let tag = tags[next];
                next += 1;
                owners[slot] = Owner::Filling(tag);
                producer = Some(Context {
                    tag,
                    slot,
                    phase: 0,
                    pending: None,
                    payload: Vec::new(),
                });
            } else {
                producer_backpressure += 1;
            }
        }
        if consumer.is_none()
            && retired < tags.len()
            && let Some(slot) = owners
                .iter()
                .position(|x| *x == Owner::Ready(tags[retired]))
        {
            let tag = tags[retired];
            owners[slot] = Owner::Reading(tag);
            consumer = Some(Context {
                tag,
                slot,
                phase: 0,
                pending: None,
                payload: vec![0.0; 32],
            });
        }
        max_live_slots = max_live_slots.max(owners.iter().filter(|x| **x != Owner::Empty).count());
        // Rotating priority; each successful issue reserves a physical port.
        for is_producer in [tick % 2 == 0, tick % 2 != 0] {
            let context = if is_producer {
                producer.as_mut()
            } else {
                consumer.as_mut()
            };
            let Some(c) = context else { continue };
            if c.pending.is_some() {
                continue;
            }
            let port = phase_port(is_producer, c.phase);
            if port.is_some_and(|p| ready_at[index(p)] > tick) {
                port_wait += 1;
                continue;
            }
            let duration = if is_producer && c.phase == 0 {
                3 // Force an input refill to contend with consumer coefficient reads.
            } else if !is_producer && c.phase == 0 {
                1
            } else if port.is_some() {
                1 + (tick + c.tag.request as u64) % 3
            } else if is_producer {
                u64::from(8 * hardware().arithmetic_cycles())
            } else {
                slow_consumer + 6
            };
            let end = tick + duration;
            if let Some(port) = port {
                ready_at[index(port)] = end;
                reservations.push(Reservation {
                    port,
                    start: tick,
                    end,
                    producer: is_producer,
                    tag: c.tag,
                });
            }
            c.pending = Some(Pending {
                phase: c.phase,
                end,
            });
        }
        tick += 1;
    }
    for (i, a) in reservations.iter().enumerate() {
        for b in &reservations[i + 1..] {
            assert!(
                a.port != b.port || a.end <= b.start || b.end <= a.start,
                "double-booked port"
            );
        }
    }
    assert!(
        states == expected && outputs == expected_outputs,
        "mixed/stale state or coefficients"
    );
    assert!(owners.iter().all(|x| *x == Owner::Empty));
    ProbeResult {
        cycles: tick,
        jobs: tags.len(),
        slots,
        producer_backpressure,
        port_wait,
        max_live_slots,
        exact: true,
        preserved_k_reduction: 256,
        sector_rows: (160, 223),
        reservations,
    }
}
#[test]
fn bounded_handoff_numeric_backpressure_and_sector_lifetime() {
    for kda in [false, true] {
        for slots in [1, 2, 4] {
            let result = probe(slots, 600, kda);
            assert!(result.max_live_slots <= slots);
            assert!(result.producer_backpressure > 0);
            if slots > 1 {
                assert!(result.port_wait > 0);
            }
            println!(
                "PROJECTION_HANDOFF_PROBE model={} {}",
                if kda { "kda" } else { "mamba" },
                serde_json::to_string(&result).unwrap()
            );
        }
    }
}
#[test]
fn bounded_handoff_fast_consumer_has_no_early_read_or_alias() {
    for kda in [false, true] {
        let result = probe(2, 0, kda);
        assert!(result.exact);
    }
}
