//! Independent clock-stepped oracle for the split-K Matrix service candidate.
//! It executes numerical PE accumulators/tree stages, not a torch matmul.
//! This probe does not replace Matrix ISA execution or certify its integration.
use half::bf16;
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Hardware {
    edge: usize,
    reduction_lanes: usize,
    mac_latency: usize,
    mac_ii: usize,
    tree_add_latency: usize,
    matrix_read_elements: usize,
    vector_read_elements: usize,
    vector_write_elements: usize,
    matrix_capacity_bytes: usize,
    vector_capacity_bytes: usize,
    accumulator: String,
    #[serde(default)]
    weight_replay: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    hardware: Hardware,
    m: usize,
    n: usize,
    k: usize,
    a: Vec<f32>,
    b: Vec<f32>,
    write_stall: usize,
}

#[derive(Default, Serialize)]
struct Counters {
    issue: usize,
    sram: usize,
    arithmetic: usize,
    dependency: usize,
    total: usize,
    matrix_read_elements: usize,
    vector_read_elements: usize,
    vector_write_elements: usize,
    issued_macs: usize,
    output_bf16_bits: Vec<u16>,
}

fn rounded(value: f32, fp32: bool) -> f32 {
    if fp32 {
        value
    } else {
        bf16::from_f32(value).to_f32()
    }
}

fn execute(r: Request) -> Counters {
    let h = &r.hardware;
    assert!(
        !h.weight_replay,
        "generic Matrix probe does not implement projection replay"
    );
    let e = h.edge;
    assert!((1..=16).contains(&e) && h.reduction_lanes > 0);
    assert!(h.reduction_lanes.is_multiple_of(e));
    let groups = h.reduction_lanes / e;
    assert!(groups.is_power_of_two());
    assert!(
        [
            h.mac_latency,
            h.mac_ii,
            h.tree_add_latency,
            h.matrix_read_elements,
            h.vector_read_elements,
            h.vector_write_elements
        ]
        .iter()
        .all(|&x| x > 0)
    );
    assert!(r.m > 0 && r.n > 0 && r.k > 0);
    assert_eq!(r.a.len(), r.m.checked_mul(r.k).unwrap());
    assert_eq!(r.b.len(), r.k.checked_mul(r.n).unwrap());
    assert!(r.a.iter().chain(&r.b).all(|x| x.is_finite()));
    assert!(matches!(h.accumulator.as_str(), "BF16" | "FP32"));
    let fp32 = h.accumulator == "FP32";
    let operand_values = e * h.reduction_lanes;
    assert!(operand_values * 2 <= h.matrix_capacity_bytes);
    assert!(operand_values * 2 + e * e * 2 <= h.vector_capacity_bytes);
    let a =
        r.a.iter()
            .map(|&x| bf16::from_f32(x).to_f32())
            .collect::<Vec<_>>();
    let b =
        r.b.iter()
            .map(|&x| bf16::from_f32(x).to_f32())
            .collect::<Vec<_>>();
    let mut c = Counters {
        output_bf16_bits: vec![0; r.m * r.n],
        ..Default::default()
    };
    for row0 in (0..r.m).step_by(e) {
        for col0 in (0..r.n).step_by(e) {
            let mut tile = vec![0.0_f32; e * e];
            for k0 in (0..r.k).step_by(h.reduction_lanes) {
                c.issue += 1;
                // Consume finite independent read ports until BOTH operand
                // latches are full. Tail lanes are supplied as explicit zeros.
                let (mut matrix_left, mut vector_left) = (operand_values, operand_values);
                while matrix_left > 0 || vector_left > 0 {
                    matrix_left = matrix_left.saturating_sub(h.matrix_read_elements);
                    vector_left = vector_left.saturating_sub(h.vector_read_elements);
                    c.sram += 1;
                }
                c.matrix_read_elements += operand_values;
                c.vector_read_elements += operand_values;
                let mut sums = vec![0.0_f32; groups * e * e];
                let mut count = vec![0_usize; e * e];
                let mut ready = (0..e * e).map(|i| i / e + i % e).collect::<Vec<_>>();
                let mut last_launch = vec![None; e * e];
                let mut clock = 0;
                let mut last_result = 0;
                while count.iter().any(|&k| k < e) || clock < last_result {
                    for pe in 0..e * e {
                        let k = count[pe];
                        if k == e || clock < ready[pe] {
                            continue;
                        }
                        if last_launch[pe].is_some_and(|t| clock < t + h.mac_ii) {
                            continue;
                        }
                        let (row, col) = (row0 + pe / e, col0 + pe % e);
                        for group in 0..groups {
                            let reduction = k0 + group * e + k;
                            let av = if row < r.m && reduction < r.k {
                                a[row * r.k + reduction]
                            } else {
                                0.0
                            };
                            let bv = if col < r.n && reduction < r.k {
                                b[reduction * r.n + col]
                            } else {
                                0.0
                            };
                            let index = group * e * e + pe;
                            // Separate RN32 multiply/add; do not use mul_add.
                            let product = av * bv;
                            sums[index] = rounded(sums[index] + product, fp32);
                            c.issued_macs += 1;
                        }
                        ready[pe] = clock + h.mac_latency;
                        last_result = last_result.max(ready[pe]);
                        last_launch[pe] = Some(clock);
                        count[pe] += 1;
                    }
                    clock += 1;
                }
                c.arithmetic += clock;
                let mut active = groups;
                while active > 1 {
                    let mut next = vec![0.0; active / 2 * e * e];
                    for group in 0..active / 2 {
                        for pe in 0..e * e {
                            next[group * e * e + pe] = rounded(
                                sums[2 * group * e * e + pe] + sums[(2 * group + 1) * e * e + pe],
                                fp32,
                            );
                        }
                    }
                    sums = next;
                    active /= 2;
                    for _ in 0..h.tree_add_latency {
                        c.arithmetic += 1;
                    }
                }
                for pe in 0..e * e {
                    tile[pe] = rounded(tile[pe] + sums[pe], fp32);
                }
                for _ in 0..h.tree_add_latency {
                    c.arithmetic += 1;
                }
            }
            // Results survive write backpressure without recomputation or RNG.
            c.issue += 1;
            for _ in 0..r.write_stall {
                c.dependency += 1;
            }
            let mut left = e * e;
            while left > 0 {
                left = left.saturating_sub(h.vector_write_elements);
                c.sram += 1;
            }
            c.vector_write_elements += e * e;
            for (pe, value) in tile.into_iter().enumerate() {
                let (row, col) = (row0 + pe / e, col0 + pe % e);
                if row < r.m && col < r.n {
                    c.output_bf16_bits[row * r.n + col] = bf16::from_f32(value).to_bits();
                }
            }
        }
    }
    c.total = c.issue + c.sram + c.arithmetic + c.dependency;
    c
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();
    let request: Request = serde_json::from_str(&input).expect("invalid Matrix probe request");
    println!("{}", serde_json::to_string(&execute(request)).unwrap());
}
