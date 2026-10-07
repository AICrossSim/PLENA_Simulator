//! Fused delta-update lane. This module does not implement the KDA prediction
//! context or a complete v2 recurrence schedule. Its timing is a conservative
//! subpacket latency, with no assumed overlap across dependent SRAM packets.
use std::sync::Mutex;

#[derive(Debug, Clone, Copy)]
pub(crate) struct UpdateConfig {
    pub lanes: u32,
    pub latency: u32,
    pub interval: u32,
    pub stochastic: bool,
    pub seed: u32,
}
impl UpdateConfig {
    pub fn from_env() -> Self {
        fn number(name: &str, default: u32) -> u32 {
            std::env::var(name).map_or(default, |s| s.parse().expect("invalid v2 lane parameter"))
        }
        let c = Self {
            lanes: number("PLENA_V2_UPDATE_LANES", 256),
            latency: number("PLENA_V2_UPDATE_LATENCY", 6),
            interval: number("PLENA_V2_UPDATE_INTERVAL", 2),
            stochastic: match std::env::var("PLENA_V2_STATE_ROUNDING").as_deref() {
                Ok("sr") => true,
                Ok("rn") | Err(_) => false,
                _ => panic!("v2 state rounding must be rn or sr"),
            },
            seed: number("PLENA_V2_SR_SEED", 0x13579bdf),
        };
        c.validate();
        c
    }
    pub fn validate(&self) {
        assert!([128, 256, 512].contains(&self.lanes));
        assert!(self.latency > 0 && self.interval > 0 && self.seed != 0);
    }
    pub fn packet_cycles(&self, values: usize) -> u32 {
        assert!(values > 0);
        self.latency + ((values as u32).div_ceil(self.lanes) - 1) * self.interval
    }
}

pub(crate) struct UpdateLane {
    pub config: UpdateConfig,
    rng: Mutex<Vec<u32>>,
}
impl UpdateLane {
    pub fn new(config: UpdateConfig) -> Self {
        config.validate();
        // Distinct deterministic per-physical-lane seeds. The GPU A2 experiment
        // uses Philox instead; its trajectory must not be called RTL-exact.
        let rng = (0..config.lanes)
            .map(|i| {
                if i == 0 {
                    config.seed
                } else {
                    config.seed.wrapping_add(i.wrapping_mul(0x9e3779b9)) | 1
                }
            })
            .collect();
        Self {
            config,
            rng: Mutex::new(rng),
        }
    }
    pub fn update(&self, lane: usize, s: f32, delta: f32, b: f32, x: f32) -> f32 {
        assert!(lane < self.config.lanes as usize);
        for value in [s, delta, b, x] {
            assert_eq!(
                value.to_bits() & 0xffff,
                0,
                "fused update inputs must be BF16"
            );
            assert!(value.is_finite(), "v2 finite-input contract");
        }
        // Deliberately separate FP32 operations: no mul_add and no BF16
        // boundary until the final store. Matches the MODE3 host reference.
        let product = delta * s;
        let decayed = s - product;
        let outer = b * x;
        let result = decayed + outer;
        let random = if self.config.stochastic {
            let mut rng = self.rng.lock().expect("v2 RNG lock poisoned");
            let random = (rng[lane] & 0xffff) as u16;
            rng[lane] = (rng[lane] >> 1) ^ if rng[lane] & 1 != 0 { 0x80200003 } else { 0 };
            Some(random)
        } else {
            None
        };
        f32::from_bits(u32::from(round_bf16(result, random)) << 16)
    }
}

pub(crate) fn round_bf16(value: f32, random: Option<u16>) -> u16 {
    let bits = value.to_bits();
    if bits & 0x7f800000 == 0x7f800000 {
        return ((bits >> 16) as u16) | if bits & 0x7fffff != 0 { 0x40 } else { 0 };
    }
    let upper = (bits >> 16) as u16;
    match random {
        Some(r) => upper.wrapping_add(u16::from(r < (bits & 0xffff) as u16)),
        None => (bits.wrapping_add(0x7fff + u32::from(upper & 1)) >> 16) as u16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_small_decay_until_final_store() {
        let lane = UpdateLane::new(UpdateConfig {
            lanes: 128,
            latency: 6,
            interval: 2,
            stochastic: false,
            seed: 1,
        });
        // Fused update cancels the tiny decay using the outer term exactly;
        // rounding an intermediate would lose that contribution.
        assert_eq!(lane.update(0, 1.0, 1.0 / 512.0, 1.0, 1.0 / 512.0), 1.0);
        assert_eq!(lane.config.packet_cycles(2048), 36);
    }
    #[test]
    fn sr_is_unbiased_over_every_threshold_for_both_signs() {
        for value in [1.003_f32, -1.003_f32, 0.000123_f32, -0.000123_f32] {
            let bits = value.to_bits();
            let top = (bits >> 16) as u16;
            let increments = (0..=u16::MAX)
                .filter(|&r| round_bf16(value, Some(r)) != top)
                .count();
            assert_eq!(increments, (bits & 0xffff) as usize);
        }
    }
    #[test]
    fn lane_zero_matches_mode3_lfsr_stream() {
        let c = UpdateConfig {
            lanes: 256,
            latency: 6,
            interval: 2,
            stochastic: true,
            seed: 0x13579bdf,
        };
        let lane = UpdateLane::new(c);
        let mut rng = c.seed;
        for _ in 0..8192 {
            let value = 1.0_f32 - 1.0 / 512.0 + 1.0 / 1024.0;
            assert_eq!(
                lane.update(0, 1.0, 1.0 / 512.0, 1.0, 1.0 / 1024.0)
                    .to_bits(),
                u32::from(round_bf16(value, Some(rng as u16))) << 16
            );
            rng = (rng >> 1) ^ if rng & 1 != 0 { 0x80200003 } else { 0 };
        }
    }
}
