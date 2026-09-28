//! Opt-in, bounded M_MV service for BF16 layer integration.
//!
//! This is a candidate implementation contract, not synthesized timing. The
//! legacy Matrix path is unchanged without an explicit profile. Each M_MV
//! consumes ONE rectangular view, retains BLEN partial sums until M_MV_WO,
//! and serializes bank read, Vector read, mini-array/tree, and final write.
use half::bf16;
use serde::Deserialize;
use std::sync::LazyLock;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct MatrixService {
    pub edge: u32,
    pub reduction_lanes: u32,
    pub mac_latency: u32,
    pub mac_ii: u32,
    pub tree_add_latency: u32,
    pub matrix_read_elements: u32,
    pub vector_read_elements: u32,
    pub vector_write_elements: u32,
    pub matrix_capacity_bytes: u32,
    pub vector_capacity_bytes: u32,
    pub accumulator: String,
    /// Opt-in finite 16 KiB BF16 weight replay; legacy M_MV stays unchanged.
    #[serde(default)]
    pub weight_replay: bool,
}

pub(crate) static PROFILE: LazyLock<Option<MatrixService>> = LazyLock::new(|| {
    std::env::var_os("PLENA_MATRIX_SERVICE_PROFILE").map(|path| {
        let value: MatrixService = serde_json::from_slice(
            &std::fs::read(path).expect("cannot read Matrix service profile"),
        )
        .expect("invalid Matrix service profile");
        value.validate();
        value
    })
});

impl MatrixService {
    pub fn validate(&self) {
        assert!((1..=16).contains(&self.edge));
        assert!(self.reduction_lanes > 0 && self.reduction_lanes.is_multiple_of(self.edge));
        assert!((self.reduction_lanes / self.edge).is_power_of_two());
        assert!(
            [
                self.mac_latency,
                self.mac_ii,
                self.tree_add_latency,
                self.matrix_read_elements,
                self.vector_read_elements,
                self.vector_write_elements
            ]
            .iter()
            .all(|&n| n > 0)
        );
        assert!(matches!(self.accumulator.as_str(), "BF16" | "FP32"));
        let bytes = u64::from(self.edge) * u64::from(self.reduction_lanes) * 2;
        assert!(bytes <= u64::from(self.matrix_capacity_bytes));
        assert!(bytes + u64::from(self.edge).pow(2) * 2 <= u64::from(self.vector_capacity_bytes));
    }

    /// Serialized bounded-latch supply, charged separately from SRAM preload.
    /// No arbitrary crossbar or extra SRAM port is assumed.
    pub fn replay_feed_cycles(&self, rows: u32, requests: u32) -> u32 {
        (rows * self.edge).div_ceil(self.matrix_read_elements)
            + (rows * requests).div_ceil(self.vector_read_elements)
    }

    pub fn rounded(&self, value: f32) -> f32 {
        if self.accumulator == "FP32" {
            value
        } else {
            bf16::from_f32(value).to_f32()
        }
    }

    pub fn arithmetic_cycles(&self) -> u32 {
        2 * (self.edge - 1)
            + (self.edge - 1) * self.mac_ii.max(self.mac_latency)
            + self.mac_latency
            + (self.reduction_lanes / self.edge).ilog2() * self.tree_add_latency
            + self.tree_add_latency
    }

    /// One live row of a padded edge x edge mini-array output tile. The other
    /// rows/columns execute zeros and still cost cycles. Inputs have already
    /// been read from real SRAM cells, never supplied by a host reference.
    pub fn column(&self, input: &[f32], weight: &[f32], previous: f32) -> f32 {
        assert_eq!(input.len(), weight.len());
        assert!(input.len() <= self.reduction_lanes as usize);
        let mut groups = vec![0.0_f32; (self.reduction_lanes / self.edge) as usize];
        for (g, sum) in groups.iter_mut().enumerate() {
            for local in 0..self.edge as usize {
                let k = g * self.edge as usize + local;
                let product = if k < input.len() {
                    input[k] * weight[k]
                } else {
                    0.0
                };
                *sum = self.rounded(*sum + product);
            }
        }
        while groups.len() > 1 {
            groups = groups
                .chunks_exact(2)
                .map(|p| self.rounded(p[0] + p[1]))
                .collect();
        }
        self.rounded(previous + groups[0])
    }
}

/// M_MM.P bounded panel configuration. Strides are in element units after
/// decoding; zero strides and reserved bits are rejected, not silently masked.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ProjectionConfig {
    pub requests: u32,
    pub input_stride: u32,
    pub output_stride: u32,
}
impl ProjectionConfig {
    pub fn decode(value: u32) -> Self {
        assert_eq!(value >> 18, 0, "M_MM.P reserved config bits must be zero");
        let config = Self {
            requests: (value & 3) + 1,
            input_stride: ((value >> 2) & 255) * 256,
            output_stride: ((value >> 10) & 255) * 32,
        };
        assert!(config.input_stride > 0 && config.output_stride > 0);
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn hardware() -> MatrixService {
        MatrixService {
            edge: 4,
            reduction_lanes: 8,
            mac_latency: 2,
            mac_ii: 1,
            tree_add_latency: 2,
            matrix_read_elements: 32,
            vector_read_elements: 32,
            vector_write_elements: 32,
            matrix_capacity_bytes: 1024,
            vector_capacity_bytes: 1024,
            accumulator: "BF16".into(),
            weight_replay: false,
        }
    }
    #[test]
    fn projection_config_decodes_units_and_rejects_reserved_or_zero_stride() {
        let config = ProjectionConfig::decode(3 | (8 << 2) | (64 << 10));
        assert_eq!(
            (config.requests, config.input_stride, config.output_stride),
            (4, 2048, 2048)
        );
        for invalid in [
            0,
            3 | (8 << 2),
            3 | (64 << 10),
            (1 << 18) | (8 << 2) | (64 << 10),
        ] {
            assert!(std::panic::catch_unwind(|| ProjectionConfig::decode(invalid)).is_err());
        }
    }

    #[test]
    fn tails_and_cross_instruction_accumulation() {
        let h = hardware();
        h.validate();
        let x = [1.0, -2.0, 3.0, 0.5, 4.0];
        let w = [2.0, 1.0, -1.0, 4.0, 0.25];
        assert_eq!(h.column(&x, &w, 0.0), 0.0);
        assert_eq!(h.column(&x, &w, 7.0), 7.0);
        assert_eq!(h.arithmetic_cycles(), 18);
    }
    #[test]
    fn feedback_latency_limits_launches() {
        let mut h = hardware();
        let fast = h.arithmetic_cycles();
        h.mac_latency = 4;
        assert_eq!(h.arithmetic_cycles() - fast, 8);
    }
}
