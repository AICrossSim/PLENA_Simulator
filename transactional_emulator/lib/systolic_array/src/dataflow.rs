//! Dataflow Types for Systolic Arrays
//!
//! This module defines the different dataflow strategies for systolic arrays.

use serde::{Deserialize, Serialize};

/// Dataflow strategy for systolic array computation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dataflow {
    /// Weight Stationary: Weights fixed, activations flow horizontally
    WeightStationary,
    /// Output Stationary: Partial sums fixed, weights/activations both flow
    OutputStationary,
    /// Activation/Input Stationary: Activations fixed, weights flow
    ActivationStationary,
}

impl Default for Dataflow {
    fn default() -> Self {
        Dataflow::WeightStationary
    }
}

impl Dataflow {
    /// Get a human-readable name for the dataflow
    pub fn name(&self) -> &'static str {
        match self {
            Dataflow::WeightStationary => "Weight Stationary",
            Dataflow::OutputStationary => "Output Stationary",
            Dataflow::ActivationStationary => "Activation Stationary",
        }
    }

    /// Get the abbreviation for the dataflow
    pub fn abbrev(&self) -> &'static str {
        match self {
            Dataflow::WeightStationary => "WS",
            Dataflow::OutputStationary => "OS",
            Dataflow::ActivationStationary => "IS",
        }
    }

    /// Describe the data movement pattern
    pub fn describe(&self) -> &'static str {
        match self {
            Dataflow::WeightStationary => {
                "Weights stationary in PEs, activations flow left-to-right"
            }
            Dataflow::OutputStationary => {
                "Partial sums stationary, weights flow top-to-bottom, activations left-to-right"
            }
            Dataflow::ActivationStationary => {
                "Activations stationary, weights flow left-to-right, psums flow top-to-bottom"
            }
        }
    }

    /// Get the number of cycles for weight loading phase
    pub fn weight_load_cycles(&self, array_size: usize, k_dim: usize) -> usize {
        match self {
            Dataflow::WeightStationary => array_size + k_dim - 1,
            Dataflow::OutputStationary => 0,
            Dataflow::ActivationStationary => 0,
        }
    }

    /// Get the number of cycles for the compute phase
    pub fn compute_cycles(&self, array_size: usize, m_dim: usize, k_dim: usize) -> usize {
        match self {
            Dataflow::WeightStationary => {
                let pipeline_latency = 2 * (array_size - 1);
                m_dim * k_dim + pipeline_latency
            }
            Dataflow::OutputStationary => {
                let pipeline_latency = 2 * (array_size - 1);
                m_dim * k_dim + pipeline_latency
            }
            Dataflow::ActivationStationary => {
                let pipeline_latency = array_size - 1;
                m_dim * k_dim + pipeline_latency
            }
        }
    }

    /// Get the number of cycles for draining results
    pub fn drain_cycles(&self, array_size: usize) -> usize {
        array_size - 1
    }

    /// Get the total cycles for a complete matrix operation
    pub fn total_cycles(&self, array_size: usize, m_dim: usize, k_dim: usize, n_dim: usize) -> usize {
        let m_tiles = (m_dim + array_size - 1) / array_size;
        let n_tiles = (n_dim + array_size - 1) / array_size;
        let k_tiles = (k_dim + array_size - 1) / array_size;

        let cycles_per_output_tile = match self {
            Dataflow::WeightStationary => {
                let weight_cycles = self.weight_load_cycles(array_size, array_size);
                let compute_cycles = self.compute_cycles(array_size, array_size, array_size);
                weight_cycles + k_tiles * compute_cycles
            }
            Dataflow::OutputStationary => {
                k_tiles * self.compute_cycles(array_size, array_size, array_size)
            }
            Dataflow::ActivationStationary => {
                self.compute_cycles(array_size, array_size, array_size) * k_tiles
            }
        };

        m_tiles * n_tiles * cycles_per_output_tile + self.drain_cycles(array_size)
    }

    /// Estimate energy efficiency (relative, lower is better)
    pub fn relative_energy(&self, array_size: usize, m_dim: usize, k_dim: usize, n_dim: usize) -> f64 {
        let total_macs = (m_dim * k_dim * n_dim) as f64;
        
        match self {
            Dataflow::WeightStationary => {
                let weight_loads = (k_dim * n_dim) as f64;
                let activation_moves = (m_dim * k_dim * n_dim / array_size) as f64;
                (weight_loads + activation_moves) / total_macs
            }
            Dataflow::OutputStationary => {
                let weight_moves = (k_dim * n_dim * m_dim / array_size) as f64;
                let activation_moves = (m_dim * k_dim * n_dim / array_size) as f64;
                (weight_moves + activation_moves) / total_macs
            }
            Dataflow::ActivationStationary => {
                let activation_loads = (m_dim * k_dim) as f64;
                let weight_moves = (k_dim * n_dim * m_dim / array_size) as f64;
                (activation_loads + weight_moves) / total_macs
            }
        }
    }
}

impl std::fmt::Display for Dataflow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::str::FromStr for Dataflow {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ws" | "weight_stationary" | "weightstationary" => Ok(Dataflow::WeightStationary),
            "os" | "output_stationary" | "outputstationary" => Ok(Dataflow::OutputStationary),
            "is" | "as" | "activation_stationary" | "activationstationary" => {
                Ok(Dataflow::ActivationStationary)
            }
            _ => Err(format!("Unknown dataflow: {}. Use WS, OS, or IS.", s)),
        }
    }
}
