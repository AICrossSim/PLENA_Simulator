//! Processing Element (PE) Model
//!
//! Each PE in the systolic array contains:
//! - A weight register
//! - An activation input register
//! - A partial sum accumulator
//! - Logic for MAC (Multiply-Accumulate) operations

use serde::{Deserialize, Serialize};

/// State of a Processing Element
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PEState {
    /// PE is idle, waiting for data
    Idle,
    /// PE has loaded weight and is ready for computation
    WeightLoaded,
    /// PE is actively computing (MAC operation)
    Computing,
}

impl Default for PEState {
    fn default() -> Self {
        PEState::Idle
    }
}

/// Processing Element for systolic array
#[derive(Debug, Clone)]
pub struct ProcessingElement {
    /// Row position in the array
    pub row: usize,
    /// Column position in the array
    pub col: usize,
    /// Weight register (stationary in WS dataflow)
    pub weight: f32,
    /// Activation input register
    pub activation: f32,
    /// Partial sum accumulator
    pub accumulator: f32,
    /// Current state of the PE
    pub state: PEState,
    /// Number of MAC operations performed
    pub mac_count: u64,
    /// Whether weight is valid (loaded)
    pub weight_valid: bool,
    /// Whether activation is valid (received this cycle)
    pub activation_valid: bool,
    /// Pipeline register for activation passing (horizontal)
    pub activation_out: f32,
    /// Pipeline register for weight passing (vertical in IS mode)
    pub weight_out: f32,
    /// Pipeline register for partial sum passing (vertical in OS mode)
    pub psum_out: f32,
    /// Valid signals for pipeline registers
    pub activation_out_valid: bool,
    pub weight_out_valid: bool,
    pub psum_out_valid: bool,
}

impl ProcessingElement {
    /// Create a new Processing Element at the given position
    pub fn new(row: usize, col: usize) -> Self {
        Self {
            row,
            col,
            weight: 0.0,
            activation: 0.0,
            accumulator: 0.0,
            state: PEState::Idle,
            mac_count: 0,
            weight_valid: false,
            activation_valid: false,
            activation_out: 0.0,
            weight_out: 0.0,
            psum_out: 0.0,
            activation_out_valid: false,
            weight_out_valid: false,
            psum_out_valid: false,
        }
    }

    /// Reset the PE to initial state
    pub fn reset(&mut self) {
        self.weight = 0.0;
        self.activation = 0.0;
        self.accumulator = 0.0;
        self.state = PEState::Idle;
        self.weight_valid = false;
        self.activation_valid = false;
        self.activation_out = 0.0;
        self.weight_out = 0.0;
        self.psum_out = 0.0;
        self.activation_out_valid = false;
        self.weight_out_valid = false;
        self.psum_out_valid = false;
        // Note: mac_count is NOT reset for statistics tracking
    }

    /// Load a weight into the PE (for Weight Stationary dataflow)
    pub fn load_weight(&mut self, weight: f32) {
        self.weight = weight;
        self.weight_valid = true;
        self.state = PEState::WeightLoaded;
    }

    /// Load an activation into the PE
    pub fn load_activation(&mut self, activation: f32) {
        self.activation = activation;
        self.activation_valid = true;
    }

    /// Load a partial sum (for Output Stationary - receiving from above)
    pub fn load_psum(&mut self, psum: f32) {
        self.accumulator = psum;
    }

    /// Perform a single MAC operation: acc += weight * activation
    /// Returns true if a valid MAC was performed
    pub fn compute_mac(&mut self) -> bool {
        if self.weight_valid && self.activation_valid {
            self.accumulator += self.weight * self.activation;
            self.mac_count += 1;
            self.state = PEState::Computing;
            true
        } else {
            false
        }
    }

    /// Compute MAC and prepare outputs for Weight Stationary dataflow
    /// - Weight stays in place
    /// - Activation is forwarded to the right
    /// - Partial sum accumulates locally
    pub fn step_weight_stationary(&mut self, activation_in: Option<f32>) -> Option<f32> {
        // Receive new activation if available
        if let Some(act) = activation_in {
            self.activation = act;
            self.activation_valid = true;
        }

        // Perform MAC if we have valid weight and activation
        if self.weight_valid && self.activation_valid {
            self.accumulator += self.weight * self.activation;
            self.mac_count += 1;
            self.state = PEState::Computing;

            // Forward activation to the right
            self.activation_out = self.activation;
            self.activation_out_valid = true;

            // Clear activation for next cycle
            self.activation_valid = false;

            Some(self.activation_out)
        } else {
            self.activation_out_valid = false;
            None
        }
    }

    /// Compute MAC and prepare outputs for Output Stationary dataflow
    /// - Partial sum stays in place
    /// - Weights flow vertically (top to bottom)
    /// - Activations flow horizontally (left to right)
    pub fn step_output_stationary(
        &mut self,
        weight_in: Option<f32>,
        activation_in: Option<f32>,
    ) -> (Option<f32>, Option<f32>) {
        // Receive weight from above
        if let Some(w) = weight_in {
            self.weight = w;
            self.weight_valid = true;
        }

        // Receive activation from left
        if let Some(act) = activation_in {
            self.activation = act;
            self.activation_valid = true;
        }

        // Perform MAC if both weight and activation are valid
        if self.weight_valid && self.activation_valid {
            self.accumulator += self.weight * self.activation;
            self.mac_count += 1;
            self.state = PEState::Computing;

            // Forward weight downward
            self.weight_out = self.weight;
            self.weight_out_valid = true;

            // Forward activation to the right
            self.activation_out = self.activation;
            self.activation_out_valid = true;

            // Clear valid flags for next inputs
            self.weight_valid = false;
            self.activation_valid = false;

            (Some(self.weight_out), Some(self.activation_out))
        } else {
            self.weight_out_valid = false;
            self.activation_out_valid = false;
            (None, None)
        }
    }

    /// Compute MAC and prepare outputs for Activation/Input Stationary dataflow
    /// - Activations stay in place
    /// - Weights flow horizontally (left to right) or vertically
    /// - Partial sums flow vertically (top to bottom)
    pub fn step_activation_stationary(
        &mut self,
        weight_in: Option<f32>,
        psum_in: Option<f32>,
    ) -> (Option<f32>, Option<f32>) {
        // Receive weight from left
        if let Some(w) = weight_in {
            self.weight = w;
            self.weight_valid = true;
        }

        // Receive partial sum from above
        let incoming_psum = psum_in.unwrap_or(0.0);

        // Perform MAC if we have valid activation and weight
        if self.activation_valid && self.weight_valid {
            let local_product = self.weight * self.activation;
            self.psum_out = incoming_psum + local_product;
            self.psum_out_valid = true;
            self.mac_count += 1;
            self.state = PEState::Computing;

            // Forward weight to the right
            self.weight_out = self.weight;
            self.weight_out_valid = true;

            // Clear weight valid for next input
            self.weight_valid = false;

            (Some(self.weight_out), Some(self.psum_out))
        } else {
            // Just pass through the partial sum
            self.psum_out = incoming_psum;
            self.psum_out_valid = psum_in.is_some();
            self.weight_out_valid = false;
            (None, if self.psum_out_valid { Some(self.psum_out) } else { None })
        }
    }

    /// Get the current accumulated result
    pub fn get_result(&self) -> f32 {
        self.accumulator
    }

    /// Clear the accumulator (for starting a new computation)
    pub fn clear_accumulator(&mut self) {
        self.accumulator = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pe_mac_operation() {
        let mut pe = ProcessingElement::new(0, 0);
        pe.load_weight(2.0);
        pe.load_activation(3.0);
        assert!(pe.compute_mac());
        assert_eq!(pe.get_result(), 6.0);

        pe.load_activation(4.0);
        pe.compute_mac();
        assert_eq!(pe.get_result(), 14.0); // 6 + 2*4
    }

    #[test]
    fn test_pe_weight_stationary() {
        let mut pe = ProcessingElement::new(0, 0);
        pe.load_weight(2.0);

        let out1 = pe.step_weight_stationary(Some(3.0));
        assert_eq!(out1, Some(3.0)); // Activation forwarded
        assert_eq!(pe.get_result(), 6.0);

        let out2 = pe.step_weight_stationary(Some(4.0));
        assert_eq!(out2, Some(4.0));
        assert_eq!(pe.get_result(), 14.0);
    }

    #[test]
    fn test_pe_reset() {
        let mut pe = ProcessingElement::new(1, 2);
        pe.load_weight(5.0);
        pe.load_activation(3.0);
        pe.compute_mac();
        assert_eq!(pe.mac_count, 1);

        pe.reset();
        assert_eq!(pe.weight, 0.0);
        assert_eq!(pe.accumulator, 0.0);
        assert_eq!(pe.state, PEState::Idle);
        assert_eq!(pe.mac_count, 1); // MAC count preserved
    }
}
