//! Systolic Array Behavioral Model
//!
//! This module provides a detailed behavioral simulation of systolic arrays
//! with support for different dataflows:
//! - Weight Stationary (WS): Weights are fixed in PEs, activations flow through
//! - Output Stationary (OS): Partial sums stay in PEs, weights and activations flow
//! - Activation/Input Stationary (IS): Activations are fixed, weights flow through
//!
//! The model includes cycle-accurate timing for data movement and computation.

mod pe;
mod dataflow;
mod array;
mod stats;

pub use pe::{ProcessingElement, PEState};
pub use dataflow::Dataflow;
pub use array::{SystolicArray, SystolicConfig, BufferState, BandwidthConfig};
pub use stats::SystolicStats;
