//! Executable V0: expert grouping, independent normal SRAM/accumulators,
//! finite double-buffered MX ingress, and one shared HBM/vector path.
//! This is a numerical architecture experiment, not the ISA or RTL runner.
mod dma_credits;
mod engine;
mod read_cache;
mod types;

// `execute` is retained for an embedding caller with an existing executor.
#[allow(unused_imports)]
pub use engine::{execute, run, validate};
pub use types::*;

#[cfg(test)]
mod tests;
