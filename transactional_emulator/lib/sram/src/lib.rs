//! SRAM models for the PLENA simulator.
//!
//! Hosts two SRAM types that share the same async-locked, cell-indexed
//! storage shape with optional per-cell delayed writes:
//!
//! - [`MatrixSram`] — 2D tile storage (`tile_size × tile_size` elements per cell)
//! - [`VectorSram`] — 1D row storage (`vlen` elements per cell)
//!
//! Both representations use the shared [`Cell`] state and [`addr_to_cell`]
//! address arithmetic exposed at the crate root.

use quantize::QuantTensor;
use tokio::sync::oneshot::Receiver;

pub mod matrix;
pub mod vector;

pub use matrix::MatrixSram;
pub use vector::VectorSram;

/// State of a single SRAM cell.
///
/// A freshly-written cell is [`Cell::Ready`]. A cell awaiting a delayed write
/// holds a [`Cell::Pending`] receiver that resolves to a `QuantTensor` when
/// the producer completes; consumers must `await` it on first read and then
/// transition the cell to `Ready`.
pub(crate) enum Cell<T> {
    Ready(T),
    Pending(Receiver<QuantTensor>),
}

/// Convert an element-address into a cell index, asserting the address is
/// aligned to `units_per_cell`.
pub(crate) fn addr_to_cell(addr: u32, units_per_cell: u32) -> usize {
    assert!(
        addr % units_per_cell == 0,
        "address {} not multiple of {}",
        addr,
        units_per_cell
    );
    (addr / units_per_cell) as usize
}
