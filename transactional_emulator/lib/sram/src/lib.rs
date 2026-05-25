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
/// `Cell::Ready(T)` holds the cell's current value. `Cell::Pending` holds a
/// [`Receiver`] that yields a [`QuantTensor`] produced by some upstream
/// async operation; the cell transitions to `Ready` on first observation
/// via [`Cell::resolve_with`] (or [`Cell::resolve`] when `T = QuantTensor`).
///
/// The `Pending` variant is hard-coded to `Receiver<QuantTensor>` because
/// both consumers receive their input from tensor-producing channels even
/// when their cell payload differs (e.g. `Vec<u8>` for `VectorSram`).
pub(crate) enum Cell<T> {
    Ready(T),
    Pending(Receiver<QuantTensor>),
}

impl<T> Cell<T> {
    /// If pending, await the channel and store `convert(tensor)` as the new
    /// ready value, then return a reference to the ready value.
    pub(crate) async fn resolve_with(&mut self, convert: impl FnOnce(QuantTensor) -> T) -> &T {
        if let Cell::Pending(fut) = self {
            *self = Cell::Ready(convert(fut.await.unwrap()));
        }
        match self {
            Cell::Ready(t) => t,
            Cell::Pending(_) => unreachable!(),
        }
    }
}

impl Cell<QuantTensor> {
    /// Convenience for cells whose payload is already a [`QuantTensor`]; no
    /// conversion needed.
    pub(crate) async fn resolve(&mut self) -> &QuantTensor {
        self.resolve_with(|t| t).await
    }
}

/// Convert an element-address into a cell index, asserting both alignment
/// (`addr` is a multiple of `units_per_cell`) and bounds (`idx < depth`).
pub(crate) fn addr_to_cell(addr: u32, units_per_cell: u32, depth: usize) -> usize {
    assert!(
        addr % units_per_cell == 0,
        "address {} not multiple of {}",
        addr,
        units_per_cell
    );
    let idx = (addr / units_per_cell) as usize;
    assert!(
        idx < depth,
        "address {} out of bounds (cell index {} >= depth {})",
        addr,
        idx,
        depth
    );
    idx
}

/// Assert that `x` is a multiple of `mul`, returning `x / mul`.
///
/// Used by callers (e.g. compute engines) that need an aligned address to be
/// reduced to its quotient — distinct from [`addr_to_cell`], which also
/// enforces a depth bound.
pub fn assert_multiple_of(x: u32, mul: u32) -> u32 {
    assert!(x.is_multiple_of(mul));
    x / mul
}

/// Decompose `x` into `(base, offset)` such that `base + offset == x`,
/// where `base` is the largest multiple of `mul` not exceeding `x`.
pub fn multiple_and_offset(x: u32, mul: u32) -> (u32, u32) {
    let d = x / mul;
    let r = x % mul;
    (d * mul, r)
}
