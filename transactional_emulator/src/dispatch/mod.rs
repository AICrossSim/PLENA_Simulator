//! Per-category opcode dispatchers for the transactional emulator.
//!
//! Each submodule provides one `impl Accelerator` block with a single
//! `dispatch_<category>` method. The top-level `Accelerator::do_ops`
//! groups match arms by category and delegates here.

mod control;
mod matrix;
mod mem;
mod scalar;
mod vector;
