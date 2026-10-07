//! Latency charge capture — the single seam through which every modeled
//! instruction latency flows.
//!
//! All per-op cycle costs in the emulator are charged through exactly two
//! sites: the `cycle!` macro (scalar/vector arms) and `MatrixCore::compute`
//! (matrix ops). Both delegate here, so the timing mode below decides what a
//! charge means:
//!
//! - [`TimingMode::Serial`]: sleep the calling task on the virtual clock
//!   (`resolve_at(PERIOD * cycles)`) — bit-identical to the historical
//!   behavior where total latency is the sum of every instruction.
//! - [`TimingMode::Scoreboard`]: accumulate the cycles into a per-op counter
//!   without sleeping; the dispatch loop drains it via [`take_charged`] and
//!   feeds the analytic scoreboard (`accelerator::scoreboard`), which models
//!   pipelined overlap (stalls only on data dependencies and structural
//!   hazards).
//!
//! Dispatch must never compute a latency itself — runtime-valued costs
//! (`V_TOPK` scales with expert count, `MatrixCoreProfile` cycle multipliers)
//! only stay correct because the same functional code produces the number in
//! both modes.
//!
//! The mode and accumulator are thread-local: the executor and every machine
//! method run on the thread that called `Executor::enter`, and exactly one
//! dispatch task exists per executor. DMA/ramulator tasks never charge
//! cycles. A future multi-accelerator run must move this into per-task state.

use std::cell::Cell;

use crate::runtime_config::PERIOD;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TimingMode {
    /// Charge by sleeping the calling task (historical serial behavior).
    Serial,
    /// Charge by accumulating; the dispatch loop applies the latency to the
    /// analytic scoreboard instead.
    Scoreboard,
}

thread_local! {
    static MODE: Cell<TimingMode> = const { Cell::new(TimingMode::Serial) };
    static CHARGED_CYCLES: Cell<u64> = const { Cell::new(0) };
}

pub(crate) fn set_timing_mode(mode: TimingMode) {
    MODE.with(|m| m.set(mode));
}

pub(crate) fn timing_mode() -> TimingMode {
    MODE.with(|m| m.get())
}

/// Charge `cycles` of latency to the current instruction.
///
/// Serial mode sleeps the calling task; scoreboard mode records the cycles
/// for [`take_charged`] without advancing the virtual clock.
pub(crate) async fn charge_cycles(cycles: u32) {
    match timing_mode() {
        TimingMode::Serial => {
            runtime::Executor::current()
                .resolve_at(PERIOD * cycles)
                .await;
        }
        TimingMode::Scoreboard => {
            CHARGED_CYCLES.with(|c| c.set(c.get() + cycles as u64));
        }
    }
}

/// Cycles charged since the last [`take_charged`] (scoreboard mode only).
///
/// Used by dispatch to assert no charge leaks across instruction boundaries.
pub(crate) fn pending_charge() -> u64 {
    CHARGED_CYCLES.with(|c| c.get())
}

/// Take-and-reset the accumulated charge for the instruction that just ran.
pub(crate) fn take_charged() -> u64 {
    CHARGED_CYCLES.with(|c| c.replace(0))
}

#[cfg(test)]
mod tests {
    use runtime::{Duration, Executor, Instant};

    use super::*;

    #[tokio::test]
    async fn serial_mode_sleeps_the_calling_task() {
        set_timing_mode(TimingMode::Serial);
        let executor = Executor::new();
        executor.spawn(async {
            charge_cycles(5).await;
            charge_cycles(3).await;
        });
        executor.enter(Instant::ETERNITY).await;
        assert_eq!(executor.now(), Instant::INIT + PERIOD * 8u32);
    }

    #[tokio::test]
    async fn scoreboard_mode_accumulates_without_sleeping() {
        set_timing_mode(TimingMode::Scoreboard);
        let executor = Executor::new();
        executor.spawn(async {
            charge_cycles(5).await;
            charge_cycles(3).await;
            assert_eq!(pending_charge(), 8);
            assert_eq!(take_charged(), 8);
            assert_eq!(pending_charge(), 0);
        });
        executor.enter(Instant::ETERNITY).await;
        assert_eq!(executor.now(), Instant::INIT + Duration::from_picos(0));
        // Restore the default for other tests on this thread.
        set_timing_mode(TimingMode::Serial);
    }
}
