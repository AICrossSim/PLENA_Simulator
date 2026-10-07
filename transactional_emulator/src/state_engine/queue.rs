use std::sync::atomic::{AtomicU32, Ordering};

use tokio::sync::Notify;

use super::generated_contract::StateStatus;

const PENDING: u32 = u32::MAX;

/// One-shot latch carrying a terminal [`StateStatus`].
///
/// Safety depends on the simulator's cooperative executor. `Notify::notified()`
/// only registers a waiter when the future is first polled, and
/// `notify_waiters()` stores no permit, so a `signal` landing between the status
/// re-check in [`CompletionLatch::wait`] and that first poll would be lost. No
/// such interleaving exists here: `Executor::enter` polls ready tasks in
/// sequence on one thread, and `wait` has no await point between the re-check
/// and `notified.await`, so a signalling task cannot run in that window.
/// Driving state-engine tasks on a genuinely parallel runtime would break this
/// argument; register with `Notified::enable()` before the re-check if that
/// ever changes.
pub struct CompletionLatch {
    status: AtomicU32,
    notify: Notify,
}

impl CompletionLatch {
    pub fn pending() -> Self {
        Self {
            status: AtomicU32::new(PENDING),
            notify: Notify::new(),
        }
    }

    pub fn completed(status: StateStatus) -> Self {
        Self {
            status: AtomicU32::new(status as u32),
            notify: Notify::new(),
        }
    }

    pub fn try_status(&self) -> Option<StateStatus> {
        let raw = self.status.load(Ordering::Acquire);
        (raw != PENDING).then(|| StateStatus::try_from(raw).unwrap_or(StateStatus::InternalError))
    }

    pub fn signal(&self, status: StateStatus) {
        if self
            .status
            .compare_exchange(PENDING, status as u32, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            self.notify.notify_waiters();
        }
    }

    pub async fn wait(&self) -> StateStatus {
        loop {
            if let Some(status) = self.try_status() {
                return status;
            }
            let notified = self.notify.notified();
            if let Some(status) = self.try_status() {
                return status;
            }
            notified.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    #[tokio::test]
    async fn wait_observes_signals_before_and_after_registration() {
        let completed = CompletionLatch::completed(StateStatus::Success);
        assert_eq!(completed.wait().await, StateStatus::Success);

        // On the current-thread test runtime the spawned task is only queued
        // here, so this covers a signal that lands before the waiter registers
        // and is recovered by the status atomic on the waiter's first poll.
        let pending = Arc::new(CompletionLatch::pending());
        let waiter = pending.clone();
        let task = tokio::spawn(async move { waiter.wait().await });
        pending.signal(StateStatus::StateHazard);
        assert_eq!(task.await.unwrap(), StateStatus::StateHazard);
    }

    #[tokio::test]
    async fn wait_wakes_a_waiter_that_is_already_blocked() {
        // The test above never blocks in `notified.await`, so the notification
        // path itself was untested. Yield first so the waiter is polled, reaches
        // `notified.await` and registers, and only then signal it.
        let pending = Arc::new(CompletionLatch::pending());
        let waiter = pending.clone();
        let task = tokio::spawn(async move { waiter.wait().await });
        tokio::task::yield_now().await;
        assert!(
            pending.try_status().is_none(),
            "waiter should still be blocked"
        );
        pending.signal(StateStatus::Success);
        assert_eq!(task.await.unwrap(), StateStatus::Success);
    }

    #[tokio::test]
    async fn only_the_first_signal_is_recorded() {
        let latch = CompletionLatch::pending();
        latch.signal(StateStatus::StateHazard);
        latch.signal(StateStatus::Success);
        assert_eq!(latch.wait().await, StateStatus::StateHazard);
    }
}
