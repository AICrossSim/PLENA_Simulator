//! Matrix compute-resource timing profiles.
//!
//! Stage-2 paper experiments need to model more than one matrix engine.  This
//! module keeps the existing numerical MatrixMachine path intact while making
//! the compute resource explicit: each MatrixMachine owns one MatrixCore, and
//! independent MatrixCores can make progress concurrently on the runtime
//! executor.

use crate::runtime_config::PERIOD;

/// Static geometry/timing metadata for one matrix compute engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MatrixCoreProfile {
    pub(crate) name: &'static str,
    pub(crate) rows: u32,
    pub(crate) cols: u32,
    pub(crate) cycle_multiplier_num: u32,
    pub(crate) cycle_multiplier_den: u32,
}

impl MatrixCoreProfile {
    pub(crate) const fn big_default() -> Self {
        Self {
            name: "big_4x1024",
            rows: 4,
            cols: 1024,
            cycle_multiplier_num: 1,
            cycle_multiplier_den: 1,
        }
    }

    pub(crate) const fn tail_4x256() -> Self {
        Self {
            name: "tail_4x256",
            rows: 4,
            cols: 256,
            cycle_multiplier_num: 1,
            cycle_multiplier_den: 1,
        }
    }

    pub(crate) fn pe_count(self) -> u32 {
        self.rows * self.cols
    }

    pub(crate) fn pe_area_fraction_vs(self, baseline: Self) -> f64 {
        self.pe_count() as f64 / baseline.pe_count() as f64
    }

    fn scale_cycles(self, cycles: u32) -> u32 {
        let num = cycles as u64 * self.cycle_multiplier_num as u64;
        let den = self.cycle_multiplier_den.max(1) as u64;
        num.div_ceil(den) as u32
    }
}

/// One matrix compute resource.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MatrixCore {
    profile: MatrixCoreProfile,
}

impl MatrixCore {
    pub(crate) const fn new(profile: MatrixCoreProfile) -> Self {
        Self { profile }
    }

    pub(crate) const fn profile(&self) -> MatrixCoreProfile {
        self.profile
    }

    pub(crate) async fn compute(&self, cycles: u32) {
        runtime::Executor::current()
            .resolve_at(PERIOD * self.profile.scale_cycles(cycles))
            .await;
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use runtime::{Duration, Executor, Instant};

    use super::*;

    #[tokio::test]
    async fn independent_matrix_cores_overlap_in_sim_time() {
        let executor = Executor::new();
        let observed = Arc::new(Mutex::new(Vec::new()));

        let big = MatrixCore::new(MatrixCoreProfile::big_default());
        let small = MatrixCore::new(MatrixCoreProfile::tail_4x256());

        let observed_big = observed.clone();
        executor.spawn(async move {
            big.compute(100).await;
            observed_big.lock().unwrap().push(Executor::current().now());
        });

        let observed_small = observed.clone();
        executor.spawn(async move {
            small.compute(80).await;
            observed_small
                .lock()
                .unwrap()
                .push(Executor::current().now());
        });

        executor.enter(Instant::ETERNITY).await;

        assert_eq!(executor.now(), Instant::INIT + Duration::from_nanos(100));
        assert_eq!(
            observed.lock().unwrap().as_slice(),
            &[
                Instant::INIT + Duration::from_nanos(80),
                Instant::INIT + Duration::from_nanos(100),
            ]
        );
    }

    #[test]
    fn tail_core_pe_area_fraction_is_explicit() {
        let big = MatrixCoreProfile::big_default();
        let tail = MatrixCoreProfile::tail_4x256();

        assert_eq!(big.pe_count(), 4096);
        assert_eq!(tail.pe_count(), 1024);
        assert_eq!(tail.pe_area_fraction_vs(big), 0.25);
    }
}
