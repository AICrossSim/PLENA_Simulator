use core::ops::{Add, AddAssign, Mul};
use std::ops::Sub;

/// Simulator time duration.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Duration(u64);

impl core::fmt::Debug for Duration {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}.{:03}ns", self.0 / 1000, self.0 % 1000)
    }
}

impl Duration {
    pub const ZERO: Self = Self(0);

    pub const fn from_picos(v: u64) -> Self {
        Self(v)
    }

    pub const fn from_nanos(v: u64) -> Self {
        Self(v * 1000)
    }

    pub const fn from_micros(v: u64) -> Self {
        Self(v * 1_000_000)
    }

    pub const fn from_millis(v: u64) -> Self {
        Self(v * 1_000_000_000)
    }

    pub const fn from_secs(v: u64) -> Self {
        Self(v * 1_000_000_000_000)
    }
}

impl Add<Duration> for Duration {
    type Output = Duration;

    fn add(self, rhs: Duration) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Mul<u32> for Duration {
    type Output = Duration;

    fn mul(self, rhs: u32) -> Self::Output {
        Self(self.0 * rhs as u64)
    }
}

impl Mul<u64> for Duration {
    type Output = Duration;

    fn mul(self, rhs: u64) -> Self::Output {
        Self(self.0 * rhs as u64)
    }
}

/// Simulator time instant.
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub struct Instant(u64);

impl core::fmt::Debug for Instant {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if *self == Self::ETERNITY {
            write!(f, "eternity")
        } else {
            Duration(self.0).fmt(f)
        }
    }
}

impl Add<Duration> for Instant {
    type Output = Instant;

    fn add(self, rhs: Duration) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, rhs: Duration) {
        self.0 += rhs.0;
    }
}

impl Sub<Instant> for Instant {
    type Output = Duration;

    fn sub(self, rhs: Instant) -> Self::Output {
        assert!(self.0 >= rhs.0);
        Duration(self.0 - rhs.0)
    }
}

impl Instant {
    /// Very first instant of the simulated system.
    ///
    /// Components that need kick-starting should use this instant to perform initialization.
    pub const INIT: Self = Self(0);

    /// The very last instant that can be tracked by the system.
    ///
    /// Effectively means "never".
    pub const ETERNITY: Self = Self(u64::MAX);

    pub const fn to_secs(&self) -> f64 {
        self.0 as f64 / (1_000_000_000_000_u64 as f64)
    }
}

pub trait Deadline {
    fn to_instant(self, now: Instant) -> Instant;
}

impl Deadline for Duration {
    fn to_instant(self, now: Instant) -> Instant {
        Instant(now.0 + self.0)
    }
}

impl Deadline for Instant {
    fn to_instant(self, _now: Instant) -> Instant {
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_duration_unit_constructors_agree() {
        assert_eq!(Duration::from_nanos(1), Duration::from_picos(1000));
        assert_eq!(Duration::from_micros(1), Duration::from_picos(1_000_000));
        assert_eq!(
            Duration::from_millis(1),
            Duration::from_picos(1_000_000_000)
        );
        assert_eq!(
            Duration::from_secs(1),
            Duration::from_picos(1_000_000_000_000)
        );
    }

    #[test]
    fn test_duration_arithmetic() {
        assert_eq!(
            Duration::from_nanos(1) + Duration::from_nanos(2),
            Duration::from_nanos(3)
        );
        assert_eq!(Duration::from_nanos(2) * 3u32, Duration::from_nanos(6));
        assert_eq!(Duration::from_nanos(2) * 3u64, Duration::from_nanos(6));
    }

    #[test]
    fn test_duration_debug_is_nanos_with_pico_fraction() {
        assert_eq!(format!("{:?}", Duration::from_picos(1500)), "1.500ns");
        assert_eq!(format!("{:?}", Duration::from_nanos(1)), "1.000ns");
        assert_eq!(format!("{:?}", Duration::ZERO), "0.000ns");
    }

    #[test]
    fn test_instant_add_sub_roundtrip() {
        let t = Instant::INIT + Duration::from_nanos(5);
        assert_eq!(t - Instant::INIT, Duration::from_nanos(5));

        let mut u = Instant::INIT;
        u += Duration::from_nanos(3);
        assert_eq!(u, Instant::INIT + Duration::from_nanos(3));
    }

    #[test]
    #[should_panic]
    fn test_instant_sub_underflow_panics() {
        // Subtracting a later instant from an earlier one must panic.
        let _ = Instant::INIT - (Instant::INIT + Duration::from_nanos(1));
    }

    #[test]
    fn test_instant_debug_and_eternity_ordering() {
        assert_eq!(format!("{:?}", Instant::ETERNITY), "eternity");
        assert_eq!(format!("{:?}", Instant::INIT), "0.000ns");
        assert_eq!(
            format!("{:?}", Instant::INIT + Duration::from_nanos(2)),
            "2.000ns"
        );
        assert!(Instant::INIT < Instant::ETERNITY);
    }

    #[test]
    fn test_instant_to_secs() {
        assert_eq!((Instant::INIT + Duration::from_secs(1)).to_secs(), 1.0);
    }

    #[test]
    fn test_deadline_relative_vs_absolute() {
        let now = Instant::INIT + Duration::from_nanos(100);
        // A `Duration` deadline is relative to `now`.
        assert_eq!(
            Duration::from_nanos(5).to_instant(now),
            now + Duration::from_nanos(5)
        );
        // An `Instant` deadline is absolute and ignores `now`.
        let fixed = Instant::INIT + Duration::from_nanos(42);
        assert_eq!(fixed.to_instant(now), fixed);
    }
}
