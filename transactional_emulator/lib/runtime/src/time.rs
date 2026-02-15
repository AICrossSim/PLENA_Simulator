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
