//! Explicit bandwidth/latency/credit model, NOT native DRAM timing.
use super::fabric::{WeightRead, WeightSource};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub bytes_per_cycle: u64,
    pub response_latency_cycles: u64,
    pub outstanding_sectors: usize,
}
#[derive(Default, Serialize)]
pub struct Counts {
    pub submitted_tiles: usize,
    pub completed_tiles: usize,
    pub read_bytes: u64,
    pub accepted_sectors: usize,
    pub completed_sectors: usize,
    pub outstanding_peak: usize,
    pub tile_peak: usize,
    pub credit_stall_cycles: u64,
    pub bus_busy_cycles: u64,
}
pub struct Analytical {
    pub config: Config,
    pub counts: Counts,
    queue: VecDeque<(usize, usize)>,
    remaining: BTreeMap<usize, usize>,
    returns: BTreeMap<u64, Vec<usize>>,
    pending: usize,
    last: Option<u64>,
}
impl Analytical {
    pub fn new(config: Config) -> Result<Self, String> {
        if config.bytes_per_cycle == 0
            || config.bytes_per_cycle > 1 << 20
            || !config.bytes_per_cycle.is_multiple_of(32)
            || config.response_latency_cycles == 0
            || config.outstanding_sectors == 0
        {
            return Err(
                "analytical source needs 32B-aligned bandwidth and positive latency/credits".into(),
            );
        }
        Ok(Self {
            config,
            counts: Counts::default(),
            queue: VecDeque::new(),
            remaining: BTreeMap::new(),
            returns: BTreeMap::new(),
            pending: 0,
            last: None,
        })
    }
    pub fn report(&self) -> serde_json::Value {
        serde_json::json!({"kind":"analytical_bandwidth_latency_credit_source_not_Ramulator","config":self.config,"counts":self.counts,"drained":self.drained()})
    }
}
impl WeightSource for Analytical {
    fn submit(&mut self, r: WeightRead<'_>) -> Result<(), String> {
        // Row-aligned sectors, same request granularity as the native adapter.
        let start = (r.k_start * 2) / 32;
        let end = ((r.k_start + r.valid_k) * 2).div_ceil(32);
        let sectors = r.valid_n * (end - start);
        if self.remaining.insert(r.id, sectors).is_some() {
            return Err("duplicate analytical transfer".into());
        }
        self.queue.push_back((r.id, sectors));
        self.counts.submitted_tiles += 1;
        self.counts.read_bytes += sectors as u64 * 32;
        self.counts.tile_peak = self.counts.tile_peak.max(self.remaining.len());
        Ok(())
    }
    fn advance(&mut self, now: u64) -> Result<Vec<usize>, String> {
        if self.last.is_some_and(|t| now != t + 1) {
            return Err("analytical source requires consecutive core clocks".into());
        }
        self.last = Some(now);
        let mut done = Vec::new();
        for id in self.returns.remove(&now).unwrap_or_default() {
            self.pending -= 1;
            self.counts.completed_sectors += 1;
            let left = self.remaining.get_mut(&id).unwrap();
            *left -= 1;
            if *left == 0 {
                self.remaining.remove(&id);
                done.push(id);
                self.counts.completed_tiles += 1;
            }
        }
        let mut issued = 0;
        for _ in 0..self.config.bytes_per_cycle / 32 {
            if self.pending >= self.config.outstanding_sectors {
                if !self.queue.is_empty() {
                    self.counts.credit_stall_cycles += 1;
                }
                break;
            }
            let Some((id, left)) = self.queue.front_mut() else {
                break;
            };
            self.returns
                .entry(now + self.config.response_latency_cycles)
                .or_default()
                .push(*id);
            *left -= 1;
            if *left == 0 {
                self.queue.pop_front();
            }
            self.pending += 1;
            issued += 1;
            self.counts.accepted_sectors += 1;
            self.counts.outstanding_peak = self.counts.outstanding_peak.max(self.pending);
        }
        self.counts.bus_busy_cycles += u64::from(issued > 0);
        Ok(done)
    }
    fn drained(&self) -> bool {
        self.queue.is_empty()
            && self.remaining.is_empty()
            && self.returns.is_empty()
            && self.pending == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sector_credits_limit_supply_and_completion_is_not_submission() {
        let mut s = Analytical::new(Config {
            bytes_per_cycle: 64,
            response_latency_cycles: 5,
            outstanding_sectors: 2,
        })
        .unwrap();
        let job = crate::moe_spatial::Job {
            expert: 0,
            m: 1,
            n: 2,
            k: 64,
            seed: 1,
        };
        s.submit(WeightRead {
            id: 7,
            job: &job,
            n_start: 0,
            k_start: 0,
            valid_n: 2,
            valid_k: 64,
            now: 0,
        })
        .unwrap();
        assert!(!s.drained());
        let mut done = Vec::new();
        for now in 0..30 {
            let x = s.advance(now).unwrap();
            if now < 20 {
                assert!(x.is_empty());
            }
            done.extend(x);
        }
        assert_eq!(done, vec![7]);
        assert!(s.drained());
        assert_eq!(s.counts.read_bytes, 256);
        assert_eq!(s.counts.outstanding_peak, 2);
        assert_eq!(s.counts.accepted_sectors, s.counts.completed_sectors);
        assert!(s.counts.credit_stall_cycles > 0);
    }
    #[test]
    fn analytical_configuration_and_clock_are_validated() {
        assert!(
            Analytical::new(Config {
                bytes_per_cycle: 31,
                response_latency_cycles: 1,
                outstanding_sectors: 1
            })
            .is_err()
        );
        let mut s = Analytical::new(Config {
            bytes_per_cycle: 32,
            response_latency_cycles: 1,
            outstanding_sectors: 1,
        })
        .unwrap();
        s.advance(0).unwrap();
        assert!(s.advance(2).is_err());
    }
}
