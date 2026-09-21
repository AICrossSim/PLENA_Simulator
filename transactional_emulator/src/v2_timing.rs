//! Resource reservations made by executed v2 micro-operations, not an opcode
//! census. Instructions retire after all reservations complete. Functional
//! reads/writes stay ordered; timestamps capture overlap inside the bounded
//! walker. No inter-instruction or DMA overlap is credited.
use std::collections::BTreeSet;

#[derive(Default)]
pub(crate) struct Calendar {
    matrix: BTreeSet<u64>,
    vector: BTreeSet<u64>,
    arithmetic: BTreeSet<u64>,
    context_read_port: BTreeSet<u64>,
    context_write_port: BTreeSet<u64>,
    next_issue: u64,
    pub end: u64,
    pub feedback_wait: u64,
    pub matrix_reads: u64,
    pub matrix_writes: u64,
    pub matrix_words: u64,
    pub vector_reads: u64,
    pub vector_writes: u64,
    pub context_reads: u64,
    pub context_writes: u64,
    pub launches: u64,
}

impl Calendar {
    fn reserve(port: &mut BTreeSet<u64>, earliest: u64, cycles: u64) -> u64 {
        assert!(cycles > 0);
        let mut t = earliest;
        loop {
            if let Some(&busy) = port.range(t..t + cycles).next() {
                t = busy + 1;
            } else {
                port.extend(t..t + cycles);
                return t + cycles;
            }
        }
    }
    pub fn matrix(&mut self, earliest: u64, cycles: u64, words: u64, write: bool) -> u64 {
        self.matrix_reads += u64::from(!write);
        self.matrix_writes += u64::from(write);
        self.matrix_words += words;
        let end = Self::reserve(&mut self.matrix, earliest, cycles.max(1));
        self.end = self.end.max(end);
        end
    }
    pub fn vector(&mut self, earliest: u64, write: bool, cycles: u64) -> u64 {
        self.vector_reads += u64::from(!write);
        self.vector_writes += u64::from(write);
        let end = Self::reserve(&mut self.vector, earliest, cycles);
        self.end = self.end.max(end);
        end
    }
    pub fn compute(
        &mut self,
        operands_ready: u64,
        feedback_ready: u64,
        latency: u32,
        ii: u32,
    ) -> u64 {
        let possible = operands_ready.max(self.next_issue);
        let start = possible.max(feedback_ready);
        self.feedback_wait += start - possible;
        self.next_issue = start + u64::from(ii);
        let end = start + u64::from(latency);
        self.arithmetic.extend(start..end);
        self.end = self.end.max(end);
        self.launches += 1;
        end
    }
    pub fn existing_arithmetic(&mut self, ready: u64, latency: u32) -> u64 {
        let end = ready + u64::from(latency);
        self.arithmetic.extend(ready..end);
        self.end = self.end.max(end);
        end
    }
    pub fn context(&mut self, ready: u64, cycles: u64, write: bool) -> u64 {
        let port = if write {
            self.context_writes += 1;
            &mut self.context_write_port
        } else {
            self.context_reads += 1;
            &mut self.context_read_port
        };
        let end = Self::reserve(port, ready, cycles);
        self.end = self.end.max(end);
        end
    }
    pub async fn retire(self) {
        let mut bank: BTreeSet<_> = self.matrix.union(&self.vector).copied().collect();
        bank.extend(&self.context_read_port);
        bank.extend(&self.context_write_port);
        let exposed_arithmetic = self.arithmetic.difference(&bank).count() as u64;
        let idle = self.end - bank.len() as u64 - exposed_arithmetic;
        crate::timing::record_v2(&self, bank.len() as u64, exposed_arithmetic, idle);
        crate::timing::charge_bank_cycles(bank.len() as u64).await;
        crate::timing::charge_arithmetic_cycles(exposed_arithmetic.try_into().unwrap()).await;
        crate::timing::charge_dependency_cycles(idle.try_into().unwrap()).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn six_cycle_pipeline_accepts_every_two_cycles() {
        let mut c = Calendar::default();
        for _ in 0..8 {
            c.compute(0, 0, 6, 2);
        }
        assert_eq!(c.end, 20);
        assert_eq!(c.launches, 8);
    }
    #[test]
    fn feedback_uses_active_slots_and_ports_cannot_double_book() {
        let mut c = Calendar::default();
        let first = c.compute(0, 0, 6, 2);
        assert_eq!(c.compute(0, first, 6, 2), 12);
        assert_eq!(c.feedback_wait, 4);
        assert_eq!(c.matrix(0, 2, 2, false), 2);
        assert_eq!(c.matrix(0, 2, 2, true), 4);
        assert_eq!(c.matrix(8, 1, 1, false), 9);
        assert_eq!(c.matrix(0, 2, 2, false), 6);
    }
}
