//! Counters shared by L_SCATTER_M ingress and X_STATE consumption.

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ProjectionBufferStats {
    pub values: u64,
    pub fifo_capacity_values: u64,
    pub fifo_peak_values: u64,
    pub fifo_spill_values: u64,
    pub fifo_backpressure_cycles: u64,
    pub write_packets: u64,
    pub write_ideal_cycles: u64,
    pub write_service_cycles: u64,
    pub read_packets: u64,
    pub read_ideal_cycles: u64,
    pub read_service_cycles: u64,
}

impl ProjectionBufferStats {
    pub fn accumulate(&mut self, other: Self) {
        self.values += other.values;
        self.fifo_capacity_values = self.fifo_capacity_values.max(other.fifo_capacity_values);
        self.fifo_peak_values = self.fifo_peak_values.max(other.fifo_peak_values);
        self.fifo_spill_values += other.fifo_spill_values;
        self.fifo_backpressure_cycles += other.fifo_backpressure_cycles;
        self.write_packets += other.write_packets;
        self.write_ideal_cycles += other.write_ideal_cycles;
        self.write_service_cycles += other.write_service_cycles;
        self.read_packets += other.read_packets;
        self.read_ideal_cycles += other.read_ideal_cycles;
        self.read_service_cycles += other.read_service_cycles;
    }
}
