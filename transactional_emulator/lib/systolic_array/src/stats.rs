//! Statistics tracking for systolic array simulation

use serde::{Deserialize, Serialize};
use crate::Dataflow;

/// Statistics collected during systolic array simulation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystolicStats {
    /// Total cycles executed
    pub total_cycles: u64,
    /// Cycles spent loading weights
    pub weight_load_cycles: u64,
    /// Cycles spent in computation
    pub compute_cycles: u64,
    /// Cycles spent draining results
    pub drain_cycles: u64,
    /// Cycles stalled due to buffer constraints
    pub buffer_stall_cycles: u64,
    /// Total MAC operations performed
    pub total_macs: u64,
    /// Number of matrix multiplications completed
    pub matmul_count: u64,
    /// Total elements processed (activations)
    pub activations_processed: u64,
    /// Total weights loaded
    pub weights_loaded: u64,
    /// PE utilization (fraction of PEs active per cycle during compute)
    pub pe_utilization: f64,
    /// Current dataflow being used
    pub dataflow: Option<Dataflow>,
    /// Array dimensions
    pub array_rows: usize,
    pub array_cols: usize,
    /// Buffer configuration
    pub weight_buffer_size: usize,
    pub activation_buffer_size: usize,
    pub output_buffer_size: usize,
}

impl SystolicStats {
    /// Create new statistics tracker
    pub fn new(rows: usize, cols: usize, dataflow: Dataflow) -> Self {
        Self {
            array_rows: rows,
            array_cols: cols,
            dataflow: Some(dataflow),
            ..Default::default()
        }
    }

    /// Create new statistics tracker with buffer configuration
    pub fn with_buffers(
        rows: usize,
        cols: usize,
        dataflow: Dataflow,
        weight_buffer: usize,
        activation_buffer: usize,
        output_buffer: usize,
    ) -> Self {
        Self {
            array_rows: rows,
            array_cols: cols,
            dataflow: Some(dataflow),
            weight_buffer_size: weight_buffer,
            activation_buffer_size: activation_buffer,
            output_buffer_size: output_buffer,
            ..Default::default()
        }
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        let rows = self.array_rows;
        let cols = self.array_cols;
        let dataflow = self.dataflow;
        let wb = self.weight_buffer_size;
        let ab = self.activation_buffer_size;
        let ob = self.output_buffer_size;
        *self = Self::default();
        self.array_rows = rows;
        self.array_cols = cols;
        self.dataflow = dataflow;
        self.weight_buffer_size = wb;
        self.activation_buffer_size = ab;
        self.output_buffer_size = ob;
    }

    /// Record weight loading cycles
    pub fn record_weight_load(&mut self, cycles: u64, weights: u64) {
        self.weight_load_cycles += cycles;
        self.total_cycles += cycles;
        self.weights_loaded += weights;
    }

    /// Record computation cycles and MACs
    pub fn record_compute(&mut self, cycles: u64, macs: u64, active_pes: u64, total_pes: u64) {
        self.compute_cycles += cycles;
        self.total_cycles += cycles;
        self.total_macs += macs;
        
        // Update running average of PE utilization
        if cycles > 0 && total_pes > 0 {
            let cycle_utilization = active_pes as f64 / (cycles * total_pes) as f64;
            let total_compute = self.compute_cycles;
            let prev_compute = total_compute - cycles;
            if total_compute > 0 {
                self.pe_utilization = (self.pe_utilization * prev_compute as f64 
                    + cycle_utilization * cycles as f64) / total_compute as f64;
            }
        }
    }

    /// Record drain cycles
    pub fn record_drain(&mut self, cycles: u64) {
        self.drain_cycles += cycles;
        self.total_cycles += cycles;
    }

    /// Record buffer stall cycles
    pub fn record_buffer_stall(&mut self, cycles: u64) {
        self.buffer_stall_cycles += cycles;
        self.total_cycles += cycles;
    }

    /// Record a completed matrix multiplication
    pub fn record_matmul(&mut self, activations: u64) {
        self.matmul_count += 1;
        self.activations_processed += activations;
    }

    /// Calculate theoretical peak throughput (MACs per cycle)
    pub fn peak_throughput(&self) -> f64 {
        (self.array_rows * self.array_cols) as f64
    }

    /// Calculate actual throughput (MACs per cycle)
    pub fn actual_throughput(&self) -> f64 {
        if self.total_cycles == 0 {
            0.0
        } else {
            self.total_macs as f64 / self.total_cycles as f64
        }
    }

    /// Calculate efficiency (actual / peak throughput)
    pub fn efficiency(&self) -> f64 {
        let peak = self.peak_throughput();
        if peak == 0.0 {
            0.0
        } else {
            self.actual_throughput() / peak
        }
    }

    /// Print a summary of the statistics
    pub fn print_summary(&self) {
        println!("=== Systolic Array Statistics ===");
        if let Some(df) = self.dataflow {
            println!("Dataflow: {} ({})", df.name(), df.abbrev());
        }
        println!("Array Size: {}x{}", self.array_rows, self.array_cols);
        println!("---");
        println!("Total Cycles: {}", self.total_cycles);
        println!("  Weight Load: {} ({:.1}%)", 
            self.weight_load_cycles, 
            100.0 * self.weight_load_cycles as f64 / self.total_cycles.max(1) as f64);
        println!("  Compute: {} ({:.1}%)", 
            self.compute_cycles,
            100.0 * self.compute_cycles as f64 / self.total_cycles.max(1) as f64);
        println!("  Drain: {} ({:.1}%)", 
            self.drain_cycles,
            100.0 * self.drain_cycles as f64 / self.total_cycles.max(1) as f64);
        println!("---");
        println!("Total MACs: {}", self.total_macs);
        println!("Matrix Multiplications: {}", self.matmul_count);
        println!("Weights Loaded: {}", self.weights_loaded);
        println!("Activations Processed: {}", self.activations_processed);
        println!("---");
        println!("Peak Throughput: {:.1} MACs/cycle", self.peak_throughput());
        println!("Actual Throughput: {:.2} MACs/cycle", self.actual_throughput());
        println!("Efficiency: {:.1}%", 100.0 * self.efficiency());
        println!("PE Utilization: {:.1}%", 100.0 * self.pe_utilization);
        println!("================================");
    }
}
