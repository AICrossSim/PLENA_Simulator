//! Systolic Array Implementation
//!
//! This module implements a cycle-accurate systolic array simulator
//! supporting multiple dataflow strategies.

use crate::{Dataflow, ProcessingElement, SystolicStats};
use tch::Tensor;

/// Buffer state for tracking occupancy and stalls
#[derive(Debug, Clone, Default)]
pub struct BufferState {
    /// Current occupancy (number of tiles/elements)
    pub occupancy: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Cycles spent waiting for data
    pub stall_cycles: u64,
    /// Bytes transferred
    pub bytes_transferred: u64,
}

impl BufferState {
    /// Create a new buffer state with given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            occupancy: 0,
            capacity,
            stall_cycles: 0,
            bytes_transferred: 0,
        }
    }

    /// Reset buffer state (clear occupancy and stats)
    pub fn reset(&mut self) {
        self.occupancy = 0;
        self.stall_cycles = 0;
        self.bytes_transferred = 0;
    }

    /// Check if buffer has space for more elements
    pub fn has_space(&self, count: usize) -> bool {
        self.occupancy + count <= self.capacity
    }

    /// Check if buffer has enough elements
    pub fn has_elements(&self, count: usize) -> bool {
        self.occupancy >= count
    }

    /// Add elements to buffer
    pub fn fill(&mut self, count: usize) {
        self.occupancy = (self.occupancy + count).min(self.capacity);
    }

    /// Remove elements from buffer
    pub fn drain(&mut self, count: usize) {
        self.occupancy = self.occupancy.saturating_sub(count);
    }
}

/// Memory bandwidth configuration
#[derive(Debug, Clone)]
pub struct BandwidthConfig {
    /// Weight memory bandwidth (bytes per cycle)
    pub weight_bw_bytes_per_cycle: f64,
    /// Activation memory bandwidth (bytes per cycle)
    pub activation_bw_bytes_per_cycle: f64,
    /// Output memory bandwidth (bytes per cycle)
    pub output_bw_bytes_per_cycle: f64,
    /// Bytes per element (e.g., 2 for FP16, 4 for FP32)
    pub bytes_per_element: usize,
}

impl Default for BandwidthConfig {
    fn default() -> Self {
        Self {
            // Default: 64 bytes/cycle (512 bits @ 1GHz)
            weight_bw_bytes_per_cycle: 64.0,
            activation_bw_bytes_per_cycle: 64.0,
            output_bw_bytes_per_cycle: 64.0,
            bytes_per_element: 2, // FP16
        }
    }
}

impl BandwidthConfig {
    /// Calculate cycles needed to transfer a given number of elements for weights
    pub fn weight_transfer_cycles(&self, elements: usize) -> u64 {
        let bytes = elements * self.bytes_per_element;
        (bytes as f64 / self.weight_bw_bytes_per_cycle).ceil() as u64
    }

    /// Calculate cycles needed to transfer a given number of elements for activations
    pub fn activation_transfer_cycles(&self, elements: usize) -> u64 {
        let bytes = elements * self.bytes_per_element;
        (bytes as f64 / self.activation_bw_bytes_per_cycle).ceil() as u64
    }

    /// Calculate cycles needed to transfer a given number of elements for outputs
    pub fn output_transfer_cycles(&self, elements: usize) -> u64 {
        let bytes = elements * self.bytes_per_element;
        (bytes as f64 / self.output_bw_bytes_per_cycle).ceil() as u64
    }
}

/// Configuration for the systolic array
#[derive(Debug, Clone)]
pub struct SystolicConfig {
    /// Number of rows in the array
    pub rows: usize,
    /// Number of columns in the array
    pub cols: usize,
    /// Dataflow strategy
    pub dataflow: Dataflow,
    /// Whether to enable detailed tracing
    pub trace_enabled: bool,
    /// Weight buffer size in multiples of array columns (from Matrix SRAM)
    /// A value of 2 enables double-buffering
    pub weight_buffer_size: usize,
    /// Activation buffer size in multiples of array rows (from Vector SRAM)
    /// A value of 2 enables double-buffering
    pub activation_buffer_size: usize,
    /// Output buffer size in multiples of array columns
    /// A value of 2 enables double-buffering
    pub output_buffer_size: usize,
    /// Bandwidth configuration for memory transfers
    pub bandwidth: Option<BandwidthConfig>,
}

impl Default for SystolicConfig {
    fn default() -> Self {
        Self {
            rows: 16,
            cols: 16,
            dataflow: Dataflow::WeightStationary,
            trace_enabled: false,
            weight_buffer_size: 2,
            activation_buffer_size: 2,
            output_buffer_size: 2,
            bandwidth: Some(BandwidthConfig::default()),
        }
    }
}

/// Systolic Array Simulator
///
/// Provides cycle-accurate simulation of matrix operations
/// with configurable dataflow strategies.
pub struct SystolicArray {
    /// Configuration
    config: SystolicConfig,
    /// 2D array of processing elements
    pes: Vec<Vec<ProcessingElement>>,
    /// Current simulation cycle
    cycle: u64,
    /// Statistics tracker
    stats: SystolicStats,
    /// Input activation buffers (for each row)
    activation_buffers: Vec<Vec<f32>>,
    /// Input weight buffers (for each column in OS/IS mode)
    weight_buffers: Vec<Vec<f32>>,
    /// Output buffers (partial sums or final results)
    output_buffers: Vec<Vec<f32>>,
    /// Weight buffer state tracking
    weight_buffer_state: BufferState,
    /// Activation buffer state tracking
    activation_buffer_state: BufferState,
    /// Output buffer state tracking
    output_buffer_state: BufferState,
}

impl SystolicArray {
    /// Create a new systolic array with the given configuration
    pub fn new(config: SystolicConfig) -> Self {
        let rows = config.rows;
        let cols = config.cols;
        let dataflow = config.dataflow;

        // Initialize PEs
        let pes: Vec<Vec<ProcessingElement>> = (0..rows)
            .map(|r| (0..cols).map(|c| ProcessingElement::new(r, c)).collect())
            .collect();

        // Calculate buffer capacities based on array size and buffer size multiplier
        // Weight buffer: stores weight tiles (each tile is rows * cols elements)
        let weight_buffer_capacity = config.weight_buffer_size * rows * cols;
        // Activation buffer: stores activation tiles
        let activation_buffer_capacity = config.activation_buffer_size * rows * cols;
        // Output buffer: stores output tiles
        let output_buffer_capacity = config.output_buffer_size * rows * cols;

        Self {
            config,
            pes,
            cycle: 0,
            stats: SystolicStats::new(rows, cols, dataflow),
            activation_buffers: vec![Vec::new(); rows],
            weight_buffers: vec![Vec::new(); cols],
            output_buffers: vec![Vec::new(); rows],
            weight_buffer_state: BufferState::new(weight_buffer_capacity),
            activation_buffer_state: BufferState::new(activation_buffer_capacity),
            output_buffer_state: BufferState::new(output_buffer_capacity),
        }
    }

    /// Create a systolic array with default configuration
    pub fn with_size(rows: usize, cols: usize, dataflow: Dataflow) -> Self {
        Self::new(SystolicConfig {
            rows,
            cols,
            dataflow,
            trace_enabled: false,
            weight_buffer_size: 2,
            activation_buffer_size: 2,
            output_buffer_size: 2,
            bandwidth: Some(BandwidthConfig::default()),
        })
    }

    /// Get the current configuration
    pub fn config(&self) -> &SystolicConfig {
        &self.config
    }

    /// Get the current dataflow
    pub fn dataflow(&self) -> Dataflow {
        self.config.dataflow
    }

    /// Set the dataflow strategy
    pub fn set_dataflow(&mut self, dataflow: Dataflow) {
        self.config.dataflow = dataflow;
        self.stats.dataflow = Some(dataflow);
    }

    /// Get the statistics
    pub fn stats(&self) -> &SystolicStats {
        &self.stats
    }

    /// Get mutable statistics
    pub fn stats_mut(&mut self) -> &mut SystolicStats {
        &mut self.stats
    }

    /// Reset the array state (but keep statistics)
    pub fn reset(&mut self) {
        for row in &mut self.pes {
            for pe in row {
                pe.reset();
            }
        }
        self.cycle = 0;
        for buf in &mut self.activation_buffers {
            buf.clear();
        }
        for buf in &mut self.weight_buffers {
            buf.clear();
        }
        for buf in &mut self.output_buffers {
            buf.clear();
        }
        // Reset buffer state tracking (keeps capacity)
        self.weight_buffer_state.reset();
        self.activation_buffer_state.reset();
        self.output_buffer_state.reset();
    }

    /// Reset everything including statistics
    pub fn reset_all(&mut self) {
        self.reset();
        self.stats.reset();
    }

    /// Get the current cycle count
    pub fn current_cycle(&self) -> u64 {
        self.cycle
    }

    /// Get the weight buffer state
    pub fn weight_buffer_state(&self) -> &BufferState {
        &self.weight_buffer_state
    }

    /// Get the activation buffer state
    pub fn activation_buffer_state(&self) -> &BufferState {
        &self.activation_buffer_state
    }

    /// Get the output buffer state
    pub fn output_buffer_state(&self) -> &BufferState {
        &self.output_buffer_state
    }

    /// Get bytes per element from bandwidth config (default to 2 for FP16)
    fn bytes_per_element(&self) -> usize {
        self.config
            .bandwidth
            .as_ref()
            .map(|b| b.bytes_per_element)
            .unwrap_or(2)
    }

    /// Calculate weight transfer cycles based on bandwidth config
    fn weight_transfer_cycles(&self, elements: usize) -> u64 {
        match &self.config.bandwidth {
            Some(bw) => bw.weight_transfer_cycles(elements),
            None => 0, // No bandwidth constraints
        }
    }

    /// Calculate activation transfer cycles based on bandwidth config
    fn activation_transfer_cycles(&self, elements: usize) -> u64 {
        match &self.config.bandwidth {
            Some(bw) => bw.activation_transfer_cycles(elements),
            None => 0, // No bandwidth constraints
        }
    }

    /// Calculate output transfer cycles based on bandwidth config
    fn output_transfer_cycles(&self, elements: usize) -> u64 {
        match &self.config.bandwidth {
            Some(bw) => bw.output_transfer_cycles(elements),
            None => 0, // No bandwidth constraints
        }
    }

    /// Check and record stall for weight buffer
    /// Returns the number of stall cycles needed to fill the buffer
    fn check_weight_buffer_stall(&mut self, needed: usize) -> u64 {
        if self.weight_buffer_state.has_elements(needed) {
            return 0;
        }

        let elements_needed = needed.saturating_sub(self.weight_buffer_state.occupancy);
        let fill_cycles = self.weight_transfer_cycles(elements_needed);

        if fill_cycles > 0 {
            self.weight_buffer_state.stall_cycles += fill_cycles;
            self.stats.record_weight_buffer_stall(fill_cycles);
            let bytes = elements_needed * self.bytes_per_element();
            self.weight_buffer_state.bytes_transferred += bytes as u64;
            self.stats.record_weight_bytes_transferred(bytes as u64);
        }

        // Fill the buffer after stall
        self.weight_buffer_state.fill(elements_needed);
        fill_cycles
    }

    /// Check and record stall for activation buffer
    /// Returns the number of stall cycles needed to fill the buffer
    fn check_activation_buffer_stall(&mut self, needed: usize) -> u64 {
        if self.activation_buffer_state.has_elements(needed) {
            return 0;
        }

        let elements_needed = needed.saturating_sub(self.activation_buffer_state.occupancy);
        let fill_cycles = self.activation_transfer_cycles(elements_needed);

        if fill_cycles > 0 {
            self.activation_buffer_state.stall_cycles += fill_cycles;
            self.stats.record_activation_buffer_stall(fill_cycles);
            let bytes = elements_needed * self.bytes_per_element();
            self.activation_buffer_state.bytes_transferred += bytes as u64;
            self.stats.record_activation_bytes_transferred(bytes as u64);
        }

        // Fill the buffer after stall
        self.activation_buffer_state.fill(elements_needed);
        fill_cycles
    }

    /// Check and record stall for output buffer (when draining)
    /// Returns the number of stall cycles needed to drain the buffer
    fn check_output_buffer_stall(&mut self, needed_space: usize) -> u64 {
        if self.output_buffer_state.has_space(needed_space) {
            return 0;
        }

        // Need to drain some elements to make space
        let elements_to_drain = self.output_buffer_state.occupancy + needed_space
            - self.output_buffer_state.capacity;
        let drain_cycles = self.output_transfer_cycles(elements_to_drain);

        if drain_cycles > 0 {
            self.output_buffer_state.stall_cycles += drain_cycles;
            self.stats.record_output_buffer_stall(drain_cycles);
            let bytes = elements_to_drain * self.bytes_per_element();
            self.output_buffer_state.bytes_transferred += bytes as u64;
            self.stats.record_output_bytes_transferred(bytes as u64);
        }

        // Drain the buffer after stall
        self.output_buffer_state.drain(elements_to_drain);
        drain_cycles
    }

    /// Load weights into the array (for Weight Stationary dataflow)
    /// 
    /// Weight matrix shape: [K, N] where K is reduction dim, N is output dim
    /// The array processes [array_rows, array_cols] tiles
    pub fn load_weights_ws(&mut self, weights: &Tensor) -> u64 {
        let k = weights.size()[0] as usize;
        let n = weights.size()[1] as usize;
        
        let rows = self.config.rows.min(k);
        let cols = self.config.cols.min(n);
        
        let weight_data: Vec<f32> = weights.flatten(0, -1).try_into().unwrap();
        
        let mut cycles = 0u64;
        
        // Load weights column by column with pipeline delay
        for c in 0..cols {
            for r in 0..rows {
                let idx = r * n + c;
                if idx < weight_data.len() {
                    self.pes[r][c].load_weight(weight_data[idx]);
                }
            }
            cycles += 1; // One cycle per column load (pipelined)
        }
        
        // Add pipeline fill time
        cycles += (rows - 1) as u64;
        
        self.cycle += cycles;
        self.stats.record_weight_load(cycles, (rows * cols) as u64);
        
        cycles
    }

    /// Load activations into the array (for Activation Stationary dataflow)
    pub fn load_activations_is(&mut self, activations: &Tensor) -> u64 {
        let m = activations.size()[0] as usize;
        let k = activations.size()[1] as usize;
        
        let rows = self.config.rows.min(m);
        let cols = self.config.cols.min(k);
        
        let act_data: Vec<f32> = activations.flatten(0, -1).try_into().unwrap();
        
        let mut cycles = 0u64;
        
        // Load activations - each row of the array gets a row of activations
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * k + c;
                if idx < act_data.len() {
                    self.pes[r][c].load_activation(act_data[idx]);
                    self.pes[r][c].activation_valid = true;
                }
            }
            cycles += 1;
        }
        
        self.cycle += cycles;
        cycles
    }

    /// Perform matrix multiplication: C = A @ B
    /// 
    /// For Weight Stationary: B is preloaded, A streams through
    /// For Output Stationary: Both A and B stream through
    /// For Activation Stationary: A is preloaded, B streams through
    ///
    /// Returns the result tensor and the number of cycles taken
    pub fn matmul(&mut self, a: &Tensor, b: &Tensor) -> (Tensor, u64) {
        match self.config.dataflow {
            Dataflow::WeightStationary => self.matmul_weight_stationary(a, b),
            Dataflow::OutputStationary => self.matmul_output_stationary(a, b),
            Dataflow::ActivationStationary => self.matmul_activation_stationary(a, b),
        }
    }

    /// Weight Stationary matrix multiplication
    ///
    /// In WS dataflow:
    /// - Weights are preloaded into PEs and remain stationary
    /// - Activations stream through horizontally
    /// - Buffer stalls occur if weight buffer can't fill fast enough
    fn matmul_weight_stationary(&mut self, a: &Tensor, b: &Tensor) -> (Tensor, u64) {
        let m = a.size()[0] as usize;
        let k = a.size()[1] as usize;
        let n = b.size()[1] as usize;

        let start_cycle = self.cycle;
        let rows = self.config.rows;
        let cols = self.config.cols;

        // Reset accumulators and buffer states
        for row in &mut self.pes {
            for pe in row {
                pe.clear_accumulator();
            }
        }
        self.weight_buffer_state.reset();
        self.activation_buffer_state.reset();
        self.output_buffer_state.reset();

        // Phase 1: Weight preload with bandwidth constraints
        // Need to load K*N weights into the array
        let weight_elements = rows.min(k) * cols.min(n);

        // Check if weight buffer can hold the weights, stall if needed
        let weight_stall = self.check_weight_buffer_stall(weight_elements);
        self.cycle += weight_stall;

        // Load weights (B matrix) into PEs
        let _load_cycles = self.load_weights_ws(b);

        // Convert tensors to vectors
        let a_data: Vec<f32> = a.flatten(0, -1).try_into().unwrap();

        // Phase 2: Activation streaming with bandwidth constraints
        // Elements needed per cycle for streaming
        let act_elements_per_cycle = rows.min(m);

        // Check if activation buffer supports streaming (double buffering)
        let can_double_buffer = self.config.activation_buffer_size >= 2;

        // For each K iteration (reduction dimension)
        for k_idx in 0..k {
            // Check activation buffer availability
            if !can_double_buffer {
                // Single buffer: may need to stall for activation fill
                let act_stall = self.check_activation_buffer_stall(act_elements_per_cycle);
                self.cycle += act_stall;
            }

            // Feed activations with proper skew
            // Row i receives its activation i cycles after row 0
            let mut active_pes = 0u64;

            for r in 0..rows.min(m) {
                // Calculate skewed timing
                if k_idx >= r {
                    let actual_k = k_idx - r;
                    if actual_k < k {
                        let act_val = a_data[r * k + actual_k];

                        // Process through the row
                        let mut current_act = Some(act_val);
                        for c in 0..cols.min(n) {
                            current_act = self.pes[r][c].step_weight_stationary(current_act);
                            if current_act.is_some() {
                                active_pes += 1;
                            }
                        }
                    }
                }
            }

            // Consume activations from buffer
            self.activation_buffer_state.drain(act_elements_per_cycle.min(self.activation_buffer_state.occupancy));

            self.cycle += 1;
            self.stats.record_compute(1, active_pes, active_pes, (rows * cols) as u64);
        }

        // Phase 3: Drain phase - let pipeline flush
        let drain_cycles = (rows + cols - 2) as u64;
        for _ in 0..drain_cycles {
            for r in 0..rows {
                let mut current_act = None;
                for c in 0..cols {
                    current_act = self.pes[r][c].step_weight_stationary(current_act);
                }
            }
            self.cycle += 1;
        }
        self.stats.record_drain(drain_cycles);

        // Phase 4: Output collection with buffer constraints
        let output_elements = rows.min(m) * cols.min(n);

        // Check output buffer space for results
        let output_stall = self.check_output_buffer_stall(output_elements);
        self.cycle += output_stall;

        // Fill output buffer with results
        self.output_buffer_state.fill(output_elements);

        // Collect results
        let mut result = vec![0.0f32; m * n];
        for r in 0..rows.min(m) {
            for c in 0..cols.min(n) {
                result[r * n + c] = self.pes[r][c].get_result();
            }
        }

        let result_tensor = Tensor::from_slice(&result).reshape([m as i64, n as i64]);

        // Record bytes transferred for bandwidth utilization
        let bytes_per_elem = self.bytes_per_element();
        self.stats.record_weight_bytes_transferred((weight_elements * bytes_per_elem) as u64);
        self.stats.record_activation_bytes_transferred((m * k * bytes_per_elem) as u64);
        self.stats.record_output_bytes_transferred((output_elements * bytes_per_elem) as u64);

        self.stats.record_matmul((m * k) as u64);

        (result_tensor, self.cycle - start_cycle)
    }

    /// Output Stationary matrix multiplication
    ///
    /// In OS dataflow:
    /// - Weights flow top to bottom (column by column)
    /// - Activations flow left to right (row by row)
    /// - Partial sums accumulate in place (outputs stay in PEs)
    /// - Both weight and activation buffers must stream simultaneously
    fn matmul_output_stationary(&mut self, a: &Tensor, b: &Tensor) -> (Tensor, u64) {
        let m = a.size()[0] as usize;
        let k = a.size()[1] as usize;
        let n = b.size()[1] as usize;

        let start_cycle = self.cycle;
        let rows = self.config.rows;
        let cols = self.config.cols;

        // Reset accumulators, pipeline state, and buffer states
        for row in &mut self.pes {
            for pe in row {
                pe.clear_accumulator();
                pe.weight_valid = false;
                pe.activation_valid = false;
            }
        }
        self.weight_buffer_state.reset();
        self.activation_buffer_state.reset();
        self.output_buffer_state.reset();

        let a_data: Vec<f32> = a.flatten(0, -1).try_into().unwrap();
        let b_data: Vec<f32> = b.flatten(0, -1).try_into().unwrap();

        // Check double buffering capability
        let weight_can_double_buffer = self.config.weight_buffer_size >= 2;
        let act_can_double_buffer = self.config.activation_buffer_size >= 2;

        // Elements needed per cycle for streaming
        let weight_elements_per_cycle = cols.min(n); // One weight per column
        let act_elements_per_cycle = rows.min(m); // One activation per row

        // Total cycles: k for the reduction dimension + pipeline fill/drain
        let total_compute_cycles = k + rows.min(m) + cols.min(n) - 2;

        for cycle in 0..total_compute_cycles {
            // Check buffer availability for this cycle
            if !weight_can_double_buffer && cycle < k {
                let weight_stall = self.check_weight_buffer_stall(weight_elements_per_cycle);
                self.cycle += weight_stall;
            }
            if !act_can_double_buffer && cycle < k {
                let act_stall = self.check_activation_buffer_stall(act_elements_per_cycle);
                self.cycle += act_stall;
            }

            let mut active_pes = 0u64;

            // Temporary storage for outputs to pass to neighbors
            let mut weight_outputs: Vec<Vec<Option<f32>>> =
                vec![vec![None; cols]; rows];
            let mut act_outputs: Vec<Vec<Option<f32>>> =
                vec![vec![None; cols]; rows];

            for r in 0..rows.min(m) {
                for c in 0..cols.min(n) {
                    // Determine inputs based on position and cycle
                    let weight_in = if r == 0 {
                        // First row gets weights from external input
                        // PE[0,c] receives B[k_idx, c] where k_idx = cycle - c (skewed)
                        let k_idx = cycle as i64 - c as i64;
                        if k_idx >= 0 && (k_idx as usize) < k {
                            Some(b_data[k_idx as usize * n + c])
                        } else {
                            None
                        }
                    } else {
                        // Interior rows use pipeline register (set by previous cycle)
                        if self.pes[r][c].weight_valid {
                            Some(self.pes[r][c].weight)
                        } else {
                            None
                        }
                    };

                    let act_in = if c == 0 {
                        // First column gets activations from external input
                        // PE[r,0] receives A[r, k_idx] where k_idx = cycle - r (skewed)
                        let k_idx = cycle as i64 - r as i64;
                        if k_idx >= 0 && (k_idx as usize) < k {
                            Some(a_data[r * k + k_idx as usize])
                        } else {
                            None
                        }
                    } else {
                        // Interior columns use pipeline register (set by previous cycle)
                        if self.pes[r][c].activation_valid {
                            Some(self.pes[r][c].activation)
                        } else {
                            None
                        }
                    };

                    let (w_out, a_out) = self.pes[r][c].step_output_stationary(weight_in, act_in);

                    weight_outputs[r][c] = w_out;
                    act_outputs[r][c] = a_out;

                    if w_out.is_some() && a_out.is_some() {
                        active_pes += 1;
                    }
                }
            }

            // Propagate outputs to next cycle's pipeline registers
            for r in 0..rows.min(m) {
                for c in 0..cols.min(n) {
                    // Pass weight down to next row
                    if r + 1 < rows.min(m) {
                        if let Some(w) = weight_outputs[r][c] {
                            self.pes[r + 1][c].weight = w;
                            self.pes[r + 1][c].weight_valid = true;
                        }
                    }
                    // Pass activation right to next column
                    if c + 1 < cols.min(n) {
                        if let Some(a) = act_outputs[r][c] {
                            self.pes[r][c + 1].activation = a;
                            self.pes[r][c + 1].activation_valid = true;
                        }
                    }
                }
            }

            // Consume from buffers
            if cycle < k {
                self.weight_buffer_state.drain(weight_elements_per_cycle.min(self.weight_buffer_state.occupancy));
                self.activation_buffer_state.drain(act_elements_per_cycle.min(self.activation_buffer_state.occupancy));
            }

            self.cycle += 1;
            self.stats.record_compute(1, active_pes, active_pes, (rows * cols) as u64);
        }

        // Output drain phase with buffer constraints
        let output_elements = rows.min(m) * cols.min(n);
        let output_stall = self.check_output_buffer_stall(output_elements);
        self.cycle += output_stall;
        self.output_buffer_state.fill(output_elements);

        // Collect results from accumulators
        let mut result = vec![0.0f32; m * n];
        for r in 0..rows.min(m) {
            for c in 0..cols.min(n) {
                result[r * n + c] = self.pes[r][c].get_result();
            }
        }

        let result_tensor = Tensor::from_slice(&result).reshape([m as i64, n as i64]);

        // Record bytes transferred
        let bytes_per_elem = self.bytes_per_element();
        self.stats.record_weight_bytes_transferred((k * n * bytes_per_elem) as u64);
        self.stats.record_activation_bytes_transferred((m * k * bytes_per_elem) as u64);
        self.stats.record_output_bytes_transferred((output_elements * bytes_per_elem) as u64);

        self.stats.record_matmul((m * k) as u64);

        (result_tensor, self.cycle - start_cycle)
    }

    /// Activation Stationary matrix multiplication
    ///
    /// In IS dataflow:
    /// - Activations are preloaded into PEs and remain stationary
    /// - Weights stream through horizontally
    /// - Partial sums flow top to bottom
    /// - Buffer stalls occur if activation buffer can't fill fast enough
    fn matmul_activation_stationary(&mut self, a: &Tensor, b: &Tensor) -> (Tensor, u64) {
        let m = a.size()[0] as usize;
        let k = a.size()[1] as usize;
        let n = b.size()[1] as usize;

        let start_cycle = self.cycle;
        let rows = self.config.rows;
        let cols = self.config.cols;

        // Reset accumulators and buffer states
        for row in &mut self.pes {
            for pe in row {
                pe.clear_accumulator();
            }
        }
        self.weight_buffer_state.reset();
        self.activation_buffer_state.reset();
        self.output_buffer_state.reset();

        // Phase 1: Activation preload with bandwidth constraints
        // Need to load M*K activations into the array
        let activation_elements = rows.min(m) * cols.min(k);

        // Check if activation buffer can hold the activations, stall if needed
        let act_stall = self.check_activation_buffer_stall(activation_elements);
        self.cycle += act_stall;

        // Load activations into PEs (A matrix)
        let _load_cycles = self.load_activations_is(a);

        let b_data: Vec<f32> = b.flatten(0, -1).try_into().unwrap();

        // Check double buffering capability for weights
        let weight_can_double_buffer = self.config.weight_buffer_size >= 2;

        // Elements needed per cycle for weight streaming
        let weight_elements_per_cycle = rows.min(m); // One weight per row for skewed input

        // Collect results for each output column
        let mut result = vec![0.0f32; m * n];

        for n_idx in 0..n {
            // Reset partial sums and pipeline state for this output column
            for row in &mut self.pes {
                for pe in row {
                    pe.psum_out = 0.0;
                    pe.psum_out_valid = false;
                    pe.weight_valid = false;
                }
            }

            // Process K dimension with weights flowing through
            // Need k cycles for weights + (rows-1) for pipeline drain
            let compute_cycles = k + rows - 1;

            // Track previous cycle's psum outputs for proper pipeline propagation
            let mut prev_psum_outputs: Vec<Vec<Option<f32>>> =
                vec![vec![None; cols.min(k)]; rows.min(m)];

            for cycle in 0..compute_cycles {
                // Check weight buffer availability
                if !weight_can_double_buffer && cycle < k {
                    let weight_stall = self.check_weight_buffer_stall(weight_elements_per_cycle);
                    self.cycle += weight_stall;
                }

                let mut active_pes = 0u64;

                // Temporary storage for this cycle's outputs
                let mut weight_outputs: Vec<Vec<Option<f32>>> =
                    vec![vec![None; cols.min(k)]; rows.min(m)];
                let mut psum_outputs: Vec<Vec<Option<f32>>> =
                    vec![vec![None; cols.min(k)]; rows.min(m)];

                for r in 0..rows.min(m) {
                    for c in 0..cols.min(k) {
                        // Weight input for first column comes from external input
                        let weight_in = if c == 0 {
                            let k_idx = cycle as i64 - r as i64;
                            if k_idx >= 0 && (k_idx as usize) < k {
                                Some(b_data[k_idx as usize * n + n_idx])
                            } else {
                                None
                            }
                        } else {
                            // Interior columns use pipeline register (set by previous cycle's propagation)
                            if self.pes[r][c].weight_valid {
                                Some(self.pes[r][c].weight)
                            } else {
                                None
                            }
                        };

                        // Partial sum input: row 0 starts fresh, others get from row above
                        let psum_in = if r == 0 {
                            Some(0.0)
                        } else {
                            // Get psum from PE above from previous cycle
                            prev_psum_outputs[r - 1][c]
                        };

                        let (w_out, p_out) = self.pes[r][c].step_activation_stationary(weight_in, psum_in);

                        weight_outputs[r][c] = w_out;
                        psum_outputs[r][c] = p_out;

                        if w_out.is_some() || p_out.is_some() {
                            active_pes += 1;
                        }
                    }
                }

                // Propagate weight outputs to next column's pipeline registers
                for r in 0..rows.min(m) {
                    for c in 0..cols.min(k) {
                        if c + 1 < cols.min(k) {
                            if let Some(w) = weight_outputs[r][c] {
                                self.pes[r][c + 1].weight = w;
                                self.pes[r][c + 1].weight_valid = true;
                            }
                        }
                    }
                }

                // Collect bottom row outputs (completed partial sums for this output column)
                let bottom_row = rows.min(m) - 1;
                for c in 0..cols.min(k) {
                    if let Some(psum) = psum_outputs[bottom_row][c] {
                        // Output row index based on skewed timing
                        let m_idx = cycle as i64 - c as i64 - (rows - 1) as i64;
                        if m_idx >= 0 && (m_idx as usize) < m {
                            result[m_idx as usize * n + n_idx] = psum;
                        }
                    }
                }

                // Consume weights from buffer
                if cycle < k {
                    self.weight_buffer_state.drain(weight_elements_per_cycle.min(self.weight_buffer_state.occupancy));
                }

                // Save this cycle's psum outputs for next cycle's input
                prev_psum_outputs = psum_outputs;

                self.cycle += 1;
                self.stats.record_compute(1, active_pes, active_pes, (rows * cols) as u64);
            }

            // Check output buffer space for this column's results
            let output_elements_per_col = rows.min(m);
            let output_stall = self.check_output_buffer_stall(output_elements_per_col);
            self.cycle += output_stall;
            self.output_buffer_state.fill(output_elements_per_col);
        }

        let result_tensor = Tensor::from_slice(&result).reshape([m as i64, n as i64]);

        // Record bytes transferred
        let bytes_per_elem = self.bytes_per_element();
        self.stats.record_activation_bytes_transferred((activation_elements * bytes_per_elem) as u64);
        self.stats.record_weight_bytes_transferred((n * k * bytes_per_elem) as u64);
        self.stats.record_output_bytes_transferred((m * n * bytes_per_elem) as u64);

        self.stats.record_matmul((m * k) as u64);

        (result_tensor, self.cycle - start_cycle)
    }

    /// Perform batched matrix multiplication
    pub fn batched_matmul(&mut self, a: &Tensor, b: &Tensor) -> (Tensor, u64) {
        let batch_size = a.size()[0] as usize;

        let mut results = Vec::with_capacity(batch_size);
        let mut total = 0u64;
        
        for i in 0..batch_size {
            let a_slice = a.select(0, i as i64);
            let b_slice = if b.dim() == 3 {
                b.select(0, i as i64)
            } else {
                b.shallow_clone()
            };
            
            let (result, cycles) = self.matmul(&a_slice, &b_slice);
            results.push(result);
            total += cycles;
        }
        
        let stacked = Tensor::stack(&results, 0);
        (stacked, total)
    }

    /// Get estimated cycles for a matmul operation without executing
    /// Takes into account dataflow, buffer sizes, and bandwidth constraints
    pub fn estimate_cycles(&self, m: usize, k: usize, n: usize) -> usize {
        let array_size = self.config.rows.min(self.config.cols);

        // Base compute cycles from dataflow model
        let compute_cycles = self.config.dataflow.total_cycles(array_size, m, k, n);

        // Calculate number of tiles needed
        let k_tiles = (k + array_size - 1) / array_size;

        // Buffer stall analysis based on dataflow
        let buffer_stall_cycles = match self.config.dataflow {
            Dataflow::WeightStationary => {
                // WS: Weights loaded once per output tile, activations stream through
                // Weight buffer affects initial load, activation buffer affects streaming
                if self.config.weight_buffer_size >= 2 {
                    // Double buffering: can overlap weight load with previous tile's compute
                    0
                } else {
                    // Single buffer: must wait for weight load before compute
                    // Stall once per K-tile for weight reload
                    let weight_load_cycles = array_size;
                    k_tiles.saturating_sub(1) * weight_load_cycles
                }
            }
            Dataflow::OutputStationary => {
                // OS: Both weights and activations stream through
                // Need both buffers to support streaming
                let weight_stall = if self.config.weight_buffer_size >= 2 { 0 } else { array_size };
                let act_stall = if self.config.activation_buffer_size >= 2 { 0 } else { array_size };
                weight_stall + act_stall
            }
            Dataflow::ActivationStationary => {
                // IS: Activations loaded once, weights stream through
                if self.config.activation_buffer_size >= 2 {
                    0
                } else {
                    // Must reload activations for each output column
                    let n_tiles = (n + array_size - 1) / array_size;
                    let act_load_cycles = array_size;
                    n_tiles.saturating_sub(1) * act_load_cycles
                }
            }
        };

        // Output buffer stall: if output buffer is too small, must stall to drain
        let output_stall_cycles = if self.config.output_buffer_size >= 2 {
            0 // Can overlap drain with next tile's compute
        } else {
            // Must drain before starting next tile
            let m_tiles = (m + array_size - 1) / array_size;
            let n_tiles = (n + array_size - 1) / array_size;
            let drain_cycles = array_size;
            (m_tiles * n_tiles).saturating_sub(1) * drain_cycles
        };

        // Bandwidth-limited transfer cycles
        let bandwidth_stall_cycles = match &self.config.bandwidth {
            Some(bw) => {
                // Estimate bandwidth stalls based on data movement requirements
                // Weight transfer
                let weight_elements = match self.config.dataflow {
                    Dataflow::WeightStationary => k * n, // Preload all weights
                    Dataflow::OutputStationary => k * n, // Stream all weights
                    Dataflow::ActivationStationary => k * n, // Stream all weights
                };
                let weight_transfer_cycles = bw.weight_transfer_cycles(weight_elements) as usize;

                // Activation transfer
                let act_elements = match self.config.dataflow {
                    Dataflow::WeightStationary => m * k, // Stream all activations
                    Dataflow::OutputStationary => m * k, // Stream all activations
                    Dataflow::ActivationStationary => m * k, // Preload all activations
                };
                let act_transfer_cycles = bw.activation_transfer_cycles(act_elements) as usize;

                // Output transfer
                let output_elements = m * n;
                let output_transfer_cycles = bw.output_transfer_cycles(output_elements) as usize;

                // Estimate how much can be overlapped with compute
                // If compute is slower than data transfer, no additional stalls
                // If data transfer is slower, add the difference
                let total_transfer = weight_transfer_cycles + act_transfer_cycles + output_transfer_cycles;
                total_transfer.saturating_sub(compute_cycles)
            }
            None => 0,
        };

        compute_cycles + buffer_stall_cycles + output_stall_cycles + bandwidth_stall_cycles
    }

    /// Print detailed PE state (for debugging)
    pub fn print_pe_state(&self) {
        println!("=== Systolic Array PE State (Cycle {}) ===", self.cycle);
        for r in 0..self.config.rows {
            for c in 0..self.config.cols {
                let pe = &self.pes[r][c];
                print!(
                    "[{:.2},{:.2},{:.2}] ",
                    pe.weight, pe.activation, pe.accumulator
                );
            }
            println!();
        }
    }

    /// Get all PE MAC counts for utilization analysis
    pub fn get_pe_mac_counts(&self) -> Vec<Vec<u64>> {
        self.pes
            .iter()
            .map(|row| row.iter().map(|pe| pe.mac_count).collect())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_matmul_ws() {
        let mut array = SystolicArray::with_size(4, 4, Dataflow::WeightStationary);
        
        // 2x3 @ 3x2 = 2x2
        let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([2, 3]);
        let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape([3, 2]);
        
        let expected = a.matmul(&b);
        let (result, _cycles) = array.matmul(&a, &b);
        
        println!("Expected:\n{}", expected);
        println!("Result:\n{}", result);
        
        // Allow small floating point differences
        let diff: f32 = (&result - &expected).abs().sum(tch::Kind::Float).try_into().unwrap();
        assert!(diff < 0.01, "Results differ by {}", diff);
    }

    #[test]
    fn test_dataflow_comparison() {
        let a = Tensor::from_slice(&[1.0f32; 16]).reshape([4, 4]);
        let b = Tensor::from_slice(&[1.0f32; 16]).reshape([4, 4]);
        
        for dataflow in [
            Dataflow::WeightStationary,
            Dataflow::OutputStationary,
            Dataflow::ActivationStationary,
        ] {
            let mut array = SystolicArray::with_size(4, 4, dataflow);
            let (result, cycles) = array.matmul(&a, &b);
            
            println!("{}: {} cycles", dataflow.name(), cycles);
            println!("Result sum: {}", result.sum(tch::Kind::Float));
        }
    }
}
