//! Nested C_LOOP_* control state for the opcode interpreter.
//!
//! This module owns both the active loop stack and the ISA-visible loop counter
//! register side effects. Keeping those updates here makes each C_LOOP_* opcode
//! handler in dispatch only choose the loop operation and apply the returned PC
//! decision.

use crate::runtime_config::MAX_LOOP_INSTRUCTIONS;

use super::registers::AcceleratorRegFile;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum LoopDecision {
    Continue,
    JumpTo(usize),
}

/// Tracks active C_LOOP_* state and updates loop counter registers.
pub(super) struct LoopState {
    stack: Vec<LoopInfo>,
}

/// Information about an active loop.
struct LoopInfo {
    start_pc: usize,
    iteration_count: u32,
    current_iteration: u32,
    instruction_count: usize,
    loop_reg: u8,
}

impl LoopState {
    pub(super) fn new() -> Self {
        Self { stack: Vec::new() }
    }

    #[cfg(test)]
    fn depth(&self) -> usize {
        self.stack.len()
    }

    /// Count one dispatched instruction against every active loop.
    pub(super) fn record_instruction(&mut self) {
        for loop_info in &mut self.stack {
            loop_info.instruction_count += 1;
            if loop_info.instruction_count > *MAX_LOOP_INSTRUCTIONS {
                tracing::error!(
                    loop_pc = loop_info.start_pc,
                    max = *MAX_LOOP_INSTRUCTIONS,
                    current_iter = loop_info.current_iteration,
                    instructions = loop_info.instruction_count,
                    "Loop exceeded max instructions limit"
                );
                panic!(
                    "Loop at PC {} exceeded max instructions limit ({}). Current iteration: {}, Instructions in this iteration: {}",
                    loop_info.start_pc,
                    *MAX_LOOP_INSTRUCTIONS,
                    loop_info.current_iteration,
                    loop_info.instruction_count
                );
            }
        }
    }

    /// Start a loop and initialize its loop counter register.
    pub(super) fn start(
        &mut self,
        pc: usize,
        loop_reg: u8,
        iteration_count: u32,
        reg_file: &mut AcceleratorRegFile,
    ) {
        assert!(
            iteration_count > 0,
            "Iteration count must be greater than 0"
        );
        reg_file.write_gp(loop_reg, iteration_count);
        self.stack.push(LoopInfo {
            start_pc: pc,
            iteration_count,
            current_iteration: iteration_count,
            instruction_count: 0,
            loop_reg,
        });
        tracing::debug!(
            "C_LOOP_START: Starting loop at PC {} with {} iterations",
            pc,
            iteration_count
        );
    }

    /// Advance or complete a loop, updating its loop counter register.
    pub(super) fn end(&mut self, loop_reg: u8, reg_file: &mut AcceleratorRegFile) -> LoopDecision {
        let Some(pos) = self.stack.iter().rposition(|l| l.loop_reg == loop_reg) else {
            tracing::error!(
                rd = loop_reg,
                loop_stack_depth = self.stack.len(),
                "C_LOOP_END: No matching C_LOOP_START found"
            );
            panic!("C_LOOP_END: No matching C_LOOP_START found for register {loop_reg}");
        };

        let reg_value = reg_file.read_gp(loop_reg);
        if reg_value > 1 {
            reg_file.write_gp(loop_reg, reg_value - 1);

            let loop_info = &mut self.stack[pos];
            loop_info.current_iteration = reg_value - 1;
            loop_info.instruction_count = 0;
            let target_pc = loop_info.start_pc + 1;

            tracing::debug!(
                "C_LOOP_END: Looping back to PC {} (remaining iterations: {})",
                target_pc,
                reg_value - 1
            );

            LoopDecision::JumpTo(target_pc)
        } else {
            reg_file.write_gp(loop_reg, 0);

            let loop_info = self.stack.remove(pos);
            tracing::debug!(
                "C_LOOP_END: Loop at PC {} completed (executed {} times)",
                loop_info.start_pc,
                loop_info.iteration_count
            );

            LoopDecision::Continue
        }
    }

    /// Break the innermost loop and clear its loop counter register.
    pub(super) fn break_innermost(&mut self, reg_file: &mut AcceleratorRegFile) {
        if let Some(loop_info) = self.stack.pop() {
            tracing::debug!("C_BREAK: Breaking out of loop at PC {}", loop_info.start_pc);
            reg_file.write_gp(loop_info.loop_reg, 0);
        } else {
            tracing::error!("C_BREAK: No active loop to break out of");
            panic!("C_BREAK: No active loop to break out of");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::registers::AcceleratorRegFile;
    use super::{LoopDecision, LoopState};

    #[test]
    fn loop_state_tracks_iterations_and_jump_targets() {
        let mut regs = AcceleratorRegFile::new();
        let mut loops = LoopState::new();

        loops.start(4, 2, 3, &mut regs);

        assert_eq!(regs.read_gp(2), 3);
        assert_eq!(loops.depth(), 1);

        assert_eq!(loops.end(2, &mut regs), LoopDecision::JumpTo(5));
        assert_eq!(regs.read_gp(2), 2);
        assert_eq!(loops.depth(), 1);

        assert_eq!(loops.end(2, &mut regs), LoopDecision::JumpTo(5));
        assert_eq!(regs.read_gp(2), 1);
        assert_eq!(loops.depth(), 1);

        assert_eq!(loops.end(2, &mut regs), LoopDecision::Continue);
        assert_eq!(regs.read_gp(2), 0);
        assert_eq!(loops.depth(), 0);
    }

    #[test]
    fn loop_state_breaks_innermost_loop() {
        let mut regs = AcceleratorRegFile::new();
        let mut loops = LoopState::new();

        loops.start(10, 1, 2, &mut regs);
        loops.start(20, 3, 4, &mut regs);
        loops.break_innermost(&mut regs);

        assert_eq!(regs.read_gp(3), 0);
        assert_eq!(regs.read_gp(1), 2);
        assert_eq!(loops.depth(), 1);
    }
}
