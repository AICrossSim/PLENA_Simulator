//! Nested C_LOOP_* control state for the opcode interpreter.
//!
//! This module owns both the active loop stack and the ISA-visible loop counter
//! register side effects. Keeping those updates here makes each C_LOOP_* opcode
//! handler in dispatch only choose the loop operation and apply the returned PC
//! decision.

use crate::runtime_config::MAX_LOOP_INSTRUCTIONS;
use crate::timing::AguStreamUpdate;

use super::registers::AcceleratorRegFile;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum LoopDecision {
    Continue,
    JumpTo(usize),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct AguBoundaryResolution {
    pub(super) marker_pc: usize,
    pub(super) target_pc: usize,
    pub(super) loop_depth: usize,
    pub(super) iteration_before: u32,
    pub(super) iteration_after: u32,
    pub(super) exiting: bool,
    pub(super) stream_updates: Vec<AguStreamUpdate>,
}

/// Tracks active C_LOOP_* state and updates loop counter registers.
pub(super) struct LoopState {
    stack: Vec<LoopInfo>,
    pending_agu_streams: Vec<AguDescriptor>,
    pending_agu_body_len: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AguDescriptor {
    gp_register: u8,
    stride: i64,
}

/// Information about an active loop.
struct LoopInfo {
    start_pc: usize,
    iteration_count: u32,
    current_iteration: u32,
    instruction_count: usize,
    loop_reg: u8,
    agu: Option<AguLoopInfo>,
}

struct AguLoopInfo {
    body_start_pc: usize,
    marker_pc: usize,
    descriptors: Vec<AguDescriptor>,
}

impl LoopState {
    pub(super) fn new() -> Self {
        Self {
            stack: Vec::new(),
            pending_agu_streams: Vec::new(),
            pending_agu_body_len: None,
        }
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
            agu: None,
        });
        tracing::debug!(
            "C_LOOP_START: Starting loop at PC {} with {} iterations",
            pc,
            iteration_count
        );
    }

    pub(super) fn configure_agu_stride(&mut self, gp_register: u8, encoded: u32) {
        assert_ne!(gp_register, 0, "gp0 cannot be bound to the loop AGU");
        assert!(
            self.pending_agu_streams.len() < 6,
            "a loop may bind at most six AGU streams"
        );
        let raw = encoded & ((1 << 17) - 1);
        let mantissa = if raw & (1 << 16) != 0 {
            i64::from(raw) - (1 << 17)
        } else {
            i64::from(raw)
        };
        let shift = encoded >> 17;
        assert_ne!(mantissa, 0, "zero-stride AGU binding is invalid in AGU v1");
        self.pending_agu_streams.push(AguDescriptor {
            gp_register,
            stride: mantissa << shift,
        });
    }

    pub(super) fn configure_agu_body_len(&mut self, body_len: u32) {
        assert!(
            body_len > 0,
            "AGU loop body must contain at least one instruction"
        );
        self.pending_agu_body_len = Some(body_len as usize);
    }

    pub(super) fn start_agu(
        &mut self,
        pc: usize,
        loop_reg: u8,
        iteration_count: u32,
        reg_file: &mut AcceleratorRegFile,
    ) {
        assert!(
            self.stack.len() < 4,
            "hardware loop nesting exceeds four levels"
        );
        let body_len = self
            .pending_agu_body_len
            .take()
            .expect("C_LOOP_START_AGU requires C_AGU_LOOP_LEN");
        let descriptors = std::mem::take(&mut self.pending_agu_streams);
        assert!(
            descriptors.iter().enumerate().all(|(index, descriptor)| {
                descriptors[..index]
                    .iter()
                    .all(|other| other.gp_register != descriptor.gp_register)
            }),
            "one AGU loop cannot bind the same GP more than once"
        );
        assert!(
            self.stack.iter().all(|active| {
                active.agu.as_ref().is_none_or(|agu| {
                    descriptors.iter().all(|pending| {
                        agu.descriptors
                            .iter()
                            .all(|current| current.gp_register != pending.gp_register)
                    })
                })
            }),
            "overlapping nested AGU frames cannot bind the same GP"
        );
        assert!(
            descriptors
                .iter()
                .all(|descriptor| descriptor.gp_register != loop_reg),
            "AGU loop counter register cannot also be an address stream"
        );
        assert!(
            iteration_count > 0,
            "Iteration count must be greater than 0"
        );
        if loop_reg != 0 {
            reg_file.write_gp(loop_reg, iteration_count);
        }
        self.stack.push(LoopInfo {
            start_pc: pc,
            iteration_count,
            current_iteration: iteration_count,
            instruction_count: 0,
            loop_reg,
            agu: Some(AguLoopInfo {
                body_start_pc: pc + 1,
                marker_pc: pc + 1 + body_len,
                descriptors,
            }),
        });
    }

    /// Resolve an AGU boundary before the static C_LOOP_END marker is fetched.
    pub(super) fn before_instruction(
        &mut self,
        pc: usize,
        reg_file: &mut AcceleratorRegFile,
    ) -> Option<AguBoundaryResolution> {
        let loop_depth = self.stack.len();
        let loop_info = self.stack.last_mut()?;
        let agu = loop_info.agu.as_mut()?;
        if pc != agu.marker_pc {
            return None;
        }
        let iteration_before = loop_info.current_iteration;
        let mut stream_updates = Vec::with_capacity(agu.descriptors.len());
        for descriptor in &mut agu.descriptors {
            let old_offset = reg_file.gp_affine_offset(descriptor.gp_register);
            reg_file.advance_gp_affine(descriptor.gp_register, descriptor.stride);
            stream_updates.push(AguStreamUpdate {
                gp_register: descriptor.gp_register,
                stride: descriptor.stride,
                old_offset,
                new_offset: reg_file.gp_affine_offset(descriptor.gp_register),
            });
        }
        let counter = if loop_info.loop_reg == 0 {
            loop_info.current_iteration
        } else {
            reg_file.read_gp(loop_info.loop_reg)
        };
        if counter > 1 {
            if loop_info.loop_reg != 0 {
                reg_file.write_gp(loop_info.loop_reg, counter - 1);
            }
            loop_info.current_iteration = counter - 1;
            loop_info.instruction_count = 0;
            Some(AguBoundaryResolution {
                marker_pc: pc,
                target_pc: agu.body_start_pc,
                loop_depth,
                iteration_before,
                iteration_after: counter - 1,
                exiting: false,
                stream_updates,
            })
        } else {
            if loop_info.loop_reg != 0 {
                reg_file.write_gp(loop_info.loop_reg, 0);
            }
            self.stack.pop();
            Some(AguBoundaryResolution {
                marker_pc: pc,
                target_pc: pc + 1,
                loop_depth,
                iteration_before,
                iteration_after: 0,
                exiting: true,
                stream_updates,
            })
        }
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

    #[test]
    fn agu_boundary_updates_offsets_and_skips_static_marker() {
        let mut regs = AcceleratorRegFile::new();
        let mut loops = LoopState::new();
        regs.write_gp(1, 100);
        loops.configure_agu_stride(1, 64);
        loops.configure_agu_body_len(2);
        loops.start_agu(4, 7, 3, &mut regs);

        assert_eq!(loops.before_instruction(6, &mut regs), None);
        let first = loops.before_instruction(7, &mut regs).unwrap();
        assert_eq!(first.target_pc, 5);
        assert_eq!(first.loop_depth, 1);
        assert_eq!(first.iteration_before, 3);
        assert_eq!(first.iteration_after, 2);
        assert!(!first.exiting);
        assert_eq!(first.stream_updates[0].old_offset, 0);
        assert_eq!(first.stream_updates[0].new_offset, 64);
        assert_eq!(regs.read_gp(1), 164);
        assert_eq!(regs.read_gp(7), 2);
        assert_eq!(loops.before_instruction(7, &mut regs).unwrap().target_pc, 5);
        assert_eq!(regs.read_gp(1), 228);
        let last = loops.before_instruction(7, &mut regs).unwrap();
        assert_eq!(last.target_pc, 8);
        assert!(last.exiting);
        assert_eq!(regs.read_gp(1), 292);
        assert_eq!(regs.read_gp(7), 0);
        assert_eq!(loops.depth(), 0);
    }

    #[test]
    fn agu_stride_decodes_negative_shifted_mantissa() {
        let mut regs = AcceleratorRegFile::new();
        let mut loops = LoopState::new();
        regs.write_gp(1, 1024);
        let encoded_minus_64 = (3 << 17) | ((-8_i32 as u32) & ((1 << 17) - 1));
        loops.configure_agu_stride(1, encoded_minus_64);
        loops.configure_agu_body_len(1);
        loops.start_agu(0, 7, 1, &mut regs);
        assert_eq!(loops.before_instruction(2, &mut regs).unwrap().target_pc, 3);
        assert_eq!(regs.read_gp(1), 960);
    }

    #[test]
    fn agu_loop_can_use_gp0_as_an_internal_counter() {
        let mut regs = AcceleratorRegFile::new();
        let mut loops = LoopState::new();

        loops.configure_agu_stride(3, 1);
        loops.configure_agu_body_len(1);
        loops.start_agu(4, 0, 3, &mut regs);

        assert_eq!(regs.read_gp(0), 0);
        assert_eq!(loops.before_instruction(6, &mut regs).unwrap().target_pc, 5);
        assert_eq!(regs.read_gp(0), 0);
        assert_eq!(loops.before_instruction(6, &mut regs).unwrap().target_pc, 5);
        assert_eq!(regs.read_gp(0), 0);
        assert_eq!(loops.before_instruction(6, &mut regs).unwrap().target_pc, 7);
        assert_eq!(regs.read_gp(0), 0);
        assert_eq!(regs.read_gp(3), 3);
    }
}
