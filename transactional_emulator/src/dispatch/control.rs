use crate::op;
use crate::{Accelerator, LoopInfo, cycle, is_quiet};

impl Accelerator {
    /// Dispatch all C_* (control / config) opcodes. Returns `Some(target_pc)`
    /// if the instruction triggers a jump (currently only `C_LOOP_END`).
    pub(crate) async fn dispatch_control(&mut self, pc: usize, op: &op::Opcode) -> Option<usize> {
        let mut jump_pc: Option<usize> = None;
        match op {
            op::Opcode::C_SET_ADDR_REG { rd, rs1, rs2 } => {
                let imm = ((self.reg_file.read_gp(*rs1) as u64) << 32)
                    | (self.reg_file.read_gp(*rs2) as u64);
                self.reg_file.hbm_addr_reg[*rd as usize] = imm;
                cycle!(1);
            }
            op::Opcode::C_SET_SCALE_REG { rd } => {
                self.reg_file.scale = self.reg_file.read_gp(*rd);
                cycle!(1);
            }
            op::Opcode::C_SET_STRIDE_REG { rd } => {
                self.reg_file.stride = self.reg_file.read_gp(*rd);
                cycle!(1);
            }
            op::Opcode::C_SET_V_MASK_REG { rd } => {
                self.reg_file.v_mask = self.reg_file.read_gp(*rd);
                cycle!(1);
            }
            op::Opcode::C_LOOP_START { rd, imm } => {
                // Store iteration count in register
                assert!(*imm > 0, "Iteration count must be greater than 0");
                let iteration_count = *imm as u32;
                self.reg_file.gp_reg[*rd as usize] = iteration_count;

                // Push new loop onto stack
                self.loop_stack.push(LoopInfo {
                    start_pc: pc,
                    iteration_count,
                    current_iteration: iteration_count,
                    instruction_count: 0,
                    loop_reg: *rd,
                });

                if !is_quiet() {
                    println!(
                        "C_LOOP_START: Starting loop at PC {} with {} iterations",
                        pc, iteration_count
                    );
                }
                cycle!(1);
            }
            op::Opcode::C_LOOP_END { rd } => {
                // Find the matching loop (most recent loop with matching register)
                if let Some(loop_info) =
                    self.loop_stack.iter_mut().rev().find(|l| l.loop_reg == *rd)
                {
                    // Decrement the register (as per spec)
                    let reg_value = self.reg_file.read_gp(*rd);
                    if reg_value > 1 {
                        // More iterations remaining, loop back
                        self.reg_file.gp_reg[*rd as usize] = reg_value - 1;

                        // Update loop state
                        loop_info.current_iteration = reg_value - 1;
                        loop_info.instruction_count = 0; // Reset instruction count for next iteration

                        // Jump back to C_LOOP_START + 1 (skip the C_LOOP_START instruction itself)
                        jump_pc = Some(loop_info.start_pc + 1);

                        if !is_quiet() {
                            println!(
                                "C_LOOP_END: Looping back to PC {} (remaining iterations: {})",
                                loop_info.start_pc + 1,
                                reg_value - 1
                            );
                        }
                    } else {
                        // Last iteration (reg_value == 1) or already done (reg_value == 0)
                        // Decrement to 0 and exit the loop
                        self.reg_file.gp_reg[*rd as usize] = 0;

                        // Loop is complete, pop it from stack
                        if !is_quiet() {
                            println!(
                                "C_LOOP_END: Loop at PC {} completed (executed {} times)",
                                loop_info.start_pc, loop_info.iteration_count
                            );
                        }
                        // Remove this loop from the stack
                        let loop_reg = loop_info.loop_reg;
                        let pos = self
                            .loop_stack
                            .iter()
                            .rposition(|l| l.loop_reg == loop_reg)
                            .unwrap();
                        self.loop_stack.remove(pos);
                    }
                } else {
                    panic!(
                        "C_LOOP_END: No matching C_LOOP_START found for register {}",
                        *rd
                    );
                }
                cycle!(1);
            }
            op::Opcode::C_BREAK => {
                // Break out of the innermost loop
                if let Some(loop_info) = self.loop_stack.pop() {
                    if !is_quiet() {
                        println!("C_BREAK: Breaking out of loop at PC {}", loop_info.start_pc);
                    }
                    // Set the loop register to 0 to indicate loop is done
                    self.reg_file.gp_reg[loop_info.loop_reg as usize] = 0;
                } else {
                    panic!("C_BREAK: No active loop to break out of");
                }
                cycle!(1);
            }
            _ => unreachable!("dispatch_control: non-control opcode {:?}", op),
        }
        jump_pc
    }
}
