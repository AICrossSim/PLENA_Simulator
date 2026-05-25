use half::bf16;

use crate::op;
use crate::{
    Accelerator, SCALAR_FP_BASIC_CYCLES, SCALAR_FP_EXP_CYCLES, SCALAR_FP_RECI_CYCLES,
    SCALAR_FP_SQRT_CYCLES, SCALAR_INT_BASIC_CYCLES, VLEN, cycle,
};

impl Accelerator {
    /// Dispatch all S_* (scalar) opcodes.
    pub(crate) async fn dispatch_scalar(&mut self, op: &op::Opcode) {
        match op {
            // Write to fp0 is a no-op.
            op::Opcode::S_ADD_FP { rd: 0, .. }
            | op::Opcode::S_SUB_FP { rd: 0, .. }
            | op::Opcode::S_MAX_FP { rd: 0, .. }
            | op::Opcode::S_MUL_FP { rd: 0, .. }
            | op::Opcode::S_EXP_FP { rd: 0, .. }
            | op::Opcode::S_RECI_FP { rd: 0, .. }
            | op::Opcode::S_SQRT_FP { rd: 0, .. } => {}

            op::Opcode::S_ADD_FP { rd, rs1, rs2 } => {
                self.reg_file.fp_reg[*rd as usize] =
                    self.reg_file.read_fp(*rs1) + self.reg_file.read_fp(*rs2);
                cycle!(*SCALAR_FP_BASIC_CYCLES);
            }
            op::Opcode::S_SUB_FP { rd, rs1, rs2 } => {
                self.reg_file.fp_reg[*rd as usize] =
                    self.reg_file.read_fp(*rs1) - self.reg_file.read_fp(*rs2);
                cycle!(*SCALAR_FP_BASIC_CYCLES);
            }
            op::Opcode::S_MAX_FP { rd, rs1, rs2 } => {
                self.reg_file.fp_reg[*rd as usize] =
                    bf16::max(self.reg_file.read_fp(*rs1), self.reg_file.read_fp(*rs2));
                cycle!(*SCALAR_FP_BASIC_CYCLES);
            }
            op::Opcode::S_MUL_FP { rd, rs1, rs2 } => {
                self.reg_file.fp_reg[*rd as usize] =
                    self.reg_file.read_fp(*rs1) * self.reg_file.read_fp(*rs2);
                cycle!(*SCALAR_FP_BASIC_CYCLES);
            }
            op::Opcode::S_EXP_FP { rd, rs1 } => {
                let val: f32 = self.reg_file.read_fp(*rs1).into();
                let clamped = val.clamp(-88.0, 88.0);
                self.reg_file.fp_reg[*rd as usize] = bf16::from_f32(clamped.exp());
                cycle!(*SCALAR_FP_EXP_CYCLES);
            }
            op::Opcode::S_RECI_FP { rd, rs1 } => {
                self.reg_file.fp_reg[*rd as usize] = bf16::ONE / self.reg_file.read_fp(*rs1);
                cycle!(*SCALAR_FP_RECI_CYCLES);
            }
            op::Opcode::S_SQRT_FP { rd, rs1 } => {
                self.reg_file.fp_reg[*rd as usize] =
                    bf16::from_f32(f32::from(self.reg_file.read_fp(*rs1)).sqrt());
                cycle!(*SCALAR_FP_SQRT_CYCLES);
            }
            op::Opcode::S_LD_FP { rd, rs1, imm } => {
                self.reg_file.fp_reg[*rd as usize] =
                    self.fpsram[(self.reg_file.read_gp(*rs1) + *imm) as usize];
                cycle!(1);
            }
            op::Opcode::S_ST_FP { rd, rs1, imm } => {
                self.fpsram[(self.reg_file.read_gp(*rs1) + *imm) as usize] =
                    self.reg_file.read_fp(*rd);
                cycle!(1);
            }
            op::Opcode::S_MAP_V_FP { rd, rs1, imm } => {
                let start_idx = (self.reg_file.read_gp(*rs1) + *imm) as usize;
                let end_idx = start_idx + *VLEN as usize;
                let f = &self.fpsram[start_idx..end_idx];
                self.v_machine
                    .vector_transfer_fp(self.reg_file.read_gp(*rd), f)
                    .await;
                cycle!(*VLEN);
            }
            op::Opcode::S_ADD_INT { rd, rs1, rs2 } => {
                self.reg_file.gp_reg[*rd as usize] = self
                    .reg_file
                    .read_gp(*rs1)
                    .wrapping_add(self.reg_file.read_gp(*rs2));
                cycle!(*SCALAR_INT_BASIC_CYCLES);
            }
            op::Opcode::S_ADDI_INT { rd, rs1, imm } => {
                self.reg_file.gp_reg[*rd as usize] =
                    self.reg_file.read_gp(*rs1).wrapping_add(*imm as u32);
                cycle!(*SCALAR_INT_BASIC_CYCLES);
            }
            op::Opcode::S_SUB_INT { rd, rs1, rs2 } => {
                self.reg_file.gp_reg[*rd as usize] = self
                    .reg_file
                    .read_gp(*rs1)
                    .wrapping_sub(self.reg_file.read_gp(*rs2));
                cycle!(*SCALAR_INT_BASIC_CYCLES);
            }
            op::Opcode::S_MUL_INT { rd, rs1, rs2 } => {
                self.reg_file.gp_reg[*rd as usize] = self
                    .reg_file
                    .read_gp(*rs1)
                    .wrapping_mul(self.reg_file.read_gp(*rs2));
                cycle!(*SCALAR_INT_BASIC_CYCLES);
            }
            op::Opcode::S_LUI_INT { rd, imm } => {
                self.reg_file.gp_reg[*rd as usize] = (*imm as u32) << 12;
                cycle!(*SCALAR_INT_BASIC_CYCLES);
            }
            op::Opcode::S_LD_INT { rd, rs1, imm } => {
                self.reg_file.gp_reg[*rd as usize] =
                    self.intsram[(self.reg_file.read_gp(*rs1) + *imm) as usize];
                cycle!(*SCALAR_INT_BASIC_CYCLES);
            }
            op::Opcode::S_ST_INT { rd, rs1, imm } => {
                self.intsram[(self.reg_file.read_gp(*rs1) + *imm) as usize] =
                    self.reg_file.read_gp(*rd);
                cycle!(*SCALAR_INT_BASIC_CYCLES);
            }
            _ => unreachable!("dispatch_scalar: non-scalar opcode {:?}", op),
        }
    }
}
