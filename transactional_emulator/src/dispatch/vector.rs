use half::bf16;

use crate::Accelerator;
use crate::op;

impl Accelerator {
    /// Dispatch all V_* (vector) opcodes. Delegates to `VectorMachine`.
    pub(crate) async fn dispatch_vector(&mut self, op: &op::Opcode) {
        match op {
            op::Opcode::V_ADD_VV {
                rd,
                rs1,
                rs2,
                rmask,
            } => {
                let mask = self.resolve_v_mask(*rmask);
                self.v_machine
                    .add(
                        self.reg_file.read_gp(*rd),
                        self.reg_file.read_gp(*rs1),
                        self.reg_file.read_gp(*rs2),
                        *rmask,
                        mask,
                    )
                    .await;
            }
            op::Opcode::V_ADD_VF {
                rd,
                rs1,
                rs2,
                rmask,
            } => {
                let mask = self.resolve_v_mask(*rmask);
                self.v_machine
                    .add_scalar(
                        self.reg_file.read_gp(*rd),
                        self.reg_file.read_gp(*rs1),
                        self.reg_file.read_fp(*rs2).into(),
                        *rmask,
                        mask,
                    )
                    .await;
            }
            op::Opcode::V_SUB_VV {
                rd,
                rs1,
                rs2,
                rmask,
            } => {
                let mask = self.resolve_v_mask(*rmask);
                self.v_machine
                    .sub(
                        self.reg_file.read_gp(*rd),
                        self.reg_file.read_gp(*rs1),
                        self.reg_file.read_gp(*rs2),
                        *rmask,
                        mask,
                    )
                    .await;
            }
            op::Opcode::V_SUB_VF {
                rd,
                rs1,
                rs2,
                rmask,
                rorder,
            } => {
                let mask = self.resolve_v_mask(*rmask);
                self.v_machine
                    .sub_scalar(
                        self.reg_file.read_gp(*rd),
                        self.reg_file.read_gp(*rs1),
                        self.reg_file.read_fp(*rs2).into(),
                        *rmask,
                        mask,
                        *rorder,
                    )
                    .await;
            }
            op::Opcode::V_MUL_VV {
                rd,
                rs1,
                rs2,
                rmask,
            } => {
                let mask = self.resolve_v_mask(*rmask);
                self.v_machine
                    .mul(
                        self.reg_file.read_gp(*rd),
                        self.reg_file.read_gp(*rs1),
                        self.reg_file.read_gp(*rs2),
                        *rmask,
                        mask,
                    )
                    .await;
            }
            op::Opcode::V_MUL_VF {
                rd,
                rs1,
                rs2,
                rmask,
            } => {
                let mask = self.resolve_v_mask(*rmask);
                self.v_machine
                    .mul_scalar(
                        self.reg_file.read_gp(*rd),
                        self.reg_file.read_gp(*rs1),
                        self.reg_file.read_fp(*rs2).into(),
                        *rmask,
                        mask,
                    )
                    .await;
            }
            op::Opcode::V_EXP_V { rd, rs1, rmask } => {
                let mask = self.resolve_v_mask(*rmask);
                self.v_machine
                    .exp(
                        self.reg_file.read_gp(*rd),
                        self.reg_file.read_gp(*rs1),
                        *rmask,
                        mask,
                    )
                    .await;
            }
            op::Opcode::V_RECI_V { rd, rs1, rmask } => {
                let mask = self.resolve_v_mask(*rmask);
                self.v_machine
                    .reciprocal(
                        self.reg_file.read_gp(*rd),
                        self.reg_file.read_gp(*rs1),
                        *rmask,
                        mask,
                    )
                    .await;
            }
            op::Opcode::V_SHIFT_V { rd, rs1, rs2 } => {
                self.v_machine
                    .shift_scalar(
                        self.reg_file.read_gp(*rd),
                        self.reg_file.read_gp(*rs1),
                        self.reg_file.read_gp(*rs2),
                    )
                    .await;
            }
            // Write to fp0 is a no-op.
            op::Opcode::V_RED_SUM { rd: 0, .. } | op::Opcode::V_RED_MAX { rd: 0, .. } => (),

            op::Opcode::V_RED_SUM { rd, rs1, rmask } => {
                let mask = self.resolve_v_mask(*rmask);
                let result = self
                    .v_machine
                    .reduce_sum(
                        self.reg_file.read_gp(*rs1),
                        self.reg_file.read_fp(*rd).into(),
                        *rmask,
                        mask,
                    )
                    .await;
                self.reg_file.fp_reg[*rd as usize] = bf16::from_f32(result);
            }
            op::Opcode::V_RED_MAX { rd, rs1, rmask } => {
                let mask = self.resolve_v_mask(*rmask);
                let result = self
                    .v_machine
                    .reduce_max(
                        self.reg_file.read_gp(*rs1),
                        self.reg_file.read_fp(*rd).into(),
                        *rmask,
                        mask,
                    )
                    .await;
                self.reg_file.fp_reg[*rd as usize] = bf16::from_f32(result);
            }
            _ => unreachable!("dispatch_vector: non-vector opcode {:?}", op),
        }
    }
}
