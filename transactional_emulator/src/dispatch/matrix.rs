use crate::Accelerator;
use crate::op;

impl Accelerator {
    /// Dispatch all M_* (matrix) opcodes. Delegates to `MatrixMachine`.
    pub(crate) async fn dispatch_matrix(&mut self, op: &op::Opcode) {
        match op {
            op::Opcode::M_MM { rs1, rs2 } => {
                self.m_machine
                    .mm(self.reg_file.gp(*rs1), self.reg_file.gp(*rs2))
                    .await;
            }
            op::Opcode::M_MM_WO { rd, rstride, imm } => {
                let stride_len = if *rstride == 0 {
                    1
                } else {
                    self.reg_file.gp(*rstride)
                };
                self.m_machine
                    .mm_wo(self.reg_file.gp(*rd) + *imm as u32, stride_len as u32)
                    .await;
            }
            op::Opcode::M_TMM { rs1, rs2 } => {
                self.m_machine
                    .tmm(self.reg_file.gp(*rs1), self.reg_file.gp(*rs2))
                    .await;
            }
            op::Opcode::M_BMM { rs1, rs2, rd: _ } => {
                self.m_machine
                    .bmm(
                        self.reg_file.gp(*rs1),
                        self.reg_file.gp(*rs2),
                        self.reg_file.bmm_scale,
                    )
                    .await;
            }
            op::Opcode::M_BTMM { rs1, rs2, rd: _ } => {
                self.m_machine
                    .btmm(
                        self.reg_file.gp(*rs1),
                        self.reg_file.gp(*rs2),
                        self.reg_file.bmm_scale,
                    )
                    .await;
            }
            op::Opcode::M_BMM_WO { rd, imm } => {
                self.m_machine
                    .bmm_wo(self.reg_file.gp(*rd) + *imm as u32)
                    .await;
            }
            op::Opcode::M_MV { rs1, rs2 } => {
                self.m_machine
                    .mv(self.reg_file.gp(*rs1), self.reg_file.gp(*rs2))
                    .await;
            }
            op::Opcode::M_TMV { rs1, rs2 } => {
                self.m_machine
                    .tmv(self.reg_file.gp(*rs1), self.reg_file.gp(*rs2))
                    .await;
            }
            op::Opcode::M_BMV { rs1, rs2, rd } => {
                self.m_machine
                    .bmv(
                        self.reg_file.gp(*rs1) + self.reg_file.gp(*rd),
                        self.reg_file.gp(*rs2),
                        self.reg_file.bmm_scale,
                    )
                    .await;
            }
            op::Opcode::M_BTMV { rs1, rs2, rd } => {
                self.m_machine
                    .btmv(
                        self.reg_file.gp(*rs1) + self.reg_file.gp(*rd),
                        self.reg_file.gp(*rs2),
                        self.reg_file.bmm_scale,
                    )
                    .await;
            }
            op::Opcode::M_MV_WO { rd, imm } => {
                self.m_machine
                    .mv_wo(self.reg_file.gp(*rd) + *imm as u32)
                    .await;
            }
            op::Opcode::M_BMV_WO { rd, imm } => {
                self.m_machine
                    .bmv_wo(self.reg_file.gp(*rd) + *imm as u32)
                    .await;
            }
            _ => unreachable!("dispatch_matrix: non-matrix opcode {:?}", op),
        }
    }
}
