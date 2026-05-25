use quantize::MxDataType;

use crate::op;
use crate::{
    Accelerator, MATRIX_KV_TYPE, MATRIX_WEIGHT_TYPE, MLEN, PREFETCH_M_AMOUNT, PREFETCH_V_AMOUNT,
    STORE_V_AMOUNT, VECTOR_ACTIVATION_TYPE, VECTOR_KV_TYPE, VLEN,
};

impl Accelerator {
    /// Dispatch all H_* (HBM transfer) opcodes.
    pub(crate) async fn dispatch_mem(&mut self, op: &op::Opcode) {
        match op {
            op::Opcode::H_PREFETCH_M {
                rd,
                rs1,
                rs2,
                rstride,
                precision,
            } => {
                // TODO: rstride support to be added
                let offset = self.reg_file.gp(*rs1);
                let addr = self.reg_file.hbm(*rs2);
                let dtype = match precision {
                    op::MatrixPrecision::Weights => *MATRIX_WEIGHT_TYPE,
                    op::MatrixPrecision::KeyValue => *MATRIX_KV_TYPE,
                };

                let scale = match dtype {
                    MxDataType::Plain(_) => 0,
                    MxDataType::Mx { elem, scale, block } => {
                        offset / (elem.size_in_bits() as u32 * block / scale.size_in_bits() as u32)
                    } // Element addr shifted by (element to scale ratio)
                };
                let xfer = self.transfer_mx_from_hbm(
                    addr + offset as u64,
                    addr + self.reg_file.scale as u64 + scale as u64,
                    dtype,
                    self.m_machine.mram.ty,
                    *rstride,
                    *MLEN,
                    *PREFETCH_M_AMOUNT,
                    *MLEN,
                );

                self.m_machine
                    .mram
                    .continous_write_delayed(self.reg_file.gp(*rd), *PREFETCH_M_AMOUNT, xfer)
                    .await;
            }
            op::Opcode::H_PREFETCH_V {
                rd,
                rs1,
                rs2,
                rstride,
                precision,
            } => {
                // TODO: rstride support to be added
                let offset = self.reg_file.gp(*rs1);
                let addr = self.reg_file.hbm(*rs2);
                let dtype = match precision {
                    op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                    op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                };

                let scale = match dtype {
                    MxDataType::Plain(_) => 0,
                    MxDataType::Mx { elem, scale, block } => {
                        offset / (elem.size_in_bits() as u32 * block / scale.size_in_bits() as u32)
                    }
                };
                let xfer = self.transfer_mx_from_hbm(
                    addr + offset as u64,
                    addr + self.reg_file.scale as u64 + scale as u64,
                    dtype,
                    self.v_machine.vram.ty(),
                    *rstride,
                    *VLEN,
                    *PREFETCH_V_AMOUNT,
                    1,
                );

                let dest = self.reg_file.gp(*rd);
                self.v_machine
                    .vram
                    .continous_write_delayed(dest, *PREFETCH_V_AMOUNT, xfer)
                    .await;
            }
            op::Opcode::H_STORE_V {
                rd,
                rs1,
                rs2,
                rstride,
                precision,
            } => {
                let src_addr = self.reg_file.gp(*rd);
                let offset = self.reg_file.gp(*rs1);
                let addr = self.reg_file.hbm(*rs2);
                let dtype = match precision {
                    op::VectorPrecision::Activation => *VECTOR_ACTIVATION_TYPE,
                    op::VectorPrecision::KeyValue => *VECTOR_KV_TYPE,
                };

                let scale = match dtype {
                    MxDataType::Plain(_) => 0,
                    MxDataType::Mx { elem, scale, block } => {
                        offset / (elem.size_in_bits() as u32 * block / scale.size_in_bits() as u32)
                    }
                };

                let element_index = addr + offset as u64;
                // Scales are stored AFTER elements, so scale_index = element_index + scale_reg + scale
                // where scale_reg is the offset from element start to scale start
                let scale_index = addr + self.reg_file.scale as u64 + scale as u64;

                self.transfer_mx_to_hbm(
                    src_addr,
                    element_index,
                    scale_index,
                    self.v_machine.vram.ty(),
                    dtype,
                    *rstride,
                    *VLEN,
                    *STORE_V_AMOUNT,
                )
                .await;
            }
            _ => unreachable!("dispatch_mem: non-mem opcode {:?}", op),
        }
    }
}
